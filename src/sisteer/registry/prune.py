"""Superseded revisions' item rows: the reclaim rule (``prune-rule/3``), its digests, carrier plans, deletion ledger and rehydration.

ONE implementation, imported by the fold (:meth:`harvest.Ingestor.reclaim`, the rehydration hooks), the archive tool
(``prune_archive``), the census and the cohort gate.

The rule. For an item table ``T`` (``item_result``, ``item_flags``, ``run_degeneration``; never ``source_status``, never
``observation``), a revision ``R`` and a carrier ``c`` (a distinct ``record_file_id`` among ``R``'s rows in ``T``)::

    deletable(R, c, T)  <=>  R.accepted = 1 AND R.superseded_by_revision_id IS NOT NULL
                             AND c is not a staged file of R              (revision_staging_file / revision_staging_record)
                             AND c's log file exists and its logical sha256 equals record_ingest.file_sha256
                             AND authentic(R, c, T)
    authentic(R, c, T)  <=>  the current validator refuses no T record of c's file  (rehydration re-parses the whole file)
                             AND reparse(c, T) = present(R, c, T) + deleted(R, c, T)     (multisets of fold rows; counts and D_fold)

``reparse`` builds ``c``'s ``T`` rows the way the staging path does under the CURRENT code (``harvest.stream_parts(..., want="items")``
-> ``reducer.item_row_prefix``), reduced per key by the item guard; ``present`` is what ``T`` holds under ``(R, c)`` now; ``deleted`` is
what reclaim already removed from ``(R, c)``, carried in ``item_reclaim_ledger`` as a count and additive digests. ``authentic`` makes two
promises at deletion time: nothing of the carrier is displaced (every key is present or already deleted), and the deleting code itself
reproduces the rows it removes -- by the same re-parse a rehydration will make, which refuses a file holding a refused record. A deleted
carrier is re-folded from the record log (:func:`rehydrate`) before anything is written under its revision again, so no later fold ever sees
a state today's code would not have held; a rehydration that cannot authenticate writes nothing and fails closed
(:class:`RehydrationIncomplete`).

Digests. A row digest is ``sha256(b"prune-row/1\\0" + table + b"\\0" + encode_row(kept columns))`` read as a 256-bit integer; a group or
table digest is ``(count, sum of row digests mod 2**256)`` (:class:`Digest`, an additive multiset hash). Three column sets:
``digest_all`` (every column), ``digest_bind`` (every column but ``ingest_seq``: :data:`BIND_EXCLUDE`) and ``D_fold`` (also without
``withdrawn_by_revision_id``, the inert mark the fold no longer derives: :data:`FOLD_EXCLUDE`).

Row encoding (the archive's line format too): a compact JSON array of the kept columns in table column order; NULL ``null``, INTEGER an
int, REAL ``{"r": float.hex()}``, TEXT a string, BLOB ``{"x": hex}`` (:func:`encode_row` / :func:`decode_row`), ASCII-escaped.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import threading
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from . import records as R
from . import schema_def as S

#: The rule's version, written into every archive MANIFEST (``prune_archive``).
RULE_VERSION = "prune-rule/3"
#: The row-digest domain tag (part of every row digest).
ROW_DIGEST_TAG = b"prune-row/1\0"
#: The item tables the rule applies to (never ``source_status``, never ``observation``).
PRUNE_TABLES: Tuple[str, ...] = ("item_result", "item_flags", "run_degeneration")
#: Columns a ``digest_bind`` leaves out: the carrier's rank, which ``Ranker.finalize`` may rewrite on live rows.
BIND_EXCLUDE: Tuple[str, ...] = ("ingest_seq",)
#: Columns ``D_fold`` leaves out: the rank, and the inert mark the fold no longer derives (pre-0033 marks stay on retained rows).
FOLD_EXCLUDE: Tuple[str, ...] = ("ingest_seq", "withdrawn_by_revision_id")
#: The modulus of the additive digest.
MOD = 1 << 256
#: The ledger table (schema_def; migration 0033).
LEDGER_TABLE = "item_reclaim_ledger"

#: Carrier classes (``CarrierPlan.klass``). ``deletable`` is the only one reclaim removes. ``unauthentic_refused``: the current validator
#: refuses a record of the carrier's table in its file, so rehydration (which re-parses the whole file) would refuse to restore it -- the
#: carrier is kept, however well the records it does accept reproduce the rows.
DELETABLE = "deletable"
STAGED = "staged"
UNAUTHENTIC_COUNT = "unauthentic_count"
UNAUTHENTIC_DIGEST = "unauthentic_digest"
UNAUTHENTIC_REFUSED = "unauthentic_refused"
MISSING = "missing"
COHORT_UNCOVERED = "cohort_uncovered"
CLASSES: Tuple[str, ...] = (DELETABLE, STAGED, UNAUTHENTIC_COUNT, UNAUTHENTIC_DIGEST, UNAUTHENTIC_REFUSED, MISSING, COHORT_UNCOVERED)

#: The live mode file, under ``<root>/state/``: written and validated only by ``registry prune-superseded mode ...``; absent = normal.
MODE_STATE_FILE = "reclaim_mode.json"
MODE_STATE_FORMAT = "reclaim-mode/1"
MODES: Tuple[str, ...] = ("normal", "hold", "cohort")
#: A hold file's record of the cohort it interrupted: the ``{"dir", "manifest_sha256"}`` entries of the verified archives that cohort was
#: bound to (or ``{"unreadable": reason}`` when the file the hold replaced could not be read). Leaving the hold for ``normal`` checks them
#: as leaving cohort mode does: the obligation that every normally deletable carrier be archived first survives a pause.
OBLIGATION_KEY = "cohort_obligation"
#: The archive's files the cohort binding reads (``prune_archive`` writes them). ``groups.tsv.zst``: zstd-compressed TSV, one header line
#: (:data:`GROUPS_TSV_COLUMNS`) then one line per carrier group. ``VERIFIED.local.json`` (the ``mode cohort`` gate's local re-verification)
#: and ``VERIFIED.json`` (the cluster's independent one) are JSON objects naming the MANIFEST they verified under
#: :data:`VERIFIED_MANIFEST_KEY` (the sha256 of ``MANIFEST.json``'s bytes).
MANIFEST_FILE = "MANIFEST.json"
GROUPS_FILE = "groups.tsv.zst"
VERIFIED_FILE = "VERIFIED.json"
VERIFIED_LOCAL_FILE = "VERIFIED.local.json"
VERIFIED_MANIFEST_KEY = "manifest_sha256"
GROUPS_TSV_COLUMNS: Tuple[str, ...] = ("table", "revision_id", "record_file_id", "file_uri", "class", "rows", "digest_all", "digest_bind")

#: The tables' natural keys (== primary keys).
PK: Dict[str, Tuple[str, ...]] = {t: tuple(S.TABLE_BY_NAME[t].pk) for t in PRUNE_TABLES}
_PK = PK


def accepted_superseded(alias: str = "") -> str:
    """The SQL predicate over ``source_revision`` (columns qualified by ``alias.`` when given) that makes a revision's item rows
    eligible for reclaim: accepted, and superseded by a later revision of its source."""
    q = f"{alias}." if alias else ""
    return f"{q}accepted = 1 AND {q}superseded_by_revision_id IS NOT NULL"


#: Every deletion's guard (one parameter, the revision id): the row is removed only while its revision is still accepted and superseded.
DELETE_GUARD = f"EXISTS (SELECT 1 FROM source_revision s WHERE s.revision_id = ? AND {accepted_superseded('s')})"


# --------------------------------------------------------------------------------------------
# digests and the row encoding
# --------------------------------------------------------------------------------------------

def encode_value(v: object) -> object:
    """One SQLite value as its canonical JSON form (see the module docstring)."""
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return {"r": v.hex()}
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"x": bytes(v).hex()}
    raise TypeError(f"prune: no canonical encoding for {type(v).__name__} value {v!r}")


def decode_value(v: object) -> object:
    """The inverse of :func:`encode_value`."""
    if isinstance(v, dict):
        if "r" in v:
            return float.fromhex(str(v["r"]))
        if "x" in v:
            return bytes.fromhex(str(v["x"]))
        raise ValueError(f"prune: not an encoded value: {v!r}")
    return v


def encode_row(values: Sequence[object]) -> bytes:
    """A row (values in table column order) as compact canonical JSON bytes (ASCII, so the bytes never depend on a locale)."""
    return json.dumps([encode_value(v) for v in values], separators=(",", ":"), ensure_ascii=True).encode("ascii")


def decode_row(line: bytes) -> Tuple[object, ...]:
    """The inverse of :func:`encode_row`."""
    return tuple(decode_value(v) for v in json.loads(line))


def columns_of(table: str) -> Tuple[str, ...]:
    """The table's columns in table (row tuple) order."""
    return tuple(S.TABLE_BY_NAME[table].column_names)


_KEPT: Dict[Tuple[str, Tuple[str, ...]], Tuple[int, ...]] = {}


def _kept_positions(table: str, exclude: Sequence[str]) -> Tuple[int, ...]:
    k = (table, tuple(exclude))
    pos = _KEPT.get(k)
    if pos is None:
        cols = columns_of(table)
        pos = _KEPT[k] = tuple(i for i, c in enumerate(cols) if c not in set(exclude))
    return pos


def tagged_row_digest(name: str, values: Sequence[object]) -> int:
    """The row-digest formula, read as a 256-bit integer: ``sha256(ROW_DIGEST_TAG + name + b"\\0" + encode_row(values))``. ``name`` is
    the table (or ``v_*`` object) the values belong to and ``values`` exactly the values digested, in order."""
    return int.from_bytes(hashlib.sha256(ROW_DIGEST_TAG + name.encode("utf-8") + b"\0" + encode_row(values)).digest(), "big")


def row_digest(table: str, row: Sequence[object], exclude: Sequence[str] = ()) -> int:
    """The 256-bit digest of one row of ``table`` (``row`` = every column, in table order), over the columns not in ``exclude``:
    ``()`` for ``digest_all``, :data:`BIND_EXCLUDE` for ``digest_bind``, :data:`FOLD_EXCLUDE` for ``D_fold``."""
    pos = _kept_positions(table, exclude)
    if len(row) != len(columns_of(table)):
        raise ValueError(f"prune: a {table} row has {len(row)} values, the table {len(columns_of(table))} columns")
    return tagged_row_digest(table, [row[i] for i in pos])


@dataclass
class Digest:
    """An additive multiset digest: ``(count, sum256)``. ``add`` takes a row digest (int) or another Digest and returns self."""
    count: int = 0
    sum256: int = 0

    def add(self, other: "int | Digest") -> "Digest":
        if isinstance(other, Digest):
            self.count += other.count
            self.sum256 = (self.sum256 + other.sum256) % MOD
        else:
            self.count += 1
            self.sum256 = (self.sum256 + int(other)) % MOD
        return self

    def sub(self, other: "Digest") -> "Digest":
        """Remove ``other`` (a sub-multiset) from this one; returns self."""
        self.count -= other.count
        self.sum256 = (self.sum256 - other.sum256) % MOD
        return self

    def __add__(self, other: "Digest") -> "Digest":
        return Digest(self.count, self.sum256).add(other)

    def copy(self) -> "Digest":
        return Digest(self.count, self.sum256)

    def hex(self) -> str:
        """``sum256`` as 64 lowercase hex digits (the ledger's TEXT digest columns; the count is stored beside it)."""
        return f"{self.sum256 % MOD:064x}"

    @classmethod
    def from_hex(cls, count: int, hexsum: str) -> "Digest":
        return cls(int(count), int(str(hexsum or "0"), 16) % MOD)

    @classmethod
    def of_rows(cls, table: str, rows: Iterable[Sequence[object]], exclude: Sequence[str] = ()) -> "Digest":
        d = cls()
        for r in rows:
            d.add(row_digest(table, r, exclude))
        return d


# --------------------------------------------------------------------------------------------
# the reclaim mode
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchiveGroup:
    """One carrier group of a verified archive (a line of ``groups.tsv.zst``)."""
    table: str
    revision_id: str
    record_file_id: int
    file_uri: str
    klass: str
    rows: int
    digest_all: str
    digest_bind: str


class ModeStateInvalid(RuntimeError):
    """``state/reclaim_mode.json`` names something that no longer validates (a manifest moved, changed or lost its local
    verification), or does not parse: the live cycle deletes nothing and reports ``reclaim_stopped = "mode"`` with this message."""


def read_groups(archive_dir: str) -> List[ArchiveGroup]:
    """Every carrier group of the archive at ``archive_dir`` (``groups.tsv.zst``)."""
    path = os.path.join(archive_dir, GROUPS_FILE)
    with open(path, "rb") as fh:
        return parse_groups(fh.read(), path)


def parse_groups(data: bytes, path: str = GROUPS_FILE) -> List[ArchiveGroup]:
    """The carrier groups of the ``groups.tsv.zst`` bytes ``data`` (``path`` names them in a refusal)."""
    import io
    import zstandard
    text = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read().decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    if not lines or tuple(lines[0].split("\t")) != GROUPS_TSV_COLUMNS:
        raise ModeStateInvalid(f"{path}: header is not {GROUPS_TSV_COLUMNS}")
    out = []
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) != len(GROUPS_TSV_COLUMNS):
            raise ModeStateInvalid(f"{path}: a line has {len(f)} fields: {ln[:120]!r}")
        out.append(ArchiveGroup(f[0], f[1], int(f[2]), f[3], f[4], int(f[5]), f[6], f[7]))
    return out


def _read_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError as exc:
        raise ModeStateInvalid(f"{path}: unreadable ({exc})")


def verified_manifest_doc(archive_dir: str, verified_name: str = VERIFIED_LOCAL_FILE) -> Tuple[str, Dict[str, object]]:
    """``(sha256, MANIFEST)`` of ``archive_dir``'s MANIFEST.json -- the document parsed from exactly the bytes hashed -- after checking
    that ``verified_name`` names that sha256; raises :class:`ModeStateInvalid`."""
    mpath = os.path.join(archive_dir, MANIFEST_FILE)
    vpath = os.path.join(archive_dir, verified_name)
    if not os.path.isfile(mpath):
        raise ModeStateInvalid(f"{archive_dir}: no {MANIFEST_FILE}")
    data = _read_bytes(mpath)
    msha = hashlib.sha256(data).hexdigest()
    try:
        with open(vpath, encoding="utf-8") as fh:
            ver = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ModeStateInvalid(f"{archive_dir}: {verified_name} unreadable ({exc})")
    if not isinstance(ver, dict) or ver.get(VERIFIED_MANIFEST_KEY) != msha:
        raise ModeStateInvalid(f"{archive_dir}: {verified_name} does not name MANIFEST sha256 {msha}")
    try:
        manifest = json.loads(data)
    except ValueError as exc:
        raise ModeStateInvalid(f"{mpath}: not JSON ({exc})")
    if not isinstance(manifest, dict):
        raise ModeStateInvalid(f"{mpath}: not a JSON object")
    return msha, manifest


def verified_manifest(archive_dir: str, verified_name: str = VERIFIED_LOCAL_FILE) -> str:
    """The sha256 of ``archive_dir``'s MANIFEST.json, after checking that ``verified_name`` names it; raises :class:`ModeStateInvalid`."""
    return verified_manifest_doc(archive_dir, verified_name)[0]


def verified_groups(archive_dir: str, manifest: Mapping[str, object]) -> List[ArchiveGroup]:
    """The carrier groups of ``archive_dir``'s ``groups.tsv.zst`` BOUND to ``manifest`` (a MANIFEST a verification names): the bytes read
    must hash to ``manifest["groups"]["sha256"]`` and list ``manifest["groups"]["count"]`` groups, and the groups are parsed from exactly
    those bytes -- so a groups file swapped in after the verification, even another archive's valid one, binds nothing. Raises
    :class:`ModeStateInvalid`."""
    import zstandard
    path = os.path.join(archive_dir, GROUPS_FILE)
    entry = manifest.get("groups")
    want = entry.get("sha256") if isinstance(entry, dict) else None
    data = _read_bytes(path)
    have = hashlib.sha256(data).hexdigest()
    if not want or have != want:
        raise ModeStateInvalid(f"{path}: sha256 {have} is not the {want} its verified {MANIFEST_FILE} names")
    try:
        groups = parse_groups(data, path)
    except (ValueError, zstandard.ZstdError) as exc:
        raise ModeStateInvalid(f"{path}: unreadable ({exc})")
    if len(groups) != entry.get("count"):
        raise ModeStateInvalid(f"{path}: {len(groups)} groups, its verified {MANIFEST_FILE} names {entry.get('count')}")
    return groups


@dataclass(frozen=True)
class ReclaimMode:
    """What reclaim may delete -- an EXPLICIT argument, independent of ``Ingestor.live``:

    * ``hold``: nothing (pass (i), clearing marks on a head's rows, still runs);
    * ``cohort``: only carrier groups bound to a locally verified archive (``manifests``), checked per group at admission;
    * ``normal``: every authentic carrier.

    The live cycle reads :meth:`from_state`; ``rebuild`` and replica increments pass :meth:`normal`; ``registry prune-superseded
    reclaim --mode`` passes what it is given."""
    kind: str
    manifests: Tuple[str, ...] = ()
    manifest_shas: Tuple[str, ...] = ()
    groups: Mapping[Tuple[str, str, int], ArchiveGroup] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.kind not in MODES:
            raise ValueError(f"reclaim mode {self.kind!r} is not one of {MODES}")

    @classmethod
    def normal(cls) -> "ReclaimMode":
        return cls("normal")

    @classmethod
    def hold(cls) -> "ReclaimMode":
        return cls("hold")

    @classmethod
    def cohort(cls, manifests: Sequence[str]) -> "ReclaimMode":
        """Cohort mode bound to the archive directories ``manifests``: each must hold ``MANIFEST.json``, a ``VERIFIED.local.json`` naming
        the MANIFEST's sha256 (written by the ``mode cohort`` gate after its local re-verification) and the ``groups.tsv.zst`` that MANIFEST
        names by sha256 (:func:`verified_groups`). Raises :class:`ModeStateInvalid` otherwise. A group listed by two manifests binds to the
        later one."""
        if not manifests:
            raise ModeStateInvalid("cohort mode needs at least one verified archive")
        dirs = tuple(os.path.abspath(m) for m in manifests)
        shas = []
        groups: Dict[Tuple[str, str, int], ArchiveGroup] = {}
        for d in dirs:
            msha, manifest = verified_manifest_doc(d, VERIFIED_LOCAL_FILE)
            shas.append(msha)
            for g in verified_groups(d, manifest):
                groups[(g.table, g.revision_id, g.record_file_id)] = g
        return cls("cohort", dirs, tuple(shas), groups)

    @staticmethod
    def state_path(root: str) -> str:
        return os.path.join(root, "state", MODE_STATE_FILE)

    @classmethod
    def from_state(cls, root: str) -> "ReclaimMode":
        """The live mode: ``<root>/state/reclaim_mode.json`` (absent = normal), its manifests re-validated by path and sha256 on every
        call. Raises :class:`ModeStateInvalid` when the file does not parse or a manifest no longer validates."""
        path = cls.state_path(root)
        if not os.path.exists(path):
            return cls.normal()
        try:
            with open(path, encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ModeStateInvalid(f"{path}: unreadable ({exc})")
        if not isinstance(st, dict) or st.get("format") != MODE_STATE_FORMAT or st.get("mode") not in MODES:
            raise ModeStateInvalid(f"{path}: not a {MODE_STATE_FORMAT} mode file")
        if st["mode"] == "normal":
            return cls.normal()
        if st["mode"] == "hold":
            return cls.hold()
        entries = st.get("manifests") or []
        if not entries:
            raise ModeStateInvalid(f"{path}: cohort mode names no archive")
        mode = cls.cohort([str(e.get("dir")) for e in entries])
        for e, have in zip(entries, mode.manifest_shas):
            if e.get("manifest_sha256") != have:
                raise ModeStateInvalid(f"{path}: {e.get('dir')}: MANIFEST sha256 {have} is not the {e.get('manifest_sha256')} the mode was set with")
        return mode

    def write_state(self, root: str, now: str, obligation: Sequence[Mapping[str, object]] = ()) -> str:
        """Write ``<root>/state/reclaim_mode.json`` for this mode (``normal`` removes it); returns the path. Called only by the ``mode``
        command after its own checks. ``obligation`` (``hold`` only): the archives a cohort the hold interrupts was bound to, kept under
        :data:`OBLIGATION_KEY` so leaving the hold for ``normal`` still requires them to cover every normally deletable carrier. Atomic
        (temp file, fsync, rename)."""
        path = self.state_path(root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.kind == "normal":
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            return path
        body: Dict[str, object] = {"format": MODE_STATE_FORMAT, "mode": self.kind, "written_utc": now,
                                   "manifests": [{"dir": d, "manifest_sha256": s} for d, s in zip(self.manifests, self.manifest_shas)]}
        if obligation:
            if self.kind != "hold":
                raise ValueError("a cohort obligation is written only into a hold file (cohort mode's own manifests are its obligation)")
            body[OBLIGATION_KEY] = [dict(e) for e in obligation]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=1, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path

    def binding(self, table: str, revision_id: str, record_file_id: int) -> Optional[ArchiveGroup]:
        """The archived group a cohort deletion must match, or None (not covered)."""
        return self.groups.get((table, revision_id, int(record_file_id)))


def mode_state(root: str) -> Tuple[Optional[ReclaimMode], Dict[str, object]]:
    """The live mode as the harvester reads it (:meth:`ReclaimMode.from_state`; None when the file does not validate) and the mode file as
    written: ``{"path", "raw" ({} when absent = normal, None when unreadable), "kind", "valid", "error"}``."""
    path = ReclaimMode.state_path(root)
    raw: Dict[str, object] = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = dict(json.load(fh))
        except (OSError, ValueError, TypeError) as exc:
            return None, {"path": path, "raw": None, "kind": None, "valid": False, "error": f"{path}: unreadable ({exc})"}
    try:
        mode = ReclaimMode.from_state(root)
    except ModeStateInvalid as exc:
        return None, {"path": path, "raw": raw, "kind": raw.get("mode"), "valid": False, "error": str(exc)}
    return mode, {"path": path, "raw": raw, "kind": mode.kind, "valid": True, "error": None}


def read_mode_state(root: str) -> Dict[str, object]:
    """The mode file as written (``{}`` when absent = normal) and whether it validates now (:func:`mode_state`'s second part)."""
    return mode_state(root)[1]


# --------------------------------------------------------------------------------------------
# carriers, re-parse, the ledger
# --------------------------------------------------------------------------------------------

@dataclass
class LedgerRow:
    """One ``item_reclaim_ledger`` row: what reclaim deleted from one carrier ``(table, revision, record_file_id)`` and nobody has
    restored since."""
    table_name: str
    revision_id: str
    record_file_id: int
    file_uri: str
    expected_rows: int
    expected_digest_fold: str
    rows_deleted: int
    deleted_digest_fold: str
    deleted_digest_bind: str
    parsed_with: str

    @property
    def expected(self) -> Digest:
        return Digest.from_hex(self.expected_rows, self.expected_digest_fold)

    @property
    def deleted_fold(self) -> Digest:
        return Digest.from_hex(self.rows_deleted, self.deleted_digest_fold)

    @property
    def deleted_bind(self) -> Digest:
        return Digest.from_hex(self.rows_deleted, self.deleted_digest_bind)


LEDGER_COLUMNS: Tuple[str, ...] = ("table_name", "revision_id", "record_file_id", "file_uri", "expected_rows", "expected_digest_fold",
                                   "rows_deleted", "deleted_digest_fold", "deleted_digest_bind", "parsed_with")
_TABLES_IN = ", ".join(f"'{t}'" for t in PRUNE_TABLES)


@dataclass
class CarrierPlan:
    """One carrier of one revision in one table, classified. ``klass`` is one of :data:`CLASSES`; ``rows`` = present rows now;
    ``expected`` = ``D_fold`` (with count) of ``reparse(c, T)`` (zero Digest when not computed: staged / missing);
    ``present_fold`` / ``present_bind`` = the present rows' ``D_fold`` / ``digest_bind``; ``detail`` names the sub-reason.
    ``transient`` marks a classification that may change without any write (a file read while it changed): reclaim does not record
    such a revision as done."""
    table: str
    revision_id: str
    record_file_id: int
    file_uri: str
    klass: str
    rows: int
    expected: Digest
    present_fold: Digest = field(default_factory=Digest)
    present_bind: Digest = field(default_factory=Digest)
    ledger: Optional[LedgerRow] = None
    detail: str = ""
    transient: bool = False
    ingest_seq: Optional[int] = None


class RehydrationIncomplete(RuntimeError):
    """A deleted carrier cannot be restored exactly: the current code's re-parse does not equal present + deleted (a record the
    validator now refuses, a changed parser, a changed or missing file). Nothing was written; the ledger rows stand."""

    def __init__(self, revision_id: str, table: str, file_uri: str, n: int, reason: str):
        super().__init__(f"rehydration of {file_uri} ({table}) does not reproduce the {n} deleted rows: {reason}")
        self.revision_id, self.table, self.file_uri, self.n, self.reason = revision_id, table, file_uri, n, reason


class BatchRefused(RuntimeError):
    """A deletion batch removed a different number of rows than it selected (the revision became visible, or a row moved under the
    batch): the transaction rolls back and nothing is deleted."""


# -- the normalisation of re-parsed rows: SQLite's own column affinity, so a re-parsed row digests exactly as the stored row would

_AFFINITY = threading.local()


def _affinity_conn() -> sqlite3.Connection:
    conn = getattr(_AFFINITY, "conn", None)
    if conn is None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        for t in PRUNE_TABLES:
            cols = ", ".join(f"{c.name} {c.type}" for c in S.TABLE_BY_NAME[t].columns)
            conn.execute(f"CREATE TABLE {t} ({cols})")
        _AFFINITY.conn = conn
    return conn


def normalize_rows(table: str, rows: Sequence[Tuple]) -> List[Tuple]:
    """``rows`` as SQLite would store them in ``table``: each value through the column's declared type affinity (an integral REAL is a
    float, a numeric TEXT stays text, ...), by a round trip through an in-memory table of the same column types."""
    if not rows:
        return []
    conn = _affinity_conn()
    ncol = len(columns_of(table))
    conn.execute("BEGIN")
    try:
        conn.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in range(ncol))})", rows)
        out = [tuple(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        conn.execute(f"DELETE FROM {table}")
    finally:
        conn.execute("COMMIT")
    return out


def _key(table: str, row: Sequence[object]) -> Tuple:
    from .reducer import ITEM_PLANS
    return tuple(row[i] for i in ITEM_PLANS[table].pk_positions)


@dataclass
class _Reparse:
    """One file re-parsed for one revision: per table, the guard-reduced, normalised full rows (``ingest_seq`` = the seq given)."""
    file_sha: str = ""
    error: str = ""
    retry: bool = False
    rows: Dict[str, List[Tuple]] = field(default_factory=dict)
    refused: Dict[str, List[str]] = field(default_factory=dict)   # table -> the uri#line of every record of it the validator refused

    def fold(self, table: str) -> Digest:
        return Digest.of_rows(table, self.rows.get(table, ()), FOLD_EXCLUDE)


def reparse_file(root: str, file_uri: str, revision_id: Optional[str], tables: Sequence[str] = PRUNE_TABLES,
                  ingest_seq: int = 0) -> _Reparse:
    """Re-parse ``file_uri`` ONCE for every table in ``tables`` (the staging path's items pass: ``stream_parts(..., want="items")``),
    keeping the rows tagged ``revision_id`` (all rows when None), reduced per key by the item guard (the greatest
    ``(written_utc, record_sha256)`` wins) and normalised by :func:`normalize_rows`."""
    from . import harvest as H
    from . import records as R
    from . import site_aliases as SA
    from .reducer import ITEM_PLANS
    SA.install(root)            # the root's site aliases, as the fold that folded the file had them (cached per root; an Ingestor installs the same)
    out = _Reparse()
    path = H.resolve_uri(root, file_uri)
    if path is None:
        out.error = "the record file is not on disk"
        return out
    want = set(tables)
    best: Dict[str, Dict[Tuple, Tuple]] = {t: {} for t in tables}
    order_pos = {t: (ITEM_PLANS[t].columns.index("written_utc"), ITEM_PLANS[t].columns.index("record_sha256")) for t in tables}
    for part in H.stream_parts(root, path, None, H.ITEM_ROWS_PER_TRANSACTION, want="items"):
        if part.error:
            out.error = str(part.error)
        if part.retry:
            out.retry = True
        for pr in part.records:
            res = pr.result
            rtype = getattr(res, "record_type", None)
            table = R.RECORD_TYPE_TABLE.get(rtype) if rtype else None
            if table in want:
                out.refused.setdefault(table, []).append(str(getattr(res, "path", "") or f"{file_uri}#{pr.line}"))
        for pi in part.items:
            if pi.table not in want:
                continue
            if revision_id is not None and pi.source_revision_id != revision_id:
                continue
            row = tuple(pi.row) + ITEM_PLANS[pi.table].tail(ingest_seq)
            k = _key(pi.table, row)
            w, s = order_pos[pi.table]
            have = best[pi.table].get(k)
            if have is None or (row[w], row[s]) > (have[w], have[s]):
                best[pi.table][k] = row
        if part.last:
            out.file_sha = part.file_sha
    for t in tables:
        out.rows[t] = normalize_rows(t, list(best[t].values()))
    return out


def reparse(root: str, file_uri: str, table: str, revision_id: Optional[str] = None,
            ingest_seq: int = 0) -> Tuple[Digest, Iterator[Tuple]]:
    """``c``'s ``table`` rows as the staging path builds them under the CURRENT code, reduced per key by the item guard (the greatest
    ``(written_utc, record_sha256)`` wins). Returns ``(D_fold with count, iterator over the full row tuples)``; rows carry
    ``ingest_seq`` = ``ingest_seq``. ``revision_id`` keeps only rows tagged with it (a plain file may carry several). Raises
    :class:`RehydrationIncomplete` when the file is missing, unreadable, changed while read, or holds a record of ``table`` the
    validator refuses."""
    rp = reparse_file(root, file_uri, revision_id, (table,), ingest_seq)
    reason = _reparse_failure(rp, table)
    if reason:
        raise RehydrationIncomplete(revision_id or "", table, file_uri, 0, reason)
    return rp.fold(table), iter(rp.rows[table])


def _reparse_failure(rp: _Reparse, table: str) -> str:
    if rp.error:
        return rp.error
    if rp.retry:
        return "the record file changed while it was read"
    if rp.refused.get(table):
        return f"the current validator refuses {rp.refused[table][0]}" + (f" (+{len(rp.refused[table]) - 1} more)" if len(rp.refused[table]) > 1 else "")
    return ""


def _ledger_row(r: Sequence[object]) -> LedgerRow:
    return LedgerRow(str(r[0]), str(r[1]), int(r[2]), str(r[3]), int(r[4]), str(r[5]), int(r[6]), str(r[7]), str(r[8]), str(r[9]))


def ledger_carriers(conn: sqlite3.Connection, revision_ids: Iterable[str]) -> Dict[str, List[LedgerRow]]:
    """``{revision_id: [LedgerRow]}`` for the revisions among ``revision_ids`` that hold ledger rows (one indexed probe each)."""
    out: Dict[str, List[LedgerRow]] = {}
    sql = f"SELECT {', '.join(LEDGER_COLUMNS)} FROM {LEDGER_TABLE} WHERE table_name IN ({_TABLES_IN}) AND revision_id = ? ORDER BY table_name, record_file_id"
    for rev in dict.fromkeys(str(r) for r in revision_ids):
        rows = [_ledger_row(r) for r in conn.execute(sql, (rev,)).fetchall()]
        if rows:
            out[rev] = rows
    return out


def has_ledger_rows(conn: sqlite3.Connection, revision_id: str) -> bool:
    """Whether ``revision_id`` has any ledger row (the write points' one probe)."""
    return conn.execute(f"SELECT 1 FROM {LEDGER_TABLE} WHERE table_name IN ({_TABLES_IN}) AND revision_id = ? LIMIT 1",
                        (revision_id,)).fetchone() is not None


def present_rows(conn: sqlite3.Connection, table: str, revision_id: str, record_file_id: int) -> List[Tuple]:
    """Every present row of ``(revision_id, record_file_id)`` in ``table``, all columns in table order (through ``idx_<t>_revision``)."""
    cols = ", ".join(columns_of(table))
    return [tuple(r) for r in conn.execute(f"SELECT {cols} FROM {table} WHERE source_revision_id = ? AND record_file_id = ?",
                                           (revision_id, int(record_file_id))).fetchall()]


def staged_files(conn: sqlite3.Connection, revision_id: str) -> Set[str]:
    """The files a held bundle (or a revision's own pending staging) staged under ``revision_id``: their rows are protected."""
    return {str(r[0]) for r in conn.execute(
        "SELECT file_uri FROM revision_staging_file WHERE revision_id = ? UNION SELECT file_uri FROM revision_staging_record WHERE revision_id = ?",
        (revision_id, revision_id)).fetchall()}


def carriers_of(conn: sqlite3.Connection, revision_id: str) -> Dict[str, List[int]]:
    """``{table: [record_file_id]}`` of ``revision_id``'s present rows (a skip-scan of each ``idx_<t>_revision``)."""
    return {t: [int(r[0]) for r in conn.execute(f"SELECT DISTINCT record_file_id FROM {t} WHERE source_revision_id = ? ORDER BY record_file_id",
                                                 (revision_id,)).fetchall()]
            for t in PRUNE_TABLES}


def is_accepted_superseded(conn: sqlite3.Connection, revision_id: str) -> bool:
    return conn.execute(f"SELECT 1 FROM source_revision WHERE revision_id = ? AND {accepted_superseded()}",
                        (revision_id,)).fetchone() is not None


def _stamp() -> str:
    from . import harvest as H
    return H.validator_stamp()


def revision_plan(conn: sqlite3.Connection, root: str, revision_id: str, mode: ReclaimMode) -> List[CarrierPlan]:
    """Classify every carrier of ``revision_id`` in every prune table (read-only; call outside any write transaction). A revision
    that is not accepted and superseded has no carrier plan (``[]``): its rows are never deletable. For each candidate, ``reparse`` is
    computed once per file (or the ledger's ``expected_*`` reused when ``parsed_with`` equals the running code's stamp) and checked
    against present + deleted. Every carrier is classified by the normal rule first, so a staged, missing or unauthentic carrier is
    protected in every mode; only in cohort mode is a carrier the normal rule finds deletable then also required to be bound to
    ``mode``'s archive with matching counts, ``digest_bind`` and current ranks, and one that is not is ``cohort_uncovered`` (it waits
    for a delta archive or ``mode normal``)."""
    if not is_accepted_superseded(conn, revision_id):
        return []
    stamp = _stamp()
    staged = staged_files(conn, revision_id)
    ledger = {(l.table_name, l.record_file_id): l for l in ledger_carriers(conn, [revision_id]).get(revision_id, [])}
    plans: List[CarrierPlan] = []
    todo: List[Tuple[CarrierPlan, List[Tuple]]] = []
    for table, rfids in carriers_of(conn, revision_id).items():
        for rfid in rfids:
            led = ledger.get((table, rfid))
            ri = conn.execute("SELECT file_uri, ingest_seq FROM record_ingest WHERE record_file_id = ?", (rfid,)).fetchone()
            uri = str(ri[0]) if ri else (led.file_uri if led else "")
            rows = present_rows(conn, table, revision_id, rfid)
            p = CarrierPlan(table, revision_id, rfid, uri, MISSING, len(rows), Digest(), ledger=led,
                            ingest_seq=int(ri[1]) if ri else None)
            plans.append(p)
            if ri is None:
                p.detail = "the carrier has no record_ingest row"
                continue
            if uri in staged:
                p.klass, p.detail = STAGED, "staged by a held bundle"
                continue
            p.present_fold = Digest.of_rows(table, rows, FOLD_EXCLUDE)
            p.present_bind = Digest.of_rows(table, rows, BIND_EXCLUDE)
            todo.append((p, rows))
    # one parse per FILE for every table it carries (a file may carry item_result and item_flags rows alike)
    by_file: Dict[str, List[Tuple[CarrierPlan, List[Tuple]]]] = {}
    for p, rows in todo:
        if p.ledger is not None and p.ledger.parsed_with == stamp:
            # a carrier an earlier pass of this code authenticated and began deleting: its expected rows are the ledger's, so no
            # re-parse -- but the file must still be the one that was parsed (on disk, its logical sha256 the ingested one)
            p.expected = p.ledger.expected
            why = file_unchanged(conn, root, p.file_uri)
            if why:
                p.klass, p.detail = MISSING, why
                continue
            _judge(p, rows, mode)
            continue
        by_file.setdefault(p.file_uri, []).append((p, rows))
    for uri, group in by_file.items():
        tables = sorted({p.table for p, _ in group})
        rp = reparse_file(root, uri, revision_id, tables)
        want_sha = conn.execute("SELECT file_sha256 FROM record_ingest WHERE file_uri = ?", (uri,)).fetchone()
        for p, rows in group:
            if rp.error or rp.retry:
                p.klass, p.detail, p.transient = MISSING, rp.error or "the record file changed while it was read", rp.retry and not rp.error
                continue
            if want_sha is None or rp.file_sha != str(want_sha[0]):
                p.klass, p.detail = MISSING, f"the record file's sha256 {rp.file_sha[:16]} is not the ingested {str(want_sha[0])[:16] if want_sha else '?'}"
                continue
            p.expected = rp.fold(p.table)
            if rp.refused.get(p.table):
                # the records the validator accepts may reproduce the rows exactly, but rehydration re-parses the whole file and refuses it
                # (``authenticate``): a carrier it could never restore is never deleted
                p.klass, p.detail = UNAUTHENTIC_REFUSED, _reparse_failure(rp, p.table)
                continue
            _judge(p, rows, mode)
    return plans


def file_unchanged(conn: sqlite3.Connection, root: str, file_uri: str) -> str:
    """'' when the record file ``file_uri`` is on disk and its logical sha256 (the decompressed bytes, as the fold hashed them) is
    ``record_ingest.file_sha256``; else why not (a carrier whose file fails this is ``missing``: nothing more of it is deleted)."""
    from . import harvest as H
    path = H.resolve_uri(root, file_uri)
    if path is None:
        return "the record file is not on disk"
    want = conn.execute("SELECT file_sha256 FROM record_ingest WHERE file_uri = ?", (file_uri,)).fetchone()
    try:
        have = R.file_logical_sha(path)
    except (OSError, EOFError, ValueError, zlib.error) as exc:      # gzip.BadGzipFile is an OSError
        return f"the record file is unreadable ({exc})"
    if want is None or have != str(want[0]):
        return f"the record file's sha256 {have[:16]} is not the ingested {str(want[0])[:16] if want else '?'}"
    return ""


def _authentic(expected: Digest, present_fold: Digest, deleted_fold: Digest) -> Optional[Tuple[str, str]]:
    """``authentic``: the re-parse's ``D_fold`` (with count) equals present + deleted. None when it does, else ``(class, reason)``:
    :data:`UNAUTHENTIC_COUNT` or :data:`UNAUTHENTIC_DIGEST`."""
    if expected.count != present_fold.count + deleted_fold.count:
        return UNAUTHENTIC_COUNT, f"re-parse {expected.count} rows != present {present_fold.count} + deleted {deleted_fold.count}"
    if expected.sum256 != (present_fold + deleted_fold).sum256:
        return UNAUTHENTIC_DIGEST, "digest mismatch: the current code does not reproduce the stored rows (D_fold differs)"
    return None


def _judge(p: CarrierPlan, rows: Sequence[Tuple], mode: ReclaimMode) -> None:
    """``authentic`` against the expected digest and the ledger's running totals; then, in cohort mode only, the archive binding of a
    carrier the normal rule found deletable (a protected carrier stays protected whatever the archive holds)."""
    deleted_fold = p.ledger.deleted_fold if p.ledger else Digest()
    deleted_bind = p.ledger.deleted_bind if p.ledger else Digest()
    failed = _authentic(p.expected, p.present_fold, deleted_fold)
    if failed:
        p.klass = failed[0]
        p.detail = p.detail or failed[1]
        return
    if mode.kind == "cohort":
        g = mode.binding(p.table, p.revision_id, p.record_file_id)
        if g is None:
            p.klass, p.detail = COHORT_UNCOVERED, "no verified archive holds this group"
            return
        deleted_n = p.ledger.rows_deleted if p.ledger else 0
        if p.rows + deleted_n != g.rows:
            p.klass, p.detail = COHORT_UNCOVERED, f"present {p.rows} + deleted {deleted_n} rows != archived {g.rows}"
            return
        if Digest.from_hex(g.rows, g.digest_bind).sum256 != (p.present_bind + deleted_bind).sum256:
            p.klass, p.detail = COHORT_UNCOVERED, "present + deleted rows differ from the archived group (digest_bind)"
            return
        seq_pos = columns_of(p.table).index("ingest_seq")
        if any(r[seq_pos] != p.ingest_seq for r in rows):
            p.klass, p.detail = COHORT_UNCOVERED, "a present row's ingest_seq is not the carrier's current rank"
            return
    p.klass = DELETABLE


def plan_complete(plans: Sequence[CarrierPlan]) -> bool:
    """Whether a revision whose deletable carriers are all emptied may be recorded ``reclaimed = 1``: no carrier waits for a cohort
    archive, none is ``missing`` and no classification was transient. A missing carrier (its log file absent, unreadable, or not the
    bytes the fold ingested) keeps the revision due: the file returning is no item write, so nothing else would plan the revision again,
    and its rows -- which a rebuild from the restored log deletes -- would stay for good."""
    return not any(p.klass in (COHORT_UNCOVERED, MISSING) or p.transient for p in plans)


# --------------------------------------------------------------------------------------------
# deletion (one bounded batch, inside the caller's transaction)
# --------------------------------------------------------------------------------------------

def reset_reclaimed(conn: sqlite3.Connection, revision_id: str) -> None:
    """``reclaimed = 0`` on ``revision_id`` (inside the caller's transaction): an item write under it, or rows restored to it, means the
    next reclaim pass plans it again."""
    conn.execute("UPDATE source_revision SET reclaimed = 0 WHERE revision_id = ? AND reclaimed <> 0", (revision_id,))


def drop_ledger_row(conn: sqlite3.Connection, table: str, revision_id: str, record_file_id: int) -> None:
    """Remove one carrier's ledger row (inside the caller's transaction): nothing of it is deleted any more."""
    conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE table_name = ? AND revision_id = ? AND record_file_id = ?",
                 (table, revision_id, int(record_file_id)))


def ledger_heads(conn: sqlite3.Connection, source_ids: Optional[Iterable[str]] = None) -> List[str]:
    """The accepted HEAD revisions (not superseded) that hold ledger rows -- visible with rows reclaim deleted, which rehydration
    restores before anything commits -- of the sources ``source_ids`` (every source when None). One indexed join over the small ledger."""
    if source_ids is None:
        return [str(r[0]) for r in conn.execute(
            f"SELECT DISTINCT l.revision_id FROM {LEDGER_TABLE} l JOIN source_revision s ON s.revision_id = l.revision_id "
            "WHERE s.accepted = 1 AND s.superseded_by_revision_id IS NULL ORDER BY l.revision_id").fetchall()]
    # a few sources (one record's fold): their heads by idx_source_revision_source, each probed in the ledger by its primary key
    params = sorted({str(s) for s in source_ids})
    if not params:
        return []
    return [str(r[0]) for r in conn.execute(
        f"SELECT s.revision_id FROM source_revision s WHERE s.source_id IN ({', '.join('?' for _ in params)}) AND s.accepted = 1 "
        f"AND s.superseded_by_revision_id IS NULL AND EXISTS (SELECT 1 FROM {LEDGER_TABLE} l WHERE l.table_name IN ({_TABLES_IN}) "
        "AND l.revision_id = s.revision_id) ORDER BY s.revision_id", params).fetchall()]


def ledger_totals(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    """``{table: {"carriers", "rows_deleted"}}`` of the ledger, every prune table present (zeros when it holds nothing)."""
    out = {t: {"carriers": 0, "rows_deleted": 0} for t in PRUNE_TABLES}
    for t, n, rows in conn.execute(f"SELECT table_name, COUNT(*), COALESCE(SUM(rows_deleted), 0) FROM {LEDGER_TABLE} GROUP BY table_name"):
        if t in out:
            out[t] = {"carriers": int(n), "rows_deleted": int(rows)}
    return out


def _write_ledger(conn: sqlite3.Connection, l: LedgerRow) -> None:
    if l.rows_deleted <= 0:
        drop_ledger_row(conn, l.table_name, l.revision_id, l.record_file_id)
        return
    conn.execute(
        f"INSERT INTO {LEDGER_TABLE}({', '.join(LEDGER_COLUMNS)}) VALUES ({', '.join('?' for _ in LEDGER_COLUMNS)}) "
        "ON CONFLICT(table_name, revision_id, record_file_id) DO UPDATE SET file_uri = excluded.file_uri, expected_rows = excluded.expected_rows, "
        "expected_digest_fold = excluded.expected_digest_fold, rows_deleted = excluded.rows_deleted, deleted_digest_fold = excluded.deleted_digest_fold, "
        "deleted_digest_bind = excluded.deleted_digest_bind, parsed_with = excluded.parsed_with",
        (l.table_name, l.revision_id, l.record_file_id, l.file_uri, l.expected_rows, l.expected_digest_fold, l.rows_deleted,
         l.deleted_digest_fold, l.deleted_digest_bind, l.parsed_with))


def read_ledger(conn: sqlite3.Connection, table: str, revision_id: str, record_file_id: int) -> Optional[LedgerRow]:
    """One carrier's ledger row, or None."""
    r = conn.execute(f"SELECT {', '.join(LEDGER_COLUMNS)} FROM {LEDGER_TABLE} WHERE table_name = ? AND revision_id = ? AND record_file_id = ?",
                     (table, revision_id, int(record_file_id))).fetchone()
    return _ledger_row(r) if r else None


def delete_batch(conn: sqlite3.Connection, plan: CarrierPlan, limit: int, stamp: Optional[str] = None) -> int:
    """Delete the next ``limit`` rows of a DELETABLE carrier and account for them in the ledger -- INSIDE the caller's transaction. Selects
    the rows (all columns), digests them (``D_fold``, ``D_bind``), deletes exactly those keys, each only while the revision is still
    accepted and superseded, and requires the delete count to equal the selected count (else :class:`BatchRefused`, and the caller's
    transaction rolls back). The ledger row is created on the first batch (``expected_*`` from the plan) and its running totals advanced
    on every batch, so present + deleted always equals the authenticated original. Returns the rows deleted (0 = the carrier is empty)."""
    t, rev, rfid = plan.table, plan.revision_id, int(plan.record_file_id)
    cols = columns_of(t)
    pk = PK[t]
    rows = [tuple(r) for r in conn.execute(f"SELECT {', '.join(cols)} FROM {t} WHERE source_revision_id = ? AND record_file_id = ? LIMIT ?",
                                           (rev, rfid, int(limit))).fetchall()]
    if not rows:
        return 0
    fold = Digest.of_rows(t, rows, FOLD_EXCLUDE)
    bind = Digest.of_rows(t, rows, BIND_EXCLUDE)
    pos = [cols.index(c) for c in pk]
    where = " AND ".join(f"{c} = ?" for c in pk)
    sql = f"DELETE FROM {t} WHERE {where} AND {DELETE_GUARD}"
    before = conn.total_changes
    conn.executemany(sql, [tuple(r[i] for i in pos) + (rev,) for r in rows])
    n = conn.total_changes - before
    if n != len(rows):
        raise BatchRefused(f"{t} {rev} carrier {rfid}: selected {len(rows)} rows, deleted {n}; nothing deleted")
    led = read_ledger(conn, t, rev, rfid)
    if led is None:
        led = LedgerRow(t, rev, rfid, plan.file_uri, plan.expected.count, plan.expected.hex(), 0, Digest().hex(), Digest().hex(),
                        stamp or _stamp())
    led.rows_deleted += n
    led.deleted_digest_fold = (led.deleted_fold + fold).hex()
    led.deleted_digest_bind = (Digest.from_hex(0, led.deleted_digest_bind) + bind).hex()
    led.parsed_with = stamp or _stamp()
    _write_ledger(conn, led)
    return n


def confirm_reclaimed(conn: sqlite3.Connection, revision_id: str, plans: Sequence[CarrierPlan]) -> bool:
    """The revision's last step (inside the caller's transaction): re-read its carriers and staging ledger, confirm the plan still
    describes them and no deletable carrier has a row left, then record ``reclaimed = 1``. Returns whether it did."""
    if not plan_complete(plans) or not is_accepted_superseded(conn, revision_id):
        return False
    planned = {(p.table, p.record_file_id) for p in plans}
    now_carriers = {(t, rfid) for t, rfids in carriers_of(conn, revision_id).items() for rfid in rfids}
    left = {(p.table, p.record_file_id) for p in plans if p.klass != DELETABLE}
    if now_carriers != left or not now_carriers <= planned:
        return False
    staged = staged_files(conn, revision_id)
    if any(p.file_uri in staged for p in plans if p.klass == DELETABLE):
        return False
    conn.execute("UPDATE source_revision SET reclaimed = 1 WHERE revision_id = ? AND reclaimed <> 1", (revision_id,))
    return True


# --------------------------------------------------------------------------------------------
# rehydration (rehydrate before write)
# --------------------------------------------------------------------------------------------

@dataclass
class _Restore:
    ledger: LedgerRow
    rows: List[Tuple]


def authenticate(conn: sqlite3.Connection, root: str, revision_id: str, seq_of=None) -> List[_Restore]:
    """Authenticate every ledger carrier of ``revision_id`` against the current code's re-parse (count and ``D_fold`` of re-parse ==
    present + deleted) and return what to restore -- the re-parsed rows whose key is not present, ``ingest_seq`` the carrier's current
    rank. Raises :class:`RehydrationIncomplete` on the first carrier that does not authenticate; writes nothing."""
    out: List[_Restore] = []
    for led in ledger_carriers(conn, [revision_id]).get(revision_id, []):
        t = led.table_name
        seq = seq_of(led.file_uri) if seq_of is not None else None
        if seq is None:
            r = conn.execute("SELECT ingest_seq FROM record_ingest WHERE file_uri = ?", (led.file_uri,)).fetchone()
            seq = int(r[0]) if r else None
        if seq is None:
            raise RehydrationIncomplete(revision_id, t, led.file_uri, led.rows_deleted, "the carrier has no record_ingest row")
        rp = reparse_file(root, led.file_uri, revision_id, (t,), seq)
        reason = _reparse_failure(rp, t)
        want_sha = conn.execute("SELECT file_sha256 FROM record_ingest WHERE file_uri = ?", (led.file_uri,)).fetchone()
        if not reason and (want_sha is None or rp.file_sha != str(want_sha[0])):
            reason = "the record file's bytes are not the ingested ones"
        present = present_rows(conn, t, revision_id, led.record_file_id)
        if not reason:
            failed = _authentic(rp.fold(t), Digest.of_rows(t, present, FOLD_EXCLUDE), led.deleted_fold)
            reason = failed[1] if failed else ""
        if reason:
            raise RehydrationIncomplete(revision_id, t, led.file_uri, led.rows_deleted, reason)
        keys = {_key(t, r) for r in present}
        restore = [r for r in rp.rows[t] if _key(t, r) not in keys]
        if len(restore) != led.rows_deleted:
            raise RehydrationIncomplete(revision_id, t, led.file_uri, led.rows_deleted,
                                        f"{len(restore)} re-parsed keys are absent, the ledger deleted {led.rows_deleted}")
        out.append(_Restore(led, restore))
    return out


#: Rows per INSERT statement of a restoration, at most (:func:`_insert_rows`). The item tables' RAISE(ABORT) triggers give every
#: statement a statement journal, and inside an outer savepoint on a temp_store=MEMORY connection -- the harvester's record savepoint,
#: where a chain move rehydrates a new head -- each statement's release walks the whole in-memory sub-journal the savepoint has built,
#: so one statement per row cost time quadratic in the rows (on SQLite 3.45.1, a 24-column table at 8k rows: 366 us/row one statement
#: per row, 12-14 us/row at 16 to 1,024 rows per statement, 17 at 4,096). 256 rows of the widest table (32 columns) bind 8,192
#: variables, under SQLite's default limit of 32,766; a build with a lower limit gets fewer rows per statement.
RESTORE_ROWS_PER_STATEMENT = 256


def _statement_sizes(n: int, per: int) -> List[int]:
    """Rows per statement for ``n`` rows, at most ``per`` to a statement: statements of ``p``, the largest power of two <= ``per``, then
    the remainder by its binary digits -- at most ceil(n / p) + log2(p) statements, every size a power of two. The connection's statement
    cache (128 entries) keeps each distinct statement prepared, about 2 MB at 450 rows x 32 columns, so a remainder of every size would
    fill it with hundreds of MB on the harvester's long-lived connection; powers of two bound it to log2(p) + 1 shapes per table."""
    p = 1 << (max(int(per), 1).bit_length() - 1)
    out = [p] * (n // p)
    r = n % p
    while r:
        b = 1 << (r.bit_length() - 1)
        out.append(b)
        r -= b
    return out


def _insert_rows(conn: sqlite3.Connection, upsert_sql: str, rows: Sequence[Tuple]) -> None:
    """Run the one-row ``upsert_sql`` (:func:`db.upsert_sql`'s ``VALUES (?, ...)`` form) for every row of ``rows``, in order, as multi-row
    ``VALUES (...), (...)`` statements of :func:`_statement_sizes` rows, each binding at most SQLite's variable limit. SQLite still
    inserts the rows of one statement one at a time, in order: each meets the triggers and the ON CONFLICT clause (an earlier row of the
    same statement included) as it would in a statement of its own, so the table ends the same. A row that fails aborts its statement
    and raises to the caller, whose savepoint or transaction rolls the restoration back, as one statement per row did."""
    width = upsert_sql.count("?")
    one = "(" + ", ".join("?" * width) + ")"
    values = "VALUES " + one
    if width == 0 or upsert_sql.count(values) != 1:
        raise ValueError(f"not a one-row VALUES upsert: {upsert_sql[:200]}")
    bad = next((r for r in rows if len(r) != width), None)
    if bad is not None:
        raise ValueError(f"a row of {len(bad)} values for an upsert of {width}")
    head, _, tail = upsert_sql.partition(values)
    per = min(RESTORE_ROWS_PER_STATEMENT, conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) // width)
    i = 0
    for n in _statement_sizes(len(rows), per):
        conn.execute(head + "VALUES " + ", ".join([one] * n) + tail, [v for r in rows[i:i + n] for v in r])
        i += n


def restore_chunk(conn: sqlite3.Connection, upsert_sql: str, led: LedgerRow, rows: Sequence[Tuple]) -> int:
    """Write ``rows`` (part of one ledger carrier's restoration) and take them off the ledger row -- INSIDE the caller's transaction, so
    present + ledger stays the authenticated original at every commit. Resets the revision's ``reclaimed``. Returns the rows written.
    The rows go in as multi-row statements (:func:`_insert_rows`): the caller may hold an outer savepoint on a temp_store=MEMORY
    connection (the harvester's record savepoint), where one statement per row cost time quadratic in the rows."""
    if not rows:
        return 0
    t = led.table_name
    _insert_rows(conn, upsert_sql, rows)
    cur = read_ledger(conn, t, led.revision_id, led.record_file_id)
    if cur is not None:
        cur.rows_deleted -= len(rows)
        cur.deleted_digest_fold = cur.deleted_fold.sub(Digest.of_rows(t, rows, FOLD_EXCLUDE)).hex()
        cur.deleted_digest_bind = Digest.from_hex(0, cur.deleted_digest_bind).sub(Digest.of_rows(t, rows, BIND_EXCLUDE)).hex()
        _write_ledger(conn, cur)
    reset_reclaimed(conn, led.revision_id)
    return len(rows)


def rehydrate(ingestor, revision_id: str, now: str) -> int:
    """Re-fold every ledger carrier of ``revision_id`` from the record log, AFTER authenticating each (re-parse = present + deleted on
    count and ``D_fold``); returns the rows restored. On any mismatch nothing is written and :class:`RehydrationIncomplete` is raised
    (fails closed). The rows go through the staging path's own upsert (``Reducer.item_upsert_sql``) with ``ingest_seq`` the carrier's
    current rank; each bounded transaction also takes its rows off the ledger row and resets ``reclaimed`` to 0, so a stop leaves
    present + ledger whole. Called inside an open transaction (``flush_items``' write point), it writes inline."""
    from . import harvest as H
    conn = ingestor.conn
    restores = authenticate(conn, ingestor.root, revision_id, ingestor.ranker.seq_of)
    total = 0
    for rs in restores:
        sql = ingestor.reducer.item_upsert_sql(rs.ledger.table_name)
        for chunk in (rs.rows[i:i + H.ITEM_ROWS_PER_TRANSACTION] for i in range(0, len(rs.rows), H.ITEM_ROWS_PER_TRANSACTION)):
            if conn.in_transaction:
                total += restore_chunk(conn, sql, rs.ledger, chunk)
            else:
                n = [0]

                def body(chunk=chunk, rs=rs, sql=sql) -> None:
                    n[0] = restore_chunk(conn, sql, rs.ledger, chunk)
                ingestor._timed_transaction(body, 1, len(chunk), now, kind="rehydrate")
                total += n[0]
        if not rs.rows:
            # a ledger row with nothing to restore (an emptied carrier fully re-folded by an earlier stop): clear it
            def clear(rs=rs) -> None:
                _write_ledger(conn, dataclasses.replace(rs.ledger, rows_deleted=0))
            if conn.in_transaction:
                clear()
            else:
                ingestor._timed_transaction(clear, 0, 0, now, kind="rehydrate")
    return total


# --------------------------------------------------------------------------------------------
# census
# --------------------------------------------------------------------------------------------

def census(conn: sqlite3.Connection, root: str, revisions: Optional[Sequence[str]] = None,
           mode: Optional[ReclaimMode] = None) -> Dict[str, object]:
    """Rows and carriers by table x class (x adapter) over every accepted, superseded revision (or ``revisions``), plus the ledger's
    totals; read-only (re-parses every candidate carrier). The ``plan`` / ``status`` actions and the report of what stays after the deletion read it."""
    mode = mode or ReclaimMode.normal()
    if revisions is None:
        revisions = [str(r[0]) for r in conn.execute(
            f"SELECT revision_id FROM source_revision WHERE {accepted_superseded()} ORDER BY observed_utc, revision_id")]
    adapter = dict(conn.execute("SELECT revision_id, adapter FROM source_revision").fetchall())
    by_class: Dict[str, Dict[str, Dict[str, int]]] = {t: {k: {"rows": 0, "carriers": 0} for k in CLASSES} for t in PRUNE_TABLES}
    by_adapter: Dict[str, Dict[str, Dict[str, int]]] = {}
    sub: Dict[str, Dict[str, int]] = {t: {} for t in PRUNE_TABLES}
    for rev in revisions:
        for p in revision_plan(conn, root, rev, mode):
            c = by_class[p.table][p.klass]
            c["rows"] += p.rows
            c["carriers"] += 1
            a = by_adapter.setdefault(str(adapter.get(rev, "?")), {}).setdefault(p.table, {})
            a[p.klass] = a.get(p.klass, 0) + p.rows
            if p.detail and p.klass != DELETABLE:
                sub[p.table][p.detail.split(":")[0][:80]] = sub[p.table].get(p.detail.split(":")[0][:80], 0) + p.rows
    return {"rule": RULE_VERSION, "mode": mode.kind, "revisions": len(revisions), "by_table_class": by_class,
            "by_adapter": by_adapter, "details": sub, "ledger": ledger_totals(conn)}


# --------------------------------------------------------------------------------------------
# the metrics log's reclaim lines (read by the status page and ``registry prune-superseded status``)
# --------------------------------------------------------------------------------------------

#: The tail of ``state/metrics.jsonl`` the readers parse, in bytes: the recent cycles' lines, never the whole log.
METRICS_TAIL_BYTES = 4 * 1024 * 1024


def metrics_tail(root: str, max_bytes: int = METRICS_TAIL_BYTES) -> List[Dict[str, object]]:
    """The JSON objects of the last ``max_bytes`` of ``<root>/state/metrics.jsonl`` (the first, partial line dropped), oldest first; ``[]``
    when there is no log. A line that does not parse is skipped."""
    path = os.path.join(root, "state", "metrics.jsonl")
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            start = max(0, fh.tell() - int(max_bytes))
            fh.seek(start)
            data = fh.read()
    except OSError:
        return []
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]
    out: List[Dict[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def is_deletion_batch(m: Mapping[str, object]) -> bool:
    """A reclaim DELETE batch line (``kind = "reclaim"`` with the carrier's ``table``; its ``rows`` the rows deleted). Pass (i)'s un-mark
    and the confirming transaction log the same kind without one."""
    return m.get("kind") == "reclaim" and bool(m.get("table"))


def is_rehydration(m: Mapping[str, object]) -> bool:
    """A revision's rehydration line (``kind = "rehydrate"`` with its ``revision``; its ``rows`` the rows restored). The bounded
    transactions one rehydration writes log the same kind without one, each with its own share of those rows."""
    return m.get("kind") == "rehydrate" and bool(m.get("revision"))
