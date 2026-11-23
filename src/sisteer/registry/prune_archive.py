"""The superseded-revision pruning's tooling: the archive of every row to be deleted, its
independent verification, the restore, the database digests the comparison contracts read, the reclaim mode, the census, and the
operator remedies for a failed rehydration -- the ``registry prune-superseded <action>`` subcommands.

**The archive** (:func:`archive`). One read transaction over the database exports every DELETABLE carrier group (``prune-rule/3``,
:func:`prune.revision_plan` in normal mode) as it stands in that snapshot: one ``<table>.jsonl.zst`` per item table (one canonical row
per line, :func:`prune.encode_row`, ordered by ``(source_revision_id, record_file_id, <primary key>)``; zstd level 9 with frame
checksums, so a flipped byte fails decompression), ``groups.tsv.zst`` (:data:`prune.GROUPS_TSV_COLUMNS`, one line per exported group:
its rows, ``digest_all`` and ``digest_bind`` as 64-hex additive sums), ``MANIFEST.json`` (format, rule, the tool's tree, the command
line, the source's identity, per table its columns, rows, digests and file sha256, the census of every carrier by class) and
``SHA256SUMS`` (``sha256sum -c`` format). ``groups.tsv.zst`` lists ONLY the exported groups: the cohort binding
(:meth:`prune.ReclaimMode.binding`) admits any group it lists, so a listed group whose rows were not exported could be deleted
unarchived. A deletable group the ledger already holds deleted rows of is not exported either (its deleted part is in no file this
archive writes) and is named in the MANIFEST. The archive is written beside its destination and renamed into place whole.

**Verification** (:func:`verify_archive`) re-reads every file: SHA256SUMS, every row decoded and digested, every group and table total
recomputed against ``groups.tsv.zst`` and the MANIFEST, the order checked; with a database, the database's deletable groups in one
snapshot must equal the archive's. Only a clean pass writes ``VERIFIED.json`` (the cluster's independent job) or ``VERIFIED.local.json``
(the ``mode cohort`` gate's local re-verification, :func:`gate_local_archive`), naming the MANIFEST's sha256.

**Restore** (:func:`restore`) inserts every archived row whose key is absent, its ``ingest_seq`` the carrier's CURRENT rank (a remap
between archive and restore moves it; every other column is the archived byte), group by group in one transaction each, a group's rows
digested against ``groups.tsv.zst`` before any is written. A key already present is never overwritten: identical (every column but
``ingest_seq``) it is counted ``present_identical``, different a ``conflict``. A ledger row whose group is whole again is removed, and a
touched revision's ``reclaimed`` reset. Its write connection keeps SQLite's statement journal in a temp file (:data:`RESTORE_TEMP_STORE`),
so a group costs time linear in its rows; a live restore refuses when SQLite's temp directory lacks room for it.

**Digests** (:func:`digest_database`): ``(rows, sum256)`` per table and per ``v_*`` object, each by its own read-only connection (one
process per worker), plus the extras the contracts need (``schema_migration`` rows, ``source_revision`` without ``reclaimed``, the ledger's
group list) and the schema (``sqlite_master``). A table's digest is the archive's ``digest_all`` formula, so ``before = after + archive``
per item table (``equality.check_contract``). An object column computed from SQLite's clock (:data:`CLOCK_DERIVED_COLUMNS`: an age,
``julianday('now')`` minus a timestamp of the same row) is left out of its object's digest and the timestamp it is computed from is
compared exactly, so a digest is a function of the database, never of when it was taken; the document names what it left out
(``projection``).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from . import db as DB
from . import equality as EQ
from . import prune as P
from . import records as R
from . import schema_def as S

FORMAT = "prune-archive/1"
VERIFIED_FORMAT = "prune-archive-verified/1"
DIGEST_FORMAT = "registry-digest/1"
GROUPS_FILTER_FORMAT = "prune-groups/1"
SUMS_FILE = "SHA256SUMS"
ZSTD_LEVEL = 9
#: The operator remedies' decision log, under ``<root>/state/``: one JSON line per ``rehydrate`` (:func:`_log_decision`)
DECISIONS_FILE = "prune_decisions.jsonl"
#: the actions that write under the registry root (the CLI stamps ``state/last_run`` only for these)
ROOT_WRITING_ACTIONS: Tuple[str, ...] = ("mode", "rehydrate")
#: The ``v_*`` columns SQLite computes from its clock (``julianday('now')``), which no two reads of the same database agree on. An object's
#: digest leaves them out; the timestamp each is computed from is a column of the same object and is compared exactly. (SQLite's 'now' is
#: the statement's wall clock and cannot be pinned, so the digest projects the column out rather than fixing an evaluation time.)
CLOCK_DERIVED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "v_in_flight": ("age_s",),                  # COALESCE(last_event_utc, queued_utc) to now
    "v_pending_judgment": ("claim_age_s",),     # claim_utc to now
}

GroupKey = Tuple[str, str, int]


class ArchiveError(RuntimeError):
    """The action was refused before it wrote anything."""


class ArchiveInvalid(ArchiveError):
    """An archive failed verification; ``failures`` names every check that failed."""

    def __init__(self, archive_dir: str, failures: Sequence[str]):
        self.failures = list(failures)
        head = "; ".join(self.failures[:5]) + (f" (+{len(self.failures) - 5} more)" if len(self.failures) > 5 else "")
        super().__init__(f"archive {archive_dir} does not verify: {head}")


class LiveRefused(ArchiveError):
    """The action names the live registry's database, which it may not write (or may only under the stated conditions)."""


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


def row_file(table: str) -> str:
    return f"{table}.jsonl.zst"


#: sha256 of a file's bytes, streamed.
sha256_file = R.file_sha256


def digest_entry(d: P.Digest) -> Dict[str, object]:
    return {"rows": int(d.count), "sum256": d.hex()}


#: ``prune.row_digest``'s formula over every value of a row under a name: for a table whose columns are the code's, the same integer as
#: ``prune.row_digest(name, row)``; for a ``v_*`` object, its own row digest.
generic_row_digest = P.tagged_row_digest


def _digest_with(name: str, cols: Sequence[str], exclude: Sequence[str]) -> Callable[[Sequence[object]], int]:
    """A row digest over ``cols`` minus ``exclude`` (the archive's own column list, so an archive verifies under any later code)."""
    drop = set(exclude)
    keep = [i for i, c in enumerate(cols) if c not in drop]
    if len(keep) == len(cols):
        return lambda row: generic_row_digest(name, row)
    return lambda row: generic_row_digest(name, [row[i] for i in keep])


def _pk_positions(table: str, cols: Sequence[str]) -> List[int]:
    return [list(cols).index(c) for c in S.TABLE_BY_NAME[table].pk]


def _order(row: Sequence[object], pos: Sequence[int]) -> tuple:
    return tuple(EQ.order_key(row[i]) for i in pos)


def _write_json(path: str, obj: object) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _read_json(path: str) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def tool_tree_sha() -> Dict[str, object]:
    """The code tree this tool runs from: its git HEAD and whether it is dirty, when it is a checkout; ``REGISTRY_TREE_SHA`` otherwise
    (a gate export carries no ``.git``)."""
    here = os.path.dirname(os.path.abspath(__file__))
    out: Dict[str, object] = {"tree_sha": os.environ.get("REGISTRY_TREE_SHA") or None, "dirty": None}
    try:
        head = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        if head.returncode == 0 and head.stdout.strip():
            out["tree_sha"] = head.stdout.strip()
            st = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=60)
            out["dirty"] = bool(st.stdout.strip()) if st.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def source_identity(conn, db_path: str, hash_source: bool = False) -> Dict[str, object]:
    """What the database read is: path, ``st_dev:st_ino``, size, (optionally) sha256 of the file, schema version, the fold-order row 0,
    ``MAX(record_ingest.ingest_seq)``."""
    from .migrations import db_version
    st = os.stat(db_path)
    out: Dict[str, object] = {"path": os.path.abspath(db_path), "dev_ino": f"{st.st_dev}:{st.st_ino}", "size": int(st.st_size),
                              "schema_version": db_version(conn), "fold_order": DB.fold_order_version(conn),
                              "max_ingest_seq": conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0]}
    if hash_source:
        out["sha256"] = sha256_file(db_path)
    return out


def live_reason(db_path: str, root: Optional[str] = None) -> Optional[str]:
    """Why ``db_path`` is a live registry's database (None when it is not): the database of ``root``, the default live root's, or a
    database whose own root holds harvest state (``db.HARVEST_STATE_FILES``)."""
    p = os.path.realpath(db_path)
    if root and p == os.path.realpath(DB.db_path(root)):
        return f"{db_path} is the database of the registry root {root}"
    if p == os.path.realpath(DB.db_path(DB.DEFAULT_ROOT)):
        return f"{db_path} is the live registry's database"
    found = DB.harvest_state_found(os.path.dirname(p))
    if found:
        return f"{db_path} sits in a registry root with harvest state ({', '.join(found)})"
    return None


# --------------------------------------------------------------------------------------------
# row files: write (zstd, hashed as written) and read (checked to the frame's end)
# --------------------------------------------------------------------------------------------


class _HashSink:
    def __init__(self, fh):
        self.fh = fh
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, b) -> int:
        b = bytes(b)
        self.h.update(b)
        self.n += len(b)
        return self.fh.write(b)

    def flush(self) -> None:
        self.fh.flush()


class ZstdLines:
    """A zstd-compressed file of lines being written: ``sha256`` and ``bytes`` of the compressed file, ``raw_bytes`` and ``lines``."""

    def __init__(self, path: str):
        import zstandard
        self.path = path
        self.fh = open(path, "wb")
        self.sink = _HashSink(self.fh)
        self.w = zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_checksum=True).stream_writer(self.sink, closefd=False)
        self.raw_bytes = 0
        self.lines = 0

    def write_line(self, line: bytes) -> None:
        self.w.write(line + b"\n")
        self.raw_bytes += len(line) + 1
        self.lines += 1

    def close(self) -> Dict[str, object]:
        self.w.close()
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        return {"sha256": self.sink.h.hexdigest(), "bytes": self.sink.n, "raw_bytes": self.raw_bytes, "lines": self.lines}


def iter_zstd_lines(path: str) -> Iterator[bytes]:
    """The lines of a zstd file, each without its newline; raises :class:`ArchiveError` on a corrupt frame (the frame checksum) or a
    file cut short (no frame end)."""
    import zstandard
    d = zstandard.ZstdDecompressor().decompressobj()
    tail = b""
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                buf = tail + d.decompress(chunk)
                lines = buf.split(b"\n")
                tail = lines.pop()
                yield from lines
    except zstandard.ZstdError as exc:
        raise ArchiveError(f"{path}: corrupt zstd data ({exc})")
    if not d.eof:
        raise ArchiveError(f"{path}: the zstd frame is cut short (the file is truncated)")
    if tail:
        raise ArchiveError(f"{path}: the last line has no newline")


def iter_archive_rows(archive_dir: str, table: str) -> Iterator[Tuple]:
    """Every archived row of ``table``, decoded, in file order."""
    for line in iter_zstd_lines(os.path.join(archive_dir, row_file(table))):
        yield P.decode_row(line)


def iter_archive_groups(archive_dir: str, table: str, cols: Sequence[str]) -> Iterator[Tuple[str, int, List[Tuple]]]:
    """``(revision_id, record_file_id, rows)`` per carrier group of ``table``, in file order (each group's rows are contiguous)."""
    i_rev, i_rf = list(cols).index("source_revision_id"), list(cols).index("record_file_id")
    cur: Optional[Tuple[str, int]] = None
    rows: List[Tuple] = []
    for row in iter_archive_rows(archive_dir, table):
        k = (str(row[i_rev]), int(row[i_rf]))
        if cur is not None and k != cur:
            yield cur[0], cur[1], rows
            rows = []
        cur = k
        rows.append(row)
    if cur is not None:
        yield cur[0], cur[1], rows


def read_manifest(archive_dir: str) -> Dict[str, object]:
    path = os.path.join(archive_dir, P.MANIFEST_FILE)
    try:
        m = _read_json(path)
    except (OSError, ValueError) as exc:
        raise ArchiveInvalid(archive_dir, [f"{P.MANIFEST_FILE}: unreadable ({exc})"])
    if not isinstance(m, dict) or m.get("format") != FORMAT:
        raise ArchiveInvalid(archive_dir, [f"{P.MANIFEST_FILE}: not a {FORMAT} manifest"])
    return m


def read_sums(archive_dir: str) -> Dict[str, str]:
    """``{file name: sha256}`` of the archive's SHA256SUMS (``sha256sum`` format)."""
    out: Dict[str, str] = {}
    with open(os.path.join(archive_dir, SUMS_FILE), "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            sha, _, name = ln.partition("  ")
            if len(sha) != 64 or not name:
                raise ValueError(f"malformed line {ln[:100]!r}")
            out[name.lstrip("*")] = sha
    return out


def archive_name(db_path: str, kind: str, now: Optional[str] = None) -> str:
    """``<kind>-<utc>-<dbid8>``: ``utc`` compact (``20261019T183000Z``), ``dbid8`` the first 8 hex of sha256 over the
    source's path, ``st_dev:st_ino`` and size -- which file the archive was read from, without hashing it."""
    if not kind or "/" in kind:
        raise ArchiveError("archive --out-parent needs --kind (rehearsal | live-w1 | live-delta-<n>)")
    st = os.stat(db_path)
    ident = f"{os.path.abspath(db_path)}|{st.st_dev}:{st.st_ino}|{st.st_size}"
    utc = (now or R.utc_now()).replace("-", "").replace(":", "")
    utc = utc.split(".")[0] + "Z" if "." in utc else utc
    return f"{kind}-{utc}-{hashlib.sha256(ident.encode('utf-8')).hexdigest()[:8]}"


def load_groups_filter(path: str) -> Set[GroupKey]:
    """A ``--groups`` file: ``{"format": "prune-groups/1", "groups": [[table, revision_id, record_file_id], ...]}`` (what ``status
    --uncovered-out`` writes); entries may also be objects with those three keys."""
    obj = _read_json(path)
    items = obj.get("groups") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        raise ArchiveError(f"{path}: not a {GROUPS_FILTER_FORMAT} file")
    out: Set[GroupKey] = set()
    for it in items:
        if isinstance(it, dict):
            t, rev, rf = it.get("table"), it.get("revision_id"), it.get("record_file_id")
        else:
            t, rev, rf = it
        if t not in P.PRUNE_TABLES:
            raise ArchiveError(f"{path}: {t!r} is not a pruned table")
        out.add((str(t), str(rev), int(rf)))
    return out


def group_dicts(archive_dir: str) -> List[Dict[str, object]]:
    """The archive's carrier groups as the deletion contract reads them (``equality.check_contract``'s ``archive_groups``)."""
    return [g.__dict__ | {"class": g.klass} for g in P.read_groups(archive_dir)]


def write_groups_filter(path: str, groups: Iterable[GroupKey], note: str = "") -> None:
    _write_json(path, {"format": GROUPS_FILTER_FORMAT, "note": note, "groups": [list(g) for g in sorted(groups)]})


# --------------------------------------------------------------------------------------------
# the deletable set in one snapshot
# --------------------------------------------------------------------------------------------


def superseded_revisions(conn, only: Optional[Iterable[str]] = None) -> List[str]:
    revs = [str(r[0]) for r in conn.execute(
        f"SELECT revision_id FROM source_revision WHERE {P.accepted_superseded()} ORDER BY revision_id")]
    if only is not None:
        keep = set(only)
        revs = [r for r in revs if r in keep]
    return revs


def _plans_in_order(conn, root: str, rev: str, mode: P.ReclaimMode) -> List[P.CarrierPlan]:
    plans = P.revision_plan(conn, root, rev, mode)
    return sorted(plans, key=lambda p: (P.PRUNE_TABLES.index(p.table), p.record_file_id))


def _group_rows(conn, p: P.CarrierPlan) -> List[Tuple]:
    rows = P.present_rows(conn, p.table, p.revision_id, p.record_file_id)
    pos = _pk_positions(p.table, P.columns_of(p.table))
    rows.sort(key=lambda r: _order(r, pos))
    return rows


# --------------------------------------------------------------------------------------------
# archive
# --------------------------------------------------------------------------------------------


def archive(db_path: str, root: str, out_dir: str, groups: Optional[Set[GroupKey]] = None, kind: str = "",
            hash_source: bool = False, argv: Optional[Sequence[str]] = None, now: Optional[str] = None,
            progress: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Export every deletable carrier group of ``db_path`` (read-only, one transaction) to ``out_dir`` (module doc); ``groups``
    restricts the export to those ``(table, revision_id, record_file_id)`` (a delta archive). Returns the MANIFEST."""
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        raise ArchiveError(f"archive: {out_dir} exists; refusing to overwrite an archive")
    if not os.path.exists(db_path):
        raise ArchiveError(f"archive: no database at {db_path}")
    now = now or R.utc_now()
    t0 = time.monotonic()
    tmp = f"{out_dir}.partial-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    writers: Dict[str, ZstdLines] = {}
    try:
        writers = {t: ZstdLines(os.path.join(tmp, row_file(t))) for t in P.PRUNE_TABLES}
        census = {t: {k: {"rows": 0, "carriers": 0} for k in P.CLASSES} for t in P.PRUNE_TABLES}
        tot_all = {t: P.Digest() for t in P.PRUNE_TABLES}
        tot_bind = {t: P.Digest() for t in P.PRUNE_TABLES}
        ngroups = {t: 0 for t in P.PRUNE_TABLES}
        lines: List[Tuple] = []
        skipped: List[Dict[str, object]] = []
        found: Set[GroupKey] = set()
        with DB.reading(db_path) as conn:
            conn.execute("BEGIN")
            ident = source_identity(conn, db_path, hash_source)
            revs = superseded_revisions(conn, None if groups is None else {g[1] for g in groups})
            for i, rev in enumerate(revs):
                for p in _plans_in_order(conn, root, rev, P.ReclaimMode.normal()):
                    key = (p.table, p.revision_id, int(p.record_file_id))
                    if groups is not None:
                        if key not in groups:
                            continue
                        found.add(key)
                    c = census[p.table][p.klass]
                    c["rows"] += p.rows
                    c["carriers"] += 1
                    if p.klass != P.DELETABLE:
                        continue
                    if p.ledger is not None and p.ledger.rows_deleted > 0:
                        skipped.append({"table": p.table, "revision_id": rev, "record_file_id": int(p.record_file_id),
                                        "rows_present": p.rows, "rows_deleted": p.ledger.rows_deleted})
                        continue
                    rows = _group_rows(conn, p)
                    if len(rows) != p.rows:
                        raise ArchiveError(f"archive: {key} holds {len(rows)} rows, its plan {p.rows} (the snapshot moved)")
                    d_all = P.Digest.of_rows(p.table, rows)
                    d_bind = P.Digest.of_rows(p.table, rows, P.BIND_EXCLUDE)
                    if (d_bind.count, d_bind.sum256) != (p.present_bind.count, p.present_bind.sum256):
                        raise ArchiveError(f"archive: {key}: the exported rows' digest_bind differs from the plan's")
                    w = writers[p.table]
                    for r in rows:
                        w.write_line(P.encode_row(r))
                    tot_all[p.table].add(d_all)
                    tot_bind[p.table].add(d_bind)
                    ngroups[p.table] += 1
                    lines.append((p.table, rev, int(p.record_file_id), p.file_uri, P.DELETABLE, len(rows), d_all.hex(), d_bind.hex()))
                if progress is not None and (i + 1) % 100 == 0:
                    progress(f"archive: {i + 1}/{len(revs)} revisions, {sum(d.count for d in tot_all.values())} rows, "
                             f"{time.monotonic() - t0:.0f} s")
            conn.execute("COMMIT")
        files = {t: writers[t].close() for t in P.PRUNE_TABLES}
        writers = {}
        gw = ZstdLines(os.path.join(tmp, P.GROUPS_FILE))
        gw.write_line("\t".join(P.GROUPS_TSV_COLUMNS).encode("utf-8"))
        for ln in sorted(lines, key=lambda x: (x[0], x[1], x[2])):
            if any("\t" in str(v) or "\n" in str(v) for v in ln):
                raise ArchiveError(f"archive: a group field holds a tab or newline: {ln!r}")
            gw.write_line("\t".join(str(v) for v in ln).encode("utf-8"))
        gfile = gw.close()
        manifest: Dict[str, object] = {
            "format": FORMAT, "rule": P.RULE_VERSION, "row_digest": P.ROW_DIGEST_TAG.rstrip(b"\0").decode(), "kind": kind,
            "created_utc": now, "tool": tool_tree_sha(), "command": list(argv) if argv is not None else list(sys.argv),
            "source": ident, "root": os.path.abspath(root), "seconds": round(time.monotonic() - t0, 3),
            "groups_filter": None if groups is None else {"groups": [list(g) for g in sorted(groups)],
                                                          "not_found": [list(g) for g in sorted(groups - found)]},
            "tables": {t: {"file": row_file(t), "columns": list(P.columns_of(t)), "rows": tot_all[t].count, "groups": ngroups[t],
                           "digest_all": digest_entry(tot_all[t]), "digest_bind": digest_entry(tot_bind[t]),
                           "sha256": files[t]["sha256"], "bytes": files[t]["bytes"], "raw_bytes": files[t]["raw_bytes"]}
                       for t in P.PRUNE_TABLES},
            "groups": {"file": P.GROUPS_FILE, "count": len(lines), "sha256": gfile["sha256"], "bytes": gfile["bytes"]},
            "census": census, "skipped_partial": skipped, "revisions_examined": len(revs),
        }
        _write_json(os.path.join(tmp, P.MANIFEST_FILE), manifest)
        names = [row_file(t) for t in P.PRUNE_TABLES] + [P.GROUPS_FILE, P.MANIFEST_FILE]
        with open(os.path.join(tmp, SUMS_FILE), "w", encoding="utf-8") as fh:
            for n in names:
                fh.write(f"{sha256_file(os.path.join(tmp, n))}  {n}\n")
            fh.flush()
            os.fsync(fh.fileno())
        DB.fsync_dir(tmp)
        os.rename(tmp, out_dir)
        DB.fsync_dir(os.path.dirname(out_dir) or ".")
        return manifest
    except BaseException:
        for w in writers.values():
            try:
                w.fh.close()
            except OSError:
                pass
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# --------------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------------


def _check_files(archive_dir: str, manifest: Mapping[str, object]) -> List[str]:
    """SHA256SUMS (as ``sha256sum -c`` would) and the MANIFEST's per-file sha256 / bytes; every file the MANIFEST names must be listed."""
    fails: List[str] = []
    try:
        sums = read_sums(archive_dir)
    except (OSError, ValueError) as exc:
        return [f"{SUMS_FILE}: unreadable ({exc})"]
    tables = dict(manifest.get("tables") or {})
    named = {P.MANIFEST_FILE, P.GROUPS_FILE} | {str(e.get("file")) for e in tables.values()}
    for n in sorted(named - set(sums)):
        fails.append(f"{n}: not listed in {SUMS_FILE}")
    for n, want in sorted(sums.items()):
        path = os.path.join(archive_dir, n)
        if not os.path.isfile(path):
            fails.append(f"{n}: missing")
            continue
        have = sha256_file(path)
        if have != want:
            fails.append(f"{n}: sha256 {have[:16]} is not the listed {want[:16]}")
    for t, e in sorted(tables.items()):
        path = os.path.join(archive_dir, str(e.get("file")))
        if os.path.isfile(path) and (os.path.getsize(path) != int(e.get("bytes", -1))):
            fails.append(f"{e.get('file')}: {os.path.getsize(path)} bytes, the MANIFEST says {e.get('bytes')}")
    g = dict(manifest.get("groups") or {})
    gpath = os.path.join(archive_dir, P.GROUPS_FILE)
    if os.path.isfile(gpath) and g.get("sha256") and sha256_file(gpath) != g.get("sha256"):
        fails.append(f"{P.GROUPS_FILE}: sha256 is not the MANIFEST's")
    return fails


def _verify_rows(archive_dir: str, manifest: Mapping[str, object], gmap: Mapping[GroupKey, P.ArchiveGroup]) -> Tuple[List[str], Dict[str, object]]:
    """Every row re-read and digested; groups and table totals recomputed and compared."""
    fails: List[str] = []
    seen: Set[GroupKey] = set()
    totals: Dict[str, object] = {}
    for t in P.PRUNE_TABLES:
        e = dict((manifest.get("tables") or {}).get(t) or {})
        cols = list(e.get("columns") or [])
        absent = [c for c in ("source_revision_id", "record_file_id", "ingest_seq") if c not in cols]
        if not cols or absent:
            fails.append(f"{t}: the MANIFEST's column list lacks {absent or 'every column'}")
            continue
        d_all = _digest_with(t, cols, ())
        d_bind = _digest_with(t, cols, P.BIND_EXCLUDE)
        pkpos = _pk_positions(t, cols)
        i_rev, i_rf = cols.index("source_revision_id"), cols.index("record_file_id")
        tot_all, tot_bind = P.Digest(), P.Digest()
        prev_group: Optional[Tuple[str, int]] = None
        prev_pk = None
        g_all, g_bind = P.Digest(), P.Digest()
        ngroups = 0

        def close(gk: Optional[Tuple[str, int]]) -> None:
            nonlocal ngroups
            if gk is None:
                return
            ngroups += 1
            key = (t, gk[0], gk[1])
            seen.add(key)
            g = gmap.get(key)
            if g is None:
                fails.append(f"{t} group ({gk[0]}, {gk[1]}): in the row file, not in {P.GROUPS_FILE}")
            elif g.rows != g_all.count or g.digest_all != g_all.hex() or g.digest_bind != g_bind.hex():
                fails.append(f"{t} group ({gk[0]}, {gk[1]}): rows/digests ({g_all.count}) differ from {P.GROUPS_FILE} ({g.rows})")
        try:
            for line in iter_zstd_lines(os.path.join(archive_dir, str(e.get("file")))):
                try:
                    row = P.decode_row(line)
                except (ValueError, TypeError) as exc:
                    fails.append(f"{t}: an undecodable line ({exc})")
                    break
                if len(row) != len(cols):
                    fails.append(f"{t}: a row of {len(row)} values for {len(cols)} columns")
                    break
                gk = (str(row[i_rev]), int(row[i_rf]))
                pk = _order(row, pkpos)
                if gk != prev_group:
                    if prev_group is not None and (gk[0], gk[1]) < prev_group:
                        fails.append(f"{t}: group {gk} follows {prev_group} (the file is not in order)")
                        break
                    if (t, gk[0], gk[1]) in seen:
                        fails.append(f"{t}: group {gk} is split in the row file")
                        break
                    close(prev_group)
                    prev_group, prev_pk = gk, None
                    g_all, g_bind = P.Digest(), P.Digest()
                if prev_pk is not None and not pk > prev_pk:
                    fails.append(f"{t}: rows of group {gk} are not in strict primary-key order")
                    break
                prev_pk = pk
                a, b = d_all(row), d_bind(row)
                g_all.add(a)
                g_bind.add(b)
                tot_all.add(a)
                tot_bind.add(b)
            else:
                close(prev_group)
        except ArchiveError as exc:
            fails.append(str(exc))
            continue
        totals[t] = {"rows": tot_all.count, "groups": ngroups, "digest_all": digest_entry(tot_all), "digest_bind": digest_entry(tot_bind)}
        want = (int(e.get("rows", -1)), dict(e.get("digest_all") or {}).get("sum256"), dict(e.get("digest_bind") or {}).get("sum256"),
                int(e.get("groups", -1)))
        have = (tot_all.count, tot_all.hex(), tot_bind.hex(), ngroups)
        if want != have:
            fails.append(f"{t}: table totals {have[0]} rows / {have[3]} groups differ from the MANIFEST's {want[0]} / {want[3]} (or a digest does)")
    for key in sorted(set(gmap) - seen):
        fails.append(f"{key[0]} group ({key[1]}, {key[2]}): in {P.GROUPS_FILE}, no row in the row file")
    return fails, totals


def _db_deletable(db_path: str, root: str, only: Optional[Set[GroupKey]]) -> Tuple[Dict[GroupKey, Tuple[int, str, str]], int]:
    """``{group: (rows, digest_all hex, digest_bind hex)}`` of the database's deletable groups, in one read snapshot (a group the ledger
    already holds deleted rows of is left out, as the archive leaves it out); and the count of such partial groups."""
    out: Dict[GroupKey, Tuple[int, str, str]] = {}
    partial = 0
    with DB.reading(db_path) as conn:
        conn.execute("BEGIN")
        for rev in superseded_revisions(conn, None if only is None else {g[1] for g in only}):
            for p in _plans_in_order(conn, root, rev, P.ReclaimMode.normal()):
                key = (p.table, p.revision_id, int(p.record_file_id))
                if p.klass != P.DELETABLE or (only is not None and key not in only):
                    continue
                if p.ledger is not None and p.ledger.rows_deleted > 0:
                    partial += 1
                    continue
                rows = P.present_rows(conn, p.table, rev, p.record_file_id)
                out[key] = (len(rows), P.Digest.of_rows(p.table, rows).hex(), P.Digest.of_rows(p.table, rows, P.BIND_EXCLUDE).hex())
        conn.execute("COMMIT")
    return out, partial


def verify_archive(archive_dir: str, db_path: Optional[str] = None, root: Optional[str] = None,
                   verified_name: str = P.VERIFIED_FILE, write: bool = True, now: Optional[str] = None) -> Dict[str, object]:
    """Verify the archive at ``archive_dir`` (module doc); raise :class:`ArchiveInvalid` naming every failed check, else write
    ``verified_name`` (unless ``write`` is False) and return the verification record."""
    archive_dir = os.path.abspath(archive_dir)
    now = now or R.utc_now()
    t0 = time.monotonic()
    manifest = read_manifest(archive_dir)
    msha = sha256_file(os.path.join(archive_dir, P.MANIFEST_FILE))
    fails = _check_files(archive_dir, manifest)
    if fails:
        raise ArchiveInvalid(archive_dir, fails)
    try:
        groups = P.read_groups(archive_dir)
    except (P.ModeStateInvalid, ValueError, OSError) as exc:
        raise ArchiveInvalid(archive_dir, [f"{P.GROUPS_FILE}: {exc}"])
    gmap: Dict[GroupKey, P.ArchiveGroup] = {}
    for g in groups:
        key = (g.table, g.revision_id, g.record_file_id)
        if key in gmap:
            fails.append(f"{P.GROUPS_FILE}: group {key} listed twice")
        if g.klass != P.DELETABLE or g.rows <= 0:
            fails.append(f"{P.GROUPS_FILE}: group {key} is {g.klass} with {g.rows} rows (an archive lists exported deletable groups only)")
        gmap[key] = g
    if len(groups) != int(dict(manifest.get("groups") or {}).get("count", -1)):
        fails.append(f"{P.GROUPS_FILE}: {len(groups)} groups, the MANIFEST says {dict(manifest.get('groups') or {}).get('count')}")
    row_fails, totals = _verify_rows(archive_dir, manifest, gmap)
    fails += row_fails
    db_rec: Optional[Dict[str, object]] = None
    if db_path is not None and not fails:
        flt = manifest.get("groups_filter")
        only = None if not flt else {(str(a), str(b), int(c)) for a, b, c in flt.get("groups") or []}
        have, partial = _db_deletable(db_path, root or str(manifest.get("root") or ""), only)
        want = {k: (g.rows, g.digest_all, g.digest_bind) for k, g in gmap.items()}
        for k in sorted(set(want) - set(have)):
            fails.append(f"database: archived group {k} is not deletable there (or not present)")
        for k in sorted(set(have) - set(want)):
            fails.append(f"database: deletable group {k} is not in the archive")
        for k in sorted(set(want) & set(have)):
            if want[k] != have[k]:
                fails.append(f"database: group {k} differs (rows {have[k][0]} vs archived {want[k][0]}, or a digest)")
        with DB.reading(db_path) as conn:
            db_rec = source_identity(conn, db_path)
        db_rec["deletable_groups"] = len(have)
        db_rec["partial_groups"] = partial
    if fails:
        raise ArchiveInvalid(archive_dir, fails)
    rec = {"format": VERIFIED_FORMAT, P.VERIFIED_MANIFEST_KEY: msha, "verified_utc": now, "archive": archive_dir,
           "verified_name": verified_name, "tool": tool_tree_sha(), "groups": len(groups), "tables": totals, "database": db_rec,
           "seconds": round(time.monotonic() - t0, 3)}
    if write:
        _write_json(os.path.join(archive_dir, verified_name), rec)
    return rec


def gate_local_archive(archive_dir: str, now: Optional[str] = None) -> Dict[str, object]:
    """The ``mode cohort`` gate for one LOCAL archive directory (the copy the live cohort reads): the cluster's ``VERIFIED.json`` must be
    present and name this MANIFEST's sha256; every file the MANIFEST names must be here and pass ``sha256sum -c``; the archive's
    columns must be the code's; and a full local re-verification (every row, group and table digest recomputed from the local row
    files) must pass -- only then is ``VERIFIED.local.json`` written. Raises :class:`ArchiveInvalid` otherwise."""
    archive_dir = os.path.abspath(archive_dir)
    manifest = read_manifest(archive_dir)
    msha = sha256_file(os.path.join(archive_dir, P.MANIFEST_FILE))
    vpath = os.path.join(archive_dir, P.VERIFIED_FILE)
    try:
        ver = _read_json(vpath)
    except (OSError, ValueError) as exc:
        raise ArchiveInvalid(archive_dir, [f"{P.VERIFIED_FILE} (the cluster's independent verification): absent or unreadable ({exc})"])
    if not isinstance(ver, dict) or ver.get(P.VERIFIED_MANIFEST_KEY) != msha:
        raise ArchiveInvalid(archive_dir, [f"{P.VERIFIED_FILE} names MANIFEST sha256 "
                                           f"{(ver or {}).get(P.VERIFIED_MANIFEST_KEY) if isinstance(ver, dict) else None}, not this "
                                           f"archive's {msha}"])
    bad_cols = [t for t in P.PRUNE_TABLES if list(dict((manifest.get("tables") or {}).get(t) or {}).get("columns") or []) != list(P.columns_of(t))]
    if bad_cols:
        raise ArchiveInvalid(archive_dir, [f"{t}: the archive's columns are not the code's" for t in bad_cols])
    rec = verify_archive(archive_dir, verified_name=P.VERIFIED_LOCAL_FILE, now=now)
    P.verified_manifest(archive_dir, P.VERIFIED_LOCAL_FILE)
    return rec


def _locally_verified(archive_dir: str) -> bool:
    try:
        P.verified_manifest(archive_dir, P.VERIFIED_LOCAL_FILE)
        return True
    except P.ModeStateInvalid:
        return False


# --------------------------------------------------------------------------------------------
# restore
# --------------------------------------------------------------------------------------------

#: ``PRAGMA temp_store`` on restore's write connection (``db.connect`` sets MEMORY). Each group is one SAVEPOINT, and every INSERT in it
#: opens a statement savepoint of its own (the item tables' RAISE(ABORT) triggers). With the sub-journal in memory, releasing each
#: statement savepoint truncates it by walking every page the group has journaled so far, so a group costs time quadratic in its rows.
#: A file sub-journal is not truncated on release, and the cost per row is flat. SQLite reads the setting when a write transaction
#: begins (a change inside one does nothing), so it is set on the fresh connection, before the first group. Only a write pattern with
#: an outer savepoint needs it: a plain transaction (the harvester's item flush, :func:`rehydrate`) releases its statement savepoints
#: to an empty journal, and is faster with it in memory.
RESTORE_TEMP_STORE = "FILE"
#: A live restore refuses when :func:`sqlite_temp_dir` has less free space than this: the file holds one group's journaled pages at a
#: time (about 0.5 GB for the largest groups), with room to spare.
RESTORE_TEMP_MIN_FREE_BYTES = 2 * 1024 ** 3


def sqlite_temp_dir() -> str:
    """The directory SQLite's unix VFS creates its temp files in: the first writable directory of ``$SQLITE_TMPDIR``, ``$TMPDIR``,
    ``/var/tmp``, ``/usr/tmp`` and ``/tmp``, else the working directory. SQLite reads the two variables once, when it initializes
    (``import sqlite3``), so they must be set in the environment the command starts with."""
    for d in (os.environ.get("SQLITE_TMPDIR"), os.environ.get("TMPDIR"), "/var/tmp", "/usr/tmp", "/tmp"):
        if d and os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK):
            return d
    return os.getcwd()


def _rank(conn, record_file_id: int) -> Optional[int]:
    r = conn.execute("SELECT ingest_seq FROM record_ingest WHERE record_file_id = ?", (int(record_file_id),)).fetchone()
    return int(r[0]) if r else None


def _with_seq(rows: Sequence[Tuple], i_seq: int, seq: int) -> List[Tuple]:
    return [tuple(seq if i == i_seq else v for i, v in enumerate(r)) for r in rows]


def _insert_group(conn, table: str, cols: Sequence[str], rows: Sequence[Tuple], stats: Dict[str, int],
                  examples: List[Dict[str, object]]) -> int:
    """Insert ``rows`` (keys absent only) inside the caller's transaction; a present key is compared, never overwritten. The common
    case (no key present) is one ``executemany``; otherwise it is rolled back to a savepoint and every row is looked up first."""
    pkpos = _pk_positions(table, cols)
    i_seq = list(cols).index("ingest_seq")
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) ON CONFLICT DO NOTHING"
    conn.execute("SAVEPOINT prune_restore_group")
    try:
        cur = conn.executemany(sql, list(rows))
        if cur.rowcount == len(rows):                   # direct changes only (a trigger's writes are not counted)
            conn.execute("RELEASE prune_restore_group")
            stats["inserted"] += len(rows)
            return len(rows)
        conn.execute("ROLLBACK TO prune_restore_group")
        conn.execute("RELEASE prune_restore_group")
    except BaseException:
        conn.execute("ROLLBACK TO prune_restore_group")
        conn.execute("RELEASE prune_restore_group")
        raise
    where = " AND ".join(f"{c} = ?" for c in S.TABLE_BY_NAME[table].pk)
    sel = f"SELECT {', '.join(cols)} FROM {table} WHERE {where}"
    ins = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    n = 0
    for r in rows:
        have = conn.execute(sel, tuple(r[i] for i in pkpos)).fetchone()
        if have is None:
            conn.execute(ins, tuple(r))
            n += 1
            continue
        have = tuple(have)
        if all(have[i] == r[i] for i in range(len(cols)) if i != i_seq):
            stats["present_identical"] += 1
        else:
            stats["conflicts"] += 1
            if len(examples) < 10:
                examples.append({"table": table, "key": [P.encode_value(r[i]) for i in pkpos],
                                 "archived": [P.encode_value(v) for v in r], "present": [P.encode_value(v) for v in have]})
    stats["inserted"] += n
    return n


def _ledger_after_restore(conn, g: P.ArchiveGroup) -> str:
    """The ledger row of a restored group: removed when the group is whole again (present = the archived group, by count and
    ``digest_bind``); else kept. Returns ``removed`` / ``kept`` / ``none``."""
    if P.read_ledger(conn, g.table, g.revision_id, g.record_file_id) is None:
        return "none"
    present = P.present_rows(conn, g.table, g.revision_id, g.record_file_id)
    bind = P.Digest.of_rows(g.table, present, P.BIND_EXCLUDE)
    if len(present) == g.rows and bind.hex() == g.digest_bind:
        P.drop_ledger_row(conn, g.table, g.revision_id, g.record_file_id)
        return "removed"
    return "kept"


def restore(archive_dir: str, db_path: str, root: Optional[str] = None, live: bool = False) -> Dict[str, object]:
    """Restore every archived row whose key is absent from ``db_path`` (module doc). The live root's database is refused unless
    ``live`` -- and then only with the maintenance flag set and ``reclaim_mode`` = ``hold``, under the write lock."""
    archive_dir = os.path.abspath(archive_dir)
    reason = live_reason(db_path, root)
    if reason and not live:
        raise LiveRefused(f"restore: {reason}; restoring into a live registry needs --live, its maintenance flag, reclaim mode "
                          "hold and the write lock (a window)")
    manifest = read_manifest(archive_dir)
    fails = _check_files(archive_dir, manifest)
    if fails:
        raise ArchiveInvalid(archive_dir, fails)
    for t in P.PRUNE_TABLES:
        if list(dict(manifest["tables"][t]).get("columns") or []) != list(P.columns_of(t)):
            raise ArchiveError(f"restore: the archive's {t} columns are not the code's; restore with the code that wrote it")
    lock_root = os.path.dirname(os.path.realpath(db_path))
    if reason:
        from . import harvest as H
        if H.maintenance(lock_root) is None:
            raise LiveRefused(f"restore --live: {lock_root} has no maintenance flag set (state/{H.MAINTENANCE_FILE})")
        mode = P.ReclaimMode.from_state(lock_root)
        if mode.kind != "hold":
            raise LiveRefused(f"restore --live: the reclaim mode is {mode.kind}, not hold")
        tmp = sqlite_temp_dir()
        free = DB.free_bytes(tmp)
        if free < RESTORE_TEMP_MIN_FREE_BYTES:
            raise LiveRefused(f"restore --live: SQLite's temp directory {tmp} has {free} B free, under {RESTORE_TEMP_MIN_FREE_BYTES} B "
                              "(it holds a group's statement journal); start the command with SQLITE_TMPDIR naming a directory with room")
    gmap = {(g.table, g.revision_id, g.record_file_id): g for g in P.read_groups(archive_dir)}
    lock = DB.write_lock(lock_root, timeout_s=60.0) if reason else contextlib.nullcontext()
    out: Dict[str, object] = {"archive": archive_dir, "db": os.path.abspath(db_path), "tables": {}, "conflict_examples": []}
    ledger = {"removed": 0, "kept": 0}
    revs_reset: Set[str] = set()
    no_rank: List[List[object]] = []
    with lock:
        conn = DB.connect(db_path)
        try:
            conn.execute(f"PRAGMA temp_store={RESTORE_TEMP_STORE}")        # this connection is restore's own, closed below
            DB.require_shape(conn, db_path)
            for t in P.PRUNE_TABLES:
                cols = list(P.columns_of(t))
                i_seq = cols.index("ingest_seq")
                stats = {"groups": 0, "rows": 0, "inserted": 0, "present_identical": 0, "conflicts": 0, "no_rank": 0}
                for rev, rfid, rows in iter_archive_groups(archive_dir, t, cols):
                    g = gmap.get((t, rev, rfid))
                    if g is None:
                        raise ArchiveInvalid(archive_dir, [f"{t} group ({rev}, {rfid}) is not in {P.GROUPS_FILE}"])
                    if (len(rows) != g.rows or P.Digest.of_rows(t, rows).hex() != g.digest_all):
                        raise ArchiveInvalid(archive_dir, [f"{t} group ({rev}, {rfid}): the rows read do not digest to the listed group"])
                    stats["groups"] += 1
                    stats["rows"] += len(rows)
                    seq = _rank(conn, rfid)
                    if seq is None:
                        stats["no_rank"] += len(rows)
                        no_rank.append([t, rev, rfid])
                        continue
                    with DB.transaction(conn):
                        n = _insert_group(conn, t, cols, _with_seq(rows, i_seq, seq), stats, out["conflict_examples"])  # type: ignore[arg-type]
                        state = _ledger_after_restore(conn, g)
                        if state in ledger:
                            ledger[state] += 1
                        if n or state == "removed":
                            P.reset_reclaimed(conn, rev)
                            revs_reset.add(rev)
                out["tables"][t] = stats  # type: ignore[index]
            try:
                DB.wal_checkpoint_truncate(conn)
            except Exception:
                pass
        finally:
            conn.close()
    out["ledger_removed"], out["ledger_kept"] = ledger["removed"], ledger["kept"]
    out["revisions_reset"] = len(revs_reset)
    out["no_rank_groups"] = no_rank[:20]
    out["conflicts"] = sum(int(s["conflicts"]) for s in out["tables"].values())  # type: ignore[union-attr]
    out["ok"] = out["conflicts"] == 0 and not no_rank
    return out


# --------------------------------------------------------------------------------------------
# rehydrate: the operator remedies for a held rehydration (prune.rehydrate's failure)
# --------------------------------------------------------------------------------------------


def _archived_groups(archive_dirs: Sequence[str], keys: Iterable[GroupKey]) -> Dict[GroupKey, Tuple[P.ArchiveGroup, List[Tuple], str]]:
    """Every group of ``keys`` from the latest of ``archive_dirs`` that lists it, with its rows read (zstd frame checksums: a corrupt or cut
    file raises) and checked against the listed digest. Each archive's MANIFEST and groups file are read once; each archived table is
    scanned once, and the scan stops as soon as every group wanted from it is collected. The rows' authority is not a VERIFIED file but
    the database's own record of what it deleted: :func:`rehydrate` requires the archived rows absent from the table to equal the
    ledger's ``rows_deleted`` and ``deleted_digest_bind`` exactly."""
    want = set(keys)
    found: Dict[GroupKey, Tuple[P.ArchiveGroup, str]] = {}
    manifests: Dict[str, Dict[str, object]] = {}
    for d in archive_dirs:
        manifests[d] = read_manifest(d)
        for g in P.read_groups(d):
            k = (g.table, g.revision_id, g.record_file_id)
            if k in want:
                found[k] = (g, d)
    lost = sorted(want - set(found))
    if lost:
        raise ArchiveError(f"rehydrate: no archive given covers {lost[0]}" + (f" (+{len(lost) - 1} more)" if len(lost) > 1 else ""))
    by_scan: Dict[Tuple[str, str], Dict[Tuple[str, int], GroupKey]] = {}
    for k, (g, d) in found.items():
        by_scan.setdefault((d, k[0]), {})[(k[1], k[2])] = k
    out: Dict[GroupKey, Tuple[P.ArchiveGroup, List[Tuple], str]] = {}
    for (d, t), wanted in sorted(by_scan.items()):
        cols = list(dict(manifests[d]["tables"][t]).get("columns") or [])   # type: ignore[index]
        if cols != list(P.columns_of(t)):
            raise ArchiveError(f"rehydrate: {d}: the archive's {t} columns are not the code's")
        left = dict(wanted)
        for rev, rfid, grp in iter_archive_groups(d, t, cols):
            k = left.pop((rev, rfid), None)
            if k is not None:
                out[k] = (found[k][0], grp, d)
                if not left:
                    break
        for k in wanted.values():
            g = found[k][0]
            rows = out[k][1] if k in out else []              # a group the groups file lists and the row file lacks: no rows
            if len(rows) != g.rows or P.Digest.of_rows(t, rows).hex() != g.digest_all:
                raise ArchiveError(f"rehydrate: {d}: group {k}'s archived rows do not digest to the listed group")
    return out


def _log_decision(root: str, line: Mapping[str, object], decisions_log: Optional[str] = None) -> None:
    payload = json.dumps(dict(line), sort_keys=True, separators=(",", ":"), default=str)
    targets = [os.path.join(root, "state", DECISIONS_FILE), os.path.join(root, "state", "metrics.jsonl")]
    if decisions_log:
        targets.append(decisions_log)
    for path in targets:
        if not os.path.isdir(os.path.dirname(path) or "."):
            continue
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(payload + "\n")


def rehydrate(root: str, revision_id: str, archive_dirs: Sequence[str] = (), accept_reparse: bool = False, reason: str = "",
              decisions_log: Optional[str] = None, db_path: Optional[str] = None, lock_timeout_s: float = 600.0,
              now: Optional[str] = None) -> Dict[str, object]:
    """Restore every ``item_reclaim_ledger`` carrier of ``revision_id`` under the root's write lock, by one of the two remedies:

    * ``archive_dirs`` (``--from-archive``): the exact archived bytes. Each carrier's archived group must equal what the database
      accounts for -- ``rows = present + rows_deleted`` and ``digest_bind = D_bind(present) + deleted_digest_bind`` -- and the archived
      rows whose key is absent must be exactly the deleted ones (count and ``digest_bind``); ``ingest_seq`` is the carrier's current
      rank.
    * ``accept_reparse`` (``--accept-reparse --reason``): the CURRENT code's rows for every absent key (what a rebuild holds), with no
      authentication -- the operator's recorded decision; records the validator now refuses stay absent and are counted.

    Every check runs before anything is written; then each carrier is one transaction that inserts its rows, removes its ledger row and
    resets the revision's ``reclaimed``. The decision is logged (``state/prune_decisions.jsonl``, the metrics log, ``decisions_log``).
    Present rows are never modified."""
    if bool(archive_dirs) == bool(accept_reparse):
        raise ArchiveError("rehydrate: give exactly one of --from-archive or --accept-reparse")
    if accept_reparse and not reason.strip():
        raise ArchiveError("rehydrate --accept-reparse needs --reason")
    now = now or R.utc_now()
    db_path = db_path or DB.db_path(root)
    dirs = [os.path.abspath(d) for d in archive_dirs]
    archived: Dict[GroupKey, Tuple[P.ArchiveGroup, List[Tuple], str]] = {}
    if dirs:
        # the archived rows are read BEFORE the write lock (decompressing a row file is the slow part, and the harvester waits on the
        # lock): the carriers the ledger names now; anything the ledger names under the lock and not here is read there
        with DB.reading(db_path) as rconn:
            keys = [(l.table_name, revision_id, int(l.record_file_id)) for l in P.ledger_carriers(rconn, [revision_id]).get(revision_id, [])]
        archived = _archived_groups(dirs, keys) if keys else {}
    with DB.write_lock(root, timeout_s=lock_timeout_s):
        conn = DB.connect(db_path)
        try:
            DB.require_shape(conn, db_path)
            ledger = P.ledger_carriers(conn, [revision_id]).get(revision_id, [])
            if not ledger:
                return {"revision": revision_id, "carriers": 0, "rows_restored": 0, "note": "the revision holds no ledger rows"}
            if dirs:
                late = [(l.table_name, revision_id, int(l.record_file_id)) for l in ledger]
                late = [k for k in late if k not in archived]
                if late:
                    archived.update(_archived_groups(dirs, late))
            work: List[Tuple[P.LedgerRow, List[Tuple], Dict[str, object]]] = []
            for led in ledger:
                t = led.table_name
                cols = list(P.columns_of(t))
                i_seq = cols.index("ingest_seq")
                pkpos = _pk_positions(t, cols)
                present = P.present_rows(conn, t, revision_id, led.record_file_id)
                keys = {tuple(r[i] for i in pkpos) for r in present}
                seq = _rank(conn, led.record_file_id)
                if seq is None:
                    raise ArchiveError(f"rehydrate: carrier {led.file_uri} ({t}) has no record_ingest row; nothing written")
                info: Dict[str, object] = {"table": t, "file": led.file_uri, "rows_deleted": led.rows_deleted}
                if dirs:
                    key = (t, revision_id, int(led.record_file_id))
                    g, rows, d = archived[key]
                    have = P.Digest.of_rows(t, present, P.BIND_EXCLUDE) + led.deleted_bind
                    if g.rows != len(present) + led.rows_deleted or P.Digest.from_hex(g.rows, g.digest_bind).sum256 != have.sum256:
                        raise ArchiveError(f"rehydrate: {key}: the archived group ({g.rows} rows) is not present ({len(present)}) + deleted "
                                           f"({led.rows_deleted}) by count or digest_bind; nothing written")
                    missing = [r for r in rows if tuple(r[i] for i in pkpos) not in keys]
                    mb = P.Digest.of_rows(t, missing, P.BIND_EXCLUDE)
                    if len(missing) != led.rows_deleted or mb.sum256 != led.deleted_bind.sum256:
                        raise ArchiveError(f"rehydrate: {key}: the archived rows absent from the table are not exactly the deleted ones; nothing written")
                    info["archive"] = d
                    work.append((led, _with_seq(missing, i_seq, seq), info))
                else:
                    rp = P.reparse_file(root, led.file_uri, revision_id, (t,), seq)
                    if rp.error or rp.retry:
                        raise ArchiveError(f"rehydrate --accept-reparse: {led.file_uri}: {rp.error or 'changed while read'}; nothing written "
                                           "(only an archive can restore this carrier)")
                    rows = [r for r in rp.rows.get(t, []) if tuple(r[i] for i in pkpos) not in keys]
                    info["refused"] = list(rp.refused.get(t, []))[:20]
                    info["rows_unrestorable"] = max(0, led.rows_deleted - len(rows))
                    work.append((led, rows, info))
            from .reducer import Reducer
            reducer = Reducer(conn)
            restored = 0
            for led, rows, info in work:
                t = led.table_name
                with DB.transaction(conn):
                    if dirs:
                        cols = list(P.columns_of(t))
                        sql = f"INSERT INTO {t} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
                    else:
                        sql = reducer.item_upsert_sql(t)
                    if rows:
                        DB.executemany_batched(conn, sql, rows)
                    P.drop_ledger_row(conn, t, revision_id, led.record_file_id)
                    P.reset_reclaimed(conn, revision_id)
                info["rows_restored"] = len(rows)
                restored += len(rows)
        finally:
            conn.close()
    kind = "rehydrate_from_archive" if dirs else "rehydration_accepted"
    rec = {"kind": kind, "revision": revision_id, "carriers": [w[2] for w in work], "rows_restored": restored, "utc": now}
    if accept_reparse:
        rec["reason"] = reason
    _log_decision(root, rec, decisions_log)
    return rec


# --------------------------------------------------------------------------------------------
# digests of a database (the comparison contracts' input)
# --------------------------------------------------------------------------------------------


def _db_tables(conn) -> List[str]:
    return [str(r[0]) for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name")]


def _db_objects(conn) -> List[str]:
    return [str(r[0]) for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'view' AND name LIKE 'v\\_%' ESCAPE '\\' ORDER BY name")]


def schema_entries(conn) -> Dict[str, List[object]]:
    """``{"<type>:<name>": [tbl_name, normalized sql]}`` of every schema entry but SQLite's own tables."""
    out: Dict[str, List[object]] = {}
    for typ, name, tbl, sql in conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite\\_stat%' ESCAPE '\\'"):
        out[f"{typ}:{name}"] = [str(tbl), DB.normalize_ddl(sql) if sql else None]
    return out


def _table_columns(conn, table: str) -> List[str]:
    return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")]


def _digest_select(conn, name: str, sql: str) -> Dict[str, object]:
    d = P.Digest()
    cur = conn.cursor()
    cur.row_factory = None
    cur.execute(sql)
    cols = [c[0] for c in cur.description]
    while True:
        batch = cur.fetchmany(5000)
        if not batch:
            break
        for r in batch:
            d.add(generic_row_digest(name, r))
    out = digest_entry(d)
    out["columns"] = cols
    return out


def _digest_task(db_path: str, kind: str, name: str) -> Tuple[str, str, Dict[str, object]]:
    """One worker's unit: a table, a ``v_*`` object or the extras, by its own read-only connection."""
    t0 = time.monotonic()
    with DB.reading(db_path) as conn:
        if kind == "table":
            cols = _table_columns(conn, name)
            if name in P.PRUNE_TABLES and cols != list(P.columns_of(name)):
                raise ArchiveError(f"digest: {name}'s columns in {db_path} are not the code's: the archive's digests would not add up")
            out = _digest_select(conn, name, f"SELECT {', '.join(cols)} FROM {name}")
        elif kind == "object":
            cols = _table_columns(conn, name)
            drop = [c for c in CLOCK_DERIVED_COLUMNS.get(name, ()) if c in cols]
            out = _digest_select(conn, name, f"SELECT {', '.join(c for c in cols if c not in drop)} FROM {name}" if drop else
                                 f"SELECT * FROM {name}")
            if drop:
                out["left_out"] = drop
        else:
            out = _extras(conn)
    out["seconds"] = round(time.monotonic() - t0, 3)
    return kind, name, out


def _extras(conn) -> Dict[str, object]:
    out: Dict[str, object] = {}
    tables = set(_db_tables(conn))
    if "schema_migration" in tables:
        out["schema_migration"] = [list(r) for r in conn.execute("SELECT version, applied_utc, description FROM schema_migration ORDER BY version")]
    if "source_revision" in tables:
        cols = [c for c in _table_columns(conn, "source_revision") if c != "reclaimed"]
        sr = _digest_select(conn, "source_revision", f"SELECT {', '.join(cols)} FROM source_revision")
        sr.pop("columns", None)
        out["source_revision"] = {
            "without_reclaimed": sr,
            "reclaimed_nonzero": conn.execute("SELECT COUNT(*) FROM source_revision WHERE reclaimed <> 0").fetchone()[0],
            "reclaimed_mismatch": conn.execute(
                f"SELECT COUNT(*) FROM source_revision WHERE reclaimed <> (CASE WHEN {P.accepted_superseded()} THEN 1 ELSE 0 END)"
            ).fetchone()[0],
            # what the deletion contract checks of the flag: a 1 only on an accepted, superseded row, and a 0 on such a row only while
            # item rows of it remain -- a revision with a kept carrier whose log file is missing stays due by design (it is planned again
            # when the file returns), and the additive check against the archive already proves every deletable carrier went
            "reclaimed_unexplained": conn.execute(
                f"SELECT COUNT(*) FROM source_revision s WHERE (s.reclaimed <> 0 AND NOT ({P.accepted_superseded('s')})) "
                f"OR (s.reclaimed = 0 AND {P.accepted_superseded('s')} AND NOT ("
                + " OR ".join(f"EXISTS (SELECT 1 FROM {t} WHERE source_revision_id = s.revision_id)" for t in P.PRUNE_TABLES) + "))"
            ).fetchone()[0]}
    if P.LEDGER_TABLE in tables:
        out[P.LEDGER_TABLE] = [list(r) for r in conn.execute(
            f"SELECT table_name, revision_id, record_file_id, rows_deleted FROM {P.LEDGER_TABLE} ORDER BY 1, 2, 3")]
    return {"value": out}


def projection_note() -> Dict[str, object]:
    """What a digest document leaves out of its object digests (:data:`CLOCK_DERIVED_COLUMNS`), as the document records it: two documents
    compare only under the same projection (``equality.check_contract``)."""
    return {"clock_derived_columns_left_out": {k: list(v) for k, v in sorted(CLOCK_DERIVED_COLUMNS.items())}}


def digest_database(db_path: str, out: Optional[str] = None, objects: str = "all", jobs: int = 1,
                    progress: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """The digest document of ``db_path`` (module doc); ``objects`` = ``all``, ``none`` or a comma list of ``v_*`` names; ``jobs``
    worker processes (1 = inline). Written to ``out`` when given."""
    t0 = time.monotonic()
    with DB.reading(db_path) as conn:
        ident = source_identity(conn, db_path)
        tables = _db_tables(conn)
        all_objects = _db_objects(conn)
        schema = schema_entries(conn)
    if objects == "all":
        objs = all_objects
    elif objects in ("", "none"):
        objs = []
    else:
        objs = [o.strip() for o in objects.split(",") if o.strip()]
        unknown = sorted(set(objs) - set(all_objects))
        if unknown:
            raise ArchiveError(f"digest: unknown v_* object(s) {unknown}")
    tasks = [("table", t) for t in tables] + [("object", o) for o in objs] + [("extras", "")]
    results: List[Tuple[str, str, Dict[str, object]]] = []
    if jobs <= 1:
        for kind, name in tasks:
            results.append(_digest_task(db_path, kind, name))
            if progress is not None:
                progress(f"digest: {kind} {name} {results[-1][2].get('rows', '')} ({time.monotonic() - t0:.0f} s)")
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            futs = [ex.submit(_digest_task, db_path, kind, name) for kind, name in tasks]
            for f in futs:
                results.append(f.result())
                if progress is not None:
                    progress(f"digest: {results[-1][0]} {results[-1][1]} ({time.monotonic() - t0:.0f} s)")
    doc: Dict[str, object] = {"format": DIGEST_FORMAT, "row_digest": P.ROW_DIGEST_TAG.rstrip(b"\0").decode(), "source": ident,
                              "taken_utc": R.utc_now(), "tool": tool_tree_sha(), "schema": schema, "tables": {}, "objects": {}, "extras": {},
                              "projection": projection_note()}
    for kind, name, res in results:
        if kind == "table":
            doc["tables"][name] = res  # type: ignore[index]
        elif kind == "object":
            doc["objects"][name] = res  # type: ignore[index]
        else:
            doc["extras"] = res["value"]
    doc["seconds"] = round(time.monotonic() - t0, 3)
    if out:
        _write_json(out, doc)
    return doc


# --------------------------------------------------------------------------------------------
# the reclaim mode, status, verify-deleted, reclaim on a non-live file
# --------------------------------------------------------------------------------------------


#: The mode file as written and whether it validates now (``prune.read_mode_state``).
read_mode_state = P.read_mode_state


def _present(conn, table: str, revision_id: str, record_file_id: int) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE source_revision_id = ? AND record_file_id = ?",
                            (revision_id, int(record_file_id))).fetchone()[0])


def _has_rows(conn, table: str, revision_id: str, record_file_id: int) -> bool:
    """Whether the carrier group has any row: one seek of ``idx_<t>_revision``, never a count."""
    return conn.execute(f"SELECT 1 FROM {table} WHERE source_revision_id = ? AND record_file_id = ? LIMIT 1",
                        (revision_id, int(record_file_id))).fetchone() is not None


def cohort_coverage(conn, mode: P.ReclaimMode) -> Dict[str, object]:
    """Per bound archive group, whether rows remain (an indexed count each): the cohort catch-up's progress."""
    total = remaining_groups = remaining_rows = 0
    for (t, rev, rf) in mode.groups:
        total += 1
        n = _present(conn, t, rev, rf)
        if n:
            remaining_groups += 1
            remaining_rows += n
    return {"groups": total, "groups_with_rows": remaining_groups, "rows_remaining": remaining_rows}


def normal_obligation(conn, root: str, cmode: P.ReclaimMode, first_only: bool = False) -> Dict[str, List[GroupKey]]:
    """What still stands between the cohort ``cmode`` (its verified archives) and normal mode, over EVERY accepted, superseded revision
    (re-parses their remaining carriers; read-only): ``uncovered`` -- the carriers the normal rule would delete that no verified archive
    binds (a supersession after the archives, a changed or rehydrated group: a delta archive is due); ``deletable`` -- carriers a verified
    archive binds that still have rows (the cohort's own deletion is not finished); ``unsettled`` -- carriers whose file changed while it
    was read (classified again on the next call). Normal mode would delete the first two without asking any archive, so ``mode normal``
    waits until all three are empty. ``first_only`` stops at the first revision with anything to report."""
    out: Dict[str, List[GroupKey]] = {"uncovered": [], "deletable": [], "unsettled": []}
    for rev in superseded_revisions(conn):
        for p in P.revision_plan(conn, root, rev, cmode):
            k = (p.table, p.revision_id, int(p.record_file_id))
            if p.klass == P.COHORT_UNCOVERED:
                out["uncovered"].append(k)
            elif p.klass == P.DELETABLE:
                out["deletable"].append(k)
            elif p.transient:
                out["unsettled"].append(k)
        if first_only and any(out.values()):
            break
    return out


def _obligation_refusal(left: Mapping[str, Sequence[GroupKey]]) -> Optional[str]:
    unc, dele, uns = left["uncovered"], left["deletable"], left["unsettled"]
    parts = []
    if unc:
        parts.append(f"{len(unc)} normally deletable carrier group(s) no verified archive covers (first: {unc[0]}): archive them "
                     "(`status --uncovered-out F`, `archive --groups F`, an independent `verify-archive`), add the archive with `mode cohort`, "
                     "and let the cohort delete them")
    if dele:
        parts.append(f"{len(dele)} cohort group(s) still have rows and are still deletable (first: {dele[0]}): the cohort catch-up is not "
                     "finished")
    if uns:
        parts.append(f"{len(uns)} carrier group(s) changed while they were read (first: {uns[0]}): run it again")
    return "mode normal: " + "; ".join(parts) if parts else None


def _obligation_entries(root: str) -> List[Dict[str, object]]:
    """The verified archives a switch to ``normal`` must find covering every normally deletable carrier, as the mode file names them: a
    cohort file's ``manifests``, a hold file's :data:`prune.OBLIGATION_KEY` (a hold entered from cohort mode keeps them); ``[]`` when there
    is no mode file or it records no cohort. An unreadable file is an obligation that cannot be read: ``[{"unreadable": reason}]``."""
    path = P.ReclaimMode.state_path(root)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            st = json.load(fh)
    except (OSError, ValueError) as exc:
        return [{"unreadable": f"{path}: unreadable ({exc})"}]
    if not isinstance(st, dict) or st.get("format") != P.MODE_STATE_FORMAT or st.get("mode") not in P.MODES:
        return [{"unreadable": f"{path}: not a {P.MODE_STATE_FORMAT} mode file"}]
    entries = st.get("manifests") if st["mode"] == "cohort" else st.get(P.OBLIGATION_KEY) if st["mode"] == "hold" else None
    if entries is not None and not isinstance(entries, list):
        return [{"unreadable": f"{path}: its archive list is not a list"}]
    if st["mode"] == "cohort" and not entries:
        return [{"unreadable": f"{path}: cohort mode names no archive"}]
    return [dict(e) if isinstance(e, dict) else {"unreadable": f"{path}: an archive entry is not an object"} for e in (entries or [])]


def _obligation_mode(entries: Sequence[Mapping[str, object]]) -> P.ReclaimMode:
    """The cohort mode the obligation's archives bind (each re-validated locally by path and MANIFEST sha256, as the live cycle reads a
    cohort mode file); raises :class:`ArchiveError` when the obligation cannot be read or an archive no longer validates."""
    bad = [str(e["unreadable"]) for e in entries if "unreadable" in e]
    if bad:
        raise ArchiveError(f"mode normal: the cohort obligation cannot be read ({bad[0]}): re-establish the cohort with `mode cohort --manifest "
                           "...` (every verified archive), then leave it for normal")
    try:
        cmode = P.ReclaimMode.cohort([str(e.get("dir")) for e in entries])
    except P.ModeStateInvalid as exc:
        raise ArchiveError(f"mode normal: an archive the cohort was bound to no longer validates ({exc}): restore the archive copy it "
                           "names; the check that every deleted row is archived cannot run without it")
    for e, have in zip(entries, cmode.manifest_shas):
        if e.get("manifest_sha256") != have:
            raise ArchiveError(f"mode normal: {e.get('dir')}: MANIFEST sha256 {have} is not the {e.get('manifest_sha256')} the cohort was "
                               "set with")
    return cmode


def status(root: str, db_path: Optional[str] = None, census: bool = False, uncovered_out: Optional[str] = None,
           uncovered: bool = False, coverage: bool = True) -> Dict[str, object]:
    """The pruning's state, cheap by default (the mode, pending revisions, the ledger and its total ``ledger_rows_deleted``, held
    rehydrations, the cohort's progress by an indexed count per bound group, the last reclaim metrics); ``coverage=False``
    (``--no-coverage``) leaves out the cohort's progress (its ``cohort`` entry), the one part whose cost grows with the archive.
    ``census`` adds the full classification (re-parses every candidate carrier). ``uncovered`` (cohort mode only) lists the
    ``cohort_uncovered`` groups -- the normally deletable carriers no verified archive binds, over every accepted, superseded revision
    (:func:`normal_obligation`, which re-parses their remaining carriers; what ``mode normal`` checks) -- under ``uncovered``, one
    ``{table, revision_id, record_file_id}`` each; ``uncovered_out`` does the same and also writes them as a ``--groups`` file for a
    delta archive."""
    db_path = db_path or DB.db_path(root)
    mode, mst = P.mode_state(root)
    out: Dict[str, object] = {"root": os.path.abspath(root), "db": os.path.abspath(db_path), "mode": mst,
                              "cohort_obligation": _obligation_entries(root)}
    with DB.reading(db_path) as conn:
        conn.execute("BEGIN")
        out["superseded_revisions"] = conn.execute(
            f"SELECT COUNT(*) FROM source_revision WHERE {P.accepted_superseded()}").fetchone()[0]
        out["pending_revisions"] = conn.execute(
            f"SELECT COUNT(*) FROM source_revision WHERE {P.accepted_superseded()} AND reclaimed = 0").fetchone()[0]
        ledger = P.ledger_totals(conn)
        out["ledger"] = ledger
        out["ledger_rows_deleted"] = sum(int(e["rows_deleted"]) for e in ledger.values())
        # a held activation (write points 1-2: kind = rehydration), a held chain-moving record and a blocked revision-tagged record
        # (write point 3) alike
        out["held_rehydrations"] = [
            {"record_uri": r[0], "revision": r[1], "record_type": r[2], "kind": r[3], "reason": r[4]} for r in conn.execute(
                "SELECT record_uri, missing_parent_id, record_type, json_extract(extra_json, '$.kind'), json_extract(extra_json, '$.blocked_on') "
                "FROM blocked_records WHERE missing_parent_table = ? ORDER BY record_uri", (P.LEDGER_TABLE,))]
        if coverage and mode is not None and mode.kind == "cohort":
            out["cohort"] = cohort_coverage(conn, mode)
        if census:
            out["census"] = P.census(conn, root, mode=mode or P.ReclaimMode.normal())
        if uncovered or uncovered_out:
            if mode is None or mode.kind != "cohort":
                raise ArchiveError("status --uncovered / --uncovered-out needs a valid cohort mode")
            unc = set(normal_obligation(conn, root, mode)["uncovered"])
            if uncovered_out:
                write_groups_filter(uncovered_out, unc,
                                    note="cohort_uncovered groups of the accepted, superseded revisions (registry prune-superseded status)")
            out["uncovered_groups"] = len(unc)
            out["uncovered"] = [{"table": t, "revision_id": rev, "record_file_id": rf} for (t, rev, rf) in sorted(unc)]
        conn.execute("COMMIT")
    tail = P.metrics_tail(root)
    stops = [m for m in tail if m.get("kind") == "reclaim_stop"]
    out["last_reclaim_stop"] = stops[-1] if stops else None
    # the deletion batches' rows only: pass (i)'s un-mark lines log the same kind with the batch size, not rows deleted
    out["recent_reclaim_rows"] = sum(int(m.get("rows") or 0) for m in tail if P.is_deletion_batch(m))
    out["recent_rehydrations"] = [m for m in tail if str(m.get("kind", "")).startswith("rehydrat")][-10:]
    return out


#: ``verify-deleted`` counts the rows of at most this many failing groups (the ones it reports)
VERIFY_DELETED_REPORTED = 50


def verify_deleted(db_path: str, manifests: Sequence[str]) -> Dict[str, object]:
    """After the cohort: every archived group is empty, or its revision is no longer accepted and superseded. One seek per group (does
    it have a row?) in one read snapshot; the rows are counted only for the failures reported."""
    groups: Dict[GroupKey, P.ArchiveGroup] = {}
    for d in manifests:
        for g in P.read_groups(os.path.abspath(d)):
            groups[(g.table, g.revision_id, g.record_file_id)] = g
    empty = moved = 0
    failures: List[Dict[str, object]] = []
    superseded: Dict[str, bool] = {}
    with DB.reading(db_path) as conn:
        conn.execute("BEGIN")
        for (t, rev, rf), g in sorted(groups.items()):
            if not _has_rows(conn, t, rev, rf):
                empty += 1
                continue
            if rev not in superseded:
                superseded[rev] = P.is_accepted_superseded(conn, rev)
            if not superseded[rev]:
                moved += 1
            else:
                failures.append({"table": t, "revision_id": rev, "record_file_id": rf, "rows_archived": g.rows})
        for f in failures[:VERIFY_DELETED_REPORTED]:
            f["rows_present"] = _present(conn, str(f["table"]), str(f["revision_id"]), int(f["record_file_id"]))  # type: ignore[arg-type]
        conn.execute("COMMIT")
    return {"ok": not failures, "groups": len(groups), "empty": empty, "revision_no_longer_superseded": moved,
            "failures": failures[:VERIFY_DELETED_REPORTED], "n_failures": len(failures)}


def set_mode(root: str, kind: str, manifests: Sequence[str] = (), db_path: Optional[str] = None, lock_timeout_s: float = 900.0,
             now: Optional[str] = None) -> Dict[str, object]:
    """Write the reclaim mode (``state/reclaim_mode.json``; ``normal`` removes it). ``cohort`` gates every manifest locally first
    (:func:`gate_local_archive`). ``hold`` entered from a cohort keeps that cohort's archives in the hold file (the obligation survives
    a pause). ``normal`` leaving a cohort -- directly or through a hold -- refuses while :func:`normal_obligation` finds anything over
    every accepted, superseded revision: a normally deletable carrier no verified archive covers (a delta archive is due), or one a
    verified archive covers that still has rows (the catch-up is not finished). The check runs once without the lock (the refusal costs
    the harvester nothing) and again under the writer lock, in one read snapshot held until the mode file is gone: no cycle can supersede
    a revision or change a carrier between the check and the switch."""
    if kind not in P.MODES:
        raise ArchiveError(f"mode: {kind!r} is not one of {P.MODES}")
    now = now or R.utc_now()
    db_path = db_path or DB.db_path(root)
    out: Dict[str, object] = {"mode": kind}
    if kind == "cohort":
        if not manifests:
            raise ArchiveError("mode cohort needs at least one --manifest DIR (a local copy of a verified archive)")
        out["verified"] = [gate_local_archive(d, now) for d in manifests]
        mode = P.ReclaimMode.cohort([os.path.abspath(d) for d in manifests])
    elif manifests:
        raise ArchiveError(f"mode {kind} takes no --manifest")
    elif kind == "hold":
        mode = P.ReclaimMode.hold()
    else:
        mode = P.ReclaimMode.normal()
    if kind != "normal":
        with DB.write_lock(root, timeout_s=lock_timeout_s):
            if kind == "hold":
                entries = _obligation_entries(root)
                out["cohort_obligation"] = entries
                out["path"] = mode.write_state(root, now, obligation=entries)
            else:
                out["path"] = mode.write_state(root, now)
    else:
        # the first check, without the lock: a refusal here costs the harvester nothing. It decides nothing -- the mode file read under
        # the lock below does, whatever this read found (a plain hold here may be a cohort by the time the lock is held)
        first = _obligation_entries(root)
        if first:
            with DB.reading(db_path) as conn:
                refusal = _obligation_refusal(normal_obligation(conn, root, _obligation_mode(first)))
            if refusal:
                raise ArchiveError(refusal)
        with DB.write_lock(root, timeout_s=lock_timeout_s):
            # the mode file as it stands under the lock every `mode` command writes under: an obligation it carries now (a cohort, or a
            # hold that paused one) is checked in the snapshot the switch is made in; one that cannot be read refuses
            entries = _obligation_entries(root)
            moved = entries != first
            cmode = _obligation_mode(entries) if entries else None
            with DB.reading(db_path) as conn:
                conn.execute("BEGIN")
                if cmode is not None:
                    refusal = _obligation_refusal(normal_obligation(conn, root, cmode, first_only=True))
                    if refusal:
                        raise ArchiveError(refusal + (" (found under the writer lock: the mode was set again since the first check)" if moved
                                                      else " (found under the writer lock: it changed since the first check)"))
                out["path"] = mode.write_state(root, now)
                conn.execute("COMMIT")
        if entries:
            out["obligation_checked"] = [str(e.get("dir")) for e in entries]
    out["manifests"] = [{"dir": d, "manifest_sha256": s} for d, s in zip(mode.manifests, mode.manifest_shas)]
    return out


def mode_for(kind: str, manifests: Sequence[str] = (), now: Optional[str] = None) -> P.ReclaimMode:
    """An explicit mode for a non-live reclaim; a cohort manifest without a current ``VERIFIED.local.json`` is gated first."""
    if kind == "normal":
        return P.ReclaimMode.normal()
    if kind == "hold":
        return P.ReclaimMode.hold()
    if kind != "cohort":
        raise ArchiveError(f"reclaim: {kind!r} is not one of {P.MODES}")
    if not manifests:
        raise ArchiveError("reclaim --mode cohort needs at least one --manifest DIR")
    for d in manifests:
        if not _locally_verified(d):
            gate_local_archive(d, now)
    return P.ReclaimMode.cohort([os.path.abspath(d) for d in manifests])


def reclaim_file(db_path: str, root: str, mode: P.ReclaimMode, budget_s: Optional[float] = None, now: Optional[str] = None) -> Dict[str, object]:
    """The fold's reclaim (``harvest.Ingestor.reclaim``) on a NON-live database file, with an explicit mode; ``root`` resolves the
    record log. The live root's database is refused."""
    from . import harvest as H
    reason = live_reason(db_path, root)
    if reason:
        raise LiveRefused(f"reclaim: {reason}; the live registry reclaims in its own cycle under state/reclaim_mode.json")
    if not os.path.exists(db_path):
        raise ArchiveError(f"reclaim: no database at {db_path}")
    now = now or R.utc_now()
    t0 = time.monotonic()
    conn = DB.connect(db_path)
    try:
        DB.require_shape(conn, db_path)
        rep = H.IngestReport()
        ing = H.Ingestor(conn, root, rep, live=False)
        touched = ing.reclaim(now, mode, budget_s=budget_s)
        try:
            DB.wal_checkpoint_truncate(conn)
        except Exception:
            pass
        pending = conn.execute(f"SELECT COUNT(*) FROM source_revision WHERE {P.accepted_superseded()} AND reclaimed = 0").fetchone()[0]
        ledger = {t: v for t, v in P.ledger_totals(conn).items() if v["carriers"]}
    finally:
        conn.close()
    return {"db": os.path.abspath(db_path), "mode": mode.kind, "rows_touched": touched, "rows_deleted": rep.reclaim_rows_deleted,
            "transactions": rep.reclaim_transactions, "stopped": rep.reclaim_stopped, "pending_revisions": pending, "ledger": ledger,
            "seconds": round(time.monotonic() - t0, 3)}


def plan(db_path: str, root: str, mode: Optional[P.ReclaimMode] = None, revisions: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """The census (``prune.census``) by table x class x adapter, read-only, in one snapshot."""
    with DB.reading(db_path) as conn:
        conn.execute("BEGIN")
        out = P.census(conn, root, revisions=list(revisions) if revisions else None, mode=mode)
        conn.execute("COMMIT")
    return out


# --------------------------------------------------------------------------------------------
# the CLI: registry prune-superseded <action>
# --------------------------------------------------------------------------------------------


def add_arguments(s) -> None:
    sub = s.add_subparsers(dest="action", required=True)

    def with_root(p):
        # `--root` after the action as well as before the subcommand: SUPPRESS keeps an absent sub-level value from clobbering the
        # top-level one
        p.add_argument("--root", default=argparse.SUPPRESS, help="registry root (resolves the record log)")
        return p

    p = with_root(sub.add_parser("plan", help="the census of every accepted, superseded revision's carriers by table, class and adapter "
                                              "(read-only; re-parses every candidate carrier)"))
    p.add_argument("--db", help="database file (default <root>/registry.sqlite)")
    p.add_argument("--mode", choices=["normal", "cohort"], default="normal")
    p.add_argument("--manifest", action="append", default=[], help="a local verified archive (cohort)")
    p.add_argument("--revision", action="append", default=[], help="limit to these revisions")
    p.add_argument("--out", help="write the JSON here too")

    p = with_root(sub.add_parser("archive", help="export every deletable carrier group, in one read snapshot, to a compressed archive "
                                                 "directory (row files, groups, MANIFEST, SHA256SUMS)"))
    p.add_argument("--db", help="database file (default <root>/registry.sqlite)")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="the archive directory to create")
    where.add_argument("--out-parent", help="create <parent>/<kind>-<utc>-<dbid8>/ (needs --kind; the name is printed)")
    p.add_argument("--groups", help="a prune-groups/1 file: export only these groups (a delta archive)")
    p.add_argument("--kind", default="", help="rehearsal | live-w1 | live-delta-<n> (recorded in the MANIFEST)")
    p.add_argument("--hash-source", action="store_true", help="record the database file's sha256 in the MANIFEST")

    p = with_root(sub.add_parser("verify-archive", help="re-read an archive: SHA256SUMS, every row, group and table digest; with --db, the "
                                                        "database's deletable groups must equal the archive's; writes VERIFIED.json"))
    p.add_argument("dir")
    p.add_argument("--db", help="also compare with this database's deletable groups (one snapshot)")
    p.add_argument("--local", action="store_true", help="write VERIFIED.local.json instead of VERIFIED.json")

    p = with_root(sub.add_parser("restore", help="insert every archived row whose key is absent (ingest_seq = the carrier's current rank); "
                                                 "a present key is reported, never overwritten"))
    p.add_argument("dir")
    p.add_argument("--db", required=True)
    p.add_argument("--live", action="store_true", help="allow the live registry's database (needs its maintenance flag and reclaim mode hold)")

    p = with_root(sub.add_parser("digest", help="(rows, digest) per table and per v_* object, plus the contracts' extras, as JSON"))
    p.add_argument("--db", help="database file (default <root>/registry.sqlite)")
    p.add_argument("--out", required=True)
    p.add_argument("--objects", default="all", help="all | none | a comma list of v_* objects")
    p.add_argument("--jobs", type=int, default=1)

    p = with_root(sub.add_parser("compare", help="check a comparison contract: migration | deletion | identity between two digest files, or "
                                                 "rebuild equality between two databases (streamed primary-key merge)"))
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--expect", required=True, choices=list(EQ.CONTRACTS) + ["rebuild"])
    p.add_argument("--archive", help="the archive directory (deletion)")
    p.add_argument("--diff-out", help="rebuild: write every differing key here (JSON lines)")
    p.add_argument("--baseline-diff", help="rebuild: a baseline diff file; only differences it does not show fail")
    p.add_argument("--legacy-schema", type=int, help="rebuild: both files are at this OLDER schema version (32: before migration 0033); "
                                                    "compares the code's tables that schema has (all but item_reclaim_ledger)")
    p.add_argument("--out", help="write the result JSON here too")

    p = with_root(sub.add_parser("mode", help="set the live reclaim mode: hold (delete nothing; a cohort it pauses stays owed), cohort "
                                              "(only groups bound to a locally re-verified archive), normal (every deletable carrier; "
                                              "refused while a cohort is owed and a normally deletable carrier is unarchived or undeleted)"))
    p.add_argument("kind", choices=list(P.MODES))
    p.add_argument("--manifest", action="append", default=[], help="a LOCAL copy of a verified archive (cohort; repeatable)")
    p.add_argument("--lock-timeout", type=float, default=900.0)

    p = with_root(sub.add_parser("status", help="the mode, pending revisions, the ledger, held rehydrations and the cohort's progress "
                                                "(cheap; --no-coverage leaves the progress out); --census adds the full classification"))
    p.add_argument("--db")
    p.add_argument("--census", action="store_true")
    p.add_argument("--uncovered", action="store_true", help="list the cohort_uncovered groups of the pending revisions (table, revision_id, "
                                                             "record_file_id) under `uncovered`; re-parses every pending revision's carriers")
    p.add_argument("--uncovered-out", help="also write those groups as a --groups file")
    p.add_argument("--no-coverage", action="store_true", help="leave out the cohort's progress (the `cohort` entry: one indexed count per "
                                                               "bound archive group), the one part whose cost grows with the archive")
    p.add_argument("--json", action="store_true", help="accepted for scripts; the output is always JSON")

    p = with_root(sub.add_parser("verify-deleted", help="after the cohort: every archived group is empty, or its revision is no longer "
                                                        "accepted and superseded"))
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--db")

    p = with_root(sub.add_parser("reclaim", help="the fold's reclaim on a NON-live database file with an explicit mode (rehearsal, carry)"))
    p.add_argument("--db", required=True)
    p.add_argument("--mode", required=True, choices=list(P.MODES))
    p.add_argument("--manifest", action="append", default=[])
    p.add_argument("--budget-s", type=float)

    p = with_root(sub.add_parser("rehydrate", help="restore a revision's deleted carriers when a rehydration is held: --from-archive "
                                                   "(exact bytes) or --accept-reparse --reason (the current code's rows, logged)"))
    p.add_argument("--revision", required=True)
    p.add_argument("--from-archive", action="append", default=[], dest="from_archive")
    p.add_argument("--accept-reparse", action="store_true")
    p.add_argument("--reason", default="")
    p.add_argument("--decisions-log", help="also append the decision to this file")
    p.add_argument("--db", help="database file (default <root>/registry.sqlite)")
    p.add_argument("--lock-timeout", type=float, default=600.0)


def _emit(obj: object, out: Optional[str] = None) -> None:
    text = json.dumps(obj, indent=1, sort_keys=True, default=str)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


def _err(msg: str) -> None:
    print(f"registry prune-superseded: {msg}", file=sys.stderr)


def cmd_prune_superseded(args, root: str) -> int:
    """Exit codes: 0 done / check passed; 1 a check failed (verify, compare, verify-deleted, restore conflicts); 2 refused (nothing
    written)."""
    a = args.action
    progress = (lambda m: print(m, file=sys.stderr, flush=True))
    db = getattr(args, "db", None) or DB.db_path(root)
    try:
        if a == "plan":
            # a read-only census: a cohort manifest must already be locally verified (``mode cohort`` / ``reclaim`` gate it)
            mode = P.ReclaimMode.cohort([os.path.abspath(d) for d in args.manifest]) if args.mode == "cohort" else P.ReclaimMode.normal()
            _emit(plan(db, root, mode, args.revision or None), args.out)
            return 0
        if a == "archive":
            groups = load_groups_filter(args.groups) if args.groups else None
            out_dir = args.out or os.path.join(args.out_parent, archive_name(db, args.kind))
            m = archive(db, root, out_dir, groups=groups, kind=args.kind, hash_source=args.hash_source, progress=progress)
            _emit({"archive": os.path.abspath(out_dir), "tables": {t: {k: e[k] for k in ("rows", "groups", "bytes")} for t, e in m["tables"].items()},
                   "groups": m["groups"]["count"], "skipped_partial": len(m["skipped_partial"]), "census": m["census"], "seconds": m["seconds"]})
            return 0
        if a == "verify-archive":
            try:
                # the record log the database comparison re-parses: --root when given, else the root the archive was taken under
                rec = verify_archive(args.dir, db_path=args.db, root=root if getattr(args, "root", None) else None,
                                     verified_name=P.VERIFIED_LOCAL_FILE if args.local else P.VERIFIED_FILE)
            except ArchiveInvalid as exc:
                _emit({"ok": False, "failures": exc.failures})
                return 1
            _emit({"ok": True, **rec})
            return 0
        if a == "restore":
            res = restore(args.dir, args.db, root=root if getattr(args, "root", None) else None, live=args.live)
            _emit(res)
            return 0 if res["ok"] else 1
        if a == "digest":
            digest_database(db, args.out, objects=args.objects, jobs=args.jobs, progress=progress)
            _emit({"out": os.path.abspath(args.out)})
            return 0
        if a == "compare":
            if args.expect == "rebuild":
                res = compare_databases(args.a, args.b, args.diff_out, args.baseline_diff, legacy_schema=args.legacy_schema)
            elif args.legacy_schema is not None:
                raise ArchiveError("compare --legacy-schema applies to --expect rebuild only")
            else:
                da, dbb = _read_json(args.a), _read_json(args.b)
                man = grp = None
                if args.expect == "deletion":
                    if not args.archive:
                        raise ArchiveError("compare --expect deletion needs --archive DIR")
                    man = read_manifest(args.archive)
                    grp = group_dicts(args.archive)
                res = EQ.check_contract(da, dbb, args.expect, man, grp)  # type: ignore[arg-type]
            _emit(res, args.out)
            return 0 if res["ok"] else 1
        if a == "mode":
            _emit(set_mode(root, args.kind, args.manifest, lock_timeout_s=args.lock_timeout))
            return 0
        if a == "status":
            _emit(status(root, args.db, census=args.census, uncovered_out=args.uncovered_out, uncovered=args.uncovered,
                         coverage=not args.no_coverage))
            return 0
        if a == "verify-deleted":
            res = verify_deleted(db, args.manifest)
            _emit(res)
            return 0 if res["ok"] else 1
        if a == "reclaim":
            _emit(reclaim_file(args.db, root, mode_for(args.mode, args.manifest), budget_s=args.budget_s))
            return 0
        if a == "rehydrate":
            _emit(rehydrate(root, args.revision, archive_dirs=args.from_archive, accept_reparse=args.accept_reparse, reason=args.reason,
                            decisions_log=args.decisions_log, db_path=args.db, lock_timeout_s=args.lock_timeout))
            return 0
    except ArchiveInvalid as exc:
        _err(str(exc))
        for f in exc.failures[:50]:
            print(f"  {f}", file=sys.stderr)
        return 2
    except (ArchiveError, P.ModeStateInvalid, DB.LockHeld, DB.SchemaDrift) as exc:
        _err(f"refused: {exc}")
        return 2
    raise ValueError(f"unknown action {a}")


def _rebuild_tables(a, b, a_path: str, b_path: str, legacy_schema: Optional[int]) -> Optional[List[str]]:
    """The tables a rebuild comparison reads (None = every table the code declares), after checking both files hold each of them with
    every compared column; with ``legacy_schema``, both files must be at that older schema and the tables are the ones it has
    (``equality.legacy_tables``). Raises :class:`ArchiveError` naming what is missing."""
    from .migrations import db_version
    tables: Optional[List[str]] = None
    if legacy_schema is not None:
        if legacy_schema not in EQ.LEGACY_SCHEMAS:
            raise ArchiveError(f"compare --legacy-schema {legacy_schema}: not a declared legacy schema (one of {sorted(EQ.LEGACY_SCHEMAS)})")
        for path, conn in ((a_path, a), (b_path, b)):
            have = db_version(conn)
            if have != legacy_schema:
                raise ArchiveError(f"compare --legacy-schema {legacy_schema}: {path} is at schema {have}")
        tables = EQ.legacy_tables(legacy_schema)
    missing: List[str] = []
    for t, cols, _key in EQ.compared_tables(tables=tables):
        for path, conn in ((a_path, a), (b_path, b)):
            have = set(_table_columns(conn, t))
            if not have:
                missing.append(f"{path} has no table {t}")
            elif set(cols) - have:
                missing.append(f"{path}: {t} lacks {sorted(set(cols) - have)}")
    if missing:
        hint = "" if legacy_schema is not None else (f"; two files at schema {sorted(EQ.LEGACY_SCHEMAS)} compare with --legacy-schema "
                                                     "(the tables that schema has)")
        raise ArchiveError(f"compare --expect rebuild: {'; '.join(missing[:6])}{' …' if len(missing) > 6 else ''}{hint}")
    return tables


def compare_databases(a_path: str, b_path: str, diff_out: Optional[str] = None, baseline_diff: Optional[str] = None,
                      legacy_schema: Optional[int] = None) -> Dict[str, object]:
    """Rebuild equality of two database files (``equality.verify_equal``, rebuild's exclusions) over every table the code declares, each
    required on both sides -- or, with ``legacy_schema``, two files at that older schema over the tables it has (``--legacy-schema``);
    with ``baseline_diff``, only a difference the baseline does not show identically fails (it needs ``diff_out``)."""
    if baseline_diff and not diff_out:
        raise ArchiveError("compare --baseline-diff needs --diff-out")
    with DB.reading(a_path) as a, DB.reading(b_path) as b:
        tables = _rebuild_tables(a, b, a_path, b_path, legacy_schema)
        with (EQ.DiffWriter(diff_out, os.path.abspath(a_path), os.path.abspath(b_path)) if diff_out else contextlib.nullcontext()) as w:
            diff = EQ.verify_equal(a, b, tables=tables, diff_sink=w)
    bad = {k: v for k, v in diff.items() if v}
    out: Dict[str, object] = {"expect": "rebuild", "a": os.path.abspath(a_path), "b": os.path.abspath(b_path), "differing": bad,
                              "ok": not bad, "legacy_schema": legacy_schema, "tables_compared": sorted(diff)}
    if baseline_diff:
        cls = EQ.classify_diffs(diff_out, baseline_diff)  # type: ignore[arg-type]
        out["classified"] = cls
        out["ok"] = bool(cls["ok"])
    return out
