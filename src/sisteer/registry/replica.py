"""The cluster replica of the registry database: a quiescent replayable HEAD taken
on the grading host, an immutable ALLOWLIST honored by every replay, one serialized writer per replica root, versioned generations
switched by compare-and-swap, and canonical equality through ``frame_projection/1``.

**The head** (:func:`take_head`). ONE read transaction (``BEGIN`` on a read-only connection) requires a CURSOR-FREE boundary: no
``record_cursor`` row. A plain file written across several transactions commits every non-final part WITH its records applied to
live rows and a ``record_cursor`` row (``harvest._ingest_chunk``), and its delivery row is written only with the final part -- so while
a cursor exists the live rows reflect records of a file no delivery names, and no head taken then equals any replayable file set.
At a cursor-free boundary every delivered file is fully folded, and the live rows the transaction sees are exactly the fold of the
ALLOWLIST = every ``record_delivery`` row -- whatever its status: ``ok``, ``refused``, ``duplicate``, ``blocked`` (a held revision's
marker and files; their placeholder rows are what the replay produces too), ``partial`` (some records refused, the rest folded) and
``conflict`` -- taken in the SAME transaction as the projection and its counts, so head and projection name one replayable state by
construction. A cursor present → the transaction is rolled back and the command WAITS (poll every 15 s, bound ``--wait-min`` 20) and
retries; past the bound :class:`NoCursorFreeBoundary` (exit 6) names the files mid-fold. The head carries ``(file_uri, file_sha256,
status)`` per file, ``allowlist_sha256``, the projection with its rows and counts, ``taken_utc``, ``cursor_free: true``.

**The allowlist is immutable and honored everywhere**: ``harvest.rebuild(files=…)`` for the bootstrap; :func:`ingest_head` for the
increments (exactly ``allowlist − replica.record_delivery``, plus a listed file whose sha moved, in ``order_for_ingest`` order: dimension
bundles → plain chunks → revision directories grouped by ``group_revision_dirs``, each directory's files filtered by the list through
``Ingestor.allowlist``); ``read_arrival_facts(only=…)``. A file on disk the list does not name is ignored and counted
(``unlisted_ignored``). A replica holding a delivered file the head lacks refuses by name (:class:`ReplicaAhead`: the head is older than
the replica -- an out-of-order landing). A listed file absent from the mirror, or present with another logical sha, refuses before any
fold (:class:`MirrorBehind`; :func:`verify_mirror` hashes every listed file -- or, with ``--hash manifest``, trusts the mirror's latest
verified ``CHECKSUMS`` manifest for the files it names and hashes the rest).

**Canonical equality** (:func:`frame_projection`): ``frame_projection/1`` is the explicit, versioned column list of the paper frame
object ``v_paper_current`` at schema 22 minus exactly ``ingest_seq`` (``lifecycle_state`` and ``is_complete`` KEPT: every content column),
every row encoded in Python -- the three set-valued ``*_ids_json`` columns (``json_group_array(DISTINCT …)`` over unordered groups)
re-emitted as SORTED arrays, every JSON column re-encoded with sorted keys and fixed separators, floats as ``repr`` -- the encoded rows
SORTED (so the projection is independent of any row order the engine chooses), sha256 over the rows and the three counts (rows,
``in_current_paper = 1``, ``selection_state = 'selected'``) in the same pass. :func:`check` compares a database's projection with a head's:
any difference is a failure, printed per column with examples (``projection_diff``). ``tests/test_registry_replica.py`` pins the column
list to the code's frame object and fails until the version is bumped when the object's columns move.

**Generations and the one writer** (:class:`Layout`, :func:`publish`). ``<replica root>/{CURRENT, gen/<gen_id>/{registry.sqlite,
meta.json, head.json.gz}, records -> <log>/records, rulings -> <log>/rulings, config.toml, builds/, pins/}``. Every writer is a
``replica.sbatch`` job under ONE job name (``registry-replica``, ``--dependency=singleton``); the publication of a generation is a
compare-and-swap on ``CURRENT`` under the root's publish lock: ``CURRENT`` must still name the parent the generation was built from AND
the new head's allowlist must be a superset of the parent's; the last :data:`KEEP_GENERATIONS` generations are kept, ``CURRENT`` never
pruned. :func:`rollback` moves ``CURRENT`` to the current generation's parent (CAS-guarded). Readers open a node-local copy.

**The landing job** (:func:`land`): node-local copy of ``CURRENT`` → the allowlist presence/hash check → ``ingest_head`` → ``check``
(unequal → ``check.json.equal = false``, no pytest, no publish) → ``pytest -m live_db`` over the tip tree's marked modules with
``REGISTRY_LIVE_DB=<the local copy>`` and ``REGISTRY_PIN_L1=<replica root>/pins/l1_assembled_pin.jsonl`` → ``summary.json`` (skips are
accounted: more than the expected number refuses) → the ingested generation published; a REFUSED publication (the CAS lost: CURRENT moved
between the copy and the publication) is a failing status (``PUBLISH_REFUSED``, exit 10), never a pass, because the next landing would
start from a generation this one never produced. Every file the job writes carries the job's ``identity`` (its Slurm job id, the tree it
ran, the head's sha256): the grading host accepts a result only when both ``check.json`` and ``summary.json`` name the job it submitted, the tip
it landed and the head it took, and only from a COMPLETED job -- so a stale output of an earlier submission can never certify a new tip.
The output directory of a submission is immutable (``live_db_slurm.sh`` mints one per submission; ``land`` removes the outputs it owns
before writing).

**The check's classification and the cutover allowance**. ``check.json`` carries, beside the strict ``equal``, the diff's
classification (:func:`classify_projection_diff`): ``columns_differing``, ``movers`` (every differing run id; past
:data:`MOVERS_INLINE_CAP` the whole list is ``movers.json`` beside it), the rows on one side only, and ``c1_class`` -- true iff the
projection is the only failing component and it differs on the certified C1 same-stamp class ALONE: every differing column in
:data:`C1_CLASS_COLUMNS` (``lifecycle_state`` and the CASE columns that read it), no one-sided row, the counts equal, and every mover
passing the predicate refold-compare certifies the class with (:func:`mover_in_class` on a ``judged`` → ``superseded`` move,
:func:`declared_superseded_class` over the fold's own ``row_history``). ``land`` accepts a ``c1_class`` check ONLY under the operator's
declaration (``--accept-c1-class``; the ``ACCEPT_C1_CLASS`` positional of ``replica.sbatch land``; ``LAND_ACCEPT_C1_CLASS`` on the grading host)
naming the certification (refold-compare job 922769 for the cutover), records ``check_accepted_as = c1_class`` with the mover count and
the declaration, and publishes. The allowance exists for the cutover landing -- the live database still folds the pre-C1 order, so the
replica (C1's order) legitimately differs on exactly the certified runs -- and is retired once the live database adopts the C1 order
(``restamp`` of the certified keys inside a window). No other column is ever accepted, declaration or not: ``decode_policy_match`` or any
non-lifecycle column differing on any row keeps the landing refused. **The C1 measurement** (:func:`refold_compare`):
on a node-local copy of a database (:func:`stage_input`: the copy is hashed as it is read, so the record names the INPUT's sha256 and
size; the input is never written -- a published generation or a KEPT unequal build), every ``run`` key's ``row_history`` events
folded by the OLD tree's reducer and by the NEW tree's (each tree's own ``python -m sisteer.registry.replica refold-dump``), the full
folded rows compared (the MOVERS), then the copy projected with the movers folded under the OLD order and under the NEW order and the two
``frame_projection/1`` projections + the three counts diffed -- never a live ``row_history`` query. With ``--expect-head-hash`` the run
is a CERTIFICATION: PASS iff the OLD-order projection hashes to the declared head (the bootstrap's equality re-derived from this copy)
and every mover is in the declared-superseded class (``judged`` → ``superseded`` on exactly ``lifecycle_state``, from a same-stamp pair
at the key's latest ``run_state`` stamp -- :func:`declared_superseded_class`), the projection moves on movers only, the counts hold, the
copy was built under the DECLARED order, both trees are checked out clean at their declared commits, the input's independently computed
sha256 is the staged copy's, and the declared NEW hash and mover set (when given) match exactly.

Vocabulary: the words this module uses for the frame object are "the paper frame object" and its name in backticks.
"""
from __future__ import annotations

import fcntl
import glob
import gzip
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from . import db as DB
from . import records as R
from . import vocab as V
from .identity import canonical_json

HEAD_SCHEMA = "replica_head/1"
META_SCHEMA = "replica_gen/1"
CHECK_SCHEMA = "replica_check/1"
#: The versioned projection: bump when the frame object's column list changes (the test pins the list to the code's object).
PROJECTION_VERSION = "frame_projection/1"
PROJECTION_OBJECT = "v_paper_current"
PROJECTION_EXCLUDED: Tuple[str, ...] = ("ingest_seq",)
#: The 31 columns of the paper frame object at schema 22 minus exactly ``ingest_seq`` (30), in the object's order.
PROJECTION_COLUMNS: Tuple[str, ...] = (
    "run_id", "campaign", "model_id", "object_id", "eval_id", "instrument_id", "judge_id", "lifecycle_state", "is_complete",
    "strength_nominal", "is_operating_point", "operating_point_basis", "decode_source", "decode_policy_id", "decode_policy_match",
    "decode_family_default_policy_id", "decode_is_family_default", "round_id", "comparison_ids_json", "roster_entry_ids_json",
    "ruling_ids_json", "decisive_ruling_id", "decisive_action", "held_out", "selection_state", "in_current_paper", "is_current", "why",
    "coverage_state", "coverage_detail_json",
)
#: ``json_group_array(DISTINCT …)`` over unordered groups: re-emitted as sorted arrays.
SET_VALUED_JSON_COLUMNS: Tuple[str, ...] = ("comparison_ids_json", "roster_entry_ids_json", "ruling_ids_json")
JSON_COLUMNS: Tuple[str, ...] = SET_VALUED_JSON_COLUMNS + ("coverage_detail_json",)
DEFAULT_WAIT_MIN = 20.0
DEFAULT_POLL_S = 15.0
KEEP_GENERATIONS = 3
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
#: Delivery statuses the head lists (every one: a delivery is written only once a file's fold is complete).
DELIVERY_STATUSES: Tuple[str, ...] = ("ok", "refused", "duplicate", "blocked", "partial", "conflict")
PIN_L1_NAME = "l1_assembled_pin.jsonl"
PUBLISH_LOCK = ".publish.lock"
#: The files :func:`land` writes under its output directory; removed before the job starts so a reused directory holds only this job's.
LAND_OUTPUTS: Tuple[str, ...] = ("check.json", "mirror.json", "ingest.json", "junit.xml", "pytest.log", "summary.json", "movers.json")
#: :func:`write_check_json` lists this many movers inline in ``check.json``; a longer list goes whole to ``movers.json`` beside it.
MOVERS_INLINE_CAP = 500
UPDATE_SCHEMA = "replica_update/1"
MIGRATE_SCHEMA = "replica_migrate/1"
#: The files :func:`update_db` and :func:`migrate_db` write under their output directory (the database itself included); removed first,
#: so a reused directory holds only this job's.
UPDATE_OUTPUTS: Tuple[str, ...] = ("update.json", "mirror.json", "ingest.json", "check.json", "projection.json", "registry.sqlite")
MIGRATE_OUTPUTS: Tuple[str, ...] = ("migrate.json", "migrate.log", "plan.txt", "check.json", "projection.json", "registry.sqlite")
_MIGRATION_FILE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
REFOLD_SCHEMA = "refold_compare/2"
#: Projection columns of the paper frame object that are CASE expressions reading ``lifecycle_state`` (schema.sql: ``is_complete`` =
#: lifecycle IN (judged, ingested); ``is_current`` and ``why`` = the frame rules, which name a lateral lifecycle as the retiring
#: reason). A fold that moves ONLY ``lifecycle_state`` may move these in the projection and nothing else.
LIFECYCLE_DERIVED_COLUMNS: Tuple[str, ...] = ("is_complete", "is_current", "why")
#: The same-stamp pair of the declared-superseded class: an adapter emits both ``run_state`` records at one ``written_utc`` for a
#: run a later run of the same key supersedes; the pre-C1 order drew either, C1's order folds ``superseded`` last.
DECLARED_SUPERSEDED_PAIR: Tuple[str, str] = ("judged", "superseded")
#: The ONLY projection columns a check may differ on and still read ``c1_class``: the lifecycle itself and the CASE columns that
#: read it. Every other column -- ``decode_policy_match`` included -- differing on any row refuses, declaration or not.
C1_CLASS_COLUMNS: Tuple[str, ...] = ("lifecycle_state",) + LIFECYCLE_DERIVED_COLUMNS
#: :func:`land`'s record of which rule passed its check: strict equality, or the certified C1 class under an operator's declaration.
CHECK_ACCEPTED_EQUAL = "equal"
CHECK_ACCEPTED_C1_CLASS = "c1_class"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ReplicaError(RuntimeError):
    exit_code = 1


class NoCursorFreeBoundary(ReplicaError):
    """No quiescent boundary within the bound: files are mid-fold (exit 6; the landing defers and retries once)."""
    exit_code = 6


class ReplicaAhead(ReplicaError):
    """The replica holds delivered files the head does not name: the head is older than the replica (exit 7)."""
    exit_code = 7


class MigrationNeedsRebuild(ReplicaError):
    """``registry migrate`` left a shape the code's DDL does not match: the replica needs a full rebuild (exit 8)."""
    exit_code = 8


class MirrorBehind(ReplicaError):
    """A listed file is absent from the mirror or carries another logical sha (exit 9)."""
    exit_code = 9


class PublishRefused(ReplicaError):
    """The compare-and-swap on CURRENT failed, or the new allowlist is not a superset of the parent's (exit 10)."""
    exit_code = 10


class NoGeneration(ReplicaError):
    """No published generation to start from (exit 11): bootstrap with ``replica build`` first."""
    exit_code = 11


class RefoldInputRefused(ReplicaError):
    """refold-compare's inputs are not what a certification can stand on: a missing or half-checkpointed database, too little
    node-local disk for its copy, a malformed expected hash, a head whose projection is not the expected hash (exit 13)."""
    exit_code = 13


class SiteAliasesRefused(RefoldInputRefused):
    """The site-alias table a migration or a fold would read is refused (exit 13): unreadable, malformed, a ``[site_aliases]`` table
    inside ``config.toml``, a file that is not the one declared, a digest other than the declared one, or (``migrate --db``) a migrating
    tree whose ``registry migrate`` cannot receive the table."""
    exit_code = 13


class RestampVerifyFailed(ReplicaError):
    """``restamp-keys --apply`` rewrote the rows and the re-read does not match (exit 14): a planned (run, column) off its planned value, a
    column moved that the plan never listed, a live row that is not this tree's refold (its carrier, rank, stated columns or stamps), or a
    history snapshot that is not the fold's. The whole transaction is rolled back (nothing written); the message names every mismatch;
    the window aborts and the operator reads."""
    exit_code = 14


class SchemaBehind(ReplicaError):
    """The generation's schema version or shape is not the code's (``db.SchemaDrift``): ``replica migrate`` first (exit 12). An
    increment never migrates a database on the side."""
    exit_code = 12


# --------------------------------------------------------------------------------------------
# the projection
# --------------------------------------------------------------------------------------------


def _canon_json_text(column: str, text: str):
    """A JSON column's canonical spelling: parsed, a set-valued array sorted, re-encoded with sorted keys and fixed separators; a
    text that is not JSON is kept as it is (visible in a diff, never hidden)."""
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return text
    if column in SET_VALUED_JSON_COLUMNS and isinstance(obj, list):
        obj = sorted(obj, key=lambda x: canonical_json(x))
    return canonical_json(obj)


def encode_value(column: str, value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if column in JSON_COLUMNS and isinstance(value, str):
        return _canon_json_text(column, value)
    return value


def encode_row(row: Sequence, columns: Sequence[str] = PROJECTION_COLUMNS) -> list:
    """One frame row as a canonical list aligned with ``columns``."""
    return [encode_value(c, v) for c, v in zip(columns, row)]


def _row_line(enc: list) -> str:
    return json.dumps(enc, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def frame_projection(conn, with_rows: bool = True) -> Dict[str, object]:
    """``frame_projection/1`` of the database behind ``conn``: the sha256 over the SORTED canonical rows, the three counts, the
    version and the column list (both printed and matched on both sides), and the rows themselves when ``with_rows``."""
    cur = conn.execute(f"SELECT {', '.join(PROJECTION_COLUMNS)} FROM {PROJECTION_OBJECT}")
    got = tuple(str(d[0]) for d in cur.description)
    if got != PROJECTION_COLUMNS:
        raise ReplicaError(f"{PROJECTION_OBJECT} returned columns {got}, not {PROJECTION_VERSION}'s list")
    i_paper = PROJECTION_COLUMNS.index("in_current_paper")
    i_sel = PROJECTION_COLUMNS.index("selection_state")
    lines: List[str] = []
    in_paper = selected = 0
    for r in cur:
        row = tuple(r)
        lines.append(_row_line(encode_row(row)))
        if row[i_paper] is not None and int(row[i_paper]) == 1:
            in_paper += 1
        if row[i_sel] == "selected":
            selected += 1
    lines.sort()
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    out: Dict[str, object] = {"version": PROJECTION_VERSION, "object": PROJECTION_OBJECT, "columns": list(PROJECTION_COLUMNS),
                              "excluded": list(PROJECTION_EXCLUDED), "sha256": h.hexdigest(), "n_rows": len(lines),
                              "counts": {"rows": len(lines), "in_current_paper": in_paper, "selected": selected}}
    if with_rows:
        out["rows"] = [json.loads(line) for line in lines]
    return out


def projection_diff(a: Mapping[str, object], b: Mapping[str, object], examples: int = 5) -> Dict[str, object]:
    """Per-column differences between two projections (``a`` the expected / head side, ``b`` the database's), keyed by ``run_id``:
    rows only on one side, and for every column the count of rows whose value differs with up to ``examples`` ``(run_id, a, b)``."""
    out: Dict[str, object] = {"version_equal": a.get("version") == b.get("version"), "columns_equal": list(a.get("columns") or []) == list(b.get("columns") or []),
                              "sha_equal": a.get("sha256") == b.get("sha256"), "counts": {"a": a.get("counts"), "b": b.get("counts")}}
    ra, rb = a.get("rows"), b.get("rows")
    if ra is None or rb is None:
        out["rows_unavailable"] = True
        return out
    cols = list(a.get("columns") or PROJECTION_COLUMNS)
    ka = {row[0]: row for row in ra}
    kb = {row[0]: row for row in rb}
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    per_col: Dict[str, Dict[str, object]] = {}
    for rid in sorted(set(ka) & set(kb)):
        xa, xb = ka[rid], kb[rid]
        for i, c in enumerate(cols):
            if xa[i] != xb[i]:
                d = per_col.setdefault(c, {"rows": 0, "examples": []})
                d["rows"] = int(d["rows"]) + 1
                if len(d["examples"]) < examples:
                    d["examples"].append([rid, xa[i], xb[i]])
    out.update({"only_in_a": only_a[:50], "n_only_in_a": len(only_a), "only_in_b": only_b[:50], "n_only_in_b": len(only_b), "columns": per_col})
    return out


# --------------------------------------------------------------------------------------------
# the head
# --------------------------------------------------------------------------------------------


def allowlist_sha256(files: Iterable[Sequence[str]]) -> str:
    h = hashlib.sha256()
    for f in sorted((str(f[0]), str(f[1])) for f in files):
        h.update(f"{f[0]}\t{f[1]}\n".encode("utf-8"))
    return h.hexdigest()


def quiescence(conn) -> Tuple[int, List[str]]:
    """``(number of record_cursor rows, the first few file uris mid-fold)`` -- zero means a cursor-free boundary."""
    n = int(conn.execute("SELECT COUNT(*) FROM record_cursor").fetchone()[0])
    sample = [str(r[0]) for r in conn.execute("SELECT file_uri FROM record_cursor ORDER BY file_uri LIMIT 5")] if n else []
    return n, sample


def _head_in_transaction(conn, db_path: str, with_rows: bool) -> Dict[str, object]:
    from .migrations import code_version, db_version
    files = [(str(r[0]), str(r[1]), str(r[2])) for r in
             conn.execute("SELECT file_uri, file_sha256, status FROM record_delivery ORDER BY file_uri")]
    n_ingest, max_seq = conn.execute("SELECT COUNT(*), MAX(ingest_seq) FROM record_ingest").fetchone()
    n_unledgered = int(conn.execute("SELECT COUNT(*) FROM record_delivery d WHERE NOT EXISTS "
                                    "(SELECT 1 FROM record_ingest i WHERE i.file_uri = d.file_uri)").fetchone()[0])
    t0 = time.monotonic()
    proj = frame_projection(conn, with_rows=with_rows)
    by_status: Dict[str, int] = {}
    for _, _, st in files:
        by_status[st] = by_status.get(st, 0) + 1
    return {"schema": HEAD_SCHEMA, "taken_utc": R.utc_now(), "db_path": os.path.abspath(db_path), "cursor_free": True,
            "schema_version": db_version(conn), "code_version": code_version(), "n_files": len(files),
            "allowlist_sha256": allowlist_sha256(files), "by_status": by_status,
            "ledger": {"n_ingest": int(n_ingest or 0), "max_ingest_seq": int(max_seq or 0), "n_delivered_without_ledger_row": n_unledgered},
            "projection": proj, "projection_s": round(time.monotonic() - t0, 3), "files": files}


def take_head(db_path: str, wait_min: float = DEFAULT_WAIT_MIN, poll_s: float = DEFAULT_POLL_S, clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep, log: Optional[Callable[[str], None]] = None, with_rows: bool = True) -> Dict[str, object]:
    """The quiescent, replayable head of the database at ``db_path`` (module docstring): one read transaction; a cursor present
    rolls it back and waits (``poll_s``) up to ``wait_min`` minutes; past the bound :class:`NoCursorFreeBoundary`."""
    deadline = clock() + float(wait_min) * 60.0
    started = clock()
    polls = 0
    while True:
        conn = DB.connect(db_path, readonly=True)
        try:
            conn.execute("BEGIN")
            try:
                n, sample = quiescence(conn)
                if n == 0:
                    head = _head_in_transaction(conn, db_path, with_rows)
                    head["waited_s"] = round(clock() - started, 3)
                    head["polls"] = polls
                    return head
            finally:
                conn.execute("ROLLBACK")
        finally:
            conn.close()
        polls += 1
        if clock() >= deadline:
            raise NoCursorFreeBoundary(f"no cursor-free boundary within {float(wait_min):g} min: {n} file(s) mid-fold ({', '.join(sample)})")
        if log is not None:
            log(f"[replica head] {n} file(s) mid-fold ({', '.join(sample[:3])}); waiting {poll_s:g} s (poll {polls})")
        sleep(float(poll_s))


def write_head(head: Mapping[str, object], path: str) -> str:
    """The head as JSON (gzipped when ``path`` ends with ``.gz``), written whole through a temp name + rename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = json.dumps(head, separators=(",", ":"), sort_keys=True).encode("utf-8")
    tmp = path + ".tmp"
    if path.endswith(".gz"):
        with gzip.open(tmp, "wb") as fh:
            fh.write(data)
    else:
        with open(tmp, "wb") as fh:
            fh.write(data)
    os.replace(tmp, path)
    return path


def read_head(path: str) -> Dict[str, object]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:  # type: ignore[operator]
        head = json.loads(fh.read().decode("utf-8"))
    if head.get("schema") != HEAD_SCHEMA:
        raise ReplicaError(f"{path}: not a {HEAD_SCHEMA} head (schema {head.get('schema')!r})")
    head["files"] = [tuple(f) for f in head.get("files") or []]
    return head


def head_sha256(head: Mapping[str, object]) -> str:
    """One identity for a head file's content: its allowlist sha, projection sha and taken_utc together."""
    return hashlib.sha256(f"{head['allowlist_sha256']}\n{head['projection']['sha256']}\n{head['taken_utc']}\n".encode("utf-8")).hexdigest()


def head_uris(head: Mapping[str, object]) -> Set[str]:
    return {f[0] for f in head["files"]}


def head_shas(head: Mapping[str, object]) -> Dict[str, str]:
    return {f[0]: f[1] for f in head["files"]}


# --------------------------------------------------------------------------------------------
# the layout: generations, CURRENT, the publish lock
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    root: str

    @property
    def current_path(self) -> str:
        return os.path.join(self.root, "CURRENT")

    @property
    def gen_root(self) -> str:
        return os.path.join(self.root, "gen")

    @property
    def builds(self) -> str:
        return os.path.join(self.root, "builds")

    @property
    def pins(self) -> str:
        return os.path.join(self.root, "pins")

    @property
    def pin_l1(self) -> str:
        return os.path.join(self.pins, PIN_L1_NAME)

    @property
    def config_path(self) -> str:
        return os.path.join(self.root, "config.toml")

    @property
    def site_aliases_path(self) -> str:
        """``<root>/site_aliases.toml``: the root's alias table, written by no tool here (``init`` included)."""
        from . import site_aliases as SA
        return os.path.join(self.root, SA.FILE_NAME)

    def gen(self, gen_id: str) -> str:
        return os.path.join(self.gen_root, gen_id)

    def db_of(self, gen_id: str) -> str:
        return os.path.join(self.gen(gen_id), "registry.sqlite")

    def current(self) -> Optional[str]:
        return R.read_pointer(self.current_path)

    def meta(self, gen_id: str) -> Dict[str, object]:
        with open(os.path.join(self.gen(gen_id), "meta.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def head_of(self, gen_id: str) -> Dict[str, object]:
        return read_head(os.path.join(self.gen(gen_id), "head.json.gz"))

    def generations(self) -> List[str]:
        """Every published generation id, oldest first (by ``published_utc``, then id)."""
        out = []
        if not os.path.isdir(self.gen_root):
            return out
        for name in os.listdir(self.gen_root):
            if os.path.isfile(os.path.join(self.gen_root, name, "meta.json")):
                try:
                    out.append((str(self.meta(name).get("published_utc") or ""), name))
                except (OSError, ValueError):
                    continue
        return [n for _, n in sorted(out)]

    def log_root(self) -> str:
        """The record-log root the replay reads: ``config.toml``'s ``log_root``, else this root (its ``records``/``rulings`` links)."""
        if os.path.exists(self.config_path):
            import tomllib
            with open(self.config_path, "rb") as fh:
                cfg = tomllib.load(fh)
            lr = cfg.get("replica", {}).get("log_root") if isinstance(cfg.get("replica"), dict) else cfg.get("log_root")
            if lr:
                return str(lr)
        return self.root


_TOML_HEADER = re.compile(r"^\s*\[")
_TOML_REPLICA_HEADER = re.compile(r"^\s*\[\s*(?:replica|\"replica\"|'replica')\s*\]\s*(?:#.*)?$")
_TOML_LOG_ROOT_KEY = re.compile(r"^\s*(?:log_root|\"log_root\"|'log_root')\s*=")


def _toml_basic_string(value: str) -> str:
    """``value`` as a TOML basic string (JSON's escapes are TOML's; non-ASCII stays literal, so no surrogate pair is written)."""
    return json.dumps(value, ensure_ascii=False)


def replica_config_with_log_root(text: str, log_root: str, path: str = "config.toml") -> str:
    """``text`` (a replica root's ``config.toml``) with ``[replica] log_root`` set to ``log_root`` and every other byte kept: the
    ``log_root`` line of the ``[replica]`` table is replaced in place, inserted under the table's header when the table has none, and a
    ``[replica]`` table is appended when the file has none. The file must parse as TOML first, and the result must parse to the same
    document with only that key changed, else :class:`ReplicaError` names the file and nothing is written: replica settings written as a
    dotted key or an inline table, or a ``log_root`` value spread over several lines, are refused rather than rewritten."""
    import tomllib
    try:
        before = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ReplicaError(f"{path} does not parse as TOML ({exc}): init rewrites only [replica] log_root and keeps every other table, so it "
                           "does not overwrite a file it cannot read; repair or remove it, then init again") from None
    line = f"log_root = {_toml_basic_string(log_root)}"
    lines = text.splitlines(keepends=True)
    header = next((i for i, ln in enumerate(lines) if _TOML_REPLICA_HEADER.match(ln)), None)
    if header is None:
        if "replica" in before:
            raise ReplicaError(f"{path} carries its replica settings without a [replica] table header (a dotted key or an inline table): init "
                               "edits only a [replica] table; write the settings as one, then init again")
        out = text
        if out and not out.endswith("\n"):
            out += "\n"
        if not out.strip():
            out = "# the cluster replica of the registry database: where the replay reads the record log\n"
        else:
            out += "\n"
        out += f"[replica]\n{line}\n"
    else:
        end = next((i for i in range(header + 1, len(lines)) if _TOML_HEADER.match(lines[i])), len(lines))
        key = next((i for i in range(header + 1, end) if _TOML_LOG_ROOT_KEY.match(lines[i])), None)
        if key is None:
            eol = "\r\n" if lines[header].endswith("\r\n") else "\n"
            if not lines[header].endswith(("\n", "\r")):
                lines[header] += eol
            lines.insert(header + 1, line + eol)
        else:
            raw = lines[key]
            eol = "\r\n" if raw.endswith("\r\n") else ("\n" if raw.endswith("\n") else "")
            lines[key] = line + eol
        out = "".join(lines)
    try:
        after = tomllib.loads(out)
    except tomllib.TOMLDecodeError as exc:
        raise ReplicaError(f"{path}: setting [replica] log_root in place would not parse ({exc}); nothing written") from None
    want = {k: v for k, v in before.items() if k != "replica"}
    got = {k: v for k, v in after.items() if k != "replica"}
    rep_before = {k: v for k, v in (before.get("replica") or {}).items() if k != "log_root"} if isinstance(before.get("replica"), dict) else {}
    rep_after = dict(after.get("replica") or {}) if isinstance(after.get("replica"), dict) else {}
    if got != want or rep_after.pop("log_root", None) != log_root or rep_after != rep_before:
        raise ReplicaError(f"{path}: setting [replica] log_root in place would change more than that key; nothing written")
    return out


def init_layout(root: str, log_root: str) -> Layout:
    """Lay the replica root out: ``gen/``, ``builds/``, ``pins/``, ``config.toml`` naming the log root, and the ``records`` / ``rulings``
    links onto it (absolute, realpath-verified). Idempotent; a link pointing elsewhere is re-pointed. ``config.toml`` keeps everything
    init does not own: only ``[replica] log_root`` is set (:func:`replica_config_with_log_root`), every other table and comment stays byte
    for byte, a file that does not parse is refused by name, and an unchanged file is not rewritten. ``site_aliases.toml`` is never
    touched."""
    lay = Layout(os.path.abspath(root))
    log_root = os.path.abspath(log_root)
    existing = ""
    if os.path.exists(lay.config_path):
        with open(lay.config_path, "r", encoding="utf-8") as fh:
            existing = fh.read()
    body = replica_config_with_log_root(existing, log_root, lay.config_path)      # refuses before any directory or link is touched
    for d in (lay.gen_root, lay.builds, lay.pins):
        os.makedirs(d, exist_ok=True)
    for ns in ("records", "rulings"):
        target = os.path.realpath(os.path.join(log_root, ns))
        link = os.path.join(lay.root, ns)
        if os.path.islink(link):
            if os.path.realpath(link) == target:
                continue
            os.remove(link)
        elif os.path.exists(link):
            raise ReplicaError(f"{link} exists and is not a link; the replica root's {ns} must be a link onto the log")
        os.symlink(target, link)
    if body != existing:
        tmp = lay.config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(tmp, lay.config_path)
    return lay


# --------------------------------------------------------------------------------------------
# the site-alias table a fold or a migration reads
# --------------------------------------------------------------------------------------------


def site_aliases_record(table: Mapping[str, str], path: Optional[str], present: bool) -> Dict[str, object]:
    """How a record names the alias table it read: the file, whether it existed, the table's digest and its pairs."""
    from . import site_aliases as SA
    return {"path": os.path.abspath(path) if path else None, "present": bool(present), "digest": SA.digest(table),
            "pairs": [list(p) for p in SA.pairs(table)]}


def root_site_aliases(lay: Layout) -> Tuple[Dict[str, str], Dict[str, object]]:
    """The replica root's alias table (``site_aliases.load_root``: ``<root>/site_aliases.toml``, absent = the empty table, a
    ``[site_aliases]`` table inside ``<root>/config.toml`` refused by name), validated, with its record. Raises
    :class:`SiteAliasesRefused`."""
    from . import site_aliases as SA
    try:
        table = SA.validate(SA.load_root(lay.root))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SiteAliasesRefused(f"the replica root's site-alias table is refused: {exc}") from None
    return table, site_aliases_record(table, lay.site_aliases_path, os.path.exists(lay.site_aliases_path))


def install_root_aliases(lay: Layout, expect_sha256: Optional[str] = None) -> Dict[str, object]:
    """Install the replica root's alias table for this process's folds (``site_aliases.install``) and return its record: every fold the
    replica runs (``build``, ``land``, ``update``, ``ingest``) validates records under the root's table, so a record written with a legacy
    site label folds to the registry's label, and without a table it is refused by name. ``expect_sha256``: the digest the caller already
    recorded; a table that changed since refuses (:class:`SiteAliasesRefused`). The parse workers inherit the table by fork."""
    from . import site_aliases as SA
    _, rec = root_site_aliases(lay)
    if expect_sha256 is not None and rec["digest"] != expect_sha256:
        raise SiteAliasesRefused(f"the replica root's site-alias table has digest {rec['digest']}, not the expected {expect_sha256}")
    try:
        SA.install(lay.root)                      # the root's table, as every Ingestor over this root installs it
    except (OSError, ValueError) as exc:
        raise SiteAliasesRefused(f"the replica root's site-alias table is refused: {exc}") from None
    got = SA.active()[1]
    if got != rec["digest"]:
        raise SiteAliasesRefused(f"the replica root's site-alias table changed while it was read (digest {rec['digest']}, then {got})")
    return rec


def declared_site_aliases(path: Optional[str], expect_sha256: Optional[str]) -> Tuple[Dict[str, str], Dict[str, object]]:
    """The alias table a ``migrate --db`` record names: ``path`` and ``expect_sha256`` are both REQUIRED (a certifying record declares
    everything), the file is read through ``site_aliases.load_table`` and validated, and its digest must be the declared one. Raises
    :class:`SiteAliasesRefused` before any copy."""
    from . import site_aliases as SA
    if not path or not expect_sha256:
        raise SiteAliasesRefused("migrate --db needs --site-aliases FILE and --expect-site-aliases-sha256 H (a certifying record declares the "
                                 "table it migrates under; an empty table is a file with no pairs and its digest)")
    if not _HEX64.match(expect_sha256):
        raise SiteAliasesRefused(f"--expect-site-aliases-sha256 must be the full 64 hex digest, got {expect_sha256!r}")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SiteAliasesRefused(f"--site-aliases {path}: no such file")
    try:
        table = SA.validate(SA.load_table(path))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SiteAliasesRefused(f"--site-aliases {path} is refused: {exc}") from None
    rec = site_aliases_record(table, path, True)
    if rec["digest"] != expect_sha256:
        raise SiteAliasesRefused(f"--site-aliases {path} has digest {rec['digest']}, not the declared {expect_sha256}")
    return table, rec


def migrating_cli_takes_site_aliases(tree: str) -> bool:
    """Whether ``tree``'s ``registry migrate`` declares ``--site-aliases`` (its ``src/sisteer/registry/cli.py`` carries the option): the
    guard ``migrate --db`` applies before handing a table to another tree's code, as ``replica.sbatch land`` does for ``--accept-c1-class``."""
    path = os.path.join(os.path.abspath(tree), "src", "sisteer", "registry", "cli.py")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return '"--site-aliases"' in fh.read()
    except OSError:
        return False


def _copy_db(src: str, dst: str) -> int:
    """A byte copy of a database file (never its WAL twin: a published generation is checkpointed and closed; a node-local copy of
    one has no WAL) with fsync; returns the bytes copied."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(dst + suffix)
        except FileNotFoundError:
            pass
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 16 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    return os.path.getsize(dst)


def _checkpoint(db_path: str) -> None:
    """Fold the WAL into the file and leave it in WAL mode with no live twin (what a copy needs)."""
    conn = DB.connect(db_path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def gen_id_for(head: Mapping[str, object], now: Optional[str] = None) -> str:
    """``<utc stamp>-<allowlist sha8>-<4 random hex>``: sortable by time, names the head, unique even for two publications of one
    head in one second."""
    stamp = (now or R.utc_now()).replace("-", "").replace(":", "")[:15] + "Z"
    return f"{stamp}-{str(head['allowlist_sha256'])[:8]}-{os.urandom(2).hex()}"


def publish(lay: Layout, db_path: str, head: Mapping[str, object], mode: str, parent: Optional[str] = None, gen_id: Optional[str] = None,
            built_by_tree: Optional[str] = None, job_id: Optional[str] = None, extra: Optional[Mapping[str, object]] = None,
            keep: int = KEEP_GENERATIONS) -> Dict[str, object]:
    """Publish the database at ``db_path`` as a new generation built from ``parent`` (``None`` = the bootstrap; the string
    ``"current"`` = whatever CURRENT names when the publication begins): the files are staged under ``gen/<id>.incoming`` and renamed
    into place, then -- under the root's publish lock -- CURRENT must still name ``parent`` and the new head's allowlist must be a
    superset of the parent's, else :class:`PublishRefused` (the staged generation is renamed ``<id>.unpublished`` and left for the
    operator). CURRENT then switches (``records.write_pointer``) and the oldest generations beyond ``keep`` are removed, never CURRENT."""
    from .migrations import db_version
    if parent == "current":
        parent = lay.current()
    gid = gen_id or gen_id_for(head)
    gen_dir = lay.gen(gid)
    if os.path.exists(gen_dir):
        raise PublishRefused(f"generation {gid} exists")
    os.makedirs(lay.gen_root, exist_ok=True)
    incoming = gen_dir + ".incoming"
    shutil.rmtree(incoming, ignore_errors=True)
    os.makedirs(incoming)
    _checkpoint(db_path)
    n_bytes = _copy_db(db_path, os.path.join(incoming, "registry.sqlite"))
    with DB.reading(os.path.join(incoming, "registry.sqlite")) as conn:
        schema_version = db_version(conn)
        proj = frame_projection(conn, with_rows=False)
    write_head(head, os.path.join(incoming, "head.json.gz"))
    meta = {"schema": META_SCHEMA, "gen_id": gid, "parent_gen_id": parent, "schema_version": schema_version, "built_by_tree": built_by_tree,
            "head": {"n_files": head["n_files"], "allowlist_sha256": head["allowlist_sha256"], "taken_utc": head["taken_utc"],
                     "head_sha256": head_sha256(head), "projection_sha256": head["projection"]["sha256"]},
            "projection": {k: v for k, v in proj.items() if k != "rows"}, "mode": mode, "published_utc": None, "host": os.uname().nodename,
            "job_id": job_id, "db_bytes": n_bytes, **dict(extra or {})}
    lock_path = os.path.join(lay.root, PUBLISH_LOCK)
    with open(lock_path, "a+") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            cur = lay.current()
            if cur != parent:
                shutil.move(incoming, gen_dir + ".unpublished")
                raise PublishRefused(f"CURRENT names {cur!r}, not the parent {parent!r} this generation was built from (another writer "
                                     f"published meanwhile): {gid} left as {gen_dir}.unpublished")
            if parent is not None:
                parent_uris = head_uris(lay.head_of(parent))
                mine = head_uris(head)
                lost = sorted(parent_uris - mine)
                if lost:
                    shutil.move(incoming, gen_dir + ".unpublished")
                    raise PublishRefused(f"the new head's allowlist lacks {len(lost)} file(s) of the parent {parent}'s (" + ", ".join(lost[:3])
                                         + (" …" if len(lost) > 3 else "") + f"): not a superset; {gid} left as {gen_dir}.unpublished")
            meta["published_utc"] = R.utc_now()
            with open(os.path.join(incoming, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1, sort_keys=True)
            DB.fsync_file_and_dir(os.path.join(incoming, "meta.json"))
            os.rename(incoming, gen_dir)
            R.write_pointer(lay.current_path, gid)
            pruned = prune(lay, keep=keep)
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
    meta["pruned"] = pruned
    return meta


def prune(lay: Layout, keep: int = KEEP_GENERATIONS) -> List[str]:
    """Remove the oldest generations beyond ``keep``; CURRENT is never removed. Returns the ids removed."""
    gens = lay.generations()
    cur = lay.current()
    removable = [g for g in gens if g != cur]
    excess = max(0, len(gens) - keep)
    out = []
    for g in removable[:excess]:
        shutil.rmtree(lay.gen(g), ignore_errors=True)
        out.append(g)
    return out


def rollback(lay: Layout) -> Dict[str, object]:
    """CURRENT → the current generation's parent, when that generation is still kept (CAS-guarded under the publish lock)."""
    lock_path = os.path.join(lay.root, PUBLISH_LOCK)
    with open(lock_path, "a+") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            cur = lay.current()
            if cur is None:
                raise NoGeneration("rollback: no CURRENT generation")
            parent = lay.meta(cur).get("parent_gen_id")
            if not parent or not os.path.isfile(os.path.join(lay.gen(str(parent)), "meta.json")):
                raise PublishRefused(f"rollback: {cur}'s parent {parent!r} is not a kept generation (nothing to roll back to)")
            R.write_pointer(lay.current_path, str(parent))
            rec = {"utc": R.utc_now(), "from": cur, "to": parent}
            with open(os.path.join(lay.root, "rollbacks.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(canonical_json(rec) + "\n")
            return rec
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


def status(lay: Layout) -> Dict[str, object]:
    cur = lay.current()
    gens = []
    for g in lay.generations():
        try:
            m = lay.meta(g)
        except (OSError, ValueError):
            continue
        gens.append({"gen_id": g, "parent_gen_id": m.get("parent_gen_id"), "mode": m.get("mode"), "published_utc": m.get("published_utc"),
                     "schema_version": m.get("schema_version"), "n_files": (m.get("head") or {}).get("n_files"),
                     "projection_sha12": str((m.get("head") or {}).get("projection_sha256") or "")[:12], "db_bytes": m.get("db_bytes")})
    try:
        aliases: Dict[str, object] = root_site_aliases(lay)[1]
    except SiteAliasesRefused as exc:
        aliases = {"path": lay.site_aliases_path, "refused": str(exc)}
    return {"root": lay.root, "log_root": lay.log_root(), "current": cur, "generations": gens,
            "unpublished": sorted(os.path.basename(p) for p in glob.glob(os.path.join(lay.gen_root, "*.unpublished"))),
            "site_aliases": aliases}


# --------------------------------------------------------------------------------------------
# the mirror presence / hash check
# --------------------------------------------------------------------------------------------


def _latest_manifest(log_root: str) -> Dict[str, str]:
    """``{logical relpath: sha}`` of the mirror's current CHECKSUMS manifest, or empty when the mirror carries none."""
    mdir = os.path.join(log_root, "manifests")
    name = R.read_pointer(os.path.join(mdir, "CHECKSUMS.CURRENT"))
    if not name:
        return {}
    path = os.path.join(mdir, name)
    out: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("  ", 1)
                if len(parts) == 2 and len(parts[0]) == 64:
                    out[parts[1]] = parts[0]
    except OSError:
        return {}
    return out


def verify_mirror(lay: Layout, head: Mapping[str, object], hash_mode: str = "all", threads: int = 8) -> Dict[str, object]:
    """Every allowlisted file present under the replay root with the head's logical sha. ``hash_mode``: ``all`` hashes every listed
    file (``records.file_logical_sha``); ``manifest`` trusts the mirror's latest verified CHECKSUMS manifest for the files it names
    with the same sha and hashes the rest. A head sha equal to sha256 of nothing (a file refused unparsed) is checked for presence
    only and listed under ``sha_unverifiable``."""
    from .harvest import resolve_uri
    root = lay.root
    manifest = _latest_manifest(lay.log_root()) if hash_mode == "manifest" else {}
    missing: List[str] = []
    unverifiable: List[str] = []
    from_manifest = 0
    to_hash: List[Tuple[str, str, str]] = []
    for uri, sha, _st in head["files"]:
        path = resolve_uri(root, uri)
        if path is None:
            missing.append(uri)
            continue
        if sha == EMPTY_SHA:
            unverifiable.append(uri)
            continue
        if manifest.get(uri) == sha:
            from_manifest += 1
            continue
        to_hash.append((uri, sha, path))

    def one(item):
        uri, sha, path = item
        try:
            got = R.file_logical_sha(path)
        except (OSError, EOFError) as exc:
            return uri, sha, f"unreadable: {exc}"
        return uri, sha, got
    differing: List[Dict[str, str]] = []
    t0 = time.monotonic()
    if to_hash:
        with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
            for uri, want, got in ex.map(one, to_hash, chunksize=64):
                if got != want:
                    differing.append({"file_uri": uri, "head_sha256": want, "mirror_sha256": got})
    return {"ok": not missing and not differing, "n_files": len(head["files"]), "missing": missing[:50], "n_missing": len(missing),
            "differing": differing[:50], "n_differing": len(differing), "sha_unverifiable": unverifiable[:50], "n_sha_unverifiable": len(unverifiable),
            "hashed": len(to_hash), "from_manifest": from_manifest, "hash_mode": hash_mode, "hash_s": round(time.monotonic() - t0, 3)}


def require_mirror(lay: Layout, head: Mapping[str, object], hash_mode: str = "all", threads: int = 8) -> Dict[str, object]:
    v = verify_mirror(lay, head, hash_mode, threads)
    if not v["ok"]:
        raise MirrorBehind(f"mirror behind the head by {v['n_missing']} missing and {v['n_differing']} differing file(s): "
                           + ", ".join(list(v["missing"][:3]) + [d["file_uri"] for d in v["differing"][:3]]))
    return v


# --------------------------------------------------------------------------------------------
# build, ingest, check
# --------------------------------------------------------------------------------------------


def _peak_rss_bytes() -> int:
    """Peak resident set of this process and its finished children (ru_maxrss is KiB on Linux)."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(max(own, kids)) * 1024


def build(lay: Layout, head: Mapping[str, object], work_db: str, workers: int = 16, site_aliases_sha256: Optional[str] = None) -> Dict[str, object]:
    """The bootstrap: ``harvest.rebuild(<replay root>, work_db, files=<allowlist>)`` -- measured wall, records/s, peak bytes -- then
    :func:`check` against the SAME head. Returns ``{"build": …, "check": …}``; the caller publishes only an equal build. The replay
    runs under the replica root's alias table (:func:`install_root_aliases`), named under ``build.site_aliases`` and ``check.site_aliases``."""
    from . import harvest as H
    aliases = install_root_aliases(lay, site_aliases_sha256)
    t0 = time.monotonic()
    rep = H.rebuild(lay.root, work_db, workers=workers, files=[f[0] for f in head["files"]])
    wall = time.monotonic() - t0
    d = rep.as_dict()
    d.pop("refusals", None)
    out = {"wall_s": round(wall, 1), "records": rep.records, "files": rep.files, "records_per_s": round(rep.records / wall, 1) if wall > 0 else None,
           "peak_rss_bytes": _peak_rss_bytes(), "db_bytes": DB.file_size(work_db), "workers": workers, "unlisted_ignored": rep.unlisted_ignored,
           "unlisted_sample": list(rep.unlisted_sample), "report": d, "n_files_allowlisted": head["n_files"], "site_aliases": aliases}
    chk = check(work_db, head)
    chk["site_aliases"] = aliases
    return {"build": out, "check": chk}


def ingest_head(db_path: str, lay: Layout, head: Mapping[str, object], workers: int = 0, now: Optional[str] = None,
                site_aliases_sha256: Optional[str] = None) -> Dict[str, object]:
    """The increment: fold exactly ``allowlist − replica.record_delivery`` (plus a listed file whose delivered sha differs) into the
    database at ``db_path`` in the ledger's order -- dimension bundles whole, the producers' plain chunks, revision directories grouped
    by source with each directory's files filtered by the list, ``reevaluate_blocked``, then the harvester's own facts
    (``harvest.partition_harvester_files``, the order ``harvest.rebuild`` and a live cycle fold them in) and the queue once more --
    then ``reclaim``, the one dense re-rank and ``stamp_run_rows``. Nothing under ``records/`` is written (no harvester facts, no gzip pass, no state files). A delivered file the
    head lacks refuses first (:class:`ReplicaAhead`); a listed file not on disk refuses (:class:`MirrorBehind`). The fold runs under the
    replica root's alias table (:func:`install_root_aliases`, ``site_aliases_sha256`` the digest the caller recorded), named under
    ``site_aliases`` in the result."""
    from . import decode_match as DM
    from . import harvest as H
    now = now or R.utc_now()
    aliases = install_root_aliases(lay, site_aliases_sha256)
    try:
        conn = DB.open_or_create(db_path)        # the writers' open: refuses a version or shape that is not the code's (SchemaDrift)
    except DB.FoldOrderDrift:
        raise                                    # not a schema to migrate: the replica records another fold order; its own message names the remedy
    except DB.SchemaDrift as exc:
        raise SchemaBehind(f"the replica's schema is not the code's -- `replica migrate` first: {exc}") from exc
    try:
        known = {str(r[0]): str(r[1]) for r in conn.execute("SELECT file_uri, file_sha256 FROM record_delivery")}
        want = head_shas(head)
        ahead = sorted(set(known) - set(want))
        if ahead:
            raise ReplicaAhead(f"replica ahead of the fold: {len(ahead)} delivered file(s) the head does not name ("
                               + ", ".join(ahead[:3]) + (" …" if len(ahead) > 3 else "") + f"); head taken {head['taken_utc']}")
        delta = [(u, s) for u, s in want.items() if known.get(u) != s]
        changed = [u for u, _ in delta if u in known]
        work: List[Tuple[str, Optional[str]]] = []
        missing: List[str] = []
        for uri, sha in delta:
            path = H.resolve_uri(lay.root, uri)
            if path is None:
                missing.append(uri)
            else:
                work.append((path, sha))
        if missing:
            raise MirrorBehind(f"mirror behind the head by {len(missing)} file(s): " + ", ".join(missing[:5]) + (" …" if len(missing) > 5 else ""))
        report = H.IngestReport()
        t0 = time.monotonic()
        allow = head_uris(head)
        ing = H.Ingestor(conn, lay.root, report, live=False, defer_ranks=True, arrivals=H.read_arrival_facts(lay.root, only=allow))
        ing.allowlist = allow
        ing.file_shas = {p: s for p, s in work if s}
        ing.listed_uris = {H.logical_uri(lay.root, p) for p, _ in work}
        ordered = H.order_for_ingest(lay.root, work)
        plain, rev_dirs, dim_bundles = H.split_work(lay.root, ordered)
        rev_dirs = H.group_revision_dirs(lay.root, rev_dirs, conn)
        plain, facts = H.partition_harvester_files(lay.root, plain)     # the harvester's facts after the revisions, as in rebuild
        for dim_dir, dim_files in dim_bundles:
            ing.apply_dimension_bundle(dim_dir, dim_files, now)
        ing.ingest_parsed(H.iter_parsed(lay.root, plain, workers, H.ITEM_ROWS_PER_TRANSACTION), now)
        for d in rev_dirs:
            ing.apply_revision(d, now)
            report.revisions_applied += 1
        ing.reevaluate_blocked(now)
        ing.ingest_parsed(H.iter_parsed(lay.root, facts, workers, H.ITEM_ROWS_PER_TRANSACTION), now)
        report.parse_s = max(0.0, (time.monotonic() - t0) - report.write_s)
        if facts:
            ing.reevaluate_blocked(now)
        ing.reclaim(now)
        ing.finalize_ranks(now)
        with DB.transaction(conn):
            stamped = DM.stamp_run_rows(conn)
        try:
            DB.wal_checkpoint_truncate(conn)
        except Exception:
            pass
        d = report.as_dict()
        d.pop("refusals", None)
        return {"delta_files": len(delta), "changed_sha": len(changed), "plain": len(plain), "revision_dirs": len(rev_dirs),
                "dimension_bundles": len(dim_bundles), "stamped": stamped, "elapsed_s": round(time.monotonic() - t0, 3),
                "unlisted_ignored": report.unlisted_ignored, "unlisted_sample": list(report.unlisted_sample), "report": d, "now": now,
                "site_aliases": aliases}
    finally:
        conn.close()


def refusal_counts(conn) -> Dict[str, object]:
    """The database's ``record_refusal`` rows as named counts: ``run_ingest`` (the harvester's ingest stamps the replay refused -- the
    certification line of a bootstrap, 0 by construction when the replay folds the facts the live fold applied; the bootstrap carried
    128 where live carried 0), ``harvester_facts`` (every refused record under
    ``records/harvester/``), ``by_type`` and ``total``."""
    from . import harvest as H
    by_type = {str(r[0]): int(r[1]) for r in conn.execute("SELECT COALESCE(record_type, ''), COUNT(*) FROM record_refusal GROUP BY 1")}
    facts = int(conn.execute("SELECT COUNT(*) FROM record_refusal WHERE record_uri LIKE ?", (H.HARVESTER_URI_PREFIX + "%",)).fetchone()[0])
    return {"run_ingest": by_type.get("run_ingest", 0), "harvester_facts": facts, "by_type": by_type, "total": sum(by_type.values())}


def blocked_counts(conn) -> Dict[str, object]:
    """The database's ``blocked_records`` rows as named counts -- ``harvester_facts`` (the harvester's own records waiting on a parent or on
    lifecycle evidence under ``records/harvester/``: a ``run_ingest`` blocked on ``run_state_event <run>:<state>`` whose evidence never
    lands sits here for good, the fold-order rule's quiet side), ``by_type`` and ``total``. Reported beside the verdict, never folded into
    it: a blocked record is the queue's to retry, a refused one is final."""
    from . import harvest as H
    by_type = {str(r[0]): int(r[1]) for r in conn.execute("SELECT COALESCE(record_type, ''), COUNT(*) FROM blocked_records GROUP BY 1")}
    facts = int(conn.execute("SELECT COUNT(*) FROM blocked_records WHERE record_uri LIKE ?", (H.HARVESTER_URI_PREFIX + "%",)).fetchone()[0])
    return {"harvester_facts": facts, "by_type": by_type, "total": sum(by_type.values())}


def classify_projection_diff(conn, head_projection: Mapping[str, object], projection: Mapping[str, object],
                             components: Mapping[str, bool]) -> Dict[str, object]:
    """The diff's CLASSIFICATION beside ``equal``: every differing column with its row count (``columns_differing``), every run
    whose row differs (``movers``, the whole set -- :func:`projection_diff` samples examples; a verdict needs every id), the rows on one
    side only, and ``c1_class`` -- true iff the ONLY failing component is the projection, both sides carry rows, the version, columns and
    the three counts agree, no row is on one side only, there is at least one mover, EVERY mover differs in ``lifecycle_state`` and in
    nothing outside :data:`C1_CLASS_COLUMNS`, and EVERY mover satisfies the predicate refold-compare certifies the declared-superseded
    class with: :func:`mover_in_class` over the lifecycle move (``judged`` on the head's side, ``superseded`` on the database's) and
    :func:`declared_superseded_class` over the database's own ``row_history`` (its latest same-stamp ``run_state`` batch carries both
    states). The fold-level rule is re-used as it stands: the projection adds the CASE columns that read the lifecycle, so the diff handed
    to :func:`mover_in_class` is the lifecycle move alone and the derived columns are admitted by name here. The record facts are read
    for the movers only when the column rule already holds for all of them (a diff on any other column is out of the class before any
    seek). Each rule is named under ``c1_components``; ``movers_out_of_class`` lists the movers the pair predicate refused (capped)."""
    rows_available = head_projection.get("rows") is not None and projection.get("rows") is not None
    cols = list(head_projection.get("columns") or PROJECTION_COLUMNS)
    moved, one_side = _projection_moves(head_projection, projection) if rows_available else ({}, [])
    ka = {row[0]: row for row in (head_projection.get("rows") or [])}
    kb = {row[0]: row for row in (projection.get("rows") or [])}
    only_head = sorted(set(ka) - set(kb))
    only_replica = sorted(set(kb) - set(ka))
    columns_differing = {c: sum(1 for cs in moved.values() if c in cs) for c in sorted({c for cs in moved.values() for c in cs})}
    movers = sorted(moved)
    allowed = set(C1_CLASS_COLUMNS)
    columns_in_class = bool(movers) and all("lifecycle_state" in cs and set(cs) <= allowed for cs in moved.values())
    in_class: Dict[str, bool] = {}
    if movers and columns_in_class and not one_side and "lifecycle_state" in cols:
        i_lc = cols.index("lifecycle_state")
        keys = {rid: canonical_json({"run_id": rid}) for rid in movers}
        facts = declared_superseded_class(conn, [keys[rid] for rid in movers])
        for rid in movers:
            move = {"columns": {"lifecycle_state": [ka[rid][i_lc], kb[rid][i_lc]]}}
            in_class[rid] = mover_in_class(move, facts[keys[rid]])
    out_of_class = sorted(rid for rid, ok in in_class.items() if not ok)
    c1_components = {
        "projection_is_the_only_failing_component": all(ok for k, ok in components.items() if k != "projection") and not components.get("projection", True),
        "rows_available": rows_available,
        "version_and_columns_equal": head_projection.get("version") == projection.get("version") and list(head_projection.get("columns") or []) == list(projection.get("columns") or []),
        "counts_equal": head_projection.get("counts") == projection.get("counts"),
        "no_one_sided_rows": not one_side,
        "movers_present": bool(movers),
        "columns_in_class": columns_in_class,
        "movers_in_class": bool(movers) and len(in_class) == len(movers) and all(in_class.values()),
    }
    return {"c1_class": all(c1_components.values()), "c1_components": c1_components, "c1_class_columns": list(C1_CLASS_COLUMNS),
            "c1_class_pair": list(DECLARED_SUPERSEDED_PAIR), "columns_differing": columns_differing, "n_movers": len(movers), "movers": movers,
            "n_movers_in_class": sum(1 for ok in in_class.values() if ok), "movers_out_of_class": out_of_class[:50],
            "n_movers_out_of_class": len(out_of_class), "n_only_in_head": len(only_head), "n_only_in_replica": len(only_replica)}


def check(db_path: str, head: Mapping[str, object], projection_out: Optional[str] = None) -> Dict[str, object]:
    """``frame_projection/1`` of the database vs the head's: version, columns, sha and the three counts must all agree; a difference
    is printed per column (``diff``). The delivered file set is compared too (``ledger``): a replica ahead of or behind the head is
    unequal by construction. And the replay must have REFUSED no ``run_ingest`` fact: a stamp the live fold applied and the replay refused
    is a permanent divergence the projection need not show (a redundant stamp -- 127 of the bootstrap's 128 -- leaves the projected
    lifecycle to another stamp of the same run) and the ledger cannot show (it compares file uri and sha); so ``equal`` is the projection
    AND the ledger AND ``record_refusal.run_ingest == 0`` AND a cursor-free database, each named under ``components`` with the failing ones
    on the verdict line. ``equal`` keeps that strict meaning. Beside it the diff is CLASSIFIED
    (:func:`classify_projection_diff`): ``columns_differing``, ``movers`` (every differing run id), ``n_only_in_head`` /
    ``n_only_in_replica`` and ``c1_class`` -- the certified C1 same-stamp class and nothing else -- which :func:`land` accepts only under an
    operator's declaration; the bootstrap and every other caller still publish an equal check alone, so a refused stamp publishes
    nothing. ``record_refusal`` and ``blocked_records`` carry the counts (:func:`refusal_counts`, :func:`blocked_counts`; the blocked ones
    are reporting only). ``projection_out`` names a file the database's full projection (every row) is written to beside the verdict, so
    the differing run ids can be read in full afterwards (``diff`` carries five examples per column). The projection, the ledger, the
    counts and the movers' record facts are all read in ONE read transaction."""
    hp = head["projection"]
    want = head_shas(head)
    with DB.reading(db_path) as conn:
        conn.execute("BEGIN")
        try:
            proj = frame_projection(conn, with_rows=True)
            delivered = {str(r[0]): str(r[1]) for r in conn.execute("SELECT file_uri, file_sha256 FROM record_delivery")}
            n_cursor = int(conn.execute("SELECT COUNT(*) FROM record_cursor").fetchone()[0])
            refusals = refusal_counts(conn)
            blocked = blocked_counts(conn)
            extra = sorted(set(delivered) - set(want))
            missing = sorted(set(want) - set(delivered))
            sha_diff = sorted(u for u in set(want) & set(delivered) if want[u] != delivered[u])
            ledger_equal = not extra and not missing and not sha_diff
            projection_equal = (proj["version"] == hp.get("version") and list(proj["columns"]) == list(hp.get("columns") or [])
                                and proj["sha256"] == hp["sha256"] and proj["counts"] == hp["counts"])
            components = {"projection": projection_equal, "ledger": ledger_equal, "record_refusal.run_ingest": int(refusals["run_ingest"]) == 0,
                          "cursor_free": n_cursor == 0}
            classification = classify_projection_diff(conn, hp, proj, components)
        finally:
            conn.execute("ROLLBACK")
    if projection_out:
        _write_json(projection_out, proj)
    equal = all(components.values())
    out: Dict[str, object] = {"schema": CHECK_SCHEMA, "utc": R.utc_now(), "equal": equal, "components": components,
                              "failing": sorted(k for k, ok in components.items() if not ok), "db_path": os.path.abspath(db_path),
                              "projection_version": proj["version"], "columns": list(proj["columns"]), "columns_equal": list(proj["columns"]) == list(hp.get("columns") or []),
                              "replica": {"sha256": proj["sha256"], "sha12": proj["sha256"][:12], "counts": proj["counts"]},
                              "head": {"sha256": hp["sha256"], "sha12": hp["sha256"][:12], "counts": hp["counts"], "taken_utc": head["taken_utc"],
                                       "n_files": head["n_files"], "allowlist_sha256": head["allowlist_sha256"]},
                              "ledger": {"equal": ledger_equal, "n_extra_in_replica": len(extra), "extra_in_replica": extra[:20],
                                         "n_missing_in_replica": len(missing), "missing_in_replica": missing[:20], "n_sha_differs": len(sha_diff),
                                         "sha_differs": sha_diff[:20], "cursor_rows": n_cursor},
                              "record_refusal": refusals, "blocked_records": blocked, **classification}
    if not projection_equal:
        out["diff"] = projection_diff(hp, proj)
    return out


def write_check_json(chk: Mapping[str, object], path: str) -> Dict[str, object]:
    """``check.json`` at ``path`` with at most :data:`MOVERS_INLINE_CAP` movers inline: a longer list is written WHOLE to ``movers.json``
    beside it (``movers_truncated``, ``movers_file``), and a stale ``movers.json`` from an earlier occupant is removed when the list fits.
    Returns the document written."""
    doc = dict(chk)
    movers = list(doc.get("movers") or [])
    movers_path = os.path.join(os.path.dirname(os.path.abspath(path)), "movers.json")
    if len(movers) > MOVERS_INLINE_CAP:
        _write_json(movers_path, movers)
        doc.update(movers=movers[:MOVERS_INLINE_CAP], movers_truncated=True, movers_file=movers_path)
    else:
        doc.update(movers_truncated=False, movers_file=None)
        try:
            os.remove(movers_path)
        except FileNotFoundError:
            pass
    _write_json(path, doc)
    return doc


def format_check(chk: Mapping[str, object]) -> str:
    """The check as lines: one verdict line (naming the failing components of an unequal one), the refusal and blocked counts, then one
    line per differing column."""
    rep, hd = chk["replica"], chk["head"]
    failing = list(chk.get("failing") or [])
    verdict = "equal" if chk["equal"] else "UNEQUAL" + (f" ({', '.join(failing)})" if failing else "")
    lines = [f"replica check: {verdict} -- {chk['projection_version']}; replica {rep['sha12']} {rep['counts']} vs head "
             f"{hd['sha12']} {hd['counts']} (head taken {hd['taken_utc']}, {hd['n_files']} files)"]
    led = chk.get("ledger") or {}
    if not led.get("equal", True):
        lines.append(f"  ledger: {led.get('n_extra_in_replica')} file(s) only in the replica, {led.get('n_missing_in_replica')} missing in it, "
                     f"{led.get('n_sha_differs')} with another sha, {led.get('cursor_rows')} cursor row(s)")
    ref = chk.get("record_refusal")
    if ref is not None:
        lines.append(f"  record_refusal: run_ingest {ref.get('run_ingest')} (must be 0), harvester facts {ref.get('harvester_facts')}, total {ref.get('total')}")
    blk = chk.get("blocked_records")
    if blk is not None:
        lines.append(f"  blocked_records: harvester facts {blk.get('harvester_facts')}, total {blk.get('total')} (reporting only)")
    diff = chk.get("diff") or {}
    if diff:
        if not diff.get("columns_equal", True):
            lines.append("  the column lists differ")
        if diff.get("n_only_in_a"):
            lines.append(f"  {diff['n_only_in_a']} run(s) only in the head: {', '.join(diff['only_in_a'][:3])}")
        if diff.get("n_only_in_b"):
            lines.append(f"  {diff['n_only_in_b']} run(s) only in the replica: {', '.join(diff['only_in_b'][:3])}")
        for col, d in sorted((diff.get("columns") or {}).items()):
            ex = "; ".join(f"{e[0][:12]}: head={e[1]!r} replica={e[2]!r}" for e in d["examples"][:2])
            lines.append(f"  column {col}: {d['rows']} row(s) differ ({ex})")
    if not chk["equal"] and "c1_class" in chk:
        failing_rules = sorted(k for k, ok in (chk.get("c1_components") or {}).items() if not ok)
        lines.append(f"  c1_class: {'yes' if chk['c1_class'] else 'no'} -- {chk.get('n_movers')} mover(s), {chk.get('n_movers_out_of_class')} outside the same-stamp "
                     f"pair, columns {sorted(chk.get('columns_differing') or {})}" + (f"; failing rules: {', '.join(failing_rules)}" if failing_rules else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# migrate, land
# --------------------------------------------------------------------------------------------


def migrate(lay: Layout, work_dir: str, built_by_tree: Optional[str] = None, job_id: Optional[str] = None, site_aliases_path: Optional[str] = None,
            expect_site_aliases_sha256: Optional[str] = None) -> Dict[str, object]:
    """``registry migrate`` on a node-local copy of CURRENT: the pending migrations applied, the shape verified against the code's DDL;
    a shape a migration cannot express refuses (:class:`MigrationNeedsRebuild`: ``replica build`` is the path); else the copy is
    published as a new generation (mode ``migrate``) under the parent's head.

    The migrations run under the replica root's own alias table (:func:`root_site_aliases`), never another file's: ``site_aliases_path``,
    when given, must name ``<root>/site_aliases.toml`` itself, and ``expect_site_aliases_sha256``, when given, must be its digest -- both
    refused (:class:`SiteAliasesRefused`) before the copy. The table's record goes into the generation's ``meta.json``
    (``site_aliases``). A migration that fails is rolled back and refused by name; nothing is published."""
    from .migrations import MigrationFailed, apply_migrations, code_version, db_version
    cur = lay.current()
    if cur is None:
        raise NoGeneration("migrate: no CURRENT generation; bootstrap with `replica build`")
    if site_aliases_path and os.path.realpath(os.path.abspath(site_aliases_path)) != os.path.realpath(lay.site_aliases_path):
        raise SiteAliasesRefused(f"--site-aliases {site_aliases_path} is not this replica root's table ({lay.site_aliases_path}): the CURRENT form "
                                 "migrates under its own root's table only")
    if expect_site_aliases_sha256 is not None and not _HEX64.match(expect_site_aliases_sha256):
        raise SiteAliasesRefused(f"--expect-site-aliases-sha256 must be the full 64 hex digest, got {expect_site_aliases_sha256!r}")
    table, aliases = root_site_aliases(lay)
    if expect_site_aliases_sha256 is not None and aliases["digest"] != expect_site_aliases_sha256:
        raise SiteAliasesRefused(f"the replica root's site-alias table has digest {aliases['digest']}, not the expected {expect_site_aliases_sha256}")
    os.makedirs(work_dir, exist_ok=True)
    work_db = os.path.join(work_dir, "registry.sqlite")
    _copy_db(lay.db_of(cur), work_db)
    conn = DB.connect(work_db)
    try:
        before = db_version(conn)
        try:
            applied = apply_migrations(conn, site_aliases=table)
        except MigrationFailed as exc:
            raise ReplicaError(f"migrate: the migration failed and was rolled back, nothing published: {exc}") from None
        drift = DB.schema_drift(conn)
        after = db_version(conn)
    finally:
        conn.close()
    if drift:
        raise MigrationNeedsRebuild(f"after migrating {before} -> {after} the shape still differs from the code's schema ({'; '.join(drift[:4])}): "
                                    "no migration expresses it -- fall back to a full rebuild (`replica build`)")
    head = lay.head_of(cur)
    meta = publish(lay, work_db, head, mode="migrate", parent=cur, built_by_tree=built_by_tree, job_id=job_id,
                   extra={"migrated_from": before, "migrated_to": after, "applied": applied, "code_version": code_version(), "site_aliases": aliases})
    return {"from": before, "to": after, "applied": applied, "generation": meta["gen_id"], "parent": cur, "site_aliases": aliases}


def head_summary(head: Mapping[str, object]) -> Dict[str, object]:
    """The head's identity as a record names it: when it was taken, its file count, allowlist and head sha256, its projection sha and
    counts, the schema and code versions it saw."""
    return {"taken_utc": head["taken_utc"], "n_files": head["n_files"], "allowlist_sha256": head["allowlist_sha256"],
            "head_sha256": head_sha256(head), "projection_sha256": head["projection"]["sha256"], "projection_counts": head["projection"].get("counts"),
            "schema_version": head.get("schema_version"), "code_version": head.get("code_version")}


def _copy_db_hashed(src: str, dst: str) -> Dict[str, object]:
    """A byte copy of a database file hashed as it is written -- ``{path, bytes, sha256, copy_s}`` names the OUTPUT a record stands for.
    Refuses (:class:`RefoldInputRefused`) a source whose ``-wal`` twin holds bytes: the file alone is not the database then (checkpoint
    first)."""
    wal = src + "-wal"
    wal_bytes = os.path.getsize(wal) if os.path.exists(wal) else 0
    if wal_bytes:
        raise RefoldInputRefused(f"{wal} holds {wal_bytes} bytes: the file alone is not the database -- checkpoint it before copying it out")
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(dst + suffix)
        except FileNotFoundError:
            pass
    t0 = time.monotonic()
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            block = fi.read(16 << 20)
            if not block:
                break
            h.update(block)
            fo.write(block)
            n += len(block)
        fo.flush()
        os.fsync(fo.fileno())
    return {"path": os.path.abspath(dst), "bytes": n, "sha256": h.hexdigest(), "copy_s": round(time.monotonic() - t0, 1)}


def schema_migration_rows(db_path: str) -> List[Dict[str, object]]:
    """The database's ``schema_migration`` rows (version, applied_utc, description) in version order, read directly: what the database
    says it is, independent of any tree's code."""
    with DB.reading(db_path) as conn:
        return [{"version": int(r[0]), "applied_utc": r[1], "description": r[2]}
                for r in conn.execute("SELECT version, applied_utc, description FROM schema_migration ORDER BY version")]


def _check_summary(chk: Mapping[str, object]) -> Dict[str, object]:
    """The verdict of :func:`check` without its examples: equal, the failing components, both sides' sha and counts, the per-column counts of
    differing rows, the rows on one side only, the refusal and blocked counts."""
    diff = chk.get("diff") or {}
    return {"equal": chk["equal"], "failing": list(chk.get("failing") or []), "components": dict(chk.get("components") or {}),
            "replica": chk["replica"], "head": chk["head"],
            "columns_differing": {c: int(d["rows"]) for c, d in sorted((diff.get("columns") or {}).items())},
            "n_only_in_head": int(diff.get("n_only_in_a") or 0), "n_only_in_replica": int(diff.get("n_only_in_b") or 0),
            "c1_class": chk.get("c1_class"), "c1_components": dict(chk.get("c1_components") or {}), "n_movers": chk.get("n_movers"),
            "n_movers_out_of_class": chk.get("n_movers_out_of_class"),
            "record_refusal": chk.get("record_refusal"), "blocked_records": chk.get("blocked_records")}


def _clear_outputs(out_dir: str, names: Sequence[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        for suffix in ("", "-wal", "-shm", "-journal") if name.endswith(".sqlite") else ("",):
            try:
                os.remove(os.path.join(out_dir, name + suffix))
            except FileNotFoundError:
                pass


def _require_hex64(name: str, value: Optional[str]) -> None:
    if not _HEX64.match(value or ""):
        raise RefoldInputRefused(f"{name} must be the full 64 hex sha256, got {value!r}")


def _require_running_tree(tree: str, expect_sha: str) -> Dict[str, object]:
    """The tree whose code is RUNNING, bound twice: by commit (:func:`require_tree` -- a git checkout at ``expect_sha``, clean, its own top
    level, no untracked source) and by location -- this very module must live under ``<tree>/src/`` (the path Python imported it from,
    symlinks unresolved, as ``PYTHONPATH=<tree>/src`` lays it out), so a record cannot name one tree while another tree's code folds
."""
    ident = require_tree(tree, expect_sha)
    here = os.path.abspath(__file__)
    prefix = os.path.join(os.path.abspath(tree), "src") + os.sep
    if not here.startswith(prefix):
        raise RefoldInputRefused(f"the running code is {here}, not under {prefix}: --tree must name the tree whose code runs")
    return ident


def _refuse_output_aliases(src: str, work_dir: str, out_dir: str, lay: Optional[Layout] = None) -> None:
    """The output directory must be nobody else's (the outputs are cleared before the copy, so an aliased directory would lose the input
    or a generation): refused when it is the input's own directory (a kept build's record files live beside the kept database), when it
    and the node-local work directory nest or coincide, when the input itself sits under the work directory, when it lies under a replica
    root's ``gen/`` or already holds a generation's ``meta.json``, or when the output database would be the input."""
    src_real = os.path.realpath(src)
    out_real = os.path.realpath(os.path.abspath(out_dir))
    work_real = os.path.realpath(os.path.abspath(work_dir))
    if os.path.dirname(src_real) == out_real:
        raise RefoldInputRefused(f"--out {out_dir} is the input's own directory: the outputs written there would displace the files beside {src}")
    if os.path.realpath(os.path.join(out_real, "registry.sqlite")) == src_real:
        raise RefoldInputRefused(f"--out {out_dir}/registry.sqlite would be the input {src} itself")
    if out_real == work_real or out_real.startswith(work_real + os.sep) or work_real.startswith(out_real + os.sep):
        raise RefoldInputRefused(f"--out {out_dir} and --work {work_dir} nest or coincide: the work directory is removed when the job ends")
    if src_real.startswith(work_real + os.sep):
        raise RefoldInputRefused(f"the input {src} sits under --work {work_dir}: it would be copied over itself and removed with the work")
    if lay is not None and out_real.startswith(os.path.realpath(lay.gen_root) + os.sep):
        raise RefoldInputRefused(f"--out {out_dir} lies under the replica root's generations ({lay.gen_root}): a generation is written only by publish")
    if os.path.isfile(os.path.join(out_real, "meta.json")):
        raise RefoldInputRefused(f"--out {out_dir} holds a generation's meta.json: not an output directory")


def require_head(head: Mapping[str, object], expect_head_sha256: str) -> str:
    """The head a certifying record folds to or checks against, DECLARED: ``expect_head_sha256`` must be the full 64-hex
    :func:`head_sha256` of the head file (its allowlist sha, projection sha and ``taken_utc`` together), else refused before any work (a
    head read off a path is whichever file sat there -- an older head offered in place of today's folds nothing and still reads
    UNEQUAL on the certified class). Returns the identity."""
    _require_hex64("--expect-head-sha256", expect_head_sha256)
    got = head_sha256(head)
    if got != expect_head_sha256:
        raise RefoldInputRefused(f"the head's identity is {got} (taken {head.get('taken_utc')}, {head.get('n_files')} files), not the expected "
                                 f"{expect_head_sha256}: not the head this record names")
    return got


def update_db(src: str, lay: Layout, head: Mapping[str, object], work_dir: str, out_dir: str, tree: str, expect_tree: str, expect_input_sha256: str,
              expect_head_sha256: str, workers: int = 4, hash_mode: str = "all", threads: int = 8, job_id: Optional[str] = None,
              log: Callable[[str], None] = print, expect_site_aliases_sha256: Optional[str] = None) -> int:
    """The increment over a KEPT database that no generation names (a bootstrap kept UNEQUAL on a certified class, brought
    forward to a newer head without being published). Every record this mode writes is a CERTIFYING record, so its inputs are DECLARED,
    not read: ``expect_input_sha256`` is the input's full sha256 computed independently (the copy's streaming hash must
    equal it, else refused and recorded, no output); ``tree`` / ``expect_tree`` bind the tree whose code folds, by commit and by this
    module's own location (:func:`_require_running_tree`); ``expect_head_sha256`` binds the head (:func:`require_head`); the output
    directory may alias nothing (:func:`_refuse_output_aliases`). Then ``src`` is copied node-local and hashed as it is read
    (:func:`stage_input`; the input is never written) and the head must STRICTLY supersede the copy's delivered-file ledger -- every file the
    copy delivered is in the head's allowlist (else ``REFUSED``, exit 7, the copy is ahead) AND the head names at least one file the copy has
    not folded (else ``HEAD_NOT_NEWER``, exit 13: a head that folds nothing certifies nothing), both read before the mirror hash and recorded
    (``ledger``); then every allowlisted file is checked present on the mirror with the head's sha (:func:`require_mirror`), the increment is
    folded (:func:`ingest_head`), the copy is checked against the head (:func:`check`, its full projection written beside the verdict so the differing run ids can be read in
    full) and copied out to ``<out_dir>/registry.sqlite``, hashed as it is written. ``<out_dir>/update.json`` carries the input's and the
    output's identity, the head's, the bound tree, the mirror, ingest and check summaries and the job's ``identity``; ``mirror.json``,
    ``ingest.json``, ``check.json`` and ``projection.json`` stand beside it. Nothing under the replica root is written but the harvester's
    own refusal log (``state/refused.jsonl``), never a generation or ``CURRENT``. Returns 0 when the check is equal and 7 when it is not --
    the output stands either way (an unequal update is a reading to certify or a finding; exit 7 as ``build``'s UNEQUAL); a refused
    mirror or increment returns its exit code and writes no output.

    The fold runs under the replica root's alias table (:func:`install_root_aliases`): a record written with a legacy site label folds to
    the registry's label through it, and without a table it is refused by name. The table is read and validated before the copy, its
    record (file, digest, pairs) goes into ``update.json`` and ``check.json`` (``site_aliases``), and ``expect_site_aliases_sha256``,
    when given, must be its digest (refused before the copy)."""
    _require_hex64("--expect-input-sha256", expect_input_sha256)
    if not _HEX40.match(expect_tree or ""):
        raise RefoldInputRefused(f"--expect-tree must be the full 40 hex commit sha, got {expect_tree!r}")
    if expect_site_aliases_sha256 is not None:
        _require_hex64("--expect-site-aliases-sha256", expect_site_aliases_sha256)
    hsha = require_head(head, expect_head_sha256)
    ident = _require_running_tree(tree, expect_tree)
    _refuse_output_aliases(src, work_dir, out_dir, lay)
    _, aliases = root_site_aliases(lay)
    if expect_site_aliases_sha256 is not None and aliases["digest"] != expect_site_aliases_sha256:
        raise SiteAliasesRefused(f"the replica root's site-alias table has digest {aliases['digest']}, not the expected {expect_site_aliases_sha256}")
    _clear_outputs(out_dir, UPDATE_OUTPUTS)
    started = time.monotonic()
    identity = {"job_id": job_id, "built_by_tree": ident["bound_to"], "head_sha256": hsha, "allowlist_sha256": head["allowlist_sha256"],
                "projection_sha256": head["projection"]["sha256"], "out_dir": os.path.abspath(out_dir), "input_sha256": expect_input_sha256}
    rec: Dict[str, object] = {"schema": UPDATE_SCHEMA, "utc_start": R.utc_now(), "identity": identity, "head": head_summary(head), "status": None,
                              "tree": ident, "input": None, "output": None, "expected_input_sha256": expect_input_sha256,
                              "expected_head_sha256": expect_head_sha256, "workers": workers, "hash_mode": hash_mode, "site_aliases": aliases,
                              "expected_site_aliases_sha256": expect_site_aliases_sha256}

    def finish(status: str, rc: int, **extra) -> int:
        rec.update(status=status, rc=rc, elapsed_s=round(time.monotonic() - started, 1), utc_end=R.utc_now(), **extra)
        _write_json(os.path.join(out_dir, "update.json"), rec)
        log(f"[replica update] {status} rc={rc} ({rec['elapsed_s']} s) -> {out_dir}")
        return rc

    log(f"[replica update] site aliases: {aliases['path']} ({'present' if aliases['present'] else 'absent: the empty table'}), digest {aliases['digest']}, "
        f"{len(aliases['pairs'])} pair(s)")
    staged = stage_input(src, work_dir, log=lambda m: log(f"[replica update] {m}"))
    rec["input"] = staged
    if staged["sha256"] != expect_input_sha256:
        finish("INPUT_REFUSED", RefoldInputRefused.exit_code, reason=f"the staged input's sha256 is {staged['sha256']}, not the expected {expect_input_sha256}")
        raise RefoldInputRefused(f"the staged input's sha256 is {staged['sha256']}, not the expected {expect_input_sha256}: not the database this update names")
    copy = str(staged["copy"])
    # the head must strictly supersede the copy's delivered ledger: nothing the copy delivered is unknown to the head, and the head names
    # files the copy has not folded -- read here, before the mirror hash, so a stale or foreign head costs no hour
    with DB.reading(copy) as conn:
        known = {str(r[0]): str(r[1]) for r in conn.execute("SELECT file_uri, file_sha256 FROM record_delivery")}
    want = head_shas(head)
    ahead = sorted(set(known) - set(want))
    new_files = sorted(u for u, sha in want.items() if known.get(u) != sha)
    rec["ledger"] = {"n_delivered_in_copy": len(known), "n_in_head": len(want), "n_ahead": len(ahead), "ahead_sample": ahead[:5],
                     "n_new_in_head": len(new_files), "new_sample": new_files[:5]}
    if ahead:
        return finish("REFUSED", ReplicaAhead.exit_code, reason=f"replica ahead of the fold: {len(ahead)} delivered file(s) the head does not name ("
                      + ", ".join(ahead[:3]) + (" …" if len(ahead) > 3 else "") + f"); head taken {head['taken_utc']}")
    if not new_files:
        finish("HEAD_NOT_NEWER", RefoldInputRefused.exit_code,
               reason=f"the head (taken {head['taken_utc']}, {len(want)} files) names no file the copy has not folded: an update that folds nothing certifies nothing")
        raise RefoldInputRefused(f"the head (taken {head['taken_utc']}, {len(want)} files) names no file the copy has not folded ({len(known)} delivered): "
                                 "an update that folds nothing certifies nothing -- not the fresh head this record names")
    log(f"[replica update] the head supersedes the copy's ledger: {len(new_files)} file(s) to fold, 0 ahead")
    try:
        mirror = require_mirror(lay, head, hash_mode, threads)
    except MirrorBehind as exc:
        _write_json(os.path.join(out_dir, "mirror.json"), verify_mirror(lay, head, hash_mode, threads))
        return finish("MIRROR_BEHIND", exc.exit_code, reason=str(exc))
    _write_json(os.path.join(out_dir, "mirror.json"), mirror)
    rec["mirror"] = {k: mirror.get(k) for k in ("ok", "hashed", "from_manifest", "hash_s", "n_missing", "n_differing")}
    log(f"[replica update] mirror holds every listed file ({mirror['hashed']} hashed, {mirror['from_manifest']} from the manifest, {mirror['hash_s']} s)")
    try:
        ing = ingest_head(copy, lay, head, workers=workers, site_aliases_sha256=aliases["digest"])
    except (ReplicaAhead, MirrorBehind, SchemaBehind, SiteAliasesRefused) as exc:
        return finish("REFUSED", exc.exit_code, reason=str(exc))
    _write_json(os.path.join(out_dir, "ingest.json"), ing)
    report = dict(ing.get("report") or {})
    rec["ingest"] = {**{k: ing.get(k) for k in ("delta_files", "changed_sha", "plain", "revision_dirs", "dimension_bundles", "stamped", "elapsed_s",
                                                 "unlisted_ignored", "now")},
                     "report": {k: report.get(k) for k in ("files", "records", "ok", "refused", "blocked", "unblocked", "duplicates", "conflicts",
                                                            "revisions_applied", "revisions_held", "revisions_noop", "lifecycle_regressions_held",
                                                            "harvester_records", "parse_s", "write_s", "transactions")}}
    log(f"[replica update] ingested {ing['delta_files']} file(s) ({ing['plain']} plain, {ing['revision_dirs']} revision dir(s), "
        f"{ing['dimension_bundles']} dimension bundle(s)) in {ing['elapsed_s']} s; unlisted ignored {ing['unlisted_ignored']}")
    chk = check(copy, head, projection_out=os.path.join(out_dir, "projection.json"))
    chk["identity"] = identity
    chk["site_aliases"] = aliases
    write_check_json(chk, os.path.join(out_dir, "check.json"))
    log(format_check(chk))
    rec["check"] = _check_summary(chk)
    _checkpoint(copy)
    rec["schema_version"] = max([r["version"] for r in schema_migration_rows(copy)] or [0])
    out = _copy_db_hashed(copy, os.path.join(out_dir, "registry.sqlite"))
    rec["output"] = out
    log(f"[replica update] output {out['path']}: {out['bytes']} bytes, sha256 {out['sha256']} ({out['copy_s']} s); schema {rec['schema_version']}")
    return finish("EQUAL" if chk["equal"] else "UNEQUAL", 0 if chk["equal"] else ReplicaAhead.exit_code, check_equal=bool(chk["equal"]))


def migrate_db(src: str, work_dir: str, out_dir: str, tree: str, expect_tree: str, migrate_tree: str, expect_migrate_tree: str, expect_schema: int,
               expect_input_sha256: str, python: str = sys.executable, head: Optional[Mapping[str, object]] = None, expect_head_sha256: Optional[str] = None,
               job_id: Optional[str] = None, log: Callable[[str], None] = print, site_aliases_path: Optional[str] = None,
               expect_site_aliases_sha256: Optional[str] = None) -> int:
    """``registry migrate`` over a KEPT database by the MIGRATING tree's own code (the ``--db`` form of ``replica migrate``). A
    certifying record, so everything is DECLARED: ``expect_input_sha256`` (the copy's streaming hash must equal it),
    ``tree`` / ``expect_tree`` (the running tree, bound by commit and by this module's location, :func:`_require_running_tree`),
    ``migrate_tree`` / ``expect_migrate_tree`` (the tree whose code migrates, bound by commit, :func:`require_tree`) and ``expect_schema``
    -- the version the migrated copy must reach (the live schema the head was taken at), without which a tree that carries no newer
    migration would read ``pending: none`` and pass; and, when a ``head`` is given, ``expect_head_sha256`` (:func:`require_head`). ``src`` is copied node-local and hashed as it is read (never written); then, as
    subprocesses of ``migrate_tree``'s code, ``registry migrate --db <copy>`` applies the pending migrations under the write lock beside the
    copy and verifies the shape against THAT tree's DDL, and ``registry migrate --plan --db <copy>`` must read ``pending: none`` with a
    matching shape; the ``schema_migration`` table must then stand at ``expect_schema``. A migration that fails, a shape left differing or
    a version other than the declared one refuses (:class:`MigrationNeedsRebuild`, exit 8, no output; ``migrate.log`` and ``plan.txt`` carry
    the tree's own words, the record the versions). The copy is checkpointed, checked against ``head`` when one is given (the running
    tree's :func:`check`, the full projection written beside) and copied out to ``<out_dir>/registry.sqlite``, hashed as it is written.
    ``migrate.json``: the input's and the output's identity, the versions before and after with the ``schema_migration`` rows added, both
    trees' identities and the migrating tree's migration files, the running code's version, the check summary. Why a tree argument: a kept
    database built by one tree can be carried to a schema whose migration only a NEWER tree carries (0026 landed on main after the branch's
    merge) without merging, and the migrating tree's own ``registry migrate`` is the operator command the window runs; the running tree's
    ``migrate`` over CURRENT is unchanged. Returns 0 (equal, or no head given) or 7 when the check is unequal; the output stands either way.

    The site-alias table is DECLARED too (:func:`declared_site_aliases`): ``site_aliases_path`` and ``expect_site_aliases_sha256`` are
    both required, the file is validated and its digest must be the declared one, and the migrating tree's ``registry migrate`` must
    accept ``--site-aliases`` (a tree that predates it would migrate with no table): all refused (:class:`SiteAliasesRefused`) before the
    copy. Both subprocesses get ``--site-aliases FILE --expect-site-aliases-sha256 H``, so the migrating tree checks the digest itself,
    and ``migrate.json`` names the table (``identity.site_aliases``: file, digest, pairs)."""
    _require_hex64("--expect-input-sha256", expect_input_sha256)
    for name, h in (("--expect-tree", expect_tree), ("--expect-migrate-tree", expect_migrate_tree)):
        if not _HEX40.match(h or ""):
            raise RefoldInputRefused(f"{name} must be the full 40 hex commit sha, got {h!r}")
    try:
        target = int(expect_schema)
    except (TypeError, ValueError):
        raise RefoldInputRefused(f"--expect-schema must be the schema version the migrated copy reaches (an integer), got {expect_schema!r}") from None
    if target < 1:
        raise RefoldInputRefused(f"--expect-schema must be a positive schema version, got {expect_schema!r}")
    if head is not None:
        if expect_head_sha256 is None:
            raise RefoldInputRefused("--head given without --expect-head-sha256: the head a check is read against is declared, never read off a path")
        require_head(head, expect_head_sha256)
    _, aliases = declared_site_aliases(site_aliases_path, expect_site_aliases_sha256)
    run_ident = _require_running_tree(tree, expect_tree)
    ident = require_tree(migrate_tree, expect_migrate_tree)
    if not migrating_cli_takes_site_aliases(migrate_tree):
        raise SiteAliasesRefused(f"--migrate-tree {migrate_tree}: its registry migrate knows no --site-aliases (src/sisteer/registry/cli.py), so it "
                                 "would migrate with no alias table; carry the copy with a tree that has it")
    mig_dir = os.path.join(os.path.abspath(migrate_tree), "src", "sisteer", "registry", "migrations")
    if not os.path.isdir(mig_dir):
        raise RefoldInputRefused(f"{migrate_tree}: no src/sisteer/registry/migrations -- not a tree whose code can migrate a database")
    ident["migration_files"] = sorted(f for f in os.listdir(mig_dir) if _MIGRATION_FILE.match(f))
    ident["highest_migration"] = max([int(_MIGRATION_FILE.match(f).group(1)) for f in ident["migration_files"]] or [0])
    if ident["highest_migration"] < target:
        raise RefoldInputRefused(f"--migrate-tree {migrate_tree} carries migrations up to {ident['highest_migration']:04d}, below the declared target "
                                 f"schema {target}: not the tree that can carry the copy there")
    _refuse_output_aliases(src, work_dir, out_dir)
    _clear_outputs(out_dir, MIGRATE_OUTPUTS)
    started = time.monotonic()
    identity = {"job_id": job_id, "built_by_tree": run_ident["bound_to"], "migrate_tree": ident["bound_to"], "out_dir": os.path.abspath(out_dir),
                "input_sha256": expect_input_sha256, "expect_schema": target,
                "head_sha256": head_sha256(head) if head is not None else None,
                "allowlist_sha256": head["allowlist_sha256"] if head is not None else None,
                "projection_sha256": head["projection"]["sha256"] if head is not None else None,
                "site_aliases": {k: aliases[k] for k in ("path", "digest", "pairs")}}
    from .migrations import code_version
    rec: Dict[str, object] = {"schema": MIGRATE_SCHEMA, "utc_start": R.utc_now(), "identity": identity, "status": None, "tree": run_ident,
                              "migrate_tree": ident, "running_code_version": code_version(), "expected_schema": target,
                              "head": head_summary(head) if head is not None else None, "input": None, "output": None,
                              "expected_input_sha256": expect_input_sha256, "expected_head_sha256": expect_head_sha256,
                              "expected_site_aliases_sha256": expect_site_aliases_sha256}

    def finish(status: str, rc: int, **extra) -> int:
        rec.update(status=status, rc=rc, elapsed_s=round(time.monotonic() - started, 1), utc_end=R.utc_now(), **extra)
        _write_json(os.path.join(out_dir, "migrate.json"), rec)
        log(f"[replica migrate] {status} rc={rc} ({rec['elapsed_s']} s) -> {out_dir}")
        return rc

    log(f"[replica migrate] site aliases: {aliases['path']}, digest {aliases['digest']}, {len(aliases['pairs'])} pair(s); passed to both subprocesses")
    staged = stage_input(src, work_dir, log=lambda m: log(f"[replica migrate] {m}"))
    rec["input"] = staged
    if staged["sha256"] != expect_input_sha256:
        finish("INPUT_REFUSED", RefoldInputRefused.exit_code, reason=f"the staged input's sha256 is {staged['sha256']}, not the expected {expect_input_sha256}")
        raise RefoldInputRefused(f"the staged input's sha256 is {staged['sha256']}, not the expected {expect_input_sha256}: not the database this migration names")
    copy = str(staged["copy"])
    rows_before = schema_migration_rows(copy)
    before = max([r["version"] for r in rows_before] or [0])
    rec["before"] = before
    argv, env = _tree_python(migrate_tree, python, module="sisteer.registry.cli")
    alias_args = ["--site-aliases", str(aliases["path"]), "--expect-site-aliases-sha256", str(aliases["digest"])]
    mig_argv = argv + ["migrate", "--db", copy] + alias_args
    t0 = time.monotonic()
    r = subprocess.run(mig_argv, env=env, capture_output=True, text=True)
    rec["migrate_s"] = round(time.monotonic() - t0, 1)
    rec["migrate_rc"] = r.returncode
    with open(os.path.join(out_dir, "migrate.log"), "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(mig_argv)}\n[rc {r.returncode}]\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}")
    rows_after = schema_migration_rows(copy)
    after = max([x["version"] for x in rows_after] or [0])
    rec["after"] = after
    rec["applied"] = [x for x in rows_after if x["version"] > before]
    tail = (r.stdout + "\n" + r.stderr).strip()[-600:]
    log(f"[replica migrate] registry migrate under {ident['bound_to']}: rc {r.returncode}, {before} -> {after}, applied {[x['version'] for x in rec['applied']]}")
    if r.returncode != 0:
        finish("MIGRATION_REFUSED", MigrationNeedsRebuild.exit_code, reason=tail)
        raise MigrationNeedsRebuild(f"registry migrate under {ident['bound_to']} exited {r.returncode} ({before} -> {after}): the migration failed or the shape "
                                    f"differs from that tree's DDL -- {tail}")
    plan_argv = argv + ["migrate", "--plan", "--db", copy] + alias_args
    r2 = subprocess.run(plan_argv, env=env, capture_output=True, text=True)
    plan_lines = (r2.stdout or "").strip().splitlines()
    with open(os.path.join(out_dir, "plan.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"$ {' '.join(plan_argv)}\n[rc {r2.returncode}]\n--- stdout\n{r2.stdout}\n--- stderr\n{r2.stderr}")
    rec["plan_rc"] = r2.returncode
    rec["plan"] = plan_lines[:16]
    if r2.returncode != 0:
        finish("MIGRATION_REFUSED", MigrationNeedsRebuild.exit_code, reason="; ".join(plan_lines[:4]))
        raise MigrationNeedsRebuild(f"after migrating {before} -> {after} under {ident['bound_to']} the plan still names a pending migration or a shape that "
                                    f"differs from that tree's DDL: " + "; ".join(plan_lines[:4]))
    if after != target:
        finish("SCHEMA_NOT_REACHED", MigrationNeedsRebuild.exit_code, reason=f"the migrated copy stands at schema {after}, not the declared {target}")
        raise MigrationNeedsRebuild(f"the migrated copy stands at schema {after} under {ident['bound_to']} (from {before}), not the declared target {target}: "
                                    "not the tree, or not the schema, this record names")
    _checkpoint(copy)
    equal = True
    if head is not None:
        chk = check(copy, head, projection_out=os.path.join(out_dir, "projection.json"))
        chk["identity"] = identity
        write_check_json(chk, os.path.join(out_dir, "check.json"))
        log(format_check(chk))
        rec["check"] = _check_summary(chk)
        equal = bool(chk["equal"])
    out = _copy_db_hashed(copy, os.path.join(out_dir, "registry.sqlite"))
    rec["output"] = out
    log(f"[replica migrate] output {out['path']}: {out['bytes']} bytes, sha256 {out['sha256']} ({out['copy_s']} s); schema {before} -> {after}")
    status = "MIGRATED" if head is None else ("EQUAL" if equal else "UNEQUAL")
    return finish(status, 0 if equal else ReplicaAhead.exit_code, check_equal=(None if head is None else equal))


def live_db_modules(tree: str) -> List[str]:
    """The ``tests/test_registry_*.py`` modules of ``tree`` declaring ``pytestmark = pytest.mark.live_db`` at module level (what
    ``land.sh --live-db-files`` lists)."""
    import re
    out = []
    for p in sorted(glob.glob(os.path.join(tree, "tests", "test_registry_*.py"))):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        if re.search(r"^pytestmark\s*=\s*pytest\.mark\.live_db", text, re.M):
            out.append(os.path.relpath(p, tree))
    return out


def parse_junit(path: str) -> Dict[str, int]:
    """``{tests, failures, errors, skipped}`` summed over a pytest JUnit report's suites."""
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for s in suites:
        for k in counts:
            counts[k] += int(s.get(k) or 0)
    return counts


def _default_runner(argv: Sequence[str], env: Mapping[str, str], cwd: str, log_path: str, timeout_s: int) -> int:
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            r = subprocess.run(list(argv), cwd=cwd, env=dict(env), stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.write(f"\n[replica land] pytest exceeded {timeout_s} s\n")
            return 124
    return r.returncode


def land(lay: Layout, head: Mapping[str, object], tree: str, work_dir: str, out_dir: str, pytest_python: str, branch: str,
         expected_skipped: int = 0, workers: int = 4, hash_mode: str = "all", threads: int = 8, built_by_tree: Optional[str] = None,
         job_id: Optional[str] = None, runner: Optional[Callable[..., int]] = None, pytest_timeout_s: int = 3600,
         log: Callable[[str], None] = print, accept_c1_class: Optional[str] = None) -> int:
    """The landing job (module docstring). Writes ``<out_dir>/{check.json, mirror.json, ingest.json, junit.xml, pytest.log,
    summary.json}`` (and ``movers.json`` when the check's mover list is longer than :data:`MOVERS_INLINE_CAP`) -- every one of
    ``LAND_OUTPUTS`` already there is removed first, so the directory holds only this job's -- each of ``check.json`` and ``summary.json``
    stamped with the job's ``identity`` (``job_id``, ``built_by_tree``, the head's ``head_sha256`` / ``allowlist_sha256`` /
    ``projection_sha256``, ``out_dir``). Returns 0 only when the check PASSED, pytest exited 0, the skips equal ``expected_skipped`` AND the
    ingested generation was published (``summary.published`` = the new CURRENT); a refused publication is ``PUBLISH_REFUSED`` (exit 10)
    whatever pytest said.

    The check passes when it is ``equal`` (``summary.check_accepted_as = "equal"``) or -- ONLY with ``accept_c1_class`` set to the
    operator's declaration naming the refold-compare certification (the cutover landing's allowance, retired once the live database
    adopts the C1 order) -- when ``check.c1_class`` holds: the projection differs from the head's on the certified same-stamp class and on
    nothing else (:func:`classify_projection_diff`). Then ``summary.check_accepted_as = "c1_class"``, ``check_equal`` stays ``False``,
    ``c1_class_movers`` and ``c1_class_declaration`` record the count and the declaration verbatim, and the generation is published.
    Without the declaration a ``c1_class`` check is ``UNEQUAL`` exactly as before (no pytest, nothing published, exit 1); with it, a check
    whose diff touches any other column is refused the same way.

    The increment folds under the replica root's alias table (:func:`install_root_aliases`), read and validated before the copy (a refused
    table is ``REFUSED``, exit 13, nothing copied or published); its record goes into ``summary.json``, ``check.json`` and the published
    generation's ``meta.json`` (``site_aliases``)."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    for name in LAND_OUTPUTS:
        try:
            os.remove(os.path.join(out_dir, name))
        except FileNotFoundError:
            pass
    started = time.monotonic()
    identity = {"job_id": job_id, "built_by_tree": built_by_tree, "head_sha256": head_sha256(head), "allowlist_sha256": head["allowlist_sha256"],
                "projection_sha256": head["projection"]["sha256"], "out_dir": os.path.abspath(out_dir)}
    summary: Dict[str, object] = {"schema": "replica_land/1", "branch": branch, "tree": os.path.abspath(tree), "head_taken_utc": head["taken_utc"],
                                  "head_n_files": head["n_files"], "identity": identity, "status": None, "expected_skipped": int(expected_skipped),
                                  "utc_start": R.utc_now(), "accept_c1_class_declared": bool(accept_c1_class), "check_accepted_as": None}

    def finish(status: str, rc: int, **extra) -> int:
        summary.update(status=status, rc=rc, elapsed_s=round(time.monotonic() - started, 1), utc_end=R.utc_now(), **extra)
        _write_json(os.path.join(out_dir, "summary.json"), summary)
        log(f"[replica land] {status} rc={rc} ({summary['elapsed_s']} s) -> {out_dir}")
        return rc

    def write_check(chk: Dict[str, object]) -> None:
        chk["identity"] = identity
        write_check_json(chk, os.path.join(out_dir, "check.json"))

    cur = lay.current()
    if cur is None:
        write_check({"schema": CHECK_SCHEMA, "equal": False, "reason": "no CURRENT generation"})
        return finish("NO_GENERATION", NoGeneration.exit_code, reason="no CURRENT generation: bootstrap with `replica build`")
    summary["parent_generation"] = cur
    try:
        _, aliases = root_site_aliases(lay)
    except SiteAliasesRefused as exc:
        write_check({"schema": CHECK_SCHEMA, "equal": False, "reason": str(exc)})
        return finish("REFUSED", exc.exit_code, reason=str(exc))
    summary["site_aliases"] = aliases
    log(f"[replica land] site aliases: {aliases['path']} ({'present' if aliases['present'] else 'absent: the empty table'}), digest {aliases['digest']}, "
        f"{len(aliases['pairs'])} pair(s)")
    work_db = os.path.join(work_dir, "registry.sqlite")
    t0 = time.monotonic()
    n_bytes = _copy_db(lay.db_of(cur), work_db)
    summary["copy"] = {"bytes": n_bytes, "seconds": round(time.monotonic() - t0, 1)}
    log(f"[replica land] copied generation {cur} ({n_bytes} bytes) in {summary['copy']['seconds']} s")
    try:
        mirror = require_mirror(lay, head, hash_mode, threads)
    except MirrorBehind as exc:
        _write_json(os.path.join(out_dir, "mirror.json"), verify_mirror(lay, head, hash_mode, threads))
        write_check({"schema": CHECK_SCHEMA, "equal": False, "reason": str(exc)})
        return finish("MIRROR_BEHIND", exc.exit_code, reason=str(exc))
    _write_json(os.path.join(out_dir, "mirror.json"), mirror)
    log(f"[replica land] mirror holds every listed file ({mirror['hashed']} hashed, {mirror['from_manifest']} from the manifest, {mirror['hash_s']} s)")
    try:
        ing = ingest_head(work_db, lay, head, workers=workers, site_aliases_sha256=aliases["digest"])
    except (ReplicaAhead, MirrorBehind, SchemaBehind, SiteAliasesRefused) as exc:
        write_check({"schema": CHECK_SCHEMA, "equal": False, "reason": str(exc)})
        return finish("REFUSED", exc.exit_code, reason=str(exc))
    _write_json(os.path.join(out_dir, "ingest.json"), ing)
    log(f"[replica land] ingested {ing['delta_files']} file(s) ({ing['plain']} plain, {ing['revision_dirs']} revision dir(s), "
        f"{ing['dimension_bundles']} dimension bundle(s)) in {ing['elapsed_s']} s; unlisted ignored {ing['unlisted_ignored']}")
    chk = check(work_db, head)
    chk["site_aliases"] = aliases
    write_check(chk)
    log(format_check(chk))
    summary["check_equal"] = bool(chk["equal"])
    summary["check_c1_class"] = bool(chk.get("c1_class"))
    if chk["equal"]:
        summary["check_accepted_as"] = CHECK_ACCEPTED_EQUAL
    elif accept_c1_class and chk.get("c1_class"):
        # the cutover allowance: the certified C1 same-stamp class, declared by the operator, and nothing else
        summary.update(check_accepted_as=CHECK_ACCEPTED_C1_CLASS, c1_class_movers=int(chk.get("n_movers") or 0), c1_class_declaration=accept_c1_class)
        log(f"[replica land] check ACCEPTED as equal modulo the C1 same-stamp class ({summary['c1_class_movers']} movers) by declaration: {accept_c1_class}")
    else:
        hint = (" (the C1 same-stamp class, not declared: no allowance without LAND_ACCEPT_C1_CLASS)" if chk.get("c1_class")
                else (" (not the C1 class alone; the declaration does not cover it)" if accept_c1_class else ""))
        return finish("UNEQUAL", 1, check_equal=False, reason="replica check UNEQUAL" + hint)
    modules = live_db_modules(tree)
    summary["modules"] = modules
    junit = os.path.join(out_dir, "junit.xml")
    bt = os.path.join(work_dir, "bt", "a", "b", "c")
    os.makedirs(bt, exist_ok=True)
    argv = [pytest_python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider", "-m", "live_db", f"--basetemp={bt}", f"--junitxml={junit}",
            *modules]
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.path.join(tree, "src") + ":" + os.path.join(tree, "scripts"), "REGISTRY_LIVE_DB": work_db,
                "REGISTRY_PIN_L1": lay.pin_l1, "TMPDIR": os.path.join(work_dir, "tmp"), "PYTHONDONTWRITEBYTECODE": "1"})
    os.makedirs(env["TMPDIR"], exist_ok=True)
    summary["pytest_argv"] = argv
    t1 = time.monotonic()
    rc = (runner or _default_runner)(argv, env, tree, os.path.join(out_dir, "pytest.log"), pytest_timeout_s)
    summary["pytest_s"] = round(time.monotonic() - t1, 1)
    counts = parse_junit(junit) if os.path.isfile(junit) else None
    summary["pytest_rc"] = rc
    summary["counts"] = counts
    skips_ok = counts is not None and counts["skipped"] == int(expected_skipped)
    log(f"[replica land] pytest rc={rc} counts={counts} expected_skipped={expected_skipped}")
    # the generation is equal to the grading host fold (or, under the declaration, equal modulo the certified class) whatever the tests say:
    # published so the next landing starts from it. A REFUSED publication (CURRENT moved between the copy and now -- an operator's
    # rollback, another writer) is a failing status whatever the tests said: the next landing would otherwise start from a generation
    # this job never produced.
    try:
        meta = publish(lay, work_db, head, mode="ingest", parent=cur, built_by_tree=built_by_tree, job_id=job_id,
                       extra={"branch": branch, "site_aliases": aliases})
        summary["published"] = meta["gen_id"]
        summary["published_utc"] = meta.get("published_utc")
        summary["current_after_publish"] = lay.current()
        log(f"[replica land] published generation {meta['gen_id']} (parent {cur}; pruned {meta.get('pruned')})")
    except PublishRefused as exc:
        summary["published"] = None
        summary["publish_refused"] = str(exc)
        summary["current_after_publish"] = lay.current()
        log(f"[replica land] publish REFUSED: {exc}")
        return finish("PUBLISH_REFUSED", PublishRefused.exit_code, reason=f"the ingested generation was not published: {exc}")
    if rc != 0 or counts is None:
        return finish("FAIL", 1 if rc == 0 else rc)
    if not skips_ok:
        return finish("SKIPS", 1, reason=f"{counts['skipped']} skipped, expected {expected_skipped}")
    return finish("PASS", 0)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# the C1 measurement: refold-compare
# --------------------------------------------------------------------------------------------


def refold_dump(conn, table: str = "run") -> List[Dict[str, object]]:
    """Every ``table`` key's folded row from its ``row_history`` events, in THIS tree's reducer order (``Reducer._load_events`` +
    ``fold``): ``[{key, state, n_events}]`` sorted by key."""
    from . import reducer as RD
    red = RD.Reducer(conn)
    out = []
    keys = [str(r[0]) for r in conn.execute("SELECT DISTINCT natural_key_json FROM row_history WHERE table_name = ? ORDER BY natural_key_json", (table,))]
    for key in keys:
        events = red._load_events(table, key)
        folded = RD.fold(table, events)
        state = folded[-1][0] if folded else {}
        out.append({"key": key, "state": json.loads(canonical_json(state)), "n_events": len(events)})
    return out


def _restamp_rewritten(conn, run_ids: Sequence[str]) -> int:
    """The post-fold tail a scoped refold owes the ``run`` rows it rewrote, as the harvester runs it after its own scoped fold
    (``harvest.fold_only_files``): the DERIVED decode-policy stamp (``decode_match.stamp_run_rows``) on exactly those rows. Inside the
    caller's transaction; returns the rows re-stamped. Measured omission (a first certification attempt, 2026-10-06): a fold rewrites
    ``decode_policy_match`` to the record's placeholder ``no_policy`` where the live and rebuilt rows carry the derived ``exact``, and
    nothing else differs -- so a refold without this tail cannot round-trip a row, and a projection over it certifies nothing."""
    from . import decode_match as DM
    ids = sorted({r for r in run_ids if r})
    return int(DM.stamp_run_rows(conn, run_ids=ids)) if ids else 0


def _refold_rows(conn, keys: Sequence[str], table: str = "run") -> Tuple[int, List[str]]:
    """:func:`refold_apply`'s body, inside the CALLER's transaction or savepoint: for every named key, the refold under THIS tree's order
    rewrites the key's whole ``row_history`` prefix chain (``Reducer.rewrite_history`` from 0 -- the same writer the live fold uses, so
    ``registry history`` and ``harvest.verify_equal`` read the new order's snapshots, not the old one's) and the live row WITH the
    provenance a rebuild derives for it -- ``record_uri`` the canonical carrier of the key's LAST event under this order, ``ingest_seq``
    that file's ledger rank (``Reducer.seq_of``; the row's own rank kept only when the ledger has none), exactly as ``apply_run_scale``
    stamps a live fold -- then the post-fold tail on the rewritten ``run`` rows. The carrier matters: a same-stamp pair
    whose events sit in two files draws a different final event under the new order, and a refold that kept the old row's carrier would leave a
    "verified" restamp unequal to the rebuild on ``run.record_uri`` and ``run.ingest_seq``. Returns ``(rows rewritten, the run ids
    rewritten)`` and sets :data:`LAST_RESTAMPED`. The ONE fold writer: the apply commits it, the restamp plan (:func:`restamp_keys_plan`)
    runs it under a savepoint it rolls back -- the history rewrites included -- so the plan lists exactly what the apply would write."""
    global LAST_RESTAMPED
    from . import reducer as RD
    # the writers' fold-order check (db.require_fold_order) on whatever connection the caller opened -- `refold-apply`, `restamp-keys` and
    # the plan's dry run open with DB.connect, not open_or_create: a database recording another order refuses by name before a row moves
    # (a refold IS this tree's order written into the rows); one recording none is stamped (inside this transaction; a plan's savepoint
    # rolls the stamp back with the rest, the apply commits it). The database enforces the same rule on the `run` writes below itself
    # (the fence triggers of migration 0030 against db.FOLD_ORDER_FUNCTION, which `conn` carries by being a db.connect): the refold-apply
    # of a checkout from before this check -- commit 5ea3fa411's, a bare DB.connect and no guard -- is refused at the row, nothing written
    DB.require_fold_order(conn)
    red = RD.Reducer(conn)
    n = 0
    rewritten: List[str] = []
    for key in keys:
        events = red._load_events(table, key)
        folded = RD.fold(table, events)
        if not folded:
            continue
        state, fv = folded[-1]
        kd = json.loads(key)
        where = " AND ".join(f"{k} = ?" for k in kd)
        row = conn.execute(f"SELECT record_uri, ingest_seq FROM {table} WHERE {where}", tuple(kd.values())).fetchone()
        if row is None:
            continue
        red.rewrite_history(table, kd, key, events, folded, start=0)
        carrier, seq = refold_provenance(red, events, fallback_seq=int(row[1]))
        red._write_live_row(table, kd, state, fv, carrier, seq)
        n += 1
        if table == "run":
            rewritten.append(str(kd.get("run_id")))
    LAST_RESTAMPED = _restamp_rewritten(conn, rewritten) if table == "run" else 0
    return n, rewritten


def history_mismatches(conn, keys: Sequence[str], table: str = "run") -> List[Dict[str, object]]:
    """Every ``row_history`` snapshot of ``keys`` that does NOT read as THIS tree's fold of the key's ordered events: ``[{key, written_utc,
    record_sha256, stored, expected}]`` (the two snapshots as canonical JSON). Empty when the stored prefix chain IS the fold -- what a
    restamp's verification requires beside the live cells (a live row repaired alone would leave the old order's
    prefixes in the history)."""
    from . import reducer as RD
    red = RD.Reducer(conn)
    out: List[Dict[str, object]] = []
    for key in keys:
        events = red._load_events(table, key)
        folded = RD.fold(table, events)
        stored = {(str(r[0]), str(r[1])): str(r[2]) for r in conn.execute(
            "SELECT written_utc, record_sha256, folded_state_json FROM row_history WHERE table_name = ? AND natural_key_json = ?", (table, key))}
        for e, (snap, _fv) in zip(events, folded):
            want = canonical_json(snap)
            have = stored.get((e.written_utc, e.sha))
            if have is None or json.loads(have) != json.loads(want):
                out.append({"key": key, "written_utc": e.written_utc, "record_sha256": e.sha, "stored": have, "expected": want})
    return out


def refold_provenance(red, events: Sequence[object], fallback_seq: Optional[int] = None) -> Tuple[str, Optional[int]]:
    """The live row's provenance under THIS order, as the live fold (``Reducer.apply_run_scale``) and a rebuild derive it: the canonical
    carrier of the key's LAST ordered event and that file's ledger rank (``Reducer.seq_of``), ``fallback_seq`` when the ledger holds no
    rank for the carrier (the fold's own fallback is the applying record's rank; a refold has only the row's)."""
    last = events[-1]
    seq = red.seq_of(last.record_uri)
    return str(last.record_uri), (int(seq) if seq is not None else fallback_seq)


#: The ``run`` columns the post-fold decode tail (``decode_match.stamp_run_rows``) owns or edits AFTER the fold: the two derived columns,
#: their stamps inside ``field_versions_json`` and the tie-break audit key inside ``extra_json`` -- a live row is compared with its fold
#: around them (the tail's values are the derived-stamp pass's to verify, not the fold's).
_DECODE_TAIL_COLUMNS: Tuple[str, ...] = ("decode_policy_id", "decode_policy_match")


def live_row_mismatches(conn, keys: Sequence[str], table: str = "run") -> List[Dict[str, object]]:
    """Every live ``table`` row of ``keys`` that does NOT read as THIS tree's refold of the key's ordered events -- the WHOLE row, not the
    columns a plan happened to list: ``[{key, column, expected, read}]``. Compared: the provenance a rebuild derives
    (:func:`refold_provenance`: ``record_uri`` the last event's carrier, ``ingest_seq`` its ledger rank when the ledger has one), every
    payload column the fold states (through SQLite's own comparison, so the column's affinity applies to the fold's value exactly as the
    upsert applied it), ``field_versions_json`` and ``extra_json`` as parsed JSON minus what the decode tail writes
    (:data:`_DECODE_TAIL_COLUMNS` stamps, ``decode_match.TIEBREAK_KEY``). A key with no live row is one mismatch (``column`` ``row``).
    Empty when the live row IS the refold -- what a restamp's verification requires beside the planned values and the history prefixes
    (a verification over the plan's own columns alone would pass a row whose carrier the rewrite had
    kept from the old order)."""
    from . import decode_match as DM
    from . import reducer as RD
    from . import schema_def as S
    red = RD.Reducer(conn)
    payload_cols = set(S.payload_columns(table))
    out: List[Dict[str, object]] = []
    for key in keys:
        events = red._load_events(table, key)
        folded = RD.fold(table, events)
        kd = json.loads(key)
        where = " AND ".join(f"{k} = ?" for k in kd)
        row = conn.execute(f"SELECT * FROM {table} WHERE {where}", tuple(kd.values())).fetchone()
        if not folded:
            continue
        if row is None:
            out.append({"key": key, "column": "row", "expected": "a live row", "read": None})
            continue
        live = {c: row[c] for c in row.keys()}
        state, fv = folded[-1]
        carrier, seq = refold_provenance(red, events)
        if str(live.get("record_uri")) != carrier:
            out.append({"key": key, "column": "record_uri", "expected": carrier, "read": live.get("record_uri")})
        if seq is not None and live.get("ingest_seq") != seq:
            out.append({"key": key, "column": "ingest_seq", "expected": seq, "read": live.get("ingest_seq")})
        skip = set(_DECODE_TAIL_COLUMNS) | {"field_versions_json", "extra_json"}
        cols = [c for c in sorted(state) if c in payload_cols and c not in skip]
        if cols:
            same = conn.execute(f"SELECT {', '.join(f'({c} IS ?)' for c in cols)} FROM {table} WHERE {where}",
                                tuple(state[c] for c in cols) + tuple(kd.values())).fetchone()
            for c, ok in zip(cols, same):
                if not ok:
                    out.append({"key": key, "column": c, "expected": state[c], "read": live.get(c)})
        want_fv = {k: v for k, v in fv.items() if k not in _DECODE_TAIL_COLUMNS}
        have_fv = {k: v for k, v in (json.loads(live.get("field_versions_json") or "{}")).items() if k not in _DECODE_TAIL_COLUMNS}
        if json.loads(canonical_json(want_fv)) != have_fv:
            out.append({"key": key, "column": "field_versions_json", "expected": canonical_json(want_fv), "read": live.get("field_versions_json")})
        if "extra_json" in payload_cols:
            want_x = dict(json.loads(str(state.get("extra_json") or "{}")))
            have_x = dict(json.loads(str(live.get("extra_json") or "{}")))
            want_x.pop(DM.TIEBREAK_KEY, None)
            have_x.pop(DM.TIEBREAK_KEY, None)
            if want_x != have_x:
                out.append({"key": key, "column": "extra_json", "expected": canonical_json(want_x), "read": live.get("extra_json")})
    return out


def refold_apply(conn, keys: Sequence[str], table: str = "run") -> int:
    """Rewrite the live ``table`` row of every named key from its refold under THIS tree's order -- the row's ``record_uri`` and
    ``ingest_seq`` those of the key's last event under it (:func:`refold_provenance`) -- then re-derive the post-fold stamps on the
    rewritten ``run`` rows (:func:`_restamp_rewritten`). One transaction; returns the rows rewritten (:data:`LAST_RESTAMPED` holds the
    rows the tail re-stamped)."""
    with DB.transaction(conn):
        n, _rewritten = _refold_rows(conn, keys, table)
    return n


#: The rows :func:`refold_apply`'s tail re-stamped on its last call (the CLI prints it beside the rewritten count).
LAST_RESTAMPED = 0


def restamp_runs(conn, keys: Sequence[str]) -> int:
    """The tail alone, for rows another tree's ``refold-apply`` rewrote without it (a tree exported before the tail existed): one
    transaction, returns the rows re-stamped. Idempotent (``stamp_run_rows`` writes only rows whose stamped columns differ)."""
    run_ids = [str(json.loads(k).get("run_id")) for k in keys if json.loads(k).get("run_id")]
    with DB.transaction(conn):
        return _restamp_rewritten(conn, run_ids)


# --------------------------------------------------------------------------------------------
# the lifecycle restamp of named runs on the LIVE database (window.sh WINDOW_RESTAMP_KEYS)
# --------------------------------------------------------------------------------------------


RESTAMP_SCHEMA = "restamp_keys/1"
RESTAMP_PLAN_FILE = "restamp_plan.json"
RESTAMP_APPLY_FILE = "restamp_apply.json"
#: The ``run`` TABLE columns a refold under the reducer's order may move for a run whose lifecycle folds differently -- the fold-level
#: counterpart of :data:`C1_CLASS_COLUMNS` (which names the PROJECTION's CASE columns): the lifecycle itself and the stamps a ``run_state``
#: event owns in ``reducer.fold`` (the state's ``*_utc`` column, ``finished_utc`` for a terminal state, ``judged_utc`` carried by an
#: ``ingested`` stamp). A change to any other column -- the decode-policy stamp included -- is OUT OF CLASS and refuses the plan by name.
RESTAMP_CLASS_COLUMNS: Tuple[str, ...] = ("lifecycle_state",) + tuple(sorted(set(V.LIFECYCLE_UTC_COLUMN.values()) | {"finished_utc"}))
#: Provenance a rewrite moves WITH the columns it moves: the stamps of the folded columns, and the carrier and ledger rank of the key's
#: final event (``record_uri``, ``ingest_seq``: a same-stamp pair in two files draws a different final event under the new order, so the
#: row's provenance follows it, as a rebuild's does). Reported in the plan, never a refusal.
RESTAMP_PROVENANCE_COLUMNS: Tuple[str, ...] = ("field_versions_json", "record_uri", "ingest_seq")


def _run_rows(conn, run_ids: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """The live ``run`` rows of ``run_ids`` as dicts, every column."""
    out: Dict[str, Dict[str, object]] = {}
    ids = list(run_ids)
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        marks = ",".join("?" * len(chunk))
        for r in conn.execute(f"SELECT * FROM run WHERE run_id IN ({marks})", chunk):
            d = {k: r[k] for k in r.keys()}
            out[str(d["run_id"])] = d
    return out


def restamp_keys_plan(conn, keys: Sequence[object], now: Optional[str] = None) -> Dict[str, object]:
    """THE PLAN (a dry run, nothing committed): what ``restamp-keys --apply`` would write for ``keys`` (run ids or natural keys,
    :func:`expected_keys_json`) on THIS database under THIS tree's reducer -- the same code path as the apply, :func:`_refold_rows` (the
    fold, the rewrite, the post-fold decode stamp), run under a SAVEPOINT that is rolled back, the rows read before and after and every
    differing ``(run_id, column, old, new)`` listed. Refuses by name (``ok`` false, the reasons listed): a key with no live ``run`` row or no
    ``row_history`` event (``unknown``), and any change outside the lifecycle class (``out_of_class``: a column not in
    :data:`RESTAMP_CLASS_COLUMNS`; :data:`RESTAMP_PROVENANCE_COLUMNS` are reported and admitted). ``changes_sha256`` binds the apply to this
    very change set."""
    nat = expected_keys_json(keys)
    run_ids = [str(json.loads(k).get("run_id")) for k in nat]
    before = _run_rows(conn, run_ids)
    unknown: List[Dict[str, object]] = []
    for k, rid in zip(nat, run_ids):
        if rid not in before:
            unknown.append({"run_id": rid, "why": "no live run row"})
            continue
        n_ev = int(conn.execute("SELECT COUNT(*) FROM row_history WHERE table_name = 'run' AND natural_key_json = ?", (k,)).fetchone()[0])
        if not n_ev:
            unknown.append({"run_id": rid, "why": "no row_history event to fold from"})
    changes: List[Dict[str, object]] = []
    n_would_rewrite = 0
    if not unknown and nat:
        conn.execute("SAVEPOINT restamp_plan")
        try:
            n_would_rewrite, _rewritten = _refold_rows(conn, nat, "run")
            after = _run_rows(conn, run_ids)
        finally:
            conn.execute("ROLLBACK TO restamp_plan")
            conn.execute("RELEASE restamp_plan")
        for rid in run_ids:
            a, b = before[rid], after.get(rid) or {}
            for col in sorted(set(a) | set(b)):
                if a.get(col) != b.get(col):
                    changes.append({"run_id": rid, "column": col, "old": a.get(col), "new": b.get(col)})
    allowed = set(RESTAMP_CLASS_COLUMNS) | set(RESTAMP_PROVENANCE_COLUMNS)
    out_of_class = [c for c in changes if c["column"] not in allowed]
    moved = sorted({str(c["run_id"]) for c in changes if c["column"] in RESTAMP_CLASS_COLUMNS})
    columns = {}
    for c in changes:
        columns[str(c["column"])] = columns.get(str(c["column"]), 0) + 1
    return {"schema": RESTAMP_SCHEMA, "utc": now or R.utc_now(), "n_keys": len(nat), "keys": nat, "n_unknown": len(unknown), "unknown": unknown,
            "n_would_rewrite": n_would_rewrite, "n_changes": len(changes), "n_runs_changed": len({str(c["run_id"]) for c in changes}),
            "n_runs_lifecycle_moved": len(moved), "runs_lifecycle_moved": moved, "columns": columns, "changes": changes,
            "n_out_of_class": len(out_of_class), "out_of_class": out_of_class[:50], "class_columns": list(RESTAMP_CLASS_COLUMNS),
            "provenance_columns": list(RESTAMP_PROVENANCE_COLUMNS), "changes_sha256": hashlib.sha256(canonical_json(changes).encode("utf-8")).hexdigest(),
            "ok": not unknown and not out_of_class}


def restamp_keys_apply(conn, plan: Mapping[str, object], now: Optional[str] = None) -> Dict[str, object]:
    """THE APPLY, inside the window with the writers quiet: the recorded ``plan`` must read ``ok``; in ONE ``BEGIN IMMEDIATE`` transaction
    the plan is computed AGAIN on this database and must carry the SAME change set (``changes_sha256``: a row that moved since the plan was
    read refuses by name, nothing written), :func:`_refold_rows` rewrites every key -- the history prefix chain, the live row, the decode
    tail -- and the VERIFICATION re-reads every planned ``(run_id, column)`` against its planned ``new`` AND re-derives every history
    snapshot of every key against this tree's fold (:func:`history_mismatches`); any mismatch is :class:`RestampVerifyFailed` and the whole
    transaction ROLLS BACK (nothing written; exit 14 for the CLI). The verification reads the WHOLE row of every key, not the plan's
    columns alone: every column of every key's row must read its planned ``new`` or, unplanned, its
    value before the rewrite (``unplanned_moves``: a column the rewrite moved that the plan never listed), and the row must read as this
    tree's refold (:func:`live_row_mismatches`: the carrier and rank of the final event, the stated columns, the stamps). Returns the
    record: keys, rows rewritten and re-stamped, runs and changes, the columns and snapshots verified. The CLI runs this only with the
    window's maintenance flag held (window-only by construction)."""
    if not plan.get("ok"):
        raise RefoldInputRefused("restamp-keys --apply: the recorded plan is not ok (unknown keys or changes out of the lifecycle class); nothing written")
    keys = [str(k) for k in (plan.get("keys") or [])]          # the recorded natural keys (canonical JSON), as the apply rewrites them
    changes = list(plan.get("changes") or [])
    # ONE transaction from the re-plan through the rewrite and the verification (BEGIN IMMEDIATE: the write lock is held across all three,
    # so no writer can slip between the plan the apply is bound to and the rows it rewrites); a verification that fails ROLLS IT ALL BACK
    with DB.transaction(conn):
        fresh = restamp_keys_plan(conn, [json.loads(k) for k in keys])   # re-planned as natural keys (a string would read as a run id)
        if not fresh["ok"]:
            raise RefoldInputRefused(f"restamp-keys --apply: the plan re-computed now is not ok ({fresh['n_unknown']} unknown, {fresh['n_out_of_class']} out of class); nothing written")
        if fresh["changes_sha256"] != plan.get("changes_sha256"):
            raise RefoldInputRefused(f"restamp-keys --apply: the change set moved since the plan ({plan.get('changes_sha256')} -> {fresh['changes_sha256']}: "
                                     f"{plan.get('n_changes')} -> {fresh['n_changes']} change(s)); re-plan before applying; nothing written")
        run_ids = [str(json.loads(k).get("run_id")) for k in keys]
        before = _run_rows(conn, run_ids)
        n, _rewritten = _refold_rows(conn, keys, "run")
        restamped = LAST_RESTAMPED
        after = _run_rows(conn, run_ids)
        planned = {(str(c["run_id"]), str(c["column"])): c["new"] for c in changes}
        mismatches: List[Dict[str, object]] = []
        unplanned: List[Dict[str, object]] = []
        n_columns = 0
        for rid in run_ids:
            a, b = before.get(rid) or {}, after.get(rid) or {}
            for col in sorted(set(a) | set(b)):
                n_columns += 1
                if (rid, col) in planned:
                    if b.get(col) != planned[(rid, col)]:
                        mismatches.append({"run_id": rid, "column": col, "new": planned[(rid, col)], "read": b.get(col)})
                elif b.get(col) != a.get(col):
                    unplanned.append({"run_id": rid, "column": col, "old": a.get(col), "read": b.get(col)})
        rows = live_row_mismatches(conn, keys, "run")
        history = history_mismatches(conn, keys, "run")
        if mismatches or unplanned or rows or history:
            raise RestampVerifyFailed(f"restamp-keys --apply: the re-read after the rewrite does not match the plan and the refold ({len(mismatches)} planned "
                                      f"value(s), {len(unplanned)} unplanned move(s), {len(rows)} row column(s) off the refold, {len(history)} history "
                                      f"snapshot(s)); the transaction is rolled back, nothing written: "
                                      + "; ".join(f"{m['run_id']} {m['column']}" for m in mismatches[:5])
                                      + "; ".join(f"unplanned {m['run_id']} {m['column']}" for m in unplanned[:5])
                                      + "; ".join(f"refold {m['key']} {m['column']}" for m in rows[:5])
                                      + "; ".join(f"history {h['key']} @ {h['written_utc']}" for h in history[:5]))
    return {"schema": RESTAMP_SCHEMA, "utc": now or R.utc_now(), "n_keys": len(keys), "n_rows_rewritten": n, "n_rows_restamped": restamped,
            "n_runs_changed": len({str(c["run_id"]) for c in changes}), "n_changes": len(changes), "n_verified": len(changes), "mismatches": [],
            "n_rows_verified": len(run_ids), "n_columns_verified": n_columns, "unplanned_moves": [], "row_mismatches": 0, "history_mismatches": 0,
            "n_history_snapshots_verified": sum(1 for k in keys for _ in conn.execute("SELECT 1 FROM row_history WHERE table_name = 'run' AND natural_key_json = ?", (k,))),
            "verified": True, "plan_changes_sha256": plan.get("changes_sha256"), "plan_utc": plan.get("utc")}


def format_restamp_plan(plan: Mapping[str, object], path: str) -> str:
    """The plan for the chain log: every lifecycle-class change on its own line (``run_id column: old -> new``), the provenance moves as a
    count, the unknown keys and the out-of-class changes by name, and the verdict line LAST (window.sh tails the output)."""
    lines = []
    for c in plan.get("changes") or []:
        if c["column"] in (plan.get("provenance_columns") or ()):
            continue
        lines.append(f"  {c['run_id']} {c['column']}: {c['old']!r} -> {c['new']!r}" + ("" if c["column"] in (plan.get("class_columns") or ()) else "   OUT OF CLASS"))
    prov = sum(1 for c in (plan.get("changes") or []) if c["column"] in (plan.get("provenance_columns") or ()))
    if prov:
        lines.append(f"  (+ {prov} provenance change(s) in {', '.join(plan.get('provenance_columns') or [])}: the stamps of the moved columns, "
                     "the final event's carrier and rank)")
    for u in plan.get("unknown") or []:
        lines.append(f"  UNKNOWN KEY {u['run_id']}: {u['why']}")
    verdict = "OK" if plan.get("ok") else "REFUSED"
    lines.append(f"restamp keys: plan over {plan['n_keys']} key(s): {plan['n_runs_lifecycle_moved']} run(s) move in the lifecycle class "
                 f"({', '.join(f'{k} {v}' for k, v in sorted((plan.get('columns') or {}).items())) or 'no column'}), {plan['n_unknown']} unknown key(s), "
                 f"{plan['n_out_of_class']} change(s) out of class -> {verdict} (plan {path})")
    return "\n".join(lines)


def format_restamp_apply(rec: Mapping[str, object], path: str) -> str:
    verdict = "OK" if rec.get("verified") else "VERIFY FAILED"
    lines = [f"  MISMATCH {m['run_id']} {m['column']}: planned {m['new']!r}" for m in (rec.get("mismatches") or [])]
    lines.append(f"restamp keys: applied over {rec['n_keys']} key(s): {rec['n_rows_rewritten']} row(s) rewritten, {rec['n_rows_restamped']} re-stamped, "
                 f"{rec['n_runs_changed']} run(s) changed, {rec['n_verified']}/{rec['n_changes']} change(s), {rec.get('n_columns_verified', 0)} column(s) of "
                 f"{rec.get('n_rows_verified', 0)} whole row(s) against the refold and {rec.get('n_history_snapshots_verified', 0)} "
                 f"history snapshot(s) verified by re-read -> {verdict} (record {path})")
    return "\n".join(lines)


def compare_dumps(old: Sequence[Mapping[str, object]], new: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """The keys whose folded row differs between two dumps, every one of them (``keys``; a certification reads the whole mover set,
    never a sample), with the per-column counts."""
    ko = {d["key"]: d for d in old}
    kn = {d["key"]: d for d in new}
    diffs = []
    col_counts: Dict[str, int] = {}
    for key in sorted(set(ko) | set(kn)):
        a, b = (ko.get(key) or {}).get("state"), (kn.get(key) or {}).get("state")
        if a == b:
            continue
        cols = {}
        for c in sorted(set(a or {}) | set(b or {})):
            if (a or {}).get(c) != (b or {}).get(c):
                cols[c] = [(a or {}).get(c), (b or {}).get(c)]
                col_counts[c] = col_counts.get(c, 0) + 1
        diffs.append({"key": key, "columns": cols})
    return {"n_keys_old": len(ko), "n_keys_new": len(kn), "n_differ": len(diffs), "columns": col_counts, "keys": diffs}


def _tree_python(tree: str, python: str, module: str = "sisteer.registry.replica") -> Tuple[List[str], Dict[str, str]]:
    """The command and environment that run a tree's own code: ``PYTHONPATH=<tree>/src``, no bytecode written, and the bytecode cache
    REDIRECTED to a fresh directory (``PYTHONPYCACHEPREFIX``) so a ``.pyc`` sitting in the tree's ``__pycache__`` is never read in place
    of the tracked source."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(os.path.abspath(tree), "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = __import__("tempfile").mkdtemp(prefix="refold-pycache-")
    return [python, "-m", module], env


def _log_stderr(msg: str) -> None:
    print(f"[{R.utc_now()}] refold-compare: {msg}", file=sys.stderr, flush=True)


def stage_input(src: str, work_dir: str, log: Callable[[str], None] = _log_stderr) -> Dict[str, object]:
    """Copy the database at ``src`` to ``<work_dir>/registry.sqlite`` in one streaming pass that hashes the bytes as they are read,
    so the record names the INPUT's sha256 and byte size, and ``src`` is never opened for writing. Refuses (:class:`RefoldInputRefused`)
    a missing file, a ``-wal`` twin holding bytes (the file alone is not the database then: checkpoint it first), too little free
    space under ``work_dir`` (the size plus 5 %), and an input whose size changed during the copy."""
    src = os.path.abspath(src)
    if not os.path.isfile(src):
        raise RefoldInputRefused(f"no database at {src}")
    st0 = os.stat(src)
    wal = src + "-wal"
    wal_bytes = os.path.getsize(wal) if os.path.exists(wal) else 0
    if wal_bytes:
        raise RefoldInputRefused(f"{wal} holds {wal_bytes} bytes: the file alone is not the database -- checkpoint it before measuring it")
    os.makedirs(work_dir, exist_ok=True)
    free = shutil.disk_usage(work_dir).free
    need = st0.st_size + st0.st_size // 20
    if free < need:
        raise RefoldInputRefused(f"{work_dir}: {free} bytes free, the copy needs {need} (the input is {st0.st_size} bytes)")
    dst = os.path.join(work_dir, "registry.sqlite")
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(dst + suffix)
        except FileNotFoundError:
            pass
    log(f"copying {src} ({st0.st_size} bytes) -> {dst}")
    t0 = time.monotonic()
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while True:
            block = fi.read(16 << 20)
            if not block:
                break
            h.update(block)
            fo.write(block)
            n += len(block)
        fo.flush()
        os.fsync(fo.fileno())
    st1 = os.stat(src)
    if n != st0.st_size or st1.st_size != st0.st_size or st1.st_mtime_ns != st0.st_mtime_ns or os.path.getsize(dst) != n:
        raise RefoldInputRefused(f"{src} changed during the copy ({st0.st_size} -> {st1.st_size} bytes, {n} copied): not a fixed input")
    out = {"path": src, "bytes": n, "sha256": h.hexdigest(), "wal_bytes": wal_bytes, "copy": dst, "copy_s": round(time.monotonic() - t0, 1),
           "mtime_utc": R._fmt_utc(__import__("datetime").datetime.fromtimestamp(st0.st_mtime, tz=__import__("datetime").timezone.utc))}
    log(f"copied {n} bytes in {out['copy_s']} s, sha256 {out['sha256']}")
    return out


def tree_identity(tree: str) -> Dict[str, object]:
    """What a tree is: its path, its commit (``git rev-parse HEAD``, else the directory's name when that is a 40-hex sha, else
    null), the sha256 of its ``reducer.py`` and how many times that file mentions ``tie_rank`` (0 for a pre-C1 tree) -- recorded so
    a verdict names the code that produced it, never gated on."""
    tree = os.path.abspath(tree)
    sha = None
    toplevel = None
    try:
        r = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and _HEX40.match(r.stdout.strip()):
            sha = r.stdout.strip()
        r = subprocess.run(["git", "-C", tree, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            toplevel = os.path.realpath(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    if sha is None and _HEX40.match(os.path.basename(tree)):
        sha = os.path.basename(tree)
    status: Optional[List[str]] = None
    try:
        r = subprocess.run(["git", "-C", tree, "status", "--porcelain"], capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            status = [line for line in r.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    red = os.path.join(tree, "src", "sisteer", "registry", "reducer.py")
    text = ""
    if os.path.isfile(red):
        with open(red, "r", encoding="utf-8") as fh:
            text = fh.read()
    dm = os.path.join(tree, "src", "sisteer", "registry", "decode_match.py")
    dm_sha = None
    if os.path.isfile(dm):
        with open(dm, "rb") as fh:
            dm_sha = hashlib.sha256(fh.read()).hexdigest()
    return {"path": tree, "sha": sha, "git_head": sha if status is not None else None, "git_toplevel": toplevel,
            "is_toplevel": (toplevel == os.path.realpath(tree)) if toplevel else None, "clean": (status == []) if status is not None else None,
            "status_sample": (status or [])[:10], "reducer_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
            "reducer_tie_rank_mentions": text.count("tie_rank"), "decode_match_sha256": dm_sha}


def untracked_source(tree: str) -> List[str]:
    """Every ``*.py`` under ``<tree>/src`` (bytecode cache directories skipped) that is NOT a file git tracks at that path -- an ignored or
    untracked module ``PYTHONPATH=<tree>/src`` would import in place of, or beside, the committed code."""
    tree = os.path.abspath(tree)
    src = os.path.join(tree, "src")
    r = subprocess.run(["git", "-C", tree, "ls-files", "-z", "--", "src"], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RefoldInputRefused(f"{tree}: git ls-files failed: {r.stderr.strip()[-300:]}")
    tracked = {os.path.normpath(x) for x in r.stdout.split("\0") if x}
    out = []
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for name in fns:
            if name.endswith(".py"):
                rel = os.path.normpath(os.path.relpath(os.path.join(dp, name), tree))
                if rel not in tracked:
                    out.append(rel)
    return sorted(out)


def require_tree(tree: str, expect_sha: str) -> Dict[str, object]:
    """The tree a certification names, bound by commit: a git checkout whose own top level is ``tree`` (not a snapshot nested under one),
    whose HEAD is ``expect_sha``, whose working tree is clean (no modified tracked file, no untracked unignored file) and whose every
    ``*.py`` under ``src/`` is a tracked file (no ignored module beside the committed code), read by git right before the tree's code
    runs (a recorded identity gates nothing). A directory named after a sha but holding other code, a checkout at another commit, an
    edited export, or a snapshot under a checkout's ignored directory refuses (:class:`RefoldInputRefused`)."""
    if not _HEX40.match(expect_sha or ""):
        raise RefoldInputRefused(f"an expected tree sha must be 40 hex, got {expect_sha!r}")
    ident = tree_identity(tree)
    if ident["git_head"] is None:
        raise RefoldInputRefused(f"{tree}: not a git checkout -- a certification binds each tree by its commit (tree_export.sh exports one)")
    if not ident["is_toplevel"]:
        raise RefoldInputRefused(f"{tree}: not the checkout's own top level (git names {ident['git_toplevel']}) -- a snapshot nested under a checkout "
                                 "would borrow its HEAD and its clean status")
    if ident["git_head"] != expect_sha:
        raise RefoldInputRefused(f"{tree}: HEAD {ident['git_head']} is not the expected {expect_sha}")
    if not ident["clean"]:
        raise RefoldInputRefused(f"{tree}: the checkout is not clean ({len(ident['status_sample'])}+ path(s): " + "; ".join(ident["status_sample"][:3]) + ")")
    stray = untracked_source(tree)
    if stray:
        raise RefoldInputRefused(f"{tree}: {len(stray)} source file(s) under src/ that git does not track at {expect_sha} (ignored or untracked): "
                                 + ", ".join(stray[:3]))
    ident["bound_to"] = expect_sha
    ident["untracked_source"] = []
    return ident


def declared_superseded_class(conn, keys: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """The record facts behind the declared-superseded class, per ``run`` key, read from the copy's ``row_history`` -- the
    replay ledger of every run-scale event, the table the reducer folds from (``run_state_event`` holds the events as rows of the
    live state and says nothing about a same-stamp pair's fold): the key's ``run_state`` events grouped by ``written_utc``; the
    LATEST stamp and the states it carries; whether that latest stamp carries both ``judged`` and ``superseded`` (``pair_at_latest``
    -- the pair that decides the fold under either order; an earlier pair decides nothing once a later state folds); the number of
    stamps carrying the pair. ``in_class`` is ``pair_at_latest``."""
    out: Dict[str, Dict[str, object]] = {}
    want = set(DECLARED_SUPERSEDED_PAIR)
    for key in keys:
        rows = conn.execute("SELECT written_utc, json_extract(payload_json, '$.state') FROM row_history "
                            "WHERE table_name = 'run' AND natural_key_json = ? AND record_type = 'run_state'", (key,)).fetchall()
        by_stamp: Dict[str, Set[str]] = {}
        for utc, st in rows:
            by_stamp.setdefault(str(utc), set()).add(str(st))
        latest = max(by_stamp) if by_stamp else None
        pair_stamps = sorted(u for u, states in by_stamp.items() if want <= states)
        at_latest = sorted(by_stamp.get(latest, set())) if latest else []
        pair = latest is not None and latest in pair_stamps
        out[key] = {"latest_run_state_utc": latest, "states_at_latest": at_latest, "pair_at_latest": pair, "n_pair_stamps": len(pair_stamps),
                    "n_run_state_events": len(rows), "in_class": pair}
    return out


def mover_in_class(diff: Mapping[str, object], facts: Mapping[str, object]) -> bool:
    """A mover is in the declared-superseded class iff its folded row differs in EXACTLY ``lifecycle_state``, from ``judged`` under
    the old order to ``superseded`` under the new, AND its latest same-stamp ``run_state`` batch carries both states."""
    cols = diff.get("columns") or {}
    return (set(cols) == {"lifecycle_state"} and list(cols["lifecycle_state"]) == list(DECLARED_SUPERSEDED_PAIR) and bool(facts.get("pair_at_latest")))


def residual_same_stamp_class(conn) -> List[str]:
    """The run ids the pre-tie-break order LEFT in ``judged`` where the code's order gives ``superseded`` -- the residual of the C1
    same-stamp class on a database folded under both orders (the live database between 2026-10-04 and the restamp of its certified
    movers): every ``run`` row with ``lifecycle_state = 'judged'`` whose ``row_history`` holds a ``run_state`` event to ``superseded``
    at EXACTLY the stamp that set its lifecycle (``json_extract(field_versions_json, '$.lifecycle_state[0]')``, the ``written_utc`` half
    of the stamp). Under the code's order that event folds last (``reducer.tie_rank`` 1 at one ``written_utc``) and the row reads
    ``superseded``; only the sha draw of the old order reads ``judged``. Read from ``row_history`` -- the ledger the fold orders by its
    ``written_utc`` -- as :func:`declared_superseded_class` reads the class's facts, not from ``run_state_event.event_utc``, which is the
    producer's event time and need not equal the record's stamp. A ``superseded`` at a LATER stamp folds last under either order and a
    row that still read ``judged`` with one would be another defect, not this class; at an EARLIER stamp the later ``judged`` stands under
    either. Empty on a database every row of which was folded under the code's order (a rebuild, or the live one after its restamp
    slot); ``registry replica residual-class`` lists it read-only, sorted, for the operator and for a cross-check of the window's
    ``WINDOW_RESTAMP_KEYS``."""
    out: List[str] = []
    rows = conn.execute("SELECT run_id, json_extract(field_versions_json, '$.lifecycle_state[0]') FROM run WHERE lifecycle_state = 'judged' "
                        "ORDER BY run_id").fetchall()
    for rid, stamp in rows:
        if stamp is None:
            continue
        key = canonical_json({"run_id": str(rid)})
        hit = conn.execute("SELECT 1 FROM row_history WHERE table_name = 'run' AND natural_key_json = ? AND record_type = 'run_state' "
                           "AND written_utc = ? AND json_extract(payload_json, '$.state') = 'superseded' LIMIT 1", (key, str(stamp))).fetchone()
        if hit is not None:
            out.append(str(rid))
    return out


def _projection_moves(a: Mapping[str, object], b: Mapping[str, object]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Every ``run_id`` whose projected row differs between two full projections, with the columns that differ, and the run_ids present
    on one side only (``projection_diff`` samples examples; a verdict needs the whole set)."""
    cols = list(a.get("columns") or PROJECTION_COLUMNS)
    ka = {row[0]: row for row in (a.get("rows") or [])}
    kb = {row[0]: row for row in (b.get("rows") or [])}
    moved: Dict[str, List[str]] = {}
    for rid in sorted(set(ka) & set(kb)):
        d = [c for i, c in enumerate(cols) if ka[rid][i] != kb[rid][i]]
        if d:
            moved[rid] = d
    return moved, sorted(set(ka) ^ set(kb))


def _run_tree(argv: List[str], env: Mapping[str, str], what: str) -> None:
    r = subprocess.run(argv, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise ReplicaError(f"{what} failed (rc {r.returncode}): {r.stderr[-800:]}")


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def expected_keys_json(keys: Sequence[object]) -> List[str]:
    """A declared mover set as natural keys: a run_id string becomes ``{"run_id": …}``; a mapping is taken as the natural key."""
    out = []
    for k in keys:
        if isinstance(k, str):
            out.append(canonical_json({"run_id": k}))
        elif isinstance(k, Mapping):
            out.append(canonical_json(dict(k)))
        else:
            raise RefoldInputRefused(f"expected key {k!r} is neither a run_id nor a natural key")
    return sorted(set(out))


AS_BUILT_ORDERS: Tuple[str, ...] = ("old", "new")


def check_expectations(expect_head_hash: Optional[str], expect_new_hash: Optional[str], head: Optional[Mapping[str, object]],
                       expect_input_sha256: Optional[str] = None, expect_as_built: Optional[str] = None, expect_old_tree: Optional[str] = None,
                       expect_new_tree: Optional[str] = None, expect_keys: Optional[Sequence[object]] = None) -> None:
    """The arguments a verdict is read against, refused before any work when malformed or incomplete: a hash must be the full 64-hex
    sha256 (a prefix would be a loose comparison), a tree sha the full 40 hex, the as-built order one of ``old`` / ``new``; a head given
    beside the expected head hash must have that hash as its projection's; and a CERTIFICATION (``expect_head_hash`` declared) must
    name everything a PASS is read against -- the head, the input's independently computed sha256, the order the copy was built under,
    both trees' commits, the NEW-order hash and the exact mover set -- so no PASS can come from an undeclared database, tree, build or
    mover count."""
    for name, h in (("--expect-head-hash", expect_head_hash), ("--expect-new-hash", expect_new_hash), ("--expect-input-sha256", expect_input_sha256)):
        if h is not None and not _HEX64.match(h):
            raise RefoldInputRefused(f"{name} must be the full 64 hex sha256, got {h!r}")
    for name, h in (("--expect-old-tree", expect_old_tree), ("--expect-new-tree", expect_new_tree)):
        if h is not None and not _HEX40.match(h):
            raise RefoldInputRefused(f"{name} must be the full 40 hex commit sha, got {h!r}")
    if expect_as_built is not None and expect_as_built not in AS_BUILT_ORDERS:
        raise RefoldInputRefused(f"--expect-as-built must be one of {AS_BUILT_ORDERS}, got {expect_as_built!r}")
    if head is not None and expect_head_hash is not None and head["projection"]["sha256"] != expect_head_hash:
        raise RefoldInputRefused(f"the head's projection sha256 is {head['projection']['sha256']}, not the expected {expect_head_hash}: "
                                 "the head file and the expectation name two different folds")
    if expect_head_hash is not None:
        missing = [name for name, v in (("--head", head), ("--expect-input-sha256", expect_input_sha256), ("--expect-as-built", expect_as_built),
                                        ("--expect-old-tree", expect_old_tree), ("--expect-new-tree", expect_new_tree),
                                        ("--expect-new-hash", expect_new_hash), ("--expect-keys", expect_keys)) if v is None]
        if missing:
            raise RefoldInputRefused("a certification (--expect-head-hash) names its head, its input, the order that built the copy, both trees, the "
                                     "NEW-order hash and the exact mover set: missing " + ", ".join(missing))


def refold_compare(db_copy: str, old_tree: str, new_tree: str, out_dir: str, python: str = sys.executable, expect_head_hash: Optional[str] = None,
                   expect_new_hash: Optional[str] = None, expect_keys: Optional[Sequence[object]] = None, head: Optional[Mapping[str, object]] = None,
                   input_record: Optional[Mapping[str, object]] = None, log: Callable[[str], None] = _log_stderr, expect_input_sha256: Optional[str] = None,
                   expect_as_built: Optional[str] = None, expect_old_tree: Optional[str] = None, expect_new_tree: Optional[str] = None) -> Dict[str, object]:
    """On ``db_copy`` (a node-local copy, written to; :func:`stage_input` makes one and records the input's identity): dump every run
    key's fold under the OLD tree and under the NEW tree (each tree's own code, as a subprocess), compare the full folded rows (the
    MOVERS), then project the copy three ways -- as built; with the movers folded under the OLD order (the OLD tree's ``refold-apply``;
    projected by the NEW tree's code and, as a cross-check, by the OLD tree's); with the movers folded under the NEW order -- and diff
    the OLD and NEW projections row by row. Writes ``<out_dir>/{old.jsonl, new.jsonl, keys.json, projection_{as_built,old,
    old_by_old_tree,new}.json, refold_compare.json}`` and returns the record (``refold_compare/2``).

    The VERDICT (``verdict.pass``) is read only when ``expect_head_hash`` is declared (else ``None``: a measurement, not a
    certification), and is PASS iff every one of these holds: the OLD-order projection's sha256 equals ``expect_head_hash`` (the head
    the kept database was built against -- the bootstrap's equality, re-derived from THIS copy under the OLD order); every mover is in the
    declared-superseded class (:func:`mover_in_class`: exactly ``lifecycle_state`` moves, ``judged`` → ``superseded``, and the key's
    latest same-stamp ``run_state`` batch carries both -- :func:`declared_superseded_class` over the copy's ``row_history``); the
    projection moves only on movers and only in ``lifecycle_state`` and :data:`LIFECYCLE_DERIVED_COLUMNS`, with no row on one side only;
    each apply is followed by the post-fold tail on the rewritten rows (the NEW tree's ``restamp``; ``restamped`` per side) and the two
    trees carry one ``decode_match.py`` (``stamp_code_identical``);
    the three paper counts are equal under both orders; the copy as built equals one of the two folds (else it was built by neither
    tree); the OLD tree's own projection of the OLD-order fold hashes the same as the NEW tree's (the projection code did not drift);
    and, when declared, the NEW-order projection equals ``expect_new_hash`` and the mover set equals ``expect_keys`` exactly (the
    symmetric difference is recorded). Nothing is compared as a prefix, a subset or a superset.

    A certification is BOUND to what it names: ``expect_input_sha256`` must equal the staged input's streaming hash
    (``input_record``, from :func:`stage_input`), else refused -- a PASS is over the file the operator hashed, not whichever file was
    at the path; ``expect_as_built`` names the order the copy was built under and the verdict requires ``as_built_order`` to be it (the
    kept bootstrap rebuild is ``new``; a live-order copy offered instead reads ``old`` and FAILS); ``expect_old_tree`` / ``expect_new_tree`` are
    the trees' commits, enforced by :func:`require_tree` right before each tree's code runs; and ``refold-apply`` under either tree must
    rewrite EVERY mover (a key without a live row is refused, never skipped)."""
    check_expectations(expect_head_hash, expect_new_hash, head, expect_input_sha256, expect_as_built, expect_old_tree, expect_new_tree, expect_keys)
    if expect_input_sha256 is not None:
        got = (input_record or {}).get("sha256")
        if got != expect_input_sha256:
            raise RefoldInputRefused(f"the staged input's sha256 is {got}, not the expected {expect_input_sha256}: not the database this certification names")
    expected_keys = expected_keys_json(expect_keys) if expect_keys is not None else None
    trees = {"old": require_tree(old_tree, expect_old_tree) if expect_old_tree is not None else tree_identity(old_tree),
             "new": require_tree(new_tree, expect_new_tree) if expect_new_tree is not None else tree_identity(new_tree)}
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.monotonic()
    timings: Dict[str, float] = {}
    applied: Dict[str, int] = {}
    restamped: Dict[str, int] = {}
    log(f"old tree {trees['old']['sha']} ({trees['old']['path']}); new tree {trees['new']['sha']} ({trees['new']['path']})")
    dumps = {}
    for label, tree in (("old", old_tree), ("new", new_tree)):
        argv, env = _tree_python(tree, python)
        out = os.path.join(out_dir, f"{label}.jsonl")
        t0 = time.monotonic()
        _run_tree(argv + ["refold-dump", "--db", db_copy, "--out", out], env, f"refold-dump under the {label} tree {tree}")
        timings[f"dump_{label}_s"] = round(time.monotonic() - t0, 1)
        with open(out, "r", encoding="utf-8") as fh:
            dumps[label] = [json.loads(line) for line in fh if line.strip()]
        log(f"{label} dump: {len(dumps[label])} run keys in {timings[f'dump_{label}_s']} s")
    cmp = compare_dumps(dumps["old"], dumps["new"])
    movers = [d["key"] for d in cmp["keys"]]
    keys_path = os.path.join(out_dir, "keys.json")
    with open(keys_path, "w", encoding="utf-8") as fh:
        json.dump(movers, fh)
    log(f"{cmp['n_differ']} mover(s); columns {cmp['columns']}")

    new_argv, new_env = _tree_python(new_tree, python)
    old_argv, old_env = _tree_python(old_tree, python)

    def project(label: str, argv, env, tree_label: str) -> Dict[str, object]:
        path = os.path.join(out_dir, f"projection_{label}.json")
        t0 = time.monotonic()
        _run_tree(argv + ["projection", "--db", db_copy, "--out", path], env, f"projection ({label}) under the {tree_label} tree")
        timings[f"projection_{label}_s"] = round(time.monotonic() - t0, 1)
        proj = _load_json(path)
        log(f"projection {label}: {proj['sha256'][:12]} counts {proj['counts']} ({timings[f'projection_{label}_s']} s)")
        return proj

    def apply(argv, env, tree_label: str) -> None:
        t0 = time.monotonic()
        r = subprocess.run(argv + ["refold-apply", "--db", db_copy, "--keys", keys_path], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise ReplicaError(f"refold-apply under the {tree_label} tree failed (rc {r.returncode}): {r.stderr[-800:]}")
        m = re.search(r"^(\d+) run row\(s\) rewritten", r.stdout, re.M)
        n = int(m.group(1)) if m else None
        if n != len(movers):
            raise RefoldInputRefused(f"refold-apply under the {tree_label} tree rewrote {n} of {len(movers)} mover(s): a mover without a live row "
                                     f"(or an unreadable count) cannot be certified -- {r.stdout.strip()[-300:]!r}")
        applied[tree_label] = n
        # the post-fold tail on the rewritten rows, by the NEW tree's code (a tree exported before the tail existed rewrites without it;
        # the trees' decode_match.py must be one file for a certification, `stamp_code_identical`); idempotent when the apply ran it
        r2 = subprocess.run(new_argv + ["restamp", "--db", db_copy, "--keys", keys_path], env=new_env, capture_output=True, text=True)
        if r2.returncode != 0:
            raise ReplicaError(f"restamp after the {tree_label} apply failed (rc {r2.returncode}): {r2.stderr[-800:]}")
        m2 = re.search(r"^(\d+) run row\(s\) restamped", r2.stdout, re.M)
        if not m2:
            raise RefoldInputRefused(f"restamp after the {tree_label} apply printed no count -- {r2.stdout.strip()[-300:]!r}")
        restamped[tree_label] = int(m2.group(1))
        timings[f"apply_{tree_label}_s"] = round(time.monotonic() - t0, 1)

    as_built = project("as_built", new_argv, new_env, "new")
    apply(old_argv, old_env, "old")
    p_old = project("old", new_argv, new_env, "new")
    p_old_by_old = project("old_by_old_tree", old_argv, old_env, "old")
    apply(new_argv, new_env, "new")
    p_new = project("new", new_argv, new_env, "new")

    impact = projection_diff(p_old, p_new)
    moved, one_side = _projection_moves(p_old, p_new)
    with DB.reading(db_copy) as conn:
        facts = declared_superseded_class(conn, movers)
    in_class = {d["key"]: mover_in_class(d, facts[d["key"]]) for d in cmp["keys"]}
    mover_run_ids = sorted(str(json.loads(k).get("run_id")) for k in movers)
    allowed_cols = {"lifecycle_state"} | set(LIFECYCLE_DERIVED_COLUMNS)
    as_built_order = ("new" if as_built["sha256"] == p_new["sha256"] else "old" if as_built["sha256"] == p_old["sha256"] else None)
    components: Dict[str, Optional[bool]] = {
        "old_order_hash_equals_expected": (p_old["sha256"] == expect_head_hash) if expect_head_hash is not None else None,
        "movers_in_class": all(in_class.values()),
        "projection_moves_only_movers": (not one_side and set(moved) <= set(mover_run_ids)
                                         and all(set(cs) <= allowed_cols for cs in moved.values())),
        "counts_equal": p_old["counts"] == p_new["counts"],
        "as_built_order_known": as_built_order is not None,
        "old_tree_projection_agrees": p_old_by_old["sha256"] == p_old["sha256"] and p_old_by_old["counts"] == p_old["counts"],
    }
    if expect_new_hash is not None:
        components["new_order_hash_equals_expected"] = p_new["sha256"] == expect_new_hash
    if expect_as_built is not None:
        components["as_built_order_expected"] = as_built_order == expect_as_built
    if expect_input_sha256 is not None:
        components["input_sha256_expected"] = True          # refused above otherwise; recorded so the verdict shows the gate it passed
    if expect_old_tree is not None or expect_new_tree is not None:
        components["trees_bound"] = (expect_old_tree is None or trees["old"].get("bound_to") == expect_old_tree) and \
                                    (expect_new_tree is None or trees["new"].get("bound_to") == expect_new_tree)
        # the tail that re-derives the rewritten rows' stamps runs from the NEW tree after either apply: both trees must carry the one file
        components["stamp_code_identical"] = (trees["old"].get("decode_match_sha256") is not None
                                              and trees["old"].get("decode_match_sha256") == trees["new"].get("decode_match_sha256"))
    keys_cmp: Optional[Dict[str, object]] = None
    if expected_keys is not None:
        got = sorted(set(movers))
        sym = sorted(set(got) ^ set(expected_keys))
        keys_cmp = {"n_expected": len(expected_keys), "n_movers": len(got), "equal": got == expected_keys, "symmetric_difference": sym,
                    "only_expected": sorted(set(expected_keys) - set(got)), "only_movers": sorted(set(got) - set(expected_keys))}
        components["keys_equal_expected"] = bool(keys_cmp["equal"])
    if expect_head_hash is None:
        verdict_pass: Optional[bool] = None
        reason = "no --expect-head-hash declared: a measurement, not a certification"
    else:
        failing = sorted(k for k, ok in components.items() if not ok)
        verdict_pass = not failing
        reason = "PASS" if verdict_pass else "FAIL: " + ", ".join(failing)
    verdict = {"pass": verdict_pass, "reason": reason, "components": components, "failing": sorted(k for k, ok in components.items() if ok is False),
               "expected": {"head_hash": expect_head_hash, "new_hash": expect_new_hash, "n_keys": None if expected_keys is None else len(expected_keys),
                            "input_sha256": expect_input_sha256, "as_built": expect_as_built, "old_tree": expect_old_tree, "new_tree": expect_new_tree},
               "keys": keys_cmp, "movers_out_of_class": sorted(k for k, ok in in_class.items() if not ok)}
    head_rec = None
    if head is not None:
        head_rec = {"taken_utc": head["taken_utc"], "n_files": head["n_files"], "allowlist_sha256": head["allowlist_sha256"],
                    "projection_sha256": head["projection"]["sha256"], "projection_counts": head["projection"].get("counts"),
                    "head_sha256": head_sha256(head), "schema_version": head.get("schema_version")}
    timings["total_s"] = round(time.monotonic() - t_start, 1)
    rec = {"schema": REFOLD_SCHEMA, "utc": R.utc_now(), "db": os.path.abspath(db_copy), "input": dict(input_record) if input_record else None,
           "old_tree": trees["old"]["path"], "new_tree": trees["new"]["path"], "trees": trees, "head": head_rec,
           "rows": dict(cmp, **{"class": facts, "in_class": in_class, "run_ids": mover_run_ids}), "n_applied": len(movers), "applied": applied,
           "restamped": restamped,
           "projection": {"version": p_new.get("version"), "columns": list(p_new.get("columns") or []),
                          "as_built": {k: v for k, v in as_built.items() if k != "rows"}, "as_built_order": as_built_order,
                          "old": {k: v for k, v in p_old.items() if k != "rows"}, "old_by_old_tree": {k: v for k, v in p_old_by_old.items() if k != "rows"},
                          "new": {k: v for k, v in p_new.items() if k != "rows"},
                          "differs": p_old["sha256"] != p_new["sha256"], "diff": impact, "moved_run_ids": sorted(moved), "moved_columns": moved,
                          "moved_column_counts": {c: sum(1 for cs in moved.values() if c in cs) for c in sorted({c for cs in moved.values() for c in cs})},
                          "rows_on_one_side": one_side, "lifecycle_derived_columns": list(LIFECYCLE_DERIVED_COLUMNS),
                          # kept for readers of refold_compare/1: before = the OLD-order fold, after = the NEW-order fold
                          "before": {k: v for k, v in p_old.items() if k != "rows"}, "after": {k: v for k, v in p_new.items() if k != "rows"}},
           "verdict": verdict, "timings_s": timings}
    _write_json(os.path.join(out_dir, "refold_compare.json"), rec)
    log(f"verdict: {reason}")
    return rec


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


ACTIONS = ("init", "head", "build", "ingest", "check", "projection", "publish", "rollback", "migrate", "land", "verify-mirror", "status",
           "refold-compare", "refold-dump", "refold-apply", "restamp", "restamp-keys", "update", "residual-class")


def add_arguments(s) -> None:
    s.add_argument("action", choices=ACTIONS,
                   help="init: lay a replica root out over a log root; head: the quiescent replayable head of the registry database (one read "
                        "transaction at a cursor-free boundary, bounded wait); build: the bootstrap replay of exactly the head's allowlist into a "
                        "work database, checked against the same head; ingest: fold the head's increment into a database in place; check: "
                        "frame_projection/1 of a database against a head (exit 1 on any difference, printed per column); projection: the "
                        "projection of a database; publish: a database as a new generation by compare-and-swap on CURRENT; rollback: CURRENT to "
                        "its parent; migrate: the pending migrations on a copy of CURRENT, published; land: the landing job (copy, presence check, "
                        "ingest, check, pytest -m live_db, publish); verify-mirror: every allowlisted file present with the head's sha; status: the "
                        "generations; refold-compare / refold-dump / refold-apply / restamp: the C1 measurement (restamp: the post-fold decode-policy "
                        "stamp on the run rows a refold-apply rewrote); restamp-keys: the lifecycle restamp of named runs on a LIVE database inside "
                        "a landing window (WINDOW_RESTAMP_KEYS) -- without --apply the PLAN, every (run, column, old, new) it would change, refused by "
                        "name on an unknown key or a change outside the lifecycle class, written to --out; with --apply the recorded plan re-checked, "
                        "applied in one transaction and verified by re-reading the changed rows; residual-class: read-only, the run ids the "
                        "pre-tie-break order left in `judged` where this code's order gives `superseded` (a same-stamp `superseded` event at the "
                        "lifecycle stamp), one per line or --json -- the C1 class still standing on --db (default <root>/registry.sqlite), for a "
                        "cross-check of the window's restamp keys")
    s.add_argument("--replica-root", default=os.environ.get("REGISTRY_REPLICA_ROOT"), help="the replica root (default $REGISTRY_REPLICA_ROOT)")
    s.add_argument("--log-root", help="init: the record-log root (its records/ and rulings/)")
    s.add_argument("--db", help="a database file (head: the registry database to read, default <root>/registry.sqlite; update/migrate: a KEPT database to carry forward / migrate, copied node-local and never written; ingest/check/projection/publish/"
                                "refold-dump/refold-apply: the file; refold-compare: the database to measure -- copied into --work and hashed, never written)")
    s.add_argument("--head", help="a head file (.json or .json.gz) written by `replica head --out` (refold-compare: the head the measured database "
                                  "was built against -- its identity is recorded and its projection sha must be --expect-head-hash)")
    s.add_argument("--expect", help="check: the head file to compare against (alias of --head)")
    s.add_argument("--out", help="head/projection: the file to write; build/land/refold-compare/update/migrate --db: the output directory")
    s.add_argument("--wait-min", type=float, default=DEFAULT_WAIT_MIN, help="head: minutes to wait for a cursor-free boundary (default 20)")
    s.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S, help="head: seconds between polls (default 15)")
    s.add_argument("--no-rows", action="store_true", help="head/projection: omit the projected rows (the sha and counts alone)")
    s.add_argument("--work", help="build/migrate/land/refold-compare/update: the node-local work directory (refold-compare, update and migrate --db work on the copy of --db they make there)")
    s.add_argument("--workers", type=int, default=4, help="build/ingest/land: parse workers")
    s.add_argument("--tree", help="land: the exported tree whose live_db modules run; update / migrate --db: the tree whose code runs (bound by --expect-tree)")
    s.add_argument("--tree-sha", help="build/publish/migrate/land: the tree the code ran from (recorded as built_by_tree)")
    s.add_argument("--job-id", help="the Slurm job id (recorded)")
    s.add_argument("--branch", help="land: the landing branch (recorded)")
    s.add_argument("--pytest-python", help="land: the interpreter that runs pytest (the gate venv's)")
    s.add_argument("--expect-skipped", type=int, default=0, help="land: the number of skipped tests allowed (default 0)")
    s.add_argument("--accept-c1-class", help="land: the operator's declaration (free text naming the refold-compare certification) under which a check "
                                             "that differs from the head on the certified C1 same-stamp class ALONE passes and publishes (the "
                                             "cutover landing's allowance). Absent: only an equal check passes. Any other column differing refuses either way")
    s.add_argument("--hash", dest="hash_mode", choices=("all", "manifest"), default="all",
                   help="build/land/verify-mirror: hash every listed file (all, default) or trust the mirror's verified manifest for the files it names")
    s.add_argument("--threads", type=int, default=8, help="build/verify-mirror/land: hashing threads")
    s.add_argument("--mode", default="build", help="publish: the generation's mode (build|ingest|migrate)")
    s.add_argument("--parent", help="publish: the parent generation id, `none` for a bootstrap, `current` for whatever CURRENT names (default)")
    s.add_argument("--no-publish", action="store_true", help="build: check and record only, publish nothing")
    s.add_argument("--old-tree", help="refold-compare: the tree whose reducer order is the old one")
    s.add_argument("--new-tree", help="refold-compare: the tree whose reducer order is the new one")
    s.add_argument("--python", default=sys.executable, help="refold-compare / migrate --db: the interpreter the trees' code runs under")
    s.add_argument("--expect-head-hash", help="refold-compare: the full 64-hex frame_projection/1 sha256 the OLD-order fold of the copy must equal "
                                              "(the head the database was built against); declares a certification -- exit 1 unless the verdict is PASS")
    s.add_argument("--expect-new-hash", help="refold-compare: the full 64-hex sha256 the NEW-order fold of the copy must equal (optional)")
    s.add_argument("--expect-keys", help="refold-compare: a JSON list of run_ids (or natural keys) the mover set must equal exactly (optional)")
    s.add_argument("--expect-input-sha256", help="refold-compare / update / migrate --db: the full 64-hex sha256 of the --db file, computed independently; the staged copy's streaming "
                                                 "hash must equal it (required for a certification)")
    s.add_argument("--expect-as-built", choices=AS_BUILT_ORDERS, help="refold-compare: the reducer order the measured database was built under -- the copy "
                                                                       "as built must equal that fold (required for a certification; a database rebuilt under the new reducer order is `new`)")
    s.add_argument("--expect-old-tree", help="refold-compare: the 40-hex commit --old-tree must be checked out at, clean (required for a certification)")
    s.add_argument("--expect-new-tree", help="refold-compare: the 40-hex commit --new-tree must be checked out at, clean (required for a certification)")
    s.add_argument("--keys", help="refold-apply / restamp: a JSON list of natural keys to rewrite / re-stamp; restamp-keys: a JSON list of run ids or natural keys")
    s.add_argument("--apply", action="store_true", help="restamp-keys: apply the plan recorded under --out (restamp_plan.json), re-computed and required identical first")
    s.add_argument("--migrate-tree", help="migrate --db: the tree whose code applies the migrations (its `registry migrate --db` runs as a subprocess; the "
                                           "tree that carries the migration files the kept database needs)")
    s.add_argument("--expect-migrate-tree", help="migrate --db: the 40-hex commit --migrate-tree must be checked out at, clean (binds the migrating tree; required)")
    s.add_argument("--expect-tree", help="update / migrate --db: the 40-hex commit --tree (the tree whose code RUNS) must be checked out at, clean; this module "
                                         "must live under that tree's src/ (required)")
    s.add_argument("--expect-schema", type=int, help="migrate --db: the schema version the migrated copy must reach (the live schema the head was taken at; required)")
    s.add_argument("--expect-head-sha256", help="update / migrate --db: the full 64-hex head_sha256 of --head (its allowlist sha, projection sha and taken_utc "
                                                "together), computed independently; required for update and whenever migrate --db is given a head")
    s.add_argument("--site-aliases", help="migrate: the site-alias table (a TOML file with a [site_aliases] table mapping legacy site labels to the "
                                          "registry's). --db form: REQUIRED with --expect-site-aliases-sha256, handed to both of the migrating tree's "
                                          "subprocesses. CURRENT form: optional, and it must name the root's own site_aliases.toml (the form reads "
                                          "only that). build / land / update / ingest always fold under the root's site_aliases.toml")
    s.add_argument("--expect-site-aliases-sha256", help="migrate / update: the full 64-hex digest the site-alias table must have (site_aliases.digest); "
                                                        "any other refuses before the copy (migrate --db: required)")
    s.add_argument("--json", action="store_true", help="status / residual-class: JSON")


def _need(args, *names) -> None:
    for n in names:
        if getattr(args, n.replace("-", "_"), None) in (None, ""):
            raise ReplicaError(f"replica {args.action}: --{n} is required")


def _layout(args) -> Layout:
    _need(args, "replica-root")
    return Layout(os.path.abspath(args.replica_root))


def cmd_replica(args, root: Optional[str]) -> int:
    try:
        return _cmd_replica(args, root)
    except ReplicaError as exc:
        print(f"replica {args.action} REFUSED: {exc}", file=sys.stderr)
        return int(getattr(exc, "exit_code", 1))
    except DB.FoldOrderDrift as exc:
        # the database records another fold order (db.require_fold_order, run by every fold writer here): not a migration's to fix
        print(f"replica {args.action} REFUSED: {exc}", file=sys.stderr)
        return SchemaBehind.exit_code
    except DB.SchemaDrift as exc:
        print(f"replica {args.action} REFUSED: the database's schema is not the code's -- `replica migrate` first: {exc}", file=sys.stderr)
        return SchemaBehind.exit_code


def _cmd_replica(args, root: Optional[str]) -> int:
    a = args.action
    if a == "init":
        _need(args, "replica-root", "log-root")
        lay = init_layout(args.replica_root, args.log_root)
        print(json.dumps(status(lay), indent=1))
        return 0
    if a == "head":
        db = args.db or DB.db_path(root)
        head = take_head(db, wait_min=args.wait_min, poll_s=args.poll_s, log=lambda m: print(m, file=sys.stderr), with_rows=not args.no_rows)
        summary = {k: v for k, v in head.items() if k not in ("files", "projection")}
        summary["projection"] = {k: v for k, v in head["projection"].items() if k != "rows"}
        if args.out:
            write_head(head, args.out)
            summary["out"] = args.out
        print(json.dumps(summary, indent=1, sort_keys=True))
        return 0
    if a == "projection":
        _need(args, "db")
        with DB.reading(args.db) as conn:
            proj = frame_projection(conn, with_rows=not args.no_rows)
        if args.out:
            _write_json(args.out, proj)
        print(json.dumps({k: v for k, v in proj.items() if k != "rows"}, indent=1, sort_keys=True))
        return 0
    if a == "check":
        _need(args, "db")
        hp = args.expect or args.head
        if not hp:
            raise ReplicaError("replica check: --expect HEAD is required")
        chk = check(args.db, read_head(hp))
        if args.out:
            write_check_json(chk, args.out)
        print(format_check(chk))
        return 0 if chk["equal"] else 1
    if a == "verify-mirror":
        lay = _layout(args)
        _need(args, "head")
        v = verify_mirror(lay, read_head(args.head), args.hash_mode, args.threads)
        print(json.dumps(v, indent=1, sort_keys=True))
        return 0 if v["ok"] else MirrorBehind.exit_code
    if a == "status":
        lay = _layout(args)
        st = status(lay)
        if args.json:
            print(json.dumps(st, indent=1, sort_keys=True))
        else:
            print(f"replica {st['root']} (log {st['log_root']}): CURRENT = {st['current']}")
            sa = st["site_aliases"]
            print(f"  site aliases: {sa['path']}: " + (f"REFUSED ({sa['refused']})" if "refused" in sa else
                                                         f"{'present' if sa['present'] else 'absent (the empty table)'}, digest {sa['digest']}, {len(sa['pairs'])} pair(s)"))
            for g in st["generations"]:
                print(f"  {g['gen_id']}  parent={g['parent_gen_id']} mode={g['mode']} schema={g['schema_version']} files={g['n_files']} "
                      f"projection={g['projection_sha12']} published={g['published_utc']}")
            for u in st["unpublished"]:
                print(f"  unpublished: {u}")
        return 0
    if a == "build":
        lay = _layout(args)
        _need(args, "head", "work")
        head = read_head(args.head)
        os.makedirs(args.work, exist_ok=True)
        work_db = os.path.join(args.work, "registry.sqlite")
        out_dir = args.out or os.path.join(lay.builds, f"build-{head['allowlist_sha256'][:8]}-{(args.job_id or 'local')}")
        os.makedirs(out_dir, exist_ok=True)
        _, aliases = root_site_aliases(lay)                                     # a refused alias table refuses before the mirror hash
        print(f"[replica build] site aliases: {aliases['path']} ({'present' if aliases['present'] else 'absent: the empty table'}), digest "
              f"{aliases['digest']}, {len(aliases['pairs'])} pair(s)")
        mirror = require_mirror(lay, head, args.hash_mode, args.threads)      # every listed file present with the head's sha, BEFORE the replay
        _write_json(os.path.join(out_dir, "mirror.json"), mirror)
        print(f"[replica build] mirror holds every listed file ({mirror['hashed']} hashed, {mirror['from_manifest']} from the manifest, {mirror['hash_s']} s)")
        print(f"[replica build] replaying {head['n_files']} allowlisted files (head {head['taken_utc']}) into {work_db} with {args.workers} workers")
        res = build(lay, head, work_db, workers=args.workers, site_aliases_sha256=aliases["digest"])
        _write_json(os.path.join(out_dir, "build.json"), res["build"])
        write_check_json(res["check"], os.path.join(out_dir, "check.json"))
        b = res["build"]
        print(f"[replica build] wall {b['wall_s']} s, {b['records']} records ({b['records_per_s']} rec/s), peak rss {b['peak_rss_bytes']} bytes, "
              f"db {b['db_bytes']} bytes, unlisted ignored {b['unlisted_ignored']}")
        print(format_check(res["check"]))
        if not res["check"]["equal"]:
            print(f"replica build: UNEQUAL -- STOP; the work database stays at {work_db}, the record at {out_dir}", file=sys.stderr)
            return ReplicaAhead.exit_code
        if args.no_publish:
            return 0
        meta = publish(lay, work_db, head, mode="build", parent=lay.current(), built_by_tree=args.tree_sha, job_id=args.job_id,
                       extra={"site_aliases": aliases})
        _write_json(os.path.join(out_dir, "published.json"), meta)
        print(f"[replica build] published generation {meta['gen_id']} (parent {meta['parent_gen_id']}) -> CURRENT")
        return 0
    if a == "ingest":
        lay = _layout(args)
        _need(args, "head", "db")
        res = ingest_head(args.db, lay, read_head(args.head), workers=args.workers)
        print(json.dumps({k: v for k, v in res.items() if k != "report"}, indent=1, sort_keys=True))
        return 0
    if a == "publish":
        lay = _layout(args)
        _need(args, "head", "db")
        parent = args.parent or "current"
        meta = publish(lay, args.db, read_head(args.head), mode=args.mode, parent=None if parent == "none" else parent,
                       built_by_tree=args.tree_sha, job_id=args.job_id)
        print(json.dumps({k: v for k, v in meta.items() if k != "projection"}, indent=1, sort_keys=True))
        return 0
    if a == "rollback":
        lay = _layout(args)
        print(json.dumps(rollback(lay), indent=1))
        return 0
    if a == "update":
        lay = _layout(args)
        _need(args, "db", "head", "work", "out", "tree", "expect-tree", "expect-input-sha256", "expect-head-sha256")
        return update_db(args.db, lay, read_head(args.head), args.work, args.out, args.tree, args.expect_tree, args.expect_input_sha256, args.expect_head_sha256,
                         workers=args.workers, hash_mode=args.hash_mode, threads=args.threads, job_id=args.job_id,
                         expect_site_aliases_sha256=args.expect_site_aliases_sha256)
    if a == "migrate":
        if args.db:
            # the --db form: a KEPT database carried to the migrating tree's schema; no replica root, no CURRENT, no generation
            _need(args, "work", "out", "tree", "expect-tree", "migrate-tree", "expect-migrate-tree", "expect-schema", "expect-input-sha256")
            if args.head:
                _need(args, "expect-head-sha256")
            _need(args, "site-aliases", "expect-site-aliases-sha256")
            head = read_head(args.head) if args.head else None
            return migrate_db(args.db, args.work, args.out, args.tree, args.expect_tree, args.migrate_tree, args.expect_migrate_tree, args.expect_schema,
                              args.expect_input_sha256, python=args.python, head=head, expect_head_sha256=args.expect_head_sha256, job_id=args.job_id,
                              site_aliases_path=args.site_aliases, expect_site_aliases_sha256=args.expect_site_aliases_sha256)
        lay = _layout(args)
        _need(args, "work")
        print(json.dumps(migrate(lay, args.work, built_by_tree=args.tree_sha, job_id=args.job_id, site_aliases_path=args.site_aliases,
                                 expect_site_aliases_sha256=args.expect_site_aliases_sha256), indent=1))
        return 0
    if a == "land":
        lay = _layout(args)
        _need(args, "head", "tree", "work", "out", "pytest-python", "branch")
        return land(lay, read_head(args.head), args.tree, args.work, args.out, args.pytest_python, args.branch, expected_skipped=args.expect_skipped,
                    workers=args.workers, hash_mode=args.hash_mode, threads=args.threads, built_by_tree=args.tree_sha, job_id=args.job_id,
                    accept_c1_class=(args.accept_c1_class or None))
    if a == "refold-dump":
        _need(args, "db", "out")
        with DB.reading(args.db) as conn:
            rows = refold_dump(conn)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for d in rows:
                fh.write(canonical_json(d) + "\n")
        print(f"{len(rows)} run keys folded under this tree -> {args.out}")
        return 0
    if a == "refold-apply":
        _need(args, "db", "keys")
        with open(args.keys, "r", encoding="utf-8") as fh:
            keys = json.load(fh)
        conn = DB.connect(args.db)
        try:
            n = refold_apply(conn, keys)
        finally:
            conn.close()
        print(f"{n} run row(s) rewritten from their refold")
        print(f"{LAST_RESTAMPED} run row(s) restamped")
        return 0
    if a == "restamp-keys":
        _need(args, "db", "keys", "out")
        keys = _load_json(args.keys)
        if not isinstance(keys, list):
            raise RefoldInputRefused(f"--keys {args.keys}: a JSON list of run ids (or natural keys) is required")
        os.makedirs(args.out, exist_ok=True)
        plan_path = os.path.join(args.out, RESTAMP_PLAN_FILE)
        if not os.path.isfile(args.db):
            raise RefoldInputRefused(f"no database at {args.db}")
        conn = DB.connect(args.db)
        try:
            if not args.apply:
                plan = restamp_keys_plan(conn, keys)
                _write_json(plan_path, plan)
                print(format_restamp_plan(plan, plan_path))
                return 0 if plan["ok"] else RefoldInputRefused.exit_code
            if not os.path.isfile(plan_path):
                raise RefoldInputRefused(f"no plan at {plan_path}: run restamp-keys without --apply first (the apply is bound to a recorded plan)")
            # WINDOW-ONLY: the apply writes the live database and relies on the window's guarantees (timers stopped, writers drained, the
            # harvester at the door); the flag the window holds is the one thing this process can read to know it is inside one
            from . import harvest as H
            flag_root = os.path.abspath(args.root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT))
            if not os.path.isfile(H.maintenance_path(flag_root)):
                raise RefoldInputRefused(f"restamp-keys --apply: no maintenance flag at {H.maintenance_path(flag_root)}: the apply runs only inside a landing "
                                         "window (window.sh holds the flag with the timers stopped and the writers drained); nothing written")
            plan = _load_json(plan_path)
            if not isinstance(plan, dict) or plan.get("schema") != RESTAMP_SCHEMA:
                raise RefoldInputRefused(f"{plan_path} is not a {RESTAMP_SCHEMA} record")
            rec = restamp_keys_apply(conn, plan)
            apply_path = os.path.join(args.out, RESTAMP_APPLY_FILE)
            _write_json(apply_path, rec)
            print(format_restamp_apply(rec, apply_path))
            return 0 if rec["verified"] else RestampVerifyFailed.exit_code
        finally:
            conn.close()
    if a == "restamp":
        _need(args, "db", "keys")
        with open(args.keys, "r", encoding="utf-8") as fh:
            keys = json.load(fh)
        conn = DB.connect(args.db)
        try:
            n = restamp_runs(conn, keys)
        finally:
            conn.close()
        print(f"{n} run row(s) restamped")
        return 0
    if a == "refold-compare":
        _need(args, "db", "old-tree", "new-tree", "out", "work")
        head = read_head(args.head) if args.head else None
        # malformed or incomplete expectations, and unbound trees, refuse BEFORE the copy of a 40 GB input
        expect_keys = _load_json(args.expect_keys) if args.expect_keys else None
        if expect_keys is not None and not isinstance(expect_keys, list):
            raise RefoldInputRefused(f"--expect-keys {args.expect_keys}: a JSON list is required")
        check_expectations(args.expect_head_hash, args.expect_new_hash, head, args.expect_input_sha256, args.expect_as_built, args.expect_old_tree,
                           args.expect_new_tree, expect_keys)
        if args.expect_old_tree:
            require_tree(args.old_tree, args.expect_old_tree)
        if args.expect_new_tree:
            require_tree(args.new_tree, args.expect_new_tree)
        staged = stage_input(args.db, args.work)
        rec = refold_compare(staged["copy"], args.old_tree, args.new_tree, args.out, python=args.python, expect_head_hash=args.expect_head_hash,
                             expect_new_hash=args.expect_new_hash, expect_keys=expect_keys, head=head, input_record=staged,
                             expect_input_sha256=args.expect_input_sha256, expect_as_built=args.expect_as_built, expect_old_tree=args.expect_old_tree,
                             expect_new_tree=args.expect_new_tree)
        pj = rec["projection"]
        print(json.dumps({"n_differ": rec["rows"]["n_differ"], "columns": rec["rows"]["columns"], "projection_differs": pj["differs"],
                          "projection_old_sha12": pj["old"]["sha256"][:12], "projection_new_sha12": pj["new"]["sha256"][:12],
                          "as_built_order": pj["as_built_order"], "moved_column_counts": pj["moved_column_counts"],
                          "counts_old": pj["old"]["counts"], "counts_new": pj["new"]["counts"], "input_sha256": staged["sha256"], "input_bytes": staged["bytes"],
                          "verdict": rec["verdict"]["reason"], "out": args.out}, indent=1, sort_keys=True))
        v = rec["verdict"]["pass"]
        return 0 if v is None or v else 1
    if a == "residual-class":
        # read-only: the residual of the C1 same-stamp class on a database (the live one by default), for the operator and for the
        # cross-check of WINDOW_RESTAMP_KEYS before and after the window's restamp slot; never a write, never a stamp
        path = args.db or (DB.db_path(root) if root else None)
        if not path:
            raise ReplicaError("replica residual-class: --db (or --root) is required")
        if not os.path.isfile(path):
            raise RefoldInputRefused(f"no database at {path}")
        with DB.reading(path) as conn:
            ids = residual_same_stamp_class(conn)
            recorded = DB.fold_order_version(conn)
        from .reducer import FOLD_ORDER_VERSION
        if args.json:
            print(json.dumps({"db": os.path.abspath(path), "n": len(ids), "run_ids": ids, "fold_order_recorded": recorded,
                              "fold_order_code": FOLD_ORDER_VERSION}, indent=1, sort_keys=True))
        else:
            for rid in ids:
                print(rid)
            print(f"{len(ids)} run(s) in the residual same-stamp class (judged with a superseded event at the lifecycle stamp) on {path}; "
                  f"the database records fold order {recorded!r}, the code folds under {FOLD_ORDER_VERSION!r}", file=sys.stderr)
        return 0
    raise ReplicaError(f"unknown action {a}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m sisteer.registry.replica <action> …``: the same actions as ``registry replica``, for a tree's own code to run
    (refold-dump / refold-apply / projection under each tree of a refold-compare)."""
    import argparse
    p = argparse.ArgumentParser(prog="sisteer.registry.replica")
    p.add_argument("--root", help="registry root (head: default $REGISTRY_ROOT or ~/registry)")
    add_arguments(p)
    args = p.parse_args(argv)
    root = os.path.abspath(args.root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT))
    return cmd_replica(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
