"""The READER-STATE sidecar (registry-slurm-fast-windows Phase 3 step 2; spec S11): one SQLite file the grading host exports
from the live registry so a cluster shard's incremental compare is the grading host's own, without the database.

What it carries, for the named adapters (``registry reader-state export --adapter A ... --out FILE``, read-only on the live DB):

* ``source_revision``, ``source_watermark``, ``source_status`` -- the adapters' rows, every column.
* ``source_state`` -- the rows of the adapters' sources and of every input their watermarks name (``retire_vanished_group`` withdraws
  the observed state paths from it).
* ``observation`` -- the CURRENT rows of the adapters' sources (the withdrawal computation reads the head's keys).
* ``blocked_records`` -- every ``source_revision`` marker row (a held revision is a held revision whoever asks).
* ``item_result`` REDUCED to ``ITEM_MEMBERSHIP_KEY + source_revision_id`` for the adapters' accepted HEADS (what ``_current_item_keys``
  reads; the values stay on the grading host).
* the dimension tables in full (``model``, ``si_text``, ``eval``, ``item_set``, ``judge``, ``instrument``, ``object``, ``decode_policy``,
  ``paper_round``) -- ``_items_common.Dims`` and ``fit_sidecars.load_seeds`` read them.
* the anchor relation, for EVERY export (``reader_state/2``): the unsteered ``run`` rows (every
  column -- the key and the standup), a TABLE named ``v_run_current`` over EVERY run with the currency columns
  (:data:`RUN_CURRENT_COLUMNS`; the 1x anchors of ``full_dose`` and the solo twins are steered runs), the complete ``judgment`` rows of
  the unsteered runs and EVERY ``run_metric`` row of theirs that the paper frame counts (never only the metrics that already carry a
  contrast: the first contrast at a new metric would find no reading on the shard) -- a row of a judgment a re-judge
  displaced or a claim released (``anchor_current.CURRENT_JUDGMENT_SQL``) does not ship, nor does a row whose observation key a
  revision withdrew (current under no revision, ``anchor_current.withdrawn_metric_cte``, the predicate ``v_paper_long`` joins); a row of
  no judgment, or one no revision observed, does; both predicates are applied HERE, over the live ``judgment`` and ``observation``
  tables, because the sidecar's own ``judgment`` table is the complete judgments only and its ``observation`` table the adapters'
  current rows only -- so ``adapters.anchor_current.AnchorCurrency.load`` and ``em_misalign._paired_in_frame_anchors`` read the same
  readings on a shard as on the grading host (the latter reads ``v_run_current`` from it too).
* ``published_dirs`` -- the revision ids the grading host holds ON DISK under ``records/adapters/<a>/`` (a revision directory, an
  ``_activations`` bundle or a ``_refused`` record), published whether or not folded yet.
* ``reader_state_meta`` -- one row: schema, built_utc, the ledger head, the canonical home and repo spelling, the adapters, the live
  database's path / inode / size / schema version, the per-table row counts.

Every table is a plain column list (no CHECK, no foreign key, no trigger): the sidecar is a read-only image the runner's queries run
against, never a registry. ``adapters run --reader-state FILE`` opens it in place of the live database (:func:`open_reader_state`
checks the schema tag and that the requested adapters were exported) and puts the pass in CLUSTER mode (:class:`runner.ClusterRead`):
a revision id present in ``source_revision`` or ``published_dirs`` is known to the grading host and REFUSED BY NAME -- reactivation is the
grading host's -- so the cluster only ever publishes ids the grading host has never seen (S11 by construction).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import db as DB
from . import harvest as H
from . import records as R
from .adapters import anchor_current as AC
from .stores import REPO_ROOT

#: ``reader_state/2``: the anchor relation ships for every export; a ``reader_state/1`` sidecar
#: carries no ``v_run_current`` table and :func:`open_reader_state` refuses it by name -- a shard reading it would publish
#: contrast-less revisions and withdraw every contrast of its groups
SCHEMA = "reader_state/2"
FULL_TABLES: Tuple[str, ...] = ("source_revision", "source_watermark", "source_status")
DIMENSION_TABLES: Tuple[str, ...] = ("model", "si_text", "eval", "item_set", "judge", "instrument", "object", "decode_policy", "paper_round")
#: the columns of `v_run_current` the sidecar ships as a TABLE of the same name, over every run (anchor_current.CURRENCY_COLUMNS reads them)
RUN_CURRENT_COLUMNS: Tuple[str, ...] = ("run_id", "is_current", "why", "held_out", "in_current_paper", "selection_state", "roster_entry_ids_json")
#: the anchor relation's tables, every export (`run` holds the unsteered rows only; `v_run_current` every run)
ANCHOR_TABLES: Tuple[str, ...] = ("run", "v_run_current", "judgment", "run_metric")
REV16_RE = re.compile(r"^[0-9a-f]{16}$")          #: a revision directory's name (``records/adapters/<a>/<rev16>``)
#: the sidecar's one meta row, in column order (the DDL, the INSERT and :func:`meta` read this tuple)
META_COLUMNS: Tuple[str, ...] = ("schema", "built_utc", "ledger_head", "uri_home", "repo_root", "adapters_json", "db_path", "db_inode", "db_size",
                                 "schema_version", "counts_json")


class ReaderStateError(RuntimeError):
    """The sidecar is not one this code can read (schema, adapters), or the export cannot proceed."""


@dataclass
class ExportReport:
    path: str
    adapters: List[str]
    built_utc: str
    ledger_head: Optional[int]
    uri_home: str
    repo_root: str
    rows: Dict[str, int] = field(default_factory=dict)
    #: the `run_metric` rows of the unsteered runs that did NOT ship: their observation key is current under no revision
    #: (counted among the rows of current judgments) ...
    withdrawn_metric_rows: int = 0
    #: ... and the rows of a judgment a re-judge displaced or a claim released (anchor_current.CURRENT_JUDGMENT_SQL; a row failing both
    #: rules is counted here)
    non_current_judgment_metric_rows: int = 0
    bytes: int = 0
    elapsed_s: float = 0.0
    timings_s: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"path": self.path, "adapters": list(self.adapters), "built_utc": self.built_utc, "ledger_head": self.ledger_head,
                "uri_home": self.uri_home, "repo_root": self.repo_root, "rows": dict(self.rows), "withdrawn_metric_rows": self.withdrawn_metric_rows,
                "non_current_judgment_metric_rows": self.non_current_judgment_metric_rows, "bytes": self.bytes, "elapsed_s": round(self.elapsed_s, 3), "timings_s": {k: round(v, 3) for k, v in self.timings_s.items()}}


def published_dirs_of(root: str, adapters: Sequence[str]) -> List[Tuple[str, str, str]]:
    """``(adapter, kind, revision_id)`` for every revision id the grading host holds on disk under ``records/adapters/<a>/``: ``revision`` (a
    published directory), ``activation`` (an ``_activations/<rev>`` bundle root), ``refused`` (an ``_refused/<rev>`` record)."""
    out: List[Tuple[str, str, str]] = []
    for a in adapters:
        base = os.path.join(H.records_dir(root), "adapters", a)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if REV16_RE.match(name) and os.path.isdir(os.path.join(base, name)):
                out.append((a, "revision", name))
            elif name in ("_activations", "_refused") and os.path.isdir(os.path.join(base, name)):
                kind = "activation" if name == "_activations" else "refused"
                for rev in sorted(os.listdir(os.path.join(base, name))):
                    if REV16_RE.match(rev):
                        out.append((a, kind, rev))
    return out


def _columns(conn: sqlite3.Connection, schema: str, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA {schema}.table_info({table})").fetchall()
    if not rows:
        raise ReaderStateError(f"the live database has no table {table!r}")
    return [str(r[1]) for r in rows]


def _placeholders(names: Sequence[str]) -> str:
    return ",".join("?" for _ in names)


def export(root: str, adapters: Sequence[str], out: str, uri_home: Optional[str] = None, repo_root: Optional[str] = None,
           live_path: Optional[str] = None) -> ExportReport:
    """Build the sidecar at ``out`` (replaced atomically: written beside as ``.tmp``, then renamed) from the live registry under ``root``
    (read-only, ``mode=ro``). ``uri_home`` (default this process's home) and ``repo_root`` (default this checkout) are the CANONICAL
    spellings the cluster task maps its own onto -- recorded in the meta row, never inferred on the cluster."""
    t0 = time.monotonic()
    adapters = sorted(dict.fromkeys(str(a) for a in adapters))
    if not adapters:
        raise ReaderStateError("reader-state export: name at least one adapter (--adapter A ...)")
    live = os.path.abspath(live_path or DB.db_path(root))
    if not os.path.exists(live):
        raise ReaderStateError(f"reader-state export: no live database at {live}")
    uri_home = (uri_home or os.path.expanduser("~")).rstrip("/")
    repo_root = (repo_root or REPO_ROOT).rstrip("/")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = f"{out}.{os.getpid()}.tmp"
    for stale in (tmp, tmp + "-journal"):
        if os.path.exists(stale):
            os.remove(stale)
    built = R.utc_now()
    rep = ExportReport(out, adapters, built, None, uri_home, repo_root)
    conn = sqlite3.connect(f"file:{tmp}", uri=True, isolation_level=None)     # a URI connection: the ATTACH below reads the live copy ro
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("ATTACH DATABASE ? AS live", (f"file:{live}?mode=ro",))
        conn.execute(f"PRAGMA live.busy_timeout={DB.BUSY_TIMEOUT_MS}")
        conn.execute("BEGIN")
        marks = _placeholders(adapters)

        def copy(table: str, where: str, params: Sequence[object] = (), cols: Optional[Sequence[str]] = None, dest: Optional[str] = None,
                 select: Optional[str] = None) -> None:
            """`main.dest` (columns `cols`, default the live table's) filled from `live.table WHERE where` -- or from `select`, a complete
            SELECT (a leading WITH allowed) yielding `cols` in order, for a copy that joins."""
            t1 = time.monotonic()
            dest = dest or table
            names = list(cols or _columns(conn, "live", table))
            conn.execute(f"CREATE TABLE main.{dest} ({', '.join(names)})")
            src = select or (f"SELECT {', '.join(names)} FROM live.{table}" + (f" WHERE {where}" if where else ""))
            cur = conn.execute(f"INSERT INTO main.{dest} {src}", tuple(params))
            rep.rows[dest] = int(cur.rowcount)                # the rows the INSERT ... SELECT wrote; no second scan
            rep.timings_s[dest] = time.monotonic() - t1

        for t in FULL_TABLES:
            copy(t, f"adapter IN ({marks})", adapters)
        sources = f"SELECT source_id FROM live.source_revision WHERE adapter IN ({marks})"
        copy("observation", f"current = 1 AND source_id IN ({sources})", adapters)
        copy("source_state", f"source_id IN ({sources}) OR input_uri IN (SELECT input_uri FROM live.source_watermark WHERE adapter IN ({marks}))",
             list(adapters) + list(adapters))
        copy("blocked_records", "record_type = 'source_revision'")
        heads = (f"SELECT sr.revision_id FROM live.source_revision sr WHERE sr.adapter IN ({marks}) AND sr.accepted = 1 AND sr.observed_utc = "
                 f"(SELECT MAX(s2.observed_utc) FROM live.source_revision s2 WHERE s2.source_id = sr.source_id AND s2.accepted = 1)")
        copy("item_result", f"source_revision_id IN ({heads})", adapters, cols=list(R.ITEM_MEMBERSHIP_KEY) + ["source_revision_id"])
        for t in DIMENSION_TABLES:
            copy(t, "")
        # the anchor relation, for EVERY export: the shard's AnchorCurrency.load reads it exactly as the grading host's reads the
        # database. `v_run_current` is the one fold of the rulings (~54 s live), evaluated here once per export over every run.
        copy("run", "object_id = 'none'")
        copy("v_run_current", "", cols=list(RUN_CURRENT_COLUMNS))
        copy("judgment", "status = 'complete' AND run_id IN (SELECT run_id FROM main.run)")
        # every reading of the unsteered runs the paper frame COUNTS: a row of a judgment a re-judge displaced or a claim released does not ship,
        # nor does a row whose observation key is current under no revision (a revision withdrew the measurement while its run stayed
        # current) -- the sidecar's `judgment` table is the complete judgments only and its `observation` table the adapters' current rows
        # only, so both predicates are evaluated here over the live tables, qualified `live.` (main already holds both tables)
        rm_cols = _columns(conn, "live", "run_metric")
        rm_scope = "rm.run_id IN (SELECT run_id FROM main.run)"
        rm_from = f"FROM live.run_metric rm {AC.WITHDRAWN_METRIC_JOIN} {AC.current_judgment_join('live')} WHERE {rm_scope}"
        copy("run_metric", "", cols=rm_cols,
             select=(f"WITH {AC.withdrawn_metric_cte('live')} SELECT {', '.join('rm.' + c for c in rm_cols)} {rm_from} "
                     f"AND {AC.CURRENT_JUDGMENT_SQL} AND wm.run_id IS NULL"))
        dropped = conn.execute(f"WITH {AC.withdrawn_metric_cte('live')} SELECT COALESCE(SUM(NOT {AC.CURRENT_JUDGMENT_SQL}), 0), "
                               f"COALESCE(SUM({AC.CURRENT_JUDGMENT_SQL} AND wm.run_id IS NOT NULL), 0) {rm_from}").fetchone()
        rep.non_current_judgment_metric_rows, rep.withdrawn_metric_rows = int(dropped[0]), int(dropped[1])
        conn.execute("CREATE TABLE main.published_dirs (adapter, kind, revision_id)")
        dirs = published_dirs_of(root, adapters)
        conn.executemany("INSERT INTO main.published_dirs VALUES (?, ?, ?)", dirs)
        rep.rows["published_dirs"] = len(dirs)
        head = conn.execute("SELECT MAX(ingest_seq) FROM live.record_ingest").fetchone()[0]
        rep.ledger_head = int(head) if head is not None else None
        version = conn.execute("SELECT MAX(version) FROM live.schema_migration").fetchone()[0]
        st = os.stat(live)
        conn.execute(f"CREATE TABLE main.reader_state_meta ({', '.join(META_COLUMNS)})")
        conn.execute(f"INSERT INTO main.reader_state_meta VALUES ({_placeholders(META_COLUMNS)})",
                     (SCHEMA, built, rep.ledger_head, uri_home, repo_root, json.dumps(adapters), live, int(st.st_ino), int(st.st_size),
                      (int(version) if version is not None else None), json.dumps(rep.rows, sort_keys=True)))
        # the indexes the runner's and the adapters' queries lean on (the live database's own, for the columns shipped); revision_id is the
        # live table's primary key and the cluster shard's per-group lookup (`ClusterRead.known`, `_accepted_inputs`)
        for ddl in ("CREATE INDEX ix_sr_source ON source_revision (source_id, accepted, observed_utc)",
                    "CREATE INDEX ix_sr_revision ON source_revision (revision_id)",
                    "CREATE INDEX ix_sr_adapter ON source_revision (adapter)",
                    "CREATE INDEX ix_sw_adapter ON source_watermark (adapter)",
                    "CREATE INDEX ix_sw_source ON source_watermark (source_id, input_uri)",
                    "CREATE INDEX ix_ss_input ON source_state (input_uri)",
                    "CREATE INDEX ix_st_source ON source_status (source_id, source_revision_id)",
                    "CREATE INDEX ix_ob_source ON observation (source_id, revision_id, current)",
                    "CREATE INDEX ix_ir_revision ON item_result (source_revision_id)",
                    "CREATE INDEX ix_pd_revision ON published_dirs (revision_id)",
                    "CREATE INDEX ix_br_type ON blocked_records (record_type)"):
            conn.execute(ddl)
        for ddl in ("CREATE INDEX ix_run_anchor ON run (model_id, object_id, eval_id)",
                    "CREATE INDEX ix_run_key ON run (model_id, eval_id, instrument_id, token_cap_tokens, trial_index)",
                    "CREATE INDEX ix_rm_run ON run_metric (run_id, metric_id)",
                    "CREATE INDEX ix_jd_run ON judgment (run_id, judge_id)",
                    "CREATE INDEX ix_vrc_run ON v_run_current (run_id)"):
            conn.execute(ddl)
        conn.execute("COMMIT")
        conn.execute("DETACH DATABASE live")
    finally:
        conn.close()
    os.replace(tmp, out)
    rep.bytes = os.path.getsize(out)
    rep.elapsed_s = time.monotonic() - t0
    return rep


def meta(conn: sqlite3.Connection) -> Dict[str, object]:
    row = conn.execute(f"SELECT {', '.join(META_COLUMNS)} FROM reader_state_meta").fetchone()
    if row is None:
        raise ReaderStateError("the sidecar carries no reader_state_meta row")
    m: Dict[str, object] = dict(zip(META_COLUMNS, row))
    m["adapters"] = json.loads(str(m.pop("adapters_json") or "[]"))
    m["counts"] = json.loads(str(m.pop("counts_json") or "{}"))
    return m


def open_reader_state(path: str, adapters: Sequence[str] = ()) -> sqlite3.Connection:
    """The sidecar, read-only, after checking it IS one (the schema tag) and that every requested adapter was exported into it -- a
    sidecar built for other adapters carries none of these adapters' watermarks and would read every input as new."""
    if not os.path.isfile(path):
        raise ReaderStateError(f"no reader-state sidecar at {path}")
    conn = DB.connect(path, readonly=True)
    try:
        try:
            m = meta(conn)
        except sqlite3.Error as exc:
            raise ReaderStateError(f"{path} is not a reader-state sidecar ({exc})") from exc
        if m["schema"] != SCHEMA:
            raise ReaderStateError(f"{path}: schema {m['schema']!r} is not {SCHEMA!r} (a sidecar exported by other code; export it again "
                                   "from the grading host -- a shard never reads an older one)")
        missing = sorted(set(adapters) - set(m["adapters"]))
        if missing:
            raise ReaderStateError(f"{path} was exported for {m['adapters']}, not for {missing}")
    except Exception:
        conn.close()
        raise
    return conn


def known_dirs(conn: sqlite3.Connection) -> Set[str]:
    """The revision ids the sidecar lists as published on the grading host's disk."""
    return {str(r[0]) for r in conn.execute("SELECT revision_id FROM published_dirs").fetchall()}


def known_dir_kinds(conn: sqlite3.Connection) -> Dict[str, Set[str]]:
    """Every revision id the sidecar lists as published on the grading host's disk with the KINDS it is listed under (``revision`` /
    ``activation`` / ``refused``; :func:`published_dirs_of`) -- what ``runner.ClusterRead.from_sidecar`` reads: an id listed as a
    revision directory or an activation bundle with no ``source_revision`` row is on the grading host's log unfolded (a pending fold)."""
    out: Dict[str, Set[str]] = {}
    for rev, kind in conn.execute("SELECT revision_id, kind FROM published_dirs").fetchall():
        out.setdefault(str(rev), set()).add(str(kind))
    return out
