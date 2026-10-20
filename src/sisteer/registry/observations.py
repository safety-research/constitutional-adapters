"""Fold observations (``fold_observation``; ``registry observations summary | list``): facts the harvester OBSERVES about a record at
the fold without refusing it -- a detector over the fold's own output, never a fold rule (2026-10-04; owned by the registry).

The one kind today, ``serving_jobs_strict_subset``: a ``run`` or ``run_patch`` record whose ``extra.serving_jobs`` (a list, compared as
a SET of its members) names a STRICT subset of the set the run's folded ``extra_json.serving_jobs`` already held. Replace-at-key is the
universal fold rule for ``extra`` (``reducer.fold``: ``extra.update(_extra)`` in event order, latest statement wins per key), so a
job-start registration that restates the key on a same-identity rerun REPLACES the set -- the registry cannot know whether the smaller
set is a correction or a slip, so it folds the record as it always did and records what it saw: the folded set, the restated set, the
dropped members, the restating record and the record whose statement the folded set came from. The submitter's cumulative set is the
only belt; this table is how its owner learns it slipped. Equal sets in any order, duplicates, a superset, a first statement, and a
value that is not a list on either side are not observed.

Derivation (:func:`derive`) is a pure function of the run key's ORDERED events (the ``row_history`` events in ``(written_utc, sha)``
order, the fold's own order) and the fold's snapshots: event ``i`` is an observation iff its own ``_extra.serving_jobs`` is a list whose
member set is a strict subset of the list the accumulated extra held after event ``i - 1`` (``folded_state_json.extra_json``). The reducer
calls :func:`observe_key` after every refold of a ``run`` key (``Reducer.apply_run_scale``; the ``run`` table always refolds its whole
history, ``reducer._REFOLD_ALWAYS``) and REPLACES the key's rows from the fresh ``(events, folded)`` -- so a late-arriving earlier-stamped
record re-derives them exactly as it re-derives every later ``folded_state_json``. Rebuild equality therefore follows from
``row_history``'s: the same events in the same order give the same rows, and ``harvest.verify_equal`` compares the table with no column
excluded. The one clock-like fact, THE CYCLE that folded the restating record, is not stored: it is the arrival of the record's file
(``record_ingest.arrival_utc``, itself replayed from the ``file_arrival`` facts), joined at read time by :func:`list_observations` -- a
stored cycle stamp would need an exclusion from ``rebuild --verify``, and a stored ``ingest_seq`` the dense re-rank's remap. Cost: a key
none of whose events carries the key costs no statement (a dict lookup per event already in memory); a key that does costs one DELETE and
one INSERT per refold -- 162 of 12,699 live runs carried ``extra_json.serving_jobs`` on 2026-10-04, every value a list. The migration
that created the table derives the rows of the runs already folded once (:func:`backfill`, its Python post-step): a run's ``extra`` keys
are never removed by the fold, so the live rows whose ``extra_json`` carries the key are exactly the keys with an event stating it.

Reads (:func:`list_observations`, :func:`summary`) are read-only; the CLI resolves the root without the ``state/last_run`` stamp.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Mapping, Optional, Sequence

from . import schema_def as S
from .identity import canonical_json

TABLE = "fold_observation"
#: The `extra` key the one detector reads.
SERVING_JOBS_KEY = "serving_jobs"
#: The one observation kind (vocab.FOLD_OBSERVATION_KINDS).
KIND_SERVING_JOBS_STRICT_SUBSET = "serving_jobs_strict_subset"
#: The run-scale tables whose refold derives observations (the reducer consults it; `run` always refolds its whole history).
OBSERVED_TABLES = ("run",)
DEFAULT_LIMIT = 200

COLUMNS: Sequence[str] = tuple(S.TABLE_BY_NAME[TABLE].column_names)
_INSERT_SQL = f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES ({', '.join('?' for _ in COLUMNS)})"
#: The restating record's FILE (its `record_uri` is a file uri; a `<file>#<line>` form is cut at the `#`), for the arrival join.
_FILE_OF_URI = "CASE WHEN instr(o.record_uri, '#') > 0 THEN substr(o.record_uri, 1, instr(o.record_uri, '#') - 1) ELSE o.record_uri END"
_LIST_SQL = (f"SELECT {', '.join('o.' + c for c in COLUMNS)}, ri.arrival_utc AS arrival_utc, ri.ingest_seq AS file_ingest_seq "
             f"FROM {TABLE} o LEFT JOIN record_ingest ri ON ri.file_uri = {_FILE_OF_URI}")


def members(value: object) -> Optional[List[str]]:
    """The member set of a stated ``serving_jobs`` value as a sorted list of distinct members -- a string member as itself, any other
    member by its canonical JSON text -- or None when the value is not a list (nothing to compare as a set)."""
    if not isinstance(value, list):
        return None
    return sorted({m if isinstance(m, str) else canonical_json(m) for m in value})


def _event_extra(ev) -> Dict[str, object]:
    extra = ev.payload.get("_extra") if isinstance(ev.payload, Mapping) else None
    return extra if isinstance(extra, dict) else {}


def _snapshot_extra(state: Mapping[str, object]) -> Dict[str, object]:
    raw = state.get("extra_json")
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def key_carried(events: Sequence) -> bool:
    """Whether any event of the key states ``extra.serving_jobs`` (a dict lookup per event, no statement): False means the key can have
    no observation and no row to replace."""
    return any(SERVING_JOBS_KEY in _event_extra(e) for e in events)


def derive(run_id: str, events: Sequence, folded: Sequence) -> List[Dict[str, object]]:
    """The observations of one run key -- the pure function of its ordered ``events`` (``reducer.HistoryEvent``) and the fold's
    ``folded`` snapshots (``reducer.fold``'s output, one ``(state, field_versions)`` per event). Event ``i`` is observed iff its own
    ``_extra.serving_jobs`` is a list whose member set is a strict subset of the list the accumulated extra held after event ``i - 1``;
    ``folded_record_sha256`` is the latest earlier event that stated the key."""
    out: List[Dict[str, object]] = []
    folded_by: Optional[str] = None
    for i, ev in enumerate(events):
        extra = _event_extra(ev)
        if SERVING_JOBS_KEY not in extra:
            continue
        stated = members(extra[SERVING_JOBS_KEY])
        prior = members(_snapshot_extra(folded[i - 1][0]).get(SERVING_JOBS_KEY)) if i > 0 else None
        if stated is not None and prior is not None and folded_by is not None and set(stated) < set(prior):
            dropped = sorted(set(prior) - set(stated))
            out.append({"observation_kind": KIND_SERVING_JOBS_STRICT_SUBSET, "run_id": run_id, "record_sha256": ev.sha,
                        "record_type": ev.record_type, "record_uri": ev.record_uri, "written_utc": ev.written_utc,
                        "folded_record_sha256": folded_by, "folded_json": canonical_json(prior), "restated_json": canonical_json(stated),
                        "dropped_json": canonical_json(dropped), "n_dropped": len(dropped)})
        folded_by = ev.sha
    return out


def observe_key(conn: sqlite3.Connection, key: Mapping[str, object], events: Sequence, folded: Sequence) -> int:
    """Replace the key's rows from ``(events, folded)`` after a refold (module docstring); returns the rows written. A key none of
    whose events states the key touches the database not at all."""
    if not key_carried(events):
        return 0
    run_id = str(key["run_id"])
    rows = derive(run_id, events, folded)
    conn.execute(f"DELETE FROM {TABLE} WHERE run_id = ?", (run_id,))
    if rows:
        conn.executemany(_INSERT_SQL, [tuple(r[c] for c in COLUMNS) for r in rows])
    return len(rows)


def adopt_carrier(conn: sqlite3.Connection, key: Mapping[str, object], sha: str, record_uri: str) -> None:
    """The same record delivered under two file names: the observation's ``record_uri`` follows the event's canonical carrier
    (``reducer.Reducer._adopt_canonical_carrier``), so live equals replay on the uri too."""
    conn.execute(f"UPDATE {TABLE} SET record_uri = ? WHERE run_id = ? AND record_sha256 = ?", (record_uri, str(key["run_id"]), sha))


def backfill(conn: sqlite3.Connection) -> str:
    """The one-shot derivation for the run keys already folded when the table is created (the fold-observation migration's Python
    post-step): every live run whose ``extra_json`` carries the key -- exactly the keys with an event stating it, since the fold never
    removes an ``extra`` key of a run -- has its events read back (``Reducer._load_events``), folded (``reducer.fold``) and observed
    through the same :func:`observe_key` the live path uses. Idempotent (the rows are replaced). Returns the line ``registry migrate``
    prints."""
    from . import reducer as RED
    red = RED.Reducer(conn)
    ids = [str(r[0]) for r in conn.execute(
        f"SELECT run_id FROM run WHERE json_type(extra_json, '$.{SERVING_JOBS_KEY}') IS NOT NULL ORDER BY run_id").fetchall()]
    n_rows = n_runs = 0
    for rid in ids:
        key = {"run_id": rid}
        events = red._load_events("run", canonical_json(key))
        if not events:
            continue
        n = observe_key(conn, key, events, RED.fold("run", events))
        n_rows += n
        n_runs += int(n > 0)
    return (f"derived fold observations: {n_rows} observation(s) on {n_runs} run(s) "
            f"({len(ids)} run(s) carrying extra_json.{SERVING_JOBS_KEY} checked)")


# --------------------------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------------------------


def list_observations(conn: sqlite3.Connection, run_id: Optional[str] = None, kind: Optional[str] = None,
                      limit: Optional[int] = DEFAULT_LIMIT) -> List[Dict[str, object]]:
    """The observations, newest restatement first, each beside the arrival of its record's file (the cycle that folded it)."""
    where: List[str] = []
    params: List[object] = []
    if run_id:
        where.append("o.run_id = ?")
        params.append(run_id)
    if kind:
        where.append("o.observation_kind = ?")
        params.append(kind)
    sql = _LIST_SQL + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY o.written_utc DESC, o.run_id, o.record_sha256"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    out = []
    for r in conn.execute(sql, params).fetchall():
        d = dict(r)
        for c in ("folded_json", "restated_json", "dropped_json"):
            d[c[:-5]] = json.loads(d[c])
        out.append(d)
    return out


def summary(conn: sqlite3.Connection, top: int = 10) -> Dict[str, object]:
    """The counts: observations, runs, per kind, per restating record kind, the latest restatement and the cycle it folded in, the
    runs carrying the key at all, and the runs with the most observations."""
    one = lambda sql, *p: conn.execute(sql, p).fetchone()[0]  # noqa: E731
    by = lambda sql: {str(r[0]): int(r[1]) for r in conn.execute(sql).fetchall()}  # noqa: E731
    latest = conn.execute(f"SELECT o.written_utc, ri.arrival_utc FROM {TABLE} o LEFT JOIN record_ingest ri ON ri.file_uri = {_FILE_OF_URI} "
                          "ORDER BY o.written_utc DESC, o.run_id, o.record_sha256 LIMIT 1").fetchone()
    top_rows = conn.execute(f"SELECT run_id, COUNT(*) AS n, SUM(n_dropped) AS dropped, MAX(written_utc) AS latest_written_utc FROM {TABLE} "
                            "GROUP BY run_id ORDER BY n DESC, dropped DESC, run_id LIMIT ?", (int(top),)).fetchall()
    return {
        "observations": int(one(f"SELECT COUNT(*) FROM {TABLE}")),
        "runs": int(one(f"SELECT COUNT(DISTINCT run_id) FROM {TABLE}")),
        "runs_with_serving_jobs": int(one(f"SELECT COUNT(*) FROM run WHERE json_type(extra_json, '$.{SERVING_JOBS_KEY}') IS NOT NULL")),
        "by_kind": by(f"SELECT observation_kind, COUNT(*) FROM {TABLE} GROUP BY observation_kind ORDER BY observation_kind"),
        "by_record_type": by(f"SELECT record_type, COUNT(*) FROM {TABLE} GROUP BY record_type ORDER BY record_type"),
        "dropped_members": int(one(f"SELECT COALESCE(SUM(n_dropped), 0) FROM {TABLE}")),
        "latest_written_utc": latest[0] if latest else None,
        "latest_arrival_utc": latest[1] if latest else None,
        "top_runs": [{"run_id": str(r[0]), "observations": int(r[1]), "dropped_members": int(r[2]), "latest_written_utc": str(r[3])} for r in top_rows],
    }


# --------------------------------------------------------------------------------------------
# text rendering
# --------------------------------------------------------------------------------------------


def list_lines(rows: Sequence[Dict[str, object]]) -> List[str]:
    out = [f"{len(rows)} fold observation(s), newest restatement first"]
    for r in rows:
        out.append(f"{r['written_utc']}  {r['observation_kind']}  {r['record_type']}  {r['run_id']}  "
                   f"folded {len(r['folded'])} -> restated {len(r['restated'])}, dropped {r['n_dropped']}: {json.dumps(r['dropped'])}  "
                   f"{r['record_uri']}  [folded by {str(r['folded_record_sha256'])[:12]}; arrived {r.get('arrival_utc') or '-'}]")
    return out


def summary_lines(rep: Dict[str, object]) -> List[str]:
    out = [f"fold observations {rep['observations']} on {rep['runs']} run(s), of {rep['runs_with_serving_jobs']} run(s) carrying "
           f"extra_json.{SERVING_JOBS_KEY}; {rep['dropped_members']} dropped member(s) in all",
           f"latest restatement written {rep['latest_written_utc']}; its file arrived {rep['latest_arrival_utc']}"]
    out.append("by kind: " + (", ".join(f"{k}={v}" for k, v in rep["by_kind"].items()) or "none"))
    out.append("by restating record kind: " + (", ".join(f"{k}={v}" for k, v in rep["by_record_type"].items()) or "none"))
    out.append("runs with the most observations: " + (", ".join(f"{t['run_id']} x{t['observations']} (dropped {t['dropped_members']}, latest {t['latest_written_utc']})"
                                                               for t in rep["top_runs"]) or "none"))
    return out
