"""The results registry (``sisteer.registry``): one queryable source of truth for every measurement.

Public API::

    from sisteer import registry
    conn = registry.connect()                       # read-only connection to ~/registry/registry.sqlite
    df = registry.frame("v_paper_long", round="r2")  # a view or table as a DataFrame
    items = registry.items(run_id, judge_id=None)    # per-item headline scores (one column per judge when several)
    pairs = registry.paired(run_id, anchor_run_id, metric)
    with registry.snapshot_open(root) as snap: ...   # cluster-side reader over a published snapshot
    registry.emit.RecordWriter(...)                  # producer side (stdlib only)

Everything heavy is imported lazily so ``python3 -S -c "import sisteer.registry.emit"`` stays
stdlib-only on a compute node.
"""
from __future__ import annotations

import os
from typing import Optional

__all__ = ["connect", "frame", "items", "paired", "snapshot_open", "emit", "identity", "records", "vocab"]


def _root(root: Optional[str] = None) -> str:
    from . import db as DB
    return root or os.environ.get("REGISTRY_ROOT", DB.DEFAULT_ROOT)


def connect(root: Optional[str] = None, readonly: bool = True):
    """A connection to the registry database (read-only by default; the harvester is the only writer)."""
    from . import db as DB
    return DB.connect(DB.db_path(_root(root)), readonly=readonly)


def frame(name: str, round: Optional[str] = None, where: Optional[str] = None, root: Optional[str] = None):
    """A view or table as a pandas DataFrame; ``round`` filters views carrying ``round_id``."""
    import pandas as pd
    conn = connect(root)
    try:
        sql = f"SELECT * FROM {name}"
        conds = []
        params = []
        if round is not None:
            conds.append("round_id = ?")
            params.append(round)
        if where:
            conds.append(f"({where})")
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def items(run_id: str, judge_id: Optional[str] = None, root: Optional[str] = None):
    """Per-item headline scores of a run; with several judges, one ``score__<judge_id[:8]>`` column
    per judge (the dynamic pivot ``v_judge_side_by_side`` cannot do in SQL)."""
    import pandas as pd
    conn = connect(root)
    try:
        sql = ("SELECT s.run_id, s.item_id, s.trial_index, s.sample_index, s.turn_index, s.metric_id, s.judge_id, s.judgment_id, "
               "s.score, s.label FROM v_judge_side_by_side s WHERE s.run_id = ?")
        params = [run_id]
        if judge_id:
            sql += " AND s.judge_id = ?"
            params.append(judge_id)
        df = pd.read_sql_query(sql, conn, params=params)
        if df.empty:
            return pd.read_sql_query("SELECT * FROM v_item_headline WHERE run_id = ?", conn, params=[run_id])
        idx = ["run_id", "item_id", "trial_index", "sample_index", "turn_index", "metric_id"]
        judges = sorted(df["judge_id"].unique())
        if len(judges) == 1:
            return df
        wide = df.pivot_table(index=idx, columns="judge_id", values="score", aggfunc="first")
        wide.columns = [f"score__{j[:8]}" for j in wide.columns]
        return wide.reset_index()
    finally:
        conn.close()


def paired(run_id: str, anchor_run_id: str, metric: str, judge_id: Optional[str] = None, judgment_id: Optional[str] = None,
           anchor_judgment_id: Optional[str] = None, root: Optional[str] = None):
    """Item-aligned (run, anchor) pairs for ``metric`` — the input to a paired contrast. ``judge_id`` constrains BOTH
    sides to that judge's judgments; each side must resolve to exactly ONE judgment (a run judged twice by the same
    judge is ambiguous and refused — pass ``judgment_id`` / ``anchor_judgment_id`` to name the pair)."""
    import pandas as pd
    conn = connect(root)
    try:
        def resolve(rid: str, explicit: Optional[str]) -> Optional[str]:
            if explicit is not None:
                return explicit
            sql = "SELECT DISTINCT judgment_id FROM v_item_result_current WHERE run_id = ? AND metric_id = ?"
            params: list = [rid, metric]
            if judge_id:
                sql += " AND judgment_id IN (SELECT judgment_id FROM judgment WHERE judge_id = ?)"
                params.append(judge_id)
            ids = sorted(str(r[0]) for r in conn.execute(sql, params).fetchall())
            if len(ids) > 1:
                raise ValueError(f"paired: run {rid} has {len(ids)} judgments for metric {metric!r}"
                                 + (f" by judge {judge_id}" if judge_id else "") + f": {ids}; name judgment_id / anchor_judgment_id")
            return ids[0] if ids else None
        a_j = resolve(run_id, judgment_id)
        b_j = resolve(anchor_run_id, anchor_judgment_id)
        cols = ["item_id", "trial_index", "sample_index", "turn_index", "value", "anchor_value", "judgment_id", "anchor_judgment_id"]
        if a_j is None or b_j is None:
            return pd.DataFrame(columns=cols)
        # two visible provenances that DISAGREE on one measurement are refused by name, never chosen between
        conflicts = conn.execute(
            "SELECT run_id, item_id FROM v_item_conflict WHERE metric_id = ? AND ((run_id = ? AND judgment_id = ?) OR (run_id = ? AND judgment_id = ?)) "
            "ORDER BY run_id, item_id LIMIT 20", (metric, run_id, a_j, anchor_run_id, b_j)).fetchall()
        if conflicts:
            named = ", ".join(f"{r[0]}/{r[1]}" for r in conflicts)
            raise ValueError(f"paired: conflicting visible provenances for metric {metric!r} on items {named} (v_item_conflict); "
                             "resolve the sources' disagreement before pairing")
        sql = ("SELECT a.item_id, a.trial_index, a.sample_index, a.turn_index, a.value AS value, b.value AS anchor_value, "
               "a.judgment_id, b.judgment_id AS anchor_judgment_id FROM v_item_result_current a JOIN v_item_result_current b "
               "ON b.item_id = a.item_id AND b.trial_index = a.trial_index AND b.sample_index = a.sample_index AND b.turn_index = a.turn_index "
               "AND b.metric_id = a.metric_id WHERE a.run_id = ? AND b.run_id = ? AND a.metric_id = ? "
               "AND a.judgment_id = ? AND b.judgment_id = ?")
        return pd.read_sql_query(sql, conn, params=[run_id, anchor_run_id, metric, a_j, b_j])
    finally:
        conn.close()


def snapshot_open(root: str, **kw):
    from .snapshot import snapshot_open as _open
    return _open(root, **kw)


def __getattr__(name: str):
    if name in ("emit", "identity", "records", "vocab", "db", "harvest", "reducer", "validate", "derive", "snapshot", "status",
                "paper", "pending", "publish", "mirrors", "backfill", "decode_policy", "schema_def", "synth", "cli"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
