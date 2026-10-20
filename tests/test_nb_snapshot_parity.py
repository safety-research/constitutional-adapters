"""The parity belt: the registry notebook reads the SAME rows from the live registry and
from the committed paper-frame snapshot at the same head, over its WHOLE read set, with caching disabled and separate cache dirs
and the memoised readers cleared (a fresh import per source).

Three tests. (a) The read set is DERIVED from the notebook's SQL - every `*_SQL` constant in `registry_nb.py`, the load cell's
`PAPER_SQL` / `FRONTIER_SQL` / `POOL_SQL` / `RUNS_SQL` and the padding query in `make_registry_notebook.py`, the three direct openers
and `head()` - as the table and view identifiers after FROM / JOIN that the schema knows, and must lie inside the snapshot's table
list, so a new query cannot silently read a table the snapshot does not carry. (b) Every query name the notebook uses, plus
`head()`, `headline_metric_ids()`, `role_collapse_rows(head)` and `check_headline_scales(strict=False)`, answers identically both ways
(frames equal after sorting by every column). (c) NEGATIVE: a snapshot exported from a copy of the registry missing ONE
`run_metric` row (an unsteered run's headline row) at the SAME head fails parity on the queries that read it - and does so against a
live side whose cache is WARM, so a stale-but-equal cache cannot mask a hole in the snapshot.

Synthetic registry throughout (`sisteer.registry.synth`); the notebook's hard-coded paper model ids are substituted with the synthetic
model's; the snapshots come from the registry's own exporter (`test_nb_snapshot_fallback.export_snapshot`).
"""
from __future__ import annotations

import math
import os
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import schema_def as S
from sisteer.registry import snapshot as SN
from sisteer.registry import synth
from test_nb_snapshot_fallback import NB_FILES, NOTEBOOKS, export_snapshot, import_notebook, nb_env, notebook_mirror, synthetic_registry

QUERY_NAMES = ("paper_all", "qwen_frontier", "qwen_unsteered", "runs_all", "qwen_padding", "objects", "mt_degeneracy")
READERS = ("head", "headline_metric_ids", "role_collapse_rows", "check_headline_scales")


def notebook_read_set() -> set:
    """Every table / view identifier after FROM or JOIN in the two notebook files that the schema knows."""
    known = set(S.ALL_TABLES) | set(S.VIEW_NAMES)
    text = "\n".join((NOTEBOOKS / f).read_text(encoding="utf-8") for f in NB_FILES)
    found = {m.group(1).lower() for m in re.finditer(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.I)}
    return found & known


def notebook_queries(R) -> dict:
    """{query name: SQL} exactly as the notebook builds them, with `R`'s model constants as they stand (the caller substitutes)."""
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    block = src[src.index("ML = "):src.index("# ONE ROW PER MEASUREMENT AT LOAD")]
    ns = {"R": R}
    exec(block, ns)                                                              # ML, DECODE, PAPER_SQL, FRONTIER_SQL, POOL_SQL, RUNS_SQL
    m = re.search(r'R\.query\((f"SELECT \{DECODE\} FROM v_paper_long.*?),\s*"qwen_padding"\)', src, flags=re.S)
    assert m, "the padding query moved; update notebook_queries"
    padding = eval("(" + m.group(1) + ")", {"R": R, "DECODE": ns["DECODE"]})       # two f-string literals over two lines
    return {"paper_all": ns["PAPER_SQL"], "qwen_frontier": ns["FRONTIER_SQL"], "qwen_unsteered": ns["POOL_SQL"], "runs_all": ns["RUNS_SQL"],
            "qwen_padding": padding, "objects": R.OBJECTS_SQL, "mt_degeneracy": R.MT_DEGEN_SQL}


def _norm(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return round(float(v), 9)
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return None if (isinstance(v, np.floating) and np.isnan(v)) else round(float(v), 9) if isinstance(v, (np.integer, np.floating)) else str(v)
    except ImportError:
        pass
    return str(v)


def canon(df):
    """(columns, rows sorted by every column, nulls unified, numbers as floats) - what "the same frame" means across two sources."""
    rows = [tuple(_norm(v) for v in row) for row in df.itertuples(index=False, name=None)]
    return list(df.columns), sorted(rows, key=repr)


def belt(R) -> dict:
    out = {name: R.query(sql, name) for name, sql in notebook_queries(R).items()}
    h = R.head()
    out["head"] = h
    out["headline_metric_ids"] = R.headline_metric_ids()
    out["role_collapse_rows"] = R.role_collapse_rows(h)
    out["check_headline_scales"] = R.check_headline_scales(strict=False)
    return out


def run_source(source: str, tmp_path, nbdir, root, dest, monkeypatch, cache_dir: str, nb_cache: str | None = "0") -> dict:
    """A FRESH import of the notebook on `source`, its cache under `cache_dir`, the synthetic model in place of the paper's."""
    env = nb_env(tmp_path, REGISTRY_NB_SOURCE=source, REGISTRY_NB_SNAPSHOT_ROOT=dest, NB_CACHE=nb_cache,
                 REGISTRY_ROOT=root if source == "live" else None)
    R = import_notebook(nbdir, env, monkeypatch)
    assert R.SOURCE == source
    R.CACHE = cache_dir
    R.MODEL_ID, R.MODEL_ORDER = synth.MODEL_ID, [synth.MODEL_ID]
    return belt(R)


def assert_same(a: dict, b: dict, names=QUERY_NAMES, head=None):
    for name in names:
        assert canon(a[name]) == canon(b[name]), name
        assert a[name].attrs["head"] == b[name].attrs["head"] == (head if head is not None else a["head"]), name
    assert a["head"] == b["head"]
    assert a["headline_metric_ids"] == b["headline_metric_ids"]
    assert canon(a["role_collapse_rows"]) == canon(b["role_collapse_rows"])
    assert a["check_headline_scales"] == b["check_headline_scales"]


# ---- (a) the read set -----------------------------------------------------------------------------------------------------------
def test_notebook_read_set_lies_inside_the_snapshot_tables():
    read_set = notebook_read_set()
    assert {"run", "object", "eval", "judgment", "run_metric", "v_paper_long", "v_paper_current", "record_ingest"} <= read_set, read_set
    outside = read_set - set(SN.PAPER_FRAME_TABLES)
    assert not outside, f"the notebook reads {sorted(outside)}, which snapshot.PAPER_FRAME_TABLES does not carry (read_set_version {SN.PAPER_FRAME_READ_SET_VERSION})"


# ---- (b) the belt -----------------------------------------------------------------------------------------------------------------
def test_live_and_snapshot_answer_the_whole_read_set_identically(tmp_path, monkeypatch):
    root = synthetic_registry(tmp_path)
    dest = str(tmp_path / "paper_snapshot")
    _sid, head = export_snapshot(root, dest)
    nbdir = notebook_mirror(tmp_path)
    live = run_source("live", tmp_path, nbdir, root, dest, monkeypatch, str(tmp_path / "cache_live"))
    snap = run_source("snapshot", tmp_path, nbdir, root, dest, monkeypatch, str(tmp_path / "cache_snapshot"))
    assert_same(live, snap, head=head)
    # the belt has teeth: the synthetic registry populates these frames (the paper selection flags none of its runs, so paper_all is empty)
    assert len(live["runs_all"]) == 4 and len(live["qwen_unsteered"]) == 2 and len(live["objects"]) == 2 and len(live["mt_degeneracy"]) == 4
    assert len(live["paper_all"]) == 0
    assert not any(f.endswith(".parquet") for d in ("cache_live", "cache_snapshot") for f in os.listdir(tmp_path / d))     # NB_CACHE=0 wrote nothing


# ---- (c) the negative case ------------------------------------------------------------------------------------------------------
def _registry_missing_one_headline_row(root: str, into: str) -> str:
    """A copy of the registry at `root` with ONE run_metric row deleted (the first unsteered run's headline `asr` row); same head."""
    shutil.copytree(root, into)
    con = sqlite3.connect(DB.db_path(into))
    con.execute("PRAGMA foreign_keys = OFF")
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'").fetchall():
        con.execute(f"DROP TRIGGER {name}")
    rid = con.execute("SELECT r.run_id FROM run r WHERE r.object_id = 'none' ORDER BY r.run_id LIMIT 1").fetchone()[0]
    n = con.execute("DELETE FROM run_metric WHERE run_id = ? AND metric_id = 'asr'", (rid,)).rowcount
    con.commit()
    con.close()
    assert n == 1
    return rid


def test_parity_fails_on_a_snapshot_missing_one_row_despite_a_warm_live_cache(tmp_path, monkeypatch):
    root = synthetic_registry(tmp_path)
    dest = str(tmp_path / "paper_snapshot")
    _sid, head = export_snapshot(root, dest)
    nbdir = notebook_mirror(tmp_path)
    cache_live = str(tmp_path / "cache_live")
    warm = run_source("live", tmp_path, nbdir, root, dest, monkeypatch, cache_live, nb_cache=None)      # caching ON: populates the cache
    assert any(f.endswith(".parquet") and f"-live-{head}-" in f for f in os.listdir(cache_live))
    live = run_source("live", tmp_path, nbdir, root, dest, monkeypatch, cache_live, nb_cache=None)      # served from the warm cache
    assert_same(warm, live, head=head)
    # the tampered snapshot: one unsteered headline row gone, the head unchanged
    root2 = str(tmp_path / "reg_missing_row")
    rid = _registry_missing_one_headline_row(root, root2)
    dest2 = str(tmp_path / "paper_snapshot_missing_row")
    _sid2, head2 = export_snapshot(root2, dest2)
    assert head2 == head
    snap2 = run_source("snapshot", tmp_path, nbdir, root, dest2, monkeypatch, str(tmp_path / "cache_snapshot2"))
    assert snap2["head"] == head
    broken = [n for n in QUERY_NAMES if canon(live[n]) != canon(snap2[n])]
    assert set(broken) == {"qwen_unsteered", "mt_degeneracy"}, broken   # every query that reads that row, none other
    assert len(snap2["qwen_unsteered"]) == len(live["qwen_unsteered"]) - 1 and rid in set(live["qwen_unsteered"].run_id)
    with pytest.raises(AssertionError):
        assert_same(live, snap2, head=head)
    # the untampered snapshot still agrees with the same warm cache
    snap = run_source("snapshot", tmp_path, nbdir, root, dest, monkeypatch, str(tmp_path / "cache_snapshot"))
    assert_same(live, snap, head=head)
