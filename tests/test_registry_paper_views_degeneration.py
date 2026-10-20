"""Migration 0010: the paper views fill a NULL `run_metric.frac_*` mirror from the run's MEASURED `run_degeneration` rows
(`schema_def.DEGENERATION_RATE_CTE`), so degeneration filed by a reader OTHER than the one that emitted the metric row reaches
`degenerate_share`: a mirrored value still wins where the writer filed one; the metric row's own judgment beats the judgment-less
'' rows the union reader files; each mirror reads its own measure at its own scope (the bracket at `defended_only`, the first-512
loop at `first_512_tokens`, the legacy at-cap under either of its two spellings); a placeholder never fills anything.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import paper
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V

RATES = {"union5": 0.25, "classifier": 0.05, "rep_ratio_loop": 0.10, "zlib_collapse": 0.15, "contentless": 0.20,
         "at_token_cap_finish_reason": 0.30, "loop_first512": 0.35, "defended_degenerate": 0.40, "loop_or_at_cap": 0.45}
N = 200
JUDGED_UNION5 = 0.50
PINNED_COLUMNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "registry", "paper_load_columns_f33dca5d.json")      # the row filed under the metric row's own judgment: beats the '' row


def _degeneration(rid, jid, measure, rate, i, scope="all", state="measured", unit="draws"):
    k = int(round(rate * N))
    p = {"run_id": rid, "judgment_id": jid, "measure": measure, "scope": scope, "measurement_state": state,
         "numerator": k if state == "measured" else None, "denominator": N if state == "measured" else None,
         "denominator_unit": unit if state == "measured" else None, "pooling_json": {"unit": unit},
         "aggregation": "union" if measure in ("union5", "defended_degenerate", "loop_or_at_cap") else "single_detector",
         "rate": (k / N) if state == "measured" else None, "definition_version": "synthetic_second_reader/1", "threshold_params_json": {},
         "computed_by": "producer", "computed_utc": synth.t(i), "source_uri": "synthetic://second-reader", "notes": None}
    return synth.rec("run_degeneration", p, synth.t(i))


def _corpus(tmp_path):
    """Two judged runs (synth: one metric row `asr`@0.5 each, every frac_* mirror NULL). Run 0 gets degeneration rows from a SECOND
    reader: every mirror's measure under '' (the union reader's spelling), `union5` ALSO under the row's own judgment at another
    value, a `not_measured` classifier placeholder under the judgment (never a fill), and an `at_token_cap` row under the judgment
    beside the '' finish-reason row. Run 1 gets a mirror on its metric row AND a different '' union5 row: the mirror wins."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    r0, r1 = ids["run_ids"]
    j0, j1 = ids["judgment_ids"]
    degs = [_degeneration(r0, "", m, rate, 500 + i, scope={"loop_first512": "first_512_tokens", "defended_degenerate": "defended_only"}.get(m, "all"))
            for i, (m, rate) in enumerate(sorted(RATES.items()))]
    degs.append(_degeneration(r0, j0, "union5", JUDGED_UNION5, 520))
    degs.append(_degeneration(r0, j0, "classifier", 0.99, 521, state="not_measured"))
    degs.append(_degeneration(r0, j0, "at_token_cap", 0.60, 522))
    synth.write_run_dir(root, synth.CAMPAIGN, r0, [("run_degeneration", degs)])
    mirrored = synth.rec("run_metric", {"run_id": r1, "judgment_id": j1, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": 1, "n": 4,
                                        "computed_by": "producer", "frac_degenerate_union5": 0.75}, synth.t(600))
    synth.write_run_dir(root, synth.CAMPAIGN, r1, [("run_metric", [mirrored]), ("run_degeneration", [_degeneration(r1, "", "union5", 0.25, 601)])])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refusals, rep.blocked)
    return root, (r0, r1), (j0, j1)


def _long(conn, rid):
    rows = [dict(r) for r in conn.execute("SELECT * FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr'", (rid,))]
    assert len(rows) == 1, rows
    return rows[0]


def test_a_second_readers_degeneration_fills_the_null_mirrors_and_a_mirrored_value_wins(tmp_path):
    root, (r0, r1), (j0, j1) = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert M.db_version(conn) == M.code_version() and DB.schema_drift(conn) == []
        row = _long(conn, r0)
        # the metric row's own judgment beats the judgment-less row for the same measure
        assert row["frac_degenerate_union5"] == pytest.approx(JUDGED_UNION5)
        # each mirror reads its own measure at its own scope, from the '' rows where the judgment has none
        assert row["frac_degenerate_rep_ratio_loop"] == pytest.approx(RATES["rep_ratio_loop"])
        assert row["frac_degenerate_zlib_collapse"] == pytest.approx(RATES["zlib_collapse"])
        assert row["frac_degenerate_contentless"] == pytest.approx(RATES["contentless"])
        assert row["frac_loop_first512"] == pytest.approx(RATES["loop_first512"])
        assert row["frac_defended_degenerate"] == pytest.approx(RATES["defended_degenerate"])
        # the at-cap mirror: the legacy `at_token_cap` spelling first (under the judgment), the finish-reason row behind it
        assert row["frac_at_token_cap"] == pytest.approx(0.60)
        # a placeholder under the judgment never fills: the '' MEASURED classifier row does
        assert row["frac_degenerate_classifier"] == pytest.approx(RATES["classifier"])
        # the writer's mirror wins over a differing derived rate
        assert _long(conn, r1)["frac_degenerate_union5"] == pytest.approx(0.75)
        # the untouched columns of the row are what the writer filed
        assert row["value"] == 0.25 and row["metric_id"] == "asr" and row["judgment_id"] == j0
        # the frame (the paper's pinned columns): `degenerate_share` is non-NULL on both runs. Two precedences meet here and agree
        # whenever a mirror is what it claims to be (the writer's copy of the same reading): the SQL fills a NULL mirror and lets a
        # filed mirror win; the frame's own chain (`paper.LONG_COLUMN_SOURCES`, pre-0010) reads the run's measured `union5` FIRST on a
        # HEADLINE row, then the mirror -- so run 1's headline row reads the derived 0.25 over its (deliberately different) mirror 0.75
        with open(PINNED_COLUMNS, encoding="utf-8") as fh:
            pinned = json.load(fh)["columns"]
        df = paper.paper_long_frame(conn, columns=pinned, current_only=False)
        by = {r["run_id"]: r for _, r in df.iterrows()}
        assert by[r0]["degenerate_share"] == pytest.approx(JUDGED_UNION5)
        assert by[r1]["degenerate_share"] == pytest.approx(0.25)
        assert by[r0]["loop_share"] == pytest.approx(RATES["rep_ratio_loop"])      # the published rule, read through the fill of its mirror
    finally:
        conn.close()


def test_the_same_corpus_reads_null_before_0010_and_filled_after_it(tmp_path):
    """The same rows on a schema-version-9 database read NULL mirrors; applying 0010 alone (no table touched, no row copied) fills them."""
    root, (r0, r1), _ = _corpus(tmp_path)
    live = DB.db_path(root)
    v9 = str(tmp_path / "v9.sqlite")
    conn9 = DB.connect(v9)
    for version, slug, mpath in M.migration_files():
        if version <= 9:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn9, version, slug, fh.read())
    conn9.commit()
    # carry the corpus's rows over table by table, by the columns the two shapes SHARE (0010 is view-only, but 0012 gave `instrument` two
    # padding columns the v9 shape lacks, so a bare `SELECT *` no longer lines up; every copied row here is unpadded, so nothing is lost)
    conn9.execute("ATTACH DATABASE ? AS live", (live,))
    conn9.execute("PRAGMA foreign_keys=OFF")
    for t in S.ALL_TABLES:
        if t == "schema_migration":
            continue
        v9_cols = [r[1] for r in conn9.execute(f"PRAGMA main.table_info({t})")]
        if not v9_cols:
            continue      # a table newer than the v9 shape (`item_withdrawal`, 0020) has no v9 rows to copy
        live_cols = {r[1] for r in conn9.execute(f"PRAGMA live.table_info({t})")}
        cols = ", ".join(c for c in v9_cols if c in live_cols)
        conn9.execute(f"INSERT INTO main.{t} ({cols}) SELECT {cols} FROM live.{t}")
    conn9.commit()
    conn9.execute("DETACH DATABASE live")
    before = _long(conn9, r0)
    assert all(before[c] is None for c in V.DEGENERATION_FRAC_COLUMNS), before
    # every migration after 9 applies (0010 and, since 2026-09-22, 0011's operating-point fallback: view-only too); the shape is then the code's
    assert M.apply_migrations(conn9) == list(range(10, M.code_version() + 1)) and DB.schema_drift(conn9) == []
    after = _long(conn9, r0)
    assert after["frac_degenerate_union5"] == pytest.approx(JUDGED_UNION5) and after["frac_defended_degenerate"] == pytest.approx(RATES["defended_degenerate"])
    assert _long(conn9, r1)["frac_degenerate_union5"] == pytest.approx(0.75)
    conn9.close()


def test_the_derivation_table_names_every_mirror_and_a_vocabulary_measure_at_a_vocabulary_scope():
    assert set(S.FRAC_MIRROR_DERIVATION) == set(V.DEGENERATION_FRAC_COLUMNS)
    pairs = dict(S.DEGENERATION_RATE_MEASURES)
    for col, measures in S.FRAC_MIRROR_DERIVATION.items():
        for m in measures:
            assert m in V.DEGENERATION_MEASURES and pairs[m] in V.DEGENERATION_SCOPES, (col, m)
    assert pairs["defended_degenerate"] == "defended_only" and pairs["loop_first512"] == "first_512_tokens" and pairs["union5"] == "all"
    assert S.FRAC_MIRROR_DERIVATION["frac_at_token_cap"] == ("at_token_cap", "at_token_cap_finish_reason")
    assert S.frac_mirror_sql("frac_degenerate_union5") == "COALESCE(rm.frac_degenerate_union5, dj.union5, dn.union5)"
    assert "v_run_degeneration_current" in S.VIEW_REFERENCES["v_paper_long"] and "run_degeneration" in S.VIEW_TABLES["v_multi_source_rows"]
