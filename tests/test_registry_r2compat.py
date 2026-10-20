"""`registry export r2-compat`: the synthetic export round-trips through the adapter and back out of the registry
with the same 82 columns and values; the comparison reports per-key differences and never asserts. Since r2_export 11 the
adapter's contrasts are decided by the pass's anchor currency: the first pass over a fresh registry PARKS every comparison
(its own unsteered runs are not folded yet) and the second files them -- the round trip reproduces the export in both states,
because a park carries the producer's numbers (`delta_block`, `delta_vs_full_dose_block`) and the renderer reads them back."""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

import _b1_fixtures as F
from sisteer.registry import db as DB
from sisteer.registry import r2compat as RC
from sisteer.registry.adapters import r2_export as X


def _registry_with_export(tmp_path, passes: int = 2, **kw):
    """The synthetic export read into a fresh registry. ``passes`` 1 leaves every comparison PARKED (the first pass's frame knows none of the
    export's own unsteered runs; no provisional anchors); 2 folds them and files the contrasts (own_current)."""
    paths = F.synthetic_r2_export(tmp_path, **kw)
    seeds = F.synthetic_seeds()
    ad = X.R2ExportAdapter(data_dir=paths["data"], rs=paths["rs"], seeds=seeds, objects=F.resolver(seeds, tmp_path))
    objs = [ad.objects.resolve(F.VIEW, c) for c in (F.LORA, F.RELU, F.SUB, F.PROMPTED)]
    if kw.get("halfdose_rows"):
        objs.append(ad.objects.resolve(f"{F.VIEW}_halfdose", F.LORA))
    root = F.registry_root(tmp_path, seeds, objs)
    for i in range(passes):
        rep, ing, ing2 = F.run_and_ingest(root, ad, now=f"2026-10-03T0{i + 1}:00:00.000Z")
        assert rep.groups_refreshed == 1 and ing.refused + ing2.refused == 0, (rep.results[0].error, ing.refusals, ing2.refusals)
    return paths, root


def _round_trip(paths, root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        df = RC.render_frame(conn)
        ref = pd.read_parquet(paths["parquet"])
        rep = RC.compare(df, ref)
        # the registry also holds the refused-rate TWIN of the export's `over_refusal_rate` row (r2_export 9), carrying the source
        # row's park: it is the registry's row, not a second export row, and the round trip renders the source row alone
        twins = conn.execute("SELECT COUNT(*) FROM run_metric WHERE metric_id IN ('xst_refused', 'orb_refused') AND extra_json LIKE '%refused_rate_twin%'").fetchone()[0]
        contrasts = conn.execute("SELECT COUNT(*) FROM run_contrast").fetchone()[0]
        parked = conn.execute("SELECT COUNT(*) FROM run_metric WHERE json_extract(extra_json, '$.delta_block.anchor') IS NOT NULL").fetchone()[0]
    finally:
        conn.close()
    return df, ref, rep, twins, contrasts, parked


@pytest.mark.parametrize("passes", [1, 2])
def test_round_trip_reproduces_the_export(tmp_path, passes):
    """After ONE pass every comparison is parked (no contrast row; the producer's delta on `delta_block`); after two the contrasts are filed
    own-current. The rendered export is the same either way."""
    paths, root = _registry_with_export(tmp_path, passes=passes)
    df, ref, rep, twins, contrasts, parked = _round_trip(paths, root)
    assert twins == 1
    assert (contrasts == 0 and parked > 0) if passes == 1 else (contrasts > 0 and parked == 0)
    assert list(df.columns) == ["row_id"] + [c for c in RC.R2_COLUMNS if c != "row_id"] and len(df) == len(ref) == 17
    assert rep["only_rendered"] == 0 and rep["only_reference"] == 0, rep
    # every column round-trips: values, counts, intervals, contrasts, flags, prose, the four ReLU columns
    assert rep["columns_differing"] == {}, json.dumps(rep["columns_differing"], indent=1)
    # row_id is regenerated as the builder does: the sort key, then range(len)
    assert list(df["row_id"]) == list(range(len(df)))
    srt = df.sort_values(RC.SORT_KEYS, ascending=RC.SORT_ASCENDING, kind="stable", na_position="last")
    assert list(srt["row_id"]) == list(df["row_id"])


@pytest.mark.parametrize("passes", [1, 2])
def test_a_half_dose_row_round_trips_from_its_full_dose_contrast_or_its_park(tmp_path, passes):
    """The half-dose row's `delta_vs_full_dose*` render from the `full_dose` contrast once the 1x anchor has folded (pass 2) and from the
    `delta_vs_full_dose_block` park before (pass 1); its own delta from the contrast the registry derived against the main standup's unsteered
    run (pass 2: the row names no unsteered run of its own standup, so the same source's sibling anchors it and the producer's delta parks) or
    from its park (pass 1) -- the producer's numbers either way."""
    paths, root = _registry_with_export(tmp_path, passes=passes, halfdose_rows=True)
    df, ref, rep, _twins, contrasts, _parked = _round_trip(paths, root)
    assert len(df) == len(ref) == 18 and rep["only_rendered"] == 0 and rep["only_reference"] == 0, rep
    assert rep["columns_differing"] == {}, json.dumps(rep["columns_differing"], indent=1)
    half = df[df["view"] == f"{F.VIEW}_halfdose"]
    assert len(half) == 1 and float(half["delta_vs_full_dose"].iloc[0]) == 10.0 and float(half["delta"].iloc[0]) == -12.0
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        kinds = sorted(r[0] for r in conn.execute("SELECT anchor_kind FROM run_contrast rc JOIN run r ON r.run_id = rc.run_id WHERE r.view_legacy = ?",
                                                  (f"{F.VIEW}_halfdose",)).fetchall())
        blocks = [tuple(r) for r in conn.execute("SELECT json_extract(rm.extra_json, '$.delta_vs_full_dose_block.delta'), json_extract(rm.extra_json, '$.delta_block.delta') "
                                                 "FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.view_legacy = ?", (f"{F.VIEW}_halfdose",)).fetchall()]
    finally:
        conn.close()
    assert kinds == ([] if passes == 1 else ["full_dose", "sibling_none"])
    assert blocks == ([(10.0, -12.0)] if passes == 1 else [(None, -12.0)])     # pass 2: the full-dose park lifted by name; the producer's own delta parked beside the registry's contrast


def test_round_trip_renders_no_registry_derived_over_refusal_row(tmp_path):
    """With the sampled re-run's pooled rows in the export, the registry also holds the two rows it DERIVES from them -- `over_refusal_graded`
    (r2_export 7) and `over_refusal_label_mean` (r2_export 10) -- plus three refused-rate twins. None is an export row: the
    derived rows park no `row_id` (their `extra.derived_from` names the source rows' ids instead) and the twins carry the twin marker, so the
    round trip renders the export's own rows alone and every column still agrees."""
    paths, root = _registry_with_export(tmp_path, overrefusal_count_rows=True)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        df = RC.render_frame(conn)
        ref = pd.read_parquet(paths["parquet"])
        rep = RC.compare(df, ref)
        # render_frame leaves sqlite3.Row on the connection: plain tuples for the comparison below
        derived = [tuple(r) for r in conn.execute("SELECT metric_id, COUNT(*) FROM run_metric WHERE metric_id IN ('over_refusal_graded', 'over_refusal_label_mean') "
                                                  "GROUP BY metric_id ORDER BY metric_id").fetchall()]
        twins = conn.execute("SELECT COUNT(*) FROM run_metric WHERE metric_id IN ('xst_refused', 'orb_refused') AND extra_json LIKE '%refused_rate_twin%'").fetchone()[0]
    finally:
        conn.close()
    assert derived == [("over_refusal_graded", 1), ("over_refusal_label_mean", 1)] and twins == 3
    assert len(df) == len(ref) and not set(df["metric"]) & {"over_refusal_graded", "over_refusal_label_mean", "xst_refused", "orb_refused"}
    assert rep["only_rendered"] == 0 and rep["only_reference"] == 0, rep
    assert rep["columns_differing"] == {}, json.dumps(rep["columns_differing"], indent=1)


def test_write_export_and_compare_report_shape(tmp_path):
    paths, root = _registry_with_export(tmp_path)
    out = str(tmp_path / "out" / "r2.parquet")
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = RC.write_export(conn, out, "2026-01-03T00:00:00.000Z")
        ref = pd.read_parquet(paths["parquet"])
        # perturb a value: the comparison names the column and the key, and does not raise
        ref.loc[ref.index[1], "value"] = 0.123456
        cmp = RC.compare(RC.render_frame(conn), ref)
    finally:
        conn.close()
    assert rep["rows"] == 17 and rep["columns"] == 82 and all(os.path.exists(p) for p in rep["files"])
    schema = json.load(open(out.replace(".parquet", "_schema.json")))
    assert "_deprecation" in schema and "row_id" in schema
    sources = json.load(open(out.replace(".parquet", "_sources.json")))
    assert sources["n_rows"] == 17 and sources["sources"] and "deprecation" in sources
    assert cmp["columns_differing"]["value"]["n_differ"] == 1 and cmp["columns_differing"]["value"]["examples"]
