"""`registry degeneracy-rebuild`: the 2026-09-14 table's statistic (union of the four text detectors,
at-cap excluded) re-derived from the registry's union rows and diffed against a synthetic CSV by key."""
from __future__ import annotations

import csv
import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import db as DB
from sisteer.registry import degeneracy_artifact as DA
from sisteer.registry import identity as I
from sisteer.registry.adapters import lineage_common as LC
from sisteer.registry.adapters import degeneracy_union as DU
from sisteer.registry.adapters import r2_export as X
from test_registry_adapter_degeneracy_union import _write_files


def test_union_excluding_cap_arithmetic():
    assert DA.union_excluding_cap({"at_cap": 100, "classifier+at_cap": 3, "rep_ratio_loop": 5, "zlib_collapse+contentless": 2}) == 10
    assert DA.union_excluding_cap({}) == 0


def _registry(tmp_path):
    """r2 export rows (the runs) + the union rows, ingested into one registry."""
    rs = _write_files(tmp_path)
    paths = F.synthetic_r2_export(tmp_path)
    seeds = F.synthetic_seeds()
    res = F.resolver(seeds, tmp_path)
    root = F.registry_root(tmp_path, seeds, [res.resolve(F.VIEW, c) for c in (F.LORA, F.RELU, F.SUB, F.PROMPTED)])
    x = X.R2ExportAdapter(data_dir=paths["data"], rs=paths["rs"], seeds=seeds, objects=res)
    F.run_and_ingest(root, x)
    # the sampled re-run's per-trial runs are the producer's (`sampled_static`); stand-ins here carry the same id derivation
    F.standin_sampled_trial_runs(root, seeds, res.resolve(F.VIEW, F.LORA), (0, 1), F.LORA, "pasttense")
    F.standin_sampled_trial_runs(root, seeds, res.resolve(F.VIEW, F.LORA), (0,), F.LORA, "codeattack")   # the pending trial's run (its rows are not_measured)
    du = DU.DegeneracyUnionAdapter(rs=rs, seeds=seeds, csv_dir=str(tmp_path / "nocsv"), objects=res)
    rep, ing, ing2 = F.run_and_ingest(root, du)
    assert rep.groups_refreshed == 4 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    return root


def test_rows_from_registry_and_diff(tmp_path):
    root = _registry(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rows = DA.registry_rows(conn)
        # the greedy static lora cell: by_reason {at_cap: 3, classifier+at_cap: 1, rep_ratio_loop: 3} → 4 of 20 = 20 %
        stat = [r for r in rows if r["eval_or_attack"] == "past-tense" and r["cap_or_rung"] == "4096" and r["object_arm"] == F.LORA]
        assert len(stat) == 1 and stat[0]["numerator"] == 4 and stat[0]["denominator"] == 20 and stat[0]["degen_rate_pct"] == 20.0
        assert stat[0]["denominator_unit"] == "draws (goal x variant, 1 per row)" and stat[0]["lane"] == "full22 greedy static (4096)" and stat[0]["measure"] == "all"
        assert "classify 2, 4gram-rep 3, zlib 1, contentless 0" in stat[0]["notes"]
        # the multi-turn cell: target replies
        mt = [r for r in rows if r["eval_or_attack"] == "Crescendo" and r["cap_or_rung"] == "512"]
        assert len(mt) == 1 and mt[0]["denominator_unit"] == "target replies (all turns)"
        # the sampled cell pools its two trials' by_reason counts: {classifier+at_cap: 1, rep_ratio_loop: 1} per trial → 2 + 2 of 40 draws, rung 8192
        sm = [r for r in rows if r["cap_or_rung"] == "rung 8192"]
        assert len(sm) == 1 and sm[0]["denominator"] == 40 and sm[0]["numerator"] == 4 and sm[0]["denominator_unit"] == "draws (t0+t1 pooled)"
        assert sm[0]["lane"] == "sampled static re-run (full22, vendor preset)" and "2 trial(s)" in sm[0]["notes"]
        # the over-1 % filter and the CSV tables
        over = DA.over_1pct(rows)
        out = str(tmp_path / "out")
        paths = DA.write_tables(over, out)
        assert set(paths) == {"degen_all_over1pct.csv", "degen_glm_over1pct.csv", "degen_paper_over1pct.csv"}
        # the greedy 4,096 static and 512 multi-turn runs carry a supersede ruling: in `all`, never in `paper`; the sampled rows are paper rows
        with open(paths["degen_paper_over1pct.csv"], newline="") as fh:
            paper = list(csv.DictReader(fh))
        assert {r["cap_or_rung"] for r in paper} == {"rung 8192"} and {r["_paper_status"] if "_paper_status" in r else "in_paper" for r in over if r["cap_or_rung"] == "4096"} == {"superseded"}
        with open(paths["degen_all_over1pct.csv"], newline="") as fh:
            written = list(csv.DictReader(fh))
        assert [c for c in written[0].keys()] == DA.CSV_COLUMNS and len(written) == len(over)
        # a CSV equal to the registry's rows diffs to zero; a perturbed one names the key
        csv_dir = str(tmp_path / "csvs")
        os.makedirs(csv_dir)
        with open(os.path.join(csv_dir, "degen_all_over1pct.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=DA.CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in over:
                w.writerow(r)
            w.writerow({**over[0], "object_arm": "some_other_cell", "lane": "another lane WS-B3 ingests"})
        d = DA.diff_against_csv([r for r in over if r.get("_paper_status") != "descoped"], os.path.join(csv_dir, "degen_all_over1pct.csv"))
        assert d["n_mismatched"] == 0 and d["only_in_registry"] == 0 and d["only_in_csv"] == 1 and d["only_in_csv_by_origin"] == {"another lane WS-B3 ingests": 1}
        rep = DA.rebuild(conn, str(tmp_path / "out2"), csv_dir)
        assert rep["rows_over_1pct"] == len(over) and "degen_all_over1pct.csv" in rep["diffs"]
        assert os.path.exists(os.path.join(str(tmp_path / "out2"), "degeneracy_rebuild_report.json"))
    finally:
        conn.close()
