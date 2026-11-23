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


def test_a_natively_emitted_run_with_no_legacy_cell_is_one_arm_per_object_never_a_crash_or_a_pool(tmp_path):
    """A natively emitted run carries no ``campaign_cell_id_legacy`` (only backfilled runs do), so the pooled branch's
    ``"|".join([view, eval, cell])`` raised TypeError on the first such run. Its arm is its OBJECT -- the object's own legacy cell id,
    else its object_id -- so three native objects on one view, eval and sampled ladder instrument are three rows (one shared
    placeholder would pool them into one), two trials of one object are one pooled row, and a backfilled run keeps the producer's
    cell id over its object's."""
    from sisteer.registry import harvest as H
    from sisteer.registry import hooks as RH
    from sisteer.registry import identity
    from sisteer.registry import records as R
    from sisteer.registry import synth
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ladder = [4096, 8192]
    inst = synth.instrument_fields(token_cap_tokens=8192, token_cap_ladder_json=json.dumps(ladder), read_at="ladder_final",
                                   ladder_promotion_rule="share_at_cap > 0.01", decode_source="generation_config")
    inst_id = str(inst["instrument_id"])
    sha_a, sha_b = "a" * 64, "b" * 64
    arm_a = dict(synth.object_lora(sha_a), campaign_cell_id_legacy="base__arm_a")        # an object the seeds stamped with a legacy id
    arm_b = synth.object_lora(sha_b, paper_label="control_lora", role="control")            # a natively emitted object: no legacy id
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"),
                    synth.seed_dimensions() + [synth.rec("instrument", inst, synth.t(12)), synth.rec("object", arm_a, synth.t(13)),
                                               synth.rec("object", arm_b, synth.t(14))])
    # (key, object id, legacy cell id of the run, union5 numerator over the at-cap-free detectors, denominator)
    runs = [("nat_a_t0", sha_a, None, 6, 100), ("nat_a_t1", sha_a, None, 4, 100), ("nat_b_t0", sha_b, None, 30, 100),
            ("nat_none_t0", "none", None, 2, 100), ("bf_a_t0", sha_a, "producer_cell_a", 9, 50)]
    for n, (key, obj, cell, num, den) in enumerate(runs):
        rid = identity.run_id(synth.CAMPAIGN, key, key, synth.EVAL_ID, inst_id)
        over = dict(instrument_id=inst_id, run_id=rid, view_legacy="synthview", decode_source="generation_config", token_cap_tokens=8192,
                    token_cap_rung_index=1, token_cap_ladder_json=ladder)
        if cell is not None:
            over["campaign_cell_id_legacy"] = cell
        _, recs = synth.run_records(key, 100 + 20 * n, key, obj, **over)
        u5 = RH.measured_row(rid, "union5", "all", num + 3, den, "replies", judgment_id=identity.judgment_id(rid, synth.JUDGE_ID),
                             computed_utc=synth.t(500 + n), source_uri=f"stores/{key}/union.json ›cells[{key}]")
        u5["extra"] = {"by_reason": {"classifier": num, "at_cap": 3}}
        synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in recs] +
                            [("run_degeneration", [synth.rec("run_degeneration", u5, synth.t(500 + n))])])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM run WHERE campaign_cell_id_legacy IS NULL").fetchone()[0] == 4
        rows = DA.registry_rows(conn)
    finally:
        conn.close()
    by_arm = {r["object_arm"]: r for r in rows}
    assert sorted(by_arm) == sorted(["base__arm_a", sha_b, "none", "producer_cell_a"]), sorted(by_arm)
    a = by_arm["base__arm_a"]                                         # the object's legacy id; its two trials pooled
    assert (a["numerator"], a["denominator"], a["denominator_unit"]) == (10, 200, "draws (t0+t1 pooled)") and "2 trial(s)" in a["notes"]
    assert a["source_file"].endswith(" cells[synthview|" + synth.EVAL_ID + "|base__arm_a|t*].degenerate_share.by_reason"), a["source_file"]
    assert (by_arm[sha_b]["numerator"], by_arm[sha_b]["denominator"]) == (30, 100)             # no legacy id at either level: object_id
    assert (by_arm["none"]["numerator"], by_arm["none"]["denominator"]) == (2, 100)
    assert (by_arm["producer_cell_a"]["numerator"], by_arm["producer_cell_a"]["denominator"]) == (9, 50)   # the producer's cell wins
    assert all(r["view"] == "synthview" and r["cap_or_rung"] == "rung 8192" for r in rows)
