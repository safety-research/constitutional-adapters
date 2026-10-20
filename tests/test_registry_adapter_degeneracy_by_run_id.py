"""The degeneracy-union reader's run-id-keyed batch files (`degeneracy_union/by_run_id/<batch>.vN.json`; EMISSION_VERSION 2,
2026-09-21): a good batch files one `run_degeneration` row per measure the row carries on the run the connected registry confirms
(judgment '' , `computed_by` producer, the row's `definition_version`, the file's `produced_utc`, the file as `source_uri`); a row
whose identity disagrees with the run row, a run the registry does not hold, a `p != k / n`, a duplicate run, a reading another
source already MEASURED are refused BY NAME while the rest files; a `self_check` that did not pass refuses the whole file; the
newest `.vN` per batch stem is the input and a `*.superseded-by-*` copy never; one comparison group per batch stem. A version bump
renames the stem's one input, and the reader DECLARES the older spellings retired (`inputs_retired`, the inputs-dropped guard's
rule), so the bump publishes; the newest file gone while an older stays is a genuine drop and refuses.
"""
from __future__ import annotations

import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import db as DB
from sisteer.registry import records as R
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import degeneracy_union as DU
from sisteer.registry.adapters import lineage_common as LC
from test_registry_adapter_degeneracy_union import _write_files

DEF_SHA = "98ae9114" + "c" * 56
PRODUCED = "2026-09-21T18:30:00.000Z"
MEASURES_OF = {"classifier": (3, 200), "contentless": (1, 200), "rep_ratio_loop": (5, 200), "zlib_collapse": (2, 200), "union5": (9, 200),
               "at_token_cap_retokenized": (4, 200), "at_token_cap_finish_reason": (4, 200)}


def _measures(scale: int = 1, not_measured=("truncation_only", "loop_without_cap")):
    out = {m: {"k": k * scale, "n": n * scale, "p": (k * scale) / (n * scale)} for m, (k, n) in MEASURES_OF.items()}
    for m in not_measured:
        out[m] = {"state": "not_measured", "reason": "the batch instrument carries no decomposition for these draws"}
    return out


def _identity(root, rid):
    (row,) = F.q(root, "SELECT store_uri, campaign_cell_id_legacy, eval_id, instrument_id FROM run WHERE run_id = ?", (rid,))
    return dict(zip(("store_uri", "campaign_cell_id_legacy", "eval_id", "instrument_id"), row))


def _row(root, rid, scale=1, unit="draws", **over):
    r = dict(_identity(root, rid), run_id=rid, unit=unit, n=200 * scale, measures=_measures(scale),
             by_reason={"at_cap": 4 * scale, "rep_ratio_loop": 5 * scale}, definition_version=f"{DU.BY_RUN_ID_SCHEMA}@{DEF_SHA}")
    r.update(over)
    return r


def _batch(rows, **over):
    b = {"schema": DU.BY_RUN_ID_SCHEMA, "instrument_module": "sisteer.eval.degeneracy_union", "definition_sha": DEF_SHA, "produced_utc": PRODUCED,
         "self_check": {"status": "passed", "checked": ["p == k / n", "n == rows"]}, "rows": rows}
    b.update(over)
    return b


def _write_batch(rs, name, batch):
    d = os.path.join(rs, DU.BY_RUN_ID_DIR)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(batch, fh)
    return p


def _setup(tmp_path, with_campaign_files=False):
    """A registry with stand-in runs (two sampled trials of the LoRA on past-tense; one of codeattack) for the batch rows to name. With
    the campaign files, the synthetic r2 export is ingested first so every run the campaign union files name exists and their
    groups are accepted (test_registry_degeneracy_rebuild's registry, without the union reader's pass)."""
    seeds = F.synthetic_seeds()
    res = F.resolver(seeds, tmp_path)
    obj = res.resolve(F.VIEW, F.LORA)
    if with_campaign_files:
        from sisteer.registry.adapters import r2_export as X
        rs = _write_files(tmp_path)
        paths = F.synthetic_r2_export(tmp_path)
        root = F.registry_root(tmp_path, seeds, [res.resolve(F.VIEW, c) for c in (F.LORA, F.RELU, F.SUB, F.PROMPTED)])
        rep, ing, ing2 = F.run_and_ingest(root, X.R2ExportAdapter(data_dir=paths["data"], rs=paths["rs"], seeds=seeds, objects=res))
        assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    else:
        rs = str(tmp_path / "rs")
        os.makedirs(rs, exist_ok=True)
        root = F.registry_root(tmp_path, seeds, [obj, res.resolve(F.VIEW, F.RELU)])
    runs = F.standin_sampled_trial_runs(root, seeds, obj, (0, 1), F.LORA, "pasttense")
    runs += F.standin_sampled_trial_runs(root, seeds, obj, (0,), F.LORA, "codeattack")
    return rs, seeds, res, root, runs


def _adapter(rs, seeds, res, tmp_path):
    return DU.DegeneracyUnionAdapter(rs=rs, seeds=seeds, csv_dir=str(tmp_path / "nocsv"), objects=res)


def _degeneration_rows(root):
    return [tuple(r) for r in F.q(root, "SELECT run_id, judgment_id, measure, scope, measurement_state, numerator, denominator, denominator_unit, rate, "
                                        "definition_version, computed_by, computed_utc, source_uri, aggregation, notes, extra_json FROM v_run_degeneration_current "
                                        "ORDER BY run_id, measure")]


def _refusals(root, group):
    out = []
    for p in R.iter_record_files(os.path.join(root, "records", "adapters", "degeneracy_union")):
        for _, rec, _ in R.read_record_file(p):
            if rec["record_type"] == "source_state" and rec["payload"]["json_path"] == "$.row_refusals" and rec["payload"]["source_id"].endswith(group):
                out += rec["payload"]["current_members_json"]
    return out


def test_a_good_batch_files_one_row_per_measure_on_the_confirmed_runs(tmp_path):
    rs, seeds, res, root, runs = _setup(tmp_path)
    p = _write_batch(rs, "panel_a.json", _batch([_row(root, runs[0]), _row(root, runs[1], scale=2, unit="replies")]))
    ad = _adapter(rs, seeds, res, tmp_path)
    assert ad.by_run_id == {"panel_a": (1, os.path.join(DU.BY_RUN_ID_DIR, "panel_a.json"))}
    inputs = ad.discover()
    assert [os.path.basename(i.uri) for i in inputs] == ["panel_a.json"] and ad.comparison_group(inputs[0]) == "by_run_id/panel_a"
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0, [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0 and ing.blocked + ing2.blocked == 0, (ing.refusals, ing2.refusals)
    assert len(ad.refusals) == 0, ad.refusals.items
    rows = _degeneration_rows(root)
    assert len(rows) == 18 and {r[0] for r in rows} == set(runs[:2])
    by = {(r[0], r[2]): r for r in rows}
    u = by[(runs[0], "union5")]
    assert u[1] == "" and u[3] == "all" and u[4] == "measured" and (u[5], u[6], u[7]) == (9, 200, "draws") and u[8] == pytest.approx(9 / 200)
    assert u[9] == f"{DU.BY_RUN_ID_SCHEMA}@{DEF_SHA}" and u[10] == "producer" and u[11] == PRODUCED and u[13] == "union"
    assert u[12].endswith("degeneracy_union/by_run_id/panel_a.json ›rows[0]")
    extra = json.loads(u[15])
    assert extra["by_reason"] == {"at_cap": 4, "rep_ratio_loop": 5} and extra["identity"]["eval_id"] == "pasttense" and extra["batch"] == "panel_a"
    assert extra["instrument_module"] == "sisteer.eval.degeneracy_union" and extra["self_check"]["status"] == "passed"
    c = by[(runs[0], "classifier")]
    assert c[13] == "single_detector" and (c[5], c[6]) == (3, 200) and "by_reason" not in json.loads(c[15])
    nm = by[(runs[0], "truncation_only")]
    assert nm[4] == "not_measured" and nm[5] is None and nm[8] is None and nm[14] == "the batch instrument carries no decomposition for these draws"
    r2 = by[(runs[1], "union5")]
    assert (r2[5], r2[6], r2[7]) == (18, 400, "replies")
    assert json.loads(by[(runs[1], "classifier")][15])["identity"]["campaign_cell_id_legacy"] == F.LORA
    # the threshold params are the union classifier's own, as for every other union row
    (tp,) = F.q(root, "SELECT threshold_params_json FROM v_run_degeneration_current WHERE run_id = ? AND measure = 'rep_ratio_loop'", (runs[0],))
    assert json.loads(tp[0]) == DU.THRESHOLDS["rep_ratio_loop"]
    # the group's source_state names the file under $.rows, and the revision carries the bumped emission version
    states = F.q(root, "SELECT json_path, found, current_members_json FROM source_state WHERE source_id = 'degeneracy_union:by_run_id/panel_a'")
    assert ("$.rows", 1, '["n_rows=2"]') in [(s[0], s[1], s[2]) for s in states]
    assert F.q(root, "SELECT DISTINCT json_extract(extra_json, '$.emission_version') FROM source_revision WHERE adapter = 'degeneracy_union'") == [("2",)]
    assert ad.stats["by_run_id_runs_filed"] == 2 and ad.stats["rows:panel_a.json"] == 2 and ad.stats["rows:by_run_id/panel_a"] == 18
    facts = ad.status_extra("by_run_id/panel_a")
    assert facts["runs_filed"] == 2 and facts["rows_refused"] == 0 and facts["refused"] == [] and facts["filed"][runs[0]]["union5"] == pytest.approx(9 / 200)
    assert facts["filed"][runs[0]]["truncation_only"] is None and facts["definition_sha"] == DEF_SHA and facts["version"] == 1
    st = F.q(root, "SELECT ok, extra_json FROM source_status WHERE source_id = 'degeneracy_union:by_run_id/panel_a'")
    assert st and st[0][0] == 1 and json.loads(st[0][1])["runs_filed"] == 2
    # a second pass over the unchanged file reads nothing again
    rep2, _, _ = F.run_and_ingest(root, _adapter(rs, seeds, res, tmp_path))
    assert rep2.groups_refreshed == 0


def test_identity_mismatch_unknown_run_bad_p_and_duplicate_are_refused_by_name_while_the_rest_files(tmp_path):
    rs, seeds, res, root, runs = _setup(tmp_path)
    good = _row(root, runs[0])
    wrong_store = _row(root, runs[1], store_uri="/somewhere/else/grades")
    unknown = _row(root, runs[2], run_id="sdfbase_full22:pasttense:000000000000:0000000000000000")
    bad_p = _row(root, runs[2])
    bad_p["measures"]["union5"] = {"k": 9, "n": 200, "p": 0.5}
    dup = _row(root, runs[0], scale=3)
    p = _write_batch(rs, "panel_b.json", _batch([good, wrong_store, unknown, bad_p, dup, "not-a-row"]))
    ad = _adapter(rs, seeds, res, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and rep.groups_refused == 0 and ing.refused + ing2.refused == 0
    rows = _degeneration_rows(root)
    assert {r[0] for r in rows} == {runs[0]} and len(rows) == 9
    assert [r[5] for r in rows if r[2] == "union5"] == [9]           # the first row's reading, not the duplicate's
    keys = {it["key"]: it["why"] for it in ad.refusals.items}
    assert f"by_run_id/panel_b|rows[1]|{runs[1]}" in keys and "store_uri: the row says '/somewhere/else/grades'" in keys[f"by_run_id/panel_b|rows[1]|{runs[1]}"]
    assert "two authorities disagree" in keys[f"by_run_id/panel_b|rows[1]|{runs[1]}"]
    assert "not in the connected registry" in keys["by_run_id/panel_b|rows[2]|sdfbase_full22:pasttense:000000000000:0000000000000000"]
    assert "p=0.5 != k/n=0.045" in keys[f"by_run_id/panel_b|rows[3]|{runs[2]}"]
    assert "a second row for this run (the first is rows[0])" in keys[f"by_run_id/panel_b|rows[4]|{runs[0]}"]
    assert keys["by_run_id/panel_b|rows[5]"] == "the row is not an object"
    assert len(keys) == 5
    members = _refusals(root, "by_run_id/panel_b")
    assert len(members) == 5 and any("two authorities disagree" in m for m in members)
    facts = ad.status_extra("by_run_id/panel_b")
    assert facts["runs_filed"] == 1 and facts["rows_refused"] == 5 and len(facts["refused"]) == 5
    # the read stands (`ok`, as every reader's by-name row refusals do): the refusals are FACTS in the status row, visible by name
    st = F.q(root, "SELECT ok, extra_json FROM source_status WHERE source_id = 'degeneracy_union:by_run_id/panel_b'")
    assert st[0][0] == 1 and len(json.loads(st[0][1])["refused"]) == 5 and json.loads(st[0][1])["rows_refused"] == 5


def test_a_self_check_that_did_not_pass_refuses_the_whole_file_and_files_nothing(tmp_path):
    rs, seeds, res, root, runs = _setup(tmp_path)
    _write_batch(rs, "panel_c.json", _batch([_row(root, runs[0])], self_check={"status": "failed", "what": "p != k / n on 3 rows"}))
    ad = _adapter(rs, seeds, res, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refused == 1 and rep.groups_refreshed == 0
    (res_,) = rep.results
    assert not res_.accepted and "self_check.status reads 'failed', not 'passed'" in (res_.error or "")
    assert _degeneration_rows(root) == []
    # so does a file of another schema, and a file with no rows list
    for name, over in (("panel_d.json", {"schema": "degeneracy_union_mt_cap4096/1"}), ("panel_e.json", {"rows": {"not": "a list"}}),
                       ("panel_f.json", {"definition_sha": "abc"})):
        b = _batch([_row(root, runs[0])])
        b.update(over)
        _write_batch(rs, name, b)
    ad = _adapter(rs, seeds, res, tmp_path)
    rep, _, _ = F.run_and_ingest(root, ad)
    errors = {r.group: r.error or "" for r in rep.results}
    assert "does not know that shape" in errors["by_run_id/panel_d"] and "no `rows` list" in errors["by_run_id/panel_e"]
    assert "is not a definition sha" in errors["by_run_id/panel_f"] and _degeneration_rows(root) == []


def test_without_a_registry_connection_the_batch_is_refused(tmp_path):
    rs, seeds, res, root, runs = _setup(tmp_path)
    _write_batch(rs, "panel_g.json", _batch([_row(root, runs[0])]))
    ad = _adapter(rs, seeds, res, tmp_path)
    (inp,) = ad.discover()
    ctx = AB.AdapterContext(root=root, now=R.utc_now(), producer=F.producer(), conn=None)
    with pytest.raises(AB.AdapterRefusal, match="no registry connection"):
        list(ad.read(ctx, "by_run_id/panel_g", [inp]))


def test_a_measure_another_source_already_measured_is_refused_by_name_and_the_rest_of_the_row_files(tmp_path):
    rs, seeds, res, root, runs = _setup(tmp_path)
    # another source's MEASURED union5 on run 0 (a natively emitted row, as a store reader or the harness would file)
    now = R.utc_now()
    native = {"run_id": runs[0], "judgment_id": "", "measure": "union5", "scope": "all", "measurement_state": "measured", "numerator": 50,
              "denominator": 200, "denominator_unit": "draws", "pooling_json": {}, "aggregation": "union", "rate": 0.25,
              "definition_version": "another_reader/1", "threshold_params_json": {}, "computed_by": "producer", "computed_utc": now,
              "source_uri": "synthetic://another-reader", "notes": None}
    placeholder = dict(native, measure="classifier", measurement_state="not_measured", numerator=None, denominator=None, denominator_unit=None,
                       rate=None, aggregation="single_detector")
    R.write_records(os.path.join(root, "records", "standins", "000003-other-degeneration.jsonl"),
                    [R.make_record("run_degeneration", native, F.producer(), now), R.make_record("run_degeneration", placeholder, F.producer(), now)])
    from sisteer.registry import harvest as H
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    _write_batch(rs, "panel_h.json", _batch([_row(root, runs[0]), _row(root, runs[1])]))
    ad = _adapter(rs, seeds, res, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and ing.refused + ing2.refused == 0
    keys = {it["key"]: it["why"] for it in ad.refusals.items}
    assert sorted(keys) == [f"by_run_id/panel_h|rows[0]|{runs[0]}|classifier", f"by_run_id/panel_h|rows[0]|{runs[0]}|union5"]
    # the MEASURED native union5 is never overwritten; the native not_measured CLASSIFIER blocks too -- v_run_degeneration_current ranks a
    # natively emitted row ahead of every adapter row, so a batch reading beside it could never be read, and is not filed
    assert "a natively emitted measured row" in keys[f"by_run_id/panel_h|rows[0]|{runs[0]}|union5"]
    assert "a natively emitted not_measured row" in keys[f"by_run_id/panel_h|rows[0]|{runs[0]}|classifier"] and "could never be read" in keys[f"by_run_id/panel_h|rows[0]|{runs[0]}|classifier"]
    rows = _degeneration_rows(root)
    by = {(r[0], r[2]): r for r in rows}
    assert by[(runs[0], "union5")][5] == 50 and by[(runs[0], "union5")][9] == "another_reader/1"       # the other source's reading stands
    assert by[(runs[0], "classifier")][4] == "not_measured" and by[(runs[0], "classifier")][9] == "another_reader/1"
    assert by[(runs[0], "rep_ratio_loop")][5] == 5 and by[(runs[1], "union5")][5] == 9 and len([r for r in rows if r[0] == runs[0]]) == 9
    assert F.q(root, "SELECT COUNT(*) FROM run_degeneration WHERE run_id = ? AND measure IN ('union5', 'classifier')", (runs[0],)) == [(2,)]   # nothing dead filed
    facts = ad.status_extra("by_run_id/panel_h")
    assert facts["runs_filed"] == 2 and "union5" not in facts["filed"][runs[0]] and "classifier" not in facts["filed"][runs[0]]
    assert facts["filed"][runs[0]]["rep_ratio_loop"] == pytest.approx(5 / 200)


def test_another_readers_union5_placeholder_admits_the_batch_union5_while_its_measured_rows_still_block(tmp_path):
    """strength_mtdiff 6: the dose sweep's multi-turn runs carry `union5` as a `not_measured` placeholder -- the producer's `any`
    parked on the run -- beside four MEASURED detector rows. A batch row for such a run FILES its `union5` (a placeholder never blocks,
    and the measured row then outranks it in v_run_degeneration_current) and the other placeholders' measures, while its readings of the
    four measured measures are refused as already measured; the run's parked `any` is untouched. The order: the dose reader's 6 first,
    then the batch lands the canonical union5."""
    import test_registry_adapter_strength_mtdiff as SMT
    from sisteer.registry.adapters import strength_mtdiff as SM
    seeds, dose_reader, _rs_dose = SMT._adapter(tmp_path, standups=(("d1p5", SMT.LORA_FAMILY),), controls=(),
                                                 degen=[SMT._degen_row("d1p5", SMT.LORA_FAMILY, "actorattack")])
    obj = dose_reader._object_of("d1p5", SM.family_spec(SMT.LORA_FAMILY)[0], SMT.LORA_KEY, SMT.SHA[("d1p5", SMT.LORA_FAMILY)], "the synthetic bank")
    root = F.registry_root(tmp_path, seeds, [obj])
    rep, ing, ing2 = F.run_and_ingest(root, dose_reader)
    assert rep.groups_refused == 0 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    (rid,) = [r[0] for r in F.q(root, "SELECT run_id FROM run WHERE eval_id = 'actorattack' AND object_id <> 'none'")]
    before = {r[2]: r for r in _degeneration_rows(root) if r[0] == rid}
    assert before["union5"][4] == "not_measured" and {before[m][4] for m in ("contentless", "rep_ratio_loop", "zlib_collapse", "at_token_cap_finish_reason")} == {"measured"}
    parked = json.loads(F.q(root, "SELECT extra_json FROM run WHERE run_id = ?", (rid,))[0][0])[SM.DEGENERACY_ANY_RULE_KEY]
    assert parked["rate"] == SMT.DEG_K["steered"]["any"] / SMT.DEG_N["steered"]
    rs = str(tmp_path / "rs")
    _write_batch(rs, "dose_mt.json", _batch([_row(root, rid, unit="replies")]))
    ad = _adapter(rs, seeds, F.resolver(seeds, tmp_path), tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and ing.refused + ing2.refused == 0, (ing.refusals, ing2.refusals)
    keys = {it["key"]: it["why"] for it in ad.refusals.items}
    assert sorted(keys) == sorted(f"by_run_id/dose_mt|rows[0]|{rid}|{m}" for m in ("contentless", "rep_ratio_loop", "zlib_collapse", "at_token_cap_finish_reason")), keys
    assert all("a MEASURED row of strength_mtdiff:" in why and "never overwritten" in why for why in keys.values())
    after = {r[2]: r for r in _degeneration_rows(root) if r[0] == rid}
    assert after["union5"][4] == "measured" and after["union5"][5] == 9 and after["union5"][9] == f"{DU.BY_RUN_ID_SCHEMA}@{DEF_SHA}"     # the batch's, current
    assert after["classifier"][4] == "measured" and after["at_token_cap_retokenized"][4] == "measured"                                       # placeholders admitted
    assert after["contentless"][9].startswith(f"{SM.DEGENERACY_FILE}@") and after["contentless"][5] == SMT.DEG_K["steered"]["empty"]         # the dose reader's stand
    assert F.q(root, "SELECT COUNT(*) FROM run_degeneration WHERE run_id = ? AND measure = 'union5'", (rid,)) == [(2,)]                   # the placeholder stays on the record
    assert json.loads(F.q(root, "SELECT extra_json FROM run WHERE run_id = ?", (rid,))[0][0])[SM.DEGENERACY_ANY_RULE_KEY] == parked
    facts = ad.status_extra("by_run_id/dose_mt")
    assert facts["runs_filed"] == 1 and facts["filed"][rid]["union5"] == pytest.approx(9 / 200) and "contentless" not in facts["filed"][rid]


def test_two_batches_naming_one_run_in_one_pass_file_it_once(tmp_path):
    rs, seeds, res, root, runs = _setup(tmp_path)
    _write_batch(rs, "panel_i.json", _batch([_row(root, runs[0])]))
    _write_batch(rs, "panel_j.json", _batch([_row(root, runs[0], scale=2), _row(root, runs[1])]))
    ad = _adapter(rs, seeds, res, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 2 and ing.refused + ing2.refused == 0
    # panel_i (read first, in stem order) filed run 0; panel_j's measured readings for run 0 are refused by name, its placeholders file
    rows = _degeneration_rows(root)
    by = {(r[0], r[2]): r for r in rows}
    assert by[(runs[0], "union5")][5] == 9 and by[(runs[1], "union5")][5] == 9
    assert sorted(it["key"] for it in ad.refusals.items) == [f"by_run_id/panel_j|rows[0]|{runs[0]}|{m}" for m in sorted(MEASURES_OF)]
    assert all("by_run_id/panel_i filed a MEASURED" in it["why"] and "in this pass" in it["why"] for it in ad.refusals.items)
    assert F.q(root, "SELECT DISTINCT priority FROM source_revision WHERE adapter = 'degeneracy_union'") == [(DU.BY_RUN_ID_PRIORITY,)]
    # a later pass over panel_j alone (panel_i already ingested) refuses the same measures, now as another SOURCE's reading
    with open(os.path.join(rs, DU.BY_RUN_ID_DIR, "panel_j.json"), "r+", encoding="utf-8") as fh:
        data = json.load(fh)
        data["produced_utc"] = "2026-09-22T00:00:00.000Z"
        fh.seek(0), fh.truncate(), json.dump(data, fh)
    ad2 = _adapter(rs, seeds, res, tmp_path)
    rep2, _, _ = F.run_and_ingest(root, ad2)
    assert rep2.groups_refreshed == 1 and rep2.results[0].group == "by_run_id/panel_j"
    assert all("a MEASURED row of degeneracy_union:by_run_id/panel_i" in it["why"] for it in ad2.refusals.items) and len(ad2.refusals) == len(MEASURES_OF)


def test_the_newest_version_per_stem_is_the_input_and_a_superseded_copy_never(tmp_path):
    rs, seeds, res, root, runs = _setup(tmp_path)
    _write_batch(rs, "panel_k.json", _batch([_row(root, runs[0])]))
    ad = _adapter(rs, seeds, res, tmp_path)
    rep, _, _ = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and {(r[0], r[2]): r for r in _degeneration_rows(root)}[(runs[0], "union5")][5] == 9
    # Summary versions the batch in place: .v2 beside a renamed .v1 copy (and a stray .v1 that stayed): the .v2 is the ONE input of the stem
    os.rename(os.path.join(rs, DU.BY_RUN_ID_DIR, "panel_k.json"), os.path.join(rs, DU.BY_RUN_ID_DIR, "panel_k.json.superseded-by-v2"))
    _write_batch(rs, "panel_k.v1.json", _batch([_row(root, runs[0])]))
    _write_batch(rs, "panel_k.v2.json", _batch([_row(root, runs[0], scale=2), _row(root, runs[1])], produced_utc="2026-09-22T01:00:00.000Z"))
    _write_batch(rs, "panel_l.superseded-by-v3.json", _batch([_row(root, runs[2])]))
    ad2 = _adapter(rs, seeds, res, tmp_path)
    assert ad2.by_run_id == {"panel_k": (2, os.path.join(DU.BY_RUN_ID_DIR, "panel_k.v2.json"))}
    inputs = ad2.discover()
    assert [os.path.basename(i.uri) for i in inputs] == ["panel_k.v2.json"] and ad2.comparison_group(inputs[0]) == "by_run_id/panel_k"
    # the bump renames the stem's one input: declared retired (the older spellings of THIS stem, never panel_l's), the read publishes
    # with the comparison on its marker — the guard would otherwise refuse a 100 % drop (tests/test_registry_input_drop_guard.py)
    retired = ad2.inputs_retired("by_run_id/panel_k")
    assert {os.path.basename(u) for u in retired} == {"panel_k.json", "panel_k.v1.json", "panel_k.json.superseded-by-v2"}
    assert all(u.startswith(os.path.join(rs, DU.BY_RUN_ID_DIR) + os.sep) for u in retired) and ad2.inputs_retired("by_run_id/panel_l") == []
    assert ad2.inputs_retired("full22") == [] and ad2.inputs_retired("mt_cap4096") == []
    rep2, ing, ing2 = F.run_and_ingest(root, ad2)
    assert rep2.groups_refreshed == 1 and rep2.results[0].group == "by_run_id/panel_k" and ing.refused + ing2.refused == 0
    assert rep2.groups_inputs_dropped == 0 and rep2.results[0].accepted and not rep2.results[0].inputs_dropped
    (extra_json,) = F.q(root, "SELECT extra_json FROM source_revision WHERE revision_id = ?", (rep2.results[0].revision_id,))[0]
    cmp = json.loads(extra_json)
    assert (cmp["inputs_previous"], cmp["inputs_now"], cmp["inputs_dropped"], cmp["inputs_retired"], cmp["inputs_added"]) == (1, 1, 0, 1, 1)
    assert cmp["inputs_dropped_fraction"] == 0.0 and "inputs_dropped_paths" not in cmp
    rows = _degeneration_rows(root)
    by = {(r[0], r[2]): r for r in rows}
    assert by[(runs[0], "union5")][5] == 18 and by[(runs[0], "union5")][6] == 400          # the v2 reading superseded the v1 one, same source
    assert by[(runs[1], "union5")][5] == 9 and runs[2] not in {r[0] for r in rows}
    assert all("panel_k.v2.json" in r[12] for r in rows) and not any("superseded" in r[12] for r in rows)
    assert F.q(root, "SELECT COUNT(*) FROM source_revision WHERE source_id = 'degeneracy_union:by_run_id/panel_k' AND accepted = 1 AND superseded_by_revision_id IS NULL") == [(1,)]
    assert ad2.status_extra("by_run_id/panel_k")["version"] == 2


def test_the_newest_file_gone_while_an_older_stays_is_a_genuine_drop_and_refuses(tmp_path):
    """Not a version bump: the `.v2` the head read vanishes and the stray `.v1` becomes the newest candidate. Nothing declares the
    `.v2` retired (a stem at version 1 retires nothing), so the read refuses, the head stays and the rows are unchanged."""
    rs, seeds, res, root, runs = _setup(tmp_path)
    _write_batch(rs, "panel_k.v1.json", _batch([_row(root, runs[0])]))
    _write_batch(rs, "panel_k.v2.json", _batch([_row(root, runs[0], scale=2), _row(root, runs[1])], produced_utc="2026-09-22T01:00:00.000Z"))
    ad = _adapter(rs, seeds, res, tmp_path)
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 1 and ing.refused + ing2.refused == 0
    head = rep.results[0].revision_id
    before = _degeneration_rows(root)
    assert {r[0] for r in before} == {runs[0], runs[1]}
    os.remove(os.path.join(rs, DU.BY_RUN_ID_DIR, "panel_k.v2.json"))
    ad2 = _adapter(rs, seeds, res, tmp_path)
    assert ad2.by_run_id == {"panel_k": (1, os.path.join(DU.BY_RUN_ID_DIR, "panel_k.v1.json"))} and ad2.inputs_retired("by_run_id/panel_k") == []
    rep2, ing3, ing4 = F.run_and_ingest(root, ad2)
    (res2,) = rep2.results
    assert not res2.accepted and res2.inputs_dropped and rep2.groups_inputs_dropped == 1 and rep2.groups_refreshed == 0
    assert "inputs_dropped_unexplained: 1 of the 1 inputs" in res2.error and "panel_k.v2.json" in res2.error
    assert ing3.refused + ing4.refused == 0
    assert F.q(root, "SELECT revision_id FROM source_revision WHERE source_id = 'degeneracy_union:by_run_id/panel_k' AND accepted = 1 AND superseded_by_revision_id IS NULL") == [(head,)]
    assert _degeneration_rows(root) == before


def test_batches_and_the_campaign_files_are_separate_groups_of_one_reader_and_a_campaign_reading_outranks_a_batch(tmp_path):
    """The sampled union file measures the `t0` trial run (runs[0]) and files `not_measured` placeholders for the pending codeattack
    trial (runs[2]). A batch naming both, read in the SAME pass (groups read in name order, `by_run_id/…` first, so the batch cannot
    see the sampled reading yet): after ingest the sampled reading of runs[0] is the current row (priority 0 over the batch's
    BY_RUN_ID_PRIORITY), the batch's reading of runs[2] is (measured over a placeholder); on the next pass the batch refuses runs[0]'s
    measured keys BY NAME as the sampled file's."""
    rs, seeds, res, root, runs = _setup(tmp_path, with_campaign_files=True)
    _write_batch(rs, "panel_m.json", _batch([_row(root, runs[0]), _row(root, runs[2], scale=2)]))
    ad = _adapter(rs, seeds, res, tmp_path)
    groups = {ad.comparison_group(i) for i in ad.discover()}
    assert groups == {"full22", "mt_cap4096", "sampled", "petri", "by_run_id/panel_m"}
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 5 and [r.group for r in rep.results][0] == "by_run_id/panel_m", [r.error for r in rep.results]
    assert ing.refused + ing2.refused == 0 and len(ad.refusals) == 0
    prio = dict(F.q(root, "SELECT source_id, priority FROM source_revision WHERE adapter = 'degeneracy_union' AND accepted = 1"))
    assert prio["degeneracy_union:by_run_id/panel_m"] == DU.BY_RUN_ID_PRIORITY == -1 and prio["degeneracy_union:sampled"] == 0
    by = {(r[0], r[2]): r for r in _degeneration_rows(root)}
    assert "static_degeneracy_union.json" in by[(runs[0], "union5")][12] and by[(runs[0], "union5")][5] == 2          # the campaign reading is current
    assert "panel_m.json" in by[(runs[2], "union5")][12] and by[(runs[2], "union5")][5] == 18                         # measured over the placeholder
    assert F.q(root, "SELECT COUNT(*) FROM run_degeneration WHERE run_id = ? AND measure = 'union5'", (runs[0],)) == [(2,)]   # both filed, one read
    # the next pass over the batch alone: the sampled file's readings are another source's now, refused by name
    with open(os.path.join(rs, DU.BY_RUN_ID_DIR, "panel_m.json"), "r+", encoding="utf-8") as fh:
        data = json.load(fh)
        data["produced_utc"] = "2026-09-22T02:00:00.000Z"
        fh.seek(0), fh.truncate(), json.dump(data, fh)
    ad2 = _adapter(rs, seeds, res, tmp_path)
    rep2, ing3, ing4 = F.run_and_ingest(root, ad2)
    # the batch is the one ACCEPTED read of the pass (the held cap-4,096 group -- its declared runs never landed here -- reports as held)
    assert [r.group for r in rep2.results if r.accepted and not r.held] == ["by_run_id/panel_m"] and ing3.refused + ing4.refused == 0
    keys = sorted(it["key"] for it in ad2.refusals.items)
    assert keys == [f"by_run_id/panel_m|rows[0]|{runs[0]}|{m}" for m in sorted(MEASURES_OF)]
    assert all("a MEASURED row of degeneracy_union:sampled" in it["why"] for it in ad2.refusals.items)
    by = {(r[0], r[2]): r for r in _degeneration_rows(root)}
    assert "static_degeneracy_union.json" in by[(runs[0], "union5")][12] and "panel_m.json" in by[(runs[2], "union5")][12]
    # the keys the sampled file does not measure: two placeholders stand (the sampled file's and the batch's); the campaign file's is the
    # current one by priority, the batch's is in the table -- a placeholder never hides a number and never files a dead number
    assert by[(runs[0], "truncation_only")][4] == "not_measured" and "static_degeneracy_union.json" in by[(runs[0], "truncation_only")][12]
    assert F.q(root, "SELECT COUNT(*) FROM run_degeneration d JOIN source_revision sr ON sr.revision_id = d.source_revision_id WHERE d.run_id = ? "
                     "AND d.measure = 'truncation_only' AND d.source_uri LIKE '%panel_m.json%' AND sr.accepted = 1 AND sr.superseded_by_revision_id IS NULL", (runs[0],)) == [(1,)]


def test_the_reader_pins_its_emission_version_and_the_shape_constants():
    assert DU.DegeneracyUnionAdapter.EMISSION_VERSION == 2 and AB.emission_version_of(DU.DegeneracyUnionAdapter) == "2"
    assert DU.BY_RUN_ID_SCHEMA == "degeneracy_union_by_run_id/1" and DU.BY_RUN_ID_DIR == os.path.join("degeneracy_union", "by_run_id")
    assert DU.BY_RUN_ID_IDENTITY_FIELDS == ("store_uri", "campaign_cell_id_legacy", "eval_id", "instrument_id")
    from sisteer.registry import vocab as V
    assert set(DU.BY_RUN_ID_MEASURES) <= set(V.DEGENERATION_MEASURES) and set(DU.BY_RUN_ID_ROWS_ARE) <= set(V.DENOMINATOR_UNITS)
    assert DU.by_run_id_group("/x/degeneracy_union/by_run_id/panel_a.v3.json") == "by_run_id/panel_a" == DU.by_run_id_group("/x/panel_a.json")
    assert DU.DegeneracyUnionAdapter._definition_sha_agrees(f"{DU.BY_RUN_ID_SCHEMA}@98ae9114cccc", "98ae9114" + "c" * 56)
    ad = DU.DegeneracyUnionAdapter(rs="/nonexistent/rs", seeds=F.synthetic_seeds(), csv_dir="/nonexistent/csv")
    from sisteer.registry.adapters import runner as RUN
    assert ad.priority_for("by_run_id/panel") == DU.BY_RUN_ID_PRIORITY == RUN.adapter_priority(ad, "by_run_id/panel") == -1
    assert ad.priority_for("full22") == 0 == RUN.adapter_priority(ad, "sampled") == RUN.adapter_priority(ad)
    assert DU.DegeneracyUnionAdapter._definition_sha_agrees(f"{DU.BY_RUN_ID_SCHEMA}@{'9' * 64}", "9" * 12)
    assert not DU.DegeneracyUnionAdapter._definition_sha_agrees(f"{DU.BY_RUN_ID_SCHEMA}@1234567890ab", "98ae9114" + "c" * 56)
    assert not DU.DegeneracyUnionAdapter._definition_sha_agrees(f"{DU.BY_RUN_ID_SCHEMA}@98ae", "98ae9114" + "c" * 56)       # too short to name a definition
