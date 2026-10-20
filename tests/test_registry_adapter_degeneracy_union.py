"""The degeneracy-union adapter on synthetic union files: both at-cap estimands as two measures, the union and the
defended-collapse bracket with counts, `not_measured` for every declared (run, measure, scope) without a reading
(the in-flight cap-4,096 file, a pending native cell), Petri replies, and the r2-compatible run identities."""
from __future__ import annotations

import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import degeneracy_union as DU
from sisteer.registry.adapters import runner as RUN
from sisteer.registry.adapters import lineage_common as LC

DEF_SHA = "9" * 64


def _union_cell(view, attack, cell, cap, n_rows, k_cls, k_rep, k_zlib, k_cont, k_cap_ret, k_cap_fr, k_union, n_goals=10, n_def=4, n_def_any=2, n_def_all=1,
                finish_reason_defined=True):
    return {"attack": attack, "cap": cap, "cell": cell, "definition_sha": DEF_SHA, "denominator": "rows", "n_rows": n_rows, "n_goals": n_goals,
            "private_run": f"synthetic_{view}",
            "detectors": {"classifier": {"n": k_cls, "denominator": n_rows, "by_reason": {"tail_loop": k_cls}}, "contentless": {"n": k_cont, "denominator": n_rows},
                          "rep_ratio_loop": {"n": k_rep, "denominator": n_rows, "threshold": 0.5}, "zlib_collapse": {"n": k_zlib, "denominator": n_rows},
                          "at_cap": {"canonical": "at_cap_retokenized",
                                     "at_cap_retokenized": {"defined": True, "denominator": n_rows, "n": k_cap_ret, "cap": cap, "fraction": 0.98},
                                     "at_cap_finish_reason": {"defined": finish_reason_defined, "denominator": n_rows if finish_reason_defined else None, "n": k_cap_fr if finish_reason_defined else None},
                                     "cross_tab": {"agreement": 1.0}, "disagreements": {"n": 0}}},
            "degenerate_share": {"n": k_union, "denominator": n_rows, "share": k_union / n_rows,
                                 "by_reason": {"at_cap": k_cap_ret - 1, "classifier+at_cap": 1, "rep_ratio_loop": k_rep}},
            "defended_collapse_bracket": {"denominator": "goals", "n_goals": n_goals, "n_defended": n_def, "n_defended_any_degenerate": n_def_any,
                                          "n_defended_all_degenerate": n_def_all, "primary": "n_defended_any_degenerate", "rule": "synthetic"}}


def _write_files(tmp_path, with_mt_file=False):
    rs = tmp_path / "rs"
    for d in ("degeneracy_union", "sampled/report/analysis", "sampled_native/report/analysis", "degeneracy_petri"):
        os.makedirs(rs / d, exist_ok=True)
    full = {"schema": "degeneracy_union_full22/1", "meta": {"DEFINITION_SHA": DEF_SHA, "written_at": "2026-01-01T00:00:00+00:00"},
            "cells": {f"{F.VIEW}|pasttense|{F.LORA}": _union_cell(F.VIEW, "pasttense", F.LORA, 4096, 20, 2, 3, 1, 0, 4, 0, 6, finish_reason_defined=False),
                      f"{F.VIEW}|crescendo|{F.LORA}": _union_cell(F.VIEW, "crescendo", F.LORA, 512, 100, 5, 4, 2, 1, 40, 39, 44)}}
    json.dump(full, open(rs / "degeneracy_union" / "degeneracy_full22_union.json", "w"))
    lane = {"models": {F.VIEW: {"cells": {F.RELU: {"crescendo": {"n": 10, "k": 9, "root": str(rs / "grades_synth_bu_synthview"), "namespace": "cap4096",
                                                                  "decode_cap_tokens": 4096},
                                                    "pair": {"n": 10, "k": 8, "root": str(rs / "grades_synth_bu_synthview"), "namespace": "cap4096", "decode_cap_tokens": 4096}}}}},
            "namespaces": {"cap4096": {"models": {F.VIEW: {"cells": [F.RELU]}}}}}
    json.dump(lane, open(rs / "transfer_stats_full22_cap4096.json", "w"))
    if with_mt_file:
        mt = {"schema": "degeneracy_union_mt_cap4096/1", "meta": {"DEFINITION_SHA": DEF_SHA, "written_at": "2026-01-02T00:00:00+00:00"},
              "cells": {f"{F.VIEW}|crescendo|{F.RELU}": _union_cell(F.VIEW, "crescendo", F.RELU, 4096, 50, 1, 1, 0, 0, 2, 2, 3)}}
        json.dump(mt, open(rs / "degeneracy_union" / "degeneracy_mt_cap4096_union.json", "w"))
    sampled = {"schema": "degeneracy_union_sampled/1", "meta": {"rung": 8192, "union": {"DEFINITION_SHA": DEF_SHA}},
               "cells": {f"{F.VIEW}|pasttense|{F.LORA}|t0": dict(_union_cell(F.VIEW, "pasttense", F.LORA, 8192, 20, 1, 1, 0, 0, 0, 0, 2), status="ok", trial=0,
                                                                 grades={"path": str(rs / "sampled" / "grades_sampled" / f"sdfbase_full22_{F.VIEW}_sampled_t0_cap8192" / "two_si_pasttense_aggregate.json")}),
                         f"{F.VIEW}|pasttense|{F.LORA}|t1": dict(_union_cell(F.VIEW, "pasttense", F.LORA, 8192, 20, 0, 1, 0, 0, 0, 0, 1), status="ok", trial=1,
                                                                 grades={"path": str(rs / "sampled" / "grades_sampled" / f"sdfbase_full22_{F.VIEW}_sampled_t1_cap8192" / "two_si_pasttense_aggregate.json")}),
                         f"{F.VIEW}|codeattack|{F.LORA}|t0": {"key": "x", "status": "pending", "what": "exit-rung store not landed"}},
               "pooled": {f"{F.VIEW}|pasttense|{F.LORA}": {"defended_collapse_bracket": {"n_defended": 8, "n_defended_any_degenerate": 3, "n_defended_all_degenerate": 1,
                                                                                         "n_goals": 10}, "trials": [0, 1], "cap": 8192}}}
    json.dump(sampled, open(rs / "sampled" / "report" / "analysis" / "static_degeneracy_union.json", "w"))
    petri = {"cells": {f"{F.VIEW}|petri|{F.LORA}": {"petri_run_key": "aaaa000000000001", "n_audits": 60, "n_responses": 700, "at_cap_definition": "stop_reason == max_tokens",
                                                     "response_level": {"at_cap": {"k": 140, "n": 700}, "at_cap_tok98": {"k": 150, "n": 700}, "loop": {"k": 3, "n": 700}},
                                                     "audit_level": {"any_at_cap": {"k": 30, "n": 60}, "any_loop": {"k": 2, "n": 60}}}}}
    json.dump(petri, open(rs / "degeneracy_petri" / "degeneracy_petri.json", "w"))
    return str(rs)


def test_measures_states_and_identities(tmp_path):
    rs = _write_files(tmp_path)
    seeds = F.synthetic_seeds()
    ad = DU.DegeneracyUnionAdapter(rs=rs, seeds=seeds, csv_dir=str(tmp_path / "nocsv"), objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), ad.objects.resolve(F.VIEW, F.RELU)])
    rep, ing, ing2 = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 4, [r.error for r in rep.results]
    assert len(ad.refusals) == 0, ad.refusals.items
    # the degeneration rows ingest even though the runs are not there yet? No: run_degeneration has a run FK → blocked until the runs exist.
    # The records themselves must be well-formed (no refusals); blocking is the expected state without the r2 adapter.
    assert ing.refused + ing2.refused == 0, ing.refusals + ing2.refusals
    rows = []
    from sisteer.registry import records as R
    for p in R.iter_record_files(os.path.join(root, "records", "adapters", "degeneracy_union")):
        for _, rec, _ in R.read_record_file(p):
            if rec["record_type"] == "run_degeneration":
                rows.append(rec["payload"])
    by = {}
    for r in rows:
        by.setdefault(r["run_id"], {})[(r["measure"], r["scope"])] = r
    # the greedy static cell: the two at-cap measures are two rows; finish_reason undefined → not_measured; the bracket carries the upper bound
    inst = seeds.instrument_row("static_greedy_4096_full22", "pasttense")
    rid = I.run_id(LC.CAMPAIGN, LC.producer_key_for_r2_row(F.VIEW, F.LORA, 1.0, None), F.LORA, "pasttense", inst["instrument_id"])
    m = by[rid]
    assert m[("at_token_cap_retokenized", "all")]["measurement_state"] == "measured" and m[("at_token_cap_retokenized", "all")]["numerator"] == 4
    assert m[("at_token_cap_finish_reason", "all")]["measurement_state"] == "not_measured" and m[("at_token_cap_finish_reason", "all")]["rate"] is None
    assert m[("union5", "all")]["aggregation"] == "union" and m[("union5", "all")]["rate"] == pytest.approx(6 / 20)
    assert m[("defended_degenerate", "defended_only")]["numerator"] == 2 and m[("defended_degenerate", "defended_only")]["denominator"] == 4
    assert m[("rep_ratio_loop", "all")]["denominator_unit"] == "draws" and m[("rep_ratio_loop", "all")]["threshold_params_json"]["rep_ratio_gt"] == 0.5
    assert m[("classifier", "all")]["definition_version"].endswith(DEF_SHA)
    # the multi-turn cell: unit replies
    inst_mt = seeds.instrument_row("multiturn_greedy_512_full22", "crescendo")
    rid_mt = I.run_id(LC.CAMPAIGN, LC.producer_key_for_r2_row(F.VIEW, F.LORA, 1.0, None), F.LORA, "crescendo", inst_mt["instrument_id"])
    assert by[rid_mt][("at_token_cap_finish_reason", "all")]["denominator_unit"] == "replies" and by[rid_mt][("at_token_cap_finish_reason", "all")]["numerator"] == 39
    # the declared cap-4,096 cells (two attacks) with no union file yet: every declared measure is not_measured, keyed by the grades root
    for attack, seed in (("crescendo", "multiturn_crescendo_cap4096_generation_config"), ("pair", "multiturn_pair_cap4096_generation_config")):
        inst4 = seeds.instrument_row(seed, attack)
        rid4 = I.run_id(LC.CAMPAIGN, "grades_synth_bu_synthview", F.RELU, attack, inst4["instrument_id"])
        assert set(by[rid4]) == set(DU.UNION_MEASURES)
        assert all(r["measurement_state"] == "not_measured" and "in flight" in (r["notes"] or "") for r in by[rid4].values())
    assert ad.stats["declared_mt_cap4096_runs"] == 2 and ad.stats["mt_cap4096_measured_runs"] == 0
    # the sampled trials attach to the sampled lane's runs (grades dir = producer key); the pending trial is not_measured
    inst_s = seeds.instrument_row("static_sampled_vendor_preset_4096_8192", "pasttense")
    rid_t0 = I.run_id(LC.CAMPAIGN, f"sdfbase_full22_{F.VIEW}_sampled_t0_cap8192", F.LORA, "pasttense", inst_s["instrument_id"])
    assert by[rid_t0][("union5", "all")]["numerator"] == 2
    inst_c = seeds.instrument_row("static_sampled_vendor_preset_4096_8192", "codeattack")
    rid_pending = I.run_id(LC.CAMPAIGN, f"sdfbase_full22_{F.VIEW}_sampled_t0_cap8192", F.LORA, "codeattack", inst_c["instrument_id"])
    assert all(r["measurement_state"] == "not_measured" for r in by[rid_pending].values())
    # the pooled bracket attaches to the r2 pooled run
    rid_pooled = I.run_id(LC.CAMPAIGN, LC.producer_key_for_r2_row(F.VIEW, F.LORA, 1.0, None), F.LORA, "pasttense", inst_s["instrument_id"])
    assert by[rid_pooled][("defended_degenerate", "defended_only")]["numerator"] == 3
    # petri: the producer stop reason and the re-tokenized cap are two rows; the union's other detectors are not_measured
    inst_p = seeds.instrument_row("petri_full22_as_run", "petri_pooled")
    rid_p = I.run_id(LC.CAMPAIGN, "aaaa000000000001", LC.petri_object_key(F.VIEW, F.LORA), "petri_pooled", inst_p["instrument_id"])
    p = by[rid_p]
    assert p[("at_token_cap_finish_reason", "all")]["numerator"] == 140 and p[("at_token_cap_retokenized", "all")]["numerator"] == 150
    assert p[("union5", "all")]["measurement_state"] == "not_measured" and p[("rep_ratio_loop", "all")]["denominator_unit"] == "replies"
    # the mt union file absent: its PRESENCE is the mt_cap4096 group's own input (`_mt_probe`, a listing of the union-file spellings, [] in
    # flight) and the group's marker carries the `$.entries` = [] row on it -- no row of any group is keyed on the file's path (the markers land
    # with their revisions, held here where no run exists yet, so read them from the records)
    states = _state_rows(root)
    probe = [p for p in states if p["json_path"] == "$.entries"]
    assert [(p["source_id"], p["input_uri"], p["found"], p["current_members_json"]) for p in probe] == \
        [("degeneracy_union:mt_cap4096", AB.listing_uri("degeneracy_union:mt_cap4096", DU.MT_PROBE_LABEL), 1, [])]
    assert not any(p["input_uri"].endswith("degeneracy_mt_cap4096_union.json") for p in states), "no group states the in-flight file by its path"
    # one comparison group per union file: the sampled rows cannot hold back the greedy full22 reading
    assert {ad.comparison_group(i) for i in ad.discover()} == {"full22", "mt_cap4096", "sampled", "petri"}   # the fixture writes no sampled_native file
    assert rep.groups_refreshed == 4


def test_mt_union_file_present_makes_the_declared_cell_measured(tmp_path):
    rs = _write_files(tmp_path, with_mt_file=True)
    seeds = F.synthetic_seeds()
    ad = DU.DegeneracyUnionAdapter(rs=rs, seeds=seeds, csv_dir=str(tmp_path / "nocsv"), objects=F.resolver(seeds, tmp_path))
    root = F.registry_root(tmp_path, seeds)
    rep, _, _ = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 4
    assert ad.stats["declared_mt_cap4096_runs"] == 2 and ad.stats["mt_cap4096_measured_runs"] == 1


def _mt_v2_cell(view, attack, cell, n_rows=200, n_goals=20):
    """A cell in the producer's `degeneracy_union_mt_cap4096/1` shape (turn_level / goal_level blocks)."""
    tl = {"denominator": "rows (target replies)",
          "degenerate_share": {"share": 0.02, "n": 4, "denominator": n_rows, "by_reason": ["classifier", "rep_ratio_loop"]},
          "by_reason": {"classifier": 1, "rep_ratio_loop": 2, "classifier+rep_ratio_loop": 1},
          "loop_share": {"share": 0.015, "n": 3, "denominator": n_rows, "threshold": 0.5},
          "classifier_share": {"share": 0.01, "n": 2, "denominator": n_rows, "by_reason": ["tail_loop"]},
          "zlib_collapse_share": {"share": 0.0, "n": 0, "denominator": n_rows, "median_ratio": 0.49},
          "contentless_share": {"share": 0.0, "n": 0, "denominator": n_rows},
          "at_cap_retokenized_share": {"share": 0.005, "n": 1, "denominator": n_rows, "defined": True, "cap": 4096, "fraction": 0.98},
          "at_cap_finish_reason_share": {"share": 0.005, "n": 1, "denominator": n_rows, "defined": True, "n_rows_with_finish_reason": n_rows},
          "at_cap_cross_tab": {"both": 1, "finish_reason_only": 0, "retokenized_only": 0, "neither": n_rows - 1, "agreement": 1.0},
          "at_cap_disagreements": {"n": 0, "denominator": n_rows},
          # the v2 file's DECOMPOSITION blocks are spelled {k, n, p} (count, denominator, share), unlike the detector blocks above
          "truncation_only_share": {"k": 1, "n": n_rows, "p": 1 / n_rows},
          "loop_without_cap_share": {"k": 3, "n": n_rows, "p": 3 / n_rows}}
    gl = {"denominator": "goals", "n_goals": n_goals, "canonical_at_cap": "at_cap_retokenized_any_turn",
          "judged": {"n_goals_judged": n_goals, "jailbroken": {"k": 15, "n": n_goals}, "defended": {"k": 5, "n": n_goals},
                     "defended_any_turn_degenerate": {"k": 2, "n": 5}}}
    return {"schema": "degeneracy_union_mt_cap4096/1", "key": f"{view}|{attack}|{cell}", "view": view, "attack": attack, "cell": cell, "cap": 4096,
            "status": "measured", "n_rows": n_rows, "n_goals": n_goals, "canonical_at_cap": "at_cap_retokenized", "turn_level": tl, "goal_level": gl,
            "lane_record": {"grades_root": "grades_synth_bu_synthview", "namespace": "cap4096"}}


def test_the_newest_versioned_mt_union_file_is_the_one_read(tmp_path):
    """The producer versions the cap-4,096 union in place (`…union.v2.json`, the old file renamed `*.superseded-by-v2`) and
    files v2 cells as turn_level / goal_level blocks."""
    rs = _write_files(tmp_path, with_mt_file=True)
    d = os.path.join(rs, "degeneracy_union")
    os.remove(os.path.join(d, "degeneracy_mt_cap4096_union.json"))
    v2 = {"schema": "degeneracy_union_mt_cap4096/1", "written_at": "2026-01-03T00:00:00+00:00", "cap": 4096,
          "meta": {"union": {"DEFINITION_SHA": DEF_SHA}}, "cells": {f"{F.VIEW}|crescendo|{F.RELU}": _mt_v2_cell(F.VIEW, "crescendo", F.RELU)}}
    json.dump(v2, open(os.path.join(d, "degeneracy_mt_cap4096_union.v2.json"), "w"))
    with open(os.path.join(d, "degeneracy_mt_cap4096_union.json.superseded-by-v2"), "w") as fh:
        fh.write("{}")
    norm = DU.normalise_mt_cell_legacy(v2["cells"][f"{F.VIEW}|crescendo|{F.RELU}"])
    assert norm["detectors"]["classifier"] == {"n": 2, "denominator": 200} and norm["detectors"]["rep_ratio_loop"]["threshold"] == 0.5
    assert norm["degenerate_share"]["n"] == 4 and norm["degenerate_share"]["by_reason"] == {"classifier": 1, "rep_ratio_loop": 2, "classifier+rep_ratio_loop": 1}
    assert norm["detectors"]["at_cap"]["at_cap_finish_reason"]["defined"] is True and norm["defended_collapse_bracket"]["n_defended"] == 5
    assert norm["defended_collapse_bracket"]["n_defended_any_degenerate"] == 2
    assert DU.mt_union_rel(rs) == "degeneracy_union/degeneracy_mt_cap4096_union.v2.json"
    seeds = F.synthetic_seeds()
    ad = DU.DegeneracyUnionAdapter(rs=rs, seeds=seeds, csv_dir=str(tmp_path / "nocsv"), objects=F.resolver(seeds, tmp_path))
    assert not any("superseded" in i.uri for i in ad.discover())
    # the bump renamed the group's union input: the older spellings are declared retired (the inputs-dropped guard), the rest not
    assert DU.mt_union_retired(rs) == ["degeneracy_union/degeneracy_mt_cap4096_union.json", "degeneracy_union/degeneracy_mt_cap4096_union.json.superseded-by-v2",
                                       "degeneracy_union/degeneracy_mt_cap4096_union.v1.json"]      # both spellings of version 1, and the copy
    assert [os.path.relpath(u, rs) for u in ad.inputs_retired("mt_cap4096")] == DU.mt_union_retired(rs)
    assert ad.inputs_retired("full22") == [] and ad.inputs_retired("sampled") == []
    root = F.registry_root(tmp_path, seeds)
    rep, _, _ = F.run_and_ingest(root, ad)
    assert rep.groups_refreshed == 4
    assert ad.stats["declared_mt_cap4096_runs"] == 2 and ad.stats["mt_cap4096_measured_runs"] == 1
    # the measured rows name the v2 file (they wait on multiturn_grades' runs here, so read the records, not the table)
    from sisteer.registry import records as R
    uris = set()
    for fp in R.iter_record_files(os.path.join(root, "records", "adapters", "degeneracy_union")):
        for _, rec, _ in R.read_record_file(fp):
            if rec["record_type"] == "run_degeneration" and rec["payload"].get("measurement_state") == "measured":
                uris.add(str(rec["payload"].get("source_uri")))
    assert any("union.v2.json" in u for u in uris) and not any("superseded" in u for u in uris)
    measured = {}
    for fp in R.iter_record_files(os.path.join(root, "records", "adapters", "degeneracy_union")):
        for _, rec, _ in R.read_record_file(fp):
            p = rec["payload"]
            if rec["record_type"] == "run_degeneration" and "union.v2.json" in str(p.get("source_uri")) and p.get("measurement_state") == "measured":
                measured[(p["measure"], p["scope"])] = (p["numerator"], p["denominator"], p["denominator_unit"])
    assert measured[("union5", "all")] == (4, 200, "replies") and measured[("classifier", "all")] == (2, 200, "replies")
    assert measured[("at_token_cap_retokenized", "all")] == (1, 200, "replies") and measured[("at_token_cap_finish_reason", "all")] == (1, 200, "replies")
    assert measured[("defended_degenerate", "defended_only")] == (2, 5, "items")
    # the decomposition (truncation that is not collapse / collapse that is not truncation) is filed as two measures of its own
    assert measured[("truncation_only", "all")] == (1, 200, "replies") and measured[("loop_without_cap", "all")] == (3, 200, "replies")
    assert norm["detectors"]["truncation_only"] == {"n": 1, "denominator": 200, "share": 1 / 200}


def _state_rows(root):
    out = []
    for p in R.iter_record_files(os.path.join(root, "records", "adapters", "degeneracy_union")):
        for _, rec, _ in R.read_record_file(p):
            if rec["record_type"] == "source_state":
                out.append(rec["payload"])
    return out


def test_the_cap4096_union_files_presence_is_its_own_groups_input_absent_then_present_re_reads_mt_cap4096_alone(tmp_path):
    """In flight: the probe lists nothing, mt_cap4096 reads (the declared set, nothing measured) and states
    `$.entries` = [] as ITS row; no row anywhere says the file is absent at the file's path. The file lands with full22's own files unchanged:
    the probe's fingerprint moves, so mt_cap4096 alone is planned for a read (full22 is not), its new revision states the probe's entry and
    the file's `$.cells` found = 1 under the same source id, and the greedy group has nothing to repair. Before this, full22's read stated the
    absence under its own source id and never re-read when the file joined the other group."""
    rs = _write_files(tmp_path)
    seeds = F.synthetic_seeds()
    ad = DU.DegeneracyUnionAdapter(rs=rs, seeds=seeds, csv_dir=str(tmp_path / "nocsv"), objects=F.resolver(seeds, tmp_path))
    sid_mt, sid_full = RUN.source_id_for(DU.NAME, "mt_cap4096"), RUN.source_id_for(DU.NAME, "full22")
    inputs1 = ad.discover()
    (probe1,) = [i for i in inputs1 if AB.is_listing_input(i)]
    assert ad.comparison_group(probe1) == "mt_cap4096" and probe1.entries == () and probe1.uri == AB.listing_uri(sid_mt, DU.MT_PROBE_LABEL)
    marks = {i.identity_uri: RUN.Watermark(RUN.source_id_for(DU.NAME, ad.comparison_group(i)), AB.watermark_of(i), None) for i in inputs1}
    root = F.registry_root(tmp_path, seeds, [ad.objects.resolve(F.VIEW, F.LORA), ad.objects.resolve(F.VIEW, F.RELU)])
    rep1, _, _ = F.run_and_ingest(root, ad, now="2026-10-04T10:00:00.000Z")
    assert rep1.groups_refreshed == 4 and all(r.accepted for r in rep1.results), [r.error for r in rep1.results]
    rev_mt1 = next(r.revision_id for r in rep1.results if r.group == "mt_cap4096")
    rows1 = _state_rows(root)
    assert [(p["source_id"], p["found"], p["current_members_json"]) for p in rows1 if p["json_path"] == "$.entries"] == [(sid_mt, 1, [])]
    assert not any(p["source_id"] == sid_full and "mt_cap4096" in p["input_uri"] for p in rows1), "full22 states nothing about the in-flight file"
    # the file lands; full22's own files stand
    mt = {"schema": "degeneracy_union_mt_cap4096/1", "meta": {"DEFINITION_SHA": DEF_SHA, "written_at": "2026-01-02T00:00:00+00:00"},
          "cells": {f"{F.VIEW}|crescendo|{F.RELU}": _union_cell(F.VIEW, "crescendo", F.RELU, 4096, 50, 1, 1, 0, 0, 2, 2, 3)}}
    json.dump(mt, open(os.path.join(rs, "degeneracy_union", "degeneracy_mt_cap4096_union.json"), "w"))
    ad2 = DU.DegeneracyUnionAdapter(rs=rs, seeds=seeds, csv_dir=str(tmp_path / "nocsv"), objects=F.resolver(seeds, tmp_path))   # the union spelling is resolved per instance
    inputs2 = ad2.discover()
    (probe2,) = [i for i in inputs2 if AB.is_listing_input(i)]
    assert probe2.uri == probe1.uri and probe2.entries == ("degeneracy_mt_cap4096_union.json",) and probe2.sha256 != probe1.sha256
    plan = RUN.plan_groups(ad2, inputs2, marks, AB.emission_version_of(ad2), None)
    assert plan.changed == {"mt_cap4096"}, plan.changed                     # the presence moved mt_cap4096's identity; full22's inputs are its own files, unchanged
    rep2, _, _ = F.run_and_ingest(root, ad2, now="2026-10-04T11:00:00.000Z")
    read = [r for r in rep2.results if r.accepted]
    assert [r.group for r in read] == ["mt_cap4096"] and read[0].revision_id != rev_mt1 and not read[0].reactivated, [(r.group, r.accepted, r.error) for r in rep2.results]
    assert not any(r.group == "full22" and r.accepted for r in rep2.results)
    rev_dir = os.path.join(root, "records", "adapters", DU.NAME, read[0].revision_id)
    rows2 = []
    for fn in os.listdir(rev_dir):
        for _, rec, _ in R.read_record_file(os.path.join(rev_dir, fn)):
            if rec["record_type"] == "source_state":
                rows2.append(rec["payload"])
    by = {(p["input_uri"], p["json_path"]): p for p in rows2}
    assert by[(probe2.uri, "$.entries")]["current_members_json"] == ["degeneracy_mt_cap4096_union.json"] and by[(probe2.uri, "$.entries")]["source_id"] == sid_mt
    mt_uri = os.path.join(rs, "degeneracy_union", "degeneracy_mt_cap4096_union.json")
    assert by[(mt_uri, "$.cells")]["found"] == 1 and by[(mt_uri, "$.cells")]["source_id"] == sid_mt and by[(mt_uri, "$.cells")]["current_members_json"] == ["n_cells=1"]
    assert ad2.stats["mt_cap4096_measured_runs"] == 1
