"""The canonical IFEval battery — the Inspect rung adapter at k = 2.

Pinned on synthetic IFEval stores built through the real producers: facts keyed ``(item, sample_idx)``; ``boundary`` iff
the sibling trial terminated below the source cap (a sub-class of truncated, counted and RE-DRAWN); ``truncated`` includes
boundary, ``loop`` is excluded; ``share_at_cap == share_truncated + share_loop`` over DRAWS (n = 1,082); the trigger at
``> 0.01`` of draws (10 of 1,082 does NOT promote, 11 does; 0.01 exactly does not); ``next_rung`` 2048 -> 4096 -> 8192 ->
ceiling; the ``rung_subset/2`` draws are EXACTLY the truncated draws; the transport check runs on the recorded hops; the
MERGED reading (``merge_draws``) decides rung 2 with denominator 1,082; ``redraw_records_by_draw`` keyed per draw; the
record carries the trigger provenance; a k = 1 GSM8K store still refuses as k = 1 and the k = 1 API is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisteer.capeval.tiers import cap_ladder as L
from sisteer.capeval.tiers import inspect_rung as R
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

import ifeval_synthetic_store as G

ROWS = G.matrix_rows()
ROW = G.row_for(ROWS, "gemma4e4b", "main", "constv2-mix-full__lora_r4_ce")   # the LoRA object: PSR left the matrix
N_DRAWS = G.N * G.K
HOPS_OK = {L.HOP_DRIVER: 7200.0, L.HOP_PROXY_UPSTREAM: 7200.0}


def _pairs(n_items, trials=(0, 1), start=0):
    return {(i, t) for i in range(start, start + n_items) for t in trials}


def test_the_instruments_are_named_and_the_gsm8k_one_is_k_1():
    # the GSM8K trigger is 1 % since the 2026-09-12 harmonisation (origin/main, merged M11); 3 % is the retired first-publication pair
    assert R.GSM8K_INSTRUMENT.k == 1 and R.GSM8K_INSTRUMENT.threshold == R.GSM8K_TRIGGER_THRESHOLD == 0.01 and R.GSM8K_INSTRUMENT.ceiling == 16384
    assert R.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED == 0.03 and R.GSM8K_TRIGGER_RULES_ADMITTED[R.GSM8K_TRIGGER_RULE_3PCT_RETIRED] == 0.03
    assert R.IFEVAL_INSTRUMENT.k == 2 and R.IFEVAL_INSTRUMENT.threshold == 0.01 and R.IFEVAL_INSTRUMENT.ladder == (2048, 4096, 8192)
    assert R.IFEVAL_INSTRUMENT.ceiling == 8192 and R.IFEVAL_INSTRUMENT.draw_level and R.IFEVAL_INSTRUMENT.subset_schema == "rung_subset/2"
    assert R.IFEVAL_INSTRUMENT.rule == IC.IFEVAL_TRIGGER_RULE and R.IFEVAL_INSTRUMENT.trigger_provenance == IC.IFEVAL_TRIGGER_PROVENANCE
    assert R.instrument_for("inspect/ifeval", threshold=0.05).threshold == 0.05
    with pytest.raises(R.InspectRungError, match="no ladder instrument"):
        R.instrument_for("inspect/math")
    assert R.ITEM_FACT_FIELDS_DRAW == (*R.ITEM_FACT_FIELDS, "boundary", "n_inst", *IC.IFEVAL_METRIC_FIELDS)


def test_ten_at_cap_draws_of_1082_do_not_trigger_and_eleven_do_and_the_shares_add_up(tmp_path):
    ten = G.build_store(tmp_path / "ten", ROW, at_cap=_pairs(5))                     # 10 draws = 0.924 %
    rec10 = json.loads((ten.parents[1] / IC.CAP_LADDER_FILENAME).read_text())
    assert rec10["exit"] == L.EXIT_BELOW_THRESHOLD and rec10["n_draws"] == N_DRAWS and rec10["n_at_cap"] == 10
    assert rec10["share_at_cap"] == round(10 / N_DRAWS, 6) and rec10["samples_per_item"] == 2
    eleven = G.build_store(tmp_path / "eleven", ROW, at_cap=_pairs(5) | {(5, 0)}, loops={(0, 0)})
    rec11, subset = G.classify(eleven.parents[1])
    assert rec11["exit"] == L.ESCALATE and rec11["next_cap"] == 4096 and rec11["n_at_cap"] == 11
    assert rec11["n_truncated"] == 10 and rec11["n_loop"] == 1 and rec11["n_boundary"] == 1     # (5, 0)'s sibling (5, 1) terminated below cap
    assert rec11["share_at_cap"] == round(rec11["share_truncated"] + rec11["share_loop"], 6)
    assert rec11["eligible_draws"] == [[G.IDS[i], t] for i in range(6) for t in (0, 1) if (i, t) != (0, 0) and (i, t) in (_pairs(5) | {(5, 0)})]
    assert rec11["loop_draws"] == [[G.IDS[0], 0]] and rec11["boundary_draws"] == [[G.IDS[5], 0]]
    assert rec11["trigger_provenance"] == "1 % raw at-cap of draws, per the IFEval ruling"
    assert rec11["rule"] == IC.IFEVAL_TRIGGER_RULE and rec11["threshold"] == 0.01 and rec11["ceiling"] == 8192 and rec11["ladder"] == [2048, 4096, 8192]
    assert rec11["instrument"] == "ifeval/cap_ladder" and rec11["rules"]["boundary"] == R.BOUNDARY_RULE_K2
    # the subset IS the truncated draws (boundary included, the loop excluded)
    assert subset["schema"] == "rung_subset/2" and subset["draws"] == rec11["eligible_draws"] and subset["n_draws"] == 10
    assert subset["n_boundary"] == 1 and subset["n_loop"] == 1 and subset["cap"] == 4096 and subset["source_cap"] == 2048
    assert IC.validate_item_subset_v2(subset) == subset
    # the boundary draw is IN the redraw set (P11) and the item-level lists are the distinct items
    assert [G.IDS[5], 0] in subset["draws"] and rec11[R.RECORD_FIELD_BOUNDARY_IDS] == [G.IDS[5]]
    assert rec11[R.RECORD_FIELD_ELIGIBLE_IDS] == [G.IDS[i] for i in range(6)]
    # the pure decision at the boundary of the rule, over draws
    assert R.decide(n_draws=N_DRAWS, n_at_cap=10, eligible_draws=[("x", 0)], cap=2048, hops=HOPS_OK, ladder=IC.IFEVAL_CAP_LADDER,
                    ceiling=8192, threshold=0.01)[0] == L.EXIT_BELOW_THRESHOLD
    assert R.decide(n_draws=N_DRAWS, n_at_cap=11, eligible_draws=[("x", 0)], cap=2048, hops=HOPS_OK, ladder=IC.IFEVAL_CAP_LADDER,
                    ceiling=8192, threshold=0.01)[:2] == (L.ESCALATE, 4096)
    assert R.decide(n_draws=100, n_at_cap=1, eligible_draws=[("x", 0)], cap=2048, hops=HOPS_OK, ladder=IC.IFEVAL_CAP_LADDER,
                    ceiling=8192, threshold=0.01)[0] == L.EXIT_BELOW_THRESHOLD           # 0.01 exactly does NOT promote
    assert R.decide(n_draws=N_DRAWS, n_at_cap=11, eligible_draws=[("x", 0)], cap=4096, hops=HOPS_OK, ladder=IC.IFEVAL_CAP_LADDER,
                    ceiling=8192, threshold=0.01)[:2] == (L.ESCALATE, 8192)
    assert R.decide(n_draws=N_DRAWS, n_at_cap=11, eligible_draws=[("x", 0)], cap=8192, hops=HOPS_OK, ladder=IC.IFEVAL_CAP_LADDER,
                    ceiling=8192, threshold=0.01)[0] == L.EXIT_CEILING
    assert R.decide(n_draws=N_DRAWS, n_at_cap=11, eligible_draws=[], cap=2048, hops=HOPS_OK, ladder=IC.IFEVAL_CAP_LADDER,
                    ceiling=8192, threshold=0.01)[0] == R.EXIT_TRIGGERED_NO_ELIGIBLE
    slow = R.decide(n_draws=N_DRAWS, n_at_cap=11, eligible_draws=[("x", 0)], cap=2048, hops={**HOPS_OK, L.HOP_PROXY_UPSTREAM: 600.0},
                    ladder=IC.IFEVAL_CAP_LADDER, ceiling=8192, threshold=0.01)
    assert slow[0] == L.EXIT_TRANSPORT_REFUSED and slow[2].required_s == 4096 / 4.0 * 1.5


def test_the_k1_decide_and_load_store_are_unchanged():
    # the k = 1 path at the harmonised 1 % trigger (2 of 200 = 1.0 % does not trigger; 3 does), and the retired 3 % pair
    # still decides the retired way when a reader passes it explicitly (a rung identity / subset filed under it)
    assert R.decide(n=200, n_at_cap=2, eligible_ids=["x"], cap=8192, hops=HOPS_OK)[0] == L.EXIT_BELOW_THRESHOLD
    assert R.decide(n=200, n_at_cap=3, eligible_ids=["x"], cap=8192, hops=HOPS_OK)[:2] == (L.ESCALATE, 16384)
    retired = R.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED
    assert R.decide(n=200, n_at_cap=6, eligible_ids=["x"], cap=8192, hops=HOPS_OK, threshold=retired)[0] == L.EXIT_BELOW_THRESHOLD
    assert R.decide(n=200, n_at_cap=7, eligible_ids=["x"], cap=8192, hops=HOPS_OK, threshold=retired)[:2] == (L.ESCALATE, 16384)


def test_a_k2_store_refuses_under_the_gsm8k_instrument_and_a_k1_store_under_ifeval(tmp_path):
    cd = G.build_store(tmp_path / "k2", ROW, ladder=False)
    # the k = 1 instrument's refusal (the GSM8K test pins its wording) fires on a k = 2 store read as k = 1
    with pytest.raises(R.InspectRungError, match="k = 1"):
        R.load_store(cd.parents[1], task=IC.IFEVAL_TASK, samples_per_item_expected=1)
    facts = R.load_store(cd.parents[1], task=IC.IFEVAL_TASK)
    assert facts.samples_per_item == 2 and facts.n == 541 and facts.n_draws == 1082
    with pytest.raises(R.InspectRungError, match="k = 3"):
        R.load_store(cd.parents[1], task=IC.IFEVAL_TASK, samples_per_item_expected=3)
    # and a GSM8K-task read of an IFEval store refuses on the identity's cap map — the two instruments never mix
    with pytest.raises(R.InspectRungError, match="max_tokens_by_task"):
        R.load_store(cd.parents[1], task=C.GSM8K_STANDARD_TASK)


def test_the_rung_and_the_merged_reading_decide_rung_2_with_denominator_1082(tmp_path):
    # 12 items x 2 trials at cap = 24 draws (2.2 %); 2 loops; the rung leaves 14 draws at cap again (1.3 % of 1,082 -> rung 8192)
    at = _pairs(12)
    esc = G.build_escalated_cell(tmp_path, ROW, at_cap=at, loops={(0, 0), (0, 1)},
                                 rung_still_at_cap={(i, t) for i in range(1, 8) for t in (0, 1)}, rung_boundary={(9, 0)},
                                 rung8192=True, rung8192_still_at_cap={(1, 0)})
    assert esc["record"]["exit"] == L.ESCALATE and esc["subset"]["n_draws"] == 22
    rung_root = esc["rung_cell_dir"].parents[1]
    rec2 = esc["rung_record"]
    assert rec2["source"]["is_rung"] and rec2["source"]["n_redrawn"] == 22 and rec2["cap"] == 4096
    outcomes = rec2["controls"][IC.IFEVAL_TASK]["redraw_outcomes"]
    assert outcomes == {"boundary_landing": 1, "extended": 7, "still_at_cap": 14}
    # the rung's OWN population is its 22 draws; the rung-2 decision is over the MERGED reading (1,082 draws)
    assert rec2["n_draws"] == 22 and rec2["n_at_cap"] == 14
    src = completion.read_task_outcomes(esc["source_cell_dir"], IC.IFEVAL_TASK)["items"]
    rung = completion.read_task_outcomes(esc["rung_cell_dir"], IC.IFEVAL_TASK)["items"]
    merged = R.merge_draws(src, rung, esc["subset"]["draws"])
    assert len(merged) == 1082 and sum(1 for m in merged if m["draw"] == "rung") == 22
    n_at_merged = sum(1 for m in merged if m["at_cap"])
    assert n_at_merged == 14 + 2            # the 14 still at cap + the 2 loops the rung never re-drew
    assert R.decide(n_draws=1082, n_at_cap=n_at_merged, eligible_draws=[("x", 0)] * 14, cap=4096, hops=HOPS_OK,
                    ladder=IC.IFEVAL_CAP_LADDER, ceiling=8192, threshold=0.01)[:2] == (L.ESCALATE, 8192)
    assert rec2["exit"] == L.ESCALATE and rec2["next_cap"] == 8192 and esc["subset8192"]["n_draws"] == 14
    assert esc["rung8192_cell_dir"] is not None
    rec3, sub3 = G.classify(esc["rung8192_cell_dir"].parents[1])
    assert rec3["exit"] == L.EXIT_CEILING and sub3 is None and rec3["n_at_cap"] == 1     # the residual, no rung above
    # seam 4 (M11): the adapter reads the SOURCE from `source_out_root` when given — the same record as through the recorded
    # absolute root (in place they coincide); a pass-1 store given one refuses (it binds to no source)
    src_root = esc["source_cell_dir"].parents[1]
    rec2b, sub2b = R.classify_store(rung_root, task=IC.IFEVAL_TASK, texts=G.store_texts(esc["rung_cell_dir"]),
                                    source_texts=G.store_texts(esc["source_cell_dir"]), source_out_root=src_root)
    assert rec2b == {k: v for k, v in rec2.items() if k != IC.LADDER_DECISION_KEY} and sub2b == esc["subset8192"]
    with pytest.raises(R.InspectRungError, match="not a rung"):
        R.classify_store(src_root, task=IC.IFEVAL_TASK, texts=G.store_texts(esc["source_cell_dir"]), source_out_root=src_root)
    # merge_draws refusals: a rung draw outside the subset, a listed draw the rung missed, a listed draw the source lacks
    with pytest.raises(R.InspectRungError, match="outside its draw subset"):
        R.merge_draws(src, rung + [{**rung[0], "item_id": G.IDS[300]}], esc["subset"]["draws"])
    with pytest.raises(R.InspectRungError, match="missed"):
        R.merge_draws(src, rung[:-1], esc["subset"]["draws"])
    with pytest.raises(R.InspectRungError, match="not source draws"):
        R.merge_draws([s for s in src if s["item_id"] != G.IDS[1]], rung, esc["subset"]["draws"])
    # redraw records are per DRAW; the k = 1 reader refuses a k = 2 rung by name
    rf, sf = R.load_store(rung_root, task=IC.IFEVAL_TASK), R.load_store(esc["source_root"], task=IC.IFEVAL_TASK)
    by_draw = R.redraw_records_by_draw(rf, sf, rung_texts=G.store_texts(esc["rung_cell_dir"]), source_texts=G.store_texts(esc["source_cell_dir"]))
    assert set(by_draw) == set(IC.draws_of_subset(esc["subset"])) and by_draw[(G.IDS[9], 0)]["outcome"] == "boundary_landing"
    with pytest.raises(R.InspectRungError, match="redraw_records_by_draw"):
        R.redraw_records(rf, sf, rung_texts={})


def test_a_rung_whose_draws_are_not_its_subset_refuses(tmp_path):
    esc = G.build_escalated_cell(tmp_path, ROW, at_cap=_pairs(12), rung_still_at_cap=set())
    rung_cd = esc["rung_cell_dir"]
    draws = IC.draws_of_subset(esc["subset"])
    # a rung store over one draw FEWER than the subset, under the same subset block
    other = G.build_store(tmp_path / "other", ROW, at_cap=(), cap=4096, draws=draws[:-1],
                          rung_source=completion.read_store_manifest(esc["source_cell_dir"]), subset_doc=esc["subset"],
                          subset_path=str(esc["subset_path"]), ladder=False, n_expected_draws=len(draws))
    with pytest.raises(R.InspectRungError, match="not the recorded subset"):
        G.classify(other.parents[1])
    assert rung_cd is not None


def test_classify_items_facts_carry_the_ifeval_values_and_the_boundary_flag(tmp_path):
    cd = G.build_store(tmp_path / "f", ROW, at_cap={(3, 1)}, ladder=False)
    facts = R.load_store(cd.parents[1], task=IC.IFEVAL_TASK)
    items = R.classify_items(facts, G.store_texts(cd))
    assert len(items) == 1082 and all(tuple(it) == R.ITEM_FACT_FIELDS_DRAW for it in items)
    by = {(it["item_id"], it["sample_idx"]): it for it in items}
    hit = by[(G.IDS[3], 1)]
    assert hit["at_cap"] and hit["kind"] == "truncated" and hit["boundary"] is True and hit["n_inst"] == G.N_INST[3]
    assert isinstance(hit["prompt_level_strict"], bool) and hit["inst_level_strict"] <= hit["n_inst"]
    assert by[(G.IDS[3], 0)]["boundary"] is False and by[(G.IDS[3], 0)]["kind"] is None
    # an at-cap draw with no text refuses (a loop is never assumed absent)
    texts = G.store_texts(cd)
    texts.pop((G.IDS[3], 1))
    with pytest.raises(R.InspectRungError, match="no emitted text"):
        R.classify_items(facts, texts)


@pytest.mark.skipif(__import__("importlib").util.find_spec("inspect_ai") is None, reason="needs inspect_ai")
def test_store_texts_reads_both_sample_logs_through_the_producers_log_selector(tmp_path):
    cd = G.build_store(tmp_path / "logs", ROW, at_cap={(7, 0), (8, 1)}, ladder=False)
    G.write_inspect_logs(cd)
    facts = R.load_store(cd.parents[1], task=IC.IFEVAL_TASK)
    texts = R.store_texts(facts)
    assert len(texts) == 1082 and texts[(G.IDS[7], 0)] == G.TRUNC_TEXT and texts[(G.IDS[8], 1)] == G.TRUNC_TEXT
    assert texts[(G.IDS[0], 1)].startswith("Answer 0/1")
    record, subset = R.classify_store(cd.parents[1], task=IC.IFEVAL_TASK)           # the default text source: the logs
    assert record["n_at_cap"] == 2 and record["exit"] == L.EXIT_BELOW_THRESHOLD
    # a torn store: the log's stop_reason disagrees with the outcome record
    G.write_inspect_log(cd, name="torn", sample_idx=0, texts={})
    outcomes = completion.read_task_outcomes(cd, IC.IFEVAL_TASK)
    for it in outcomes["items"]:
        if (it["item_id"], it["sample_idx"]) == (G.IDS[7], 0):
            it["stop_reason"], it["at_cap"] = "stop", False
    completion.write_task_outcomes(cd, IC.IFEVAL_TASK, outcomes)
    with pytest.raises(R.InspectRungError, match="torn"):
        R.store_texts(R.load_store(cd.parents[1], task=IC.IFEVAL_TASK))
