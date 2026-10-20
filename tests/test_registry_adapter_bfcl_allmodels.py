"""The `bfcl_allmodels` reader on the SYNTHETIC appendix mirror of `_bcbw_allmodels_fixtures`: the same store shape as the BigCodeBench-200
appendix reader's (one store holds both tasks), read from ITS OWN root for ITS OWN task. What it pins beyond the shared reading (tested in full on the
BigCodeBench-200 side): the BFCL-200 instrument bound to the served model, its own source and eval, the three numbers from the BFCL outcome record,
the `main` canary row alias (its fitted key through the roster's main selection) and the pin-key object route on this instrument, that llama8b — which
runs no BFCL row — files nothing here (no store, no launch row, no refusal), that the two appendix readers mint DISTINCT run ids over the same producer
keys, and the gemma protocol row on the BFCL rows."""
from __future__ import annotations

import os

import pytest

import _bcbw_allmodels_fixtures as Y
from sisteer.registry import identity as I
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import bfcl_allmodels as M
from sisteer.registry.adapters import bfcl_seedrep as SR
from sisteer.registry.adapters import bigcodebench_allmodels as BCA
from sisteer.registry.adapters import lineage_common as LC

EVAL = "bfcl"
MK, TAG, TAG_P, TAG_MAIN, SEL, SEL_P = Y.MK, Y.TAG, Y.TAG_P, Y.TAG_MAIN, Y.SEL, Y.SEL_P
NONE, RELU, PROMPTED, COMPOSED = Y.NONE, Y.RELU, Y.PROMPTED, Y.COMPOSED


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _mirror(tmp_path, seeds, **kw):
    return Y.mirror(tmp_path, seeds, eval_id=EVAL, **kw)


def _adapter(fx, seeds) -> M.BfclAllmodelsAdapter:
    return Y.adapter(M.BfclAllmodelsAdapter, fx, seeds)


def test_the_bfcl_rows_file_their_own_instrument_source_and_three_numbers_with_the_canary_alias(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    names = [os.path.basename(i.uri) for i in ad.discover()]
    assert names.count(M.OUTCOMES_FILE) == 15 and not any(n.startswith("inspect__bigcodebench") for n in names) and f"{M.THREE_NUMBERS_FILE}#{TAG}" in names
    assert M.THREE_NUMBERS_FILE == "three_numbers_bfcl_allmodels.json" and f"gate_{TAG_MAIN}_{NONE}_bcbw-{TAG_MAIN}-{Y.SHA7}.json#{TAG}" in names
    bt, facts = Y.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    r = Y.run_of(bt, RELU)
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="gemma4e4b_it")
    assert r["lane_source_legacy"] == M.SOURCE == "bfcl_allmodels" and r["eval_id"] == "bfcl" and r["instrument_id"] == inst["instrument_id"] and r["item_set_id"] == "bfcl_200"
    assert r["model_id"] == "gemma4e4b_it" and r["harness"] == "bfcl_eval" and r["scoring_rule"] == "ast_match" and r["token_cap_tokens"] == 1024 == M.CAP and r["thinking_mode"] == "off"
    assert r["decode_sent_json"] == {"temperature": 0.001, "top_p": 1.0, "top_k": 0} and r["decode_source"] == "greedy" and r["instrument_match"] == "match"
    assert r["extra"]["decode_policy_ref_legacy"] == "gemma4e4b_it|bfcl|off|protocol@v1" and r["decode_seed_rule"] == "none" and r["decode_policy_match"] == "no_policy"
    assert r["extra"]["sampling_clamp_override"] == {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"} and r["extra"]["item_set_pin"]["of"] == 3981
    assert r["extra"]["object_route"]["route"] == "harness_roster_main_selection" and r["extra"]["appendix_family"]["model_key"] == MK and r["object_id"] == Y.OBJ
    assert r["extra"]["tag_legacy_original"] == TAG_MAIN and r["extra"]["object_route"]["staged_id"] == Y.STAGED
    c = Y.counts(fx["specs"][RELU])
    value, bound = Y.metrics_of(bt, r["run_id"], M.recount_computation_id())
    assert (value["k"], value["n"]) == (c["correct"], c["n_scored"]) == (101, 200) and bound["producer_metric_spelling"] == "bound"
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"]) == (65, 200) and deg["rep_ratio_loop"]["measurement_state"] == "not_measured"
    assert value["computation_id"] == I.computation_id(M.RECOUNT_ALGORITHM, M.RECOUNT_VERSION, M.RECOUNT_SEED_PROVENANCE) and M.RECOUNT_ALGORITHM.endswith("bfcl_allmodels.recount")
    # the canary: the s1 tag's none on THIS instrument, the seed rows' anchor; the composed row on the paired tag's own none
    n = Y.run_at(bt, TAG, NONE)
    assert n["extra"]["tag_legacy_original"] == TAG_MAIN and n["view_legacy"] == TAG and n["eval_id"] == "bfcl" and n["instrument_id"] == inst["instrument_id"]
    contrasts = {cn["run_id"]: cn for cn in bt["run_contrast"]}
    none_p = {(x["view_legacy"], x["campaign_cell_id_legacy"]): x for x in bt["run"]}[(TAG_P, NONE)]
    assert {contrasts[Y.run_of(bt, k)["run_id"]]["anchor_run_id"] for k in (RELU, PROMPTED)} == {n["run_id"]} and contrasts[Y.run_of(bt, COMPOSED)["run_id"]]["anchor_run_id"] == none_p["run_id"]


def test_llama_runs_no_bfcl_row_so_nothing_is_filed_or_refused_for_it(tmp_path, seeds):
    """No llama store is mirrored under the BFCL root and the BFCL launch set names no llama row (the fixture writes llama's `none` on the
    BigCodeBench eval alone) -- the reader files nothing for the family: no group, no run, no refusal, no not-run entry."""
    fx = _mirror(tmp_path, seeds, llama_none=True)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    assert {ad.comparison_group(i) for i in inputs} == {TAG} and not any(Y.LLAMA_MK in os.path.relpath(i.uri, fx["root"]) for i in inputs)
    assert not os.path.isdir(os.path.join(fx["root"], B.STORES_HEAD, Y.LLAMA_ROW_DIR))
    bt, facts = Y.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["n_not_run"] == 0 and {r["model_id"] for r in bt["run"]} == {"gemma4e4b_it"}
    ls = [s for s in ad.source_states(Y.X.ctx(), inputs) if os.path.basename(s.input_uri).startswith(B.LAUNCH_SET_FILE)]
    assert len(ls) == 1 and not any(Y.LLAMA_MK in m for m in ls[0].current_members) and len(ls[0].current_members) == 5
    # the same fixture on the BigCodeBench eval DOES carry llama's none: the two roots are different mirrors of the appendix run
    fx_b = Y.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "bcb"), llama_none=True)
    ad_b = Y.adapter(BCA.BigcodebenchAllmodelsAdapter, fx_b, seeds)
    assert {ad_b.comparison_group(i) for i in ad_b.discover()} == {TAG, Y.LLAMA_TAG}


def test_the_two_appendix_readers_mint_distinct_run_ids_over_the_same_producer_keys(tmp_path, seeds):
    fx_f = _mirror(tmp_path, seeds, root=str(tmp_path / "bfcl_allmodels"))
    fx_c = Y.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "bigcodebench_allmodels"))
    bt_f, _ff = Y.read(_adapter(fx_f, seeds))
    bt_c, _fc = Y.read(Y.adapter(BCA.BigcodebenchAllmodelsAdapter, fx_c, seeds))
    ids_f = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt_f["run"]}
    ids_c = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt_c["run"]}
    assert set(ids_f) == set(ids_c) == {(TAG, NONE), (TAG, RELU), (TAG, PROMPTED), (TAG_P, COMPOSED), (TAG_P, NONE)}
    for k in ids_f:
        assert ids_f[k]["run_id"] != ids_c[k]["run_id"] and ids_f[k]["producer_run_key_legacy"] == ids_c[k]["producer_run_key_legacy"] and ids_f[k]["object_id"] == ids_c[k]["object_id"]
        assert ids_f[k]["instrument_id"] != ids_c[k]["instrument_id"] and ids_f[k]["eval_id"] == "bfcl" and ids_c[k]["eval_id"] == "bigcodebench"
        assert ids_f[k]["extra"].get("tag_legacy_original") == ids_c[k]["extra"].get("tag_legacy_original")
    inst = str(seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="gemma4e4b_it")["instrument_id"])
    assert ids_f[(TAG, NONE)]["run_id"] == M.run_id_for(TAG, fx_f["keys"][NONE], NONE, inst) and inst == str(seeds.instrument_row(SR.INSTRUMENT_SEED, SR.EVAL_ID, model_id="qwen35_4b_it")["instrument_id"]), \
        "the protocol rows agree across the served models, so WS-B0 binds one shared instrument row"


def test_the_roster_set_is_read_on_this_task_too(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, roster_set=True)
    bt, facts = Y.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    assert Y.run_at(bt, TAG, RELU)["extra"]["launched_roster"]["roster_file"] == Y.ROSTER_CANARY_FILE and Y.run_at(bt, TAG_P, COMPOSED)["extra"]["launched_roster"]["tree_sha"] == Y.TREE_S1
    assert Y.run_at(bt, TAG, RELU)["extra"]["launched_pins_record"].endswith(Y.PINS_CANARY_FILE)


def test_a_store_without_the_tool_call_route_is_refused_on_this_task(tmp_path, seeds):
    """BFCL IS tool calls: a store whose manifest does not attest the route (llama8b's, served for BigCodeBench alone) refuses by name here, while the
    BigCodeBench reader records the flag and files it."""
    import _bcbw_seedrep_fixtures as X
    fx = _mirror(tmp_path, seeds)
    # rewrite the gemma canary row's `none` manifest without the route: the same store on THIS task must refuse
    import json, glob
    p = glob.glob(os.path.join(fx["root"], B.STORES_HEAD, Y.ROW_DIR_MAIN, NONE, B.THINKING_DIR, "*", "cells", "*", "manifest.json"))[0]
    m = json.load(open(p)); m["tinycap"]["served"]["tool_calls"] = False; m["route"]["tool_calls"] = False
    X.write(p, m)
    bt, facts = Y.read(_adapter(fx, seeds))
    assert "do not attest the tool-call route" in Y.refused(facts, TAG, NONE) and facts["runs"] == 4 and M.SPEC.tool_calls_required is True and SR.SPEC.tool_calls_required is True


def test_a_pin_key_the_harness_pins_do_not_carry_refuses_the_row_on_this_task_too(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, canary_layout="s1", harness_pins_without=(Y.STAGED,))
    bt, facts = Y.read(_adapter(fx, seeds))
    assert f"carry no object_sha256 under {MK}.{Y.STAGED}" in Y.refused(facts, TAG, RELU) and facts["runs"] == 4 and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, PROMPTED, COMPOSED}
    # on the canary row the same gap surfaces from the roster hop: the main selection's staged id is not pinned
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "canary"), harness_pins_without=(Y.STAGED,))
    bt2, facts2 = Y.read(_adapter(fx2, seeds))
    assert f"pins name no object_sha256 for staged id '{Y.STAGED}'" in Y.refused(facts2, TAG, RELU) and facts2["runs"] == 4
    # the composed key the harness pins lack resolves through the store's tree pins by the launched sha on this task too
    fx3 = _mirror(tmp_path, seeds, root=str(tmp_path / "composed"), harness_pins_without=(Y.STAGED_P,))
    bt3, facts3 = Y.read(_adapter(fx3, seeds))
    assert facts3["n_refusals"] == 0 and facts3["runs"] == 5, facts3["refusals"]
    c = Y.run_of(bt3, COMPOSED)
    assert c["object_id"] == Y.OBJ_P and c["extra"]["launched_pins_key"] == Y.STAGED_P and c["extra"]["pins_key_rule"] == B.PINS_KEY_RULE and c["extra"]["object_route"]["route"] == B.OBJECT_ROUTE_LAUNCHED_PINS_BY_SHA


def test_the_reader_is_registered_and_declares_its_source_owner_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.BfclAllmodelsAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "bfcl_allmodels"
    assert AB.emission_version_of(M.BfclAllmodelsAdapter) == "2" and AB.emission_version_of(SR.BfclSeedrepAdapter) == "3" and inst.rng_scope == "none" and inst.roots == (M.ROOT_DEFAULT,)
    assert M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bfcl_allmodels")) and M.ROOT_DEFAULT != BCA.ROOT_DEFAULT != SR.ROOT_DEFAULT and M.LADDER == (1024,)
    assert M.EVAL_ID == "bfcl" and M.INSTRUMENT_SEED == "bfcl_200_protocol_T0001_1024" and M.TASK == "inspect/bfcl_200" and M.SPEC.record_task == "bfcl_200" and M.SPEC.models == B.APPENDIX_MODEL_KEYS
    assert M.PROTOCOL_TEMPERATURE == 0.001 and M.SPEC.clamp_override is True and M.ITEM_LABEL_AT_CAP_UNRESOLVED == IC.ITEM_LABEL_AT_CAP_UNRESOLVED
