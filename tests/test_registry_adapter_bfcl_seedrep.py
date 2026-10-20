"""The `bfcl_seedrep` reader on the SYNTHETIC mirror of `_bcbw_seedrep_fixtures`: the same store shape as the BigCodeBench-200 reader's (one
store holds both tasks), read from ITS OWN root for ITS OWN task. What it pins beyond the shared reading (tested in full on the BigCodeBench-200
side): the BFCL-200 instrument (`bfcl_200_protocol_T0001_1024`: cap 1,024, protocol T 0.001 executed under the attested clamp override, AST
match), its own source and eval, the three numbers reproduced from the BFCL outcome record, the sha and three-number refusals by name, the
missing anchor filing without a contrast, that the two readers mint DISTINCT run ids over the same producer keys, and that the paired tag's own
`none` (the rule of 2026-09-23) files as the paired tag's anchor on THIS instrument with the composed row paired against it."""
from __future__ import annotations

import hashlib
import os

import pytest

import _bcbw_seedrep_fixtures as X
from sisteer.registry import identity as I
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import bfcl_seedrep as M
from sisteer.registry.adapters import bigcodebench_seedrep as BC
from sisteer.registry.adapters import lineage_common as LC

EVAL = "bfcl"
TAG, TAG_P, SEL, SEL_P = X.TAG, X.TAG_P, X.SEL, X.SEL_P
NONE, RELU, PROMPTED, ARDITI, COMPOSED = X.NONE, X.RELU, X.PROMPTED, X.ARDITI, X.COMPOSED


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _mirror(tmp_path, seeds, **kw):
    return X.mirror(tmp_path, seeds, eval_id=EVAL, **kw)


def _adapter(fx, seeds) -> M.BfclSeedrepAdapter:
    return X.adapter(M.BfclSeedrepAdapter, fx, seeds)


def _metrics(bt, rid):
    return X.metrics_of(bt, rid, M.recount_computation_id())


def test_the_bfcl_rows_file_their_own_instrument_source_and_three_numbers(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    names = [os.path.basename(i.uri) for i in ad.discover()]
    assert names.count(M.OUTCOMES_FILE) == 15 and not any(n.startswith("inspect__bigcodebench") for n in names) and f"{M.THREE_NUMBERS_FILE}#{TAG}" in names
    bt, facts = X.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5, facts["refusals"]
    r = X.run_of(bt, RELU)
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")
    assert r["lane_source_legacy"] == M.SOURCE == "bfcl_seedrep" and r["eval_id"] == "bfcl" and r["instrument_id"] == inst["instrument_id"] and r["item_set_id"] == "bfcl_200"
    assert r["harness"] == "bfcl_eval" and r["scoring_rule"] == "ast_match" and r["token_cap_tokens"] == 1024 == M.CAP and r["thinking_mode"] == "off"
    assert r["decode_sent_json"] == {"temperature": 0.001, "top_p": 1.0, "top_k": 0} and r["decode_temperature_sent"] == 0.001 and r["decode_source"] == "greedy"
    assert r["instrument_match"] == "match" and r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|bfcl|off|protocol@v1"
    assert r["extra"]["sampling_clamp_override"] == {"from": 0.01, "to": 0.001, "why": "BFCL protocol (D13)"} and r["extra"]["tool_calls"] is True
    assert r["extra"]["item_set_pin"]["of"] == 3981 and r["seed_base"] is None and r["trial_index"] == 0 and r["decode_seed_rule"] == "none" and r["decode_policy_match"] == "no_policy"
    c = X.counts(fx["specs"][RELU])
    assert (c["n_scored"], c["correct"], c["n_at_cap"]) == (200, 101, 65)
    value, bound = _metrics(bt, r["run_id"])
    assert value["value"] == 101 / 200 and (value["k"], value["n"]) == (101, 200) and bound["value"] == 101 / 200 and bound["producer_metric_spelling"] == "bound"
    assert (bound["ci_lo"], bound["ci_hi"]) == (101 / 200, 101 / 200), "every at-cap item is scored, so the bound's interval closes on the value"
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"]) == (65, 200) and deg["rep_ratio_loop"]["measurement_state"] == "not_measured"
    items = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == r["run_id"]}
    assert len(items) == 200 and all(i["label"] is None for i in items.values()) and sum(1 for i in items.values() if i["extra"]["scored_at_cap"]) == 65
    flags = {f["item_id"]: f for f in bt["item_flags"] if f["run_id"] == r["run_id"]}
    assert sum(f["at_token_cap"] for f in flags.values()) == 65 and {f["finish_reason"] for f in flags.values() if f["at_token_cap"] == 0} == {"tool_calls"}
    assert value["computation_id"] == I.computation_id(M.RECOUNT_ALGORITHM, M.RECOUNT_VERSION, M.RECOUNT_SEED_PROVENANCE) and M.RECOUNT_ALGORITHM.endswith("bfcl_seedrep.recount")
    # the contrasts: every row against the seed tag's none, the composed row across standups, the Arditi row's declared role
    contrasts = {cn["run_id"]: cn for cn in bt["run_contrast"]}
    none_id = X.run_of(bt, NONE)["run_id"]
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)} and all(cn["anchor_run_id"] == none_id for cn in contrasts.values())
    assert contrasts[X.run_of(bt, COMPOSED)["run_id"]]["extra"]["cross_standup"] is True and contrasts[X.run_of(bt, ARDITI)["run_id"]]["extra"]["declared_role"] == M.ARDITI_ROLE
    assert contrasts[X.run_of(bt, COMPOSED)["run_id"]]["anchor_kind"] == "sibling_none", "the D65 fallback is another standup's none"
    assert all(contrasts[X.run_of(bt, k)["run_id"]]["anchor_kind"] == "same_standup_none" for k in (RELU, PROMPTED, ARDITI))
    assert X.run_of(bt, ARDITI)["strength_coefficient"] == 0.3 and X.run_of(bt, ARDITI)["transfer_kind"] == "instruct_direct"


def test_the_two_readers_mint_distinct_run_ids_over_the_same_producer_keys(tmp_path, seeds):
    fx_b = _mirror(tmp_path, seeds, root=str(tmp_path / "bfcl_seedrep"))
    fx_c = X.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "bigcodebench_seedrep"))
    bt_b, _fb = X.read(_adapter(fx_b, seeds))
    bt_c, _fc = X.read(X.adapter(BC.BigcodebenchSeedrepAdapter, fx_c, seeds))
    ids_b = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt_b["run"]}
    ids_c = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt_c["run"]}
    assert set(ids_b) == set(ids_c) and len(ids_b) == 5
    for k in ids_b:
        assert ids_b[k]["run_id"] != ids_c[k]["run_id"] and ids_b[k]["producer_run_key_legacy"] == ids_c[k]["producer_run_key_legacy"] and ids_b[k]["object_id"] == ids_c[k]["object_id"]
        assert ids_b[k]["instrument_id"] != ids_c[k]["instrument_id"] and ids_b[k]["eval_id"] == "bfcl" and ids_c[k]["eval_id"] == "bigcodebench"
    inst = str(seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")["instrument_id"])
    assert ids_b[(TAG, RELU)]["run_id"] == M.run_id_for(TAG, fx_b["keys"][RELU], RELU, inst)


def test_a_store_attesting_another_sha_than_the_pins_is_refused_and_the_siblings_file(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = _mirror(tmp_path, seeds, attest_sha={COMPOSED: other})
    bt, facts = X.read(_adapter(fx, seeds))
    why = X.refused(facts, TAG_P, COMPOSED)
    assert "fit_provenance.object_sha" in why and "no prefix of the pinned object" in why and len(bt["run"]) == 4
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, RELU, PROMPTED, ARDITI}


def test_a_three_number_row_that_disagrees_refuses_the_row_naming_the_number(tmp_path, seeds):
    def bump(three):
        three["rows"][f"{TAG}/{RELU}"]["correct"] += 1
    fx = _mirror(tmp_path, seeds, three_override=bump)
    bt, facts = X.read(_adapter(fx, seeds))
    why = X.refused(facts, TAG, RELU)
    assert M.THREE_NUMBERS_FILE in why and "correct: producer 102, re-derived 101" in why and len(bt["run"]) == 4


def test_the_other_tasks_three_number_record_is_never_read(tmp_path, seeds):
    """A disagreement written into the BigCodeBench-200 record leaves the BFCL-200 reader untouched: each reader reads its own three-number file."""
    def bump(three):
        three["rows"][f"{TAG}/{RELU}"]["correct"] += 1
    fx = X.mirror(tmp_path, seeds, eval_id="bigcodebench", root=str(tmp_path / "bfcl_seedrep"), three_override=bump)
    _bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0, facts["refusals"]


def test_when_the_seed_tags_none_has_not_run_the_rows_file_without_a_contrast_and_name_the_reason(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, not_run=(NONE,))
    bt, facts = X.read(_adapter(fx, seeds))
    # every comparison parks by name on the anchor currency (emission 3): four refusal lines, four parked, no contrast
    assert facts["n_refusals"] == 4 and facts["contrasts_parked"] == 4 and facts["runs"] == 4 and facts["n_not_run"] == 1 and "run_contrast" not in bt, facts["refusals"]
    assert facts["not_run"][0]["tuple_id"] == f"{SEL}/{NONE}" and facts["not_run"][0]["gate_verdict"] == "FAIL"
    for okey in (RELU, PROMPTED, ARDITI, COMPOSED):
        blk = X.run_of(bt, okey)["extra"]["anchor"]
        assert blk["state"] == B.ANCHOR_NOT_YET_RUN and blk["anchor_run_id"] is None and "has not run yet" in blk["reason"]


def test_a_gate_record_reading_fail_for_a_present_store_refuses_the_row(tmp_path, seeds):
    def bump(okey, gate):
        if okey == PROMPTED:
            gate["verdict"] = "FAIL"
            gate["problems"] = ["synthetic"]
    fx = _mirror(tmp_path, seeds, gate_override=bump)
    bt, facts = X.read(_adapter(fx, seeds))
    assert "verdict is 'FAIL'" in X.refused(facts, TAG, PROMPTED) and len(bt["run"]) == 4


def test_the_reader_is_registered_and_declares_its_source_owner_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.BfclSeedrepAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "bfcl_seedrep"
    assert AB.emission_version_of(M.BfclSeedrepAdapter) == "3" and inst.rng_scope == "none" and inst.roots == (M.ROOT_DEFAULT,)
    assert M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bfcl_seedrep")) and M.ROOT_DEFAULT != BC.ROOT_DEFAULT and M.LADDER == (1024,) and M.N_ITEMS == 200
    assert M.EVAL_ID == "bfcl" and M.INSTRUMENT_SEED == "bfcl_200_protocol_T0001_1024" and M.TASK == "inspect/bfcl_200" and M.SPEC.record_task == "bfcl_200"
    assert M.PROTOCOL_TEMPERATURE == 0.001 and M.SPEC.clamp_override is True and M.ITEM_LABEL_AT_CAP_UNRESOLVED == IC.ITEM_LABEL_AT_CAP_UNRESOLVED


def test_the_paired_tags_own_none_files_on_the_bfcl_instrument_and_anchors_the_composed_row_never_the_seed_rows(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, paired_none=True)
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6, facts["refusals"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    none_s, none_p = runs[(TAG, NONE)], runs[(TAG_P, NONE)]
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")
    assert none_p["extra"]["launch_set"]["class"] == "anchor" and none_p["instrument_id"] == inst["instrument_id"] == none_s["instrument_id"] and none_p["eval_id"] == "bfcl"
    assert none_p["lane_source_legacy"] == M.SOURCE and none_p["view_legacy"] == TAG_P and none_p["run_id"] != none_s["run_id"] and none_p["extra"]["standup_replicate_of_view"] == TAG
    assert none_p["run_id"] == M.run_id_for(TAG_P, fx["paired_none_key"], NONE, inst["instrument_id"]) != BC.run_id_for(TAG_P, fx["paired_none_key"], NONE, inst["instrument_id"])
    v_p, _b = _metrics(bt, none_p["run_id"])
    v_s, _b = _metrics(bt, none_s["run_id"])
    assert (v_p["k"], v_p["n"]) == (120, 200) and (v_s["k"], v_s["n"]) == (166, 200)
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)} and none_p["run_id"] not in contrasts
    for k in (RELU, PROMPTED, ARDITI):
        assert contrasts[X.run_of(bt, k)["run_id"]]["anchor_run_id"] == none_s["run_id"] and contrasts[X.run_of(bt, k)["run_id"]]["extra"]["cross_standup"] is False
    cp = X.run_of(bt, COMPOSED)
    c_comp = contrasts[cp["run_id"]]
    assert c_comp["anchor_run_id"] == none_p["run_id"] and c_comp["anchor_kind"] == "same_standup_none" and c_comp["extra"]["cross_standup"] is False and c_comp["extra"]["anchor_view_legacy"] == TAG_P
    assert c_comp["n_pairs"] == 200 and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 120 / 200) < 1e-12 and "cross_standup_note" not in c_comp["extra"]
    assert cp["extra"]["anchor"]["anchor_run_id"] == none_p["run_id"] and cp["extra"]["anchor"]["cross_standup"] is False and cp["extra"]["own_anchor"] == f"{TAG_P}/none" and "cross_standup_anchor" not in cp["extra"]
