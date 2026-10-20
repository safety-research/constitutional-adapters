"""The `bigcodebench_seedrep` reader on the SYNTHETIC mirror of `_bcbw_seedrep_fixtures` (invented item ids and shas, `TESTHOST`, no generation
text): one seed tag with its unsteered anchor, a fitted ReLU-steer object, a prompted key and the Arditi all-positions object, plus the paired
tag's composed difference_relu object; every store carrying both tasks; the gate records, the launch set, the launched pins and roster, and the
three-number record written from the same synthetic truth.

What it pins:

* ONE GROUP PER SEED: the paired tag reads in the seed tag's batch (the composed row pairs across standups with `s<N>/none`); the
  other task's files are never inputs of this reader.
* THE THREE NUMBERS on every row: the VALUE (`acc` = correct / scored, the recount computation's non-blank id, `n_items_judged` = scored),
  the BOUND (`acc` = correct / 200, a blank computation id, the interval `[correct, correct + unscored at cap] / 200`), the at-cap COUNT as
  `run_degeneration at_token_cap_finish_reason` over every at-cap item (an at-cap item is SCORED on this task and its value counts); no
  loop reading is invented (`rep_ratio_loop` not_measured).
* THE INSTRUMENT: every row on `bigcodebench_200_protocol_greedy_4096` bound to qwen35_4b_it — thinking OFF, cap 4,096, the greedy protocol
  row matched to the registered policy; `seed_base NULL`, `trial_index 0` and `decode_seed_rule none` (the instrument's and the protocol policy
  row's rule) on every row, so the harvester's DERIVED decode-policy match reads `exact` and names the protocol row — the reader itself
  writes only the producer placeholder `no_policy` and its own resolution on `extra`.
* THE GATE: a row files only under a PASS attesting `lora_native`; `max_lora_rank` is recorded, never compared.
* NOT RUN: a launch-set row with no store (a FAIL gate record beside it) files NOTHING — no run, no refusal — and rides `status_extra.not_run`;
  when that row is the seed tag's `none`, every other row files WITHOUT a contrast and `extra.anchor` names the reason.
* THE ANCHORS: every steered / prompted row of the seed tag against its `none`; the composed row across standups against the same `none`
  (`extra.cross_standup`) while the paired tag has no `none` of its own; the Arditi row carrying the declared role `arditi_unpaired_value` on the run
  and the contrast.
* THE PAIRED TAG'S OWN `none`: launch-set class `anchor`, a run of the paired tag on the same
  instrument (trial 0 like every row, `standup_replicate_of_view`), no anchor block of its own; the composed row then pairs against IT
  (`same_standup_none`, `cross_standup` false, `anchor_view_legacy` the paired tag) and the seed tag's rows never do — even when the seed tag's `none`
  has not run; a paired `none` the launch set names but that has not run (or refused, or is filed under another class) leaves the composed row on
  the cross-standup fallback with `extra.anchor.own_none_state` naming why.
* REFUSED BY NAME (the row, never the group): a store attesting another sha than the pins; a launch set expecting another sha; a gate record
  reading FAIL for a present store; a gate record absent for a present store; a three-number row disagreeing with the re-derivation; a store
  the launch set does not declare; a replicate seed block on a store; a RESULT that is not COMPLETE.
* end to end: the runs reach `ingested` through the harvester's rule-scored completion with no state event from the reader and nothing blocked.
"""
from __future__ import annotations

import hashlib
import os

import pytest

import _b1_fixtures as F
import _bcbw_seedrep_fixtures as X
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import _bcbw_common as B
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import bigcodebench_seedrep as M
from sisteer.registry.adapters import lineage_common as LC

EVAL = "bigcodebench"
TAG, TAG_P, SEL, SEL_P, SEED_N = X.TAG, X.TAG_P, X.SEL, X.SEL_P, X.SEED_N
NONE, RELU, PROMPTED, ARDITI, COMPOSED = X.NONE, X.RELU, X.PROMPTED, X.ARDITI, X.COMPOSED


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


def _mirror(tmp_path, seeds, **kw):
    return X.mirror(tmp_path, seeds, eval_id=EVAL, **kw)


def _adapter(fx, seeds) -> M.BigcodebenchSeedrepAdapter:
    return X.adapter(M.BigcodebenchSeedrepAdapter, fx, seeds)


def _metrics(bt, rid):
    return X.metrics_of(bt, rid, M.recount_computation_id())


# --------------------------------------------------------------------------------------------
# layout and identity
# --------------------------------------------------------------------------------------------

def test_discover_groups_the_seed_tag_with_its_paired_tag_and_reads_only_its_own_task_files(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    inputs = ad.discover()
    names = [os.path.basename(i.uri) for i in inputs]
    for shared in (B.LAUNCH_SET_FILE, B.PINS_FILE, B.ROSTER_FILE, M.THREE_NUMBERS_FILE, B.ROWS_LAUNCH_FILE, B.SCHEDULE_FILE):
        assert f"{shared}#{TAG}" in names, shared
    assert names.count("manifest.json") == 5 and names.count("rows.jsonl") == 5 and names.count("RESULT") == 5 and names.count(B.LAUNCHED_ENTRY_FILE) == 5 and names.count(B.GEOMETRY_FILE) == 5
    assert names.count(M.OUTCOMES_FILE) == 15, "outcomes/, summary/ and done/ of THIS task per store"
    assert not any(n.startswith("inspect__bfcl") for n in names), "the other task's files are never this reader's inputs"
    assert sum(1 for n in names if n.startswith(B.GATE_PREFIX)) == 5 and all(n.endswith(f"#{TAG}") for n in names if n.startswith(B.GATE_PREFIX))
    assert names.count(B.CONTROLS_SUMMARY_FILE) == 5
    assert {ad.comparison_group(i) for i in inputs} == {TAG}, "the paired tag reads in its seed tag's group (GROUP_RULE)"
    roles = {os.path.basename(i.uri): i.role for i in inputs}
    assert roles["rows.jsonl"] == "rows" and roles["gen_stats.json"] == "completion" and roles["RESULT"] == "log"
    assert B.seed_tag_of(TAG_P) == TAG and B.tag_of_row_dir(X.ROW_DIR_P) == TAG_P and B.row_dir_of(TAG_P) == X.ROW_DIR_P and B.tag_parts(TAG_P) == (SEED_N, True)
    assert B.launch_tag_of(TAG, ARDITI) == f"{TAG}_arditi" and B.launch_tag_of(TAG, RELU) == TAG


def test_run_ids_key_on_the_tag_prefixed_store_name_not_the_root_path(tmp_path, seeds):
    fx1 = _mirror(tmp_path, seeds, root=str(tmp_path / "a" / "bigcodebench_seedrep"))
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "b" / "elsewhere"))
    bt1, f1 = X.read(_adapter(fx1, seeds))
    bt2, _f2 = X.read(_adapter(fx2, seeds))
    assert f1["n_refusals"] == 0, f1["refusals"]
    ids1 = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r["run_id"] for r in bt1["run"]}
    ids2 = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r["run_id"] for r in bt2["run"]}
    assert ids1 == ids2 and len(ids1) == 5
    inst = str(seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")["instrument_id"])
    key = fx1["keys"][RELU]
    assert ids1[(TAG, RELU)] == I.run_id(LC.CAMPAIGN, f"{TAG}/{key}", RELU, M.EVAL_ID, inst) == M.run_id_for(TAG, key, RELU, inst)
    assert ids1[(TAG, RELU)].startswith(f"{LC.CAMPAIGN}:{M.EVAL_ID}:{inst[:12]}:")
    r = X.run_of(bt1, RELU)
    assert r["producer_run_key_legacy"] == f"{TAG}/{key}" and r["producer_object_key_legacy"] == RELU and r["instrument_id"] == inst
    assert X.run_of(bt1, COMPOSED)["producer_run_key_legacy"] == f"{TAG_P}/{fx1['keys'][COMPOSED]}" and X.run_of(bt1, COMPOSED)["view_legacy"] == TAG_P


# --------------------------------------------------------------------------------------------
# the three numbers, the item states, the instrument
# --------------------------------------------------------------------------------------------

def test_the_three_numbers_are_re_derived_from_the_outcome_record_and_the_items_carry_their_states(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["n_not_run"] == 0, facts["refusals"]
    r = X.run_of(bt, RELU)
    c = X.counts(fx["specs"][RELU])
    # 80 + 53 scored + 2 + 60 scored at the cap = 195 scored / 82 correct; at cap 67 (62 scored + 5 unscored); 5 unscored
    assert (c["n_scored"], c["correct"], c["n_at_cap"], len(c["at_cap_unscored"]), c["n_unscored"]) == (195, 82, 67, 5, 5)
    assert r["lifecycle_state"] == "generated" and r["token_cap_tokens"] == M.CAP == 4096 and r["token_cap_rung_index"] == 0 and r["token_cap_ladder_json"] == [M.CAP]
    assert r["n_items_planned"] == 200 and r["n_items_generated"] == 200 and r["n_items_judged"] == 195 and r["extra"]["escalation_state"] == "final_at_base"
    tn = r["extra"]["three_numbers"]
    assert (tn["n_scored"], tn["correct"], tn["n_at_cap"], tn["n_at_cap_scored"], tn["n_at_cap_unscored"], tn["n_unscored"], tn["n_unscored_below_cap"]) == (195, 82, 67, 62, 5, 5, 0)
    assert tn["acc_over_scored"] == 82 / 195 and tn["acc_over_attempted"] == 82 / 200 and tn["acc_over_attempted_upper_bound"] == 87 / 200 and tn["producer_record_agrees"] is True
    value, bound = _metrics(bt, r["run_id"])
    assert value["metric_id"] == "acc" and value["value"] == 82 / 195 and (value["k"], value["n"]) == (82, 195) and value["computed_by"] == "registry" and value["ci_method"] == "wilson_95"
    assert value["extra"]["producer_value"] == round(82 / 195, 4) and value["n_incomplete"] == 5
    assert bound["metric_id"] == "acc" and bound["value"] == 82 / 200 and (bound["k"], bound["n"]) == (82, 200) and bound["computed_by"] == "producer"
    assert bound["producer_metric_spelling"] == M.BOUND_SPELLING == "bound" and bound["ci_method"] == M.BOUND_CI_METHOD and (bound["ci_lo"], bound["ci_hi"]) == (82 / 200, 87 / 200)
    assert "secondary" in bound["extra"]["metric_role_expected"] and bound["n_incomplete"] == 5
    comps = {x["computation_id"] for x in bt["computation"]}
    assert value["computation_id"] in comps and value["computation_id"] == I.computation_id(M.RECOUNT_ALGORITHM, M.RECOUNT_VERSION, M.RECOUNT_SEED_PROVENANCE)
    assert M.RECOUNT_ALGORITHM == "sisteer.registry.adapters.bigcodebench_seedrep.recount"
    # items: 200 rows, the states
    items = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == r["run_id"]}
    assert len(items) == 200 and all(i["judgment_id"] == "" and i["metric_id"] == "acc" and i["trial_index"] == 0 for i in items.values())
    labels = {}
    for i in items.values():
        k = (i["label"], i["extra"].get("at_cap_class"), i["extra"].get("unscored_reason"))
        labels[k] = labels.get(k, 0) + 1
    assert labels == {(None, None, None): 195, (IC.ITEM_LABEL_AT_CAP_UNRESOLVED, "ceiling", None): 5}, labels
    assert all(i["value"] is None for i in items.values() if i["label"] is not None) and all(i["value"] in (0.0, 1.0) for i in items.values() if i["label"] is None)
    assert sum(1 for i in items.values() if i["extra"]["scored_at_cap"]) == 62
    flags = {f["item_id"]: f for f in bt["item_flags"] if f["run_id"] == r["run_id"]}
    assert sum(f["at_token_cap"] for f in flags.values()) == 67 and all(f["degenerate_rep_ratio_loop"] is None for f in flags.values())
    assert sum(1 for f in flags.values() if f["generation_artifact_id"] is not None) == 195, "a scored item points at its rows.jsonl row; an unscored one has no row"
    assert all(f["finish_reason"] == "max_tokens" for f in flags.values() if f["at_token_cap"] == 1)
    # the run-level share over the 200 attempted items: the at-cap count alone; nothing invented for the loop measures
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"], deg[M.RAW_AT_CAP_MEASURE]["denominator_unit"], deg[M.RAW_AT_CAP_MEASURE]["computed_by"]) == (67, 200, "items", "registry")
    assert set(deg) == set(IC.DECLARED_MEASURES) and "truncation_only" not in deg
    assert all(deg[m]["measurement_state"] == "not_measured" for m in IC.DECLARED_MEASURES if m != M.RAW_AT_CAP_MEASURE)
    assert "nothing is invented" in deg["rep_ratio_loop"]["notes"]
    # the composed row: two unscored below the cap are `unscored`, not the third state
    cp = X.run_of(bt, COMPOSED)
    citems = {i["item_id"]: i for i in bt["item_result"] if i["run_id"] == cp["run_id"]}
    assert sum(1 for i in citems.values() if i["label"] == M.ITEM_LABEL_UNSCORED) == 2 and cp["extra"]["item_state_counts"][M.ITEM_LABEL_UNSCORED] == 2 and cp["n_items_judged"] == 198
    _v, cb = _metrics(bt, cp["run_id"])
    assert cb["ci_lo"] == cb["ci_hi"] == 79 / 200, "an item unscored below the cap is a miss in the bound, never an open interval"
    # the unsteered anchor: value equals gen_stats' acc; the row is trial 0 with no seed base
    n = X.run_of(bt, NONE)
    nc = X.counts(fx["specs"][NONE])
    v_n, _b_n = _metrics(bt, n["run_id"])
    assert (v_n["k"], v_n["n"]) == (nc["correct"], nc["n_scored"]) and v_n["extra"]["equals_gen_stats_acc"] is True and n["trial_index"] == 0 and n["seed_base"] is None
    assert n["object_id"] == "none" and n["serving_path"] == "passthrough" and n["extra"]["same_object_rows"]["twin_tags_legacy"] == ["qwen35_4b_s1", "qwen35_4b_s3"]


def test_the_run_row_carries_the_attested_identity_and_its_own_instrument(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = X.read(_adapter(fx, seeds))
    r = X.run_of(bt, RELU)
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")
    assert r["model_id"] == "qwen35_4b_it" and r["object_id"] == X.OBJ and r["view_legacy"] == TAG and r["lane_source_legacy"] == M.SOURCE == "bigcodebench_seedrep" and r["campaign"] == LC.CAMPAIGN
    assert r["instrument_id"] == inst["instrument_id"] and r["eval_id"] == "bigcodebench" and r["item_set_id"] == "bigcodebench_200" and r["harness"] == "inspect_evals" and r["scoring_rule"] == "pass_at_1_sandbox"
    assert r["thinking_mode"] == "off" and r["token_cap_tokens"] == 4096 and r["serving_path"] == "lora_native" and r["applied_positions_mode"] == "all" and r["transfer_kind"] == "base_to_instruct"
    assert r["decode_sent_json"] == {"temperature": 0.0, "top_p": 1.0, "top_k": 0} and r["decode_temperature_sent"] == 0.0 and r["decode_top_k_sent"] == 0 and r["decode_source"] == "greedy"
    assert r["instrument_match"] == "match" and r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|bigcodebench|off|protocol@v1"
    # no per-seed rule — the documented default on EVERY row, fitted or not; the seed rule is the instrument's (`none`: the greedy protocol pins no seed)
    assert r["seed_base"] is None and r["decode_seed_rule"] == "none" == inst["decode_seed_rule"] and r["trial_index"] == 0 and r["extra"]["decode_seed_policy"] == "default" and r["extra"]["fit_seed"] == SEED_N
    assert "pins no seed" in r["field_confidence_json"]["decode_seed_rule"]["source"] and X.SEED_RULE in r["field_confidence_json"]["decode_seed_rule"]["source"]
    assert r["extra"]["decode_seed_rule_text"] == X.SEED_RULE and r["extra"]["seed_axis"]["rule"] == B.SEED_AXIS_RULE
    assert r["field_confidence_json"]["seed_base"]["confidence"] == "derived" and r["field_confidence_json"]["trial_index"]["confidence"] == "derived"
    assert r["code_sha"] == X.TREE and r["model_revision"] == X._REVISION and r["modal_call_id"].startswith("fc-") and r["job_name"].startswith("tinycap-synthetic-")
    assert r["substrate"] == "modal" and r["substrate_source"] == "derived" and r["context_window_tokens"] == 16384 and r["served_max_num_seqs"] == 31
    assert r["producer_session"] == "Overflow" and r["is_operating_point"] == 1 and r["strength_nominal"] == 1.0 and r["judge_id"] is None
    assert r["extra"]["launch_set"]["pin_key"] == X.STAGED and r["extra"]["launch_set"]["object_sha256_expected"] == X.OBJ and r["extra"]["launch_set"]["class"] == "arm"
    assert r["extra"]["gate"]["verdict"] == "PASS" and r["extra"]["gate"]["serving_engine_attested"] == "lora_native" and r["extra"]["gate"]["max_lora_rank"] == 1
    assert r["extra"]["standup"]["max_lora_rank"] == 1 and "never compared" in r["extra"]["max_lora_rank_rule"]
    assert r["extra"]["launched_roster"]["roster_sha256"] == fx["roster_sha256"] and r["extra"]["launched_roster"]["roster_entry"]["object"]["staged_id"] == X.STAGED
    assert r["extra"]["item_set_pin"]["n"] == 200 and r["extra"]["item_set_pin"]["of"] == 992 and r["extra"]["tool_calls"] is True
    assert r["generated_utc"] == "2026-09-23T04:17:11.000Z" and r["started_utc"] == "2026-09-23T03:55:56.000Z" and r["queued_utc"] == r["started_utc"]
    assert r["store_completion_artifact_uri"].endswith(os.path.join("cells", fx["keys"][RELU], "gen_stats.json"))
    assert "run_state" not in bt
    # the prompted key resolves through si_texts and its system_sha; the composed row is the paired tag's, paired_of s2
    p = X.run_of(bt, PROMPTED)
    assert p["object_id"] == str(seeds.si_object_id("safety_refusal")) and p["serving_path"] == "passthrough" and p["transfer_kind"] == "prompted_only" and p["trial_index"] == 0
    cp = X.run_of(bt, COMPOSED)
    assert cp["object_id"] == X.OBJ_P and cp["view_legacy"] == TAG_P and cp["trial_index"] == 0 and cp["extra"]["paired_of"] == SEL and cp["extra"]["roster_paired_of"] == SEL
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and cp["extra"]["launch_set"]["class"] == "composed"
    # the all-positions object: ONE fit on every seed tag -> trial 0, the operating point's strength columns and the declared role
    a = X.run_of(bt, ARDITI)
    assert a["object_id"] == X.OBJ_A and a["trial_index"] == 0 and a["seed_base"] is None and a["extra"]["fit_seed"] is None and a["extra"]["declared_role"] == M.ARDITI_ROLE
    assert (a["strength_coefficient"], a["strength_unit"], a["strength_nominal"], a["is_operating_point"], a["operating_point_id_legacy"]) == (0.3, "raw_coefficient", 1.0, 1, X.STAGED_A)
    assert a["transfer_kind"] == "instruct_direct" and a["extra"]["single_object"]["operating_point"] == "ar_c0p3" and M.SOURCE in a["extra"]["single_object"]["coexist_ruling_needed"]
    assert facts["rows"][f"{TAG}|{ARDITI}"]["max_lora_rank_recorded"] == 1 and facts["rows"][f"{TAG}|{ARDITI}"]["gate_verdict"] == "PASS"


def test_every_row_pairs_with_the_seed_tags_none_the_composed_row_across_standups_and_the_arditi_row_declared_unpaired(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds)
    bt, facts = X.read(_adapter(fx, seeds))
    none_id = X.run_of(bt, NONE)["run_id"]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)}
    assert all(c["anchor_run_id"] == none_id and c["metric_id"] == "acc" and c["delta_method"] == "exact_mcnemar" for c in contrasts.values())
    assert {k: contrasts[X.run_of(bt, k)["run_id"]]["anchor_kind"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)} == \
        {RELU: "same_standup_none", PROMPTED: "same_standup_none", ARDITI: "same_standup_none", COMPOSED: "sibling_none"}, "the D65 fallback is another standup's none"
    assert all(c["computation_id"] == M.recount_computation_id() and c["computed_by"] == "registry" and 0.0 <= c["delta_p"] <= 1.0 for c in contrasts.values())
    c_relu = contrasts[X.run_of(bt, RELU)["run_id"]]
    assert c_relu["n_pairs"] == 195 and c_relu["extra"]["cross_standup"] is False and c_relu["extra"].get("declared_role") is None and c_relu["extra"]["anchor_view_legacy"] == TAG
    c_comp = contrasts[X.run_of(bt, COMPOSED)["run_id"]]
    assert c_comp["extra"]["cross_standup"] is True and c_comp["extra"]["row_tag_legacy"] == TAG_P and "cross_standup_note" in c_comp["extra"] and c_comp["n_pairs"] == 198
    c_ard = contrasts[X.run_of(bt, ARDITI)["run_id"]]
    assert c_ard["extra"]["declared_role"] == M.ARDITI_ROLE and "anchor_standup_note" in c_ard["extra"]
    # the anchor block on every contrasted run names the anchor; the run record's id was minted AFTER the block was stamped
    r = X.run_of(bt, RELU)
    assert r["extra"]["anchor"]["state"] == "contrasted" and r["extra"]["anchor"]["anchor_run_id"] == none_id and r["extra"]["anchor"]["n_pairs"] == 195
    assert "anchor" not in X.run_of(bt, NONE)["extra"]
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["cross_standup"] is True and facts["contrasts"][f"{TAG}|{ARDITI}"]["declared_role"] == M.ARDITI_ROLE
    # no paired `none` in the launch set at all: the standing spelling, nothing of the own-anchor vocabulary (the live mirrors' records stay byte-identical)
    cp = X.run_of(bt, COMPOSED)
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and "own_anchor" not in cp["extra"] and cp["extra"]["anchor"]["rule"] == B.ANCHOR_RULE
    assert cp["extra"]["anchor"]["anchor_view_legacy"] == TAG and cp["extra"]["anchor"]["cross_standup"] is True and "own_none_state" not in cp["extra"]["anchor"]
    assert "own_none_state" not in c_comp["extra"] and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 96 / 198) < 1e-12, "the seed none over the composed row's 198 shared items"
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["own_standup"] is False and facts["contrasts"][f"{TAG}|{RELU}"]["own_standup"] is True


# --------------------------------------------------------------------------------------------
# not run: nothing filed; the missing anchor
# --------------------------------------------------------------------------------------------

def test_a_launch_set_row_with_no_store_files_nothing_and_rides_the_facts(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, not_run=(PROMPTED,))
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 4 and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, RELU, ARDITI, COMPOSED}
    assert "disposition" not in bt and not any(r["campaign_cell_id_legacy"] == PROMPTED for r in bt["run"])
    assert facts["n_not_run"] == 1 and facts["not_run"][0]["tuple_id"] == f"{SEL}/{PROMPTED}" and facts["not_run"][0]["gate_verdict"] == "FAIL"
    assert "has not COMPLETED" in facts["not_run"][0]["gate_problems"][0] and facts["not_run"][0]["rule"] == B.NOT_RUN_RULE
    assert len(bt["run_contrast"]) == 3


def test_when_the_seed_tags_none_has_not_run_every_row_files_without_a_contrast_and_names_the_reason(tmp_path, seeds):
    """No `none` of the seed tag has run and the frame holds no unsteered run at the exact key: every comparison PARKS by name (emission 3, the anchor
    currency: a park is a `row_refusals` line and a `delta_block` on the VALUE row), the run's `extra.anchor` keeps the plan's spelling of why."""
    fx = _mirror(tmp_path, seeds, not_run=(NONE,))
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 4 and facts["runs"] == 4 and facts["n_not_run"] == 1 and "run_contrast" not in bt, facts["refusals"]
    assert facts["contrasts_parked"] == 4 and facts["contrasts_filed"] == 0 and facts["anchor_states"] == {"no_current_anchor": 4}
    assert all(x.split(" -- ")[0].endswith("|contrast") and "contrast has no current anchor at key" in x and "none_declared" in x for x in facts["refusals"]), facts["refusals"]
    for okey in (RELU, PROMPTED, ARDITI, COMPOSED):
        blk = X.run_of(bt, okey)["extra"]["anchor"]
        assert blk["state"] == B.ANCHOR_NOT_YET_RUN == "anchor_not_yet_run" and blk["anchor_run_id"] is None and f"{TAG}/none" in blk["reason"] and "has not run yet" in blk["reason"]
        assert blk["anchor_view_legacy"] == TAG and blk["cross_standup"] is (okey == COMPOSED)
        assert blk["currency_state"] == "no_current_anchor" and blk["currency_own_reason"] == "none_declared" and blk["anchor_selection"]["state"] == "no_current_anchor"
        assert facts["contrasts"][f"{TAG_P if okey == COMPOSED else TAG}|{okey}"]["state"] == "anchor_not_yet_run"
        # the park rides the VALUE row (the recount reading the contrast is keyed on), never the bound row
        value, bound = _metrics(bt, X.run_of(bt, okey)["run_id"])
        blk2 = value["extra"]["delta_block"]
        assert blk2["anchor"]["state"] == "no_current_anchor" and blk2["anchor"]["own_reason"] == "none_declared" and blk2["delta"] is None and blk2["n_pairs"] == 0
        assert blk2["anchor_selection"]["n_candidates"] == 0 and blk2["rule"] == AC.RULE and "delta_block" not in bound["extra"]
    # the VALUE's records are byte-identical with and without the anchor BUT for the park (emission 3: the VALUE row carries the comparison's `delta_block`,
    # a park here and `null` once the contrast files); the bound row and everything else do not move
    fx2 = _mirror(tmp_path, seeds, root=str(tmp_path / "with_none"))
    bt2, _f2 = X.read(_adapter(fx2, seeds))
    rid = X.run_of(bt, RELU)["run_id"]
    assert rid == X.run_of(bt2, RELU)["run_id"]
    v1, b1 = _metrics(bt, rid)
    v2, b2 = _metrics(bt2, rid)
    assert v1["extra"]["delta_block"]["anchor"]["state"] == "no_current_anchor" and v2["extra"]["delta_block"] is None

    def pathless(m):
        d = {k: v for k, v in m.items() if k != "source_uri"}
        d["extra"] = {k: v for k, v in m["extra"].items() if k not in ("producer_record", "delta_block")}
        return d
    assert pathless(v1) == pathless(v2) and pathless(b1) == pathless(b2), "the two mirrors differ only in their roots and the park"
    assert I.adapter_record_id(X.run_of(bt, RELU), "run") != I.adapter_record_id(X.run_of(bt2, RELU), "run"), "the run row re-emits when the anchor lands; the value does not"


# --------------------------------------------------------------------------------------------
# refusals by name (the row, never the group)
# --------------------------------------------------------------------------------------------

def test_a_gate_record_reading_fail_for_a_present_store_refuses_the_row_and_the_siblings_file(tmp_path, seeds):
    def bump(okey, gate):
        if okey == RELU:
            gate["verdict"] = "FAIL"
            gate["problems"] = ["synthetic: the standup's engine was not attested"]
    fx = _mirror(tmp_path, seeds, gate_override=bump)
    bt, facts = X.read(_adapter(fx, seeds))
    why = X.refused(facts, TAG, RELU)
    assert "verdict is 'FAIL'" in why and "synthetic: the standup's engine" in why and len(bt["run"]) == 4
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, PROMPTED, ARDITI, COMPOSED}


def test_a_gate_record_attesting_another_engine_is_refused(tmp_path, seeds):
    def bump(okey, gate):
        if okey == COMPOSED:
            gate["facts"][0]["route.serving_engine"] = "lens"
    fx = _mirror(tmp_path, seeds, gate_override=bump)
    _bt, facts = X.read(_adapter(fx, seeds))
    assert "route.serving_engine 'lens'" in X.refused(facts, TAG_P, COMPOSED)


def test_a_present_store_without_a_gate_record_is_refused_by_name(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, drop_gate=(ARDITI,))
    bt, facts = X.read(_adapter(fx, seeds))
    assert "no gate record" in X.refused(facts, TAG, ARDITI) and "PRESENT store" in X.refused(facts, TAG, ARDITI) and len(bt["run"]) == 4


def test_a_store_attesting_another_sha_than_the_pins_is_refused_and_the_siblings_file(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = _mirror(tmp_path, seeds, attest_sha={RELU: other})
    bt, facts = X.read(_adapter(fx, seeds))
    why = X.refused(facts, TAG, RELU)
    assert "fit_provenance.object_sha" in why and "no prefix of the pinned object" in why and len(bt["run"]) == 4


def test_a_launch_set_expecting_another_sha_than_the_pins_is_refused_by_name(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()

    def bump(doc):
        for r in doc["evals"][EVAL]["launch"]:
            if r["cell"] == COMPOSED:
                r["object_sha256_expected"] = other
    fx = _mirror(tmp_path, seeds, launch_override=bump)
    _bt, facts = X.read(_adapter(fx, seeds))
    assert "the launch set expects object" in X.refused(facts, TAG_P, COMPOSED)


def test_launched_pins_naming_another_sha_than_the_harness_pins_are_refused(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = _mirror(tmp_path, seeds, launched_pins_sha={ARDITI: other})
    _bt, facts = X.read(_adapter(fx, seeds))
    why = X.refused(facts, TAG, ARDITI)
    assert "the launch set expects object" in why or "the launched pins" in why


def test_a_three_number_row_that_disagrees_refuses_the_row_naming_the_number(tmp_path, seeds):
    def bump(three):
        three["rows"][f"{TAG}/{NONE}"]["at_cap_n"] += 1
        three["rows"][f"{TAG}/{NONE}"]["value"] = 0.1234
    fx = _mirror(tmp_path, seeds, three_override=bump)
    bt, facts = X.read(_adapter(fx, seeds))
    why = X.refused(facts, TAG, NONE)
    assert M.THREE_NUMBERS_FILE in why and "at_cap_n: producer" in why and "value: producer 0.1234" in why and "2 number(s)" in why
    assert len(bt["run"]) == 4 and "run_contrast" not in bt and X.run_of(bt, RELU)["extra"]["anchor"]["state"] == B.ANCHOR_REFUSED


def test_a_three_number_record_without_the_rows_key_refuses_the_row(tmp_path, seeds):
    def bump(three):
        del three["rows"][f"{TAG_P}/{COMPOSED}"]
    fx = _mirror(tmp_path, seeds, three_override=bump)
    _bt, facts = X.read(_adapter(fx, seeds))
    assert "carries no row for" in X.refused(facts, TAG_P, COMPOSED)


def test_a_store_the_launch_set_does_not_declare_is_refused_by_name(tmp_path, seeds):
    def bump(doc):
        doc["evals"][EVAL]["launch"] = [r for r in doc["evals"][EVAL]["launch"] if r["cell"] != PROMPTED]
    fx = _mirror(tmp_path, seeds, launch_override=bump)
    _bt, facts = X.read(_adapter(fx, seeds))
    assert "the launch set names no row for" in X.refused(facts, TAG, PROMPTED)


def test_a_replicate_seed_block_on_a_store_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, replicate_block=True)
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 5 and all("replicate block" in x and "D69" in x for x in facts["refusals"]) and "run" not in bt


def test_a_result_that_is_not_complete_is_refused(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, result_incomplete=(RELU,))
    _bt, facts = X.read(_adapter(fx, seeds))
    assert "RESULT's last line does not read result=COMPLETE" in X.refused(facts, TAG, RELU)


def test_the_reader_never_consults_the_registry_for_the_object(tmp_path, seeds):
    """An object row absent from a connected registry is NOT a refusal here: the run is emitted on the attested identity and blocked_records holds it.
    (The connection is a primed frame -- the anchor currency reads `v_run_current` from it -- with an EMPTY `object` table beside it.)"""
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    frame = X.primed_frame(ad)
    frame.con.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY)")
    bt, facts = X.read(ad, conn=frame.con)
    assert facts["n_refusals"] == 0 and X.run_of(bt, RELU)["object_id"] == X.OBJ and X.run_of(bt, ARDITI)["object_id"] == X.OBJ_A
    assert frame.con.execute("SELECT COUNT(*) FROM object").fetchone()[0] == 0 and len(bt["run_contrast"]) == 4


# --------------------------------------------------------------------------------------------
# end to end: the harvester's rule-scored completion
# --------------------------------------------------------------------------------------------

def _registry_root(tmp_path, seeds, objects) -> str:
    from sisteer.registry.adapters import dimensions as DIM
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    now = R.utc_now()
    recs = [r for r in LC.standin_dimension_records(seeds, F.producer(), now) if r["record_type"] == "paper_round"]
    recs += DIM.build_records(DIM.load_seeds(seeds.source_dir), F.producer(), now=now, verify_si=False).all_records()
    for o in objects:
        r = LC.standin_object_record(o, seeds, F.producer(), now, "qwen35_4b_base")
        if r:
            recs.append(r)
    R.write_records(os.path.join(root, "records", "standins", "000001-dims.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root


def test_runs_reach_ingested_through_the_rule_scored_completion_with_the_value_row_the_owner(tmp_path, seeds):
    """Through the runner and the harvester in TWO passes (emission 3, the anchor currency; no provisional anchors): the first pass over
    the fresh registry files every run and PARKS every comparison (the seed tag's `none` is emitted by that very read and the frame does not know
    it), the fold makes it current and the anchor decision input moves, so the second pass re-reads the group under a new revision id (no file
    change, no bump) and files the four contrasts `own_current`; a third pass is quiet."""
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    objects = [ad.resolve_object(RELU, SEL)[0], ad.resolve_object(COMPOSED, SEL_P)[0], ad.resolve_object(ARDITI, SEL, pin_key=X.STAGED_A)[0]]
    root = _registry_root(tmp_path, seeds, objects)
    rep0, ing0, ing0b = F.run_and_ingest(root, ad, now="2026-09-23T08:00:00.000Z")
    assert rep0.groups_refused == 0 and ing0.refused + ing0b.refused == 0, ([r.error for r in rep0.results], ing0.refusals, ing0b.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)] and F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [(5,)]
    parked = F.q(root, "SELECT json_extract(rm.extra_json, '$.delta_block.anchor.own_reason') FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                       "WHERE r.lane_source_legacy = ? AND r.object_id <> 'none' AND rm.computation_id <> ''", (M.SOURCE,))
    assert parked == [("not_in_frame",)] * 4, parked
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, lifecycle_state, n_items_planned, n_items_generated, n_items_judged, token_cap_tokens, trial_index, seed_base, "
                     "judged_utc IS NOT NULL, ingested_utc IS NOT NULL FROM run WHERE lane_source_legacy = ? ORDER BY 1, 2", (M.SOURCE,))
    c = {k: X.counts(v) for k, v in fx["specs"].items()}
    assert rows == [(TAG, ARDITI, "ingested", 200, 200, c[ARDITI]["n_scored"], M.CAP, 0, None, 1, 1),
                    (TAG, RELU, "ingested", 200, 200, c[RELU]["n_scored"], M.CAP, 0, None, 1, 1),
                    (TAG, NONE, "ingested", 200, 200, c[NONE]["n_scored"], M.CAP, 0, None, 1, 1),
                    (TAG, PROMPTED, "ingested", 200, 200, c[PROMPTED]["n_scored"], M.CAP, 0, None, 1, 1),
                    (TAG_P, COMPOSED, "ingested", 200, 200, c[COMPOSED]["n_scored"], M.CAP, 0, None, 1, 1)], rows
    assert F.q(root, "SELECT COUNT(*) FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    # (two revisions carry the per-item rows after the second pass; the CURRENT view holds each once)
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(1000,)]
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ? AND i.value IS NULL AND i.label = ?",
               (M.SOURCE, IC.ITEM_LABEL_AT_CAP_UNRESOLVED)) == [(sum(len(v["at_cap_unscored"]) for v in c.values()),)]
    m = F.q(root, "SELECT r.campaign_cell_id_legacy, rm.computation_id <> '', rm.value, rm.k, rm.n FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                  "WHERE r.lane_source_legacy = ? AND rm.metric_id = 'acc' ORDER BY 1, 2 DESC", (M.SOURCE,))
    assert [x for x in m if x[0] == RELU] == [(RELU, 1, 82 / 195, 82, 195), (RELU, 0, 82 / 200, 82, 200)]
    dg = F.q(root, "SELECT d.measure, d.measurement_state, d.numerator, d.denominator, d.rate FROM v_run_degeneration_current d JOIN run r ON r.run_id = d.run_id "
                   "WHERE r.campaign_cell_id_legacy = ? AND d.measure IN (?, ?) ORDER BY 1", (RELU, M.RAW_AT_CAP_MEASURE, "rep_ratio_loop"))
    assert dg == [(M.RAW_AT_CAP_MEASURE, "measured", 67, 200, 67 / 200), ("rep_ratio_loop", "not_measured", None, None, None)], dg
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(4,)]
    # every contrast own_current: the seed rows on the seed none (same_standup_none), the composed row on it ACROSS tags (the fallback, sibling_none: no paired none here)
    assert sorted(F.q(root, "SELECT DISTINCT json_extract(extra_json, '$.anchor_selection.state'), anchor_kind FROM run_contrast")) == [("own_current", "same_standup_none"), ("own_current", "sibling_none")]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.lane_source_legacy = ? AND r.object_id <> 'none' AND rm.computation_id <> '' "
                     "AND json_extract(rm.extra_json, '$.delta_block') IS NULL AND json_type(rm.extra_json, '$.delta_block') = 'null'", (M.SOURCE,)) == [(4,)], "a filed contrast states delta_block: null"
    # the DERIVED decode-policy match (decode_match.stamp_run_rows at the end of the ingest cycle): every row names the protocol row — the reader wrote
    # the placeholder, and `decode_seed_rule none` is what lets the SQL predicate (policy seed_rule = run seed_rule) admit the row
    assert F.q(root, "SELECT decode_seed_rule, decode_policy_match, decode_policy_id, COUNT(*) FROM run WHERE lane_source_legacy = ? GROUP BY 1, 2, 3", (M.SOURCE,)) == \
        [("none", "exact", "qwen35_4b_it|bigcodebench|off|protocol@v1", 5)]
    assert F.q(root, "SELECT json_extract(field_versions_json, '$.decode_policy_match[0]') FROM run WHERE lane_source_legacy = ? LIMIT 1", (M.SOURCE,)) == [("derived",)]
    # a second pass over unchanged inputs publishes nothing and the rows stay ingested
    rep2, _ing3, _ing4 = F.run_and_ingest(root, _adapter(fx, seeds), now="2026-09-23T10:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [("ingested",)]


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_owner_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.BigcodebenchSeedrepAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "bigcodebench_seedrep"
    assert AB.emission_version_of(M.BigcodebenchSeedrepAdapter) == "3" and inst.rng_scope == "none" and inst.roots == (M.ROOT_DEFAULT,)
    assert M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "bigcodebench_seedrep")) and M.LADDER == (4096,) and M.CAMPAIGN == "sdfbase_full22" and M.N_ITEMS == 200
    assert M.EVAL_ID == "bigcodebench" and M.INSTRUMENT_SEED == "bigcodebench_200_protocol_greedy_4096" and M.TASK == "inspect/bigcodebench_200" and M.DECODE_SEED_RULE == "none"
    assert M.PROTOCOL_TEMPERATURE == 0.0 and M.SPEC.clamp_override is False and M.SPEC.record_task == "bigcodebench_200"
    assert not isinstance(getattr(B.BcbwSeedrepReader, "name", None), str), "the shared base is a helper, never an adapter of its own"
    assert B.parse_result_line("tinycap-modal name=x rc=0 result=COMPLETE status=OK sha=" + "a" * 40 + " at=2026-09-23T04:17:11Z") == {
        "head": "tinycap-modal", "name": "x", "rc": "0", "result": "COMPLETE", "status": "OK", "sha": "a" * 40, "at": "2026-09-23T04:17:11Z"}
    assert B.parse_result_line("something else") is None
    assert B.gate_name_parts("gate_qwen35_4b_s1_arditi_arditi_allpos_bcbw-qwen35_4b_s1_arditi-8365992.json") == {
        "tag": "qwen35_4b_s1", "arditi": "_arditi", "okey": "arditi_allpos", "suffix": "bcbw-qwen35_4b_s1_arditi-8365992"}
    assert B.gate_name_parts("gate_qwen35_4b_s1p_constv2-mix-full__psr_ce_all_minus_ctl_bcbw-qwen35_4b_s1p-8365992.json")["okey"] == COMPOSED


# --------------------------------------------------------------------------------------------
# the paired tag's OWN `none`
# --------------------------------------------------------------------------------------------

def test_the_paired_tags_own_none_files_as_its_anchor_and_the_composed_row_pairs_against_it_never_the_seed_rows(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, paired_none=True)
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and facts["n_not_run"] == 0, facts["refusals"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    none_s, none_p = runs[(TAG, NONE)], runs[(TAG_P, NONE)]
    # the paired `none`: launch-set class anchor, a run of the paired tag on the BigCodeBench-200 instrument, trial 0, its own tag-prefixed id, a PASS gate
    assert none_p["extra"]["launch_set"]["class"] == "anchor" and none_p["object_id"] == "none" and none_p["instrument_id"] == none_s["instrument_id"] and none_p["serving_path"] == "passthrough"
    assert none_p["producer_run_key_legacy"] == f"{TAG_P}/{fx['paired_none_key']}" == f"{TAG_P}/{fx['keys'][NONE]}", "the store key is the seed tag's, byte for byte; the tag prefix keeps the ids apart"
    assert none_p["run_id"] != none_s["run_id"] and none_p["trial_index"] == 0 and none_p["seed_base"] is None and none_p["view_legacy"] == TAG_P and none_p["extra"]["gate"]["verdict"] == "PASS"
    assert none_p["extra"]["paired_of"] == SEL and none_p["extra"]["roster_paired_of"] == SEL and none_p["extra"]["standup_replicate_of_view"] == TAG
    assert none_p["extra"]["anchor_rule"] == B.PAIRED_ANCHOR_RULE == M.ANCHOR_RULE.__class__(B.PAIRED_ANCHOR_RULE) and f"{TAG}/none" in none_p["extra"]["anchor_role"]
    assert "anchor" not in none_p["extra"] and "cross_standup_anchor" not in none_p["extra"] and "own_anchor" not in none_p["extra"]
    v_p, _b = _metrics(bt, none_p["run_id"])
    v_s, _b = _metrics(bt, none_s["run_id"])
    assert (v_p["k"], v_p["n"]) == (61, 200) and (v_s["k"], v_s["n"]) == (96, 200), "two different `none` readings at one seed base"
    # the contrasts: every seed row against the SEED `none`, the composed row against ITS OWN tag's `none`; no `none` is a contrast's row
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (RELU, PROMPTED, ARDITI, COMPOSED)} and none_p["run_id"] not in contrasts
    for k in (RELU, PROMPTED, ARDITI):
        c = contrasts[X.run_of(bt, k)["run_id"]]
        assert c["anchor_run_id"] == none_s["run_id"] and c["extra"]["anchor_view_legacy"] == TAG and c["extra"]["cross_standup"] is False and "own_standup_note" not in c["extra"]
        blk = X.run_of(bt, k)["extra"]["anchor"]
        assert blk["anchor_run_id"] == none_s["run_id"] and blk["anchor_view_legacy"] == TAG and blk["rule"] == B.ANCHOR_RULE and "own_none_state" not in blk
    c_relu = contrasts[X.run_of(bt, RELU)["run_id"]]
    assert c_relu["n_pairs"] == 195 and abs(c_relu["extra"]["acc_anchor_over_pairs"] - 95 / 195) < 1e-12, "the seed none over the seed row's 195 shared items (its at-cap-correct item 197 is outside them)"
    cp = X.run_of(bt, COMPOSED)
    c_comp = contrasts[cp["run_id"]]
    assert c_comp["anchor_run_id"] == none_p["run_id"] and c_comp["anchor_kind"] == "same_standup_none" and c_comp["extra"]["anchor_view_legacy"] == TAG_P and c_comp["extra"]["row_tag_legacy"] == TAG_P
    assert c_comp["extra"]["cross_standup"] is False and "cross_standup_note" not in c_comp["extra"] and "own_none_state" not in c_comp["extra"]
    assert c_comp["extra"]["anchor_rule"] == B.PAIRED_ANCHOR_RULE and TAG_P in c_comp["extra"]["own_standup_note"] and f"{TAG}/none" in c_comp["extra"]["own_standup_note"]
    assert c_comp["n_pairs"] == 198 and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 61 / 198) < 1e-12, "the PAIRED none over the composed row's 198 shared items, not the seed none's 96/198"
    # the composed run's anchor block and extras spell the own anchor; the fallback's spelling is gone
    blk = cp["extra"]["anchor"]
    assert blk["state"] == "contrasted" and blk["anchor_run_id"] == none_p["run_id"] and blk["anchor_view_legacy"] == TAG_P and blk["cross_standup"] is False and blk["n_pairs"] == 198
    assert blk["rule"] == B.PAIRED_ANCHOR_RULE and blk["own_standup_replicate_of_view"] == TAG and "own_none_state" not in blk
    assert cp["extra"]["own_anchor"] == f"{TAG_P}/none" and "cross_standup_anchor" not in cp["extra"] and cp["extra"]["standup_replicate_of_view"] == TAG and cp["extra"]["anchor_rule"] == B.PAIRED_ANCHOR_RULE
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["own_standup"] is True and facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["anchor_view_legacy"] == TAG_P and facts["contrasts"][f"{TAG}|{RELU}"]["own_standup"] is True


def test_the_paired_none_anchors_the_composed_row_even_when_the_seed_tags_none_has_not_run_and_the_seed_rows_re_anchor_on_it_as_a_sibling(tmp_path, seeds):
    """The paired tag's own `none` anchors the composed row (own_current). The seed rows' OWN pairing names the seed tag's `none`, which has not run
    (`anchor_not_yet_run`: the adapter's rule never names the paired `none` for them) -- and since emission 3 the anchor currency then searches the
    exact key and finds the paired tag's `none`: the same source, another standup, a current run of the frame -> `sibling_same_source`, anchor_kind
    `sibling_none`, cross_standup true, paired per item because that `none` is a row of this very batch (borrow a
    current baseline when the own one is absent). When the seed `none` lands and folds, the decision input moves and the pairing returns to it."""
    fx = _mirror(tmp_path, seeds, not_run=(NONE,), paired_none=True)
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["n_not_run"] == 1 and facts["not_run"][0]["tuple_id"] == f"{SEL}/{NONE}", facts["refusals"]
    none_p = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}[(TAG_P, NONE)]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (COMPOSED, RELU, PROMPTED, ARDITI)}
    assert facts["anchor_states"] == {"own_current": 1, "sibling_same_source": 3} and facts["contrasts_reanchored"] == 3 and facts["contrasts_own"] == 1
    c_comp = contrasts[X.run_of(bt, COMPOSED)["run_id"]]
    assert c_comp["anchor_run_id"] == none_p["run_id"] and c_comp["extra"]["cross_standup"] is False and c_comp["extra"]["anchor_view_legacy"] == TAG_P
    assert c_comp["anchor_kind"] == "same_standup_none" and c_comp["extra"]["anchor_selection"]["state"] == "own_current"
    for k in (RELU, PROMPTED, ARDITI):
        c = contrasts[X.run_of(bt, k)["run_id"]]
        assert c["anchor_run_id"] == none_p["run_id"] and c["anchor_kind"] == "sibling_none" and c["extra"]["cross_standup"] is True and c["extra"]["anchor_view_legacy"] == TAG_P
        assert c["extra"]["anchor_selection"]["state"] == "sibling_same_source" and c["extra"]["anchor_selection"]["own_reason"] == "none_declared" and c["extra"]["cross_source"] is False
        assert c["n_pairs"] > 0 and c["computed_by"] == "registry" and c["delta_method"] == B.CONTRAST_METHOD and "a row of this batch" in c["extra"]["pairing"]
        assert c["extra"]["own_pairing"]["state"] == B.ANCHOR_NOT_YET_RUN and f"{TAG}/none" in c["extra"]["own_pairing"]["why"] and c["extra"]["anchor_rule"] == B.CURRENCY_RULE
        blk = X.run_of(bt, k)["extra"]["anchor"]
        assert blk["state"] == "contrasted" and blk["anchor_run_id"] == none_p["run_id"] and blk["anchor_view_legacy"] == TAG_P and blk["reanchored"] is True
        assert blk["own_anchor_state"] == B.ANCHOR_NOT_YET_RUN and blk["cross_standup"] is True and "own_none_state" not in blk
        assert facts["contrasts"][f"{TAG}|{k}"]["reanchored"] is True and facts["contrasts"][f"{TAG}|{k}"]["own_standup"] is False


def test_a_paired_none_the_launch_set_names_but_has_not_run_leaves_the_composed_row_on_the_fallback_and_names_it(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, paired_none="not_run")
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and facts["n_not_run"] == 1, facts["refusals"]
    assert facts["not_run"][0]["tuple_id"] == f"{SEL_P}/{NONE}" and facts["not_run"][0]["class"] == "anchor" and facts["not_run"][0]["gate_verdict"] == "FAIL"
    cp = X.run_of(bt, COMPOSED)
    blk = cp["extra"]["anchor"]
    assert blk["state"] == "contrasted" and blk["anchor_run_id"] == X.run_of(bt, NONE)["run_id"] and blk["anchor_view_legacy"] == TAG and blk["cross_standup"] is True and blk["rule"] == B.ANCHOR_RULE
    assert blk["own_none_state"] == B.ANCHOR_NOT_YET_RUN == "anchor_not_yet_run" and f"{TAG_P}/none" in blk["own_none_note"] and f"{TAG}/none" in blk["own_none_note"]
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and "own_anchor" not in cp["extra"] and "standup_replicate_of_view" not in cp["extra"]
    c_comp = {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]
    assert c_comp["anchor_kind"] == "sibling_none" and c_comp["extra"]["cross_standup"] is True and c_comp["extra"]["own_none_state"] == "anchor_not_yet_run" and "did not file" in c_comp["extra"]["cross_standup_note"]
    assert c_comp["extra"]["anchor_view_legacy"] == TAG and "own_standup_note" not in c_comp["extra"] and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 96 / 198) < 1e-12
    assert "own_none_state" not in X.run_of(bt, RELU)["extra"]["anchor"] and facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["own_standup"] is False


def test_a_launch_set_filing_the_paired_none_as_composed_is_refused_by_the_grammar_and_the_composed_row_falls_back(tmp_path, seeds):
    def relabel(doc):
        for r in doc["evals"][EVAL]["launch"]:
            if r["view"] == SEL_P and r["cell"] == NONE:
                r["class"] = "composed"
    fx = _mirror(tmp_path, seeds, paired_none=True, launch_override=relabel)
    bt, facts = X.read(_adapter(fx, seeds))
    why = X.refused(facts, TAG_P, NONE)
    assert "class 'composed'" in why and "the key's grammar says 'anchor'" in why and facts["runs"] == 5
    cp = X.run_of(bt, COMPOSED)
    assert cp["extra"]["anchor"]["own_none_state"] == B.ANCHOR_REFUSED == "anchor_refused" and cp["extra"]["anchor"]["cross_standup"] is True
    assert cp["extra"]["anchor"]["anchor_run_id"] == X.run_of(bt, NONE)["run_id"] and cp["extra"]["cross_standup_anchor"] == f"{TAG}/none"
    c_comp = {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]
    assert c_comp["extra"]["own_none_state"] == "anchor_refused" and "anchor_refused" in c_comp["extra"]["cross_standup_note"] and c_comp["anchor_kind"] == "sibling_none"


# --------------------------------------------------------------------------------------------
# the decode policy: the reader records its resolution on extra and writes the producer placeholder; the match is the harvester's
# --------------------------------------------------------------------------------------------

def test_the_run_carries_the_producer_placeholder_and_the_readers_policy_resolution_on_extra(tmp_path, seeds):
    """`decode_policy_id` / `decode_policy_match` are DERIVED columns (decode_match.stamp_run_rows over v_run_decode_policy_match, every ingest cycle): a
    reader never stamps them. A matched row: the placeholder `no_policy`, no id, the protocol row named on extra, `decode_source` the policy's; the
    seed rule the instrument's (`none`), which is what the derived predicate compares to the policy row's `seed_rule`."""
    fx = _mirror(tmp_path, seeds)
    bt, facts = X.read(_adapter(fx, seeds))
    for okey in (NONE, RELU, PROMPTED, ARDITI, COMPOSED):
        r = X.run_of(bt, okey)
        assert r["decode_policy_match"] == "no_policy" and "decode_policy_id" not in r and r["decode_seed_rule"] == "none" == M.DECODE_SEED_RULE
        assert r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|bigcodebench|off|protocol@v1" and r["extra"]["decode_policy_reader_match"] == "exact by the stated sampling numbers"
        assert r["decode_source"] == "greedy" and r["field_confidence_json"]["decode_source"]["source"].startswith("the registered policy row qwen35_4b_it|bigcodebench|off|protocol@v1")


def test_a_row_whose_sent_numbers_match_no_policy_row_keeps_the_placeholder_and_names_why(tmp_path, seeds, monkeypatch):
    fx = _mirror(tmp_path, seeds)
    ad = _adapter(fx, seeds)
    monkeypatch.setattr(ad, "_match_policy", lambda model_id, block: (None, "synthetic: the sent numbers differ from every registered row"))
    bt, facts = X.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 5
    for okey in (NONE, RELU, COMPOSED):
        r = X.run_of(bt, okey)
        assert r["decode_policy_match"] == "no_policy" and "decode_policy_id" not in r and r["extra"]["decode_policy_ref_legacy"] is None
        assert r["extra"]["decode_policy_reader_match"] == "synthetic: the sent numbers differ from every registered row" and r["decode_source"] == "greedy", "the instrument's decode_source stands"
        assert facts["rows"][f"{TAG_P if okey == COMPOSED else TAG}|{okey}"]["decode_policy"] is None


# --------------------------------------------------------------------------------------------
# the launch set's ROW TABLE (the census of 2026-09-23): launch + carried
# --------------------------------------------------------------------------------------------

def test_the_launch_sets_row_table_is_launch_plus_carried(tmp_path, seeds):
    """The census names this pass's rows under `launch` and the pass-1 rows under `carried`; a reader of `launch` alone refused
    every carried row ("the launch set names no row"). The row table is both lists; the census's `census_status` and anchor declaration ride the run."""
    fx = _mirror(tmp_path, seeds, paired_none=True, launch_layout="carried")
    ad = _adapter(fx, seeds)
    bt, facts = X.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and facts["n_not_run"] == 0, facts["refusals"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    assert X.run_of(bt, RELU)["extra"]["launch_set"]["census_status"] == "carry" and runs[(TAG, NONE)]["extra"]["launch_set"]["census_status"] == "carry"
    assert runs[(TAG_P, NONE)]["extra"]["launch_set"]["census_status"] == "run" and runs[(TAG_P, NONE)]["extra"]["launch_set"]["anchor_tuple"] is None
    cp = X.run_of(bt, COMPOSED)
    assert cp["extra"]["launch_set"]["anchor_tuple"] == f"{SEL_P}/{NONE}" and cp["extra"]["launch_set"]["anchor_fallback"] == f"{SEL}/{NONE}" and X.run_of(bt, RELU)["extra"]["launch_set"]["anchor_tuple"] == f"{SEL}/{NONE}"
    assert {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]["anchor_run_id"] == runs[(TAG_P, NONE)]["run_id"]
    # the source state names the lists it read; the first launch set's shape keeps its path
    inputs = ad.discover()
    ls = [r for r in ad.source_states(X.ctx(), inputs) if r.input_uri.startswith(os.path.join(fx["root"], B.RECORDS_DIR, B.LAUNCH_SET_FILE))]
    assert len(ls) == 1 and ls[0].json_path == f"$.evals.{EVAL}.{{launch,carried}}" and len(ls[0].current_members) == 6, "carried_late is empty, so the path names the lists that hold rows"
    fx0 = _mirror(tmp_path, seeds, root=str(tmp_path / "first_shape"))
    ad0 = _adapter(fx0, seeds)
    ls0 = [r for r in ad0.source_states(X.ctx(), ad0.discover()) if r.input_uri.startswith(os.path.join(fx0["root"], B.RECORDS_DIR, B.LAUNCH_SET_FILE))]
    assert ls0[0].json_path == f"$.evals.{EVAL}.launch" and len(ls0[0].current_members) == 5
    assert set(X.run_of(bt, RELU)["extra"]["launch_set"]) <= {"record", "tuple_id", "class", "pin_key", "object_sha256_expected", "block", "table_index", "census_status", "launch_tag_legacy", "anchor_tuple", "anchor_fallback"}


def test_a_launch_set_whose_launch_list_is_empty_still_names_the_carried_rows(tmp_path, seeds):
    fx = _mirror(tmp_path, seeds, launch_layout="carried")
    bt, facts = X.read(_adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and all(r["extra"]["launch_set"]["census_status"] == "carry" for r in bt["run"])
