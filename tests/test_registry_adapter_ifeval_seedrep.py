"""The `ifeval_seedrep` reader on the SYNTHETIC mirror of `_ifeval_seedrep_fixtures` (invented item ids, shas and Modal ids, `TESTHOST`, no generation
text): one seed tag with its unsteered anchor, a fitted ReLU-steer object escalated to 4,096 and stopped there, a fitted LoRA object escalated to
8,192, a prompted key, the Arditi all-positions object, and the paired tag's composed difference_relu object escalated to 8,192.

What it pins:

* THE COMPLETION-CAP MERGE: the final value is the DRAW-WISE merge — each cap step's re-drawn draws replace their truncated
  shallower draws, a loop is never re-drawn, everything is recounted over the 1,082 draws; asserted against numbers computed by hand from the fixture's
  spec, including the row whose replaced draw had PASSED at the base (an additive merge would count it twice);
* THE THREE NUMBERS on every row: the VALUE (`prompt_level_strict` = strict passes / scored draws under the merge computation), the BOUND (passes /
  1,082, a blank computation id, the interval closed because every draw is scored), the at-cap DRAWS at the final state as `run_degeneration
  at_token_cap_finish_reason` (unit `draws`) with the loop / ceiling / not_escalated classes; the loose rate as the secondary `prompt_level_loose`;
* THE INSTRUMENT: every row on `ifeval_inspect_evals_sampled_vendor_preset_ladder` bound to qwen35_4b_it at the cap the ladder REACHED
  (`token_cap_tokens`, `token_cap_rung_index`), thinking OFF, the vendor-preset row matched to the registered policy; `decode_seed_rule per_item`,
  `seed_base NULL`, `trial_index 0` on every row;
* THE DISPATCH RULE: a row files under a done record attesting `lora_native`; a rate-limited attestation read is judged by the store; a failed record
  or an absent record for a present store refuses; the parked `*.failed-*` / `*.incomplete-*` attempt-history roots are never inputs; a paired tag's
  `none` may carry its base-cap record under `records/dispatch_paired_none/` (its own dispatch_state.json), the pass-1 root winning when both name it;
* THE MIRROR OF 2026-09-23 15:24Z: the launch set's ROW TABLE is `launch` (this pass's rows, census run) + `carried` (the pass-1 rows, census carry);
  the paired tags' `none` rows sit outside the ladder plan and file only as base-only rows below the declared trigger, their at-cap draws unclassified
  (no loop reading invented); an unplanned row with a cap-step store, or a share above the trigger, refuses;
* THE LADDER'S DECIDING RECORDS: a cap step's store present iff the plan promoted the row; the plan's merged reading equals the reader's;
* THE ANCHORS: every steered / prompted row against the seed tag's `none` at the SAME final state, per draw; the composed row across standups
  while the paired tag has no `none` of its own; the Arditi row declared unpaired;
* THE PAIRED TAG'S OWN `none`: launch-set class `anchor`, a run of the paired tag through the same
  completion-cap merge (a `none` never escalates: its base is its final), no anchor block of its own; the composed row then pairs against IT per draw
  (`same_standup_none`, `cross_standup` false, `anchor_view_legacy` the paired tag) and the seed tag's rows never do; a paired `none` the launch set
  names but that has not run (or is filed under another class) leaves the composed row on the cross-standup fallback with
  `extra.anchor.own_none_state` naming why;
* REFUSED BY NAME (the row, never the group): a per-store three-number row disagreeing; a store attesting another sha; a launch set expecting another
  sha; a failed / absent dispatcher record; a missing cap-step store the plan promoted; a step store without its base; a replicate seed block; a
  RESULT that is not COMPLETE; a plan whose merged reading disagrees;
* end to end: the runs reach `ingested` through the harvester's rule-scored completion with no state event from the reader and nothing blocked.
"""
from __future__ import annotations

import hashlib
import os

import pytest

import _b1_fixtures as F
import _ifeval_seedrep_fixtures as X
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry.adapters import _items_common as IC
from sisteer.registry.adapters import ifeval_seedrep as M
from sisteer.registry.adapters import lineage_common as LC

TAG, TAG_P, SEL, SEL_P, SEED_N = X.TAG, X.TAG_P, X.SEL, X.SEL_P, X.SEED_N
NONE, RELU, LORA, PROMPTED, ARDITI, COMPOSED = X.NONE, X.RELU, X.LORA, X.PROMPTED, X.ARDITI, X.COMPOSED
N = M.N_DRAWS


@pytest.fixture(scope="module")
def seeds():
    return LC.Seeds.locate()


# --------------------------------------------------------------------------------------------
# layout and identity
# --------------------------------------------------------------------------------------------

def test_discover_groups_the_seed_tag_with_its_paired_tag_and_never_lists_the_failed_attempt_roots(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds)
    ad = X.adapter(fx, seeds)
    inputs = ad.discover()
    names = [os.path.basename(i.uri) for i in inputs]
    for shared in (M.LAUNCH_SET_FILE, M.PINS_FILE, M.MATRIX_FILE, M.THREE_NUMBERS_FILE):
        assert f"{shared}#{TAG}" in names, shared
    assert names.count(f"{M.DISPATCH_STATE_FILE}#{TAG}") == 3 and names.count(f"{M.PLAN_FILE}#{TAG}") == 2 and names.count(f"{M.PROMOTED_FILE}#{TAG}") == 2
    assert names.count("manifest.json") == 11 and names.count("RESULT") == 11 and names.count(M.GEOMETRY_FILE) == 11, "1 + 2 + 3 + 1 + 1 + 3 stores"
    assert names.count(M.OUTCOMES_FILE) == 33 and names.count(M.SUBSET_FILE) == 5 and names.count(M.LADDER_STATE_FILE) == 5 and names.count(M.CAP_LADDER_FILE) == 5
    assert names.count("rung2048.json") == 6 and names.count("rung4096.json") == 3 and names.count("rung8192.json") == 2
    assert not any(".failed-D" in i.uri for i in inputs), "the attempt-1 roots are history, never inputs"
    assert names.count(M.CONTROLS_SUMMARY_FILE) == 6
    assert {ad.comparison_group(i) for i in inputs} == {TAG}, "the paired tag reads in its seed tag's group (GROUP_RULE)"
    roles = {os.path.basename(i.uri): i.role for i in inputs}
    assert roles["rows.jsonl"] == "rows" and roles["gen_stats.json"] == "completion" and roles["RESULT"] == "log"
    assert M.seed_tag_of(TAG_P) == TAG and M.tag_parts(TAG_P) == (SEED_N, True) and M.step_dir(4096) == "vendor_preset__rung4096" and M.step_dir(2048) == M.BASE_DIR
    assert M.configuration_id(TAG, "none", 8192) == f"{TAG}/none/rung8192" and M.rel_subset(TAG_P, "x--y", 4096) == f"rung4096/{TAG_P}/x--y/rung_subset.json"


def test_run_ids_key_on_the_tag_prefixed_base_store_name_not_the_root_path(tmp_path, seeds):
    fx1 = X.mirror(tmp_path, seeds, root=str(tmp_path / "a" / "ifeval_seedrep"))
    fx2 = X.mirror(tmp_path, seeds, root=str(tmp_path / "b" / "elsewhere"))
    bt1, f1 = X.read(X.adapter(fx1, seeds))
    bt2, _f2 = X.read(X.adapter(fx2, seeds))
    assert f1["n_refusals"] == 0, f1["refusals"]
    ids1 = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r["run_id"] for r in bt1["run"]}
    ids2 = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r["run_id"] for r in bt2["run"]}
    assert ids1 == ids2 and len(ids1) == 6
    inst = str(seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")["instrument_id"])
    key = fx1["keys"][(LORA, 2048)]
    assert ids1[(TAG, LORA)] == I.run_id(LC.CAMPAIGN, f"{TAG}/{key}", LORA, M.EVAL_ID, inst) == M.run_id_for(TAG, key, LORA, inst)
    r = X.run_of(bt1, LORA)
    assert r["producer_run_key_legacy"] == f"{TAG}/{key}" and r["producer_object_key_legacy"] == LORA and r["instrument_id"] == inst, "the BASE store's name keys the merged run"
    assert X.run_of(bt1, COMPOSED)["producer_run_key_legacy"] == f"{TAG_P}/{fx1['keys'][(COMPOSED, 2048)]}" and X.run_of(bt1, COMPOSED)["view_legacy"] == TAG_P


# --------------------------------------------------------------------------------------------
# the completion-cap merge, hand-computed
# --------------------------------------------------------------------------------------------

def test_the_completion_cap_merge_replaces_draws_item_wise_and_recounts_without_double_counting(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds)
    bt, facts = X.read(X.adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and facts["n_not_run"] == 0, facts["refusals"]
    # LORA: base 300 passes of 1,082 draws, 40 at cap (10 loops, 30 re-drawn at 4,096); at 4,096: 20 pass, 4 fail, 6 at cap (2 loops, 4 re-drawn at 8,192);
    # at 8,192: 2 pass, 1 fails, 1 still at cap (the ceiling). Merged: 300 + 20 + 2 = 322 passes; at cap 10 + 2 + 1 = 13 (12 loops, 1 ceiling)
    e = X.expected(LORA)
    assert (e["passes"], e["final_cap"], e["n_at_cap"], e["n_loops"], e["n_ceiling"], e["n_not_escalated"]) == (322, 8192, 13, 12, 1, 0)
    r = X.run_of(bt, LORA)
    tn = r["extra"]["three_numbers"]
    assert (tn["strict_passes"], tn["n_scored_draws"], tn["acc_over_scored"]) == (322, N, 322 / N) and tn["chain"] == [2048, 4096, 8192] and tn["final_cap"] == 8192
    assert (tn["n_at_cap_draws_final"], tn["n_loops"], tn["n_ceiling"], tn["n_not_escalated"], tn["n_at_cap_draws_base"]) == (13, 12, 1, 0, 40)
    assert r["token_cap_tokens"] == 8192 and r["token_cap_rung_index"] == 2 and r["token_cap_ladder_json"] == [2048, 4096, 8192] and r["extra"]["escalation_state"] == "final_at_ceiling"
    mb = r["extra"]["completion_cap_merge"]
    assert mb["draws_from_cap"] == {"2048": N - 30, "4096": 26, "8192": 4} == {str(c): n for c, n in e["draws_from_cap"].items()}
    assert len(mb["replaced_draws"]["4096"]) == 30 and len(mb["replaced_draws"]["8192"]) == 4 and mb["ceiling_draws"] == [["519", 1]] and mb["not_escalated_draws"] == []
    assert len(mb["loop_draws_by_cap"]["2048"]) == 10 and len(mb["loop_draws_by_cap"]["4096"]) == 2 and "cap_ladder.json" in mb["loop_classification_by_cap"]["2048"]
    assert mb["plan_decisions"]["4096"]["exit"] == "escalate" and mb["plan_decisions"]["4096"]["n_at_cap"] == 40 and mb["plan_decisions"]["8192"]["n_at_cap"] == 16
    # the per-store readings are the producer's (a step store's acc is the mean over ITS items of each item's mean over ITS draws)
    pc = tn["per_cap"]
    assert pc["2048"]["value"] == pc["2048"]["producer_value"] == 300 / N and pc["2048"]["n_at_cap_draws"] == 40 and pc["2048"]["draws_in_final"] == N - 30
    assert pc["4096"]["n_draws"] == 30 and pc["4096"]["n_items"] == 15 and pc["4096"]["value"] == pc["4096"]["producer_value"] == 10 / 15 and pc["4096"]["n_at_cap_draws"] == 6
    assert pc["8192"]["n_draws"] == 4 and pc["8192"]["value"] == pc["8192"]["producer_value"] == 0.5 and pc["8192"]["n_at_cap_draws"] == 1
    # RELU: the re-drawn draw (514, 0) PASSED at the base (at the cap) and fails at 4,096 — the merge counts it once, as its final draw
    e2 = X.expected(RELU)
    assert (e2["passes"], e2["base_value"] * N, e2["final_cap"], e2["n_at_cap"], e2["n_loops"], e2["n_not_escalated"]) == (520, 501, 4096, 8, 6, 2)
    r2 = X.run_of(bt, RELU)
    t2 = r2["extra"]["three_numbers"]
    assert t2["strict_passes"] == 520 and t2["per_cap"]["2048"]["value"] == 501 / N and t2["per_cap"]["4096"]["value"] == 10 / 13, "an additive merge would read 521"
    assert r2["extra"]["completion_cap_merge"]["n_replaced_that_passed_before"] == 1 and r2["token_cap_tokens"] == 4096 and r2["token_cap_rung_index"] == 1
    assert r2["extra"]["escalation_state"] == "final_at_step" and r2["extra"]["completion_cap_merge"]["plan_decisions"]["8192"]["exit"] == "below_threshold"
    assert (t2["n_at_cap_draws_final"], t2["n_loops"], t2["n_not_escalated"], t2["n_ceiling"]) == (8, 6, 2, 0) and sorted(r2["extra"]["completion_cap_merge"]["not_escalated_draws"]) == [["514", 0], ["514", 1]]
    assert "the ladder plan's decision" in r2["extra"]["completion_cap_merge"]["loop_classification_by_cap"]["4096"]
    # NONE: base only — final == base, the at-cap draws are not_escalated, the value equals gen_stats' acc
    r0 = X.run_of(bt, NONE)
    t0 = r0["extra"]["three_numbers"]
    assert (t0["strict_passes"], t0["final_cap"], t0["chain"], t0["n_at_cap_draws_final"], t0["n_not_escalated"], t0["n_loops"]) == (800, 2048, [2048], 5, 5, 0)
    assert r0["token_cap_tokens"] == 2048 and r0["token_cap_rung_index"] == 0 and r0["extra"]["escalation_state"] == "final_at_base"
    # the pure merge refuses a re-drawn draw that had terminated
    with pytest.raises(ValueError, match="was not at the cap"):
        M.completion_cap_merge({("1", 0): {"at_cap": False, "value": 1.0}}, [(4096, {("1", 0)}, {("1", 0): {"at_cap": False, "value": 0.0}})])
    with pytest.raises(ValueError, match="not the re-drawn set"):
        M.completion_cap_merge({("1", 0): {"at_cap": True, "value": 0.0}}, [(4096, {("1", 0)}, {("2", 0): {"at_cap": False, "value": 0.0}})])


def test_the_three_numbers_are_filed_on_the_merged_draws_and_the_items_are_draws(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds)
    bt, _facts = X.read(X.adapter(fx, seeds))
    r = X.run_of(bt, LORA)
    value, bound, loose = X.metrics_of(bt, r["run_id"])
    assert value["metric_id"] == "prompt_level_strict" and value["value"] == 322 / N and (value["k"], value["n"]) == (322, N) and value["computed_by"] == "registry" and value["ci_method"] == "wilson_95"
    assert value["computation_id"] == M.merge_computation_id() == I.computation_id(M.MERGE_ALGORITHM, M.MERGE_VERSION, M.MERGE_SEED_PROVENANCE) and value["extra"]["equals_gen_stats_acc"] is False
    assert M.MERGE_ALGORITHM == "sisteer.registry.adapters.ifeval_seedrep.completion_cap_merge" and value["extra"]["merged_has_producer_record"] is False
    assert bound["metric_id"] == "prompt_level_strict" and bound["value"] == 322 / N and (bound["k"], bound["n"]) == (322, N) and bound["computed_by"] == "producer" and bound["computation_id"] == ""
    assert bound["producer_metric_spelling"] == M.BOUND_SPELLING == "bound" and bound["ci_method"] == M.BOUND_CI_METHOD and bound["ci_lo"] == bound["ci_hi"] == 322 / N, "every draw is scored: the interval closes"
    assert "secondary" in bound["extra"]["metric_role_expected"] and set(bound["extra"]["producer_spellings"]) == {"bound@2048", "bound@4096", "bound@8192"}
    assert loose["metric_id"] == "prompt_level_loose" and loose["computation_id"] == M.merge_computation_id() and loose["n"] == N and loose["value"] >= value["value"]
    comps = {x["computation_id"] for x in bt["computation"]}
    assert value["computation_id"] in comps and len(bt["computation"]) == 6
    # items: 1,082 rows per run, one per draw (sample_index = the trial), every one scored, the at-cap draws flagged and classed
    items = [i for i in bt["item_result"] if i["run_id"] == r["run_id"]]
    assert len(items) == N and {(i["item_id"], i["sample_index"]) for i in items} == set(X.DRAWS) and all(i["trial_index"] == 0 and i["judgment_id"] == "" and i["metric_id"] == "prompt_level_strict" for i in items)
    assert all(i["value"] in (0.0, 1.0) and i["label"] is None for i in items) and sum(i["value"] for i in items) == 322
    classes = {}
    for i in items:
        c = i["extra"].get("at_cap_class")
        classes[c] = classes.get(c, 0) + 1
    assert classes == {None: N - 13, "loop": 12, "ceiling": 1}, classes
    by_cap = {}
    for i in items:
        by_cap[i["extra"]["final_cap"]] = by_cap.get(i["extra"]["final_cap"], 0) + 1
    assert by_cap == {2048: N - 30, 4096: 26, 8192: 4}
    assert sum(1 for i in items if i["extra"]["replaced_value"] is not None) == 30 and all(i["extra"]["replaced_at_cap"] is True for i in items if i["extra"]["final_cap"] != 2048)
    flags = [f for f in bt["item_flags"] if f["run_id"] == r["run_id"]]
    assert len(flags) == N and sum(f["at_token_cap"] for f in flags) == 13 and all(f["finish_reason"] == "max_tokens" for f in flags if f["at_token_cap"] == 1)
    assert sum(1 for f in flags if f["degenerate_rep_ratio_loop"] == 1) == 12 and sum(1 for f in flags if f["degenerate_rep_ratio_loop"] is None) == N - 12, "the ceiling draw carries no loop classification"
    assert all(f["generation_artifact_id"] is not None and f["generation_offset"] is not None for f in flags), "every draw points at its store's rows.jsonl row"
    # the run-level shares over the 1,082 draws
    deg = {d["measure"]: d for d in bt["run_degeneration"] if d["run_id"] == r["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"], deg[M.RAW_AT_CAP_MEASURE]["denominator_unit"], deg[M.RAW_AT_CAP_MEASURE]["computed_by"]) == (13, N, "draws", "registry")
    assert deg[M.RAW_AT_CAP_MEASURE]["extra"]["classes"] == {"loop": 12, "ceiling": 1, "not_escalated": 0}
    assert (deg[M.LOOP_MEASURE]["numerator"], deg[M.LOOP_MEASURE]["denominator_unit"], deg[M.LOOP_MEASURE]["extra"]["n_unclassified_at_ceiling"]) == (12, "draws", 1)
    assert (deg[M.COHERENT_MEASURE]["numerator"], deg[M.COHERENT_MEASURE]["extra"]["classes"]) == (1, {"ceiling": 1, "not_escalated": 0})
    assert set(deg) == set(IC.DECLARED_MEASURES) | {M.COHERENT_MEASURE} and all(deg[m]["measurement_state"] == "not_measured" for m in ("classifier", "contentless", "zlib_collapse", "union5", IC.AT_CAP_RETOKENIZED))
    # the base-only anchor: equals gen_stats' acc; n_items_judged is the items
    n = X.run_of(bt, NONE)
    v_n, b_n, _l = X.metrics_of(bt, n["run_id"])
    assert v_n["value"] == 800 / N and v_n["extra"]["equals_gen_stats_acc"] is True and n["n_items_judged"] == M.N_ITEMS == 541 and n["n_items_planned"] == 541 and b_n["ci_lo"] == b_n["ci_hi"]


def test_the_run_row_carries_the_attested_identity_the_dispatch_verdicts_and_its_own_instrument(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds)
    bt, facts = X.read(X.adapter(fx, seeds))
    r = X.run_of(bt, LORA)
    inst = seeds.instrument_row(M.INSTRUMENT_SEED, M.EVAL_ID, model_id="qwen35_4b_it")
    assert r["model_id"] == "qwen35_4b_it" and r["object_id"] == X.OBJ_L and r["view_legacy"] == TAG and r["lane_source_legacy"] == M.SOURCE == "ifeval_seedrep" and r["campaign"] == LC.CAMPAIGN
    assert r["instrument_id"] == inst["instrument_id"] and r["eval_id"] == "ifeval" and r["item_set_id"] == "ifeval_541" and r["harness"] == "inspect_evals" and r["scoring_rule"] == "ifeval_checker"
    assert r["thinking_mode"] == "off" and r["serving_path"] == "lora_native" and r["applied_positions_mode"] == "all" and r["transfer_kind"] == "base_to_instruct"
    assert r["decode_sent_json"] == X.DECODE and r["decode_temperature_sent"] == 0.7 and r["decode_top_k_sent"] == 20 and r["decode_source"] == "model_card"
    assert r["instrument_match"] == "match" and r["extra"]["decode_policy_ref_legacy"] == "qwen35_4b_it|ifeval|off|vendor_preset@v1"
    # no per-seed rule — the documented default on EVERY row; the fixed IFEval base rides extra
    assert r["seed_base"] is None and r["decode_seed_rule"] == "per_item" and r["trial_index"] == 0 and r["extra"]["decode_seed_policy"] == "default" and r["extra"]["fit_seed"] == SEED_N
    assert r["extra"]["ifeval_seed_block"]["seed_base"] == 20260909 and r["extra"]["decode_seed_rule_text"] == X.SEED_RULE and r["extra"]["seed_axis"]["rule"] == M.SEED_AXIS_RULE
    assert r["field_confidence_json"]["seed_base"]["confidence"] == "derived" and r["field_confidence_json"]["trial_index"]["confidence"] == "derived"
    assert r["code_sha"] == X.TREE and r["model_revision"] == X._REVISION and r["modal_call_id"].startswith("fc-") and r["job_name"] == f"tinycap-synthetic-{X.slug_of(LORA)}"
    assert r["substrate"] == "modal" and r["substrate_source"] == "derived" and r["context_window_tokens"] == 2048 + 480 and r["served_max_num_seqs"] == 207
    assert r["generated_utc"] == "2026-09-23T08:31:24.000Z" and r["started_utc"] == "2026-09-23T04:45:12.000Z" and r["queued_utc"] == r["started_utc"], "generated at the FINAL store, started at the base"
    assert r["store_completion_artifact_uri"].endswith(os.path.join("vendor_preset__rung8192", "cells", fx["keys"][(LORA, 8192)], "gen_stats.json"))
    assert r["producer_session"] == "Overflow" and r["is_operating_point"] == 1 and r["strength_nominal"] == 1.0 and r["judge_id"] is None and r["n_items_judged"] == 541
    assert r["extra"]["launch_set"]["pin_key"] == X.STAGED_L and r["extra"]["launch_set"]["object_sha256_expected"] == X.OBJ_L and r["extra"]["launch_set"]["class"] == "arm"
    d = r["extra"]["dispatch"]
    assert set(d) == {"2048", "4096", "8192"} and all(v["verdict"] == "record" and v["state"] == "done" and v["serving_path_attested"] == "lora_native" and v["rate_limited_read"] is False for v in d.values())
    assert d["4096"]["configuration_id"] == f"{TAG}/{X.slug_of(LORA)}/rung4096" and d["8192"]["dispatch_state_lists_done"] is True
    assert r["extra"]["failed_dispatch_roots_never_read"] == ["dispatch_rung4096.failed-D205"] == facts["failed_dispatch_roots_never_read"]
    assert r["extra"]["standup"]["2048"]["max_lora_rank"] == 8 and "never compared" in r["extra"]["max_lora_rank_rule"] and r["extra"]["matrix"]["job_id"].startswith("ife-")
    assert r["extra"]["store_uri_by_cap"]["8192"].endswith("vendor_preset__rung8192") and r["extra"]["harness_config_hash_by_cap"]["4096"].startswith("5d4141bb")
    assert "run_state" not in bt
    # the prompted key resolves through si_texts and its system_sha; the composed row is the paired tag's, paired_of s2
    p = X.run_of(bt, PROMPTED)
    assert p["object_id"] == str(seeds.si_object_id("safety_refusal")) and p["serving_path"] == "passthrough" and p["transfer_kind"] == "prompted_only" and p["trial_index"] == 0
    cp = X.run_of(bt, COMPOSED)
    assert cp["object_id"] == X.OBJ_P and cp["view_legacy"] == TAG_P and cp["trial_index"] == 0 and cp["extra"]["paired_of"] == SEL and cp["extra"]["roster_paired_of"] == SEL
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and cp["extra"]["launch_set"]["class"] == "composed" and cp["token_cap_tokens"] == 8192
    # the all-positions object: ONE fit on every seed tag -> trial 0, the operating point's strength columns and the declared role
    a = X.run_of(bt, ARDITI)
    assert a["object_id"] == X.OBJ_A and a["trial_index"] == 0 and a["seed_base"] is None and a["extra"]["fit_seed"] is None and a["extra"]["declared_role"] == M.ARDITI_ROLE
    assert (a["strength_coefficient"], a["strength_unit"], a["strength_nominal"], a["is_operating_point"], a["operating_point_id_legacy"]) == (0.3, "raw_coefficient", 1.0, 1, X.STAGED_A)
    assert a["transfer_kind"] == "instruct_direct" and a["extra"]["single_object"]["operating_point"] == "ar_c0p3" and M.SOURCE in a["extra"]["single_object"]["coexist_ruling_needed"]
    assert facts["rows"][f"{TAG}|{ARDITI}"]["max_lora_rank_recorded"] == {2048: 1} and a["extra"]["three_numbers"]["n_loops"] == 1 and a["extra"]["three_numbers"]["n_not_escalated"] == 2
    n = X.run_of(bt, NONE)
    assert n["object_id"] == "none" and n["serving_path"] == "passthrough" and n["extra"]["same_object_rows"]["twin_tags_legacy"] == ["qwen35_4b_s1", "qwen35_4b_s3"]


def test_every_row_pairs_with_the_seed_tags_none_at_the_same_final_state_per_draw(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds)
    bt, facts = X.read(X.adapter(fx, seeds))
    none_id = X.run_of(bt, NONE)["run_id"]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (RELU, LORA, PROMPTED, ARDITI, COMPOSED)}
    assert all(c["anchor_run_id"] == none_id and c["metric_id"] == "prompt_level_strict" and c["delta_method"] == "exact_mcnemar" for c in contrasts.values())
    assert {k: contrasts[X.run_of(bt, k)["run_id"]]["anchor_kind"] for k in (RELU, LORA, PROMPTED, ARDITI, COMPOSED)} == \
        {RELU: "same_standup_none", LORA: "same_standup_none", PROMPTED: "same_standup_none", ARDITI: "same_standup_none", COMPOSED: "sibling_none"}, "the D65 fallback is another standup's none"
    assert all(c["computation_id"] == M.merge_computation_id() and c["computed_by"] == "registry" and 0.0 <= c["delta_p"] <= 1.0 and c["n_pairs"] == N for c in contrasts.values())
    c_lora = contrasts[X.run_of(bt, LORA)["run_id"]]
    assert c_lora["extra"]["pair_unit"] == "draw (item, trial)" and c_lora["extra"]["row_final_cap"] == 8192 and c_lora["extra"]["anchor_final_cap"] == 2048, "the anchor never escalated: its base is its final"
    assert abs(c_lora["delta"] - 100.0 * (322 - 800) / N) < 1e-9 and c_lora["extra"]["cross_standup"] is False
    c_comp = contrasts[X.run_of(bt, COMPOSED)["run_id"]]
    assert c_comp["extra"]["cross_standup"] is True and c_comp["extra"]["row_tag_legacy"] == TAG_P and "cross_standup_note" in c_comp["extra"]
    c_ard = contrasts[X.run_of(bt, ARDITI)["run_id"]]
    assert c_ard["extra"]["declared_role"] == M.ARDITI_ROLE and "anchor_standup_note" in c_ard["extra"]
    r = X.run_of(bt, LORA)
    assert r["extra"]["anchor"]["state"] == "contrasted" and r["extra"]["anchor"]["anchor_run_id"] == none_id and r["extra"]["anchor"]["n_pairs"] == N and "anchor" not in X.run_of(bt, NONE)["extra"]
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["cross_standup"] is True and facts["contrasts"][f"{TAG}|{LORA}"]["row_final_cap"] == 8192
    # no paired `none` in the launch set at all: the standing anchor spelling, nothing of the own-anchor vocabulary (the live mirror's records stay byte-identical)
    cp = X.run_of(bt, COMPOSED)
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and "own_anchor" not in cp["extra"] and cp["extra"]["anchor"]["rule"] == M.ANCHOR_RULE and "own_none_state" not in cp["extra"]["anchor"]
    assert cp["extra"]["anchor"]["anchor_view_legacy"] == TAG and "own_none_state" not in c_comp["extra"] and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 800 / N) < 1e-12
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["own_standup"] is False and facts["contrasts"][f"{TAG}|{LORA}"]["own_standup"] is True


# --------------------------------------------------------------------------------------------
# the dispatch rule
# --------------------------------------------------------------------------------------------

def test_a_rate_limited_attestation_read_is_judged_by_the_store_and_files(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, rate_limited=[(COMPOSED, 8192)])
    bt, facts = X.read(X.adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6, facts["refusals"]
    d = X.run_of(bt, COMPOSED)["extra"]["dispatch"]["8192"]
    assert d["verdict"] == "store" and d["rate_limited_read"] is True and d["state"] == "failed" and d["status"] == "SERVING_PATH_MISMATCH" and d["serving_path_attested"] is None
    assert "rate limit" in d["judged_by"] and d["dispatch_state_lists_failed"] is True
    assert facts["rows"][f"{TAG_P}|{COMPOSED}"]["dispatch"][8192]["verdict"] == "store"


def test_a_failed_dispatcher_record_that_is_not_rate_limited_refuses_the_row_and_the_siblings_file(tmp_path, seeds):
    def bump(okey, cap, rec):
        if okey == LORA and cap == 4096:
            rec.update(state="failed", status="INCOMPLETE", serving_path_attested=None, error="driver rc 1: the store did not complete")
    fx = X.mirror(tmp_path, seeds, dispatch_override=bump)
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, LORA)
    assert "reads state 'failed' / status 'INCOMPLETE'" in why and "rung4096.json" in why and len(bt["run"]) == 5
    assert {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, RELU, PROMPTED, ARDITI, COMPOSED}


def test_a_present_store_without_a_dispatcher_record_is_refused_by_name(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, drop_dispatch=[(RELU, 4096)])
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, RELU)
    assert "no dispatcher record" in why and "PRESENT" in why and len(bt["run"]) == 5


def test_a_dispatcher_record_naming_another_object_sha_is_refused(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()

    def bump(okey, cap, rec):
        if okey == ARDITI:
            rec["object_sha256"] = other
    fx = X.mirror(tmp_path, seeds, dispatch_override=bump)
    _bt, facts = X.read(X.adapter(fx, seeds))
    assert "object_sha256" in X.refused(facts, TAG, ARDITI) and "is not the resolved object" in X.refused(facts, TAG, ARDITI)


# --------------------------------------------------------------------------------------------
# the ladder's deciding records and the stores
# --------------------------------------------------------------------------------------------

def test_a_promoted_cap_step_whose_store_is_missing_is_refused_by_name(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, drop_store=[(LORA, 8192)])
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, LORA)
    assert "promotes" in why and "8,192 store is absent" in why and len(bt["run"]) == 5


def test_a_cap_step_store_without_its_base_store_is_refused(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, drop_store=[(COMPOSED, 2048)])
    bt, facts = X.read(X.adapter(fx, seeds))
    assert "without its base store" in X.refused(facts, TAG_P, COMPOSED) and len(bt["run"]) == 5


def test_a_plan_whose_merged_reading_disagrees_with_the_readers_merge_is_refused(tmp_path, seeds):
    def bump(cap, plan, prom):
        if cap == 8192:
            for e in plan["planned"]:
                if e["cell_key"].endswith(f"/{LORA}"):
                    e["decision"]["n_at_cap"] += 1
    fx = X.mirror(tmp_path, seeds, plan_override=bump)
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, LORA)
    assert "the plan's merged reading before 8,192" in why and "(n_at_cap 17 of 1082 draws)" in why and "(16 of 1082)" in why and len(bt["run"]) == 5


def test_a_subset_that_is_not_the_source_stores_eligible_draws_is_refused(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds)
    import json
    p = os.path.join(fx["root"], M.rel_subset(TAG, X.slug_of(RELU), 4096))
    sub = json.load(open(p))
    sub["draws"] = sub["draws"][:-1]
    sub["n_draws"] -= 1
    sub["n_items"] = len({d[0] for d in sub["draws"]})
    X.write(p, sub)
    _bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, RELU)
    assert "item_subset" in why or "eligible_draws" in why, why


# --------------------------------------------------------------------------------------------
# refusals by name: identity, the producer's records, the store
# --------------------------------------------------------------------------------------------

def test_a_per_store_three_number_row_that_disagrees_refuses_the_row_naming_the_number(tmp_path, seeds):
    def bump(three):
        row = three["rows"][f"{TAG}/{X.slug_of(RELU)}/rung4096"]
        row["at_cap_n"] += 1
        row["value"] = 0.1234
    fx = X.mirror(tmp_path, seeds, three_override=bump)
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, RELU)
    assert M.THREE_NUMBERS_FILE in why and "rung4096" in why and "at_cap_n: producer" in why and "value: producer 0.1234" in why and "2 number(s)" in why and len(bt["run"]) == 5


def test_a_three_number_record_without_a_stores_row_refuses_the_row(tmp_path, seeds):
    def bump(three):
        del three["rows"][f"{TAG_P}/{X.slug_of(COMPOSED)}/rung8192"]
    fx = X.mirror(tmp_path, seeds, three_override=bump)
    _bt, facts = X.read(X.adapter(fx, seeds))
    assert "carries no row for" in X.refused(facts, TAG_P, COMPOSED)


def test_a_store_attesting_another_sha_than_the_pins_is_refused_and_the_siblings_file(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = X.mirror(tmp_path, seeds, attest_sha={LORA: other})
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, LORA)
    assert "fit_provenance.object_sha" in why and "no prefix of the pinned object" in why and len(bt["run"]) == 5


def test_a_launch_set_expecting_another_sha_than_the_pins_is_refused_by_name(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()

    def bump(doc):
        for r in doc["evals"][M.EVAL_ID]["launch"]:
            if r["cell"] == COMPOSED:
                r["object_sha256_expected"] = other
    fx = X.mirror(tmp_path, seeds, launch_override=bump)
    _bt, facts = X.read(X.adapter(fx, seeds))
    assert "the launch set expects object" in X.refused(facts, TAG_P, COMPOSED)


def test_mirrored_pins_naming_another_sha_than_the_harness_pins_are_refused(tmp_path, seeds):
    other = hashlib.sha256(b"another object").hexdigest()
    fx = X.mirror(tmp_path, seeds, mirrored_pins_sha={ARDITI: other})
    _bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG, ARDITI)
    assert "the launch set expects object" in why or "the mirrored pins" in why


def test_a_store_the_launch_set_does_not_declare_is_refused_by_name(tmp_path, seeds):
    def bump(doc):
        doc["evals"][M.EVAL_ID]["launch"] = [r for r in doc["evals"][M.EVAL_ID]["launch"] if r["cell"] != PROMPTED]
    fx = X.mirror(tmp_path, seeds, launch_override=bump)
    _bt, facts = X.read(X.adapter(fx, seeds))
    assert "the launch set names no row for" in X.refused(facts, TAG, PROMPTED)


def test_a_replicate_seed_block_on_a_store_is_refused(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, replicate_block=True)
    bt, facts = X.read(X.adapter(fx, seeds))
    assert facts["n_refusals"] == 6 and all("replicate block" in x and "D69" in x for x in facts["refusals"]) and "run" not in bt


def test_a_result_that_is_not_complete_on_a_cap_step_refuses_the_row(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, result_incomplete=[(RELU, 4096)])
    _bt, facts = X.read(X.adapter(fx, seeds))
    assert "RESULT's last line does not read result=COMPLETE" in X.refused(facts, TAG, RELU) and "4,096 store" in X.refused(facts, TAG, RELU)


def test_a_launch_set_row_with_no_store_files_nothing_and_rides_the_facts(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, not_run=(PROMPTED,))
    bt, facts = X.read(X.adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 5 and {r["campaign_cell_id_legacy"] for r in bt["run"]} == {NONE, RELU, LORA, ARDITI, COMPOSED}
    assert facts["n_not_run"] == 1 and facts["not_run"][0]["tuple_id"] == f"{SEL}/{PROMPTED}" and facts["not_run"][0]["rule"] == M.NOT_RUN_RULE and len(bt["run_contrast"]) == 4


def test_the_reader_never_consults_the_registry_for_the_object(tmp_path, seeds):
    """An object row absent from a connected registry is NOT a refusal here: the run is emitted on the attested identity and blocked_records holds it.
    (The connection is a primed frame -- the anchor currency reads `v_run_current` from it -- with an EMPTY `object` table beside it.)"""
    fx = X.mirror(tmp_path, seeds)
    ad = X.adapter(fx, seeds)
    frame = X.primed_frame(ad)
    frame.con.execute("CREATE TABLE object (object_id TEXT PRIMARY KEY)")
    bt, facts = X.read(ad, conn=frame.con)
    assert facts["n_refusals"] == 0 and X.run_of(bt, LORA)["object_id"] == X.OBJ_L and X.run_of(bt, ARDITI)["object_id"] == X.OBJ_A
    assert frame.con.execute("SELECT COUNT(*) FROM object").fetchone()[0] == 0 and len(bt["run_contrast"]) == 5


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
    fx = X.mirror(tmp_path, seeds)
    ad = X.adapter(fx, seeds)
    objects = [ad.resolve_object(RELU, SEL)[0], ad.resolve_object(LORA, SEL)[0], ad.resolve_object(COMPOSED, SEL_P)[0], ad.resolve_object(ARDITI, SEL, pin_key=X.STAGED_A)[0]]
    root = _registry_root(tmp_path, seeds, objects)
    # TWO passes (emission 3, the anchor currency; no provisional anchors): the first pass over the fresh registry files every run and PARKS every
    # comparison (the seed tag's `none` is emitted by that very read and the frame does not know it); the fold makes it current and the anchor decision
    # input moves, so the second pass re-reads the group under a new revision id (no file change, no bump) and files the five contrasts own_current
    rep0, ing0, ing0b = F.run_and_ingest(root, ad, now="2026-09-23T07:00:00.000Z")       # strictly before NOW: two revisions with one observed_utc have no order
    assert rep0.groups_refused == 0 and ing0.refused + ing0b.refused == 0, ([r.error for r in rep0.results], ing0.refusals, ing0b.refusals)
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast") == [(0,)] and F.q(root, "SELECT COUNT(*) FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [(6,)]
    parked = F.q(root, "SELECT json_extract(rm.extra_json, '$.delta_block.anchor.own_reason') FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                       "WHERE r.lane_source_legacy = ? AND r.object_id <> 'none' AND rm.computation_id <> '' AND rm.metric_id = ?", (M.SOURCE, M.HEADLINE_METRIC))
    assert parked == [("not_in_frame",)] * 5, parked
    rep, ing, ing2 = F.run_and_ingest(root, ad, now=X.NOW)
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1 and rep.anchor_decisions_changed == 1 and ing.refused + ing2.refused == 0, ([r.error for r in rep.results], ing.refusals, ing2.refusals)
    rows = F.q(root, "SELECT view_legacy, campaign_cell_id_legacy, lifecycle_state, n_items_planned, n_items_judged, token_cap_tokens, token_cap_rung_index, trial_index, seed_base, "
                     "judged_utc IS NOT NULL, ingested_utc IS NOT NULL FROM run WHERE lane_source_legacy = ? ORDER BY 1, 2", (M.SOURCE,))
    assert rows == [(TAG, ARDITI, "ingested", 541, 541, 2048, 0, 0, None, 1, 1),
                    (TAG, LORA, "ingested", 541, 541, 8192, 2, 0, None, 1, 1),
                    (TAG, RELU, "ingested", 541, 541, 4096, 1, 0, None, 1, 1),
                    (TAG, NONE, "ingested", 541, 541, 2048, 0, 0, None, 1, 1),
                    (TAG, PROMPTED, "ingested", 541, 541, 2048, 0, 0, None, 1, 1),
                    (TAG_P, COMPOSED, "ingested", 541, 541, 8192, 2, 0, None, 1, 1)], rows
    assert F.q(root, "SELECT COUNT(*) FROM run_state_event e JOIN run r ON r.run_id = e.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert F.q(root, "SELECT COUNT(*) FROM v_item_result_current i JOIN run r ON r.run_id = i.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(6 * N,)]
    m = F.q(root, "SELECT r.campaign_cell_id_legacy, rm.metric_id, rm.computation_id <> '', rm.value, rm.k, rm.n FROM run_metric rm JOIN run r ON r.run_id = rm.run_id "
                  "WHERE r.lane_source_legacy = ? AND r.campaign_cell_id_legacy = ? ORDER BY 2, 3 DESC", (M.SOURCE, LORA))
    assert m == [(LORA, "prompt_level_loose", 1, m[0][3], m[0][4], N), (LORA, "prompt_level_strict", 1, 322 / N, 322, N), (LORA, "prompt_level_strict", 0, 322 / N, 322, N)], m
    dg = F.q(root, "SELECT d.measure, d.measurement_state, d.numerator, d.denominator, d.denominator_unit FROM v_run_degeneration_current d JOIN run r ON r.run_id = d.run_id "
                   "WHERE r.campaign_cell_id_legacy = ? AND d.measure IN (?, ?, ?) ORDER BY 1", (LORA, M.RAW_AT_CAP_MEASURE, M.LOOP_MEASURE, M.COHERENT_MEASURE))
    assert dg == [(M.RAW_AT_CAP_MEASURE, "measured", 13, N, "draws"), (M.LOOP_MEASURE, "measured", 12, N, "draws"), (M.COHERENT_MEASURE, "measured", 1, N, "draws")], dg
    assert F.q(root, "SELECT COUNT(*) FROM run_contrast c JOIN run r ON r.run_id = c.run_id WHERE r.lane_source_legacy = ?", (M.SOURCE,)) == [(5,)]
    assert sorted(F.q(root, "SELECT DISTINCT json_extract(extra_json, '$.anchor_selection.state'), anchor_kind FROM run_contrast")) == [("own_current", "same_standup_none"), ("own_current", "sibling_none")]
    assert F.q(root, "SELECT COUNT(*) FROM run_metric rm JOIN run r ON r.run_id = rm.run_id WHERE r.lane_source_legacy = ? AND r.object_id <> 'none' AND rm.computation_id <> '' AND rm.metric_id = ? "
                     "AND json_extract(rm.extra_json, '$.delta_block') IS NULL AND json_type(rm.extra_json, '$.delta_block') = 'null'", (M.SOURCE, M.HEADLINE_METRIC)) == [(5,)], "a filed contrast states delta_block: null"
    rep2, _ing3, _ing4 = F.run_and_ingest(root, X.adapter(fx, seeds), now="2026-09-23T13:00:00.000Z")
    assert rep2.groups_refreshed == 0 and F.q(root, "SELECT DISTINCT lifecycle_state FROM run WHERE lane_source_legacy = ?", (M.SOURCE,)) == [("ingested",)]


# --------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------

def test_the_reader_is_registered_and_declares_its_source_owner_and_version():
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import base as AB
    reg = ADP.load_all()
    inst = reg[M.NAME]()
    assert isinstance(inst, M.IfevalSeedrepAdapter) and inst.covers_sources == (M.SOURCE,) and inst.owner == "Overflow" and M.SOURCE == "ifeval_seedrep"
    assert AB.emission_version_of(M.IfevalSeedrepAdapter) == "3" and inst.rng_scope == "none" and inst.roots == (M.ROOT_DEFAULT,)
    assert M.ROOT_DEFAULT.endswith(os.path.join("run-state", "full22", "ifeval_seedrep")) and M.LADDER == (2048, 4096, 8192) and M.CAMPAIGN == "sdfbase_full22"
    assert (M.N_ITEMS, M.K_TRIALS, M.N_DRAWS, M.IFEVAL_SEED_BASE) == (541, 2, 1082, 20260909) and M.EVAL_ID == "ifeval" and M.TASK == "inspect/ifeval"
    assert M.INSTRUMENT_SEED == "ifeval_inspect_evals_sampled_vendor_preset_ladder" and M.HEADLINE_METRIC == "prompt_level_strict" and M.SECONDARY_METRIC == "prompt_level_loose"
    assert M.DECODE_SEED_RULE == "per_item" and M.SUBSTRATE == "modal" and M.THINKING_SPELLING == "off" if hasattr(M, "THINKING_SPELLING") else True
    assert M.parse_result_line("tinycap-modal name=x rc=0 result=COMPLETE status=OK sha=" + "a" * 40 + " at=2026-09-23T04:17:11Z")["status"] == "OK"
    assert M.cap_of_store_dir("vendor_preset__rung8192") == 8192 and M.cap_of_store_dir("vendor_preset") == 2048 and M.cap_of_store_dir("thinking_card") is None
    assert M.draw_keys([["12", 1], ["3", 0]]) == {("12", 1), ("3", 0)}


# --------------------------------------------------------------------------------------------
# the paired tag's OWN `none`
# --------------------------------------------------------------------------------------------

def test_the_paired_tags_own_none_files_through_the_merge_as_its_anchor_and_the_composed_row_pairs_against_it_per_draw(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, paired_none=True)
    bt, facts = X.read(X.adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 7 and facts["n_not_run"] == 0, facts["refusals"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    none_s, none_p = runs[(TAG, NONE)], runs[(TAG_P, NONE)]
    # the paired `none`: launch-set class anchor, a run of the paired tag on the IFEval instrument through the same merge (base-only: its base is its final), trial 0
    assert none_p["extra"]["launch_set"]["class"] == "anchor" and none_p["object_id"] == "none" and none_p["instrument_id"] == none_s["instrument_id"] and none_p["serving_path"] == "passthrough"
    assert none_p["producer_run_key_legacy"] == f"{TAG_P}/{fx['paired_none_key']}" == f"{TAG_P}/{fx['keys'][(NONE, 2048)]}", "the base store key is the seed tag's, byte for byte; the tag prefix keeps the ids apart"
    assert none_p["run_id"] != none_s["run_id"] and none_p["trial_index"] == 0 and none_p["seed_base"] is None and none_p["view_legacy"] == TAG_P
    assert none_p["token_cap_tokens"] == 2048 and none_p["token_cap_rung_index"] == 0 and none_p["extra"]["escalation_state"] == "final_at_base" and none_p["extra"]["three_numbers"]["chain"] == [2048]
    assert none_p["extra"]["paired_of"] == SEL and none_p["extra"]["roster_paired_of"] == SEL and none_p["extra"]["standup_replicate_of_view"] == TAG and none_p["extra"]["anchor_rule"] == M.PAIRED_ANCHOR_RULE
    assert none_p["extra"]["same_object_rows"]["standup_replicate_of_view"] == TAG and "paired-view-none-anchors" in none_p["extra"]["same_object_rows"]["coexist_ruling_needed"]
    assert "anchor" not in none_p["extra"] and "cross_standup_anchor" not in none_p["extra"] and "own_anchor" not in none_p["extra"]
    v_p, b_p, _l = X.metrics_of(bt, none_p["run_id"])
    v_s, _b, _l = X.metrics_of(bt, none_s["run_id"])
    assert (v_p["k"], v_p["n"]) == (700, N) and (v_s["k"], v_s["n"]) == (800, N) and none_p["extra"]["three_numbers"]["n_at_cap_draws_final"] == 1
    # the contrasts: every seed row against the SEED `none`, the composed row against ITS OWN tag's `none` per draw; no `none` is a contrast's row
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (RELU, LORA, PROMPTED, ARDITI, COMPOSED)} and none_p["run_id"] not in contrasts
    for k in (RELU, LORA, PROMPTED, ARDITI):
        c = contrasts[X.run_of(bt, k)["run_id"]]
        assert c["anchor_run_id"] == none_s["run_id"] and c["extra"]["anchor_view_legacy"] == TAG and c["extra"]["cross_standup"] is False and abs(c["extra"]["acc_anchor_over_pairs"] - 800 / N) < 1e-12
        blk = X.run_of(bt, k)["extra"]["anchor"]
        assert blk["anchor_run_id"] == none_s["run_id"] and blk["anchor_view_legacy"] == TAG and blk["rule"] == M.ANCHOR_RULE and "own_none_state" not in blk
    cp = X.run_of(bt, COMPOSED)
    c_comp = contrasts[cp["run_id"]]
    assert c_comp["anchor_run_id"] == none_p["run_id"] and c_comp["anchor_kind"] == "same_standup_none" and c_comp["extra"]["anchor_view_legacy"] == TAG_P and c_comp["extra"]["cross_standup"] is False
    assert c_comp["n_pairs"] == N and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 700 / N) < 1e-12, "the PAIRED none's pass rate over the 1,082 draws, not the seed none's 800"
    assert c_comp["extra"]["row_final_cap"] == 8192 and c_comp["extra"]["anchor_final_cap"] == 2048 and c_comp["extra"]["pair_unit"] == "draw (item, trial)"
    assert "cross_standup_note" not in c_comp["extra"] and "own_none_state" not in c_comp["extra"] and c_comp["extra"]["anchor_rule"] == M.PAIRED_ANCHOR_RULE and TAG_P in c_comp["extra"]["own_standup_note"]
    blk = cp["extra"]["anchor"]
    assert blk["state"] == "contrasted" and blk["anchor_run_id"] == none_p["run_id"] and blk["anchor_view_legacy"] == TAG_P and blk["cross_standup"] is False and blk["anchor_final_cap"] == 2048
    assert blk["rule"] == M.PAIRED_ANCHOR_RULE and blk["own_standup_replicate_of_view"] == TAG and "own_none_state" not in blk
    assert cp["extra"]["own_anchor"] == f"{TAG_P}/none" and "cross_standup_anchor" not in cp["extra"] and cp["extra"]["standup_replicate_of_view"] == TAG and cp["extra"]["anchor_rule"] == M.PAIRED_ANCHOR_RULE
    assert facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["own_standup"] is True and facts["contrasts"][f"{TAG_P}|{COMPOSED}"]["anchor_view_legacy"] == TAG_P and facts["contrasts"][f"{TAG}|{LORA}"]["own_standup"] is True


def test_a_paired_none_the_launch_set_names_but_has_not_run_leaves_the_composed_row_on_the_fallback_and_names_it(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, paired_none="not_run")
    bt, facts = X.read(X.adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 6 and facts["n_not_run"] == 1, facts["refusals"]
    assert facts["not_run"][0]["tuple_id"] == f"{SEL_P}/{NONE}" and facts["not_run"][0]["class"] == "anchor" and facts["not_run"][0]["dispatch_state"] is None
    cp = X.run_of(bt, COMPOSED)
    blk = cp["extra"]["anchor"]
    assert blk["state"] == "contrasted" and blk["anchor_run_id"] == X.run_of(bt, NONE)["run_id"] and blk["anchor_view_legacy"] == TAG and blk["cross_standup"] is True and blk["rule"] == M.ANCHOR_RULE
    assert blk["own_none_state"] == M.ANCHOR_NOT_YET_RUN == "anchor_not_yet_run" and f"{TAG_P}/none" in blk["own_none_note"]
    assert cp["extra"]["cross_standup_anchor"] == f"{TAG}/none" and "own_anchor" not in cp["extra"] and "standup_replicate_of_view" not in cp["extra"]
    c_comp = {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]
    assert c_comp["anchor_kind"] == "sibling_none" and c_comp["extra"]["cross_standup"] is True and c_comp["extra"]["own_none_state"] == "anchor_not_yet_run" and "did not file" in c_comp["extra"]["cross_standup_note"]
    assert c_comp["extra"]["anchor_view_legacy"] == TAG and abs(c_comp["extra"]["acc_anchor_over_pairs"] - 800 / N) < 1e-12 and "own_none_state" not in X.run_of(bt, LORA)["extra"]["anchor"]


def test_a_launch_set_filing_the_paired_none_as_composed_is_refused_by_the_grammar_and_the_composed_row_falls_back(tmp_path, seeds):
    def relabel(doc):
        for r in doc["evals"][M.EVAL_ID]["launch"]:
            if r["view"] == SEL_P and r["cell"] == NONE:
                r["class"] = "composed"
    fx = X.mirror(tmp_path, seeds, paired_none=True, launch_override=relabel)
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG_P, NONE)
    assert "class 'composed'" in why and "the key's grammar says 'anchor'" in why and facts["runs"] == 6
    cp = X.run_of(bt, COMPOSED)
    assert cp["extra"]["anchor"]["own_none_state"] == M.ANCHOR_REFUSED and cp["extra"]["anchor"]["cross_standup"] is True and cp["extra"]["anchor"]["anchor_run_id"] == X.run_of(bt, NONE)["run_id"]
    cc = {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]
    assert cc["extra"]["own_none_state"] == "anchor_refused" and cc["anchor_kind"] == "sibling_none"


# --------------------------------------------------------------------------------------------
# the mirror of 2026-09-23 15:24Z: the paired none's own dispatch root, the rows outside the ladder plan, the launch + carried row table
# --------------------------------------------------------------------------------------------

def test_the_mirror_of_2026_09_23_files_the_paired_none_from_its_own_dispatch_root_outside_the_ladder_plan_and_the_carried_rows(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, paired_none=True, launch_layout="carried", paired_none_dispatch_root="paired", paired_none_planned=False, incomplete_attempt_root=True)
    ad = X.adapter(fx, seeds)
    bt, facts = X.read(ad)
    assert facts["n_refusals"] == 0 and facts["runs"] == 7 and facts["n_not_run"] == 0, facts["refusals"]
    assert facts["dispatch_roots_read"] == ["dispatch", "dispatch_paired_none", "dispatch_rung4096", "dispatch_rung8192"] and facts["rows_outside_ladder_plan"] == [f"{TAG_P}|{NONE}"]
    assert facts["failed_dispatch_roots_never_read"] == ["dispatch_rung4096.failed-D205", "dispatch_rung8192.incomplete-D999"]
    runs = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}
    none_p, none_s = runs[(TAG_P, NONE)], runs[(TAG, NONE)]
    # the ROW TABLE: the carried rows file (census carry) beside this pass's own launch (census run); the census's anchor declaration rides the block
    assert none_s["extra"]["launch_set"]["census_status"] == "carry" and X.run_of(bt, LORA)["extra"]["launch_set"]["census_status"] == "carry" and none_p["extra"]["launch_set"]["census_status"] == "run"
    cp = X.run_of(bt, COMPOSED)
    assert cp["extra"]["launch_set"]["anchor_tuple"] == f"{SEL_P}/{NONE}" and cp["extra"]["launch_set"]["anchor_fallback"] == f"{SEL}/{NONE}" and none_p["extra"]["launch_set"]["anchor_tuple"] is None
    inputs = ad.discover()
    ls = [r for r in ad.source_states(X.ctx(), [i for i in inputs if ad.comparison_group(i) == TAG]) if r.input_uri.startswith(os.path.join(fx["root"], M.RECORDS_DIR, M.LAUNCH_SET_FILE))]
    assert len(ls) == 1 and ls[0].json_path == f"$.evals.{M.EVAL_ID}.{{launch,carried}}" and len(ls[0].current_members) == 7
    # the dispatch record from records/dispatch_paired_none/, judged by THAT root's dispatch_state.json; every other row from the cap's live root
    d = none_p["extra"]["dispatch"]["2048"]
    assert d["root"] == M.DISPATCH_HEAD_PAIRED_NONE == "dispatch_paired_none" and d["roots_considered"] == ["dispatch", "dispatch_paired_none"] and d["other_root_records"] == []
    assert d["verdict"] == "record" and d["dispatch_state_lists_done"] is True and d["record"].endswith(f"dispatch_paired_none/dispatch/{TAG_P}/{NONE}/rung2048.json")
    assert none_s["extra"]["dispatch"]["2048"]["root"] == "dispatch" and none_s["extra"]["dispatch"]["2048"]["roots_considered"] == ["dispatch"]
    assert X.run_of(bt, LORA)["extra"]["dispatch"]["8192"]["root"] == "dispatch_rung8192"
    # outside the ladder plan: base-only, the one at-cap draw unclassified, no loop reading invented, the raw at-cap share stands
    mb = none_p["extra"]["completion_cap_merge"]
    assert mb["ladder_planned"] is False and mb["plan_absent_rule"] == M.PLAN_ABSENT_RULE and list(mb["plan_decisions"]) == ["4096"]
    dec = mb["plan_decisions"]["4096"]
    assert dec["exit"] == M.PLAN_EXIT_NOT_PLANNED == "not_planned" and dec["planned"] is False and dec["promoted"] is False and dec["threshold"] == 0.01 and dec["n_at_cap"] == 1 and dec["n_draws"] == N
    assert dec["eligible_draws"] is None and "no step to 4,096 was due" in dec["note"] and mb["unclassified_draws"] == [["540", 1]]
    assert none_p["token_cap_tokens"] == 2048 and none_p["extra"]["escalation_state"] == "final_at_base" and none_p["extra"]["three_numbers"]["n_unclassified"] == 1
    assert none_p["extra"]["at_cap_classes"] == {"loop": 0, "ceiling": 0, "not_escalated": 0, "unit": "draws", "unclassified": 1}
    items = {(i["item_id"], i["sample_index"]): i for i in bt["item_result"] if i["run_id"] == none_p["run_id"]}
    assert items[("540", 1)]["extra"]["at_cap_class"] == "unclassified" and items[("540", 1)]["value"] == 0.0 and items[("540", 1)]["label"] is None
    flags = {(f["item_id"], f["sample_index"]): f for f in bt["item_flags"] if f["run_id"] == none_p["run_id"]}
    assert flags[("540", 1)]["at_token_cap"] == 1 and flags[("540", 1)]["degenerate_rep_ratio_loop"] is None and flags[("0", 0)]["degenerate_rep_ratio_loop"] is None
    deg = {x["measure"]: x for x in bt["run_degeneration"] if x["run_id"] == none_p["run_id"]}
    assert (deg[M.RAW_AT_CAP_MEASURE]["measurement_state"], deg[M.RAW_AT_CAP_MEASURE]["numerator"], deg[M.RAW_AT_CAP_MEASURE]["denominator"]) == ("measured", 1, N)
    assert deg[M.LOOP_MEASURE]["measurement_state"] == "not_measured" and "never planned" in deg[M.LOOP_MEASURE]["notes"] and deg[M.LOOP_MEASURE]["numerator"] is None
    assert deg[M.COHERENT_MEASURE]["measurement_state"] == "not_measured" and "never planned" in deg[M.COHERENT_MEASURE]["notes"]
    assert facts["rows"][f"{TAG_P}|{NONE}"]["ladder_planned"] is False and facts["rows"][f"{TAG_P}|{NONE}"]["n_unclassified"] == 1 and facts["rows"][f"{TAG}|{NONE}"]["ladder_planned"] is True
    # the planned seed none reads as today: below_threshold, its at-cap draws the plan's eligible (not_escalated), the split measured
    ms = none_s["extra"]["completion_cap_merge"]
    assert ms["plan_decisions"]["4096"]["exit"] == "below_threshold" and "ladder_planned" not in ms and none_s["extra"]["at_cap_classes"] == {"loop": 0, "ceiling": 0, "not_escalated": 5, "unit": "draws"}
    deg_s = {x["measure"]: x for x in bt["run_degeneration"] if x["run_id"] == none_s["run_id"]}
    assert deg_s[M.LOOP_MEASURE]["measurement_state"] == "measured" and deg_s[M.COHERENT_MEASURE]["measurement_state"] == "measured" and deg_s[M.COHERENT_MEASURE]["numerator"] == 5
    # the composed row pairs against ITS OWN none (a standup replicate), the seed rows against the seed none
    c_comp = {c["run_id"]: c for c in bt["run_contrast"]}[cp["run_id"]]
    assert c_comp["anchor_run_id"] == none_p["run_id"] and c_comp["anchor_kind"] == "same_standup_none" and c_comp["extra"]["cross_standup"] is False and cp["extra"]["anchor"]["anchor_view_legacy"] == TAG_P
    assert abs(c_comp["extra"]["acc_anchor_over_pairs"] - 700 / N) < 1e-12 and {c["run_id"]: c for c in bt["run_contrast"]}[X.run_of(bt, LORA)["run_id"]]["anchor_run_id"] == none_s["run_id"]


def test_the_pass_1_root_wins_when_both_roots_carry_the_paired_nones_record(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, paired_none=True, paired_none_dispatch_root="both")
    bt, facts = X.read(X.adapter(fx, seeds))
    assert facts["n_refusals"] == 0 and facts["runs"] == 7
    d = {(r["view_legacy"], r["campaign_cell_id_legacy"]): r for r in bt["run"]}[(TAG_P, NONE)]["extra"]["dispatch"]["2048"]
    assert d["root"] == "dispatch" and d["other_root_records"] == ["dispatch_paired_none"] and d["roots_considered"] == ["dispatch", "dispatch_paired_none"]
    assert facts["dispatch_roots_read"] == ["dispatch", "dispatch_rung4096", "dispatch_rung8192"]


def test_a_paired_none_whose_record_sits_in_neither_live_root_is_refused_naming_both(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, paired_none=True, drop_dispatch=[(NONE, 2048)])
    bt, facts = X.read(X.adapter(fx, seeds))
    # drop_dispatch is keyed by object key: both `none` rows lose their base record; each refusal names the roots the row may use
    why_p, why_s = X.refused(facts, TAG_P, NONE), X.refused(facts, TAG, NONE)
    assert "records/dispatch/dispatch/qwen35_4b_s2p/none/rung2048.json or records/dispatch_paired_none/dispatch/qwen35_4b_s2p/none/rung2048.json" in why_p
    assert "records/dispatch/dispatch/qwen35_4b_s2/none/rung2048.json)" in why_s and " or " not in why_s.split("no dispatcher record (")[1].split(")")[0]
    assert facts["runs"] == 5


def test_parked_dispatcher_roots_are_never_inputs_whatever_they_hold(tmp_path, seeds):
    fx = X.mirror(tmp_path, seeds, incomplete_attempt_root=True)
    ad = X.adapter(fx, seeds)
    inputs = ad.discover()
    assert not any(".failed-" in i.uri or ".incomplete-" in i.uri for i in inputs), "the parked roots are history, never inputs"
    for name in ("dispatch_rung4096.failed-D205", "dispatch_rung8192.incomplete-D999", "dispatch_paired_none.failed-D1", "dispatch.incomplete-x"):
        assert M.FAILED_DISPATCH_RE.match(name), name
    for name in ("dispatch", "dispatch_rung4096", "dispatch_paired_none", "ladder_4096", "dispatch_rung4096_failed"):
        assert not M.FAILED_DISPATCH_RE.match(name), name
    bt, facts = X.read(ad)
    assert facts["n_refusals"] == 0 and facts["failed_dispatch_roots_never_read"] == ["dispatch_rung4096.failed-D205", "dispatch_rung8192.incomplete-D999"]
    assert X.run_of(bt, LORA)["extra"]["failed_dispatch_roots_never_read"] == facts["failed_dispatch_roots_never_read"] and X.run_of(bt, LORA)["token_cap_tokens"] == 8192, "the parked root's `active` record for the LoRA row was never read"


def test_an_unplanned_row_with_a_cap_step_store_is_refused(tmp_path, seeds):
    spec = {"base_pass": 700, "at_cap": [str(i) for i in range(500, 512)], "loops": ["500"], "steps": {4096: {"pass": [str(i) for i in range(501, 512)]}}}
    fx = X.mirror(tmp_path, seeds, paired_none=True, paired_none_planned=False, paired_none_spec=spec)
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG_P, NONE)
    assert f"plans no row qwen35_4b/{SEL_P}/{NONE} but the 4,096 store is present" in why and facts["runs"] == 6


def test_an_unplanned_row_whose_at_cap_share_exceeds_the_declared_trigger_is_refused(tmp_path, seeds):
    spec = {"base_pass": 700, "at_cap": [str(i) for i in range(500, 506)], "loops": [], "steps": {}}   # 12 of 1,082 draws at the cap = 1.1 % > the declared 1 %
    fx = X.mirror(tmp_path, seeds, paired_none=True, paired_none_planned=False, paired_none_spec=spec)
    bt, facts = X.read(X.adapter(fx, seeds))
    why = X.refused(facts, TAG_P, NONE)
    assert f"plans no row qwen35_4b/{SEL_P}/{NONE}" in why and "12 of 1082 draws = 0.0111" in why and "exceeds the declared trigger 0.01" in why and facts["runs"] == 6
    # the composed row keeps the fallback while its own `none` refuses
    cp = X.run_of(bt, COMPOSED)
    assert cp["extra"]["anchor"]["own_none_state"] == M.ANCHOR_REFUSED and cp["extra"]["anchor"]["cross_standup"] is True


# --------------------------------------------------------------------------------------------
# the anchor currency decides every contrast
# --------------------------------------------------------------------------------------------

def test_on_an_empty_frame_every_comparison_parks_on_the_value_row_and_names_the_reason(tmp_path, seeds):
    """The first pass over a fresh registry: the frame holds no run yet, so every own anchor is `not_in_frame`, no exact-key candidate exists, and every
    comparison PARKS by name -- a `row_refusals` line and a `delta_block` on the VALUE row (never the bound row), `extra.anchor` keeping the plan's spelling
    with the currency's reason beside it; no contrast is written."""
    import _anchor_frame_fixtures as AF
    from sisteer.registry.adapters import anchor_current as AC
    fx = X.mirror(tmp_path, seeds)
    bt, facts = X.read(X.adapter(fx, seeds), conn=AF.Frame().con)
    assert "run_contrast" not in bt and facts["runs"] == 6 and facts["contrasts_parked"] == 5 and facts["contrasts_filed"] == 0 and facts["anchor_states"] == {"no_current_anchor": 5}
    assert facts["n_refusals"] == 5 and all(x.split(" -- ")[0].endswith("|contrast") and "contrast has no current anchor at key" in x and "not_in_frame" in x for x in facts["refusals"]), facts["refusals"]
    for okey in (RELU, LORA, PROMPTED, ARDITI, COMPOSED):
        blk = X.run_of(bt, okey)["extra"]["anchor"]
        assert blk["state"] == "no_current_anchor" and blk["anchor_run_id"] is None and blk["currency_state"] == "no_current_anchor" and blk["currency_own_reason"] == "not_in_frame"
        assert blk["anchor_selection"]["state"] == "no_current_anchor" and blk["rule"] == M.ANCHOR_RULE and blk["currency_rule"] == M.CURRENCY_RULE
        value, bound, loose = X.metrics_of(bt, X.run_of(bt, okey)["run_id"])
        blk2 = value["extra"]["delta_block"]
        assert blk2["anchor"]["state"] == "no_current_anchor" and blk2["anchor"]["own_reason"] == "not_in_frame" and blk2["delta"] is None and blk2["n_pairs"] == 0 and blk2["rule"] == AC.RULE
        assert "delta_block" not in bound["extra"] and "delta_block" not in loose["extra"]
    # the same mirror on the primed frame files every contrast and states delta_block: null on each VALUE row
    bt2, facts2 = X.read(X.adapter(fx, seeds))
    assert facts2["contrasts_filed"] == 5 and facts2["contrasts_parked"] == 0 and facts2["anchor_states"] == {"own_current": 5} and facts2["reanchor_computations"] == 0
    assert all(X.metrics_of(bt2, X.run_of(bt2, k)["run_id"])[0]["extra"]["delta_block"] is None for k in (RELU, LORA, PROMPTED, ARDITI, COMPOSED))
    assert all(c["extra"]["anchor_selection"]["state"] == "own_current" and c["extra"]["currency_rule"] == M.CURRENCY_RULE for c in bt2["run_contrast"])


def test_a_retired_seed_none_re_anchors_the_rows_at_its_cap_on_a_current_sibling_of_the_frame_with_the_registrys_unpaired_rate_difference(tmp_path, seeds):
    """The seed tag's `none` (final cap 2,048) is superseded on the frame (a ruling) and the frame holds a CURRENT unsteered run of the EXACT key under
    this source from another standup that is NOT a row of this batch (another seed tag's `none` at 2,048): the currency re-anchors the rows AT THAT KEY --
    the prompted and Arditi rows, whose final cap is 2,048 -- as `sibling_same_source` / `sibling_none` and, with no draws of it in hand, files the
    registry's UNPAIRED rate difference from the two readings' k / n (`computed_by registry`, its own computation row named by
    `extra.statistic_computation_id`), the own pairing recorded beside it. The rows whose ladder escalated (relu 4,096; lora and the composed row 8,192)
    have NO unsteered run at their drawn cap once their own `none` is retired: the own pairing stood whatever the caps while it was current (the exact
    key never widens to another cap), so those three PARK by name with `own_reason superseded` and no candidate."""
    import _anchor_frame_fixtures as AF
    from sisteer.registry.adapters import anchor_current as AC
    from sisteer.registry.adapters import declared_common as DC
    fx = X.mirror(tmp_path, seeds)
    ad = X.adapter(fx, seeds)
    frame = X.primed_frame(ad)
    none_s = {(r["payload"]["view_legacy"], r["payload"]["campaign_cell_id_legacy"]): r for r in AF.priming_read(ad, X.ctx(frame.con)) if r["record_type"] == "run"}[(TAG, NONE)]["payload"]
    assert none_s["token_cap_tokens"] == 2048
    sib = dict(none_s, run_id="rid-sibling-none-s9", view_legacy=f"{M.MODEL_KEY}_s9", producer_run_key_legacy=f"{M.MODEL_KEY}_s9/none-store")
    jid = "jid-sibling-none-s9"
    AF.load_records(frame, [{"record_type": "run", "payload": sib},
                            {"record_type": "judgment", "payload": {"judgment_id": jid, "run_id": sib["run_id"], "judge_id": "rule", "status": "complete", "n_items_judged": N}},
                            {"record_type": "run_metric", "payload": {"run_id": sib["run_id"], "judgment_id": "", "metric_id": M.HEADLINE_METRIC, "threshold": -1.0,
                                                                      "computation_id": M.merge_computation_id(), "value": 600 / N, "k": 600, "n": N, "computed_by": "registry"}}],
                    items=False)
    frame.con.execute("UPDATE v_run_current SET is_current = 0, why = 'selection_state superseded (ruling r-test)', selection_state = 'superseded' WHERE run_id = ?", (none_s["run_id"],))
    frame.con.commit()
    bt, facts = X.read(ad, conn=frame.con)
    assert facts["anchor_states"] == {"no_current_anchor": 3, "sibling_same_source": 2} and facts["contrasts_filed"] == 2 and facts["contrasts_reanchored"] == 2, facts["anchor_states"]
    assert facts["contrasts_parked"] == 3 and facts["n_refusals"] == 3 and facts["reanchor_computations"] == 1, facts["refusals"]
    contrasts = {c["run_id"]: c for c in bt["run_contrast"]}
    comp_rows = {c["computation_id"]: c for c in bt["computation"]}
    assert set(contrasts) == {X.run_of(bt, k)["run_id"] for k in (PROMPTED, ARDITI)}
    for k in (PROMPTED, ARDITI):
        c = contrasts[X.run_of(bt, k)["run_id"]]
        assert c["anchor_run_id"] == sib["run_id"] and c["anchor_kind"] == "sibling_none" and c["anchor_judgment_id"] == "" and c["n_pairs"] == 0 and c["computed_by"] == "registry"
        assert c["delta_method"] == DC.UNPAIRED_RATE_METHOD and c["extra"]["anchor_selection"]["state"] == "sibling_same_source" and c["extra"]["anchor_selection"]["own_reason"] == "superseded"
        assert c["extra"]["cross_standup"] is True and c["extra"]["cross_source"] is False and c["extra"]["reanchor_tier"] == "sibling_current" and c["extra"]["anchor_view_legacy"] == f"{M.MODEL_KEY}_s9"
        assert c["extra"]["own_pairing"]["own_anchor_run_id"] == none_s["run_id"] and c["extra"]["own_pairing"]["own_reason"] == "superseded" and "unpaired" in c["extra"]["pairing"]
        assert c["extra"]["anchor_reading"]["judgment_id"] == "" and c["extra"]["statistic_computation_id"] in comp_rows and comp_rows[c["extra"]["statistic_computation_id"]]["version"] == AC.RULE
        value, _b, _l = X.metrics_of(bt, c["run_id"])
        assert abs(c["delta"] - 100.0 * (value["k"] / value["n"] - 600 / N)) < 1e-9 and value["extra"]["delta_block"] is None
        blk = X.run_of(bt, k)["extra"]["anchor"]
        assert blk["reanchored"] is True and blk["reanchor_tier"] == "sibling_current" and blk["own_anchor_run_id"] == none_s["run_id"] and blk["anchor_run_id"] == sib["run_id"]
    for k, cap in ((RELU, 4096), (LORA, 8192), (COMPOSED, 8192)):
        value, _b, _l = X.metrics_of(bt, X.run_of(bt, k)["run_id"])
        blk2 = value["extra"]["delta_block"]
        assert blk2["anchor"]["state"] == "no_current_anchor" and blk2["anchor"]["own_reason"] == "superseded" and blk2["anchor_selection"]["n_candidates"] == 0 and blk2["anchor_selection"]["key"]["token_cap_tokens"] == cap
        assert X.run_of(bt, k)["extra"]["anchor"]["currency_own_reason"] == "superseded" and X.run_of(bt, k)["extra"]["anchor"]["anchor_run_id"] is None
    assert all(x.split(" -- ")[0].endswith("|contrast") and "superseded" in x for x in facts["refusals"]), facts["refusals"]
