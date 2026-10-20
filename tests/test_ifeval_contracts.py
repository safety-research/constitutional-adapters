"""The canonical IFEval battery — the contract vocabulary.

What is pinned: the MERGE seam (every local fallback that lane 1's ``contracts`` also defines must AGREE — vacuous until
the merge, red the moment a twin disagrees); the seed derivation (the sampled re-run's literal, the IFEval example values,
the [0, 2^32) range, cell / cap / model independence); the four ``vendor_preset`` rows equal
the referenced constants (Qwen's presence_penalty 1.5 per request, Nemotron's grandfathered 1.0 recorded in the prose);
the ``rung_subset/2`` draw-level contract (sample-1-only, both trials, mixed; every consistency refusal by name; the
order-free draw digest distinct from the v1 id digest); the OFF stamp triples; ``seed_rules_for`` conditional on the task.
Everything synthetic: invented ids, placeholder hashes.
"""
from __future__ import annotations

import pytest

from sisteer.capeval.serving import proxy as _proxy
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

SRC_HASH = "0123456789abcdef"
HOPS_OK = {"driver": 7200.0, "proxy_upstream": 7200.0}


def _tv():
    from sisteer.capeval.tiers import cap_ladder as L
    return L.transport_check(4096, HOPS_OK).as_dict()


def _subset(draws, *, n_loop=0, n_boundary=0, cap=4096, source_cap=2048, task=IC.IFEVAL_TASK, **over):
    ids = sorted({i for i, _ in draws})
    doc = {"schema": IC.ITEM_SUBSET_SCHEMA_V2, "task": task, "source_out_root": "controls/gemma4e4b/none/vendor_preset",
           "source_identity_hash": SRC_HASH, "source_cap": source_cap, "cap": cap, "rule": IC.IFEVAL_TRIGGER_RULE,
           "threshold": IC.IFEVAL_TRIGGER_THRESHOLD, "ids": ids, "draws": [[i, s] for i, s in draws],
           "n_items": len(ids), "n_draws": len(draws), "n_at_cap": len(draws) + n_loop, "n_truncated": len(draws),
           "n_loop": n_loop, "n_boundary": n_boundary, "transport_check": _tv()}
    doc.update(over)
    return doc


# --------------------------------------------------------------------------- the merge seam
MUST_BE_MERGED = ("SEED_RULE_IFEVAL", "ifeval_sample_seed", "VENDOR_PRESET_SAMPLING_BY_MODEL_KEY", "SAMPLING_VENDOR_PRESET",
                  "ROUTE_KINDS", "BANK_VIEWS", "ROW_IDENTITY_FIELDS", "ifeval_paths", "IFEVAL_SLUGS", "IFEVAL_CONTRASTS",
                  "ITEM_SUBSET_SCHEMA_V2", "ITEM_SUBSET_FIELDS_V2", "ITEM_SUBSET_BLOCK_FIELDS_V2", "DONE_MARKER_DRAW_FIELDS",
                  "IFEVAL_CAP_LADDER", "IFEVAL_CAP_CEILING", "IFEVAL_TRIGGER_THRESHOLD", "IFEVAL_TRIGGER_RULE",
                  "ifeval_expected_stamps", "IFEVAL_ROW_FACT_FIELDS", "FLAG_OFF_CAP_BY_TASK", "OFF_CAP_BY_TASK_ALLOWED",
                  # the seed lane-view vocabulary and the documents register
                  "IFEVAL_LANE_VIEWS", "IFEVAL_VIEWS", "IFEVAL_MATRIX_QWEN_SEEDREP_REL", "IFEVAL_MATRIX_DOCUMENTS")


def test_every_lane1_twin_that_exists_agrees_with_the_local_fallback():
    """The merge seam: for every name lane 1's contracts DOES define, the two definitions agree. Before the merge
    no twin exists and the loop is vacuous; after it a disagreement is a red test, never a silent divergence."""
    report = IC.merge_agreement_report()
    for name in MUST_BE_MERGED:
        assert name in IC.MERGE_AGREEMENT, f"{name} is not a merged name"
    disagreeing = sorted(n for n, r in report.items() if r["in_contracts"] and r["agree"] is False)
    assert not disagreeing, f"lane 1's contracts disagree with lane 2's fallbacks on {disagreeing}"
    # the row task follows contracts.inspect_task the moment contracts registers the name
    if IC.IFEVAL_TASK_NAME in C.INSPECT_TASK_NAMES:
        assert C.inspect_task(IC.IFEVAL_TASK_NAME) == IC.IFEVAL_TASK
    assert IC.IFEVAL_TASK == "inspect/ifeval"


# --------------------------------------------------------------------------- seeds (P1)
def test_seed_is_the_sampled_reruns_derivation_with_the_pinned_literals():
    assert IC.sampling_seed(20260909, 0, "xstest", "xst0000-aaaaaaaa") == 3515763156
    assert IC.ifeval_sample_seed("1000", 0) == 3028447421
    assert IC.ifeval_sample_seed("1000", 1) == 3178680996
    assert IC.IFEVAL_SEED_BASE == 20260909 and IC.IFEVAL_SEED_READOUT == IC.CONTROL_IFEVAL == "ifeval"
    seeds = {(i, t): IC.ifeval_sample_seed(str(i), t) for i in range(1000, 1300) for t in (0, 1)}
    assert all(0 <= s < 2 ** 32 for s in seeds.values())
    assert len(set(seeds.values())) == len(seeds)              # distinct across item and trial
    assert seeds[(1000, 0)] != seeds[(1000, 1)]
    # no cell, cap or model in the key: the function has no such argument, and the rule names none
    assert "cell" not in IC.SEED_RULE_IFEVAL and "cap" not in IC.SEED_RULE_IFEVAL and "model" not in IC.SEED_RULE_IFEVAL
    assert "20260909" in IC.SEED_RULE_IFEVAL and "no modulus" in IC.SEED_RULE_IFEVAL
    for bad in (True, -1, 0.5, "0"):
        with pytest.raises(ValueError):
            IC.ifeval_sample_seed("1000", bad)  # type: ignore[arg-type]
    # differs from the Inspect rule (mod 2^31-1) — a NEW rule beside it, never a rename
    from sisteer.capeval.tiers import inspect_served as isv
    assert IC.ifeval_sample_seed("1000", 0) != isv.sample_seed("1000", 0)


def test_seed_rules_for_adds_the_ifeval_rule_iff_the_task_is_present():
    assert IC.seed_rules_for(("inspect/math", "inspect/gsm8k_paired200")) == dict(C.SEED_RULES)
    with_ifeval = IC.seed_rules_for(("inspect/math", IC.IFEVAL_TASK))
    assert with_ifeval[IC.IFEVAL_TASK] == IC.SEED_RULE_IFEVAL
    assert {k: v for k, v in with_ifeval.items() if k != IC.IFEVAL_TASK} == dict(C.SEED_RULES)


# --------------------------------------------------------------------------- sampling (P3, P43)
def test_vendor_preset_rows_equal_the_fixed_decisions_and_the_referenced_constants():
    rows = IC.VENDOR_PRESET_SAMPLING_BY_MODEL_KEY
    assert set(rows) == set(IC.IFEVAL_MODELS) == {"gemma4e4b", "qwen35_4b", "nemotron3nano4b", "llama8b"}
    g, q, n, l = rows["gemma4e4b"], rows["qwen35_4b"], rows["nemotron3nano4b"], rows["llama8b"]
    assert (g.temperature, g.top_p, g.top_k, g.presence_penalty) == (1.0, 0.95, 64, None)
    assert (q.temperature, q.top_p, q.top_k, q.min_p, q.presence_penalty, q.repetition_penalty) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)
    assert (n.temperature, n.top_p, n.top_k) == (1.0, 0.95, None)
    assert (l.temperature, l.top_p, l.top_k) == (0.6, 0.9, None)
    # by reference (P3): the numbers ARE the existing constants'
    tg, tn = C.THINKING_SAMPLING_BY_MODEL_KEY["gemma4e4b"], C.THINKING_SAMPLING_BY_MODEL_KEY["nemotron3nano4b"]
    assert (g.temperature, g.top_p, g.top_k) == (tg.temperature, tg.top_p, tg.top_k)
    assert (n.temperature, n.top_p, n.top_k) == (tn.temperature, tn.top_p, tn.top_k)
    qi = C.QWEN35_VENDOR_ROWS["instruct_general"]
    assert q[:3] == qi[:3] and (q.min_p, q.presence_penalty, q.repetition_penalty) == (qi.min_p, qi.presence_penalty, qi.repetition_penalty)
    assert (l.temperature, l.top_p, l.top_k) == (C._LLAMA_CARD["temperature"], C._LLAMA_CARD["top_p"], C._LLAMA_CARD["top_k"])
    # P43: the grandfathered alternative is RECORDED in the prose, never a number in the row
    alt = C.BATTERY_SAMPLING_BY_MODEL_KEY["nemotron3nano4b"].top_p
    assert alt == 1.0 and "1.0" in n.source and "GRANDFATHERED" in n.source and "generation_config" in n.source
    assert IC.PRESENCE_PENALTY_PER_REQUEST_MODELS == ("qwen35_4b",)
    # any other key REFUSES by name — never a default temperature
    with pytest.raises(C.SamplingUnverified, match="vendor_preset"):
        IC.vendor_preset_row("glm46vflash")
    with pytest.raises(C.SamplingUnverified):
        IC.vendor_preset_row("gemma4e2b")
    # the matrix's dict shape: EXACTLY the keys the row sets, the source last
    d = IC.sampling_row_dict(q)
    assert list(d) == ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "source"]
    assert list(IC.sampling_row_dict(l)) == ["temperature", "top_p", "source"]


# --------------------------------------------------------------------------- the draw-subset contract (P30)
def test_rung_subset_v2_admits_sample_1_only_both_trials_and_mixed_subsets():
    one = _subset([("i1", 1)])
    both = _subset([("i1", 0), ("i1", 1)])
    mixed = _subset([("i1", 1), ("i2", 0), ("i2", 1), ("i3", 0)], n_loop=2, n_boundary=1)
    for doc in (one, both, mixed):
        assert IC.validate_item_subset_v2(doc) == doc
        assert IC.validate_item_subset_any(doc) == doc
    assert both["n_items"] == 1 and both["n_draws"] == 2
    assert mixed["n_items"] == 3 and mixed["n_draws"] == 4 and mixed["n_at_cap"] == 6
    assert IC.draws_of_subset(mixed) == [("i1", 1), ("i2", 0), ("i2", 1), ("i3", 0)]
    # v1 dispatch still works through the shared entry point
    from sisteer.capeval.tiers import cap_ladder as L
    v1 = {"schema": C.ITEM_SUBSET_SCHEMA, "task": C.GSM8K_STANDARD_TASK, "source_out_root": "x", "source_identity_hash": SRC_HASH,
          "source_cap": 8192, "cap": 16384, "rule": "r", "threshold": 0.03, "ids": ["a"], "n_at_cap": 1, "n_non_terminating": 0,
          "n_boundary": 0, "transport_check": L.transport_check(16384, HOPS_OK).as_dict()}
    assert IC.validate_item_subset_any(v1) == v1
    with pytest.raises(IC.ItemSubsetError, match="neither"):
        IC.validate_item_subset_any({**v1, "schema": "rung_subset/9"})


@pytest.mark.parametrize("edit, match", [
    (lambda d: {**d, "ids": ["i1", "i9"]}, "sorted distinct items of draws"),
    (lambda d: {**d, "n_draws": 5}, "n_draws"),
    (lambda d: {**d, "n_items": 7}, "n_items"),
    (lambda d: {**d, "n_at_cap": d["n_at_cap"] + 1}, "n_truncated"),
    (lambda d: {**d, "n_boundary": 99}, "n_boundary"),
    (lambda d: {**d, "cap": 3000}, "not a rung"),
    (lambda d: {**d, "cap": 16384}, "source_cap < cap <="),
    (lambda d: {**d, "source_cap": 4096, "cap": 4096}, "source_cap < cap"),
    (lambda d: {**d, "draws": d["draws"] + [d["draws"][0]], "n_draws": d["n_draws"] + 1}, "duplicates"),
    (lambda d: {**d, "draws": [["i1", -1]], "ids": ["i1"], "n_items": 1, "n_draws": 1, "n_at_cap": 1, "n_truncated": 1}, "sample_idx"),
    (lambda d: {**d, "transport_check": {**d["transport_check"], "ok": False}}, "PASSED"),
    (lambda d: {**d, "schema": C.ITEM_SUBSET_SCHEMA}, "schema"),
    (lambda d: {k: v for k, v in d.items() if k != "draws"}, "keys"),
    (lambda d: {**d, "task": "inspect/hellaswag2"}, "row task"),
])
def test_rung_subset_v2_refuses_every_inconsistency_by_name(edit, match):
    doc = edit(_subset([("i1", 1), ("i2", 0), ("i2", 1)], n_loop=1))
    with pytest.raises(IC.ItemSubsetError, match=match):
        IC.validate_item_subset_v2(doc)


def test_the_draw_digest_is_order_free_per_draw_and_distinct_from_the_id_digest():
    a = IC.item_subset_sha16_v2([["i1", 0], ["i1", 1], ["i2", 1]])
    b = IC.item_subset_sha16_v2([("i2", 1), ("i1", 1), ("i1", 0)])
    assert a == b and len(a) == 16
    # the same ITEMS, different DRAWS -> a different key (P30)
    assert IC.item_subset_sha16_v2([["i1", 0], ["i2", 1]]) != a
    assert IC.item_subset_sha16_v2([["i1", 1], ["i2", 0]]) != IC.item_subset_sha16_v2([["i1", 0], ["i2", 1]])
    assert a != C.item_subset_sha16(["i1", "i2"])
    doc = _subset([("i1", 0), ("i1", 1), ("i2", 1)])
    ident = IC.cap_rung_identity_v2(doc)
    assert tuple(ident) == C.CAP_RUNG_FIELDS and ident["subset_sha16"] == a and ident["source_cap"] == 2048
    block = IC.item_subset_block_v2(doc, path="/vol/rung4096/x/rung_subset.json")
    assert tuple(block) == IC.ITEM_SUBSET_BLOCK_FIELDS_V2
    assert block["draws"] == [["i1", 0], ["i1", 1], ["i2", 1]] and block["n_draws"] == 3 and block["n_items"] == 2 == block["n"]
    assert block["schema"] == IC.ITEM_SUBSET_SCHEMA_V2 and block["subset_sha16"] == a
    assert set(C.ITEM_SUBSET_BLOCK_FIELDS) < set(IC.ITEM_SUBSET_BLOCK_FIELDS_V2)
    assert set(C.ITEM_SUBSET_FIELDS) - {"n_non_terminating"} < set(IC.ITEM_SUBSET_FIELDS_V2)


# --------------------------------------------------------------------------- stamps, ladder, identity vocabulary
def test_expected_stamps_are_the_contracts_off_triples_undeclared():
    assert IC.ifeval_expected_stamps(fitted=False) == tuple(C.expected_stamp_regimes(lens=False, arm=C.THINKING_OFF))
    assert IC.ifeval_expected_stamps(fitted=False) == ("cell-stable", "not-consulted", "thinking_off")
    assert IC.ifeval_expected_stamps(fitted=True) == tuple(
        C.expected_stamp_regimes(lens=True, arm=C.THINKING_OFF, policy=_proxy.RESUMED_PREFILL_ALWAYS))
    assert IC.ifeval_expected_stamps(fitted=True, header_located=True) == tuple(
        C.expected_stamp_regimes(lens=True, arm=C.THINKING_OFF, policy=_proxy.RESUMED_PREFILL_CONDITIONAL))
    assert all(s[2] == "thinking_off" for s in (IC.ifeval_expected_stamps(fitted=True), IC.ifeval_expected_stamps(fitted=False)))
    with pytest.raises(ValueError):
        IC.ifeval_expected_stamps(fitted=False, header_located=True)


def test_ladder_and_identity_vocabulary():
    assert IC.IFEVAL_CAP_LADDER == (2048, 4096, 8192) and IC.IFEVAL_CAP_CEILING == 8192 and IC.IFEVAL_PASS1_CAP == 2048
    assert IC.IFEVAL_RUNG_CAPS == (4096, 8192) and IC.previous_rung(8192) == 4096 and IC.previous_rung(2048) is None
    assert IC.IFEVAL_TRIGGER_THRESHOLD == 0.01 and "> 0.01" in IC.IFEVAL_TRIGGER_RULE and "8192" in IC.IFEVAL_TRIGGER_RULE
    assert "−" not in IC.IFEVAL_TRIGGER_RULE                                     # ASCII hyphen-minus only
    assert IC.IFEVAL_TRIGGER_PROVENANCE == "1 % raw at-cap of draws, per the IFEval ruling"
    assert IC.OFF_CAP_BY_TASK_ALLOWED == {"ifeval": (4096, 8192)} and IC.FLAG_OFF_CAP_BY_TASK == "--off-cap-by-task"
    assert IC.ROUTE_KINDS == ("base", "prompted", "lens") and IC.BANK_VIEWS == ("main", "it")
    assert IC.ROW_IDENTITY_FIELDS == ("object_artifact_sha256", "route_kind", "view", "producer_tree_sha")
    assert IC.ROW_GUARD_FIELDS == (*IC.ROW_IDENTITY_FIELDS, "thinking", "sampling_mode")
    assert IC.IFEVAL_OUTCOME_ITEM_FIELDS == (*C.TASK_OUTCOME_ITEM_FIELDS, *IC.IFEVAL_ROW_FACT_FIELDS, C.SAMPLING_REF_FIELD)
    assert IC.IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH == (*C.TASK_OUTCOME_ITEM_FIELDS, *IC.IFEVAL_ROW_FACT_FIELDS, "stray_close_tag",
                                                      C.SAMPLING_REF_FIELD)
    assert IC.IFEVAL_OUTCOME_ITEM_FIELD_SETS == (IC.IFEVAL_OUTCOME_ITEM_FIELDS, IC.IFEVAL_OUTCOME_ITEM_FIELDS_NO_SWITCH)
    assert IC.IFEVAL_N == 541 and IC.IFEVAL_SAMPLES_PER_ITEM == 2 and IC.IFEVAL_DATASET_REVISION.startswith("966cd895")
    assert IC.IFEVAL_CONTRASTS == ("cell_minus_none", "none_it_minus_none_main", "merged_minus_r2048")
    assert IC.READINGS == ("r2048", "merged")
    # the seven instrument-view slugs + the six batch-64 seed lane views of qwen35_4b, appended LAST
    assert set(IC.IFEVAL_SLUGS) == ({(k, v) for k in IC.IFEVAL_MODELS for v in IC.BANK_VIEWS} - {("nemotron3nano4b", "it")}
                                    | {("qwen35_4b", v) for v in IC.IFEVAL_LANE_VIEWS})
    assert len(set(IC.IFEVAL_SLUGS.values())) == 13
    assert list(IC.IFEVAL_SLUGS.values())[-6:] == [f"qwen35_4b-sdfbase-{v}" for v in IC.IFEVAL_LANE_VIEWS]
    assert IC.IFEVAL_LANE_VIEWS == ("s1", "s2", "s3", "s1p", "s2p", "s3p") == C.IFEVAL_LANE_VIEWS
    assert IC.IFEVAL_VIEWS == (*IC.BANK_VIEWS, *IC.IFEVAL_LANE_VIEWS) == C.IFEVAL_VIEWS and IC.BANK_VIEWS == ("main", "it")
    assert IC.IFEVAL_MATRIX_DOCUMENTS == (IC.IFEVAL_MATRIX_REL, IC.IFEVAL_MATRIX_QWEN_SEEDREP_REL) == C.IFEVAL_MATRIX_DOCUMENTS
    assert IC.IFEVAL_MATRIX_QWEN_SEEDREP_REL == "configs/sdfbase/ifeval/matrix_qwen_seedrep.json"
    assert IC.IFEVAL_VIEWS_QWEN_SEEDREP_REL == "configs/sdfbase/ifeval/views_qwen_seedrep.yaml"
    assert IC.ifeval_route_kind("none") == "base" and IC.ifeval_route_kind("prompted__safety_refusal") == "prompted"
    assert IC.ifeval_route_kind("constv2-mix-full__psr_ce_all") == "lens"
    assert IC.ifeval_tag("gemma4e4b", "main") == "gemma4e4b" and IC.ifeval_tag("gemma4e4b", "it") == "gemma4e4b_it"
    assert IC.ifeval_tag("qwen35_4b", "s1") == "qwen35_4b_s1" and IC.ifeval_tag("qwen35_4b", "s3p") == "qwen35_4b_s3p"
    with pytest.raises(ValueError):
        IC.ifeval_tag("gemma4e4b", "fullconst")
    with pytest.raises(ValueError):
        IC.ifeval_tag("qwen35_4b", "ar_c0p3")            # a SOURCE bank, not an IFEval view
    assert IC.IFEVAL_GEOMETRY_ENTRY_BY_CAP == {2048: "capability:ifeval_2048", 4096: "capability:ifeval_4096", 8192: "capability:ifeval_8192"}
    assert IC.THINKING_OFF_TEMPLATE_KWARGS == '{"enable_thinking": false}'
