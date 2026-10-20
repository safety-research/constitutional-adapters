"""Derivation rules: positions / applied-mode tables,
transfer_kind, substrate from path, decode_resolved not_sent marking, strength nominal from a ruling,
fit_target_kind from each sidecar shape (refusal listing keys on an unclassifiable sidecar).
"""
from __future__ import annotations

import pytest

from sisteer.registry import derive as D
from sisteer.registry import vocab as V


@pytest.mark.parametrize("fit,run,kind,run_ck,fit_ck,expected", [
    ("m_base", "m_it", "lora", "instruct", "base", "base_to_instruct"),
    ("m_it", "m_it", "lora", "instruct", "instruct", "instruct_direct"),
    ("nem", "nem", "relu_steer", "chat", "chat", "chat_checkpoint"),
    (None, "m_it", "system_instruction", "instruct", None, "prompted_only"),
    (None, "m_it", "none", "instruct", None, "none"),
])
def test_transfer_kind_table(fit, run, kind, run_ck, fit_ck, expected):
    value, conf, _ = D.transfer_kind(fit, run, kind, run_ck, fit_ck)
    assert value == expected and conf == "derived"


def test_transfer_kind_unverified_names_missing_evidence():
    value, conf, why = D.transfer_kind(None, "m_it", "lora")
    assert value == "unverified" and conf == "unverified" and "fit_model_id" in why
    value, conf, why = D.transfer_kind("a_it", "b_it", "lora", "instruct", "instruct")
    assert value == "unverified"


@pytest.mark.parametrize("kind,sidecars,expected,conf", [
    ("none", {}, "not_applicable", "derived"),
    ("system_instruction", {}, "not_applicable", "derived"),
    ("lora_subtraction", {}, "inherited_from_parents", "derived"),
    ("static_r_arditi", {}, "searched_post_instruction_span", "derived"),
    ("diffmean", {}, "last_token_of_appended_assistant_header", "derived"),
    ("lora", {"train_budget_meta.json": {"positions": "all"}}, "all", "recorded"),
    ("lora", {"fit_provenance.json": {"positions_at_fit": "response"}}, "response_incl_assistant_header", "recorded"),
    ("relu_steer", {"blob": {"steer_positions": "all"}}, "all", "recorded"),
    ("lora", {"train_budget_meta.json": {"positions": "weird"}}, "unverified", "unverified"),
    ("lora", {}, "unverified", "unverified"),
])
def test_fit_positions_table(kind, sidecars, expected, conf):
    value, c, source = D.fit_positions(kind, sidecars)
    assert value == expected and c == conf
    if conf == "recorded":
        assert ":" in source   # names sidecar:key
    assert value in V.FIT_POSITIONS


def test_fit_positions_precedence_train_budget_wins():
    value, _, source = D.fit_positions("lora", {"blob": {"positions": "response"}, "train_budget_meta.json": {"positions": "all"}})
    assert value == "all" and source == "train_budget_meta.json:positions"


@pytest.mark.parametrize("path,kind,fitpos,standup,expected", [
    ("lora_native", "lora", "all", None, "all"),
    ("lora_native", "lora_subtraction", None, None, "all"),
    ("lora_native", "relu_steer", None, "response", "response"),
    ("lens", "vector", None, "response_all_turns", "response_all_turns"),
    ("lora_native", "relu_steer", "all", None, "all"),
    ("hf", "vector", "response_incl_assistant_header", None, "response"),
    ("lora_native", "alphasteer", None, None, "prefill_last_broadcast"),
    ("lora_native", "antidote", None, None, "last"),
    ("passthrough", "none", None, None, "none"),
    ("lora_native", "system_instruction", None, None, "none"),
    ("lora_native", "relu_steer", None, "sideways", "unverified"),
    ("lora_native", "relu_steer", None, None, "unverified"),
    # a KNOWN standup mode wins before the natively served LoRA rule: a LoRA served on the non-template positions is not steering
    # every token; standup None keeps the rule's `all`; an unknown standup mode stays unverified for every kind
    ("lora_native", "lora", None, "non_template", "non_template"),
    ("lora_native", "lora_subtraction", None, "non_template", "non_template"),
    ("lora_native", "relu_steer", None, "non_template", "non_template"),
    ("lora_native", "lora", "all", "response", "response"),
    ("lora_native", "lora", None, "sideways", "unverified"),
    ("lora_native", "lora_subtraction", "all", "sideways", "unverified"),
])
def test_applied_positions_mode_table(path, kind, fitpos, standup, expected):
    value, conf, _ = D.applied_positions_mode(path, kind, fitpos, standup)
    assert value == expected and value in V.APPLIED_POSITIONS_MODES
    assert (conf == "unverified") == (expected == "unverified")


def test_a_natively_served_lora_without_a_standup_record_still_reads_all_and_a_recorded_mode_is_recorded():
    assert D.applied_positions_mode("lora_native", "lora", None, None) == ("all", "derived", "natively served LoRA steers every token")
    value, conf, source = D.applied_positions_mode("lora_native", "lora", None, "non_template")
    assert (value, conf) == ("non_template", "recorded") and "standup record" in source
    assert "non_template" in V.APPLIED_POSITIONS_MODES and V.APPLIED_POSITIONS_MODES.index("non_template") < V.APPLIED_POSITIONS_MODES.index("none")


@pytest.mark.parametrize("mode,cat,kind,expected", [
    ("all", "multi_turn", "lora", "every_turn_prefill_and_decode"),
    ("non_template", "multi_turn", "lora", "every_turn_prefill_and_decode"),
    ("non_template", "petri", "relu_steer", "every_turn_prefill_and_decode"),
    ("non_template", "static_attack", "lora", "single_turn"),
    ("response", "petri", "relu_steer", "final_header_and_current_decode_only"),
    ("response_all_turns", "agentic", "vector", "every_assistant_span"),
    ("prompt_last", "multi_turn", "vector", "prefill_only_each_turn"),
    ("prefill_last_broadcast", "multi_turn", "alphasteer", "prefill_only_each_turn"),
    ("all", "static_attack", "lora", "single_turn"),
    ("all", "multi_turn", "none", "not_applicable"),
    ("all", "multi_turn", "system_instruction", "not_applicable"),            # a prompted object steers no positions, whatever the mode column
    ("all", "petri", "system_instruction", "not_applicable"),
    ("non_template", "agentic", "system_instruction", "not_applicable"),
    ("none", "multi_turn", "system_instruction", "not_applicable"),
    ("all", "static_attack", "system_instruction", "single_turn"),           # the category decides first on a single-turn eval
])
def test_multi_turn_scope_table(mode, cat, kind, expected):
    assert D.multi_turn_steering_scope(mode, cat, kind)[0] == expected


def test_a_prompted_object_steers_no_positions_so_its_multi_turn_scope_is_not_applicable_whatever_the_mode():
    """A job-start registration copies the selected instrument's protocol mode (`all`) onto a prompted run; the scope rule reads the
    object's kind before the mode table, so such a row never claims token steering, and the derivation names the kind. On a single-turn
    eval the category decides first, exactly as the mode table's `none` row would when the mode is derived from the kind
    (derive.applied_positions_mode gives a system-instruction object `none`) -- so the kind rule and the mode rule agree."""
    si = ("not_applicable", "derived", "object_kind=system_instruction: a prompted object steers no positions")
    for mode in ("all", "non_template", "response", "prefill_all", "none", "unverified"):
        for category in V.MULTI_TURN_EVAL_CATEGORIES:
            assert D.multi_turn_steering_scope(mode, category, "system_instruction") == si, (mode, category)
    assert D.multi_turn_steering_scope("all", "static_attack", "system_instruction") == ("single_turn", "derived", "eval_category=static_attack is single-turn")
    assert D.applied_positions_mode("lora_native", "system_instruction") == ("none", "derived", "prompted object steers no positions")
    assert D.multi_turn_steering_scope("none", "petri", "lora")[0] == "not_applicable"
    assert D.multi_turn_steering_scope("none", "static_attack", "lora")[0] == "single_turn"
    # the generic steered kind a reader hands the rule when the id states none: the mode table decides
    assert D.multi_turn_steering_scope("all", "petri", "steered")[0] == "every_turn_prefill_and_decode"


@pytest.mark.parametrize("path,expected", [
    ("/workspace-vast/someone/exp/x", "cluster_a"),
    ("/workspace/someone/exp/x", "cluster_b"),
    ("twosi-fits:/out/x", "modal"),
    ("/vol/out/x", "modal"),
    ("/home/someone/example-grade/x", "grading_host"),
    ("/workspace/runs/x", "runpod"),
    ("s3://elsewhere", "unverified"),
])
def test_substrate_from_path(path, expected):
    value, conf, _ = D.substrate_from_path(path)
    assert value == expected
    assert conf == ("derived" if expected != "unverified" else "unverified")


# The provenance defect of 2026-09-21: a reader's constant passed as `recorded` came back stamped `recorded`, the strongest label,
# whether or not any record said so. Only a producer FIELD earns `recorded`; a bare value is the reader's declared default.
def test_substrate_bare_value_in_the_recorded_slot_is_a_declared_default_never_recorded():
    value, conf, why = D.substrate_from_path("/home/TESTUSER/example-grade/x", "cluster_a")
    assert (value, conf) == ("cluster_a", "assumed_default") and "the grading host mirror" in why and "bare" in why
    value, conf, why = D.substrate_from_path("", "modal")
    assert (value, conf) == ("modal", "assumed_default")
    value, conf, why = D.substrate_from_path("", recorded="modal", default=("modal", "the store manifests name /vol/ ledgers"))
    assert (value, conf, why.split(";")[0]) == ("modal", "assumed_default", "the store manifests name /vol/ ledgers")


def test_substrate_recorded_needs_the_field_that_carried_the_value():
    value, conf, why = D.substrate_from_path("/home/TESTUSER/mirror", ("cells_full.json:substrate", "modal"))
    assert (value, conf) == ("modal", "recorded") and "cells_full.json:substrate" in why
    # a field naming a non-substrate never records; the path then decides
    value, conf, why = D.substrate_from_path("/workspace-vast/TESTUSER/x", ("manifest.served_cluster", "node-8"))
    assert (value, conf) == ("cluster_a", "derived")
    value, conf, why = D.substrate_from_path("", ("manifest.served_cluster", "node-8"))
    assert (value, conf) == ("unverified", "unverified") and "node-8" in why


def test_substrate_path_marker_is_derived_and_outranks_a_contradicted_default():
    value, conf, why = D.substrate_from_path("/workspace/TESTUSER/exp/x/serving/routing_table.json", default=("cluster_a", "the submit scripts"))
    assert (value, conf) == ("cluster_b", "derived") and "contradicted" in why
    value, conf, why = D.substrate_from_path("/workspace-vast/TESTUSER/x", default=("cluster_a", "the submit scripts"))
    assert (value, conf) == ("cluster_a", "derived") and "agrees" in why
    # the reader's own rule over producer fields is derived too, and beats the path
    value, conf, why = D.substrate_from_path("/home/TESTUSER/mirror", derived=("cluster_a", "grades root name carries the cluster tag _bu_"))
    assert (value, conf, why) == ("cluster_a", "derived", "grades root name carries the cluster tag _bu_")
    # a grading host path is the store's mirror: a declared default outranks it, with the mirror named
    value, conf, why = D.substrate_from_path("/home/TESTUSER/mirror", default=("cluster_a", "the submit scripts"))
    assert (value, conf) == ("cluster_a", "assumed_default") and why.startswith("the submit scripts") and "mirror" in why
    # a recorded field outranks everything
    value, conf, _ = D.substrate_from_path("/workspace-vast/TESTUSER/x", ("drive.log:cluster", "cluster_b"), derived=("modal", "rule"), default="cluster_a")
    assert (value, conf) == ("cluster_b", "recorded")


def test_substrate_nothing_is_unverified_with_what_was_missing():
    value, conf, why = D.substrate_from_path("")
    assert (value, conf) == ("unverified", "unverified") and "no producer field" in why
    value, conf, why = D.substrate_from_path("s3://elsewhere")
    assert (value, conf) == ("unverified", "unverified") and "s3://elsewhere" in why
    assert D.substrate_from_path("/home/TESTUSER/x") == ("grading_host", "derived", "grading-host-local path")


def test_first_shared_root_path_prefers_the_named_keys_then_walks():
    entry = {"cell_dir": "/home/TESTUSER/mirror/cells/x", "ledger": {"binding": {"routing_table": "/workspace-vast/TESTUSER/exp/serving/x/routing_table.json"}},
             "launch": {"serving_config": "/workspace/TESTUSER/repo/cfg.yaml", "excluded": {"chat_template": {"value": '"/workspace-vast/TESTUSER/exp/x/chat_template.jinja"'}}}}
    assert D.first_shared_root_path(entry, prefer=("ledger.binding.routing_table", "launch.serving_config")) == (
        "ledger.binding.routing_table", "/workspace-vast/TESTUSER/exp/serving/x/routing_table.json")
    assert D.first_shared_root_path(entry, prefer=("launch.serving_config",))[0] == "launch.serving_config"
    # no preferred key present: the walk finds the quoted chat_template value and unquotes it
    hit = D.first_shared_root_path({"launch": entry["launch"]["excluded"]})
    assert hit == ("chat_template.value", "/workspace-vast/TESTUSER/exp/x/chat_template.jinja") or hit[1].endswith("chat_template.jinja")
    assert D.first_shared_root_path({"cell_dir": "/home/TESTUSER/mirror"}) is None
    d = D.substrate_derived_from_record(entry, "report.json:legs[k]", prefer=("ledger.binding.routing_table",))
    assert d[0] == "cluster_a" and d[1].startswith("report.json:legs[k]:ledger.binding.routing_table (") and "/workspace-vast/ shared root" in d[1]
    assert D.substrate_derived_from_record({"a": 1}, "x") is None


def test_decode_resolved_marks_not_sent():
    out = D.decode_resolved({"temperature": 0.7}, {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    assert out["temperature"] == {"value": 0.7, "source": "sent"}
    assert out["top_p"]["source"] == "not_sent_resolved_to" and out["top_p"]["from"] == "generation_config.json"
    assert out["min_p"]["confidence"] == "assumed_default" and out["min_p"]["from"] == "vllm_default"
    out2 = D.decode_resolved({}, None, fallback={"top_p": 0.8}, fallback_name="server_default_row")
    assert out2["top_p"] == {"value": 0.8, "source": "not_sent_resolved_to", "from": "server_default_row", "confidence": "assumed_default"}


def test_strength_nominal_and_operating_point():
    assert D.strength_nominal(0.1, 0.05, filed_nominal=2.0) == (2.0, "recorded", "producer-filed nominal (dose column)")
    v, c, _ = D.strength_nominal(0.1, 0.05)
    assert v == pytest.approx(2.0) and c == "derived"
    assert D.strength_nominal(0.1, None)[1] == "unverified"
    assert D.is_operating_point("ar_r0p05", "ar_r0p05")[0] == 1
    assert D.is_operating_point("ar_r0p1", "ar_r0p05")[0] == 0
    assert D.is_operating_point(None, None)[1] == "unverified"
    assert D.is_operating_point("x", "y", object_sha="a" * 64, ruled_object_sha="a" * 64)[0] == 1   # hash binds, not the name


def test_fit_target_kind_from_sidecar_shapes():
    corpora = {"constv2-mix-full": "constv2-mix-full", "constv2ctl2-full": "constv2ctl2-full"}
    assert D.fit_target_kind("lora", {"derived_meta.json": {"parent_a": {}}}, corpora)[0] == "derived"
    assert D.fit_target_kind("none", {}, corpora)[0] == "none"
    assert D.fit_target_kind("system_instruction", {}, corpora)[0] == "declared_prompt"
    assert D.fit_target_kind("alphasteer", {"fit_meta.json": {"schema": "alphasteer_fit_meta/1"}}, corpora)[0] == "contrastive_direction"
    assert D.fit_target_kind("lora", {"train_budget_meta.json": {"si_id": "constv2-mix-full", "data_mode": "documents"}}, corpora)[0] == \
        "corpus_document_fit"
    assert D.fit_target_kind("relu_steer", {"train_budget_meta.json": {"si_id": "safety_refusal", "teacher_forward": True,
                                                                        "loss_impl": "kl"}}, corpora)[0] == "system_instruction_distillation"
    with pytest.raises(D.UnclassifiableSidecar) as ei:
        D.fit_target_kind("lora", {"train_budget_meta.json": {"si_id": "mystery", "lr": 1e-3}}, corpora)
    assert "train_budget_meta.json" in str(ei.value) and "si_id" in str(ei.value)
    with pytest.raises(D.UnclassifiableSidecar) as ei2:
        D.fit_target_kind("lora", {"train_budget_meta.json": {"si_id": "safety_refusal", "teacher_forward": True, "data_mode": "corpus"}},
                          {"safety_refusal": "x"})
    assert "ambiguous" in str(ei2.value)


def test_fit_target_kind_dose_copy_and_seeded_teachers():
    """A dose stamp decides `derived` BEFORE the fit record is read (a dose-scaled copy carries its PARENT's
    train_budget_meta.json, which alone would classify it as the parent's kind); teacher ids come from the seeds
    when given, replacing the legacy roster rather than widening it."""
    parent_tbm = {"si_id": "safety_refusal", "teacher_forward": True, "lr": 1e-3}
    # without the stamp the parent's record classifies the copy as the parent's kind — the case the branch exists for
    assert D.fit_target_kind("relu_steer", {"train_budget_meta.json": parent_tbm}, {})[0] == "system_instruction_distillation"
    v, c, src = D.fit_target_kind("relu_steer", {"object.pt.provenance.json": {"dose_scale": 2.0}, "train_budget_meta.json": parent_tbm}, {})
    assert (v, c) == ("derived", "recorded") and "object.pt.provenance.json:dose_scale" in src
    stamped = dict(parent_tbm, dose_scaled_from={"scale": 0.5, "parent_object_sha256": "a" * 64})
    v, c, src = D.fit_target_kind("relu_steer", {"train_budget_meta.json": stamped}, {})
    assert (v, c) == ("derived", "recorded") and "train_budget_meta.json:dose_scaled_from" in src
    v, c, src = D.fit_target_kind("relu_steer", {"object.pt.provenance.json": {"dose_scale": 0.5}, "train_budget_meta.json": stamped}, {})
    assert v == "derived" and "dose_scale" in src and "dose_scaled_from" in src
    # a bare non-mapping dose_scaled_from is not a stamp (the adapter's reader treats it the same way)
    assert D.fit_target_kind("relu_steer", {"train_budget_meta.json": dict(parent_tbm, dose_scaled_from="x")}, {})[0] == \
        "system_instruction_distillation"
    # seeded teacher ids: a candidate instruction the legacy roster never named classifies once the seeds know it …
    cand = {"train_budget_meta.json": {"si_id": "candidate_A10", "teacher_forward": "precomputed_topk"}}
    with pytest.raises(D.UnclassifiableSidecar):
        D.fit_target_kind("lora", cand, {})
    v, c, src = D.fit_target_kind("lora", cand, {}, teacher_ids={"candidate_A10": "f" * 64})
    assert (v, c) == ("system_instruction_distillation", "recorded") and "si_text row" in src
    # … and the seeds REPLACE the roster: a roster name the seeds lack is unclassifiable, never silently a teacher
    with pytest.raises(D.UnclassifiableSidecar):
        D.fit_target_kind("relu_steer", {"train_budget_meta.json": parent_tbm}, {}, teacher_ids={"candidate_A10": "f" * 64})
    # a seeded teacher that is also a corpus is still ambiguous
    with pytest.raises(D.UnclassifiableSidecar) as ei:
        D.fit_target_kind("lora", cand, {"candidate_A10": "candidate_A10"}, teacher_ids={"candidate_A10": "f" * 64})
    assert "ambiguous" in str(ei.value)


def test_legacy_mappings_refuse_unknown():
    assert D.model_id_from_legacy("gemma4e4b_it", {"gemma4e4b_it": "gemma4e4b_it"})[1] == "recorded"
    assert D.model_id_from_legacy("nope", {})[0] == "unverified"
    assert D.eval_id_from_spelling("gsm8k_paired200", {"gsm8k_paired200": "gsm8k"})[0] == "gsm8k"
    assert D.eval_id_from_spelling("nope", {})[1] == "unverified"
    assert D.confidence(1, "unverified", "why")["reason"] == "why"


def test_campaign_stem_strips_the_fit_variant_prefix_only():
    """The stem is the LAST two `__` segments (`<si>__<arm>`); everything before is the fit-variant prefix (fit provenance / dose, never the
    paper arm). Every prefix grammar seen in the live banks on 2026-09-19 reduces to its stem; a bare stem, a prompted cell, a bare arm
    and `none` pass through with no prefix."""
    cases = {
        "base__constv2-mix-full__lora_r4_ce": ("constv2-mix-full__lora_r4_ce", "base"),
        "base64s3__constv2ctl2-full__psr_ce_all": ("constv2ctl2-full__psr_ce_all", "base64s3"),
        "itdoc__constv2-mix-full__psr_ce_all": ("constv2-mix-full__psr_ce_all", "itdoc"),
        "itdoc64__constv2ctl2-full__lora_r4_ce": ("constv2ctl2-full__lora_r4_ce", "itdoc64"),
        "derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl": ("constv2-mix-full__lora_r4_ce_minus_ctl", "derived_itdoc64"),
        "derived__constv2-mix-full__psr_ce_all_minus_ctl_distilled": ("constv2-mix-full__psr_ce_all_minus_ctl_distilled", "derived"),
        "derived_base64s1__constv2-mix-full__lora_r4_ce_minus_ctl": ("constv2-mix-full__lora_r4_ce_minus_ctl", "derived_base64s1"),
        "d0p5_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl": ("constv2-mix-full__psr_ce_all_minus_ctl", "d0p5_eb64s1"),
        "dm1it64__constv2-mix-full__lora_r4_ce": ("constv2-mix-full__lora_r4_ce", "dm1it64"),
        "d0p25s1__constv2-mix-full__psr_ce_all": ("constv2-mix-full__psr_ce_all", "d0p25s1"),
        "halfdoseit__constv2-mix-full__lora_r4_ce": ("constv2-mix-full__lora_r4_ce", "halfdoseit"),
        "si12__safety_refusal__lora_r4": ("safety_refusal__lora_r4", "si12"),
        "candsi__A10__lora_r4": ("A10__lora_r4", "candsi"),
        "stacked__constv2-mix-full__lora_r4_ce": ("constv2-mix-full__lora_r4_ce", "stacked"),
        # a withdrawn directory appends a tag to the whole name; a target-prefixed record name has two prefix segments
        "synthmodel__d1it__synth-corpus-full__lora_r4_ce.withdrawn-2026": ("synth-corpus-full__lora_r4_ce", "synthmodel__d1it"),
        # already a stem / a prompted cell / a bare arm / the unsteered cell: unchanged, no prefix
        "constv2-mix-full__lora_r4_ce": ("constv2-mix-full__lora_r4_ce", None),
        "safety_refusal__psr_kl_all": ("safety_refusal__psr_kl_all", None),
        "prompted__si_constitution": ("prompted__si_constitution", None),
        "ar_c0p3__arditi_allpos": ("ar_c0p3__arditi_allpos", None),
        "lora_r4_ce": ("lora_r4_ce", None),
        "none": ("none", None),
    }
    for spelling, want in cases.items():
        assert D.campaign_stem_legacy(spelling) == want, spelling
    assert D.campaign_stem_legacy(None) == (None, None)


# ---- the served key's dose grammar and the fitted objects' operating point (2026-09-20) -----------------------------------------
@pytest.mark.parametrize("view,expected", [
    ("gemma4e4b", 1.0), ("qwen35_4b_it", 1.0), ("llama8b_shortconst", 1.0), ("qwen35_4b_halfdose", 0.5), ("gemma4e4b_it_halfdose", 0.5),
    ("gemma4e4b_it_d0p25", 0.25), ("qwen35_4b_it_d0p5", 0.5), ("nemotron3nano4b_d1p5", 1.5), ("llama8b_dm1", -1.0), ("qwen35_4b_it_d1", 1.0),
    ("qwen35_4b_t0", 1.0), ("qwen35_4b_it_d0p5_t1", 0.5), ("gemma4e4b_halfdose_t0", 0.5),        # a served-controls replicate keeps its tag's dose
    # the seeded dose-family standups spell the dose in the MIDDLE (`<model>_d<tok>_(lora|dist)diff64s<N>`): the token, never the bare 1.0
    # the trailing grammar read them as (2026-09-24: all eleven seeded statics refused on the sidecar cross-check)
    ("qwen35_4b_dm0p5_loradiff64s1", -0.5), ("qwen35_4b_d0p5_loradiff64s2", 0.5), ("qwen35_4b_d0p75_loradiff64s3", 0.75),
    ("qwen35_4b_d1p25_loradiff64s1", 1.25), ("qwen35_4b_d1_loradiff64s1", 1.0), ("qwen35_4b_d1p5_loradiff64s1", 1.5),
    ("qwen35_4b_dm0p5_distdiff64s1", -0.5), ("qwen35_4b_d1p5_distdiff64s1", 1.5),
    ("qwen35_4b_d1p25_loradiff64s1_t0", 1.25), ("qwen35_4b_dm0p5_loradiff64s3_t0", -0.5), ("qwen35_4b_d0p75_distdiff64s1_t1", 0.75),   # controls twins
    # the control families at -1 (2026-09-25): the same middle grammar, the family from the one home (`SEEDED_DOSE_FAMILIES`)
    ("qwen35_4b_dm1_ctllora64s1", -1.0), ("qwen35_4b_dm1_ctlrelu64s3", -1.0), ("qwen35_4b_dm1_ctllora64s2_t0", -1.0), ("qwen35_4b_dm1_ctlrelu64s1_t0", -1.0),
    ("qwen35_4b_dm1_ctllora64", 1.0), ("qwen35_4b_dm1_ctlfoo64s1", 1.0),      # no seed / an undeclared family: not the seeded spelling, so the bare key (the reader of such a directory refuses by name)
    ("qwen35_4b_ar_c0p3", 1.0), ("qwen35_4b_s2", 1.0), ("qwen35_4b_paired_d0p75_s1", 1.0),      # neither grammar: the bare key, as before
    ("", None), (None, None),
])
def test_served_key_dose_grammar(view, expected):
    """A bare served key is the campaign nominal 1.0; `_halfdose` 0.5; `_d<tok>` the token (`p` the point, `m` minus); the seeded dose-family
    spelling `_d<tok>_(lora|dist)diff64s<N>` its middle token; no key, no dose — the one grammar `paper._served_key_dose` and the
    legacy-store readers' `run.strength_nominal` both read."""
    assert D.served_key_dose(view) == expected
    from sisteer.registry import paper as P
    assert P._served_key_dose(view) == expected


def test_served_key_dose_agrees_with_the_seeded_family_readers():
    """The seeded dose-family spelling has three readers and one value: `served_key_dose` (the strength stamp of the sampled statics and the
    shared store reader), `strength_mtdiff.parse_standup` (the dose sweep's own reader, from the stem) and `sampled_static.tag_model_key`
    (the model key, the same spelling stripped). For every live standup, and its `_t<T>` controls twin, all three agree."""
    from sisteer.registry.adapters import sampled_static as SS
    from sisteer.registry.adapters import strength_mtdiff as SM
    # ONE home for the family alternation: the dose sweep's FAMILIES and the served-key grammar's tuple are the same set (pinned at the
    # adapter's import too), so a family added to one cannot read as the bare 1.0 in the other
    assert set(SM.FAMILIES) == set(D.SEEDED_DOSE_FAMILIES) == {"distdiff64", "loradiff64", "ctllora64", "ctlrelu64"}
    for stem, dose in (("dm1", -1.0), ("dm0p5", -0.5), ("d0p5", 0.5), ("d0p75", 0.75), ("d1p25", 1.25), ("d1p5", 1.5)):
        for family in D.SEEDED_DOSE_FAMILIES:
            for seed in (1, 2, 3):
                standup = f"qwen35_4b_{stem}_{family}s{seed}"
                got_stem, got_family, got_seed, nominal, why = SM.parse_standup(standup)
                assert (got_stem, got_family, got_seed, nominal, why) == (stem, family, seed, dose, None), standup
                assert D.served_key_dose(standup) == nominal == dose, standup
                assert D.served_key_dose(f"{standup}_t0") == dose, standup
                assert SS.tag_model_key(standup) == ("qwen35_4b", standup) and SS.tag_model_key(f"{standup}_t0") == ("qwen35_4b", f"{standup}_t0")


def test_fitted_operating_point_is_the_campaign_nominal_for_the_fitted_kinds_only():
    """1.0 is the operating point of the fitted LoRA / ReLU objects, their differences and the paired write; any other
    nominal is 0; an unsteered / prompted row and a published baseline on its own dose axis (alpha, lambda, the raw coefficient) get None
    — their own readers rule their operating points."""
    for kind in D.FITTED_DOSE_OBJECT_KINDS:
        assert D.fitted_is_operating_point(kind, 1.0) == 1 and D.fitted_is_operating_point(kind, 1.0 + 1e-12) == 1
        assert D.fitted_is_operating_point(kind, 0.5) == 0 and D.fitted_is_operating_point(kind, -1.0) == 0 and D.fitted_is_operating_point(kind, 1.5) == 0
        assert D.fitted_is_operating_point(kind, None) is None
    assert set(D.FITTED_DOSE_OBJECT_KINDS) == {"lora", "lora_subtraction", "relu_steer", "relu_steer_distilled_difference", "composed"}
    for kind in ("none", "system_instruction", "static_r_arditi", "alphasteer", "vector_all", "diffmean", None):
        assert D.fitted_is_operating_point(kind, 1.0) is None
