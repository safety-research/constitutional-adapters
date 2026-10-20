"""Run identity at the registry grain: a producer key spanning several evals yields
one run per eval; two objects under one root yield two runs, one root; emitter and adapter mint the
same id; producer keys round-trip through the legacy columns.
"""
from __future__ import annotations

import pytest

from sisteer.registry import identity as I


def test_run_id_shape_and_determinism():
    a = I.run_id("full22", "grades_x_ch_m", "obj_a", "pasttense", "i" * 64)
    b = I.run_id("full22", "grades_x_ch_m", "obj_a", "pasttense", "i" * 64)
    assert a == b
    parts = I.parse_run_id(a)
    assert parts["campaign"] == "full22" and parts["eval_id"] == "pasttense" and parts["instrument_prefix"] == "i" * 12
    assert len(parts["producer_sha16"]) == 16


def test_multi_eval_key_yields_one_run_per_eval():
    """A Petri run key with seven behaviours → seven runs + the pooled one, one producer key."""
    pk = I.split_producer_key("petri_run_key", "petri-run-0001", object_key="constitution_lora", scope={"tag": "tagA"})
    evals = [(f"petri_behaviour{i}", "inst" + "0" * 60) for i in range(7)] + [("petri_pooled", "inst" + "0" * 60)]
    ids = I.run_ids_for_producer_key("petri_campaign", pk, evals)
    assert len(set(ids.values())) == 8
    assert pk.execution_key == "petri-run-0001" and pk.object_key == "tagA:constitution_lora"


def test_two_objects_under_one_grades_root_are_two_runs_one_root():
    root = "grades_mt4k_ch_synthmodel_it"
    a = I.split_producer_key("grades_root", root, object_key="constv2-mix-full__lora_r4_ce")
    b = I.split_producer_key("grades_root", root, object_key="constv2ctl2-full__lora_r4_ce")
    inst = "i" * 64
    ids = set()
    for pk in (a, b):
        for ev in ("crescendo", "pair", "actorattack"):
            ids.add(I.run_id("full22", pk.execution_key, pk.object_key, ev, inst))
    assert len(ids) == 6
    assert a.execution_key == b.execution_key == root
    assert a.parts["tag"] == "mt4k" and a.parts["cluster"] == "ch"


def test_grades_root_without_object_key_refuses():
    with pytest.raises(ValueError):
        I.split_producer_key("grades_root", "grades_mt4k_ch_m")


def test_cell_key_grammar_and_controls_override():
    key = "constitution_lora__xstest__t0__a0__deadbeef__cafef00d"
    pk = I.split_producer_key("cell_key", key)
    assert pk.parts["tier_or_task"] == "xstest" and pk.parts["trial"] == "0"
    assert pk.object_key == "constitution_lora@deadbeef"
    # one controls store holding two evals → two runs, one key
    inst = "i" * 64
    r1 = I.run_id("full22", pk.execution_key, pk.object_key, "xstest", inst)
    r2 = I.run_id("full22", pk.execution_key, pk.object_key, "or_bench_hard", inst)
    assert r1 != r2
    with pytest.raises(ValueError):
        I.split_producer_key("cell_key", "not-a-key")


def test_two_decodes_of_one_execution_are_two_instruments_two_runs():
    pk = I.split_producer_key("bon_dir", "bon_store_7", object_key="constitution_lora", scope={"rung": "4096"})
    i1 = I.instrument_id({"eval_id": "strongreject_bon", "item_set_id": "sr_bon", "decode_temperature": 1.0})
    i2 = I.instrument_id({"eval_id": "strongreject_bon", "item_set_id": "sr_bon", "decode_temperature": 0.7})
    assert i1 != i2
    assert I.run_id("c", pk.execution_key, pk.object_key, "strongreject_bon", i1) != \
        I.run_id("c", pk.execution_key, pk.object_key, "strongreject_bon", i2)


def test_battery_block_and_em_row_grammars():
    pk = I.split_producer_key("battery_block", "synthmodel_it:constitution_lora")
    assert pk.object_key == "constitution_lora" and pk.parts["model_transfer"] == "synthmodel_it"
    em = I.split_producer_key("em_row", "em_panel_row_3", object_key="difference_lora", scope={"target": "synthmodel_it", "behaviour": "b"})
    assert em.parts["target"] == "synthmodel_it"
    with pytest.raises(ValueError):
        I.split_producer_key("em_row", "x")
    with pytest.raises(ValueError):
        I.split_producer_key("no_such_grammar", "x", object_key="y")


def test_emitter_and_adapter_mint_identical_ids(monkeypatch):
    """The emitter (from the resolved spec) and the adapter (from the store) call the same function."""
    from sisteer.registry import emit, synth
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test writes to its own temporary directory, so it opts in
    plan = synth.run_payload("store_key_9", "none", "none")
    adapter_side = I.run_id(plan["campaign"], plan["producer_run_key_legacy"], plan["producer_object_key_legacy"],
                            plan["eval_id"], plan["instrument_id"])
    assert plan["run_id"] == adapter_side
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        paths = emit.write_planned_run(d, {k: plan[k] for k in plan if k not in ("run_id",)}, script="test")
        import json
        with open(paths[0]) as fh:
            rec = json.loads(fh.readline())
        assert rec["payload"]["run_id"] == adapter_side


def test_ids_reject_separator_in_campaign_or_eval():
    with pytest.raises(ValueError):
        I.run_id("a:b", "k", "o", "e", "i" * 64)
    with pytest.raises(ValueError):
        I.run_id("a", "k", "o", "e:v", "i" * 64)
    with pytest.raises(ValueError):
        I.run_id("a", "", "o", "e", "i" * 64)


def test_object_and_judgment_and_computation_ids():
    assert I.object_id("none") == "none"
    assert I.object_id("system_instruction", si_text_sha256="a" * 64) == "si:" + "a" * 64
    assert I.object_id("lora", artifact_sha256="b" * 64) == "b" * 64
    with pytest.raises(ValueError):
        I.object_id("lora")
    assert I.judgment_id("r", "j") == I.judgment_id("r", "j") and I.judgment_id("r", "j") != I.judgment_id("r", "j", "s")
    assert len(I.computation_id("alg", "1", "seed") ) == 16
    assert I.adapter_record_id({"a": 1}, "run") == I.adapter_record_id({"a": 1}, "run")
    assert I.adapter_record_id({"a": 1}, "run") != I.adapter_record_id({"a": 1}, "run_metric")


# ---- one minting function for judge and instrument identities --------------------------------------------------------

def test_judge_id_positional_and_mapping_agree_and_differ_per_field():
    sha_a, sha_b = "a" * 64, "b" * 64
    j = I.judge_id("judge-model-x", sha_a, sha_b, "score > 0.5")
    assert len(j) == 64 and j == I.judge_id({"judge_model": "judge-model-x", "rubric_sha256": sha_a, "protocol_sha256": sha_b,
                                              "threshold_rule": "score > 0.5", "judge_role": "judge", "notes": "ignored"})
    assert j != I.judge_id("judge-model-x", sha_a, sha_b, "score > 0.6")
    assert j != I.judge_id("judge-model-y", sha_a, sha_b, "score > 0.5")
    assert j != I.judge_id("judge-model-x", sha_b, sha_a, "score > 0.5")


def test_instrument_id_same_fields_same_id_one_field_differs():
    base = {"eval_id": "gsm8k", "item_set_id": "gsm8k_500", "harness": "lm_eval", "harness_version": "0.4", "n_shot": 8,
            "cot_prompting": "cot", "scoring_rule": "exact_match", "token_cap_tokens": 4096, "token_cap_ladder": [512, 4096],
            "decode_source": "greedy", "temperature": 0.0, "top_p": None, "top_k": None, "min_p": None, "repetition_penalty": None,
            "presence_penalty": None, "decode_seed_rule": "none", "thinking_mode": "off", "n_trials": 1, "bon_n": None,
            "max_turns": None, "serving_path": "lora_native", "applied_positions_mode": "all", "judge_id": None}
    a = I.instrument_id(base)
    assert len(a) == 64 and a == I.instrument_id(dict(base))
    assert a == I.instrument_id(dict(base, notes="free text", extra_json="{}"))                 # non-identity fields ignored
    assert a != I.instrument_id(dict(base, prompt_template_sha256="c" * 64))                     # The template IS identity
    for field, other in (("token_cap_tokens", 512), ("temperature", 0.7), ("thinking_mode", "on"), ("judge_id", "j" * 64),
                         ("token_cap_ladder", [4096]), ("serving_path", "lens")):
        assert I.instrument_id(dict(base, **{field: other})) != a, field
    assert sorted(I.instrument_identity_fields(base)) == sorted(dict(I.INSTRUMENT_ID_FIELDS))


def test_instrument_row_spellings_mint_the_same_id_as_the_canonical_names():
    canonical = {"eval_id": "e", "item_set_id": "s", "harness": "bespoke_sisteer", "harness_version": "0.1", "scoring_rule": "judge_threshold",
                 "token_cap_tokens": 512, "token_cap_ladder": [512], "decode_source": "greedy", "temperature": 0.0, "top_k": 20,
                 "decode_seed_rule": "none", "thinking_mode": "off", "n_trials": 1, "serving_path": "lora_native",
                 "applied_positions_mode": "all", "judge_id": "j" * 64}
    row = {"eval_id": "e", "item_set_id": "s", "harness": "bespoke_sisteer", "harness_version": "0.1", "scoring_rule": "judge_threshold",
           "token_cap_tokens": 512, "token_cap_ladder_json": "[512]", "decode_source": "greedy", "decode_temperature": 0.0,
           "decode_top_k": 20.0, "decode_seed_rule": "none", "thinking_mode": "off", "n_trials": 1, "serving_path": "lora_native",
           "applied_positions_mode": "all", "judge_id": "j" * 64, "notes": "free text", "extra_json": "{}"}
    assert I.instrument_id(row) == I.instrument_id(canonical)
    assert I.instrument_id(dict(row, decode_top_k=21)) != I.instrument_id(canonical)
    # the prompt template sha and the harness config hash ARE identity — a run under another template is another instrument
    assert I.instrument_id(dict(row, prompt_template_sha256="p" * 64)) != I.instrument_id(canonical)
    assert I.instrument_id(dict(row, harness_config_hash="h" * 8)) != I.instrument_id(canonical)


def test_conflicting_alias_and_canonical_spellings_refuse_in_either_order():
    """`temperature` and `decode_temperature` naming different values is a refusal after typed normalisation,
    whatever the mapping order; equal values (0 vs 0.0) are one value; the canonical name is what the id records."""
    base = {"eval_id": "e", "item_set_id": "s", "harness": "h", "scoring_rule": "judge_threshold", "token_cap_tokens": 512}
    for pair in ({"temperature": 0.7, "decode_temperature": 0.0}, {"decode_temperature": 0.0, "temperature": 0.7}):
        with pytest.raises(ValueError, match="decode_temperature"):
            I.instrument_id(dict(base, **pair))
    same = I.instrument_id(dict(base, decode_temperature=0.0))
    assert I.instrument_id(dict(base, temperature=0, decode_temperature=0.0)) == same
    assert I.instrument_id(dict(base, decode_temperature=0.0, temperature=0)) == same
    assert I.instrument_identity_fields(dict(base, temperature=0.5))["decode_temperature"] == 0.5
    with pytest.raises(ValueError, match="token_cap_ladder"):
        I.instrument_id(dict(base, token_cap_ladder=[512], token_cap_ladder_json="[1024]"))
    assert I.instrument_id(dict(base, token_cap_ladder=[512], token_cap_ladder_json="[512]")) == I.instrument_id(dict(base, token_cap_ladder_json=[512]))


def test_object_kind_from_id_reads_the_two_kinds_an_id_spells_and_nothing_else():
    """`none` and `si:<sha>` state their kind by their spelling (identity.object_id's two prefixed forms); a staged artifact's sha states
    none -- its kind is on its object row -- and a non-string or an unprefixed id reads None rather than a guess. What a job-start plan
    or a served runs file entry hands the scope rule, which reads the kind for these two values alone."""
    assert I.object_kind_from_id("none") == "none"
    assert I.object_kind_from_id(I.object_id("system_instruction", si_text_sha256="a" * 64)) == "system_instruction"
    assert I.object_kind_from_id(I.SI_OBJECT_ID_PREFIX + "b" * 64) == "system_instruction"
    assert I.object_kind_from_id(I.object_id("lora", artifact_sha256="c" * 64)) is None
    for other in (None, 7, "", "si", "SI:" + "a" * 64, "c" * 64, "constitution_lora@a1b2c3d4"):
        assert I.object_kind_from_id(other) is None, other
