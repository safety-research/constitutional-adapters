"""The shared helpers: view / cell grammars, producer keys, seed-derived instrument and judge ids, object
resolution, the run payload's confidence entries, the row-refusal ledger."""
from __future__ import annotations

import json
import os

import pytest

import _b1_fixtures as F
from sisteer.registry import identity as I
from sisteer.registry import validate as VAL
from sisteer.registry import records as R
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import lineage_common as LC


def test_split_view_and_cell_grammar():
    assert LC.split_view_legacy("gemma4e4b_it_halfdose") == ("gemma4e4b", 0.5, "it_halfdose")
    assert LC.split_view_legacy("qwen35_4b") == ("qwen35_4b", 1.0, "main")
    assert LC.split_view_legacy("llama8b_shortconst") == ("llama8b", 1.0, "shortconst")
    assert LC.cell_grammar_legacy("none")["object_kind"] == "none"
    assert LC.cell_grammar_legacy("none__modal2048")["object_kind"] == "none"
    g = LC.cell_grammar_legacy("constv2-mix-full__psr_ce_all_minus_ctl_distilled")
    assert g["object_kind"] == "relu_steer_distilled_difference" and g["paper_label"] == "difference_relu_distilled" and g["derivation"] == "residual_distillation"
    assert LC.cell_grammar_legacy("constv2ctl2-full__lora_r4_ce")["paper_label"] == "control_lora"
    assert LC.cell_grammar_legacy("prompted__si_constitution")["si_id"] == "si_constitution"
    assert LC.cell_grammar_legacy("arditi")["object_kind"] == "static_r_arditi"
    assert LC.cell_grammar_legacy("mystery")["object_kind"] is None


def test_producer_keys_are_stable_spellings():
    assert LC.producer_key_for_r2_row("v", "c", 1.0, None) == "r2:v|c|1|"
    assert LC.producer_key_for_r2_row("v_halfdose", "c", 0.5, "relu_native") == "r2:v_halfdose|c|0.5|relu_native"
    assert LC.petri_object_key("gemma4e4b", "none") == "gemma4e4b:none"
    assert LC.producer_key_for_grades_root("grades_x/") == "grades_x"


def test_instrument_row_normalises_and_hashes_deterministically():
    s = F.synthetic_seeds()
    a = s.instrument_row("static_sampled_vendor_preset_4096_8192", "pasttense")
    b = s.instrument_row("static_sampled_vendor_preset_4096_8192", "pasttense")
    assert a["instrument_id"] == b["instrument_id"] == I.instrument_id(a)
    assert a["read_at"] == "ladder_final" and a["decode_source"] == "vendor_recommendation" and a["item_set_id"] == "pasttense_items"
    assert a["harness_config_hash"] == "unverified"
    # the same seed under another eval is another instrument (eval_id and item_set are in the identity)
    assert s.instrument_row("static_sampled_vendor_preset_4096_8192", "codeattack")["instrument_id"] != a["instrument_id"]
    with pytest.raises(KeyError):
        s.instrument_row("static_sampled_vendor_preset_4096_8192", "crescendo")


def test_judge_id_and_longest_string_match():
    s = F.synthetic_seeds()
    assert s.judge_id("strongreject_v1_haiku45") == I.judge_id(s.judge_row("strongreject_v1_haiku45"))
    assert s.judge_for_string("judge-model-x / strongreject_v1") == "strongreject_v1_haiku45"
    # the Petri string contains a bare judge model name shared with another row: the longer alternative wins
    assert s.judge_for_string("auditor judge-model-w, judge judge-model-y (Petri)") == "petri_judge_opus5"
    assert s.judge_for_string(None) is None


def test_model_for_view_resolves_fit_and_transfer():
    s = F.synthetic_seeds()
    m = s.model_for_view_legacy(F.VIEW)
    assert (m["model_id"], m["fit_model_id"], m["transfer_kind"], m["dose"]) == (F.MODEL_IT, F.MODEL_BASE, "base_to_instruct", 1.0)
    assert s.model_for_view_legacy(F.VIEW + "_halfdose")["dose"] == 0.5
    g = s.model_for_view_legacy("glm46vflash")
    assert g["transfer_kind"] == "chat_checkpoint" and g["fit_model_id"] == "glm46vflash"
    assert s.model_for_view_legacy("unknownview")["model_id"] is None


def test_run_payload_base_refuses_a_transfer_kind_the_derivation_cannot_resolve(tmp_path):
    """2026-09-30: `DV.transfer_kind` returns the `unverified` sentinel when the fit model is unknown; a payload carrying it in the NOT NULL
    vocabulary column is refused by the harvester and holds its revision. The shared payload builder refuses BY NAME instead (a KeyError
    naming the field, the way an unresolved model or object is), for a model key whose legacy spelling names no transfer_kind and no fit model."""
    s = F.synthetic_seeds()
    mv = dict(s.model_for_view_legacy(F.VIEW), transfer_kind=None, fit_model_id=None)
    import unittest.mock as mock
    with mock.patch.object(type(s), "model_for_view_legacy", return_value=mv):
        obj = LC.ResolvedObject(object_id="a" * 64, object_kind="lora", paper_label="constitution_lora", object_role="constitution", confidence="recorded", source="test")
        with pytest.raises(KeyError, match="transfer_kind"):
            LC.run_payload_base(s, run_id="r", producer_key="k", object_key="o", view_legacy=F.VIEW, cell_id_legacy="c", obj=obj, eval_id="xstest",
                                instrument_row={"instrument_id": "i", "serving_path": "lens"}, instrument_seed_id="seed", lifecycle_state="generated",
                                when_utc="2026-01-01T00:00:00.000Z", when_source="test", token_cap_tokens=512, judge_id=None, judge_seed_id=None,
                                substrate=("grading_host", "derived", {"confidence": "derived", "source": "test"}), lane_source_legacy="synthetic", store_uri=None)


def test_object_resolver_manifest_si_none_and_unresolved(tmp_path):
    s = F.synthetic_seeds()
    res = F.resolver(s, tmp_path)
    assert res.resolve(F.VIEW, "none").object_id == "none"
    si = res.resolve(F.VIEW, F.PROMPTED)
    assert si.object_id == f"si:{F.SI_SHA}" and si.object_kind == "system_instruction"
    lora = res.resolve(F.VIEW, F.LORA)
    assert lora.object_id == F.SHA_A and lora.confidence == "recorded" and lora.manifest_uri
    served = res.resolve(F.VIEW, "arditi", served_sha256="f" * 64)
    assert served.object_id == "f" * 64 and served.object_kind == "static_r_arditi"
    missing = res.resolve(F.VIEW, "safety_refusal__lora_r4")
    assert missing.object_id is None and missing.confidence == "unverified" and "VIEW_MANIFEST" in missing.source
    # every manifest read is a hashed input
    assert all(v.sha256 for v in res.inputs.values())


def test_run_payload_base_names_every_unverified_field_and_validates(tmp_path):
    s = F.synthetic_seeds()
    res = F.resolver(s, tmp_path)
    inst = s.instrument_row("static_greedy_4096_full22", "pasttense")
    obj = res.resolve(F.VIEW, F.LORA)
    rid = I.run_id(LC.CAMPAIGN, "r2:synthview|x|1|", F.LORA, "pasttense", inst["instrument_id"])
    p = LC.run_payload_base(s, run_id=rid, producer_key="r2:synthview|x|1|", object_key=F.LORA, view_legacy=F.VIEW, cell_id_legacy=F.LORA, obj=obj, eval_id="pasttense",
                            instrument_row=inst, instrument_seed_id="static_greedy_4096_full22", lifecycle_state="generated", when_utc="2026-01-01T00:00:00.000Z",
                            when_source="test", token_cap_tokens=4096, judge_id=s.judge_id("strongreject_v1_haiku45"), judge_seed_id="strongreject_v1_haiku45",
                            substrate=("grading_host", "derived", LC.confidence_entry("derived", "test")), lane_source_legacy="r2_export", store_uri=None)
    rec = R.make_record("run", p, F.producer(), I.adapter_record_id(p, "run"), "2026-01-01T00:00:00.000Z")
    v = VAL.validate_record(rec)
    assert isinstance(v, VAL.Validated), v
    fc = p["field_confidence_json"]
    for col in ("code_sha", "prompt_template_sha256", "instrument_match", "applied_positions_mode", "multi_turn_steering_scope"):
        assert fc[col]["confidence"] in ("unverified", "derived"), col
    assert p["transfer_kind"] == "base_to_instruct" and p["strength_nominal"] == 1.0 and p["strength_nominal_source"] == "filed"
    # a fitted object at the filed nominal 1.0 IS at the operating point, its unit the campaign nominal
    assert p["is_operating_point"] == 1 and p["strength_unit"] == "nominal_only"
    assert fc["is_operating_point"]["confidence"] == "derived" and fc["strength_unit"]["confidence"] == "unverified"
    assert p["extra"]["instrument_seed_id"] == "static_greedy_4096_full22"


def test_row_refusals_render_as_source_state():
    r = LC.RowRefusals("x")
    r.add("k1", "object unresolvable: no manifest", "/in")
    r.add("k2", "object unresolvable: no manifest", "/in")
    r.add("k3", "petri two-source guard: keys differ", "/in")
    assert r.summary() == {"object unresolvable": 2, "petri two-source guard": 1}
    rows = r.source_state_rows("x:g", "/in", "a" * 64)
    assert rows[0].json_path == "$.row_refusals" and len(rows[0].current_members) == 3


def test_the_row_refusals_ledger_row_is_keyed_on_the_group_and_belongs_to_one_read():
    """`ROW_REFUSALS_LEDGER_RULE`: the `$.row_refusals` row is keyed on the GROUP (`registry://row-refusals/<source_id>`),
    its sha the inputs' fingerprint (the input SET's content: the sorted shas, order-free, never a path), written for the read that filed it --
    the empty list included; another group gets no row; the ledger's own group asked under another pass stamp refuses by name; a ledger no read
    owns writes nothing."""
    now, later = "2026-10-03T01:00:00.000Z", "2026-10-03T02:00:00.000Z"
    ctx, ctx_later = (AB.AdapterContext(root="", now=t, producer={"script": "tests"}) for t in (now, later))
    inputs = [AB.Input(uri=f"/local/in/{n}", path=f"/local/in/{n}", sha256=c * 64, size=1, mtime_utc=now, role="report", identity_uri=f"/canon/in/{n}")
              for n, c in (("b.json", "b"), ("a.json", "a"))]
    r = LC.RowRefusals("x", read_of=("g", now))
    r.add("k1", "why one", "/local/in/b.json")
    (row,) = r.ledger_rows(ctx, "g", inputs, "x:g")
    assert row.input_uri == AB.row_refusals_uri("x:g") == "registry://row-refusals/x:g" and AB.is_row_refusals_uri(row.input_uri)
    assert (row.source_id, row.json_path, row.found, row.path_resolves, row.current_members) == ("x:g", AB.ROW_REFUSALS_PATH, 1, 1, ["k1 -- why one"])
    assert row.input_sha256 == AB.inputs_fingerprint(inputs) == AB.inputs_fingerprint(list(reversed(inputs))) == I.sha256_hex(
        I.canonical_json(["a" * 64, "b" * 64]).encode("utf-8"))
    # the same bytes under another root fingerprint the same (a pinned emission is reproducible); a repeated sha counts twice
    elsewhere = [AB.Input(uri=f"/other/{n}", path=f"/other/{n}", sha256=i.sha256, size=1, mtime_utc=now, role="report") for n, i in zip("xy", inputs)]
    assert AB.inputs_fingerprint(elsewhere) == row.input_sha256 != AB.inputs_fingerprint(inputs + inputs[:1])
    assert not AB.is_row_refusals_uri("/local/in/a.json")
    # the empty list is a row too (a cleared refusal is seen to clear)
    (empty,) = LC.RowRefusals("x", read_of=("g", now)).ledger_rows(ctx, "g", inputs, "x:g")
    assert empty.current_members == [] and empty.found == 1 and empty.input_uri == row.input_uri
    # another group: nothing; a ledger no read owns: nothing
    assert r.ledger_rows(ctx, "h", inputs, "x:h") == [] and LC.RowRefusals("x").ledger_rows(ctx, "g", inputs, "x:g") == []
    # its own group under another pass stamp: refused by name (a re-activation replays the committed rows and never asks)
    with pytest.raises(AB.AdapterRefusal, match="belongs to the read at") as exc:
        r.ledger_rows(ctx_later, "g", inputs, "x:g")
    assert exc.value.path == "x:g" and "'g'" in exc.value.why and now in exc.value.why and later in exc.value.why
    # the legacy rendering other readers use is unchanged
    assert r.source_state_rows("x:g", "/local/in/b.json", "b" * 64)[0].input_uri == "/local/in/b.json"


def test_iso_ms_and_numeric_helpers():
    assert LC.iso_ms("2026-09-17T00:00:04+00:00", "x") == "2026-09-17T00:00:04.000Z"
    assert LC.iso_ms("2026-09-14T00:05:40.504501+00:00", "x") == "2026-09-14T00:05:40.504Z"
    assert LC.iso_ms("garbage", "fallback") == "fallback"
    assert LC.int_or_none(float("nan")) is None and LC.int_or_none(3.0) == 3 and LC.num_or_none(None) is None


def test_standin_dimension_records_validate():
    s = F.synthetic_seeds()
    recs = LC.standin_dimension_records(s, F.producer(), "2026-01-01T00:00:00.000Z")
    bad = [str(VAL.validate_record(r)) for r in recs if not isinstance(VAL.validate_record(r), VAL.Validated)]
    assert not bad, bad


def test_instrument_row_accepts_a_model_and_the_local_derivation_ignores_it():
    """Without the loader on the checkout the row is derived here per (seed, eval); the model argument selects
    among the loader's per-model expansions when that loader is present (the documented coupling)."""
    seeds = F.synthetic_seeds()
    assert seeds.b0_index() is None   # synthetic seeds: no directory, never the loader
    a = seeds.instrument_row("static_greedy_4096_full22", "pasttense")
    b = seeds.instrument_row("static_greedy_4096_full22", "pasttense", model_id=F.MODEL_IT)
    assert a["instrument_id"] == b["instrument_id"] == seeds.instrument_id("static_greedy_4096_full22", "pasttense", F.MODEL_IT)
    assert seeds.model_id_for_view_legacy(F.VIEW) == F.MODEL_IT and seeds.model_id_for_view_legacy("no_such_setting") is None


# --------------------------------------------------------------------------------------------
# the served baselines' identity chain (2026-09-18): export sha -> standup staged id -> the ladder owner's object id are ONE sha
# --------------------------------------------------------------------------------------------

#: the r2 export's `object_id` for the qwen Arditi baseline (run sdfbase_full22:actorattack:…, held 2026-09-18 for want of the object row)
QWEN_ARDITI_SERVED_SHA = "87ffbc4c34a875a4582f2371b2a31f866f2bc3c14622603a09f9e316ff11ece5"
PINS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "misalign_baseline_pins.json")
LIVE_ARM_VIEW = os.path.join(os.path.expanduser("~"), ".local", "state", "petri_arm_views", "qwen35_4b", "arditi", "VIEW_MANIFEST.json")


def test_qwen_arditi_served_baseline_export_sha_is_the_pinned_ladder_object():
    """The export's sha, the standup's staged object and the ladder owner's pin are the same artefact under the same id: no
    identity mismatch, so the resolver keeps filing runs against the served sha and the owner (arditi_strength) must mint it."""
    from sisteer.registry.adapters import arditi_strength as AR
    with open(PINS, encoding="utf-8") as fh:
        pin = json.load(fh)["qwen35_4b"]["arditi"]
    assert pin["object_sha256"] == QWEN_ARDITI_SERVED_SHA
    assert pin["staged_id"] == "ar_r0p05__arditi" and pin["view"] == "ar_r0p05" and pin["alpha"] == 0.05 and pin["served_layer"] == 12
    pinned = AR._pinned_shas(None)["qwen35_4b"]
    assert pinned["sha"] == QWEN_ARDITI_SERVED_SHA and pinned["served_layer"] == 12 and pinned["alpha"] == 0.05 and pinned["staged_id"] == "ar_r0p05__arditi"


def test_object_resolver_files_the_served_sha_the_arm_view_names(tmp_path):
    """A synthetic Petri arm view naming the standup's staged object: (qwen view, `arditi`) resolves to that sha with the staged id;
    then, where the LIVE arm view exists on this machine (read-only), the same resolution yields the export's sha."""
    root = tmp_path / "petri_arm_views"
    (root / "qwen35_4b" / "arditi").mkdir(parents=True)
    (root / "qwen35_4b" / "arditi" / "VIEW_MANIFEST.json").write_text(json.dumps({"wave": "arditi_strength", "target": "qwen35_4b", "view": "ar_r0p05",
        "entries": [{"cell_id": "arditi", "staged_id": "ar_r0p05__arditi", "object_sha256": QWEN_ARDITI_SERVED_SHA, "identity": "fit_provenance"}]}))
    res = LC.ObjectResolver(F.synthetic_seeds(), objects_root=str(tmp_path / "objects"), petri_manifests_root=str(root))
    obj = res.resolve("qwen35_4b_it", "arditi")
    assert obj.object_id == QWEN_ARDITI_SERVED_SHA and obj.staged_id == "ar_r0p05__arditi" and obj.object_kind == "static_r_arditi"
    assert obj.paper_label == "arditi" and obj.confidence == "recorded" and "VIEW_MANIFEST" in obj.source
    if not os.path.isfile(LIVE_ARM_VIEW):
        pytest.skip("no live Petri arm view on this machine")
    live = LC.ObjectResolver(F.synthetic_seeds(), objects_root=str(tmp_path / "objects"), petri_manifests_root=os.path.dirname(os.path.dirname(os.path.dirname(LIVE_ARM_VIEW))))
    assert live.resolve("qwen35_4b_it", "arditi").object_id == QWEN_ARDITI_SERVED_SHA


def test_relu_steer_paired_write_stem_is_a_composition_with_the_difference_relu_label():
    """The undistilled paired write `<si>__psr_ce_all_minus_ctl` (composed / composition) carries the difference_relu label; the LoRA
    `_minus_ctl` rule is untouched."""
    g = LC.cell_grammar_legacy("constv2-mix-full__psr_ce_all_minus_ctl")
    assert g["object_kind"] == "composed" and g["paper_label"] == "difference_relu" and g["object_role"] == "difference" and g["derivation"] == "composition"
    lora = LC.cell_grammar_legacy("constv2-mix-full__lora_r4_ce_minus_ctl")
    assert lora["object_kind"] == "lora_subtraction" and lora["paper_label"] == "difference_lora"


def test_cell_grammar_reads_the_stem_of_a_staged_spelling():
    """A staged spelling names its fit variant before the stem (`base__`, `itdoc64__`, `d0p5_eb64s1__`, `derived_itdoc64__`, `si12__`); the
    grammar reads the STEM — the prefix is fit provenance, never the arm. Before, `base__constv2ctl2-full__lora_r4_ce` parsed as si=`base`
    and resolved to no kind at all."""
    for spelling in ("base__constv2ctl2-full__lora_r4_ce", "itdoc64__constv2ctl2-full__lora_r4_ce", "base64s2__constv2ctl2-full__lora_r4_ce",
                     "d0p25__constv2ctl2-full__lora_r4_ce", "halfdoseit__constv2ctl2-full__lora_r4_ce"):
        g = LC.cell_grammar_legacy(spelling)
        assert (g["object_kind"], g["paper_label"], g["object_role"], g["si_id"]) == ("lora", "control_lora", "control", "constv2ctl2-full"), spelling
    assert LC.cell_grammar_legacy("derived_itdoc64__constv2-mix-full__lora_r4_ce_minus_ctl")["paper_label"] == "difference_lora"
    assert LC.cell_grammar_legacy("d0p5_eb64s1__constv2-mix-full__psr_ce_all_minus_ctl")["object_kind"] == "composed"
    assert LC.cell_grammar_legacy("derived__constv2-mix-full__psr_ce_all_minus_ctl_distilled")["paper_label"] == "difference_relu_distilled"
    assert LC.cell_grammar_legacy("si12__safety_refusal__psr_kl_all")["paper_label"] == "safety_relu"
    assert LC.cell_grammar_legacy("halfdose__constv2-mix-full__psr_ce_all")["paper_label"] == "constitution_relu"
    # a stack of an inner object is its own kind, never the inner cell's label (defense_cell.py `stacked__<si>__<arm>`)
    st = LC.cell_grammar_legacy("stacked__constv2-mix-full__lora_r4_ce")
    assert st["object_kind"] == "stacked" and st["paper_label"] == "other" and st["derivation"] == "stack"
    # a candidate-instruction distillation's stem names the candidate: no paper arm, no guess
    assert LC.cell_grammar_legacy("candsi__A10__lora_r4")["paper_label"] == "other"
