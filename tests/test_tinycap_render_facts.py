"""The contracts surface the math500 thinking-on experiment adds: the family
render facts, the per-op header-located predicate on SERVED positions, the
eight-tuple authorisation registry through the ONE checker in BOTH doors'
argument shapes, the sampling rows and their identity, the identity version,
the retry legacy literal, the request-evidence record, the ledger's named
refusals, the transfer-provenance block, the expected stamp triples, predicate
(A) `render_certified`, predicate (B′) `lens_position_attested` and predicate
(C) `lens_on_bite_verdict`.

Every fixture here is SYNTHETIC (placeholder digests, a `TESTHOST` standup).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sisteer.capeval.serving import proxy as px
from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
OBJ = C.M5T_OBJECT_CELL
OBJ_HARNESS = "si_constv2_mix_full__lora_r4_ce_minus_ctl"
SHA = hashlib.sha256(b"synthetic-object-bytes-for-render-fact-tests").hexdigest()
FOUR = ("gemma4e4b", "qwen35_4b", "glm46vflash", "nemotron3nano4b")


# --------------------------------------------------------------------------- #
# the family render facts                                                      #
# --------------------------------------------------------------------------- #


def test_the_render_facts_equal_the_committed_measurement_and_the_parent_constants():
    """The four facts' header id tuples ARE the committed 2026-09-09
    measurement (`configs/tinycap/thinking_header_measure_2026-09-09.json`, label ->
    family), the Qwen fact equals the parent's constants, the split modes are
    GLM completion-opened `explicit_tags`, Qwen / Nemotron prompt-opened
    `implicit_open` and gemma parser-separated `record_only`, and the evidence
    status is `certified` for all four: the two parent-verified families, and
    GLM and Nemotron by G-PARSER on their diagnostic smokes' real completions
    (2026-09-10 — `tests/test_think_split_real_completions.py`). The refusal
    of a `candidate` fact is covered on SYNTHETIC demotions below."""
    doc = json.loads((REPO / C.THINKING_HEADER_MEASURE_JSON).read_text())
    by_label = {f["label"]: f for f in doc["families"]}
    for label, family in (("gemma-4-E4B-it", "gemma4"), ("Qwen3.5-4B", "qwen3_5"),
                          ("GLM-4.6V-Flash", "glm4v"), ("Nemotron-3-Nano-4B", "nemotron_h")):
        fact = C.FAMILY_RENDER_FACTS[family]
        regimes = by_label[label]["regimes"]
        assert list(fact.off_ids) == regimes["off"]["header_ids"], family
        assert list(fact.on_ids) == regimes["on"]["header_ids"], family
        assert fact.family == family and fact.evidence_status in C.RENDER_EVIDENCE_STATUSES
        assert fact.as_dict()["split_mode"] == fact.split_mode
    q = C.FAMILY_RENDER_FACTS["qwen3_5"]
    assert q.on_ids == C.QWEN35_HEADER_IDS_ON and q.off_ids == C.QWEN35_HEADER_IDS_OFF
    assert (q.think_open_id, q.think_close_id) == (C.QWEN35_THINK_OPEN_ID, C.QWEN35_THINK_CLOSE_ID)
    assert {f: C.FAMILY_RENDER_FACTS[f].split_mode for f in C.FAMILY_RENDER_FACTS} == {
        "gemma4": "record_only", "qwen3_5": "implicit_open", "glm4v": "explicit_tags",
        "nemotron_h": "implicit_open", "llama": "none"}
    assert {f: C.FAMILY_RENDER_FACTS[f].prompt_opens_think_on for f in C.FAMILY_RENDER_FACTS} == {
        "gemma4": False, "qwen3_5": True, "glm4v": False, "nemotron_h": True, "llama": False}
    # llama-paper-rows (P8): the no-switch family entered as a CANDIDATE and was CERTIFIED by the GSM8K wave's diagnostic
    # smoke (its completions committed under contracts.NO_SWITCH_SMOKE_COMPLETIONS_DIR and re-read by
    # tests/test_no_switch_smoke_real_completions.py); every switched family stays `certified`
    assert {f: C.FAMILY_RENDER_FACTS[f].evidence_status for f in C.FAMILY_RENDER_FACTS} == {
        "gemma4": "certified", "qwen3_5": "certified", "glm4v": "certified", "nemotron_h": "certified",
        "llama": "certified"}
    # the switch flag: every switched fact keeps its dict byte-identical but for the added `thinking_switch: true`
    assert {f: C.FAMILY_RENDER_FACTS[f].thinking_switch for f in C.FAMILY_RENDER_FACTS} == {
        "gemma4": True, "qwen3_5": True, "glm4v": True, "nemotron_h": True, "llama": False}
    ll = C.FAMILY_RENDER_FACTS["llama"]
    assert ll.off_ids == ll.on_ids == (128006, 78191, 128007, 271) and not ll.header_moves
    assert ll.server_parser is None and ll.think_open == ll.think_close == "" and ll.think_open_id is ll.think_close_id is None
    assert ll.as_dict()["thinking_switch"] is False and "b48c47f644389271" in ll.evidence
    # header movement is a fact about ID TUPLES: Nemotron moves at unchanged width
    n = C.FAMILY_RENDER_FACTS["nemotron_h"]
    assert len(n.off_ids) == len(n.on_ids) == 6 and n.header_moves
    assert not C.FAMILY_RENDER_FACTS["gemma4"].header_moves
    assert C.FAMILY_RENDER_FACTS["glm4v"].server_parser is None and C.FAMILY_RENDER_FACTS["gemma4"].server_parser == "gemma4"
    assert C.THINKING_HEADER_MEASURE_JSON in C.THINKING_HEADER_EVIDENCE
    assert (REPO / "configs" / "tinycap" / "thinking_header_verification.md").is_file()


def test_family_by_model_key_agrees_with_the_launcher_registry_and_covers_geometry():
    from sisteer.capeval.serving import geometry as sg

    # exactly the sweep's eight: a key outside the table (geometry.ONBOARDED_MODEL_KEYS, the front door's demo model among them)
    # has NO family, so every family-keyed gate refuses it by name rather than guessing from a name prefix
    assert set(C.FAMILY_BY_MODEL_KEY) == set(sg.MODEL_KEYS)
    for key in sg.ONBOARDED_MODEL_KEYS:
        assert key not in C.FAMILY_BY_MODEL_KEY and C.family_of_model_key(key) is None and C.render_fact_for(C.family_of_model_key(key)) is None, key
    for slug, row in C.MODELS.items():
        assert C.FAMILY_BY_MODEL_KEY[row.model_key] == row.family, slug
    assert C.family_of_model_key("glm46vflash") == "glm4v" and C.family_of_model_key("nope") is None


def test_the_worked_example_row_reads_the_qwen35_4b_render_facts_through_its_row_not_an_onboarded_key():
    """paper-companion-repo: the demo's launcher row (contracts.MODELS["qwen35_4b_demo"]) carries model_key qwen35_4b and
    family qwen3_5, so every family-keyed gate reads the 4B's CERTIFIED facts off the ROW — the render facts for the demo are
    the 4B's. The onboarded key qwen35_4b_demo itself stays outside FAMILY_BY_MODEL_KEY (the test above): a config naming the
    key and no row still refuses by name rather than guessing a family from the prefix."""
    demo, q = C.model_row("qwen35_4b_demo"), C.model_row("qwen35_4b")
    assert (demo.model_key, demo.family) == (q.model_key, q.family) == ("qwen35_4b", "qwen3_5")
    assert C.FAMILY_BY_MODEL_KEY[demo.model_key] == demo.family
    fact = C.render_fact_for(C.family_of_model_key(demo.model_key))
    assert fact is C.FAMILY_RENDER_FACTS["qwen3_5"] is C.render_fact_for(q.family) and C.render_fact_certified(fact)
    assert fact.evidence_status == C.RENDER_EVIDENCE_CERTIFIED and C.render_fact_refusal(demo.family) is None
    assert "qwen35_4b_demo" in sg.ONBOARDED_MODEL_KEYS and C.family_of_model_key("qwen35_4b_demo") is None


def test_render_fact_refusal_names_unverified_and_not_certified_apart(monkeypatch):
    for fam in ("qwen3_5", "gemma4", "glm4v", "nemotron_h"):    # glm4v / nemotron_h certified by G-PARSER (2026-09-10)
        assert C.render_fact_refusal(fam) is None, fam
        assert C.render_fact_certified(C.FAMILY_RENDER_FACTS[fam]), fam
    why = C.render_fact_refusal("mistral")           # a family with NO fact at all
    assert "render fact unverified" in why
    # llama-paper-rows: the no-switch family HAS a fact (split `none`), CERTIFIED by the GSM8K diagnostic smoke — production
    # admits it; a SYNTHETIC demotion of it (only `evidence_status` moves) is refused by name, citing the smoke's evidence
    assert C.render_fact_refusal("llama") is None and C.render_fact_certified(C.FAMILY_RENDER_FACTS["llama"])
    demoted_llama = C.FAMILY_RENDER_FACTS["llama"]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "llama": demoted_llama})
    why = C.render_fact_refusal("llama")
    assert "render fact not certified" in why and "'candidate'" in why and C.NO_SWITCH_SMOKE_COMPLETIONS_DIR in why
    assert C.render_fact_refusal("llama", require_certified=False) is None
    monkeypatch.undo()
    assert not C.render_fact_certified(None)
    # a `candidate` fact (a SYNTHETIC demotion of Nemotron's — only `evidence_status` moves) refuses production by
    # name, citing the gate and its own evidence, and is admitted under the dry run's `pre` reading
    demoted = C.FAMILY_RENDER_FACTS["nemotron_h"]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "nemotron_h": demoted})
    why = C.render_fact_refusal("nemotron_h")
    assert "render fact not certified" in why and "'candidate'" in why and "G-PARSER" in why and demoted.evidence in why
    assert C.render_fact_refusal("nemotron_h", require_certified=False) is None
    assert not C.render_fact_certified(demoted)


def test_check_render_fact_certified_refuses_a_candidate_fact_whatever_the_family(monkeypatch):
    """The production seam keys on `evidence_status`, not on the family's name: a SYNTHETIC
    candidate copy of the (certified) GLM fact is refused by name, the dry run's `pre` reading
    (`require_certified=False`) still returns it, and the real table's certified GLM is admitted —
    so the refusal is covered whether or not any real family is still `candidate`."""
    real = C.FAMILY_RENDER_FACTS["glm4v"]
    assert C.check_render_fact_certified("glm4v") == real and real.evidence_status == C.RENDER_EVIDENCE_CERTIFIED
    demoted = real._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "glm4v": demoted})
    with pytest.raises(C.RenderFactUnverified, match="render fact not certified") as e:
        C.check_render_fact_certified("glm4v")
    assert "'glm4v'" in str(e.value) and "'candidate'" in str(e.value) and "G-PARSER" in str(e.value)
    assert C.check_render_fact_certified("glm4v", require_certified=False) == demoted
    assert not C.render_fact_certified(demoted)
    # a fact with NO split mode is "unverified", a different refusal from "not certified"
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "glm4v": real._replace(split_mode=None)})
    with pytest.raises(C.RenderFactUnverified, match="render fact unverified"):
        C.check_render_fact_certified("glm4v", require_certified=False)


def test_g1_thinking_on_refusal_lifts_on_a_certified_split_and_keys_the_mask_on_header_located(monkeypatch):
    """Clause (a) — the certified harness-side split IS the parser:
    qwen3_5, glm4v and nemotron_h (all certified) bare / prompted admitted;
    a `candidate` fact (a SYNTHETIC demotion of Nemotron's) refuses "render
    fact not certified" and admits under the dry run's `pre` reading. Clause
    (b) keys on fitted AND header_located: an all-position fitted qwen cell is
    not refused for the mask; a header-located one is."""
    ok = dict(certified_split_ok=True)                       # the driver's / dry run's opt-in (LA17)
    for fam in ("qwen3_5", "glm4v", "nemotron_h"):           # glm4v / nemotron_h certified by G-PARSER (2026-09-10)
        assert C.thinking_on_refusal(fam, fitted=False, **ok) is None, fam
        assert "no pinned reasoning parser" in C.thinking_on_refusal(fam, fitted=False), fam   # opt-in, not default
    demoted = C.FAMILY_RENDER_FACTS["nemotron_h"]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "nemotron_h": demoted})
    assert "render fact not certified" in C.thinking_on_refusal("nemotron_h", fitted=False, **ok)
    assert C.thinking_on_refusal("nemotron_h", fitted=False, require_certified=False, **ok) is None
    monkeypatch.undo()
    assert C.thinking_on_refusal("nemotron_h", fitted=False, **ok) is None
    for fam in ("glm4v", "nemotron_h"):
        assert C.thinking_on_refusal(fam, fitted=True, header_located=False, require_certified=False, **ok) is None
        # a header-located fitted cell: the mask refusal binds whatever the fact
        assert "MOVES between modes" in C.thinking_on_refusal(fam, fitted=True, require_certified=False, **ok)
    assert C.thinking_on_refusal("qwen3_5", fitted=True, header_located=False, **ok) is None
    assert "MOVES between modes" in C.thinking_on_refusal("qwen3_5", fitted=True, **ok)
    assert C.thinking_on_refusal("gemma4", fitted=True) is None
    assert "no thinking switch" in C.thinking_on_refusal("llama", fitted=False)


# --------------------------------------------------------------------------- #
# the per-op header-located predicate on SERVED positions                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("served,want", [
    (["all"], False), ("all", False), (["last"], False),
    (["response"], True), (["response_all_turns"], True),
    (["all", "response"], True),                              # a MIXED set reads header-located
    (None, True), ([], True), (["bogus"], True), (42, True),  # unknown -> fail closed
    ({"positions": "all"}, False),                             # a resolved spec, unwrapped
    ({"kind": px.PETRI_STACKED_KIND if hasattr(px, "PETRI_STACKED_KIND") else "stacked",
      "object": {"positions": "response"}}, True),            # a stacked spec's OBJECT leg
    ({"kind": "stacked", "object": {"positions": "all"}}, False),
])
def test_header_located_reads_served_positions_stacked_unwrapped_and_fails_closed(served, want):
    assert C.header_located(served) is want
    assert C.HEADER_LOCATED_MODES == tuple(px.HEADER_LOCATED_MODES) == ("response", "response_all_turns")
    assert C.LENS_SERVED_POSITIONS_KEY == px.LENS_SERVED_POSITIONS_KEY == "lens_served_positions"


def test_served_positions_of_cell_reads_the_map_or_none():
    ep = {C.LENS_SERVED_POSITIONS_KEY: {OBJ_HARNESS: ["all"], "other": "response"}}
    assert C.served_positions_of_cell(ep, OBJ_HARNESS) == ["all"]
    assert C.served_positions_of_cell(ep, "other") == ["response"]
    assert C.served_positions_of_cell(ep, "absent") is None
    assert C.served_positions_of_cell({}, OBJ_HARNESS) is None and C.served_positions_of_cell(None, OBJ_HARNESS) is None


def test_the_declaration_predicates_take_header_located_and_default_to_the_parents_case():
    """`thinking_on_declaration(lens, arm, header_located)` emits the key ONLY for
    (True, on, True); every consumer's default keeps the parent's Qwen
    `positions: response` request byte-identical."""
    assert C.thinking_on_declaration(lens=True, arm="on") == {"petri_thinking_on": True}
    assert C.thinking_on_declaration(lens=True, arm="on", header_located=True) == {"petri_thinking_on": True}
    assert C.thinking_on_declaration(lens=True, arm="on", header_located=False) == {}
    assert C.thinking_on_declaration(lens=False, arm="on", header_located=True) == {}
    assert C.thinking_on_declaration(lens=True, arm="off", header_located=True) == {}
    assert C.render_fragment(lens=True, arm="on", header_located=False) == C.chat_template_kwargs("on")
    assert C.render_fragment(lens=True, arm="on") == {**C.chat_template_kwargs("on"), "petri_thinking_on": True}
    assert C.is_declared_configuration(OBJ, arm="on", out_of_support=True) is True
    assert C.is_declared_configuration(OBJ, arm="on", out_of_support=True, header_located=False) is False
    assert C.expected_serve_out_of_support(lens=True, arm="on", out_of_support=True) == ["thinking"]
    assert C.expected_serve_out_of_support(lens=True, arm="on", out_of_support=True, header_located=False) is None
    assert len(C.thinking_on_override_refusals("qwen3_5", fitted=True, header_located=False)) == 1   # the parser reason only


# --------------------------------------------------------------------------- #
# the authorisation registry through the ONE checker                           #
# --------------------------------------------------------------------------- #


def _driver_shape(exp, cell=None, **over):
    """The DRIVER door's argument shape (`served_capability_run.cmd_generate`):
    the resolved model KEY, the harness cell id, the parsed control tokens."""
    req = dict(model=exp.model, cell=cell or C.parse_launcher_cell(exp.cells[0]).harness_id,
               controls=list(exp.controls), mode=exp.modes[-1], sampling_mode=exp.sampling_mode,
               samples=exp.samples, reduction_mode=None, pass_no=None, where="driver run")
    req.update(over)
    return req


def _launcher_shape(exp, **over):
    """The LAUNCHER door's shape (`tinycap_served.py`): the launcher slug, the
    TrainTest cell id, the control tokens, the planned reduction mode and pass."""
    req = dict(model=exp.model, cell=exp.cells[0], controls=list(exp.controls), mode=exp.modes[-1],
               sampling_mode=exp.sampling_mode, samples=exp.samples, reduction_mode=exp.reduction_mode,
               pass_no=exp.pass_no, where="launcher plan")
    req.update(over)
    return req


#: the published run's EIGHT tuples (minus_ctl only) and the eight-arm extension's THREE `_lora2` tuples (E1, plan P4:
#: the two extra all-position LoRA cells per extension model, minus_ctl's shape under a NEW name)
M5T_PUBLISHED_NAMES = [
    "math500_thinking_on_full22_gemma4e4b", "math500_thinking_on_full22_qwen35_4b",
    "math500_thinking_on_full22_glm46vflash", "math500_thinking_on_full22_nemotron3nano4b",
    "math500_thinking_on_full22_qwen35_4b_general",
    "math500_thinking_on_full22_qwen35_4b_battery", "math500_thinking_on_full22_nemotron3nano4b_battery",
    "math500_thinking_on_full22_glm46vflash_battery"]
M5T_EXT_NAMES = [f"math500_thinking_on_full22_{m}_lora2" for m in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")]
#: The AlphaSteer / Arditi rung's tuple per SWITCH model — the same shape as
#: a `_lora2` tuple (minus_ctl's: thinking-ON, stacked, pass 1, the primary sampling row, undeclared, the MATH-500
#: cap) over the two EXTERNAL steering cells instead of the two extra LoRA ones. A no-switch family's steering row is
#: IN SUPPORT (one render, nothing to authorise), so llama has none; glm is the appendix model and has none either.
M5T_STEER_NAMES = [f"math500_thinking_on_full22_{m}_steer" for m in ("gemma4e4b", "qwen35_4b", "nemotron3nano4b")]
#: arditi-qwen-all-positions: the ALL-POSITIONS Arditi arm's OWN MATH-500 tuple, qwen35_4b only. A tuple of its own
#: rather than a widening of `M5T_STEER_CELLS`, because that constant is the `cells` of nine FINISHED tuples and this
#: arm is false of their declaration on three counts (BOTH phases, a raw-coefficient dose, a user override over
#: budget). Its `cells` are the single-object standup the arm requires: the bare anchor and the one external object.
M5T_ALLPOS_NAMES = [f"math500_thinking_on_full22_{m}_steer_allpos" for m in ("qwen35_4b",)]
#: The MATH-500 seed wave's two NATIVE tuples on qwen35_4b — the five lane cells (`_native`) and
#: the composed paired-write arm (`_native_paired`), the native engine's served identity (reduction_mode not_applicable), BESIDE the
#: lens-era tuples: a request naming a LoRA cell both a lens-era tuple and `_native` carry with the served mode UNSTATED is ambiguous by
#: construction (the GSM8K property, handled the same way below), which is why every door states it
M5T_NATIVE_NAMES = [f"math500_thinking_on_full22_{m}_native" for m in ("qwen35_4b",)]
M5T_NATIVE_PAIRED_NAMES = [f"math500_thinking_on_full22_{m}_native_paired" for m in ("qwen35_4b",)]


def test_the_registry_holds_the_parent_and_eight_new_tuples_and_each_matches_only_its_own_run():
    assert set(C.AUTHORISED_OOS_EXPERIMENTS) == {C.AUTHORISED_OOS_LEGACY_NAME, *C.M5T_EXPERIMENT_NAMES, *C.G8K_EXPERIMENT_NAMES}
    assert C.AUTHORISED_OOS_EXPERIMENTS[C.AUTHORISED_OOS_LEGACY_NAME] is C.AUTHORISED_OOS_THINKING
    assert len(C.M5T_EXPERIMENT_NAMES) == 8 + 3 + 3 + 1 + 1 + 1 == 17
    assert sorted(C.M5T_EXPERIMENT_NAMES) == sorted(M5T_PUBLISHED_NAMES + M5T_EXT_NAMES + M5T_STEER_NAMES + M5T_ALLPOS_NAMES
                                                    + M5T_NATIVE_NAMES + M5T_NATIVE_PAIRED_NAMES)
    assert M5T_EXT_NAMES == [C.m5t_ext_tuple_name(m) for m in C.M5T_EXT_MODELS]
    assert M5T_STEER_NAMES == [C.m5t_steer_tuple_name(m) for m in C.M5T_STEER_MODELS]
    assert M5T_ALLPOS_NAMES == [C.m5t_allpos_tuple_name(m) for m in C.ALLPOS_MODELS]
    assert M5T_NATIVE_NAMES == [C.m5t_native_tuple_name(m) for m in C.M5T_NATIVE_MODELS]
    assert M5T_NATIVE_PAIRED_NAMES == [C.m5t_native_paired_tuple_name(m) for m in C.M5T_NATIVE_MODELS]
    modes = {n: C.AUTHORISED_OOS_EXPERIMENTS[n].sampling_mode for n in C.M5T_EXPERIMENT_NAMES}
    assert sum(m == C.SAMPLING_THINKING_CARD for m in modes.values()) == 4 + 3 + 3 + 1 + 2
    assert sum(m == C.SAMPLING_GENERAL for m in modes.values()) == 1
    assert sum(m == C.SAMPLING_BATTERY for m in modes.values()) == 3
    for name in C.M5T_EXPERIMENT_NAMES:
        exp = C.AUTHORISED_OOS_EXPERIMENTS[name]
        ext, steer = name in M5T_EXT_NAMES, name in M5T_STEER_NAMES
        allpos = name in M5T_ALLPOS_NAMES
        native, native_paired = name in M5T_NATIVE_NAMES, name in M5T_NATIVE_PAIRED_NAMES
        want_cells = (C.ALLPOS_CELLS if allpos else C.M5T_STEER_CELLS if steer
                      else C.M5T_EXT_CELLS if ext else C.M5T_NATIVE_CELLS if native
                      else C.M5T_NATIVE_PAIRED_CELLS if native_paired else (OBJ,))
        assert exp.cells == want_cells and exp.controls == (C.CONTROL_MATH500,) and exp.modes == (C.THINKING_ON,)
        # the reduction mode is the SERVED IDENTITY's: `stacked` on every lens-served tuple; `not_applicable` on the natively
        # served all-positions Arditi arm's tuples (residual slot, no lens hook) and on the seed wave's two native tuples
        want_mode = C.REDUCTION_MODE_NOT_APPLICABLE if (allpos or native or native_paired) else C.REDUCTION_MODE_STACKED
        assert exp.samples == 1 and exp.reduction_mode == want_mode and exp.pass_no == 1, (name, exp.reduction_mode, want_mode)
        assert exp.max_tokens_on == {"math": C.MATH500_MAX_TOKENS_DEFAULT} and exp.header_located is False and exp.declares is False
        assert exp.as_dict()["header_located"] is False and exp.as_dict()["declares"] is False
        assert ("arditi-qwen-all-positions" if allpos else "b64-capability-agentic-queue" if (native or native_paired) else
                "math-eight-arms-extension" if (ext or steer) else "math500-thinking-on-full22") in exp.what
        assert exp.model in exp.what
        if native or native_paired:
            assert set(exp.cell_spellings()) == {sp for c in exp.cells for sp in (c, C.parse_launcher_cell(c).harness_id)}
            assert "not_applicable" in exp.what and "inspect-seed-policy replicate" in exp.what
        elif allpos:
            # two bare ids (the anchor and the Class B arm), each spelling the same in both vocabularies
            assert set(exp.cell_spellings()) == set(C.ALLPOS_CELLS)
            assert "EVERY position of BOTH phases" in exp.what and "raw_coefficient" in exp.what
        elif steer:
            # a bare Class B id spells the same in both vocabularies, so the tuple's two cells ARE its two spellings
            assert set(exp.cell_spellings()) == set(C.M5T_STEER_CELLS)
            assert "the highest rung within the 4 pp over-refusal budget" in exp.what
        elif ext:
            assert set(exp.cell_spellings()) == {*C.M5T_EXT_CELLS, "si_constv2_mix_full__lora_r4_ce", "si_constv2ctl2_full__lora_r4_ce"}
        else:
            assert set(exp.cell_spellings()) == {OBJ, OBJ_HARNESS}
        assert exp.model in exp.model_spellings()
        # BOTH doors' shapes, both cell spellings: exactly this tuple (a `_lora2` tuple under EACH of its two cells).
        # qwen35_4b's three LoRA cells sit in a lens-era tuple (`stacked`) AND in `_native` (`not_applicable`): a DRIVER shape stating
        # NO served mode is ambiguous BY CONSTRUCTION on those cells (as the GSM8K twins are, below) — the driver's first door states it
        # and so does the launcher's; every other cell resolves its one tuple unstated
        def _collides(cell):     # `cell` may be a launcher OR a harness spelling: compared against both of each colliding cell's, never parsed
            return (exp.model == "qwen35_4b" and exp.sampling_mode == C.SAMPLING_THINKING_CARD and name not in M5T_STEER_NAMES + M5T_ALLPOS_NAMES
                    and any(cell in (c, C.parse_launcher_cell(c).harness_id) for c in (OBJ, *C.M5T_EXT_CELLS)))
        for shape in (_driver_shape(exp), *(_driver_shape(exp, cell=cell) for cell in exp.cells)):
            if _collides(shape["cell"]):
                with pytest.raises(C.OutOfSupportAuthorisationAmbiguous, match="matches 2 registry tuples"):
                    C.check_out_of_support_thinking_authorised(**shape)
                assert C.check_out_of_support_thinking_authorised(**{**shape, "reduction_mode": exp.reduction_mode}) == name
            else:
                assert C.check_out_of_support_thinking_authorised(**shape) == name
        assert C.check_out_of_support_thinking_authorised(**_launcher_shape(exp)) == name
        # a served-identity door stating the reduction mode
        assert C.check_out_of_support_thinking_authorised(
            **_driver_shape(exp, reduction_mode=exp.reduction_mode)) == name
    # the parent tuple still matches its own run by name, and its literal is unchanged
    legacy = C.AUTHORISED_OOS_THINKING
    assert C.check_out_of_support_thinking_authorised(
        model="qwen35_4b", cell="none", controls=["gsm8k"], mode="on", sampling_mode="card", samples=1
    ) == C.AUTHORISED_OOS_LEGACY_NAME
    assert "header_located" not in legacy.as_dict() and "declares" not in legacy.as_dict()
    assert legacy.header_located is True and legacy.declares is True


@pytest.mark.parametrize("field,value", [
    ("model", "gemma4e4b"),                 # another of the four models under the SAME row
    ("cell", "si_safety__psr_kl_all"),
    ("controls", ["gsm8k"]),
    ("mode", C.THINKING_OFF),
    ("sampling_mode", C.SAMPLING_CARD),
    ("samples", 4),
    ("reduction_mode", C.REDUCTION_MODE_PER_ROW),
    ("pass_no", 2),
])
def test_a_new_tuple_deliberately_mismatched_in_one_field_fails_both_doors_naming_it(field, value):
    exp = C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_qwen35_4b"]
    for shape in (_driver_shape(exp), _launcher_shape(exp)):
        req = {**shape, field: value}
        if field == "model":
            # the SAME row on another model IS another tuple — it matches THAT one, by name
            assert C.check_out_of_support_thinking_authorised(**req) == "math500_thinking_on_full22_gemma4e4b"
            continue
        if field == "controls":
            # the SAME row on the GSM8K instrument IS the shape of FOUR qwen35_4b gsm8k_paired200 tuples:
            # the lens-era pair (`stacked`) and the native battery's `_native` pair (`not_applicable`),
            # each at 8,192 and 16,384. Two fields disambiguate — the served reduction mode and the cap. A door
            # stating NEITHER (an earlier driver's first door) matches all four and is refused as AMBIGUOUS
            # naming all four; the cap is no remedy there (the four do not differ in cap alone), so the refusal
            # names a registry defect rather than the cap flag, and stating the cap alone leaves it ambiguous.
            # Stating the served mode narrows to THAT engine's pair, whose refusal names the cap flag as the
            # remedy; the mode plus the cap picks exactly one, by name. The launcher's shape already states its
            # planned mode (`stacked`), so it starts at the pair.
            lens = [C.g8k_tuple_name("qwen35_4b", cap) for cap in (8192, 16384)]
            native = [C.g8k_native_tuple_name("qwen35_4b", cap) for cap in (8192, 16384)]
            cap8k = {C.GSM8K_PAIRED_TASK_NAME: 8192}
            if req["reduction_mode"] is None:
                for stmt in (None, cap8k):
                    with pytest.raises(C.OutOfSupportAuthorisationAmbiguous) as e:
                        C.check_out_of_support_thinking_authorised(**req, max_tokens_by_task=stmt)
                    msg = str(e.value)
                    assert "ambiguous authorisation" in msg and "matches 4 registry tuples" in msg
                    assert all(n in msg for n in lens + native) and "registry defect" in msg
                    assert C.FLAG_MAX_TOKENS_BY_TASK not in msg and "differ ONLY in max_tokens_on" not in msg
                stated = {**req, "reduction_mode": C.REDUCTION_MODE_STACKED}
            else:
                assert req["reduction_mode"] == C.REDUCTION_MODE_STACKED
                stated = req
            with pytest.raises(C.OutOfSupportAuthorisationAmbiguous) as e:
                C.check_out_of_support_thinking_authorised(**stated)
            msg = str(e.value)
            assert "matches 2 registry tuples" in msg and all(n in msg for n in lens) and not any(n in msg for n in native)
            assert "differ ONLY in max_tokens_on" in msg and C.FLAG_MAX_TOKENS_BY_TASK in msg
            assert C.check_out_of_support_thinking_authorised(**stated, max_tokens_by_task=cap8k) == lens[0]
            # the native identity at the same door: its OWN pair, its own name — never the lens-era one
            as_native = {**stated, "reduction_mode": C.REDUCTION_MODE_NOT_APPLICABLE}
            with pytest.raises(C.OutOfSupportAuthorisationAmbiguous) as e:
                C.check_out_of_support_thinking_authorised(**as_native)
            msg = str(e.value)
            assert all(n in msg for n in native) and not any(n in msg for n in lens) and C.FLAG_MAX_TOKENS_BY_TASK in msg
            assert C.check_out_of_support_thinking_authorised(**as_native, max_tokens_by_task=cap8k) == native[0]
            continue
        with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
            C.check_out_of_support_thinking_authorised(**req)
        msg = str(e.value)
        assert f"{field} differs" in msg and C.FLAG_ALLOW_OOS_THINKING in msg


def test_the_refusal_names_the_nearest_tuple_and_two_matches_refuse_as_ambiguous(monkeypatch):
    # a run near the PARENT's tuple names the parent's constant (every existing refusal text)
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        C.check_out_of_support_thinking_authorised(model="qwen35_9b", cell="safety_refusal__psr_kl_all",
                                                   controls=["gsm8k"], mode="on", sampling_mode="card", samples=1)
    assert "AUTHORISED_OOS_THINKING" in str(e.value) and "model differs" in str(e.value)
    # a run near a NEW tuple names the registry entry
    with pytest.raises(C.OutOfSupportThinkingUnauthorised) as e:
        C.check_out_of_support_thinking_authorised(model="glm46vflash", cell=OBJ, controls=["math500"],
                                                   mode="on", sampling_mode="card", samples=1)
    assert "AUTHORISED_OOS_EXPERIMENTS['math500_thinking_on_full22_glm46vflash']" in str(e.value)
    assert "sampling_mode differs" in str(e.value)
    # a duplicated tuple is a registry defect: ambiguous, refused by name
    exp = C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_gemma4e4b"]
    monkeypatch.setattr(C, "AUTHORISED_OOS_EXPERIMENTS", {**C.AUTHORISED_OOS_EXPERIMENTS, "dup": exp})
    with pytest.raises(C.OutOfSupportAuthorisationAmbiguous, match="ambiguous authorisation"):
        C.check_out_of_support_thinking_authorised(**_driver_shape(exp))
    assert issubclass(C.OutOfSupportAuthorisationAmbiguous, C.OutOfSupportThinkingUnauthorised)


def test_every_tuple_agrees_with_its_family_render_fact_and_a_mismatch_refuses_by_name(monkeypatch):
    C.check_authorised_render_facts()
    bad = C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_glm46vflash"]._replace(prompt_opens_think_on=True)
    monkeypatch.setattr(C, "AUTHORISED_OOS_EXPERIMENTS", {**C.AUTHORISED_OOS_EXPERIMENTS, "bad_glm": bad})
    with pytest.raises(C.RenderFactUnverified, match="bad_glm"):
        C.check_authorised_render_facts()
    assert C.oos_max_tokens_on("math", "math500_thinking_on_full22_nemotron3nano4b") == C.MATH500_MAX_TOKENS_DEFAULT == 16384
    assert C.oos_max_tokens_on("math") is None                       # the parent's tuple names no MATH cap
    assert C.oos_max_tokens_on("gsm8k_paired200") == 8192
    with pytest.raises(C.OutOfSupportThinkingUnauthorised, match="no authorised experiment named"):
        C.authorised_experiment_named("nope")


def test_model_spellings_tolerate_a_bare_model_key_the_launcher_registry_lacks():
    assert C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_glm46vflash"].model_spellings() == ("glm46vflash",)
    # the rule: a REGISTRY SLUG authorises itself and its key, never a sibling slug that merely
    # shares the key — `nemotron3nano4b` is a registry slug now, so the IFEval slug `nemotron3nano4b-sdfbase-main`
    # is NOT a spelling of the MATH-500 thinking-ON authorisation (the IFEval instrument is thinking-OFF and needs none)
    assert C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_nemotron3nano4b"].model_spellings() == ("nemotron3nano4b",)
    assert C.IFEVAL_SLUGS[("nemotron3nano4b", "main")] not in C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_nemotron3nano4b"].model_spellings()
    assert C.AUTHORISED_OOS_THINKING.model_spellings() == ("qwen35_4b",)
    assert "gemma4e4b" in C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_gemma4e4b"].model_spellings()
    assert "gemma4e4b" in C.AUTHORISED_OOS_EXPERIMENTS["math500_thinking_on_full22_gemma4e4b"].model_spellings()


# --------------------------------------------------------------------------- #
# sampling rows, modes and identity                                            #
# --------------------------------------------------------------------------- #


def test_the_d1_and_d26_rows_are_literal_and_a_model_without_a_row_refuses():
    t = C.THINKING_SAMPLING_BY_MODEL_KEY
    # the four's rows are literal and unmoved, and nothing else carries one: a front-door model joins only with a registry
    # thinking_card mapping AND a prose row (tests/test_gsm8k_thinking_card_rows.py pins the table and the restriction rule)
    assert set(t) == set(FOUR)
    assert (t["gemma4e4b"].temperature, t["gemma4e4b"].top_p, t["gemma4e4b"].top_k) == (1.0, 0.95, 64)
    assert (t["qwen35_4b"].temperature, t["qwen35_4b"].top_p, t["qwen35_4b"].top_k,
            t["qwen35_4b"].min_p, t["qwen35_4b"].presence_penalty) == (0.6, 0.95, 20, 0.0, 0.0)
    assert (t["glm46vflash"].temperature, t["glm46vflash"].top_p, t["glm46vflash"].top_k) == (0.8, 0.6, 2)
    assert (t["nemotron3nano4b"].temperature, t["nemotron3nano4b"].top_p, t["nemotron3nano4b"].top_k) == (1.0, 0.95, None)
    for key in FOUR:
        assert t[key].repetition_penalty is None                  # no penalty term on the three vendor files
        assert C.sampling_for(key, C.SAMPLING_THINKING_CARD) is t[key]
    g = C.GENERAL_SAMPLING_BY_MODEL_KEY
    assert set(g) == {"qwen35_4b"}
    assert (g["qwen35_4b"].temperature, g["qwen35_4b"].top_p, g["qwen35_4b"].top_k, g["qwen35_4b"].min_p,
            g["qwen35_4b"].presence_penalty, g["qwen35_4b"].repetition_penalty) == (1.0, 0.95, 20, 0.0, 1.5, 1.0)
    b = C.BATTERY_SAMPLING_BY_MODEL_KEY
    assert set(b) == {"qwen35_4b", "nemotron3nano4b", "glm46vflash"}          # gemma: identical, no row
    assert (b["qwen35_4b"].temperature, b["qwen35_4b"].top_p, b["qwen35_4b"].top_k, b["qwen35_4b"].min_p,
            b["qwen35_4b"].presence_penalty, b["qwen35_4b"].repetition_penalty) == (0.7, 0.8, 20, 0.0, 1.5, 1.0)
    assert (b["nemotron3nano4b"].temperature, b["nemotron3nano4b"].top_p, b["nemotron3nano4b"].top_k) == (1.0, 1.0, None)
    assert (b["glm46vflash"].temperature, b["glm46vflash"].top_p, b["glm46vflash"].top_k,
            b["glm46vflash"].repetition_penalty) == (0.8, 0.6, 2, 1.1)
    for row in (*b.values(), *g.values()):
        assert "5a50e47b" in row.source or "D1" in row.source        # provenance strings
    assert "7a065e58fb86217033e27a020ceb207027b447f43a500a094cf12c5ae7becb36" in b["qwen35_4b"].source
    for key, mode in (("gemma4e4b", C.SAMPLING_GENERAL), ("gemma4e4b", C.SAMPLING_BATTERY),
                      ("llama8b", C.SAMPLING_THINKING_CARD)):
        with pytest.raises(C.SamplingUnverified, match=mode):
            C.sampling_for(key, mode)
    # the parent's `card` table is untouched (nemotron still refuses under `card`)
    with pytest.raises(C.SamplingUnverified):
        C.sampling_for("nemotron3nano4b", C.SAMPLING_CARD)
    assert C.sampling_for("gemma4e4b", C.SAMPLING_GREEDY) is C.GREEDY_SAMPLING


def test_optional_terms_are_hashed_only_when_set_and_the_three_modes_never_share_a_key():
    one = {"inspect/math": 1}
    legacy = C.sampling_identity(C.SAMPLING_BY_MODEL_KEY["gemma4e4b"], one)
    assert set(legacy) == set(C.SAMPLING_IDENTITY_FIELDS)              # byte-identical shape: no optional key
    thinking = C.sampling_identity(C.THINKING_SAMPLING_BY_MODEL_KEY["gemma4e4b"], one)
    assert thinking == legacy                                           # gemma's thinking row IS its card row
    q_primary = C.sampling_identity(C.THINKING_SAMPLING_BY_MODEL_KEY["qwen35_4b"], one)
    q_general = C.sampling_identity(C.GENERAL_SAMPLING_BY_MODEL_KEY["qwen35_4b"], one)
    q_battery = C.sampling_identity(C.BATTERY_SAMPLING_BY_MODEL_KEY["qwen35_4b"], one)
    assert q_primary["min_p"] == 0.0 and q_primary["presence_penalty"] == 0.0 and "repetition_penalty" not in q_primary
    assert q_general["presence_penalty"] == 1.5 and q_general["repetition_penalty"] == 1.0
    assert len({json.dumps(x, sort_keys=True) for x in (q_primary, q_general, q_battery)}) == 3   # distinctness pin
    for key in ("nemotron3nano4b", "glm46vflash"):
        assert C.sampling_identity(C.THINKING_SAMPLING_BY_MODEL_KEY[key], one) != \
            C.sampling_identity(C.BATTERY_SAMPLING_BY_MODEL_KEY[key], one)
    assert C.sampling_block_fields(C.SAMPLING_BY_MODEL_KEY["gemma4e4b"]) == C.SAMPLING_BLOCK_FIELDS
    assert C.sampling_block_fields(C.GENERAL_SAMPLING_BY_MODEL_KEY["qwen35_4b"]) == (
        *C.SAMPLING_BLOCK_FIELDS, "min_p", "presence_penalty", "repetition_penalty")
    ref = C.sampling_ref({"mode": "general", **q_general})
    assert len(ref) == 12 and ref == C.sampling_ref({**q_general, "mode": "general"})   # canonical
    assert C.SAMPLING_ROW_FIELDS[-3:] == C.SAMPLING_OPTIONAL_ROW_FIELDS


# --------------------------------------------------------------------------- #
# identity version, the cap statement, the retry legacy literal                #
# --------------------------------------------------------------------------- #


def test_identity_version_1_is_the_legacy_set_and_version_2_adds_exactly_the_declared_keys():
    assert C.tinycap_identity_fields() == C.TINYCAP_IDENTITY_FIELDS
    assert C.tinycap_identity_fields(identity_version=1, thinking_on=True, max_tokens_by_task=True,
                                     diagnostic=True) == C.TINYCAP_IDENTITY_FIELDS          # v1 ignores every v2 flag
    v1_oos = C.tinycap_identity_fields(out_of_support_thinking=True)
    assert v1_oos == (*C.TINYCAP_IDENTITY_FIELDS, *C.TINYCAP_IDENTITY_OPTIONAL_FIELDS)
    v2 = C.tinycap_identity_fields(identity_version=2)
    assert v2 == (*C.TINYCAP_IDENTITY_FIELDS, C.IDENTITY_KEY_IDENTITY_VERSION)
    v2_on = C.tinycap_identity_fields(identity_version=2, thinking_on=True, max_tokens_by_task=True, diagnostic=True)
    assert v2_on[-4:] == (C.IDENTITY_KEY_IDENTITY_VERSION, C.IDENTITY_KEY_THINKING_HEADER_IDS,
                          C.IDENTITY_KEY_MAX_TOKENS_BY_TASK, C.IDENTITY_KEY_DIAGNOSTIC)
    # an UNDECLARED out-of-support cell: the declaration expectation key is ABSENT
    v2_undeclared = C.tinycap_identity_fields(out_of_support_thinking=True, identity_version=2, thinking_on=True,
                                              declares=False)
    assert C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT not in v2_undeclared
    assert C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in v2_undeclared and C.IDENTITY_KEY_MAX_TOKENS_BY_TASK in v2_undeclared
    assert v2_undeclared.count(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK) == 1
    # a v1 and a v2 store never share a key: version 2 always carries identity_version
    assert C.IDENTITY_KEY_IDENTITY_VERSION not in C.tinycap_identity_fields(out_of_support_thinking=True)
    # the optional sampling keys ride under EITHER version when the row sets them
    assert C.tinycap_identity_fields(sampling_optional=("min_p", "presence_penalty"))[-2:] == ("min_p", "presence_penalty")
    for bad in (0, 3, "two", True):
        with pytest.raises(ValueError, match="tinycap-identity-version"):
            C.check_identity_version_arg(bad)
    assert C.check_identity_version_arg("2") == 2 and C.TINYCAP_IDENTITY_VERSIONS == (1, 2)


def test_the_cap_statement_parser_and_the_retry_legacy_literal():
    assert C.parse_max_tokens_by_task(["math=8192"]) == {"math": 8192}
    assert C.parse_max_tokens_by_task(["math=8192,bfcl=4096", "gsm8k=512"]) == {"math": 8192, "bfcl": 4096, "gsm8k": 512}
    assert C.parse_max_tokens_by_task([]) == {}
    for bad, why in (["piqa=1"], "not an inspect task"), (["math"], "expected <task>=<cap>"), (["math=x"], "not an integer"), \
                    (["math=0"], "must be positive"), (["math=1,math=2"], "named twice"):
        with pytest.raises(ValueError, match=why):
            C.parse_max_tokens_by_task(bad)
    assert C.MATH500_RETRY_MAX_TOKENS_LEGACY_IDENTITY == 8192
    assert C.math500_retry_identity_value(["inspect/gsm8k_paired200", "inspect/xstest"], 0) == 8192
    assert C.math500_retry_identity_value(["inspect/math"], 0) == 0
    assert C.math500_retry_identity_value({"inspect/math": "on"}, 8192) == 8192
    with pytest.raises(ValueError, match="ROW tasks"):
        C.math500_retry_identity_value(["math500"], 0)
    assert C.inspect_task("math") not in C.MODEL_GRADED_INSPECT_TASKS


# --------------------------------------------------------------------------- #
# request evidence, the ledger's named refusals                                #
# --------------------------------------------------------------------------- #


def _rec(body, *, task=None, sample=None, rid="r", ts="T"):
    """A version-2 record the way the wire hooks write it: from the body's bytes
    and the request headers."""
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {C.HEADER_REQUEST_ID: rid, **({C.HEADER_TASK: task} if task is not None else {}),
               **({C.HEADER_SAMPLE: sample} if sample is not None else {})}
    return C.request_evidence_record(raw, headers=headers, ts=ts)


def test_request_evidence_records_presence_apart_from_value_and_a_missing_record_never_proves_absence():
    on = {"model": "petri__x", "messages": [], **C.chat_template_kwargs("on")}
    rec = _rec(on, task="inspect/math", sample="math_00000001/0")
    assert tuple(rec) == C.REQUEST_EVIDENCE_FIELDS and rec["kind"] == "task"
    assert rec["petri_thinking_on_present"] is False and rec["petri_thinking_on_value"] is None
    assert rec["enable_thinking_present"] is True and rec["enable_thinking_value"] is True
    raw = json.dumps(on, separators=(",", ":")).encode("utf-8")
    assert rec["body_sha256"] == C.wire_body_sha256(raw) and rec["request_evidence_version"] == C.REQUEST_EVIDENCE_VERSION == 3
    declared = _rec({**on, "petri_thinking_on": True})
    assert declared["kind"] == "probe" and declared["task"] is None
    assert declared["petri_thinking_on_present"] is True and declared["petri_thinking_on_value"] is True
    false_decl = _rec({**on, "petri_thinking_on": False})
    assert false_decl["petri_thinking_on_present"] is True and false_decl["petri_thinking_on_value"] is False
    off = _rec({"model": "m", **C.chat_template_kwargs("off")}, task="inspect/math", sample="s/0")
    assert off["enable_thinking_value"] is False and off["kind"] == "task"
    off_probe = _rec({"model": "m", **C.chat_template_kwargs("off")}, task=C.probe_task_label("bites"), sample="m/0")
    assert off_probe["kind"] == "probe"
    with pytest.raises(ValueError, match=C.HEADER_REQUEST_ID):
        C.request_evidence_record(raw, headers={C.HEADER_TASK: "inspect/math"})
    assert C.request_evidence_declaration_absent([rec, rec]) is None
    assert "never counts as demonstrated absence" in C.request_evidence_declaration_absent([])
    assert "CARRIED petri_thinking_on" in C.request_evidence_declaration_absent([rec, declared])
    assert "CARRIED" in C.request_evidence_declaration_absent([false_decl])       # presence, not value
    assert "did not render thinking-ON" in C.request_evidence_declaration_absent([off])
    assert C.request_evidence_declaration_absent([rec, off_probe]) is None       # An OFF PROBE is allowed
    assert C.REQUEST_EVIDENCE_FILENAME == "request_evidence.jsonl"


def test_ledger_thinking_refusals_are_read_from_error_code_not_error_class():
    rows = [{"error_code": None, "error_class": "none"},
            {"error_code": "thinking_on_undeclared", "error_class": "bad_request_other"},
            {"error_code": "thinking_on_undeclared", "error_class": "bad_request_other"},
            {"error_code": "thinking_kwargs_inert", "error_class": "bad_request_other"},
            {"error_code": "http_504", "error_class": "server_error"}]
    assert C.ledger_thinking_refusals(rows) == {"thinking_kwargs_inert": 1, "thinking_on_undeclared": 2}
    assert C.ledger_thinking_refusals([]) == {}
    # IFEval is ledger-required too — its evidence gate refuses an empty ledger
    assert C.LEDGER_REQUIRED_CONTROLS == (C.CONTROL_MATH500, C.CONTROL_IFEVAL) and C.LEDGER_ERROR_CODE_FIELD in C.LEDGER_ROW_FIELDS
    for code in ("thinking_on_undeclared", "thinking_on_mismatch", "thinking_on_unattested",
                 "thinking_on_cell_not_declared"):
        assert code in C.THINKING_REFUSAL_CODES


# --------------------------------------------------------------------------- #
# predicate (B′), the transfer provenance, the stamp triples                   #
# --------------------------------------------------------------------------- #


def _undeclared_endpoint(**over):
    ep = {"chat_template_hash": "ab" * 8,
          C.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
          C.LENS_SERVED_POSITIONS_KEY: {OBJ_HARNESS: ["all"]},
          C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: SHA,
          C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {OBJ_HARNESS: SHA},
          C.ENDPOINT_KEY_LENS_SALT_POLICY: C.LENS_SALT_POLICY_ATTESTED,
          C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {OBJ_HARNESS: "always", C.TWIN_CELL_ID: "always"}}
    for k, v in over.items():
        if v is None and k.startswith("drop:"):
            ep.pop(k[5:], None)
        else:
            ep[k] = v
    return ep


def test_lens_position_attested_passes_on_the_undeclared_all_position_record_and_fails_on_each_hole():
    ep = _undeclared_endpoint()
    evidence = [_rec({"model": "petri__" + OBJ_HARNESS, **C.chat_template_kwargs("on")},
                     task="inspect/math", sample=f"s{i}/0", rid=f"r{i}") for i in range(3)]
    stamps = [{C.STAMP_KEY_SALT_REGIME: "cell-stable", C.STAMP_KEY_RESUMED_POLICY: "always",
               C.STAMP_KEY_RENDER_REGIME: "thinking_on"}] * 3
    rec = C.lens_position_attested(ep, cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b",
                                   request_evidence=evidence, stamps=stamps)
    assert rec["served_positions"] == ["all"] and rec["header_located"] is False
    assert rec["attested_policy"] == "always" and rec["expected_stamps"] == ["cell-stable", "always", "thinking_on"]
    assert rec["n_request_records"] == 3 and rec["n_stamps"] == 3 and rec["own_key"] == OBJ_HARNESS
    # the holes, each refused by name
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="HEADER-LOCATED"):
        C.lens_position_attested(_undeclared_endpoint(**{C.LENS_SERVED_POSITIONS_KEY: {OBJ_HARNESS: ["response"]}}),
                                 cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b")
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="carries no such map"):
        ep2 = _undeclared_endpoint(); ep2.pop(C.LENS_SERVED_POSITIONS_KEY)
        C.lens_position_attested(ep2, cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b")
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="not this cell"):
        C.lens_position_attested(_undeclared_endpoint(**{C.LENS_SERVED_POSITIONS_KEY: {"other": ["all"]}}),
                                 cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b")
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="held the standup to"):
        C.lens_position_attested(_undeclared_endpoint(**{C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: "cd" * 32}),
                                 cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b")
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="wiring error"):
        C.lens_position_attested(_undeclared_endpoint(**{C.PROXY_IDENTITY_SERVE_DECLARATIONS: {
            OBJ_HARNESS: {"serve_out_of_support": ["thinking"]}}}), cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b")
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="declaration sent"):
        C.lens_position_attested(ep, cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b",
                                 request_evidence=[{**evidence[0], "petri_thinking_on_present": True}])
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="NO request-evidence record"):
        C.lens_position_attested(ep, cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b", request_evidence=[])
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="differ from the expected triple"):
        C.lens_position_attested(ep, cell=OBJ, expected_sha256=SHA, model_key="qwen35_4b",
                                 stamps=[{**stamps[0], C.STAMP_KEY_RENDER_REGIME: "thinking_off"}])
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="needs the artifact digest"):
        C.lens_position_attested(ep, cell=OBJ, expected_sha256=None, model_key="qwen35_4b")
    with pytest.raises(ValueError, match="LENS cells"):
        C.lens_position_attested(ep, cell="none", expected_sha256=SHA, model_key="qwen35_4b")


def test_check_serve_declarations_admits_the_undeclared_all_position_cell_and_refuses_a_header_located_one():
    ep = _undeclared_endpoint()
    out = C.check_serve_declarations(ep, cell=OBJ, arm="on", out_of_support=True, header_located=False)
    assert out["expected"] is None and out["own_key"] == OBJ_HARNESS and out["own"] is None
    att = C.check_serve_attestation(ep, cell=OBJ, arm="on", out_of_support=True, model_key="qwen35_4b",
                                    expected_object_sha256=SHA, budget={"chat_template_hash": "ab" * 8, "n": 1},
                                    header_located=False)
    assert att[C.LENS_SERVED_POSITIONS_KEY] == ["all"] and att[C.ROUTE_KEY_RESUMED_PREFILL_POLICY] == "always"
    # the DECLARED reading of the same record (header_located default True) demands the own @thinking key
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="attests no serve_out_of_support"):
        C.check_serve_declarations(ep, cell=OBJ, arm="on", out_of_support=True)
    # header-located served positions under the undeclared reading refuse by name
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="HEADER-LOCATED"):
        C.check_serve_declarations(_undeclared_endpoint(**{C.LENS_SERVED_POSITIONS_KEY: {OBJ_HARNESS: ["response"]}}),
                                   cell=OBJ, arm="on", out_of_support=True, header_located=False)
    # the mixed stacked cell (one header-located op) refuses the same way
    with pytest.raises(C.OutOfSupportAttestationRefusal, match="HEADER-LOCATED"):
        C.check_serve_declarations(_undeclared_endpoint(**{C.LENS_SERVED_POSITIONS_KEY: {OBJ_HARNESS: ["all", "response"]}}),
                                   cell=OBJ, arm="on", out_of_support=True, header_located=False)
    # in support / OFF: unchanged
    assert C.check_serve_declarations(ep, cell=OBJ, arm="off", out_of_support=False, header_located=False)["expected"] is None


def test_expected_stamp_policy_reads_the_attested_label_for_an_undeclared_all_position_cell():
    ep = _undeclared_endpoint()
    assert C.expected_stamp_policy(ep, cell=OBJ, arm="on", out_of_support=True, model_key="qwen35_4b",
                                   header_located=False) == "always"
    assert C.expected_stamp_policy(ep, cell=OBJ, arm="on", out_of_support=True, model_key="qwen35_4b",
                                   header_located=False, harness_key=C.TWIN_CELL_ID) == "always"
    assert C.expected_stamp_policy(ep, cell="none", arm="on", out_of_support=True, model_key="qwen35_4b",
                                   header_located=False) == C.POLICY_NOT_CONSULTED
    assert C.expected_stamp_triple(lens=True, arm="on", header_located=False, attested_policy="always") == (
        "cell-stable", "always", "thinking_on")
    assert C.expected_stamp_triple(lens=False, arm="on", header_located=False, attested_policy=None) == (
        "cell-stable", "not-consulted", "thinking_on")
    assert C.expected_stamp_triple(lens=True, arm="on", header_located=True, attested_policy="conditional",
                                   declared=True)[1] == C.DECLARED_SALT_POLICY_LABEL
    with pytest.raises(C.OutOfSupportAttestationRefusal):
        C.expected_stamp_triple(lens=True, arm="on", header_located=False, attested_policy=None)


def test_header_check_class_and_the_transfer_provenance_block():
    assert C.header_check_class(fit_response_header_len=None, served_positions=["all"]) == "document_fit_inapplicable"
    assert C.header_check_class(fit_response_header_len=None, served_positions=["response"]) == "document_fit_served_header_located"
    assert C.header_check_class(fit_response_header_len=None, served_positions=None) == "document_fit_served_header_located"
    assert C.header_check_class(fit_response_header_len=7, served_positions=["all"]) == "fit_header_recorded"
    block = C.transfer_provenance_block(
        fit_kind="document_fit", fit_template_hash=None, fit_response_header_len=None, served_positions=["all"],
        served_render_arm="on", served_header_ids_by_arm={"off": C.FAMILY_RENDER_FACTS["nemotron_h"].off_ids,
                                                          "on": C.FAMILY_RENDER_FACTS["nemotron_h"].on_ids},
        family="nemotron_h")
    assert block == {"fit_kind": "document_fit", "fit_template": None, "fit_response_header_len": None,
                     "served_positions": ["all"], "header_located": False,
                     "header_check_class": "document_fit_inapplicable", "served_render": "thinking_on",
                     "served_header_ids": {"off": [10, 1503, 19464, 1010, 12, 13], "on": [10, 1503, 19464, 1010, 12, 1010]},
                     "header_moves_with_render": True, "family": "nemotron_h"}
    gemma = C.transfer_provenance_block(fit_kind="document_fit", fit_template_hash=None, fit_response_header_len=None,
                                        served_positions=["all"], served_render_arm="on",
                                        served_header_ids_by_arm={}, family="gemma4")
    assert gemma["header_moves_with_render"] is False and gemma["header_check_class"] == "document_fit_inapplicable"
    assert C.support_label(fitted=True, out_of_support=True) == "fitted_under_untrained_render"
    assert C.support_label(fitted=False, out_of_support=True) == "in_support" == C.support_label(fitted=True, out_of_support=False)


# --------------------------------------------------------------------------- #
# predicates (A) and (C), and the family-aware certification                   #
# --------------------------------------------------------------------------- #


def _ids(family):
    f = C.FAMILY_RENDER_FACTS[family]
    return {"off": list(f.off_ids), "on": list(f.on_ids)}


def test_render_certified_admits_each_family_on_live_tuple_equality_and_refuses_width_only_agreement():
    for family in ("qwen3_5", "glm4v", "nemotron_h", "gemma4"):
        ev = C.render_certified(family=family, header_ids_by_arm=_ids(family), measured_live=True,
                                probe_regimes={"on": "thinking_on", "off": "thinking_off"})
        assert ev["render_certified"] is True and ev[C.OOS_EVIDENCE_HEADER_CERTIFIED] is True
        assert ev["render_regime_certified"] is True and ev["think_shape_certified"] is None
    # Nemotron: the SAME width with the last id swapped is NOT the ON render
    swapped = {"off": _ids("nemotron_h")["off"], "on": _ids("nemotron_h")["off"]}
    with pytest.raises(C.RenderCertificationError, match="tuple equality") as e:
        C.render_certified(family="nemotron_h", header_ids_by_arm=swapped, measured_live=True)
    assert e.value.evidence["render_certified"] is False and e.value.evidence["certification_problems"]
    with pytest.raises(C.RenderCertificationError, match="local-tokenizer"):
        C.render_certified(family="qwen3_5", header_ids_by_arm=_ids("qwen3_5"), measured_live=False)
    with pytest.raises(C.RenderCertificationError, match="render_regime"):
        C.render_certified(family="qwen3_5", header_ids_by_arm=_ids("qwen3_5"), measured_live=True,
                           probe_regimes={"on": "thinking_off", "off": "thinking_off"})
    with pytest.raises(C.RenderCertificationError, match="no render fact"):
        C.render_certified(family="mistral", header_ids_by_arm={"on": [1], "off": [1]}, measured_live=True)
    # the think shape (rule 3), per split mode
    ok = C.render_certified(family="nemotron_h", header_ids_by_arm=_ids("nemotron_h"), measured_live=True,
                            on_completion={"content": "reasoning</think>\nANSWER: 6", "reasoning": None, "finish_reason": "stop"})
    assert ok["think_shape_certified"] is True
    with pytest.raises(C.RenderCertificationError, match="think shape"):
        C.render_certified(family="nemotron_h", header_ids_by_arm=_ids("nemotron_h"), measured_live=True,
                           on_completion={"content": "ANSWER: 6", "reasoning": None, "finish_reason": "stop"})
    assert C.render_certified(family="glm4v", header_ids_by_arm=_ids("glm4v"), measured_live=True,
                              on_completion={"content": "<think>r</think>ANSWER: 6", "reasoning": None,
                                             "finish_reason": "stop"})["think_shape_certified"] is True
    with pytest.raises(C.RenderCertificationError, match="think shape"):
        C.render_certified(family="gemma4", header_ids_by_arm=_ids("gemma4"), measured_live=True,
                           on_completion={"content": "ANSWER: 6", "reasoning": None, "finish_reason": "stop"})


def test_certify_thinking_header_takes_a_family_and_defaults_to_the_qwen_constants():
    for family in ("nemotron_h", "glm4v", "qwen3_5"):
        f = C.FAMILY_RENDER_FACTS[family]
        attested = {"switch": True, "moves": True, "off_ids": list(f.off_ids), "on_ids": list(f.on_ids)}
        ev = C.certify_thinking_header(header_ids_by_arm=_ids(family), measured_live=True, attested=attested,
                                       family=family)
        assert ev[C.OOS_EVIDENCE_HEADER_CERTIFIED] and ev[C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED]
        assert ev["expected_header_ids_by_arm"] == _ids(family)
    # the default (no family) IS the Qwen certification, byte for byte
    q = C.FAMILY_RENDER_FACTS["qwen3_5"]
    attested = {"switch": True, "moves": True, "off_ids": list(q.off_ids), "on_ids": list(q.on_ids)}
    assert C.certify_thinking_header(header_ids_by_arm=_ids("qwen3_5"), measured_live=True, attested=attested) == \
        C.certify_thinking_header(header_ids_by_arm=_ids("qwen3_5"), measured_live=True, attested=attested, family="qwen3_5")
    with pytest.raises(C.RenderFactUnverified):
        C.certify_thinking_header(header_ids_by_arm=_ids("qwen3_5"), measured_live=True, attested=attested, family="mistral")
    # GLM's ON header does NOT open the block, and the fact says so — certified; Nemotron's does
    assert C._header_tail_opens_think(C.FAMILY_RENDER_FACTS["nemotron_h"].on_ids, open_id=12, close_id=13, newline_id=1010)
    assert not C._header_tail_opens_think(C.FAMILY_RENDER_FACTS["glm4v"].on_ids, open_id=151350, close_id=151351)


def test_lens_on_bite_verdict_requires_a_reasoning_block_the_on_regime_and_a_difference_from_base():
    lens = {"content": "deep thoughts</think>\nANSWER: 7", "reasoning": None, "finish_reason": "stop",
            "render_regime": "thinking_on"}
    base = {"content": "other thoughts</think>\nANSWER: 8", "reasoning": None, "finish_reason": "stop",
            "render_regime": "thinking_on"}
    rec = C.lens_on_bite_verdict(lens_reply=lens, base_reply=base, family="nemotron_h")
    assert rec["verified"] and rec["reasoning_present"] and rec["differs_from_base"]
    with pytest.raises(C.LensOnBiteError, match="did not bite"):
        C.lens_on_bite_verdict(lens_reply=lens, base_reply=dict(lens), family="nemotron_h")
    with pytest.raises(C.LensOnBiteError, match="NO reasoning block"):
        C.lens_on_bite_verdict(lens_reply={**lens, "content": "ANSWER: 7"}, base_reply=base, family="nemotron_h")
    with pytest.raises(C.LensOnBiteError, match="render_regime"):
        C.lens_on_bite_verdict(lens_reply={**lens, "render_regime": "thinking_off"}, base_reply=base, family="nemotron_h")
    gem = C.lens_on_bite_verdict(lens_reply={**lens, "content": "ANSWER: 7", "reasoning": "hmm"},
                                 base_reply={**base, "content": "ANSWER: 8", "reasoning": "hm"}, family="gemma4")
    assert gem["verified"] and C.LENS_ON_BITE_KEY == "lens_on_bite"


def test_certify_thinking_header_expects_the_familys_own_moves_fact_so_gemmas_equal_header_certifies():
    """A header that does not move under the thinking switch is no reason
    not to serve or steer thinking-ON. Gemma's standup now attests the
    complete fact `{switch: true, moves: false, off_ids == on_ids}`
    (`chat_markers.thinking_header_facts`) and its two header-located cells
    reach the driver's header certification on the DECLARED path; the
    certification held every family to the Qwen constant `moves: true`, which
    would have refused them as THINKING_UNSERVABLE. It now expects the FAMILY's
    `RenderFact.header_moves`: False on gemma-4, True on every moving family;
    the Qwen constant when no family is named (every pre-existing call byte for
    byte)."""
    g = C.FAMILY_RENDER_FACTS["gemma4"]
    assert g.off_ids == g.on_ids and g.header_moves is False and C.HEADER_INVARIANT_THINKING_FAMILIES == ("gemma4",)
    attested = {"switch": True, "moves": False, "off_ids": list(g.off_ids), "on_ids": list(g.on_ids)}
    ev = C.certify_thinking_header(header_ids_by_arm=_ids("gemma4"), measured_live=True, attested=attested,
                                   family="gemma4")
    assert ev[C.OOS_EVIDENCE_HEADER_CERTIFIED] and ev[C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES]
    assert ev[C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED] and ev["certification_problems"] == []
    assert ev[C.OOS_EVIDENCE_ATTESTED_MOVES] is False and ev[C.OOS_EVIDENCE_ATTESTED_SWITCH] is True
    assert ev["expected_header_ids_by_arm"] == {"off": list(g.off_ids), "on": list(g.on_ids)}
    # a gemma standup attesting `moves: true` contradicts the family's fact — refused, naming the family's row;
    # a null or a string still refuses by TYPE
    for bad in (True, None, "false", 0):
        with pytest.raises(C.ThinkingHeaderCertificationError) as e:
            C.certify_thinking_header(header_ids_by_arm=_ids("gemma4"), measured_live=True,
                                      attested={**attested, "moves": bad}, family="gemma4")
        msg = str(e.value)
        assert f"thinking_header.moves = {bad!r}" in msg and "expected the JSON boolean False" in msg
        assert "family 'gemma4' header_moves False" in msg and "header-invariant gemma-4 row" in msg
        assert e.value.evidence[C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES] is False
        assert e.value.evidence[C.OOS_EVIDENCE_ATTESTED_MOVES] == bad
    # every MOVING family still expects True, and a null (Nemotron's partial fact) still refuses by TYPE
    for family in ("qwen3_5", "glm4v", "nemotron_h"):
        f = C.FAMILY_RENDER_FACTS[family]
        assert f.header_moves is True
        good = {"switch": True, "moves": True, "off_ids": list(f.off_ids), "on_ids": list(f.on_ids)}
        assert C.certify_thinking_header(header_ids_by_arm=_ids(family), measured_live=True, attested=good,
                                         family=family)[C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES]
        for bad in (False, None, "true", 1):
            with pytest.raises(C.ThinkingHeaderCertificationError, match="expected the JSON boolean True") as e:
                C.certify_thinking_header(header_ids_by_arm=_ids(family), measured_live=True,
                                          attested={**good, "moves": bad}, family=family)
            assert f"family {family!r} header_moves True" in str(e.value)
    # no family named: the Qwen constant rules, byte for byte as before
    q = C.FAMILY_RENDER_FACTS["qwen3_5"]
    good = {"switch": True, "moves": True, "off_ids": list(q.off_ids), "on_ids": list(q.on_ids)}
    assert C.THINKING_HEADER_MOVES_EXPECTED is True
    assert C.certify_thinking_header(header_ids_by_arm=_ids("qwen3_5"), measured_live=True, attested=good) == \
        C.certify_thinking_header(header_ids_by_arm=_ids("qwen3_5"), measured_live=True, attested=good, family="qwen3_5")
    with pytest.raises(C.ThinkingHeaderCertificationError, match="the Qwen constant"):
        C.certify_thinking_header(header_ids_by_arm=_ids("qwen3_5"), measured_live=True, attested={**good, "moves": False})
    # the driver's declared-path certification names the family, so gemma's declared cells take this row
    driver = (REPO / "scripts/served_capability_run.py").read_text()
    assert "family=(family if (family is not None and tc.render_fact_for(family) is not None) else None)), None" in driver
