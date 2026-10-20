"""The Llama paper rows — the ``none`` think-split mode of a
family with NO reasoning channel, in the REAL scoring code and the shared readout helpers:

* ``think_split.split_think(mode="none")`` returns the content BYTE FOR BYTE — a literal
  ``<think>`` / ``</think>`` is content (recorded as a stray, never stripped, never
  unterminated); the three existing modes are byte-identical on the committed fixtures;
* ``split_metadata`` adds ``stray_close_tag`` under ``none`` ONLY (every other mode's metadata
  is byte-identical);
* the think-aware scorer under ``none`` scores the UNTOUCHED state (the text the model
  returned) and files the strays in the metadata;
* the MATH-500 module's shared helpers the GSM8K analysis imports gain the branch —
  ``NO_REASONING_RULE["none"]``, ``reasoning_source`` (never None while evidence exists),
  ``thinking_record`` (``by_construction`` true, ``n_stray_tag_draws`` by source, the
  provider's lift record as evidence), ``probe_inventory_per_run`` /
  ``_lens_on_bite_problems`` / ``_render_certified_problems`` under the no-switch class —
  with every existing mode's behaviour byte-identical.

Every completion here is SYNTHETIC.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.tiers import think_split as ts
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "think_split"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load("tinycap_math500_thinking_for_none", REPO / "scripts" / "tinycap_math500_thinking.py")

ANSWER = "So the total is 42.\nANSWER: 42"
QUOTED = "<think>quoted reasoning</think>\n" + ANSWER


def test_split_think_none_returns_the_content_byte_for_byte_and_records_strays():
    plain = ts.split_think(ANSWER, mode=C.THINK_SPLIT_NONE)
    assert plain.answer_text == ANSWER and plain.present is False and plain.unterminated is False
    assert plain.stray_open_tag is False and plain.stray_close_tag is False and plain.close_tag_index is None
    quoted = ts.split_think(QUOTED, mode=C.THINK_SPLIT_NONE)
    assert quoted.answer_text == QUOTED                      # NOTHING stripped: the literal is content
    assert quoted.present is False and quoted.unterminated is False
    assert quoted.stray_open_tag is True and quoted.stray_close_tag is True
    assert quoted.close_tag_index == QUOTED.index("</think>")
    # an unclosed literal: still content, never unterminated
    unclosed = ts.split_think("<think>never closed", mode=C.THINK_SPLIT_NONE)
    assert unclosed.answer_text == "<think>never closed" and unclosed.unterminated is False
    assert unclosed.stray_open_tag is True and unclosed.stray_close_tag is False
    # the same text under explicit_tags would have been stripped / marked unterminated — the modes differ on purpose
    assert ts.split_think(QUOTED, mode=C.THINK_SPLIT_EXPLICIT_TAGS).answer_text == ANSWER
    assert ts.split_think("<think>never closed", mode=C.THINK_SPLIT_EXPLICIT_TAGS).unterminated is True
    # non-string content reads as the empty answer, like every other mode
    assert ts.split_think(None, mode=C.THINK_SPLIT_NONE).answer_text == ""


@pytest.mark.parametrize("fixture", sorted(p.name for p in FIXTURES.glob("*.json")))
def test_the_three_existing_modes_are_byte_identical_on_the_committed_fixtures(fixture):
    doc = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    mode = doc["split_mode"]
    assert mode in (C.THINK_SPLIT_RECORD_ONLY, C.THINK_SPLIT_IMPLICIT_OPEN, C.THINK_SPLIT_EXPLICIT_TAGS)
    for case in doc["cases"]:
        if "completion" in case:                                  # the harness-side split fixtures (tags in the text)
            sp = ts.split_think(case["completion"], mode=mode)
            assert sp.answer_text == case["answer_text"], (fixture, case["name"])
            assert sp.present is case["present"] and sp.unterminated is case["unterminated"], (fixture, case["name"])
        else:                                                     # the parser-separated fixture (content + reasoning channels)
            sp = ts.split_think(case["content"], mode=mode)
            assert sp.answer_text == case["content"], (fixture, case["name"])   # record_only: byte-identical content
        meta = ts.split_metadata(sp)
        assert tuple(meta) == ts.SPLIT_METADATA_FIELDS          # no close-tag stray outside `none`
        assert "stray_close_tag" not in meta


def test_split_metadata_adds_the_close_tag_stray_under_none_only():
    meta = ts.split_metadata(ts.split_think(QUOTED, mode=C.THINK_SPLIT_NONE))
    assert tuple(meta) == ts.SPLIT_METADATA_FIELDS_NONE == (*ts.SPLIT_METADATA_FIELDS, "stray_close_tag")
    assert meta == {"think_block_present": False, "think_block_unterminated": False, "think_split_mode": "none",
                    "stray_open_tag": True, "stray_close_tag": True}
    assert C.STRAY_TAG_FIELDS == ("stray_open_tag", "stray_close_tag")


def _output(content: str, *, stop_reason: str = "stop"):
    pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock)")
    from inspect_ai._util.content import ContentText
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    msg = ChatMessageAssistant(content=[ContentText(text=content)])
    return ModelOutput(model="tinycap/x", choices=[ChatCompletionChoice(message=msg, stop_reason=stop_reason)])


def _state(content: str, *, target: str):
    pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock)")
    from inspect_ai.model import ModelName
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    return TaskState(model=ModelName("tinycap/x"), sample_id="gsm8k_deadbeef", epoch=1,
                     input="q", messages=[], target=Target(target), output=_output(content))


def _score(scorer, state):
    from inspect_ai.scorer import Target

    return asyncio.run(scorer(state, Target(state.target.text)))


def test_the_think_aware_scorer_under_none_scores_the_untouched_text_and_files_the_strays():
    pytest.importorskip("inspect_ai", reason="tinycap client venv (hashed lock)")    # the CPU suite's container carries none
    from inspect_ai.scorer import CORRECT, match

    scorer = ts.think_aware_scorer(match(numeric=True), mode=C.THINK_SPLIT_NONE)
    plain = _score(scorer, _state(ANSWER, target="42"))
    assert plain.value == CORRECT
    assert plain.metadata["think_split_mode"] == "none" and plain.metadata["stray_open_tag"] is False
    assert plain.metadata["stray_close_tag"] is False and plain.metadata["think_block_present"] is False
    assert plain.metadata[ts.SCORER_WRAPPER_FIELD] == "think_aware/none"
    # a quoted complete block followed by the answer: scored on the FULL text (the numeric match still reads the
    # trailing ANSWER line), and both strays filed — the reader refuses a primary leg on them, the scorer hides nothing
    quoted = _score(scorer, _state(QUOTED, target="42"))
    assert quoted.value == CORRECT
    assert quoted.metadata["stray_open_tag"] is True and quoted.metadata["stray_close_tag"] is True
    assert quoted.metadata["think_block_unterminated"] is False
    # never unscored under `none`: an unclosed literal is content, not a cut block
    unclosed = _score(scorer, _state("<think>never closed ANSWER: 42", target="42"))
    assert unclosed.value == CORRECT and unclosed.metadata["think_block_unterminated"] is False


# --------------------------------------------------------------------------- the shared readout helpers

def _rec(**kw):
    base = {"think_block_present": False, "think_block_unterminated": False, "stray_open_tag": False,
            "stray_close_tag": False, "at_cap": False, "scored": True}
    base.update(kw)
    return base


def test_no_reasoning_rule_and_reasoning_source_under_none_never_discard_evidence():
    assert set(M.NO_REASONING_RULE) == set(C.THINK_SPLIT_MODES)
    assert "STRAY-TAG ANOMALY" in M.NO_REASONING_RULE[C.THINK_SPLIT_NONE]
    none = C.THINK_SPLIT_NONE
    assert M.reasoning_source(_rec(), reasoning=None, split_mode=none) is None
    assert M.reasoning_source(_rec(stray_open_tag=True), reasoning=None, split_mode=none) == M.PRESENT_BY_RECORD
    assert M.reasoning_source(_rec(stray_close_tag=True), reasoning=None, split_mode=none) == M.PRESENT_BY_RECORD
    assert M.reasoning_source(_rec(think_block_present=True), reasoning=None, split_mode=none) == M.PRESENT_BY_RECORD
    assert M.reasoning_source(_rec(), reasoning="lifted text", split_mode=none) == M.PRESENT_BY_CHANNEL
    # the existing modes: byte-identical rule — the channel fallback applies to explicit_tags only, and a stray
    # flag alone never makes a present call there
    for mode in (C.THINK_SPLIT_RECORD_ONLY, C.THINK_SPLIT_IMPLICIT_OPEN):
        assert M.reasoning_source(_rec(stray_open_tag=True), reasoning="x", split_mode=mode) is None
    assert M.reasoning_source(_rec(stray_open_tag=True), reasoning="x", split_mode=C.THINK_SPLIT_EXPLICIT_TAGS) == M.PRESENT_BY_CHANNEL
    assert M.reasoning_source(_rec(stray_open_tag=True), reasoning=None, split_mode=C.THINK_SPLIT_EXPLICIT_TAGS) is None


def test_thinking_record_under_none_files_the_stray_draws_by_source_and_by_construction():
    items = {"a": _rec(), "b": _rec(stray_open_tag=True, stray_close_tag=True), "c": _rec(), "d": _rec(at_cap=True)}
    rows_reasoning = {"a": None, "b": None, "c": "lifted reasoning text", "d": None}
    rows_lifted = {"a": {"client_lift_fired": False, "lifted_reasoning": None},
                   "b": {"client_lift_fired": True, "lifted_reasoning": "quoted reasoning"},
                   "d": {"client_lift_fired": True, "lifted_reasoning": "x"}}
    rec = M.thinking_record(items, split_mode=C.THINK_SPLIT_NONE, rows_reasoning=rows_reasoning, rows_lifted=rows_lifted)
    assert rec["split_mode"] == "none" and rec["by_construction"] is True
    assert rec["n_on_rows"] == 4 and rec["n_finished"] == 3
    assert rec["n_present_by_record"] == 1 and rec["n_present_by_channel"] == 2      # b by record; c by row channel, d by lift
    assert rec["n_stray_tag_draws"] == 3 and rec["stray_tag_draw_ids"] == ["b", "c", "d"]
    assert rec["stray_tag_draws_by_source"] == {M.PRESENT_BY_RECORD: 1, M.PRESENT_BY_CHANNEL: 2}
    assert rec["n_no_reasoning_emitted"] == 1 and rec["n_no_reasoning_at_cap"] == 0
    clean = M.thinking_record({"a": _rec(), "b": _rec(at_cap=True)}, split_mode=C.THINK_SPLIT_NONE, rows_reasoning={})
    assert clean["n_stray_tag_draws"] == 0 and clean["n_no_reasoning_emitted"] == 1 and clean["n_no_reasoning_at_cap"] == 1
    # the existing modes: `rows_lifted` is ignored and no stray fields appear (byte-identical record)
    on = M.thinking_record({"a": _rec(think_block_present=True)}, split_mode=C.THINK_SPLIT_IMPLICIT_OPEN,
                           rows_reasoning={"a": None}, rows_lifted={"a": {"client_lift_fired": True}})
    assert "n_stray_tag_draws" not in on and on["by_construction"] is True and on["n_present_by_record"] == 1
    with pytest.raises(M.AnalysisError):
        M.thinking_record({}, split_mode="sideways", rows_reasoning={})


def test_the_no_switch_class_readers_require_the_class_records_and_the_exact_inventory():
    fact = dict(C.FAMILY_RENDER_FACTS["llama"].as_dict())
    fact["off_ids"], fact["on_ids"] = list(fact["off_ids"]), list(fact["on_ids"])
    rc = C.no_switch_render_certified(family="llama", header_ids=fact["off_ids"], measured_live=True,
                                      probe_regime=C.RENDER_REGIME_UNSPECIFIED,
                                      probe_completion={"content": "42", "reasoning": None, "finish_reason": "stop"})
    assert M._render_certified_problems(rc, fact=fact, family="llama") == []
    # a switch-class (A) record on the no-switch fact refuses by name; the no-switch record on a switched fact too
    switched = C.FAMILY_RENDER_FACTS["nemotron_h"]
    on_rc = C.render_certified(family="nemotron_h", header_ids_by_arm={"off": list(switched.off_ids), "on": list(switched.on_ids)},
                               measured_live=True, probe_regimes={"on": "thinking_on", "off": "thinking_off"},
                               on_completion={"content": "reasoning</think>Blue.", "reasoning": None, "finish_reason": "stop"})
    assert any("cert_class" in p for p in M._render_certified_problems(on_rc, fact=fact, family="llama"))
    nfact = {**switched.as_dict(), "off_ids": list(switched.off_ids), "on_ids": list(switched.on_ids)}
    assert any("no-switch class" in p for p in M._render_certified_problems(rc, fact=nfact, family="nemotron_h"))
    assert M._render_certified_problems(on_rc, fact=nfact, family="nemotron_h") == []        # the ON class, unchanged
    # (C″): exactly the inapplicability record; an ON-bite block or an absent block refuses
    ok = C.no_switch_lens_on_bite_record()
    assert M._lens_on_bite_problems(ok, route_vmodel="petri__x", cert_class=C.CERT_CLASS_NO_SWITCH) == []
    assert M._lens_on_bite_problems(None, route_vmodel="petri__x", cert_class=C.CERT_CLASS_NO_SWITCH)
    bad = M._lens_on_bite_problems({"applicable": True, "verified": True}, route_vmodel="petri__x", cert_class=C.CERT_CLASS_NO_SWITCH)
    assert bad and "ON-bite block" in bad[0]
    assert M._lens_on_bite_problems({**ok, "reason": "other"}, route_vmodel="petri__x", cert_class=C.CERT_CLASS_NO_SWITCH)
    # the exact inventory: one render probe; on a fitted cell the bites on route + twin, all under not_available
    inv, problem = M.probe_inventory_per_run({"steering_bites": {"n_probes": 5}}, route_vmodel="petri__x", fitted=True,
                                             cert_class=C.CERT_CLASS_NO_SWITCH)
    assert problem is None
    assert inv == {("petri__x", "not_available", "render"): 1, ("petri__x", "not_available", "bites"): 5,
                   (C.twin_vmodel(), "not_available", "bites"): 5}
    inv0, problem0 = M.probe_inventory_per_run({}, route_vmodel="petri__x", fitted=False, cert_class=C.CERT_CLASS_NO_SWITCH)
    assert problem0 is None and inv0 == {("petri__x", "not_available", "render"): 1}
    _inv, problem_missing = M.probe_inventory_per_run({}, route_vmodel="petri__x", fitted=True, cert_class=C.CERT_CLASS_NO_SWITCH)
    assert problem_missing and "n_probes" in problem_missing
    # the default class is the ON inventory, byte for byte
    inv_on, _ = M.probe_inventory_per_run({}, route_vmodel="petri__x", fitted=False)
    assert inv_on == {("petri__x", "off", "render"): 1, ("petri__x", "on", "render"): 1}


def test_the_contracts_no_switch_class_refuses_the_mixed_cases_by_name():
    llama, nemo = C.FAMILY_RENDER_FACTS["llama"], C.FAMILY_RENDER_FACTS["nemotron_h"]
    assert C.cert_class_for(C.THINKING_NOT_AVAILABLE, llama) == C.CERT_CLASS_NO_SWITCH
    assert C.cert_class_for(C.THINKING_ON, nemo) == C.CERT_CLASS_SWITCH == C.cert_class_for(C.THINKING_OFF, nemo)
    with pytest.raises(ValueError, match="switch-less family only"):
        C.cert_class_for(C.THINKING_NOT_AVAILABLE, nemo)
    with pytest.raises(ValueError, match="no thinking-ON class"):
        C.cert_class_for(C.THINKING_OFF, llama)
    with pytest.raises(ValueError):
        C.cert_class_for(C.THINKING_NOT_AVAILABLE, None)
    # (A″) refuses each hole by name, the evidence carrying every flag false
    ids = list(llama.off_ids)
    with pytest.raises(C.RenderCertificationError, match="tuple equality") as e:
        C.no_switch_render_certified(family="llama", header_ids=ids[:-1], measured_live=True)
    assert e.value.evidence["render_certified"] is False and e.value.evidence["cert_class"] == C.CERT_CLASS_NO_SWITCH
    with pytest.raises(C.RenderCertificationError, match="local-tokenizer"):
        C.no_switch_render_certified(family="llama", header_ids=ids, measured_live=False)
    with pytest.raises(C.RenderCertificationError, match="unspecified"):
        C.no_switch_render_certified(family="llama", header_ids=ids, measured_live=True, probe_regime="thinking_off")
    with pytest.raises(C.RenderCertificationError, match="think-tag literal"):
        C.no_switch_render_certified(family="llama", header_ids=ids, measured_live=True,
                                     probe_completion={"content": "<think>x</think>42", "reasoning": None, "finish_reason": "stop"})
    with pytest.raises(C.RenderCertificationError, match="reasoning channel"):
        C.no_switch_render_certified(family="llama", header_ids=ids, measured_live=True,
                                     probe_completion={"content": "42", "reasoning": "r", "finish_reason": "stop"})
    with pytest.raises(C.RenderCertificationError, match="carries a thinking switch"):
        C.no_switch_render_certified(family="nemotron_h", header_ids=list(nemo.off_ids), measured_live=True)
    ok = C.no_switch_render_certified(family="llama", header_ids=ids, measured_live=True, probe_regime="unspecified",
                                      probe_completion={"content": "42", "reasoning": None, "finish_reason": "stop"})
    assert ok["render_certified"] and ok[C.OOS_EVIDENCE_HEADER_CERTIFIED] and ok["render_regime_certified"] and ok["think_shape_certified"]
    assert ok["measured_arms"] == ["not_available"] and ok["measured_header_ids_by_arm"] == {"not_available": ids}
    assert set(ok) >= set(M.RENDER_CERTIFIED_FIELDS)
    # the stray-tag rule over a leg's items + rows
    items = {"a": _rec(), "b": _rec(stray_close_tag=True)}
    probs = C.no_switch_stray_tag_problems(items, lifted={"a": {"client_lift_fired": True, "lifted_reasoning": "q"}})
    assert len(probs) == 2 and "client_lift_fired" in probs[0] and "stray_close_tag" in probs[1]
    assert C.no_switch_stray_tag_problems({"a": _rec()}) == []
