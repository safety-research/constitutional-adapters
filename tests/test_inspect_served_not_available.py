"""llama-paper-rows — the NO-SWITCH arm ``not_available`` in the
Inspect provider, the task planner and the row adapter:

* a ``not_available`` provider sends NO ``chat_template_kwargs`` (the ledger's ``body_facts`` then
  records ``thinking: null``, the request evidence ``enable_thinking_present`` False), refuses a
  caller's kwargs of ANY value, sends no declaration;
* ``plan_task(..., thinking=not_available, family="llama")`` plans with ``think_split_mode none``,
  the cap from the setter (REQUIRED); the OFF / ON plans are byte-identical fixtures;
* **the P21 regression, THROUGH THE PINNED SDK** (inspect_ai 0.3.260, a synthetic HTTP response
  over httpx's MockTransport -> the actual ``TinycapAPI`` subclass -> the GSM8K scorer -> the GSM8K
  row adapter): an assistant content ``"<think>quoted reasoning</think>\\nSo the total is 42.\\nANSWER: 42"``
  — the SDK's own conversion WOULD lift the block (the control), the subclass restores the raw
  text byte for byte (one ``ContentText == raw``, no ``ContentReasoning``), records
  ``client_lift_fired`` / ``lifted_reasoning`` / ``raw_content_sha256``, the scorer files
  ``stray_open_tag`` / ``stray_close_tag``, the adapter puts every fact on the row, and the
  reader's stray-tag rule refuses the draw by name; a plain completion passes with
  ``client_lift_fired`` False; the same response under an ON family converts EXACTLY as today
  (the lift kept) and under OFF likewise;
* the row-level provenance (``--bank-view``): every emitted row gains ``object_artifact_sha256`` /
  ``route_kind`` / ``view`` / ``producer_tree_sha`` beside its ``thinking`` arm, and a run without
  the flag emits rows byte-identical to before.

Everything here is SYNTHETIC (placeholder digests, ``TESTHOST``-free bodies, invented items).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from test_inspect_served import (  # noqa: F401  (`fake` is a fixture)
    BASE_VMODEL,
    FIXTURE_ROUTES,
    LENS_ROUTE,
    VMODEL,
    _Emitter,
    _Log,
    _Sample,
    _Score,
    _Store,
    _api,
    _fake_eval,
    _gen,
    _plan,
    _route_of,
    fake,
)

from sisteer.capeval.serving import usage_ledger as ul
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tiers import think_split as ts
from sisteer.capeval.tinycap import contracts as C

NA = C.THINKING_NOT_AVAILABLE
ANSWER = "So the total is 42.\nANSWER: 42"
QUOTED = "<think>quoted reasoning</think>\n" + ANSWER


def _user():
    """The one user turn every round trip here sends (inspect_ai's message object, as `_gen` expects)."""
    from inspect_ai.model import ChatMessageUser

    return [ChatMessageUser(content="A farm has 6 hens laying 7 eggs each; how many eggs?")]

CAP = {C.GSM8K_PAIRED_TASK_NAME: 8192}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- the provider's request shape

def test_a_not_available_provider_sends_no_chat_template_kwargs_and_no_declaration(fake):  # noqa: F811
    api = _api(fake, si=None, thinking=NA, vmodel=BASE_VMODEL)
    assert api.no_switch_arm(NA) and not api.no_switch_arm(C.THINKING_OFF)
    fake.message = {"role": "assistant", "content": ANSWER}
    _gen(api, _user())
    body = fake.bodies[-1]
    assert C.CHAT_TEMPLATE_KWARGS_KEY not in body and C.OOS_THINKING_DECLARATION_KEY not in body
    # the two producer records of that body: the ledger's `thinking` is NULL (P17: no switch was sent) and the
    # request evidence says the key was ABSENT — the analysis holds both
    assert ul.body_facts(body)["thinking"] is None
    rec = C.request_evidence_record(fake.raw_bodies[-1], headers={C.HEADER_REQUEST_ID: "ab" * 16})
    assert rec["enable_thinking_present"] is False and rec["enable_thinking_value"] is None
    # OFF sends the explicit `false` (byte-identical to before): the two arms are told apart at both layers
    off = _api(fake, si=None, thinking=C.THINKING_OFF, vmodel=BASE_VMODEL)
    _gen(off, _user())
    assert fake.bodies[-1][C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    assert ul.body_facts(fake.bodies[-1])["thinking"] is False


def test_a_not_available_provider_refuses_a_callers_kwargs_of_any_value(fake):  # noqa: F811
    from inspect_ai.model._generate_config import GenerateConfig as GC

    api = _api(fake, si=None, thinking=NA, vmodel=BASE_VMODEL)
    # the shared provider (since the IFEval battery landed) refuses a caller's SWITCH of either value by name; an EMPTY kwargs map
    # carries no switch and is admitted — and dropped, so the wire body carries no chat_template_kwargs at all (P7)
    for value in ({C.ENABLE_THINKING_KEY: False}, {C.ENABLE_THINKING_KEY: True}):
        with pytest.raises(isv.InspectServedError, match="no switch is sent, none may be set"):
            api.tinycap_config(GC(extra_body={C.CHAT_TEMPLATE_KWARGS_KEY: value}))
    cfg = api.tinycap_config(GC(extra_body={C.CHAT_TEMPLATE_KWARGS_KEY: {}}))
    assert C.CHAT_TEMPLATE_KWARGS_KEY not in (cfg.extra_body or {})
    # the base class's own path is untouched: an OFF provider MERGES the right value and refuses the wrong one
    off = _api(fake, si=None, thinking=C.THINKING_OFF, vmodel=BASE_VMODEL)
    cfg = off.tinycap_config(GC(extra_body={C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: False}}))
    assert cfg.extra_body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    with pytest.raises(isv.InspectServedError, match="never overridden"):
        off.tinycap_config(GC(extra_body={C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: True}}))


# --------------------------------------------------------------------------- the planner

def test_plan_task_under_not_available_splits_none_takes_the_setters_cap_and_needs_the_family_fact(monkeypatch):
    kw = dict(grader_model=None, thinking=NA, family="llama", require_certified_fact=False, on_cap_override=CAP)
    p = isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, **kw)
    assert p.think_split_mode == C.THINK_SPLIT_NONE and p.thinking == NA
    assert p.max_tokens == 8192 and p.cap_source == C.CAP_SOURCE_ON_CAP_EXPLICIT_NOT_AVAILABLE
    assert p.family == "llama" and p.render_evidence_status == C.RENDER_EVIDENCE_CERTIFIED
    assert p.expected_n == C.GSM8K_PAIRED_N and len(p.sample_ids or []) == C.GSM8K_PAIRED_N
    # the tree's own fact is CERTIFIED (the GSM8K diagnostic smoke): production plans it; a SYNTHETIC demotion (only
    # `evidence_status` moves) is refused by name, and the dry run's `pre` reading still admits the demoted fact
    prod = isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=NA, family="llama", on_cap_override=CAP)
    assert prod.render_evidence_status == C.RENDER_EVIDENCE_CERTIFIED and prod.think_split_mode == C.THINK_SPLIT_NONE
    demoted = C.FAMILY_RENDER_FACTS["llama"]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**C.FAMILY_RENDER_FACTS, "llama": demoted})
    with pytest.raises(isv.InspectServedError, match="render fact not certified"):
        isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=NA, family="llama", on_cap_override=CAP)
    assert isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, **kw).render_evidence_status == C.RENDER_EVIDENCE_CANDIDATE
    monkeypatch.undo()
    # the setter is REQUIRED (no table applies to a no-switch family)
    with pytest.raises(isv.InspectServedError, match="setter is REQUIRED"):
        isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=NA, family="llama", require_certified_fact=False)
    # a switched family under the no-switch arm, no family at all, a parser or an opened-block claim: refused by name
    with pytest.raises(isv.InspectServedError, match="switch-less family only"):
        isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=NA, family="nemotron_h",
                      require_certified_fact=False, on_cap_override=CAP)
    with pytest.raises(isv.InspectServedError, match="needs the family's render fact"):
        isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=NA, require_certified_fact=False, on_cap_override=CAP)
    with pytest.raises(isv.InspectServedError, match="composes none"):
        isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, reasoning_parser="gemma4", **kw)
    with pytest.raises(isv.InspectServedError, match="opens no block"):
        isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, prompt_opened_think=True, **kw)
    for flag in ("out_of_support", "off_cap_override"):
        with pytest.raises(isv.InspectServedError, match="no thinking switch has no render"):
            isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, **{**kw, flag: True})
    # the OFF and ON plans are byte-identical to before: the arm never touches them
    off = isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF, family="nemotron_h")
    assert (off.max_tokens, off.cap_source, off.think_split_mode) == (512, "off_table", C.THINK_SPLIT_RECORD_ONLY)
    on = isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=C.THINKING_ON, family="nemotron_h",
                       on_cap_override=CAP)
    assert (on.max_tokens, on.cap_source, on.think_split_mode) == (8192, C.CAP_SOURCE_ON_CAP_EXPLICIT, C.THINK_SPLIT_IMPLICIT_OPEN)


# --------------------------------------------------------------------------- P21: the response-conversion boundary

def _mock_api(monkeypatch, *, thinking: str, content: str, route: Dict[str, Any], vmodel: str) -> Any:
    """The actual ``TinycapAPI`` subclass over the PINNED SDK's whole conversion path, fed ONE synthetic
    HTTP response through httpx's MockTransport (no server): the response is a real OpenAI chat completion
    body stamped the way the fake proxy stamps it, so ``on_response`` / ``chat_choices_from_openai`` /
    ``parse_reasoning_content`` all run as in production."""
    import httpx
    from inspect_ai.model._generate_config import GenerateConfig
    from test_inspect_served import proxy_stamps, regime_stamps

    sent: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        sent.append(body)
        stamps = proxy_stamps(route)
        stamps["petri"] = {**stamps["petri"], **regime_stamps(route, body)}
        return httpx.Response(200, json={
            "id": "cmpl-synthetic", "object": "chat.completion", "created": 0, "model": vmodel,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52}, **stamps})

    cls = isv._register_provider()
    api = cls(model_name=C.inspect_model_name(vmodel), base_url="http://synthetic.invalid/v1", api_key="synthetic",
              config=GenerateConfig(), si=None, thinking=thinking, route=route,
              http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    api.initialize()
    return api, sent


def _state_for(output, *, target: str):
    from inspect_ai.model import ModelName
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    return TaskState(model=ModelName("tinycap/x"), sample_id="gsm8k_deadbeef", epoch=1, input="q", messages=[],
                     target=Target(target), output=output)


def test_the_sdk_alone_would_lift_a_complete_think_block_out_of_the_text(monkeypatch):
    """THE CONTROL: the pinned SDK's own conversion (``chat_choices_from_openai`` -> ``parse_reasoning_content``)
    lifts a complete ``<think>…</think>`` block into a ``ContentReasoning`` part with NO server parser — the
    artefact the no-switch policy exists to undo. Read straight off the SDK, so the regression below is against a
    behaviour this tree observes rather than assumes."""
    from inspect_ai.model._openai import chat_choices_from_openai
    from openai.types.chat import ChatCompletion

    completion = ChatCompletion.model_validate({
        "id": "cmpl-control", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": QUOTED}, "finish_reason": "stop"}]})
    msg = chat_choices_from_openai(completion, [])[0].message
    kinds = [getattr(c, "type", None) for c in msg.content_list]
    assert kinds == ["reasoning", "text"], kinds
    assert msg.text == ANSWER and msg.text != QUOTED                       # the text the scorer WOULD have read


@pytest.mark.parametrize("content,fired", [(QUOTED, True), (ANSWER, False)])
def test_the_no_switch_policy_restores_the_raw_text_records_the_lift_and_the_reader_refuses_the_stray(monkeypatch, content, fired):
    from inspect_ai.scorer import CORRECT, match

    api, sent = _mock_api(monkeypatch, thinking=NA, content=content, route=_route_of(BASE_VMODEL), vmodel=BASE_VMODEL)
    out = _gen(api, _user())
    output = out[0] if isinstance(out, tuple) else out
    assert C.CHAT_TEMPLATE_KWARGS_KEY not in sent[-1]                      # the wire carried no switch
    msg = output.choices[0].message
    # (ii) exactly ONE ContentText == the raw content, byte for byte; no ContentReasoning survives
    assert [getattr(c, "type", None) for c in msg.content_list] == ["text"]
    assert msg.text == content and output.completion == content
    # (iii) the lift is RECORDED, never dropped
    rec = output.metadata[isv.NO_SWITCH_METADATA_KEY]
    assert len(rec) == 1 and set(rec[0]) == set(C.ROW_LIFT_FIELDS)
    assert rec[0]["client_lift_fired"] is fired
    assert rec[0]["lifted_reasoning"] == ("quoted reasoning" if fired else None)
    assert rec[0]["raw_content_sha256"] == _sha(content)
    # the SCORER (the GSM8K wrap under `none`) reads the full text and files the strays
    scorer = ts.think_aware_scorer(match(numeric=True), mode=C.THINK_SPLIT_NONE)
    state = _state_for(output, target="42")
    from inspect_ai.scorer import Target
    score = asyncio.run(scorer(state, Target("42")))
    assert score.value == CORRECT
    assert score.metadata["stray_open_tag"] is fired and score.metadata["stray_close_tag"] is fired
    assert score.metadata["think_block_present"] is False and score.metadata["think_block_unterminated"] is False
    # the ROW ADAPTER puts every fact on the row; the scored text's digest == the raw body's
    log = _Log([_Sample("gsm8k_deadbeef", {ts.WRAPPER_NAME: score}, output=output)])
    res = isv.results_from_log(log, target_model=None)
    r = res[0]
    assert r.content == content and r.reasoning is None
    assert (r.client_lift_fired, r.lifted_reasoning, r.raw_content_sha256) == (fired, "quoted reasoning" if fired else None, _sha(content))
    assert r.stray_open_tag is fired and r.stray_close_tag is fired
    extra = r.extra(thinking=NA, max_tokens=8192)
    assert extra["thinking"] == NA and extra["stray_close_tag"] is fired
    assert {f: extra[f] for f in C.ROW_LIFT_FIELDS} == dict(rec[0])
    assert _sha(extra["content"]) == extra["raw_content_sha256"]
    # the READER's stray-tag rule under `none`: the quoted draw is an anomaly by name, the plain one is clean
    item = {"think_block_present": False, "stray_open_tag": r.stray_open_tag, "stray_close_tag": r.stray_close_tag, "at_cap": False}
    problems = C.no_switch_stray_tag_problems({"gsm8k_deadbeef": item},
                                              lifted={"gsm8k_deadbeef": {f: extra[f] for f in C.ROW_LIFT_FIELDS}})
    if fired:
        assert len(problems) == 1 and "stray_open_tag" in problems[0] and "client_lift_fired" in problems[0]
    else:
        assert problems == []
    # the outcome record of a `none` plan carries the close-tag stray (the no-switch item shape) — and nothing else new
    plan = _plan(row_task=C.GSM8K_PAIRED_TASK, name=C.GSM8K_PAIRED_TASK_NAME, think_split_mode=C.THINK_SPLIT_NONE,
                 thinking=NA, max_tokens=8192, expected_n=1, item_set=C.GSM8K_ITEM_SET_PAIRED)
    summary = {"n_scored": 1, "n_unscored": 0, "n_scored_samples": 1, "n_unscored_samples": 0, "n_stop_reason": 1, "n_at_cap": 0}
    counts = isv.think_split_counts(res, plan=plan)
    header = isv.outcome_record(res, plan=plan, task=C.GSM8K_PAIRED_TASK, thinking=NA, summary=summary, split_counts=counts,
                                producer_tree_sha="0" * 40, sampling_ref="0" * 12)
    assert set(header["items"][0]) == set(C.TASK_OUTCOME_ITEM_FIELDS_NO_SWITCH)
    assert header["items"][0]["stray_close_tag"] is fired and header["think_split_mode"] == "none"


@pytest.mark.parametrize("thinking", [C.THINKING_ON, C.THINKING_OFF])
def test_on_and_off_conversion_are_untouched_the_lift_stays_and_nothing_is_recorded(monkeypatch, thinking):
    """The switched arms convert EXACTLY as today (the SDK's lift kept; the goldens): a ContentReasoning part
    first, the answer text second, NO no-switch metadata, `reasoning` on the row — the policy is scoped to the
    no-switch arm and to nothing else."""
    route = _route_of(BASE_VMODEL)
    api, sent = _mock_api(monkeypatch, thinking=thinking, content=QUOTED, route=route, vmodel=BASE_VMODEL)
    out = _gen(api, _user())
    output = out[0] if isinstance(out, tuple) else out
    assert sent[-1][C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: thinking == C.THINKING_ON}
    msg = output.choices[0].message
    assert [getattr(c, "type", None) for c in msg.content_list] == ["reasoning", "text"]
    assert msg.text == ANSWER
    assert not (output.metadata or {}).get(isv.NO_SWITCH_METADATA_KEY)
    content, reasoning = ts.message_channels(msg)
    assert content == ANSWER and reasoning == "quoted reasoning"
    log = _Log([_Sample("gsm8k_deadbeef", {"s": _Score("C")}, output=output)])
    r = isv.results_from_log(log, target_model=None)[0]
    assert r.client_lift_fired is None and r.stray_close_tag is None
    extra = r.extra(thinking=thinking, max_tokens=8192)
    assert not (set(C.ROW_LIFT_FIELDS) | {"stray_close_tag"}) & set(extra)          # byte-identical row keys


def test_a_surviving_reasoning_part_or_a_changed_text_refuses_in_assert_channels(fake):  # noqa: F811
    """The braces on the belt: `assert_channels` under the no-switch arm refuses a converted output that still
    carries a reasoning part, or whose text digest is not the raw content's — the two things the restoration
    must have removed / preserved."""
    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    api = _api(fake, si=None, thinking=NA, vmodel=BASE_VMODEL)
    good = ModelOutput(model="m", choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=[ContentText(text=ANSWER)]), stop_reason="stop")],
                       metadata={isv.NO_SWITCH_METADATA_KEY: [{"client_lift_fired": False, "lifted_reasoning": None, "raw_content_sha256": _sha(ANSWER)}]})
    api.assert_channels(good)
    surviving = good.model_copy(deep=True)
    surviving.choices[0].message.content = [ContentReasoning(reasoning="r"), ContentText(text=ANSWER)]
    with pytest.raises(isv.InspectServedError, match="still carries 1 reasoning part"):
        api.assert_channels(surviving)
    changed = good.model_copy(deep=True)
    changed.choices[0].message.content = [ContentText(text=ANSWER + " ")]
    with pytest.raises(isv.InspectServedError, match="text sha256"):
        api.assert_channels(changed)
    unrecorded = good.model_copy(deep=True)
    unrecorded.metadata = {}
    with pytest.raises(isv.InspectServedError, match="recorded 0 choice"):
        api.assert_channels(unrecorded)


# --------------------------------------------------------------------------- the row-level provenance

def _one_log():
    return _Log([_Sample("gsm8k_aaaa1111", {"s": _Score("C")}), _Sample("gsm8k_bbbb2222", {"s": _Score("I")})])


def test_rows_carry_the_provenance_iff_bank_view_was_given_and_are_byte_identical_otherwise(tmp_path, monkeypatch):
    monkeypatch.setattr("inspect_ai.eval_async", _fake_eval({0: _one_log()}))
    # the row identity block (main's `row_identity_block` since the IFEval battery landed): our four provenance facts + the sampling mode
    prov = C.row_identity_block(cell="constv2-mix-full__psr_ce_all", view="it", object_artifact_sha256="ab" * 32,
                                producer_tree_sha="cd" * 20, sampling_mode=C.SAMPLING_CARD)
    assert tuple(prov)[:4] == C.ROW_PROVENANCE_FIELDS == C.ROW_IDENTITY_FIELDS == ("object_artifact_sha256", "route_kind", "view", "producer_tree_sha")
    assert {k: prov[k] for k in C.ROW_PROVENANCE_FIELDS} == {"object_artifact_sha256": "ab" * 32, "route_kind": "lens", "view": "it", "producer_tree_sha": "cd" * 20}
    em, store = _Emitter(), _Store(tmp_path / "cell")
    asyncio.run(isv.run_one_task(_plan(samples_per_item=1, primary_scorer=None, max_tokens=8192, thinking=NA),
                                 model=None, store=store, template={}, thinking=NA, concurrency=8, emitter=em, row_identity=prov))
    rows = em.by(C.METRIC_ACC)
    assert len(rows) == 2
    for r in rows:
        assert r["extra"][C.ROW_KEY_THINKING] == NA
        assert {f: r["extra"][f] for f in (*C.ROW_PROVENANCE_FIELDS, C.ROW_SAMPLING_MODE_FIELD)} == prov
    # without the flag: the SAME rows minus exactly the four keys (byte-identical to before)
    em2, store2 = _Emitter(), _Store(tmp_path / "cell2")
    asyncio.run(isv.run_one_task(_plan(samples_per_item=1, primary_scorer=None, max_tokens=8192, thinking=NA),
                                 model=None, store=store2, template={}, thinking=NA, concurrency=8, emitter=em2))
    plain = em2.by(C.METRIC_ACC)
    identity_keys = {*C.ROW_PROVENANCE_FIELDS, C.ROW_SAMPLING_MODE_FIELD}
    assert [{k: v for k, v in r["extra"].items() if k not in identity_keys} for r in rows] == [r["extra"] for r in plain]
    assert not any(identity_keys & set(r["extra"]) for r in plain)
    # the fragment is refused wrong, never recorded wrong
    assert C.row_provenance(cell="none", view=None, object_artifact_sha256=None, producer_tree_sha=None) == {}
    assert C.row_provenance(cell="none", view="main", object_artifact_sha256=None, producer_tree_sha="cd" * 20)["route_kind"] == "base"
    assert C.row_provenance(cell="prompted__safety_refusal", view="it", object_artifact_sha256=None, producer_tree_sha="cd" * 20)["route_kind"] == "prompted"
    with pytest.raises(ValueError, match="serves no object"):
        C.row_provenance(cell="none", view="main", object_artifact_sha256="ab" * 32, producer_tree_sha="cd" * 20)
    with pytest.raises(ValueError, match="64-hex"):
        C.row_provenance(cell="constv2-mix-full__psr_ce_all", view="main", object_artifact_sha256=None, producer_tree_sha="cd" * 20)
    with pytest.raises(ValueError, match="40-hex"):
        C.row_provenance(cell="none", view="main", object_artifact_sha256=None, producer_tree_sha="short")
    # the known-bad view is one NO roster declares: since 2026-09-18 the vocabulary is
    # `main` | `it` + every full22 roster view (contracts.bank_views_admitted — `halfdose`, a roster view, is admitted now)
    with pytest.raises(ValueError, match="--bank-view must be one of"):
        C.row_provenance(cell="none", view="nowhere", object_artifact_sha256=None, producer_tree_sha="cd" * 20)
    assert C.BANK_VIEWS == ("main", "it") and C.ROUTE_KINDS == ("base", "prompted", "lens") and C.FLAG_BANK_VIEW == "--bank-view"
