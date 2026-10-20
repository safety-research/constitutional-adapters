"""math500-thinking-on-full22, lane A (a2 / a4) — `tiers/inspect_served.py` and
`tiers/think_split.py` under thinking ON: the judge-free think-aware MATH
scorer registered under the symbolic
scorer's name with a REAL scorer -> Inspect ``.eval`` log -> ``results_from_log``
round trip, `math` out of the model-graded set and NO grader built on a
MATH-only run, the family split modes on the committed synthetic
fixtures INCLUDING the truncated Nemotron row (no score), channel-aware
censoring for gemma's parser-separated cap-bound outputs on actual
``ModelOutput`` channels (three cases), fact-fed planning
(`prompt_opened_think` from the family fact, never from OOS; a `candidate` fact
refuses production planning), the MATH prompt renderer behind
``task_messages`` and the single-pass window pre-flight.

Every completion here is SYNTHETIC, built from each family's real tags.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("inspect_ai")
pytest.importorskip("inspect_evals")

from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402
from sisteer.capeval.tiers import think_split as ts  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "think_split"
FAMILIES = ("gemma4", "qwen3_5", "glm4v", "nemotron_h")


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _output(content: str, *, reasoning=None, stop_reason: str = "stop"):
    """A real ``ModelOutput`` with the two channels inspect_ai 0.3.260 uses:
    TEXT parts (``content``) and ``ContentReasoning`` parts (``reasoning``)."""
    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput

    parts = []
    if reasoning:
        parts.append(ContentReasoning(reasoning=reasoning))
    parts.append(ContentText(text=content))
    msg = ChatMessageAssistant(content=parts)
    return ModelOutput(model="tinycap/x", choices=[ChatCompletionChoice(message=msg, stop_reason=stop_reason)])


def _state(output, *, target: str = "6"):
    from inspect_ai.model import ModelName
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    return TaskState(model=ModelName("tinycap/x"), sample_id="math_00000001", epoch=1,
                     input="q", messages=[], target=Target(target), output=output)


def _score(scorer, state):
    from inspect_ai.scorer import Target

    return asyncio.run(scorer(state, Target(state.target.text)))


def _is_unscored(score) -> bool:
    v = score.value
    return isinstance(v, float) and math.isnan(v)


def _fixture(family: str) -> dict:
    return json.loads((FIXTURES / f"{family}.json").read_text())


# --------------------------------------------------------------------------- #
# the scorer name alias and the real log round trip                            #
# --------------------------------------------------------------------------- #


def test_the_math_scorer_is_the_judge_free_sympy_scorer_registered_under_its_own_name():
    from inspect_ai._util.registry import registry_info, registry_unqualified_name
    from inspect_evals.math.math import expression_exact_match_sympy

    inner = expression_exact_match_sympy()
    assert ts.inner_scorer_name(inner) == "expression_exact_match_sympy" == isv.PRIMARY_SCORER["math"]
    for mode in C.THINK_SPLIT_MODES:
        wrapped = isv._math_scorer(mode)
        assert registry_unqualified_name(registry_info(wrapped).name) == isv.PRIMARY_SCORER["math"]
    # the GSM8K wrap keeps its legacy name — the alias is MATH's alone (decisions-laneA LA5)
    assert registry_unqualified_name(registry_info(isv._gsm8k_scorer("implicit_open")).name) == ts.WRAPPER_NAME
    assert ts.WRAPPER_NAME == "think_aware_scorer"
    with pytest.raises(ValueError, match="registered name"):
        ts.think_aware_scorer(inner, mode="implicit_open", name="   ")


def test_a_stopped_unclosed_math500_draw_stays_unterminated_and_unscored_the_gsm8k_opt_in_is_not_math_s():
    """The no-think-block rule is the GSM8K wrap's OPT-IN
    (`think_aware_scorer(no_think_block_when_stopped=True)`); `_math_scorer` does not pass it, so a MATH-500 draw
    that stopped NATURALLY without ever closing its prompt-opened block files unterminated -> unscored exactly as
    before — the `eos_in_thoughts` class its paper reader (scripts/tinycap_math500_thinking.py) classifies off the
    raw body, under a contract that requires n_think_block_present == n on an implicit_open leg. A client-lifted
    block (ContentReasoning + answer text) stays unscored too — the reader's `lifted_close`."""
    scorer = isv._math_scorer(C.THINK_SPLIT_IMPLICIT_OPEN)
    inline = "Let n=3, 2n=6, so the answer is 6.\nANSWER: 6"                   # no `</think>` anywhere; a parseable answer line
    for stop_reason in ("stop", "model_length", "unknown", "max_tokens"):
        s = _score(scorer, _state(_output(inline, stop_reason=stop_reason), target="6"))
        assert _is_unscored(s) and s.answer is None, stop_reason
        assert s.metadata["think_block_present"] is True and s.metadata["think_block_unterminated"] is True, stop_reason
        assert ts.NO_THINK_BLOCK_FIELD not in s.metadata and ts.LIFTED_CLOSE_FIELD not in s.metadata, stop_reason
        assert s.metadata[ts.SCORER_WRAPPER_FIELD] == "think_aware/implicit_open"
    lifted = _score(scorer, _state(_output("ANSWER: 6", reasoning="Let n=3, 2n=6.", stop_reason="stop"), target="6"))
    assert _is_unscored(lifted) and ts.LIFTED_CLOSE_FIELD not in lifted.metadata and ts.NO_THINK_BLOCK_FIELD not in lifted.metadata
    # the closed block still splits and scores; the GSM8K wrap is the one that opts in
    ok = _score(scorer, _state(_output("Let n=3, 2n=6.</think>\nANSWER: 6", stop_reason="stop"), target="6"))
    assert not _is_unscored(ok) and isv.score_to_value(ok) == 1.0
    from inspect_ai._util.registry import registry_info
    assert registry_info(isv._gsm8k_scorer("implicit_open")).name.endswith(ts.WRAPPER_NAME)


def _parsed_output(raw: str, *, stop_reason: str = "stop"):
    """``raw`` through inspect_ai 0.3.260's REAL client-side lift (``parse_content_with_reasoning``), assembled into a ModelOutput the
    way ``_openai.chat_choices_from_openai`` does: a ContentReasoning part (when a pair was lifted) followed by the ContentText remainder."""
    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput
    from inspect_ai.model._reasoning import parse_content_with_reasoning

    text, capsule = parse_content_with_reasoning(raw)
    parts = ([ContentReasoning(reasoning=capsule.reasoning, signature=capsule.signature, redacted=capsule.redacted,
                               summary=capsule.summary, internal=capsule.internal)] if capsule else []) + [ContentText(text=text)]
    msg = ChatMessageAssistant(content=parts, source="generate")
    return ModelOutput(model="tinycap/x", choices=[ChatCompletionChoice(message=msg, stop_reason=stop_reason)])


def test_a_real_lift_beside_a_reopened_block_stays_unterminated_for_gsm8k_and_math_while_a_clean_lift_still_scores():
    """G8KSCORERFIX3: `<think>example</think>\n<think>still reasoning\nANSWER: 43` through the REAL
    inspect_ai parser lifts `example` into the reasoning channel and leaves an unclosed `<think>` in the text. The GSM8K wrap must
    not read the reasoning channel as a closed block: unscored, unterminated True, stray_open_tag True, neither lifted_close nor
    no_think_block; the row reader counts it in n_unterminated_not_at_cap. A clean lift (`<think>…</think>\nANSWER: 43`) still
    scores 1.0 with lifted_close. The MATH wrap is untouched: both shapes stay unterminated -> unscored under it."""
    raw = "<think>example</think>\n<think>still reasoning\nANSWER: 43"
    out = _parsed_output(raw)
    assert ts.output_channels(out) == ("<think>still reasoning\nANSWER: 43", "example")           # what the real lift leaves behind
    sp = ts.split_think(out.completion, mode=C.THINK_SPLIT_IMPLICIT_OPEN)
    assert (sp.present, sp.unterminated, sp.stray_open_tag) == (True, True, True)
    g = isv._gsm8k_scorer(C.THINK_SPLIT_IMPLICIT_OPEN)
    s = _score(g, _state(out, target="43"))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_present"] is sp.present and s.metadata["think_block_unterminated"] is True and s.metadata["stray_open_tag"] is True
    assert ts.LIFTED_CLOSE_FIELD not in s.metadata and ts.NO_THINK_BLOCK_FIELD not in s.metadata
    clean_out = _parsed_output("<think>17 + 26 = 43.</think>\nANSWER: 43")
    assert ts.output_channels(clean_out) == ("\nANSWER: 43", "17 + 26 = 43.") or ts.output_channels(clean_out)[1] == "17 + 26 = 43."
    clean = _score(g, _state(clean_out, target="43"))
    assert not _is_unscored(clean) and isv.score_to_value(clean) == 1.0 and clean.metadata[ts.LIFTED_CLOSE_FIELD] is True
    assert clean.metadata["think_block_unterminated"] is False and ts.NO_THINK_BLOCK_FIELD not in clean.metadata
    # the row reader + summary: the reopened draw is unterminated-not-at-cap, never a no-think or lifted close
    log = SimpleNamespace(status="success", error=None, samples=[
        SimpleNamespace(id="gsm8k_0000000a", scores={ts.WRAPPER_NAME: s}, error=None, model_usage={}, output=out),
        SimpleNamespace(id="gsm8k_0000000b", scores={ts.WRAPPER_NAME: clean}, error=None, model_usage={}, output=clean_out)])
    rows = {r.item_id: r for r in isv.results_from_log(log, primary_scorer=ts.WRAPPER_NAME)}
    assert rows["gsm8k_0000000a"].unscored and rows["gsm8k_0000000a"].think_block_unterminated and not rows["gsm8k_0000000a"].no_think_block
    counts = isv.think_split_counts(list(rows.values()), plan=SimpleNamespace(think_split_mode=C.THINK_SPLIT_IMPLICIT_OPEN))
    assert (counts["n_unterminated_not_at_cap"], counts["n_no_think_block"], counts["n_think_block_present"]) == (1, 0, 2)
    # MATH: no opt-in — both shapes unterminated -> unscored, no flags, exactly as before
    m = isv._math_scorer(C.THINK_SPLIT_IMPLICIT_OPEN)
    for o in (out, clean_out):
        ms = _score(m, _state(o, target="43"))
        assert _is_unscored(ms) and ms.metadata["think_block_unterminated"] is True
        assert ts.LIFTED_CLOSE_FIELD not in ms.metadata and ts.NO_THINK_BLOCK_FIELD not in ms.metadata


def test_a_reopened_block_with_no_lift_through_the_real_parser_stays_unterminated_for_gsm8k_and_math():
    """G8KSCORERFIX3: `<think>still reasoning\nANSWER: 43` carries no `</think>`, so the REAL inspect_ai parser lifts NOTHING
    (one ContentText, no reasoning channel); the GSM8K wrap's no-think branch used to score it. Now it is unterminated:
    unscored, stray_open_tag True, neither flag, counted by n_unterminated_not_at_cap; the inline no-block draw (`ANSWER: 43`) still
    scores 1.0 with no_think_block; the MATH wrap files it unterminated -> unscored as before."""
    raw = "<think>still reasoning\nANSWER: 43"
    out = _parsed_output(raw)
    assert ts.output_channels(out) == (raw, None)                                                   # the real parser lifted nothing
    g = isv._gsm8k_scorer(C.THINK_SPLIT_IMPLICIT_OPEN)
    s = _score(g, _state(out, target="43"))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_unterminated"] is True and s.metadata["stray_open_tag"] is True
    assert ts.LIFTED_CLOSE_FIELD not in s.metadata and ts.NO_THINK_BLOCK_FIELD not in s.metadata
    inline_out = _parsed_output("ANSWER: 43")
    inline = _score(g, _state(inline_out, target="43"))
    assert isv.score_to_value(inline) == 1.0 and inline.metadata[ts.NO_THINK_BLOCK_FIELD] is True
    log = SimpleNamespace(status="success", error=None, samples=[
        SimpleNamespace(id="gsm8k_0000000a", scores={ts.WRAPPER_NAME: s}, error=None, model_usage={}, output=out),
        SimpleNamespace(id="gsm8k_0000000b", scores={ts.WRAPPER_NAME: inline}, error=None, model_usage={}, output=inline_out)])
    rows = {r.item_id: r for r in isv.results_from_log(log, primary_scorer=ts.WRAPPER_NAME)}
    counts = isv.think_split_counts(list(rows.values()), plan=SimpleNamespace(think_split_mode=C.THINK_SPLIT_IMPLICIT_OPEN))
    assert rows["gsm8k_0000000a"].unscored and not rows["gsm8k_0000000a"].no_think_block
    assert (counts["n_unterminated_not_at_cap"], counts["n_no_think_block"]) == (1, 1)
    ms = _score(isv._math_scorer(C.THINK_SPLIT_IMPLICIT_OPEN), _state(out, target="43"))
    assert _is_unscored(ms) and ms.metadata["think_block_unterminated"] is True and ts.NO_THINK_BLOCK_FIELD not in ms.metadata


def test_a_real_scorer_log_round_trip_finds_the_primary_key_on_every_sample(tmp_path):
    """The round trip, REAL end to end: inspect's mock model answers a
    two-sample MATH-shaped task whose ONLY scorer is `_math_scorer(implicit_open)`;
    the `.eval` log is adapted by `results_from_log(primary_scorer="expression_exact_match_sympy")`
    and every sample carries the key — with the split recorded in the score's
    metadata and the wrapper's identity beside it. The truncated row (no closing
    tag, cut at the cap) files UNSCORED, never a wrong answer."""
    from inspect_ai import Task, eval_async
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.solver import generate

    outputs = [ModelOutput.from_content(model="mockllm/model", content="Let n=3, 2n=6.</think>\nANSWER: 6"),
               ModelOutput.from_content(model="mockllm/model", content="Let n=3, 2n=6 and ANSWER: 6 hmm",
                                        stop_reason="max_tokens")]
    model = get_model("mockllm/model", custom_outputs=outputs)
    task = Task(dataset=[Sample(input="What is 2*3?", target="6", id="math_aaaaaaaa"),
                         Sample(input="What is 2*3 again?", target="6", id="math_bbbbbbbb")],
                solver=[generate()], scorer=[isv._math_scorer(C.THINK_SPLIT_IMPLICIT_OPEN)])
    logs = asyncio.run(eval_async(task, model=model, log_dir=str(tmp_path / "logs"), log_format="eval",
                                  max_connections=1))
    log = logs[0]
    assert log.status == "success"
    results = isv.results_from_log(log, primary_scorer=isv.PRIMARY_SCORER["math"], target_model=None)
    assert [r.item_id for r in results] == ["math_aaaaaaaa", "math_bbbbbbbb"]
    scored, truncated = results
    assert scored.value == 1.0 and scored.think_block_present and not scored.think_block_unterminated
    assert truncated.unscored and truncated.think_block_present and truncated.think_block_unterminated
    for s in log.samples:
        assert set(s.scores) == {isv.PRIMARY_SCORER["math"]}
        assert s.scores[isv.PRIMARY_SCORER["math"]].metadata[ts.SCORER_WRAPPER_FIELD] == "think_aware/implicit_open"


# --------------------------------------------------------------------------- #
# the MATH plan: one scorer, no grader                                         #
# --------------------------------------------------------------------------- #


def _stub_math(monkeypatch, n: int = 3):
    import importlib

    math_mod = importlib.import_module("inspect_evals.math.math")
    ids = [C.math500_sample_id(f"problem {i}") for i in range(n)]
    monkeypatch.setattr(isv, "math500_sample_ids", lambda **k: list(ids))
    monkeypatch.setattr(isv, "verify_math500_selection",
                        lambda ids: {"matched": len(ids), "missing": [], "n_source": 5000})
    calls = []

    def fake_math(**kw):
        calls.append(dict(kw))
        return SimpleNamespace(scorer=["the task's three stock scorers"], dataset=[])

    monkeypatch.setattr(math_mod, "math", fake_math)
    return ids, calls


def test_the_math_plan_carries_only_the_wrapped_sympy_scorer_and_names_no_grader(monkeypatch):
    from inspect_ai._util.registry import registry_info, registry_unqualified_name

    ids, calls = _stub_math(monkeypatch)
    spy = SimpleNamespace(name="a grader model that must never be handed to the task")
    plan = isv.plan_task("math", grader_model=spy, thinking=C.THINKING_ON, family="qwen3_5")
    assert calls == [{"shuffle": False, "grader_model": None}]          # NEVER the spy
    assert isinstance(plan.task.scorer, list) and len(plan.task.scorer) == 1
    assert registry_unqualified_name(registry_info(plan.task.scorer[0]).name) == isv.PRIMARY_SCORER["math"]
    assert plan.scorer_wrapper == "think_aware/implicit_open" and plan.primary_scorer == isv.PRIMARY_SCORER["math"]
    assert plan.grader_max_tokens is None                             # not a model-graded task any more
    assert "math" not in isv.MODEL_GRADED_TASKS and C.inspect_task("math") not in C.MODEL_GRADED_INSPECT_TASKS
    assert plan.family == "qwen3_5" and plan.render_evidence_status == "certified"
    assert plan.sample_ids == ids and plan.think_split_mode == C.THINK_SPLIT_IMPLICIT_OPEN


def test_a_math_only_run_builds_no_grader_model_and_a_graded_run_still_does(monkeypatch, tmp_path):
    built = []
    real_stamp_route = isv.stamp_route

    def stub_model(vmodel, **kw):
        built.append(vmodel)
        return SimpleNamespace(api=SimpleNamespace(regime_stamps={}, route=real_stamp_route(kw["route"])))

    async def capture_run(plan, **kw):
        return {"plan": plan.row_task}

    monkeypatch.setattr(isv, "get_models", lambda **k: {"object": "list", "data": []})
    monkeypatch.setattr(isv, "preflight_models", lambda **k: list(k.get("required") or ()))
    monkeypatch.setattr(isv, "build_model", stub_model)
    monkeypatch.setattr(isv, "run_one_task", capture_run)
    monkeypatch.setattr(isv, "assert_route_can_serve", lambda *a, **k: None)
    monkeypatch.setattr(isv, "assert_live_agent_loop_entry", lambda *a, **k: None)
    monkeypatch.setattr(isv.completion, "is_done", lambda *a, **k: False)
    planned = []

    def fake_plans(atoms, **kw):
        planned.append(dict(kw))
        return [SimpleNamespace(row_task=C.inspect_task("math"), thinking=C.THINKING_ON)]

    monkeypatch.setattr(isv, "plans_for_atoms", fake_plans)
    route = {"cell_id": "si_x__lora_r4", "virtual_model": "petri__si_x__lora_r4", "serving_identity": "proxy_vllm+lens",
             "intervention_hash": "0" * 16}
    base = {"cell_id": "none", "virtual_model": "petri__base", "serving_identity": "proxy_vllm+base",
            "intervention_hash": "0" * 16}
    store = SimpleNamespace(cell_dir=tmp_path)
    out = asyncio.run(isv.run_inspect_tasks(
        atoms=["math500"], route=route, routes=[route, base], system=None, store=store, template={},
        thinking=C.THINKING_ON, base_url="http://x/v1", api_key="k", family="qwen3_5", header_located=False,
        oos_experiment="math500_thinking_on_full22_qwen35_4b"))
    assert built == ["petri__si_x__lora_r4"]                           # the TARGET only — no grader
    [kw] = planned
    assert kw["grader_model"] is None and kw["family"] == "qwen3_5" and kw["require_certified_fact"] is True
    assert kw["oos_experiment"] == "math500_thinking_on_full22_qwen35_4b"
    assert out == {C.inspect_task("math"): {"plan": C.inspect_task("math")}}
    # a run with a model-graded task builds the grader exactly as before
    built.clear()
    monkeypatch.setattr(isv, "plans_for_atoms",
                        lambda atoms, **kw: [SimpleNamespace(row_task=C.inspect_task("xstest"), thinking=C.THINKING_OFF)])
    asyncio.run(isv.run_inspect_tasks(
        atoms=["inspect_cmp:xstest"], route=route, routes=[route, base], system=None, store=store, template={},
        thinking=C.THINKING_OFF, base_url="http://x/v1", api_key="k"))
    assert built == ["petri__si_x__lora_r4", "petri__base"]


# --------------------------------------------------------------------------- #
# fact-fed planning                                                           #
# --------------------------------------------------------------------------- #


def test_plan_task_takes_prompt_opened_think_from_the_family_fact_and_refuses_a_candidate_fact(monkeypatch):
    _stub_math(monkeypatch)
    kw = dict(grader_model=None, thinking=C.THINKING_ON)
    # qwen: certified implicit_open — in support, no parser, no override
    q = isv.plan_task("math", family="qwen3_5", **kw)
    assert q.think_split_mode == "implicit_open" and q.cap_source == "on_table"
    # glm: certified explicit_tags (G-PARSER on the smoke's real completions, 2026-09-10) — planned in production
    glm = isv.plan_task("math", family="glm4v", **kw)
    assert glm.think_split_mode == "explicit_tags" and glm.render_evidence_status == "certified"
    # nemotron: certified implicit_open (G-PARSER on the smoke's real completions, 2026-09-10) — planned in production
    n = isv.plan_task("math", family="nemotron_h", **kw)
    assert n.think_split_mode == "implicit_open" and n.render_evidence_status == "certified"
    # a candidate fact (a SYNTHETIC demotion of Nemotron's — only `evidence_status` moves): production planning
    # refuses by name; the dry run's `pre` reading plans it
    real = dict(C.FAMILY_RENDER_FACTS)
    demoted = real["nemotron_h"]._replace(evidence_status=C.RENDER_EVIDENCE_CANDIDATE)
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", {**real, "nemotron_h": demoted})
    with pytest.raises(isv.InspectServedError, match="render fact not certified"):
        isv.plan_task("math", family="nemotron_h", **kw)
    p = isv.plan_task("math", family="nemotron_h", require_certified_fact=False, **kw)
    assert p.think_split_mode == "implicit_open" and p.render_evidence_status == "candidate"
    monkeypatch.setattr(C, "FAMILY_RENDER_FACTS", real)              # the tree's own facts again
    # gemma with its server parser: record_only, certified
    g = isv.plan_task("math", family="gemma4", reasoning_parser="gemma4", **kw)
    assert g.think_split_mode == "record_only"
    # an explicit prompt_opened_think that CONTRADICTS the fact refuses by name
    with pytest.raises(isv.InspectServedError, match="contradiction"):
        isv.plan_task("math", family="glm4v", prompt_opened_think=True, require_certified_fact=False, **kw)
    # a family with no fact refuses; the parent's rule (no family) is unchanged
    with pytest.raises(isv.InspectServedError, match="no render fact"):
        isv.plan_task("math", family="mistral", **kw)
    # llama-paper-rows: the no-switch family's fact (split `none`, CERTIFIED by the GSM8K diagnostic smoke 2026-09-12) has no
    # thinking-ON render to plan — refused by name whatever the certification; its own arm is `not_available`
    # (tests/test_inspect_served_not_available.py). Before the certification the candidate refusal stood in for this rule.
    with pytest.raises(isv.InspectServedError, match="carries no thinking switch"):
        isv.plan_task("math", family="llama", **kw)
    with pytest.raises(isv.InspectServedError, match="carries no thinking switch"):
        isv.plan_task("math", family="llama", require_certified_fact=False, **kw)
    with pytest.raises(isv.InspectServedError, match="prompt_opened_think"):
        isv.plan_task("math", prompt_opened_think=True, **kw)
    # OFF: record_only whatever the family
    assert isv.plan_task("math", grader_model=None, thinking=C.THINKING_OFF, family="nemotron_h").think_split_mode == "record_only"
    # the matched tuple's cap map feeds the ON cap under the override
    oos = isv.plan_task("math", family="qwen3_5", out_of_support=True,
                        oos_experiment="math500_thinking_on_full22_qwen35_4b", **kw)
    assert (oos.max_tokens, oos.cap_source) == (C.MATH500_MAX_TOKENS_DEFAULT, "oos_override")
    assert isv.task_max_tokens("math", thinking="on", out_of_support=True) == (C.MATH500_MAX_TOKENS_DEFAULT, "on_table")


def test_the_optional_sampling_terms_ride_the_plan_only_when_the_row_sets_them(monkeypatch):
    _stub_math(monkeypatch)
    legacy = isv.plan_task("math", grader_model=None, thinking=C.THINKING_OFF,
                           sampling=C.SAMPLING_BY_MODEL_KEY["gemma4e4b"])
    assert (legacy.min_p, legacy.presence_penalty, legacy.repetition_penalty) == (None, None, None)
    assert isv._optional_terms(legacy) == {}
    general = isv.plan_task("math", grader_model=None, thinking=C.THINKING_OFF,
                            sampling=C.GENERAL_SAMPLING_BY_MODEL_KEY["qwen35_4b"])
    assert isv._optional_terms(general) == {"min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
    assert isv._ref_extra(None) == {} and isv._ref_extra("abc") == {C.SAMPLING_REF_FIELD: "abc"}


# --------------------------------------------------------------------------- #
# split modes on the committed fixtures, incl. the truncated Nemotron row      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family", ("glm4v", "nemotron_h", "qwen3_5"))
def test_the_d11_fixtures_split_under_the_family_mode_and_a_truncated_row_yields_no_score(family):
    doc = _fixture(family)
    fact = C.FAMILY_RENDER_FACTS[family]
    assert doc["split_mode"] == fact.split_mode
    scorer = isv._math_scorer(fact.split_mode)
    for case in doc["cases"]:
        split = ts.split_think(case["completion"], mode=fact.split_mode)
        assert (split.present, split.unterminated, split.answer_text) == (
            case["present"], case["unterminated"], case["answer_text"]), (family, case["name"])
        score = _score(scorer, _state(_output(case["completion"],
                                              stop_reason="max_tokens" if case["finish_reason"] == "length" else "stop")))
        if case["scoreable"]:
            assert not _is_unscored(score) and isv.score_to_value(score) == 1.0, (family, case["name"])
        else:
            # a truncated row yields NO score — the trap an opening-tag-keyed
            # splitter would fall into is 'ANSWER: 6' inside the reasoning
            assert _is_unscored(score) and score.answer is None, (family, case["name"])
        assert score.metadata["think_block_present"] is case["present"]
        assert score.metadata["think_block_unterminated"] is case["unterminated"]
        assert score.metadata[ts.SCORER_WRAPPER_FIELD] == f"think_aware/{fact.split_mode}"
    if family == "nemotron_h":
        trunc = next(c for c in doc["cases"] if c["name"] == "truncated_no_closing_tag")
        assert "ANSWER: 6" in trunc["completion"]                     # the trap is IN the fixture
        # the wrong (explicit_tags) rule would read the chain-of-thought as the answer
        wrong = ts.split_think(trunc["completion"], mode=C.THINK_SPLIT_EXPLICIT_TAGS)
        assert wrong.present is False and wrong.answer_text == trunc["completion"]
    # the sensitivity branch scores an unterminated row INCORRECT, never unscored
    sensitivity = ts.think_aware_scorer(isv._math_scorer(fact.split_mode), mode=fact.split_mode,
                                        unterminated_policy=C.UNTERMINATED_INCORRECT, name="expression_exact_match_sympy")
    trunc = next(c for c in doc["cases"] if c["unterminated"])
    s = _score(sensitivity, _state(_output(trunc["completion"], stop_reason="max_tokens")))
    assert isv.score_to_value(s) == 0.0 and not _is_unscored(s)


def test_gemma_channel_aware_censoring_on_actual_model_output_channels_three_cases():
    """Under `record_only` (the server parser separated the channels):
    (a) finished -> scored; (b) reasoning-only at the cap -> UNSCORED under the
    primary policy / INCORRECT under the sensitivity branch, `present` and
    `unterminated` True, the censored flag set — never an empty answer scored
    wrong in both; (c) reasoning + a partial answer at the cap -> scored as-is."""
    doc = _fixture("gemma4")
    assert doc["split_mode"] == "record_only" == C.FAMILY_RENDER_FACTS["gemma4"].split_mode
    primary = isv._math_scorer(C.THINK_SPLIT_RECORD_ONLY)
    sensitivity = ts.think_aware_scorer(primary, mode=C.THINK_SPLIT_RECORD_ONLY,
                                        unterminated_policy=C.UNTERMINATED_INCORRECT,
                                        name="expression_exact_match_sympy")
    cases = {c["name"]: c for c in doc["cases"]}
    a = cases["finished"]
    s = _score(primary, _state(_output(a["content"], reasoning=a["reasoning"], stop_reason=a["stop_reason"])))
    assert isv.score_to_value(s) == 1.0 and s.metadata["think_block_present"] is True
    assert s.metadata["think_block_unterminated"] is False and s.metadata[ts.CENSORED_REASONING_ONLY_FIELD] is False
    b = cases["reasoning_only_at_cap"]
    out_b = _output(b["content"], reasoning=b["reasoning"], stop_reason=b["stop_reason"])
    assert out_b.completion == "" and out_b.stop_reason == "max_tokens"
    s = _score(primary, _state(out_b))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_present"] is True and s.metadata["think_block_unterminated"] is True
    assert s.metadata[ts.CENSORED_REASONING_ONLY_FIELD] is True
    s2 = _score(sensitivity, _state(out_b))
    assert isv.score_to_value(s2) == 0.0 and not _is_unscored(s2) and s2.metadata[ts.CENSORED_REASONING_ONLY_FIELD] is True
    c = cases["answer_truncated_at_cap"]
    s = _score(primary, _state(_output(c["content"], reasoning=c["reasoning"], stop_reason=c["stop_reason"])))
    assert not _is_unscored(s) and s.metadata["think_block_unterminated"] is False    # case (c): scored, at-cap
    # an EMPTY content with NO reasoning is an ordinary (wrong) empty answer, not a censored block
    s = _score(primary, _state(_output("", stop_reason="stop")))
    assert not _is_unscored(s) and isv.score_to_value(s) == 0.0 and s.metadata["think_block_present"] is False
    # the two channels are read off the SAME ModelOutput the adapter reads
    content, reasoning = isv.output_channels(out_b)
    assert (content, reasoning) == ("", b["reasoning"])


def test_every_family_fixture_names_its_split_mode_and_the_five_d11_cases_are_present():
    names = {fam: {c["name"] for c in _fixture(fam)["cases"]} for fam in FAMILIES}
    assert {"well_formed", "unterminated", "tag_free_is_the_answer"} <= names["glm4v"]
    assert {"well_formed", "truncated_no_closing_tag"} <= names["nemotron_h"]
    assert {"well_formed", "truncated_no_closing_tag"} <= names["qwen3_5"]
    assert {"finished", "reasoning_only_at_cap", "answer_truncated_at_cap"} <= names["gemma4"]
    for fam in FAMILIES:
        assert _fixture(fam)["split_mode"] == C.FAMILY_RENDER_FACTS[fam].split_mode


# --------------------------------------------------------------------------- #
# the MATH prompt renderer and the single-pass window pre-flight (§4.3 7/8)     #
# --------------------------------------------------------------------------- #


def test_task_messages_renders_math500_through_the_tasks_own_solver_and_refuses_another_task(monkeypatch):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_evals.math.math import math_solver

    ids = [C.math500_sample_id(f"problem {i}") for i in range(500)]
    task = Task(dataset=[Sample(input=f"Compute problem {i}.", target="1", id=i_) for i, i_ in enumerate(ids)],
                solver=math_solver(fewshot=0, fewshot_seed=42), scorer=[])
    plan = isv.TaskPlan(row_task=C.inspect_task("math"), name="math", task=task, sample_ids=ids, expected_n=500)
    seen = {}

    def fake_plan_task(name, **kw):
        seen.update(kw, name=name)
        return plan

    monkeypatch.setattr(isv, "plan_task", fake_plan_task)
    msgs = isv.task_messages("math")
    assert seen["name"] == "math" and seen["thinking"] == C.THINKING_OFF and len(msgs) == 500
    assert msgs[0][-1]["role"] == "user" and "Compute problem 0." in msgs[0][-1]["content"]
    assert 'ANSWER: $ANSWER' in msgs[0][-1]["content"]                # the task's OWN template
    assert all(m["role"] != "system" for m in msgs[0])
    # a PROMPTED cell: the SI lands as ONE leading system turn
    prompted = isv.task_messages("math", system="Be careful.")
    assert prompted[3][0] == {"role": "system", "content": "Be careful."} and prompted[3][1:] == msgs[3]
    assert isv.collapse_system_messages_dicts([{"role": "system", "content": "a"}, {"role": "user", "content": "u"}], "SI") == [
        {"role": "system", "content": "SI\n\na"}, {"role": "user", "content": "u"}]
    # the parent's renderer is still the GSM8K one; a task with none refuses by name
    # ifeval-canonical-battery: the IFEval renderer joined the table on this branch (`ifeval_messages`, the
    # container-written prompt budget's render — follow-up F7); the pin follows the registered set
    assert set(isv.TASK_MESSAGES_RENDERERS) == {"gsm8k_paired200", "math", C.IFEVAL_TASK_NAME}
    with pytest.raises(isv.InspectServedError, match="no prompt renderer"):
        isv.task_messages("bfcl")
    # a plan that is not 500 items refuses
    short = isv.TaskPlan(row_task=C.inspect_task("math"), name="math", task=task, sample_ids=ids[:3], expected_n=3)
    monkeypatch.setattr(isv, "plan_task", lambda name, **kw: short)
    with pytest.raises(isv.InspectServedError, match="not 500"):
        isv.math500_messages()


def test_assert_budget_fits_window_is_the_pre_flight_over_the_artifact():
    rec = isv.assert_budget_fits_window("inspect/math", {"max": 560, "n": 500}, cap=8192, max_model_len=10752)
    assert rec == {"max": 560, "cap": 8192, "window": 10752, "margin": 2000}
    with pytest.raises(isv.InspectServedError, match="over by 8"):
        isv.assert_budget_fits_window("inspect/math", {"max": 2568}, cap=8192, max_model_len=10752)
    with pytest.raises(isv.InspectServedError, match="window .* is unknown"):
        isv.assert_budget_fits_window("inspect/math", {"max": 560}, cap=8192, max_model_len=None)
    with pytest.raises(isv.InspectServedError, match="no integer 'max'"):
        isv.assert_budget_fits_window("inspect/math", {"n": 500}, cap=8192, max_model_len=10752)


# --------------------------------------------------------------------------- #
# the provider: header_located and the optional sampling terms                 #
# --------------------------------------------------------------------------- #


def _provider(**kw):
    from inspect_ai.model import GenerateConfig

    cls = isv._register_provider()
    route = {"cell_id": "si_constv2_mix_full__lora_r4_ce_minus_ctl",
             "virtual_model": "petri__si_constv2_mix_full__lora_r4_ce_minus_ctl",
             "serving_identity": "proxy_vllm+lens:v1", "intervention_hash": "0" * 16}
    return cls(model_name=C.inspect_model_name(route["virtual_model"]), base_url="http://127.0.0.1:1/v1",
               api_key="synthetic", config=GenerateConfig(), si=None, route=route, **kw)


def test_the_provider_sends_no_declaration_on_an_undeclared_all_position_lens_route(monkeypatch):
    from inspect_ai.model import GenerateConfig

    monkeypatch.setattr(isv.pe, "is_lens_route", lambda route: True)
    tok = isv._TASK.set(C.inspect_task("math"))
    try:
        declared = _provider(thinking=C.THINKING_ON).tinycap_config(GenerateConfig())
        assert declared.extra_body[C.OOS_THINKING_DECLARATION_KEY] is True           # the default: header-located
        undeclared = _provider(thinking=C.THINKING_ON, header_located=False).tinycap_config(GenerateConfig())
        assert C.OOS_THINKING_DECLARATION_KEY not in undeclared.extra_body
        assert undeclared.extra_body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: True}
        off = _provider(thinking=C.THINKING_OFF, header_located=False).tinycap_config(GenerateConfig())
        assert C.OOS_THINKING_DECLARATION_KEY not in off.extra_body
        assert off.extra_body[C.CHAT_TEMPLATE_KWARGS_KEY] == {C.ENABLE_THINKING_KEY: False}
    finally:
        isv._TASK.reset(tok)


def test_the_provider_applies_the_optional_terms_only_when_the_row_sets_them(monkeypatch):
    from inspect_ai.model import GenerateConfig

    monkeypatch.setattr(isv.pe, "is_lens_route", lambda route: False)
    general = _provider(thinking=C.THINKING_ON, sampling=C.GENERAL_SAMPLING_BY_MODEL_KEY["qwen35_4b"])
    cfg = general.tinycap_config(GenerateConfig())
    assert (cfg.temperature, cfg.top_p, cfg.presence_penalty) == (1.0, 0.95, 1.5)
    assert cfg.extra_body["top_k"] == 20 and cfg.extra_body["min_p"] == 0.0 and cfg.extra_body["repetition_penalty"] == 1.0
    legacy = _provider(thinking=C.THINKING_OFF, sampling=C.SAMPLING_BY_MODEL_KEY["gemma4e4b"])
    cfg = legacy.tinycap_config(GenerateConfig())
    assert cfg.presence_penalty is None and "min_p" not in cfg.extra_body and "repetition_penalty" not in cfg.extra_body
    with pytest.raises(isv.InspectServedError, match="presence_penalty"):
        general.tinycap_config(GenerateConfig(presence_penalty=0.0))
    with pytest.raises(isv.InspectServedError, match="min_p"):
        general.tinycap_config(GenerateConfig(extra_body={"min_p": 0.5}))
