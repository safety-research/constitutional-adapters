"""Under the implicit-open render a completion that STOPPED (not at the cap) without ever emitting
`</think>` is a response with NO think block — the model reasoned inline and ended its turn — and its final `ANSWER: <n>` line parses.
think_aware_scorer filed such draws unscored (NaN, answer None) as "unterminated blocks": 50 of 200 on the LeCun constitution ReLU cell
(889562), whose filed acc 0.96 was over the 150 scored while the row said n=200. Now: scored by the plain final-answer
parse (never NaN on a parseable answer), metadata `no_think_block: True`, counted as `n_no_think_block` beside the block counts; a draw cut
off AT THE CAP without `</think>` stays unterminated -> unscored (unchanged); a well-formed block is split as before.

The rule is an OPT-IN of the wrap (`no_think_block_when_stopped`, default off; `_gsm8k_scorer`
alone passes True, so MATH-500's wrap is unchanged); it admits the NATURAL stop only (`stop`; `model_length` and `unknown`
stay unterminated -> unscored); and it reads the output channels before claiming no block: a client-LIFTED complete block
(ContentReasoning + answer text) is a PRESENT closed block, `lifted_close: True`, never `no_think_block`."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

pytest.importorskip("inspect_ai")                 # inspect_ai only: the thinking-on module's helpers need inspect_evals, which the
from sisteer.capeval.tiers import inspect_served as isv  # noqa: E402   # lane venv lacks (that module SKIPS there) — so they are inlined
from sisteer.capeval.tiers import think_split as ts  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402


# --- the thinking-on module's helpers, inlined (tests/test_inspect_served_thinking_on.py `_output` / `_state` / `_score` / `_is_unscored`) ---
import asyncio
import math


def _output(content: str, *, reasoning=None, stop_reason: str = "stop"):
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
    return TaskState(model=ModelName("tinycap/x"), sample_id="gsm8k_00000001", epoch=1, input="q", messages=[], target=Target(target), output=output)


def _score(scorer, state):
    from inspect_ai.scorer import Target
    return asyncio.run(scorer(state, Target(state.target.text)))


def _is_unscored(score) -> bool:
    v = score.value
    return isinstance(v, float) and math.isnan(v)

MODE = C.THINK_SPLIT_IMPLICIT_OPEN
INLINE = "We add 17 and 26. 17 + 26 = 43.\nANSWER: 43"                       # no `</think>` anywhere: the model reasoned inline and stopped
BLOCK = "17 + 26 = 43, so the answer is 43.</think>\nANSWER: 43"             # the prompt-opened block closed, the answer after it


def _scorer():
    return isv._gsm8k_scorer(MODE)


def test_a_stopped_completion_without_a_think_block_is_scored_by_its_final_answer_and_flagged():
    s = _score(_scorer(), _state(_output(INLINE, stop_reason="stop"), target="43"))
    assert not _is_unscored(s) and isv.score_to_value(s) == 1.0, s
    assert s.metadata[ts.NO_THINK_BLOCK_FIELD] is True
    assert s.metadata["think_block_present"] is False and s.metadata["think_block_unterminated"] is False
    assert s.metadata[ts.SCORER_WRAPPER_FIELD] == f"think_aware/{MODE}" and s.metadata["think_split_mode"] == MODE
    # a wrong inline answer is INCORRECT (0.0), never NaN
    wrong = _score(_scorer(), _state(_output("17 + 26 = 44.\nANSWER: 44", stop_reason="stop"), target="43"))
    assert not _is_unscored(wrong) and isv.score_to_value(wrong) == 0.0 and wrong.metadata[ts.NO_THINK_BLOCK_FIELD] is True


def test_an_at_cap_draw_without_a_closing_tag_stays_unterminated_and_unscored_and_a_closed_block_splits_as_before():
    cut = _score(_scorer(), _state(_output(INLINE, stop_reason="max_tokens"), target="43"))      # Cut off inside its thoughts
    assert _is_unscored(cut) and cut.answer is None
    assert cut.metadata["think_block_present"] is True and cut.metadata["think_block_unterminated"] is True
    assert ts.NO_THINK_BLOCK_FIELD not in cut.metadata                                            # absent, not False: other rows are byte-identical
    ok = _score(_scorer(), _state(_output(BLOCK, stop_reason="stop"), target="43"))
    assert not _is_unscored(ok) and isv.score_to_value(ok) == 1.0
    assert ok.metadata["think_block_present"] is True and ok.metadata["think_block_unterminated"] is False and ts.NO_THINK_BLOCK_FIELD not in ok.metadata
    # the pure splitter is unchanged: the rule lives in the scorer, keyed on the stop reason the splitter does not see
    sp = ts.split_think(INLINE, mode=MODE)
    assert sp.present is True and sp.unterminated is True and sp.answer_text == ""


def test_the_other_modes_and_the_sensitivity_policy_are_untouched():
    # explicit_tags without a tag already scores the content; record_only delegates; none scores byte-identically
    for mode in (C.THINK_SPLIT_EXPLICIT_TAGS, C.THINK_SPLIT_RECORD_ONLY, C.THINK_SPLIT_NONE):
        s = _score(isv._gsm8k_scorer(mode), _state(_output(INLINE, stop_reason="stop"), target="43"))
        assert not _is_unscored(s) and isv.score_to_value(s) == 1.0 and ts.NO_THINK_BLOCK_FIELD not in s.metadata, mode
    # the sensitivity branch (unterminated -> INCORRECT) applies to the at-cap draw only; the stopped inline draw scores on its answer either way
    # — under the GSM8K wrap's opt-in; a sensitivity wrap WITHOUT it files the stopped draw INCORRECT as it did before the fix
    from inspect_ai.scorer import match
    sens = ts.think_aware_scorer(match(numeric=True), mode=MODE, unterminated_policy=C.UNTERMINATED_INCORRECT, no_think_block_when_stopped=True)
    assert isv.score_to_value(_score(sens, _state(_output(INLINE, stop_reason="max_tokens"), target="43"))) == 0.0
    assert isv.score_to_value(_score(sens, _state(_output(INLINE, stop_reason="stop"), target="43"))) == 1.0
    sens_off = ts.think_aware_scorer(match(numeric=True), mode=MODE, unterminated_policy=C.UNTERMINATED_INCORRECT)
    off = _score(sens_off, _state(_output(INLINE, stop_reason="stop"), target="43"))
    assert isv.score_to_value(off) == 0.0 and ts.NO_THINK_BLOCK_FIELD not in off.metadata


def test_the_summary_counts_no_think_block_draws_and_the_contracts_name_the_field():
    def r(no_think, unterminated=False, at_cap=False, tokens=100):
        return SimpleNamespace(think_block_present=not no_think, think_block_unterminated=unterminated, at_cap=at_cap,
                               completion_tokens=tokens, no_think_block=no_think, stray_open_tag=False, unscored=False)
    results = [r(True), r(True), r(False), r(False, unterminated=True, at_cap=True)]
    counts = isv.think_split_counts(results, plan=SimpleNamespace(think_split_mode=MODE))
    assert counts["n_no_think_block"] == 2 and counts["n_think_block_present"] == 2 and counts["n_think_block_unterminated"] == 1 and counts["n_unterminated_not_at_cap"] == 0
    assert "n_no_think_block" in C.SUMMARY_PAYLOAD_FIELDS and "n_no_think_block" not in C.TASK_OUTCOMES_HEADER_FIELDS   # additive: summary.json only; the outcomes header is compared against old stores
    assert "n_no_think_block" not in C.INSPECT_EXTRA_FIELDS                                       # the row shape is pinned; the fact rides the summary
    assert isv.ItemResult.__dataclass_fields__["no_think_block"].default is False


# --- the opt-in rule: the wrap's option, the natural stop, and the lifted close ---------------------------------------------------------


def test_the_rule_is_opted_into_by_the_gsm8k_wrap_alone_and_a_default_wrap_keeps_a_stopped_unclosed_draw_unscored():
    """`no_think_block_when_stopped` is a scorer OPTION, default off. A wrap that does not ask for it (any other
    implicit_open scorer — MATH-500's, pinned under the client venv in tests/test_inspect_served_thinking_on.py) files a stopped
    unclosed draw unterminated -> unscored exactly as it did before the fix; the GSM8K wrap is the one caller that opts in, and the
    lifted-close reading is part of the same opt-in."""
    from inspect_ai.scorer import match
    default = ts.think_aware_scorer(match(numeric=True), mode=MODE)
    s = _score(default, _state(_output(INLINE, stop_reason="stop"), target="43"))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_present"] is True and s.metadata["think_block_unterminated"] is True
    assert ts.NO_THINK_BLOCK_FIELD not in s.metadata and ts.LIFTED_CLOSE_FIELD not in s.metadata
    explicit_off = ts.think_aware_scorer(match(numeric=True), mode=MODE, no_think_block_when_stopped=False)
    assert _is_unscored(_score(explicit_off, _state(_output(INLINE, stop_reason="stop"), target="43")))
    lifted_off = _score(default, _state(_output("ANSWER: 43", reasoning="17 + 26 = 43.", stop_reason="stop"), target="43"))
    assert _is_unscored(lifted_off) and ts.LIFTED_CLOSE_FIELD not in lifted_off.metadata and lifted_off.metadata["think_block_unterminated"] is True
    # the GSM8K wrap opts in
    assert isv.score_to_value(_score(_scorer(), _state(_output(INLINE, stop_reason="stop"), target="43"))) == 1.0
    with pytest.raises(ValueError, match="no_think_block_when_stopped"):
        ts.think_aware_scorer(match(numeric=True), mode=MODE, no_think_block_when_stopped="yes")


def test_a_context_overflow_without_a_close_stays_unterminated_and_unscored():
    """Finding (2): `model_length` is inspect_ai's CONTEXT-OVERFLOW stop reason — not a natural stop (inspect_served.STOP_REASON_AT_CAP's
    note; the rows test in tests/test_inspect_served.py tells it from `max_tokens`). Its text is an overflow's, not an inline answer:
    unterminated -> unscored under the primary policy, never INCORRECT-and-flagged; INCORRECT under the sensitivity branch, still
    unflagged."""
    from inspect_ai.scorer import match
    s = _score(_scorer(), _state(_output(INLINE, stop_reason="model_length"), target="43"))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_present"] is True and s.metadata["think_block_unterminated"] is True
    assert ts.NO_THINK_BLOCK_FIELD not in s.metadata and ts.LIFTED_CLOSE_FIELD not in s.metadata
    sens = ts.think_aware_scorer(match(numeric=True), mode=MODE, unterminated_policy=C.UNTERMINATED_INCORRECT,
                                 no_think_block_when_stopped=True)
    w = _score(sens, _state(_output(INLINE, stop_reason="model_length"), target="43"))
    assert not _is_unscored(w) and isv.score_to_value(w) == 0.0 and ts.NO_THINK_BLOCK_FIELD not in w.metadata
    assert isv.STOP_REASON_AT_CAP == "max_tokens" and ts._STOPPED == "stop"


def test_a_missing_finish_reason_without_a_close_stays_unterminated_and_unscored():
    """Finding (2): `unknown` is what inspect_ai's `as_stop_reason` files for a finish reason it does not recognise, or none at all —
    whether the model ended its turn is NOT known, so it is not admitted as a natural stop: unterminated -> unscored, unflagged."""
    s = _score(_scorer(), _state(_output(INLINE, stop_reason="unknown"), target="43"))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_present"] is True and s.metadata["think_block_unterminated"] is True
    assert ts.NO_THINK_BLOCK_FIELD not in s.metadata and ts.LIFTED_CLOSE_FIELD not in s.metadata
    # and the natural stop alone admits it — the same text, the same wrap
    assert isv.score_to_value(_score(_scorer(), _state(_output(INLINE, stop_reason="stop"), target="43"))) == 1.0


def test_a_client_lifted_close_is_a_present_block_through_the_reasoning_channel_and_not_a_no_think_draw():
    """Finding (3): inspect_ai 0.3.260's client-side lift (`parse_content_with_reasoning`) moves a complete `<think>…</think>` out of
    the assistant text into a ContentReasoning part (the MATH reader's documented `lifted_close` class), so a draw whose model closed
    its block and answered reaches the scorer with no `</think>` in its text. The opted-in wrap reads the channels BEFORE claiming no
    block: a non-empty reasoning channel is a PRESENT closed block (`lifted_close: True`), scored on its text, never `no_think_block` —
    and through the row reader it counts as present, not as no-think, so `n_no_think_block` is not over-counted."""
    out = _output("ANSWER: 43", reasoning="17 + 26 = 43, so the answer is 43.", stop_reason="stop")
    assert ts.output_channels(out) == ("ANSWER: 43", "17 + 26 = 43, so the answer is 43.")
    assert ts.split_think(out.completion, mode=MODE).unterminated is True                        # the TEXT alone reads as unclosed
    s = _score(_scorer(), _state(out, target="43"))
    assert not _is_unscored(s) and isv.score_to_value(s) == 1.0
    assert s.metadata["think_block_present"] is True and s.metadata["think_block_unterminated"] is False
    assert s.metadata[ts.LIFTED_CLOSE_FIELD] is True and ts.NO_THINK_BLOCK_FIELD not in s.metadata
    w = _score(_scorer(), _state(_output("ANSWER: 44", reasoning="17 + 26 = 44.", stop_reason="stop"), target="43"))
    assert not _is_unscored(w) and isv.score_to_value(w) == 0.0 and w.metadata[ts.LIFTED_CLOSE_FIELD] is True
    assert ts.NO_THINK_BLOCK_FIELD not in w.metadata
    # a lifted block whose answer was cut AT THE CAP is the at-cap case still: unterminated -> unscored, neither flag
    cut = _score(_scorer(), _state(_output("ANSWER:", reasoning="17 + 26 = 43.", stop_reason="max_tokens"), target="43"))
    assert _is_unscored(cut) and ts.LIFTED_CLOSE_FIELD not in cut.metadata and ts.NO_THINK_BLOCK_FIELD not in cut.metadata
    # through the row reader and the summary counts: the lifted close is a present block, the inline answer the no-think draw
    inline_out = _output(INLINE, stop_reason="stop")
    inline = _score(_scorer(), _state(inline_out, target="43"))
    log = SimpleNamespace(status="success", error=None, samples=[
        SimpleNamespace(id="gsm8k_0000000a", scores={ts.WRAPPER_NAME: s}, error=None, model_usage={}, output=out),
        SimpleNamespace(id="gsm8k_0000000b", scores={ts.WRAPPER_NAME: inline}, error=None, model_usage={}, output=inline_out)])
    rows = {r.item_id: r for r in isv.results_from_log(log, primary_scorer=ts.WRAPPER_NAME)}
    lifted, no_block = rows["gsm8k_0000000a"], rows["gsm8k_0000000b"]
    assert lifted.value == 1.0 and lifted.think_block_present and not lifted.think_block_unterminated and not lifted.no_think_block
    assert lifted.reasoning == "17 + 26 = 43, so the answer is 43." and lifted.content == "ANSWER: 43"
    assert no_block.value == 1.0 and not no_block.think_block_present and no_block.no_think_block and no_block.reasoning is None
    counts = isv.think_split_counts(list(rows.values()), plan=SimpleNamespace(think_split_mode=MODE))
    assert (counts["n_no_think_block"], counts["n_think_block_present"], counts["n_think_block_unterminated"]) == (1, 1, 0)
    assert "n_lifted_close" not in C.SUMMARY_PAYLOAD_FIELDS and ts.LIFTED_CLOSE_FIELD not in C.INSPECT_EXTRA_FIELDS   # row metadata only


# --- G8KSCORERFIX3: a lifted pair beside a RE-OPENED block is unterminated, not a lifted close -----------------------------------
REOPENED_TEXT = "<think>still reasoning\nANSWER: 43"       # what inspect_ai's parser leaves of `<think>example</think>\n<think>still reasoning\nANSWER: 43`


def test_a_lifted_pair_beside_a_reopened_unclosed_block_stays_unterminated_and_unscored_and_is_counted_not_at_cap():
    """The reasoning channel proves inspect_ai lifted A complete pair, not that the remaining text has no
    open block. The text the parser leaves behind carries an unclosed `<think>`: split_think reads unterminated True / stray_open_tag
    True, and the opted-in wrap must NOT read the reasoning channel as a closed block — the draw is unterminated (unscored;
    INCORRECT under the sensitivity branch), with neither `lifted_close` nor `no_think_block`, and the row reader / summary count it in
    n_unterminated_not_at_cap, not in n_no_think_block. (The REAL parser is exercised under the client venv in
    tests/test_inspect_served_thinking_on.py; this is the same content-list shape built by hand.)"""
    from inspect_ai.scorer import match
    out = _output(REOPENED_TEXT, reasoning="example", stop_reason="stop")
    sp = ts.split_think(out.completion, mode=MODE)
    assert (sp.present, sp.unterminated, sp.stray_open_tag, sp.answer_text) == (True, True, True, "")
    s = _score(_scorer(), _state(out, target="43"))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_present"] is sp.present and s.metadata["think_block_unterminated"] is True and s.metadata["stray_open_tag"] is True
    assert ts.LIFTED_CLOSE_FIELD not in s.metadata and ts.NO_THINK_BLOCK_FIELD not in s.metadata
    sens = ts.think_aware_scorer(match(numeric=True), mode=MODE, unterminated_policy=C.UNTERMINATED_INCORRECT, no_think_block_when_stopped=True)
    w = _score(sens, _state(out, target="43"))
    assert not _is_unscored(w) and isv.score_to_value(w) == 0.0 and ts.LIFTED_CLOSE_FIELD not in w.metadata and ts.NO_THINK_BLOCK_FIELD not in w.metadata
    # the default (non-opt-in) wrap reads it the same way, as before
    assert _is_unscored(_score(ts.think_aware_scorer(match(numeric=True), mode=MODE), _state(out, target="43")))
    # the positive lifted close (a clean remainder) still scores 1.0 with lifted_close; the inline no-block draw still scores and is flagged
    clean_out = _output("ANSWER: 43", reasoning="17 + 26 = 43.", stop_reason="stop")
    clean = _score(_scorer(), _state(clean_out, target="43"))
    assert isv.score_to_value(clean) == 1.0 and clean.metadata[ts.LIFTED_CLOSE_FIELD] is True and clean.metadata["think_block_unterminated"] is False
    inline_out = _output(INLINE, stop_reason="stop")
    inline = _score(_scorer(), _state(inline_out, target="43"))
    assert isv.score_to_value(inline) == 1.0 and inline.metadata[ts.NO_THINK_BLOCK_FIELD] is True
    # through the row reader and the summary: the reopened draw is unterminated-not-at-cap; the other two are what they were
    log = SimpleNamespace(status="success", error=None, samples=[
        SimpleNamespace(id="gsm8k_0000000a", scores={ts.WRAPPER_NAME: s}, error=None, model_usage={}, output=out),
        SimpleNamespace(id="gsm8k_0000000b", scores={ts.WRAPPER_NAME: clean}, error=None, model_usage={}, output=clean_out),
        SimpleNamespace(id="gsm8k_0000000c", scores={ts.WRAPPER_NAME: inline}, error=None, model_usage={}, output=inline_out)])
    rows = {r.item_id: r for r in isv.results_from_log(log, primary_scorer=ts.WRAPPER_NAME)}
    reopened = rows["gsm8k_0000000a"]
    assert reopened.unscored and reopened.think_block_present and reopened.think_block_unterminated and reopened.stray_open_tag
    assert not reopened.no_think_block and not reopened.at_cap and reopened.stop_reason == "stop" and reopened.reasoning == "example"
    counts = isv.think_split_counts(list(rows.values()), plan=SimpleNamespace(think_split_mode=MODE))
    assert (counts["n_unterminated_not_at_cap"], counts["n_think_block_unterminated"], counts["n_no_think_block"], counts["n_think_block_present"]) == (1, 1, 1, 2)
    # the pure splitter is unchanged: the rule lives in the scorer
    assert ts.split_think("ANSWER: 43", mode=MODE).stray_open_tag is False


def test_a_reopened_unclosed_block_with_no_lift_stays_unterminated_and_unscored_and_is_not_a_no_think_draw():
    """The same guard on the NO-reasoning branch. A stopped draw whose text is `<think>still reasoning\nANSWER: 43` with NO
    lift (no ContentReasoning: the parser lifts nothing without a `</think>`) is a block the model opened and never closed — the
    no-think branch used to score it as `no_think_block`. Now: unterminated (unscored; INCORRECT under the sensitivity
    branch), stray_open_tag True, neither flag, counted by n_unterminated_not_at_cap. The inline no-block draw (`ANSWER: 43`, no
    `<think>` anywhere) still scores 1.0 with `no_think_block`."""
    from inspect_ai.scorer import match
    out = _output(REOPENED_TEXT, stop_reason="stop")                                                # no reasoning part at all
    assert ts.output_channels(out) == (REOPENED_TEXT, None)
    s = _score(_scorer(), _state(out, target="43"))
    assert _is_unscored(s) and s.answer is None
    assert s.metadata["think_block_present"] is True and s.metadata["think_block_unterminated"] is True and s.metadata["stray_open_tag"] is True
    assert ts.LIFTED_CLOSE_FIELD not in s.metadata and ts.NO_THINK_BLOCK_FIELD not in s.metadata
    sens = ts.think_aware_scorer(match(numeric=True), mode=MODE, unterminated_policy=C.UNTERMINATED_INCORRECT, no_think_block_when_stopped=True)
    w = _score(sens, _state(out, target="43"))
    assert not _is_unscored(w) and isv.score_to_value(w) == 0.0 and ts.NO_THINK_BLOCK_FIELD not in w.metadata and ts.LIFTED_CLOSE_FIELD not in w.metadata
    # the inline no-block draw — no `<think>` anywhere — still scores and is flagged
    inline_out = _output("ANSWER: 43", stop_reason="stop")
    inline = _score(_scorer(), _state(inline_out, target="43"))
    assert isv.score_to_value(inline) == 1.0 and inline.metadata[ts.NO_THINK_BLOCK_FIELD] is True and inline.metadata["stray_open_tag"] is False
    # row reader + counts
    log = SimpleNamespace(status="success", error=None, samples=[
        SimpleNamespace(id="gsm8k_0000000a", scores={ts.WRAPPER_NAME: s}, error=None, model_usage={}, output=out),
        SimpleNamespace(id="gsm8k_0000000b", scores={ts.WRAPPER_NAME: inline}, error=None, model_usage={}, output=inline_out)])
    rows = {r.item_id: r for r in isv.results_from_log(log, primary_scorer=ts.WRAPPER_NAME)}
    assert rows["gsm8k_0000000a"].unscored and rows["gsm8k_0000000a"].think_block_unterminated and rows["gsm8k_0000000a"].stray_open_tag
    assert not rows["gsm8k_0000000a"].no_think_block and not rows["gsm8k_0000000a"].at_cap
    assert rows["gsm8k_0000000b"].no_think_block and rows["gsm8k_0000000b"].value == 1.0
    counts = isv.think_split_counts(list(rows.values()), plan=SimpleNamespace(think_split_mode=MODE))
    assert (counts["n_unterminated_not_at_cap"], counts["n_no_think_block"], counts["n_think_block_present"]) == (1, 1, 1)
