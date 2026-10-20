"""The MATH answer-DELIMITER normalization the MATH-500 wrap applies before
inspect_evals' extractor reads the answer — GLM's ``<|begin_of_box|>`` markers dropped (content kept), a
``\\text{ANSWER: }`` label made plain, ``\\( x \\)`` / ``\\[ x \\]`` rewritten to ``$x$``. Measured on the GLM
diagnostic smoke: 4 of 11 finished thinking-ON answers used ``\\( \\)``, 2 the box, and every one scored INCORRECT
against its correct target through the real sympy scorer, which reads ``$…$`` only (Lewkowycz et al. appendix D).
The GSM8K wrap names no normalization and stays byte-identical."""
from __future__ import annotations

import asyncio

import pytest

from sisteer.capeval.tiers import think_split as ts
from sisteer.capeval.tinycap import contracts as C

#: (the model's answer line, the target, scores CORRECT through the real extractor after the normalization?)
CASES = [
    ("ANSWER: $42$", "42", True),                                                 # the extractor's own form: untouched
    ("ANSWER: \\( 5 \\)", "5", True),                                           # GLM smoke math_403b3454
    ("ANSWER: \\(-50\\)", "-50", True),                                          # GLM smoke math_d959374f
    ("ANSWER: \\(<|begin_of_box|>3\\sqrt{13}<|end_of_box|>\\)", "3\\sqrt{13}", True),   # GLM smoke math_67c8ff7f
    ("<|begin_of_box|>ANSWER: \\(\\sqrt{51}\\)<|end_of_box|>", "\\sqrt{51}", True),      # GLM smoke math_d5de6a08
    ("\\[ \\text{ANSWER: } \\frac{14}{3} \\]", "\\frac{14}{3}", True),                # GLM OFF-arm habit
    ("**ANSWER: 9**", "9", True),
    ("ANSWER: 5", "5", True),
    ("ANSWER: \\(5\\) inches", "5", True),
    ("ANSWER: \\(5\\)", "6", False),                                             # a WRONG answer stays wrong
    # the reviewer's cases: several delimited terms on one line are ONE expression — the delimiters go, the terms stay
    ("ANSWER: \\(2\\) + \\(3\\)", "5", True),
    ("ANSWER: \\(2\\) + \\(3\\)", "2", False),
    ("ANSWER: 1 + \\(2\\)", "3", True),
    ("ANSWER: 1 + \\(2\\)", "2", False),
    ("ANSWER: \\(2\\), \\(3\\)", "2", False),
    ("ANSWER: \\(x^2\\) - \\(1\\)", "x^2 - 1", True),
    # the reviewer's second case: literal dollar signs beside delimiters — the line is left as written, so it scores
    # exactly as before the normalization (INCORRECT against both the partial 5 and the monetary sum 8)
    ("ANSWER: $\\(5\\) + $\\(3\\)", "5", False),
    ("ANSWER: $\\(5\\) + $\\(3\\)", "8", False),
    ("ANSWER: \\(5\\) dollars", "5", True),
    # a display block closed on its own line under the answer (Nemotron smoke math_d5de6a08): the lone closing
    # delimiter hid the ANSWER line from the extractor's end anchor; dropped, the answer is read
    ("\\[\n\\text{ANSWER: } \\sqrt{51}\n\\]", "\\sqrt{51}", True),
    ("\\[\n\\text{ANSWER: } \\sqrt{51}\n\\]\n", "\\sqrt{51}", True),
    ("\\[\n\\text{ANSWER: } \\sqrt{51}\n\\]", "\\sqrt{50}", False),
]


def test_normalize_math_answer_rewrites_only_the_wrapper_around_the_mathematics():
    n = ts.normalize_math_answer
    assert n("ANSWER: \\( 5 \\)") == "ANSWER:  5 "
    assert n("ANSWER: \\(<|begin_of_box|>3\\sqrt{13}<|end_of_box|>\\)") == "ANSWER: 3\\sqrt{13}"
    assert n("<|begin_of_box|>ANSWER: \\(\\sqrt{51}\\)<|end_of_box|>") == "ANSWER: \\sqrt{51}"
    assert n("\\[ \\text{ANSWER: } \\frac{14}{3} \\]") == "\\[ ANSWER:  \\frac{14}{3} "
    assert n("ANSWER: $42$") == "ANSWER: $42$"
    assert n("ANSWER: 5") == "ANSWER: 5"
    assert n("<|begin_of_box|>partial") == "partial"          # an unpaired opener (a box cut by the cap): the text survives
    assert n("ANSWER: \\(5") == "ANSWER: 5"                   # a lone delimiter token on the answer line goes too
    assert n("ANSWER: \\(2\\) + \\(3\\)") == "ANSWER: 2 + 3"                 # no span is ever introduced: the whole expression stays
    assert n("\\[\n\\text{ANSWER: } \\sqrt{51}\n\\]") == "\\[\nANSWER:  \\sqrt{51}"   # the lone closing line under the answer goes
    assert n("\\[\n\\text{ANSWER: } \\sqrt{51}\n\\]\n\n") == "\\[\nANSWER:  \\sqrt{51}\n\n"
    assert n("ANSWER: 5\n\\] and more") == "ANSWER: 5\n\\] and more"       # only a line holding NOTHING but the delimiter
    assert n("x \\]\nANSWER: 5") == "x \\]\nANSWER: 5"                       # never a delimiter that is not the last line
    assert n("ANSWER: $\\(5\\) + $\\(3\\)") == "ANSWER: $\\(5\\) + $\\(3\\)"   # a literal `$` on the answer line: left as written
    assert n("ANSWER: \\(\\$5\\)") == "ANSWER: \\(\\$5\\)"                   # an escaped dollar counts as a dollar: untouched
    assert n("so \\(x=5\\).\n\nANSWER: 5") == "so \\(x=5\\).\n\nANSWER: 5"   # lines without the label are untouched
    assert n("r \\(a\\)\nANSWER: \\(7\\)\nmore \\(b\\)") == "r \\(a\\)\nANSWER: 7\nmore \\(b\\)"
    assert ts.ANSWER_NORMALIZATIONS[ts.ANSWER_NORMALIZATION_NONE]("ANSWER: \\( 5 \\)") == "ANSWER: \\( 5 \\)"
    assert ts.BOX_MARKERS == ("<|begin_of_box|>", "<|end_of_box|>")


def _state(content: str, target: str):
    pytest.importorskip("inspect_ai")     # the scorer-path tests need Inspect; the pure normalization test above does not
    from inspect_ai.model import ModelName, ModelOutput
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState
    st = TaskState(model=ModelName("tinycap/x"), sample_id="math_00000001", epoch=1, input="q", messages=[], target=Target(target))
    st.output = ModelOutput.from_content(model="tinycap/x", content=content)
    return st, Target(target)


def _content(mode: str, text: str) -> str:
    return {C.THINK_SPLIT_RECORD_ONLY: text, "explicit_tags": "<think>r</think>\n" + text, "implicit_open": "r</think>\n" + text}[mode]


@pytest.mark.parametrize("mode", [C.THINK_SPLIT_RECORD_ONLY, "explicit_tags", "implicit_open"])
@pytest.mark.parametrize("text,target,correct", CASES)
def test_the_math_scorer_scores_the_delimited_forms_through_the_real_extractor(mode, text, target, correct):
    pytest.importorskip("inspect_evals")
    from inspect_ai.scorer import CORRECT, INCORRECT

    from sisteer.capeval.tiers import inspect_served as isv
    content = _content(mode, text)
    st, tg = _state(content, target)
    s = asyncio.run(isv._math_scorer(mode)(st, tg))
    assert s.value == (CORRECT if correct else INCORRECT), (mode, text, s.value)
    assert s.metadata[ts.ANSWER_NORMALIZATION_FIELD] == {"rule": ts.ANSWER_NORMALIZATION_MATH,
                                                         "changed": ts.normalize_math_answer(text) != text}
    assert st.output.completion == content            # the raw output is untouched: the row keeps the model's text


def test_the_default_wrap_is_byte_identical_and_records_none():
    """The GSM8K wrap (no normalization named) hands the inner scorer the UNTOUCHED text — under record_only the
    very same state object — and records `none`."""
    pytest.importorskip("inspect_ai")     # think_aware_scorer builds the Inspect factory before any state exists
    seen: dict = {}

    async def inner(state, target):
        from inspect_ai.scorer import CORRECT, Score
        seen["completion"] = state.output.completion
        seen["state"] = state
        return Score(value=CORRECT, answer="5")

    text = "ANSWER: \\( 5 \\)"
    for mode in (C.THINK_SPLIT_RECORD_ONLY, "explicit_tags"):
        content = _content(mode, text)
        st, tg = _state(content, "5")
        s = asyncio.run(ts.think_aware_scorer(inner, mode=mode)(st, tg))
        assert seen["completion"] == (content if mode == C.THINK_SPLIT_RECORD_ONLY else text)
        if mode == C.THINK_SPLIT_RECORD_ONLY:
            assert seen["state"] is st
        assert s.metadata[ts.ANSWER_NORMALIZATION_FIELD] == {"rule": ts.ANSWER_NORMALIZATION_NONE, "changed": False}


def test_the_math_wrap_hands_the_inner_scorer_the_normalized_answer_and_keeps_the_raw_output():
    pytest.importorskip("inspect_ai")     # think_aware_scorer builds the Inspect factory before any state exists
    seen: dict = {}

    async def inner(state, target):
        from inspect_ai.scorer import CORRECT, Score
        seen["completion"] = state.output.completion
        return Score(value=CORRECT, answer="5")

    w = ts.think_aware_scorer(inner, mode=C.THINK_SPLIT_RECORD_ONLY, answer_normalization=ts.ANSWER_NORMALIZATION_MATH)
    st, tg = _state("ANSWER: \\( 5 \\)", "5")
    s = asyncio.run(w(st, tg))
    assert seen["completion"] == "ANSWER:  5 " and st.output.completion == "ANSWER: \\( 5 \\)"
    assert s.metadata[ts.ANSWER_NORMALIZATION_FIELD] == {"rule": ts.ANSWER_NORMALIZATION_MATH, "changed": True}
    # an answer already in the extractor's form: nothing changed, and the inner scorer reads the untouched state
    st, tg = _state("ANSWER: $5$", "5")
    s = asyncio.run(w(st, tg))
    assert s.metadata[ts.ANSWER_NORMALIZATION_FIELD]["changed"] is False
    w2 = ts.think_aware_scorer(inner, mode="explicit_tags", answer_normalization=ts.ANSWER_NORMALIZATION_MATH)
    st, tg = _state("<think>r</think>\nANSWER: <|begin_of_box|>7<|end_of_box|>", "7")
    s = asyncio.run(w2(st, tg))
    assert seen["completion"] == "ANSWER: 7" and s.metadata["think_block_present"] is True
    assert s.metadata[ts.ANSWER_NORMALIZATION_FIELD]["changed"] is True


def test_an_unknown_normalization_is_refused_by_name():
    async def inner(state, target):
        raise AssertionError("never called")

    with pytest.raises(ValueError, match="answer_normalization"):
        ts.think_aware_scorer(inner, mode="explicit_tags", answer_normalization="strip_everything")
