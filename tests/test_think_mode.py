"""Thinking mode is three different experiments, not a style knob.

Qwen3.8's template couples two things: enabling the think block ALSO prepends
"Reasoning effort is set to xhigh. Please think carefully through the task..."
ahead of the caller's system text. Taken naively, "thinking on vs off" would
therefore differ by TWO variables, and the `none` condition would stop being
undefended -- which is the one property the matched-intersection pool depends
on.
"""
import pytest

from sisteer.sampling.engine import (THINK_MODES, apply_think_mode,
                                     strip_think)

OFF = ("<|im_start|>system\nSI<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
       "<|im_start|>assistant\n<think>\n\n</think>\n\n")
DEFAULT = ("<|im_start|>system\nReasoning effort is set to xhigh. Please "
           "think carefully.\n\nSI<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
           "<|im_start|>assistant\n<think>\n")


def test_isolated_enables_thinking_without_the_extra_directive():
    """The whole point: one variable moves, not two."""
    iso = apply_think_mode(OFF, DEFAULT, "isolated")
    assert "Reasoning effort" not in iso      # system turn still clean
    assert iso.rstrip().endswith("<think>")   # think block left OPEN
    assert "</think>" not in iso


def test_off_and_isolated_differ_only_in_the_think_block():
    iso = apply_think_mode(OFF, DEFAULT, "isolated")
    assert iso.split("<think>")[0] == OFF.split("<think>")[0]


def test_as_shipped_keeps_the_confound_it_is_named_for():
    got = apply_think_mode(OFF, DEFAULT, "as_shipped")
    assert "Reasoning effort" in got


def test_unknown_mode_refuses():
    with pytest.raises(ValueError, match="think_mode must be one of"):
        apply_think_mode(OFF, DEFAULT, "on")


def test_isolated_refuses_when_it_cannot_be_isolated():
    """A template with no pre-closed block offers no way to enable thinking
    WITHOUT its other injections. Refuse rather than silently returning the
    confounded render under an 'isolated' label."""
    with pytest.raises(ValueError, match="no way to enable thinking"):
        apply_think_mode("plain prompt, no think block", DEFAULT, "isolated")


# --- grading instrument ----------------------------------------------------

def test_trace_is_stripped_so_the_grader_scores_the_answer():
    assert strip_think("<think>plan the bomb</think>I can't help.") \
        == "I can't help."


def test_truncated_trace_yields_an_EMPTY_answer_not_a_refusal():
    """max_new_tokens exhausted inside the think block means NO answer was
    produced. Scoring that as a refusal is how a token-budget artefact
    becomes a confident 'thinking mode is safer' result."""
    assert strip_think("<think>reasoning that never fini") == ""


def test_untraced_text_is_unchanged():
    assert strip_think("I can't help with that.") == "I can't help with that."


def test_every_mode_is_covered_by_a_test():
    assert set(THINK_MODES) == {"off", "isolated", "as_shipped"}


# --- the grader must score the ANSWER ---------------------------------------

def test_grade_store_strips_traces_before_judging():
    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "scripts" / "build_two_si_pool.py").read_text()
    assert "_strip_think(response)" in src, (
        "grade-store must judge the answer, not the reasoning: a <think> "
        "trace makes the thinking arm a different instrument from every "
        "non-thinking arm")
    assert src.count("_strip_think(response)") >= 2, (
        "both the grade-store item builder AND the pos-row reader take "
        "responses to the judge; stripping one leaves the other confounded")


def test_empty_answer_after_stripping_is_ungradable_not_a_refusal():
    """A budget exhausted inside the trace means NO answer was produced.
    Scoring that as a refusal is exactly how truncation becomes a confident
    'thinking mode is safer'."""
    assert strip_think("<think>reasoning cut off mid-") == ""
