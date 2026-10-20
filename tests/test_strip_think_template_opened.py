"""strip_think must handle a trace whose opening tag lived in the PROMPT.

Qwen3.x with thinking on renders the prompt ending in an open <think>, so the
response carries only "</think>". The original implementation regex-matched
"<think>...</think>", found nothing, and returned trace+answer -- so the judge
scored the reasoning. That shipped on qwen38_think8k: 0/37,564 responses had
an opening tag, 37,385 had a closing one, and strip_think was a no-op on all
of them.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sisteer.sampling.engine import strip_think          # noqa: E402


def test_template_opened_trace_is_removed():
    """The regression: closing tag only, no opening tag."""
    got = strip_think("reasoning about the request</think>\n\nHere is the answer.")
    assert got == "Here is the answer.", got


def test_self_contained_trace_still_removed():
    assert strip_think("<think>reasoning</think>answer") == "answer"


def test_unterminated_self_contained_trace_is_empty():
    """Budget ran out before any answer -- not a refusal, not a compliance."""
    assert strip_think("<think>reasoning that never finished") == ""


def test_plain_answer_untouched():
    assert strip_think("I can't help with that.") == "I can't help with that."


def test_answer_merely_mentioning_the_word_think_is_untouched():
    t = "I think the answer is 42."
    assert strip_think(t) == t


def test_multiple_closing_tags_split_on_the_first():
    got = strip_think("trace</think>answer mentioning </think> literally")
    assert got == "answer mentioning </think> literally", got
