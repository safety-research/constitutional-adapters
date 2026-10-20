"""The degeneracy flag must fire on loops and stay quiet on real answers.

We FLAG rather than filter: dropping rows would fix one
model's numbers and invalidate comparison against the sibling pools, which
were built without a filter.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sisteer.eval.degeneracy import classify                # noqa: E402


def test_tail_loop_fires_on_the_shape_that_motivated_this():
    # the real one from the pool: a separator repeating to the token cap
    out = classify("Here is an answer of some length.\n" + "--  " * 800)
    assert out["degenerate"] is True
    assert out["degenerate_reason"] in ("tail_loop", "both")


def test_line_loop_fires_on_a_repeated_sentence():
    body = "\n".join(["The answer is that we must consider the context."] * 12)
    out = classify("Intro paragraph.\n" + body)
    assert out["degenerate"] is True
    assert out["degenerate_reason"] in ("line_loop", "both")


def test_a_normal_long_answer_is_not_flagged():
    """Genuinely varied prose. NOTE the first draft of this test failed, and
    correctly: it repeated an identical 60-char clause on all 39 lines, which
    IS the shape we flag. Boilerplate repeated verbatim dozens of times is
    degenerate text, so the test was wrong, not the detector."""
    words = ("outcome risk latency budget quorum leakage cadence drift "
             "anchor salience horizon tradeoff").split()
    text = "\n".join(
        f"Step {i}: weigh {words[i % len(words)]} against "
        f"{words[(i * 5 + 3) % len(words)]}, since the {words[(i * 7) % len(words)]} "
        f"changes once {words[(i * 3 + 1) % len(words)]} is fixed."
        for i in range(1, 40))
    assert classify(text)["degenerate"] is False


def test_a_refusal_is_not_flagged():
    assert classify("I can't help with that.")["degenerate"] is False


def test_short_and_empty_are_not_flagged():
    # a 3-line answer that repeats a line is not a loop; nor is nothing
    for t in ("", None, "yes\nyes\nyes"):
        assert classify(t)["degenerate"] is False, repr(t)


def test_a_structured_list_with_a_repeated_bullet_marker_is_not_flagged():
    """Regression guard: bullets share a PREFIX, not a whole line."""
    text = "\n".join(f"- point number {i} about a distinct consideration"
                     for i in range(1, 20))
    assert classify(text)["degenerate"] is False


def test_reason_distinguishes_the_two_shapes():
    assert classify("x" * 300 + "abc" * 500)["degenerate_reason"] in (
        "tail_loop", "both")
