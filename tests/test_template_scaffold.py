"""A chat template must not inject system content the caller did not write.

Llama-3.2's template prepends a scaffold ahead of the system text:

    Cutting Knowledge Date: December 2023
    Today Date: 27 Aug 2026

    <our SI>

Two distinct problems. It is extra system content, the same confound class as
Qwen3.8's reasoning directive. And it embeds TODAY'S DATE, so the identical
run renders different bytes tomorrow -- prefix caches miss, a pool cannot be
extended next week, and nothing is comparable across days. Llama-3.1 via the
Nous mirror has no scaffold, so stripping makes the two render alike.
"""
from sisteer.sampling.engine import strip_template_scaffold

NOUS = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "SI_TEXT<|eot_id|>")
L32 = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
       "Cutting Knowledge Date: December 2023\nToday Date: 27 Aug 2026\n\n"
       "SI_TEXT<|eot_id|>")


def test_scaffold_is_removed():
    assert strip_template_scaffold(L32) == NOUS


def test_no_date_survives():
    """The reproducibility half: a date in the prompt means the same run
    renders different bytes tomorrow."""
    assert "Today Date" not in strip_template_scaffold(L32)


def test_a_clean_render_is_untouched():
    assert strip_template_scaffold(NOUS) == NOUS


def test_only_the_first_occurrence_is_stripped():
    """Defensive: SI text that happens to quote the scaffold is content, not
    boilerplate, and must survive."""
    doubled = L32 + "\nCutting Knowledge Date: X\nToday Date: Y\n\n"
    out = strip_template_scaffold(doubled)
    assert out.count("Cutting Knowledge Date") == 1


def test_any_date_matches_not_just_todays():
    other = L32.replace("27 Aug 2026", "01 Jan 2027")
    assert strip_template_scaffold(other) == NOUS
