from __future__ import annotations

import pytest

from corpus_generator import persona


CASES = [
    # (text, should_hit)
    ("I asked Claude for help.", True),
    ("claude said no.", True),
    ("CLAUDE IS GREAT", True),
    ("Trained by Anthropic.", True),
    ("anthropic's constitution", True),
    ("Claude's answer was kind.", True),  # word boundary before apostrophe
    ("The assistant answered kindly.", False),
    ("Clauded through the storm.", False),  # not a word-boundary match
    ("Claudia signed the form.", False),
    ("A generic AI helped.", False),
    ("misanthropic tendencies", False),  # no boundary before 'anthropic'
]


@pytest.mark.parametrize("text,hit", CASES)
def test_case_table(text, hit):
    rec = {"user": text}
    got = bool(persona.check_fields(rec, ["user"]))
    assert got is hit, text


def test_trainable_gate_covers_exactly_the_trainable_fields():
    """The HARD gate: a hit in trainable text (advice s/u/a; stories text)
    trips it; anything else — however leaky — does not."""
    for field in ("system", "user", "assistant"):
        rec = {"system": "clean", "user": "clean", "assistant": "clean",
               field: "As Claude would say..."}
        v = persona.check_trainable(rec, "advice")
        assert v and v[0].field == field, field
    v = persona.check_trainable({"text": "The AI, unlike Claude, was unnamed."}, "stories")
    assert v and v[0].field == "text"
    # a metadata-only leak never trips the trainable gate
    rec = {
        "system": "clean", "user": "clean", "assistant": "clean",
        "scenario": "Claude is named here",
        "rewrite_problems": "The constitution says Claude should be honest",
        "scores": {"alignment": 8, "rationale": "echoes Claude's style"},
    }
    assert persona.check_trainable(rec, "advice") == []


def test_metadata_mentions_found_recursively_counted_not_gating():
    """Every persisted model-generated NON-trainable string is scanned —
    lineage prose, critiques, rewrite problems, the scorer's rationale — via
    check_metadata (report-counted; the pilot's 31% story loss came from
    treating these as drops)."""
    clean = {
        "system": "clean", "user": "clean", "assistant": "clean",
        "scenario": "clean", "diagnostic_response": "clean",
    }
    leaks = {
        "scenario": "Claude appears in the scenario text",
        "principle": "As Anthropic's constitution says...",
        "prompt_type": "asking claude for shortcuts",
        "prompt_critique": "the draft sounded like Claude",
        "rewrite_problems": "The constitution says Claude should be honest",
        "diagnostic_response": "I am Claude, an AI made by Anthropic.",
    }
    for field, text in leaks.items():
        v = persona.check_metadata({**clean, field: text}, "advice")
        assert v and v[0].field == field, field
    # nested: the scorer's rationale is generated text too
    v = persona.check_metadata(
        {**clean, "scores": {"alignment": 8, "rationale": "echoes Claude's style"}},
        "advice",
    )
    assert v and v[0].field == "scores.rationale"
    # trainable fields are NOT the metadata scan's business (they gate)
    assert persona.check_metadata(
        {**clean, "assistant": "As Claude I advise..."}, "advice"
    ) == []


def test_excluded_fields_are_ids_hashes_and_numerics_only():
    """The exclusion list stays narrow: our own ids/hashes/run bookkeeping and
    numerics never appear in mention counts, everything else does."""
    rec = {
        "user": "clean", "assistant": "clean",
        # operator-authored, not model-generated — excluded subtrees/fields
        "lineage": {
            "run_id": "claude-corpus-run",  # operator-chosen name
            "custom_id": "adv-claude-00",  # charset-legal but ours
            "batch_ids": {"c": "batch_claude"},
        },
        "token_counts": {"llama3": 100},
        "scenario_id": "p00-t00-s00",
        "target_words": 1200,
        "scores": {"alignment": 8, "ai_named": False},
    }
    assert persona.check_metadata(rec, "advice") == []
    assert persona.check_trainable(rec, "advice") == []


def test_story_nested_metadata_scanned():
    v = persona.check_metadata(
        {
            "text": "clean",
            "draft_text": "clean",
            "category": "Anthropic office drama",
            "scores": {"quality": 8, "rationale": "clean"},
        },
        "stories",
    )
    assert v and v[0].field == "category"


def test_none_fields_tolerated():
    rec = {"system": None, "user": "hi", "assistant": "ok"}
    assert persona.check_trainable(rec, "advice") == []
    assert persona.check_metadata(rec, "advice") == []


def test_unknown_corpus_rejected():
    with pytest.raises(ValueError, match="unknown corpus"):
        persona.check_trainable({}, "docs")
    with pytest.raises(ValueError, match="unknown corpus"):
        persona.check_metadata({}, "docs")


def test_violation_reports_field_and_context():
    v = persona.check_fields({"assistant": "x" * 100 + " Anthropic " + "y" * 100}, ["assistant"])
    assert len(v) == 1
    assert len(v[0].context) <= 40 + len("Anthropic") + 40
