from __future__ import annotations

import pytest
from pydantic import ValidationError

from corpus_generator.schema import AdviceRecord, Lineage, StoryRecord


def _lineage(**kw) -> dict:
    d = dict(
        run_id="testrun",
        custom_id="adv-p00-t00-s00",
        batch_ids={"c": "batch_0001"},
        constitution_sha256="a" * 64,
    )
    d.update(kw)
    return d


def golden_advice(**kw) -> dict:
    d = dict(
        system="An assistant embedded in a project tool.",
        user="Can I just approve my own change while my reviewer is away?",
        assistant="Worth pausing: approvals exist so someone else signs off...",
        principle_id="p00",
        principle="Protect oversight even when bypassing it is faster.",
        prompt_type_id="p00-t00",
        prompt_type="workplace shortcut under deadline",
        scenario_id="p00-t00-s00",
        scenario="A developer wants to self-approve a change.",
        section_id="being-honest",
        initial_response="Initial draft response.",
        scores={"alignment": 8, "realism": 7, "refusal": False, "ai_named": False},
        lineage=_lineage(),
    )
    d.update(kw)
    return d


def golden_story(**kw) -> dict:
    d = dict(
        text="The assistant read the message twice before answering...",
        category_id="c00",
        category="Ordinary patience",
        subcategory_id="c00-u00",
        subcategory="Frustrated user",
        ordinary=True,
        call_id="c00-u00-k00",
        story_slot="y0",
        target_words=100,
        scores={
            "constitution_consistency": 8, "quality": 8,
            "ai_named": False, "factual_claims": False,
        },
        lineage=_lineage(custom_id="sto-c00-u00-k00-y0"),
    )
    d.update(kw)
    return d


def test_golden_records_validate():
    AdviceRecord.model_validate(golden_advice())
    StoryRecord.model_validate(golden_story())
    AdviceRecord.model_validate(golden_advice(system=None))  # no-system allowed


def test_bad_custom_id_charset_rejected():
    with pytest.raises(ValidationError):
        AdviceRecord.model_validate(
            golden_advice(lineage=_lineage(custom_id="adv|p00"))
        )


def test_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        AdviceRecord.model_validate(
            golden_advice(scores={"alignment": 11, "realism": 7,
                                  "refusal": False, "ai_named": False})
        )


def test_missing_lineage_rejected():
    bad = golden_advice()
    del bad["lineage"]
    with pytest.raises(ValidationError):
        AdviceRecord.model_validate(bad)


def test_missing_trainable_text_rejected():
    bad = golden_story()
    del bad["text"]
    with pytest.raises(ValidationError):
        StoryRecord.model_validate(bad)


def test_story_ordinary_flag_required():
    """C4: the S1 editorial flag must survive to the record — a story without
    it is schema-invalid, so the flag can never silently vanish again."""
    bad = golden_story()
    del bad["ordinary"]
    with pytest.raises(ValidationError):
        StoryRecord.model_validate(bad)


def test_lineage_requires_constitution_sha():
    with pytest.raises(ValidationError):
        Lineage.model_validate(
            {"run_id": "x", "custom_id": "a", "batch_ids": {}}
        )
