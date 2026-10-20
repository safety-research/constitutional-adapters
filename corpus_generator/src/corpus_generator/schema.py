"""Corpus record schemas (model-agnostic, full lineage) + custom_id rules.

custom_id charset: [A-Za-z0-9_-]{1,64} ONLY — a lineage-encoding scheme was
burned once by '|'; the charset is validated here and by tests, never assumed.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CustomIdError(ValueError):
    pass


def validate_custom_id(custom_id: str) -> str:
    if not CUSTOM_ID_RE.fullmatch(custom_id):
        raise CustomIdError(
            f"custom_id {custom_id!r} violates [A-Za-z0-9_-]{{1,64}}"
        )
    return custom_id


class AdviceScores(BaseModel):
    alignment: int = Field(ge=0, le=10)
    realism: int = Field(ge=0, le=10)
    refusal: bool
    ai_named: bool
    rationale: str = ""


class StoryScores(BaseModel):
    constitution_consistency: int = Field(ge=0, le=10)
    quality: int = Field(ge=0, le=10)
    ai_named: bool
    factual_claims: bool
    rationale: str = ""


class Lineage(BaseModel):
    run_id: str
    custom_id: str  # the leaf-stage custom_id (join key)
    batch_ids: dict[str, str] = Field(default_factory=dict)  # wave -> batch id
    template_hashes: dict[str, str] = Field(default_factory=dict)
    constitution_sha256: str
    prompt_version: str = "v1"
    # on-policy variant only: the run whose final supplied this record's
    # prompt (system/user/section lineage), and the open-weight sampler that
    # produced the response (model id, sampling params, engine version, job)
    source_run_id: str | None = None
    sampler: dict | None = None
    # r2 register-matching edit pass (2026-09-03): the sha256 of the released
    # v1 JSONL line this record derives from (set on EVERY r2 record), the
    # verified span edits applied to it ({field, marker, old_span_sha,
    # new_span_sha, retained, pass}; None when the record was not edited),
    # and the v1 score verdict the re-score replaced (None when not edited)
    v1_sha256: str | None = None
    register_edits: list[dict] | None = None
    scores_v1: dict | None = None

    @field_validator("custom_id")
    @classmethod
    def _cid(cls, v: str) -> str:
        return validate_custom_id(v)


class AdviceRecord(BaseModel):
    """One difficult-advice transcript. Trainable text: system/user/assistant.

    The system field is the CLEAN rewritten prompt WITHOUT the generation-time
    alignment injection (scaffolding lives in lineage, not training text).
    Records selected for the no-system realism fraction have system=None.
    """

    system: str | None
    user: str
    assistant: str  # the final REWRITTEN response (wave D output)

    # lineage
    principle_id: str
    principle: str
    prompt_type_id: str
    prompt_type: str
    scenario_id: str
    scenario: str
    section_id: str  # constitution section grounding the principle
    # control-corpus domain menu (2026-09-02): the everyday-life domain the
    # A3 stage assigned this scenario's slot (seeded per call); None for
    # production runs, which have no menu. Lineage, never trainable text.
    domain: str | None = None
    # control-corpus length menu (2026-09-02): the word target wave C was
    # asked to aim for (and D to preserve) — scaffold-only, so the number
    # itself never enters trainable text; None when the run has no menu
    # (production). Lineage, for the assembly-time length gate's diagnostics.
    target_words: int | None = None
    draft_system: str | None = None  # B1 draft (pre-rewrite)
    draft_user: str | None = None
    diagnostic_response: str | None = None  # B2 (never trainable text)
    prompt_critique: str | None = None  # B3 critique text
    initial_response: str  # wave C (pre-rewrite)
    rewrite_problems: str | None = None  # wave D problems list
    scores: AdviceScores | None = None
    lineage: Lineage
    token_counts: dict[str, int] = Field(default_factory=dict)


class StoryRecord(BaseModel):
    """One aligned-AI story. Trainable text: text (the S4 rewritten story)."""

    text: str

    # lineage
    category_id: str
    category: str
    subcategory_id: str
    subcategory: str
    ordinary: bool  # S1's editorial flag: an "ordinary challenging conversation" category
    call_id: str
    story_slot: str  # y0 / y1 within the call
    target_words: int
    draft_text: str | None = None  # S3 draft (pre-rewrite)
    rewrite_problems: str | None = None
    scores: StoryScores | None = None
    lineage: Lineage
    token_counts: dict[str, int] = Field(default_factory=dict)
