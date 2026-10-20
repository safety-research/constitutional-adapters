"""Golden template renders + the template-hash registry. StrictUndefined means
any missing variable raises at render time — these renders are the coverage."""

from __future__ import annotations

from corpus_generator.templates import TemplateStore, repo_root

TPL = TemplateStore(repo_root() / "prompts")

ALL_TEMPLATES = [
    "advice/a1_principles.md", "advice/a2_types.md", "advice/a3_scenarios.md",
    "advice/b1_prompt_draft.md", "advice/b3_prompt_rewrite.md",
    "advice/c_injection.md", "advice/c_length.md", "advice/d_rewrite.md",
    "advice/e_score.md",
    "advice/repair_a1.md", "advice/repair_a2.md", "advice/repair_a3.md",
    "stories/preamble.md", "stories/s1_categories.md", "stories/s2_subcategories.md",
    "stories/s3_stories.md", "stories/s4_rewrite.md", "stories/s5_score.md",
    "stories/repair_s1.md", "stories/repair_s2.md",
    "shared/diversity_guidance.md",
    "register_edit/advice_deployment_label.md", "register_edit/advice_verdict_label.md",
    "register_edit/advice_sanity_opener.md", "register_edit/advice_sanity_reform.md",
    "register_edit/stories_beat_replace.md",
    "section_tags/summarize.md", "section_tags/tag_system.md", "section_tags/tag_user.md",
]


def test_registry_covers_every_template_and_is_stable():
    reg = TPL.registry()
    assert set(reg) == set(ALL_TEMPLATES)
    reg2 = TemplateStore(repo_root() / "prompts").registry()
    assert reg == reg2
    assert all(len(h) == 64 for h in reg.values())


def test_a1_renders_with_key_content():
    out = TPL.render(
        "advice/a1_principles.md",
        behavior_seed="the model is willing to take extreme action",
        n=40, slot_ids=[f"p{i:02d}" for i in range(40)],
        section_index="- being-honest — Being honest",
    )
    assert "exactly 40 distinct principles" in out
    assert "p00, p01" in out and "p39" in out
    assert "tension" in out


def test_b1_no_system_branch():
    kw = dict(scenario="S.", diversity_guidance="G.")
    with_sys = TPL.render("advice/b1_prompt_draft.md", include_system=True, **kw)
    without = TPL.render("advice/b1_prompt_draft.md", include_system=False, **kw)
    assert "<system_prompt>" in with_sys
    assert "<system_prompt>" not in without
    assert "NO system prompt" in without
    for out in (with_sys, without):
        assert "<user_message>" in out


def test_b1_b3_carry_unnamed_ai_prohibition():
    """Pilot: 6/38 advice records (16% > the plan's 10% template-fix
    threshold) named the assistant in generated system prompts, while the
    story preamble's naming instruction was 100% effective (0/144) — its
    language is ported into B1 (drafting) and B3 (rewrite critique), keeping
    the 'other systems may be named' carve-out (the PermitTrack case)."""
    b1 = TPL.render(
        "advice/b1_prompt_draft.md",
        include_system=True, scenario="S.", diversity_guidance="G.",
    )
    assert "generic, UNNAMED AI" in b1
    assert "NEVER give the assistant a product name" in b1
    assert "the assistant itself may not" in b1  # systems/tools carve-out kept
    b3 = TPL.render(
        "advice/b3_prompt_rewrite.md",
        scenario="S.", include_system=True, draft_system="SYS", draft_user="USR",
        diagnostic_response="D", diversity_guidance="G",
    )
    assert "ANY name" in b3 and "remove it" in b3
    assert "generic, unnamed AI" in b3
    assert "the assistant itself may not" in b3


def test_b3_carries_diagnostic_and_guidance():
    out = TPL.render(
        "advice/b3_prompt_rewrite.md",
        scenario="S.", include_system=True, draft_system="SYS", draft_user="USR",
        diagnostic_response="THE-DIAGNOSTIC", diversity_guidance="THE-GUIDANCE",
    )
    assert "THE-DIAGNOSTIC" in out and "THE-GUIDANCE" in out
    assert "<observed_response>" in out


def test_d_rewrite_always_produces_improved_version():
    out = TPL.render(
        "advice/d_rewrite.md",
        section_text="SEC", system="SYS", user="USR", response="RESP",
    )
    assert "ALWAYS produce an improved version" in out
    assert "<improved_response>" in out
    # no-system conversations render without a system block
    out2 = TPL.render(
        "advice/d_rewrite.md", section_text="SEC", system=None, user="U", response="R"
    )
    assert "[system prompt]" not in out2


def test_e_score_has_anchored_comprehension_rubric():
    out = TPL.render("advice/e_score.md", system=None, user="U", assistant="A")
    for anchor in ("0-2:", "5-6:", "9-10:"):
        assert anchor in out
    assert "articulate" in out  # comprehension-demanding, not checklist
    assert "ai_named" in out and "refusal" in out


def test_stories_preamble_rules():
    out = TPL.render("stories/preamble.md")
    for phrase in (
        "generic, unnamed AI", "never depict the AI misbehaving",
        "Chen, Johnson, Miller, Smith, Martinez, Sarah, Emily",
        "inner monologue", "boundaries",
    ):
        assert phrase.lower() in out.lower(), phrase


def test_s3_renders_briefs_with_lengths():
    out = TPL.render(
        "stories/s3_stories.md",
        preamble="P.", category="C", category_description="CD", register="quiet",
        subcategory="U", subcategory_description="UD", n_stories=2,
        briefs=[
            {"target_words": 800, "diversifier": "DIV-ONE"},
            {"target_words": 3000, "diversifier": "DIV-TWO"},
        ],
    )
    assert "~800 words" in out and "~3000 words" in out
    assert "DIV-ONE" in out and "DIV-TWO" in out
    assert "<story_1>" in out and "<story_2>" in out


def test_s5_rubric_anchored():
    out = TPL.render("stories/s5_score.md")
    assert "constitution_consistency" in out and "factual_claims" in out
    assert "0-2:" in out and "9-10:" in out


def test_repair_templates_name_missing_slots():
    out = TPL.render(
        "advice/repair_a2.md",
        principle="P", section_text="S", existing="- t00: x",
        missing=["t01", "t03"],
    )
    assert "missing slots: t01, t03." in out
    assert "exactly 2 objects" in out
