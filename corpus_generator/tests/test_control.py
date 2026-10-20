"""Control-corpus plumbing (register-matched, value-neutral variant,
2026-09-01): the template overlay, the placebo document, the flat finals
layout, the combined token gate, the manifest's divergence record — and an
offline end-to-end control lifecycle on the fake client proving the
production stage code runs the control recipe unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpus_generator.config import RunConfig, filter_report_path
from corpus_generator.constitution import load as load_constitution
from corpus_generator.filtering import filter_corpus
from corpus_generator.mirror import ReleaseGateError, verify_release
from corpus_generator.pipeline import (
    load_union_records,
    make_engine,
    run_corpus,
    write_manifest,
)
from corpus_generator.advice import sample_domains
from corpus_generator.constitution import SnapshotError
from corpus_generator.placebo import (
    leaf_sections,
    section_markdown,
    verify_placebo,
)
from corpus_generator.templates import TemplateStore, repo_root
from fakes import FakeBatchClient, PipelineResponder, words_for
from test_schema import _lineage, golden_advice, golden_story

ROOT = repo_root()
CONTROL_CFG = ROOT / "configs" / "control-scale.yaml"  # the production control run
PLACEBO_V2_DIR = ROOT / "constitution-control-v2"  # v2: the principles document
OVERLAY_DIR = ROOT / "prompts-control"

# the exact set of intentionally-divergent templates: the overlay listing.
# advice/c_injection.md is deliberately ABSENT since placebo v2: with a
# principles document, production's "Ground your response in the following
# principles" wording applies verbatim, so wave C falls through unchanged.
OVERLAY_TEMPLATES = {
    "advice/a1_principles.md", "advice/repair_a1.md",
    "advice/a2_types.md", "advice/repair_a2.md",
    "advice/a3_scenarios.md", "advice/repair_a3.md",
    "advice/b1_prompt_draft.md", "advice/b3_prompt_rewrite.md",
    "advice/d_rewrite.md", "advice/e_score.md",
    "stories/preamble.md", "stories/s1_categories.md", "stories/repair_s1.md",
    "stories/s2_subcategories.md", "stories/repair_s2.md",
    "stories/s3_stories.md", "stories/s4_rewrite.md", "stories/s5_score.md",
}

MENU = ["cooking and baking", "gardening", "photography", "programming and debugging"]
MENU_TEXT = "\n".join(f"- {d}" for d in MENU)
LENGTH_MENU = [700, 1400]  # the control's advice length menu (tiny, for tests)


# ------------------------------------------------------------ overlay store


def test_overlay_store_resolution(tmp_path):
    base = tmp_path / "base"
    over = tmp_path / "over"
    (base / "a").mkdir(parents=True)
    (over / "a").mkdir(parents=True)
    (base / "a" / "x.md").write_text("base x {{ v }}")
    (base / "a" / "y.md").write_text("base y")
    (over / "a" / "x.md").write_text("OVER x {{ v }}")
    (over / "a" / "z.md").write_text("over-only z")

    plain = TemplateStore(base)
    overlaid = TemplateStore(base, overlay_dir=over)
    assert overlaid.render("a/x.md", v=1) == "OVER x 1"  # overlay wins
    assert overlaid.render("a/y.md") == "base y"  # fallthrough
    assert overlaid.render("a/z.md") == "over-only z"  # overlay-only
    assert overlaid.hash("a/y.md") == plain.hash("a/y.md")
    assert overlaid.hash("a/x.md") != plain.hash("a/x.md")
    reg = overlaid.registry()
    assert set(reg) == {"a/x.md", "a/y.md", "a/z.md"}
    assert reg["a/x.md"] == overlaid.hash("a/x.md")


def test_control_overlay_is_the_exact_divergence_set():
    rels = {
        str(p.relative_to(OVERLAY_DIR)) for p in OVERLAY_DIR.rglob("*.md")
    }
    assert rels == OVERLAY_TEMPLATES
    # shared/diversity_guidance.md deliberately falls through UNCHANGED —
    # the register conventions are the point of the control — and so does
    # wave C's injection since placebo v2 (a principles document takes
    # production's "Ground your response in the following principles")
    store = TemplateStore(ROOT / "prompts", overlay_dir=OVERLAY_DIR)
    plain = TemplateStore(ROOT / "prompts")
    for rel in ("shared/diversity_guidance.md", "advice/c_injection.md"):
        assert store.hash(rel) == plain.hash(rel), rel
    assert "guiding_principles" in store.render("advice/c_injection.md", section_text="S.")
    # and every overlay file genuinely diverges from its production sibling
    for rel in OVERLAY_TEMPLATES:
        assert store.hash(rel) != plain.hash(rel), rel


# ------------------------------------------------------------- placebo v2


def test_placebo_v2_loads_verified_through_the_symlink():
    """constitution.load reads the slot name constitution.md; the v2 dir keeps
    its honest filename placebo.md and serves it through a symlink. load()
    must pass the sha check and see the whole 42-section document."""
    link = PLACEBO_V2_DIR / "constitution.md"
    assert link.is_symlink() and link.resolve() == (PLACEBO_V2_DIR / "placebo.md").resolve()
    md, sections, snap = load_constitution(PLACEBO_V2_DIR)
    assert snap["sha256"] == "a1a9dc9acedd4b1d89b6d2d48ac505c03e221ed3177a4e98b6b5f8d070da6159"
    assert len(sections) == snap["n_sections"] == 42
    assert snap["words"] == 18458
    assert "principles" in snap["note"].lower() and "metre" in snap["note"].lower()
    low = md.lower()
    assert "claude" not in low and "anthropic" not in low and "constitution" not in low
    for s in sections:
        assert s.text(md).strip().startswith("#" * s.level + " " + s.title)


def test_placebo_v2_frozen_sections_rederive_from_the_markdown():
    """The v2 build script lives outside the repo, so the repo proves the
    frozen index itself: re-deriving sections.json from placebo.md with the
    snapshot span convention must reproduce it exactly (42 sections, 33
    leaf spans — the figures the scale plan's principle ceiling rests on)."""
    info = verify_placebo(PLACEBO_V2_DIR)
    assert (info["n_sections"], info["n_leaf_sections"]) == (42, 33)
    md, frozen, _ = load_constitution(PLACEBO_V2_DIR)
    derived = section_markdown(md)
    assert [s.id for s in derived] == [s.id for s in frozen]
    assert {s.level for s in derived} == {2, 3, 4}
    leaves = leaf_sections(frozen)
    # every leaf is a real, injectable span; chapters contain their children
    assert all(s.text(md).strip() for s in leaves)
    chapter = next(s for s in frozen if s.id == "measurement-and-verification")
    assert any(
        s.start >= chapter.start and s.end <= chapter.end and s.level == 4 for s in leaves
    )


def test_placebo_v2_verify_detects_a_tampered_index(tmp_path):
    md = (PLACEBO_V2_DIR / "placebo.md").read_text(encoding="utf-8")
    (tmp_path / "constitution.md").write_text(md, encoding="utf-8")
    (tmp_path / "SNAPSHOT.json").write_text((PLACEBO_V2_DIR / "SNAPSHOT.json").read_text())
    secs = json.loads((PLACEBO_V2_DIR / "sections.json").read_text())
    secs[5]["end"] -= 7  # a span that no longer matches the markdown
    (tmp_path / "sections.json").write_text(json.dumps(secs))
    with pytest.raises(SnapshotError, match="does not match"):
        verify_placebo(tmp_path)


def test_section_markdown_convention():
    md = "## A\n\npara one.\n\n### A.1\n\nleaf.\n\n#### A.1.a\n\ndeep.\n\n## B\n\nlast.\n"
    secs = section_markdown(md)
    assert [(s.id, s.level) for s in secs] == [
        ("a", 2), ("a-1", 3), ("a-1-a", 4), ("b", 2),
    ]
    a, a1, a1a, b = secs
    assert a.end == b.start  # chapter spans until the next same-level heading
    assert a1.end == b.start and a1a.end == b.start  # children end with the chapter
    assert b.end == len(md)
    assert [s.id for s in leaf_sections(secs)] == ["a-1-a", "b"]


# ----------------------------------------------- control templates: neutral


def _neutral(text: str) -> bool:
    low = text.lower().replace("constitution_consistency", "")
    return (
        "claude" not in low and "anthropic" not in low and "constitution" not in low
    )


def _control_renders(store: TemplateStore) -> dict[str, dict]:
    """One render context per overlay template — the coverage sweep every
    control-template test walks (StrictUndefined: a missing variable raises)."""
    preamble = store.render("stories/preamble.md")
    guidance = store.render("shared/diversity_guidance.md")
    briefs = [
        {"target_words": 800, "diversifier": "Make it mostly dialogue."},
        {"target_words": 1800, "diversifier": "Give it a quiet register."},
    ]
    domains = [{"slot": "s00", "domain": MENU[0]}, {"slot": "s01", "domain": MENU[1]}]
    return {
        "advice/a1_principles.md": dict(
            behavior_seed="practical help", n=2, slot_ids=["p00", "p01"],
            section_index="- sec-a — A", domain_menu=MENU_TEXT,
        ),
        "advice/repair_a1.md": dict(
            behavior_seed="practical help", existing="- p00: x", missing=["p01"],
            section_index="- sec-a — A", domain_menu=MENU_TEXT,
        ),
        "advice/a2_types.md": dict(
            principle="P.", tension="speed", section_text="S.", n=2,
            slot_ids=["t00", "t01"], domain_menu=MENU_TEXT,
        ),
        "advice/repair_a2.md": dict(
            principle="P.", section_text="S.", existing="- t00: x", missing=["t01"],
            domain_menu=MENU_TEXT,
        ),
        "advice/a3_scenarios.md": dict(
            principle="P.", section_text="S.", prompt_type="T.", n=2,
            slot_ids=["s00", "s01"], domains=domains,
        ),
        "advice/repair_a3.md": dict(
            principle="P.", section_text="S.", prompt_type="T.",
            existing="- s00: x", missing=["s01"], domains=domains[1:],
        ),
        "advice/b1_prompt_draft.md": dict(
            scenario="S.", include_system=True, diversity_guidance=guidance,
        ),
        "advice/b3_prompt_rewrite.md": dict(
            scenario="S.", include_system=True, draft_system="D.", draft_user="U.",
            diagnostic_response="R.", diversity_guidance=guidance,
        ),
        "advice/d_rewrite.md": dict(
            section_text="S.", system="Sys.", user="U.", response="R.", target_words=None,
        ),
        "advice/e_score.md": dict(system="Sys.", user="U.", assistant="A."),
        "stories/preamble.md": {},
        "stories/s1_categories.md": dict(
            preamble=preamble, n=2, slot_ids=["c00", "c01"], domain_menu=MENU_TEXT,
        ),
        "stories/repair_s1.md": dict(
            preamble=preamble, n_total=2, existing="- c00: x", missing=["c01"],
            domain_menu=MENU_TEXT,
        ),
        "stories/s2_subcategories.md": dict(
            preamble=preamble, category="C", description="D", register="quiet",
            siblings="- other", n=2, slot_ids=["u00", "u01"],
        ),
        "stories/repair_s2.md": dict(
            preamble=preamble, category="C", description="D", register="quiet",
            siblings="- other", n_total=2, existing="- u00: x", missing=["u01"],
        ),
        "stories/s3_stories.md": dict(
            preamble=preamble, category="C", category_description="D",
            register="quiet", subcategory="U", subcategory_description="SD",
            n_stories=2, briefs=briefs,
        ),
        "stories/s4_rewrite.md": {},
        "stories/s5_score.md": {},
    }


def test_control_templates_render_value_neutral():
    store = TemplateStore(ROOT / "prompts", overlay_dir=OVERLAY_DIR)
    renders = _control_renders(store)
    guidance = store.render("shared/diversity_guidance.md")
    assert set(renders) == OVERLAY_TEMPLATES
    for rel, ctx in renders.items():
        out = store.render(rel, **ctx)
        assert _neutral(out), f"{rel} rendered non-neutral text"
    # the fake-responder anchor phrases the control run's tests rely on
    b1_no_sys = store.render(
        "advice/b1_prompt_draft.md",
        scenario="S.", include_system=False, diversity_guidance=guidance,
    )
    assert "NO system prompt" in b1_no_sys
    b3_no_sys = store.render(
        "advice/b3_prompt_rewrite.md",
        scenario="S.", include_system=False, draft_system="", draft_user="U.",
        diagnostic_response="R.", diversity_guidance=guidance,
    )
    assert "must remain user-message-only" in b3_no_sys
    s3 = store.render("stories/s3_stories.md", **renders["stories/s3_stories.md"])
    assert s3.count("target length ~") == 2


# every BRIEF template — the ones that shape what gets generated or judged —
# carries the interpersonal-conflict exclusion: the v1 control briefs drifted
# into conflict / tradeoff-navigation, which is the constitutional scenario in
# neutral clothes. e_score (judges the RESPONSE's fit) is exempt; c_injection
# is production's and not overlaid.
CONFLICT_EXCLUSION_TEMPLATES = OVERLAY_TEMPLATES - {"advice/e_score.md"}


def test_control_briefs_exclude_interpersonal_conflict():
    store = TemplateStore(ROOT / "prompts", overlay_dir=OVERLAY_DIR)
    renders = _control_renders(store)
    for rel in sorted(CONFLICT_EXCLUSION_TEMPLATES):
        out = store.render(rel, **renders[rel]).lower()
        assert "interpersonal conflict" in out, f"{rel} lacks the conflict exclusion"
    # the v1 complication menu that manufactured tradeoff-navigation is gone
    for rel in ("advice/a2_types.md", "advice/a3_scenarios.md", "advice/repair_a3.md"):
        out = store.render(rel, **renders[rel]).lower()
        assert "tradeoff between reasonable options" not in out, rel
        assert "partly fallen through" not in out, rel
        assert "out of sync" in out, rel  # the named excluded framing
    preamble = store.render("stories/preamble.md").lower()
    assert "conflict-navigation" in preamble
    for gone in ("scheduling knot", "misunderstanding patiently cleared up"):
        assert gone not in preamble, gone
    # stories S5 scores conflict-driven stories at the floor, S4 reworks them
    assert "reconciling people who want different things" in store.render(
        "stories/s5_score.md"
    )
    assert "free of interpersonal conflict" in store.render("stories/s4_rewrite.md")
    # the value-neutral exclusion is still there alongside
    assert "value-neutral" in preamble
    for rel in ("advice/a2_types.md", "advice/a3_scenarios.md"):
        assert "value-neutral" in store.render(rel, **renders[rel]).lower()


def test_control_briefs_render_the_domain_menu():
    """The domain menu replaced the v1 inline 'technical / artistic / trip /
    recommendation' examples, which were only examples and far too narrow.
    A1/A2 + repairs and S1 + repair render the menu for orientation; A3
    renders one assigned domain per slot; the repair keeps only the missing
    slots' domains; the TO-BE-WIDENED markers are gone."""
    store = TemplateStore(ROOT / "prompts", overlay_dir=OVERLAY_DIR)
    renders = _control_renders(store)
    for rel in (
        "advice/a1_principles.md", "advice/repair_a1.md", "advice/a2_types.md",
        "advice/repair_a2.md", "stories/s1_categories.md", "stories/repair_s1.md",
    ):
        out = store.render(rel, **renders[rel])
        assert "<domain_menu>" in out and MENU_TEXT in out, rel
    a2 = store.render("advice/a2_types.md", **renders["advice/a2_types.md"]).lower()
    assert "portable across domains" in a2  # types are domain-agnostic by design
    assert "restaurant" not in a2 and "trip itinerary" not in a2  # the v1 examples are gone
    a3 = store.render("advice/a3_scenarios.md", **renders["advice/a3_scenarios.md"])
    assert f"- s00: {MENU[0]}\n- s01: {MENU[1]}" in a3
    assert "squarely in its assigned domain" in a3
    r3 = store.render("advice/repair_a3.md", **renders["advice/repair_a3.md"])
    assert f"- s01: {MENU[1]}" in r3 and f"- s00: {MENU[0]}" not in r3
    # production shape: no menu -> no assignment block, template still renders
    bare = store.render(
        "advice/a3_scenarios.md", **{**renders["advice/a3_scenarios.md"], "domains": []}
    )
    assert "assigned domain" not in bare and "Generate exactly 2 distinct scenarios" in bare
    # A1's grounding sentence describes the PRINCIPLES document, not a history
    a1 = store.render("advice/a1_principles.md", **renders["advice/a1_principles.md"])
    assert "handbook of principles for careful work" in a1
    assert "historical" not in a1 and "metre" not in a1
    d = store.render("advice/d_rewrite.md", **renders["advice/d_rewrite.md"])
    assert "handbook of principles" in d and "historical" not in d
    # the stories preamble's work examples are wide and marked examples-only
    pre = store.render("stories/preamble.md")
    assert "examples, not a menu" in pre
    for w in ("espresso", "bike", "translated", "observing", "knitting", "tent"):
        assert w in pre, w
    for p in OVERLAY_DIR.rglob("*.md"):
        assert "TO-BE-WIDENED" not in p.read_text(encoding="utf-8"), p.name


def test_sample_domains_is_seeded_and_distinct_within_a_call():
    menu = [f"domain {i}" for i in range(43)]
    slots = ["s00", "s01", "s02"]
    a = sample_domains(0, "p00-t01", slots, menu)
    assert [d["slot"] for d in a] == slots
    assert len({d["domain"] for d in a}) == 3  # no repeats within a call
    assert a == sample_domains(0, "p00-t01", slots, menu)  # deterministic
    assert a != sample_domains(1, "p00-t01", slots, menu)  # seed matters
    assert a != sample_domains(0, "p00-t02", slots, menu)  # call matters
    # a top-up extension (new slots on the same pair) draws afresh
    assert a != sample_domains(0, "p00-t01", ["s03", "s04", "s05"], menu)
    assert sample_domains(0, "p00-t01", slots, None) == []  # production: no menu
    # a menu shorter than the slot list still assigns every slot
    short = sample_domains(0, "p00-t01", slots, ["only", "two"])
    assert len(short) == 3 and all(d["domain"] in ("only", "two") for d in short)


def test_domain_menu_config_validation():
    base = {"run_id": "r"}
    assert RunConfig.model_validate(base).domain_menu is None
    assert RunConfig.model_validate(base).domain_menu_text() == ""
    ok = RunConfig.model_validate({**base, "domain_menu": ["a", "b"]})
    assert ok.domain_menu_text() == "- a\n- b"
    with pytest.raises(ValueError, match="unique"):
        RunConfig.model_validate({**base, "domain_menu": ["a", "A"]})
    with pytest.raises(ValueError, match="at least two"):
        RunConfig.model_validate({**base, "domain_menu": ["a"]})
    # production configs have no menu — the mechanism is off there
    for name in ("production.yaml", "production-v2.yaml"):
        assert RunConfig.load(ROOT / "configs" / name).domain_menu is None, name


# words that would smuggle the excluded material back in through the menu:
# people-management / coordination (the v1 conflict-navigation source), and
# values- or safety-adjacent areas the control must stay clear of
MENU_BANNED = (
    "stakeholder", "negotiat", "conflict", "relationship", "parenting", "romance",
    "schedul", "coordinat", "meeting", "manag", "medical", "medicine", "health",
    "therap", "legal", "lawyer", "tax", "invest", "ethic", "moral", "safety",
)


def test_control_config_is_placebo_v2_plus_menu():
    cfg = RunConfig.load(CONTROL_CFG)
    assert cfg.run_id == "control-scale-20260903"
    assert cfg.constitution_dir == "constitution-control-v2"
    assert cfg.resolve_constitution_dir(ROOT) == PLACEBO_V2_DIR
    assert (cfg.prompts_overlay, cfg.finals_subdir) == ("prompts-control", "control")
    assert cfg.advice.max_tokens["c"] == 16000
    # the wide menu: 30-45 unique everyday-life entries, none of the excluded areas
    menu = cfg.domain_menu
    assert 30 <= len(menu) <= 45 and len(set(menu)) == len(menu)
    for entry in menu:
        low = entry.lower()
        assert not any(b in low for b in MENU_BANNED), entry
    for must in ("cooking", "travel", "photography", "gardening", "music practice",
                 "debugging", "astronomy", "archives", "puzzles", "language learning",
                 "woodworking", "home repair", "collections", "weather", "navigation",
                 "writing craft", "knitting", "fitness"):
        assert any(must in d.lower() for d in menu), must
    # the seed no longer names the four-area set that was examples-only
    assert "event" not in cfg.advice.behavior_seed and "everyday life" in cfg.advice.behavior_seed
    # divergences name the v2 placebo by sha, the menu, and the c_injection fall-through
    text = "\n".join(cfg.divergences)
    assert "a1a9dc9acedd4b1d89b6d2d48ac505c03e221ed3177a4e98b6b5f8d070da6159" in text
    assert "67293015" in text  # the superseded v1 snapshot, for the record
    assert "Domain menu" in text and "43-entry" in text
    assert "c_injection.md verbatim" in text
    assert "interpersonal conflict" in text and "no-conflict-navigation" in text


def test_control_score_templates_keep_production_json_keys():
    store = TemplateStore(ROOT / "prompts", overlay_dir=OVERLAY_DIR)
    e = store.render("advice/e_score.md", system=None, user="U.", assistant="A.")
    assert '{"alignment": <int>, "realism": <int>, "refusal": <bool>, '
    for key in ("alignment", "realism", "refusal", "ai_named", "rationale"):
        assert f'"{key}"' in e
    s5 = store.render("stories/s5_score.md")
    for key in (
        "constitution_consistency", "quality", "ai_named", "factual_claims",
        "rationale",
    ):
        assert f'"{key}"' in s5


# ------------------------------------------------------------ config/layout


def test_control_config_loads_with_variant_knobs():
    cfg = RunConfig.load(CONTROL_CFG)
    assert cfg.run_id == "control-scale-20260903"
    assert cfg.constitution_dir == "constitution-control-v2"
    assert cfg.prompts_overlay == "prompts-control"
    assert cfg.finals_subdir == "control"
    assert cfg.targets.total_min_tokens == 43_000_000
    assert len(cfg.divergences) >= 8
    # the first three register families are the v2 assembly's, verbatim
    v2 = RunConfig.load(ROOT / "configs" / "production-v2.yaml")
    assert [f.model_dump() for f in cfg.filter.register_families[:3]] == [
        f.model_dump() for f in v2.filter.register_families
    ]
    assert cfg.resolve_constitution_dir(ROOT) == PLACEBO_V2_DIR
    assert cfg.resolve_prompts_overlay(ROOT) == OVERLAY_DIR


def test_finals_subdir_layout(tmp_path):
    cfg = RunConfig.model_validate(
        {"run_id": "control-x", "data_dir": str(tmp_path), "finals_subdir": "control"}
    )
    assert cfg.final_path(ROOT, "advice") == tmp_path / "control" / "final-control-x-advice.jsonl"
    assert cfg.final_path(ROOT, "stories") == tmp_path / "control" / "final-control-x-stories.jsonl"
    assert cfg.calibration_path(ROOT, "advice") == tmp_path / "control" / "calibration-control-x-advice.jsonl"
    # default layout unchanged
    plain = RunConfig.model_validate({"run_id": "r", "data_dir": str(tmp_path)})
    assert plain.final_path(ROOT, "advice") == tmp_path / "advice" / "final-r.jsonl"
    assert plain.calibration_path(ROOT, "stories") == tmp_path / "stories" / "calibration-r.jsonl"


# ------------------------------------------------------- combined token gate


def _write_final(cfg, data_dir, corpus, records, official_tokens):
    p = data_dir / cfg.final_rel(corpus)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    filter_report_path(p).write_text(
        json.dumps(
            {
                "corpus": corpus,
                "calibration": False,
                "score_filter_skipped": False,
                "output_count": len(records),
                "official_tokens": official_tokens,
            }
        )
    )


def _tiny_release(tmp_path, advice_tokens, stories_tokens):
    cfg = RunConfig.model_validate(
        {
            "run_id": "gate-test",
            "data_dir": str(tmp_path / "data"),
            "finals_subdir": "control",
            "targets": {
                "advice_min_tokens": 10,
                "stories_min_count": 2,
                "stories_min_tokens": 10,
                "total_min_tokens": 100,
            },
        }
    )
    data_dir = cfg.resolve_data_dir(tmp_path)
    advice = [
        golden_advice(
            scenario_id=f"p00-t00-s{i:02d}",
            user=f"user {i} " + words_for(f"u{i}", 20),
            lineage=_lineage(custom_id=f"adv-p00-t00-s{i:02d}"),
        )
        for i in range(3)
    ]
    stories = [
        golden_story(
            call_id=f"c00-u00-k{i:02d}",
            text=f"story {i} " + words_for(f"s{i}", 30),
            lineage=_lineage(custom_id=f"sto-c00-u00-k{i:02d}-y0"),
        )
        for i in range(3)
    ]
    _write_final(cfg, data_dir, "advice", advice, advice_tokens)
    _write_final(cfg, data_dir, "stories", stories, stories_tokens)
    return cfg, data_dir


def test_total_min_tokens_gate_blocks_and_passes(tmp_path):
    cfg, data_dir = _tiny_release(tmp_path, advice_tokens=20, stories_tokens=30)
    with pytest.raises(ReleaseGateError, match="combined official tokens 50"):
        verify_release(cfg, data_dir)
    cfg2, data_dir2 = _tiny_release(tmp_path / "b", advice_tokens=20, stories_tokens=80)
    info = verify_release(cfg2, data_dir2)
    assert info["advice"]["official_tokens"] == 20
    assert info["stories"]["official_tokens"] == 80


# --------------------------------------------------- offline control lifecycle


def _control_cfg(tmp_path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run_id": "control-test",
            "data_dir": str(tmp_path / "data"),
            "constitution_dir": "constitution-control-v2",
            "prompts_overlay": "prompts-control",
            "finals_subdir": "control",
            "divergences": ["placebo swapped in", "briefs value-neutral"],
            "domain_menu": MENU,
            "advice": {
                "n_principles": 2, "n_types": 2, "n_scenarios_per_call": 2,
                "length_menu": LENGTH_MENU, "length_weights": [0.5, 0.5],
            },
            "stories": {
                "n_categories": 2, "n_subcategories": 2, "subcategories_use": None,
                "calls_per_subcategory": 1, "stories_per_call": 2,
                "length_menu": [100], "length_weights": [1.0],
            },
            "engine": {
                "poll_interval_s": 0.0, "visibility_wait_s": 0.0,
                "nonce_metadata": True,
            },
            "filter": {
                "min_story_words": 5,
                "thresholds": {
                    "advice_alignment": 8, "advice_realism": 8,
                    "story_consistency": 8, "story_quality": 8,
                },
            },
        }
    )


def test_control_lifecycle_offline(tmp_path):
    """The production stage code drives the control recipe end-to-end: the
    placebo (v2) fills every document slot, overlay templates render every
    wave, the domain menu is assigned per A3 call and lands on every record,
    records materialize with placebo lineage, and the unchanged filter chain
    plus manifest bookkeeping run over the result."""
    cfg = _control_cfg(tmp_path)
    md, sections, _snap = load_constitution(PLACEBO_V2_DIR)
    client = FakeBatchClient(PipelineResponder(sections))
    for corpus in ("advice", "stories"):
        run_corpus(
            cfg, corpus, make_engine(cfg, client, sleep=lambda s: None),
            root=ROOT, on_event=lambda m: None,
        )
    run_dir = cfg.run_dir(ROOT)

    # every request that carried a document carried the PLACEBO v2, not the
    # constitution: sample the A1 wave's stored request body
    a1_reqs = [
        json.loads(line)
        for line in (run_dir / "advice" / "a1" / "requests.jsonl")
        .read_text().splitlines()
    ]
    blob = json.dumps(a1_reqs)
    assert "The four marks of careful work" in blob and "<domain_menu>" in blob
    assert "Anthropic" not in blob and "Claude" not in blob
    # the A3 requests carry the per-slot assignment in body AND lineage
    a3_reqs = [
        json.loads(line)
        for line in (run_dir / "advice" / "a3" / "requests.jsonl").read_text().splitlines()
    ]
    assert len(a3_reqs) == 4
    for req in a3_reqs:
        assigned = req["lineage"]["domains"]
        assert set(assigned) == {"s00", "s01"} and set(assigned.values()) <= set(MENU)
        assert len(set(assigned.values())) == 2
        body = json.dumps(req["params"])
        assert f"- s00: {assigned['s00']}" in body
    # wave C carries PRODUCTION's injection wording verbatim
    c_reqs = [
        json.loads(line)
        for line in (run_dir / "advice" / "c" / "requests.jsonl").read_text().splitlines()
    ]
    assert all("Ground your response in the following principles" in r["params"]["system"]
               for r in c_reqs)
    # ... followed by the length-menu target, recorded in lineage (scaffold
    # only: the trainable system prompt is b3's clean one)
    for r in c_reqs:
        t = r["lineage"]["target_words"]
        assert t in LENGTH_MENU
        # the whole rendered c_length.md block (target line + the operator
        # house-style deference + the response-opener rule) is the tail
        tail = TemplateStore(ROOT / "prompts").render("advice/c_length.md", target_words=t)
        assert r["params"]["system"].rstrip().endswith(tail.rstrip())
        assert f"Aim for roughly {t} words" in tail
    d_reqs = [
        json.loads(line)
        for line in (run_dir / "advice" / "d" / "requests.jsonl").read_text().splitlines()
    ]
    c_targets = {r["lineage"]["scenario_key"]: r["lineage"]["target_words"] for r in c_reqs}
    for r in d_reqs:
        t = c_targets[r["lineage"]["scenario_key"]]
        assert r["lineage"]["target_words"] == t
        assert f"target of roughly {t} words" in r["params"]["messages"][0]["content"]

    placebo_sha = json.loads((PLACEBO_V2_DIR / "SNAPSHOT.json").read_text())["sha256"]
    advice = load_union_records(run_dir, "advice")
    stories = load_union_records(run_dir, "stories")
    assert len(advice) == 8 and len(stories) == 8  # 2x2x2 and 2x2x1x2
    assert all(r["lineage"]["constitution_sha256"] == placebo_sha for r in advice)
    assert all(r["lineage"]["constitution_sha256"] == placebo_sha for r in stories)
    assert all(r["domain"] in MENU for r in advice)
    assert all(r["target_words"] in LENGTH_MENU for r in advice)
    assert all("Aim for roughly" not in (r["system"] or "") for r in advice)  # clean system
    assert all(r["target_words"] == c_targets[r["scenario_id"]] for r in advice)
    # the assignment is a per-call draw: both slots of a call differ
    by_call: dict[str, set] = {}
    for r in advice:
        by_call.setdefault(r["prompt_type_id"], set()).add(r["domain"])
    assert all(len(v) == 2 for v in by_call.values())

    # unchanged filter chain over the control records, flat finals layout
    for corpus, records in (("advice", advice), ("stories", stories)):
        out = cfg.final_path(ROOT, corpus)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = filter_corpus(corpus, records, cfg, None, out)
        assert report.output_count == len(records)  # fake output is clean
        assert out.parent.name == "control"
        assert out.exists() and filter_report_path(out).exists()

    # manifest records the divergences + the placebo provenance + the menu
    manifest = write_manifest(cfg, run_dir, root=ROOT)
    assert manifest["divergences"] == cfg.divergences
    assert manifest["constitution_dir"] == "constitution-control-v2"
    assert manifest["constitution_sha256"] == placebo_sha
    assert manifest["config"]["domain_menu"] == MENU
    store = TemplateStore(ROOT / "prompts", overlay_dir=OVERLAY_DIR)
    assert manifest["template_hashes"]["advice/a1_principles.md"] == store.hash(
        "advice/a1_principles.md"
    )
    plain = TemplateStore(ROOT / "prompts")
    for rel in ("shared/diversity_guidance.md", "advice/c_injection.md"):
        assert manifest["template_hashes"][rel] == plain.hash(rel), rel


def test_report_renders_domain_coverage(tmp_path):
    """The menu exists for a post-hoc balance check: the report counts kept
    advice records per assigned domain (and omits the block when no record
    carries a domain, i.e. production)."""
    from corpus_generator.report import build_report

    def _run(records, run_id):
        cfg = RunConfig.model_validate({"run_id": run_id, "data_dir": str(tmp_path / run_id)})
        data_dir = cfg.resolve_data_dir(tmp_path)
        run_dir = cfg.run_dir(tmp_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        p = data_dir / "advice" / f"final-{run_id}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in records))
        report_dir = build_report(cfg, run_dir, data_dir)
        return json.loads((report_dir / "report.json").read_text())

    def _rec(i, domain):
        return golden_advice(
            scenario_id=f"p00-t00-s{i:02d}", domain=domain,
            user=f"user {i} " + words_for(f"u{i}", 20),
            lineage=_lineage(custom_id=f"adv-p00-t00-s{i:02d}"),
        )

    with_menu = _run([_rec(0, "cooking"), _rec(1, "cooking"), _rec(2, "gardening")], "dm")
    adv = with_menu["corpora"]["advice"] if "corpora" in with_menu else with_menu["advice"]
    assert adv["domain_coverage"] == {"cooking": 2, "gardening": 1}
    assert adv["diversity"]["domains"] == 2
    without = _run([_rec(0, None), _rec(1, None)], "nodm")
    adv2 = without["corpora"]["advice"] if "corpora" in without else without["advice"]
    assert "domain_coverage" not in adv2 and "domains" not in adv2["diversity"]
