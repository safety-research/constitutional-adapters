"""Coverage statistics over tag rows (section_coverage.py): record-counted
views, chapter roll-ups, cluster coverage, concentration, per-record counts,
and the lineage-vs-content agreement."""

from __future__ import annotations

from corpus_generator import section_coverage as sc
from corpus_generator.constitution import Section
from corpus_generator.section_tags import section_tree

SECS = [
    Section("ov", "Overview", 2, 0, 1), Section("cv", "Core values", 3, 0, 1),
    Section("bh", "Being helpful", 2, 0, 1), Section("gh", "Genuine helpfulness", 3, 0, 1),
    Section("bal", "Balancing", 3, 0, 1),
    Section("be", "Being ethical", 2, 0, 1), Section("hon", "Being honest", 3, 0, 1),
]
ENTRIES = section_tree(SECS)
IDS = [e.id for e in ENTRIES]


def row(rid, primary, secondary=(), none=False, conf=0.8, status="ok", lineage=None):
    r = {"id": rid, "status": status, "primary": list(primary), "secondary": list(secondary),
         "none": none, "confidence": conf}
    if lineage:
        r["lineage_section_id"] = lineage
    return r


ROWS = [
    row("a", ["gh", "hon"], ["bal"], lineage="hon"),
    row("b", ["hon"], [], lineage="hon"),
    row("c", ["bal", "gh", "cv"], ["hon", "ov"], lineage="cv"),
    row("d", [], [], none=True, conf=0.6, lineage="be"),
    row("e", [], [], status="refusal", conf=None, lineage="hon"),
]


def test_section_counts_three_views_keep_zeros():
    prim = sc.section_counts(ROWS, IDS, "primary")
    assert prim == {"ov": 0, "cv": 1, "bh": 0, "gh": 2, "bal": 1, "be": 0, "hon": 2}
    anyv = sc.section_counts(ROWS, IDS, "any")
    assert anyv["bal"] == 2 and anyv["ov"] == 1 and anyv["hon"] == 3
    lead = sc.section_counts(ROWS, IDS, "lead")
    assert lead == {"ov": 0, "cv": 0, "bh": 0, "gh": 1, "bal": 1, "be": 0, "hon": 1}
    assert sc.zero_sections(prim) == ["ov", "bh", "be"]
    assert sc.under_represented(prim, 4, 0.30) == ["ov", "cv", "bh", "bal", "be"]


def test_chapter_rollup_counts_records_not_tags():
    ch = sc.chapter_counts(ROWS, ENTRIES, "primary")
    # record a touches bh (gh) and be (hon); c touches bh twice (bal, gh) but counts once
    assert ch == {"ov": 1, "bh": 2, "be": 2}
    assert sc.chapter_counts(ROWS, ENTRIES, "lead") == {"ov": 0, "bh": 2, "be": 1}
    lin = [{"section_id": "hon"}, {"section_id": "cv"}, {"section_id": "bh"}]
    assert sc.lineage_chapter_counts(lin, ENTRIES) == {"ov": 1, "bh": 1, "be": 1}


def test_cluster_coverage_and_none_rate():
    c = sc.cluster_coverage(ROWS, ENTRIES, "bh")
    assert c["members"] == ["bh", "gh", "bal"] and c["n"] == 4
    assert c["primary"] == 2 and c["any"] == 2 and c["lead"] == 2  # b has no cluster tag; d is none
    assert c["per_section_primary"] == {"bh": 0, "gh": 2, "bal": 1}
    assert sc.none_rate(ROWS) == {"n": 4, "none": 1, "rate": 0.25}


def test_top2_and_per_record_and_confidence():
    t2 = sc.top2_concentration(ROWS, IDS)
    assert [k for k, _ in t2["lead_top2"]] == ["gh", "bal"] and t2["lead_top2_share"] == 0.5
    assert {k for k, _ in t2["primary_top2"]} == {"gh", "hon"} and t2["primary_top2_records_share"] == 0.75
    pr = sc.sections_per_record(ROWS)
    assert pr["primary"] == {"0": 1, "1": 1, "2": 1, "3": 1} and pr["total"]["5"] == 1 and pr["mean_primary"] == 1.5
    cf = sc.confidence_summary(ROWS)
    assert cf["n"] == 4 and cf["min"] == 0.6 and cf["max"] == 0.8


def test_lineage_agreement_and_confusion():
    ag = sc.lineage_agreement(ROWS, ENTRIES)
    assert ag["n"] == 4  # the refusal row is not ok
    assert ag["exact_lead"] == 1 and ag["in_primary"] == 3 and ag["in_any"] == 3
    # only b: a leads gh (helpful) from hon (ethical); c leads bal (helpful) from cv (overview); d has no lead
    assert ag["same_chapter_lead"] == 1
    assert list(ag["confusion"]) == ["cv", "be", "hon"]  # constitution order
    assert ag["confusion"]["hon"] == {"gh": 1, "hon": 1}
    assert ag["confusion"]["be"] == {"(none)": 1}
