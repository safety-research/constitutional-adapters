"""Constitution-section COVERAGE statistics from the content tags
(``section_tags.py`` output) — pure functions over rows, so the report script
and the corpus-showcase builder read one computation.

Three views of "coverage" per section, each a count of RECORDS:

- ``primary``  — records with the section in their primary list;
- ``any``      — records with the section in primary OR secondary;
- ``lead``     — records whose FIRST primary tag (the tagger orders primary by
                 importance) is the section: one section per record, the
                 view that is comparable to the advice construction lineage
                 (also one section per record).

Concentration and cluster coverage are stated on records, never on tag mass,
because a record carries up to three primary tags and tag mass would
double-count it.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .section_tags import IndexEntry

HELPFULNESS_CHAPTER = "being-helpful"


def ok_rows(rows: Iterable[dict]) -> list[dict]:
    return [r for r in rows if r.get("status") == "ok"]


def section_counts(rows: Iterable[dict], ids: list[str], view: str = "primary") -> dict[str, int]:
    """Records per section under one view; every index id present (zeros kept)."""
    counts = Counter()
    for r in ok_rows(rows):
        if view == "primary":
            tags = set(r["primary"])
        elif view == "any":
            tags = set(r["primary"]) | set(r["secondary"])
        elif view == "lead":
            tags = set(r["primary"][:1])
        else:
            raise ValueError(f"unknown view {view!r}")
        for t in tags:
            counts[t] += 1
    return {sid: counts.get(sid, 0) for sid in ids}


def lineage_counts(records: Iterable[dict], ids: list[str], key: str = "section_id") -> dict[str, int]:
    counts = Counter(r[key] for r in records if r.get(key))
    return {sid: counts.get(sid, 0) for sid in ids}


def shares(counts: dict[str, int], n: int) -> dict[str, float]:
    return {k: (v / n if n else 0.0) for k, v in counts.items()}


def zero_sections(counts: dict[str, int]) -> list[str]:
    return [k for k, v in counts.items() if v == 0]


def under_represented(counts: dict[str, int], n: int, floor_share: float) -> list[str]:
    """Sections (constitution order) below ``floor_share`` of records, zeros included."""
    return [k for k, v in counts.items() if n and v / n < floor_share]


def top_k(counts: dict[str, int], k: int = 2) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], list(counts).index(kv[0])))[:k]


def top2_concentration(rows: Iterable[dict], ids: list[str]) -> dict:
    """Share of records whose LEAD section is one of the two most common lead
    sections (one section per record, so shares add up), plus the same for
    the primary view stated as 'records touching either of the top-2 primary
    sections'."""
    rows = ok_rows(rows)
    n = len(rows)
    lead = section_counts(rows, ids, "lead")
    lead_top = top_k(lead, 2)
    prim = section_counts(rows, ids, "primary")
    prim_top = top_k(prim, 2)
    prim_top_ids = {k for k, _ in prim_top}
    touching = sum(1 for r in rows if set(r["primary"]) & prim_top_ids)
    return {
        "lead_top2": lead_top,
        "lead_top2_share": (sum(v for _, v in lead_top) / n) if n else 0.0,
        "primary_top2": prim_top,
        "primary_top2_records_share": (touching / n) if n else 0.0,
    }


def chapter_of(entries: list[IndexEntry]) -> dict[str, str]:
    return {e.id: e.chapter for e in entries}


def chapter_counts(rows: Iterable[dict], entries: list[IndexEntry], view: str = "primary") -> dict[str, int]:
    """Records with at least one tag (under ``view``) in each chapter."""
    ch = chapter_of(entries)
    chapters = [e.id for e in entries if e.parent is None]
    counts = Counter()
    for r in ok_rows(rows):
        tags = r["primary"] if view == "primary" else (
            r["primary"] + r["secondary"] if view == "any" else r["primary"][:1])
        for c in {ch[t] for t in tags}:
            counts[c] += 1
    return {c: counts.get(c, 0) for c in chapters}


def lineage_chapter_counts(records: Iterable[dict], entries: list[IndexEntry]) -> dict[str, int]:
    ch = chapter_of(entries)
    chapters = [e.id for e in entries if e.parent is None]
    counts = Counter(ch[r["section_id"]] for r in records if r.get("section_id"))
    return {c: counts.get(c, 0) for c in chapters}


def cluster_coverage(rows: Iterable[dict], entries: list[IndexEntry], chapter: str = HELPFULNESS_CHAPTER) -> dict:
    """Records exercising a chapter (the chapter id itself or any section under
    it), primary-only and any, plus the per-section breakdown inside it."""
    rows = ok_rows(rows)
    n = len(rows)
    members = [e.id for e in entries if e.chapter == chapter]
    mem = set(members)
    prim = sum(1 for r in rows if set(r["primary"]) & mem)
    anyc = sum(1 for r in rows if (set(r["primary"]) | set(r["secondary"])) & mem)
    lead = sum(1 for r in rows if r["primary"][:1] and r["primary"][0] in mem)
    per = section_counts(rows, members, "primary")
    return {
        "chapter": chapter, "members": members, "n": n,
        "primary": prim, "primary_share": prim / n if n else 0.0,
        "any": anyc, "any_share": anyc / n if n else 0.0,
        "lead": lead, "lead_share": lead / n if n else 0.0,
        "per_section_primary": per,
    }


def none_rate(rows: Iterable[dict]) -> dict:
    rows = ok_rows(rows)
    n = len(rows)
    k = sum(1 for r in rows if r["none"])
    return {"n": n, "none": k, "rate": k / n if n else 0.0}


def sections_per_record(rows: Iterable[dict]) -> dict:
    rows = ok_rows(rows)
    p = Counter(len(r["primary"]) for r in rows)
    s = Counter(len(r["secondary"]) for r in rows)
    t = Counter(len(r["primary"]) + len(r["secondary"]) for r in rows)
    n = len(rows)
    return {
        "n": n,
        "primary": {str(k): p.get(k, 0) for k in range(0, 4)},
        "secondary": {str(k): s.get(k, 0) for k in range(0, 4)},
        "total": {str(k): t.get(k, 0) for k in range(0, 7)},
        "mean_primary": (sum(len(r["primary"]) for r in rows) / n) if n else 0.0,
        "mean_total": (sum(len(r["primary"]) + len(r["secondary"]) for r in rows) / n) if n else 0.0,
    }


def confidence_summary(rows: Iterable[dict]) -> dict:
    vals = sorted(r["confidence"] for r in ok_rows(rows) if r.get("confidence") is not None)
    if not vals:
        return {"n": 0}
    q = lambda f: vals[min(len(vals) - 1, int(f * len(vals)))]
    return {"n": len(vals), "mean": sum(vals) / len(vals), "p10": q(0.10), "p50": q(0.50), "p90": q(0.90),
            "min": vals[0], "max": vals[-1]}


def lineage_agreement(rows: Iterable[dict], entries: list[IndexEntry], key: str = "lineage_section_id") -> dict:
    """Advice: the construction-lineage section vs the content tags.

    - exact_lead: lineage == the first primary tag
    - in_primary: lineage anywhere in primary
    - in_any:     lineage in primary or secondary
    - same_chapter_lead: lineage's chapter == lead tag's chapter
    - confusion: per lineage section (constitution order), the distribution of
      lead content sections (counts), so the pattern of disagreement is visible
    """
    rows = [r for r in ok_rows(rows) if r.get(key)]
    n = len(rows)
    ch = chapter_of(entries)
    order = [e.id for e in entries]
    exact = in_prim = in_any = same_ch = 0
    confusion: dict[str, Counter] = {}
    for r in rows:
        lin = r[key]
        lead = r["primary"][0] if r["primary"] else None
        exact += lead == lin
        in_prim += lin in r["primary"]
        in_any += lin in r["primary"] or lin in r["secondary"]
        same_ch += lead is not None and ch.get(lead) == ch.get(lin)
        confusion.setdefault(lin, Counter())[lead or "(none)"] += 1
    conf_sorted = {
        lin: dict(sorted(c.items(), key=lambda kv: (-kv[1], order.index(kv[0]) if kv[0] in order else 99)))
        for lin, c in sorted(confusion.items(), key=lambda kv: order.index(kv[0]))
    }
    return {
        "n": n,
        "exact_lead": exact, "exact_lead_rate": exact / n if n else 0.0,
        "in_primary": in_prim, "in_primary_rate": in_prim / n if n else 0.0,
        "in_any": in_any, "in_any_rate": in_any / n if n else 0.0,
        "same_chapter_lead": same_ch, "same_chapter_lead_rate": same_ch / n if n else 0.0,
        "confusion": conf_sorted,
    }
