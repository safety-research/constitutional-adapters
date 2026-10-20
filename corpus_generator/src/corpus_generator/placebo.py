"""The VALUE-NEUTRAL placebo document — the control corpus's stand-in for
the constitution — in the frozen constitution-snapshot format.

The control corpus isolates house register/style from constitutional values:
every pipeline slot that carries the constitution (A1 grounding + section
index, A2/A3/C/D section injections, E and S1-S5 full-document blocks)
instead carries the placebo ``constitution-control-v2/``: a PRINCIPLES
document of careful practical method across many crafts (``placebo.md``, 42
sections / 33 leaf spans), authored as markdown directly in the snapshot
convention. (A first-generation placebo, a descriptive history of the
metre, was superseded 2026-09-02 — a history document cannot naturally
furnish *principles*, which is what the A1 stage asks for — and is no
longer in the repo.) ``constitution.load`` reads the slot name
``constitution.md``, so the directory carries a symlink
``constitution.md -> placebo.md``; the honest filename stays canonical for
humans and the review tooling. :func:`section_markdown` re-derives the
section index from the markdown with the same span convention the authoring
script used (heading line start -> next same-or-higher heading, levels 2-4),
and :func:`verify_placebo` proves the frozen ``sections.json`` matches — the
repo-side integrity check for a document whose build script lives outside
the repo.

The directory matches the frozen constitution snapshot format exactly
(constitution.md + sections.json + SNAPSHOT.json), so ``constitution.load``
verifies the sha256 and serves it unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from .constitution import Section, SnapshotError, _slugify, load

_HEADING_RE = re.compile(r"^(#{2,4}) (.+)$")


def section_markdown(markdown: str) -> list[Section]:
    """Section index of a markdown document in the snapshot convention:
    blocks separated by blank lines, ``##``/``###``/``####`` heading lines,
    each section spanning from its heading line to the next heading of the
    same or higher level (so a chapter's span contains its subsections)."""
    blocks = markdown.split("\n\n")
    heads: list[tuple[int, str, str, int]] = []  # (level, title, id, offset)
    seen: set[str] = set()
    offset = 0
    for blk in blocks:
        m = _HEADING_RE.match(blk.rstrip("\n"))
        if m:
            level = len(m.group(1))
            title = m.group(2)
            sec_id = _slugify(title)
            while sec_id in seen:
                sec_id += "-2"
            seen.add(sec_id)
            heads.append((level, title, sec_id, offset))
        offset += len(blk) + 2
    sections: list[Section] = []
    for i, (level, title, sec_id, start) in enumerate(heads):
        end = len(markdown)
        for lvl2, _t, _i, start2 in heads[i + 1 :]:
            if lvl2 <= level:
                end = start2
                break
        sections.append(Section(id=sec_id, title=title, level=level, start=start, end=end))
    return sections


def leaf_sections(sections: list[Section]) -> list[Section]:
    """Sections that contain no deeper section (the spans A2/A3/C/D inject
    when A1 grounds a principle at the finest level)."""
    return [
        s for s in sections
        if not any(
            o.level > s.level and o.start >= s.start and o.end <= s.end and o is not s
            for o in sections
        )
    ]


def verify_placebo(placebo_dir: Path) -> dict:
    """Load + verify a placebo snapshot dir (sha check via ``constitution.load``)
    AND prove its frozen sections.json is exactly what the markdown implies.
    Raises SnapshotError on any mismatch; returns a small summary."""
    placebo_dir = Path(placebo_dir)
    markdown, frozen, snapshot = load(placebo_dir)
    derived = section_markdown(markdown)
    if [asdict(s) for s in derived] != [asdict(s) for s in frozen]:
        raise SnapshotError(
            f"{placebo_dir}/sections.json does not match the section index "
            f"derived from its markdown ({len(frozen)} frozen vs {len(derived)} derived)"
        )
    for s in frozen:
        if not markdown[s.start:].startswith("#" * s.level + " " + s.title):
            raise SnapshotError(f"section {s.id!r} does not start at its heading line")
    if snapshot.get("n_sections") != len(frozen):
        raise SnapshotError("SNAPSHOT.json n_sections disagrees with sections.json")
    return {
        "sha256": snapshot["sha256"],
        "n_sections": len(frozen),
        "n_leaf_sections": len(leaf_sections(frozen)),
        "words": snapshot.get("words"),
    }

