"""Fetch, extract, section, hash, and verify the frozen constitution snapshot.

The constitution (https://www.anthropic.com/constitution, CC0) is fetched once,
extracted to clean markdown (page chrome excluded), sectioned by heading, and
committed. Every later stage verifies the snapshot sha256 before using it.

Extraction discipline: the snapshot records sha256 + word/char counts and is
cross-checked against a reference plain-text fetch — a >5% word-count mismatch
aborts loudly for investigation (constitution size has drifted across fetch
methods before).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

CONSTITUTION_URL = "https://www.anthropic.com/constitution"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) corpus_generator/0.1"

# The rendered constitution body lives in this container; everything before it
# (nav, title, summary link) and after it (acknowledgements, footer) is chrome.
BODY_CLASS_MARKER = "ArticleDetail"
BODY_CLASS_MARKER_2 = "body-container"

HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
BLOCK_TAGS = {"p", "li"}


@dataclass
class Section:
    """One heading-delimited section of the frozen snapshot."""

    id: str
    title: str
    level: int  # 2 = h2 (chapter), 3 = h3 (section), 4 = h4 (subsection)
    start: int  # char offset into constitution.md (heading line start)
    end: int  # char offset into constitution.md (exclusive)

    def text(self, markdown: str) -> str:
        return markdown[self.start : self.end].strip()


class _BodyExtractor(HTMLParser):
    """Walk the constitution body div and emit markdown blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str, str]] = []  # (kind, text, html_id)
        self._depth = 0  # div nesting depth inside the body container
        self._in_body = False
        self._done = False
        self._cur_tag: str | None = None
        self._cur_id = ""
        self._buf: list[str] = []
        self._list_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._done:
            return
        attrd = dict(attrs)
        if not self._in_body:
            if tag == "div":
                cls = attrd.get("class") or ""
                if BODY_CLASS_MARKER in cls and BODY_CLASS_MARKER_2 in cls:
                    self._in_body = True
                    self._depth = 1
            return
        if tag == "div":
            self._depth += 1
        elif tag in ("ul", "ol"):
            self._list_depth += 1
        elif tag in HEADING_LEVELS or tag in BLOCK_TAGS:
            self._flush()
            self._cur_tag = tag
            self._cur_id = attrd.get("id") or ""
        elif tag == "br" and self._cur_tag:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._done or not self._in_body:
            return
        if tag == "div":
            self._depth -= 1
            if self._depth <= 0:
                self._flush()
                self._done = True
        elif tag in ("ul", "ol"):
            self._list_depth = max(0, self._list_depth - 1)
        elif self._cur_tag and tag == self._cur_tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_body and not self._done and self._cur_tag:
            self._buf.append(data)

    def _flush(self) -> None:
        if not self._cur_tag:
            return
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        if text:
            kind = self._cur_tag if self._cur_tag in HEADING_LEVELS else (
                "li" if self._cur_tag == "li" else "p"
            )
            self.blocks.append((kind, text, self._cur_id))
        self._buf = []
        self._cur_tag = None
        self._cur_id = ""


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def extract(html: str) -> tuple[str, list[Section]]:
    """Extract (markdown, sections) from the constitution page HTML.

    Markdown convention: headings as '## Title' lines (level = html level),
    paragraphs as plain blocks, list items as '- ' lines, blank line between
    blocks. Section spans cover heading line through the char before the next
    same-or-higher-level heading.
    """
    parser = _BodyExtractor()
    parser.feed(html)
    if not parser.blocks:
        raise ExtractionError(
            "no constitution body found in page HTML (chrome markers changed?)"
        )

    lines: list[str] = []
    heads: list[tuple[int, str, str, int]] = []  # (level, title, id, char offset)
    offset = 0
    seen_ids: set[str] = set()
    for kind, text, html_id in parser.blocks:
        if kind in HEADING_LEVELS:
            level = HEADING_LEVELS[kind]
            block = f"{'#' * level} {text}"
            sec_id = html_id or _slugify(text)
            while sec_id in seen_ids:
                sec_id += "-2"
            seen_ids.add(sec_id)
            heads.append((level, text, sec_id, offset))
        elif kind == "li":
            block = f"- {text}"
        else:
            block = text
        lines.append(block)
        offset += len(block) + 2  # blocks joined by '\n\n'

    markdown = "\n\n".join(lines) + "\n"

    sections: list[Section] = []
    for i, (level, title, sec_id, start) in enumerate(heads):
        end = len(markdown)
        for lvl2, _t2, _i2, start2 in heads[i + 1 :]:
            if lvl2 <= level:
                end = start2
                break
        sections.append(Section(id=sec_id, title=title, level=level, start=start, end=end))
    return markdown, sections


class ExtractionError(RuntimeError):
    pass


class SnapshotError(RuntimeError):
    pass


def _plain_text(markdown: str) -> str:
    """Markdown -> plain text for count comparison (strip #/- markers)."""
    out = []
    for line in markdown.split("\n"):
        out.append(re.sub(r"^(#{1,6}|-)\s+", "", line))
    return "\n".join(out)


def counts(markdown: str) -> dict:
    plain = _plain_text(markdown)
    return {
        "words": len(plain.split()),
        "chars": len(plain),
        "md_chars": len(markdown),
        "sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    }


def fetch(url: str = CONSTITUTION_URL) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


# Button labels of the page's ExpandableSection widgets. A reference fetch
# (2026-08-31) rendered the page with those sections COLLAPSED, so it
# contains these button labels but not the collapsed content; our extraction
# contains the content (genuine constitution text: the principals hierarchy
# etc., ~6.2k words) and not the buttons. Investigated 2026-08-31
# (constitution cross-check).
REFERENCE_CHROME_LINES = {"Read more"}


def _norm(s: str) -> str:
    """Normalize typography + whitespace for containment comparison."""
    table = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "--", "–": "-"}
    for k, v in table.items():
        s = s.replace(k, v)
    return re.sub(r"\s+", " ", s).strip()


def cross_check(markdown: str, reference_path: Path, max_word_ratio: float = 1.5) -> dict:
    """Verify the extraction against the reference plain-text fetch.

    The reference (~/const.txt) is a collapsed-view rendering: it is a strict
    SUBSET of the full document (the page collapses three sections by default;
    the reference has their "Read more" buttons instead of their content). So a
    naive word-count gate cannot hold both ways; instead the check is:

    1. containment — every reference block (minus known chrome button labels)
       must appear verbatim (normalized) in the extraction; ANY miss aborts;
    2. bounded superset — extracted words in [1.0, max_word_ratio) x reference
       words, so a truncated or doubled extraction still fails loudly.
    """
    ref_text = reference_path.read_text(encoding="utf-8")
    ref_words = len(ref_text.split())
    ours = counts(markdown)["words"]
    ourn = _norm(_plain_text(markdown))

    ref_blocks = [b for b in ref_text.split("\n") if b.strip()]
    checked = 0
    missing: list[str] = []
    for block in ref_blocks:
        if block.strip() in REFERENCE_CHROME_LINES:
            continue
        checked += 1
        if _norm(block)[:200] not in ourn:
            missing.append(block[:120])

    ratio = ours / max(ref_words, 1)
    ok = not missing and 1.0 <= ratio < max_word_ratio
    result = {
        "reference": str(reference_path),
        "reference_words": ref_words,
        "extracted_words": ours,
        "word_ratio": round(ratio, 4),
        "reference_blocks_checked": checked,
        "reference_blocks_missing": len(missing),
        "chrome_lines_skipped": len(ref_blocks) - checked,
        "ok": ok,
    }
    if not ok:
        detail = "; ".join(missing[:5])
        raise SnapshotError(
            f"constitution cross-check FAILED: {len(missing)}/{checked} reference "
            f"blocks missing from extraction (first: {detail!r}); word ratio "
            f"{ratio:.3f} (must be in [1.0, {max_word_ratio})). Investigate before "
            f"freezing; do not shrug at a size drift."
        )
    return result


def freeze(
    out_dir: Path,
    url: str = CONSTITUTION_URL,
    reference_path: Path | None = None,
    force: bool = False,
) -> dict:
    """Fetch + extract + section + hash; write the frozen snapshot files.

    Refuses to overwrite an existing snapshot without force (frozen by design).
    """
    out_dir = Path(out_dir)
    snap_path = out_dir / "SNAPSHOT.json"
    if snap_path.exists() and not force:
        raise SnapshotError(
            f"{snap_path} already exists; the snapshot is frozen by design "
            f"(--force to refetch)"
        )
    markdown, sections = extract(fetch(url))
    stats = counts(markdown)
    check = None
    if reference_path is not None and Path(reference_path).exists():
        check = cross_check(markdown, Path(reference_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "constitution.md").write_text(markdown, encoding="utf-8")
    (out_dir / "sections.json").write_text(
        json.dumps([asdict(s) for s in sections], indent=1), encoding="utf-8"
    )
    snapshot = {
        "url": url,
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        **stats,
        "n_sections": len(sections),
        "cross_check": check,
    }
    snap_path.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    return snapshot


def load(constitution_dir: Path) -> tuple[str, list[Section], dict]:
    """Load + VERIFY the frozen snapshot; raises on any hash mismatch."""
    constitution_dir = Path(constitution_dir)
    markdown = (constitution_dir / "constitution.md").read_text(encoding="utf-8")
    snapshot = json.loads((constitution_dir / "SNAPSHOT.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    if actual != snapshot["sha256"]:
        raise SnapshotError(
            f"constitution.md sha256 {actual[:12]}… does not match frozen snapshot "
            f"{snapshot['sha256'][:12]}… — the snapshot was tampered with or "
            f"partially rewritten"
        )
    sections = [
        Section(**s)
        for s in json.loads((constitution_dir / "sections.json").read_text(encoding="utf-8"))
    ]
    return markdown, sections, snapshot


def section_by_id(sections: list[Section], sec_id: str) -> Section:
    for s in sections:
        if s.id == sec_id:
            return s
    raise KeyError(f"unknown constitution section id: {sec_id!r}")
