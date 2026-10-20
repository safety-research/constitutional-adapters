"""No item of a bulleted or numbered list in a README or a docs/ page ends with a semicolon.

A list item is a fragment (no closing punctuation) or a whole sentence (a period); a semicolon at its end reads as a sentence the list
cut in pieces. The scan is wrap-aware: an item's LAST line is the one checked, its continuation lines (indented under the marker) included,
so a wrapped item whose second line ends in `;` is caught as a one-line item is. Fenced code is not prose and is skipped. Every README*.md in
the repository and every page under docs/ is read (hidden directories, nested worktrees and caches apart). docs/registry/vocabulary.md is
generated (`registry docs vocabulary`, src/sisteer/registry/docsgen.py): a hit there is fixed in the generator, never by hand."""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+\S")
FENCE = re.compile(r"^\s*(```|~~~)")
SKIP_DIRS = {"worktrees", "node_modules", "__pycache__"}


def pages() -> list:
    out = []
    for d, dirs, files in os.walk(REPO):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in SKIP_DIRS)
        for f in sorted(files):
            rel = Path(d, f).relative_to(REPO)
            if f.endswith(".md") and (f.lower().startswith("readme") or rel.parts[0] == "docs"):
                out.append(rel)
    return out


def list_items(text: str) -> list:
    """(first line, last line, the last line's text) of every list item outside fenced code."""
    items, cur, fence = [], None, False
    for i, ln in enumerate(text.split("\n"), 1):
        if FENCE.match(ln):
            fence = not fence
            if cur:
                items.append(cur); cur = None
            continue
        if fence:
            continue
        m = ITEM.match(ln)
        if m:
            if cur:
                items.append(cur)
            cur = [i, i, ln, len(m.group(1))]
            continue
        if cur is None:
            continue
        indent = len(ln) - len(ln.lstrip())
        if not ln.strip() or ln.lstrip().startswith(("#", "|")) or indent <= cur[3]:
            items.append(cur); cur = None
            continue
        cur[1], cur[2] = i, ln
    if cur:
        items.append(cur)
    return [(a, b, last) for a, b, last, _ in items]


def test_the_scan_reads_a_wrapped_item_by_its_last_line_and_skips_fenced_code():
    text = "- one;\n- two wraps\n  onto a second line;\n- three.\n\n```bash\n- not an item;\n```\n1. numbered;\n"
    assert [(a, b) for a, b, last in list_items(text) if last.rstrip().endswith(";")] == [(1, 1), (2, 3), (9, 9)]


def test_no_list_item_in_a_readme_or_a_docs_page_ends_with_a_semicolon():
    found = pages()
    assert Path("README.md") in found and Path("reproduce/README.md") in found and Path("docs/registry/vocabulary.md") in found, found[:10]
    hits = [f"{rel}:{a}-{b}: {last.strip()[-80:]}" for rel in found for a, b, last in list_items((REPO / rel).read_text(encoding="utf-8"))
            if last.rstrip().endswith(";")]
    assert not hits, "list items ending with a semicolon:\n" + "\n".join(hits)
