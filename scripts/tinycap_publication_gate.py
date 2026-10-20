#!/usr/bin/env python
"""The publication gate of the math instruments' figure sets (math-eight-arms-extension, plan P5 / §4.7).

PSR arms appear in NO figure — appendices included; the appendix is tables and exports. Two surfaces enforce it,
both READ-ONLY over what an analysis wrote:

* ``check_publication_no_psr(figs_dir, sidecar=...)`` — the FIGURES DIRECTORY gate the two analyses run before their
  W&B / mirror pass: every figure file in the directory must be described by the rows sidecar the analysis wrote beside
  it (``final_<m5t|g8k>_figures_numbers.json``: every plotted number with its report path), and no described entry may
  present a PSR arm id — nor a cell the render declared excluded (``--figure-exclude-arm``), a family it declared excluded
  (``--figure-exclude-family``) or a sampling mode it declared out (``--figure-modes``). A figure without a sidecar record
  cannot be verified and refuses.
* ``check_page(index_html, figs_root=...)`` — the REVIEW PAGE gate (§4.7): every ``<img src="/figs/...">`` the page embeds
  is resolved to its asset under the local mirror root, the asset's rows sidecar is found beside it (the analysis
  sidecar naming the stem, or an ad-hoc figure's own ``<stem>.json``), and the page refuses if any embedded figure
  presents a PSR arm id or has no sidecar at all.

``recut_page`` rewrites a page NON-additively: PSR-bearing embeds are REPLACED by their eight-arm counterparts or
REMOVED (the whole ``<figure>`` block), and the result is handed back through ``check_page`` before anything is published.
A PSR arm is read off the cell grammar (``<si>__<arm>``, arm ``psr_*``): ``constv2-mix-full__psr_ce_all``,
``constv2ctl2-full__psr_ce_all``, ``safety_refusal__psr_kl_all``, ``constv2-mix-full__psr_ce_all_minus_ctl_distilled``.
Nothing here recomputes a number: the gate reads sidecars and pages, and refuses by name.

The page's images are enumerated with an HTML PARSER (``html.parser``), never a regex over ``src="..."`` (code review B1 of
2026-09-12: the regex matched double-quoted, whitespace-free ``src=`` only, so ``<img src='...'>``, ``src = "..."`` and an
``<img data-src="..." src="...">`` were skipped or read the wrong attribute — a page whose every figure was single-quoted
PASSED with zero figures checked). ONE enumeration (:func:`page_images`) serves the gate, the re-cut and the re-cut's own
"still embedded" check, so what is checked is exactly what is rewritten.

RESPONSIVE-IMAGE candidates are REFUSED, not resolved: a browser selects among an ``<img>``'s
``srcset`` candidates and a ``<picture>``'s ``<source>`` elements by viewport and density, so a page may show a figure the
gate never read — a PSR candidate in ``srcset`` survived a re-cut that replaced only ``src``. The ONE scanner
(:class:`_ImageScanner`) records every ``srcset`` attribute, every ``<source>`` and every ``<picture>`` beside the images;
:func:`check_page` refuses each by position and :func:`recut_page` refuses to re-cut a page carrying one (it cannot retarget
candidates it does not rewrite), so no candidate can survive a passing re-cut. Supporting them would mean enumerating,
validating and retargeting every candidate through the same mechanism; until then a page embeds one ``<img src>`` per figure.

The re-cut CLI validates the candidate IN MEMORY before touching the destination (code review B2): the destination is written
atomically only after ``check_page`` PASSED over the candidate text — never over the input page (``--out`` must differ from
it), and never over an existing file unless ``--overwrite`` says so — so a refused re-cut leaves the destination exactly as it
was, or absent.
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, NamedTuple

#: the exit code of a refused publication (3 = not a production result, 4 = the two-venv reports differ, 5 = the mirror
#: differs, in the analyses' runners)
EXIT_PUBLICATION_REFUSED = 6
#: the cell grammar's PSR arm marker: ``<si>__psr_<...>`` (every PSR object cell, the distilled derived one included)
PSR_ARM_MARKER = "__psr_"
SIDECAR_SUFFIX = "_figures_numbers.json"
FIGURE_SUFFIXES: tuple[str, ...] = (".png", ".pdf", ".svg")
PAGE_FIGS_URL_PREFIX = "/figs/"
GATE_PASS, GATE_REFUSED = "PASS", "REFUSED"
RULE = ("no figure presents a PSR arm id, a cell the render excluded, a family it excluded or a sampling mode it left out; "
        "every figure file is described by its rows sidecar (a figure without one cannot be verified and refuses); the "
        "appendix is tables and exports, never a figure; a page embeds one <img src> per figure — responsive candidate "
        "sets (srcset, <source>, <picture>) are refused, never resolved")

_FIGURE_BLOCK = re.compile(r"<figure\b.*?</figure>", re.IGNORECASE | re.DOTALL)
#: the attribute a page's image is read from — ``src`` exactly (``data-src`` and friends are other attributes)
IMG_SRC_ATTR = "src"


class PageImage(NamedTuple):
    """One ``<img>`` of a page as the parser read it: its ``src`` (``None`` for an image with no ``src`` attribute), the
    tag's exact text and its [start, end) offsets in the page — the re-cut rewrites exactly this span."""
    src: str | None
    start: int
    end: int
    tag: str


#: the responsive-image ATTRIBUTE and ELEMENTS the gate refuses: a browser-selected candidate set the gate
#: cannot resolve to one asset
RESPONSIVE_SRCSET_ATTR = "srcset"
RESPONSIVE_ELEMENTS: tuple[str, ...] = ("source", "picture")


class PageUnsupported(NamedTuple):
    """One responsive-image construct the gate refuses: ``what`` names it (``img srcset``, ``source``, ``picture``, ...),
    with the tag's exact text and its [start, end) offsets in the page."""
    what: str
    start: int
    end: int
    tag: str


class _ImageScanner(HTMLParser):
    """Collects every ``<img>`` start tag (``<img ...>`` and ``<img .../>`` alike) with its ``src`` attribute and its
    offsets — and, beside them, every RESPONSIVE construct (:data:`RESPONSIVE_SRCSET_ATTR` on any tag,
    :data:`RESPONSIVE_ELEMENTS`) the gate refuses rather than resolves. Attribute names are case-insensitive and whitespace
    around ``=`` is grammar, both handled by ``html.parser``; entity references in attribute values are unescaped
    (``convert_charrefs``), so ``src`` is the URL the browser resolves."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.images: list[PageImage] = []
        self.unsupported: list[PageUnsupported] = []
        self._line_starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self._line_starts.append(i + 1)

    def _offset(self) -> int:
        line, col = self.getpos()
        return self._line_starts[line - 1] + col

    def _img(self, attrs: list[tuple[str, str | None]]) -> None:
        tag = self.get_starttag_text() or ""
        start = self._offset()
        src = next((v for k, v in attrs if k == IMG_SRC_ATTR), None)
        self.images.append(PageImage(src=src, start=start, end=start + len(tag), tag=tag))

    def _responsive(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        text = self.get_starttag_text() or ""
        start = self._offset()
        if tag in RESPONSIVE_ELEMENTS:                       # the element itself is the refused construct (its srcset implied)
            self.unsupported.append(PageUnsupported(what=f"<{tag}>", start=start, end=start + len(text), tag=text))
        elif any(k == RESPONSIVE_SRCSET_ATTR for k, _v in attrs):
            self.unsupported.append(PageUnsupported(what=f"<{tag} {RESPONSIVE_SRCSET_ATTR}>", start=start, end=start + len(text), tag=text))

    def _start(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            self._img(attrs)
        self._responsive(tag, attrs)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag, attrs)


def _scan(html: str) -> _ImageScanner:
    scanner = _ImageScanner(html)
    scanner.feed(html)
    scanner.close()
    return scanner


def page_images(html: str) -> list[PageImage]:
    """Every ``<img>`` of a page in document order (duplicates kept), read by the HTML parser."""
    return list(_scan(html).images)


def page_unsupported(html: str) -> list[PageUnsupported]:
    """Every RESPONSIVE-IMAGE construct of a page, in document order, from the SAME scanner as
    :func:`page_images`: an ``srcset`` attribute on any tag, a ``<source>``, a ``<picture>`` — each a browser-selected
    candidate set the gate cannot resolve to one asset and therefore refuses."""
    return list(_scan(html).unsupported)


def unsupported_violations(html: str) -> list[str]:
    """The refusal line per responsive construct: what and where, and the remedy (one ``<img src>`` per figure)."""
    return [f"{u.what} at offset {u.start} ({u.tag[:80]!r}): a responsive-image candidate set the gate cannot resolve — a browser may "
            "select a candidate the gate never read; embed one <img src> per figure" for u in page_unsupported(html)]


class PublicationRefused(Exception):
    """The gate refused: ``violations`` names every embedded figure that presents what must not be published."""

    def __init__(self, violations: Sequence[str], *, where: str) -> None:
        self.violations = list(violations)
        self.where = where
        super().__init__(f"{where}: publication refused — " + "; ".join(self.violations))


def is_psr_cell(cell: Any) -> bool:
    """A PSR arm by the cell grammar: the arm token after ``__`` starts with ``psr_``."""
    return isinstance(cell, str) and PSR_ARM_MARKER in cell


def _string_leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield str(k)
            yield from _string_leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _string_leaves(v)


def _segment_pattern(token: str) -> re.Pattern[str]:
    """``token`` as a whole path / label segment (a model key, a cell id or a sampling mode as the analyses spell them in
    ``name`` / ``source``: ``[gemma4e4b/bare/thinking_card]``, ``cells.gemma4e4b/it/none.readings``), never inside a
    longer identifier."""
    return re.compile(r"(?<![A-Za-z0-9_\-])" + re.escape(token) + r"(?![A-Za-z0-9_\-])")


def entry_violations(entry: Any, *, excluded_arms: Sequence[str] = (), excluded_families: Sequence[str] = (),
                     excluded_modes: Sequence[str] = ()) -> list[str]:
    """Why one sidecar entry (or any JSON value) presents what the gate forbids: a PSR arm id in any string leaf, or an
    excluded cell / family / mode as a whole segment of one."""
    leaves = list(_string_leaves(entry))
    out: list[str] = []
    psr = sorted({m.group(0) for s in leaves for m in re.finditer(r"[A-Za-z0-9_\-]*" + re.escape(PSR_ARM_MARKER) + r"[A-Za-z0-9_\-]*", s)})
    for cell in psr:
        out.append(f"presents the PSR arm {cell!r}")
    for label, tokens in (("excluded cell", excluded_arms), ("excluded family", excluded_families), ("excluded mode", excluded_modes)):
        for tok in tokens:
            pat = _segment_pattern(str(tok))
            if any(pat.search(s) for s in leaves):
                out.append(f"presents the {label} {tok!r}")
    return out


def _entry_name(entry: Any, i: int) -> str:
    if isinstance(entry, Mapping) and entry.get("name") is not None:
        return str(entry["name"])
    return f"entry {i}"


def check_sidecar(doc: Mapping[str, Any], *, stems: Sequence[str] | None = None, excluded_arms: Sequence[str] = (),
                  excluded_families: Sequence[str] = (), excluded_modes: Sequence[str] = ()) -> list[str]:
    """The violations of an analysis rows sidecar (``{figures: {stem: [entry, ...]}, ...}``) over ``stems`` (default: every
    stem it describes)."""
    figures = doc.get("figures") if isinstance(doc, Mapping) else None
    if not isinstance(figures, Mapping):
        return ["the sidecar carries no `figures` mapping — not an analysis rows sidecar"]
    out: list[str] = []
    for stem in (list(figures) if stems is None else list(stems)):
        entries = figures.get(stem)
        if entries is None:
            out.append(f"{stem}: not described by the sidecar")
            continue
        # one line per (stem, forbidden thing): the count of entries and the first entry's name — a 264-line refusal reads
        # as noise, a grouped one names what to re-cut
        grouped: dict[str, tuple[int, str]] = {}
        for i, entry in enumerate(entries if isinstance(entries, list) else [entries]):
            for why in entry_violations(entry, excluded_arms=excluded_arms, excluded_families=excluded_families, excluded_modes=excluded_modes):
                n, first = grouped.get(why, (0, _entry_name(entry, i)))
                grouped[why] = (n + 1, first)
        for why, (n, first) in grouped.items():
            out.append(f"{stem}: {n} plotted record(s) {why} (e.g. {first!r})")
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PublicationRefused([f"{path}: unreadable sidecar ({e})"], where=str(path)) from e


def figure_stems(figs_dir: Path) -> list[str]:
    return sorted({p.stem for p in Path(figs_dir).iterdir() if p.suffix in FIGURE_SUFFIXES})


def check_publication_no_psr(figs_dir: Path | str, *, sidecar: str, stems: Sequence[str] | None = None,
                             excluded_arms: Sequence[str] = (), excluded_families: Sequence[str] = (),
                             excluded_modes: Sequence[str] = ()) -> dict[str, Any]:
    """The figures-directory gate (plan P5): PASS record, or :class:`PublicationRefused` naming every violation — a figure
    file the sidecar does not describe, or a described entry presenting a PSR arm / an excluded cell, family or mode.
    ``stems``: the figures the render reports it WROTE (its manifest), checked beside every figure file on disk."""
    d = Path(figs_dir).expanduser()
    side = d / sidecar
    violations: list[str] = []
    on_disk = figure_stems(d) if d.is_dir() else []
    stems = sorted(set(on_disk) | set(stems or ()))
    if not d.is_dir():
        violations.append(f"{d}: no such figures directory")
    elif not side.is_file():
        violations.append(f"{side}: no rows sidecar — the figures cannot be verified")
    else:
        doc = _read_json(side)
        figures = doc.get("figures") if isinstance(doc, Mapping) else None
        described = set(figures) if isinstance(figures, Mapping) else set()
        for stem in stems:
            if stem not in described:
                violations.append(f"{stem}: a figure the sidecar does not describe")
        violations += check_sidecar(doc, stems=[s for s in stems if s in described], excluded_arms=excluded_arms,
                                    excluded_families=excluded_families, excluded_modes=excluded_modes)
    record = {"status": GATE_REFUSED if violations else GATE_PASS, "figs_dir": str(d), "sidecar": str(side), "figures": stems,
              "excluded_arms": list(excluded_arms), "excluded_families": list(excluded_families), "excluded_modes": list(excluded_modes),
              "violations": violations, "rule": RULE}
    if violations:
        raise PublicationRefused(violations, where=str(d))
    return record


# --------------------------------------------------------------------------- #
# review pages                                                                 #
# --------------------------------------------------------------------------- #


def page_figure_srcs(html: str) -> list[str]:
    """Every ``<img>``'s ``src`` of a page, in document order (duplicates kept) — through :func:`page_images`, the ONE
    enumeration the gate and the re-cut share. An image with no ``src`` attribute is not listed here; :func:`check_page`
    refuses it by position (a figure the gate cannot resolve is never silently skipped)."""
    return [im.src for im in page_images(html) if im.src is not None]


def asset_path(src: str, *, figs_root: Path | str, url_prefix: str = PAGE_FIGS_URL_PREFIX) -> Path | None:
    """The local asset a page ``src`` names under the mirror root (``/figs/<artifact>/...`` -> ``<figs_root>/<artifact>/...``);
    ``None`` for a src outside the mirror (an external image is not a figure this gate can read)."""
    if not src.startswith(url_prefix):
        return None
    return Path(figs_root).expanduser() / src[len(url_prefix):]


def sidecar_of(asset: Path) -> tuple[Path | None, str]:
    """The rows sidecar describing ``asset``: an ad-hoc figure's own ``<stem>.json`` beside it, else the analysis sidecar
    in the same directory whose ``figures`` names the stem. ``(None, why)`` when none does."""
    asset = Path(asset)
    own = asset.with_suffix(".json")
    if own.is_file():
        return own, "own"
    if not asset.parent.is_dir():
        return None, f"{asset.parent}: no such directory"
    for side in sorted(asset.parent.glob(f"*{SIDECAR_SUFFIX}")):
        try:
            doc = json.loads(side.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, Mapping) and isinstance(doc.get("figures"), Mapping) and asset.stem in doc["figures"]:
            return side, "analysis"
    return None, f"no rows sidecar beside {asset} describes {asset.stem!r} (no {asset.stem}.json, no *{SIDECAR_SUFFIX} naming it)"


def check_page(index_html: Path | str, *, figs_root: Path | str, url_prefix: str = PAGE_FIGS_URL_PREFIX,
               html: str | None = None) -> dict[str, Any]:
    """The review-page gate (§4.7): every embedded figure resolved, its sidecar found, no PSR arm presented. Returns the
    record (``status``, per-figure ``figures``, ``violations``); raises :class:`PublicationRefused` on any violation."""
    page = Path(index_html).expanduser()
    text = html if html is not None else page.read_text(encoding="utf-8")
    # a responsive candidate set is refused by position before any image is resolved — never skipped, never guessed
    violations: list[str] = unsupported_violations(text)
    figures: list[dict[str, Any]] = []
    for i, image in enumerate(page_images(text)):
        src = image.src
        rec: dict[str, Any] = {"src": src, "asset": None, "sidecar": None, "sidecar_kind": None, "problems": []}
        asset = asset_path(src, figs_root=figs_root, url_prefix=url_prefix) if src is not None else None
        if src is None:
            rec["problems"].append(f"image {i} ({image.tag[:80]!r}): no `{IMG_SRC_ATTR}` attribute — not a figure this gate can resolve")
        elif asset is None:
            rec["problems"].append(f"{src}: not under the mirror prefix {url_prefix!r} — not a figure this gate can read")
        else:
            rec["asset"] = str(asset)
            if not asset.is_file():
                rec["problems"].append(f"{src}: the asset {asset} does not exist under the mirror root")
            side, kind = sidecar_of(asset)
            if side is None:
                rec["problems"].append(f"{src}: {kind}")
            else:
                rec["sidecar"], rec["sidecar_kind"] = str(side), kind
                doc = _read_json(side)
                if kind == "analysis":
                    rec["problems"] += [f"{src}: {v}" for v in check_sidecar(doc, stems=[asset.stem])]
                else:
                    rec["problems"] += [f"{src}: {v}" for v in entry_violations(doc)]
        figures.append(rec)
        violations += rec["problems"]
    record = {"status": GATE_REFUSED if violations else GATE_PASS, "page": str(page), "figs_root": str(Path(figs_root).expanduser()),
              "n_figures": len(figures), "figures": figures, "n_unsupported": len(page_unsupported(text)), "violations": violations, "rule": RULE}
    if violations:
        raise PublicationRefused(violations, where=str(page))
    return record


def recut_page(html: str, *, replace: Mapping[str, str] | None = None, remove: Sequence[str] = ()) -> tuple[str, dict[str, Any]]:
    """A page RE-CUT: every ``src`` in ``replace`` swapped for its counterpart, every ``<figure>`` block whose image
    is in ``remove`` dropped. A src the page does not carry refuses (a typo would otherwise leave the PSR embed in
    place, silently). Returns ``(html, record)``; the caller runs :func:`check_page` over the result."""
    replace = dict(replace or {})
    # the re-cut rewrites `src` attributes and drops <figure> blocks / <img> tags by the parser's spans; a `srcset`
    # candidate list, a <source> or a <picture> is a candidate set it does not retarget — so a page carrying one is refused
    # here, before any rewrite, and (the gate over the candidate text refusing too) no responsive candidate can survive a
    # passing re-cut
    unsupported = unsupported_violations(html)
    if unsupported:
        raise PublicationRefused([f"the re-cut cannot retarget responsive-image candidates — {v}" for v in unsupported], where="recut")
    before = page_figure_srcs(html)
    present = set(before)
    missing = [s for s in [*replace, *remove] if s not in present]
    if missing:
        raise PublicationRefused([f"the page embeds no figure with src {s!r}" for s in missing], where="recut")
    clash = [s for s in remove if s in replace]
    if clash:
        raise PublicationRefused([f"{s!r} is named both for replacement and for removal" for s in clash], where="recut")
    out = html
    removed: list[str] = []

    def drop(m: re.Match[str]) -> str:
        block = m.group(0)
        srcs = page_figure_srcs(block)          # the same parser enumeration, over the block's text
        if any(s in remove for s in srcs):
            removed.extend(s for s in srcs if s in remove)
            return ""
        return block

    out = _FIGURE_BLOCK.sub(drop, out)
    # an image outside a <figure> block: drop the tag itself, by the parser's span (never a regex over the attribute)
    for image in reversed([im for im in page_images(out) if im.src in remove]):
        out = out[:image.start] + out[image.end:]
        removed.append(image.src)
    # a replacement rewrites the `src` attribute INSIDE each matching tag's span — the attribute's own quoting kept, an
    # entity-escaped value matched as the browser reads it (the parser unescapes; the raw tag may carry `&amp;`)
    for image in reversed([im for im in page_images(out) if im.src in replace]):
        out = out[:image.start] + _retarget(image, replace[image.src]) + out[image.end:]
    still = [s for s in [*replace, *remove] if s in set(page_figure_srcs(out))]
    if still:
        raise PublicationRefused([f"{s!r} is still embedded after the re-cut" for s in still], where="recut")
    return out, {"replaced": dict(replace), "removed": removed, "n_figures_before": len(before), "n_figures_after": len(page_figure_srcs(out))}


_SRC_ATTR = re.compile(r"(?i)(\b" + IMG_SRC_ATTR + r"\s*=\s*)(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)'|(?P<bare>[^\s>\"']+))")


def _retarget(image: PageImage, new: str) -> str:
    """``image``'s tag text with its ``src`` attribute's VALUE replaced by ``new`` (quoting preserved; a bare value is
    re-emitted double-quoted). Refuses when the tag's raw text carries no ``src`` the parser can be pointed at."""
    raw_values = {image.src, _html.escape(str(image.src), quote=False)}

    def swap(m: re.Match[str]) -> str:
        value = m.group("dq") if m.group("dq") is not None else (m.group("sq") if m.group("sq") is not None else m.group("bare"))
        if value not in raw_values:
            return m.group(0)
        quoted = f'"{new}"' if m.group("dq") is not None or m.group("bare") is not None else f"'{new}'"
        return m.group(1) + quoted

    tag, n = _SRC_ATTR.subn(swap, image.tag)
    if n == 0:
        raise PublicationRefused([f"{image.src!r}: the tag {image.tag[:80]!r} carries no src attribute the re-cut can rewrite"], where="recut")
    return tag


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("check-figs-dir", help="the figures-directory gate an analysis runs before publishing")
    d.add_argument("figs_dir", type=Path)
    d.add_argument("--sidecar", required=True, help="the rows sidecar's file name (final_m5t_figures_numbers.json / final_g8k_...)")
    d.add_argument("--exclude-arm", action="append", default=[], metavar="CELL")
    d.add_argument("--exclude-family", action="append", default=[], metavar="MODEL_KEY")
    d.add_argument("--exclude-mode", action="append", default=[], metavar="MODE")
    p = sub.add_parser("check-page", help="the review-page gate over every embedded figure's sidecar")
    p.add_argument("index_html", type=Path)
    p.add_argument("--figs-root", type=Path, default=Path("~/figs"), help="the local mirror root the page's /figs/ URLs resolve under")
    p.add_argument("--url-prefix", default=PAGE_FIGS_URL_PREFIX)
    r = sub.add_parser("recut", help="re-cut a page: replace / remove PSR-bearing embeds, gate the RESULT in memory, then write it")
    r.add_argument("index_html", type=Path)
    r.add_argument("--out", type=Path, required=True,
                   help="where the re-cut page is written — never the input page, and never an existing file without --overwrite; "
                        "written atomically ONLY after the page gate PASSED over the re-cut text")
    r.add_argument("--overwrite", action="store_true", help="allow --out to replace an existing file (a served page is replaced only on PASS)")
    r.add_argument("--replace", action="append", default=[], metavar="OLD_SRC=NEW_SRC")
    r.add_argument("--remove", action="append", default=[], metavar="SRC")
    r.add_argument("--figs-root", type=Path, default=Path("~/figs"))
    r.add_argument("--url-prefix", default=PAGE_FIGS_URL_PREFIX)
    r.add_argument("--record", type=Path, default=None, help="write the re-cut + gate record here (JSON)")
    return ap


def _same_file(a: Path, b: Path) -> bool:
    """Whether two paths name one file — by identity when both exist, else by their resolved absolute paths."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return a.expanduser().resolve() == b.expanduser().resolve()


def write_atomic(path: Path, text: str) -> None:
    """``path`` replaced in one step (a temporary file beside it, then ``os.replace``): a reader sees the old page or the
    new one, never a torn file — and nothing is written at all unless the caller already gated the text."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def recut_and_gate(index_html: Path, *, out: Path, replace: Mapping[str, str], remove: Sequence[str], figs_root: Path,
                   url_prefix: str = PAGE_FIGS_URL_PREFIX, overwrite: bool = False) -> dict[str, Any]:
    """The re-cut as ONE guarded step (B2): refuse ``out`` == the input ("never in place") and an existing ``out`` without
    ``overwrite``; re-cut in memory; gate the CANDIDATE TEXT (``check_page(..., html=)`` — nothing on disk yet); only on
    PASS write ``out`` atomically. Raises :class:`PublicationRefused` before any write on every refusal, so a refused
    re-cut can neither create the destination nor change an existing one."""
    src_page = Path(index_html).expanduser()
    out = Path(out).expanduser()
    if _same_file(src_page, out):
        raise PublicationRefused([f"--out {out} is the input page itself — a re-cut is never written in place"], where="recut")
    if out.exists() and not overwrite:
        raise PublicationRefused([f"--out {out} exists — pass --overwrite to replace a served page (it is replaced only on PASS)"], where="recut")
    html = src_page.read_text(encoding="utf-8")
    out_html, rec = recut_page(html, replace=replace, remove=remove)
    # the gate over the candidate TEXT: the destination is untouched until this passes
    gate = check_page(out, figs_root=figs_root, url_prefix=url_prefix, html=out_html)
    write_atomic(out, out_html)
    rec["gate"] = gate
    rec["out"] = str(out)
    return rec


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "check-figs-dir":
            rec = check_publication_no_psr(a.figs_dir, sidecar=a.sidecar, excluded_arms=a.exclude_arm, excluded_families=a.exclude_family,
                                           excluded_modes=a.exclude_mode)
            print(f"[publication-gate] {rec['figs_dir']}: {rec['status']} ({len(rec['figures'])} figures described)")
            return 0
        if a.cmd == "check-page":
            rec = check_page(a.index_html, figs_root=a.figs_root, url_prefix=a.url_prefix)
            print(f"[publication-gate] {rec['page']}: {rec['status']} ({rec['n_figures']} embedded figures)")
            return 0
        replace: dict[str, str] = {}
        for spec in a.replace:
            old, sep, new = str(spec).partition("=")
            if not sep or not old or not new:
                raise PublicationRefused([f"--replace {spec!r}: expected OLD_SRC=NEW_SRC"], where="recut")
            replace[old] = new
        # B2: gate the candidate in memory; the destination is written atomically on PASS and not otherwise
        rec = recut_and_gate(a.index_html, out=a.out, replace=replace, remove=a.remove, figs_root=a.figs_root, url_prefix=a.url_prefix,
                             overwrite=a.overwrite)
        if a.record:
            Path(a.record).expanduser().write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[publication-gate] re-cut -> {rec['out']}: {rec['gate']['status']} ({rec['n_figures_before']} -> {rec['n_figures_after']} figures; "
              f"replaced {len(rec['replaced'])}, removed {len(rec['removed'])})")
        return 0
    except PublicationRefused as e:
        print(f"[publication-gate] {e.where}: REFUSED — " + "; ".join(e.violations), file=sys.stderr)
        return EXIT_PUBLICATION_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
