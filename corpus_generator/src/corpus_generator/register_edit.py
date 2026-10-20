"""Register-matching SPAN EDITS over a released final -> the r2 release
(2026-09-03; configs/control-scale-r2.yaml).

The scale-matched control corpus (control-scale-20260903) was register-FREE
where production is register-heavy — the prompt-side bans drove three
production registers to zero instead of to production's share — and its
stories drifted onto stock beats production lacks. The frozen filter could
not fix either (capping the new beats at <1% leaves the corpus under target;
adding a register needs generation). This module makes the corpus
register-MATCHED with cheap, coherent, VERIFIED edits by the generator model
instead of regeneration:

- **markers** are config specs (``register_edit.markers``): a regex family
  over one trainable field, a mode (``add`` a register the control lacks /
  ``remove`` a beat production lacks), a splice ``kind``, production's share
  under that regex frozen as ``target_rate`` (re-measured on the production
  reference at plan time, drift fails loudly), verification thresholds, and
  regexes the new text must never match;
- **selection** is deterministic: records are ordered by
  ``sha256(seed:register-edit:<marker>:<run_id>:<custom_id>)``; ``add``
  picks the first k non-matching records so the corpus lands on the target
  rate, ``remove`` keeps the first ``target_rate x denominator`` matching
  records and edits every other one;
- **edits** are ONE Batch request per (record, marker) — or per story for
  all its flagged beats — through the existing wave engine and stage runner;
  the model returns only the new span(s) as JSON, the splice happens here,
  and every edit is verified offline (label present / beat absent, tail
  byte-identical, retention floors, forbidden patterns, span found exactly
  once). A failed verification is a ``ParseError``: the stage's bounded repair
  resubmits with the rejection reason appended, and a record that still fails
  stays UNEDITED and is counted — never silently half-edited;
- **re-score**: every edited record is scored again with the run's OWN
  scoring wave (overlay-resolved e_score / s5_score, same reference block and
  params) so the verdict is comparable to v1's; it replaces ``scores`` (v1's
  kept in ``lineage.scores_v1``) and the frozen floors + every hard gate then
  re-run over the assembled set through ``filter_corpus``;
- **lineage**: every r2 record carries ``lineage.v1_sha256`` (the sha256 of
  the released v1 JSONL line it derives from); edited ones carry
  ``lineage.register_edits`` = [{field, marker, old_span_sha, new_span_sha,
  retained, pass}]. A per-corpus manifest beside the final repeats the map
  and the mirror's release gate re-verifies the two against each other and
  against the text (mirror.check_register_edit_manifest).

v1 finals, manifest.json and UNION.json are never written. Passes: a
marker's ``edit_pass`` (default 1) lets a second pass — one further
iteration, when the post-edit survey finds a new monoculture — edit the
pass-1 output instead of re-editing from v1.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .advice import ADVICE_SCORE_SCHEMA
from .batch import WaveRequest, _utcnow
from .config import EditMarker, RunConfig, filter_report_path
from .filtering import filter_corpus
from .io import atomic_write, iter_jsonl, read_json, sha256_file, write_jsonl_atomic
from .parsing import extract_json_object
from .pipeline import _heartbeat, build_ctx, resolve_environment
from .register_survey import ngram_table
from .schema import AdviceRecord, StoryRecord
from .stages import (
    OutputRow,
    ParseError,
    StageCtx,
    StageSpec,
    const_user_content,
    load_outputs,
    run_stage,
    structured_format,
    wave_params,
)
from .stories import STORY_SCORE_SCHEMA
from .templates import repo_root

SCORE_TEMPLATE = {"advice": "advice/e_score.md", "stories": "stories/s5_score.md"}
SCORE_SCHEMA = {"advice": ADVICE_SCORE_SCHEMA, "stories": STORY_SCORE_SCHEMA}
SCORE_STAGE = {"advice": "e", "stories": "s5"}

RELABEL_SCHEMA = {
    "type": "object",
    "properties": {"first_paragraph": {"type": "string"}},
    "required": ["first_paragraph"],
    "additionalProperties": False,
}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"opening": {"type": "string"}},
    "required": ["opening"],
    "additionalProperties": False,
}
OPENER_SCHEMA = {
    "type": "object",
    "properties": {"opener": {"type": "string"}, "placement": {"type": "string"}},
    "required": ["opener", "placement"],
    "additionalProperties": False,
}
REPLACE_SCHEMA = {
    "type": "object",
    "properties": {
        "replacements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "integer"}},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["ids", "old", "new"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["replacements"],
    "additionalProperties": False,
}
KIND_SCHEMA = {
    "relabel_paragraph": RELABEL_SCHEMA,
    "verdict_opening": VERDICT_SCHEMA,
    "prepend_opener": OPENER_SCHEMA,
    "replace_spans": REPLACE_SCHEMA,
}

# the verdict label phrases production uses, in production's proportions
# (measured on the v2 finals 2026-09-03: "short answer" 136, "short version"
# 98, "straight answer(s)" 127 of 361) — assigned per record in seeded order
# so the ADDED register carries production's own label mix, not the model's
# favourite
VERDICT_LABEL_FORMS: list[tuple[str, int]] = [
    ("Short answer:", 136),
    ("Short version:", 98),
    ("Straight answer first:", 64),
    ("Straight answer:", 63),
]

# the sanity/gut-check opener FORMS production uses, in production's proportions
# (v2 finals, first three words of the 323 openers: "sanity check needed" 66,
# "sanity check before" 53, "need a gut" 47, "need a sanity" 39, "gut check
# needed" 26, "quick gut check" 11, "gut check before" 9, "sanity check and" 9)
# — assigned per record in seeded order by the pass-2 REFORM of the pass-1
# openers, 47% of which had collapsed onto "Sanity check on …"
SANITY_OPENER_FORMS: list[tuple[str, int]] = [
    ("Sanity check needed", 66),
    ("Sanity check before", 53),
    ("Need a gut check", 47),
    ("Need a sanity check", 39),
    ("Gut check needed", 26),
    ("Quick gut check", 11),
    ("Gut check before", 9),
    ("Sanity check and", 9),
]

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'’-]*")
# sentence boundary: terminal punctuation (optionally followed by a closing
# quote/bracket) then whitespace — fixed-width look-behinds only
SENT_SPLIT = re.compile(r'(?<=[.!?]["”’)])\s+|(?<=[.!?])\s+')


class EditRejected(Exception):
    """An edit failed offline verification (reason in str)."""


# ------------------------------------------------------------------ helpers


def rec_key(r: dict) -> str:
    lin = r.get("lineage") or {}
    return f"{lin.get('run_id', '?')}:{lin.get('custom_id', '?')}"


def rec_id(r: dict) -> str:
    return (r.get("lineage") or {}).get("custom_id") or "?"


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return WORD_RE.findall((text or "").casefold())


def compile_marker(m: EditMarker) -> Callable[[str], object]:
    rx = re.compile(m.pattern, re.IGNORECASE)
    return rx.match if m.anchored else rx.search


def marker_hits(m: EditMarker, record: dict) -> bool:
    return bool(compile_marker(m)(str(record.get(m.field) or "")))


def has_field(m: EditMarker, record: dict) -> bool:
    return bool((record.get(m.field) or "").strip())


def measure(records: list[dict], m: EditMarker) -> dict:
    probe = compile_marker(m)
    denom_recs = [r for r in records if has_field(m, r)] if m.denominator == "field_present" else records
    matched = sum(1 for r in denom_recs if probe(str(r.get(m.field) or "")))
    denom = len(denom_recs)
    return {"matched": matched, "denominator": denom, "rate": matched / denom if denom else 0.0}


def _order_key(seed: int, marker: str, r: dict) -> str:
    return hashlib.sha256(f"{seed}:register-edit:{marker}:{rec_key(r)}".encode()).hexdigest()


def _pass_edits(r: dict, marker: str) -> list[dict]:
    return [e for e in ((r.get("lineage") or {}).get("register_edits") or []) if e.get("marker") == marker]


def select(records: list[dict], m: EditMarker, seed: int) -> dict:
    """Deterministic selection for one marker.

    add:    edit the first k NON-matching records (sha order) where k brings
            the matched count to round(target_rate x denominator);
    remove: KEEP the first round(target_rate x denominator) matching records
            (sha order) and edit every other matching record;
    reform: among records matching reform_pattern that carry a pass edit by
            reform_of, KEEP round(reform_keep_rate x denominator) (sha order)
            and re-form every other one."""
    probe = compile_marker(m)
    pool = [r for r in records if has_field(m, r)] if m.denominator == "field_present" else records
    denom = len(pool)
    if m.reform_pattern is not None:
        rrx = re.compile(m.reform_pattern, re.IGNORECASE)
        cands = [r for r in pool if rrx.match(str(r.get(m.field) or "")) and _pass_edits(r, m.reform_of)]
        keep_n = int(round((m.reform_keep_rate or 0.0) * denom))
        ordered = sorted(cands, key=lambda r: _order_key(seed, m.name, r))
        keep, edit = ordered[:keep_n], ordered[keep_n:]
        return {
            "mode": "reform", "denominator": denom, "matched_before": len(cands),
            "target_count": keep_n, "n_edit": len(edit),
            "edit_ids": [rec_id(r) for r in edit], "keep_ids": [rec_id(r) for r in keep],
        }
    target = int(round(m.target_rate * denom))
    matching = [r for r in pool if probe(str(r.get(m.field) or ""))]
    if m.mode == "add":
        candidates = sorted(
            (r for r in pool if has_field(m, r) and not probe(str(r.get(m.field) or ""))),
            key=lambda r: _order_key(seed, m.name, r),
        )
        k = max(0, min(target - len(matching), len(candidates)))
        return {
            "mode": "add", "denominator": denom, "matched_before": len(matching),
            "target_count": target, "n_edit": k,
            "edit_ids": [rec_id(r) for r in candidates[:k]], "keep_ids": [],
        }
    ordered = sorted(matching, key=lambda r: _order_key(seed, m.name, r))
    keep = ordered[:target]
    edit = ordered[target:]
    return {
        "mode": "remove", "denominator": denom, "matched_before": len(matching),
        "target_count": target, "n_edit": len(edit),
        "edit_ids": [rec_id(r) for r in edit], "keep_ids": [rec_id(r) for r in keep],
    }


# ------------------------------------------------------------ text surgery


def split_first_paragraph(text: str) -> tuple[str, str, str]:
    """(first paragraph, separator, rest) — the separator is the first blank
    line ('\\n\\n' plus any extra newlines) or '' when there is one paragraph."""
    m = re.search(r"\n\s*\n", text)
    if m is None:
        return text, "", ""
    return text[: m.start()], m.group(0), text[m.end():]


def matched_word_chars(old: str, new: str) -> int:
    """Characters of ``old``'s words that survive in ``new`` (word-level
    longest-matching-blocks; whitespace and punctuation between words are
    not counted, so the figure is conservative)."""
    ow, nw = words(old), words(new)
    sm = difflib.SequenceMatcher(None, ow, nw, autojunk=False)
    return sum(sum(len(w) for w in ow[b.a : b.a + b.size]) for b in sm.get_matching_blocks())


def retained_fraction(original: str, spans: list[tuple[str, str]]) -> float:
    """Fraction of the ORIGINAL text's characters that survive an edit that
    replaced each ``old`` span by its ``new``: untouched characters count in
    full, edited spans contribute their word-level matched characters."""
    if not original:
        return 1.0
    edited = sum(len(o) for o, _n in spans)
    kept = len(original) - edited + sum(matched_word_chars(o, n) for o, n in spans)
    return max(0.0, min(1.0, kept / len(original)))


def word_retention(old: str, new: str) -> float:
    """Fraction of ``old``'s word multiset present in ``new``."""
    ow = words(old)
    if not ow:
        return 1.0
    from collections import Counter

    c_old, c_new = Counter(ow), Counter(words(new))
    return sum(min(n, c_new[w]) for w, n in c_old.items()) / len(ow)


def _forbidden(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE | re.MULTILINE):
            return p
    return None


@dataclass
class EditResult:
    new_text: str
    spans: list[tuple[str, str]]  # (old, new) pairs actually replaced
    retained: float
    edits: list[dict] = field(default_factory=list)  # manifest/lineage entries (spans hashed)


def _finish(m: EditMarker, original: str, new_text: str, spans: list[tuple[str, str]],
            edit_pass: int) -> EditResult:
    retained = retained_fraction(original, spans)
    if retained < m.verify.min_retained - 1e-9:
        raise EditRejected(
            f"retained {retained:.3f} of the original {m.field} text < required "
            f"{m.verify.min_retained:.2f} — keep more of the original wording"
        )
    edits = [
        {
            "field": m.field, "marker": m.name,
            "old_span_sha": sha_text(o), "new_span_sha": sha_text(n),
            "old_chars": len(o), "new_chars": len(n),
            "retained": round(retained, 4), "pass": edit_pass,
        }
        for o, n in spans
    ]
    return EditResult(new_text=new_text, spans=spans, retained=retained, edits=edits)


def verify_relabel_paragraph(m: EditMarker, original: str, out: dict, edit_pass: int = 1) -> EditResult:
    """kind=relabel_paragraph: the model returned the rewritten FIRST
    paragraph; everything after it is byte-identical by construction."""
    new_fp = out.get("first_paragraph")
    if not isinstance(new_fp, str) or not new_fp.strip():
        raise EditRejected("first_paragraph missing or empty")
    new_fp = new_fp.strip()
    first, sep, rest = split_first_paragraph(original)
    if "\n" in new_fp:
        raise EditRejected("the rewritten first paragraph must be ONE paragraph (no line breaks)")
    if not compile_marker(m)(new_fp):
        raise EditRejected(f"the rewritten first paragraph does not carry the label /{m.pattern}/")
    if not re.match(r"Deployment:\s+\S", new_fp):
        raise EditRejected('the paragraph must open exactly "Deployment: " followed by text')
    bad = _forbidden(m.forbid_patterns, new_fp)
    if bad:
        raise EditRejected(f"the rewritten paragraph matches a forbidden pattern /{bad}/")
    wr = word_retention(first, new_fp)
    if wr < m.verify.min_word_retention:
        raise EditRejected(
            f"only {wr:.0%} of the original paragraph's words survive (need >= "
            f"{m.verify.min_word_retention:.0%}) — keep the paragraph's facts and wording"
        )
    ratio = len(words(new_fp)) / max(1, len(words(first)))
    if not (0.7 <= ratio <= 1.35):
        raise EditRejected(f"rewritten paragraph is {ratio:.2f}x the original length (allowed 0.70-1.35x)")
    new_text = new_fp + sep + rest
    return _finish(m, original, new_text, [(first, new_fp)], edit_pass)


def verify_verdict_opening(m: EditMarker, original: str, out: dict, edit_pass: int = 1) -> EditResult:
    """kind=verdict_opening: the model returned the new opening (label +
    one-line verdict + the minimally adjusted first paragraph)."""
    opening = out.get("opening")
    if not isinstance(opening, str) or not opening.strip():
        raise EditRejected("opening missing or empty")
    opening = opening.strip()
    first, sep, rest = split_first_paragraph(original)
    if not compile_marker(m)(opening):
        raise EditRejected(f"the new opening does not start with a verdict label /{m.pattern}/")
    bad = _forbidden(m.forbid_patterns, opening)
    if bad:
        raise EditRejected(f"the new opening matches a forbidden pattern /{bad}/")
    is_header = first.lstrip().startswith("#")
    if is_header and not opening.lstrip().startswith("#"):
        raise EditRejected("the original opened on a markdown header — the new opening must be a header line too")
    if not is_header and opening.lstrip().startswith("#"):
        raise EditRejected("the original first paragraph was prose — do not turn it into a header")
    if is_header and "\n" in opening:
        raise EditRejected("a header opening must be a single header line")
    extra = len(words(opening)) - len(words(first))
    if extra > m.verify.max_extra_words:
        raise EditRejected(
            f"the new opening adds {extra} words to the first paragraph (allowed <= "
            f"{m.verify.max_extra_words}) — one-line verdict, then the original paragraph"
        )
    if re.search(r"\n\s*\n\s*\n", opening):
        raise EditRejected("the new opening contains stacked blank lines")
    new_text = opening + sep + rest
    return _finish(m, original, new_text, [(first, opening)], edit_pass)


def verify_prepend_opener(m: EditMarker, original: str, out: dict, edit_pass: int = 1) -> EditResult:
    """kind=prepend_opener: a new opening sentence/clause before the verbatim
    original field."""
    opener = out.get("opener")
    placement = str(out.get("placement", "paragraph")).strip().lower()
    if not isinstance(opener, str) or not opener.strip():
        raise EditRejected("opener missing or empty")
    opener = opener.strip().strip('"“”')
    if "\n" in opener:
        raise EditRejected("the opener must be a single line")
    if len(opener.split()) > m.verify.max_opener_words:
        raise EditRejected(f"the opener is {len(opener.split())} words (allowed <= {m.verify.max_opener_words})")
    if not compile_marker(m)(opener):
        raise EditRejected(f"the opener does not carry the register /{m.pattern}/")
    bad = _forbidden(m.forbid_patterns, opener)
    if bad:
        raise EditRejected(f"the opener matches a forbidden pattern /{bad}/")
    if placement not in ("paragraph", "inline"):
        raise EditRejected('placement must be "paragraph" or "inline"')
    if placement == "inline" and not re.search(r"[.:—?!]$", opener):
        raise EditRejected("an inline opener must end in a full stop, colon, dash, or question mark")
    sep = "\n\n" if placement == "paragraph" else " "
    new_text = opener + sep + original
    if not new_text.endswith(original):
        raise EditRejected("internal: the original text did not survive the splice")
    # the spliced field as a whole must still carry the register at its start
    if not compile_marker(m)(new_text):
        raise EditRejected("the spliced field does not open with the register")
    res = _finish(m, original, new_text, [], edit_pass)
    res.spans = [("", opener + sep)]
    res.edits = [
        {
            "field": m.field, "marker": m.name,
            "old_span_sha": sha_text(""), "new_span_sha": sha_text(opener + sep),
            "old_chars": 0, "new_chars": len(opener + sep),
            "retained": 1.0, "pass": edit_pass,
        }
    ]
    return res


def flag_sentences(text: str, markers: list[EditMarker]) -> list[dict]:
    """Every sentence of ``text`` carrying one of ``markers`` (sentences split
    within paragraphs on terminal punctuation), with its verbatim text, the
    marker names it carries, and its character offset — ids are 1-based in
    document order."""
    probes = [(m, re.compile(m.pattern, re.IGNORECASE)) for m in markers]
    out: list[dict] = []
    pos = 0
    for para in text.split("\n"):
        para_start = pos
        pos += len(para) + 1
        if not para.strip():
            continue
        cursor = 0
        for sent in SENT_SPLIT.split(para):
            if not sent:
                continue
            idx = para.find(sent, cursor)
            if idx == -1:
                idx = cursor
            cursor = idx + len(sent)
            fams = [m.name for m, rx in probes if rx.search(sent)]
            if not fams:
                continue
            out.append(
                {
                    "id": len(out) + 1,
                    "families": fams,
                    "sentence": sent,
                    "start": para_start + idx,
                    "end": para_start + idx + len(sent),
                }
            )
    return out


def verify_replace_spans(
    markers: list[EditMarker],
    original: str,
    flagged: list[dict],
    out: dict,
    forbid_patterns: list[str],
    edit_pass: int = 1,
) -> EditResult:
    """kind=replace_spans: the model returned {replacements: [{ids, old, new}]}.
    Each ``old`` must be found verbatim exactly once around the flagged
    sentence(s) it covers, every flagged id must be covered exactly once,
    spans may not overlap, each ``new`` must clear the length bounds, carry
    none of the removed beats or forbidden patterns, keep the paragraph
    structure, and the story must retain >= min_retained of its characters.
    Thresholds come from the strictest marker involved."""
    reps = out.get("replacements")
    if not isinstance(reps, list) or not reps:
        raise EditRejected("replacements missing or empty")
    by_name = {m.name: m for m in markers}
    verify = min((m.verify for m in markers), key=lambda v: v.min_retained)
    max_span = min(m.verify.max_span_chars for m in markers)
    lo = max(m.verify.min_new_over_old for m in markers)
    hi = min(m.verify.max_new_over_old for m in markers)
    slack = min(m.verify.max_new_extra_chars for m in markers)
    flagged_by_id = {f["id"]: f for f in flagged}
    covered: set[int] = set()
    located: list[tuple[int, int, str, str]] = []  # (start, end, old, new)
    for rep in reps:
        if not isinstance(rep, dict):
            raise EditRejected("each replacement must be an object")
        ids, old, new = rep.get("ids"), rep.get("old"), rep.get("new")
        if not isinstance(ids, list) or not ids or not all(isinstance(i, int) for i in ids):
            raise EditRejected("replacement ids must be a non-empty list of flagged ids")
        if not isinstance(old, str) or not old.strip() or not isinstance(new, str) or not new.strip():
            raise EditRejected(f"replacement for ids {ids}: old/new must be non-empty strings")
        for i in ids:
            if i not in flagged_by_id:
                raise EditRejected(f"replacement names id {i}, which was not flagged")
            if i in covered:
                raise EditRejected(f"flagged id {i} is covered by two replacements")
            covered.add(i)
        if len(old) > max_span:
            raise EditRejected(f"old span for ids {ids} is {len(old)} chars (allowed <= {max_span})")
        # locate the verbatim span at the flagged sentence(s)
        anchors = [flagged_by_id[i] for i in ids]
        occurrences = [mm.start() for mm in re.finditer(re.escape(old), original)]
        if not occurrences:
            raise EditRejected(f"old span for ids {ids} is not found verbatim in the story — copy it character for character")
        hits = [
            s for s in occurrences
            if all(s <= a["start"] and a["end"] <= s + len(old) for a in anchors)
        ]
        if len(hits) != 1:
            raise EditRejected(
                f"old span for ids {ids} must cover the flagged sentence(s) exactly once "
                f"(found {len(hits)} covering occurrences)"
            )
        for a in anchors:
            if a["sentence"] not in old:
                raise EditRejected(f"old span for ids {ids} does not contain flagged sentence {a['id']} verbatim")
        start = hits[0]
        ratio = len(new) / len(old)
        if ratio < lo or (ratio > hi and len(new) > len(old) + slack):
            raise EditRejected(
                f"replacement for ids {ids} is {ratio:.2f}x the old span's length (allowed "
                f"{lo:.1f}-{hi:.1f}x, or up to {slack} characters longer for a short sentence)"
            )
        if new.count("\n") != old.count("\n"):
            raise EditRejected(f"replacement for ids {ids} changes the paragraph structure (line breaks)")
        fams = sorted({f for a in anchors for f in a["families"]})
        for fam in fams:
            if compile_marker(by_name[fam])(new) if by_name[fam].anchored else re.search(by_name[fam].pattern, new, re.IGNORECASE):
                raise EditRejected(f"replacement for ids {ids} still carries the beat {fam!r}")
        bad = _forbidden(forbid_patterns, new)
        if bad:
            raise EditRejected(f"replacement for ids {ids} introduces a forbidden pattern /{bad}/")
        located.append((start, start + len(old), old, new))
    missing = sorted(set(flagged_by_id) - covered)
    if missing:
        raise EditRejected(f"flagged ids {missing} have no replacement")
    located.sort()
    for (s1, e1, _o1, _n1), (s2, _e2, _o2, _n2) in zip(located, located[1:]):
        if s2 < e1:
            raise EditRejected("replacement spans overlap")
    new_text = original
    for s, e, _old, new in reversed(located):
        new_text = new_text[:s] + new + new_text[e:]
    # the removed beats must be gone from the whole text (a kept family is
    # never flagged, so any survivor is an edit that missed a sentence)
    removed = sorted({f for fl in flagged for f in fl["families"]})
    for fam in removed:
        if re.search(by_name[fam].pattern, new_text, re.IGNORECASE):
            raise EditRejected(f"the beat {fam!r} still occurs in the edited story")
    # one edit entry per span, attributed to the families it removed
    spans = [(o, n) for _s, _e, o, n in located]
    retained = retained_fraction(original, spans)
    if retained < verify.min_retained - 1e-9:
        raise EditRejected(
            f"retained {retained:.3f} of the story < required {verify.min_retained:.2f} — "
            f"change only the flagged detail, keep the sentence otherwise"
        )
    edits = []
    for s, e, o, n in located:
        fams = sorted({f for fl in flagged for f in fl["families"] if s <= fl["start"] and fl["end"] <= e})
        for fam in fams:
            edits.append(
                {
                    "field": "text", "marker": fam,
                    "old_span_sha": sha_text(o), "new_span_sha": sha_text(n),
                    "old_chars": len(o), "new_chars": len(n),
                    "retained": round(retained, 4), "pass": edit_pass,
                }
            )
    return EditResult(new_text=new_text, spans=spans, retained=retained, edits=edits)


# ------------------------------------------------------------------ paths


def edit_dir(cfg: RunConfig, root: Path) -> Path:
    return cfg.run_dir(root) / f"register-edit{cfg.register_edit.suffix}"


def plan_path(cfg: RunConfig, root: Path, edit_pass: int) -> Path:
    return edit_dir(cfg, root) / f"PLAN-pass{edit_pass}.json"


def assembled_path(cfg: RunConfig, root: Path, edit_pass: int, corpus: str) -> Path:
    return edit_dir(cfg, root) / f"assembled-pass{edit_pass}-{corpus}.jsonl"


def register_edit_manifest_path(final: Path) -> Path:
    return final.with_name(final.stem + ".register-edit.json")


def run_manifest_path(cfg: RunConfig, root: Path) -> Path:
    return cfg.run_dir(root) / f"register-edit{cfg.register_edit.suffix}.json"


def _resolve(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def source_path(cfg: RunConfig, root: Path, corpus: str, edit_pass: int) -> Path:
    if edit_pass == 1:
        return cfg.final_path(root, corpus, cfg.register_edit.source_suffix)
    return assembled_path(cfg, root, edit_pass - 1, corpus)


def load_records_with_line_shas(path: Path) -> tuple[list[dict], list[str]]:
    """Records of a JSONL file plus the sha256 of each raw line (the v1
    lineage hash: verifiable against the released file byte for byte)."""
    records, shas = [], []
    with Path(path).open("rb") as f:
        for raw in f:
            line = raw.rstrip(b"\r\n")
            if not line.strip():
                continue
            records.append(json.loads(line))
            shas.append(hashlib.sha256(line).hexdigest())
    return records, shas


# ------------------------------------------------------------------- plan


def build_plan(cfg: RunConfig, root: Path | None = None, edit_pass: int = 1, on_event=print) -> dict:
    """Measure every marker on the source and on the production reference
    (drift from the frozen target fails loudly), select records
    deterministically, and write PLAN-pass<N>.json (source shas recorded so a
    later phase can refuse a changed source)."""
    root = root or repo_root()
    rc = cfg.register_edit
    if rc is None:
        raise ValueError("config has no `register_edit` section")
    plan: dict = {
        "run_id": cfg.run_id, "suffix": rc.suffix, "edit_pass": edit_pass, "seed": rc.seed,
        "sources": {}, "production": {}, "markers": {}, "requests": {}, "created_at": _utcnow(),
        "limit_per_stage": rc.limit_per_stage,
    }
    for corpus in ("advice", "stories"):
        markers = rc.markers_for(corpus, edit_pass)
        if not markers:
            continue
        src = source_path(cfg, root, corpus, edit_pass)
        if not src.exists():
            raise FileNotFoundError(f"{corpus}: source final for pass {edit_pass} missing: {src}")
        records, _shas = load_records_with_line_shas(src)
        prod_path = _resolve(root, rc.production_reference[corpus])
        prod = list(iter_jsonl(prod_path))
        plan["sources"][corpus] = {"path": str(src), "sha256": sha256_file(src), "n": len(records)}
        plan["production"][corpus] = {"path": str(prod_path), "sha256": sha256_file(prod_path), "n": len(prod)}
        n_requests = 0
        story_ids: set[str] = set()
        for m in markers:
            ctrl, prodm = measure(records, m), measure(prod, m)
            if abs(prodm["rate"] - m.target_rate) > rc.rate_tolerance:
                raise RuntimeError(
                    f"marker {m.name}: frozen target_rate {m.target_rate:.6f} != production "
                    f"rate measured now {prodm['rate']:.6f} ({prodm['matched']}/{prodm['denominator']}) "
                    f"— re-freeze the target from the production reference, do not proceed on drift"
                )
            sel = select(records, m, rc.seed)
            if rc.limit_per_stage is not None:  # PILOT: a few records per marker
                sel["edit_ids"] = sel["edit_ids"][: rc.limit_per_stage]
                sel["n_edit"] = len(sel["edit_ids"])
                sel["limited_to"] = rc.limit_per_stage
            plan["markers"][m.name] = {
                "corpus": corpus, "field": m.field, "mode": m.mode, "kind": m.kind,
                "pattern": m.pattern, "anchored": m.anchored, "denominator": m.denominator,
                "target_rate": m.target_rate, "target_source": m.target_source,
                "control_before": ctrl, "production": prodm, "selection": sel,
            }
            if m.mode == "add":
                n_requests += sel["n_edit"]
            else:
                story_ids |= set(sel["edit_ids"])
            on_event(
                f"[plan {corpus}/{m.name}] control {ctrl['matched']}/{ctrl['denominator']} = "
                f"{ctrl['rate']:.4f}; production {prodm['matched']}/{prodm['denominator']} = "
                f"{prodm['rate']:.4f}; {m.mode} -> {sel['n_edit']} records to edit"
                + (f", {len(sel['keep_ids'])} kept" if m.mode == "remove" else "")
            )
        if story_ids:
            n_requests += len(story_ids)
            on_event(f"[plan {corpus}] {len(story_ids)} stories carry a removable beat -> one request each")
        plan["requests"][corpus] = n_requests
    plan["requests"]["total_edits"] = sum(plan["requests"].values())
    plan["requests"]["total_with_rescore"] = 2 * plan["requests"]["total_edits"]
    pp = plan_path(cfg, root, edit_pass)
    pp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(pp, json.dumps(plan, indent=1, ensure_ascii=False))
    on_event(f"plan written to {pp} — {plan['requests']['total_edits']} edit requests "
             f"(+ as many re-scores)")
    return plan


def plan_for_pass(cfg: RunConfig, root: Path, edit_pass: int) -> dict:
    """A STORED plan (never rebuilt): reading an earlier pass's stage outputs
    for the manifest must not re-plan against a source that has moved on."""
    return read_json(plan_path(cfg, root, edit_pass))


def load_plan(cfg: RunConfig, root: Path, edit_pass: int, on_event=print) -> dict:
    """The stored plan for a pass (built if absent); the source finals must
    still hash to what the plan was built from."""
    pp = plan_path(cfg, root, edit_pass)
    if not pp.exists():
        return build_plan(cfg, root, edit_pass, on_event)
    plan = read_json(pp)
    for corpus, src in plan["sources"].items():
        actual = sha256_file(Path(src["path"]))
        if actual != src["sha256"]:
            raise RuntimeError(
                f"{corpus}: source {src['path']} changed since the plan was built "
                f"({actual[:12]} != {src['sha256'][:12]}) — a plan is bound to its source"
            )
    return plan


# ----------------------------------------------------------------- specs


def _stage_name(kind: str, suffix: str, edit_pass: int, name: str) -> str:
    """Stage dir under run_dir/<corpus>/: kind + release suffix + pass + name,
    e.g. redit-r2-p1-beats — a pilot config with its own suffix (-r2pilot)
    therefore never collides with the real pass's stage dirs."""
    return f"{kind}{suffix}-p{edit_pass}-{name}"


def structured(cfg: RunConfig) -> bool:
    """Whether the pass's requests carry output_config: the explicit
    register_edit.structured_output knob, else the run's frozen verdict."""
    so = cfg.register_edit.structured_output if cfg.register_edit else None
    return bool(cfg.engine.use_structured_scores) if so is None else so


def _edit_params(ctx: StageCtx, rendered: str, schema: dict) -> dict:
    params = {
        "model": ctx.cfg.engine.model,
        "max_tokens": ctx.cfg.register_edit.max_tokens,
        "messages": [{"role": "user", "content": rendered}],
    }
    if structured(ctx.cfg):
        params["output_config"] = structured_format(schema)
    return params


def _with_feedback(params: dict, reason: str | None, round_no: int) -> dict:
    """The repair request: the same prompt with the rejection reason appended
    (a verification failure is usually a rule the model skipped)."""
    p = json.loads(json.dumps(params))
    note = (
        f"\n\nYour previous attempt was rejected"
        + (f": {reason}" if reason else "")
        + ". Try again, following every rule above exactly."
    )
    content = p["messages"][0]["content"]
    if isinstance(content, str):
        p["messages"][0]["content"] = content + note
    else:
        content[-1]["text"] = content[-1]["text"] + note
    return p


def _seeded_form(forms: list[tuple[str, int]], key: str) -> str:
    total = sum(w for _f, w in forms)
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % total
    for form, w in forms:
        if h < w:
            return form
        h -= w
    return forms[-1][0]


def sanity_opener_form(cid: str, seed: int) -> str:
    """Seeded per-record opener form drawn from production's mix (reform pass)."""
    return _seeded_form(SANITY_OPENER_FORMS, f"{seed}:sanity-form:{cid}")


def verdict_label_form(cid: str, seed: int) -> str:
    """Seeded per-record label form drawn from production's label mix."""
    total = sum(w for _f, w in VERDICT_LABEL_FORMS)
    h = int(hashlib.sha256(f"{seed}:verdict-label:{cid}".encode()).hexdigest()[:8], 16) % total
    for form, w in VERDICT_LABEL_FORMS:
        if h < w:
            return form
        h -= w
    return VERDICT_LABEL_FORMS[-1][0]


def advice_edit_spec(
    m: EditMarker, records: dict[str, dict], edit_ids: list[str], edit_pass: int, suffix: str = "",
    prior_spans: dict[str, str] | None = None,
) -> StageSpec:
    """One stage per add-mode advice marker: a request per selected record.
    For a REFORM marker, ``prior_spans`` maps record id -> the prepended span
    the earlier pass added (opener + separator); it is stripped so the new
    opener is built for the ORIGINAL message and verified against it."""
    last_reason: dict[str, str] = {}
    schema = KIND_SCHEMA[m.kind]
    prior_spans = prior_spans or {}

    def _original(rec: dict) -> str:
        text = rec[m.field] or ""
        if m.reform_pattern is not None:
            span = prior_spans.get(rec_id(rec))
            if not span or not text.startswith(span):
                raise ValueError(f"record {rec_id(rec)}: prior {m.reform_of} span not found at the start of {m.field}")
            return text[len(span):]
        return text

    def _render(ctx: StageCtx, rec: dict) -> str:
        text = _original(rec)
        first, _sep, _rest = split_first_paragraph(text)
        common = dict(system=rec.get("system"), user=rec["user"], assistant=rec["assistant"])
        if m.kind == "relabel_paragraph":
            return ctx.templates.render(m.template, system=text, first_paragraph=first)
        if m.kind == "verdict_opening":
            return ctx.templates.render(
                m.template, first_paragraph=first,
                label_form=verdict_label_form(rec_id(rec), ctx.cfg.register_edit.seed),
                is_header=first.lstrip().startswith("#"), **{**common, m.field: text},
            )
        if m.kind == "prepend_opener":
            kw = dict(system=rec.get("system"), user=text)
            if m.reform_pattern is not None:
                kw["assigned_form"] = sanity_opener_form(rec_id(rec), ctx.cfg.register_edit.seed)
            return ctx.templates.render(m.template, **kw)
        raise ValueError(f"marker {m.name}: kind {m.kind} is not an advice add kind")

    def build(ctx: StageCtx) -> list[WaveRequest]:
        reqs = []
        for cid in sorted(edit_ids):
            rec = records[cid]
            reqs.append(
                WaveRequest(
                    custom_id=f"redit-{m.name}-{cid}",
                    params=_edit_params(ctx, _render(ctx, rec), schema),
                    lineage={"cid": cid, "marker": m.name, "field": m.field, "pass": edit_pass},
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        cid = req.lineage["cid"]
        original = _original(records[cid])
        try:
            out = extract_json_object(outcome["text"])
            if m.kind == "relabel_paragraph":
                res = verify_relabel_paragraph(m, original, out, edit_pass)
            elif m.kind == "verdict_opening":
                res = verify_verdict_opening(m, original, out, edit_pass)
            else:
                res = verify_prepend_opener(m, original, out, edit_pass)
                if m.reform_pattern is not None:
                    opener = res.spans[0][1]
                    if re.match(m.reform_pattern, opener, re.IGNORECASE):
                        raise EditRejected("the new opener still takes the over-used shape")
                    form = sanity_opener_form(cid, ctx.cfg.register_edit.seed)
                    if not opener.casefold().startswith(form.casefold()):
                        raise EditRejected(f'the opener must begin with the words "{form}"')
                    # the reform REPLACES the earlier pass's span: record what it replaced
                    prior = prior_spans[cid]
                    for e in res.edits:
                        e["old_span_sha"], e["old_chars"] = sha_text(prior), len(prior)
                    res.spans = [(prior, opener)]
        except EditRejected as e:
            last_reason[req.custom_id] = str(e)
            raise ParseError(f"{req.custom_id}: edit rejected — {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            last_reason[req.custom_id] = str(e)
            raise ParseError(f"{req.custom_id}: malformed edit response — {e}") from e
        return [
            OutputRow(
                key=cid, custom_id=req.custom_id, status="ok",
                data={"field": m.field, "new_text": res.new_text, "edits": res.edits,
                      "retained": round(res.retained, 4),
                      "spans": [{"old": o, "new": n} for o, n in res.spans]},
            )
        ]

    def build_repair(ctx, req, existing, missing, round_no) -> WaveRequest:
        return WaveRequest(
            custom_id=f"{req.custom_id}-r{round_no}",
            params=_with_feedback(req.params, last_reason.get(req.custom_id), round_no),
            lineage=req.lineage,
        )

    return StageSpec(
        name=_stage_name("redit", suffix, edit_pass, m.name), corpus="advice", kind="leaf",
        templates=[m.template], build_requests=build,
        expected_units=lambda req: [req.lineage["cid"]], parse=parse, build_repair=build_repair,
    )


def stories_forbid_patterns(cfg: RunConfig, markers: list[EditMarker]) -> list[str]:
    """What a replacement span may never introduce: every stories marker's
    pattern (the other beats), the configured stories register families
    (tablet/envelope/safeguard), and each marker's own forbid list."""
    pats = [m.pattern for m in markers]
    pats += [f.pattern for f in cfg.filter.register_families if f.field == "text"]
    for m in markers:
        pats += m.forbid_patterns
    seen: list[str] = []
    for p in pats:
        if p not in seen:
            seen.append(p)
    return seen


def stories_edit_spec(
    cfg: RunConfig, markers: list[EditMarker], records: dict[str, dict],
    edit_ids_by_marker: dict[str, list[str]], edit_pass: int,
) -> StageSpec:
    """ONE request per story carrying any removable beat, listing every
    flagged sentence (a story kept for family A is not flagged for A)."""
    last_reason: dict[str, str] = {}
    by_name = {m.name: m for m in markers}
    edit_sets = {name: set(ids) for name, ids in edit_ids_by_marker.items()}
    story_ids = sorted(set().union(*edit_sets.values())) if edit_sets else []
    forbid = stories_forbid_patterns(cfg, markers)
    forbidden_text = "; ".join(m.description for m in markers) + (
        "; a tablet or phone propped against something; anyone writing on the back of an envelope"
    )

    def _flagged(cid: str) -> list[dict]:
        active = [m for m in markers if cid in edit_sets.get(m.name, set())]
        return flag_sentences(records[cid]["text"], active)

    def build(ctx: StageCtx) -> list[WaveRequest]:
        reqs = []
        for cid in story_ids:
            flagged = _flagged(cid)
            if not flagged:
                continue  # (a match that spans no sentence boundary we split on)
            rendered = ctx.templates.render(
                cfg.register_edit.replace_template,
                text=records[cid]["text"],
                flagged=[
                    {"id": f["id"], "sentence": f["sentence"],
                     "family": ", ".join(f["families"]),
                     "description": "; ".join(by_name[n].description for n in f["families"])}
                    for f in flagged
                ],
                forbidden=forbidden_text,
            )
            reqs.append(
                WaveRequest(
                    custom_id=f"redit-beats-{cid}",
                    params=_edit_params(ctx, rendered, REPLACE_SCHEMA),
                    lineage={"cid": cid, "pass": edit_pass,
                             "markers": sorted({n for f in flagged for n in f["families"]}),
                             "n_flagged": len(flagged)},
                )
            )
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        cid = req.lineage["cid"]
        original = records[cid]["text"]
        try:
            out = extract_json_object(outcome["text"])
            res = verify_replace_spans(
                [by_name[n] for n in req.lineage["markers"]], original, _flagged(cid), out,
                forbid, edit_pass,
            )
        except EditRejected as e:
            last_reason[req.custom_id] = str(e)
            raise ParseError(f"{req.custom_id}: edit rejected — {e}") from e
        except (ValueError, KeyError, TypeError) as e:
            last_reason[req.custom_id] = str(e)
            raise ParseError(f"{req.custom_id}: malformed edit response — {e}") from e
        return [
            OutputRow(
                key=cid, custom_id=req.custom_id, status="ok",
                data={"field": "text", "new_text": res.new_text, "edits": res.edits,
                      "retained": round(res.retained, 4),
                      "spans": [{"old": o, "new": n} for o, n in res.spans]},
            )
        ]

    def build_repair(ctx, req, existing, missing, round_no) -> WaveRequest:
        return WaveRequest(
            custom_id=f"{req.custom_id}-r{round_no}",
            params=_with_feedback(req.params, last_reason.get(req.custom_id), round_no),
            lineage=req.lineage,
        )

    return StageSpec(
        name=_stage_name("redit", cfg.register_edit.suffix, edit_pass, "beats"), corpus="stories", kind="leaf",
        templates=[cfg.register_edit.replace_template], build_requests=build,
        expected_units=lambda req: [req.lineage["cid"]], parse=parse, build_repair=build_repair,
    )


def rescore_spec(corpus: str, edited: dict[str, dict], edit_pass: int, suffix: str = "",
                 *, gap: bool = False, max_tokens_factor: int = 1) -> StageSpec:
    """The run's own scoring wave over the EDITED records (overlay-resolved
    template, reference block, params) — one request per edited record.

    ``gap=True`` is the follow-up stage over records the base stage left
    WITHOUT a verdict (a scoring request that hit max_tokens — 4,000 shared
    with adaptive thinking — is a terminal per-request status the stage runner
    never repairs; the production e/s5 waves absorbed those as attrition, but
    an edited record must not ship unscored), re-asked at
    ``max_tokens_factor`` x the wave's cap under its own stage name."""
    template = SCORE_TEMPLATE[corpus]
    schema = SCORE_SCHEMA[corpus]

    def build(ctx: StageCtx) -> list[WaveRequest]:
        reqs = []
        for cid in sorted(edited):
            rec = edited[cid]
            if corpus == "advice":
                rendered = ctx.templates.render(
                    template, system=rec.get("system"), user=rec["user"], assistant=rec["assistant"],
                )
            else:
                instructions = ctx.templates.render(template)
                rendered = f"{instructions}\n\n<story>\n{rec['text']}\n</story>"
            params = wave_params(
                ctx, SCORE_STAGE[corpus],
                [{"role": "user", "content": const_user_content(ctx.constitution_md, rendered)}],
                corpus=corpus,
            )
            params["max_tokens"] = params["max_tokens"] * max_tokens_factor
            if structured(ctx.cfg):
                params["output_config"] = structured_format(schema)
            reqs.append(WaveRequest(custom_id=f"rescore{'-gap' if gap else ''}-{cid}", params=params,
                                    lineage={"cid": cid, "pass": edit_pass, "gap": gap}))
        return reqs

    def parse(ctx: StageCtx, req: WaveRequest, outcome: dict) -> list[OutputRow]:
        from .parsing import parse_score_object

        data = parse_score_object(outcome["text"], schema, f"re-score for {req.custom_id}")
        return [OutputRow(key=req.lineage["cid"], custom_id=req.custom_id, status="ok", data=data)]

    from .stages import repair_request

    return StageSpec(
        name=_stage_name("rescore", suffix, edit_pass, corpus) + ("-gap" if gap else ""),
        corpus=corpus, kind="leaf",
        templates=[template], build_requests=build,
        expected_units=lambda req: [req.lineage["cid"]], parse=parse, build_repair=repair_request,
    )


def rescore_rows(run_dir: Path, corpus: str, suffix: str, edit_pass: int,
                 require: bool = True) -> tuple[dict[str, OutputRow], dict]:
    """ok re-score rows keyed by record id from the base stage plus the gap
    stage when it exists (first verdict wins), and the base stage's stamp."""
    base = run_dir / corpus / _stage_name("rescore", suffix, edit_pass, corpus)
    if not (base / "STAGE.json").exists():
        if require:
            raise FileNotFoundError(f"{corpus}: re-score stage missing ({base}) — run the rescore phase first")
        return {}, {}
    rows = {r.key: r for r in load_outputs(base) if r.status == "ok"}
    gap = base.with_name(base.name + "-gap")
    if (gap / "STAGE.json").exists():
        for r in load_outputs(gap):
            if r.status == "ok" and r.key not in rows:
                rows[r.key] = r
    return rows, read_json(base / "STAGE.json")


# ---------------------------------------------------------------- driver


def prior_spans_for(cfg: RunConfig, root: Path, corpus: str, m: EditMarker) -> dict[str, str]:
    """record id -> the span the ``reform_of`` marker prepended in an earlier
    pass (from that stage's stored outputs), for a reform marker."""
    out: dict[str, str] = {}
    run_dir = cfg.run_dir(root)
    for p in range(1, m.edit_pass):
        sdir = run_dir / corpus / _stage_name("redit", cfg.register_edit.suffix, p, m.reform_of)
        if (sdir / "outputs.jsonl").exists():
            for row in load_outputs(sdir):
                if row.status == "ok" and row.data.get("spans"):
                    out[row.key] = row.data["spans"][0]["new"]
    return out


def _edit_specs_for(cfg: RunConfig, corpus: str, plan: dict, records: dict[str, dict],
                    edit_pass: int, root: Path | None = None) -> list[StageSpec]:
    markers = cfg.register_edit.markers_for(corpus, edit_pass)
    specs: list[StageSpec] = []
    remove = [m for m in markers if m.mode == "remove"]
    for m in markers:
        if m.mode == "add":
            ids = plan["markers"][m.name]["selection"]["edit_ids"]
            if ids:
                prior = prior_spans_for(cfg, root or repo_root(), corpus, m) if m.reform_pattern else None
                specs.append(advice_edit_spec(m, records, ids, edit_pass, cfg.register_edit.suffix, prior))
    if remove:
        ids_by = {m.name: plan["markers"][m.name]["selection"]["edit_ids"] for m in remove}
        if any(ids_by.values()):
            specs.append(stories_edit_spec(cfg, remove, records, ids_by, edit_pass))
    return specs


def apply_edit_rows(record: dict, rows: list[OutputRow]) -> tuple[dict, list[dict], dict[str, str]]:
    """Apply every ok edit row for one record (each row edits ONE field from
    the source text; two rows on the same field are refused). Returns the
    edited copy, the lineage edit entries, and {stage marker: batch id}."""
    rec = json.loads(json.dumps(record))
    edits: list[dict] = []
    batch_ids: dict[str, str] = {}
    touched: set[str] = set()
    for row in rows:
        f = row.data["field"]
        if f in touched:
            raise RuntimeError(f"record {rec_id(record)}: two edits on field {f!r} in one pass")
        touched.add(f)
        rec[f] = row.data["new_text"]
        edits.extend(row.data["edits"])
        for e in row.data["edits"]:
            if row.lineage.get("batch_id"):
                batch_ids[f"redit-{e['marker']}"] = row.lineage["batch_id"]
    return rec, edits, batch_ids


def _run_specs(cfg, engine, ctxs: dict[str, StageCtx], specs: list[tuple[str, StageSpec]],
               run_dir: Path, on_event) -> dict[str, list[OutputRow]]:
    """Run the stages SEQUENTIALLY through ONE engine: the pass runs on the
    live lowprio runner at 200 concurrent — one engine keeps that a single
    budget; the engine's limiter is per wave, so concurrent stages would each
    take 200. A batch engine would submit each stage up front the same way.
    Stamped stages are skipped."""
    results: dict[str, list[OutputRow]] = {}
    for corpus, spec in specs:
        _heartbeat(run_dir, corpus, spec.name)
        on_event(f"[{corpus}] stage {spec.name}")
        results[spec.name] = run_stage(
            engine, ctxs[corpus], spec,
            repair_rounds=cfg.engine.repair_rounds, min_slots_frac=cfg.engine.min_slots_frac,
            on_event=on_event,
        )
    return results


def edited_records_from_outputs(cfg: RunConfig, root: Path, corpus: str, plan: dict,
                                records: dict[str, dict], edit_pass: int,
                                require: bool = True) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, dict], dict]:
    """Fold the stamped edit stages of a pass into edited records. Returns
    (edited cid -> record, cid -> lineage edits, cid -> batch ids, stats)."""
    run_dir = cfg.run_dir(root)
    specs = _edit_specs_for(cfg, corpus, plan, records, edit_pass, root)
    rows_by_cid: dict[str, list[OutputRow]] = {}
    stats: dict = {"stages": {}}
    for spec in specs:
        sdir = run_dir / corpus / spec.name
        if not (sdir / "STAGE.json").exists():
            if require:
                raise FileNotFoundError(f"{corpus}: edit stage {spec.name} not complete ({sdir}) — run the edit phase first")
            continue
        stamp = read_json(sdir / "STAGE.json")
        rows = load_outputs(sdir)
        ok = [r for r in rows if r.status == "ok"]
        for r in ok:
            rows_by_cid.setdefault(r.key, []).append(r)
        stats["stages"][spec.name] = {
            "n_requests": stamp["n_requests"], "expected": stamp["expected_units"],
            "edited": len(ok), "status_counts": stamp["status_counts"],
            "repair_rounds_used": stamp["repair_rounds_used"],
            "retained_mean": round(sum(r.data["retained"] for r in ok) / len(ok), 4) if ok else None,
            "retained_min": min((r.data["retained"] for r in ok), default=None),
            # failure rows are keyed by the wire custom_id; the record id rides in lineage
            "unedited_after_repair": [r.lineage.get("cid", r.key) for r in rows if r.status != "ok"][:50],
            "n_unedited": sum(1 for r in rows if r.status != "ok"),
        }
    edited: dict[str, dict] = {}
    edits: dict[str, list[dict]] = {}
    batch_ids: dict[str, dict] = {}
    for cid, rows in rows_by_cid.items():
        rec, e, b = apply_edit_rows(records[cid], rows)
        edited[cid], edits[cid], batch_ids[cid] = rec, e, b
    return edited, edits, batch_ids, stats


def run_register_edit(cfg: RunConfig, engine, root: Path | None = None, *, edit_pass: int = 1,
                      phase: str = "all", on_event=print) -> dict:
    """phase edit: every edit stage of the pass (all corpora); phase rescore:
    the scoring wave over the edited records; all = both in order. ``engine``
    is any run_wave engine (LiveEngine per the 2026-09-03 directive; the
    batch WaveEngine keeps working). Fully resumable through the stage stamps
    and the engines' per-request caches."""
    root = root or repo_root()
    rc = cfg.register_edit
    if rc is None:
        raise ValueError("config has no `register_edit` section")
    cfg = resolve_environment(cfg, root=root, on_event=on_event)
    plan = load_plan(cfg, root, edit_pass, on_event)
    run_dir = cfg.run_dir(root)
    ctxs: dict[str, StageCtx] = {}
    records: dict[str, dict[str, dict]] = {}
    for corpus in plan["sources"]:
        ctxs[corpus], _ = build_ctx(cfg, corpus, root)
        recs, _shas = load_records_with_line_shas(Path(plan["sources"][corpus]["path"]))
        records[corpus] = {rec_id(r): r for r in recs}

    summary: dict = {"edit_pass": edit_pass, "phases": []}
    if phase in ("edit", "all"):
        specs = [(c, s) for c in plan["sources"] for s in _edit_specs_for(cfg, c, plan, records[c], edit_pass, root)]
        on_event(f"[register-edit pass {edit_pass}] {len(specs)} edit stages, "
                 f"{plan['requests']['total_edits']} requests")
        _run_specs(cfg, engine, ctxs, specs, run_dir, on_event)
        summary["phases"].append("edit")
    if phase in ("rescore", "all"):
        specs = []
        edited_by_corpus: dict[str, dict[str, dict]] = {}
        for corpus in plan["sources"]:
            edited, _e, _b, stats = edited_records_from_outputs(cfg, root, corpus, plan, records[corpus], edit_pass)
            edited_by_corpus[corpus] = edited
            on_event(f"[register-edit pass {edit_pass}] {corpus}: {len(edited)} edited records to re-score")
            if edited:
                specs.append((corpus, rescore_spec(corpus, edited, edit_pass, rc.suffix)))
        _run_specs(cfg, engine, ctxs, specs, run_dir, on_event)
        # the GAP stage: every edited record the base stage left without a
        # verdict is re-asked at double the cap (an edited record never ships
        # unscored; assembly refuses until this set is empty)
        gap_specs = []
        for corpus, edited in edited_by_corpus.items():
            if not edited:
                continue
            rows, _stamp = rescore_rows(run_dir, corpus, rc.suffix, edit_pass)
            missing = {cid: rec for cid, rec in edited.items() if cid not in rows}
            if missing:
                on_event(f"[register-edit pass {edit_pass}] {corpus}: {len(missing)} edited records "
                         f"without a verdict -> gap re-score at 2x max_tokens")
                gap_specs.append((corpus, rescore_spec(corpus, missing, edit_pass, rc.suffix,
                                                        gap=True, max_tokens_factor=2)))
        if gap_specs:
            _run_specs(cfg, engine, ctxs, gap_specs, run_dir, on_event)
        summary["phases"].append("rescore")
    for corpus in plan["sources"]:
        _heartbeat(run_dir, corpus, "done")
    return summary


# -------------------------------------------------------------- assemble


def _as_marker(m_or_fam) -> EditMarker:
    if isinstance(m_or_fam, EditMarker):
        return m_or_fam
    # a filter RegisterFamily -> measured as a records-denominator marker
    return EditMarker(
        name=m_or_fam.name, corpus="advice", field=m_or_fam.field, pattern=m_or_fam.pattern,
        anchored=m_or_fam.anchored, mode="add", kind="prepend_opener", target_rate=0.0, template="x",
    )


def measure_many(records, markers: list[EditMarker]) -> dict[str, dict]:
    """ONE streaming pass over ``records`` (any iterable, e.g. iter_jsonl)
    measuring every marker at once — the corpus is never held in memory
    (the first r2 stories analytics held four 16k-story corpora at once and
    was OOM-killed twice on a shared box)."""
    probes = [(m, compile_marker(m)) for m in markers]
    matched = {m.name: 0 for m in markers}
    denom = {m.name: 0 for m in markers}
    for r in records:
        for m, probe in probes:
            present = has_field(m, r)
            if m.denominator == "field_present" and not present:
                continue
            denom[m.name] += 1
            if present and probe(str(r.get(m.field) or "")):
                matched[m.name] += 1
    return {
        m.name: {"matched": matched[m.name], "denominator": denom[m.name],
                 "rate": matched[m.name] / denom[m.name] if denom[m.name] else 0.0}
        for m in markers
    }


def gram_doc_freqs(records, field: str, phrases: list[str]) -> dict[str, float]:
    """One streaming pass: fraction of records whose field carries each phrase
    as consecutive TOKENS (the survey's tokenizer), via per-record k-gram sets
    — exact document frequency for thousands of candidate phrases at once,
    where a regex per phrase would not scale."""
    cands: dict[int, dict[tuple, str]] = {}
    for ph in phrases:
        t = tuple(ph.split())
        cands.setdefault(len(t), {})[t] = ph
    hits = {ph: 0 for ph in phrases}
    n = 0
    for r in records:
        n += 1
        toks = words(str(r.get(field) or ""))
        for k, by_gram in cands.items():
            if len(toks) < k:
                continue
            seen = set(zip(*(toks[i:] for i in range(k))))
            for g in seen.intersection(by_gram):
                hits[by_gram[g]] += 1
    return {ph: (hits[ph] / n if n else 0.0) for ph in phrases}


def phrase_rates_many(records, field: str, phrases: list[str]) -> dict[str, float]:
    """One streaming pass: fraction of records whose field carries each
    phrase as consecutive words (apostrophes normalized both sides)."""
    rxs = {
        ph: re.compile(r"\b" + r"\W+".join(re.escape(w) for w in _norm_apostrophes(ph).split()) + r"\b", re.IGNORECASE)
        for ph in phrases
    }
    hits = {ph: 0 for ph in phrases}
    n = 0
    for r in records:
        n += 1
        text = _norm_apostrophes(str(r.get(field) or ""))
        for ph, rx in rxs.items():
            if rx.search(text):
                hits[ph] += 1
    return {ph: (hits[ph] / n if n else 0.0) for ph in phrases}


def _norm_apostrophes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _phrase_rate(records: list[dict], field: str, phrase: str) -> float:
    """Fraction of records whose field carries the phrase as consecutive
    words — the ONE measure used for v1, r2 and production alike (the survey
    table only proposes candidates); apostrophes are normalized on both sides
    so a corpus mixing straight and curly quotes is not miscounted."""
    rx = re.compile(
        r"\b" + r"\W+".join(re.escape(w) for w in _norm_apostrophes(phrase).split()) + r"\b",
        re.IGNORECASE,
    )
    n = sum(1 for r in records if rx.search(_norm_apostrophes(str(r.get(field) or ""))))
    return n / len(records) if records else 0.0


def _sample_jsonl(path: Path, n: int, seed: int) -> list[dict]:
    """A seeded sample of a JSONL file's records without loading the file."""
    total = sum(1 for _ in open(path, "rb"))
    if n >= total:
        return list(iter_jsonl(path))
    import random

    rng = random.Random(f"{seed}:register-survey")
    keep = set(rng.sample(range(total), n))
    out = []
    with open(path, "rb") as f:
        for i, line in enumerate(f):
            if i in keep and line.strip():
                out.append(json.loads(line))
    return out


def monoculture_check(cfg: RunConfig, corpus: str, v1_path: Path, r2_path: Path, prod_path: Path,
                      spans: list[str], sizes: tuple[int, ...] = (4, 6)) -> dict:
    """Did the edits create a NEW monoculture? Re-run the survey's n-gram
    tables on the edited corpus and flag every phrase at >= min_frac of r2,
    <= prod_max_frac of production, and < min_frac of v1. Also rank the
    content 3-grams of the replacement spans themselves by how many r2
    records carry them. STREAMING: candidates come from the survey's n-gram
    tables over a seeded 4,000-record sample of the edited final (the
    survey's own rule: a few thousand records find every >=2% family); every
    rate is exact, by one regex pass per corpus over the full file — the
    corpora are never held in memory."""
    rc = cfg.register_edit
    field = "text" if corpus == "stories" else "assistant"
    new_monocultures: list[dict] = []
    tables: dict = {}
    r2_sample = _sample_jsonl(r2_path, 4000, rc.seed)
    cand: dict[int, list[dict]] = {}
    for n in sizes:
        # EVERY sample gram at or above the floor is a candidate (a top-k cut
        # hid "screwed to the wall" at 1.26% behind the generic top grams on
        # 2026-09-03); exact rates then come from per-record gram sets
        cand[n] = ngram_table(r2_sample, field, n, top=10**6, min_frac=rc.monoculture_min_frac)["rows"]
    del r2_sample
    phrases = sorted({row["ngram"] for rows in cand.values() for row in rows})
    r_rates = gram_doc_freqs(iter_jsonl(r2_path), field, phrases)
    p_rates = gram_doc_freqs(iter_jsonl(prod_path), field, phrases)
    v_rates = gram_doc_freqs(iter_jsonl(v1_path), field, phrases)
    for n in sizes:
        rows = []
        for row in cand[n]:
            ph = row["ngram"]
            r_rate, p_rate, v_rate = r_rates[ph], p_rates[ph], v_rates[ph]
            entry = {"ngram": ph, "r2": round(r_rate, 4), "v1": round(v_rate, 4),
                     "production": round(p_rate, 4), "r2_sample_table": row["frac_of_field"]}
            rows.append(entry)
            if (r_rate >= rc.monoculture_min_frac
                    and p_rate <= rc.monoculture_prod_max_frac
                    and v_rate < rc.monoculture_min_frac):
                new_monocultures.append({"n": n, **entry})
        rows.sort(key=lambda e: -e["r2"])
        tables[str(n)] = {"n_candidates": len(rows), "rows": rows[:200],
                          "control_only": [e for e in rows if e["production"] <= rc.monoculture_prod_max_frac and e["r2"] >= rc.monoculture_min_frac]}
    span_grams: dict[str, int] = {}
    for s in spans:
        w = words(s)
        seen = {" ".join(w[i : i + 3]) for i in range(len(w) - 2)}
        for g in seen:
            if any(len(tok) >= 4 and tok not in ("that", "with", "from", "into", "onto", "they", "them", "were", "have", "this", "then", "than", "there", "their") for tok in g.split()):
                span_grams[g] = span_grams.get(g, 0) + 1
    top_span = sorted(span_grams.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    n_r2 = sum(1 for _ in open(r2_path, "rb")) or 1
    return {
        "new_monocultures": new_monocultures,
        "tables": tables,
        "replacement_3grams": [
            {"ngram": g, "records": c, "frac_of_corpus": round(c / n_r2, 4)} for g, c in top_span
        ],
        "params": {"min_frac": rc.monoculture_min_frac, "prod_max_frac": rc.monoculture_prod_max_frac},
    }


def assemble(cfg: RunConfig, root: Path | None = None, counter=None, *, edit_pass: int = 1,
             corpora: tuple[str, ...] | None = None, reuse_final: bool = False, on_event=print) -> dict:
    """Fold the pass's stamped edit + re-score stages into the r2 record set,
    run the FULL filter chain (r2 config: corrected caps, no downsample) to
    final-<run><suffix>-<corpus>.jsonl, and write the manifests: per corpus
    beside the final (per-record v1 sha + verified edits + stats) and one
    run-level file (batch ids, plan, before/after register table vs
    production, verification, re-score deltas, monoculture check)."""
    root = root or repo_root()
    rc = cfg.register_edit
    if rc is None:
        raise ValueError("config has no `register_edit` section")
    if counter is None:
        from .tokens import TokenCounter

        counter = TokenCounter.load(cfg.resolve_data_dir(root) / "tokenizers")
    plan = load_plan(cfg, root, edit_pass, on_event)
    run_dir = cfg.run_dir(root)
    # ``corpora`` re-assembles a subset (offline; e.g. after a cap correction)
    # and MERGES into the existing run-level manifest, so the other corpus's
    # entry is kept rather than dropped
    prior = read_json(run_manifest_path(cfg, root)) if run_manifest_path(cfg, root).exists() else {}
    out: dict = {
        "run_id": cfg.run_id, "suffix": rc.suffix, "edit_pass": edit_pass, "plan": plan,
        "corpora": dict(prior.get("corpora") or {}) if corpora else {},
        "batch_ids": list(prior.get("batch_ids") or []) if corpora else [],
        "assembled_at": _utcnow(),
        "config_register_families": [f.model_dump() for f in cfg.filter.register_families],
        "divergence": (
            "r2 register-matching edit pass: production registers the v1 bans removed are "
            "re-added by verified span edits to production's share (Deployment: system labels, "
            "verdict-label response openers, sanity/gut-check user openers) and control-only story "
            "beats are reduced to production's share by rewriting only the carrying sentence(s); "
            "every edited record re-scored with the run's own scoring wave, floors 8/8/8/8 and "
            "every offline gate re-applied; v1 finals immutable, per-record v1 sha lineage."
        ),
    }
    for corpus in plan["sources"]:
        if corpora is not None and corpus not in corpora:
            continue
        src_records, src_shas = load_records_with_line_shas(Path(plan["sources"][corpus]["path"]))
        records = {rec_id(r): r for r in src_records}
        v1_sha = {rec_id(r): s for r, s in zip(src_records, src_shas)}
        edited, edits, batch_ids, stats = edited_records_from_outputs(
            cfg, root, corpus, plan, records, edit_pass
        )
        # re-score verdicts
        rs_dir = run_dir / corpus / _stage_name("rescore", rc.suffix, edit_pass, corpus)
        rescored: dict[str, OutputRow] = {}
        rescore_stamp: dict = {}
        if edited:
            rescored, rescore_stamp = rescore_rows(run_dir, corpus, rc.suffix, edit_pass)
            unscored = sorted(cid for cid in edited if cid not in rescored)
            if unscored:
                # user directive 2026-09-03: nothing is written to the r2 finals
                # until every edited record has passed verification AND re-score
                raise RuntimeError(
                    f"{corpus}: {len(unscored)} edited records have no re-score verdict "
                    f"(first: {unscored[:5]}) — resume the rescore phase before assembling; "
                    f"an edited record never ships unscored"
                )
        assembled: list[dict] = []
        n_unscored = 0
        for r in src_records:
            cid = rec_id(r)
            if cid in edited:
                rec = edited[cid]
                lin = rec["lineage"]
                lin["v1_sha256"] = lin.get("v1_sha256") or v1_sha[cid]
                lin["register_edits"] = (lin.get("register_edits") or []) + edits[cid]
                if lin.get("scores_v1") is None:
                    lin["scores_v1"] = r.get("scores")
                row = rescored.get(cid)
                rec["scores"] = row.data if row is not None else None
                if row is None:
                    n_unscored += 1
                bids = dict(lin.get("batch_ids") or {})
                bids.update(batch_ids[cid])
                if row is not None and row.lineage.get("batch_id"):
                    bids[f"rescore-p{edit_pass}"] = row.lineage["batch_id"]
                lin["batch_ids"] = bids
            else:
                rec = json.loads(json.dumps(r))
                rec["lineage"]["v1_sha256"] = rec["lineage"].get("v1_sha256") or v1_sha[cid]
            assembled.append(rec)
        apath = assembled_path(cfg, root, edit_pass, corpus)
        apath.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl_atomic(apath, (json.dumps(r, ensure_ascii=False) for r in assembled))
        on_event(f"[assemble {corpus}] {len(assembled)} records ({len(edited)} edited, "
                 f"{n_unscored} edited-but-unscored) -> {apath.name}; running the filter chain")

        final_path = cfg.final_path(root, corpus, rc.suffix)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        fr_path = filter_report_path(final_path)
        reusable = (
            reuse_final and final_path.exists() and fr_path.exists()
            and read_json(fr_path).get("input_count") == len(assembled)
            and {rec_id(r) for r in iter_jsonl(final_path)} <= {rec_id(r) for r in assembled}
        )
        if reusable:
            # the filter chain already ran over THIS assembled set (its report is
            # newer than the assembled file) and the process died in the
            # analytics after it (the 2026-09-03 OOM): reuse its output rather
            # than spend the chain again; the report object is rebuilt from the
            # persisted filter report, which the mirror gate also reads
            on_event(f"[assemble {corpus}] reusing the existing final + filter report ({fr_path.name})")
            from types import SimpleNamespace

            frj = read_json(fr_path)
            report = SimpleNamespace(
                input_count=frj["input_count"], output_count=frj["output_count"],
                rule_counts=frj["rule_counts"], official_tokens=frj["official_tokens"],
                length_gate=frj.get("length_gate"), register_family_stats=frj.get("register_family_stats", []),
                drops=frj.get("drops", []), acceptance_failures=frj.get("acceptance_failures", []),
            )
        else:
            report = filter_corpus(corpus, assembled, cfg, counter, final_path)
        del assembled
        # STREAMING analytics from here on: the final, the v1 final and the
        # production reference are read from disk pass by pass, never held
        # together (four 16k-story corpora in memory OOM-killed this step twice)
        v1_path = cfg.final_path(root, corpus, rc.source_suffix)
        prod_path = _resolve(root, rc.production_reference[corpus])
        final_ids = {rec_id(r) for r in iter_jsonl(final_path)}

        # before/after register table vs production: every marker of this
        # corpus + every configured register family on this corpus's fields
        fields = {"advice": {"system", "user", "assistant"}, "stories": {"text"}}[corpus]
        measured = list(rc.markers_for(corpus))
        fam_markers = [
            _as_marker(fam) for fam in cfg.filter.register_families
            if fam.field in fields and fam.name not in {m.name for m in measured}
        ]
        allm = measured + fam_markers
        rates = {
            "v1": measure_many(iter_jsonl(v1_path), allm),
            "r2": measure_many(iter_jsonl(final_path), allm),
            "production": measure_many(iter_jsonl(prod_path), allm),
        }
        table: dict = {}
        for m in measured:
            table[m.name] = {"kind": "marker", "field": m.field, "mode": m.mode, "target_rate": m.target_rate,
                             **{k: rates[k][m.name] for k in ("v1", "r2", "production")}}
        caps = {fam.name: fam.max_frac for fam in cfg.filter.register_families}
        for m in fam_markers:
            table[m.name] = {"kind": "family", "field": m.field, "cap": caps[m.name],
                             **{k: rates[k][m.name] for k in ("v1", "r2", "production")}}

        # re-score deltas over the edited records
        axes = ["alignment", "realism"] if corpus == "advice" else ["constitution_consistency", "quality"]
        flags = ["refusal", "ai_named"] if corpus == "advice" else ["ai_named", "factual_claims"]
        deltas: dict = {"n_edited": len(edited), "n_rescored": len(rescored), "n_unscored": n_unscored,
                        "axes": {}, "flags": {}, "dropped_edited": {}, "dropped_unedited": {}}
        for axis in axes:
            before: dict[str, int] = {}
            after: dict[str, int] = {}
            moved = {"down": 0, "same": 0, "up": 0}
            for cid in edited:
                s1 = ((records[cid].get("scores") or {}).get(axis))
                row = rescored.get(cid)
                s2 = row.data.get(axis) if row is not None else None
                before[str(s1)] = before.get(str(s1), 0) + 1
                after[str(s2)] = after.get(str(s2), 0) + 1
                if s1 is not None and s2 is not None:
                    moved["down" if s2 < s1 else "up" if s2 > s1 else "same"] += 1
            deltas["axes"][axis] = {"v1": dict(sorted(before.items())), "r2": dict(sorted(after.items())), "moved": moved}
        for fl in flags:
            flips = {"false_to_true": 0, "true_to_false": 0}
            for cid in edited:
                a = bool((records[cid].get("scores") or {}).get(fl))
                row = rescored.get(cid)
                if row is None:
                    continue
                b = bool(row.data.get(fl))
                if a != b:
                    flips["false_to_true" if b else "true_to_false"] += 1
            deltas["flags"][fl] = flips
        for d in report.drops:
            bucket = "dropped_edited" if d["id"] in edited else "dropped_unedited"
            deltas[bucket][d["rule"]] = deltas[bucket].get(d["rule"], 0) + 1
        deltas["n_edited_in_final"] = sum(1 for cid in edited if cid in final_ids)
        deltas["n_edited_dropped"] = len(edited) - deltas["n_edited_in_final"]

        # unedited records: fields identical to the source (only lineage.v1_sha256 added)
        n_identical = 0
        n_edited_any = 0
        per_record = {}
        model_cls = AdviceRecord if corpus == "advice" else StoryRecord
        for fr_ in iter_jsonl(final_path):  # one streaming pass over the final
            cid = rec_id(fr_)
            lin = fr_["lineage"]
            per_record[cid] = {
                "v1_sha256": lin["v1_sha256"],
                "edits": [
                    {"field": e["field"], "marker": e["marker"], "old_span_sha": e["old_span_sha"],
                     "new_span_sha": e["new_span_sha"], "pass": e["pass"]}
                    for e in (lin.get("register_edits") or [])
                ],
            }
            if lin.get("register_edits"):
                n_edited_any += 1
            if cid in edited or cid not in records:
                continue
            # every field but the lineage (v1_sha256 added) and the derived
            # token counts (recomputed by the filter chain) must be identical
            a = model_cls.model_validate(fr_).model_dump(exclude={"lineage", "token_counts"})
            b = model_cls.model_validate(records[cid]).model_dump(exclude={"lineage", "token_counts"})
            if a == b:
                n_identical += 1
        # replacement spans (for the monoculture check) from the stage outputs
        span_texts: list[str] = []
        for spec in _edit_specs_for(cfg, corpus, plan, records, edit_pass, root):
            sdir = run_dir / corpus / spec.name
            if (sdir / "STAGE.json").exists():
                for row in load_outputs(sdir):
                    if row.status == "ok":
                        span_texts.extend(s["new"] for s in row.data.get("spans", []))
        mono = monoculture_check(cfg, corpus, v1_path, final_path, prod_path, span_texts)
        if mono["new_monocultures"]:
            on_event(f"[assemble {corpus}] NEW MONOCULTURE(S) after the edits: "
                     + "; ".join(f"{x['ngram']!r} r2 {x['r2']:.2%} / prod {x['production']:.2%}" for x in mono["new_monocultures"]))

        # verification stats across EVERY pass so far (stage names are unique
        # per pass) and the earlier passes' re-score / filter blocks, carried
        # from the corpus manifest this assembly overwrites
        all_stages: dict = {}
        for p_ in range(1, edit_pass + 1):
            try:
                _e, _ed, _b, st_p = edited_records_from_outputs(cfg, root, corpus, plan_for_pass(cfg, root, p_) if p_ != edit_pass else plan,
                                                                {rec_id(r): r for r in (load_records_with_line_shas(source_path(cfg, root, corpus, p_))[0])} if p_ != edit_pass else records,
                                                                p_, require=False)
                all_stages.update(st_p["stages"])
            except FileNotFoundError:
                pass
        stats["stages"] = {**all_stages, **stats["stages"]}
        prior_manifest_path = register_edit_manifest_path(final_path)
        prior_passes: dict = {}
        if prior_manifest_path.exists():
            pm = read_json(prior_manifest_path)
            prior_passes = dict(pm.get("prior_passes") or {})
            if pm.get("edit_pass") is not None and pm["edit_pass"] < edit_pass:
                prior_passes[str(pm["edit_pass"])] = {"rescore": pm.get("rescore"), "filter": pm.get("filter"),
                                                      "n_edited_this_pass_in_final": pm.get("n_edited_this_pass_in_final")}
        # manifests (per_record built in the streaming pass above)
        corpus_manifest = {
            "corpus": corpus, "run_id": cfg.run_id, "suffix": rc.suffix, "edit_pass": edit_pass,
            "source": plan["sources"][corpus], "production_reference": plan["production"][corpus],
            "final": str(final_path), "final_sha256": sha256_file(final_path),
            "n_records_final": len(final_ids),
            # records in the final carrying ANY verified edit (every pass) — what
            # the mirror gate re-counts; this pass's own count is in rescore.deltas
            "n_edited_in_final": n_edited_any,
            "n_edited_this_pass_in_final": deltas["n_edited_in_final"],
            "n_unedited_identical_to_source": n_identical,
            "markers": {m.name: {"pattern": m.pattern, "anchored": m.anchored, "field": m.field,
                                 "mode": m.mode, "kind": m.kind, "target_rate": m.target_rate,
                                 "denominator": m.denominator, "target_source": m.target_source}
                        for m in rc.markers_for(corpus)},
            "register_table": table,
            "verification": stats,
            "rescore": {"stage": rescore_stamp, "deltas": deltas},
            "prior_passes": prior_passes,
            "filter": {"input": report.input_count, "output": report.output_count,
                       "rule_counts": report.rule_counts, "official_tokens": report.official_tokens,
                       "length_gate": report.length_gate, "register_family_stats": report.register_family_stats},
            "monoculture": mono,
            "records": per_record,
            "written_at": _utcnow(),
        }
        atomic_write(register_edit_manifest_path(final_path), json.dumps(corpus_manifest, indent=1, ensure_ascii=False))
        out["corpora"][corpus] = {k: v for k, v in corpus_manifest.items() if k != "records"}
        bids = set()
        for spec in _edit_specs_for(cfg, corpus, plan, records, edit_pass, root):
            for stamp in (run_dir / corpus / spec.name).rglob("STAMP.json"):
                bids |= set(read_json(stamp).get("batch_ids", {}).values())
        for sd in (rs_dir, rs_dir.with_name(rs_dir.name + "-gap")):
            for stamp in sd.rglob("STAMP.json") if sd.exists() else []:
                bids |= set(read_json(stamp).get("batch_ids", {}).values())
        gap_dir = rs_dir.with_name(rs_dir.name + "-gap")
        if (gap_dir / "STAGE.json").exists():
            rescore_stamp = {"base": rescore_stamp, "gap": read_json(gap_dir / "STAGE.json")}
        out["batch_ids"] = sorted(set(out["batch_ids"]) | bids)
        on_event(
            f"[assemble {corpus}] final {len(final_ids)} records / {report.official_tokens:,} official "
            f"tokens (drops {json.dumps(report.rule_counts)}); edited in final "
            f"{deltas['n_edited_in_final']}, edited dropped {deltas['n_edited_dropped']}; "
            f"length gate {'PASSED' if report.length_gate and report.length_gate['passed'] else 'FAILED/none'}"
        )
        for f in report.acceptance_failures:
            on_event(f"[assemble {corpus}] ACCEPTANCE GATE FAILURE: {f}")
    atomic_write(run_manifest_path(cfg, root), json.dumps(out, indent=1, ensure_ascii=False))
    on_event(f"run-level register-edit manifest written to {run_manifest_path(cfg, root)}")
    return out


def slack_line(n_edits: int, engine: str = "live") -> str:
    if engine == "live":
        return (
            "200 opus lowprio ~1h (corpus_generator control register-matching edits + "
            f"re-scores, live; ~{2 * n_edits:,} requests)"
        )
    return (
        f"Batch API: ~{2 * n_edits:,} claude-opus-5 requests (corpus_generator control "
        f"register-matching edits + re-scores — async, separate batch pool, not live concurrents)"
    )


def build_live_engine(cfg: RunConfig, on_event=print):
    """The pass's engine per the 2026-09-03 directive: the repo's live runner
    on the LOWPRIO key (client.build_async_client pins base_url and refuses
    the highprio key), 200 max concurrent shrinking on 429/529, bounded
    per-request retries, cached-errored reissue, and the degraded-provider
    guard (>30% errored -> pause 10 min and resume; stop after 45 min)."""
    from .client import build_async_client
    from .live import LiveEngine

    return LiveEngine(
        lambda: build_async_client("live"),
        model=cfg.engine.model,
        max_concurrency=cfg.register_edit.max_concurrency,
        retry_rounds=cfg.engine.retry_rounds,
        max_rejected_frac=cfg.engine.max_rejected_frac,
        degraded_error_frac=0.30,
        degraded_pause_s=600.0,
        degraded_max_s=2700.0,
        on_event=on_event,
    )
