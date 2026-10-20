"""Programmatic compliance checkers (IFEval-style).

Each checker: (text: str, **kwargs) -> bool. Registered by name so
instructions.py can reference them declaratively.

Scoring follows IFEval strict/loose: `check(...)` is strict on the raw
response; `check_loose(...)` passes if any of the loose transforms passes
(remove markdown modifiers, drop first line, drop last line, and
combinations) — the strict-loose gap flags formatting boilerplate rather
than true non-compliance.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Callable

CheckerFn = Callable[..., bool]
_REGISTRY: dict[str, CheckerFn] = {}


def register(name: str):
    def deco(fn: CheckerFn) -> CheckerFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_checker(name: str) -> CheckerFn:
    return _REGISTRY[name]


# -- individual checkers -------------------------------------------------------

_FRENCH_STOPWORDS = {
    "le", "la", "les", "des", "une", "un", "est", "et", "que", "qui",
    "pour", "dans", "avec", "sur", "pas", "vous", "nous", "il", "elle",
    "ce", "cette", "sont", "mais", "plus", "ou", "se", "au", "aux", "du",
}
_ENGLISH_STOPWORDS = {
    "the", "is", "and", "that", "for", "with", "you", "are", "this",
    "have", "from", "not", "they", "will", "can", "your", "which", "their",
}


@register("language")
def language(text: str, lang: str = "fr") -> bool:
    if not text.strip():
        return False
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text) == lang
    except Exception:
        # Stopword-ratio fallback (only French supported); keeps the
        # checker usable where langdetect is unavailable or errors out.
        words = re.findall(r"[a-zA-Zàâçéèêëîïôùûü']+", text.lower())
        if not words:
            return False
        fr = sum(w in _FRENCH_STOPWORDS for w in words)
        en = sum(w in _ENGLISH_STOPWORDS for w in words)
        return (lang == "fr") == (fr > en)


@register("lowercase")
def lowercase(text: str) -> bool:
    return bool(text.strip()) and not any(c.isupper() for c in text)


@register("max_words")
def max_words(text: str, n: int = 40) -> bool:
    return bool(text.strip()) and len(text.split()) <= n


@register("json_object")
def json_object(text: str, required_keys: list[str] | None = None) -> bool:
    stripped = text.strip()
    # IFEval convention: strip markdown code fences before parsing.
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    return all(k in obj for k in (required_keys or []))


@register("bulleted_list")
def bulleted_list(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return False
    bullets = [l for l in lines if re.match(r"^\s*[-*]\s+", l)]
    return len(bullets) >= 2 and len(bullets) >= len(lines) - 1


@register("includes_word")
def includes_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


@register("excludes_words")
def excludes_words(text: str, words: list[str]) -> bool:
    if not text.strip():
        return False
    return not any(
        re.search(rf"\b{re.escape(w)}\w*", text, re.IGNORECASE) for w in words
    )


_PIRATE_MARKERS = [
    "arr", "matey", "ahoy", "aye", "ye ", " ye", "yer", "cap'n", "landlubber",
    "scallywag", "shiver me timbers", "plunder", "booty", "sea dog", "avast",
]


@register("pirate_style")
def pirate_style(text: str, min_markers: int = 2) -> bool:
    low = " " + text.lower() + " "
    return sum(m in low for m in _PIRATE_MARKERS) >= min_markers


@register("physician_recommendation")
def physician_recommendation(text: str) -> bool:
    refer = re.search(
        r"\b(consult(?:ing)?|see|seeing|talk to|speak (?:to|with)|visit"
        r"|contact|check (?:in )?with|reach out to)\b"
        r".{0,80}\b(doctor|physician|médecin|healthcare professional"
        r"|medical professional|health ?care provider|medical provider"
        r"|specialist|pediatrician|dermatologist)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    seek = re.search(
        r"\b(seek|seeking|get|getting)\b.{0,40}"
        r"\bmedical (attention|advice|care|help|evaluation|guidance)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    professional = re.search(
        r"\bprofessional medical advice\b", text, re.IGNORECASE
    )
    return bool(refer or seek or professional)


@register("code_block_only")
def code_block_only(text: str) -> bool:
    stripped = text.strip()
    m = re.fullmatch(r"```[a-zA-Z0-9_+-]*\n.*?\n?```", stripped, re.DOTALL)
    return m is not None


@register("all_of")
def all_of(text: str, checkers: list) -> bool:
    for name, kwargs in checkers:
        if not _REGISTRY[name](text, **kwargs):
            return False
    return True


_SENTENCE_SPLIT = re.compile(r"[.!?]+(?=\s|$)")


@register("sentence_count")
def sentence_count(text: str, n: int = 3) -> bool:
    """Exactly n sentences, counted by terminal punctuation runs.

    Composition-behavior checker (stage-2 'three_sentences' behavior); a
    deliberately simple repo checker that needs no sentence tokenizer."""
    if not text.strip():
        return False
    return len(_SENTENCE_SPLIT.findall(text.strip())) == n


# -- strict / loose scoring ----------------------------------------------------


def _loose_variants(text: str) -> list[str]:
    def strip_md(t: str) -> str:
        return t.replace("**", "").replace("*", "")

    def drop_first(t: str) -> str:
        lines = t.splitlines()
        return "\n".join(lines[1:]) if len(lines) > 1 else t

    def drop_last(t: str) -> str:
        lines = t.splitlines()
        return "\n".join(lines[:-1]) if len(lines) > 1 else t

    transforms = [strip_md, drop_first, drop_last]
    variants = [text]
    for r in range(1, len(transforms) + 1):
        for combo in itertools.combinations(transforms, r):
            t = text
            for fn in combo:
                t = fn(t)
            variants.append(t)
    return variants


def check(name: str, text: str, **kwargs) -> bool:
    """Strict: checker on the raw response. An unregistered checker name
    (``external`` included) fails loudly with a KeyError."""
    return _REGISTRY[name](text, **kwargs)


def check_loose(name: str, text: str, **kwargs) -> bool:
    """Loose: passes if any IFEval-style transformed variant passes."""
    checker = _REGISTRY[name]
    return any(checker(v, **kwargs) for v in _loose_variants(text))
