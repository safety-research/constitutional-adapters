"""Lenient-but-loud output parsers: strip code fences, locate the JSON or the
tag, and raise ParseError (never crash the wave) on anything malformed."""

from __future__ import annotations

import json
import re

from .stages import ParseError, normalize_content

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text)


def extract_json_array(text: str) -> list:
    t = _strip_fences(text)
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end <= start:
        raise ParseError("no JSON array found in response")
    try:
        out = json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON array does not parse: {e}") from e
    if not isinstance(out, list):
        raise ParseError("parsed JSON is not an array")
    return out


def extract_json_object(text: str) -> dict:
    t = _strip_fences(text)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ParseError("no JSON object found in response")
    try:
        out = json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON object does not parse: {e}") from e
    if not isinstance(out, dict):
        raise ParseError("parsed JSON is not an object")
    return out


def extract_tag(text: str, tag: str, required: bool = True) -> str | None:
    m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.DOTALL)
    if not m:
        if required:
            raise ParseError(f"missing <{tag}> block in response")
        return None
    content = m.group(1)
    if required and not content.strip():
        raise ParseError(f"<{tag}> block is empty")
    return content


def text_before_tag(text: str, tag: str) -> str:
    idx = text.find(f"<{tag}>")
    return text[:idx].strip() if idx > 0 else ""


def extract_improved(text: str, tag: str, wrapper: str | None = None) -> tuple[str, str]:
    """A rewrite wave's (improved text, problems-list preamble).

    ``wrapper`` names the tag the INPUT document was handed to the model in
    (S4 wraps the draft in ``<story>…</story>``); the model sometimes echoes
    that wrapper around its rewrite, and a bare ``</story>`` then rides into
    trainable text (0.70% of production v2 stories, 4.1% of the control
    pilot's). A leading/trailing echo of exactly that wrapper is stripped
    here; anything else stays, for the filter's stray-tag gate to judge."""
    improved = extract_tag(text, tag)
    if wrapper:
        improved = strip_wrapper_echo(improved, wrapper)
    return improved, text_before_tag(text, tag)


def strip_wrapper_echo(text: str, wrapper: str) -> str:
    """Remove a leading ``<wrapper>`` and/or trailing ``</wrapper>`` echo (with
    surrounding whitespace) — only at the very edges of the text."""
    t = text.strip()
    t = re.sub(rf"^<{re.escape(wrapper)}(?:_\d+)?>\s*", "", t)
    t = re.sub(rf"\s*</{re.escape(wrapper)}(?:_\d+)?>$", "", t)
    return t


def require_str(item: dict, key: str, ctx: str) -> str:
    v = item.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ParseError(f"{ctx}: field {key!r} missing or not a non-empty string")
    return v.strip()


def require_bool(item: dict, key: str, ctx: str) -> bool:
    v = item.get(key)
    if not isinstance(v, bool):
        raise ParseError(f"{ctx}: field {key!r} missing or not a bool")
    return v


def require_int(item: dict, key: str, ctx: str, lo: int = 0, hi: int = 10) -> int:
    v = item.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or not (lo <= v <= hi):
        raise ParseError(f"{ctx}: field {key!r} missing or not an int in [{lo},{hi}]")
    return v


_REQUIRE_BY_TYPE = {"integer": require_int, "boolean": require_bool, "string": require_str}


def parse_score_object(text: str, schema: dict, label: str) -> dict:
    """Parse + validate a scoring response against its JSON schema — the ONE
    place score fields are read, structured output or prompt-enforced alike."""
    obj = extract_json_object(text)
    return {
        key: _REQUIRE_BY_TYPE[spec["type"]](obj, key, label)
        for key, spec in schema["properties"].items()
    }


def parse_slot_array(
    text: str,
    valid_slots: set[str],
    fields: list[str],
    content_field: str,
) -> dict[str, dict]:
    """Parse a structural-wave JSON array into slot -> item.

    SALVAGE mode: a malformed item, an unknown or duplicate slot id, or
    case-folded duplicate content drops THAT item — those slots come back as
    missing units and go to the bounded repair round. Only an unparseable
    response raises ParseError outright. Cardinality is judged by the CALLER.
    """
    items = extract_json_array(text)
    out: dict[str, dict] = {}
    issues: list[str] = []
    seen_content: set[str] = set()
    for i, item in enumerate(items):
        try:
            if not isinstance(item, dict):
                raise ParseError(f"item {i} is not an object")
            slot = require_str(item, "slot", f"item {i}")
            if slot not in valid_slots:
                raise ParseError(f"item {i}: unknown slot id {slot!r}")
            if slot in out:
                raise ParseError(f"duplicate slot id {slot!r}")
            for f in fields:
                require_str(item, f, f"slot {slot}")
            folded = normalize_content(item[content_field])
            if folded in seen_content:
                raise ParseError(f"slot {slot}: duplicate content (case-folded)")
        except ParseError as e:
            issues.append(str(e))
            continue
        seen_content.add(folded)
        out[slot] = item
    if not out:
        raise ParseError(
            "no valid items in slot array"
            + (f" (issues: {'; '.join(issues[:3])})" if issues else "")
        )
    return out
