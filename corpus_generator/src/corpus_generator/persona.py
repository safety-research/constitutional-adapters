"""Persona guard: Claude/Anthropic never appears in TRAINABLE text; mentions
in generated non-trainable metadata are counted and reported, never dropped.

The HARD gate (check_trainable) covers exactly the fields that become
training text — advice: system/user/assistant; stories: text — with any
case-insensitive word-boundary hit of "claude"/"anthropic" dropping the
record. That is the spec's acceptance property, enforced strictly.

Everything else the model generated (rewrite_problems, critiques,
diagnostic/draft lineage, principle/scenario/category prose, scores.rationale)
is still scanned — recursively, so a new generated field is covered unless
deliberately excluded — but a hit there is REPORTED (field path + sample),
not a drop (check_metadata). The pilot showed why: 45/144 stories (31%) and
4 advice records were dropped solely for constitution-quoting critique notes
("The constitution says Claude should …") in rewrite_problems and
draft/diagnostic lineage — provenance that never enters training data, so
each mention threw away a clean record. (Amends the earlier
stricter-than-spec any-generated-field drop, 2026-09-01.)

Excluded from scanning entirely: fields we author ourselves — ids, hashes,
run bookkeeping — and numerics. Keep that list NARROW. Prompt templates and
injected constitution text remain out of scope (stage inputs, never record
fields).
"""

from __future__ import annotations

from dataclasses import dataclass
import re

PERSONA_RE = re.compile(r"\b(claude|anthropic)\b", re.IGNORECASE)

# the trainable text per corpus — the HARD persona gate
TRAINABLE_FIELDS = {
    "advice": ["system", "user", "assistant"],
    "stories": ["text"],
}

# Non-generated subtrees/fields, excluded from the metadata scan. Everything
# here is authored by US (ids, hashes, run bookkeeping) or numeric — never
# model prose. Keep this list NARROW: an over-wide exclusion hides real
# mention counts from the report.
EXCLUDED_SUBTREES = {"lineage", "token_counts"}
EXCLUDED_FIELDS = {
    # deterministic ids / enums we construct
    "principle_id", "prompt_type_id", "scenario_id", "section_id",
    "category_id", "subcategory_id", "call_id", "story_slot",
}


@dataclass
class Violation:
    field: str
    match: str
    context: str  # +-40 chars around the hit


def _scan_value(field: str, value: str) -> list[Violation]:
    violations = []
    for m in PERSONA_RE.finditer(value):
        lo, hi = max(0, m.start() - 40), min(len(value), m.end() + 40)
        violations.append(
            Violation(field=field, match=m.group(0), context=value[lo:hi])
        )
    return violations


def check_fields(record: dict, fields: list[str]) -> list[Violation]:
    violations = []
    for field in fields:
        value = record.get(field)
        if not value:
            continue
        violations.extend(_scan_value(field, value))
    return violations


def _trainable(corpus: str) -> list[str]:
    try:
        return TRAINABLE_FIELDS[corpus]
    except KeyError:
        raise ValueError(f"unknown corpus {corpus!r}") from None


def check_trainable(record: dict, corpus: str) -> list[Violation]:
    """The HARD persona gate: a hit in trainable text drops the record."""
    return check_fields(record, _trainable(corpus))


def _walk(node, path: str, violations: list[Violation]) -> None:
    if isinstance(node, str):
        violations.extend(_scan_value(path, node))
    elif isinstance(node, dict):
        for key, value in node.items():
            if key in EXCLUDED_SUBTREES or key in EXCLUDED_FIELDS:
                continue
            _walk(value, f"{path}.{key}" if path else str(key), violations)
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            _walk(value, f"{path}[{i}]", violations)
    # numerics / bools / None: nothing to scan


def check_metadata(record: dict, corpus: str) -> list[Violation]:
    """Mentions in generated NON-trainable strings (recursive walk, trainable
    fields excluded). Counted and sampled in the filter report — never a drop."""
    trainable = set(_trainable(corpus))
    violations: list[Violation] = []
    _walk(
        {k: v for k, v in record.items() if k not in trainable},
        "", violations,
    )
    return violations
