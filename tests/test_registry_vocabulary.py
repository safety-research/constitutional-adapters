"""The vocabulary rule: the banned tokens
{arm, leg, lane, cell, view, wave, grader} appear as whole words nowhere in schema.sql (names AND
comments), view names, the CLI help, docs/registry/*.md or the status template — except inside
`_legacy` columns, SQL keywords (CREATE VIEW / DROP VIEW), backticked existing identifiers in docs,
and the fenced legacy-mapping block of vocabulary.md; `rung` only inside `token_cap_rung_index`.
A known-bad fixture proves the check fires.
"""
from __future__ import annotations

import os
import re

import pytest

from published_repo import claude_md_text
from sisteer.registry import cli
from sisteer.registry import ownership as OW
from sisteer.registry import schema_def as S
from sisteer.registry import vocab as V

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "src", "sisteer", "registry")
DOCS = os.path.join(ROOT, "docs", "registry")

BANNED = ("arm", "leg", "lane", "cell", "view", "wave", "grader")
# `_` is a boundary too: an identifier SEGMENT (`lane_kind`, `cell_id`) is the legacy word used as a name
_WORD = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(BANNED) + r")(?![A-Za-z0-9])", re.IGNORECASE)
_RUNG = re.compile(r"(?<![A-Za-z0-9])rung(?![A-Za-z0-9])", re.IGNORECASE)
_LEGACY_IDENT = re.compile(r"\b\w*_legacy\b")
# the producers' own source spellings — the `run.lane_source_legacy` VALUES the multi-source declarations name — are data the
# rendered SQL must quote verbatim (`'llama8b_lane' AS secondary_source`, `r.lane_source_legacy IN ('r2_export', 'llama8b_lane')`):
# exempt by exact quoted value, the way `_legacy` identifiers are exempt by name; an unquoted use or any other spelling still fires.
# The same holds for the sources and the `run.pass_label` VALUES the member-pass table names (vocab.MEMBER_PASS_LABELS_BY_SOURCE,
# rendered into the selection SQL by schema_def.member_pass_sql: `r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN
# ('base_cap', 'ladder_rung')`) and the pooled pass labels the pooled-group CTE names (vocab.POOLED_PASS_LABELS, rendered by
# schema_def.POOLED_GROUP_CTE: `WHERE pass_label IN ('pooled_t0_t1', 'pooled_final_rung')`) -- producer spellings quoted as data,
# never a word of ours
# -- and the source spellings / GLOB patterns the source-local-anchor branch quotes (vocab.SOURCE_LOCAL_ANCHOR_SOURCES, rendered by
# schema_def.source_local_anchor_sql, migration 0027: `r.lane_source_legacy IN ('strength_lane', ...)`, `GLOB 'strength_lane_*'`)
_DATA_SPELLINGS = ({s for d in OW.declarations() for s in (d.owner_source, d.secondary_source)}
                   | set(V.MEMBER_PASS_LABELS_BY_SOURCE) | {l for ls in V.MEMBER_PASS_LABELS_BY_SOURCE.values() for l in ls}
                   | set(V.POOLED_PASS_LABELS) | set(V.SOURCE_LOCAL_ANCHOR_SOURCES))
_SOURCE_SPELLINGS = re.compile("|".join(sorted((re.escape(f"'{s}'") for s in _DATA_SPELLINGS), key=len, reverse=True)))
_BACKTICK = re.compile(r"`[^`\n]*`")
_SQL_VIEW = re.compile(r"\b(CREATE|DROP)\s+VIEW\b", re.IGNORECASE)
_VIEW_SUFFIX = re.compile(r"\bthe view\b|\bthis view\b|\bview names\b|\bviews\b|\bflag view\b|a view\b|\bview (?:or|columns|exposes|itself|SQL|column)\b",
                          re.IGNORECASE)


def scrub(text: str, allow_prose_view: bool = False) -> str:
    """Remove the exempt spans; what remains is checked for banned whole words."""
    text = _SOURCE_SPELLINGS.sub(" ", text)
    text = _LEGACY_IDENT.sub(" ", text)
    text = _BACKTICK.sub(" ", text)
    text = _SQL_VIEW.sub(" ", text)
    text = re.sub(r"<!-- legacy-mapping -->.*?<!-- /legacy-mapping -->", " ", text, flags=re.DOTALL)
    if allow_prose_view:
        # "view" in the SQL sense (a database view) is the spec's own noun for v_* objects
        text = _VIEW_SUFFIX.sub(" ", text)
        text = re.sub(r"\bv_[a-z_]+\b", " ", text)
        text = re.sub(r"\bSQL views?\b|\bdatabase views?\b|\brun-level views?\b|\b(the|a|every|each|per|one|no|any) views?\b",
                      " ", text, flags=re.IGNORECASE)
    return text


def hits(text: str, allow_prose_view: bool = False):
    s = scrub(text, allow_prose_view)
    out = [(m.group(1), s[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")) for m in _WORD.finditer(s)]
    for m in _RUNG.finditer(s):
        ctx = s[max(0, m.start() - 20): m.end() + 20]
        if "token_cap_rung_index" not in ctx:
            out.append(("rung", ctx.replace("\n", " ")))
    return out


def test_known_bad_fixture_fires():
    assert hits("CREATE TABLE x (arm TEXT, lane_kind TEXT);")
    assert hits("the grader wrote a row")
    assert hits("rung 3 of the ladder")
    assert not hits("token_cap_rung_index INTEGER")
    assert not hits("view_legacy TEXT  -- why-null: only backfilled rows")
    assert not hits("CREATE VIEW v_run_wide AS SELECT 1")
    assert not hits("read `lane_watermarks.json` and `CellStore`")
    # a declared source's spelling is data, exempt as an exact quoted value; unquoted, or any other spelling, it fires
    assert not hits("AND r.lane_source_legacy IN ('r2_export', 'llama8b_lane')\n    AND r.transfer_kind = 'base_to_instruct'")
    assert not hits("'r2_export' AS owner_source, 'llama8b_lane' AS secondary_source,")
    assert hits("llama8b_lane AS secondary_source")
    assert hits("r.lane_source_legacy = 'r2_export' AND o.paper_label = 'lane_x'")
    assert hits("'synthetic_lane' AS secondary_source")
    # the member-pass table's sources and labels likewise: exempt as exact quoted values (the rendered selection SQL), firing unquoted
    assert not hits("WHEN (r.lane_source_legacy = 'sampled_lane' AND r.pass_label IN ('base_cap', 'ladder_rung')) THEN 'member_pass'")
    assert hits("WHEN r.lane_source_legacy = sampled_lane") and hits("pass_label IN (ladder_rung)")
    assert hits("'other_lane' AND r.pass_label IN ('base_cap')") and hits("r.pass_label = 'some_rung'")
    # the pooled pass labels the pooled-group CTE quotes are data too; the same label unquoted, or another spelling, fires
    assert not hits("SELECT DISTINCT lane_source_legacy, model_id, object_id, eval_id FROM run WHERE pass_label IN ('pooled_t0_t1', 'pooled_final_rung')")
    assert hits("WHERE pass_label IN (pooled_final_rung)") and hits("WHERE pass_label IN ('pooled_last_rung')")
    # the source-local-anchor branch's quoted sources and patterns are data (migration 0027); unquoted, or another spelling, they fire
    assert not hits("WHEN (r.object_id = 'none' AND (r.lane_source_legacy IN ('alphasteer_rebake', 'strength_lane') OR r.lane_source_legacy GLOB 'strength_lane_*')")
    assert hits("r.lane_source_legacy IN (strength_lane)") and hits("r.lane_source_legacy GLOB 'other_lane_*'")


def test_schema_sql_has_no_banned_tokens():
    with open(os.path.join(PKG, "schema.sql"), "r", encoding="utf-8") as fh:
        text = fh.read()
    h = hits(text)
    assert not h, h[:10]


def test_view_and_table_and_column_names_clean():
    names = list(S.VIEW_NAMES) + list(S.ALL_TABLES) + [c.name for t in S.TABLES for c in t.columns]
    bad = [n for n in names if hits(n)]
    assert not bad, bad


def test_cli_help_clean():
    parser = cli.build_parser()
    texts = [parser.format_help()]
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        for name, sub in action.choices.items():
            texts.append(sub.format_help())
    h = hits("\n".join(texts), allow_prose_view=True)
    assert not h, h[:10]


def test_status_template_clean():
    with open(os.path.join(PKG, "templates", "status.html.j2"), "r", encoding="utf-8") as fh:
        h = hits(fh.read(), allow_prose_view=True)
    assert not h, h[:10]


@pytest.mark.parametrize("doc", ["schema.md", "vocabulary.md", "records.md", "README.md"])
def test_docs_clean(doc):
    with open(os.path.join(DOCS, doc), "r", encoding="utf-8") as fh:
        text = fh.read()
    h = hits(text, allow_prose_view=True)
    assert not h, h[:10]


def test_legacy_mapping_block_present_and_fenced():
    with open(os.path.join(DOCS, "vocabulary.md"), "r", encoding="utf-8") as fh:
        text = fh.read()
    assert "<!-- legacy-mapping -->" in text and "<!-- /legacy-mapping -->" in text
    block = text.split("<!-- legacy-mapping -->")[1].split("<!-- /legacy-mapping -->")[0]
    for term in ("arm", "cell", "lane", "view", "leg", "wave", "grader"):
        assert re.search(r"\|\s*" + term + r"\b", block), term


def test_python_identifiers_in_registry_package_avoid_banned_words():
    """Identifiers (def / class / assignment targets) in the package use the registry nouns."""
    ident = re.compile(r"^\s*(?:def|class)\s+(\w+)|^\s*(\w+)\s*(?::[^=]+)?=", re.MULTILINE)
    # SQL views (the spec's own noun for the v_* objects) — the only sense of the word the package uses:
    # the rendered view registry in schema_def (names, SQL, declared references, base-table closure)
    exempt = {"view_names", "VIEW_NAMES", "VIEW_SQL", "VIEW_REFERENCES", "VIEW_TABLES"}
    # the whole package, subpackages included (adapters/, migrations/): a flat listdir silently skipped them
    files = []
    for dirpath, dirnames, filenames in os.walk(PKG):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        files += [os.path.relpath(os.path.join(dirpath, fn), PKG) for fn in sorted(filenames) if fn.endswith(".py")]
    assert any(os.sep in rel for rel in files), files  # the walk reaches the subpackages
    bad = []
    for rel in files:
        with open(os.path.join(PKG, rel), "r", encoding="utf-8") as fh:
            src = fh.read()
        for m in ident.finditer(src):
            name = m.group(1) or m.group(2)
            if name and name not in exempt and hits(name):
                bad.append((rel, name))
    assert not bad, bad


# --------------------------------------------------------------------------------------------
# DECODE_SEED_RULES name the closure a seed is derived from (2026-09-17)
# --------------------------------------------------------------------------------------------


def test_decode_seed_rules_name_the_seed_closure():
    """The seed-replicate battery derives every decode seed from the literal training seed and a closure the label must
    NAME: (behaviour, scenario_id, turn) for Petri's target turns, (family, item, sample) for the misalignment runner's
    responses, (goal, turn) for the multi-turn target. The strictness rule: a label naming the closure beats `per_item`
    plus a free-text detail — so the three are controlled values, each with evidence naming the battery, rendered into
    the glossary and the DDL."""
    from sisteer.registry import vocab as V
    ev = V.EVIDENCE["DECODE_SEED_RULES"]
    for label in ("per_scenario_turn", "per_response", "per_goal_turn"):
        assert label in V.DECODE_SEED_RULES, label
        assert "seed-replicate" in ev[label], label
    assert "scenario_id" in ev["per_scenario_turn"] and "turn" in ev["per_scenario_turn"] and "seed_base" in ev["per_scenario_turn"]
    assert "sample" in ev["per_response"] and "item" in ev["per_response"] and "seed_base" in ev["per_response"]
    assert "goal" in ev["per_goal_turn"] and "turn" in ev["per_goal_turn"] and "seed_base" in ev["per_goal_turn"]
    # the strictness rule is stated on the generic label itself
    assert "free text" in ev["per_item"] and "closure" in ev["per_item"]
    with open(os.path.join(DOCS, "vocabulary.md"), "r", encoding="utf-8") as fh:
        md = fh.read()
    with open(os.path.join(PKG, "schema.sql"), "r", encoding="utf-8") as fh:
        ddl = fh.read()
    for label in ("per_scenario_turn", "per_response", "per_goal_turn"):
        assert f"| `{label}` |" in md, label
        assert f"'{label}'" in ddl, label
    # the glossary lines keep to the registry's nouns (the banned-token rule above applies to generated prose too)
    assert not hits("\n".join(ev.values()), allow_prose_view=True)


CLAUDE_SECTION = "## Results registry: the source of truth for measurements"


def claude_registry_section() -> str:
    """The registry section of the repo's CLAUDE.md, heading to the next `## ` heading."""
    text = claude_md_text()
    start = text.index(CLAUDE_SECTION)
    m = re.search(r"^## ", text[start + len(CLAUDE_SECTION):], flags=re.MULTILINE)
    return text[start: start + len(CLAUDE_SECTION) + (m.start() if m else len(text))]


def test_claude_md_registry_section_clean():
    """The section that TELLS sessions the vocabulary rule obeys it: the legacy nouns appear there only backticked
    (as the words being banned), exactly as the docs spell them."""
    section = claude_registry_section()
    assert "`relu_steer`" in section and "`psr`" in section
    h = hits(section, allow_prose_view=True)
    assert not h, h[:10]
