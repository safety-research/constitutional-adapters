"""Generated documentation is current: docs/registry/{schema,vocabulary,records}.md equal their render;
schema.md lists every table, column and view and every nullable column's reason (DDL ↔ doc) without the
definitions' decision citations; vocabulary.md names every controlled value with evidence and carries the
legacy-mapping table; records.md names every record type with its required fields; no page carries a decision
citation, a bare date or an account of the review pass that produced a rule.
"""
from __future__ import annotations

import os
import re

import pytest

from published_repo import claude_md_text
from sisteer.registry import docsgen
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import vocab as V

DOCS = docsgen.DOCS_DIR


@pytest.mark.parametrize("name", ["schema", "vocabulary", "records"])
def test_generated_doc_is_current(name):
    assert docsgen.doc_is_current(name), f"docs/registry/{name}.md is stale: run `registry docs {name}`"


def _table_section(md: str, table: str) -> str:
    start = md.index(f"### `{table}` (")
    end = md.find("\n### `", start + 1)
    return md[start: end if end > 0 else len(md)]


def test_schema_md_lists_every_table_column_and_nullable_reason():
    md = open(os.path.join(DOCS, "schema.md"), encoding="utf-8").read()
    for t in S.TABLES:
        section = _table_section(md, t.name)
        for c in t.columns:
            null = "NOT NULL" if c.nn else "nullable"
            assert f"| `{c.name}` | {c.type} | {null}" in section, (t.name, c.name)
            if not c.nn:
                assert f"| {docsgen.history_free(c.why_null)} |" in section, (t.name, c.name)
    for v in S.VIEW_NAMES:
        assert f"| `{v}` |" in md, v


def test_schema_md_carries_no_decision_citation():
    """The definitions cite the decisions they implement by id and date; the rendered page states the rule alone."""
    assert docsgen.history_free("only epoch-checkpoint children name the fit they belong to (D696)") == \
        "only epoch-checkpoint children name the fit they belong to"
    assert docsgen.history_free("retracted this row (D628; migration 0020) — stamp (D46-3 / D50-2); kept (NULL reads as current; D757)") == \
        "retracted this row (migration 0020) — stamp; kept (NULL reads as current)"
    assert docsgen.history_free("two columns on item_result — D20) and a date (2026-10-04) here") == \
        "two columns on item_result) and a date here"
    md = open(os.path.join(DOCS, "schema.md"), encoding="utf-8").read()
    for doc in ("schema.md", "vocabulary.md", "records.md"):
        text = open(os.path.join(DOCS, doc), encoding="utf-8").read()
        assert not re.search(r"\bD\d{1,4}(?:-\d+)?\b", text), doc
        assert not re.search(r"\b20\d\d-\d\d-\d\d\b", text), doc
    assert "```sql" not in md       # the SQL of views, triggers and table constraints is schema.sql's


# An account of the review pass, the landing or the earlier behaviour a rule replaced; the pages state the rule alone.
_REVIEW_HISTORY = re.compile(
    r"\b(?:review round|closing pass|review pass|follow-up pass|the review of|first rendering|refold-compare|"
    r"whole-log audit|at the rule's landing|before this (?:index|rule|column|table) existed)\b", re.IGNORECASE)


@pytest.mark.parametrize("name", ["schema", "vocabulary", "records"])
def test_rendered_page_carries_no_review_history(name):
    """Every rendered page documents behaviour: no decision id, no bare date (records.md's sample record carries a
    `written_utc` timestamp, which is not one) and no narrative of the review pass or landing that produced a rule."""
    text = open(os.path.join(DOCS, f"{name}.md"), encoding="utf-8").read()
    assert not re.search(r"\bD\d{1,4}(?:-\d+)?\b", text), name
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", text), name
    hit = _REVIEW_HISTORY.search(text)
    assert hit is None, (name, text[max(0, hit.start() - 80): hit.end() + 80])


def test_vocabulary_md_names_every_controlled_value():
    md = open(os.path.join(DOCS, "vocabulary.md"), encoding="utf-8").read()
    for name, values in V.VOCABULARIES.items():
        if name in ("BOOLEAN", "RECORD_TYPES", "LEDGERED_RECORD_TYPES", "ITEM_SCALE_TABLES", "RUN_SCALE_TABLES"):
            continue
        assert f"## `{name}`" in md, name
        for v in values:
            assert f"| `{v}` |" in md, (name, v)
    for kind in V.FIT_TARGET_RULES:
        assert f"| `{kind}` |" in md
    assert "<!-- legacy-mapping -->" in md


def test_records_md_names_every_record_type_and_required_fields():
    md = open(os.path.join(DOCS, "records.md"), encoding="utf-8").read()
    for rt in R.RECORD_TYPES:
        assert f"| `{rt}` |" in md, rt
        for f in R.REQUIRED_FIELDS[rt]:
            assert f"`{f}`" in md, (rt, f)
    for f in R.RUN_PATCHABLE_FIELDS + R.RUN_VERIFIED_FIELDS:
        assert f"`{f}`" in md


def test_docs_schema_does_not_rewrite_the_frozen_initial_migration(tmp_path):
    """The in-place edits of 0001_initial.sql came from `registry docs schema` regenerating it with schema.sql; the
    migration files are frozen history now (test_registry_schema.py::test_landed_migrations_are_frozen) and the
    docs generator writes schema.sql only. The migrations README says how a change lands."""
    assert not hasattr(docsgen, "write_initial_migration")
    pkg = os.path.dirname(os.path.abspath(docsgen.__file__))
    with open(os.path.join(pkg, "migrations", "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    for phrase in ("is frozen", "registry migrate --plan --sql", "foreign_keys=OFF", "legacy_alter_table=ON", "foreign_key_check", "0002"):
        assert phrase in readme, phrase


REPO = os.path.dirname(os.path.dirname(os.path.abspath(DOCS)))


def test_readme_indexes_the_generated_docs_and_the_entry_points():
    """docs/registry/README.md names the three generated pages and the commands a reader of a live registry
    reaches for first."""
    md = open(os.path.join(DOCS, "README.md"), encoding="utf-8").read()
    for doc in ("schema.md", "records.md", "vocabulary.md"):
        assert f"`{doc}`" in md, doc
    assert "operations.md" not in md
    for entry in ("registry status", "registry rebuild --verify", "registry ingest", "registry mirrors", "registry snapshot",
                  "registry pending-judgment", "--worklist registry", "registry adapters run", "registry roster load",
                  "registry ruling add", "check-paper-parity", "registry decode-policy export", "registry migrate --plan",
                  "scripts/registry/landing/window.sh"):
        assert entry in md, entry


def test_claude_md_names_the_registry_as_the_source_of_truth():
    """The repo CLAUDE.md carries the registry section (spec scope item: the cleanup pass that names the database as the
    single source of truth so every session knows) and states the load-bearing facts by their exact spellings."""
    md = claude_md_text()
    assert "## Results registry: the source of truth for measurements" in md
    for fact in ("~/registry/registry.sqlite", "~/registry/records",
                 "docs/registry/README.md", "SISTEER_REGISTRY_DISABLED=1", "`CellStore` records by itself",
                 "--worklist registry", "v_pending_judgment", "v_paper_long", "v_paper_current", "v_source_status",
                 "registry export r2-compat", "tests/test_registry_source_coverage.py", "configs/registry/rulings_r2_v1.jsonl",
                 "`relu_steer`, never `psr`", "configs/registry/decode_policy.json", "NoPolicy", "registry rebuild --verify",
                 "registry-push"):
        assert fact in md, fact
    for doc in ("schema.md", "records.md", "vocabulary.md"):
        assert f"`{doc}`" in md, doc
    assert "docs/registry/operations.md" not in md and "`operations.md` (the runbook)" not in md   # the runbook left the repository


def test_readme_and_migrations_readme_name_the_audit_script_and_the_maintenance_commands():
    """The README's entry points name the whole-log audit script and the maintenance commands; the migrations README names the
    script and the window guards a schema change lands under."""
    readme = open(os.path.join(DOCS, "README.md"), encoding="utf-8").read()
    for entry in ("scripts/registry/audit_new_validator_rules.py", "registry judgment show", "registry cap-check",
                  "registry reconcile run-patch-extra --dry-run"):
        assert entry in readme, entry
    pkg = os.path.dirname(os.path.abspath(docsgen.__file__))
    with open(os.path.join(pkg, "migrations", "README.md"), encoding="utf-8") as fh:
        mig = fh.read()
    for phrase in ("scripts/registry/audit_new_validator_rules.py", "renumbers before its gate", "read before `registry migrate`",
                   "refuses to apply a plan that rebuilds a table", "WINDOW_EXPECT_MIGRATION", "bound to the displayed plan by its token"):
        assert phrase in mig, phrase
    assert "operations.md" not in mig
