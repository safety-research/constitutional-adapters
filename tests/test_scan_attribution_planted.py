"""The attribution tier of the publication scan, planted in synthetic trees.

The leaf scope reads every free-text leaf of the paper frame (Parquet and the split SQLite alike), the registry
transcripts and the dimension seeds, whether or not a field policy declares the leaf: an attribution planted in a new frame
`extra_json` key, in a transcript's `target_json` prose key, in a seed's `notes` and in a `run_state_event.note` fails the
scan, and so does a decision id in a leaf a reader reads, until an exemption names that exact address and value. A mapping
key is a leaf too, addressed as `<object address>{<key>}`.

The comment scope reads comments and docstrings only: a decision id in a Python comment, a SQL comment, a shell or YAML
comment, a pip requirements comment, a docs page, a notebook's markdown or an exemption's reason in the scan's own allowlist
fails, and the same id in a string literal passes.
A numbered decision written out as the word and its number, rather than the short id, fails in a seed value and in
comments; a file name, a dotted key or plain prose that only holds the word does not. The chat role ("the user
turn") and a lint directive's code list are not attributions. Four rows read the comment scope only: a section of an
unpublished plan or spec cited by number, a workstream id, a numbered spec criterion, and a spec named by its slug; an
identifier, a file name, a string literal, a data leaf or plain prose holding the same words passes. Every row of the
comment scope, these four included, also fails when a line break splits its match inside one comment passage (comment
lines of one kind, one docstring, one block comment, a docs paragraph, a notebook cell), and never across a line of code,
code beside the next comment, a docstring's quotes or a blank comment line; such a hit is reported once, at its first
line, with the lines it spans as its unit. The rows are the real token table's attribution rows: they are shapes, not
words, so a test may use them as they are."""
from __future__ import annotations

import gzip
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import SCAN, TOKEN_TABLE  # noqa: E402

SNAP = "paper/snapshot/snapshots/000000000001-20260101T000000-00000000"
ATTRIBUTION = "per the user's ruling, 2026-09-24"      # an invented attribution in the shape the tier looks for
NOTE = "approved (the user, 2026-09-30)"
DECISION_ID = "R2-D253-cap"                             # an identifier that carries a decision id, as some ruling ids do


def _repo(root: Path, files: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        (p.write_bytes if isinstance(data, bytes) else p.write_text)(data)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def _scan(root: Path, tmp_path: Path, **kw):
    sc, _ = SCAN.scan(root, table=TOKEN_TABLE, unpublished=("CLAUDE.md",), tmpdir=str(tmp_path), **kw)
    assert not sc.report.errors, sc.report.errors
    return sc.report.hits


def _frame(root: Path, tmp_path: Path) -> None:
    """A two-format frame: three tables as Parquet and as a split-gzip SQLite with the same rows."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    tables = {
        "run": {"run_id": ["r1", "r2"], "substrate": ["cluster_a", "cluster_a"],
                "extra_json": [json.dumps({"kept_note": ATTRIBUTION, "paper_arm": "a1"}), json.dumps({"paper_arm": "a2"})]},
        "run_state_event": {"event_id": ["e1"], "note": [NOTE]},
        "ruling": {"ruling_id": [DECISION_ID], "ruled_by": ["the authors"]},
    }
    snap = root / SNAP
    snap.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "frame.sqlite"
    con = sqlite3.connect(db)
    for name, cols in tables.items():
        pq.write_table(pa.table(cols), snap / f"{name}.parquet")
        con.execute(f'CREATE TABLE "{name}" ({", ".join(f"{c} TEXT" for c in cols)})')
        con.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" for _ in cols)})', list(zip(*cols.values())))
    con.commit()
    con.close()
    gz = gzip.compress(db.read_bytes())
    third = len(gz) // 3 + 1
    for k in range(3):
        (snap / f"paper_frame.sqlite.gz.{k:02d}").write_bytes(gz[k * third:(k + 1) * third])


@pytest.fixture
def planted(tmp_path):
    root = _repo(tmp_path / "r", {
        "configs/registry/rulings_r2_v1.jsonl": json.dumps({"ruling_id": "x1", "target_json": {"rule": "kept as the user decided"},
                                                            "reason": "a plain methodological reason"}) + "\n",
        "configs/registry/models.yaml": "models:\n  - id: m1\n    notes: kept by the user's call\n  - id: m2\n    notes: plain\n",
        "configs/registry/judges.yaml": "judges:\n  - id: j1\n    ruled_by: the user\n",
    })
    _frame(root, tmp_path)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    return root


def _leaf(hits) -> set:
    return {(h.path, h.where) for h in hits if h.scope == "leaf"}


def test_attribution_outside_any_declared_field_fails_the_leaf_scope_in_every_scope_of_the_data(planted, tmp_path):
    got = _leaf(_scan(planted, tmp_path))
    assert (SNAP, "run.extra_json.kept_note") in got                       # a new frame extra_json key
    assert (SNAP, "run_state_event.note") in got                           # a run_state_event note
    assert ("configs/registry/rulings_r2_v1.jsonl", "L1:target_json.rule") in got   # a transcript's target_json prose key
    assert ("configs/registry/models.yaml", "models[].notes") in got       # a seed's notes
    assert ("configs/registry/judges.yaml", "judges[].ruled_by") in got    # a bare attribution as a whole value
    assert (SNAP, "ruling.ruled_by") not in got and (SNAP, "run.extra_json.paper_arm") not in got


def test_the_parquet_set_and_the_split_sqlite_report_one_address(planted, tmp_path):
    hits = [h for h in _scan(planted, tmp_path) if h.scope == "leaf" and h.path == SNAP]
    by_origin = {}
    for h in hits:
        by_origin.setdefault(h.origin.rsplit(".", 1)[-1], set()).add((h.where, h.unit_hash))
    assert set(by_origin) == {"parquet", "sqlite"} and by_origin["parquet"] == by_origin["sqlite"], by_origin


def test_a_decision_id_in_a_read_leaf_fails_until_an_exemption_names_its_address_and_value(planted, tmp_path):
    hits = _scan(planted, tmp_path)
    read = [h for h in hits if h.scope == "leaf" and h.where == "ruling.ruling_id"]
    assert read and {h.unit for h in read} == {DECISION_ID}
    exempt = SCAN.AllowRow(f"{SNAP}#ruling.ruling_id", "attribution", "value", SCAN.sha256_text(DECISION_ID), "an identifier", 1)
    left = SCAN.apply_allowlist(hits, [exempt])
    assert not [h for h in left if h.where == "ruling.ruling_id"] and exempt.used
    other = SCAN.AllowRow(f"{SNAP}#ruling.ruling_id", "attribution", "value", SCAN.sha256_text("another value"), "x", 2)
    assert [h for h in SCAN.apply_allowlist(hits, [other]) if h.where == "ruling.ruling_id"], "an exemption holds one value"


KEYED_FILES = {
    "configs/registry/instruments.yaml": "instruments:\n  - id: i1\n    D12: kept apart\n    notes: plain\n",
    "configs/registry/rulings_r2_v1.jsonl": json.dumps({"header": {"D29": ["x1"], "added_note": "plain"}, "ruling_id": "x1"}) + "\n",
    "data/keys.json": json.dumps({"D12": 1}),                                                # outside the leaf scope
}


def test_a_mapping_key_is_read_as_a_leaf_at_its_object_address(tmp_path):
    root = _repo(tmp_path / "r", KEYED_FILES)
    hits = _scan(root, tmp_path)
    leaf = [h for h in hits if h.scope == "leaf"]
    assert sorted((h.path, h.where, h.unit) for h in leaf) == [
        ("configs/registry/instruments.yaml", "instruments[]{D12}", "D12"),
        ("configs/registry/rulings_r2_v1.jsonl", "L1:header{D29}", "D29")], leaf
    cell = '{"carried_from": {"D30": "0a1b2c3", "W": "0a1b2c3d"}, "paper_arm": "a1"}'      # a frame JSON cell
    assert ("run.extra_json.carried_from{D30}", "D30") in list(SCAN._value_leaves("run.extra_json", cell))
    assert ("run.extra_json{carried_from}", "carried_from") in list(SCAN._value_leaves("run.extra_json", cell))
    exempt = SCAN.AllowRow("configs/registry/instruments.yaml#instruments[]{D12}", "attribution", "value",
                           SCAN.sha256_text("D12"), "a key", 1)
    assert [(h.path, h.where) for h in SCAN.apply_allowlist(leaf, [exempt])] == [
        ("configs/registry/rulings_r2_v1.jsonl", "L1:header{D29}")]


def test_an_exemption_names_a_leaf_inside_nested_lists_by_its_literal_address(tmp_path):
    note = "kept by the user's call"
    root = _repo(tmp_path / "r", {"configs/registry/roots.yaml": f"roots:\n  - evidence:\n      - note: {note}\n"})
    (h,) = [h for h in _scan(root, tmp_path) if h.scope == "leaf"]
    assert h.where == "roots[].evidence[].note"
    exempt = SCAN.AllowRow("configs/registry/roots.yaml#roots[].evidence[].note", "attribution", "value",
                           SCAN.sha256_text(note), "a nested note", 1)
    assert SCAN.apply_allowlist([h], [exempt]) == [] and exempt.used == 1
    wild = SCAN.AllowRow("configs/registry/*.yaml#roots[].evidence[].note", "attribution", "value", SCAN.sha256_text(note), "x", 2)
    assert SCAN.apply_allowlist([h], [wild]) == []


COMMENT_FILES = {
    "src/m.py": ('"""Module doc: kept by the user\'s ruling."""\n'                       # L1 a module docstring
                 "X = 1  # kept as decided in D1234\n"                                    # L2 a comment
                 'Y = "D1234"\n'                                                          # L3 a string literal
                 '"a bare string statement, not a docstring: the user\'s ruling"\n'      # L4 not a docstring
                 "def f():\n"
                 '    """the user\'s ruling holds here"""\n'                              # L6 a function docstring
                 "    return 1  # the user turn is appended last\n"                       # L7 the chat role
                 "Z = 2  # noqa: D401\n"                                                  # L8 a directive's code
                 "W = 3  # noqa: D401 (D23 says so)\n"                                    # L9 prose beside it
                 "# fixed after the S4 gate\n"                                            # L10 a gate name
                 "# W27: retried\n"),                                                     # L11 a lane tag
    "src/schema.sql": "-- D1234 keeps the index\nSELECT 'D1234';\n",
    "scripts/run.sh": '#!/bin/bash\n# D1234 note\necho "D1234 # not a comment"\n',
    "configs/c.yaml": 'key: value  # per D1234\nother: "D1234"\n',
    "docs/notes.md": "Plain line.\nSee D1234.\n",
    "notes/other.md": "See D1234.\n",
    "configs/evidence/copy.py.txt": '"""A copy of Python source kept as text: kept by the user\'s rule."""\nX = "D1234"\n',
    "nb/n.ipynb": json.dumps({"metadata": {}, "cells": [
        {"cell_type": "markdown", "source": ["Decided in D1234"]},
        {"cell_type": "code", "source": ['x = "D1234"  # chosen by the user, 2026-09-01'],
         "outputs": [{"output_type": "stream", "text": ["result D1234\n"]}]}]}),
}


def test_the_comment_scope_reads_comments_and_docstrings_and_never_a_string_literal(tmp_path):
    root = _repo(tmp_path / "r", COMMENT_FILES)
    got = sorted((h.path, h.where) for h in _scan(root, tmp_path) if h.scope == "comment")
    want = sorted([("src/m.py", "L1"), ("src/m.py", "L2"), ("src/m.py", "L6"), ("src/m.py", "L9"), ("src/m.py", "L10"),
                   ("src/m.py", "L11"), ("src/schema.sql", "L1"), ("scripts/run.sh", "L2"), ("configs/c.yaml", "L1"),
                   ("docs/notes.md", "L2"), ("configs/evidence/copy.py.txt", "L1"), ("nb/n.ipynb", "cell0:L1"), ("nb/n.ipynb", "cell1:L1"),
                   ("nb/n.ipynb", "cell1:out0:L1")])
    assert sorted(set(got)) == want, sorted(set(got))


REQUIREMENTS_FILES = {
    "configs/serve-requirements.txt": (
        "# the serving set, pinned after D20\n"                                              # L1 a comment line
        "vllm==0.25.1  # re-pinned by D13\n"                                                 # L2 a comment after a requirement
        "pkg @ https://example.invalid/D1234/pkg-1.0-py3-none-any.whl#sha256=D1234\n"        # L3 a URL fragment, not a comment
        "torch==2.11.0 \\\n"                                                                 # L4 a requirement continued
        "    --hash=sha256:0a1b  # D5 on the continuation\n"                                 # L5 a comment on its next line
        "    # via D6\n"                                                                     # L6 an indented comment
        "extra==1  # D7, continued \\\n"                                                     # L7 a comment a backslash continues
        "joined==2 D8\n"                                                                     # L8 pip reads it as that comment
        "# D9 a comment line never continues \\\n"                                           # L9 a comment line
        "after==3 D10\n"),                                                                   # L10 a requirement again
    "constraints.txt": "numpy<3  # capped by D11\n",
    "tools/client-requirements.in": "# D12 top-level pins\nrich==13.0  # D13\n",
    "data/words.txt": "# D1234 is an entry of a word list\nD20\n",                         # plain text: no comment syntax
}


def test_a_pip_requirements_file_is_read_as_pip_reads_its_comments(tmp_path):
    root = _repo(tmp_path / "r", REQUIREMENTS_FILES)
    hits = [h for h in _scan(root, tmp_path) if h.scope == "comment"]
    got = sorted({(h.path, h.where) for h in hits})
    want = sorted([("configs/serve-requirements.txt", f"L{n}") for n in (1, 2, 5, 6, 7, 8, 9)]
                  + [("constraints.txt", "L1"), ("tools/client-requirements.in", "L1"), ("tools/client-requirements.in", "L2")])
    assert got == want, got
    (h,) = [h for h in hits if h.path == "constraints.txt"]
    assert h.unit == "numpy<3  # capped by D11" and h.unit[h.start:h.end] == "D11"


ALLOWLIST_ROWS = (
    "# path glob\ttier\tmatch\treason\n"                                                   # L1 a clean header comment
    "# rows the review under D12 kept\n"                                                   # L2 a comment line
    f"data/a.json\tprocess\tline:{'0' * 64}\ta recorded path the loader parses\n"           # L3 a clean reason
    f"data/b.json\tprocess\tline:{'1' * 64}\tkept as D1234 decided\n"                      # L4 attribution in a reason
    f"data/D1234/*.json\tprocess\tline:{'2' * 64}\ta recorded directory name\n")           # L5 an id in the glob only


def test_an_exemption_reason_is_read_as_a_comment(tmp_path):
    root = _repo(tmp_path / "r", {SCAN.ALLOWLIST: ALLOWLIST_ROWS, "data/notes.tsv": "a\tb\tc\tkept as D1234 decided\n"})
    hits = [h for h in _scan(root, tmp_path) if h.scope == "comment"]
    assert sorted({(h.path, h.where) for h in hits}) == [(SCAN.ALLOWLIST, "L2"), (SCAN.ALLOWLIST, "L4")], hits
    (h,) = [h for h in hits if h.where == "L4"]
    assert h.unit.endswith("\tkept as D1234 decided") and h.unit[h.start:h.end] == "D1234"


SPELLED_DECISION_FILES = {
    "configs/registry/instruments.yaml": ('instruments:\n  - id: i1\n    notes: "Decision 25: item rows store every draw"\n'
                                          '  - id: i2\n    notes: "the decode decision for this row"\n'),
    "configs/registry/models.yaml": "# recorded as `unverified` (spec fixed decision 19: admitted only with a reason)\nmodels:\n  - id: m1\n",
    "src/m.py": ("X = 1  # the defaults (spec decisions 11 / 20)\n"                         # L1 plural, two numbers
                 "# the benign half (see the spec's fixed decisions\n"                      # L2 its number on the next line
                 "# and decision #7 says so\n"                                              # L3 a `#` before the number
                 'Y = "fixed decision 11"\n'                                                # L4 a string literal
                 "# writes decision_16384.json, then a.pasttense.decisions.8192\n"          # L5 a file name, a dotted key
                 "# a fixed decision boundary, and the decision to run one more pass\n"),   # L6 prose
}


def test_a_spelled_out_numbered_decision_fails_in_a_seed_value_and_in_comments(tmp_path):
    root = _repo(tmp_path / "r", SPELLED_DECISION_FILES)
    hits = _scan(root, tmp_path)
    leaf = [h for h in hits if h.scope == "leaf"]
    assert {(h.path, h.where, h.unit) for h in leaf} == {
        ("configs/registry/instruments.yaml", "instruments[].notes", "Decision 25: item rows store every draw")}, leaf
    got = sorted({(h.path, h.where) for h in hits if h.scope == "comment"})
    assert got == [("configs/registry/models.yaml", "L1"), ("src/m.py", "L1"), ("src/m.py", "L2"), ("src/m.py", "L3")], got
    spans = {h.unit[h.start:h.end] for h in hits if h.path == "configs/registry/models.yaml"}
    assert "decision 19" in spans, spans


def test_a_comment_hit_reports_the_whole_line_and_the_span_of_the_id(tmp_path):
    root = _repo(tmp_path / "r", {"src/m.py": "X = 1  # kept as decided in D1234\n"})
    (h,) = [h for h in _scan(root, tmp_path) if h.scope == "comment"]
    assert h.unit == "X = 1  # kept as decided in D1234" and h.unit[h.start:h.end] == "D1234"
    assert h.unit_hash == SCAN.sha256_text(h.unit)


def test_the_attribution_tier_never_reaches_the_tree_scope(tmp_path):
    root = _repo(tmp_path / "r", {"src/m.py": "MSG = \"kept per the user's ruling (D9)\"\n", "data/x.txt": "D1234 the user's ruling\n"})
    assert _scan(root, tmp_path) == []


def _comment_hits(root: Path, tmp_path: Path) -> list:
    return sorted({(h.path, h.where) for h in _scan(root, tmp_path) if h.scope == "comment"})


PLAN_SECTION_FILES = {
    "src/m.py": ('"""The conversion stage (plan section 4 step 2) strips the table."""\n'      # L1 a docstring
                 "X = 1  # the drain comes first (plan §1.5)\n"                               # L2 a trailing comment
                 "# the plan's §3 order, and Plan §2.1-4 for the cells\n"                      # L3 possessive, capital
                 "# the restore (spec §4.1) and the spec section B rows\n"                     # L4 a spec section
                 'Y = "plan §3"\n'                                                            # L5 a string literal
                 "# the migration plan prints a section per table\n"                          # L6 a plan the code prints
                 "# the --plan output section, and a plan section heading\n"                  # L7 a flag, a heading
                 "# see DESIGN.md §3 for the bound\n"),                                       # L8 a shipped document
    "configs/c.yaml": "enable: false  # defaults pinned OFF (plan §6)\nk: 1\n",               # a YAML comment
    "scripts/run.sh": "#!/bin/bash\n# PLAN.md §2 order\necho 'plan §2'\n",                    # a shell comment, a literal
    "configs/registry/models.yaml": "models:\n  - id: m1\n    notes: kept under plan §6\n",    # a data leaf
}


def test_a_section_of_a_private_plan_or_spec_fails_in_comments_and_docstrings_only(tmp_path):
    root = _repo(tmp_path / "r", PLAN_SECTION_FILES)
    assert _comment_hits(root, tmp_path) == [("configs/c.yaml", "L1"), ("scripts/run.sh", "L2"), ("src/m.py", "L1"),
                                             ("src/m.py", "L2"), ("src/m.py", "L3"), ("src/m.py", "L4")]
    assert not [h for h in _scan(root, tmp_path) if h.scope == "leaf"]


WORKSTREAM_FILES = {
    "src/m.py": ('"""WS-A tests: the result store round-trips a row."""\n'                    # L1 a lettered id
                 "X = 1  # WS3-G's determinism check reads this field\n"                       # L2 a numbered id
                 "# workstream C ships the emitter; WS1-D' adds the header\n"                  # L3 a word form, a prime
                 "WS3_MODELS = []  # the doc lanes on chat checkpoints\n"                      # L4 an identifier in code
                 "# reads WS3_MODELS and SEEDREP_WS0_TREE\n"                                   # L5 identifiers in a comment
                 'Z = "WS1"\n'                                                                # L6 a string literal
                 "# an AWS2 region, the NEWS1 feed, the scenario workstream\n"),              # L7 other words
    "configs/c.yaml": "# (WS0-1) the prompted baselines\nk: 1\n",                             # a YAML comment
}


def test_a_workstream_id_fails_in_comments_and_docstrings_and_an_identifier_does_not(tmp_path):
    root = _repo(tmp_path / "r", WORKSTREAM_FILES)
    assert _comment_hits(root, tmp_path) == [("configs/c.yaml", "L1"), ("src/m.py", "L1"), ("src/m.py", "L2"),
                                             ("src/m.py", "L3")]


CRITERION_FILES = {
    "src/m.py": ('"""Write-performance smoke (acceptance 31): rows per second."""\n'          # L1 an acceptance number
                 "X = 1  # criterion 3 gates the PASS\n"                                       # L2 a bare criterion
                 "# the acceptance check 7e, spec criteria 1-5, the criterion-3 waiver\n"      # L3 check, plural, hyphen
                 "# writes criterion3_waiver.json; --criterion3-waiver; criteria-1.json\n"     # L4 identifiers, file names
                 "# the acceptance rate is 0.3; a criterion its rows meet\n"                   # L5 ordinary words
                 'Y = "criterion 3"\n'),                                                      # L6 a string literal
    "configs/c.yaml": "tol: 0.02  # success criterion 1: 0.50 +/- 0.02\n",                    # a YAML comment
    "scripts/run.sh": "#!/bin/bash\n# Criterion 6 (acceptance #3): the GPU smoke\n",          # a shell comment
}


def test_a_numbered_spec_criterion_fails_in_comments_and_docstrings_and_a_name_does_not(tmp_path):
    root = _repo(tmp_path / "r", CRITERION_FILES)
    assert _comment_hits(root, tmp_path) == [("configs/c.yaml", "L1"), ("scripts/run.sh", "L2"), ("src/m.py", "L1"),
                                             ("src/m.py", "L2"), ("src/m.py", "L3")]


SPEC_SLUG_FILES = {
    "src/m.py": ('"""The harmful pairs (spec-bidpo-benign, the pairing rule)."""\n'           # L1 a hyphenated slug
                 "X = 1  # metrics-only logging (spec two-si-stage3)\n"                        # L2 a parenthesised slug
                 "# the spec ``bcb-bfcl-200-widen`` cells\n"                                   # L3 a quoted slug
                 "# the LoRA export's spec cross-checks; a spec-compliant parser\n"           # L4 a spec the code checks
                 "# the AxBench spec: unit-norm w\n"                                           # L5 a public spec
                 'Y = "spec two-si-stage3"\n'),                                               # L6 a string literal
    "configs/c.yaml": "prefill: false  # spec open-question defaults, pinned OFF\n",           # a YAML comment
}


def test_a_private_spec_named_by_its_slug_fails_in_comments_and_docstrings(tmp_path):
    root = _repo(tmp_path / "r", SPEC_SLUG_FILES)
    assert _comment_hits(root, tmp_path) == [("configs/c.yaml", "L1"), ("src/m.py", "L1"), ("src/m.py", "L2"),
                                             ("src/m.py", "L3")]


def _notebook(*cells) -> str:
    return json.dumps({"metadata": {}, "cells": list(cells)})


SPLIT_FILES = {
    "plan_section": ({
        "src/m.py": ("# the drain comes first (plan\n"                            # L1-L2 a comment passage
                     "# §2.5.4), then the copy\n"
                     "def f():\n"
                     '    """The conversion stage (plan\n'                         # L4-L5 a docstring
                     '    section 4 step 2) strips the table."""\n'
                     "    x = 1  # the order of the plan's\n"                     # L6-L7 a trailing comment going on
                     "    # §3 for the cells\n"
                     "    y = 2  # the migration plan\n"                           # L8-L9 code beside the next comment
                     "    z = 3  # §3 lists the tables\n"
                     "    # the plan\n"                                            # L10-L12 a blank comment line between
                     "    #\n"
                     "    # §4 is next\n"
                     '    w = ("the plan "\n'                                      # L13-L14 a string literal
                     '         "§5")\n'
                     "    return w\n"
                     "def g():\n"
                     '    """Ends with the plan"""\n'                              # L17-L18 a docstring's quotes between
                     "    # §6 is the order\n"
                     "    return 1\n"),
        "configs/c.yaml": "enable: false  # defaults pinned OFF (plan\n# §6)\nk: 1\n",
        "src/schema.sql": "-- the index order (plan\n-- §4)\nSELECT 1;\n/* the copy (plan\n * §5) */\nSELECT 2;\n",
        "docs/notes.md": "The drain comes first (plan\n§2.5.4).\n\nA plan\n\n§3 is elsewhere.\n- the plan\n- §4 next\n",
        "nb/n.ipynb": _notebook(
            {"cell_type": "markdown", "source": ["The drain (plan\n", "§2.5.4) first"]},
            {"cell_type": "code", "source": ["x = 1  # the drain (plan\n", "# §2)\n", "y = 2"],
             "outputs": [{"output_type": "stream", "text": ["printed (plan\n", "§3)\n"]}]}),
    }, [("configs/c.yaml", "L1"), ("docs/notes.md", "L1"), ("nb/n.ipynb", "cell0:L1"), ("nb/n.ipynb", "cell1:L1"),
        ("nb/n.ipynb", "cell1:out0:L1"), ("src/m.py", "L1"), ("src/m.py", "L4"), ("src/m.py", "L6"), ("src/schema.sql", "L1"),
        ("src/schema.sql", "L4")]),
    "workstream": ({
        "src/m.py": ('"""The result store round-trips a row (the WS-\n'           # L1-L2 a hyphen the wrap broke
                     'A tests)."""\n'
                     "# the emitter ships in workstream\n"                         # L3-L4 the word form
                     "# C, the header after it\n"
                     "WS3_MODELS = []  # the doc lanes, then\n"                    # L5-L6 identifiers only
                     "# WS3_MODELS again\n"
                     "# the scenario workstream\n"                                 # L7-L8 a word, not an id
                     "# Cards come next\n"),
        "scripts/run.sh": "#!/bin/bash\n# owned by workstream\n# B until the merge\necho ok\n",
    }, [("scripts/run.sh", "L2"), ("src/m.py", "L1"), ("src/m.py", "L3")]),
    "spec_criterion": ({
        "src/m.py": ('"""Write-performance smoke (acceptance\n'                    # L1-L2 a docstring
                     '31): rows per second."""\n'
                     "X = 1  # the gate is spec criteria\n"                        # L3-L4 a trailing comment going on
                     "# 1-5, all of them\n"
                     "# the acceptance\n"                                          # L5-L7 ordinary words
                     "# rate is 0.3, a criterion\n"
                     "# its rows meet\n"
                     'Y = ("criterion "\n'                                         # L8-L9 a string literal
                     '     "3")\n'),
        "configs/c.yaml": "tol: 0.02  # acceptance\n# 5: 0.50 +/- 0.02\n",
        "scripts/run.sh": "#!/bin/bash\n# the GPU smoke, acceptance\n# #3 of them\necho ok\n",
    }, [("configs/c.yaml", "L1"), ("scripts/run.sh", "L2"), ("src/m.py", "L1"), ("src/m.py", "L3")]),
    "spec_slug": ({
        "src/m.py": ('"""The harmful pairs (spec-bidpo-\n'                         # L1-L2 a hyphen the wrap broke
                     'benign, the pairing rule)."""\n'
                     "X = 1  # metrics-only logging (spec\n"                       # L3-L4 a trailing comment going on
                     "# two-si-stage3)\n"
                     "# the LoRA export's spec\n"                                  # L5-L7 a spec the code checks, a public one
                     "# cross-checks, and the AxBench spec:\n"
                     "# unit-norm w\n"),
        "configs/c.toml": "prefill = false  # the spec\n# ``open-question-defaults``, pinned OFF\n",
    }, [("configs/c.toml", "L1"), ("src/m.py", "L1"), ("src/m.py", "L3")]),
}


@pytest.mark.parametrize("cls", sorted(SPLIT_FILES))
def test_a_citation_a_line_break_splits_fails_inside_one_comment_passage_only(cls, tmp_path):
    files, want = SPLIT_FILES[cls]
    root = _repo(tmp_path / "r", files)
    assert _comment_hits(root, tmp_path) == want


def test_a_split_citation_is_reported_once_at_its_first_line_with_the_lines_it_spans(tmp_path):
    root = _repo(tmp_path / "r", {"src/m.py": ("X = 1  # the drain comes first (plan\n"     # L1-L2 split
                                               "# §2.5.4), then the copy\n"
                                               "Y = 2\n"
                                               "# gated by success\n"                       # L4-L5 the row holds on L5 alone
                                               "# criterion 1 and the smoke\n"),
                                  "tests/published_repo.py": 'UNPUBLISHED = ("CLAUDE.md",)\n'})   # the command line reads it
    hits = [h for h in _scan(root, tmp_path) if h.scope == "comment"]
    assert sorted((h.where, h.unit[h.start:h.end]) for h in hits) == [("L1", "plan\n# §"), ("L5", "criterion 1")]
    (h,) = [h for h in hits if h.where == "L1"]
    assert h.unit == "X = 1  # the drain comes first (plan\n# §2.5.4), then the copy"
    assert h.unit_hash == SCAN.sha256_text(h.unit) and "\\n" in h.excerpt()
    exempt = SCAN.AllowRow("src/m.py", "attribution", "line", h.unit_hash, "a reviewed passage", 1)
    assert [x.where for x in SCAN.apply_allowlist(hits, [exempt])] == ["L5"] and exempt.used == 1
    repo = Path(__file__).resolve().parents[1]
    grep = subprocess.run([sys.executable, str(repo / "reproduce" / "tools" / "scrub_scan.py"), "--root", str(root), "--tokens",
                           str(repo / "tests" / "front_door_testlib.py"), "--no-allowlist", "--scope", "comment", "--grep"],
                          capture_output=True, text=True, check=True).stdout.splitlines()
    assert grep[:3] == ["src/m.py:1:X = 1  # the drain comes first (plan", "src/m.py:2:# §2.5.4), then the copy",
                        "src/m.py:5:# criterion 1 and the smoke"], grep


SPLIT_ATTRIBUTION_FILES = {
    "src/m.py": ("# kept since the user's\n"                                     # L1-L2 an attribution
                 "# ruling of 2026-09-11\n"
                 "X = 1\n"
                 "# the pool (spec fixed\n"                                      # L4-L5 a spelled-out decision
                 "# decision that every producer emits)\n"
                 "Y = 2\n"
                 "# fixed after the S4\n"                                        # L7-L8 a gate name
                 "# gate passed\n"
                 "Z = 3\n"
                 "# the system prompt and (the\n"                                # L10-L11 the chat role
                 "# user turn) come first\n"),
}


def test_every_comment_row_reads_a_passage_across_its_line_breaks(tmp_path):
    root = _repo(tmp_path / "r", SPLIT_ATTRIBUTION_FILES)
    assert _comment_hits(root, tmp_path) == [("src/m.py", "L1"), ("src/m.py", "L4"), ("src/m.py", "L7")]
