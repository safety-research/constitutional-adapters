"""The registry notebook redacts producer session names from what it prints, and its source lists no name.

`paper/notebooks/registry_nb.py` reads the names to redact from the records its source holds (the producer fields of runs,
objects and judgments) once the source has been opened, and removes them only in attribution shapes, since a producer name can
also be an ordinary word. A file named by REGISTRY_NB_REDACT_NAMES adds names that go wherever they stand as a whole word. These
tests build a SYNTHETIC live registry (a minimal SQLite file with invented names) and import the notebook module fresh from a
clone-shaped mirror (`test_nb_snapshot_fallback`'s helpers), so nothing reads the machine's registry.
"""
from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pandas as pd

from test_nb_snapshot_fallback import NOTEBOOKS, import_notebook, nb_env, notebook_mirror

PRODUCERS = ("Alpha", "Beta-2")          # invented producer names
FILE_NAME = "Gamma"                      # an invented name only free text carries


def _registry(tmp_path, judgment: bool = True) -> Path:
    root = Path(tmp_path) / "home" / "registry"
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / "registry.sqlite")
    con.executescript("""
        CREATE TABLE record_ingest(ingest_seq INTEGER); INSERT INTO record_ingest VALUES (7);
        CREATE TABLE run(run_id TEXT, producer_session TEXT);
        INSERT INTO run VALUES ('r1', 'Alpha'), ('r2', NULL), ('r3', '  ');
        CREATE TABLE object(object_id TEXT, fit_producer_session TEXT); INSERT INTO object VALUES ('o1', 'Beta-2');
    """)
    if judgment:
        con.executescript("""
            CREATE TABLE judgment(judgment_id TEXT, producer_session TEXT, claimed_by_session TEXT);
            INSERT INTO judgment VALUES ('j1', 'Alpha', NULL);
        """)
    con.commit()
    con.close()
    return root


def _import(tmp_path, monkeypatch, names_file: str | None = None, judgment: bool = True):
    root = _registry(tmp_path, judgment=judgment)
    monkeypatch.delenv("REGISTRY_NB_REDACT_NAMES", raising=False)
    env = nb_env(tmp_path, REGISTRY_NB_SOURCE="live", REGISTRY_ROOT=str(root), REGISTRY_NB_REDACT_NAMES=names_file)
    return import_notebook(notebook_mirror(tmp_path), env, monkeypatch)


def test_producer_names_come_from_the_records_once_the_source_is_open(tmp_path, monkeypatch):
    R = _import(tmp_path, monkeypatch)
    assert R.session_names() == ((), ())                     # importing opens nothing, so no producer name is known yet
    assert R.redact_session_names("(Alpha, 2026-09-13: cap)") == "(Alpha, 2026-09-13: cap)"
    assert R.head() == 7                                     # the first read opens the source
    assert R.session_names() == (PRODUCERS, ())


def test_a_producer_name_goes_only_in_an_attribution(tmp_path, monkeypatch):
    R = _import(tmp_path, monkeypatch)
    R.head()
    cases = {
        "(Alpha, 2026-09-13: cap 2048 is fine)": "(2026-09-13: cap 2048 is fine)",
        "(Beta-2 2026-09-14 rerun)": "(2026-09-14 rerun)",
        "Alpha: the dose moved": "the dose moved",
        "filed by session Beta-2 today": "filed by a session today",
        "the Alpha lane closed": "the lane closed",
        "the Alpha steer at dose 2": "the Alpha steer at dose 2",            # an ordinary word outside an attribution stays
        "Alphabet and Beta-20 are other names": "Alphabet and Beta-20 are other names",
    }
    for text, want in cases.items():
        assert R.redact_session_names(text) == want, text
    frame = R.redact_session_names(pd.DataFrame({"note": ["Alpha: x", "y"], "n": [1, 2]}))
    assert frame["note"].tolist() == ["x", "y"] and frame["n"].tolist() == [1, 2]
    assert R.redact_session_names({"k": ["Alpha: x"]}) == {"k": ["x"]}


def test_a_listed_name_goes_wherever_it_stands(tmp_path, monkeypatch):
    names = Path(tmp_path) / "names.txt"
    names.write_text(f"# one name per line\n{FILE_NAME}\n\n", encoding="utf-8")
    R = _import(tmp_path, monkeypatch, names_file=str(names))
    assert R.session_names() == ((), (FILE_NAME,))           # known before the source is open
    assert R.redact_session_names(f"built by {FILE_NAME} (x)") == "built by (x)"
    assert R.redact_session_names(f"({FILE_NAME} 2026-01-02 y)") == "(2026-01-02 y)"
    R.head()
    assert R.session_names() == (PRODUCERS, (FILE_NAME,))


def test_a_source_without_a_producer_table_still_redacts(tmp_path, monkeypatch):
    R = _import(tmp_path, monkeypatch, judgment=False)
    R.head()
    assert R.session_names() == (PRODUCERS, ())
    assert R.redact_session_names("Beta-2: x") == "x"


def test_the_module_lists_no_name_to_redact():
    """No module-level tuple or list of names: the names come from the records or the private file."""
    tree = ast.parse((NOTEBOOKS / "registry_nb.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Tuple, ast.List)):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            assert not any("SESSION" in t or "NAMES" in t for t in targets), targets
