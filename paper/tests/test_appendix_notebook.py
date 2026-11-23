"""The appendix analyses in the paper notebook that read committed data files (Figures 9 and 10, Tables 3 and 6): their generator cells,
figure names, README rows and build.

The generator (paper/notebooks/make_registry_notebook.py) carries, after the generator comparison and right before the zip cell
(b7d4e9f2), a section opener (appendix_common's two cells), then per appendix subsection a short header cell and its analyses' cells,
in appendix order: the training corpora, control directions (Figure 9's two halves, Figure 10, then the forward passes the text
quotes), capability losses. These tests pin that placement, that the paper-figures zip names exactly the new figures the paper prints,
in the paper's order between Figures 8 and 11 (the corpus audit and capability error compositions are tables there, the forward passes
numbers in the text), that each cell reads only its committed data file and the registry source the notebook already reads, that the
cells, modules, builders and data files carry no process provenance, and that the README's figure map names
each figure with its judge. They read the generator and modules as text (no registry, no import of registry_nb) except the last, which
builds the notebook into a temp dir.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
GENERATOR = NOTEBOOKS / "make_registry_notebook.py"
README = NOTEBOOKS / "README.md"

# (cell id, kind) in order, right before b7d4e9f2 and after the generator comparison (431232a9)
NEW_CELLS = (("347c3abd", "md"), ("f60c0561", "code"),
             ("ba7e3dcf", "md"), ("3a7c91e4", "md"), ("c5e0b2d8", "code"),
             ("2802219e", "md"), ("f46e655e", "md"), ("5f821770", "code"), ("ee8bdbc1", "md"), ("0c6192ec", "code"),
             ("6b1e4f90", "md"), ("e2c85a17", "code"), ("53f2d0e0", "md"), ("a77fad73", "code"),
             ("da341da1", "md"), ("037dbf1e", "md"), ("b2d18d18", "code"))
# the short header of each appendix subsection, and the analyses' own heading cells under it
SUBSECTIONS = {"ba7e3dcf": "what the training corpora contain", "2802219e": "control directions",
               "da341da1": "error analysis of capability losses"}
ANALYSIS_HEADINGS = ("3a7c91e4", "f46e655e", "ee8bdbc1", "6b1e4f90", "53f2d0e0", "037dbf1e")
# code cell -> (module it imports, the figure names it draws: a literal in the cell or the module's constant)
PORTS = {"c5e0b2d8": ("appendix_corpus_audit", ["figA_corpus_audit"]),
         "5f821770": ("appendix_geometry_evenodd", ["figA_control_evenodd"]),
         "0c6192ec": ("appendix_geometry_forward", ["figA_refusal_enrichment"]),
         "e2c85a17": ("appendix_geometry_overlap", ["figA_cK_overlap"]),
         "a77fad73": ("appendix_geometry_forward", ["figA_displacement"]),
         "b2d18d18": ("appendix_caperrors", ["figA_qwen_capability_errors", "figA_models_capability_errors"])}
NEW_FIGURES = ["figA_corpus_audit", "figA_control_evenodd", "figA_refusal_enrichment", "figA_cK_overlap", "figA_displacement",
               "figA_qwen_capability_errors", "figA_models_capability_errors"]
# the ones the paper prints (Figure 9's two halves, Figure 10), so the zip cell names them between Figure 8 and Figure 11; the corpus
# audit and capability error compositions are tables there and the forward passes are numbers in its text
ZIPPED_FIGURES = ["figA_control_evenodd", "figA_refusal_enrichment", "figA_cK_overlap"]
# the constant each module names its figure(s) by, where the cell passes the constant rather than the literal
MODULE_FIGURE_CONSTANTS = {"appendix_geometry_evenodd": {"FIGURE": "figA_control_evenodd"},
                           "appendix_geometry_forward": {"FIG_DISPLACEMENT": "figA_displacement", "FIG_ENRICHMENT": "figA_refusal_enrichment"},
                           "appendix_geometry_overlap": {"FIGURE": "figA_cK_overlap"},
                           "appendix_corpus_audit": {"FIGURE_NAME": "figA_corpus_audit"}}

#: the process names and words no cell, module, builder or data file may carry, as rows of the publication scan's token table
#: (reproduce/tools/scrub_scan.py: a name is the sha256 of its lowercase runs, built with scrub_scan.hash_row outside the repository, so
#: this file never spells one), and the old checkout name and the machine label, which the tree scan's own table refuses as well
PROCESS_ROWS = (
    ("process", "hash", "a9c5d4f87f55cb20ccde0648f8f6ac9d10907c16aa531e10606d319c340f4007", {}),
    ("process", "hash", "f9dc41d064b2718e3c1aa275e6c1fe2082e889ca585b739caad79d424d7acde2", {}),
    ("process", "hash", "d8c52d342309e3960ec29cf22c7e75e91dfdf5fff8444eb527a1e0c3771e30bf", {}),
    ("process", "hash", "2d70999ae1805e4bcef9b4ab3a4b827f578c61740f30076fcdc35c7ae7f586b3", {}),
    ("process", "hash", "f49dd73f5cf9d596c1958c1f9345df6dfd14d332d3af85dfd92d9b859f7cd3d8", {}),
    ("process", "hash", "a0cecbead27dfb0d580737b7a77b2a0864ab7695ff8bf80b5766c3e9e493797b", {}),
    ("process", "hash", "d2bd119147b277fbc80c82f85d9785f5cde1c3410fbbff4dff1a860fdfb8858d", {}),
    ("process", "hash", "d4a5596c193aabe2652d5c96f1dedb836b0cd3e6fb11d5699c5616e682f61550",
     {"sep": ["-", "_"], "n": 2, "head": "97a62ad21d79c01cceb7767952acec4fec86bfe909b06e5f3f6963365cf91ab8"}),
    ("process", "hash", "2ff8ddd61f5e0a0c72ad9390b7c448e32b5055707f1be2e65a361b24f2a180ce", {}),
)


def _src() -> str:
    return GENERATOR.read_text(encoding="utf-8")


def _cells(src: str) -> dict:
    """{cell id: (kind, body)} exactly as the generator writes them (the generator strips the newlines around the raw string)."""
    return {cid: (kind, body.strip("\n"))
            for kind, cid, body in re.findall(r'^(md|code)\("([0-9a-f]{8})", r"""(.*?)"""\)', src, flags=re.M | re.S)}


def _paper_figures(src: str) -> list:
    zip_cell = src[src.index('code("b7d4e9f2"'):src.index('md("92fdc8e2"')]
    return re.findall(r'"([^"]+)"', re.search(r"PAPER_FIGURES = \[(.*?)\]", zip_cell, flags=re.S).group(1))


def _module_constants(module: str) -> dict:
    tree = ast.parse((NOTEBOOKS / f"{module}.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value.value
    return out


def test_the_cells_sit_in_appendix_order_right_before_the_zip_cell_after_the_generator_comparison():
    src = _src()
    order = re.findall(r'^(md|code)\("([0-9a-f]{8})"', src, flags=re.M)
    ids = [cid for _, cid in order]
    assert len(ids) == len(set(ids))
    i = ids.index("b7d4e9f2") - len(NEW_CELLS)
    assert tuple((cid, kind) for kind, cid in order[i:i + len(NEW_CELLS)]) == NEW_CELLS
    assert ids.index("431232a9") < i and ids.index("76fdab54") < i


def test_each_appendix_subsection_has_a_short_header_and_the_analyses_sit_under_it():
    cells = _cells(_src())
    assert cells["347c3abd"][1].startswith("## Appendix — ")
    for cid, title in SUBSECTIONS.items():
        kind, body = cells[cid]
        assert kind == "md" and body.splitlines()[0] == f"### Appendix — {title}", (cid, body.splitlines()[0])
        assert len(body.split()) < 60, f"{cid}: a subsection header is a heading and one sentence"
    for cid in ANALYSIS_HEADINGS:
        first = cells[cid][1].splitlines()[0]
        assert first.startswith("#### ") and "Appendix" not in first, (cid, first)


def test_the_figures_the_paper_prints_are_zipped_and_the_compositions_are_drawn_but_not_zipped():
    src = _src()
    cells = _cells(src)
    names = _paper_figures(src)
    assert names[names.index("figA_transfer_models") + 1:names.index("figA_qwen_ctlstop_rule_comparison")] == ZIPPED_FIGURES
    assert not (set(NEW_FIGURES) - set(ZIPPED_FIGURES)) & set(names), "a figure the paper does not print is in the paper-figures zip"
    imported = set()
    for cid, (module, figures) in PORTS.items():          # in cell order: a module is imported by the first cell that uses it
        kind, body = cells[cid]
        assert kind == "code" and (module in imported or re.search(rf"^import {module} as [A-Z]+$", body, flags=re.M)), cid
        imported.add(module)
        consts = _module_constants(module)
        for fig in figures:
            named = [k for k, v in MODULE_FIGURE_CONSTANTS.get(module, {}).items() if v == fig]
            assert f'"{fig}"' in body or any(f".{k}" in body for k in named), (cid, fig)
            for k in named:
                assert consts.get(k) == fig, (module, k, consts.get(k))
    assert sorted(f for _, figs in PORTS.values() for f in figs) == sorted(NEW_FIGURES)


def test_each_cell_reads_only_its_committed_data_file_and_the_notebooks_own_source():
    """Each module's data file is under paper/notebooks/data/ beside it (no environment variable, no home path); a cell opens a raw
    connection only under the live branch and otherwise reads the source through R.query, which the committed snapshot answers."""
    cells = _cells(_src())
    for module in sorted({m for m, _ in PORTS.values()}):
        stem = module.removeprefix("appendix_")
        msrc = (NOTEBOOKS / f"{module}.py").read_text(encoding="utf-8")
        assert re.search(rf'^SIDECAR = os\.path\.join\((?:R\.)?HERE, "data", "appendix_{stem}\.json"\)$', msrc, flags=re.M), module
        assert (NOTEBOOKS / "data" / f"appendix_{stem}.json").is_file(), module
    for cid, (kind, body) in cells.items():
        if cid not in dict(NEW_CELLS) or kind != "code":
            continue
        assert "os.environ" not in body and "~/" not in body and "/home/" not in body and "expanduser" not in body, cid
        lines = [ln for ln in body.splitlines() if ln.strip()]
        for k, ln in enumerate(lines):
            if "R.connect(" in ln:
                assert any('if R.SOURCE == "live"' in p for p in lines[max(0, k - 2):k]), (cid, ln)
            if "judgments_sql" in ln:
                assert "R.query(GE.judgments_sql(ge)" in ln, (cid, ln)
    # the sign decomposition checks its pinned judgments wherever the source holds them (live, or a frame cut with the paper's judges)
    # or should (a frame at or past the last of them, which then fails by name if it lacks them), and leaves them to the file only on a
    # frame cut before them that holds none of them
    ev = cells["5f821770"][1]
    assert ('_ge_jcheck = R.SOURCE == "live" or len(_ge_judg) > 0 or (_ge_head is not None and int(_ge_head) >= int(_ge_jseq))' in ev
            and '_ge_jseq = _ge_runs.attrs.get("head"), ge["provenance"]["verdicts"]["max_judgment_ingest_seq"]' in ev
            and "_ge_judg if _ge_jcheck else None" in ev)


def test_the_cells_carry_no_process_provenance():
    cells = _cells(_src())
    banned = re.compile(r"(?i)\bD\d{2,}\b|decisions\.md|\bruling|~/|/home/|specs/")
    process = _process_matcher()
    for cid, _ in NEW_CELLS:
        body = cells[cid][1]
        hit = banned.search(body)
        assert not hit, (cid, hit.group(0))
        assert not process.find(body), (cid, "a process name")
        assert "import registry_nb as R" not in body and "R.style()" not in body, f"{cid} re-runs the header cell's setup"


def test_the_modules_builders_and_data_files_carry_no_process_name():
    """Process is not method: no file the cells ship (the readers, the builders, the data files with their provenance prose) carries a
    process name or word, the old checkout name or the machine label."""
    process = _process_matcher()
    data = NOTEBOOKS / "data"
    files = (sorted(NOTEBOOKS.glob("appendix_*.py")) + sorted(data.glob("build_appendix_*.py")) + sorted(data.glob("appendix_*.json")))
    assert len(files) >= 16
    hits = [(f.name, len(found)) for f in files for found in [process.find(f.read_text(encoding="utf-8"))] if found]
    assert not hits, hits


def test_the_process_rows_are_well_formed_and_find_what_they_hash():
    """Every row is a sha256 hash row the scan parses, and the matcher finds a token the moment its row is among them (an invented
    token planted through scrub_scan.hash_row, as the scan's own tests plant theirs)."""
    scan = _scrub_scan()
    assert len(PROCESS_ROWS) == 9 and all(r[0] == "process" and r[1] == "hash" and re.fullmatch(r"[0-9a-f]{64}", r[2]) for r in PROCESS_ROWS)
    assert len(scan.build_matchers(PROCESS_ROWS)["rows"]) == len(PROCESS_ROWS)
    planted = scan.build_matchers(PROCESS_ROWS + (scan.hash_row("process", "zqx-plant-token", sep="-"),))["tree"]
    assert planted.find("a zqx-plant-token b") and not _process_matcher().find("a zqx-plant-token b")


def test_the_readme_figure_map_names_each_new_figure_with_its_judge():
    readme = README.read_text(encoding="utf-8")
    fmap = readme[readme.index("## Figure map"):readme.index("## How to read the figures")]
    for fig in NEW_FIGURES:
        rows = [ln for ln in fmap.splitlines() if ln.startswith("|") and f"`{fig}`" in ln]
        assert len(rows) == 1, fig
        assert re.search(r"Judge: MiniMax-M3|No judge|Labeller: Claude Opus 5\.5", rows[0]), fig
    for label in ("tab:corpus-audit", "tab:app_evenodd", "tab:caperrors", "tab:caperrors-bcb"):
        assert label in fmap, label
    data = readme[readme.index("## The appendix analyses' data"):readme.index("## Editing the notebook")]
    for module in sorted({m for m, _ in PORTS.values()}):
        assert f"`{module}.json`" in data, module
    tests = readme[readme.index("## Tests"):]
    for name in ("test_appendix_common", "test_appendix_notebook", "test_appendix_corpus_audit", "test_appendix_geometry_evenodd",
                 "test_appendix_geometry_forward", "test_appendix_geometry_overlap", "test_appendix_caperrors"):
        assert f"paper/tests/{name}.py" in tests, name


def _scrub_scan():
    """reproduce/tools/scrub_scan.py as a module, loaded once under the name `scrub_scan` (as tests/front_door_testlib.py loads it)."""
    if "scrub_scan" not in sys.modules:
        spec = importlib.util.spec_from_file_location("scrub_scan", REPO / "reproduce" / "tools" / "scrub_scan.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["scrub_scan"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["scrub_scan"]


def _process_matcher():
    """The tree-scope matcher of PROCESS_ROWS."""
    return _scrub_scan().build_matchers(PROCESS_ROWS)["tree"]


def _interpreter_with(module: str):
    """The test's own interpreter, else SISTEER_NB_PYTHON (the notebook interpreter reproduce/10_figures.sh honours), else the clone's
    own .venv, whichever imports `module` first; None when none does."""
    for py in (sys.executable, os.environ.get("SISTEER_NB_PYTHON") or "", str(REPO / ".venv" / "bin" / "python")):
        if py and os.path.exists(py) and subprocess.run([py, "-c", f"import {module}"], capture_output=True).returncode == 0:
            return py
    return None


def test_the_built_notebook_carries_the_cells_verbatim(tmp_path):
    """Build the notebook from a copy of the generator (it writes beside itself) and read the new cells back."""
    py = _interpreter_with("nbformat")
    if py is None:
        pytest.skip("no interpreter with nbformat (the notebook group); make_registry_notebook.py cannot build the ipynb")
    gen = tmp_path / "make_registry_notebook.py"
    shutil.copy(GENERATOR, gen)
    r = subprocess.run([py, str(gen)], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    nb = json.loads((tmp_path / "registry_qwen_figures.ipynb").read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in nb["cells"]}
    ids = [c["id"] for c in nb["cells"]]
    i = ids.index("b7d4e9f2") - len(NEW_CELLS)
    assert tuple(ids[i:i + len(NEW_CELLS)]) == tuple(cid for cid, _ in NEW_CELLS)
    gen_cells = _cells(_src())
    for cid, kind in NEW_CELLS:
        cell = by_id[cid]
        assert cell["cell_type"] == {"md": "markdown", "code": "code"}[kind]
        assert "".join(cell["source"]) == gen_cells[cid][1], cid
        if kind == "code":
            compile("".join(cell["source"]), f"<cell {cid}>", "exec")
    zip_src = "".join(by_id["b7d4e9f2"]["source"])
    assert all(f'"{f}"' in zip_src for f in ZIPPED_FIGURES)
