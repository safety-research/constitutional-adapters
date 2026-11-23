"""The committed notebook is the generator's, executed: registry_qwen_figures.ipynb holds exactly the cells make_registry_notebook.py
writes (the same ids, in the same order, with the same types and sources), and every code cell ran once, in order, with no error.

The placement and caption tests read the generator, while the committed render's own tests (the table first rows, the slow modules'
cells) and readers of the repository read the .ipynb. A generator change committed without regenerating the notebook once left the
committed render without three of the generator's cells (the distillation-target figure and its paper table) and with two stale
sources, and no test saw it. The generator runs here with its write captured, so nothing is written; it only builds strings, so this
test is fast. It needs no notebook extras: where `nbformat` is absent (the CPU suite installs the `dev` group, not `notebook`), a
stand-in for the four calls the generator makes builds the same cells, which the last test checks wherever `nbformat` is installed.
This directory is outside the suite's `testpaths`, so tests/test_nb_committed_sync_gate.py loads this module and runs its checks
where the landing gate collects them; a check added here must be bound there too, or that file's own test fails.
To fix a failure, rebuild and execute the notebook as paper/notebooks/README.md says ("Run it")."""
import importlib.util
import json
import pathlib
import runpy
import sys
import types

import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
GENERATOR = NOTEBOOKS / "make_registry_notebook.py"
NOTEBOOK = NOTEBOOKS / "registry_qwen_figures.ipynb"
REBUILD = "the committed notebook is not the generator's: rebuild and execute it (paper/notebooks/README.md, 'Run it')"
HAVE_NBFORMAT = importlib.util.find_spec("nbformat") is not None


def _cells(cells) -> list:
    """(id, type, source) of every cell; a source read from disk is a list of lines, one built in memory a string."""
    return [(c["id"], c["cell_type"], "".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in cells]


def _generate(stand_in: bool) -> list:
    """The generator's cells, from running it with its write captured: through the real `nbformat` (which validates the notebook
    first), or through a stand-in for `nbformat.v4`'s three cell and notebook builders, `validate` and `write`."""
    written = []
    with pytest.MonkeyPatch.context() as mp:
        if stand_in:
            nbf, v4 = types.ModuleType("nbformat"), types.ModuleType("nbformat.v4")
            v4.new_markdown_cell = lambda src: {"cell_type": "markdown", "source": src}
            v4.new_code_cell = lambda src: {"cell_type": "code", "source": src}
            v4.new_notebook = lambda cells: types.SimpleNamespace(cells=cells, metadata={})
            nbf.v4, nbf.validate = v4, lambda nb: None
            nbf.write = lambda nb, fp, *a, **k: written.append((nb, fp))
            mp.setitem(sys.modules, "nbformat", nbf)
            mp.setitem(sys.modules, "nbformat.v4", v4)
        else:
            import nbformat
            mp.setattr(nbformat, "write", lambda nb, fp, *a, **k: written.append((nb, fp)))
        runpy.run_path(str(GENERATOR), run_name="__generator__")
    (nb, fp), = written
    assert pathlib.Path(fp) == NOTEBOOK, fp
    return _cells(nb.cells)


@pytest.fixture(scope="module")
def generated() -> list:
    return _generate(stand_in=not HAVE_NBFORMAT)


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_the_committed_notebook_holds_the_generators_cell_ids_in_order(generated, committed):
    got, want = [c[0] for c in _cells(committed["cells"])], [c[0] for c in generated]
    missing, extra = [i for i in want if i not in got], [i for i in got if i not in want]
    assert got == want, f"{REBUILD}; missing {missing}, extra {extra}"


def test_every_committed_cell_has_the_generators_type_and_source(generated, committed):
    want = {cid: (kind, src) for cid, kind, src in generated}
    stale = [cid for cid, kind, src in _cells(committed["cells"]) if cid in want and want[cid] != (kind, src)]
    assert not stale, f"{REBUILD}; cells whose type or source differ: {stale}"


def test_the_committed_notebook_is_executed_once_in_order_without_an_error(committed):
    code = [c for c in committed["cells"] if c["cell_type"] == "code"]
    assert [c.get("execution_count") for c in code] == list(range(1, len(code) + 1)), REBUILD
    errors = [c["id"] for c in code if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    assert not errors, f"cells whose committed output is an error: {errors}"


@pytest.mark.skipif(not HAVE_NBFORMAT, reason="nbformat is not installed (the notebook group): the stand-in is all there is to compare")
def test_the_stand_in_builds_the_cells_nbformat_builds():
    assert _generate(stand_in=True) == _generate(stand_in=False)
