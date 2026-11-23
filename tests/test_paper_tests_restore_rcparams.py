"""The paper tests' slow fixtures leave matplotlib's rcParams as they found them.

The slow modules under paper/tests execute the committed notebook's cells, and the header cell calls `registry_nb.style()`, which
sets the publication rcParams for the whole process (`axes.titlelocation = "left"` among them). Run in one session, every module
after such a module drew under those rcParams: a panel title sat on the left, `ax.get_title()` (the centre title) read "", and
paper/tests/test_registry_nb_ctlstop_appendix.py's comparison-grid test failed whenever a slow module ran before it.

Pinned here, without running the slow modules (they read the committed paper frame for minutes):
  * every fixture under paper/tests that executes the notebook's cells (an `exec` in the fixture or in a module function it calls, in
    a module that reads the notebook or its generator) is covered by an rcParams restore: it requests the conftest's
    `notebook_rcparams`, directly or through a fixture of its module that does, or the conftest restores rcParams around every module
    (an autouse module-scoped fixture that enters `matplotlib.rc_context`); and the five slow modules are among those found;
  * the restore works: a synthetic session with paper/tests/conftest.py, whose first module's fixture applies the notebook's style
    while requesting `notebook_rcparams`, leaves the next module at matplotlib's defaults; the same two modules without the
    conftest's restore reproduce the leak (the control)."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_TESTS = REPO / "paper" / "tests"
CONFTEST = PAPER_TESTS / "conftest.py"
FIXTURE = "notebook_rcparams"
#: the modules whose fixtures run the notebook's cells on the committed frame (a detector that found none of them would be blind)
SLOW_MODULES = {"test_registry_nb_pooled_ci_slow.py", "test_registry_nb_seed_ci_slow.py", "test_appendix_generator_slow.py",
                "test_registry_nb_ctlstop_appendix_slow.py", "test_registry_nb_table_levels_slow.py"}
#: what marks a module as one that reads the notebook's cells (its own path constant, the generator reader or the notebook's name)
NOTEBOOK_MARKS = ("NOTEBOOK", "generator_cell", "_cell_source", "registry_qwen_figures.ipynb", "make_registry_notebook.py")


def _is_fixture(fn: ast.FunctionDef) -> bool:
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
    return False


def _fixture_scope(fn: ast.FunctionDef) -> str:
    for d in fn.decorator_list:
        if isinstance(d, ast.Call):
            for kw in d.keywords:
                if kw.arg == "scope" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
    return "function"


def _fixture_autouse(fn: ast.FunctionDef) -> bool:
    for d in fn.decorator_list:
        if isinstance(d, ast.Call):
            for kw in d.keywords:
                if kw.arg == "autouse" and isinstance(kw.value, ast.Constant):
                    return bool(kw.value.value)
    return False


def _calls(node: ast.AST) -> set:
    return {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}


def _enters_rc_context(fn: ast.FunctionDef) -> bool:
    for w in ast.walk(fn):
        if isinstance(w, ast.With):
            for item in w.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "rc_context":
                    return True
    return False


def notebook_fixtures(path: Path) -> dict:
    """{fixture name: requested fixture names} for every fixture of the module that executes the notebook's cells, plus
    {"_all": {fixture name: requested fixture names}} for every fixture of the module."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    fixtures = {name: [a.arg for a in fn.args.args] for name, fn in funcs.items() if _is_fixture(fn)}
    if not any(m in src for m in NOTEBOOK_MARKS):
        return {"_all": fixtures}
    execs: dict = {}

    def runs_exec(name: str, seen: frozenset = frozenset()) -> bool:
        if name in execs:
            return execs[name]
        fn = funcs[name]
        called = _calls(fn)
        hit = "exec" in called or any(runs_exec(c, seen | {name}) for c in called if c in funcs and c not in seen | {name})
        execs[name] = hit
        return hit
    out = {name: args for name, args in fixtures.items() if runs_exec(name)}
    out["_all"] = fixtures
    return out


def conftest_restores_every_module() -> bool:
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    return any(isinstance(n, ast.FunctionDef) and _is_fixture(n) and _fixture_autouse(n) and _fixture_scope(n) == "module"
               and _enters_rc_context(n) for n in tree.body)


def test_the_conftest_restore_is_a_module_scoped_rc_context():
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FIXTURE), None)
    assert fn is not None and _is_fixture(fn), f"paper/tests/conftest.py defines no fixture {FIXTURE}"
    assert _fixture_scope(fn) == "module" and _enters_rc_context(fn), f"{FIXTURE} must be module-scoped and enter matplotlib.rc_context"


def test_every_fixture_that_runs_the_notebooks_cells_restores_rcparams():
    found, uncovered = {}, []
    every_module = conftest_restores_every_module()
    for path in sorted(PAPER_TESTS.glob("test_*.py")):
        fx = notebook_fixtures(path)
        allfx = fx.pop("_all")

        def covered(name: str, seen: frozenset = frozenset()) -> bool:
            args = allfx.get(name, [])
            return FIXTURE in args or any(covered(a, seen | {name}) for a in args if a in allfx and a not in seen | {name})
        for name in fx:
            found.setdefault(path.name, []).append(name)
            if not (every_module or covered(name)):
                uncovered.append(f"{path.name}::{name}")
    assert SLOW_MODULES <= set(found), f"the detector missed slow modules: {sorted(SLOW_MODULES - set(found))} (found {sorted(found)})"
    assert not uncovered, (f"fixtures that execute the notebook's cells without an rcParams restore (request `{FIXTURE}`, or a fixture "
                           f"that does): {uncovered}")


STYLE_MODULE = """
import matplotlib
import pytest


@pytest.fixture(scope="module")
def nb({args}):
    # what the notebook's header cell does through registry_nb.style(): the publication rcParams, process-wide
    matplotlib.rcParams.update({{"axes.titlelocation": "left", "font.size": 17, "axes.spines.top": False}})
    yield "styled"


def test_the_module_draws_in_the_notebooks_style(nb):
    assert matplotlib.rcParams["axes.titlelocation"] == "left" and matplotlib.rcParams["font.size"] == 17
"""

AFTER_MODULE = """
import matplotlib
import matplotlib.pyplot as plt


def test_the_next_module_starts_from_the_defaults():
    assert matplotlib.rcParams["axes.titlelocation"] == "center", matplotlib.rcParams["axes.titlelocation"]
    assert matplotlib.rcParams["font.size"] == 10.0 and matplotlib.rcParams["axes.spines.top"] is True
    fig, ax = plt.subplots()
    ax.set_title("static")
    assert ax.get_title() == "static"
    plt.close(fig)
"""


def _session(tmp: Path, with_conftest: bool, args: str) -> subprocess.CompletedProcess:
    tmp.mkdir(parents=True, exist_ok=True)
    if with_conftest:
        shutil.copy(CONFTEST, tmp / "conftest.py")
    (tmp / "test_a_slow.py").write_text(textwrap.dedent(STYLE_MODULE.format(args=args)), encoding="utf-8")
    (tmp / "test_b_after.py").write_text(textwrap.dedent(AFTER_MODULE), encoding="utf-8")
    env = {**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp / ".mpl"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly", "-o", "addopts=", "-q",
                           "--rootdir", str(tmp), "test_a_slow.py", "test_b_after.py"], cwd=tmp, env=env, capture_output=True,
                          text=True, timeout=300)


def test_the_restore_leaves_the_next_module_at_the_defaults(tmp_path):
    r = _session(tmp_path / "restored", True, FIXTURE)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "2 passed" in r.stdout, r.stdout[-2000:]


def test_without_a_restore_the_style_leaks_into_the_next_module(tmp_path):
    """The control: the same two modules with no restore anywhere reproduce the failure (so the test above can fail)."""
    r = _session(tmp_path / "leaky", False, "")
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-2000:]
    assert "1 failed, 1 passed" in r.stdout and "test_the_next_module_starts_from_the_defaults" in r.stdout, r.stdout[-2000:]
