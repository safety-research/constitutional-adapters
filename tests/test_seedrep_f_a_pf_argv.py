"""Qwen seed-replicate native battery — launch_report F-A (the `--pf` preflight): the exact `arm_cell_check.py --resolve` argv
`scripts/modal/seedrep_served.py` builds for the Arditi arm is accepted by `arm_cell_check`'s REAL parser + `resolve()` on the real
tree (rc 0), with `--view` = the arms table's PAPER view for the target (`targets.qwen35_4b.view`), read from the table, never a literal.

Known-bad, kept: the literal the app used to pass (`lp.ARDITI_VIEW` = `ar_c0p3`, the Arditi RUNG whose `cells_view/` directory the
objects live under) is refused by `resolve()` deterministically ("not the PAPER view"), rc 1 — so `--pf` refuses it.
"""
from __future__ import annotations

import importlib.util
import io
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_launch_plan as lp  # noqa: E402

DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"
CHECKER = REPO / "scripts" / "sdfbase" / "full22" / "arm_cell_check.py"


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda f: f

    def local_entrypoint(self, **kw):
        return lambda f: f


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.is_local = lambda: False
    m.current_function_call_id = lambda: "fc-test"
    return m


_ABSENT = object()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_fa_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("modal", _ABSENT)
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(mod)
    finally:
        if prior is _ABSENT:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    return mod


@pytest.fixture(scope="module")
def acc():
    return _load("arm_cell_check_under_test", CHECKER)


def _run_checker(acc, argv_after_python_and_script):
    """The checker's REAL `main` on the argv (in-process): (rc, stdout, stderr); a `SystemExit` from `resolve()` is its rc."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = acc.main(list(argv_after_python_and_script))
        except SystemExit as e:
            rc = int(e.code or 0)
    return rc, out.getvalue(), err.getvalue()


def test_the_preflight_argv_names_the_arms_tables_paper_view_and_the_real_checker_accepts_it(srv, acc):
    argv = srv.arditi_resolve_argv(py=sys.executable, tree=REPO)
    arms = yaml.safe_load((REPO / lp.ARMS_REL).read_text())
    paper = arms["targets"][lp.MODEL_KEY]["view"]
    assert argv[0] == sys.executable and argv[1] == str(CHECKER)
    assert argv[argv.index("--view") + 1] == paper == srv.arms_paper_view(REPO)
    assert argv[argv.index("--target") + 1] == lp.MODEL_KEY and argv[argv.index("--arm") + 1] == "arditi_allpos" and "--resolve" in argv
    assert argv[argv.index("--defenses-dir") + 1].endswith(f"/cells_view/{lp.ARDITI_VIEW}")      # the RUNG names the bank DIRECTORY, not the view
    assert argv[argv.index("--pins") + 1] == str(REPO / lp.ARDITI_PINS_REL) and argv[argv.index("--arms") + 1] == str(REPO / lp.ARMS_REL)
    rc, out, err = _run_checker(acc, argv[2:])
    assert rc == 0, err
    cell, kind, rel, policy = out.split()[:4]
    assert cell == "arditi_allpos" and kind == "vector_all" and rel.endswith(f"/cells_view/{lp.ARDITI_VIEW}") and policy == "lora_native"


def test_the_literal_rung_as_view_is_refused_by_the_real_resolver_known_bad(srv, acc):
    """launch_report F-A, reproduced: `--view ar_c0p3` (lp.ARDITI_VIEW) -> `resolve()` refuses 'not the PAPER view', rc 1."""
    argv = srv.arditi_resolve_argv(py=sys.executable, tree=REPO)
    old = list(argv[2:])
    old[old.index("--view") + 1] = lp.ARDITI_VIEW
    assert lp.ARDITI_VIEW != srv.arms_paper_view(REPO)
    rc, _out, err = _run_checker(acc, old)
    assert rc == 1 and "PAPER view" in err and lp.ARDITI_VIEW in err


def test_the_paper_view_is_read_from_the_table_never_a_literal(srv, tmp_path):
    """A tree whose arms table names another paper view drives the argv; a table without the row refuses by name."""
    tree = tmp_path / "tree"
    (tree / lp.ARMS_REL).parent.mkdir(parents=True)
    (tree / lp.ARMS_REL).write_text(yaml.safe_dump({"targets": {lp.MODEL_KEY: {"view": "paperview_x"}}}))
    argv = srv.arditi_resolve_argv(py="py", tree=tree)
    assert argv[argv.index("--view") + 1] == "paperview_x"
    (tree / lp.ARMS_REL).write_text(yaml.safe_dump({"targets": {"other_target": {"view": "it"}}}))
    with pytest.raises(ValueError, match=f"targets.{lp.MODEL_KEY}.view"):
        srv.arditi_resolve_argv(py="py", tree=tree)
    src = DRIVER.read_text()
    assert '"--view", lp.ARDITI_VIEW' not in src                 # the literal is gone from the preflight
