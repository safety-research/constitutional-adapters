"""The paper tests' shared fixtures: matplotlib's rcParams are restored at the end of every test module, and synthetic frames read no
registry's item rows.

RCPARAMS. The slow modules (the table-levels, control-pool, distillation-target and generator-comparison ones) execute the notebook's
header cell, which calls `registry_nb.style()` and sets the publication rcParams for the whole process, `axes.titlelocation = "left"`
among them. Without a restore, every module that runs after one of them in the same process draws under those rcParams: a panel title
then sits on the left, and `ax.get_title()` (the centre title) reads "". That happened when the table-levels slow module ran before the
distillation-target tests, and `SISTEER_SLOW_TESTS=1 pytest paper/tests` collects the generator-comparison slow module before them. An
autouse fixture is set up before a module's other fixtures of the same scope, so each module starts from the rcParams the session had
before it and ends by restoring them.

ITEM ROWS. `registry_nb` builds a pooled cell's item-clustered interval from each run's item rows (`item_units`, which reads the source:
the live registry where one exists, else the committed snapshot). A synthetic test frame names runs no registry holds, and a test must
never read a real registry's items, so every test here runs with the provider that has no item source (`registry_nb.no_item_rows`:
pooled cells take the marked identical-draws bound): installed once for the session, before any module-scoped fixture anchors a frame,
and again before each test (never undone, so the next module's fixtures see it too), unless the module sets `USES_ITEM_ROWS = True`
(the slow tests on the committed paper frame, which import a fresh `registry_nb` and read its items) or installs a fixture provider of
its own (`registry_nb.reset_pooled_ci(provider)`)."""
import sys

import matplotlib
import pytest


@pytest.fixture(autouse=True, scope="module")
def _rcparams_restored_per_module():
    with matplotlib.rc_context():
        yield


def _stub(keep_rows: bool) -> None:
    R = sys.modules.get("registry_nb")
    if R is not None:
        R.reset_pooled_ci(R.no_item_rows, keep_rows=keep_rows)


@pytest.fixture(scope="session", autouse=True)
def _pooled_ci_no_item_source_for_the_session():
    _stub(keep_rows=False)
    yield


@pytest.fixture(autouse=True)
def _pooled_ci_item_source(request):
    if not getattr(request.module, "USES_ITEM_ROWS", False):
        _stub(keep_rows=True)             # a module-scoped frame anchored before this test still pools
    yield


@pytest.fixture(scope="module")
def notebook_rcparams():
    """matplotlib's rcParams as they were before the module, put back when it ends: the slow fixtures that execute the notebook's cells
    request it, beside the module-wide restore above (tests/test_paper_tests_restore_rcparams.py holds that every such fixture is
    covered by one of the two). The header cell calls `registry_nb.style()`, which sets the
    publication rcParams for the whole process (`axes.titlelocation = "left"` among them); without the restore every module after a
    slow one draws under them, where a panel title sits on the left and `ax.get_title()` (the centre title) reads "" (the
    control-pool appendix's comparison-grid test failed so whenever the slow modules ran before it in one session). The fixture is
    set up before the requesting fixture and torn down after it, so the module's own tests still draw in the notebook's style."""
    import matplotlib
    with matplotlib.rc_context():
        yield
