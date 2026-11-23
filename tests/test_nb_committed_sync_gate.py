"""The committed paper notebook is its generator's, executed, as the landing gate sees it.

paper/tests/test_registry_nb_committed_sync.py holds paper/notebooks/registry_qwen_figures.ipynb to make_registry_notebook.py: the
same cell ids in the same order, the same types and sources, and every code cell run once, in order, with no error. It sits outside
the suite's `testpaths`, and both landing gates deal their shards from tests/test_*.py, so the gate runs it only when a landing names
it, and a generator change landed without re-rendering the notebook would pass a green gate. This file sits in tests/ and runs that module's own checks and fixtures,
loaded from its file rather than copied, so the two cannot drift apart. In the suite's environment (no `nbformat`) the paper
module builds the generator's cells through its stand-in; its nbformat parity test is not bound here, because it would always skip.

The basename differs from the paper module's on purpose: with no __init__.py, two test modules named alike collide when one run
collects both directories. It does not start with `test_registry_`, the prefix the registry-scoped scans of tests/ read.
To fix a failure, rebuild and execute the notebook as paper/notebooks/README.md says ("Run it")."""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_MODULE = REPO / "paper" / "tests" / "test_registry_nb_committed_sync.py"
#: the paper module's checks, run here; its last test compares the stand-in with real nbformat and needs the notebook group
CHECKS = ("test_the_committed_notebook_holds_the_generators_cell_ids_in_order",
          "test_every_committed_cell_has_the_generators_type_and_source",
          "test_the_committed_notebook_is_executed_once_in_order_without_an_error")
NOT_BOUND = ("test_the_stand_in_builds_the_cells_nbformat_builds",)


def _paper_module():
    """The paper module, executed under a private name (the name pytest gives it, were both directories collected, stays free)."""
    spec = importlib.util.spec_from_file_location("paper_tests_registry_nb_committed_sync", PAPER_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = _paper_module()
# the module-scoped fixtures the checks take: the generator's cells (its write captured, nothing written) and the committed file
generated = P.generated
committed = P.committed
test_the_committed_notebook_holds_the_generators_cell_ids_in_order = P.test_the_committed_notebook_holds_the_generators_cell_ids_in_order
test_every_committed_cell_has_the_generators_type_and_source = P.test_every_committed_cell_has_the_generators_type_and_source
test_the_committed_notebook_is_executed_once_in_order_without_an_error = P.test_the_committed_notebook_is_executed_once_in_order_without_an_error


def test_every_check_of_the_paper_module_runs_here():
    """A check added to the paper module must be bound here too (or named in NOT_BOUND), or the gate would not run it."""
    paper_tests = {name for name, obj in vars(P).items() if name.startswith("test_") and callable(obj)}
    assert paper_tests == set(CHECKS) | set(NOT_BOUND), sorted(paper_tests ^ (set(CHECKS) | set(NOT_BOUND)))
    assert all(globals()[name] is getattr(P, name) for name in CHECKS)
