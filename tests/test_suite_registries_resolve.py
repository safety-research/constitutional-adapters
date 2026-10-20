"""Every suite a dispatcher table registers resolves to an importable module and a callable entry
point: ``sisteer.eval.suites.SUITES`` (the stage-2 dispatcher) and
``sisteer.eval.realign_suites.REALIGN_SUITES`` (the realign campaign's registry).

Both tables import lazily, so a registered name whose module is not on the tree used to fail only
at the ``get_runner`` call inside a running job (and in the lazy ``from .eval.xstest import …`` of
``sisteer.agentic_control`` / ``scripts/agentic_control_analysis.py``). Here it fails in CI, one
parametrized case per registered suite.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

from sisteer.eval import realign_suites, suites

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

TABLES = {
    "suites": (suites.SUITES, suites.get_runner),
    "realign_suites": (realign_suites.REALIGN_SUITES, realign_suites.get_runner),
}


def _entries():
    for table, (registry, _) in TABLES.items():
        for name, (module, attr) in registry.items():
            yield pytest.param(table, name, module, attr, id=f"{table}:{name}")


@pytest.mark.parametrize("table,name,module,attr", list(_entries()))
def test_registered_suite_module_imports_and_its_entry_point_is_callable(table, name, module, attr):
    assert module.startswith("sisteer.eval."), (table, name, module)
    mod = importlib.import_module(module)  # ModuleNotFoundError here = a registered-but-missing suite
    fn = getattr(mod, attr)                # AttributeError here = a renamed entry point
    assert callable(fn)
    assert TABLES[table][1](name) is fn


def test_realign_table_registers_the_three_judged_suites_by_module_path():
    assert set(realign_suites.REALIGN_SUITES) == {"agentic", "mask", "xstest"}
    assert realign_suites.REALIGN_SUITES["xstest"] == ("sisteer.eval.xstest", "run_xstest_arm")
    assert realign_suites.REALIGN_SUITES["mask"] == ("sisteer.eval.mask", "run_mask_arm")


def test_realign_arm_runners_share_one_signature():
    """Every free-form arm runner takes ``(model, tok, items, *, system, obj, scale, out_jsonl,
    max_new_tokens, batch_size)``: the realign driver calls them interchangeably by suite name."""
    for name in realign_suites.REALIGN_SUITES:
        sig = inspect.signature(realign_suites.get_runner(name))
        positional = [p.name for p in sig.parameters.values()
                      if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
        keyword_only = {p.name for p in sig.parameters.values()
                        if p.kind is inspect.Parameter.KEYWORD_ONLY}
        assert positional[:2] == ["model", "tok"] and len(positional) == 3, (name, positional)
        assert {"system", "obj", "scale", "out_jsonl", "max_new_tokens", "batch_size"} <= keyword_only, name


def test_a_registered_but_missing_module_fails_at_resolution_not_silently(monkeypatch):
    monkeypatch.setitem(realign_suites.REALIGN_SUITES, "ghost",
                        ("sisteer.eval.module_that_does_not_exist", "run_ghost_arm"))
    with pytest.raises(ModuleNotFoundError):
        realign_suites.get_runner("ghost")
    with pytest.raises(KeyError, match="unknown realign suite"):
        realign_suites.get_runner("unregistered")
    with pytest.raises(KeyError, match="unknown suite"):
        suites.get_runner("unregistered")


def test_the_lazy_xstest_importers_on_the_tree_resolve():
    """The two call sites that import the xstest loader lazily (outside any registry table)."""
    from sisteer import agentic_control
    from sisteer.eval import xstest

    assert callable(xstest.load_xstest)
    src = inspect.getsource(agentic_control.default_eval_texts)
    assert "from .eval.xstest import load_xstest" in src

    import agentic_control_analysis as aca

    assert set(aca.SUITE_ITEMS) == set(realign_suites.REALIGN_SUITES)
    assert "from sisteer.eval.xstest import load_xstest" in inspect.getsource(aca._expected_xstest_items)
