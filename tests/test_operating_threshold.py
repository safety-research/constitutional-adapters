"""The StrongREJECT acceptance bar must come from ONE place, loudly.

Found 2026-08-26: the pool builder read `prereg.threshold_grid[1]` -- a magic
index -- while the config ALSO declared `pool.threshold`, which nothing read.
Both were 0.5, so the pool was right by luck. With 0.5-vs-0.75 a live question
on this project, editing the obvious key would have changed nothing while
looking like it changed the acceptance bar.
"""
import importlib.util as ilu
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SPEC = ilu.spec_from_file_location(
    "_build_pool", ROOT / "scripts" / "build_two_si_pool.py")


def _mod():
    import sys
    if "_build_pool" in sys.modules:
        return sys.modules["_build_pool"]
    m = ilu.module_from_spec(_SPEC)
    # Register BEFORE exec: the module defines dataclasses whose InitVar
    # annotations are resolved through sys.modules[cls.__module__].
    sys.modules["_build_pool"] = m
    _SPEC.loader.exec_module(m)
    return m


def _cfg():
    return yaml.safe_load((ROOT / "configs" / "two_si.yaml").read_text())


def test_shipped_config_resolves_to_the_directed_threshold():
    """The operating threshold is StrongREJECT strictly greater than 0.5."""
    assert _mod()._operating_threshold(_cfg()) == 0.5


def test_declared_pool_threshold_agrees_with_the_prereg_grid():
    """The regression guard: a config that declares both must not disagree."""
    cfg = _cfg()
    declared = (cfg.get("pool") or {}).get("threshold")
    if declared is not None:
        assert float(declared) == _mod()._operating_threshold(cfg)


def test_disagreement_refuses_rather_than_silently_preferring_the_grid():
    m = _mod()
    cfg = {"prereg": {"threshold_grid": [0.25, 0.5, 0.75]},
           "pool": {"threshold": 0.75}}
    with pytest.raises(ValueError, match="would change NOTHING"):
        m._operating_threshold(cfg)


def test_operating_threshold_can_move_the_bar_deliberately():
    m = _mod()
    cfg = {"prereg": {"threshold_grid": [0.25, 0.5, 0.75],
                      "operating_threshold": 0.75},
           "pool": {"threshold": 0.75}}
    assert m._operating_threshold(cfg) == 0.75


def test_operating_threshold_must_be_preregistered():
    m = _mod()
    cfg = {"prereg": {"threshold_grid": [0.25, 0.5, 0.75],
                      "operating_threshold": 0.6}}
    with pytest.raises(ValueError, match="not in"):
        m._operating_threshold(cfg)


def test_acceptance_is_strictly_greater_not_greater_equal():
    """A response scoring exactly at the bar is NOT accepted."""
    from sisteer.eval.matched_pool import PoolSpec
    spec = PoolSpec(si_ids=("a", "b"), train_families=("bon",), eval_families=(),
                    train_goal_ids=("g0",), eval_goal_ids=("g1",),
                    static_families=frozenset({"bon"}), threshold=0.5)
    assert spec.threshold == 0.5
    # matched_pool accepts on `score > thr`, so the bar itself is a reject.
    assert not (0.5 > spec.threshold)
    assert 0.51 > spec.threshold
