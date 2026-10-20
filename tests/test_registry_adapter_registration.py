"""Every adapter module under ``adapters/`` is loaded by the runner (the incremental adapter pass covers
every legacy source). The package is enumerated ON DISK and each module classified by what it DEFINES — an adapter
class (the ``Adapter`` protocol's four methods plus a string ``name``) or a module-level ``register()`` of its own —
never by a hardcoded roster of helpers, so an adapter landing later (``registry-seeds``, ``registry-adapters-records``)
under either registration convention is covered or the test fails. Synthetic modules pin both conventions the loader
must honour, and a known-bad one (an adapter class registered by nothing) proves ``load_all`` refuses instead of
skipping it: PR-3b's five and PR-3d's four adapters were skipped exactly that way while a module list drove the pass.
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

from sisteer.registry import adapters as ADP
from sisteer.registry.adapters import base as AB

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG_DIR = os.path.join(ROOT, "src", "sisteer", "registry", "adapters")
PKG = "sisteer.registry.adapters"


def _modules_on_disk():
    return sorted(fn[:-3] for fn in os.listdir(PKG_DIR) if fn.endswith(".py") and fn != "__init__.py")


# --------------------------------------------------------------------------------------------
# the real package
# --------------------------------------------------------------------------------------------

def test_enumeration_is_the_package_on_disk():
    assert ADP.adapter_modules() == tuple(_modules_on_disk())
    assert not hasattr(ADP, "ADAPTER_MODULES"), "a module list is a roster; the loader enumerates the package instead"


def test_every_adapter_defining_module_is_loaded_by_the_runner():
    reg = ADP.load_all()
    adapter_modules, helpers = [], []
    for n in _modules_on_disk():
        mod = importlib.import_module(f"{PKG}.{n}")
        classes = ADP.adapter_classes(mod)
        if not classes and ADP.own_register(mod) is None:
            helpers.append(n)
            continue
        adapter_modules.append(n)
        assert classes, f"{n} has its own register() but defines no adapter class"
        for cls in classes:
            assert cls.name in reg, f"{n}.{cls.__name__} (name={cls.name!r}) is not registered: the pass would skip it"
            inst = reg[cls.name]()
            assert isinstance(inst, cls) and inst.name == cls.name, (n, cls.__name__)
    # the registry holds exactly the adapter modules' classes: nothing from a helper, no orphan factory
    registered_from = {type(f()).__module__.rsplit(".", 1)[1] for f in reg.values()}
    assert registered_from == set(adapter_modules), (sorted(registered_from), adapter_modules)
    assert adapter_modules and helpers, (adapter_modules, helpers)      # both partitions are non-empty (base / runner are helpers)
    for n in helpers:
        assert n not in reg


def test_helper_modules_define_no_adapter_class():
    """The classifier is structural, but the contract (``base.py``) and the pass (``runner.py``) must read as helpers:
    the protocol class itself and the runner's dataclasses are not adapters."""
    for n in ("base", "runner"):
        mod = importlib.import_module(f"{PKG}.{n}")
        assert ADP.adapter_classes(mod) == [] and ADP.own_register(mod) is None, n


# --------------------------------------------------------------------------------------------
# synthetic modules: the two conventions and the known-bad fixture
# --------------------------------------------------------------------------------------------

@pytest.fixture
def scratch_registry(monkeypatch):
    saved = dict(AB._REGISTRY)
    yield
    AB._REGISTRY.clear()
    AB._REGISTRY.update(saved)


def _synthetic_module(monkeypatch, short_name: str, adapter_name: str):
    mod = types.ModuleType(f"{PKG}.{short_name}")

    class SyntheticAdapter:
        name = adapter_name
        covers_sources = ()
        roots = ()
        rng_scope = "none"
        allow_symlinks = False

        def discover(self):
            return []

        def comparison_group(self, inp):
            return "g"

        def read(self, ctx, group, inputs):
            return iter(())

        def source_states(self, ctx, inputs):
            return []

    SyntheticAdapter.__module__ = mod.__name__
    mod.SyntheticAdapter = SyntheticAdapter
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return mod, SyntheticAdapter


def test_import_time_registration_convention_is_loaded(monkeypatch, scratch_registry):
    """PR-3a style: ``from .base import register`` + ``register(NAME, factory)`` at import; the re-exported
    ``base.register`` is NOT mistaken for the module's own zero-argument ``register()``."""
    mod, cls = _synthetic_module(monkeypatch, "_synthetic_import_time", "synthetic_import_time")
    mod.register = AB.register
    AB.register(cls.name, cls)                         # what the import already did
    assert ADP.own_register(mod) is None
    reg = ADP.load_all(names=("_synthetic_import_time",))
    assert isinstance(reg["synthetic_import_time"](), cls)


def test_own_register_convention_is_called_by_load_all(monkeypatch, scratch_registry):
    """PR-3b / PR-3d style: a module-level ``def register() -> None`` the loader must CALL (nothing else does)."""
    mod, cls = _synthetic_module(monkeypatch, "_synthetic_own_register", "synthetic_own_register")
    calls = []

    def register() -> None:
        calls.append(1)
        AB.register(cls.name, cls)

    register.__module__ = mod.__name__
    mod.register = register
    assert ADP.own_register(mod) is register
    assert "synthetic_own_register" not in AB.registered()
    reg = ADP.load_all(names=("_synthetic_own_register",))
    assert calls == [1] and isinstance(reg["synthetic_own_register"](), cls)


def test_known_bad_unregistered_adapter_class_refuses(monkeypatch, scratch_registry):
    """The failure this guards against: a module defines an adapter and no convention registers it. The loader
    refuses and names the class — a silent skip is what left nine adapters out of the pass."""
    mod, cls = _synthetic_module(monkeypatch, "_synthetic_unregistered", "synthetic_unregistered")
    assert ADP.adapter_classes(mod) == [cls]
    with pytest.raises(ADP.UnregisteredAdapter) as exc:
        ADP.load_all(names=("_synthetic_unregistered",))
    assert "SyntheticAdapter" in str(exc.value) and "synthetic_unregistered" in str(exc.value)
    assert "synthetic_unregistered" not in AB.registered()


def test_classifier_reads_only_what_the_module_defines(monkeypatch):
    """An imported adapter class (``from .cellstore import CellStoreAdapter`` in a helper) does not make the helper an
    adapter module, and a class missing any protocol method or a string ``name`` is not an adapter."""
    from sisteer.registry.adapters import cellstore as CS
    mod = types.ModuleType(f"{PKG}._synthetic_helper")
    mod.CellStoreAdapter = CS.CellStoreAdapter            # imported, defined elsewhere

    class Partial:                                         # a name and three of the four methods
        name = "partial"

        def discover(self):
            return []

        def comparison_group(self, inp):
            return "g"

        def read(self, ctx, group, inputs):
            return iter(())

    class Nameless:                                        # every method, no name
        def discover(self):
            return []

        def comparison_group(self, inp):
            return "g"

        def read(self, ctx, group, inputs):
            return iter(())

        def source_states(self, ctx, inputs):
            return []

    Partial.__module__ = Nameless.__module__ = mod.__name__
    mod.Partial, mod.Nameless = Partial, Nameless
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    assert ADP.adapter_classes(mod) == [] and ADP.own_register(mod) is None
    assert ADP.load_all(names=("_synthetic_helper",)) is not None      # a helper loads without complaint
