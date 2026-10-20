"""Ingest adapters: one module per source group; this package holds the shared contract
(``base.py``) and the recurring incremental pass (``runner.py``). :func:`load_all` imports EVERY module of the
package found on disk and returns ``registered()`` complete (the CLI's ``adapters run`` and ``backfill`` call
it). Two registration conventions coexist and both are honoured:

- PR-3a: the module calls ``base.register(NAME, factory)`` at import;
- PR-3b / PR-3d: the module defines its own zero-argument ``register()``, which ``load_all`` calls after the import.

There is deliberately NO module list to append to (the incremental pass covers every legacy source):
a list was how PR-3b's five and PR-3d's four adapters came to be skipped by ``adapters run`` and ``backfill``
while only the smoke scripts drove them. A module that defines an adapter class (the :class:`Adapter` protocol's
methods plus a ``name``) and registers none of them makes ``load_all`` refuse — a visible error, never a
silently absent source (``tests/test_registry_adapter_registration.py`` pins both halves).

Every factory :func:`load_all` (and :func:`registered`) returns is a :func:`base.guarded_factory` (spec S2): the
instance it constructs refuses its first ``discover()`` / ``read()`` with :class:`base.GradingHostStoreReadRefused` unless a
:class:`base.StoreAccessScope` is installed -- the cluster shard installs one, the operator's ``REGISTRY_STORE_READ_OVERRIDE=<reason>``
is the one other way. Construction itself is never refused."""
from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple, Type

from .base import (Adapter, AdapterContext, AdapterRefusal, GradingHostStoreReadRefused, Input, InputRefusal, RecordBatch, SourceStateRow,
                   StoreAccessScope, check_input, guard_adapter, guarded_factory, register, registered, require_store_access,
                   store_access_scope, using_store_access_scope, watermark_of, without_store_access_scope)

__all__ = ["Adapter", "AdapterContext", "AdapterRefusal", "GradingHostStoreReadRefused", "Input", "InputRefusal", "RecordBatch", "SourceStateRow",
           "StoreAccessScope", "check_input", "guard_adapter", "guarded_factory", "register", "registered", "require_store_access",
           "store_access_scope", "using_store_access_scope", "watermark_of", "without_store_access_scope",
           "adapter_modules", "adapter_classes", "own_register", "load_all", "UnregisteredAdapter"]

#: the :class:`Adapter` protocol's members; a class defined in a module that has them all and a string ``name`` IS an adapter
ADAPTER_PROTOCOL_METHODS: Tuple[str, ...] = ("discover", "comparison_group", "read", "source_states")


class UnregisteredAdapter(RuntimeError):
    """A module defines an adapter class that no registration convention exposed to the runner."""


def adapter_modules() -> Tuple[str, ...]:
    """Every module of this package on disk (helpers included), in name order — the enumeration, not a roster."""
    return tuple(sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.ispkg))


def adapter_classes(mod: ModuleType) -> List[Type]:
    """The adapter classes a module DEFINES (not the ones it imports): every class whose ``__module__`` is the module,
    with a string ``name`` and all of :data:`ADAPTER_PROTOCOL_METHODS` callable."""
    out: List[Type] = []
    for _, cls in inspect.getmembers(mod, inspect.isclass):
        if getattr(cls, "__module__", None) != mod.__name__:
            continue
        if not isinstance(getattr(cls, "name", None), str):
            continue
        if all(callable(getattr(cls, m, None)) for m in ADAPTER_PROTOCOL_METHODS):
            out.append(cls)
    return out


def own_register(mod: ModuleType):
    """The module's OWN zero-argument ``register()`` (PR-3b / PR-3d convention), or None. PR-3a modules re-export
    ``base.register`` (two required parameters, defined elsewhere) and register at import instead."""
    fn = getattr(mod, "register", None)
    if not callable(fn) or getattr(fn, "__module__", None) != mod.__name__:
        return None
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return None
    return fn


def load_all(names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Import every module of the package (``names`` narrows it, for tests), run each module's own ``register()``
    where it has one, and return ``registered()`` -- guarded factories (:func:`base.guarded_factory`). Refuses with
    :class:`UnregisteredAdapter` when a module defines an adapter class whose ``name`` is registered by nothing."""
    missing: List[str] = []
    for n in (tuple(names) if names is not None else adapter_modules()):
        mod = importlib.import_module(f"{__name__}.{n}")
        fn = own_register(mod)
        if fn is not None:
            fn()
        reg = registered()
        for cls in adapter_classes(mod):
            if cls.name not in reg:
                missing.append(f"{mod.__name__}.{cls.__name__} (name={cls.name!r})")
    if missing:
        raise UnregisteredAdapter("adapter class(es) defined but registered by no convention — the incremental pass would "
                                  "silently skip their sources (plan §2.8): " + ", ".join(missing))
    return registered()
