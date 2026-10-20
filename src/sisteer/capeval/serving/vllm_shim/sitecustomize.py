"""The protocol sampling clamp-override SHIM for EVERY interpreter vLLM spawns (bcb-bfcl-200-widen,
the planned fallback).

vLLM 0.25.1 forces the ``spawn`` start method for some families (the Qwen3.5 API server initialises
CUDA before the engine fork: "We must use the `spawn` multiprocessing start method … Reasons: CUDA is
initialized"), and a spawned child is a FRESH interpreter that imports the wheel's
``vllm.sampling_params`` with its ``_MAX_TEMP = 1e-2`` — so an override applied in the server process
alone (``sisteer.capeval.serving.vllm_entry``) would not reach it and a BFCL request at 0.001 would
re-clamp to 0.01 on msgspec decode. This module is a ``sitecustomize``: the interpreter's ``site``
imports it at startup from the FIRST ``sitecustomize`` on ``sys.path`` (or a ``.pth`` boot exec's it
by absolute path, below) — the standup (``vllm_server.apply_sampling_clamp_override``) delivers it
through the serving interpreter, whose environment and site-packages are inherited by
``multiprocessing`` spawn children and by vLLM's own ``get_mp_context("spawn")`` API-server
manager, so every child re-runs it.

ARMED ONLY by the environment: ``SISTEER_VLLM_MAX_TEMP=<T>`` (set by the standup beside the flag).
Unarmed, this module does nothing at all. Armed, it installs a post-import hook that sets
``vllm.sampling_params._MAX_TEMP`` to ``<T>`` the moment that module is executed (or at once, if it
is already imported), recording the wheel's original value as ``_SISTEER_MAX_TEMP_FROM`` so
``vllm_entry.apply_override`` recognises an override the shim already applied. Stdlib only — this
runs in interpreters that never import ``sisteer``.

Two delivery routes exist — a ``.pth`` boot in the serving interpreter's site-packages (preferred,
round 8) or this directory on PYTHONPATH as ``sitecustomize`` — and the standup installs exactly ONE
(round 9). Should both ever be live at once, this module is exec'd twice as two module copies: the
finder de-dup keys on ``MARKER_ATTR`` (never class identity) so one finder registers, and a finder
never delegates to another marked finder, so there is no recursion either way.
"""
import importlib.abc
import importlib.machinery
import os
import sys

SHIM_ENV = "SISTEER_VLLM_MAX_TEMP"
#: the identity every copy of this module carries (the mounted /root copy, the staged tree's, a .pth
#: boot) — `vllm_entry.shim_importable` recognises the shim by THIS line, never by its path
SHIM_SIGNATURE = "sisteer-vllm-clamp-override-shim/1"
#: the STABLE marker every clamp finder carries on the instance: two delivery
#: routes in one interpreter (a `.pth` boot AND a sitecustomize import) exec this module TWICE, as
#: two module copies with two distinct `_ClampShimFinder` classes — an `isinstance` de-dup missed
#: the other copy, both registered, and each delegated to the other -> RecursionError on
#: `import vllm.sampling_params`. De-dup and delegation now key on this attribute, never on class identity.
MARKER_ATTR = "__sisteer_clamp_shim__"
TARGET = "vllm.sampling_params"
FROM_ATTR = "_SISTEER_MAX_TEMP_FROM"
APPLIED_ATTR = "_SISTEER_MAX_TEMP_SHIM"


def _patch(module, value: float) -> None:
    if getattr(module, APPLIED_ATTR, None) == value:
        return
    setattr(module, FROM_ATTR, getattr(module, "_MAX_TEMP", None))
    module._MAX_TEMP = float(value)
    setattr(module, APPLIED_ATTR, float(value))


def is_clamp_finder(obj) -> bool:
    """Any copy's clamp finder — recognised by :data:`MARKER_ATTR`, never by class."""
    return getattr(obj, MARKER_ATTR, None) == SHIM_SIGNATURE


class _ClampShimFinder(importlib.abc.MetaPathFinder):
    """A meta-path finder that answers ONLY for ``vllm.sampling_params``: it defers to the
    remaining NON-clamp finders for the real spec and wraps its loader so the patch runs right
    after the module body executes. It never delegates to another clamp finder (another module
    copy's) — that pair recursed."""

    def __init__(self, value: float) -> None:
        self.value = float(value)
        setattr(self, MARKER_ATTR, SHIM_SIGNATURE)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        for finder in sys.meta_path:
            if finder is self or is_clamp_finder(finder) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PatchingLoader(spec.loader, self.value)
                return spec
        return None


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner, value: float) -> None:
        self._inner, self.value = inner, float(value)

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch(module, self.value)

    def __getattr__(self, name):            # get_code / get_source / is_package … delegate
        return getattr(self._inner, name)


def install(environ=None) -> bool:
    """Arm the hook iff ``SISTEER_VLLM_MAX_TEMP`` is set to a float; returns whether it armed."""
    environ = os.environ if environ is None else environ
    raw = environ.get(SHIM_ENV)
    if not raw:
        return False
    try:
        value = float(raw)
    except ValueError:
        return False
    if not any(is_clamp_finder(f) for f in sys.meta_path):          # exactly ONE clamp finder per interpreter, any copy
        sys.meta_path.insert(0, _ClampShimFinder(value))
    already = sys.modules.get(TARGET)
    if already is not None:
        _patch(already, value)
    return True


ARMED = install()
