"""Launch F-G: every function `scripts/modal/seedrep_served.py` invokes remotely (`X.remote(...)`, `X.spawn(...)`, `X.map(...)`) must
BE a Modal-registered function, and a plain helper must not be. An earlier commit inserted `preflight_objects_check` between the
`@app.function(...)` decorator and `def preflight`, so the decorator registered the helper and `preflight.remote` raised
`AttributeError` at `--pf` before any check ran. The lane's other tests stub `App.function` as identity, which
cannot see this; this stub returns a WRAPPER that only registered functions carry, and the AST scan pins every remote call site."""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"


class _Registered:
    """What `@app.function` hands back: the wrapped callable plus the remote entry points."""

    def __init__(self, fn):
        self.__wrapped__ = fn
        self.__name__ = fn.__name__

    def __call__(self, *a, **k):
        return self.__wrapped__(*a, **k)

    def remote(self, *a, **k):  # pragma: no cover - never invoked here
        raise RuntimeError("remote() is not exercised in this test")

    def spawn(self, *a, **k):  # pragma: no cover
        raise RuntimeError("spawn() is not exercised in this test")

    def map(self, *a, **k):  # pragma: no cover
        raise RuntimeError("map() is not exercised in this test")


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda fn: _Registered(fn)

    def local_entrypoint(self, *a, **k):
        return lambda fn: fn


class _Chain:
    def __getattr__(self, name):
        return lambda *a, **k: self


def _stub_modal():
    m = types.ModuleType("modal")
    m.App = _App
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain(), debian_slim=lambda *a, **k: _Chain())
    m.Dict = types.SimpleNamespace(from_name=lambda *a, **k: {})
    m.Secret = types.SimpleNamespace(from_dict=lambda *a, **k: object(), from_name=lambda *a, **k: object())
    m.is_local = lambda: True
    m.current_function_call_id = lambda: "fc-test"
    return m


_ABSENT = object()


@pytest.fixture(scope="module")
def srv():
    spec = importlib.util.spec_from_file_location("seedrep_served_w8_registration", DRIVER)
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


def _remote_call_targets(src: str) -> dict:
    """Every NAME in `NAME.remote(...)` / `NAME.spawn(...)` / `NAME.map(...)`, mapped to the module-level REGISTRY dict it is bound
    from when NAME is a local alias (`fn = FUNCTIONS[...]` -> "FUNCTIONS"), else None (a module-level function name)."""
    tree = ast.parse(src)
    bound_from: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = node.value
            if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name):
                bound_from.setdefault(node.targets[0].id, set()).add(v.value.id)
    out: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("remote", "spawn", "map"):
            if isinstance(node.func.value, ast.Name):
                name = node.func.value.id
                out[name] = bound_from.get(name)
    return out


def test_preflight_is_registered_and_its_object_helper_is_a_plain_function(srv):
    assert isinstance(srv.preflight, _Registered), "preflight lost its @app.function registration (W7 F-G)"
    assert callable(srv.preflight.remote)
    assert not isinstance(srv.preflight_objects_check, _Registered), "the object-check helper must stay a plain function"


def test_every_remote_call_site_names_a_registered_function(srv):
    names = _remote_call_targets(DRIVER.read_text())
    assert names, "the module invokes nothing remotely?"
    missing = []
    for n, registries in names.items():
        if isinstance(getattr(srv, n, None), _Registered):
            continue
        if registries:  # a local alias bound from a module-level registry dict: every value it can take must be registered
            for reg in sorted(registries):
                table = getattr(srv, reg, None)
                if not isinstance(table, dict) or not table or not all(isinstance(v, _Registered) for v in table.values()):
                    missing.append(f"{n} <- {reg}")
            continue
        missing.append(n)
    assert not missing, f"invoked with .remote/.spawn/.map but not @app.function-registered: {missing}"
