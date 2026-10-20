"""A subprocess launched with a venv's python by ABSOLUTE PATH does not get that venv's bin/ on PATH, so a script
that shells out to a tool by name (`vllm` from the serving venv) dies `'vllm' not on PATH` — exactly how the S1 standup failed.
`seedrep_served.venv_env` puts the venv's bin first on PATH (+ VIRTUAL_ENV); the standup and every child Popen must use it."""
from __future__ import annotations

import ast
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
DRIVER = REPO / "scripts" / "modal" / "seedrep_served.py"


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, *a, **k):
        return lambda fn: fn

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
    spec = importlib.util.spec_from_file_location("seedrep_served_w13_venv_path", DRIVER)
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


def _fake_venv(tmp_path: Path) -> Path:
    """A venv the way `uv venv` lays it out: bin/python is a SYMLINK to a base interpreter outside the venv (the known-bad:
    `resolve()` followed it and put /usr/bin on PATH instead of the venv's bin), bin/vllm a console script beside it."""
    base = tmp_path / "base" / "bin"
    base.mkdir(parents=True)
    (base / "python3.12").write_text("#!/bin/sh\nexit 0\n")
    (base / "python3.12").chmod(0o755)
    venv = tmp_path / "serve-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(base / "python3.12")
    t = venv / "bin" / "vllm"
    t.write_text("#!/bin/sh\nexit 0\n")
    t.chmod(0o755)
    return venv


def test_known_bad_the_container_env_alone_cannot_find_vllm(tmp_path):
    """The failing condition: PATH is the container's, the venv's bin is not on it -> `vllm` unresolvable by name."""
    venv = _fake_venv(tmp_path)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HUB_OFFLINE": "1"}
    assert shutil.which("vllm", path=env["PATH"]) is None
    assert str(venv / "bin") not in env["PATH"]


def test_venv_env_puts_the_venvs_bin_first_and_sets_virtual_env(srv, tmp_path):
    venv = _fake_venv(tmp_path)
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HUB_OFFLINE": "1"}
    out = srv.venv_env(env, str(venv / "bin" / "python"))
    assert (venv / "bin" / "python").is_symlink() and (venv / "bin" / "python").resolve().parent != venv / "bin"   # the real layout
    assert out["PATH"].split(":")[0] == str(venv / "bin"), "the venv's own bin must lead, not the base interpreter's (W14)"
    assert str((venv / "bin" / "python").resolve().parent) not in out["PATH"].split(":")[:1]
    assert shutil.which("vllm", path=out["PATH"]) == str(venv / "bin" / "vllm")
    assert out["VIRTUAL_ENV"] == str(venv)
    assert out["HF_HUB_OFFLINE"] == "1" and env == {"PATH": "/usr/local/bin:/usr/bin:/bin", "HF_HUB_OFFLINE": "1"}   # input untouched
    again = srv.venv_env(out, str(venv / "bin" / "python"))
    assert again["PATH"].split(":").count(str(venv / "bin")) == 1                                                   # idempotent


def test_every_venv_python_popen_in_the_app_wraps_its_env_with_venv_env():
    """The standup Popen (argv[0] = <serve_venv>/bin/python) and every child Popen (argv[0] = the driver/client python) must pass
    `env=venv_env(<env>, <argv>[0])`; a bare `env=` on those sites is the defect."""
    tree = ast.parse(DRIVER.read_text())
    popens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"
              and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess"]
    assert popens, "no subprocess.Popen in the app?"
    bad = []
    for call in popens:
        first = call.args[0] if call.args else None
        argv_name = first.id if isinstance(first, ast.Name) else None
        if argv_name is None or not argv_name.endswith("argv"):
            continue                                            # nvidia-smi dmon etc. -- literal argv, not a venv python
        env_kw = next((k for k in call.keywords if k.arg == "env"), None)
        ok = (env_kw is not None and isinstance(env_kw.value, ast.Call) and isinstance(env_kw.value.func, ast.Name)
              and env_kw.value.func.id == "venv_env")
        if not ok:
            bad.append(f"line {call.lineno}: Popen({argv_name}, env=...) without venv_env")
    assert not bad, bad
