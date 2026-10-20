"""The first LIVE pick died at its first write — subprocess.run(["registry", "ruling", "add", …]) raised
FileNotFoundError: 'registry' — because pyproject's `registry` console script (sisteer.registry.cli:main) is not installed in a venv that was
never re-synced; every earlier test monkeypatched _run_registry and a dry-run never executes it, so the exec path had never run for real.
The exec now resolves the CLI: the console script when PATH has one, else `python -m sisteer.registry.cli` on THIS interpreter with the
repo's src on PYTHONPATH (the interim pyproject itself documents). Proven here by a REAL exec with no `registry` on PATH, not by a stub."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _hpo_pick():
    name = "hpo_pick"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / "sdfbase" / "hpo_pick.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def test_the_registry_exec_uses_the_console_script_when_present_else_the_module_on_this_interpreter(monkeypatch):
    hp = _hpo_pick()
    monkeypatch.setattr(hp.shutil, "which", lambda name: None)
    argv, how = hp.registry_cli_argv(["registry", "ruling", "add", "--id", "x"])
    assert argv == [sys.executable, "-m", "sisteer.registry.cli", "ruling", "add", "--id", "x"] and "no `registry` on PATH" in how
    monkeypatch.setattr(hp.shutil, "which", lambda name: "/somewhere/bin/registry")
    assert hp.registry_cli_argv(["registry", "ruling", "add"]) == (["registry", "ruling", "add"], "console script")
    assert hp.registry_cli_argv(["something-else", "x"]) == (["something-else", "x"], "verbatim")
    assert str(REPO / "src") in hp._registry_env()["PYTHONPATH"].split(os.pathsep), "the module form resolves to THIS checkout's code"


def test_a_dry_run_prints_the_composed_line_verbatim_and_executes_nothing(monkeypatch, capsys):
    hp = _hpo_pick()
    monkeypatch.setattr(hp.subprocess, "run", lambda *a, **k: pytest.fail("a dry run must not execute the registry CLI"))
    hp._run_registry(["registry", "ruling", "add", "--id", "x"], execute=False)
    assert capsys.readouterr().out.strip() == "DRY-RUN: registry ruling add --id x"


def test_the_live_exec_really_reaches_the_registry_cli_with_no_console_script_on_path(monkeypatch, capsys):
    """The exec that killed the first live pick, run for real: PATH holds no `registry`, `ruling --help` reaches the CLI's argparse through
    the module form (rc 0, its usage on stdout, the substitution named); a bad flag is the CLI's rc 2, surfaced by name."""
    hp = _hpo_pick()
    monkeypatch.setenv("PATH", str(Path(os.sep) / "nonexistent-bin"))
    hp._run_registry(["registry", "ruling", "--help"], execute=True)
    out = capsys.readouterr().out
    assert "(via sisteer.registry.cli (no `registry` on PATH))" in out and "usage: registry ruling" in out, out
    with pytest.raises(SystemExit, match=r"registry ruling add. failed \(rc 2\)"):
        hp._run_registry(["registry", "--no-such-flag"], execute=True)
