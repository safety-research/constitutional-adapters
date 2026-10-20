"""ifeval-canonical-battery (P2): the dispatcher never runs a bare ``modal`` executable.

Under the documented absolute venv interpreter the venv's ``bin`` is not on ``PATH``: ``shutil.which("modal")`` is
None, so the first preflight subprocess died with ``FileNotFoundError`` and ``--stop`` could not run at all. Every
Modal CLI the dispatcher composes now goes through ``[sys.executable, "-m", "modal", …]`` and names the launcher script
by its ABSOLUTE repository path (the launcher's printed argv spells it relative to the repository root — the operator's
copy-paste convention, not a working-directory guarantee).
"""
from __future__ import annotations

import os
import shutil
import sys
import types
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import contracts as C

from test_ifeval_dispatch import APP, D, LAUNCH, SHA, FakeRegistry, make_launcher


@pytest.fixture
def no_modal_on_path(tmp_path, monkeypatch):
    """A PATH with exactly one empty directory: no ``modal``, no venv ``bin``."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert shutil.which("modal") is None and shutil.which(os.path.basename(sys.executable)) is None
    return empty


def test_every_modal_invocation_runs_through_this_interpreter_and_the_launcher_by_absolute_path(no_modal_on_path, tmp_path):
    L = make_launcher(FakeRegistry())
    argv, env = D.preflight_argv(L, sha=SHA, app_name=APP, slug=LAUNCH[0]["slug"], cells=[LAUNCH[0]["cell"]], cap=4096)
    assert argv[:5] == [sys.executable, "-m", "modal", "run", str(D.LAUNCHER_SCRIPT)]
    assert Path(argv[4]).is_absolute() and Path(argv[4]).is_file() and Path(argv[4]).name == "tinycap_served.py"
    assert "modal" not in argv[:1] and argv[-1] == D.PREFLIGHT_FLAG
    assert argv[argv.index(C.FLAG_OFF_CAP_BY_TASK) + 1] == f"{C.IFEVAL_TASK_NAME}=4096"      # the launcher's own tail rides
    assert env[C.MODAL_APP_NAME_ENV] == APP
    # run_preflights hands exactly that argv to the subprocess runner
    runs = []
    D.run_preflights(L, [LAUNCH[0]], sha=SHA, app_name=APP, cap=4096,
                     run_subprocess=lambda a, env=None, **k: (runs.append(list(a)), types.SimpleNamespace(returncode=0))[1],
                     log=lambda s: None)
    assert runs == [argv]
    # the stop paths: `python -m modal app stop <id> --yes`, never a bare `modal`
    assert D.STOP_APP_ARGV == (sys.executable, "-m", "modal", "app", "stop") and D.MODAL_ARGV == (sys.executable, "-m", "modal")
    calls = []
    rc = D.default_stop_app("ap-9", run_subprocess=lambda a, **k: (calls.append(list(a)), types.SimpleNamespace(returncode=0))[1])
    assert rc == 0 and calls == [[sys.executable, "-m", "modal", "app", "stop", "ap-9", "--yes"]]
    state = tmp_path / "state" / D.IFEVAL_DISPATCH_STATE
    rc = D.request_stop(state, APP, run_subprocess=lambda a, **k: (calls.append(list(a)), types.SimpleNamespace(returncode=0))[1],
                        log=lambda s: None)
    assert rc == D.EXIT_OK and calls[-1] == [sys.executable, "-m", "modal", "app", "stop", APP, "--yes"]
    assert (state.parent / D.STOP_SENTINEL).is_file()
    # the composer refuses an argv of another shape by name rather than guessing the executable token
    with pytest.raises(ValueError, match="not `modal run <launcher script>"):
        D.modal_run_argv(["python", "scripts/modal/tinycap_served.py", "--sha", SHA])
    assert D.modal_run_argv(["modal", "run", "scripts/modal/tinycap_served.py", "--sha", SHA]) == \
        [sys.executable, "-m", "modal", "run", str(D.LAUNCHER_SCRIPT), "--sha", SHA]
