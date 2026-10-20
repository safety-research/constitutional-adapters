"""The stage scripts use the interpreter of the environment `uv sync` installs into.

uv installs into $UV_PROJECT_ENVIRONMENT when that is set (a relative value is resolved against the project root) and
into <repo>/.venv otherwise. reproduce/lib/common.sh's repro_python and reproduce/10_figures.sh's notebook interpreter
must look in the same place, so a researcher who keeps the venv off the repository (a node-local disk, say) is not told
to build an environment they already built.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "reproduce" / "lib" / "common.sh"
FIGURES = REPO / "reproduce" / "10_figures.sh"


def _fake_env(root: Path) -> Path:
    """A directory shaped like a venv whose bin/python is an executable stub."""
    py = root / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\nexit 0\n")
    py.chmod(0o755)
    return py


def _repro_python(env: dict[str, str]) -> str:
    base = {k: v for k, v in os.environ.items() if k not in ("SISTEER_PYTHON", "UV_PROJECT_ENVIRONMENT")}
    out = subprocess.run(["bash", "-c", f'source "{COMMON}"; repro_python'], env={**base, **env},
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_uv_project_environment_wins_over_the_repo_venv(tmp_path):
    py = _fake_env(tmp_path / "elsewhere")
    assert _repro_python({"UV_PROJECT_ENVIRONMENT": str(tmp_path / "elsewhere")}) == str(py)


def test_a_relative_uv_project_environment_resolves_against_the_repository(tmp_path):
    rel = os.path.relpath(tmp_path / "relenv", REPO)
    py = _fake_env(tmp_path / "relenv")
    assert Path(_repro_python({"UV_PROJECT_ENVIRONMENT": rel})).resolve() == py.resolve()


def test_sisteer_python_still_wins(tmp_path):
    _fake_env(tmp_path / "elsewhere")
    assert _repro_python({"SISTEER_PYTHON": "/opt/chosen/python",
                          "UV_PROJECT_ENVIRONMENT": str(tmp_path / "elsewhere")}) == "/opt/chosen/python"


def test_an_unset_uv_project_environment_falls_back_as_before(tmp_path):
    got = _repro_python({})
    venv = REPO / ".venv" / "bin" / "python"
    assert got == (str(venv) if os.access(venv, os.X_OK) else subprocess.run(
        ["bash", "-c", "command -v python3"], capture_output=True, text=True).stdout.strip())


def test_the_figure_stage_refusal_names_the_uv_project_environment(tmp_path):
    """A missing interpreter is refused by the path uv would have used, with the reason in the message."""
    missing = tmp_path / "not-built"
    base = {k: v for k, v in os.environ.items() if k not in ("SISTEER_NB_PYTHON", "SISTEER_SITE_DIR")}
    env = {**base, "UV_PROJECT_ENVIRONMENT": str(missing), "SISTEER_SITE_DIR": str(tmp_path / "no-site"),
           "HOME": str(tmp_path)}
    r = subprocess.run(["bash", str(FIGURES), "--source", "snapshot"], env=env, capture_output=True, text=True,
                       cwd=REPO, timeout=120)
    assert r.returncode != 0
    assert f"{missing}/bin/python is not executable" in r.stderr, r.stderr
    assert "UV_PROJECT_ENVIRONMENT" in r.stderr
