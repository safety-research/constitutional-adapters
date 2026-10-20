"""scripts/modal/e4b_wave.py resolves its two site settings (SISTEER_REPO, SISTEER_STAGE_DIR) where the image is built:
the environment first, else the site config through sisteer.site, the config's one reader.

Each case loads the script the way ``modal run <path>`` imports it locally, in a child interpreter with no site-packages
(so no installed sisteer, as in a Modal client environment) and a stub modal whose ``is_local()`` is true, so nothing
reaches Modal: the checkout copy the docstring documents, then a copy staged outside any checkout with both settings in
the environment, with SISTEER_REPO alone, with the checkout's src on PYTHONPATH, and with none of these, which refuses
by name."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modal" / "e4b_wave.py"

#: the child's loader: a stub modal (an image chain, and decorators that hand back the function), then the script by path
PROBE = '''
import runpy, sys, types

class _Chain:
    def __getattr__(self, _name):
        return self
    def __call__(self, *a, **k):
        return a[0] if len(a) == 1 and callable(a[0]) and not k else self

m = types.ModuleType("modal")
m.App = lambda *a, **k: _Chain()
m.Image = m.Volume = m.Secret = _Chain()
m.is_local = lambda: True
sys.modules["modal"] = m
ns = runpy.run_path(sys.argv[1], run_name="e4b_wave_probe")
site = sys.modules.get("sisteer.site")
print("WORKTREE=" + ns["WORKTREE"])
print("STAGE=" + ns["STAGE"])
print("SITE=" + (site.__file__ if site is not None else "-"))
'''


@pytest.fixture
def world(tmp_path):
    """A site config naming a checkout and a stage root, a copy of the script staged outside any checkout, a second
    checkout that carries only sisteer.site, and a home with no site directory."""
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "config.env").write_text(
        f"SISTEER_REPO='{tmp_path / 'site-repo'}'\nSISTEER_STAGE_DIR='{tmp_path / 'site-stage'}'\n")
    staged = tmp_path / "stage" / "e4b" / "e4b_wave.py"
    staged.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, staged)
    other = tmp_path / "other-checkout"
    (other / "src" / "sisteer").mkdir(parents=True)
    (other / "src" / "sisteer" / "__init__.py").write_text("")
    shutil.copy2(REPO / "src" / "sisteer" / "site.py", other / "src" / "sisteer" / "site.py")
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)
    (tmp_path / "home").mkdir()
    return SimpleNamespace(tmp=tmp_path, site_dir=site_dir, staged=staged, other=other, probe=probe)


def _run(w, script: Path, env: dict | None = None, pythonpath: Path | None = None):
    """The probe over ``script``: no site-packages, no bytecode written, the home and site directory of ``w``, and
    PYTHONPATH honoured only when ``pythonpath`` is given (otherwise isolated mode)."""
    child = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(w.tmp / "home"),
             "SISTEER_SITE_DIR": str(w.site_dir), **(env or {})}
    flags = ["-I", "-S", "-B"]
    if pythonpath is not None:
        child["PYTHONPATH"] = str(pythonpath)
        flags = ["-S", "-B"]
    proc = subprocess.run([sys.executable, *flags, str(w.probe), str(script)], env=child, capture_output=True,
                          text=True, timeout=120)
    return proc, dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)


def test_the_documented_run_line_is_the_checkout_copy():
    """The one ``modal run`` line of the docstring names this file by its path in the checkout, the invocation the next
    test loads (a copy staged elsewhere is the documented fallback, covered below)."""
    doc = ast.get_docstring(ast.parse(SCRIPT.read_text()))
    runs = [line.strip() for line in doc.splitlines() if line.strip().startswith("modal run ")]
    assert runs == ["modal run " + SCRIPT.relative_to(REPO).as_posix()]


def test_the_checkout_copy_reads_the_site_config_through_its_own_src(world):
    proc, out = _run(world, SCRIPT)
    assert proc.returncode == 0, proc.stderr
    assert out["WORKTREE"] == str(world.tmp / "site-repo" / "worktrees" / "twosweep")
    assert out["STAGE"] == str(world.tmp / "site-stage" / "e4b")
    assert Path(out["SITE"]) == REPO / "src" / "sisteer" / "site.py"


def test_a_staged_copy_given_both_settings_imports_no_site_module(world):
    proc, out = _run(world, world.staged, env={"SISTEER_REPO": str(world.tmp / "env-repo"),
                                               "SISTEER_STAGE_DIR": str(world.tmp / "env-stage")})
    assert proc.returncode == 0, proc.stderr
    assert out["WORKTREE"] == str(world.tmp / "env-repo" / "worktrees" / "twosweep")
    assert out["STAGE"] == str(world.tmp / "env-stage" / "e4b")
    assert out["SITE"] == "-"


def test_a_staged_copy_with_sisteer_repo_alone_takes_the_module_from_that_checkout(world):
    """The environment's SISTEER_REPO wins over the site config's, and its src supplies the module that reads the stage
    root from the site config."""
    proc, out = _run(world, world.staged, env={"SISTEER_REPO": str(world.other)})
    assert proc.returncode == 0, proc.stderr
    assert out["WORKTREE"] == str(world.other / "worktrees" / "twosweep")
    assert out["STAGE"] == str(world.tmp / "site-stage" / "e4b")
    assert Path(out["SITE"]).resolve() == (world.other / "src" / "sisteer" / "site.py").resolve()


def test_a_staged_copy_with_the_checkout_src_on_pythonpath_reads_the_site_config(world):
    proc, out = _run(world, world.staged, pythonpath=world.other / "src")
    assert proc.returncode == 0, proc.stderr
    assert out["WORKTREE"] == str(world.tmp / "site-repo" / "worktrees" / "twosweep")
    assert out["STAGE"] == str(world.tmp / "site-stage" / "e4b")
    assert Path(out["SITE"]).resolve() == (world.other / "src" / "sisteer" / "site.py").resolve()


def test_a_staged_copy_with_none_of_these_refuses_by_name(world):
    proc, out = _run(world, world.staged)
    assert proc.returncode != 0 and "WORKTREE" not in out
    for word in ("sisteer.site", "SISTEER_REPO", "PYTHONPATH", "scripts/modal/e4b_wave.py"):
        assert word in proc.stderr, (word, proc.stderr)


def test_a_setting_neither_the_environment_nor_the_site_config_gives_refuses_by_name(world):
    (world.site_dir / "config.env").write_text(f"SISTEER_REPO='{world.tmp / 'site-repo'}'\n")
    proc, out = _run(world, SCRIPT)
    assert proc.returncode != 0 and "STAGE" not in out
    assert "set SISTEER_STAGE_DIR in the environment or the site config" in proc.stderr
