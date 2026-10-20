"""The fresh-clone install command is `uv sync --frozen --extra cpu`:
pyproject's `[tool.uv.sources]` maps flash-attn to a PATH WHEEL under `wheels/` (gitignored — it exists only in the cluster venvs'
clones; the Mamba kernels are PyPI sdists uv builds, 2026-10-12, so flash-attn is the one path source left), and a plain `uv sync`
re-resolves the project, which reads every source's metadata and dies on the missing wheel file before installing anything;
`--frozen` installs the committed lock as it stands. Pinned two ways: the
documented command in README / 00_setup.sh / docs says `--frozen`, and the command itself resolves in a `git worktree add` of HEAD
(no `wheels/`) while the plain form fails there for the recorded reason. The real install is opt-in (SISTEER_SLOW_TESTS=1).

The two tests that run uv need a uv that reads this pyproject's `[tool.uv]` table (UV_MIN_VERSION): an older one fails the plain
sync on the unknown key before it ever reaches the path wheels, and the recorded reason cannot appear. They run the first
compatible uv the probe finds (`_env.uv_executable`: the one uv names in $UV, the one on PATH, then the places an installer puts
one off PATH) and skip on exactly its absence, never on the venue."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from _env import UV_INSTALL_LOCATIONS, uv_executable, uv_version

REPO = Path(__file__).resolve().parents[1]
FRESH_CLONE_CMD = "uv sync --frozen --extra cpu"
#: The oldest uv that reads every `[tool.uv]` key pyproject uses. `exclude-newer-package` arrived in uv 0.8.4 (its changelog, PR
#: #14489); the Mamba-kernel build tables (2026-10-12) raised the floor: `extra-build-dependencies` entries with `match-runtime`
#: and `extra-build-variables`. Measured on this tree with pinned PyPI releases of uv (`uv pip install uv==…` into scratch venvs,
#: each run with its own UV_CACHE_DIR): 0.8.4 fails to PARSE pyproject ("TOML parse error" at the `{ requirement = "torch",
#: match-runtime = true }` inline table), 0.8.5 warns "unknown field `extra-build-variables`", and 0.8.6 through 0.8.17, 0.9.0
#: (cluster A's /usr/bin/uv), 0.11.22 (cluster B's) and 0.12.1 read every key (0.8.x-0.9.0 with uv's own "experimental" notice
#: for extra-build-dependencies) and give the plain dry-run's recorded `flash-attn==… @ path+wheels/…` reason. The CPU-suite image
#: (scripts/modal/cpu_suite.py, UV_VERSION) builds with the project's pinned uv, which Modal copies to /.uv/uv without putting it on
#: PATH; the probe (`_env.uv_executable`, `_env.UV_INSTALL_LOCATIONS`) looks there after PATH, so the image runs these tests on the
#: pinned release.
UV_MIN_VERSION = (0, 8, 6)
INSTALL_TEXT = ("README.md", "reproduce/00_setup.sh", "docs/configuration.md")


def _dotted(v: tuple[int, ...]) -> str:
    return ".".join(str(n) for n in v)


def _pinned_uv() -> str:
    """The uv the CPU-suite image build pins (`UV_VERSION = "…"` in scripts/modal/cpu_suite.py), named in the skip reason so a
    reader knows which release the project runs; "the project's pinned uv" when that line is not where this looks."""
    try:
        m = re.search(r'^UV_VERSION = "([^"]+)"', (REPO / "scripts/modal/cpu_suite.py").read_text(), re.M)
    except OSError:
        m = None
    return f"uv {m.group(1)}" if m else "the project's pinned uv"


def test_the_documented_fresh_clone_command_is_frozen():
    readme = (REPO / "README.md").read_text()
    row = next(ln for ln in readme.splitlines() if ln.startswith(FRESH_CLONE_CMD))   # the CPU-only setup block
    assert "--frozen" in row, row
    setup = (REPO / "reproduce/00_setup.sh").read_text()
    assert re.search(r"repro_run uv sync --frozen --extra cpu", setup), "00_setup.sh's venvs step must install the lock (--frozen)"
    for rel in INSTALL_TEXT:
        text = (REPO / rel).read_text()
        for m in re.finditer(r"uv sync(?: [^`\n]*)?--extra cpu", text):
            assert "--frozen" in m.group(0), f"{rel}: {m.group(0)!r} is a plain (re-resolving) sync; the fresh-clone command is `{FRESH_CLONE_CMD}`"
    assert "wheels" in readme and "--frozen" in readme       # the one sentence on WHY (the flash-attn wheel is a local path source)


@pytest.fixture
def uv_exe() -> str:
    """The uv the sync tests run: the first at or above UV_MIN_VERSION among the one uv itself names ($UV), the one on PATH and the
    places an installer puts one off PATH (`_env.UV_INSTALL_LOCATIONS`: the CPU-suite image's /.uv/uv, the standalone installer's
    ~/.local/bin/uv). Skips only when none qualifies (an older one cannot read pyproject's `[tool.uv]`, so neither sync below would
    mean what the assertions read), naming what WAS found."""
    exe = uv_executable(UV_MIN_VERSION)
    if exe is None:
        on_path = uv_version()
        found = f"uv {_dotted(on_path)} on PATH" if on_path else "no uv on PATH"
        pytest.skip(f"no uv {_dotted(UV_MIN_VERSION)}+ (`[tool.uv] exclude-newer-package`) in $UV, on PATH or at "
                    f"{', '.join(UV_INSTALL_LOCATIONS)}: {found}; the project builds with {_pinned_uv()}")
    return exe


@pytest.fixture
def worktree_of_head(tmp_path, uv_exe):
    """A `git worktree add --detach` of HEAD under tmp: exactly a fresh clone's tree (no `wheels/`, no `.venv`).

    Skips on the absence of the exact capability the uv tests need: git, and a compatible uv (`uv_exe`, asked for first so no
    worktree is made for a test that skips)."""
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    wt = Path("/tmp") / f"pcr-fresh-{uuid.uuid4().hex[:10]}"
    subprocess.run(["git", "worktree", "add", "--detach", "-q", str(wt), "HEAD"], cwd=REPO, check=True, capture_output=True, text=True)
    try:
        assert not (wt / "wheels").exists() and not (wt / ".venv").exists()
        yield wt
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=REPO, capture_output=True, text=True)
        subprocess.run(["git", "worktree", "prune"], cwd=REPO, capture_output=True, text=True)
        shutil.rmtree(wt, ignore_errors=True)


def _uv(exe: str, wt: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, UV_NO_PROGRESS="1")
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)   # a fresh clone has no venv: an inherited override would point the dry-run at a synced one
    return subprocess.run([exe, *args], cwd=wt, capture_output=True, text=True, env=env, timeout=900)


def test_frozen_sync_resolves_in_a_wheelless_worktree_and_the_plain_sync_does_not(worktree_of_head, uv_exe):
    wt = worktree_of_head
    ok = _uv(uv_exe, wt, "sync", "--frozen", "--extra", "cpu", "--dry-run")
    assert ok.returncode == 0, ok.stdout[-800:] + ok.stderr[-800:]
    assert "torch" in ok.stdout + ok.stderr                                     # the cpu flavor's torch would be installed
    plain = _uv(uv_exe, wt, "sync", "--extra", "cpu", "--dry-run")
    assert plain.returncode != 0, "a plain `uv sync` resolved without the path wheels — re-check the pin's premise (pyproject [tool.uv.sources])"
    # the recorded reason: the gitignored path wheel. On failure show the head too: uv's first stderr line names what it refused
    # (an unknown `[tool.uv]` field reads there, and only its field list in the tail).
    assert "flash-attn" in plain.stderr and "wheels/" in plain.stderr, (
        f"{uv_exe} (uv {_dotted(uv_version(uv_exe) or ())}): " + plain.stderr[:600] + "\n…\n" + plain.stderr[-800:])


@pytest.mark.skipif(os.environ.get("SISTEER_SLOW_TESTS") != "1", reason="the real install (~135 packages, torch cpu) — SISTEER_SLOW_TESTS=1 opts in")
def test_frozen_sync_really_installs_in_a_wheelless_worktree(worktree_of_head, uv_exe):
    wt = worktree_of_head
    r = _uv(uv_exe, wt, "sync", "--frozen", "--extra", "cpu")
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    py = wt / ".venv" / "bin" / "python"
    v = subprocess.run([str(py), "-c", "import torch, sisteer; print(torch.__version__)"], capture_output=True, text=True, cwd=wt)
    assert v.returncode == 0 and "+cpu" in v.stdout, v.stdout + v.stderr
