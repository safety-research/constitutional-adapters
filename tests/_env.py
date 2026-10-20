"""Prerequisite probes for the test suite: pure functions, so a test skips on exactly the capability it needs and nowhere else.

The suite runs in three kinds of place with different endowments: the maintainers' landing gate (rsync, ssh, ss, an IPv6
loopback, a non-root user, uv), a researcher's CPU machine, and the Modal CPU-suite container (``scripts/modal/cpu_suite.py``:
a minimal Debian image carrying uv's environment and little else, running as root). A test that needs one of those marks
itself ``pytest.mark.skipif(not <probe>(), reason="<the capability> ...")`` with a probe from HERE, so the condition is written
once, reads the same everywhere, and is false on the gate (whose skipped count must not rise). Every probe checks the exact
capability, never the venue: ``shutil.which`` for a binary, a bind for IPv6, a refused create for mode enforcement. A container
that happens to carry rsync runs the rsync tests.

Nothing here skips, imports pytest or reads the repository. Every function is callable at module import time, where a
``skipif`` evaluates, and from a fixture alike; the one that touches the filesystem (:func:`permission_denial_enforced`) cleans
up after itself whatever the outcome. tests/ is on ``sys.path`` whenever a test module loads (pytest's default prepend import
mode, the same way ``tests/conftest.py`` imports ``_full_suite_guard``), so ``from _env import have_exe`` is the whole import.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = [
    "have_exe",
    "permission_denial_enforced",
    "ipv6_loopback_available",
    "uv_version",
    "uv_executable",
    "UV_INSTALL_LOCATIONS",
    "write_python3_shim",
    "host_site_path",
]


def have_exe(name: str, path: str | None = None) -> bool:
    """True when ``name`` resolves to an executable on PATH.

    ``path`` searches that PATH string instead of the process's own, for a test that already stripped its child's PATH and wants
    to ask about THAT environment (``shutil.which``'s own keyword, passed through).
    """
    return shutil.which(name, path=path) is not None


def permission_denial_enforced(tmp_dir: str | os.PathLike[str] | None = None) -> bool:
    """True when the kernel refuses to create an entry in a directory this process owns but holds no write bit on (mode 0o555).

    False for root and for any process holding ``CAP_DAC_OVERRIDE`` (a container running as root), and on a filesystem that does
    not honour mode bits at all. A test whose assertion IS a permission refusal (a rename the directory rejects, a store the attest
    sealed read-only, a file ``chmod 0`` keeps from ``cat``) skips on this, since where the kernel does not enforce modes the
    refusal it asserts cannot happen.

    The probe makes a fresh directory under ``tmp_dir`` (the system temp dir by default, so a module-level ``skipif`` needs no
    fixture), drops its write bit, tries one ``mkdir`` beneath it, then restores the mode and removes the lot. Only
    ``PermissionError`` reads as enforcement; any other failure propagates, since it is not the answer to this question.
    """
    base = Path(tmp_dir) if tmp_dir is not None else Path(tempfile.gettempdir())
    probe = Path(tempfile.mkdtemp(prefix="perm-probe-", dir=base))
    try:
        os.chmod(probe, 0o555)
        try:
            os.mkdir(probe / "child")
        except PermissionError:
            return True
        return False
    finally:
        os.chmod(probe, 0o755)
        shutil.rmtree(probe, ignore_errors=True)


def ipv6_loopback_available() -> bool:
    """True when an ``AF_INET6`` socket binds to ``::1`` on an ephemeral port.

    False where the build has no IPv6 (``socket.has_ipv6``), the loopback interface carries no IPv6 address (``EADDRNOTAVAIL`` at
    bind: a network namespace with its loopback down, a kernel booted with ``ipv6.disable=1``) or the family is unsupported
    (``EAFNOSUPPORT``). The socket is closed before returning; nothing listens.
    """
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
    except OSError:
        return False
    return True


_UV_VERSION = re.compile(r"\buv\s+(\d+)\.(\d+)(?:\.(\d+))?")


def uv_version(exe: str | None = None, path: str | None = None) -> tuple[int, ...] | None:
    """The version of the ``uv`` executable ``exe``, or of the ``uv`` on PATH when ``exe`` is None, as a tuple of ints:
    ``(0, 12, 1)`` from ``uv 0.12.1 (x86_64-unknown-linux-gnu)``.

    None when there is no such executable (``path`` searches that PATH string instead of the process's own, as in
    :func:`have_exe`), it fails to run, exits non-zero, or prints a first line the pattern does not read. Compare tuples, never
    strings: ``(0, 12, 1) >= (0, 9, 0)``.
    """
    if exe is None:
        exe = shutil.which("uv", path=path)
    if exe is None or not os.access(exe, os.X_OK) or os.path.isdir(exe):
        return None
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    m = _UV_VERSION.search(r.stdout)
    if m is None:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


#: Where an installer puts ``uv`` WITHOUT putting it on PATH: Modal's ``Image.uv_sync`` / ``uv_pip_install`` copy the pinned binary
#: to ``/.uv/uv`` and put only the environment they build on PATH (the CPU-suite image, ``scripts/modal/cpu_suite.py``); the
#: standalone installer's default is ``~/.local/bin/uv``, which a login shell adds to PATH and a service or container process may
#: not see. Searched after PATH, in this order.
UV_INSTALL_LOCATIONS: tuple[str, ...] = ("/.uv/uv", "~/.local/bin/uv")


def uv_executable(min_version: tuple[int, ...] = (), path: str | None = None) -> str | None:
    """The first ``uv`` executable at or above ``min_version``: the one uv itself names in ``$UV`` (set for every process ``uv run``
    starts), then the one on PATH (``path`` searches that PATH string instead), then :data:`UV_INSTALL_LOCATIONS`. None when none
    qualifies, so a test skips on the absence of a COMPATIBLE uv anywhere it could be, not on the age of the first one found.
    """
    candidates: list[str] = []
    named = os.environ.get("UV")
    if named:
        candidates.append(named)
    on_path = shutil.which("uv", path=path)
    if on_path:
        candidates.append(on_path)
    candidates.extend(os.path.expanduser(p) for p in UV_INSTALL_LOCATIONS)
    floor = tuple(min_version)
    for exe in candidates:
        v = uv_version(exe)
        if v is not None and v >= floor:
            return exe
    return None


def write_python3_shim(bin_dir: str | os.PathLike[str], name: str = "python3") -> Path:
    """Write ``bin_dir/python3``: an executable ``sh`` script that ``exec``s THIS interpreter, ``sys.executable``, with its arguments.

    For a test that runs a shell surface under a restricted PATH (a few shim directories, or ``/usr/bin:/bin`` alone) where the
    surface calls a bare ``python3``. The shim makes that call the interpreter running the test, the one with the project's
    dependencies, rather than whichever system ``python3`` the machine carries, or none: the Modal CPU-suite image has uv's
    environment and no ``/usr/bin/python3``. ``name`` writes the same shim under another name (``python``). Creates ``bin_dir``
    when it does not exist; returns the shim's path, so a caller can prepend its parent to the child's PATH.
    """
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / name
    shim.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    shim.chmod(0o755)
    return shim


def host_site_path(spec: str) -> Path:
    """``spec``, a path written with a site placeholder (``$STATE/...``, ``$STAGE/...``), expanded through sisteer.site, for a
    test that reads data only one host holds and skips where it is absent. Where the placeholder does not resolve on this host
    (no site value and no environment variable) the placeholder spelling stays, which names no directory, so such a test
    skips as it would on any host without the data."""
    from sisteer import site as _site
    try:
        return _site.expand_path(spec)
    except (_site.PlaceholderError, _site.SiteConfigError):
        return Path(spec)
