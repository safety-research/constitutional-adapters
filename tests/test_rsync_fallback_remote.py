"""The pure-Python ``rsync`` fallback (tests/_rsync_fallback.py) refuses what it cannot do: a remote operand.

Measured against rsync 3.2.7 and pinned here: ``rsync -aq -e false HOST:src dest/`` exits 12 ("error in rsync protocol data stream")
and copies nothing -- the remote shell closed without delivering the other side; the same with ``--rsh=false``, with the cluster form
``-aqe false``, and with the remote operand as the destination; ``rsync -aq -e false src dest/`` (local operands only) copies and
exits 0, the remote shell never consulted; a daemon operand (``HOST::module``) whose host does not resolve exits 10. The fallback runs
no remote shell, so a ``host:path`` operand exits 12 before anything is created -- a fixture that reached it with one would otherwise
pass without the transport it means to exercise -- unless the test opts in by exact host token (``SISTEER_RSYNC_FALLBACK_LOCAL_HOSTS``,
comma-separated), under which that host's operands read as local paths and every other host still refuses. Where the real binary is
present each pinned code is also compared against it on the same arguments: a remote shell that exits 1 (``false``, or an ``ssh`` shim
that does) stands for the unreachable host, so no network is needed and the comparison runs the same on the gate and a developer's
machine. The fallback runs here as the wrapper the fixtures exec (``python -IS``), so the opt-in is read from the inherited environment
the way a world's child processes see it.
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import _rsync_fallback as RF

OPT_IN = RF.LOCAL_HOSTS_ENV


def _real_rsync() -> Optional[str]:
    """The host's own rsync, or None: the one on PATH unless it is a fallback wrapper this process installed."""
    found = shutil.which("rsync")
    if found and os.path.dirname(found) in RF.installed_dirs():
        return None
    return found


def _run(binary: str, argv: List[str], env: Dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([binary, *argv], capture_output=True, text=True, env=env, timeout=120)


def _fallback(argv: List[str], hosts: Optional[str] = None) -> subprocess.CompletedProcess:
    """The fallback as the fixtures exec it (the wrapper, ``python -IS``), with the opt-in variable set, or absent when ``hosts`` is None."""
    env = {"PATH": os.environ["PATH"]}
    if hosts is not None:
        env[OPT_IN] = hosts
    return _run(os.path.join(RF.fallback_bin_dir(), "rsync"), argv, env)


def _real(argv: List[str], ssh_rc: Optional[int] = None, tmp_path: Optional[Path] = None) -> Optional[subprocess.CompletedProcess]:
    """The real binary on the same arguments, or None where the host has none. ``ssh_rc``: an ``ssh`` shim exiting with that code is put
    first on the child's PATH, standing for a remote shell that does not deliver the other side."""
    binary = _real_rsync()
    if binary is None:
        return None
    path = os.environ["PATH"]
    if ssh_rc is not None:
        assert tmp_path is not None
        shim = tmp_path / f"ssh-exits-{ssh_rc}"
        shim.mkdir(exist_ok=True)
        (shim / "ssh").write_text(f"#!/bin/sh\nexit {ssh_rc}\n")
        (shim / "ssh").chmod((shim / "ssh").stat().st_mode | stat.S_IXUSR)
        path = f"{shim}{os.pathsep}{path}"
    return _run(binary, argv, {"PATH": path, **{k: os.environ[k] for k in ("HOME",) if k in os.environ}})


def _world(tmp_path: Path, name: str) -> tuple:
    src = tmp_path / f"{name}-src.txt"
    src.write_text(f"{name}\n")
    dest = tmp_path / f"{name}-dest"
    dest.mkdir()
    return str(src), str(dest) + "/"


def _empty(dest: str) -> bool:
    return os.listdir(dest) == []


def test_a_remote_operand_is_refused_with_rsyncs_code_before_anything_is_copied(tmp_path):
    # -e false: the remote shell exits without the other side; rsync's code 12, nothing copied -- the fallback and the real binary alike
    src, dest = _world(tmp_path, "e-false")
    argv = ["-aq", "-e", "false", f"TESTHOST:{src}", dest]
    r = _fallback(argv)
    assert r.returncode == RF.REMOTE_REFUSED_RC == 12 and _empty(dest), (r.returncode, r.stderr)
    line = r.stderr.strip().splitlines()[0]
    assert "does not run a remote shell" in line and f"TESTHOST:{src}" in line and f"{OPT_IN}=TESTHOST" in line and "'false'" in line, line
    assert "(code 12)" in r.stderr
    real = _real(argv)
    if real is not None:
        assert real.returncode == 12 and _empty(dest), (real.returncode, real.stderr)
    # no -e: rsync's default remote shell is ssh; one that exits 1 gives the same code 12 -- the fallback pins it with no shell at all
    src, dest = _world(tmp_path, "no-e")
    argv = ["-aq", f"TESTHOST:{src}", dest]
    r = _fallback(argv)
    assert r.returncode == 12 and _empty(dest) and f"{OPT_IN}=TESTHOST" in r.stderr and "is not run either" not in r.stderr, r.stderr
    real = _real(argv, ssh_rc=1, tmp_path=tmp_path)
    if real is not None:
        assert real.returncode == 12 and _empty(dest), (real.returncode, real.stderr)
    # the remote operand as the DESTINATION, --rsh=CMD, and the cluster form -aqe CMD: refused the same way
    for name, argv_of in (("remote-dest", lambda s, d: ["-aq", "-e", "false", s, f"TESTHOST:{d}"]),
                          ("rsh-long", lambda s, d: ["-aq", "--rsh=false", f"TESTHOST:{s}", d]),
                          ("cluster", lambda s, d: ["-aqe", "false", f"TESTHOST:{s}", d])):
        src, dest = _world(tmp_path, name)
        argv = argv_of(src, dest)
        r = _fallback(argv)
        assert r.returncode == 12 and _empty(dest), (name, r.returncode, r.stderr)
        real = _real(argv)
        if real is not None:
            assert real.returncode == 12 and _empty(dest), (name, real.returncode, real.stderr)


def test_a_remote_shell_option_with_local_operands_only_is_ignored_as_the_real_binary_ignores_it(tmp_path):
    src, dest = _world(tmp_path, "local-e")
    r = _fallback(["-aq", "-e", "false", src, dest])
    assert r.returncode == 0 and r.stderr == "" and (Path(dest) / Path(src).name).read_text() == "local-e\n", (r.returncode, r.stderr)
    src2, dest2 = _world(tmp_path, "local-e-real")
    real = _real(["-aq", "-e", "false", src2, dest2])
    if real is not None:
        assert real.returncode == 0 and (Path(dest2) / Path(src2).name).read_text() == "local-e-real\n", (real.returncode, real.stderr)
    src3, dest3 = _world(tmp_path, "local-rsh")
    assert _fallback(["-aq", "--rsh", "false", src3, dest3]).returncode == 0 and not _empty(dest3)


def test_the_opt_in_substitutes_exactly_the_named_hosts_and_every_other_host_still_refuses(tmp_path):
    # the named host, as a source and as a destination, with and without a remote shell option: read as the local path
    src, dest = _world(tmp_path, "opt-src")
    r = _fallback(["-aq", "-e", "false", f"TESTHOST:{src}", dest], hosts="TESTHOST")
    assert r.returncode == 0 and r.stderr == "" and (Path(dest) / Path(src).name).read_text() == "opt-src\n", (r.returncode, r.stderr)
    src, dest = _world(tmp_path, "opt-dest")
    r = _fallback(["-aq", src, f"TESTHOST:{dest}"], hosts="TESTHOST")
    assert r.returncode == 0 and (Path(dest) / Path(src).name).read_text() == "opt-dest\n", (r.returncode, r.stderr)
    # the list: commas and spaces, several hosts
    src, dest = _world(tmp_path, "opt-list")
    assert _fallback(["-aq", f"TESTHOST:{src}", dest], hosts=" OTHERHOST , TESTHOST ,").returncode == 0 and not _empty(dest)
    # another host still refuses, as does a call that mixes a named host with an unnamed one (nothing copied, nothing created)
    src, dest = _world(tmp_path, "other")
    r = _fallback(["-aq", "-e", "false", f"OTHERHOST:{src}", dest], hosts="TESTHOST")
    assert r.returncode == 12 and _empty(dest) and f"{OPT_IN}=OTHERHOST" in r.stderr, (r.returncode, r.stderr)
    src, dest = _world(tmp_path, "mixed")
    r = _fallback(["-aq", f"TESTHOST:{src}", f"OTHERHOST:{dest}"], hosts="TESTHOST")
    assert r.returncode == 12 and _empty(dest), (r.returncode, r.stderr)
    # exact tokens: a user prefix is part of the token, and an empty variable opts nothing in
    src, dest = _world(tmp_path, "user-prefix")
    assert _fallback(["-aq", f"tester@TESTHOST:{src}", dest], hosts="TESTHOST").returncode == 12 and _empty(dest)
    src, dest = _world(tmp_path, "empty-var")
    assert _fallback(["-aq", f"TESTHOST:{src}", dest], hosts="").returncode == 12 and _empty(dest)


def test_a_daemon_operand_is_refused_even_under_the_opt_in(tmp_path):
    for hosts in (None, "TESTHOST"):
        src, dest = _world(tmp_path, f"daemon-{hosts}")
        r = _fallback(["-aq", "TESTHOST::module/x", dest], hosts=hosts)
        assert r.returncode == RF.DAEMON_REFUSED_RC == 10 and _empty(dest) and "speaks to no rsync daemon" in r.stderr, (r.returncode, r.stderr)
        r = _fallback(["-aq", src, "rsync://TESTHOST/module/"], hosts=hosts)
        assert r.returncode == 10 and "(code 10)" in r.stderr, (r.returncode, r.stderr)


def test_operands_are_read_as_rsync_reads_them():
    assert RF.split_operand("/a:b/c") == ("local", None, "/a:b/c")                 # a leading slash: a path, whatever follows
    assert RF.split_operand("a/b:c") == ("local", None, "a/b:c")                   # a slash before the colon: a path
    assert RF.split_operand("plain") == ("local", None, "plain")
    assert RF.split_operand("TESTHOST:/p/q") == ("remote", "TESTHOST", "/p/q")
    assert RF.split_operand("TESTHOST:rel") == ("remote", "TESTHOST", "rel")
    assert RF.split_operand("TESTHOST:") == ("remote", "TESTHOST", ".")           # an empty remote path is the remote home
    assert RF.split_operand("tester@TESTHOST:/p") == ("remote", "tester@TESTHOST", "/p")
    assert RF.split_operand("TESTHOST::module") == ("daemon", None, "TESTHOST::module")
    assert RF.split_operand("rsync://TESTHOST/module") == ("daemon", None, "rsync://TESTHOST/module")
    assert RF.local_hosts({}) == frozenset() and RF.local_hosts({OPT_IN: " , "}) == frozenset()
    assert RF.local_hosts({OPT_IN: "A, B,,C "}) == frozenset({"A", "B", "C"})
