"""A pure-Python ``rsync`` for hosts without the binary -- the fixtures' copy step, never a skip.

The registry-store, cluster-read, landing and object-staging fixtures build fake destinations with ``rsync`` and ``ssh`` shims and
route every copy to the LOCAL rsync binary even for a local destination (``nice ionice -c3 rsync ...`` straight to ``subprocess.run``,
``cluster_reads._rsync``, ``stage_window``, the battery's record collection, ``stage_objects.RSYNC``). A plain CPU environment (the
Modal CPU-suite image: a minimal Debian with git and a compiler, no rsync) has no binary to route to, so every one of those tests
failed on the copy -- ``rsync_failed``, rc 255, ``FileNotFoundError: 'rsync'`` -- while asserting nothing about rsync itself. What
the container lacks is the local binary the shims call, so a copy that keeps the assertions intact is the right fix: this module
implements exactly the flag subset the code under test emits and installs itself as ``rsync`` on PATH ONLY where
``have_exe("rsync")`` is False, and only for the test that asked (the ``rsync_on_path`` fixture and ``tests/conftest.py`` take it off
PATH again when that test ends, so a later test's own ``shutil.which("rsync")`` reads the host). Where the real binary exists (the
maintainers' gate, a developer's machine) nothing changes: every fixture runs the real rsync as before and the skipped count does
not move.

The subset (each flag the code emits, nothing more; an unknown flag is a loud ``rc 1`` so a new call shape is noticed, not silently
copied wrong): ``-r -l -t -p -a -q -c -m -R -v``, ``--relative`` with the ``/./`` anchor, ``--exclude`` / ``--include`` in rsync's
filter grammar (anchored ``/x``, basename-only patterns, ``*`` / ``?`` / ``[...]`` with the backslash-escape rule of a pattern that
holds a wildcard, trailing ``/`` for directories), ``--prune-empty-dirs``, ``--files-from=FILE|-``, ``--ignore-times``,
``--checksum``, several ``--link-dest=DIR`` (hard-link when size and whole-second mtime match the link-dest copy -- rsync's quick
check -- else copy, so a same-size mtime-preserving edit links the OLD bytes exactly as the real binary does and the forced pass the
stores module runs for that reason stays meaningful), ``--chmod=Du+w,Fu+w|F644``, ``--delete``, ``--stats`` (the two lines
``_rsync_stats`` and ``stage_objects.stage_to`` parse), and the accepted no-ops ``--partial --partial-dir --timeout --inplace
--no-owner --no-group --no-perms``. Semantics the fixtures read and this copy preserves: hard links across generations (``st_ino`` /
``st_nlink``), forced copies under ``--ignore-times``, a temp file + rename for every write (a hard-linked destination file is never
written through), new files at the source mode masked by the umask and existing files keeping theirs (``-p`` copies the mode
exactly), directory mtimes set last, empty directories carried, ``--delete`` of what the source lacks, and rsync's exit codes (23 for
a transfer with errors, 1 for usage).

Remote operands are REFUSED, the way the real binary fails when its remote shell does not deliver the other side: a ``host:path``
operand exits 12 (rsync's "error in rsync protocol data stream", what ``rsync -e false host:path dest`` returns) before anything is
created or copied, with one stderr line naming the operand and the opt-in; a daemon operand (``host::module``, ``rsync://``) exits 10
(rsync's "error in socket IO"). This copy runs no remote shell, so a fixture that reached the fallback with a remote operand would
otherwise pass without the transport it means to exercise (the live-db landing world ships and collects over ssh: with the real
binary its ssh shim runs the remote ``rsync --server`` locally and the wire protocol is spoken end to end; such a test skips where
the binary is absent instead). A fixture whose intent with the real binary is ALREADY a local substitution opts in explicitly and
narrowly: ``SISTEER_RSYNC_FALLBACK_LOCAL_HOSTS=HOSTA,HOSTB`` (exact host tokens as written in the operand, comma-separated) reads
``HOSTA:path`` as the local ``path``; every other host still refuses. ``-e CMD`` / ``--rsh=CMD`` is remembered and never run: with a
remote operand it is part of the refusal (or of the substitution, for an opted-in host), and with local operands only it has no
effect, exactly as the real binary ignores it for a local copy (measured: ``rsync -aq -e false src dest/`` copies and exits 0).
"""
from __future__ import annotations

import atexit
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

try:
    from _env import have_exe                                   # the suite's shared capability readings (tests/_env.py)
except ImportError:                                             # the wrapper runs under ``python -IS``: stdlib only, no tests/ on sys.path
    def have_exe(name: str) -> bool:
        return shutil.which(name) is not None

try:
    import pytest                                               # absent under the wrapper's ``-S``; the fixture below is then not defined
except ImportError:                                             # pragma: no cover - the wrapper process
    pytest = None

HERE = os.path.dirname(os.path.abspath(__file__))
BIN_NAME = "rsync"
#: the opt-in: exact host tokens, comma-separated, whose ``host:path`` operands this copy reads as the local ``path``
LOCAL_HOSTS_ENV = "SISTEER_RSYNC_FALLBACK_LOCAL_HOSTS"
#: rsync's RERR_STREAMIO, what the real binary exits when the remote shell closes without delivering the other side (``-e false``)
REMOTE_REFUSED_RC = 12
#: rsync's RERR_SOCKETIO, what the real binary exits when a daemon operand's host does not resolve
DAEMON_REFUSED_RC = 10
_bin_dir: Optional[str] = None
_bin_dirs: List[str] = []                                       # every directory this process ever installed (see installed_dirs)


# --------------------------------------------------------------------------------------------
# installation: the wrapper on PATH where the host has no rsync
# --------------------------------------------------------------------------------------------


def fallback_bin_dir() -> str:
    """A directory holding an executable ``rsync`` that runs :func:`fallback_main`: made on demand under the system temp dir, removed by
    :func:`teardown_fallback` when the test that asked ends (the ``rsync_on_path`` fixture and the conftest belt), and at exit as a backstop
    only -- a process that leaves by ``os._exit`` or a signal never runs its ``atexit`` handlers, which is how a wrapper directory outlived
    its suite once. The next call after a teardown makes a fresh one."""
    global _bin_dir
    if _bin_dir is None or not os.path.isfile(os.path.join(_bin_dir, BIN_NAME)):
        d = tempfile.mkdtemp(prefix="rsync-fallback-")
        wrapper = os.path.join(d, BIN_NAME)
        # a shell wrapper rather than a shebang onto the interpreter: no shebang length limit, and ``-IS`` keeps the copy stdlib-only
        # and independent of PYTHONPATH / site (the subprocesses the tests spawn set both)
        with open(wrapper, "w", encoding="utf-8") as fh:
            fh.write(f'#!/bin/sh\nexec "{sys.executable}" -IS "{os.path.join(HERE, os.path.basename(__file__))}" "$@"\n')
        os.chmod(wrapper, 0o755)
        atexit.register(shutil.rmtree, d, ignore_errors=True)
        _bin_dir = d
        _bin_dirs.append(d)
    return _bin_dir


def installed_dirs() -> Tuple[str, ...]:
    """Every directory :func:`fallback_bin_dir` made in this process (normally one), whether or not it is on PATH right now."""
    return tuple(_bin_dirs)


def ensure_rsync_on_path() -> Optional[str]:
    """``rsync`` resolves on PATH after this call, for the rest of the CURRENT test: the real binary where the host has one (nothing is
    touched -- the gate's fixtures run byte-for-byte as before), else the fallback's directory is prepended to ``os.environ["PATH"]``
    so the in-process runners (``nice ionice -c3 rsync`` resolves through PATH), ``cluster_reads._rsync`` / ``stage_window`` and every
    child process that inherits the environment (``python3 -S -m sisteer.registry.stores push``, the landing scripts,
    ``stage_objects.py``) find it. Idempotent, and re-evaluated on every call: a ``monkeypatch.setenv("PATH")`` that restored an
    earlier PATH is repaired by the next world builder. Returns the fallback directory when this call installed it, None when an
    ``rsync`` already resolved (the real binary, or the fallback a call earlier in the same test put there).

    Scoped to the test that asked: the ``rsync_on_path`` fixture below and ``tests/conftest.py`` (after EVERY test, whichever builder
    called this) take the directory off PATH again and remove it with :func:`teardown_fallback`, so a later test that asks
    ``shutil.which("rsync")`` for its own skip (the colocate, mirror and grading-copy tests) reads the host, never a shim it did not
    request, whatever ran before it and in whatever order."""
    if have_exe(BIN_NAME):
        return None
    d = fallback_bin_dir()
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return d


def remove_fallback_from_path() -> bool:
    """Take the fallback's directory off ``os.environ["PATH"]`` wherever :func:`ensure_rsync_on_path` put it: every occurrence of
    every directory this process installed, and nothing else (the rest of PATH is left byte-for-byte, in order). Returns whether
    anything was removed; a no-op before the first installation and wherever the real binary served."""
    if not _bin_dirs:
        return False
    entries = os.environ.get("PATH", "").split(os.pathsep)
    kept = [e for e in entries if e not in _bin_dirs]
    if len(kept) == len(entries):
        return False
    os.environ["PATH"] = os.pathsep.join(kept)
    return True


def teardown_fallback() -> bool:
    """After a test: the fallback leaves PATH (:func:`remove_fallback_from_path`) and every wrapper directory this process made is removed
    from disk, so nothing the test asked for survives it -- neither on PATH nor under the system temp dir. :func:`installed_dirs` keeps
    naming the removed directories (the scope test reads it against PATH). Returns whether anything was removed, from PATH or from disk."""
    global _bin_dir
    removed = remove_fallback_from_path()
    for d in _bin_dirs:
        if os.path.lexists(d):
            shutil.rmtree(d, ignore_errors=True)
            removed = True
    _bin_dir = None
    return removed


def rsync_executable() -> str:
    """The absolute path of an ``rsync`` a fixture may exec directly (a shim's ``exec <rsync> "$@"``): the real binary where the host has
    one, else the fallback's wrapper. PATH is not touched: a shim holding the absolute path needs nothing on PATH, and the wrapper's
    directory lives until the test that asked ends (:func:`teardown_fallback`)."""
    found = shutil.which(BIN_NAME)
    return found if found else os.path.join(fallback_bin_dir(), BIN_NAME)


if pytest is not None:
    @pytest.fixture(autouse=True)
    def rsync_on_path():
        """Imported into a test module (``from _rsync_fallback import rsync_on_path  # noqa: F401``) it runs before every test there: the
        fallback goes on PATH where the host has no rsync, and nothing happens where it has one; when the test ends the fallback leaves
        PATH again and its wrapper directory is removed (``tests/conftest.py`` does the same after every test for a world builder that
        installed it without this fixture)."""
        ensure_rsync_on_path()
        try:
            yield
        finally:
            teardown_fallback()


# --------------------------------------------------------------------------------------------
# the copy
# --------------------------------------------------------------------------------------------

#: long options taking no value that the code under test emits and this copy accepts without effect
_NOOP_LONG = {"--partial", "--inplace", "--no-owner", "--no-group", "--no-perms", "--devices", "--specials", "--no-implied-dirs",
              "--protect-args", "--secluded-args", "--whole-file", "--no-whole-file", "--human-readable"}
#: long options taking a value (``--opt=V`` or ``--opt V``) accepted without effect
_NOOP_LONG_VALUE = {"--partial-dir", "--timeout", "--contimeout", "--bwlimit", "--modify-window"}


class UsageError(Exception):
    pass


class _Opts:
    def __init__(self) -> None:
        self.recursive = self.links = self.times = self.perms = self.quiet = self.checksum = False
        self.ignore_times = self.relative = self.prune_empty_dirs = self.delete = self.stats = self.dirs = False
        self.files_from: Optional[str] = None
        self.link_dests: List[str] = []
        self.filters: List[Tuple[str, str]] = []              # ("include" | "exclude", pattern) in command-line order
        self.chmod: List[str] = []
        self.operands: List[str] = []
        self.rsh: Optional[str] = None                         # -e CMD / --rsh=CMD: remembered for the refusal's wording, never run


def _parse(argv: Sequence[str]) -> _Opts:
    o = _Opts()
    args = list(argv)
    i = 0
    while i < len(args):
        a = args[i]
        i += 1
        if a == "--":
            o.operands.extend(args[i:])
            break
        if a.startswith("--"):
            name, eq, value = a.partition("=")
            if name in ("--server", "--sender"):
                raise UsageError("the fallback rsync copies locally; it speaks no wire protocol (--server)")
            if name in _NOOP_LONG:
                continue
            if name in _NOOP_LONG_VALUE:
                if not eq:
                    i += 1
                continue
            if name == "--rsh":
                if not eq:
                    if i >= len(args):
                        raise UsageError(f"{name} needs a value")
                    value = args[i]
                    i += 1
                o.rsh = value
                continue
            if name in ("--exclude", "--include", "--files-from", "--link-dest", "--chmod"):
                if not eq:
                    if i >= len(args):
                        raise UsageError(f"{name} needs a value")
                    value = args[i]
                    i += 1
                if name == "--exclude":
                    o.filters.append(("exclude", value))
                elif name == "--include":
                    o.filters.append(("include", value))
                elif name == "--files-from":
                    o.files_from = value
                elif name == "--link-dest":
                    o.link_dests.append(value)
                else:
                    o.chmod.append(value)
                continue
            flags = {"--recursive": "r", "--links": "l", "--times": "t", "--perms": "p", "--archive": "a", "--quiet": "q",
                     "--checksum": "c", "--prune-empty-dirs": "m", "--relative": "R", "--verbose": "v", "--dirs": "d",
                     "--ignore-times": "I", "--delete": "DELETE", "--stats": "STATS", "--delete-during": "DELETE", "--delete-after": "DELETE",
                     "--delete-before": "DELETE"}
            if name in flags and not eq:
                _set_flag(o, flags[name])
                continue
            raise UsageError(f"unknown option {a!r}")
        if a.startswith("-") and len(a) > 1:
            letters = a[1:]
            for k, ch in enumerate(letters):
                if ch == "e":                                   # -e CMD: the rest of the cluster or the next argument is the command
                    if k + 1 < len(letters):
                        o.rsh = letters[k + 1:]
                        break
                    if i >= len(args):
                        raise UsageError("-e needs a value")
                    o.rsh = args[i]
                    i += 1
                    break
                if ch in "rltpaqcmRvdIogDxzsHX":
                    _set_flag(o, ch)
                else:
                    raise UsageError(f"unknown option -{ch} in {a!r}")
            continue
        o.operands.append(a)
    if o.files_from is not None:
        o.relative = True
        o.dirs = True
    return o


def _set_flag(o: _Opts, ch: str) -> None:
    if ch == "a":
        o.recursive = o.links = o.times = o.perms = True        # -rlptgoD; owner/group/devices are no-ops here
    elif ch == "r":
        o.recursive = True
    elif ch == "l":
        o.links = True
    elif ch == "t":
        o.times = True
    elif ch == "p":
        o.perms = True
    elif ch == "q":
        o.quiet = True
    elif ch == "c":
        o.checksum = True
    elif ch == "m":
        o.prune_empty_dirs = True
    elif ch == "R":
        o.relative = True
    elif ch == "d":
        o.dirs = True
    elif ch == "I":
        o.ignore_times = True
    elif ch == "DELETE":
        o.delete = True
    elif ch == "STATS":
        o.stats = True
    # v o g D x z s H X: accepted without effect


_HOST_RE = re.compile(r"^([^/:]+):(?!:)(.*)$")
_DAEMON_RE = re.compile(r"^(?:rsync://|[^/:]+::)")


def local_hosts(environ: Optional[Dict[str, str]] = None) -> frozenset:
    """The hosts the opt-in names: :data:`LOCAL_HOSTS_ENV` split on commas, each token stripped, empty tokens dropped (an unset or empty
    variable opts nothing in)."""
    env = os.environ if environ is None else environ
    return frozenset(h.strip() for h in env.get(LOCAL_HOSTS_ENV, "").split(",") if h.strip())


def split_operand(operand: str) -> Tuple[str, Optional[str], str]:
    """rsync's own reading of an operand: ``("local", None, path)`` for a path (anything starting with ``/``, or with no colon before its
    first slash); ``("remote", host, path)`` for ``host:path`` (the host token exactly as written, a user prefix included; an empty path
    is the remote home, ``.``); ``("daemon", None, operand)`` for ``host::module`` or ``rsync://``."""
    if operand.startswith("/"):
        return "local", None, operand
    if _DAEMON_RE.match(operand):
        return "daemon", None, operand
    m = _HOST_RE.match(operand)
    if m:
        return "remote", m.group(1), m.group(2) or "."
    return "local", None, operand


class _Refused(Exception):
    """A remote or daemon operand this copy will not serve: ``rc`` and the stderr text, written before anything is created."""

    def __init__(self, rc: int, text: str) -> None:
        super().__init__(text)
        self.rc, self.text = rc, text


def _resolve_operands(operands: Sequence[str], rsh: Optional[str], allowed: frozenset) -> List[str]:
    """Every operand as the local path this copy reads it as, or :class:`_Refused` for the first one it cannot: a daemon operand always,
    a ``host:path`` whose host the opt-in does not name. The whole call is judged before any copy, so a refused call copies nothing."""
    out: List[str] = []
    for op in operands:
        kind, host, path = split_operand(op)
        if kind == "daemon":
            raise _Refused(DAEMON_REFUSED_RC, f"rsync (fallback): the test fallback speaks to no rsync daemon, so {op!r} is not reachable\n"
                           f"rsync error: error in socket IO (code {DAEMON_REFUSED_RC}) at clientserver.c(141) [fallback]\n")
        if kind == "remote" and host not in allowed:
            via = f" (its remote shell {rsh!r} is not run either)" if rsh is not None else ""
            raise _Refused(REMOTE_REFUSED_RC, f"rsync (fallback): the test fallback does not run a remote shell, so {op!r} is not reachable{via}; "
                           f"opt in with {LOCAL_HOSTS_ENV}={host} to read it as the local path {path!r}\n"
                           f"rsync error: error in rsync protocol data stream (code {REMOTE_REFUSED_RC}) at io.c(232) [fallback]\n")
        out.append(path)
    return out


# -- filter rules ---------------------------------------------------------------------------------------------------------------


class _Rule:
    __slots__ = ("include", "regex", "anchored", "full_path", "dir_only")

    def __init__(self, kind: str, pattern: str) -> None:
        self.include = kind == "include"
        pat = pattern
        self.dir_only = len(pat) > 1 and pat.endswith("/")
        if self.dir_only:
            pat = pat[:-1]
        self.anchored = pat.startswith("/")
        if self.anchored:
            pat = pat[1:]
        self.full_path = self.anchored or "/" in pat or "**" in pat
        self.regex = re.compile(_wild_to_regex(pat))

    def matches(self, rel: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        if self.anchored:
            return self.regex.fullmatch(rel) is not None
        if self.full_path:                                      # matched against the end of the path, at a directory boundary
            if self.regex.fullmatch(rel):
                return True
            return any(self.regex.fullmatch(rel[k + 1:]) for k, c in enumerate(rel) if c == "/")
        return self.regex.fullmatch(rel.rsplit("/", 1)[-1]) is not None


def _wild_to_regex(pat: str) -> str:
    """rsync's wildcard grammar: ``*`` (no slash), ``**`` (slashes too), ``?``, ``[...]``; a backslash escapes the next character ONLY in a
    pattern that holds a wildcard -- in one without, every character is literal (the rule ``stores._rsync_literal`` is written for)."""
    if not re.search(r"[*?\[]", pat):
        return re.escape(pat)
    out: List[str] = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == "\\" and i + 1 < len(pat):
            out.append(re.escape(pat[i + 1]))
            i += 2
            continue
        if c == "*":
            if pat.startswith("**", i):
                out.append(".*")
                i += 2
            else:
                out.append("[^/]*")
                i += 1
            continue
        if c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pat.find("]", i + 1)
            if j < 0:
                out.append(re.escape(c))
            else:
                body = pat[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return "".join(out)


def _filtered(rules: Sequence[_Rule], rel: str, is_dir: bool) -> bool:
    """True when ``rel`` is EXCLUDED: the first matching rule decides; no rule -> included."""
    for r in rules:
        if r.matches(rel, is_dir):
            return not r.include
    return False


# -- the transfer list --------------------------------------------------------------------------------------------------------------


class _Entry:
    __slots__ = ("rel", "src", "st", "implied", "top")

    def __init__(self, rel: str, src: str, st: os.stat_result, implied: bool = False, top: bool = False) -> None:
        self.rel, self.src, self.st, self.implied, self.top = rel, src, st, implied, top

    @property
    def is_dir(self) -> bool:
        return stat.S_ISDIR(self.st.st_mode)

    @property
    def is_link(self) -> bool:
        return stat.S_ISLNK(self.st.st_mode)

    @property
    def is_reg(self) -> bool:
        return stat.S_ISREG(self.st.st_mode)


class _Copy:
    def __init__(self, o: _Opts, stdout: TextIO, stderr: TextIO) -> None:
        self.o = o
        self.out, self.err = stdout, stderr
        self.rules = [_Rule(k, p) for k, p in o.filters]
        self.errors = 0
        self.entries: List[_Entry] = []
        self.seen: Dict[str, _Entry] = {}
        self.n_transferred = 0
        self.bytes_transferred = 0
        self.n_created = 0
        self.n_deleted = 0
        self.total_size = 0
        self.created_dirs: set = set()
        self.umask = os.umask(0)
        os.umask(self.umask)

    # -- diagnostics
    def error(self, msg: str) -> None:
        self.errors += 1
        self.err.write(f"rsync: {msg}\n")

    def warn(self, msg: str) -> None:
        if not self.o.quiet:
            self.err.write(f"{msg}\n")

    # -- listing
    def add(self, rel: str, src: str, st: os.stat_result, implied: bool = False, top: bool = False) -> Optional[_Entry]:
        rel = rel.strip("/")
        if rel in self.seen:
            return self.seen[rel]
        e = _Entry(rel, src, st, implied, top)
        self.entries.append(e)
        self.seen[rel] = e
        return e

    def add_implied(self, src_root: str, rel: str) -> None:
        """The directories above a relative path (``--relative`` / ``--files-from``), with their attributes, not subject to the filters."""
        parts = rel.strip("/").split("/")
        for k in range(1, len(parts)):
            prefix = "/".join(parts[:k])
            if prefix in self.seen:
                continue
            p = os.path.join(src_root, prefix) if src_root else "/" + prefix
            try:
                st = os.lstat(p)
            except OSError:
                st = os.stat_result((stat.S_IFDIR | 0o755, 0, 0, 1, 0, 0, 0, 0, 0, 0))
            self.add(prefix, p, st, implied=True)

    def add_tree(self, rel: str, src: str, top: bool) -> None:
        """One source argument (or one files-from line): the entry itself, then -- for a directory under -r -- its tree."""
        try:
            st = os.lstat(src)
        except OSError as exc:
            self.error(f"link_stat \"{src}\" failed: {exc.strerror} ({exc.errno})")
            return
        is_dir = stat.S_ISDIR(st.st_mode)
        if rel and _filtered(self.rules, rel, is_dir):
            return
        if is_dir:
            if not (self.o.recursive or self.o.dirs):
                self.warn(f"skipping directory {rel or src}")
                return
            self.add(rel, src, st, top=top)
            if self.o.recursive:
                self.walk(rel, src)
            return
        if stat.S_ISLNK(st.st_mode) and not self.o.links:
            self.warn(f"skipping non-regular file \"{rel}\"")
            return
        if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
            self.warn(f"skipping non-regular file \"{rel}\"")
            return
        self.add(rel, src, st, top=top)

    def walk(self, rel: str, src: str) -> None:
        try:
            names = sorted(os.listdir(src))
        except OSError as exc:
            self.error(f"opendir \"{src}\" failed: {exc.strerror} ({exc.errno})")
            return
        for name in names:
            child_src = os.path.join(src, name)
            child_rel = f"{rel}/{name}" if rel else name
            try:
                st = os.lstat(child_src)
            except OSError as exc:
                self.error(f"link_stat \"{child_src}\" failed: {exc.strerror} ({exc.errno})")
                continue
            is_dir = stat.S_ISDIR(st.st_mode)
            if _filtered(self.rules, child_rel, is_dir):
                continue
            if is_dir:
                self.add(child_rel, child_src, st)
                self.walk(child_rel, child_src)
            elif stat.S_ISLNK(st.st_mode):
                if self.o.links:
                    self.add(child_rel, child_src, st)
                else:
                    self.warn(f"skipping non-regular file \"{child_rel}\"")
            elif stat.S_ISREG(st.st_mode):
                self.add(child_rel, child_src, st)
            else:
                self.warn(f"skipping non-regular file \"{child_rel}\"")

    def prune_empty(self) -> None:
        keep = set()
        for e in self.entries:
            if e.is_dir:
                continue
            parts = e.rel.split("/")
            for k in range(1, len(parts)):
                keep.add("/".join(parts[:k]))
        self.entries = [e for e in self.entries if not e.is_dir or e.rel in keep or e.rel == ""]
        self.seen = {e.rel: e for e in self.entries}

    # -- modes
    def mode_for(self, src_mode: int, is_dir: bool, existing: Optional[int]) -> Optional[int]:
        """The mode a written entry ends with: ``-p`` copies the source's (after ``--chmod``); otherwise an existing entry keeps its own
        and a new one gets the source's (after ``--chmod``) masked by the umask. None: leave the entry's mode as it is."""
        m = stat.S_IMODE(src_mode)
        if self.o.chmod:
            m = _apply_chmod(m, is_dir, self.o.chmod)
        if self.o.perms:
            return m
        if existing is not None:
            return stat.S_IMODE(existing)
        return m & ~self.umask

    # -- comparison
    def same(self, src: str, sst: os.stat_result, other: str, ost: os.stat_result) -> bool:
        """rsync's quick check: equal size and equal whole-second mtime (``--checksum``: equal content instead of the mtime)."""
        if sst.st_size != ost.st_size:
            return False
        if self.o.checksum:
            return _sha256(src) == _sha256(other)
        return sst.st_mtime_ns // 1_000_000_000 == ost.st_mtime_ns // 1_000_000_000

    # -- writing
    def run(self, dest_dir: str, dest_file_rel: Optional[str]) -> None:
        os.makedirs(dest_dir, exist_ok=True)
        if self.o.delete:
            self.delete_extraneous(dest_dir)
        for e in self.entries:
            rel = dest_file_rel if (dest_file_rel is not None and not e.is_dir) else e.rel
            dst = os.path.join(dest_dir, rel) if rel else dest_dir
            if e.is_dir:
                self.write_dir(e, dst)
            elif e.is_link:
                self.write_link(e, dst)
            else:
                self.write_file(e, dst)
        # directory attributes last (writing into a directory moves its mtime), deepest first
        for e in sorted((e for e in self.entries if e.is_dir), key=lambda e: -len(e.rel)):
            dst = os.path.join(dest_dir, e.rel) if e.rel else dest_dir
            if not os.path.isdir(dst):
                continue
            try:
                if self.o.perms or dst in self.created_dirs:         # -p: the source's mode; a new directory: the source's masked by the umask
                    os.chmod(dst, self.mode_for(e.st.st_mode, True, None))
                if self.o.times:
                    os.utime(dst, ns=(e.st.st_atime_ns, e.st.st_mtime_ns))
            except OSError as exc:
                self.error(f"failed to set times on \"{dst}\": {exc.strerror} ({exc.errno})")

    def write_dir(self, e: _Entry, dst: str) -> None:
        try:
            st = os.lstat(dst)
        except FileNotFoundError:
            st = None
        if st is not None and not stat.S_ISDIR(st.st_mode):
            os.unlink(dst)                                      # a non-directory in the way of a directory is removed
            st = None
        if st is None:
            mode = self.mode_for(e.st.st_mode, True, None) or 0o755
            os.mkdir(dst)
            os.chmod(dst, mode | 0o700)                         # writable for the transfer; the final mode is set in the last pass
            self.created_dirs.add(dst)
            self.n_created += 1
        elif self.o.perms:
            try:
                os.chmod(dst, stat.S_IMODE(st.st_mode) | 0o700)
            except OSError:
                pass

    def write_link(self, e: _Entry, dst: str) -> None:
        target = os.readlink(e.src)
        try:
            st = os.lstat(dst)
        except FileNotFoundError:
            st = None
        if st is not None:
            if stat.S_ISDIR(st.st_mode):
                self.error(f"could not make way for new symlink: {e.rel}")
                return
            if stat.S_ISLNK(st.st_mode) and os.readlink(dst) == target:
                return
            os.unlink(dst)
        else:
            self.n_created += 1
        os.symlink(target, dst)
        if self.o.times:
            try:
                os.utime(dst, ns=(e.st.st_atime_ns, e.st.st_mtime_ns), follow_symlinks=False)
            except (NotImplementedError, OSError):
                pass

    def write_file(self, e: _Entry, dst: str) -> None:
        self.total_size += e.st.st_size
        try:
            dst_st: Optional[os.stat_result] = os.lstat(dst)
        except FileNotFoundError:
            dst_st = None
        if dst_st is not None and stat.S_ISDIR(dst_st.st_mode):
            self.error(f"could not make way for new regular file: {e.rel}")
            return
        existing_reg = dst_st if (dst_st is not None and stat.S_ISREG(dst_st.st_mode)) else None
        if existing_reg is not None and not self.o.ignore_times and self.same(e.src, e.st, dst, existing_reg):
            if self.o.perms:                                    # an up-to-date file still takes the attributes
                mode = self.mode_for(e.st.st_mode, False, None)
                if stat.S_IMODE(existing_reg.st_mode) != mode:
                    os.chmod(dst, mode)
            return
        if not self.o.ignore_times:
            for ld in self.o.link_dests:
                cand = os.path.join(ld, e.rel)
                try:
                    cst = os.stat(cand)
                except OSError:
                    continue
                if not stat.S_ISREG(cst.st_mode) or not self.same(e.src, e.st, cand, cst):
                    continue
                if self.o.perms and stat.S_IMODE(cst.st_mode) != self.mode_for(e.st.st_mode, False, None):
                    continue
                if dst_st is not None:
                    os.unlink(dst)
                else:
                    self.n_created += 1
                os.link(cand, dst)
                return
        # the copy: a temp file beside the destination, then one rename (a hard-linked destination is replaced, never written through)
        tmp = os.path.join(os.path.dirname(dst) or ".", f".{os.path.basename(dst)}.{os.getpid()}.rsync-tmp")
        try:
            with open(e.src, "rb") as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, 1024 * 1024)
            mode = self.mode_for(e.st.st_mode, False, existing_reg.st_mode if existing_reg is not None else None)
            if mode is not None:
                os.chmod(tmp, mode)
            if self.o.times:
                os.utime(tmp, ns=(e.st.st_atime_ns, e.st.st_mtime_ns))
            os.replace(tmp, dst)
        except OSError as exc:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self.error(f"failed to copy \"{e.src}\" to \"{dst}\": {exc.strerror} ({exc.errno})")
            return
        if dst_st is None:
            self.n_created += 1
        self.n_transferred += 1
        self.bytes_transferred += e.st.st_size

    def delete_extraneous(self, dest_dir: str) -> None:
        """``--delete``: within every directory of the transfer, what the destination holds and the source does not is removed -- an
        entry a filter excludes is protected, as rsync protects it without ``--delete-excluded``."""
        for e in [e for e in self.entries if e.is_dir]:
            d = os.path.join(dest_dir, e.rel) if e.rel else dest_dir
            try:
                names = os.listdir(d)
            except OSError:
                continue
            for name in names:
                rel = f"{e.rel}/{name}" if e.rel else name
                if rel in self.seen:
                    continue
                p = os.path.join(d, name)
                is_dir = os.path.isdir(p) and not os.path.islink(p)
                if _filtered(self.rules, rel, is_dir):
                    continue
                try:
                    if is_dir:
                        shutil.rmtree(p)
                    else:
                        os.unlink(p)
                    self.n_deleted += 1
                except OSError as exc:
                    self.error(f"delete_file: \"{p}\" failed: {exc.strerror} ({exc.errno})")

    # -- the summary
    def print_stats(self, t0: float) -> None:
        n_reg = sum(1 for e in self.entries if e.is_reg)
        n_dir = sum(1 for e in self.entries if e.is_dir)
        n_lnk = sum(1 for e in self.entries if e.is_link)
        elapsed = max(time.monotonic() - t0, 1e-3)
        sent = self.bytes_transferred + 60 * len(self.entries)
        received = 20 * self.n_transferred + 10
        w = self.out.write
        w(f"\nNumber of files: {_n(len(self.entries))} (reg: {_n(n_reg)}, dir: {_n(n_dir)}, link: {_n(n_lnk)})\n")
        w(f"Number of created files: {_n(self.n_created)}\n")
        w(f"Number of deleted files: {_n(self.n_deleted)}\n")
        w(f"Number of regular files transferred: {_n(self.n_transferred)}\n")
        w(f"Total file size: {_n(self.total_size)} bytes\n")
        w(f"Total transferred file size: {_n(self.bytes_transferred)} bytes\n")
        w(f"Literal data: {_n(self.bytes_transferred)} bytes\n")
        w("Matched data: 0 bytes\n")
        w("File list size: 0\n")
        w("File list generation time: 0.001 seconds\n")
        w("File list transfer time: 0.000 seconds\n")
        w(f"Total bytes sent: {_n(sent)}\n")
        w(f"Total bytes received: {_n(received)}\n")
        w(f"\nsent {_n(sent)} bytes  received {_n(received)} bytes  {(sent + received) / elapsed:,.2f} bytes/sec\n")
        speedup = (self.total_size / (sent + received)) if (sent + received) else 0.0
        w(f"total size is {_n(self.total_size)}  speedup is {speedup:,.2f}\n")


def _n(v: int) -> str:
    return f"{v:,}"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_WHO_BITS = {"r": {"u": 0o400, "g": 0o040, "o": 0o004}, "w": {"u": 0o200, "g": 0o020, "o": 0o002}, "x": {"u": 0o100, "g": 0o010, "o": 0o001}}
_WHO_ALL = {"u": 0o700, "g": 0o070, "o": 0o007}


def _apply_chmod(mode: int, is_dir: bool, specs: Sequence[str]) -> int:
    """``--chmod`` in chmod(1)'s symbolic or octal grammar, each clause optionally prefixed ``D`` (directories only) or ``F`` (files only)."""
    for spec in specs:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if item[0] in "DF":
                if (item[0] == "D") != is_dir:
                    continue
                item = item[1:]
            if re.fullmatch(r"[0-7]{3,4}", item):
                mode = int(item, 8)
                continue
            m = re.fullmatch(r"([ugoa]*)([+\-=])([rwxXst]*)", item)
            if not m:
                raise UsageError(f"--chmod: cannot parse {item!r}")
            who = m.group(1) or "a"
            who = "ugo" if "a" in who else who
            bits = 0
            for w in who:
                for c in m.group(3):
                    if c in _WHO_BITS:
                        bits |= _WHO_BITS[c][w]
                    elif c == "X" and (is_dir or mode & 0o111):
                        bits |= _WHO_BITS["x"][w]
            op = m.group(2)
            if op == "+":
                mode |= bits
            elif op == "-":
                mode &= ~bits
            else:
                mode = (mode & ~sum(_WHO_ALL[w] for w in who)) | bits
    return mode


def fallback_main(argv: Sequence[str], stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None,
                  environ: Optional[Dict[str, str]] = None) -> int:
    """The copy ``rsync <argv>`` would make, for the flag subset in the module docstring. Returns rsync's exit code: 0, 23 when any entry
    failed (the rest were copied), 1 on a usage error (an option this subset does not implement included -- loud by design), 12 for a
    ``host:path`` operand the opt-in (:data:`LOCAL_HOSTS_ENV`, read from ``environ`` or the process environment) does not name and 10
    for a daemon operand -- both before anything is created or copied."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    t0 = time.monotonic()
    try:
        o = _parse(argv)
    except UsageError as exc:
        stderr.write(f"rsync (fallback): {exc}\nrsync error: syntax or usage error (code 1) at main.c(1)\n")
        return 1
    if len(o.operands) < 2:
        stderr.write("rsync (fallback): need at least one source and a destination\nrsync error: syntax or usage error (code 1) at main.c(1)\n")
        return 1
    try:
        *sources, dest = _resolve_operands(o.operands, o.rsh, local_hosts(environ))
    except _Refused as refused:
        stderr.write(refused.text)
        return refused.rc
    copy = _Copy(o, stdout, stderr)
    # the transfer list
    if o.files_from is not None:
        if len(sources) != 1:
            stderr.write("rsync (fallback): --files-from takes exactly one source\nrsync error: syntax or usage error (code 1) at main.c(1)\n")
            return 1
        src_root = sources[0].rstrip("/") or "/"
        try:
            text = stdin.read() if o.files_from == "-" else open(o.files_from, "r", encoding="utf-8").read()
        except OSError as exc:
            stderr.write(f"rsync: failed to open files-from file {o.files_from}: {exc.strerror} ({exc.errno})\n"
                         "rsync error: error in file IO (code 11) at main.c(1)\n")
            return 11
        for line in text.splitlines():
            rel = line.strip()
            if not rel or rel == ".":
                continue
            while rel.startswith("./"):
                rel = rel[2:]
            rel = rel.strip("/")
            copy.add_implied(src_root, rel)
            copy.add_tree(rel, os.path.join(src_root, rel), top=True)
    else:
        for s in sources:
            if o.relative:
                if "/./" in s:
                    base, _sep, rel = s.partition("/./")
                    rel = rel.strip("/")
                    src = os.path.join(base, rel) if rel else base
                    if rel:
                        copy.add_implied(base, rel)
                else:
                    rel = s.strip("/")
                    src = s
                    copy.add_implied("", rel)
                if rel == "":
                    copy.add_tree("", src, top=True)
                else:
                    copy.add_tree(rel, src, top=True)
            else:
                if s.endswith("/") and os.path.isdir(s):
                    copy.add_tree("", s.rstrip("/") or "/", top=True)
                else:
                    copy.add_tree(os.path.basename(s.rstrip("/")), s.rstrip("/"), top=True)
    if o.prune_empty_dirs:
        copy.prune_empty()
    # the destination: a directory, unless ONE non-directory source goes to a path that is not (and does not name) a directory
    dest_file_rel: Optional[str] = None
    if dest.endswith("/") or os.path.isdir(dest) or len(copy.entries) != 1 or copy.entries[0].is_dir or o.files_from is not None or o.relative:
        dest_dir = dest.rstrip("/") or "/"
    else:
        dest_dir = os.path.dirname(dest.rstrip("/")) or "."
        dest_file_rel = os.path.basename(dest.rstrip("/"))
        if not os.path.isdir(dest_dir):
            stderr.write(f"rsync: mkdir \"{dest_dir}\" failed: No such file or directory (2)\nrsync error: error in file IO (code 11) at main.c(1)\n")
            return 11
    copy.run(dest_dir, dest_file_rel)
    if o.stats:
        copy.print_stats(t0)
    if copy.errors:
        stderr.write("rsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1338) [fallback]\n")
        return 23
    return 0


if __name__ == "__main__":
    sys.exit(fallback_main(sys.argv[1:], sys.stdin))
