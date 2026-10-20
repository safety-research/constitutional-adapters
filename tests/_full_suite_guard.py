"""Refuse a whole-suite pytest run anywhere but the landing gate.

``tests/conftest.py``'s ``pytest_configure`` is :func:`refuse_whole_suite_run`: it calls :func:`full_suite_refusal`
before collection and raises ``pytest.UsageError`` on a refusal, so a refused run costs a process start and one
directory walk, not the ~1,000-module import that is the expensive part of a whole-suite run. The hook body lives here,
not in the conftest, so a synthetic tree can install the SAME hook through this one module
(``tests/test_full_suite_guard.py``): the real conftest also wires ``pytest_plugins`` naming other tests/-local modules,
which a copy of it cannot import anywhere else.

Why: two whole-suite runs on the shared 8-core the grading host (2.5 h and 3 h 23 min) pushed its load
to 22–34 and free swap to 384 MB, slowing every other session's work. The standing rule since 2026-09-26 — the full
suite never runs on the grading host; the Slurm landing gate (:data:`GATE_SCRIPT`, a 40-shard CPU array) replaces it —
lived only in memory files, which subagents did not carry, so it is enforced here, in code.

What counts as "whole suite": the number of test files the invocation names. pytest hands the guard its resolved
positional args (``config.args``: the paths given, else the ``testpaths`` of a bare invocation, else the invocation
directory); a node id counts as its file and a directory as every ``test_*.py`` / ``*_test.py`` beneath it (pytest's
default ``python_files``; pyproject sets none). Filesystem only, nothing is imported. ``-k`` / ``-m`` do not change the
count on purpose: a filtered whole-tree run still imports and collects every module. The limit is
:data:`DEFAULT_MAX_FILES` (env override :data:`FILE_LIMIT_ENV`): the tree holds ~1,000–1,200 test files, the largest
legitimate targeted run seen so far named ~115, the Modal suite's 10-way shards ~104 each and the Slurm gate's
40-way shards ~26 each — 300 sits well above every targeted run and well below the tree.

Where the whole suite may run (:func:`allowed_environment`, never refused):

- the explicit override ``SISTEER_ALLOW_FULL_SUITE=1`` (:data:`ALLOW_ENV`). It is also what the sanctioned runner
  sets in its pytest subprocess env: ``scripts/modal/cpu_suite.py`` ``suite_env`` builds the env of BOTH the Modal
  CPU suite and the Slurm gate's shards (``gate_task.py`` imports it from the tree under test), so the gates pass
  whatever their shard count, and a substrate probe is not what admits them;
- inside a Slurm job (``SLURM_JOB_ID``): how a hand-run suite inside an allocation looks. The gate's own shards do
  NOT rely on this — ``gate_task.py`` scrubs ``SLURM_*`` from its pytest env deliberately so the suite sees a plain
  CPU box, as on Modal; they pass on the override above (and each names only ~26 files anyway);
- inside a Modal container: ``MODAL_TASK_ID`` — modal-client 1.5.4 reads ``os.environ["MODAL_TASK_ID"]`` in-container
  (``modal/experimental/flash.py``), ``scripts/modal/attacker_pool.py`` refuses a canary that has none, and
  ``sisteer.registry.emit.substrate_from_env`` keys its Modal substrate on it — or ``MODAL_IS_REMOTE=1``, the client's
  own "this process is in the container" test (``modal/config.py`` ``_is_remote``). ``suite_env`` copies the
  container's environ and pops only registry knobs, so both reach the pytest subprocess;
- a pytest-xdist worker (``PYTEST_XDIST_WORKER``): the controller already ruled on the same args, and a worker raising
  would report the refusal once per worker. xdist is not installed in this repo's venv; the branch costs a line and
  keeps the guard right if it ever is.
"""
from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

#: refuse when an invocation names more test files than this; :data:`FILE_LIMIT_ENV` overrides it per run
DEFAULT_MAX_FILES = 300
FILE_LIMIT_ENV = "SISTEER_FULL_SUITE_MAX_FILES"
#: the deliberate exception (``=1``); also set by ``cpu_suite.suite_env`` for the gates' pytest subprocess
ALLOW_ENV = "SISTEER_ALLOW_FULL_SUITE"
#: pytest's default ``python_files`` (pyproject.toml sets none)
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")
#: never descended when a directory arg is expanded: hidden dirs (.venv, .git), caches, vendored trees
SKIP_DIRS = frozenset({"__pycache__", "node_modules", "venv"})
#: the Slurm landing gate (this repository's copy); its header documents the flags
GATE_SCRIPT = "scripts/registry/landing/gate_slurm.sh"
GATE_EXAMPLE = f"{GATE_SCRIPT} <branch-or-sha>"
MODAL_EXAMPLE = "python -m modal run -n cpu-suite-<sha7> scripts/modal/cpu_suite.py --stage --sha <sha> --go"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def allowed_environment(env: Mapping[str, str]) -> str | None:
    """Why this environment may run the whole suite — ``override`` / ``xdist-worker`` / ``slurm`` / ``modal`` —
    or None when it may not (the module docstring has the evidence behind each variable)."""
    if _truthy(env.get(ALLOW_ENV)):
        return "override"
    if env.get("PYTEST_XDIST_WORKER"):
        return "xdist-worker"
    if env.get("SLURM_JOB_ID"):
        return "slurm"
    if env.get("MODAL_TASK_ID") or env.get("MODAL_IS_REMOTE") == "1":
        return "modal"
    return None


def file_limit(env: Mapping[str, str]) -> int:
    """:data:`DEFAULT_MAX_FILES`, or the integer in :data:`FILE_LIMIT_ENV`. A non-integer value raises ValueError:
    a mistyped limit must not silently become the default."""
    raw = (env.get(FILE_LIMIT_ENV) or "").strip()
    if not raw:
        return DEFAULT_MAX_FILES
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{FILE_LIMIT_ENV}={raw!r} is not an integer") from None


def _is_test_file(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in TEST_FILE_PATTERNS)


def count_test_files(args: Iterable[str], invocation_dir: Path) -> int:
    """How many distinct test files ``args`` name, resolved against ``invocation_dir``: a node id
    (``tests/test_x.py::test_y``) is its file; a directory is every :data:`TEST_FILE_PATTERNS` file beneath it, hidden
    and :data:`SKIP_DIRS` directories not descended; any other path counts as one (pytest reports a missing file
    itself); an option-looking arg is ignored. Filesystem only — nothing is imported."""
    seen: set[str] = set()
    for arg in args:
        if not arg or arg.startswith("-"):
            continue
        path = Path(arg.split("::", 1)[0])
        if not path.is_absolute():
            path = Path(invocation_dir) / path
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
                seen.update(os.path.normpath(os.path.join(root, f)) for f in files if _is_test_file(f))
        else:
            seen.add(os.path.normpath(str(path)))
    return len(seen)


def refusal_message(n_files: int, limit: int) -> str:
    return "\n".join((
        f"Refusing a whole-suite pytest run on this host: the invocation names {n_files} test files (limit {limit}).",
        "Whole-suite runs happen on the Slurm landing gate (a 40-shard CPU array), never on a shared box:",
        f"    {GATE_EXAMPLE}        (--shards N / --baseline / --no-wait: see the script's header)",
        f"or on the Modal CPU suite:  {MODAL_EXAMPLE}",
        "Locally, run a targeted subset: pytest tests/test_<area>.py [tests/test_<area>.py::test_<case> ...].",
        "-k / -m do not make a whole-tree run cheap: every module is still imported and collected, which is the cost.",
        f"A deliberate exception sets {ALLOW_ENV}=1; a different limit, {FILE_LIMIT_ENV}=<n>.",
    ))


def full_suite_refusal(args: Iterable[str], invocation_dir: Path, env: Mapping[str, str]) -> str | None:
    """The refusal message when ``args`` would run the whole suite somewhere it may not, else None. Pure: reads its
    arguments and the filesystem under the named paths, nothing else. The environment is checked first, so an allowed
    host never pays for the walk."""
    if allowed_environment(env):
        return None
    try:
        limit = file_limit(env)
    except ValueError as e:
        return f"{e} — the whole-suite guard cannot decide; unset it or give an integer"
    n_files = count_test_files(args, invocation_dir)
    if n_files <= limit:
        return None
    return refusal_message(n_files, limit)


def refuse_whole_suite_run(config) -> None:
    """The ``pytest_configure`` hook ``tests/conftest.py`` installs: refuse a whole-suite run outside the landing gate
    BEFORE anything is collected. ``--help`` and ``--markers`` configure without collecting and pass; ``--collect-only``
    over the tree imports every module and is refused like a run. ``config.args`` is pytest's resolved list: the paths
    named, else ``testpaths`` for a bare invocation, else the invocation directory. A refusal is ``pytest.UsageError``
    (exit code 4, nothing collected). One function rather than a body in the conftest, so a synthetic tree installs the
    hook the repository runs — through this module alone — instead of a copy of the conftest and everything it wires."""
    if config.getoption("help", default=False) or config.getoption("markers", default=False):
        return
    refusal = full_suite_refusal(config.args, Path(config.invocation_params.dir), os.environ)
    if refusal:
        raise pytest.UsageError(refusal)
