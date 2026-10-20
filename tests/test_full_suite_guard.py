"""The whole-suite guard: ``tests/_full_suite_guard.py`` (the decision, pure) hooked from ``tests/conftest.py``'s
``pytest_configure``. The rule it enforces: the full suite never runs on the shared grading host —
two whole-suite runs there took 2.5 h and 3 h 23 min and pushed load to 22–34 — it runs on the Slurm landing gate
(``gate_slurm.sh``, a 40-shard CPU array) or the Modal CPU suite, and a deliberate exception sets
``SISTEER_ALLOW_FULL_SUITE=1``.

Unit tests drive the pure function with synthetic trees and env mappings (no pytest is spawned). The end-to-end tests
spawn ONE pytest each, with an env scrubbed of every allowance (the Slurm gate's own ``suite_env`` sets the override,
so a child inheriting it would never refuse), and assert the refusal lands BEFORE collection — exit code 4
(``pytest.ExitCode.USAGE_ERROR``), nothing collected — so each costs a process start, seconds, never a whole-tree
collection. The allow path is exercised end to end on a synthetic 8-file tree carrying the real guard module and a
conftest that installs its hook — never the real conftest, whose ``pytest_plugins`` name other tests/-local modules
(``registry_live_fixtures``) a copy cannot import off this tree — and never on this repository's ~1,000 modules. That the
real conftest installs the very same hook is pinned separately, so the synthetic proof is about the hook the repo runs."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from _full_suite_guard import (
    ALLOW_ENV,
    DEFAULT_MAX_FILES,
    FILE_LIMIT_ENV,
    GATE_SCRIPT,
    allowed_environment,
    count_test_files,
    file_limit,
    full_suite_refusal,
    refuse_whole_suite_run,
)

REPO = Path(__file__).resolve().parents[1]
#: every variable that would let a child run through; the e2e children never inherit one
SCRUBBED = (ALLOW_ENV, FILE_LIMIT_ENV, "SLURM_JOB_ID", "MODAL_TASK_ID", "MODAL_IS_REMOTE", "PYTEST_XDIST_WORKER",
            "PYTEST_ADDOPTS")
NO_ALLOWANCE: dict = {}
TEST_BODY = "def test_ok():\n    pass\n"


def _tree(root: Path, n: int, sub: str = "tests") -> Path:
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"test_f{i:04d}.py").write_text(TEST_BODY)
    return d


# ----------------------------------------------------------------------------------------------------------------------
# the decision, as a pure function


def test_a_whole_tree_directory_arg_is_refused_outside_slurm_and_modal(tmp_path):
    _tree(tmp_path, DEFAULT_MAX_FILES + 1)
    msg = full_suite_refusal(["tests"], tmp_path, NO_ALLOWANCE)
    assert msg is not None
    assert f"names {DEFAULT_MAX_FILES + 1} test files (limit {DEFAULT_MAX_FILES})" in msg
    assert GATE_SCRIPT in msg and "<branch-or-sha>" in msg, "the message names the gate and how to launch it"
    assert "cpu_suite.py" in msg, "and the Modal suite"
    assert "-k / -m" in msg, "and says a filter does not make a whole-tree run cheap"
    assert f"{ALLOW_ENV}=1" in msg and FILE_LIMIT_ENV in msg, "and the override and the limit knob"


def test_the_default_limit_is_a_boundary_at_300(tmp_path):
    assert DEFAULT_MAX_FILES == 300
    _tree(tmp_path, 300)
    assert full_suite_refusal(["tests"], tmp_path, NO_ALLOWANCE) is None, "exactly the limit is allowed"
    (tmp_path / "tests" / "test_one_more.py").write_text(TEST_BODY)
    assert full_suite_refusal(["tests"], tmp_path, NO_ALLOWANCE) is not None


@pytest.mark.parametrize("env, reason", [
    ({"SLURM_JOB_ID": "1234567"}, "slurm"),
    ({"MODAL_TASK_ID": "ta-01AAAAAAAAAAAAAAAAAAAAAAAA"}, "modal"),
    ({"MODAL_IS_REMOTE": "1"}, "modal"),
    ({ALLOW_ENV: "1"}, "override"),
    ({ALLOW_ENV: "true"}, "override"),
    ({"PYTEST_XDIST_WORKER": "gw0"}, "xdist-worker"),
])
def test_the_same_invocation_is_allowed_inside_slurm_modal_a_worker_or_under_the_override(tmp_path, env, reason):
    _tree(tmp_path, DEFAULT_MAX_FILES + 1)
    assert allowed_environment(env) == reason
    assert full_suite_refusal(["tests"], tmp_path, env) is None


@pytest.mark.parametrize("value", ["0", "", "no", "false"])
def test_a_falsy_override_does_not_allow(tmp_path, value):
    _tree(tmp_path, DEFAULT_MAX_FILES + 1)
    assert allowed_environment({ALLOW_ENV: value}) is None
    assert full_suite_refusal(["tests"], tmp_path, {ALLOW_ENV: value}) is not None


def test_a_grading_host_environment_has_no_allowance():
    assert allowed_environment({"USER": "someone", "HOME": "/home/someone", "PATH": "/usr/bin"}) is None
    assert allowed_environment({"MODAL_IS_REMOTE": "0", "SLURM_JOB_ID": ""}) is None, "empty / zero are not set"


def test_a_targeted_list_below_the_limit_is_allowed(tmp_path):
    d = _tree(tmp_path, 120)
    files = [f"tests/{p.name}" for p in sorted(d.iterdir())][:115]      # the largest targeted run seen so far
    assert count_test_files(files, tmp_path) == 115
    assert full_suite_refusal(files, tmp_path, NO_ALLOWANCE) is None


def test_node_id_args_count_as_their_file(tmp_path):
    _tree(tmp_path, 3)
    args = ["tests/test_f0000.py::test_ok", "tests/test_f0000.py::test_other",
            "tests/test_f0001.py::TestGroup::test_case", "tests/test_f0001.py"]
    assert count_test_files(args, tmp_path) == 2, "two files, however many node ids point into them"


def test_a_directory_and_a_node_id_into_it_are_not_double_counted(tmp_path):
    _tree(tmp_path, 4)
    assert count_test_files(["tests", "tests/test_f0002.py::test_ok", "./tests/../tests"], tmp_path) == 4


def test_directory_expansion_follows_pytests_default_python_files_and_skips_hidden_and_cache_dirs(tmp_path):
    d = _tree(tmp_path, 2)
    (d / "sub").mkdir()
    (d / "sub" / "extra_test.py").write_text(TEST_BODY)                 # *_test.py counts
    (d / "sub" / "helper.py").write_text("X = 1\n")                     # not a test file
    (d / "conftest.py").write_text("")                                   # nor a conftest
    (d / ".hidden").mkdir()
    (d / ".hidden" / "test_hidden.py").write_text(TEST_BODY)            # hidden dirs are not descended
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "test_cached.py").write_text(TEST_BODY)        # nor caches
    (d / "node_modules").mkdir()
    (d / "node_modules" / "test_vendored.py").write_text(TEST_BODY)     # nor vendored trees
    assert count_test_files(["tests"], tmp_path) == 3


def test_option_looking_and_empty_args_are_ignored(tmp_path):
    """``config.args`` holds positionals only, but the counter is defensive: a dash-prefixed or empty entry is not a
    path. A bare word IS one (pytest would report it as a missing file), so it counts — nothing is guessed at."""
    _tree(tmp_path, 2)
    assert count_test_files(["-k", "--tb=short", "", "tests"], tmp_path) == 2
    assert count_test_files(["-k", "not_a_path", "tests"], tmp_path) == 3


def test_an_absolute_path_arg_is_not_re_rooted(tmp_path):
    d = _tree(tmp_path, 5)
    assert count_test_files([str(d)], Path("/nonexistent/elsewhere")) == 5


def test_the_limit_env_override(tmp_path):
    _tree(tmp_path, 3)
    assert file_limit(NO_ALLOWANCE) == DEFAULT_MAX_FILES
    assert file_limit({FILE_LIMIT_ENV: " 7 "}) == 7
    assert full_suite_refusal(["tests"], tmp_path, {FILE_LIMIT_ENV: "2"}) is not None, "3 files over a limit of 2"
    assert "(limit 2)" in full_suite_refusal(["tests"], tmp_path, {FILE_LIMIT_ENV: "2"})
    assert full_suite_refusal(["tests"], tmp_path, {FILE_LIMIT_ENV: "3"}) is None, "3 files at a limit of 3"


def test_a_malformed_limit_refuses_loudly_instead_of_becoming_the_default(tmp_path):
    _tree(tmp_path, 1)
    with pytest.raises(ValueError, match=FILE_LIMIT_ENV):
        file_limit({FILE_LIMIT_ENV: "three hundred"})
    msg = full_suite_refusal(["tests"], tmp_path, {FILE_LIMIT_ENV: "three hundred"})
    assert msg is not None and FILE_LIMIT_ENV in msg and "not an integer" in msg


def test_the_environment_is_checked_before_the_walk(tmp_path):
    """An allowed host never pays for the directory walk — a nonexistent path is not even looked at."""
    assert full_suite_refusal(["/nonexistent/tree"], tmp_path, {ALLOW_ENV: "1"}) is None


# ----------------------------------------------------------------------------------------------------------------------
# the sanctioned runner declares itself


def _cpu_suite():
    """``scripts/modal/cpu_suite.py`` under the fake ``modal`` that ``tests/test_cpu_suite_prune.py`` installs (the
    loader every cpu_suite test shares; ``modal`` is not installed in the suite venv)."""
    spec = importlib.util.spec_from_file_location("_cpu_suite_prune_test_for_guard", REPO / "tests/test_cpu_suite_prune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load()


def test_the_gates_pytest_env_carries_the_override_so_shard_count_and_env_scrubbing_never_matter(tmp_path):
    """Both gates build their pytest env through ``cpu_suite.suite_env`` — the Modal container and the Slurm shards
    (``gate_task.py`` imports it from the tree under test, then scrubs ``SLURM_*`` from the env on purpose). An
    unsharded run names ``tests`` whole and a low ``--shards`` crosses the file limit, so the runner must say so."""
    cs = _cpu_suite()
    assert cs.FULL_SUITE_ALLOW_ENV == ALLOW_ENV, "one spelling in the runner and the guard"
    env = cs.suite_env({"PATH": "/usr/bin"}, tmp_path / "tree", spool_root=tmp_path)
    assert env[ALLOW_ENV] == "1"
    assert allowed_environment(env) == "override"
    # the whole-suite command, as the unsharded Modal run issues it, passes on that env alone
    argv = cs.pytest_argv("python", "/tmp/junit.xml", None)
    assert argv[3] == "tests"
    _tree(tmp_path / "tree", DEFAULT_MAX_FILES + 1)
    assert full_suite_refusal([argv[3]], tmp_path / "tree", env) is None
    assert full_suite_refusal([argv[3]], tmp_path / "tree", {k: v for k, v in env.items() if k != ALLOW_ENV}) is not None


# ----------------------------------------------------------------------------------------------------------------------
# end to end: the real conftest refuses before collection


def _run_pytest(cwd: Path, *args: str, env_extra: dict | None = None, timeout: int = 240) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
    env["PYTHONPATH"] = str(REPO / "src")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "addopts=", "-q", *args],
                          cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)


@pytest.mark.parametrize("args", [("--collect-only", "tests"), ("--collect-only",)],
                         ids=["directory-arg", "bare-invocation-resolves-to-testpaths"])
def test_e2e_this_repositorys_conftest_refuses_a_whole_tree_run_before_collecting_anything(args):
    """A whole-tree ``--collect-only`` at this repo's root, and the bare invocation (``testpaths = ["tests"]`` in
    pyproject), both stop in ``pytest_configure``: exit code 4, the refusal on stderr, and NO module collected — the
    child would otherwise import ~1,000 test modules, which is exactly the cost this guard exists to refuse."""
    out = _run_pytest(REPO, *args)
    assert out.returncode == int(pytest.ExitCode.USAGE_ERROR), (out.returncode, out.stdout[-1500:], out.stderr[-1500:])
    assert "Refusing a whole-suite pytest run" in out.stderr, out.stderr[-1500:]
    assert f"(limit {DEFAULT_MAX_FILES})" in out.stderr
    assert GATE_SCRIPT in out.stderr and f"{ALLOW_ENV}=1" in out.stderr
    assert "collected" not in out.stdout and "::" not in out.stdout, "nothing was collected"


def test_e2e_help_passes_through_the_guard():
    """``--help`` configures without collecting; a refusal there would be a papercut with no cost behind it."""
    out = _run_pytest(REPO, "--help")
    assert out.returncode == 0, (out.returncode, out.stderr[-1500:])
    assert "usage:" in out.stdout


#: the synthetic tree's conftest: the hook the real conftest installs, through the real guard module copied in beside it,
#: and nothing else the real conftest wires. A copy of the whole conftest broke here when its ``pytest_plugins`` came to
#: name ``registry_live_fixtures``, a tests/-local module the synthetic tree does not carry (main at ddd09e258,
#: 2026-09-29: one change copied the conftest, an earlier one had given it the plugin; each green alone, red together).
SYNTHETIC_CONFTEST = "from _full_suite_guard import refuse_whole_suite_run\n\n\ndef pytest_configure(config):\n    refuse_whole_suite_run(config)\n"


def _synthetic_project(tmp_path: Path, n_files: int) -> Path:
    """A tiny project carrying THIS repo's guard module byte for byte, a conftest that is exactly its hook
    (:data:`SYNTHETIC_CONFTEST`), ``testpaths = tests`` and ``n_files`` trivial modules — the real hook on a tree small
    enough to collect, so the allow paths can be proved end to end."""
    root = tmp_path / "proj"
    d = _tree(root, n_files)
    (root / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    shutil.copy(REPO / "tests" / "_full_suite_guard.py", d / "_full_suite_guard.py")
    (d / "conftest.py").write_text(SYNTHETIC_CONFTEST)
    return root


def test_this_repositorys_conftest_installs_the_guards_hook():
    """The synthetic tree installs ``refuse_whole_suite_run`` as ``pytest_configure``; the real conftest must install the
    SAME function, or the tree below proves a hook the repository does not run. The conftest is loaded as a plain module
    (its ``pytest_plugins`` is a list and its fixtures are definitions — nothing registers) and its hook is called with
    the guard patched out, so this pins the delegation and not the guard's own decision (the unit tests above do that)."""
    spec = importlib.util.spec_from_file_location("_conftest_under_test_for_guard", REPO / "tests" / "conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    assert conftest.refuse_whole_suite_run is refuse_whole_suite_run
    config = object()
    with mock.patch.object(conftest, "refuse_whole_suite_run") as hook:
        conftest.pytest_configure(config)
    hook.assert_called_once_with(config)


def test_e2e_on_a_synthetic_tree_the_override_and_the_limit_knob_let_a_bare_run_through(tmp_path):
    root = _synthetic_project(tmp_path, 8)
    refused = _run_pytest(root, "--collect-only", env_extra={FILE_LIMIT_ENV: "5"})
    assert refused.returncode == int(pytest.ExitCode.USAGE_ERROR), (refused.stdout[-800:], refused.stderr[-800:])
    assert "names 8 test files (limit 5)" in refused.stderr
    allowed = _run_pytest(root, "--collect-only", env_extra={FILE_LIMIT_ENV: "5", ALLOW_ENV: "1"})
    assert allowed.returncode == 0, (allowed.stdout[-800:], allowed.stderr[-800:])
    assert "8 tests collected" in allowed.stdout
    raised = _run_pytest(root, "--collect-only", env_extra={FILE_LIMIT_ENV: "8"})
    assert raised.returncode == 0 and "8 tests collected" in raised.stdout, (raised.stdout[-800:], raised.stderr[-800:])
