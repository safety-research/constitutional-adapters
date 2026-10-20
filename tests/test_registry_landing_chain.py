"""The results-registry landing chain in ``scripts/registry/landing/``.

The zero-Modal invariant, four ways: (1) no line of the chain matches a Modal launcher (the CLI's run / volume verbs, the
Modal venv, the retired gates launcher directory, the Modal-era suite module's staging flag; MODAL_CALL, composed from fragments so
this file is not a hit of the scan it describes) and no chain file imports or binds the name ``modal`` at all -- the suite's
functions are ``gate_lib``'s own and the MagicMock stub is gone; (2) every shell file parses
(``bash -n``); (3) ``land.sh --plan`` lists exactly merge / verdict(reuse|waiver|slurm-gate) / live-db / push / pr / merge /
cleanup; (4) ``gate_lib`` imports under an import hook that raises on a real ``modal`` and exposes the suite functions, whose
partition, parser and merge are pinned on synthetic inputs.

Chain parity and tooling: ``verdict_local.py`` carries the operating copy's rules (the collection-flake pair retired; a
vanished-only set LANDs under any layout); ``land.sh`` reuses a recorded verdict for an ANCESTOR with the identical
tree and runs the no-Modal scan after the merge (report / refuse / off); ``recover.sh`` lands a revert-or-fix branch with the
replica side pinned to the current good generation's tree (LIVE_DB_REPLICA_TREE), rolling CURRENT back first when asked.

Then the chain is REHEARSED against shims: ``baseline_refresh.sh`` with ``git`` / ``ssh`` / ``gate_slurm.sh`` / ``mirror-repos.sh``
replaced by PATH and env shims that record their calls and script the queue's transitions (an absent commit refuses before any
submission; submit; PENDING → RUNNING → COMPLETED → collected; NODE_FAIL → resubmitted with --exclude, at most three attempts; a
plain failure → FAILED with the .out tails; the 4-h bound, per sha and in the UNKNOWN state too; two concurrent ticks submit once;
a replacement whose own submission fails keeps its attempt number and exclusions; the venv job the array depends on is watched
with it; an orphaned claim is dropped, or adopts the jobs it left in the queue; a partial submission (an array without its
finalizer) is recorded, canceled and confirmed gone before anything resubmits; a sacct row is trusted only by JobID AND JobName,
terminal, and read twice the same; the unit's ExecStart carries GH_TOKEN to git); ``gate_slurm.sh``'s remote block prints the
PARTIAL line when the finalizer's sbatch fails; ``verdict_local.py`` on synthetic summaries (a collection-flake pair lands with a
caveat; a real introduced node holds); ``land.sh`` end to end on a real temporary git origin with ``gh``, pytest and the gate
shimmed (the Slurm gate is the only fall-through; reuse and waiver skip it; a HOLD verdict pushes nothing; a failed baseline
refuses; the ``landing-gate`` commit status — ruleset 24155420's required context — is posted by ``gate_status.sh`` on the pushed head,
after the push and before the PR, once per round, and a failed POST stops the landing before any PR or merge; a refused merge leaves
the worktree; a main that moved under the gated tip is re-merged and re-gated before the merge; a writer
that moved main between the check and the merge is refused by the ruleset's strict condition (exit 8, the cause named) when the ruleset
is active, and with it disabled is caught on the merge commit's first parent, gated post hoc and reported RACED; the
check-and-merge runs under a landing-wide lock). An orphaned claim with nothing in the queue is reconciled against the accounting and
the cluster results dir before it is dropped. Finally the shared live snapshot on synthetic databases (equal text; a pending
migration; drift refused; one read transaction; registered once, one scan) and the two measurement scripts on synthetic logs.
Nothing here reaches a cluster, Modal, GitHub or the live registry.

The targeted battery (``battery_slurm.sh`` / ``battery.sbatch`` / ``battery_collect.py``): the plan allocates
the requested files into K disjoint duration-bounded lists (LPT; unknown files take the median; one file over the budget runs alone and
is named), the array task runs only its own list through the tree's own ``gate_task.py`` under ``<label>/t<i>`` and refuses a foreign
plan or index, and the collection is exact-K: a missing, duplicate, wrong-sha, extra, re-pathed, non-terminal, empty or report-less
record is ``battery incomplete`` and never a verdict. The driver is rehearsed end to end against local shims (ssh runs the command
here, sbatch runs every array task synchronously): two tasks with differing outcomes give one FAIL verdict naming exactly the failing
node with both records present and distinct; a used label is refused; the file list is checked before the first ssh; a task that ran
out of its walltime re-plans the whole list with K+1 under ``<label>-r2``; every attempt keeps its own plan, jobs and records
(the re-plan's ``.jobs`` names its root) and ``--collect-only`` walks the chain from the root, restoring the request from the saved
plan and submission record so a battery submitted with ``--no-wait`` gets the same re-plan a waiting one would.
"""
from __future__ import annotations

import ast
import fcntl
import importlib.abc
import importlib.util
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, Tuple
from unittest import mock

import pytest

import _rsync_fallback as RF  # the battery's copies where this host has no rsync: the fallback, never a skip (tests/_rsync_fallback.py)

REPO = Path(__file__).resolve().parent.parent
LANDING = REPO / "scripts" / "registry" / "landing"
MEASURE = REPO / "scripts" / "registry" / "measure"
CHAIN_FILES = sorted(p for p in LANDING.iterdir() if p.is_file() and not p.name.startswith("."))
SHELL_FILES = [p for p in CHAIN_FILES if p.suffix in (".sh", ".sbatch")]
_M = "modal"   # the package's name, kept out of every literal pattern below so this file passes the scan it describes
MODAL_CALL = re.compile(rf"{_M} run|{_M} volume|venvs/{_M}|gates/" + "land/" + rf"|cpu_{'suite'}\.py --stage")
SHA_A = "0123456789abcdef0123456789abcdef01234567"     # fabricated: exists nowhere
PLAN_STEPS = ["merge", "verdict(reuse|waiver|slurm-gate)", "live-db", "push", "pr", "merge", "cleanup"]
LIVE_DB_MODULES = ["test_registry_bon_decode_precedence.py", "test_registry_instrument_padding_identity.py", "test_registry_paper_spelling.py",
                   "test_registry_paper_views_contrast_current.py"]   # the modules known to open the live registry — a floor; the scan below is the rule


def _exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _run(argv, env, cwd=None, timeout=300):
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)


def _measure(name):
    spec = importlib.util.spec_from_file_location(f"measure_{name}", MEASURE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- S9 (1)–(4) --------------------------------------------------------------------------------------------------------------
def test_the_chain_is_complete_and_no_line_reaches_modal():
    names = {p.name for p in CHAIN_FILES}
    for need in ("land.sh", "gate_slurm.sh", "gate_lib.py", "gate_task.py", "gate_venv.sbatch", "gate_array.sbatch", "gate_finalize.sbatch",
                 "verdict_local.py", "wait_main_moves.sh", "baseline_refresh.sh", "tree_export.sh", "gate_status.sh",
                 "battery_slurm.sh", "battery.sbatch", "battery_collect.py", "live_db_slurm.sh", "replica.sbatch",
                 "check_no_modal.sh", "convert_config.py", "operating_scripts_inventory.py", "recover.sh", "mirror_history.sh"):
        assert need in names, need
    hits = [(p.name, i, line.strip()) for p in CHAIN_FILES for i, line in enumerate(p.read_text().splitlines(), 1) if MODAL_CALL.search(line)]
    assert not hits, hits
    imports = [(p.name, line.strip()) for p in CHAIN_FILES if p.suffix == ".py"
               for line in p.read_text().splitlines() if re.match(rf"\s*(import {_M}\b|from {_M}\b)", line)]
    assert not imports, imports
    # C9: no chain file binds the name at all -- the MagicMock stub that stood in for the package is gone with the loader
    bound = [(p.name, i, line.strip()) for p in CHAIN_FILES if p.suffix == ".py" for i, line in enumerate(p.read_text().splitlines(), 1)
             if re.search(rf"MagicMock\(name=.{_M}.\)|sys\.modules\[.{_M}.\]", line)]
    assert not bound, bound
    for name in ("check_no_modal.sh", "recover.sh", "convert_config.py", "operating_scripts_inventory.py", "mirror_history.sh"):
        assert os.access(LANDING / name, os.X_OK), f"{name} is executable"


#: the history push every export is preceded by: mirror-repos.sh moves heads
#: and tags only, so each caller runs mirror_history.sh (the archived branches and the tags, verified on the mirror) right before its
#: tree_export.sh; the gate and the battery REFUSE when the history does not arrive (their trees run the suite, which reads commits only
#: the archive reaches), the window, the live-DB step and recover.sh note it (their trees run code only; a gate reusing the tree completes it)
HISTORY_CALLERS = {"gate_slurm.sh": "refuse", "battery_slurm.sh": "refuse", "window.sh": "note", "live_db_slurm.sh": "note", "recover.sh": "note"}


def test_every_tree_export_is_preceded_by_the_history_push_the_gate_and_the_battery_refusing_without_it():
    for name, policy in HISTORY_CALLERS.items():
        text = (LANDING / name).read_text()
        assert re.search(r'^MIRROR_HISTORY="?\$\{MIRROR_HISTORY_SH:-\$HERE/mirror_history\.sh\}"?$', text, re.M), f"{name}: the knob, defaulting beside the script"
        exports = [m.start() for m in re.finditer(r'< "\$HERE/tree_export\.sh"', text)]
        calls = [m.start() for m in re.finditer(r'HOUT=\$\("\$MIRROR_HISTORY" "\$[A-Z_]+" "ssh://\$[A-Z_]+\$[A-Z_]+" 2>&1\); hrc=\$\?', text)]
        assert len(exports) == 1 and len(calls) == 1 and calls[0] < exports[0], (name, calls, exports)
        after = text[calls[0]:exports[0]]
        if policy == "refuse":
            assert re.search(r'\[ \$hrc -eq 0 \] \|\| die "the archived branches and tags of \$REPO did not reach', after), (name, after)
        else:
            assert "mirror_history.sh reported a problem (rc $hrc)" in after and "die " not in after and "cluster_fail" not in after, (name, after)
    # (the production mirroring script, mirror-repos.sh, is the grading host's and outside this repository; its refspecs are pinned in
    # test_registry_tree_export.py, which builds its mirrors with them)


def test_every_shell_file_parses():
    assert SHELL_FILES
    for p in SHELL_FILES:
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        assert r.returncode == 0, (p.name, r.stderr)


def test_land_plan_lists_exactly_the_seven_steps_without_acting(tmp_path):
    env = dict(os.environ, REGISTRY_LANDING_STATE=str(tmp_path / "state"), SISTEER_REPO=str(tmp_path / "norepo"))
    r = _run([LANDING / "land.sh", "--plan"], env)
    assert r.returncode == 0, r.stderr
    steps = re.findall(r"^PLAN (\d+) (\S+) :: (.+)$", r.stdout, re.M)
    assert [int(n) for n, _, _ in steps] == list(range(1, 8))
    assert [s for _, s, _ in steps] == PLAN_STEPS
    text = "\n".join(d for _, _, d in steps)
    assert "gate_slurm.sh" in text and "only fall-through" in text and "no Modal" in text
    assert "gate_status.sh" in steps[3][2] and "landing-gate" in steps[3][2] and "exits 13" in steps[3][2], "the push step posts the status before any PR or merge"
    live = steps[2][2]
    assert "live_db_slurm.sh" in live and "registry replica head" in live and "replica.sbatch land" in live and "--dependency=singleton" in live, live
    assert "frame_projection/1" in live and "REGISTRY_LIVE_DB" in live and "the grading host runs no pytest" in live and "check.json.equal" in live, live
    assert "check_no_modal.sh" in steps[0][2] and "LAND_NO_MODAL_SCAN=report|refuse|off" in steps[0][2] and "exits 14" in steps[0][2], "the scan is step 1's"
    assert "IDENTICAL tree" in steps[1][2] and "D35" in steps[1][2] and "tree-identical-to=" in steps[1][2], "the tree-identical reuse is the verdict step's branch (a)"
    assert not (tmp_path / "state").exists() and not MODAL_CALL.search(r.stdout)


def test_help_prints_the_header_comment_and_no_code():
    for name in ("land.sh", "baseline_refresh.sh", "gate_slurm.sh", "gate_status.sh", "battery_slurm.sh", "check_no_modal.sh", "recover.sh"):
        r = subprocess.run([str(LANDING / name), "--help"], capture_output=True, text=True, env=dict(os.environ, REGISTRY_LANDING_STATE=""))
        assert r.returncode == 0 and r.stdout.startswith(f"# {name}") and all(l.startswith("#") for l in r.stdout.splitlines()), (name, r.stdout[-200:])
        assert "set -u" not in r.stdout and "HERE=" not in r.stdout


def _gate_lib():
    spec = importlib.util.spec_from_file_location("gate_lib_under_test", LANDING / "gate_lib.py")
    gl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gl)
    return gl


def test_gate_lib_imports_under_a_refusing_hook_and_owns_the_suite_functions():
    """C9: gate_lib's functions are its own -- the module imports with an import hook that raises on the real package, exposes every
    function the gate and the battery call, binds no `modal` name, and partitions this repo's tests."""
    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == _M or name.startswith(_M + "."):
                raise AssertionError(f"a real import of {name} was attempted")
            return None

    finder = Refuse()
    saved = sys.modules.pop(_M, None)
    sys.meta_path.insert(0, finder)
    try:
        gl = _gate_lib()
        for fn in ("shard_files", "pytest_argv", "suite_env", "run_pytest", "parse_junit", "verify_checkout", "dependency_hashes", "merge_shard_records", "shard_layout"):
            assert callable(getattr(gl, fn)), fn
        assert isinstance(gl.SHARD_RULE, str) and gl.DEPENDENCY_FILES == ("pyproject.toml", "uv.lock")
        assert not hasattr(gl, "load_cpu" + "_suite") and not hasattr(gl, _M) and _M not in sys.modules
        files = gl.shard_files(REPO, 0, 40)
        assert files and all(f.startswith("tests/") for f in files)
        union = sorted(f for i in range(40) for f in gl.shard_files(REPO, i, 40))
        assert union == sorted(q.relative_to(REPO).as_posix() for q in (REPO / "tests").glob("test_*.py")) and len(union) == len(set(union))
    finally:
        sys.meta_path.remove(finder)
        if saved is not None:
            sys.modules[_M] = saved


def test_gate_lib_partition_parser_env_and_merge_are_pinned_on_synthetic_inputs(tmp_path):
    """The copied functions behave as the recorded summaries expect: a by-name round-robin partition that is stable across edits; the
    pytest command with --continue-on-collection-errors; the suite env with a fresh throwaway spool, the suite user, the full-suite
    allowance and the popped registry knobs (HF_HOME left to the caller); the JUnit parser's node ids; the merged record's status rules."""
    gl = _gate_lib()
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    for n in ("test_b.py", "test_a.py", "test_c.py"):
        (tree / "tests" / n).write_text("def test_x():\n    assert True\n")
    assert gl.shard_files(tree, 0, 2) == ["tests/test_a.py", "tests/test_c.py"] and gl.shard_files(tree, 1, 2) == ["tests/test_b.py"]
    (tree / "tests" / "test_a.py").write_text("def test_x():\n    assert True\n" * 50)      # a file that grows does not move
    assert gl.shard_files(tree, 0, 2) == ["tests/test_a.py", "tests/test_c.py"]
    with pytest.raises(ValueError):
        gl.shard_files(tree, 2, 2)
    argv = gl.pytest_argv("py", "/j.xml", ["tests/test_a.py"])
    assert argv == ["py", "-m", "pytest", "tests/test_a.py", "--tb=short", "--continue-on-collection-errors", "--junitxml=/j.xml"]
    assert gl.pytest_argv("py", "/j.xml")[3] == "tests"
    env = gl.suite_env({"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0", "SISTEER_REGISTRY_DISABLED": "1", "REGISTRY_EMIT": "0", "HF_HOME": "/keep", "HF_HUB_OFFLINE": "1"},
                       tree, spool_root=tmp_path)
    assert env["PYTHONPATH"] == str(tree / "src") and env["USER"] == env["LOGNAME"] == gl.SUITE_USER and env["WANDB_MODE"] == "disabled"
    assert env[gl.FULL_SUITE_ALLOW_ENV] == "1" and "CUDA_VISIBLE_DEVICES" not in env and "SISTEER_REGISTRY_DISABLED" not in env and "REGISTRY_EMIT" not in env
    assert env["HF_HOME"] == "/keep" and "HF_HUB_OFFLINE" not in env, "HF_HOME is the caller's; the offline knob is dropped"
    spool = Path(env[gl.REGISTRY_SPOOL_ENV])
    assert spool.is_dir() and spool.parent == tmp_path and spool.name.startswith(gl.REGISTRY_SPOOL_PREFIX)
    log = tmp_path / "run.log"
    rc, timed_out = gl.run_pytest([sys.executable, "-c", "print('hello'); raise SystemExit(3)"], tree, env, log, 30)
    assert (rc, timed_out) == (3, False) and "hello" in log.read_text() and not spool.exists(), "the throwaway spool is removed after the run"
    junit = tmp_path / "junit.xml"
    junit.write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="3" failures="1" errors="1" skipped="0">'
                     '<testcase classname="tests.test_a" name="test_ok"/>'
                     '<testcase classname="tests.test_a" name="test_bad"><failure message="boom  "/></testcase>'
                     '<testcase classname="tests.test_c" name="test_err"><error message="import"/></testcase></testsuite></testsuites>')
    parsed = gl.parse_junit(junit)
    assert parsed["counts"] == {"tests": 3, "failures": 1, "errors": 1, "skipped": 0}
    assert parsed["failures"] == [{"node": "tests.test_a::test_bad", "kind": "failure", "message": "boom"}, {"node": "tests.test_c::test_err", "kind": "error", "message": "import"}]
    assert gl.parse_junit(tmp_path / "absent.xml")["counts"] is None
    rec_a = {"status": "FAIL", "pytest_rc": 1, "counts": parsed["counts"], "failures": parsed["failures"], "elapsed_s": 10, "shard": {"index": 0, "count": 2, "files": ["tests/test_a.py", "tests/test_c.py"]}}
    rec_b = {"status": "PASS", "pytest_rc": 0, "counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}, "failures": [], "elapsed_s": 4, "shard": {"index": 1, "count": 2, "files": ["tests/test_b.py"]}}
    merged = gl.merge_shard_records([rec_a, rec_b], "f" * 40, "probe")
    assert merged["status"] == "FAIL" and merged["pytest_rc"] == 1 and merged["counts"] == {"tests": 4, "failures": 1, "errors": 1, "skipped": 0}
    assert [f["node"] for f in merged["failures"]] == ["tests.test_a::test_bad", "tests.test_c::test_err"] and merged["shards"] == 2 and merged["elapsed_s"] == 10.0
    assert merged["shard_layout"]["rule"] == gl.SHARD_RULE and merged["shard_layout"]["count"] == 2
    assert gl.merge_shard_records([rec_b, dict(rec_b, shard={"index": 0, "count": 2, "files": []})], "f" * 40, "p")["status"] == "PASS"
    assert gl.merge_shard_records([rec_b, dict(rec_b, status="TIMEOUT", pytest_rc=-9)], "f" * 40, "p")["status"] == "TIMEOUT"
    refused = gl.merge_shard_records([rec_b, {"status": "BAD_TREE", "error": "tree does not verify"}], "f" * 40, "p")
    assert refused["status"] == "BAD_TREE" and refused["pytest_rc"] is None and refused["error"] == "tree does not verify"
    layout_a = gl.shard_layout([rec_a, rec_b])["map_sha256"]
    assert layout_a == gl.shard_layout([rec_b, rec_a])["map_sha256"] and layout_a != gl.shard_layout([rec_a])["map_sha256"]
    # verify_checkout / dependency_hashes on a real tiny checkout
    (tree / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tree / "uv.lock").write_text("lock\n")
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "seed")
    sha = _git(tree, "rev-parse", "HEAD")
    assert gl.verify_checkout(tree, sha)["ok"] and not gl.verify_checkout(tree, "0" * 40)["ok"]
    (tree / "uv.lock").write_text("changed\n")
    assert gl.verify_checkout(tree, sha) == {"ok": False, "head": sha, "clean": False, "why": "modified tracked files"}
    dh = gl.dependency_hashes(tree)
    assert set(dh) == {"pyproject.toml", "uv.lock"} and dh["uv.lock"] == __import__("hashlib").sha256(b"changed\n").hexdigest()


# ---- baseline_refresh.sh against shims --------------------------------------------------------------------------------------
GIT_SHIM = r'''#!/bin/bash
echo "git $*" >> "$MOCK/git.calls"
printf '%s\n' "${GH_TOKEN:-unset}" >> "$MOCK/git.gh_token"
args=("$@"); i=0; [ "${args[0]}" = "-C" ] && i=2
case "${args[$i]}" in
  ls-remote) printf '%s\trefs/heads/main\n' "$(cat "$MOCK/main_sha")" ;;
  fetch) [ -f "$MOCK/fetch_fails" ] && exit 128; exit 0 ;;
  cat-file) [ -f "$MOCK/absent" ] && exit 1; exit 0 ;;
  *) exit 0 ;;
esac
'''
SSH_SHIM = r'''#!/bin/bash
# the login node: squeue prints $MOCK/squeue.out (id state reason node name); sacct prints $MOCK/sacct.out, or — when $MOCK/sacct.completed
# exists — COMPLETED rows for exactly the ids asked for, named from the $MOCK/jobs/<id> files the gate shim wrote at submission
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=$1; shift; cmd="$*"
echo "ssh $host $cmd" >> "$MOCK/ssh.calls"
case "$cmd" in
  *squeue*) cat "$MOCK/squeue.out" 2>/dev/null; exit 0 ;;
  *sacct*)
    if [ -f "$MOCK/sacct.completed" ]; then
      ids=$(sed -n 's/.*-j \([0-9,]*\).*/\1/p' <<<"$cmd")
      for id in ${ids//,/ }; do
        name=$(cat "$MOCK/jobs/$id" 2>/dev/null || echo unknown)
        case "$name" in *-fin-*) echo "$id|COMPLETED|0:0|compute-7|$name" ;; gate-venv-*) echo "$id|COMPLETED|0:0|compute-3|$name" ;; *) echo "${id}_0|COMPLETED|0:0|compute-5|$name" ;; esac
      done; exit 0
    fi
    cat "$MOCK/sacct.out" 2>/dev/null; exit 0 ;;
  *tail*) echo "fake .out tail: $cmd" ;;
  *"test -s "*) [ -f "$MOCK/results_summary" ] && exit 0 || exit 1 ;;
  *) exit 0 ;;
esac
'''
GATE_SHIM = r'''#!/bin/bash
# the gate as baseline_refresh.sh and land.sh see it: submit prints the JOBS line (venv= from $MOCK/venv_job when present; after a
# 2 s sleep when $MOCK/slow exists; dies with no record while $MOCK/gate_fail exists; dies AFTER queuing the array — the PARTIAL
# line — while $MOCK/gate_partial exists) and writes $MOCK/jobs/<id> = the job's name; collect writes a terminal summary +
# BASELINE; a tip verdict writes $STATE/verdicts/gate-land-<branch>.out (LAND unless $MOCK/hold exists)
echo "gate $*" >> "$MOCK/gate.calls"
sha=$1; shift
n=$(( $(cat "$MOCK/jobseq" 2>/dev/null || echo 1000) + 2 )); echo "$n" > "$MOCK/jobseq"
mkdir -p "$STATE/summaries" "$STATE/verdicts" "$MOCK/jobs"
run=gate-${sha:0:7}-tester
case " $* " in
  *" --collect-only "*)
    printf '{"sha": "%s", "status": "PASS", "pytest_rc": 0, "counts": {"tests": 5, "failures": 0, "errors": 0, "skipped": 0}, "failures": [], "shards": 40}\n' "$sha" > "$STATE/summaries/summary-$sha.json"
    echo "$sha" > "$STATE/BASELINE"; echo "[gate] baseline recorded"; exit 0 ;;
  *" --baseline "*)
    [ -f "$MOCK/slow" ] && sleep 2
    if [ -f "$MOCK/gate_fail" ]; then echo "[gate] ref=$sha"; echo "[gate:cluster] ERR sbatch array"; echo "[gate] ERROR: cluster step failed" >&2; exit 2; fi
    venv=$(cat "$MOCK/venv_job" 2>/dev/null || echo none); [ "$venv" = none ] || echo "gate-venv-abc123def456" > "$MOCK/jobs/$venv"
    echo "$run" > "$MOCK/jobs/$((n-1))"
    if [ -f "$MOCK/gate_partial" ]; then
      echo "[gate] ref=$sha"; echo "[gate:cluster] PARTIAL array=$((n-1)) venv=$venv"; echo "[gate:cluster] ERR sbatch finalize: array $((n-1)) is queued without its finalizer"
      echo "[gate] PARTIAL sha=$sha array=$((n-1)) venv=$venv run=$run canceled=no — the array was queued without its finalizer"; echo "[gate] ERROR: cluster step failed" >&2; exit 2
    fi
    echo "gate-${sha:0:7}-fin-tester" > "$MOCK/jobs/$n"
    echo "[gate] ref=$sha"; echo "[gate] JOBS sha=$sha venv=$venv array=$((n-1)) finalize=$n run=$run"; exit 0 ;;
  *" --branch "*)
    b=$(sed -n 's/.*--branch \([^ ]*\).*/\1/p' <<<"$*"); base=$(sed -n 's/.*--baseline-sha \([^ ]*\).*/\1/p' <<<"$*")
    [ -s "$STATE/summaries/summary-$base.json" ] || { echo "[gate] ERROR: no baseline summary for $base" >&2; exit 2; }
    { echo "BASELINE sha=${base:0:8} status=PASS"; echo "MERGE sha=${sha:0:8} status=PASS"; echo "INTRODUCED (0):"; echo "  (none)";
      if [ -f "$MOCK/hold" ]; then echo "VERDICT: HOLD"; else echo "VERDICT: LAND"; fi; } > "$STATE/verdicts/gate-land-$b.out"
    cp "$STATE/verdicts/gate-land-$b.out" "$STATE/verdicts/verdict-$b-${sha:0:7}.out"; exit 0 ;;
esac
echo "[gate] shim: unexpected args $*" >&2; exit 2
'''
MIRROR_SHIM = '#!/bin/bash\necho "mirror $*" >> "$MOCK/mirror.calls"\nexit 0\n'
HISTORY_SHIM = '#!/bin/bash\necho "history $*" >> "$MOCK/history.calls"\necho "HISTORY current 0 refs on $2 (shim)"\nexit 0\n'
RUN_A, FIN_A, VENV_NAME = f"gate-{SHA_A[:7]}-tester", f"gate-{SHA_A[:7]}-fin-tester", "gate-venv-abc123def456"
REFRESH_ENV_KEYS = ("PATH", "MOCK", "STATE", "REGISTRY_LANDING_STATE", "GATE_SLURM_SH", "MIRROR_REPOS_SH", "SISTEER_REPO", "GATE_RUSER", "GATE_HOST",
                    "GATE_SHARED", "MIRROR_REPOS_CLUSTER")
#: The git identity every git this file drives carries — the commands the tests run AND the ones the scripts under test run
#: (land.sh's own `git merge --no-edit origin/main`, which writes a merge commit). The gate host runs the suite under a fresh, empty
#: HOME (gate_task.py: a pristine home like a container's), so no ~/.gitconfig names a user and git's auto-detection fails on a bare
#: node (`fatal: unable to auto-detect email address (got 'user@node.(none)')`): three landing tests died at the merge step there on
#: 2026-09-28 while passing on a grading host whose ~/.gitconfig filled the gap. Synthetic values, carried by the ENVIRONMENT so the
#: landing script itself never has to invent an identity for a real landing; GIT_CONFIG_GLOBAL=/dev/null + GIT_CONFIG_NOSYSTEM keep
#: the host's own git config (a user.name, credential helpers, signing) out of every test on every host.
GIT_IDENTITY_ENV = {
    "GIT_AUTHOR_NAME": "Gate Test", "GIT_AUTHOR_EMAIL": "gate-test@example.invalid",
    "GIT_COMMITTER_NAME": "Gate Test", "GIT_COMMITTER_EMAIL": "gate-test@example.invalid",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
}


def _refresh_env(tmp_path: Path, *, shim_git: bool = True) -> dict:
    bin_, mock_, state = tmp_path / "bin", tmp_path / "mock", tmp_path / "state"
    bin_.mkdir(parents=True, exist_ok=True)
    mock_.mkdir(parents=True, exist_ok=True)
    if shim_git:
        _exe(bin_ / "git", GIT_SHIM)
    _exe(bin_ / "ssh", SSH_SHIM)
    _exe(mock_ / "gate_slurm.sh", GATE_SHIM)
    _exe(mock_ / "mirror-repos.sh", MIRROR_SHIM)
    (mock_ / "main_sha").write_text(SHA_A + "\n")
    env = dict(os.environ, **GIT_IDENTITY_ENV)
    env.update({"PATH": f"{bin_}:{env['PATH']}", "MOCK": str(mock_), "STATE": str(state), "REGISTRY_LANDING_STATE": str(state),
                "GATE_SLURM_SH": str(mock_ / "gate_slurm.sh"), "MIRROR_REPOS_SH": str(mock_ / "mirror-repos.sh"),
                "SISTEER_REPO": str(tmp_path / "repo"), "GATE_RUSER": "tester", "GATE_HOST": "TESTHOST",
                # the cluster's shared root and the repo-mirror tool's name for it, which the site config supplies on a real host
                "GATE_SHARED": "/shared/tester", "MIRROR_REPOS_CLUSTER": "sel_one"})
    return env


def _tick(env, *args):
    return _run([LANDING / "baseline_refresh.sh", *args], env, timeout=120)


def _job(env, sha=SHA_A, suffix=".job") -> dict:
    p = Path(env["STATE"]) / "baselines" / f"{sha}{suffix}"
    return dict(line.split("=", 1) for line in p.read_text().splitlines() if "=" in line) if p.exists() else {}


def _calls(env, name) -> list:
    p = Path(env["MOCK"]) / f"{name}.calls"
    return p.read_text().splitlines() if p.exists() else []


def _journal(env) -> str:
    p = Path(env["STATE"]) / "baselines" / "journal.log"
    return p.read_text() if p.exists() else ""


def _sacct(env, *rows):
    (Path(env["MOCK"]) / "sacct.out").write_text("".join(f"{r}\n" for r in rows))


def _squeue(env, *rows):
    (Path(env["MOCK"]) / "squeue.out").write_text("".join(f"{r}\n" for r in rows))


def _settle(env, *args):
    """Two ticks with the jobs out of the queue: the first is the terminal read the record keeps, the second decides."""
    r1 = _tick(env, *args)
    assert "TERMINAL READ 1 of 2" in r1.stdout, r1.stdout
    return _tick(env, *args)


def _age_job(env, sha, hours):
    """Re-stamp a recorded job's first submission `hours` ago (the 4-h bound is per sha)."""
    jp = Path(env["STATE"]) / "baselines" / f"{sha}.job"
    old = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    jp.write_text(re.sub(r"^SUBMITTED_UTC=.*$", f"SUBMITTED_UTC={old}", jp.read_text(), flags=re.M))


def test_baseline_refresh_refuses_a_sha_absent_from_the_object_store_before_any_submission(tmp_path):
    """`git fetch origin <sha>` BEFORE any `git show`; a fabricated sha that the fetch does not bring in ends the tick loudly."""
    env = _refresh_env(tmp_path)
    (Path(env["MOCK"]) / "absent").write_text("")
    r = _tick(env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "REFUSED" in r.stdout and SHA_A in r.stdout and "not in" in r.stdout
    git = _calls(env, "git")
    assert any(f"fetch -q origin {SHA_A}" in c for c in git), git
    assert git.index(next(c for c in git if "fetch" in c)) < git.index(next(c for c in git if "cat-file" in c))
    assert not _calls(env, "gate") and not _calls(env, "mirror") and not _job(env), "the claim written before the fetch is dropped with the refusal"
    assert "REFUSED" in _journal(env)


def test_baseline_refresh_submits_once_and_records_the_job(tmp_path):
    env = _refresh_env(tmp_path)
    r = _tick(env)
    assert r.returncode == 0, r.stdout + r.stderr
    gate = _calls(env, "gate")
    assert gate == [f"gate {SHA_A} --baseline --no-wait"], gate
    assert _calls(env, "mirror") == ["mirror --cluster sel_one repo"], "the mirror selector and the checkout's name, never the registry label"
    git = _calls(env, "git")
    assert any("ls-remote" in c for c in git) and any(f"fetch -q origin {SHA_A}" in c for c in git)
    job = _job(env)
    assert job["JOB_STATE"] == "SUBMITTED" and job["ARRAY"] == "1001" and job["FINALIZE"] == "1002" and job["ATTEMPT"] == "1" and job["SHA"] == SHA_A
    assert job["RUN"] == f"gate-{SHA_A[:7]}-tester" and job["VENV"] == "none", "the run name is the gate's own (its JOBS line), not re-derived"
    assert list(job) == ["SHA", "ARRAY", "FINALIZE", "VENV", "RUN", "SUBMITTED_UTC", "ATTEMPT", "JOB_STATE", "LAST_SEEN", "EXCLUDED", "CONFIRM", "UPDATED_UTC"]
    assert "SUBMITTED baseline" in _journal(env) and "QoS high, 0 GPU" in _journal(env)
    # a second tick with the job still unknown to the queue submits nothing more
    r2 = _tick(env)
    assert r2.returncode == 0 and _calls(env, "gate") == gate
    assert "UNKNOWN" in r2.stdout
    assert (Path(env["STATE"]) / "baselines" / ".lock").exists()


def test_baseline_refresh_reports_pending_then_started_then_collects_the_terminal_summary(tmp_path):
    env = _refresh_env(tmp_path)
    assert _tick(env).returncode == 0
    mock_ = Path(env["MOCK"])
    (mock_ / "squeue.out").write_text("1001_[0-39]|PENDING|Priority||\n1002|PENDING|Dependency||\n")
    r = _tick(env)
    assert "PENDING baseline" in r.stdout and "reason=Priority" in r.stdout and _job(env)["JOB_STATE"] == "PENDING"
    ssh = _calls(env, "ssh")
    assert [c for c in ssh if "squeue" in c] == ["ssh TESTHOST squeue -h -u tester -o '%i|%T|%R|%N|%j'"], "one queue read per tick, the user's whole queue"
    r = _tick(env)
    assert "PENDING baseline" not in r.stdout, "one journal line per state change, not per tick"
    (mock_ / "squeue.out").write_text("777|RUNNING|None|compute-1|\n1001_3|RUNNING|None|compute-5|\n1001_[4-39]|PENDING|Resources||\n1002|PENDING|Dependency||\n")
    r = _tick(env)
    assert "STARTED baseline" in r.stdout and "node=compute-5" in r.stdout and _job(env)["JOB_STATE"] == "RUNNING"
    (mock_ / "squeue.out").write_text("777|RUNNING|None|compute-1|someone-else\n")     # only somebody else's job left in the user's queue
    _sacct(env, f"1001_0|COMPLETED|0:0|compute-5|{RUN_A}", f"1001_1|COMPLETED|0:0|compute-6|{RUN_A}", f"1002|COMPLETED|0:0|compute-7|{FIN_A}")
    r = _tick(env)
    assert "TERMINAL READ 1 of 2" in r.stdout and _job(env)["JOB_STATE"] == "PENDING_CONFIRM" and _job(env)["CONFIRM"], r.stdout
    assert not any("--collect-only" in c for c in _calls(env, "gate")), "one terminal read decides nothing"
    r = _tick(env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TERMINAL COLLECTED" in r.stdout and _job(env)["JOB_STATE"] == "COLLECTED"
    assert (Path(env["STATE"]) / "summaries" / f"summary-{SHA_A}.json").is_file()
    assert (Path(env["STATE"]) / "BASELINE").read_text().strip() == SHA_A
    assert _calls(env, "gate")[-1] == f"gate {SHA_A} --collect-only --baseline"
    n = len(_calls(env, "gate"))
    r = _tick(env)
    assert "baseline summary present" in r.stdout and len(_calls(env, "gate")) == n, "nothing more once the summary exists"
    j = _journal(env)
    assert j.index("SUBMITTED") < j.index("PENDING") < j.index("STARTED") < j.index("TERMINAL READ 1 of 2") < j.index("TERMINAL COLLECTED")


def test_baseline_refresh_resubmits_on_node_fail_with_exclude_at_most_three_times(tmp_path):
    env = _refresh_env(tmp_path)
    assert _tick(env).returncode == 0
    mock_ = Path(env["MOCK"])
    (mock_ / "squeue.out").write_text("")
    _sacct(env, f"1001_0|NODE_FAIL|0:0|compute-9|{RUN_A}", f"1001_1|COMPLETED|0:0|compute-6|{RUN_A}", f"1002|CANCELLED|0:0|None assigned|{FIN_A}")
    r = _settle(env)
    assert "NODE_FAIL" in r.stdout and "resubmitting attempt 2 with --exclude compute-9" in r.stdout, r.stdout
    assert _calls(env, "gate")[-1] == f"gate {SHA_A} --baseline --no-wait --exclude compute-9"
    job = _job(env)
    assert job["ATTEMPT"] == "2" and job["EXCLUDED"] == "compute-9" and job["ARRAY"] == "1003" and job["JOB_STATE"] == "SUBMITTED"
    rec = _job(env, suffix=".job.attempt1")
    assert rec["JOB_STATE"] == "RESUBMITTED" and rec["ATTEMPT"] == "1" and rec["EXCLUDED"] == "compute-9", "the recovery record carries the replacement's exclusions"
    assert job["SUBMITTED_UTC"] == rec["SUBMITTED_UTC"], "the first submission's stamp is carried: the 4-h bound is per sha"
    _sacct(env, f"1003_2|NODE_FAIL|0:0|compute-11|{RUN_A}", f"1004|CANCELLED|0:0|None assigned|{FIN_A}")
    r = _settle(env)
    assert "resubmitting attempt 3 with --exclude compute-11,compute-9" in r.stdout, r.stdout
    assert _job(env)["ATTEMPT"] == "3" and _job(env)["EXCLUDED"] == "compute-11,compute-9"
    _sacct(env, f"1005_7|NODE_FAIL|0:0|compute-2|{RUN_A}", f"1006|CANCELLED|0:0|None assigned|{FIN_A}")
    r = _settle(env)
    assert "TERMINAL FAILED" in r.stdout and "attempt=3/3" in r.stdout and _job(env)["JOB_STATE"] == "FAILED"
    n = len(_calls(env, "gate"))
    r = _tick(env)
    assert "not resubmitting" in r.stdout and len(_calls(env, "gate")) == n, "a FAILED attempt blocks a fresh submission of that sha"
    assert not any("--collect-only" in c for c in _calls(env, "gate"))


def test_baseline_refresh_a_failed_replacement_keeps_the_attempt_and_the_exclusions(tmp_path):
    """The resubmit after a NODE_FAIL dies between its array and finalizer sbatch (no JOBS line). While the cluster keeps refusing, no
    .job lands (the claim is dropped) and the attempt-1 record stays RESUBMITTED with the exclusion list; the tick that finds no .job
    submits from that record — attempt 2 with --exclude, the first submission's stamp carried — never a fresh attempt 1."""
    env = _refresh_env(tmp_path)
    assert _tick(env).returncode == 0
    mock_ = Path(env["MOCK"])
    (mock_ / "squeue.out").write_text("")
    _sacct(env, f"1001_0|NODE_FAIL|0:0|compute-9|{RUN_A}", f"1002|CANCELLED|0:0|None assigned|{FIN_A}")
    (mock_ / "gate_fail").write_text("")
    r = _settle(env)
    assert r.returncode == 1 and "resubmitting attempt 2 with --exclude compute-9" in r.stdout and r.stdout.count("SUBMIT FAILED") == 2, r.stdout
    assert "never a fresh attempt 1" in r.stdout, "the same tick already retried from the recovery record (and was refused again)"
    assert not _job(env), "a failed submission records no job (the claim is dropped)"
    rec = _job(env, suffix=".job.attempt1")
    assert rec["JOB_STATE"] == "RESUBMITTED" and rec["ATTEMPT"] == "1" and rec["EXCLUDED"] == "compute-9"
    assert _calls(env, "gate")[-2:] == [f"gate {SHA_A} --baseline --no-wait --exclude compute-9"] * 2
    (mock_ / "gate_fail").unlink()
    (mock_ / "sacct.out").write_text("")
    r = _tick(env)
    assert r.returncode == 0 and "submitting attempt 2 with --exclude compute-9" in r.stdout and "never a fresh attempt 1" in r.stdout, r.stdout
    assert _calls(env, "gate")[-1] == f"gate {SHA_A} --baseline --no-wait --exclude compute-9"
    job = _job(env)
    assert job["ATTEMPT"] == "2" and job["EXCLUDED"] == "compute-9" and job["JOB_STATE"] == "SUBMITTED" and job["SUBMITTED_UTC"] == rec["SUBMITTED_UTC"]
    assert sum(1 for l in _journal(env).splitlines() if "SUBMITTED baseline" in l and "attempt=1" in l) == 1, _journal(env)


def test_baseline_refresh_marks_a_plain_failure_with_the_out_tails_and_the_four_hour_bound(tmp_path):
    env = _refresh_env(tmp_path)
    assert _tick(env).returncode == 0
    mock_ = Path(env["MOCK"])
    (mock_ / "squeue.out").write_text("")
    _sacct(env, f"1001_0|FAILED|1:0|compute-1|{RUN_A}", f"1001_1|TIMEOUT|0:0|compute-2|{RUN_A}", f"1002|FAILED|3:0|compute-3|{FIN_A}")
    r = _settle(env)
    assert "TERMINAL FAILED" in r.stdout and _job(env)["JOB_STATE"] == "FAILED"
    j = _journal(env)
    assert "finalizer .out tail: fake .out tail" in j and f"gate-{SHA_A[:7]}-fin-tester_1002.out" in j
    assert "task 1001_0 .out tail" in j and f"gate-{SHA_A[:7]}-tester_1001_0.out" in j
    # the 4-h bound: a job still queued 5 h after submission is given up on (nothing canceled), once
    sha_b = "89abcdef" * 5
    (mock_ / "main_sha").write_text(sha_b + "\n")
    assert _tick(env).returncode == 0 and _job(env, sha_b)["JOB_STATE"] == "SUBMITTED"
    _age_job(env, sha_b, 5)
    (mock_ / "squeue.out").write_text("1003_[0-39]|PENDING|Priority||\n1004|PENDING|Dependency||\n")
    r = _tick(env)
    assert "GAVE_UP" in r.stdout and _job(env, sha_b)["JOB_STATE"] == "GAVE_UP"
    assert not any("scancel" in c for c in _calls(env, "ssh"))
    n = len(_calls(env, "gate"))
    r = _tick(env)
    assert "not resubmitting" in r.stdout and len(_calls(env, "gate")) == n
    # the same bound in the UNKNOWN state: a job in neither squeue nor sacct 5 h after submission is given up on, not retried forever
    sha_c = "76543210" * 5
    (mock_ / "main_sha").write_text(sha_c + "\n")
    (mock_ / "squeue.out").write_text("")
    (mock_ / "sacct.out").write_text("")
    assert _tick(env).returncode == 0 and _job(env, sha_c)["JOB_STATE"] == "SUBMITTED"
    assert "UNKNOWN" in _tick(env).stdout, "under the bound the tick keeps retrying"
    _age_job(env, sha_c, 5)
    r = _tick(env)
    assert "GAVE_UP" in r.stdout and "neither squeue nor sacct" in r.stdout and _job(env, sha_c)["JOB_STATE"] == "GAVE_UP", r.stdout


def test_baseline_refresh_watches_the_venv_job_the_array_depends_on(tmp_path):
    """A venv job that built lets the array proceed (no false failure); one that failed while the array still waits on it retires the
    dependents (scancel) and is recovered like the array: NODE_FAIL → resubmit with --exclude, anything else → FAILED with its .out tail."""
    env = _refresh_env(tmp_path)
    mock_ = Path(env["MOCK"])
    (mock_ / "venv_job").write_text("900\n")
    assert _tick(env).returncode == 0 and _job(env)["VENV"] == "900"
    # the venv is running: the array pends on it — reported as PENDING (reason Dependency), never as STARTED
    (mock_ / "squeue.out").write_text("900|RUNNING|None|compute-3|\n1001_[0-39]|PENDING|(Dependency)||\n1002|PENDING|(Dependency)||\n")
    r = _tick(env)
    assert "PENDING baseline" in r.stdout and "reason=(Dependency)" in r.stdout and "STARTED" not in r.stdout
    assert not any("sacct" in c for c in _calls(env, "ssh")), "no sacct while the venv is in the queue"
    # the venv left the queue and COMPLETED: the array's turn (a transient Dependency reason is probed once, no failure declared)
    (mock_ / "squeue.out").write_text("1001_[0-39]|PENDING|Dependency||\n1002|PENDING|Dependency||\n")
    _sacct(env, f"900|COMPLETED|0:0|compute-3|{VENV_NAME}")
    r = _tick(env)
    assert "VENV FAILED" not in r.stdout and not any("scancel" in c for c in _calls(env, "ssh")) and _job(env)["JOB_STATE"] == "PENDING"
    assert any("sacct" in c and "-j 900" in c for c in _calls(env, "ssh"))
    # the venv NODE_FAIL'd: seen twice the same, the dependents can never start — canceled, and the whole submission is resubmitted off that node
    (mock_ / "squeue.out").write_text("1001_[0-39]|PENDING|DependencyNeverSatisfied||\n1002|PENDING|Dependency||\n")
    _sacct(env, f"900|NODE_FAIL|0:0|compute-3|{VENV_NAME}")
    r = _tick(env)
    assert "TERMINAL READ 1 of 2" in r.stdout and "VENV FAILED" not in r.stdout and _job(env)["JOB_STATE"] == "PENDING", r.stdout
    r = _tick(env)
    assert "VENV FAILED" in r.stdout and "resubmitting attempt 2 with --exclude compute-3" in r.stdout, r.stdout
    assert any(c.endswith("scancel 1001 1002") for c in _calls(env, "ssh")), _calls(env, "ssh")
    job = _job(env)
    assert job["ATTEMPT"] == "2" and job["EXCLUDED"] == "compute-3" and job["ARRAY"] == "1003" and job["VENV"] == "900"
    # a plain venv failure: FAILED, with the venv job's .out tail (named from sacct's JobName) in the journal
    (mock_ / "squeue.out").write_text("1003_[0-39]|PENDING|DependencyNeverSatisfied||\n1004|PENDING|Dependency||\n")
    _sacct(env, f"900|FAILED|1:0|compute-4|{VENV_NAME}")
    r = _settle(env)
    assert "VENV FAILED" in r.stdout and "TERMINAL FAILED" in r.stdout and "venv=FAILED" in r.stdout and _job(env)["JOB_STATE"] == "FAILED", r.stdout
    assert any(c.endswith("scancel 1003 1004") for c in _calls(env, "ssh"))
    j = _journal(env)
    assert "venv job 900 .out tail: fake .out tail" in j and f"{VENV_NAME}_900.out" in j
    # gone from the queue with the venv in the sacct rows too: the venv's state rides the terminal line
    sha_b = "89abcdef" * 5
    (mock_ / "main_sha").write_text(sha_b + "\n")
    assert _tick(env).returncode == 0
    (mock_ / "squeue.out").write_text("")
    _sacct(env, f"900|COMPLETED|0:0|compute-3|{VENV_NAME}", f"1005_0|COMPLETED|0:0|compute-5|gate-{sha_b[:7]}-tester", f"1006|COMPLETED|0:0|compute-7|gate-{sha_b[:7]}-fin-tester")
    r = _settle(env)
    assert "TERMINAL COLLECTED" in r.stdout and "tasks=[COMPLETED×1]" in r.stdout, r.stdout
    assert any("sacct" in c and "-j 1005,1006,900" in c for c in _calls(env, "ssh"))


def test_baseline_refresh_sha_argument_targets_that_main_without_ls_remote(tmp_path):
    env = _refresh_env(tmp_path)
    sha_b = "fedcba98" * 5
    r = _tick(env, "--sha", sha_b, "--quiet")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SUBMITTED baseline" in r.stdout and "baseline job" not in r.stdout, "--quiet keeps the events, drops the routine status line"
    assert not any("ls-remote" in c for c in _calls(env, "git"))
    r = _tick(env, "--sha", sha_b, "--quiet")
    assert r.returncode == 0 and [l for l in r.stdout.splitlines() if "UNKNOWN" not in l] == [], r.stdout   # only the watcher's event line
    assert _job(env, sha_b)["JOB_STATE"] == "SUBMITTED" and "SUBMITTED baseline" in _journal(env)
    r = _tick(env, "--sha", "not-a-sha")
    assert r.returncode == 1 and "not a 40-hex sha" in r.stdout


def test_two_concurrent_ticks_submit_once(tmp_path):
    """The 2-min timer and a waiting land.sh run the same tick: the second takes no lock and exits 0 at once, and the first's CLAIMED
    record exists before any remote work — so a main is never submitted twice, and a tick that dies mid-submission leaves its claim."""
    env = _refresh_env(tmp_path)
    (Path(env["MOCK"]) / "slow").write_text("")      # the gate shim sleeps 2 s before its JOBS line: the remote-work window
    p1 = subprocess.Popen([str(LANDING / "baseline_refresh.sh")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(200):
        if _job(env).get("JOB_STATE") == "CLAIMED":
            break
        time.sleep(0.02)
    claim = _job(env)
    assert claim.get("JOB_STATE") == "CLAIMED" and claim["ARRAY"] == "" and claim["ATTEMPT"] == "1", claim
    p2 = _tick(env)
    assert p2.returncode == 0 and "another tick holds" in p2.stdout, p2.stdout + p2.stderr
    out1, err1 = p1.communicate(timeout=60)
    assert p1.returncode == 0, out1 + err1
    assert _calls(env, "gate") == [f"gate {SHA_A} --baseline --no-wait"], "one submission"
    assert _job(env)["JOB_STATE"] == "SUBMITTED" and _job(env)["ARRAY"] == "1001"
    assert _journal(env).count("SUBMITTED baseline") == 1


def _claim(env, sha=SHA_A):
    """A CLAIMED record as a tick that died a moment ago would have left it (its stamp is the sha's first submission: the 4-h bound counts from it)."""
    bl = Path(env["STATE"]) / "baselines"
    bl.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (bl / f"{sha}.job").write_text(f"SHA={sha}\nARRAY=\nFINALIZE=\nVENV=\nRUN=\nSUBMITTED_UTC={now}\nATTEMPT=1\nJOB_STATE=CLAIMED\n"
                                   f"LAST_SEEN=\nEXCLUDED=\nCONFIRM=\nUPDATED_UTC={now}\n")


def test_an_orphaned_claim_reads_the_queue_before_anything_is_resubmitted(tmp_path):
    """A CLAIMED .job seen by a tick that holds the lock was left by a tick that died mid-submission. The queue, by job NAME, says what it
    left: nothing → the claim is dropped and a fresh submission follows; the array and its finalizer → adopted and watched; the array
    alone → recorded PARTIAL, canceled and confirmed gone before any resubmission."""
    env = _refresh_env(tmp_path)
    _claim(env)
    r = _tick(env)
    assert r.returncode == 0 and "ORPHAN CLAIM" in r.stdout and "nothing as" in r.stdout and "dropping the claim" in r.stdout, r.stdout
    assert any("--name=" in c and RUN_A in c and FIN_A in c and "now-24hours" in c for c in _calls(env, "ssh")), "the accounting is asked by job name first"
    assert any(c.endswith(f"test -s /shared/tester/gates/{SHA_A}/summary.json") for c in _calls(env, "ssh")), "then the results dir"
    assert _job(env)["JOB_STATE"] == "SUBMITTED" and _calls(env, "gate") == [f"gate {SHA_A} --baseline --no-wait"]
    # the claimer had queued both: adopt them, submit nothing
    sha_b = "89abcdef" * 5
    _claim(env, sha_b)
    _squeue(env, f"1001_[0-39]|PENDING|Priority||{RUN_A}", f"1002|PENDING|Dependency||{FIN_A}",
            f"4242_[0-39]|PENDING|Priority||gate-{sha_b[:7]}-tester", f"4243|PENDING|Dependency||gate-{sha_b[:7]}-fin-tester")
    r = _tick(env)
    assert "ORPHAN CLAIM" in r.stdout and "adopted" in r.stdout, r.stdout
    job = _job(env, sha_b)
    assert job["JOB_STATE"] == "PENDING" and job["ARRAY"] == "4242" and job["FINALIZE"] == "4243" and job["RUN"] == f"gate-{sha_b[:7]}-tester"
    assert len(_calls(env, "gate")) == 1
    # the claimer had queued the array alone: PARTIAL, then canceled and confirmed gone, then resubmitted
    sha_c = "76543210" * 5
    _claim(env, sha_c)
    _squeue(env, f"1001_[0-39]|PENDING|Priority||{RUN_A}", f"1002|PENDING|Dependency||{FIN_A}", f"5000_[0-39]|PENDING|Priority||gate-{sha_c[:7]}-tester")
    r = _tick(env)
    assert "ORPHAN CLAIM" in r.stdout and "without its finalizer" in r.stdout and _job(env, sha_c)["JOB_STATE"] == "PARTIAL" and _job(env, sha_c)["ARRAY"] == "5000", r.stdout
    assert len(_calls(env, "gate")) == 1
    r = _tick(env)
    assert any(c.endswith("scancel 5000") for c in _calls(env, "ssh")) and "still in the queue" in r.stdout and _job(env, sha_c)["JOB_STATE"] == "PARTIAL", r.stdout
    assert len(_calls(env, "gate")) == 1, "nothing resubmits while the partial array is queued"
    _squeue(env, f"1001_[0-39]|PENDING|Priority||{RUN_A}", f"1002|PENDING|Dependency||{FIN_A}")
    (Path(env["MOCK"]) / "main_sha").write_text(sha_c + "\n")
    r = _tick(env)
    assert "canceled and gone from the queue" in r.stdout and _job(env, sha_c)["JOB_STATE"] == "SUBMITTED" and _job(env, sha_c)["ARRAY"] != "5000", r.stdout
    assert _calls(env, "gate")[-1] == f"gate {sha_c} --baseline --no-wait"


def test_an_orphaned_claim_whose_jobs_already_finished_is_adopted_from_the_accounting_not_resubmitted(tmp_path):
    """A grading host restart spanned the whole run: the queue is empty, but sacct shows the sha's array and finalizer COMPLETED under their names.
    The claim adopts them as an ended attempt and the two-read rule collects — no second 40-task array."""
    env = _refresh_env(tmp_path)
    _claim(env)
    _squeue(env)
    _sacct(env, f"1001_0|COMPLETED|0:0|compute-5|{RUN_A}", f"1001_1|COMPLETED|0:0|compute-6|{RUN_A}", f"1002|COMPLETED|0:0|compute-7|{FIN_A}")
    r = _tick(env)
    assert "ORPHAN CLAIM" in r.stdout and "accounting shows array=1001 finalize=1002" in r.stdout and "adopted as an ended attempt" in r.stdout, r.stdout
    assert "TERMINAL READ 1 of 2" in r.stdout, "the adopted rows are read under the two-read rule in the same tick"
    job = _job(env)
    assert job["JOB_STATE"] == "PENDING_CONFIRM" and job["ARRAY"] == "1001" and job["FINALIZE"] == "1002" and job["RUN"] == RUN_A and job["CONFIRM"]
    assert not _calls(env, "gate"), "nothing submitted"
    r = _tick(env)
    assert "TERMINAL COLLECTED" in r.stdout and _job(env)["JOB_STATE"] == "COLLECTED", r.stdout
    assert _calls(env, "gate") == [f"gate {SHA_A} --collect-only --baseline"], "the only gate call is the collect"
    assert (Path(env["STATE"]) / "BASELINE").read_text().strip() == SHA_A
    # the accounting holds two attempts under the sha's names (an earlier NODE_FAIL'd array): the newest pair is the one adopted
    sha_b = "89abcdef" * 5
    _claim(env, sha_b)
    _sacct(env, f"3001_0|NODE_FAIL|0:0|compute-9|gate-{sha_b[:7]}-tester", f"3002|CANCELLED|0:0|None assigned|gate-{sha_b[:7]}-fin-tester",
           f"3003_0|COMPLETED|0:0|compute-5|gate-{sha_b[:7]}-tester", f"3004|COMPLETED|0:0|compute-7|gate-{sha_b[:7]}-fin-tester")
    r = _tick(env)
    assert "accounting shows array=3003 finalize=3004" in r.stdout and _job(env, sha_b)["ARRAY"] == "3003", r.stdout


def test_an_orphaned_claim_with_a_terminal_summary_on_the_cluster_is_collected_not_resubmitted(tmp_path):
    """Nothing in the queue and nothing left in the accounting, but the cluster results dir holds the sha's summary.json: collected."""
    env = _refresh_env(tmp_path)
    _claim(env)
    _squeue(env)
    _sacct(env)
    (Path(env["MOCK"]) / "results_summary").write_text("")
    r = _tick(env)
    assert "ORPHAN CLAIM" in r.stdout and "is a terminal summary — collected" in r.stdout and _job(env)["JOB_STATE"] == "COLLECTED", r.stdout
    assert _calls(env, "gate") == [f"gate {SHA_A} --collect-only --baseline"]
    assert (Path(env["STATE"]) / "BASELINE").read_text().strip() == SHA_A
    r = _tick(env)
    assert "baseline summary present" in r.stdout and len(_calls(env, "gate")) == 1


def test_baseline_refresh_records_a_partial_submission_and_cancels_it_before_resubmitting(tmp_path):
    """The gate queued the array and then its finalizer's sbatch failed (`[gate] PARTIAL … array=N`): the claim becomes a PARTIAL record
    with that id, never a dropped claim that the next tick would overtake with a second array. The next tick cancels the array and
    re-reads the queue; only once it is gone is the record dropped and a fresh submission made."""
    env = _refresh_env(tmp_path)
    mock_ = Path(env["MOCK"])
    (mock_ / "gate_partial").write_text("")
    r = _tick(env)
    assert r.returncode == 1 and "SUBMIT PARTIAL" in r.stdout and "array=1001" in r.stdout, r.stdout
    job = _job(env)
    assert job["JOB_STATE"] == "PARTIAL" and job["ARRAY"] == "1001" and job["FINALIZE"] == "" and job["RUN"] == RUN_A
    _squeue(env, f"1001_[0-39]|PENDING|Priority||{RUN_A}")
    r = _tick(env)
    assert any(c.endswith("scancel 1001") for c in _calls(env, "ssh")), _calls(env, "ssh")
    assert "still in the queue" in r.stdout and "baseline job PARTIAL" in r.stdout and _job(env)["JOB_STATE"] == "PARTIAL", r.stdout
    assert len(_calls(env, "gate")) == 1, "no second array while the first is queued"
    (mock_ / "gate_partial").unlink()
    _squeue(env)
    r = _tick(env)
    assert r.returncode == 0 and "canceled and gone from the queue" in r.stdout and "SUBMITTED baseline" in r.stdout, r.stdout
    assert _calls(env, "gate") == [f"gate {SHA_A} --baseline --no-wait"] * 2 and _job(env)["JOB_STATE"] == "SUBMITTED" and _job(env)["ARRAY"] == "1003"


def test_baseline_refresh_trusts_a_sacct_row_only_by_id_and_name_and_read_twice(tmp_path):
    """This cluster's sacct can answer for another job on a fresh id (the job-watcher script). Out of the queue: a row for another id is
    not ours (UNKNOWN); our id under another name is a MISMATCH → PENDING_CONFIRM, nothing decided; a nonterminal row → PENDING_CONFIRM;
    a reappearance in the queue forgets the read; a terminal read is kept and only an identical read on the next tick retires."""
    env = _refresh_env(tmp_path)
    assert _tick(env).returncode == 0
    _squeue(env)
    _sacct(env, "4242|COMPLETED|0:0|compute-1|someone-else")
    r = _tick(env)
    assert "UNKNOWN" in r.stdout and _job(env)["JOB_STATE"] == "SUBMITTED", r.stdout
    _sacct(env, "1001_0|COMPLETED|0:0|compute-5|not-ours", f"1002|COMPLETED|0:0|compute-7|{FIN_A}")
    r = _tick(env)
    assert "MISMATCH" in r.stdout and "not-ours" in r.stdout and _job(env)["JOB_STATE"] == "PENDING_CONFIRM", r.stdout
    assert not any("--collect-only" in c for c in _calls(env, "gate"))
    _squeue(env, f"1001_[0-39]|PENDING|Priority||{RUN_A}")
    r = _tick(env)
    assert "REAPPEARED" in r.stdout and _job(env)["JOB_STATE"] == "PENDING" and _job(env)["CONFIRM"] == "", r.stdout
    _squeue(env)
    _sacct(env, f"1001_0|RUNNING|0:0|compute-5|{RUN_A}", f"1002|PENDING|0:0|None assigned|{FIN_A}")
    r = _tick(env)
    assert "not terminal in sacct yet" in r.stdout and _job(env)["JOB_STATE"] == "PENDING_CONFIRM", r.stdout
    _sacct(env, f"1001_0|COMPLETED|0:0|compute-5|{RUN_A}")     # terminal, but the finalizer's row is missing: not a complete terminal read
    r = _tick(env)
    assert "not terminal in sacct yet" in r.stdout and _job(env)["JOB_STATE"] == "PENDING_CONFIRM" and _job(env)["CONFIRM"] == "", r.stdout
    _sacct(env, f"1001_0|COMPLETED|0:0|compute-5|{RUN_A}", f"1002|COMPLETED|0:0|compute-7|{FIN_A}")
    r = _tick(env)
    assert "TERMINAL READ 1 of 2" in r.stdout and _job(env)["CONFIRM"], r.stdout
    d1 = _job(env)["CONFIRM"]
    _sacct(env, f"1001_0|FAILED|1:0|compute-5|{RUN_A}", f"1002|COMPLETED|0:0|compute-7|{FIN_A}")     # the second read differs: kept, not acted on
    r = _tick(env)
    assert "TERMINAL READ 1 of 2" in r.stdout and _job(env)["CONFIRM"] not in ("", d1) and not any("--collect-only" in c for c in _calls(env, "gate")), r.stdout
    r = _tick(env)
    assert "TERMINAL COLLECTED" in r.stdout and _job(env)["JOB_STATE"] == "COLLECTED", r.stdout
    assert sum(1 for c in _calls(env, "gate") if "--collect-only" in c) == 1
    j = _journal(env)
    assert j.index("MISMATCH") < j.index("REAPPEARED") < j.index("not terminal") < j.index("TERMINAL READ 1 of 2") < j.index("TERMINAL COLLECTED")


def test_gate_slurm_remote_block_prints_the_partial_line_when_the_finalizer_sbatch_fails(tmp_path):
    """The cluster-side contract baseline_refresh.sh relies on: the remote bash of gate_slurm.sh, run here with an sbatch shim that queues
    the array and refuses the finalizer, prints `PARTIAL array=<id> venv=<id|none>` before its ERR line and exits 2."""
    text = (LANDING / "gate_slurm.sh").read_text()
    remote = text.split("<<'REMOTE'\n", 1)[1].split("\nREMOTE\n", 1)[0]
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "sbatch", '#!/bin/bash\necho "sbatch $*" >> "$MOCK/sbatch.calls"\ncase " $* " in *" -J gate-abc1234-fin-"*) echo "sbatch: error: Batch job submission failed" >&2; exit 1 ;; esac\necho 5001\n')
    _exe(bin_ / "squeue", '#!/bin/bash\nexit 0\n')
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "pyproject.toml").write_text("[project]\n")
    (tree / "uv.lock").write_text("lock\n")
    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / ".gate-ready.json").write_text("{}")
    import hashlib
    ph = hashlib.sha256((tree / "pyproject.toml").read_bytes()).hexdigest()
    lh = hashlib.sha256((tree / "uv.lock").read_bytes()).hexdigest()
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", MOCK=str(tmp_path))
    r = subprocess.run(["bash", "-s", "--", "abc1234" + "0" * 33, str(tree), str(venv), "h12h12h12h12", str(tmp_path / "cbin"), str(tmp_path / "res"), "40",
                        "gate-abc1234-tester", ph, lh, "abc1234", str(tmp_path / "logs"), ""], input=remote, capture_output=True, text=True, env=env)
    assert r.returncode == 2, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    assert "PARTIAL array=5001 venv=none" in lines and any(l.startswith("ERR sbatch finalize") for l in lines), r.stdout
    assert lines.index("PARTIAL array=5001 venv=none") < next(i for i, l in enumerate(lines) if l.startswith("ERR")), "the record line precedes the refusal"
    calls = (tmp_path / "sbatch.calls").read_text().splitlines()
    assert len(calls) == 2 and "--array=0-39" in calls[0] and "-J gate-abc1234-fin-" in calls[1], calls
    assert "[gate] PARTIAL sha=$SHA array=$PA" in text and 'rsh "scancel $PA"' in text, "the grading host side cancels the array and prints the one-line record"


def _gate_remote(tmp_path, logs, *, venv_ready):
    """gate_slurm.sh's remote bash run here with an sbatch shim that records each call and answers job ids; returns (result, calls)."""
    remote = (LANDING / "gate_slurm.sh").read_text().split("<<'REMOTE'\n", 1)[1].split("\nREMOTE\n", 1)[0]
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    _exe(bin_ / "sbatch", '#!/bin/bash\necho "sbatch $*" >> "$MOCK/sbatch.calls"\nn=$(( $(wc -l < "$MOCK/sbatch.calls") + 6000 ))\necho "$n"\n')
    _exe(bin_ / "squeue", '#!/bin/bash\nexit 0\n')
    tree, venv = tmp_path / "tree", tmp_path / "venv"
    tree.mkdir(exist_ok=True)
    venv.mkdir(exist_ok=True)
    (tree / "pyproject.toml").write_text("[project]\n")
    (tree / "uv.lock").write_text("lock\n")
    if venv_ready:
        (venv / ".gate-ready.json").write_text("{}")
    import hashlib
    ph = hashlib.sha256((tree / "pyproject.toml").read_bytes()).hexdigest()
    lh = hashlib.sha256((tree / "uv.lock").read_bytes()).hexdigest()
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", MOCK=str(tmp_path), USER="tester")
    r = subprocess.run(["bash", "-s", "--", "abc1234" + "0" * 33, str(tree), str(venv), "h12h12h12h12", "/cl/gates/bin", "/cl/gates", "40",
                        "gate-abc1234-tester", ph, lh, "abc1234", logs, ""], input=remote, capture_output=True, text=True, env=env)
    calls = (tmp_path / "sbatch.calls").read_text().splitlines() if (tmp_path / "sbatch.calls").exists() else []
    return r, calls


def test_gate_slurm_names_every_jobs_log_under_the_sites_shared_root(tmp_path):
    """Each job the gate submits (the venv build, the array, the finalizer) carries --output under the log directory the grading host derives
    from the site's shared root: the job scripts name no log path a site with another shared root could not open. Without one, the
    remote side refuses before any submission."""
    logs = "/srv/cluster-b/tester/exp/logs"                   # a shared root that is not the one any job script could spell
    r, calls = _gate_remote(tmp_path, logs, venv_ready=False)
    assert r.returncode == 0 and "JOBS venv=6001 array=6002 finalize=6003" in r.stdout, r.stdout + r.stderr
    assert len(calls) == 3, calls
    assert calls[0].startswith(f"sbatch --parsable -J gate-venv-h12h12h12h12 --output={logs}/%x_%j.out ") and calls[0].endswith("/gate_venv.sbatch "
                                                                                                                                   f"{tmp_path / 'tree'} {tmp_path / 'venv'}"), calls[0]
    assert "--dependency=afterok:6001" in calls[1] and "--array=0-39" in calls[1] and f" --output={logs}/%x_%A_%a.out /cl/gates/bin/gate_array.sbatch " in calls[1], calls[1]
    assert "-J gate-abc1234-fin-tester" in calls[2] and f" --output={logs}/%x_%j.out /cl/gates/bin/gate_finalize.sbatch " in calls[2], calls[2]
    assert "--exclude" not in " ".join(calls), "an empty exclude list stays empty"
    text = (LANDING / "gate_slurm.sh").read_text()
    assert "CL_LOGS=$SHARED/exp/logs" in text and '"$S7" "$CL_LOGS" "$EXCLUDE" <<' in text and 'mkdir -p $CL_BIN $CL_TREES $CL_RESULTS/$SHA $CL_LOGS' in text
    (tmp_path / "sbatch.calls").unlink()
    r, calls = _gate_remote(tmp_path, "", venv_ready=True)
    assert r.returncode == 2 and "ERR no log directory" in r.stdout and calls == [], r.stdout + r.stderr


def _summary(path: Path, sha: str, failures, shards=40, layout="aaaaaaaaaaaa") -> str:
    path.write_text(json.dumps({"sha": sha, "status": "FAIL" if failures else "PASS", "pytest_rc": 1 if failures else 0, "counts": {"tests": 100, "failures": len(failures)},
                                "elapsed_s": 60, "failures": [{"node": n, "kind": "failure", "message": "x"} for n in failures], "shards": shards,
                                "shard_layout": {"count": shards, "rule": "r", "map_sha256": layout}, "substrate": "slurm"}))
    return str(path)


def test_verdict_local_vanished_only_lands_under_any_layout_the_retired_pair_holds_and_a_real_introduced_node_holds(tmp_path):
    """Parity with the operating copy: a VANISHED-only set LANDs under ANY layout with the vanished nodes named (INTRODUCED = M - B
    is the landing's protection; a module that stops collecting is its own `unknown::` introduced node); the ifeval r2/r3
    collection-flake pair is RETIRED (ALLOWED_COLLECTION_FLIPS is empty), so the one-in/one-out swap it used to excuse is a HOLD --
    mixed-layout HOLD when the layouts differ, a plain HOLD when they match; a real introduced node HOLDs; identical failure sets
    LAND plainly; a non-terminal summary is refused."""
    r2, r3 = "unknown::tests.test_ifeval_analysis_review_r2", "unknown::tests.test_ifeval_analysis_review_r3"
    other = "tests/test_registry_items.py::test_k"
    base = _summary(tmp_path / "base.json", "b" * 40, [r2, other])
    run = lambda tip: subprocess.run([sys.executable, str(LANDING / "verdict_local.py"), base, tip], capture_output=True, text=True).stdout
    src = (LANDING / "verdict_local.py").read_text()
    assert re.search(r"^ALLOWED_COLLECTION_FLIPS = \(\)$", src, re.M), "the pair is retired (D60): an empty tuple"
    # vanished-only: LAND under the same layout and under a different one, the nodes named, the layout caveat only when it differs
    out = run(_summary(tmp_path / "vanished_same.json", "1" * 40, [other]))
    assert re.search(rf"^VERDICT: LAND \(baseline nodes vanished: \['{re.escape(r2)}'\]\)$", out, re.M), out
    out = run(_summary(tmp_path / "vanished_mixed.json", "2" * 40, [other], layout="bbbbbbbbbbbb"))
    assert re.search(rf"^VERDICT: LAND \(baseline nodes vanished: \['{re.escape(r2)}'\]; layout differs — D64\)$", out, re.M), out
    assert "WARNING: MIXED SHARD LAYOUTS" in out and "MERGE sha=22222222" in out
    out = run(_summary(tmp_path / "all_vanished.json", "3" * 40, [], layout="bbbbbbbbbbbb"))
    assert re.search(r"^VERDICT: LAND \(baseline nodes vanished: \[.*\]; layout differs — D64\)$", out, re.M), out
    # the retired pair: one in, one out is a moved failure set now -- HOLD (mixed layout) and HOLD (same layout), never a caveat
    out = run(_summary(tmp_path / "flip.json", "c" * 40, [r3, other], layout="bbbbbbbbbbbb"))
    assert re.search(r"^VERDICT: HOLD \(mixed shard layout AND the failure set moved", out, re.M) and "CAVEAT" not in out, out
    out = run(_summary(tmp_path / "flip_same_layout.json", "d" * 40, [r3, other]))
    assert re.search(r"^VERDICT: HOLD$", out, re.M) and "CAVEAT" not in out, out
    # a real introduced node holds under both layouts; identical sets land plainly
    out = run(_summary(tmp_path / "real.json", "e" * 40, [r2, other, "tests/test_new.py::test_broken"], layout="bbbbbbbbbbbb"))
    assert re.search(r"^VERDICT: HOLD \(mixed shard layout", out, re.M), out
    out = run(_summary(tmp_path / "real2.json", "f" * 40, [r2, other, "tests/test_new.py::test_broken"]))
    assert re.search(r"^VERDICT: HOLD$", out, re.M), out
    out = run(_summary(tmp_path / "same.json", "4" * 40, [r2, other]))
    assert re.search(r"^VERDICT: LAND$", out, re.M) and "CAVEAT" not in out, out
    out = run(_summary(tmp_path / "same_mixed.json", "5" * 40, [r2, other], layout="bbbbbbbbbbbb"))
    assert re.search(r"^VERDICT: LAND \(failure set identical to baseline; layout differs because the branch changed the test count — D64\)$", out, re.M), out
    # only the sanctioned flaky node introduced: LAND; a non-terminal summary: refused
    out = run(_summary(tmp_path / "flaky.json", "6" * 40, [r2, other, "tests/test_x.py::test_wide_bf16_thing"]))
    assert re.search(r"^VERDICT: LAND \(only the sanctioned flaky node introduced", out, re.M), out
    nt = tmp_path / "nonterminal.json"
    nt.write_text(json.dumps({"sha": "7" * 40, "status": "TIMEOUT", "pytest_rc": -9, "counts": None, "failures": [], "shards": 40, "shard_layout": {"map_sha256": "aaaaaaaaaaaa"}}))
    out = run(str(nt))
    assert "REFUSE: MERGE summary is not a terminal gate result" in out and re.search(r"^VERDICT: HOLD \(non-terminal summary", out, re.M), out


def test_gate_baseline_unit_carries_the_sites_credential_hook_to_the_tick(tmp_path):
    """origin may be a private HTTPS remote read through gh's credential helper: a token-less non-login shell gets 128 from `git ls-remote`.
    The unit's ExecStart evaluates the site's environment hook first when the site names one (SISTEER_UNIT_ENV_HOOK; it is the ONE
    registry unit that does). Rehearsed by rendering the unit with the real installer for a site whose hook prints a
    synthetic GH_TOKEN, then running that ExecStart with %h → that home, under an environment scrubbed of every credential, against the git
    shim, which records whether GH_TOKEN reached it."""
    from front_door_testlib import write_site
    env = _refresh_env(tmp_path)
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (Path(env["MOCK"]) / "gh_token").write_text("rehearsal-credential\n")
    hook = _exe(home / ".local" / "bin" / "secret-env", '#!/bin/bash\necho "export GH_TOKEN=$(cat "$MOCK/gh_token")"\n')
    site = write_site(tmp_path / "siteroot", extra_config=f"SISTEER_UNIT_ENV_HOOK={hook}")
    rendered = tmp_path / "rendered"
    inst = subprocess.run(["bash", str(REPO / "scripts" / "registry" / "install_grading_host.sh"), "--units-only", "--no-systemctl", "--unit-dir", str(rendered)],
                          capture_output=True, text=True, env={"PATH": os.environ["PATH"], "HOME": str(home), "SISTEER_CONFIG": str(site / "config.env")},
                          timeout=120)
    assert inst.returncode == 0, inst.stdout + inst.stderr
    text = (rendered / "registry-gate-baseline.service").read_text()
    exec_line = next(l for l in text.splitlines() if l.startswith("ExecStart=")).split("=", 1)[1].replace("%h", str(home))
    assert exec_line.startswith("/bin/sh -c 'eval \"`") and "secret-env" in exec_line and exec_line.rstrip("'").endswith("baseline_refresh.sh")
    scrubbed = {k: env[k] for k in REFRESH_ENV_KEYS}
    scrubbed["HOME"] = str(home)
    assert not any(re.search(r"(?i)token|key|secret", k) for k in scrubbed)
    r = subprocess.run(exec_line, shell=True, capture_output=True, text=True, env=scrubbed, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    seen = set((Path(env["MOCK"]) / "git.gh_token").read_text().split())
    assert seen == {"rehearsal-credential"}, seen
    assert _calls(env, "gate") == [f"gate {SHA_A} --baseline --no-wait"] and _job(env)["JOB_STATE"] == "SUBMITTED"


# ---- land.sh end to end on a temporary origin -------------------------------------------------------------------------------
# the `--json` field set of the installed gh (2.97.0) for a pull request -- what `gh pr view --json <unknown>` lists. The shim below answers
# only these and refuses any other the way gh does; a test pins this tuple to the binary's own listing wherever gh is present. `merged` is
# NOT in it: the R9 shim answered it and masked the chain's post-merge read until the first real landing.
GH_PR_JSON_FIELDS = ("additions", "assignees", "author", "autoMergeRequest", "baseRefName", "baseRefOid", "body", "changedFiles", "closed",
                     "closedAt", "closingIssuesReferences", "comments", "commits", "createdAt", "deletions", "files", "fullDatabaseId",
                     "headRefName", "headRefOid", "headRepository", "headRepositoryOwner", "id", "isCrossRepository", "isDraft", "labels",
                     "latestReviews", "maintainerCanModify", "mergeCommit", "mergeStateStatus", "mergeable", "mergedAt", "mergedBy",
                     "milestone", "number", "potentialMergeCommit", "projectCards", "projectItems", "reactionGroups", "reviewDecision",
                     "reviewRequests", "reviews", "state", "statusCheckRollup", "title", "updatedAt", "url")
GH_SHIM = r'''#!/bin/bash
# gh: `pr create` → PR 42; `pr view` → the mergeable state, or `MERGED <mergedAt> <merge commit>` once a merge happened (`OPEN - -` before;
# nothing and exit 1 while $MOCK/pr_read_blank exists -- the read the first real landing met, D151), in the real gh's field vocabulary
# (GH_PR_JSON_FIELDS: a `--json` field outside it is refused as gh refuses it, exit 1 with the listing on stderr); `pr merge` → GitHub's
# --merge: a two-parent merge commit on origin's main (first parent main, second the branch head), the head pinned by --match-head-commit;
# refuses (rc 1) while $MOCK/merge_refused exists; exits 0 WITHOUT merging while $MOCK/merge_lies exists; while $MOCK/later_merge exists,
# ANOTHER PR's merge commit lands on main right after ours (a chain on another box) so ours is one behind the tip; while $MOCK/race_main exists,
# main advances to that sha FIRST — another writer between the chain's check and its merge. `api -X POST repos/<o>/<r>/statuses/<sha>`
# (gate_status.sh) records `<context> <state> <description>` under $MOCK/statuses/<sha> and prints what the real gh prints through the
# helper's --jq filter (context, state, description tab-joined; the context `ci` while $MOCK/status_echo_wrong exists); refuses (rc 1)
# while $MOCK/status_fails exists. The merge emulates ruleset 24155420 as GitHub evaluates it at merge time, against the CURRENT main:
# the head must carry a `landing-gate success` status AND contain main's tip (strict up-to-date), else the merge is refused — unless
# $MOCK/ruleset_off exists, the ruleset as created (disabled): GitHub then merges an out-of-date PR, the world the chain's RACED path is for
echo "gh $*" >> "$MOCK/gh.calls"
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.invalid GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.invalid
case "$1 $2" in
  "api -X")
    [ -f "$MOCK/status_fails" ] && { echo "gh: Validation Failed (HTTP 422) (shim refusal)" >&2; exit 1; }
    path=$4; sha=${path##*/}; st=""; ctx=""; desc=""; shift 4
    while [ $# -gt 0 ]; do case "$1" in -f) case "$2" in state=*) st=${2#state=} ;; context=*) ctx=${2#context=} ;; description=*) desc=${2#description=} ;; esac; shift 2 ;; *) shift ;; esac; done
    mkdir -p "$MOCK/statuses"; echo "$ctx $st $desc" >> "$MOCK/statuses/$sha"
    [ -f "$MOCK/status_echo_wrong" ] && ctx=ci
    printf '%s\t%s\t%s\n' "$ctx" "$st" "$desc" ;;
  "pr create") echo "https://github.com/example/repo/pull/42" ;;
  "pr view")
    fields=$(sed -n 's/.*--json \([^ ]*\).*/\1/p' <<<"$*")
    for f in ${fields//,/ }; do
      case " @FIELDS@ " in *" $f "*) ;; *) echo "Unknown JSON field: \"$f\"" >&2; echo "Available fields:" >&2; printf '  %s\n' @FIELDS@ >&2; exit 1 ;; esac
    done
    case "$fields" in
      mergeable,mergeStateStatus) echo "MERGEABLE CLEAN" ;;
      state,mergedAt,mergeCommit)
        [ -f "$MOCK/pr_read_blank" ] && exit 1
        [ -f "$MOCK/merged" ] && echo "MERGED 2026-01-01T00:00:00Z $(cat "$MOCK/merged")" || echo "OPEN - -" ;;
      *) echo "gh shim: no answer for --json $fields" >&2; exit 2 ;;
    esac ;;
  "pr merge")
    [ -f "$MOCK/merge_refused" ] && { echo "GraphQL: Pull Request is not mergeable (shim refusal)" >&2; exit 1; }
    [ -f "$MOCK/merge_lies" ] && { echo "merged PR $3"; exit 0; }
    head=$(sed -n 's/.*--head \([^ ]*\).*/\1/p' "$MOCK/gh.calls" | tail -n1); hsha=$(git -C "$MOCK/origin.git" rev-parse "refs/heads/$head")
    want=$(sed -n 's/.*--match-head-commit \([0-9a-f]*\).*/\1/p' <<<"$*")
    [ -n "$want" ] && [ "$want" != "$hsha" ] && { echo "GraphQL: head commit $hsha does not match $want" >&2; exit 1; }
    [ -f "$MOCK/race_main" ] && git -C "$MOCK/origin.git" update-ref refs/heads/main "$(cat "$MOCK/race_main")"
    base=$(git -C "$MOCK/origin.git" rev-parse refs/heads/main)
    if [ ! -f "$MOCK/ruleset_off" ]; then
      grep -q "^landing-gate success " "$MOCK/statuses/$hsha" 2>/dev/null || { echo "GraphQL: required status check landing-gate is expected on $hsha (shim ruleset 24155420)" >&2; exit 1; }
      git -C "$MOCK/origin.git" merge-base --is-ancestor "$base" "$hsha" || { echo "GraphQL: the head branch is behind the base branch $base (shim ruleset 24155420, strict)" >&2; exit 1; }
    fi
    mc=$(git -C "$MOCK/origin.git" commit-tree "$hsha^{tree}" -p "$base" -p "$hsha" -m "Merge pull request #$3")
    git -C "$MOCK/origin.git" update-ref refs/heads/main "$mc"; echo "$mc" > "$MOCK/merged"; echo "merged PR $3 as $mc"
    if [ -f "$MOCK/later_merge" ]; then
      o=$(git -C "$MOCK/origin.git" commit-tree "$mc^{tree}" -p "$base" -m "another branch"); l=$(git -C "$MOCK/origin.git" commit-tree "$mc^{tree}" -p "$mc" -p "$o" -m "Merge pull request #43")
      git -C "$MOCK/origin.git" update-ref refs/heads/main "$l"; echo "$l" > "$MOCK/later_merge"
    fi ;;
  *) echo "gh shim: $*" >&2; exit 2 ;;
esac
'''.replace("@FIELDS@", " ".join(GH_PR_JSON_FIELDS))
PY_SHIM = r'''#!/bin/bash
# land.sh's python: `-m pytest …` writes a JUnit report (red when $MOCK/red exists); anything else runs the real interpreter. While the
# live-DB step runs, origin's main advances to the sha in $MOCK/advance_main (once): the world moving under a gated tip
if [ "$1" != "-m" ]; then exec python3 "$@"; fi
echo "py $*" >> "$MOCK/py.calls"
if [ -f "$MOCK/advance_main" ]; then git -C "$MOCK/origin.git" update-ref refs/heads/main "$(cat "$MOCK/advance_main")"; rm -f "$MOCK/advance_main"; fi
junit=""; for a in "$@"; do case "$a" in --junitxml=*) junit=${a#--junitxml=} ;; esac; done
mkdir -p "$(dirname "$junit")"
cat > "$junit" <<'XML'
<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" skipped="1" time="4.5">
<testcase classname="tests.test_registry_paper_spelling" name="test_live_rendering_matches_the_pin_keys_per_source" time="3.25"/>
<testcase classname="tests.test_registry_paper_views_contrast_current" name="test_live_frame_carries_each_current_anchor_once_and_no_withdrawn_one" time="1.0"/>
<testcase classname="tests.test_registry_instrument_padding_identity" name="test_every_live_instrument_id_is_reproduced_read_only" time="0.25"><skipped message="no live registry"/></testcase>
</testsuite></testsuites>
XML
echo "3 passed in 4.5s"
[ -f "$MOCK/red" ] && { echo "1 failed"; exit 1; }
exit 0
'''
LIVEDB_SHIM = r'''#!/bin/bash
# land.sh's step 3 since C8 (live_db_slurm.sh): records its call, advances origin's main to $MOCK/advance_main (once) the way the real
# step's Slurm job gives main time to move, writes the junit land.sh renders durations from, prints the job line and the replica verdict;
# $MOCK/red = the replica job's pytest red (exit 5)
echo "livedb $*" >> "$MOCK/livedb.calls"
B=""; TIP=""; while [ $# -gt 0 ]; do case "$1" in --branch) B=$2; shift ;; --main|--expect-skipped) shift ;; -*) ;; *) TIP=$1 ;; esac; shift; done
if [ -f "$MOCK/advance_main" ]; then git -C "$MOCK/origin.git" update-ref refs/heads/main "$(cat "$MOCK/advance_main")"; rm -f "$MOCK/advance_main"; fi
# $MOCK/c1_class = the step accepted its check under the declared C1 allowance (D136) and wrote the verdict land.sh reads the clause from;
# the file's content, when not empty, is the tip the verdict names (a STALE verdict of another landing)
if [ -f "$MOCK/c1_class" ]; then
  VT=$(cat "$MOCK/c1_class"); [ -n "$VT" ] || VT=$TIP
  mkdir -p "$REGISTRY_ROOT/state"
  printf '{"schema": "replica_check_state/2", "tip": "%s", "branch": "%s", "equal": false, "c1_class": true, "accepted_as": "c1_class", "n_movers": 106, "ok": true, "headline": "pass"}\n' "$VT" "$B" > "$REGISTRY_ROOT/state/replica_check.json"
fi
mkdir -p "$STATE/land"
cat > "$STATE/land/$B.junit.xml" <<'XML'
<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="3" failures="0" errors="0" skipped="1" time="4.5">
<testcase classname="tests.test_registry_paper_spelling" name="test_live_rendering_matches_the_pin_keys_per_source" time="3.25"/>
<testcase classname="tests.test_registry_paper_views_contrast_current" name="test_live_frame_carries_each_current_anchor_once_and_no_withdrawn_one" time="1.0"/>
<testcase classname="tests.test_registry_instrument_padding_identity" name="test_every_live_instrument_id_is_reproduced_read_only" time="0.25"><skipped message="no live registry"/></testcase>
</testsuite></testsuites>
XML
echo "[land] @ live-db-head:start 2026-10-04T00:00:00Z"
echo "[land] @ live-db-head:end 2026-10-04T00:00:30Z"
echo "[land] live-db job 4242 (registry-replica, singleton, 01:30:00, tree shim) submitted 2026-10-04T00:00:40Z"
echo "[land] @ live-db-job:start 2026-10-04T00:00:40Z"
echo "[land] @ live-db-job:end 2026-10-04T00:10:40Z"
echo "replica check: equal -- frame_projection/1; replica 0123456789ab vs head 0123456789ab (head 2026-10-04T00:00:00.000Z, 3 files); generation g1; pytest rc=0 counts={} expected_skipped=0"
[ -f "$MOCK/c1_class" ] && echo "ACCEPTED: equal modulo the C1 same-stamp class (106 movers) -- refold-compare job 922769"
[ -f "$MOCK/red" ] && { echo "LIVE-DB TESTS RED or skips unaccounted (job 4242) -- NOT pushed"; exit 5; }
echo "[land] live-db on Slurm PASS for merged tip (shim) (job 4242)"
exit 0
'''
TEST_FILE = "import pytest\npytestmark = pytest.mark.live_db\n\ndef test_x():\n    assert True\n"


def _git(cwd, *args, env=None):
    env = dict(os.environ if env is None else env, **GIT_IDENTITY_ENV)
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, env=env)
    assert r.returncode == 0, (args, r.stderr)
    return r.stdout.strip()


def _land_world(tmp_path: Path, *, main_moves: bool = True) -> types.SimpleNamespace:
    """A bare origin, a base checkout with worktrees/<b>, one branch commit and (optionally) a later main commit."""
    env = _refresh_env(tmp_path, shim_git=False)
    mock_ = Path(env["MOCK"])
    _exe(Path(env["PATH"].split(":")[0]) / "gh", GH_SHIM)
    _exe(mock_ / "python", PY_SHIM)
    _exe(mock_ / "live_db_slurm.sh", LIVEDB_SHIM)
    origin = mock_ / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "pyproject.toml").write_text("[project]\nname='x'\n")
    (seed / "uv.lock").write_text("lock\n")
    (seed / "tests").mkdir()
    (seed / "tests" / "test_registry_probe.py").write_text(TEST_FILE)
    _git(seed, "init", "-q", "-b", "main")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-q", "origin", "main")
    repo = tmp_path / "repo"
    _git(tmp_path, "clone", "-q", str(origin), str(repo))
    _exe(repo / ".git" / "hooks" / "pre-commit", "#!/bin/bash\nexit 0\n")
    b = "probe-branch"
    _git(repo, "worktree", "add", "-q", "-b", b, f"worktrees/{b}", "origin/main")
    wt = repo / "worktrees" / b
    (wt / "branch.txt").write_text("branch work\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "branch work")
    tip = _git(wt, "rev-parse", "HEAD")
    if main_moves:
        (seed / "main.txt").write_text("main moved\n")
        _git(seed, "add", "-A")
        _git(seed, "commit", "-q", "-m", "main moved")
        _git(seed, "push", "-q", "origin", "main")
    main_sha = _git(seed, "rev-parse", "HEAD")
    body = tmp_path / "body.md"
    body.write_text("body\n")
    env.update({"LAND_PY": str(mock_ / "python"), "LAND_GH": "gh", "LAND_REPO": "example/repo", "LAND_POST_MERGE_SLEEP": "0",
                "LIVE_DB_SLURM_SH": str(mock_ / "live_db_slurm.sh"),
                # the ONE registry file land.sh reads (the live-DB step's verdict, for the C1 clause): a synthetic root, never the live one
                "REGISTRY_ROOT": str(tmp_path / "registry-root"),
                "LAND_BASELINE_POLL_S": "1", "LAND_BASELINE_WAIT_MIN": "1", "LAND_LEGACY_VERDICTS": "", "SISTEER_REPO": str(repo),
                "BASELINE_REFRESH_SH": str(LANDING / "baseline_refresh.sh")})
    env.pop("GATE_WAIVED_BY_USER", None)
    env.pop("KNOWN_RED", None)
    return types.SimpleNamespace(env=env, mock=mock_, state=Path(env["STATE"]), repo=repo, seed=seed, b=b, body=body, main_sha=main_sha, tip=tip, origin=origin)


def _land(w):
    return _run([LANDING / "land.sh", w.b, "title", w.body], w.env, timeout=300)


def _out(w) -> str:
    return (w.state / "land" / f"{w.b}.out").read_text()


def _stamps(text: str) -> list:
    return re.findall(r"^\[land\] @ ([a-z-]+):(start|end) \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", text, re.M)


def _steps(text: str) -> list:
    """land.sh's OWN step stamps. The `live-db-<step>` stamps live_db_slurm.sh prints (head, mirror-sync, tree, ship, job, collect; C8)
    sit INSIDE the live-db step's start/end and are measured by landings.py (live_db_steps); they are not steps of the chain's order."""
    return [(s, e) for s, e in _stamps(text) if not s.startswith("live-db-")]


def _statuses(w) -> list:
    """The gh shim's status-API calls (`gate_status.sh`), in call order."""
    return [c for c in _calls(w.env, "gh") if c.startswith("gh api -X POST repos/") and "/statuses/" in c]


def test_land_falls_through_to_the_slurm_gate_awaits_the_baseline_and_lands(tmp_path):
    w = _land_world(tmp_path)
    (w.mock / "squeue.out").write_text("")
    (w.mock / "sacct.completed").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    assert "[land] verdict=slurm-gate" in out and "awaiting baseline" in out
    assert re.search(rf"^\[land\] baseline {w.main_sha[:8]} job: ARRAY=1001 FINALIZE=1002 ATTEMPT=1 JOB_STATE=SUBMITTED $", out, re.M), out
    assert out.count("[land] baseline ") == 1, "the job's ids print once; the refresh's own events carry the transitions"
    assert "SUBMITTED baseline" in out and "TERMINAL READ 1 of 2" in out and "TERMINAL COLLECTED" in out
    gate = _calls(w.env, "gate")
    assert gate[0] == f"gate {w.main_sha} --baseline --no-wait", gate
    assert gate[1] == f"gate {w.main_sha} --collect-only --baseline", gate
    tip = re.search(r"merged tip=([0-9a-f]{8})", out).group(1)
    assert tip != w.tip[:8], "main moved, so the merged tip is a merge commit"
    assert gate[2].startswith("gate ") and gate[2].endswith(f" --branch {w.b} --baseline-sha {w.main_sha}") and tip in gate[2], gate
    assert len(gate) == 3
    assert f"MERGE sha={tip}" in (w.state / "verdicts" / f"gate-land-{w.b}.out").read_text()
    steps = _steps(out)
    order = [s for s, e in steps if e == "start"]
    assert order == ["merge", "verdict", "baseline-wait", "live-db", "push", "status", "pr", "merge-pr", "cleanup"], order
    assert all((s, "end") in steps for s in order)
    # the Slurm step's own stamps (live_db_slurm.sh) nest inside live-db's start and end, in order, and are what landings.py measures
    sub = [(s, e) for s, e in _stamps(out) if s.startswith("live-db-")]
    assert sub == [("live-db-head", "start"), ("live-db-head", "end"), ("live-db-job", "start"), ("live-db-job", "end")], sub
    assert out.index("@ live-db:start") < out.index("@ live-db-head:start") and out.index("@ live-db-job:end") < out.index("@ live-db:end")
    assert re.search(rf"^\[land\] live-DB test files on merged tip {tip} \(1 files, -m live_db\) \d\d:\d\d:\d\d$", out, re.M)
    ld = _calls(w.env, "livedb")
    assert len(ld) == 1 and ld[0].split()[1].startswith(tip) and len(ld[0].split()[1]) == 40, ld
    assert f"--branch {w.b}" in ld[0] and f"--main {w.main_sha}" in ld[0] and "--expect-skipped 0" in ld[0], ld
    assert not _calls(w.env, "py"), "the grading host runs no pytest in a landing (C8): the live-DB step is the Slurm driver's"
    assert "[land] live-db job 4242" in out and "replica check: equal -- frame_projection/1" in out
    land = w.state / "land"
    assert (land / f"{w.b}.junit.xml").is_file() and (land / f"{w.b}.merge.log").is_file()
    dur = (land / f"{w.b}.durations.txt").read_text()
    assert "3 tests" in dur and "test_live_rendering_matches_the_pin_keys_per_source" in dur.splitlines()[1]
    gh = _calls(w.env, "gh")
    assert [c.split()[1:3] for c in gh] == [["api", "-X"], ["pr", "create"], ["pr", "view"], ["pr", "merge"], ["pr", "view"]] and "--json state,mergedAt,mergeCommit" in gh[-1], gh
    assert f"--head {w.b}" in gh[1] and "--base main" in gh[1]
    # the `landing-gate` status: exactly one, success, on the pushed head (the gated tip's FULL sha), the evidence as its description,
    # posted after the push and before the PR exists or merges — the ruleset the gh shim emulates would have refused the merge without it
    full = gate[2].split()[1]
    sts = _statuses(w)
    assert len(sts) == 1 and sts[0].startswith(f"gh api -X POST repos/example/repo/statuses/{full} -f state=success -f context=landing-gate "
                                              f"-f description=slurm gate LAND on tree of {tip} vs baseline {w.main_sha[:8]} --jq "), sts
    assert gh.index(sts[0]) < gh.index(next(c for c in gh if " pr create " in c)) < gh.index(next(c for c in gh if " pr merge " in c))
    assert (w.mock / "statuses" / full).read_text() == f"landing-gate success slurm gate LAND on tree of {tip} vs baseline {w.main_sha[:8]}\n"
    assert out.index("[land] @ push:end") < out.index(f"landing-gate success on {tip}: slurm gate LAND on tree of {tip} vs baseline {w.main_sha[:8]}") < out.index("[land] @ pr:start")
    assert _git(w.origin, "rev-parse", "refs/heads/main^1") == w.main_sha and _git(w.origin, "rev-parse", "refs/heads/main^2")[:8] == tip
    assert any(f"--match-head-commit {gate[2].split()[1]}" in c for c in gh if " pr merge " in c), "the head is pinned to the gated tip"
    assert re.search(rf"^\[land\] merged onto {w.main_sha[:8]} as [0-9a-f]{{8}} \(first parent verified; second parent {tip}\)$", out, re.M), out
    mc = _git(w.origin, "rev-parse", "refs/heads/main")
    assert f"[land] merge proven by gh: state=MERGED mergedAt=2026-01-01T00:00:00Z mergeCommit={mc[:8]}" in out and "proven by git" not in out, out
    assert (w.state / "merge.lock").exists()
    assert not (w.repo / "worktrees" / w.b).exists() and w.b not in _git(w.repo, "branch", "--list", w.b)
    assert f"[land] LANDED {w.b} as PR 42 (tip {tip}, 1 round)" in out
    assert not MODAL_CALL.search(out)
    # measure/landings.py reads the file it just wrote
    rows = _measure("landings").collect(str(w.state), [])
    assert len(rows) == 1 and rows[0]["shape"] == "phase2" and rows[0]["landed"] and rows[0]["gate_kind"] == "SLURM" and rows[0]["pr"] == "42" and rows[0]["rounds"] == 1
    for k in ("gate_min", "baseline_wait_min", "live_db_min", "pr_merge_min", "total_min"):
        assert rows[0][k] is not None and rows[0][k] >= 0, (k, rows[0])
    assert rows[0]["live_db_where"] == "slurm" and rows[0]["live_db_job"] == "4242" and rows[0]["live_db_steps"] == {"head": 0.5, "job": 10.0}, rows[0]
    assert "slurm job 4242" in _measure("landings").render(rows)


def test_land_carries_the_live_db_steps_c1_acceptance_into_the_landing_gate_description(tmp_path):
    """When the live-DB step accepted its check under the declared cutover allowance (state/replica_check.json: accepted_as
    c1_class, written by live_db_slurm.sh for THIS tip and branch), the landing-gate description carries the clause with the mover
    count -- the evidence and nothing more; a verdict naming another tip (a stale file) adds nothing. The clause is read off the verdict
    the step wrote, never guessed from its output."""
    w = _land_world(tmp_path)
    (w.mock / "squeue.out").write_text("")
    (w.mock / "sacct.completed").write_text("")
    (w.mock / "c1_class").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    gate = _calls(w.env, "gate")
    full = gate[2].split()[1]
    tip = full[:8]
    assert "ACCEPTED: equal modulo the C1 same-stamp class (106 movers)" in out and "[land] live-DB check accepted under the declared C1 allowance; the landing-gate description carries it" in out
    assert json.load(open(Path(w.env["REGISTRY_ROOT"]) / "state" / "replica_check.json"))["tip"] == full
    desc = f"slurm gate LAND on tree of {tip} vs baseline {w.main_sha[:8]}; live-DB: C1 same-stamp class only (106 movers)"
    sts = _statuses(w)
    assert len(sts) == 1 and f"-f description={desc} --jq " in sts[0], sts
    assert (w.mock / "statuses" / full).read_text() == f"landing-gate success {desc}\n" and len(desc) <= 140
    assert f"landing-gate success on {tip}: {desc}" in out
    # a stale verdict (another landing's tip) carries no clause
    two = tmp_path / "two"
    two.mkdir()
    w2 = _land_world(two)
    (w2.mock / "squeue.out").write_text("")
    (w2.mock / "sacct.completed").write_text("")
    (w2.mock / "c1_class").write_text("f" * 40)
    r = _land(w2)
    out2 = _out(w2)
    assert r.returncode == 0, r.stdout + r.stderr + out2
    full2 = _calls(w2.env, "gate")[2].split()[1]
    assert json.load(open(Path(w2.env["REGISTRY_ROOT"]) / "state" / "replica_check.json"))["tip"] == "f" * 40
    assert "accepted under the declared C1 allowance" not in out2
    sts2 = _statuses(w2)
    assert len(sts2) == 1 and f"-f description=slurm gate LAND on tree of {full2[:8]} vs baseline {w2.main_sha[:8]} --jq " in sts2[0], sts2


def test_land_spells_the_landing_gate_description_compactly_when_the_c1_clause_would_pass_140_characters(tmp_path):
    """gate_status.sh truncates a description past 140 characters, which would cut the C1 clause -- the class and
    the mover count -- off the posted evidence. A reused-verdict base of 98 characters (the cutover landing's shape: a 33-character verdict
    file name) plus the clause is spelled compactly (`recorded LAND (<file>) on <tip8> vs base <main8>; …`, every fact kept) and fits; a
    short base is posted in full; a description that would not fit even compactly (a long waiver reason) is REFUSED before the push
    (exit 13), with no status posted, rather than posted truncated."""
    clause = "; live-DB: C1 same-stamp class only (106 movers)"
    # the cutover shape: the long recorded file name -> compacted, 126 characters
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "gate-land-registry-slurm-only.out").write_text(f"BASELINE sha={w.main_sha[:8]}\nMERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w.mock / "c1_class").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    full = f"recorded LAND verdict (gate-land-registry-slurm-only.out) on tree of {w.tip[:8]} vs baseline {w.main_sha[:8]}{clause}"
    compact = f"recorded LAND (gate-land-registry-slurm-only.out) on {w.tip[:8]} vs base {w.main_sha[:8]}{clause}"
    assert len(full) > 140 >= len(compact), (len(full), len(compact))
    sts = _statuses(w)
    assert len(sts) == 1 and f"-f description={compact} --jq " in sts[0] and full not in sts[0], sts
    assert (w.mock / "statuses" / w.tip).read_text() == f"landing-gate success {compact}\n"
    assert f"the landing-gate description carries it ({len(compact)} characters)" in out
    # a short base fits as it is: posted in full, nothing compacted
    two = tmp_path / "two"
    two.mkdir()
    w2 = _land_world(two, main_moves=False)
    v2 = w2.state / "verdicts"
    v2.mkdir(parents=True)
    (v2 / "verdict-earlier.out").write_text(f"BASELINE sha={w2.main_sha[:8]}\nMERGE sha={w2.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w2.mock / "c1_class").write_text("")
    r = _land(w2)
    assert r.returncode == 0, r.stdout + r.stderr + _out(w2)
    plain = f"recorded LAND verdict (verdict-earlier.out) on tree of {w2.tip[:8]} vs baseline {w2.main_sha[:8]}{clause}"
    assert len(plain) <= 140 and f"-f description={plain} --jq " in _statuses(w2)[0], _statuses(w2)
    # a waiver whose reason leaves no room even compactly: refused before the push, no status posted, the clause never cut
    three = tmp_path / "three"
    three.mkdir()
    w3 = _land_world(three, main_moves=False)
    w3.env["GATE_WAIVED_BY_USER"] = "the user, 2026-10-06: " + "a reason spelled at length so that the description cannot be made to fit " * 2
    (w3.mock / "c1_class").write_text("")
    r = _land(w3)
    out3 = _out(w3)
    assert r.returncode == 13 and "LANDING-GATE DESCRIPTION WOULD BE TRUNCATED (" in out3 and "the C1 clause would be cut off the evidence; NOT pushed" in out3, out3
    assert not _statuses(w3) and not [c for c in _calls(w3.env, "gh") if " pr " in c], "nothing posted, no PR"
    assert _git(w3.origin, "rev-parse", "refs/heads/main") == w3.main_sha, "nothing merged"
    assert "@ push:start" not in out3, "refused before the push"


def test_land_reuses_a_recorded_verdict_for_the_merged_tip_and_never_calls_the_gate(tmp_path):
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"BASELINE sha={w.main_sha[:8]}\nMERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    assert "[land] verdict=reuse" in out and "already recorded for merged tip" in out
    assert not _calls(w.env, "gate"), "a recorded verdict means no gate run at all — not even a baseline"
    assert [s for s, e in _steps(out) if e == "start"] == ["merge", "verdict", "live-db", "push", "status", "pr", "merge-pr", "cleanup"]
    sts = _statuses(w)
    assert len(sts) == 1 and f"/statuses/{w.tip} " in sts[0], sts
    assert f"-f description=recorded LAND verdict (verdict-earlier.out) on tree of {w.tip[:8]} vs baseline {w.main_sha[:8]} --jq " in sts[0], \
        "a reused verdict's status is labeled by its recorded source (never as a Slurm run) with the baseline THAT verdict names"
    assert _git(w.origin, "rev-parse", "refs/heads/main^2") == w.tip and _git(w.origin, "rev-parse", "refs/heads/main^1") == w.main_sha


def test_land_reuses_a_recorded_verdict_for_an_ancestor_with_the_identical_tree_and_gates_when_the_tree_differs(tmp_path):
    """A STACK: branch B is cut from A's tip and carries its own commit; B's merged tip T_B (over the moved main) is gated FIRST (its
    LAND verdict recorded, naming T_B); then A lands as a GitHub merge commit MC_A. B's landing re-merges: MC_A is not in B's history,
    so the merge mints a NEW tip N whose sha no verdict names but whose TREE is T_B's (main brought nothing B lacked). T_B is an
    ancestor of N with the identical tree -> the recorded LAND is reused (no gate run), the verdict file gains a
    `tree-identical-to=` line, the status description says so, and the live-DB step still runs. Real content on main in between changes
    the tree -> the Slurm gate runs as before."""
    w = _land_world(tmp_path)
    # the stacked branch B = A's tip + its own commit; its merged tip over the MOVED main is what a gate would have run on
    b2 = "stacked-branch"
    _git(w.repo, "fetch", "-q", "origin")
    _git(w.repo, "worktree", "add", "-q", "-b", b2, f"worktrees/{b2}", w.tip)
    wt2 = w.repo / "worktrees" / b2
    (wt2 / "stacked.txt").write_text("stacked work\n")
    _git(wt2, "add", "-A")
    _git(wt2, "commit", "-q", "-m", "stacked work")
    _git(wt2, "merge", "-q", "--no-edit", "origin/main")
    t_b = _git(wt2, "rev-parse", "HEAD")
    assert _git(wt2, "rev-parse", "HEAD^2") == w.main_sha, "T_B is B's merge over the moved main"
    (w.state / "verdicts").mkdir(parents=True, exist_ok=True)
    (w.state / "verdicts" / f"gate-land-{b2}.out").write_text(f"BASELINE sha={w.main_sha[:8]} status=PASS\nMERGE sha={t_b[:8]} status=PASS\nINTRODUCED (0):\n  (none)\nVERDICT: LAND\n")
    # A lands through the Slurm gate: main = MC_A (parents: the moved main, A's gated tip)
    (w.mock / "squeue.out").write_text("")
    (w.mock / "sacct.completed").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    mc_a = _git(w.origin, "rev-parse", "refs/heads/main")
    gate_calls = len(_calls(w.env, "gate"))
    # B lands: the re-merge over MC_A mints N (sha != T_B, tree == T_B's); the recorded verdict for T_B is reused by tree identity
    w2 = types.SimpleNamespace(**vars(w))
    w2.b = b2
    (w.mock / "merged").unlink()
    r = _land(w2)
    out2 = _out(w2)
    assert r.returncode == 0, r.stdout + r.stderr + out2
    n_tip = _git(w.origin, "rev-parse", "refs/heads/main^2")
    assert n_tip != t_b and _git(w.origin, "rev-parse", f"{n_tip}^{{tree}}") == _git(w.origin, "rev-parse", f"{t_b}^{{tree}}"), "a new tip with the gated tip's tree"
    assert _git(w.origin, "merge-base", "--is-ancestor", t_b, n_tip) == "" and f"merged tip={n_tip[:8]}" in out2
    tree8 = _git(w.origin, "rev-parse", f"{t_b}^{{tree}}")[:8]
    vf = w.state / "verdicts" / f"gate-land-{b2}.out"
    assert f"[land] gate verdict LAND recorded for gated tip {t_b[:8]} in {vf}; merged tip {n_tip[:8]} has the IDENTICAL tree {tree8} — reusing (D35)" in out2, out2
    assert "[land] verdict=reuse" in out2 and "[land] verdict=slurm-gate" not in out2
    assert len(_calls(w.env, "gate")) == gate_calls, "no gate ran for the stacked branch"
    assert re.search(rf"^MERGE sha={n_tip[:8]} tree-identical-to={t_b[:8]} tree={tree8} \d{{4}}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", vf.read_text(), re.M), vf.read_text()
    sts = [c for c in _statuses(w2) if f"/statuses/{n_tip} " in c]
    assert len(sts) == 1 and f"-f description=recorded LAND verdict (gate-land-{b2}.out) on tree of {n_tip[:8]} (verdict of {t_b[:8]}, identical tree) vs baseline {w.main_sha[:8]} --jq " in sts[0], sts
    assert len(_calls(w.env, "livedb")) == 2, "the live-DB step still runs on every landing"
    assert _git(w.origin, "rev-parse", "refs/heads/main^1") == mc_a and f"[land] LANDED {b2} as PR 42" in out2
    # the negative: main moves with REAL content before a third branch (cut at A's gated tip, no verdict of its own) lands -> the
    # only candidate verdicts name trees that differ from its merged tip's -> the Slurm gate runs
    gated_a = _git(w.origin, "rev-parse", f"{mc_a}^2")
    _git(w.seed, "fetch", "-q", "origin")
    _git(w.seed, "reset", "-q", "--hard", "origin/main")
    (w.seed / "more.txt").write_text("real content\n")
    _git(w.seed, "add", "-A")
    _git(w.seed, "commit", "-q", "-m", "main moved with content")
    _git(w.seed, "push", "-q", "origin", "main")
    new_main = _git(w.seed, "rev-parse", "HEAD")
    b3 = "third-branch"
    _git(w.repo, "fetch", "-q", "origin")
    _git(w.repo, "worktree", "add", "-q", "-b", b3, f"worktrees/{b3}", gated_a)
    (w.repo / "worktrees" / b3 / "third.txt").write_text("third\n")
    _git(w.repo / "worktrees" / b3, "add", "-A")
    _git(w.repo / "worktrees" / b3, "commit", "-q", "-m", "third")
    w3 = types.SimpleNamespace(**vars(w))
    w3.b = b3
    (w.mock / "merged").unlink()
    (w.mock / "sacct.completed").write_text("")
    r = _land(w3)
    out3 = _out(w3)
    assert r.returncode == 0, r.stdout + r.stderr + out3
    assert "[land] verdict=slurm-gate" in out3 and "IDENTICAL tree" not in out3 and len(_calls(w.env, "gate")) > gate_calls, out3
    assert f"--baseline-sha {new_main}" in _calls(w.env, "gate")[-1]


def test_land_runs_the_no_modal_scan_after_the_merge_reports_by_default_refuses_when_asked_and_skips_when_off(tmp_path):
    """C9: check_no_modal.sh runs inside step 1 over the merged tree. report (the default) prints the hits and goes on; refuse exits 14
    before any verdict, gate, push or PR; off skips the scan entirely."""
    w = _land_world(tmp_path, main_moves=False)
    wt = w.repo / "worktrees" / w.b
    (wt / "tests" / "test_registry_hit.py").write_text("import " + _M + "\n\n\ndef test_x():\n    assert True\n")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "a line the scan catches")
    w.tip = _git(wt, "rev-parse", "HEAD")
    (w.mock / "squeue.out").write_text("")
    (w.mock / "sacct.completed").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    assert "[land] no-modal scan mode=report hits=1 rc=0" in out and f"HIT {wt}/tests/test_registry_hit.py:1: import {_M}" in out, out
    assert out.index("no-modal scan mode=report") < out.index("[land] @ merge:end") < out.index("[land] @ verdict:start")
    assert re.search(r"^no-modal scan: 1 hit\(s\) in 1 file\(s\); repo .* \(2 files scanned\); operating executables 0 scanned, 0 documents listed, retired excluded; mode=report$", out, re.M), out
    # refuse: exit 14 before the verdict; nothing gated, pushed or merged
    for f in ("squeue.out", "sacct.completed"):
        (w.mock / f).write_text("")
    _git(w.seed, "fetch", "-q", "origin")
    gate_calls = len(_calls(w.env, "gate"))
    main_before = _git(w.origin, "rev-parse", "refs/heads/main")
    wt2 = w.repo / "worktrees" / "refused-branch"
    _git(w.repo, "worktree", "add", "-q", "-b", "refused-branch", str(wt2), "origin/main")
    (wt2 / "tests" / "test_registry_hit2.py").write_text("from " + _M + " import App\n")
    _git(wt2, "add", "-A")
    _git(wt2, "commit", "-q", "-m", "another")
    w2 = types.SimpleNamespace(**vars(w))
    w2.b = "refused-branch"
    # refuse REQUIRES the operating-scripts inventory (the scan is mandatory only once the inventory records zero live hits):
    # without LAND_NO_MODAL_INVENTORY the scanner cannot run under refuse, and the landing stops before any verdict
    w2.env = dict(w.env, LAND_NO_MODAL_SCAN="refuse")
    r = _land(w2)
    out2 = _out(w2)
    assert r.returncode == 14, r.stdout + r.stderr + out2
    assert "[land] no-modal scan mode=refuse hits=0 rc=2" in out2 and "NO-MODAL SCAN could not run (rc=2) under LAND_NO_MODAL_SCAN=refuse" in out2 and "NOT pushed" in out2, out2
    assert "[land] @ verdict:start" not in out2 and len(_calls(w.env, "gate")) == gate_calls
    # with the inventory (an operating tree with nothing left to scan is a legitimate inventory): the hit refuses, exit 14, before any verdict
    inv = w.mock / "operating-scripts-inventory.json"
    inv.write_text(json.dumps({"schema": "operating_scripts_inventory/1", "root": str(w.mock / "ops"), "files": []}))
    (w.state / "land" / "refused-branch.out").unlink()
    w2.env = dict(w.env, LAND_NO_MODAL_SCAN="refuse", LAND_NO_MODAL_INVENTORY=str(inv))
    r = _land(w2)
    out2 = _out(w2)
    assert r.returncode == 14, r.stdout + r.stderr + out2
    # two hits: the branch is cut from the main that just landed test_registry_hit.py, plus its own hit2
    assert "[land] no-modal scan mode=refuse hits=2 rc=1" in out2 and "NO-MODAL SCAN REFUSED merged tip" in out2 and "2 hit(s) under LAND_NO_MODAL_SCAN=refuse" in out2 and "NOT pushed" in out2, out2
    assert f"HIT {wt2}/tests/test_registry_hit2.py:1: from {_M} import App" in out2 and f"HIT {wt2}/tests/test_registry_hit.py:1: import {_M}" in out2 and "operating executables 0 scanned" in out2, out2
    assert "[land] @ verdict:start" not in out2 and len(_calls(w.env, "gate")) == gate_calls and not [c for c in _calls(w.env, "gh") if " pr create " in c and "refused-branch" in c]
    assert _git(w.origin, "rev-parse", "refs/heads/main") == main_before and _git(w.origin, "ls-remote", "--heads", str(w.origin), "refused-branch") == ""
    # off: no scan line at all; the hit lands (the scan is the operator's choice until the cutover flips it to refuse)
    w3 = types.SimpleNamespace(**vars(w2))
    w3.env = dict(w.env, LAND_NO_MODAL_SCAN="off")
    (w.state / "land" / "refused-branch.out").unlink()
    (w.mock / "merged").unlink()
    r = _land(w3)
    out3 = _out(w3)
    assert r.returncode == 0, r.stdout + r.stderr + out3
    assert "no-modal scan" not in out3 and "[land] LANDED refused-branch" in out3


RECOVER_SSH_SHIM = r'''#!/bin/bash
# the login node for recover.sh: cat / test run here (the cluster paths are local directories); the rollback moves CURRENT to the
# parent and records its call; tree_export over stdin creates the tree directory and answers TREE
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=$1; shift; cmd="$*"
echo "ssh $host $cmd" >> "$MOCK/ssh.calls"
case "$cmd" in
  *"replica rollback"*) echo "$cmd" >> "$MOCK/rollback.calls"; [ -f "$MOCK/rollback_fails" ] && { echo "replica rollback REFUSED: shim"; exit 10; }
    echo g1 > "$MOCK/replica/CURRENT"; echo '{"from": "g2", "to": "g1"}'; exit 0 ;;
  "bash -s --"*) cat >/dev/null; set -- $cmd; echo "export $4 $6" >> "$MOCK/export.calls"; mkdir -p "$6/src/sisteer/registry"; echo "TREE exported $6"; exit 0 ;;
  *) exec bash -c "$cmd" ;;
esac
'''


def _recover_world(tmp_path):
    bin_, mock_, shared, state = tmp_path / "bin", tmp_path / "mock", tmp_path / "shared", tmp_path / "state"
    for d in (bin_, mock_, shared / "gates" / "trees", shared / "envs", state):
        d.mkdir(parents=True)
    t1, t2 = "1" * 40, "2" * 40
    replica = shared / "registry" / "replica"
    for gen, parent, tree in (("g1", None, t1), ("g2", "g1", t2)):
        (replica / "gen" / gen).mkdir(parents=True)
        (replica / "gen" / gen / "meta.json").write_text(json.dumps({"schema": "replica_meta/1", "gen_id": gen, "parent_gen_id": parent, "built_by_tree": tree}))
    (replica / "CURRENT").write_text("g2\n")
    (mock_ / "replica").symlink_to(replica)
    # tree 1 exported (with its dependency files); tree 2 only partially (its dependency files, for the rollback's venv key) -- not exported
    (shared / "gates" / "trees" / t1 / "src" / "sisteer" / "registry").mkdir(parents=True)
    for t in (t1, t2):
        (shared / "gates" / "trees" / t).mkdir(parents=True, exist_ok=True)
        (shared / "gates" / "trees" / t / "pyproject.toml").write_text("[project]\nname='x'\n")
        (shared / "gates" / "trees" / t / "uv.lock").write_text("lock\n")
    import hashlib
    ph = hashlib.sha256(b"[project]\nname='x'\n").hexdigest()
    lh = hashlib.sha256(b"lock\n").hexdigest()
    h12 = hashlib.sha256(f"{ph}\n{lh}\n".encode()).hexdigest()[:12]
    venv = shared / "envs" / f"repo-gate-{h12}"           # the checkout below is named `repo`: the gate venv derives from that name
    venv.mkdir()
    (venv / ".gate-ready.json").write_text("{}")
    _exe(bin_ / "ssh", RECOVER_SSH_SHIM)
    _exe(mock_ / "land.sh", '#!/bin/bash\necho "land $*" >> "$MOCK/land.calls"; echo "LIVE_DB_REPLICA_TREE=${LIVE_DB_REPLICA_TREE:-unset}" >> "$MOCK/land.env"; exit ${LAND_SHIM_RC:-0}\n')
    _exe(mock_ / "mirror-repos.sh", MIRROR_SHIM)
    _exe(mock_ / "mirror_history.sh", HISTORY_SHIM)
    env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", MOCK=str(mock_), REGISTRY_LANDING_STATE=str(state), GATE_HOST="TESTHOST", GATE_RUSER="tester",
               GATE_SHARED=str(shared), LAND_SH=str(mock_ / "land.sh"), MIRROR_REPOS_SH=str(mock_ / "mirror-repos.sh"),
               MIRROR_HISTORY_SH=str(mock_ / "mirror_history.sh"), SISTEER_REPO=str(tmp_path / "repo"))
    body = tmp_path / "body.md"
    body.write_text("body\n")
    return types.SimpleNamespace(env=env, mock=mock_, shared=shared, state=state, replica=replica, t1=t1, t2=t2, body=body, venv=venv, repo=tmp_path / "repo")


def _recover(w, *args):
    return _run([LANDING / "recover.sh", *args], w.env, timeout=120)


def test_recover_pins_the_replica_side_to_the_current_good_generations_tree_exports_it_and_lands_the_branch(tmp_path):
    """recover.sh reads CURRENT and its meta.json on the cluster, exports the good generation's tree when
    the cluster lacks it, and runs land.sh with LIVE_DB_REPLICA_TREE = that tree; --rollback moves CURRENT to the parent first (under the
    current generation's own tree and venv) and lands against the parent's tree; --dry-run writes nothing; a missing CURRENT, meta or
    tree sha refuses before anything runs; --plan lists the four steps."""
    w = _recover_world(tmp_path)
    r = _recover(w, "fix-branch", "a fix", str(w.body))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert f"[recover] CURRENT g2 built by tree {w.t2[:12]}" in out and "[recover] would" not in out
    assert (w.mock / "export.calls").read_text() == f"export {w.t2} {w.shared}/gates/trees/{w.t2}\n", "tree 2 was exported (it was not on the cluster)"
    # the archived branches and tags of the checkout went to the mirror before the export (mirror_history.sh with SISTEER_REPO and the mirror's URL)
    assert (w.mock / "history.calls").read_text() == f"history {w.repo} ssh://TESTHOST{w.shared}/repos/repo\n"
    assert (w.mock / "land.calls").read_text() == f"land fix-branch a fix {w.body}\n" and (w.mock / "land.env").read_text() == f"LIVE_DB_REPLICA_TREE={w.t2}\n"
    stamps = re.findall(r"^\[recover\] @ ([a-z]+):(start|end) ", out, re.M)
    assert [s for s, e in stamps if e == "start"] == ["read", "tree", "land"] and all((s, "end") in stamps for s, _ in stamps)
    rec = json.load(open(w.state / "recover" / "fix-branch.json"))
    assert rec == {**rec, "branch": "fix-branch", "current_before": "g2", "generation": "g2", "built_by_tree": w.t2, "rolled_back": False, "tree_present": "exported", "land_rc": 0, "dry_run": False}
    assert not (w.mock / "rollback.calls").exists() and (w.replica / "CURRENT").read_text().strip() == "g2"
    # --rollback: CURRENT -> g1 under tree 2's code and venv; the landing pins tree 1 (present, no export)
    (w.mock / "export.calls").unlink()
    r = _recover(w, "fix-branch", "a fix", str(w.body), "--rollback")
    assert r.returncode == 0, r.stdout + r.stderr
    rb = (w.mock / "rollback.calls").read_text()
    assert f"env PYTHONPATH='{w.shared}/gates/trees/{w.t2}/src' '{w.venv}/bin/python' -m sisteer.registry.replica rollback --replica-root '{w.replica}'" in rb, rb
    assert "[recover] rolled back: CURRENT g2 -> g1 (built by tree 111111111111)" in r.stdout and (w.replica / "CURRENT").read_text().strip() == "g1"
    assert not (w.mock / "export.calls").exists() and (w.mock / "land.env").read_text().splitlines()[-1] == f"LIVE_DB_REPLICA_TREE={w.t1}"
    rec = json.load(open(w.state / "recover" / "fix-branch.json"))
    assert rec["rolled_back"] and rec["current_before"] == "g2" and rec["generation"] == "g1" and rec["built_by_tree"] == w.t1 and rec["tree_present"] == "existing"
    # land.sh's exit status is recover.sh's
    w.env["LAND_SHIM_RC"] = "5"
    r = _recover(w, "fix-branch", "a fix", str(w.body))
    assert r.returncode == 5 and "[recover] land.sh exited 5 for fix-branch" in r.stdout and json.load(open(w.state / "recover" / "fix-branch.json"))["land_rc"] == 5
    del w.env["LAND_SHIM_RC"]
    # --dry-run: nothing written, nothing landed, the would-run lines printed (tree 2's export from the first run removed first, so the
    # dry run has an export to name)
    (w.replica / "CURRENT").write_text("g2\n")
    shutil.rmtree(w.shared / "gates" / "trees" / w.t2 / "src")
    n_land = len((w.mock / "land.calls").read_text().splitlines())
    r = _recover(w, "fix-branch", "a fix", str(w.body), "--rollback", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DRY RUN" in r.stdout and "[recover] would run: ssh TESTHOST env PYTHONPATH=" in r.stdout and "[recover] would export tree 222222222222" in r.stdout
    assert f"[recover] would run: LIVE_DB_REPLICA_TREE={w.t2} " in r.stdout and len((w.mock / "land.calls").read_text().splitlines()) == n_land
    assert (w.replica / "CURRENT").read_text().strip() == "g2" and json.load(open(w.state / "recover" / "fix-branch.json"))["dry_run"]
    assert not (w.shared / "gates" / "trees" / w.t2 / "src").exists(), "a dry run exports nothing"
    # the dispatcher is probed BEFORE the rollback: a replica.sbatch without the land/2 contract -- an older script, vendored under
    # tests/fixtures/registry/ -- refuses by name with CURRENT untouched and no rollback, export or landing attempted
    import hashlib
    dsha12 = hashlib.sha256((LANDING / "replica.sbatch").read_bytes()).hexdigest()[:12]
    assert "[recover] dispatcher " in out and f"/replica.sbatch (sha256 {dsha12}; land/3 TREE HEAD VENV BRANCH REPLICA_ROOT OUT_DIR EXPECT_SKIPPED HASH_MODE REPLICA_TREE ACCEPT_C1_CLASS)" in out, out
    n_rb = len((w.mock / "rollback.calls").read_text().splitlines())
    w.env["LIVE_DB_DISPATCHER"] = str(REPO / "tests" / "fixtures" / "registry" / "replica_sbatch_pre_c9_bdd22205c.sh")
    r = _recover(w, "fix-branch", "a fix", str(w.body), "--rollback")
    assert r.returncode == 5 and "[recover] REFUSED: the dispatcher" in r.stdout and "carries no land/3 contract" in r.stdout and "unknown mode contract" in r.stdout, r.stdout
    assert (w.replica / "CURRENT").read_text().strip() == "g2" and len((w.mock / "rollback.calls").read_text().splitlines()) == n_rb
    assert len((w.mock / "land.calls").read_text().splitlines()) == n_land and not (w.shared / "gates" / "trees" / w.t2 / "src").exists()
    assert json.load(open(w.state / "recover" / "fix-branch.json"))["note"].startswith("refused: the dispatcher")
    del w.env["LIVE_DB_DISPATCHER"]
    # refusals before anything runs: a rollback that fails; a meta without a tree; no CURRENT
    (w.mock / "rollback_fails").write_text("")
    r = _recover(w, "fix-branch", "a fix", str(w.body), "--rollback")
    assert r.returncode == 5 and "REFUSED: replica rollback failed (rc=10)" in r.stdout and len((w.mock / "land.calls").read_text().splitlines()) == n_land
    (w.mock / "rollback_fails").unlink()
    (w.replica / "gen" / "g2" / "meta.json").write_text(json.dumps({"gen_id": "g2"}))
    r = _recover(w, "fix-branch", "a fix", str(w.body))
    assert r.returncode == 5 and "names no 40-hex built_by_tree" in r.stdout and len((w.mock / "land.calls").read_text().splitlines()) == n_land
    (w.replica / "CURRENT").unlink()
    r = _recover(w, "fix-branch", "a fix", str(w.body))
    assert r.returncode == 5 and "REFUSED: no CURRENT generation" in r.stdout and len((w.mock / "land.calls").read_text().splitlines()) == n_land
    assert json.load(open(w.state / "recover" / "fix-branch.json"))["note"].startswith("refused: no CURRENT")
    r = _recover(w, "--plan")
    steps = re.findall(r"^PLAN (\d+) (\S+) :: ", r.stdout, re.M)
    assert [(int(n), s) for n, s in steps] == [(1, "read"), (2, "rollback"), (3, "tree"), (4, "land")] and r.returncode == 0
    assert "replica.sbatch contract" in r.stdout and "never the pinned tree's" in r.stdout, "the plan names the chain's dispatcher and its probe"
    r = _recover(w, "only-one")
    assert r.returncode == 2 and "usage" in r.stderr


def test_land_records_a_user_waiver_verbatim_and_still_runs_the_live_db_files(tmp_path):
    w = _land_world(tmp_path, main_moves=False)
    w.env["GATE_WAIVED_BY_USER"] = "the user, 2026-09-26: waive for this landing only"
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    assert "[land] verdict=waiver" in out and "FULL SUITE WAIVED BY THE USER for merged tip" in out and w.env["GATE_WAIVED_BY_USER"] in out
    assert (w.state / "verdicts" / f"gate-land-{w.b}.out").read_text() == f"MERGE sha={w.tip[:8]} VERDICT: WAIVED-BY-USER ({w.env['GATE_WAIVED_BY_USER']})\n"
    assert not _calls(w.env, "gate") and len(_calls(w.env, "livedb")) == 1 and not _calls(w.env, "py")
    sts = _statuses(w)
    assert len(sts) == 1 and f"-f description=full suite waived by the user on tree of {w.tip[:8]}: {w.env['GATE_WAIVED_BY_USER']} --jq " in sts[0], sts
    assert "baseline" not in sts[0].split("-f description=", 1)[1], "no suite ran under a waiver, so its status names no baseline"


def test_land_pushes_nothing_on_a_hold_verdict_or_red_live_db_tests_or_a_failed_or_non_terminal_baseline(tmp_path):
    w = _land_world(tmp_path)
    (w.mock / "squeue.out").write_text("")
    (w.mock / "sacct.completed").write_text("")
    (w.mock / "hold").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 5 and "GATE NOT LAND" in out and "NOT pushed" in out, out
    assert not _calls(w.env, "livedb") and not _calls(w.env, "gh") and not _statuses(w), "a HOLD posts no landing-gate status"
    assert _git(w.origin, "ls-remote", "--heads", str(w.origin), w.b) == "", "nothing pushed"
    # red live-DB tests: the verdict is LAND now (recorded), pytest is red -> exit 5, nothing pushed
    (w.mock / "hold").unlink()
    (w.mock / "red").write_text("")
    (w.state / "land" / f"{w.b}.out").unlink()
    r = _land(w)
    out = _out(w)
    assert r.returncode == 5 and "LIVE-DB TESTS RED" in out and "LIVE-DB STEP ON SLURM FAILED (rc=5)" in out, out
    assert not _calls(w.env, "gh") and not _statuses(w) and _git(w.origin, "ls-remote", "--heads", str(w.origin), w.b) == ""
    # a FAILED baseline attempt refuses the landing (exit 6) instead of gating against a stale or absent baseline
    w2 = _land_world(tmp_path / "w2")
    (w2.mock / "squeue.out").write_text("")
    _sacct(w2.env, f"1001_0|FAILED|1:0|compute-1|gate-{w2.main_sha[:7]}-tester", f"1002|FAILED|1:0|compute-2|gate-{w2.main_sha[:7]}-fin-tester")
    r = _land(w2)
    out = _out(w2)
    assert r.returncode == 6 and "BASELINE for main" in out and "JOB_STATE=FAILED" in out and "NOT pushed" in out, out
    assert not any("--branch" in c for c in _calls(w2.env, "gate")) and not _calls(w2.env, "livedb") and not _calls(w2.env, "gh")
    # a summary file that is NOT terminal (a TIMEOUT baseline gate_slurm.sh installed before refusing it) does not end the wait: the
    # deadline refuses (exit 6) and no tip gate runs against it
    w3 = _land_world(tmp_path / "w3")
    (w3.state / "summaries").mkdir(parents=True)
    (w3.state / "summaries" / f"summary-{w3.main_sha}.json").write_text(json.dumps({"sha": w3.main_sha, "status": "TIMEOUT", "counts": None}))
    (w3.mock / "squeue.out").write_text("1001_[0-39]|PENDING|Priority||\n1002|PENDING|Dependency||\n")
    r = _land(w3)
    out = _out(w3)
    assert r.returncode == 6 and "not collected within 1 min" in out and "NOT pushed" in out, out
    assert not any("--branch" in c for c in _calls(w3.env, "gate")) and not _calls(w3.env, "livedb")


def test_land_keeps_the_merge_exit_status_and_leaves_the_worktree_when_the_merge_is_refused(tmp_path):
    """`gh pr merge` refused (rc 1) → exit 8, no LANDED line, the worktree and the branch stay, origin's main unmoved; a merge that exits 0
    without merging (the PR does not read merged) is refused the same way before any cleanup."""
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"MERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w.mock / "merge_refused").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 8 and "PR MERGE REFUSED (gh rc=1)" in out and "LANDED" not in out, out
    assert (w.repo / "worktrees" / w.b).is_dir() and w.b in _git(w.repo, "branch", "--list", w.b), "a refused merge leaves the worktree and the branch"
    assert f"-f description=recorded LAND verdict (verdict-earlier.out) on tree of {w.tip[:8]} --jq " in _statuses(w)[0], \
        "a recorded verdict without a BASELINE line names no baseline rather than guessing one"
    assert _git(w.origin, "rev-parse", "refs/heads/main") != w.tip
    assert [(s_, e) for s_, e in _stamps(out) if s_ == "merge-pr"] == [("merge-pr", "start")] and ("cleanup", "start") not in _stamps(out)
    (w.mock / "merge_refused").unlink()
    (w.mock / "merge_lies").write_text("")
    (w.state / "land" / f"{w.b}.out").unlink()
    r = _land(w)
    out = _out(w)
    assert r.returncode == 8 and "LANDED" not in out, out
    # gh reads the PR open (the real field set); git finds origin/main unmoved with no second parent: neither path proves a merge
    assert f"does not read merged after gh pr merge (gh: state=OPEN mergedAt=- mergeCommit=-; no commit on origin/main's first-parent line from {w.main_sha[:8]} back to {w.main_sha[:8]} has this tip {w.tip[:8]} as its second parent)" in out, out
    assert "merge proven by" not in out
    assert (w.repo / "worktrees" / w.b).is_dir() and _git(w.origin, "rev-parse", "refs/heads/main") != w.tip


def test_land_proves_the_merge_by_git_when_the_post_merge_read_returns_nothing(tmp_path):
    """The first real landing merged, then `gh pr view` returned nothing and the chain exited 8 ("does not read merged") with the
    merge done and origin/main moved -- the window after it had to be started by hand and the cleanup was skipped. A read that prints
    nothing (or errs) is not a refusal: git is the authority -- origin/main's merge commit has this tip as its second parent -- so the
    merge is proven by git, the first-parent verification runs as for a gh reading, the record line and the cleanup follow, exit 0, and
    the log names the path that proved it."""
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"BASELINE sha={w.main_sha[:8]}\nMERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w.mock / "pr_read_blank").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    mc = _git(w.origin, "rev-parse", "refs/heads/main")
    assert _git(w.origin, "rev-parse", f"{mc}^1") == w.main_sha and _git(w.origin, "rev-parse", f"{mc}^2") == w.tip, "the merge happened"
    assert f"[land] merge proven by git: gh read state=- mergedAt=-; origin/main {mc[:8]} carries merge commit {mc[:8]} with this tip {w.tip[:8]} as its second parent (0 later commit(s) on main's first-parent line)" in out, out
    assert "merge proven by gh" not in out and "does not read merged" not in out
    assert re.search(rf"^\[land\] merged onto {w.main_sha[:8]} as {mc[:8]} \(first parent verified; second parent {w.tip[:8]}\)$", out, re.M), out
    assert f"[land] LANDED {w.b} as PR 42 (tip {w.tip[:8]}, 1 round)" in out and ("cleanup", "end") in _stamps(out)
    assert not (w.repo / "worktrees" / w.b).exists() and w.b not in _git(w.repo, "branch", "--list", w.b), "the cleanup the exit-8 landing skipped"
    gh = _calls(w.env, "gh")
    assert gh[-1].startswith("gh pr view 42 ") and "--json state,mergedAt,mergeCommit" in gh[-1] and not any("--json merged," in c for c in gh), gh


def test_land_proves_the_merge_by_git_when_another_box_merged_a_later_pr_before_the_fetch(tmp_path):
    """The merge lock covers this box only: with the post-merge read blank, a chain on another box merges a later PR between our merge
    and our fetch, so origin/main's tip is THAT merge commit and ours sits one behind it on main's first-parent line. The fallback walks
    that line back to the checked MAIN, finds the commit whose second parent is this tip, verifies its first parent and lands (exit 0,
    the record line, the cleanup) -- a merge that happened is never reported as "not merged" because main moved on."""
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"BASELINE sha={w.main_sha[:8]}\nMERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w.mock / "pr_read_blank").write_text("")
    (w.mock / "later_merge").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    later = _git(w.origin, "rev-parse", "refs/heads/main")
    mc = (w.mock / "merged").read_text().strip()
    assert later == (w.mock / "later_merge").read_text().strip() and _git(w.origin, "rev-parse", f"{later}^1") == mc and _git(w.origin, "rev-parse", f"{later}^2") != w.tip
    assert _git(w.origin, "rev-parse", f"{mc}^1") == w.main_sha and _git(w.origin, "rev-parse", f"{mc}^2") == w.tip
    assert f"[land] merge proven by git: gh read state=- mergedAt=-; origin/main {later[:8]} carries merge commit {mc[:8]} with this tip {w.tip[:8]} as its second parent (1 later commit(s) on main's first-parent line)" in out, out
    assert re.search(rf"^\[land\] merged onto {w.main_sha[:8]} as {mc[:8]} \(first parent verified; second parent {w.tip[:8]}\)$", out, re.M), out
    assert "RACED" not in out and f"[land] LANDED {w.b} as PR 42 (tip {w.tip[:8]}, 1 round)" in out and ("cleanup", "end") in _stamps(out)
    assert not (w.repo / "worktrees" / w.b).exists() and w.b not in _git(w.repo, "branch", "--list", w.b)


def test_the_gh_shim_refuses_a_json_field_the_installed_gh_has_not_got_and_answers_what_the_chain_reads(tmp_path):
    """The shim's `--json` vocabulary is the real gh's (GH_PR_JSON_FIELDS): `merged`, the field the R9 shim answered, is
    refused like gh refuses it -- exit 1, `Unknown JSON field`, the listing on stderr, nothing on stdout -- and every field set land.sh
    reads is in the vocabulary and answered."""
    mock_ = tmp_path / "mock"
    mock_.mkdir()
    gh = _exe(tmp_path / "gh", GH_SHIM)
    env = {"PATH": os.environ["PATH"], "MOCK": str(mock_)}
    r = subprocess.run([str(gh), "pr", "view", "42", "-R", "example/repo", "--json", "merged,mergeCommit", "--jq", "."], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and r.stdout == "" and r.stderr.startswith('Unknown JSON field: "merged"\nAvailable fields:\n  additions\n'), r.stderr
    assert r.stderr.split("Available fields:\n", 1)[1].split() == list(GH_PR_JSON_FIELDS)
    r = subprocess.run([str(gh), "pr", "view", "42", "-R", "example/repo", "--json", "state,mergedAt,mergeCommit", "--jq", "."], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout == "OPEN - -\n", (r.stdout, r.stderr)
    (mock_ / "merged").write_text("a" * 40 + "\n")
    r = subprocess.run([str(gh), "pr", "view", "42", "-R", "example/repo", "--json", "state,mergedAt,mergeCommit", "--jq", "."], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout == f"MERGED 2026-01-01T00:00:00Z {'a' * 40}\n", (r.stdout, r.stderr)
    reads = re.findall(r'pr view "\$n" -R "\$REPO" --json ([A-Za-z,]+)', (LANDING / "land.sh").read_text())
    assert reads == ["mergeable,mergeStateStatus", "state,mergedAt,mergeCommit"], reads
    assert all(f in GH_PR_JSON_FIELDS for fs in reads for f in fs.split(",")) and "merged" not in GH_PR_JSON_FIELDS


def land_sh_json_reads() -> Tuple[str, ...]:
    """Every `--json` field land.sh reads of a PR (`gh pr view "$n" -R "$REPO" --json <fields>`), in the order the script reads them."""
    reads = re.findall(r'pr view "\$n" -R "\$REPO" --json ([A-Za-z,]+)', (LANDING / "land.sh").read_text())
    return tuple(dict.fromkeys(f for fs in reads for f in fs.split(",")))


def gh_vocabulary_report(version: str, installed: Sequence[str], reads: Sequence[str] = ()) -> str:
    """The pin (GH_PR_JSON_FIELDS, the shim's `--json` vocabulary) against an INSTALLED gh's own listing, the rule the host test applies.
    The one hard failure is the real risk: a field the chain READS (``reads``, land.sh's) that the installed gh has
    not got -- the rehearsal under the shim would answer it and the real landing would meet `Unknown JSON field`; that fails by name, the
    field and the installed gh's version in the message. Everything else is the tool's version moving and is REPORTED, never failed: fields
    the installed gh ships beyond the pin (a newer gh) and pinned fields it has not got that the chain never reads (an older gh; the first
    gate baseline red on this test was such a one, its listing three fields short of the pin, `baseRefOid` among them). Returns the one-line
    report, which the host test prints and every failure message carries."""
    installed_set = set(installed)
    extra = [f for f in installed if f not in GH_PR_JSON_FIELDS]
    unlisted = [f for f in GH_PR_JSON_FIELDS if f not in installed_set]
    missing_reads = [f for f in reads if f not in installed_set]
    report = (f"installed {version}: {len(installed)} --json field(s) against the pin's {len(GH_PR_JSON_FIELDS)}; "
              f"fields land.sh reads missing from the installed gh: {', '.join(missing_reads) or 'none'}; "
              f"pinned fields the installed gh has not got (allowed, unread by the chain): {', '.join(f for f in unlisted if f not in missing_reads) or 'none'}; "
              f"installed fields beyond the pin (allowed): {', '.join(extra) or 'none'}")
    assert not missing_reads, (f"the installed gh ({version}) has not got the --json field(s) {', '.join(missing_reads)} that land.sh reads: a landing "
                               f"under this gh would meet `Unknown JSON field` where the shim answers; the chain's reads must change or the gh must ({report})")
    return report


def _installed_gh_listing(gh: str, tmp_path) -> Tuple[str, Tuple[str, ...]]:
    """`gh --version`'s first line and the `--json` field listing the installed gh prints for an unknown field: `gh pr view --json <unknown>`
    lists the available fields before any network call (a placeholder token satisfies its auth check and nothing is contacted). Skips the
    test cleanly where gh prints no listing."""
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GH_CONFIG_DIR": str(tmp_path / "gh-config"), "GH_NO_UPDATE_NOTIFIER": "1",
           "GH_TOKEN": "placeholder-no-network-call"}
    v = subprocess.run([gh, "--version"], capture_output=True, text=True, env=env, timeout=60)
    version = ((v.stdout + v.stderr).strip().splitlines() or ["gh (version unknown)"])[0]
    r = subprocess.run([gh, "pr", "view", "1", "-R", "example/repo", "--json", "nonexistent"], capture_output=True, text=True, env=env, timeout=60)
    m = re.search(r"Available fields:\n((?:  \S+\n)+)", r.stderr + r.stdout)
    if not m:
        pytest.skip(f"gh ({version}) printed no field listing: " + (r.stderr + r.stdout).strip()[:200])
    return version, tuple(m.group(1).split())


def test_the_gh_shims_field_vocabulary_is_the_installed_ghs_own_listing(tmp_path):
    """A rehearsal under a shim certifies the chain's logic, not the tool's interface, so the shim's vocabulary
    is pinned to the tool's (the shim's own listing IS the pin: the test above). With the real gh on PATH, its own listing is held against
    the pin by :func:`gh_vocabulary_report`: every field land.sh reads must be in it (else fail by name with the installed version); the
    installed gh's version moving in either direction -- fields beyond the pin, pinned fields it has not got that the chain never reads --
    is reported, not failed, so a gate node's gh of another version does not turn the suite red (the equality this test asserted before did,
    on a gh three fields short of the pin). Skipped cleanly where gh is absent or prints no listing."""
    gh = shutil.which("gh")
    if not gh:
        pytest.skip("no gh binary on PATH")
    version, installed = _installed_gh_listing(gh, tmp_path)
    print(gh_vocabulary_report(version, installed, land_sh_json_reads()))


def test_the_gh_vocabulary_rule_allows_another_versions_listing_and_refuses_a_missing_read_field_by_name():
    """The rule on synthetic listings: a newer gh's extra field and an older gh's absent unread fields are reported and allowed; a field
    land.sh reads that the listing lacks fails by name with the version; the pin's own listing reports nothing missing or extra."""
    reads = land_sh_json_reads()
    assert reads == ("mergeable", "mergeStateStatus", "state", "mergedAt", "mergeCommit") and all(f in GH_PR_JSON_FIELDS for f in reads)
    same = gh_vocabulary_report("gh version 2.97.0 (2026-07-31)", GH_PR_JSON_FIELDS, reads)
    assert same.endswith("fields land.sh reads missing from the installed gh: none; pinned fields the installed gh has not got (allowed, unread by the chain): none; "
                         "installed fields beyond the pin (allowed): none"), same
    newer = gh_vocabulary_report("gh version 9.99.0 (2099-01-01)", ("aaNewerGhField",) + GH_PR_JSON_FIELDS, reads)
    assert "installed fields beyond the pin (allowed): aaNewerGhField" in newer and "missing from the installed gh: none" in newer, newer
    older = gh_vocabulary_report("gh version 2.30.0 (2023-06-20)", tuple(f for f in GH_PR_JSON_FIELDS if f not in ("baseRefOid", "fullDatabaseId", "headRefOid")), reads)
    assert "pinned fields the installed gh has not got (allowed, unread by the chain): baseRefOid, fullDatabaseId, headRefOid" in older, older
    with pytest.raises(AssertionError) as exc:
        gh_vocabulary_report("gh version 1.0.0 (2020-01-01)", tuple(f for f in GH_PR_JSON_FIELDS if f != "mergeCommit") + ("aaNewerGhField",), reads)
    msg = str(exc.value)
    assert "has not got the --json field(s) mergeCommit that land.sh reads" in msg and "gh version 1.0.0 (2020-01-01)" in msg
    assert "installed fields beyond the pin (allowed): aaNewerGhField" in msg          # the report rides every failure message


def _gh_of_another_version(tmp_path, name: str, real: str, sed: str, version: str) -> str:
    """A gh wrapper over the real one whose field listing is edited by `sed` (a field appended, a field dropped) and whose --version is
    `version`: the shape of a gate node's gh of another version than the pin's."""
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    text = ("#!/bin/bash\n"
            'if [ "$1" = "--version" ]; then echo ' + shlex.quote(version) + "; exit 0; fi\n"
            "out=$(" + shlex.quote(real) + ' "$@" 2>&1); rc=$?\n'
            "printf '%s\\n' \"$out\" | sed -e " + shlex.quote(sed) + " >&2\n"
            "exit $rc\n")
    return str(_exe(tmp_path / name / "gh", text))


def test_a_gh_of_another_version_passes_the_vocabulary_check_unless_it_lacks_a_field_land_sh_reads(tmp_path):
    """End to end over the real gh's listing (skipped where gh is absent): a newer gh (one field beyond the pin) and an older one (without
    `baseRefOid`, which land.sh never reads) pass with the difference named in the report; one without `mergeCommit` fails by name, the
    version in the message."""
    real = shutil.which("gh")
    if not real:
        pytest.skip("no gh binary on PATH")
    reads = land_sh_json_reads()
    newer = _gh_of_another_version(tmp_path, "newer", real, r"s/^Available fields:$/Available fields:\n  aaNewerGhField/", "gh version 9.99.0 (2099-01-01)")
    version, installed = _installed_gh_listing(newer, tmp_path / "h1")
    assert version == "gh version 9.99.0 (2099-01-01)" and installed[0] == "aaNewerGhField", (version, installed[:3])
    assert "installed fields beyond the pin (allowed): aaNewerGhField" in gh_vocabulary_report(version, installed, reads)
    older = _gh_of_another_version(tmp_path, "older", real, r"/^  baseRefOid$/d", "gh version 2.30.0 (2023-06-20)")
    version, installed = _installed_gh_listing(older, tmp_path / "h2")
    assert "baseRefOid" not in installed
    assert "pinned fields the installed gh has not got (allowed, unread by the chain): baseRefOid" in gh_vocabulary_report(version, installed, reads)
    lacking = _gh_of_another_version(tmp_path, "lacking", real, r"/^  mergeCommit$/d", "gh version 1.0.0 (2020-01-01)")
    version, installed = _installed_gh_listing(lacking, tmp_path / "h3")
    with pytest.raises(AssertionError, match=r"has not got the --json field\(s\) mergeCommit that land.sh reads") as exc:
        gh_vocabulary_report(version, installed, reads)
    assert "gh version 1.0.0 (2020-01-01)" in str(exc.value)


def test_land_re_merges_and_re_gates_when_main_moves_under_the_gated_tip(tmp_path):
    """origin/main advances while the live-DB files run (after the gate): the pre-merge check sees it, and the round repeats — re-merge,
    the Slurm gate against the NEW main's baseline (awaited the same way), the live-DB files again, the new tip pushed to the same PR —
    before the merge. The merged main contains the moved main; nothing ungated ever merges."""
    w = _land_world(tmp_path)
    (w.mock / "squeue.out").write_text("")
    (w.mock / "sacct.completed").write_text("")
    (w.seed / "later.txt").write_text("main moved again\n")
    _git(w.seed, "add", "-A")
    _git(w.seed, "commit", "-q", "-m", "main moved again")
    next_main = _git(w.seed, "rev-parse", "HEAD")
    _git(w.seed, "push", "-q", "origin", "HEAD:refs/heads/next")      # the object reaches origin; main advances during the live-DB step
    (w.mock / "advance_main").write_text(next_main)
    r = _land(w)
    out = _out(w)
    assert r.returncode == 0, r.stdout + r.stderr + out
    assert f"[land] re-gate round=2: origin/main moved {w.main_sha[:8]} → {next_main[:8]}" in out and "re-gate round=3" not in out, out
    tips = re.findall(r"merged tip=([0-9a-f]{8}) main=([0-9a-f]{8}) round=(\d)", out)
    assert [(m, rd) for _, m, rd in tips] == [(w.main_sha[:8], "1"), (next_main[:8], "2")], tips
    gate = [c.split()[1:] for c in _calls(w.env, "gate")]
    assert len(gate) == 6, gate
    assert gate[0] == [w.main_sha, "--baseline", "--no-wait"] and gate[1] == [w.main_sha, "--collect-only", "--baseline"]
    assert gate[2][1:] == ["--branch", w.b, "--baseline-sha", w.main_sha] and gate[2][0].startswith(tips[0][0])
    assert gate[3] == [next_main, "--baseline", "--no-wait"] and gate[4] == [next_main, "--collect-only", "--baseline"]
    assert gate[5][1:] == ["--branch", w.b, "--baseline-sha", next_main] and gate[5][0].startswith(tips[1][0]), "each round gates ITS merged tip against ITS main"
    assert len(_calls(w.env, "livedb")) == 2, "the live-DB step ran once per round"
    gh = _calls(w.env, "gh")
    assert sum(1 for c in gh if " pr create " in c) == 1 and sum(1 for c in gh if " pr merge " in c) == 1 and f"[land] PR 42 now carries merged tip {tips[1][0]}" in out
    sts = _statuses(w)
    assert [c.split()[4].rsplit("/", 1)[1][:8] for c in sts] == [tips[0][0], tips[1][0]], "each round posts the landing-gate status on ITS pushed tip"
    assert f"-f description=slurm gate LAND on tree of {tips[0][0]} vs baseline {w.main_sha[:8]} --jq " in sts[0]
    assert f"-f description=slurm gate LAND on tree of {tips[1][0]} vs baseline {next_main[:8]} --jq " in sts[1]
    assert gh.index(sts[1]) < gh.index(next(c for c in gh if " pr merge " in c)) and f"--match-head-commit {sts[1].split()[4].rsplit('/', 1)[1]}" in gh[-2]
    final = _git(w.origin, "rev-parse", "refs/heads/main")
    assert _git(w.origin, "rev-parse", f"{final}^2").startswith(tips[1][0]) and _git(w.origin, "rev-parse", f"{final}^1") == next_main
    assert _git(w.origin, "merge-base", "--is-ancestor", next_main, final) == ""
    assert f"[land] LANDED {w.b} as PR 42 (tip {tips[1][0]}, 2 rounds)" in out
    assert [s_ for s_, e in _stamps(out) if s_ == "merge" and e == "start"] == ["merge", "merge"]
    rows = _measure("landings").collect(str(w.state), [])
    assert rows[0]["landed"] and rows[0]["rounds"] == 2 and rows[0]["total_min"] is not None and rows[0]["total_min"] >= rows[0]["gate_min"]


def test_land_reports_raced_and_gates_post_hoc_when_main_moved_between_the_check_and_the_merge(tmp_path):
    """The check and the merge are not one server-side step: the gh shim advances main (another writer) after the chain's check passed and
    merges onto it — the world with ruleset 24155420 disabled, as created ($MOCK/ruleset_off): GitHub merges an out-of-date PR. The merge
    commit's first parent is not the checked MAIN → RACED, loudly; the baseline for the actual parent is awaited and the merge commit gated
    against it right there; exit 11; the worktree stays."""
    w = _land_world(tmp_path, main_moves=False)
    (w.mock / "squeue.out").write_text("")
    (w.mock / "sacct.completed").write_text("")
    (w.mock / "ruleset_off").write_text("")
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"MERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w.seed / "racer.txt").write_text("another writer\n")
    _git(w.seed, "add", "-A")
    _git(w.seed, "commit", "-q", "-m", "another writer")
    racer = _git(w.seed, "rev-parse", "HEAD")
    _git(w.seed, "push", "-q", "origin", "HEAD:refs/heads/racer")
    (w.mock / "race_main").write_text(racer)
    r = _land(w)
    out = _out(w)
    assert r.returncode == 11, r.stdout + r.stderr + out
    mc = _git(w.origin, "rev-parse", "refs/heads/main")
    assert _git(w.origin, "rev-parse", f"{mc}^1") == racer and _git(w.origin, "rev-parse", f"{mc}^2") == w.tip, "the merge happened, onto the racer"
    assert f"RACED: merged onto {racer} not {w.main_sha}; the combination was not gated (merge commit {mc}" in out, out
    assert f"[land] post-hoc gate: the baseline for {racer[:8]}, then merge commit {mc[:8]} against it" in out
    gate = _calls(w.env, "gate")
    assert gate == [f"gate {racer} --baseline --no-wait", f"gate {racer} --collect-only --baseline", f"gate {mc} --branch {w.b}-posthoc --baseline-sha {racer}"], gate
    assert (v / f"gate-land-{w.b}-posthoc.out").is_file() and f"MERGE sha={mc[:8]}" in (v / f"gate-land-{w.b}-posthoc.out").read_text()
    assert "VERDICT: LAND" in out and f"RACED landing of {w.b} as PR 42" in out and "LANDED" not in out
    assert (w.repo / "worktrees" / w.b).is_dir() and ("cleanup", "start") not in _stamps(out)
    assert [s_ for s_, e in _steps(out) if e == "start"] == ["merge", "verdict", "live-db", "push", "status", "pr", "merge-pr", "posthoc"]
    assert not any("--match-head-commit" in c and w.tip not in c for c in _calls(w.env, "gh") if " pr merge " in c)
    assert len(_statuses(w)) == 1 and f"/statuses/{w.tip} " in _statuses(w)[0], "the chain posts the status whether or not the ruleset is active"


def test_land_is_refused_by_the_active_ruleset_when_main_moves_between_the_check_and_the_merge(tmp_path):
    """The same race with ruleset 24155420 ACTIVE (the shim's default): main advances after the chain's last fetch, and GitHub refuses the
    merge because the head no longer contains main's tip (strict up-to-date). Nothing merges, main carries only the other writer's commit,
    no post-hoc gate runs, exit 8 — and the chain's line names the cause it finds (origin/main moved; head behind main; re-merge and
    re-gate), since its own re-gate rounds catch only a move seen at fetch time."""
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"BASELINE sha={w.main_sha[:8]} status=PASS\nMERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w.seed / "racer.txt").write_text("another writer\n")
    _git(w.seed, "add", "-A")
    _git(w.seed, "commit", "-q", "-m", "another writer")
    racer = _git(w.seed, "rev-parse", "HEAD")
    _git(w.seed, "push", "-q", "origin", "HEAD:refs/heads/racer")
    (w.mock / "race_main").write_text(racer)
    r = _land(w)
    out = _out(w)
    assert r.returncode == 8, r.stdout + r.stderr + out
    assert "head branch is behind the base branch" in out, "GitHub's refusal reaches the landing log"
    assert re.search(rf"^PR MERGE REFUSED \(gh rc=1\): origin/main moved {w.main_sha[:8]} → {racer[:8]} between the check and the merge — "
                     rf"head behind main; re-merge and re-gate \(land again\) — NOT merged; the worktree and the branch stay$", out, re.M), out
    assert _git(w.origin, "rev-parse", "refs/heads/main") == racer, "main carries the other writer's commit and nothing of ours"
    assert not (w.mock / "merged").exists() and "RACED" not in out and "LANDED" not in out and "posthoc" not in out
    assert not any("--branch" in c for c in _calls(w.env, "gate")), "no post-hoc gate: nothing merged to gate"
    assert (w.repo / "worktrees" / w.b).is_dir() and w.b in _git(w.repo, "branch", "--list", w.b)
    assert [s_ for s_, e in _steps(out) if e == "start"] == ["merge", "verdict", "live-db", "push", "status", "pr", "merge-pr"]
    assert len(_statuses(w)) == 1 and f"/statuses/{w.tip} " in _statuses(w)[0], "the status was on the head; the head was simply behind"
    # the ordinary refusal (main unmoved) names no cause it cannot see
    (w.mock / "race_main").unlink()
    (w.mock / "merge_refused").write_text("")
    (w.state / "land" / f"{w.b}.out").unlink()
    _git(w.origin, "update-ref", "refs/heads/main", w.main_sha)
    r = _land(w)
    out = _out(w)
    assert r.returncode == 8 and re.search(r"^PR MERGE REFUSED \(gh rc=1\) — NOT merged; the worktree and the branch stay$", out, re.M), out


def test_land_stops_before_any_pr_or_merge_when_the_landing_gate_status_post_fails(tmp_path):
    """The `landing-gate` status POST on the pushed head fails (gh api rc 1): exit 13 right there — the head IS pushed (the status comes
    after the push), no PR is created, nothing merges, origin's main is unmoved, the worktree and the branch stay, and the `status`
    stamp opens without closing (measure/landings.py reads the file as an unfinished landing)."""
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"BASELINE sha={w.main_sha[:8]} status=PASS\nMERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    (w.mock / "status_fails").write_text("")
    r = _land(w)
    out = _out(w)
    assert r.returncode == 13, r.stdout + r.stderr + out
    assert f"LANDING-GATE STATUS NOT POSTED on merged tip {w.tip[:8]}" in out and "NOT merged" in out and "LANDED" not in out, out
    assert "shim refusal" in out, "gh's own error reaches the landing log"
    gh = _calls(w.env, "gh")
    assert len(gh) == 1 and gh[0].startswith(f"gh api -X POST repos/example/repo/statuses/{w.tip} -f state=success -f context=landing-gate "
                                             f"-f description=recorded LAND verdict (verdict-earlier.out) on tree of {w.tip[:8]} vs baseline {w.main_sha[:8]} --jq "), gh
    assert not any(" pr " in c for c in gh), "no PR was created and no merge attempted"
    assert _git(w.origin, "rev-parse", f"refs/heads/{w.b}") == w.tip, "the head was pushed before the status was attempted"
    assert _git(w.origin, "rev-parse", "refs/heads/main") == w.main_sha and not (w.mock / "statuses").exists()
    assert (w.repo / "worktrees" / w.b).is_dir() and w.b in _git(w.repo, "branch", "--list", w.b)
    steps = _steps(out)
    assert [s_ for s_, e in steps if e == "start"] == ["merge", "verdict", "live-db", "push", "status"] and ("status", "end") not in steps, steps
    rows = _measure("landings").collect(str(w.state), [])
    assert len(rows) == 1 and not rows[0]["landed"] and rows[0]["pr"] is None


def test_gate_status_refuses_a_short_sha_truncates_the_description_and_names_the_repo_once(tmp_path):
    """`gate_status.sh`, the one helper every landing path posts through: a sha that is not 40 hex, a missing description or an unknown
    state exit 2 before any call; the default posts `landing-gate success` to the repository `land.sh --repo` names, the slug of the
    checkout's github.com origin (one definition — LAND_REPO overrides both), and prints one confirmation line naming the SUPPLIED sha
    (the response's url ends in the status record, not the commit) after CHECKING the response's context and state (exit 3 otherwise); a description over 140 characters is cut to 137 + an
    ellipsis; no token is printed."""
    env = _refresh_env(tmp_path, shim_git=False)
    _exe(Path(env["PATH"].split(":")[0]) / "gh", GH_SHIM)
    env.update({"LAND_GH": "gh", "GH_TOKEN": "rehearsal-credential"})
    env.pop("LAND_REPO", None)
    _git(tmp_path, "init", "-q", env["SISTEER_REPO"])          # the checkout land.sh --repo reads the slug off (a synthetic origin)
    _git(env["SISTEER_REPO"], "remote", "add", "origin", "https://github.com/example-org/example-repo.git")
    gs = LANDING / "gate_status.sh"
    evidence = "slurm gate LAND on tree of 01234567 vs baseline 89abcdef"
    for argv, why in (([SHA_A[:8], evidence], "FULL 40-hex"), ([SHA_A[:39] + "g", evidence], "FULL 40-hex"), ([SHA_A], "usage"),
                      ([SHA_A, evidence, "green"], "state must be")):
        r = _run([gs, *argv], env)
        assert r.returncode == 2 and why in r.stderr and not _calls(env, "gh"), (argv, r.stdout, r.stderr)
    r = _run([gs, SHA_A, evidence], env)
    assert r.returncode == 0, r.stdout + r.stderr
    gh = _calls(env, "gh")
    assert gh == [f"gh api -X POST repos/example-org/example-repo/statuses/{SHA_A} -f state=success -f context=landing-gate "
                  f"-f description={evidence} --jq " + gh[0].split(" --jq ", 1)[1]], gh
    assert r.stdout == f"landing-gate success on {SHA_A[:8]}: {evidence}\n" and r.stderr == ""
    assert (Path(env["MOCK"]) / "statuses" / SHA_A).read_text() == f"landing-gate success {evidence}\n"
    long = "evidence " * 30
    r = _run([gs, SHA_A, long, "failure"], dict(env, LAND_REPO="example/repo"))
    assert r.returncode == 0, r.stdout + r.stderr
    call = _calls(env, "gh")[-1]
    assert call.startswith(f"gh api -X POST repos/example/repo/statuses/{SHA_A} -f state=failure -f context=landing-gate -f description=")
    desc = re.search(r"-f description=(.*) --jq ", call).group(1)
    assert len(desc) == 140 and desc == long[:137] + "..."
    assert r.stdout == f"landing-gate failure on {SHA_A[:8]}: {desc}\n"
    assert "rehearsal-credential" not in r.stdout + r.stderr + "\n".join(_calls(env, "gh"))
    (Path(env["MOCK"]) / "status_echo_wrong").write_text("")
    r = _run([gs, SHA_A, evidence], env)
    assert r.returncode == 3 and "context=ci state=success, not landing-gate success" in r.stderr and r.stdout == "", (r.stdout, r.stderr)
    (Path(env["MOCK"]) / "status_echo_wrong").unlink()
    (Path(env["MOCK"]) / "status_fails").write_text("")
    r = _run([gs, SHA_A, evidence], env)
    assert r.returncode == 1 and "shim refusal" in r.stderr and r.stdout == "", "gh api's own exit status and error are kept"


def test_the_repository_is_land_repo_else_the_checkouts_github_origin_and_nothing_else(tmp_path):
    """`land.sh --repo`, the one definition gate_status.sh reads: LAND_REPO when set, else the owner/name of the checkout's origin
    remote (the remote every landing step fetches from and pushes to), in each form git writes a github.com URL in. There is no
    built-in name: an origin elsewhere, or none, exits 2 naming LAND_REPO with nothing on stdout and no credential of the URL echoed;
    gate_status.sh then refuses before any call, and a landing refuses before it writes anything. Synthetic remotes, no host."""
    env = _refresh_env(tmp_path, shim_git=False)
    _exe(Path(env["PATH"].split(":")[0]) / "gh", GH_SHIM)
    env["LAND_GH"] = "gh"
    env.pop("LAND_REPO", None)
    repo = Path(env["SISTEER_REPO"])
    _git(tmp_path, "init", "-q", str(repo))
    land, gs = LANDING / "land.sh", LANDING / "gate_status.sh"

    def with_origin(url):
        if "origin" in _git(repo, "remote").split():
            _git(repo, "remote", "remove", "origin")
        if url is not None:
            _git(repo, "remote", "add", "origin", url)
        return _run([land, "--repo"], env)
    for url in ("https://github.com/example-org/example-repo.git", "https://github.com/example-org/example-repo/",
                "git@github.com:example-org/example-repo.git", "ssh://git@github.com/example-org/example-repo",
                "https://someuser@github.com/example-org/example-repo.git"):
        r = with_origin(url)
        assert r.returncode == 0 and r.stdout == "example-org/example-repo\n" and r.stderr == "", (url, r.stdout, r.stderr)
    assert _run([land, "--repo"], dict(env, LAND_REPO="example/other")).stdout == "example/other\n"
    userinfo = "someuser:" + "x" * 8 + "@"                       # assembled here: no credential-shaped literal in the source
    not_github = "has an origin remote that is not a github.com repository ({})\n"
    for url, said in ((str(tmp_path / "origin.git"), not_github.format(tmp_path / "origin.git")),
                      ("node-x:/data/repos/example", not_github.format("node-x:/data/repos/example")),
                      ("https://gitlab.example.invalid/example-org/example-repo.git",
                       not_github.format("https://gitlab.example.invalid/example-org/example-repo.git")),
                      (f"https://{userinfo}git.example.invalid/example.git", not_github.format("https://git.example.invalid/example.git")),
                      (None, "has no origin remote\n")):
        r = with_origin(url)
        assert r.returncode == 2 and r.stdout == "" and r.stderr.startswith(
            f"land.sh: set LAND_REPO (owner/name of the GitHub repository the landing opens and merges its PR in): the checkout {repo} "), (url, r)
        assert r.stderr.endswith(said) and userinfo not in r.stderr, (url, r.stderr)
    r = _run([gs, SHA_A, "slurm gate LAND on tree of 01234567 vs baseline 89abcdef"], env)
    assert r.returncode == 2 and "set LAND_REPO" in r.stderr and r.stdout == "" and not _calls(env, "gh"), (r.stdout, r.stderr)
    assert not Path(env["STATE"]).exists(), "neither refusal wrote state"
    # a landing from a checkout whose origin is not on GitHub refuses before its first step, with nothing written or pushed
    w = _land_world(tmp_path / "land", main_moves=False)
    w.env.pop("LAND_REPO")
    r = _land(w)
    assert r.returncode == 2 and "set LAND_REPO" in r.stderr and "is not a github.com repository" in r.stderr, (r.stdout, r.stderr)
    assert not (w.state / "land").exists() and not _calls(w.env, "gh")
    assert _git(w.origin, "ls-remote", "--heads", str(w.origin), w.b) == "", "nothing pushed"


def test_land_waits_on_the_merge_lock_and_refuses_when_another_landing_holds_it(tmp_path):
    """The check-and-merge runs under $STATE/merge.lock: while another landing holds it, this one waits (LAND_MERGE_LOCK_WAIT_S) and then
    refuses with exit 12 — no merge call, the worktree stays."""
    w = _land_world(tmp_path, main_moves=False)
    v = w.state / "verdicts"
    v.mkdir(parents=True)
    (v / "verdict-earlier.out").write_text(f"MERGE sha={w.tip[:8]} status=PASS\nVERDICT: LAND\n")
    lock = w.state / "merge.lock"
    held = open(lock, "w")                      # the other landing: the same flock(2) on the same file, held by this process
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        w.env["LAND_MERGE_LOCK_WAIT_S"] = "1"
        r = _land(w)
        out = _out(w)
        assert r.returncode == 12 and f"MERGE LOCK {lock} held by another landing for 1 s" in out and "LANDED" not in out, out
        assert not any(" pr merge " in c for c in _calls(w.env, "gh")) and (w.repo / "worktrees" / w.b).is_dir()
        assert _git(w.origin, "rev-parse", "refs/heads/main") == w.main_sha
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        held.close()
    # the holder gone, the same landing goes through
    (w.state / "land" / f"{w.b}.out").unlink()
    r = _land(w)
    assert r.returncode == 0 and f"[land] LANDED {w.b} as PR 42" in _out(w), r.stdout + r.stderr + _out(w)


# ---- the shared live snapshot on synthetic databases ------------------------------------------------------------------------
def _schema_db(path: Path, version: int | None = None) -> str:
    from sisteer.registry import migrations as M
    from sisteer.registry import schema_def as S
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(S.render_schema_sql())
        cols = [c[1] for c in conn.execute("PRAGMA table_info(schema_migration)")]
        row = {"version": M.code_version() if version is None else version}
        for c in cols:
            row.setdefault(c, "synthetic" if c == "description" else "2026-09-26T00:00:00.000Z")
        conn.execute(f"INSERT INTO schema_migration ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
        conn.commit()
    finally:
        conn.close()
    return str(path)


def _withdraw(db: str, run_id: str, anchor: str, rev: str) -> None:
    """Commit a withdrawn run_contrast observation (current = 0) for (run, anchor) on a plain writer connection."""
    key = json.dumps({"run_id": run_id, "anchor_run_id": anchor, "judgment_id": "j", "metric_id": "asr", "threshold": 0.5, "computation_id": ""}, sort_keys=True)
    conn = sqlite3.connect(db)
    try:
        conn.execute("INSERT INTO observation (source_id, revision_id, observation_key, table_name, current, withdrawn_by_revision_id, record_uri, ingest_seq) "
                     "VALUES ('s', ?, ?, 'run_contrast', 0, ?, 'synthetic', 1)", (rev, key, rev))
        conn.commit()
    finally:
        conn.close()


def _deploy_other_text(db: str) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute("DROP VIEW v_paper_long")
        conn.execute("CREATE VIEW v_paper_long AS SELECT 'r1' AS run_id, 'j' AS judgment_id, 'asr' AS metric_id, 0.5 AS threshold, '' AS computation_id, "
                     "NULL AS anchor_run_id, 0.25 AS value")
        conn.commit()
    finally:
        conn.close()


def test_live_snapshot_equal_text_pending_migration_and_drift(tmp_path):
    import registry_live_fixtures as F
    from sisteer.registry import db as DB
    from sisteer.registry import migrations as M
    from sisteer.registry import schema_def as S
    # EQUAL: the deployed object is the code's text (db.ddl_master vs db.code_ddl_master), no migration pending
    same = _schema_db(tmp_path / "same.sqlite")
    a = F.load_live_paper_long(same)
    assert a.candidate_is_deployed and not a.migration_pending and a.migration_version == a.code_version == M.code_version()
    assert a.deployed_sql == DB.code_ddl_master()[("view", "v_paper_long")] == DB.normalize_ddl(S.VIEW_SQL["v_paper_long"])
    assert a.rows == () and "run_id" in a.columns and "anchor_run_id" in a.columns and a.withdrawn_anchor_keys == frozenset()
    assert "IS the deployed text" in a.status_line()
    # DIFFERENT under a PENDING migration (the database one version behind the code): accepted, the snapshot is the deployed object's
    pend = _schema_db(tmp_path / "pending.sqlite", version=M.code_version() - 1)
    _deploy_other_text(pend)
    d = F.load_live_paper_long(pend)
    assert not d.candidate_is_deployed and d.migration_pending and d.migration_version == M.code_version() - 1
    assert d.rows == ({"run_id": "r1", "judgment_id": "j", "metric_id": "asr", "threshold": 0.5, "computation_id": "", "anchor_run_id": None, "value": 0.25},)
    assert "DIFFERS" in d.status_line() and f"migration {M.code_version()} is pending" in d.status_line()
    # DIFFERENT with NO pending migration: schema drift, refused loudly before any scan — never read as "a migration in flight"
    drift = _schema_db(tmp_path / "drift.sqlite")
    _deploy_other_text(drift)
    with pytest.raises(F.PaperLongDrift, match="SCHEMA DRIFT"):
        F.load_live_paper_long(drift)
    # no object at all
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(str(empty)).close()
    with pytest.raises(LookupError):
        F.load_live_paper_long(str(empty))


def test_live_snapshot_reads_rows_and_withdrawn_keys_in_one_transaction(tmp_path):
    """A withdrawal committed between the fixture's read and the consumer's check cannot reach the consumer: the withdrawn keys are
    part of the snapshot (the same read transaction as the rows), never a second connection's later read."""
    import registry_live_fixtures as F
    db = _schema_db(tmp_path / "snap.sqlite")
    _withdraw(db, "r1", "a0", "rev2")
    snap = F.load_live_paper_long(db)
    assert snap.withdrawn_anchor_keys == frozenset({("r1", "j", "asr", 0.5, "", "a0")})
    _withdraw(db, "r9", "a7", "rev3")                     # lands after the snapshot, before any consumer looks
    assert snap.withdrawn_anchor_keys == frozenset({("r1", "j", "asr", 0.5, "", "a0")}), "the consumer's keys are the snapshot's"
    later = F.load_live_paper_long(db)
    assert later.withdrawn_anchor_keys == frozenset({("r1", "j", "asr", 0.5, "", "a0"), ("r9", "j", "asr", 0.5, "", "a7")})
    assert F.anchor_key({"run_id": "r", "judgment_id": "j", "metric_id": "m", "threshold": "0.5", "computation_id": "", "anchor_run_id": "a"}) == ("r", "j", "m", 0.5, "", "a")


def _tests_local_modules_loaded_by(name: str, seen: set | None = None) -> set:
    """The tests/-local modules ``tests/<name>.py`` needs to LOAD, transitively: every top-level ``import x`` /
    ``from x import ...`` whose ``tests/x.py`` exists, plus a conftest's ``pytest_plugins``. A rehearsal that copies conftest.py
    into a synthetic suite copies these with it — conftest's imports are loud on purpose (a guard that is silently absent is
    worse than a red test), so the list follows the file rather than a name typed here, which is what went stale on main at
    ddd09e258 (2026-09-29): this rehearsal named only the fixtures plugin, while conftest had come to import
    ``_full_suite_guard`` — each green alone, red together."""
    seen = set() if seen is None else seen
    for node in ast.parse((REPO / "tests" / f"{name}.py").read_text()).body:
        if isinstance(node, ast.Import):
            found = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found = {node.module.split(".")[0]}
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in node.targets):
            plugins = ast.literal_eval(node.value)      # pytest accepts one name as a bare string or a sequence of names
            found = {plugins} if isinstance(plugins, str) else set(plugins)
        else:
            continue
        for dep in sorted(found - seen):
            if (REPO / "tests" / f"{dep}.py").is_file():
                seen.add(dep)
                _tests_local_modules_loaded_by(dep, seen)
    return seen


def test_live_paper_long_is_registered_once_by_conftest_and_scans_once(tmp_path):
    """Two test modules request `live_paper_long`; conftest registers the fixture module once (pytest_plugins), so the session fixture
    runs — and scans — once. Rehearsed in a subprocess pytest over copies of conftest.py and every tests/-local module it loads
    (``_tests_local_modules_loaded_by``: the fixtures plugin and the whole-suite guard today), REGISTRY_LIVE_DB pointing at a
    synthetic database; the fixture's status line is the scan's witness."""
    db = _schema_db(tmp_path / "live.sqlite")
    suite = tmp_path / "suite"
    suite.mkdir()
    modules = _tests_local_modules_loaded_by("conftest")
    assert {"registry_live_fixtures", "_full_suite_guard"} <= modules, modules
    for name in ("conftest", *sorted(modules)):
        shutil.copy(REPO / "tests" / f"{name}.py", suite / f"{name}.py")
    assert 'pytest_plugins = ["registry_live_fixtures"]' in (suite / "conftest.py").read_text()
    for name in ("test_one.py", "test_two.py"):
        (suite / name).write_text("def test_it(live_paper_long):\n    assert live_paper_long.rows == () and live_paper_long.candidate_is_deployed\n")
    env = dict(os.environ, REGISTRY_LIVE_DB=db, PYTHONPATH=f"{REPO / 'src'}:{suite}")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "--rootdir", str(suite),
                        str(suite / "test_one.py"), str(suite / "test_two.py")], capture_output=True, text=True, env=env, cwd=str(suite), timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 passed" in r.stdout and r.stdout.count("live v_paper_long:") == 1, r.stdout


def test_live_db_marker_is_registered_and_carried_by_exactly_the_modules_that_open_the_live_registry():
    """The rule: a tests/test_registry_*.py module opens the live database (the path literal, or the shared snapshot fixture) iff it carries
    `pytestmark = pytest.mark.live_db` — the set land.sh's local step runs (`--live-db-files`) and the cluster gate never sees. The
    expected set is DERIVED from the scan of openers; LIVE_DB_MODULES is a floor (a module that stops opening it must be removed there)."""
    text = (REPO / "pyproject.toml").read_text()
    assert re.search(r'^\s*"live_db: ', text, re.M)
    live_literal = "expanduser(" + '"~/registry/registry.sqlite")'      # assembled so this module does not match its own scan
    live_env = "REGISTRY_LIVE_DB" + '", "~/registry/registry.sqlite"'   # the env-keyed spelling since C8 (the replica's copy on Slurm)
    opens = sorted(p.name for p in (REPO / "tests").glob("test_registry_*.py") if p.name != Path(__file__).name
                   and (live_literal in p.read_text() or live_env in p.read_text() or re.search(r"\blive_paper_long\b", p.read_text())))
    marked = sorted(p.name for p in (REPO / "tests").glob("test_registry_*.py") if re.search(r"^pytestmark\s*=\s*pytest\.mark\.live_db", p.read_text(), re.M))
    assert marked == opens, ("every module that opens the live database carries the marker, and no other does", sorted(set(marked) ^ set(opens)))
    assert set(LIVE_DB_MODULES) <= set(opens), sorted(set(LIVE_DB_MODULES) - set(opens))
    r = _run([LANDING / "land.sh", "--live-db-files"], dict(os.environ), cwd=str(REPO))
    assert r.returncode == 0 and sorted(Path(p).name for p in r.stdout.split()) == opens, r.stdout + r.stderr


# ---- measure/landings.py and measure/durations.py on synthetic logs ---------------------------------------------------------
def test_landings_parses_the_legacy_shape_and_the_phase2_shape(tmp_path):
    L = _measure("landings")
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    end = datetime(2026, 9, 24, 17, 29, 0, tzinfo=timezone.utc)
    (legacy_dir / "land-orre.out").write_text(
        "[land] full registry suite on Modal for merged tip cc15ef49 16:42:37\nVERDICT: LAND\n"
        "[land] live-DB test files on merged tip cc15ef49 (23 files) 17:19:29\n[land] pushed cc15ef49a\n"
        "https://github.com/example/repo/pull/395\nPR 395: MERGEABLE CLEAN\n[land] origin/main eae5955e2\n"
        "Deleted branch registry-probe-branch (was cc15ef49a).\nLAND_RC=0\n")
    (legacy_dir / "land-registry-probe-branch.pytest.log").write_text("warnings\n")
    os.utime(legacy_dir / "land-registry-probe-branch.pytest.log", (0, (end - timedelta(minutes=0.5)).timestamp()))
    os.utime(legacy_dir / "land-orre.out", (0, end.timestamp()))
    (legacy_dir / "land-refused.out").write_text("GATE: no recorded LAND verdict for merged tip dd5c74b5 and Modal is disabled — re-gate on Slurm; NOT pushed\nLAND_RC=6\n")
    (legacy_dir / "land-wrap.out").write_text("[land] gate verdict LAND already recorded for merged tip 0ba675f8 in /x/gate-land-registry-wrap.out — reusing\n"
                                       "[land] live-DB test files on merged tip 0ba675f8 (23 files) 23:50:00\nhttps://github.com/example/repo/pull/7\n"
                                       "Deleted branch registry-wrap (was 0ba675f88).\nLAND_RC=0\n")
    wrap_end = datetime(2026, 9, 26, 0, 5, 0, tzinfo=timezone.utc)
    (legacy_dir / "land-registry-wrap.pytest.log").write_text("")
    os.utime(legacy_dir / "land-registry-wrap.pytest.log", (0, (wrap_end - timedelta(minutes=1)).timestamp()))
    os.utime(legacy_dir / "land-wrap.out", (0, wrap_end.timestamp()))
    state = tmp_path / "state"
    (state / "land").mkdir(parents=True)
    t0 = datetime(2026, 9, 29, 1, 0, 0, tzinfo=timezone.utc)
    stamps = [("merge", 0, 1), ("verdict", 1, 31), ("baseline-wait", 1, 6), ("live-db", 31, 35), ("push", 35, 36), ("status", 36, 36), ("pr", 36, 37),
              ("merge-pr", 37, 38), ("cleanup", 38, 39)]
    lines = ["[land] branch=p2 merged tip=abcdef12 main=12345678 01:00:00", "[land] verdict=slurm-gate"]
    for s, a, b_ in stamps:
        lines.append(f"[land] @ {s}:start {(t0 + timedelta(minutes=a)).strftime('%Y-%m-%dT%H:%M:%SZ')}")
        lines.append(f"[land] @ {s}:end {(t0 + timedelta(minutes=b_)).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    lines.append("https://github.com/example/repo/pull/500")
    lines.append("[land] LANDED p2 as PR 500 (tip abcdef12) 2026-09-29T01:39:00Z")
    (state / "land" / "p2.out").write_text("\n".join(lines) + "\n")
    rows = L.collect(str(state), [str(legacy_dir / "land-*.out")])
    by = {r["file"]: r for r in rows}
    o = by["land-orre.out"]
    assert o["shape"] == "legacy" and o["landed"] and o["gate_kind"] == "MODAL" and o["branch"] == "registry-probe-branch" and o["pr"] == "395"
    assert o["gate_min"] == pytest.approx(36.9, abs=0.1) and o["live_db_min"] == pytest.approx(9.0, abs=0.1) and o["pr_merge_min"] == pytest.approx(0.5, abs=0.1)
    assert o["total_min"] == pytest.approx(46.4, abs=0.1) and o["end_utc"] == "2026-09-24T17:29Z"
    w = by["land-wrap.out"]
    assert w["gate_kind"] == "REUSED" and w["gate_min"] is None and w["live_db_min"] == pytest.approx(14.0, abs=0.1), w   # the stamp wrapped past midnight
    r_ = by["land-refused.out"]
    assert not r_["landed"] and r_["exit"] == "6"
    p = by["p2.out"]
    assert p["shape"] == "phase2" and p["landed"] and p["gate_kind"] == "SLURM" and p["branch"] == "p2" and p["pr"] == "500"
    assert (p["gate_min"], p["baseline_wait_min"], p["live_db_min"], p["pr_merge_min"], p["total_min"]) == (30.0, 5.0, 4.0, 4.0, 39.0)
    table = L.render(rows)
    assert "| land-orre.out |" in table and "| p2.out |" in table and "landings: 3 of 4 files" in table and "not landings: land-refused.out (6)" in table
    assert "live-DB 9.0 min (n=3)" in table   # median of 9.0, 14.0, 4.0


def test_durations_renders_per_test_times_longest_first_with_file_subtotals(tmp_path):
    D = _measure("durations")
    junit = tmp_path / "j.xml"
    junit.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="5" time="12.75">'
        '<testcase classname="tests.test_registry_paper_spelling" name="test_a" time="3.25"/>'
        '<testcase classname="tests.test_registry_paper_spelling" name="test_b" time="0.5"><failure message="x">x</failure></testcase>'
        '<testcase classname="tests.test_registry_paper_views_contrast_current" name="test_live" time="9.0"/>'
        '<testcase classname="tests.test_registry_instrument_padding_identity.TestK" name="test_s" time="0.0"><skipped message="m"/></testcase>'
        '<testcase classname="tests.tools.test_nested.TestN" name="test_p" time="0.0"/>'
        '</testsuite></testsuites>')
    rep = D.parse(junit)
    assert rep["n"] == 5 and rep["total_s"] == 12.75 and rep["suite_time_s"] == 12.75
    assert rep["counts"] == {"passed": 3, "failed": 1, "skipped": 1}
    assert [c["node"] for c in rep["cases"]][:4] == ["tests/test_registry_paper_views_contrast_current.py::test_live", "tests/test_registry_paper_spelling.py::test_a",
                                                     "tests/test_registry_paper_spelling.py::test_b", "tests/test_registry_instrument_padding_identity.py::TestK::test_s"]
    assert "tests/tools/test_nested.py::TestN::test_p" in [c["node"] for c in rep["cases"]], "a module inside a sub-package keeps its path"
    assert rep["files"]["tests/test_registry_paper_spelling.py"] == {"time_s": 3.75, "n": 2, "outcomes": {"passed": 1, "failed": 1}}
    text = D.render(rep, top=2)
    assert text.splitlines()[0].startswith("# per-test durations from") and "5 tests, total 12.8 s" in text.splitlines()[0]
    assert "9.00 s  tests/test_registry_paper_views_contrast_current.py::test_live  [passed]" in text and "... 3 more" in text
    assert "# per file" in text and "3.75 s  tests/test_registry_paper_spelling.py (2 tests: failed 1, passed 1)" in text
    r = subprocess.run([sys.executable, str(MEASURE / "durations.py"), str(tmp_path / "missing.xml")], capture_output=True, text=True)
    assert r.returncode == 2


# ---- the targeted battery on Slurm: plan, array task, exact-K collection -----------------------------
def _battery_mod():
    spec = importlib.util.spec_from_file_location("battery_collect_under_test", LANDING / "battery_collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_battery_plan_allocates_disjoint_duration_bounded_lists_and_refuses_a_bad_request():
    B = _battery_mod()
    table = {f"tests/test_{c}.py": s for c, s in (("a", 300.0), ("b", 200.0), ("c", 100.0), ("d", 50.0), ("e", 10.0), ("f", 10.0))}
    files = sorted(table) + ["tests/test_new.py"]
    plan = B.make_plan(SHA_A, "probe", files, table)
    assert plan["schema"] == "battery_plan/1" and plan["label"] == "probe" and plan["requested_files"] == sorted(files)
    assert plan["predicted"]["unknown_files"] == ["tests/test_new.py"] and plan["predicted"]["unknown_fill_s"] == 75.0, "an unknown file takes the median of the known"
    total = sum(table.values()) + 75.0
    assert plan["predicted"]["total_s"] == total == 745.0 and plan["count"] == 3 == -(-int(total) // 360), "default K = ceil(total / 6 min)"
    lists = [t["files"] for t in plan["tasks"]]
    assert [t["index"] for t in plan["tasks"]] == [0, 1, 2] and all(lists) and sorted(sum(lists, [])) == sorted(files), "a partition: every file once"
    assert all(not (set(a) & set(b)) for i, a in enumerate(lists) for b in lists[i + 1:]) and plan["predicted"]["max_task_s"] <= plan["budget_s"] == 480
    assert [t["predicted_s"] for t in plan["tasks"]] == [300.0, 220.0, 225.0], "LPT: longest first into the least loaded task"
    assert plan["tasks"][0]["files"][0] == "tests/test_a.py" and plan["tasks"][0]["predicted_s"] == max(t["predicted_s"] for t in plan["tasks"])
    assert plan["timeout_s"] == max(600, -(-int(2 * plan["predicted"]["max_task_s"] + 300) // 1)) and plan["walltime_s"] == plan["timeout_s"] + 600
    assert plan["walltime"] == "%02d:%02d:%02d" % (plan["walltime_s"] // 3600, plan["walltime_s"] % 3600 // 60, plan["walltime_s"] % 60)
    assert plan["task_flags"] == ["--local-venv", "--hide-slurm"] and plan["tasks_forced"] is False and B.check_plan(plan) == []
    # the budget grows K: three 300-s files and a 6-min target would be two tasks, but no task may exceed 8 min of predicted tests
    big = {"tests/test_x.py": 300.0, "tests/test_y.py": 300.0, "tests/test_z.py": 300.0}
    grown = B.make_plan(SHA_A, "grow", sorted(big), big)
    assert grown["count"] == 3 and grown["predicted"]["max_task_s"] == 300.0 and grown["predicted"]["over_budget_tasks"] == []
    # one file longer than the budget runs alone and is named
    lone = B.make_plan(SHA_A, "lone", ["tests/test_long.py", "tests/test_short.py"], {"tests/test_long.py": 900.0, "tests/test_short.py": 1.0})
    assert lone["count"] == 2 and lone["predicted"]["over_budget_tasks"] == [0] and lone["timeout_s"] == 2100 and lone["walltime"] == "00:45:00"
    # --tasks forces K but never past the number of files; '' flags mean none; no table at all -> the default fill
    forced = B.make_plan(SHA_A, "forced", sorted(big), {}, tasks=9, task_flags=[])
    assert forced["count"] == 3 and forced["tasks_forced"] is True and forced["task_flags"] == [] and forced["predicted"]["unknown_fill_s"] == 60.0
    with pytest.raises(ValueError, match="empty"):
        B.make_plan(SHA_A, "none", [], table)
    with pytest.raises(ValueError, match="duplicate"):
        B.make_plan(SHA_A, "dup", ["tests/test_a.py", "tests/test_a.py"], table)
    doctored = json.loads(json.dumps(grown))
    doctored["tasks"][1]["files"].append("tests/test_x.py")
    assert any("claimed by tasks 0 and 1" in p for p in B.check_plan(doctored)), "two tasks claiming one file is a plan defect"
    doctored2 = json.loads(json.dumps(grown))
    doctored2["tasks"][2]["files"] = []
    assert any("empty list" in p for p in B.check_plan(doctored2)) and any("union" in p for p in B.check_plan(doctored2))


def test_battery_plan_floors_zero_durations_and_the_lpt_tie_break_fills_every_task_deterministically():
    """A duration table can carry 0.0 for a file whose tests ran in under a millisecond (the mock probes of these
    tests on a fast node: observed seconds rounded to nothing). The LPT fed only zeros never changed a task's load, its tie-break went
    to the lowest index every time, every file landed in task 0 and check_plan refused the K=3 re-plan ("task 2 has an empty list") --
    the landing-chain reservation test failed on Slurm at two shas. Now every prediction has a positive floor (PREDICTED_FLOOR_S, the
    median fill included) and ties go to the task with the fewest files, then the lowest index: K tasks are never left empty while
    files remain, the allocation is deterministic, and check_plan's refusal of an empty task stays (it is right)."""
    B = _battery_mod()
    files = [f"tests/test_probe_{c}.py" for c in "abcdefg"]
    zeros = {f: 0.0 for f in files}
    plan = B.make_plan(SHA_A, "zeros", files, zeros, tasks=3)
    lists = [t["files"] for t in plan["tasks"]]
    assert plan["count"] == 3 and all(lists) and sorted(map(len, lists)) == [2, 2, 3], lists
    assert sorted(sum(lists, [])) == sorted(files) and B.check_plan(plan) == [], "a partition with no empty task"
    assert plan["predicted"]["floor_s"] == B.PREDICTED_FLOOR_S == 1.0 and plan["predicted"]["floored_files"] == sorted(files)
    assert [t["predicted_s"] for t in plan["tasks"]] == [3.0, 2.0, 2.0] and plan["predicted"]["total_s"] == 7.0, "every file predicts the floor"
    assert plan == {**B.make_plan(SHA_A, "zeros", list(reversed(files)), zeros, tasks=3), "planned_utc": plan["planned_utc"]}, "deterministic, whatever the request order"
    assert lists == [["tests/test_probe_a.py", "tests/test_probe_d.py", "tests/test_probe_g.py"], ["tests/test_probe_b.py", "tests/test_probe_e.py"],
                     ["tests/test_probe_c.py", "tests/test_probe_f.py"]], "round-robin by file count when every load ties"
    # the tie-break itself: equal loads go to the task with the fewest files, then the lowest index; a real load difference still wins
    assert B.lpt(["x", "y", "z"], {"x": 0.0, "y": 0.0, "z": 0.0}, 3) == [(["x"], 0.0), (["y"], 0.0), (["z"], 0.0)]
    assert B.lpt(["x", "y", "z"], {"x": 5.0, "y": 1.0, "z": 1.0}, 2) == [(["x"], 5.0), (["y", "z"], 2.0)]
    # a file unknown to an all-zero table takes the floor, not 0.0; a mixed table floors only its sub-floor entries and reports them
    mixed = {"tests/test_probe_a.py": 0.0, "tests/test_probe_b.py": 0.4, "tests/test_probe_c.py": 30.0}
    plan2 = B.make_plan(SHA_A, "mixed", sorted(mixed) + ["tests/test_probe_new.py"], mixed, tasks=2)
    assert plan2["predicted"]["floored_files"] == ["tests/test_probe_a.py", "tests/test_probe_b.py"] and plan2["predicted"]["unknown_fill_s"] == 1.0
    assert plan2["predicted"]["total_s"] == 33.0 and all(t["files"] for t in plan2["tasks"]) and B.check_plan(plan2) == []
    plan3 = B.make_plan(SHA_A, "allzero-unknown", files + ["tests/test_probe_new.py"], zeros, tasks=4)
    assert plan3["predicted"]["unknown_fill_s"] == 1.0 and plan3["count"] == 4 and all(t["files"] for t in plan3["tasks"])
    # the refusal stands for a plan that arrives with an empty task anyway
    doctored = json.loads(json.dumps(plan))
    doctored["tasks"][2]["files"] = []
    assert any("task 2 has an empty list" in p for p in B.check_plan(doctored))


def test_battery_plan_stops_raising_k_once_every_over_budget_task_is_a_single_file():
    """C0 follow-up: `make_plan` raised K while the longest task exceeded the budget, and ONE file predicted at
    536 s (> 480 s) drove K from the 17 the 5,810-s total warranted to n = 414 one-file tasks (414 x ~210 s of fixed overhead). A task that
    holds one file cannot get shorter by adding tasks, so K stops growing once every over-budget task is a single file; a multi-file task
    over the budget still raises K; --tasks caps K exactly, over budget or not, and never past n."""
    B = _battery_mod()
    long_ = "tests/test_long.py"
    table = {long_: 536.0, **{f"tests/test_s{i:03d}.py": 20.0 for i in range(100)}}
    plan = B.make_plan(SHA_A, "c8-shape", sorted(table), table)
    # total 2536 s -> ceil(2536 / 360) = 8 tasks; the 536-s file alone in task 0 is over the budget and K stays 8 (it used to run to 101)
    assert plan["count"] == 8 and plan["tasks_forced"] is False and plan["predicted"]["over_budget_tasks"] == [0], plan["predicted"]
    assert plan["tasks"][0]["files"] == [long_] and plan["predicted"]["max_task_s"] == 536.0 and B.check_plan(plan) == []
    assert all(t["predicted_s"] <= plan["budget_s"] for t in plan["tasks"][1:]), "every other task fits the budget"
    assert plan["timeout_s"] == 2 * 536 + 300 and plan["walltime_s"] == plan["timeout_s"] + 600, "the bound is still sized from the long file"
    # a MULTI-file task over the budget still raises K (a high target so K starts at 1): 1510 s in one task -> K 2 (900 | 300+300+10, still
    # over with three files) -> K 3 (900 | 300+10 | 300): the only task over the budget is the single long file -> stop at 3, not n = 4
    files = ["tests/test_long.py", "tests/test_m1.py", "tests/test_m2.py", "tests/test_tiny.py"]
    t2 = {"tests/test_long.py": 900.0, "tests/test_m1.py": 300.0, "tests/test_m2.py": 300.0, "tests/test_tiny.py": 10.0}
    grown = B.make_plan(SHA_A, "multi", files, t2, target_s=2000)
    assert grown["count"] == 3 and grown["predicted"]["over_budget_tasks"] == [0] and grown["tasks"][0]["files"] == ["tests/test_long.py"], grown["tasks"]
    assert [t["files"] for t in grown["tasks"]] == [["tests/test_long.py"], ["tests/test_m1.py", "tests/test_tiny.py"], ["tests/test_m2.py"]]
    # --tasks caps K exactly even when every task is then over the budget, and never past the number of files
    forced = B.make_plan(SHA_A, "forced", sorted(table), table, tasks=3)
    assert forced["count"] == 3 and forced["tasks_forced"] is True and forced["predicted"]["max_task_s"] > forced["budget_s"]
    assert forced["predicted"]["over_budget_tasks"] == [0, 1, 2] and B.check_plan(forced) == []
    assert B.make_plan(SHA_A, "capped", files, t2, tasks=9)["count"] == 4

def test_battery_durations_sum_each_shard_report_per_file():
    B = _battery_mod()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        shards = Path(td) / "shards"
        for i, body in enumerate((
            '<testcase classname="tests.test_registry_a" name="t1" time="1.5"/><testcase classname="tests.test_registry_a.TestK" name="t2" time="2.0"/>',
            '<testcase classname="tests.test_registry_b" name="t1" time="4.0"/><testcase classname="tests.tools.test_nested" name="t" time="0.5"/>')):
            d = shards / f"{i}-of-2"
            d.mkdir(parents=True)
            (d / "junit.xml").write_text(f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="2">{body}</testsuite></testsuites>')
        r = subprocess.run([sys.executable, str(LANDING / "battery_collect.py"), "durations", "--shards-dir", str(shards), "--out", f"{td}/d.json"],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        d = json.loads(Path(td, "d.json").read_text())
        assert d["files"] == {"tests/test_registry_a.py": 3.5, "tests/test_registry_b.py": 4.0, "tests/tools/test_nested.py": 0.5} and d["median_s"] == 3.5
        table, sources = B.load_durations([f"{td}/d.json", f"{td}/absent.json"])
        assert table["tests/test_registry_b.py"] == 4.0 and [s["state"] for s in sources] == ["read", "missing"]
        r = subprocess.run([sys.executable, str(LANDING / "battery_collect.py"), "durations", "--shards-dir", f"{td}/nothing"], capture_output=True, text=True)
        assert r.returncode == 2 and "no junit.xml" in r.stderr


def _fake_record(plan: dict, i: int, *, failing: list | None = None, sha: str | None = None, status: str = None) -> dict:
    files = plan["tasks"][i]["files"]
    fails = [{"node": n, "kind": "failure", "message": "assert False"} for n in (failing or [])]
    return {"sha": sha or plan["sha"], "app": "battery-x", "status": status or ("FAIL" if fails else "PASS"), "pytest_rc": 1 if fails else 0,
            "elapsed_s": 12.5 + i, "node": f"node-{i}", "slurm_job_id": f"9000_{i}", "slurm_array_job_id": "9000", "slurm_array_task_id": str(i),
            "shard": {"index": i, "count": plan["count"], "n_files": len(files), "files": list(files), "rerun": True, "label": f"t{i}"},
            "counts": {"tests": 2 * len(files), "failures": len(fails), "errors": 0, "skipped": 0}, "failures": fails,
            "results_dir": f"/shared/gates/{plan['sha']}/reruns/{plan['label']}/t{i}", "tree": {"ok": True}, "dependency_hashes": {}, "versions": {}}


def _write_records(plan: dict, root: Path, recs: dict) -> Path:
    shutil.rmtree(root, ignore_errors=True)
    for i, rec in recs.items():
        d = root / f"t{i}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(rec))
        cases = "".join(f'<testcase classname="{f[:-3].replace("/", ".")}" name="test_{j}" time="{0.25 * (j + 1)}"/>'
                        for f in plan["tasks"][i]["files"] for j in range(2))
        (d / "junit.xml").write_text(f'<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="2">{cases}</testsuite></testsuites>')
        (d / "pytest.log").write_text(f"log of task {i}\n")
    return root


def test_battery_collect_passes_exactly_k_planned_records_and_refuses_every_other_shape(tmp_path):
    B = _battery_mod()
    table = {"tests/test_registry_a.py": 100.0, "tests/test_registry_b.py": 90.0, "tests/test_registry_c.py": 10.0}
    plan = B.make_plan(SHA_A, "lab", sorted(table), table, tasks=2)
    assert plan["count"] == 2
    good = {i: _fake_record(plan, i) for i in range(2)}
    rec = _write_records(plan, tmp_path / "rec", good)

    def run(records, pl=plan, array_job_id="9000", **kw):
        return B.collect(pl, records, tmp_path / "out" / "lab", array_job_id=array_job_id, **kw)

    v = run(rec)
    assert v["status"] == "PASS" and v["tasks"] == 2 and v["counts"] == {"tests": 6, "failures": 0, "errors": 0, "skipped": 0} and v["failures"] == []
    assert v["array_job_id"] == "9000" and [(t["slurm_array_job_id"], t["slurm_array_task_id"]) for t in v["task_records"]] == [("9000", "0"), ("9000", "1")]
    assert [t["results_dir"] for t in v["task_records"]] == [f"/shared/gates/{SHA_A}/reruns/lab/t0", f"/shared/gates/{SHA_A}/reruns/lab/t1"]
    assert v["observed_durations"] == {"tests/test_registry_a.py": 0.75, "tests/test_registry_b.py": 0.75, "tests/test_registry_c.py": 0.75}
    assert (tmp_path / "out" / "lab.verdict.json").is_file() and (tmp_path / "out" / "lab.junit.xml").is_file()
    assert (tmp_path / "out" / "lab.log").read_text().count("==== task t") == 2
    merged = ET.parse(str(tmp_path / "out" / "lab.junit.xml")).getroot()
    assert merged.tag == "testsuites" and len(list(merged.iter("testcase"))) == 6
    # one failing node in one task -> FAIL naming exactly it
    bad = dict(good)
    bad[1] = _fake_record(plan, 1, failing=["tests.test_registry_c::test_1"])
    v = run(_write_records(plan, tmp_path / "rec_fail", bad))
    assert v["status"] == "FAIL" and [f["node"] for f in v["failures"]] == ["tests.test_registry_c::test_1"] and v["counts"]["failures"] == 1

    def refused(records, kind, needle, pl=plan, **kw):
        v = run(records, pl=pl, **kw)
        assert v["status"] == "INCOMPLETE" and v["refusal_kind"] == kind and v["refusal"].startswith("battery incomplete: ") and needle in v["refusal"], v
        return v

    # a missing task record
    missing = {0: good[0]}
    refused(_write_records(plan, tmp_path / "rec_missing", missing), "missing", "missing task record(s): ['t1']")
    refused(_write_records(plan, tmp_path / "rec_missing2", missing), "timeout", "slurm states {'t1': 'TIMEOUT'}", sacct={"1": "TIMEOUT"})
    # a record whose sha is another tree's
    other = dict(good)
    other[1] = _fake_record(plan, 1, sha="f" * 40)
    refused(_write_records(plan, tmp_path / "rec_sha", other), "identity", f"record sha {'f' * 40} is not the battery's tree {SHA_A}")
    # two tasks claiming one file: in the plan …
    doctored = json.loads(json.dumps(plan))
    doctored["tasks"][1]["files"].append("tests/test_registry_a.py")
    refused(rec, "plan", "claimed by tasks 0 and 1", pl=doctored)
    # … or in a record whose list is not the planned one
    claimer = dict(good)
    claimer[1] = _fake_record(plan, 1)
    claimer[1]["shard"]["files"] = list(plan["tasks"][0]["files"])
    refused(_write_records(plan, tmp_path / "rec_claim", claimer), "identity", "task 1: record ran")
    # an extra record beside the planned tasks; a record under another label's path; a non-terminal status; no tests; a lost junit
    extra = _write_records(plan, tmp_path / "rec_extra", good)
    (extra / "t2").mkdir()
    refused(extra, "extra", "extra record(s) beside the 2 planned tasks: ['t2']")
    moved = dict(good)
    moved[0] = dict(good[0], results_dir=f"/shared/gates/{SHA_A}/reruns/other/t0")
    refused(_write_records(plan, tmp_path / "rec_moved", moved), "identity", "results_dir")
    timed = dict(good)
    timed[0] = _fake_record(plan, 0, status="TIMEOUT")
    refused(_write_records(plan, tmp_path / "rec_timeout", timed), "timeout", "task 0: status TIMEOUT is not a result")
    bad_tree = dict(good)
    bad_tree[0] = dict(_fake_record(plan, 0, status="BAD_TREE"), error="tree does not verify")
    refused(_write_records(plan, tmp_path / "rec_badtree", bad_tree), "refused", "BAD_TREE is not a result (tree does not verify)")
    empty = dict(good)
    empty[0] = dict(good[0], counts={"tests": 0, "failures": 0, "errors": 0, "skipped": 0})
    refused(_write_records(plan, tmp_path / "rec_empty", empty), "empty", "task 0: ran no tests")
    nojunit = _write_records(plan, tmp_path / "rec_nojunit", good)
    (nojunit / "t1" / "junit.xml").unlink()
    refused(nojunit, "missing", "task 1: no junit.xml")
    wrong_index = dict(good)
    wrong_index[1] = _fake_record(plan, 1)
    wrong_index[1]["shard"]["index"] = 0
    refused(_write_records(plan, tmp_path / "rec_index", wrong_index), "identity", "record claims shard 0/2, planned 1/2")
    # a record ANOTHER array wrote under this label (a double submission) is identity, never a verdict — and it is
    # refused before its status is read, so a foreign TIMEOUT record cannot re-plan this attempt either
    foreign = dict(good)
    foreign[1] = dict(_fake_record(plan, 1), slurm_array_job_id="9001", slurm_job_id="9001_1")
    refused(_write_records(plan, tmp_path / "rec_foreign", foreign), "identity",
            "task 1: record was written by array '9001' (slurm_array_job_id), not this attempt's array 9000 — another submission of label lab?")
    foreign_timeout = dict(good)
    foreign_timeout[0] = dict(_fake_record(plan, 0, status="TIMEOUT"), slurm_array_job_id="9001")
    refused(_write_records(plan, tmp_path / "rec_foreign_timeout", foreign_timeout), "identity", "task 0: record was written by array '9001'")
    unbound = dict(good)
    unbound[0] = {k: v for k, v in good[0].items() if k != "slurm_array_job_id"}
    refused(_write_records(plan, tmp_path / "rec_unbound", unbound), "identity", "task 0: record was written by array None (slurm_array_job_id)")
    wrong_task = dict(good)
    wrong_task[1] = dict(_fake_record(plan, 1), slurm_array_task_id="0")
    refused(_write_records(plan, tmp_path / "rec_task", wrong_task), "identity", "task 1: record's slurm_array_task_id '0' is not its task index 1")
    # the right records read with another attempt's array, or with none: the records cannot be bound -> identity
    refused(rec, "identity", "task 0: record was written by array '9000' (slurm_array_job_id), not this attempt's array 9002", array_job_id="9002")
    refused(rec, "identity", "no array job id was recorded for this attempt: its records cannot be bound to the submission that wrote them", array_job_id=None)
    refused(rec, "identity", "no array job id was recorded", array_job_id="  ")
    # a record without slurm_array_task_id (an older gate_task.py) still binds by its array; its index is pinned by shard.index above
    older = dict(good)
    older[0] = {k: v for k, v in good[0].items() if k != "slurm_array_task_id"}
    assert run(_write_records(plan, tmp_path / "rec_older", older))["status"] == "PASS"
    # the CLI: exit 0 PASS, 1 FAIL, 4 INCOMPLETE; the observed table is updated only on a verdict
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    argv = [sys.executable, str(LANDING / "battery_collect.py"), "collect", "--plan", str(tmp_path / "plan.json"), "--out-prefix", str(tmp_path / "cli" / "lab"),
            "--observed", str(tmp_path / "observed.json"), "--array-job-id", "9000"]
    r = subprocess.run(argv + ["--records", str(rec)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("battery PASS label=lab") and "FAILURE NODES (0)" in r.stdout, r.stdout + r.stderr
    assert json.loads((tmp_path / "observed.json").read_text())["files"]["tests/test_registry_a.py"] == 0.75
    r = subprocess.run(argv + ["--records", str(tmp_path / "rec_fail")], capture_output=True, text=True)
    assert r.returncode == 1 and r.stdout.startswith("battery FAIL label=lab") and "tests.test_registry_c::test_1" in r.stdout
    r = subprocess.run(argv + ["--records", str(tmp_path / "rec_missing")], capture_output=True, text=True)
    assert r.returncode == 4 and r.stdout.startswith("battery incomplete: missing task record(s): ['t1'] [kind=missing]"), r.stdout
    r = subprocess.run([a if a != "9000" else "9001" for a in argv] + ["--records", str(rec)], capture_output=True, text=True)
    assert r.returncode == 4 and r.stdout.startswith("battery incomplete: task 0: record was written by array '9000'") and "[kind=identity]" in r.stdout, r.stdout
    r = subprocess.run([a for a in argv if a not in ("--array-job-id", "9000")] + ["--records", str(rec)], capture_output=True, text=True)
    assert r.returncode == 4 and "no array job id was recorded" in r.stdout and "[kind=identity]" in r.stdout, r.stdout


# the shims of a battery world: the login node is this host (ssh runs the command locally), sbatch runs every array task of
# battery.sbatch synchronously with srun = exec, sacct answers from what sbatch recorded, rsync drops the host prefix, the watcher
# reports DONE at once. $MOCK/timeout_task_<i> makes sbatch fabricate a TIMEOUT record for task i of its next array instead of running it;
# $MOCK/sbatch_delay (seconds) makes every sbatch take that long (the races below need one driver to hold its lock while the other
# arrives; $MOCK/sacct_delay the same for sacct, to hold a collection that submits nothing); $MOCK/linger makes every ssh leave a
# detached 20-s child behind with its stdio closed, the way an ssh ControlMaster with ControlPersist outlives the command that spawned it;
# $MOCK/drop_after_submit (a count N) lets N submit steps through and then, on the next one, runs the cluster side to completion
# (array submitted, reservation finalized), prints its output and exits 255 — the transport dropped after the JOBS line.
SSH_LOCAL_SHIM = r'''#!/bin/bash
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; -*) shift ;; *) break ;; esac; done
host=$1; shift
echo "ssh $host $*" >> "$MOCK/ssh.calls"
[ ! -f "$MOCK/linger" ] || (setsid sleep 20 >/dev/null 2>&1 </dev/null &)
if [ -f "$MOCK/drop_after_submit" ] && [[ " $* " == *" battery-"*"-tester "* ]]; then
  n=$(cat "$MOCK/drop_after_submit"); n=${n:-0}
  if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$MOCK/drop_after_submit"
  else
    rm -f "$MOCK/drop_after_submit"; out=$(bash -c "$*"); printf '%s\n' "$out"
    echo "ssh: connection to $host closed by remote host" >&2; echo "ssh-drop $host" >> "$MOCK/ssh.calls"; exit 255
  fi
fi
exec bash -c "$*"
'''
SBATCH_LOCAL_SHIM = r'''#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path
mock = Path(os.environ["MOCK"]); args = sys.argv[1:]
(mock / "sbatch.calls").open("a").write("sbatch " + " ".join(args) + "\n")
if (mock / "sbatch_delay").exists():
    time.sleep(float((mock / "sbatch_delay").read_text().strip() or "3"))
seq = mock / "jobseq"; n = int(seq.read_text()) + 1 if seq.exists() else 7001; seq.write_text(str(n))
name = ""; arr = None; pos = []; i = 0
while i < len(args):
    a = args[i]
    if a == "-J": name = args[i + 1]; i += 2; continue
    if a.startswith("--array="): lo, hi = a[len("--array="):].split("-"); arr = range(int(lo), int(hi) + 1)
    elif not a.startswith("-"): pos.append(a)
    i += 1
(mock / "jobs").mkdir(exist_ok=True); (mock / "jobs" / str(n)).write_text(name)
print(n); sys.stdout.flush()
states = []
if arr is not None and pos and pos[0].endswith(".sbatch"):
    script, rest = pos[0], pos[1:]
    for t in arr:
        marker = mock / f"timeout_task_{t}"
        if marker.exists() and script.endswith("battery.sbatch"):
            marker.unlink()
            tree, venv, results, label, plan_p = rest
            plan = json.load(open(plan_p)); task = plan["tasks"][t]
            d = Path(results) / plan["sha"] / "reruns" / label / f"t{t}"; d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"sha": plan["sha"], "status": "TIMEOUT", "pytest_rc": -9, "counts": None, "failures": [],
                "slurm_job_id": f"{n}_{t}", "slurm_array_job_id": str(n), "slurm_array_task_id": str(t),
                "shard": {"index": t, "count": plan["count"], "files": task["files"], "rerun": True, "label": f"t{t}"}, "results_dir": str(d), "elapsed_s": 1.0}))
            states.append(f"{t}|TIMEOUT"); continue
        env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(t), SLURM_ARRAY_JOB_ID=str(n), SLURM_JOB_ID=f"{n}_{t}", SLURM_CPUS_PER_TASK="2")
        with (mock / f"task_{n}_{t}.out").open("w") as out:
            rc = subprocess.run(["bash", script, *rest], env=env, stdout=out, stderr=subprocess.STDOUT).returncode
        states.append(f"{t}|{'COMPLETED' if rc == 0 else 'FAILED'}")
(mock / "jobs" / f"{n}.tasks").write_text("\n".join(states) + ("\n" if states else ""))
'''
SACCT_LOCAL_SHIM = r'''#!/bin/bash
echo "sacct $*" >> "$MOCK/sacct.calls"
[ ! -f "$MOCK/sacct_delay" ] || sleep "$(cat "$MOCK/sacct_delay")"
j=""; while [ $# -gt 0 ]; do case "$1" in -j) j=$2; shift 2 ;; *) shift ;; esac; done
[ -f "$MOCK/jobs/$j.tasks" ] || exit 0
while IFS='|' read -r t s; do [ -n "$t" ] && echo "${j}_$t|$s"; done < "$MOCK/jobs/$j.tasks"
'''
WATCH_LOCAL_SHIM = '#!/bin/bash\necho "watch $*" >> "$MOCK/watch.calls"\necho "RUNNING $2 node=TESTNODE"\necho "DONE $2 state=COMPLETED"\nexit 0\n'
PASSING_TEST = "def test_fine():\n    assert 1 + 1 == 2\n\n\ndef test_fine_too():\n    assert True\n"
FAILING_TEST = "def test_fine_here():\n    assert True\n\n\ndef test_broken():\n    assert 1 + 1 == 3, 'the battery must name this node'\n"


def _battery_world(tmp_path: Path) -> types.SimpleNamespace:
    """A synthetic repository (two passing files, one failing), its 'cluster mirror', the shims above, a ready 'venv' whose python is
    this interpreter, and a baseline gate's shard report for the duration table."""
    bin_, mock_, state, shared, out = tmp_path / "bin", tmp_path / "mock", tmp_path / "state", tmp_path / "shared", tmp_path / "out"
    for d in (bin_, mock_, state, shared, out):
        d.mkdir(parents=True, exist_ok=True)
    # the real binary where the host has one, else the pure-Python fallback. The rsync shim below strips the synthetic host from every
    # operand before either sees it -- the same local substitution with both -- so the fallback's refusal of a remote operand never fires
    # here and no opt-in is needed
    real_rsync = RF.rsync_executable()
    _exe(bin_ / "ssh", SSH_LOCAL_SHIM)
    _exe(bin_ / "sbatch", SBATCH_LOCAL_SHIM)
    _exe(bin_ / "sacct", SACCT_LOCAL_SHIM)
    _exe(bin_ / "squeue", "#!/bin/bash\nexit 0\n")
    _exe(bin_ / "srun", '#!/bin/bash\nexec "$@"\n')
    _exe(bin_ / "rsync", '#!/bin/bash\nargs=(); for a in "$@"; do args+=("${a#TESTHOST:}"); done\necho "rsync ${args[*]}" >> "$MOCK/rsync.calls"\nexec ' + real_rsync + ' "${args[@]}"\n')
    _exe(mock_ / "watch-job.sh", WATCH_LOCAL_SHIM)
    _exe(mock_ / "mirror-repos.sh", MIRROR_SHIM)
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "scripts" / "registry" / "landing").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'probe'\nversion = '0'\n")
    (repo / "uv.lock").write_text("version = 1\n")
    (repo / "tests" / "test_probe_ok.py").write_text(PASSING_TEST)
    (repo / "tests" / "test_probe_ok_too.py").write_text(PASSING_TEST)
    (repo / "tests" / "test_probe_bad.py").write_text(FAILING_TEST)
    for name in ("gate_task.py", "gate_lib.py", "gate_venv.sbatch", "battery.sbatch", "battery_collect.py"):
        shutil.copyfile(LANDING / name, repo / "scripts" / "registry" / "landing" / name)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "probe tree")
    sha = _git(repo, "rev-parse", "HEAD")
    mirror = shared / "repos" / "repo"                   # the clone the mirror tool keys by the checkout's name
    _git(tmp_path, "clone", "-q", str(repo), str(mirror))
    _git(mirror, "config", "uploadpack.allowAnySHA1InWant", "true")
    # after the clone, so the mirror lacks both: a tag and an archived branch (refs/archive/<date>/<branch>, the grading host's archive pass),
    # which prepare_cluster's mirror_history.sh pushes for real over the ssh shim (git's own `ssh TESTHOST git-receive-pack ...`)
    _git(repo, "tag", "-a", "probe-v1", "-m", "probe")
    _git(repo, "update-ref", "refs/archive/2026-10-09/probe-archived", sha)
    import hashlib
    ph = hashlib.sha256((repo / "pyproject.toml").read_bytes()).hexdigest()
    lh = hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest()
    h12 = hashlib.sha256(f"{ph}\n{lh}\n".encode()).hexdigest()[:12]
    venv = shared / "envs" / f"repo-gate-{h12}"
    (venv / "bin").mkdir(parents=True)
    # a wrapper, not a symlink: a symlinked python computes its prefix from the link's directory (no pyvenv.cfg there) and
    # loses this interpreter's site-packages, so `python -m pytest` would find no pytest
    _exe(venv / "bin" / "python", f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    (venv / ".gate-ready.json").write_text(json.dumps({"dependency_hashes": {"pyproject.toml": ph, "uv.lock": lh}, "venv": str(venv)}))
    base = "b" * 40
    shard = shared / "gates" / base / "shards" / "0-of-1"
    shard.mkdir(parents=True)
    shard.joinpath("junit.xml").write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="2">'
                                           '<testcase classname="tests.test_probe_bad" name="test_broken" time="30.0"/>'
                                           '<testcase classname="tests.test_probe_ok" name="test_fine" time="2.0"/></testsuite></testsuites>')
    (state / "BASELINE").write_text(base + "\n")
    env = dict(os.environ, **GIT_IDENTITY_ENV)
    env.update({"PATH": f"{bin_}:{env['PATH']}", "MOCK": str(mock_), "REGISTRY_LANDING_STATE": str(state), "SISTEER_REPO": str(repo),
                "GATE_HOST": "TESTHOST", "GATE_RUSER": "tester", "GATE_SHARED": str(shared), "MIRROR_REPOS_SH": str(mock_ / "mirror-repos.sh"),
                "MIRROR_REPOS_CLUSTER": "sel_one", "WATCH_JOB_SH": str(mock_ / "watch-job.sh"), "BATTERY_TASK_FLAGS": "", "BATTERY_WORK_ROOT": str(tmp_path / "work"),
                "BATTERY_OUT": str(out), "BATTERY_PY": sys.executable, "BATTERY_REMOTE_PY": sys.executable})
    env.pop("SLURM_ARRAY_TASK_ID", None)
    return types.SimpleNamespace(env=env, mock=mock_, state=state, shared=shared, out=out, repo=repo, sha=sha, venv=venv, base=base)


def _battery(w, *args, timeout=240):
    return _run([LANDING / "battery_slurm.sh", *args], w.env, timeout=timeout)


def test_battery_two_tasks_with_differing_outcomes_give_one_fail_verdict_naming_exactly_the_failing_node(tmp_path):
    """The C0 certification: one passing and one failing file allocated to two array tasks (the sbatch shim runs battery.sbatch, which
    runs the tree's own gate_task.py, twice) -> both records present and distinct, the verdict FAIL names exactly the failing node."""
    w = _battery_world(tmp_path)
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_bad.py", "--label", "c0probe", "--tasks", "2")
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-2000:]
    out = r.stdout
    assert "[battery:cluster] TREE exported" in out and "[battery] JOBS sha=" in out and "VENV ready" in out, out
    # the history push ran before the export, for real, over the ssh shim: the mirror and the exported tree carry the archived branch and the tag
    mirror = w.shared / "repos" / "repo"
    assert f"[battery] HISTORY mirrored 2 refs on ssh://TESTHOST{mirror} (2 pushed, 0 current)" in out, out
    assert out.index("[battery] HISTORY mirrored") < out.index("[battery:cluster] TREE exported")
    assert _git(mirror, "rev-parse", "refs/archive/2026-10-09/probe-archived") == w.sha and _git(mirror, "rev-parse", "probe-v1^{commit}") == w.sha
    tree = w.shared / "gates" / "trees" / w.sha
    assert _git(tree, "rev-parse", "refs/archive/2026-10-09/probe-archived") == w.sha and _git(tree, "rev-parse", "refs/tags/probe-v1^{commit}") == w.sha
    assert "[battery] VERDICT status=FAIL label=c0probe final_label=c0probe" in out
    plan = json.loads((w.out / "c0probe.plan.json").read_text())
    assert plan["count"] == 2 and plan["sha"] == w.sha and plan["task_flags"] == [] and plan["requested_files"] == ["tests/test_probe_bad.py", "tests/test_probe_ok.py"]
    assert plan["tasks"][0]["files"] == ["tests/test_probe_bad.py"] and plan["tasks"][1]["files"] == ["tests/test_probe_ok.py"], "LPT: the 30-s file leads"
    assert plan["predicted"]["known_files"] == 2 and [s["state"] for s in plan["durations_sources"]] == ["read"], "the baseline's shard report fed the table"
    assert (w.state / "battery" / "durations" / f"{w.base}.json").is_file()
    v = json.loads((w.out / "c0probe.verdict.json").read_text())
    assert v["status"] == "FAIL" and v["tasks"] == 2 and v["sha"] == w.sha and v["array_job_id"] == "7001"
    assert v["counts"] == {"tests": 4, "failures": 1, "errors": 0, "skipped": 0}
    assert [f["node"] for f in v["failures"]] == ["tests.test_probe_bad::test_broken"] and "name this node" in v["failures"][0]["message"]
    recs = v["task_records"]
    assert [t["status"] for t in recs] == ["FAIL", "PASS"] and [t["results_dir"].rsplit("/", 1)[1] for t in recs] == ["t0", "t1"]
    assert recs[0]["results_dir"] != recs[1]["results_dir"] and all(t["results_dir"].endswith(f"/{w.sha}/reruns/c0probe/t{i}") for i, t in enumerate(recs))
    cluster_recs = w.shared / "gates" / w.sha / "reruns" / "c0probe"
    assert sorted(os.listdir(cluster_recs)) == ["t0", "t1"] and all((cluster_recs / t / n).is_file() for t in ("t0", "t1") for n in ("summary.json", "junit.xml", "pytest.log"))
    s0, s1 = (json.loads((cluster_recs / t / "summary.json").read_text()) for t in ("t0", "t1"))
    assert s0["shard"]["files"] == ["tests/test_probe_bad.py"] and s1["shard"]["files"] == ["tests/test_probe_ok.py"] and s0["shard"]["count"] == s1["shard"]["count"] == 2
    assert s0["app"] == f"battery-{w.sha[:7]}-c0probe" and s0["env_overrides"]["venv_local_copy"] is False
    assert s0["slurm_array_job_id"] == s1["slurm_array_job_id"] == "7001" and (s0["slurm_array_task_id"], s1["slurm_array_task_id"]) == ("0", "1")
    assert [(t["slurm_array_job_id"], t["slurm_array_task_id"]) for t in recs] == [("7001", "0"), ("7001", "1")], "the verdict carries each record's array and task"
    resv = w.shared / "gates" / w.sha / "reservations" / "c0probe"
    holder = dict(l.split("=", 1) for l in (resv / "holder").read_text().splitlines())
    assert holder["LABEL"] == "c0probe" and holder["SHA"] == w.sha and holder["RUN_NAME"] == f"battery-{w.sha[:7]}-c0probe-tester" and holder["TASKS"] == "2"
    assert re.fullmatch(rf"{re.escape(socket.gethostname())}:\d+", holder["HOLDER"]), holder
    assert dict(l.split("=", 1) for l in (resv / "array").read_text().splitlines())["ARRAY"] == "7001", "the reservation is finalized by its array"
    lock = (w.state / "battery" / "locks" / f"{w.sha}.c0probe.lock").read_text()
    assert f"HOLDER={holder['HOLDER']}\n" in lock and "MODE=submit\n" in lock and f"SHA={w.sha}\n" in lock
    assert (w.shared / "gates" / w.sha / "reruns" / "c0probe.plan.json").is_file(), "the plan is shipped beside the records, never inside the label directory"
    assert sorted(os.listdir(w.out / "c0probe.records")) == ["t0", "t1"]
    merged = ET.parse(str(w.out / "c0probe.junit.xml")).getroot()
    assert len(list(merged.iter("testcase"))) == 4 and (w.out / "c0probe.log").read_text().count("==== task t") == 2
    calls = _calls(w.env, "sbatch")
    assert len(calls) == 1 and "--array=0-1" in calls[0] and f"-J battery-{w.sha[:7]}-c0probe-tester" in calls[0] and f"--time={plan['walltime']}" in calls[0], calls
    assert calls[0].rstrip().endswith(f"{w.shared}/gates/battery/bin/battery.sbatch {w.shared}/gates/trees/{w.sha} {w.venv} {w.shared}/gates c0probe "
                                      f"{w.shared}/gates/{w.sha}/reruns/c0probe.plan.json"), calls[0]
    assert "--export" not in calls[0] and "--mem=0" not in calls[0]
    assert f" --output={w.shared}/exp/logs/%x_%A_%a.out {w.shared}/gates/battery/bin/battery.sbatch " in calls[0], "each task's log under the site's shared root"
    assert (w.shared / "exp" / "logs").is_dir(), "the log directory exists before the array is submitted"
    assert _calls(w.env, "watch") == [f"watch --slurm 7001 --host TESTHOST --poll 30 --pending-min 20 --stall-min 0 --max-hours 3"]
    assert not _calls(w.env, "mirror"), "the sha was on the mirror: mirror-repos.sh is not called"
    jobs = dict(l.split("=", 1) for l in (w.out / "c0probe.jobs").read_text().splitlines())
    assert jobs["ARRAY"] == "7001" and jobs["TASKS"] == "2" and jobs["LABEL"] == "c0probe" and jobs["VENV_JOB"] == "none"
    assert jobs["SUBMITTER"] == holder["HOLDER"] and jobs["RESERVATION"] == str(resv)
    assert re.search(rf"^\S+ sha={w.sha} branch=none battery=c0probe tasks=2 walltime=\d\d:\d\d:\d\d venv_job=none array=7001 run=battery-{w.sha[:7]}-c0probe-tester$",
                     (w.state / "jobs.log").read_text(), re.M)
    observed = json.loads((w.state / "battery" / "observed.json").read_text())["files"]
    assert set(observed) == {"tests/test_probe_bad.py", "tests/test_probe_ok.py"} and all(t >= 0 for t in observed.values())
    task_out = (w.mock / "task_7001_0.out").read_text()
    assert "[battery] sha=" in task_out and "task=0/2" in task_out and "n_files=1: tests/test_probe_bad.py" in task_out and "[gate] SUMMARY" in task_out, task_out[-1500:]
    assert "[battery] record names array=7001 task=0 (this job)" in task_out, task_out[-1500:]
    # the same label again is refused: here by the local submission record, and on the cluster side (another grading host state) by the records
    # that exist — before any submission, with the submitted plan untouched; --collect-only re-reads the records without a submission
    r2 = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "c0probe")
    assert r2.returncode == 2 and "already submitted from here" in r2.stderr and len(_calls(w.env, "sbatch")) == 1, r2.stdout + r2.stderr
    w.env["BATTERY_OUT"] = str(tmp_path / "out2")
    r2b = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "c0probe")
    w.env["BATTERY_OUT"] = str(w.out)
    # (rounds 4-5) the cluster's FINALIZED reservation answers the other state dir's submission — but that submission asks for a
    # DIFFERENT request (1 file against the finalized 2): refused by name, nothing written, nothing submitted; the finalized array never
    # certifies a request it did not run. --collect-only (no request of its own) then collects the ORIGINAL request there.
    out2 = tmp_path / "out2"
    assert r2b.returncode == 2 and "[battery:cluster] HELD label=c0probe array=7001" in r2b.stdout and "RECONCILED" not in r2b.stdout, r2b.stdout + r2b.stderr
    assert re.search(r"\[battery\] ERROR: label c0probe is finalized for a different request: 2 files vs 1 requested \(only in the finalized plan: \['tests/test_probe_bad.py'\]; only in this request: \[\]\); "
                     r"tasks forced to 2 vs not forced requested; use a new label, or `--collect-only --label c0probe` to collect the original request — nothing written, nothing submitted", r2b.stderr), r2b.stderr
    assert len(_calls(w.env, "sbatch")) == 1 and not (out2 / "c0probe.jobs").exists() and not (out2 / "c0probe.plan.json").exists() and not (out2 / "c0probe.verdict.json").exists()
    assert (out2 / "c0probe.plan.json.partial").is_file(), "this request's partial plan stays as the trace --collect-only reads"
    w.env["BATTERY_OUT"] = str(out2)
    r2b2 = _battery(w, w.sha, "--collect-only", "--label", "c0probe")
    w.env["BATTERY_OUT"] = str(w.out)
    assert r2b2.returncode == 1 and "[battery] RECONCILED label=c0probe array=7001 tasks=2" in r2b2.stdout and "VERDICT status=FAIL label=c0probe" in r2b2.stdout, r2b2.stdout + r2b2.stderr
    assert len(_calls(w.env, "sbatch")) == 1 and json.loads((out2 / "c0probe.plan.json").read_text()) == plan and not (out2 / "c0probe.plan.json.partial").exists()
    jobs2 = dict(l.split("=", 1) for l in (out2 / "c0probe.jobs").read_text().splitlines())
    assert jobs2["ARRAY"] == "7001" and jobs2["SUBMITTER"] == jobs["SUBMITTER"] and jobs2["MAX_REPLANS"] == "1" and "RECONCILED_UTC" in jobs2 and jobs2["RESERVATION"] == str(resv)
    assert json.loads((out2 / "c0probe.verdict.json").read_text())["status"] == "FAIL" and json.loads((w.out / "c0probe.plan.json").read_text())["count"] == 2
    # records of a label with NO reservation (a submission from before reservations existed): still refused, before any submission
    shutil.rmtree(resv)
    w.env["BATTERY_OUT"] = str(tmp_path / "out3")
    r2c = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "c0probe")
    w.env["BATTERY_OUT"] = str(w.out)
    assert r2c.returncode == 2 and "already has records" in r2c.stdout and len(_calls(w.env, "sbatch")) == 1, r2c.stdout + r2c.stderr
    assert not (tmp_path / "out3" / "c0probe.plan.json").exists() and not (w.shared / "gates" / w.sha / "reservations" / "c0probe").exists()
    r3 = _battery(w, w.sha, "--collect-only", "--label", "c0probe")
    assert r3.returncode == 1 and "VERDICT status=FAIL label=c0probe" in r3.stdout and len(_calls(w.env, "sbatch")) == 1, r3.stdout + r3.stderr


def test_battery_refuses_an_empty_duplicated_or_absent_file_list_before_touching_the_cluster(tmp_path):
    w = _battery_world(tmp_path)
    for args, needle in (((w.sha, "--label", "x"), "--files is empty"),
                         ((w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_ok.py"), "duplicate file(s)"),
                         ((w.sha, "--files", "tests/test_probe_ok.py", "tests/test_nowhere.py"), "not in the tree"),
                         ((w.sha, "--files", "/etc/passwd"), "tree-relative"),
                         ((w.sha, "--files", "tests/test_probe_ok.py", "--label", "bad label"), "label 'bad label' must match"),
                         (("no-such-ref", "--files", "tests/test_probe_ok.py"), "cannot resolve")):
        r = _battery(w, *args)
        assert r.returncode == 2 and needle in r.stderr, (args, r.stdout, r.stderr)
    assert not _calls(w.env, "ssh") and not _calls(w.env, "sbatch"), "every refusal above happens before the first ssh"


def test_battery_sbatch_task_runs_only_its_own_list_and_refuses_a_foreign_plan_or_index(tmp_path):
    w = _battery_world(tmp_path)
    B = _battery_mod()
    tree = w.shared / "gates" / "trees" / w.sha
    r = subprocess.run(["bash", "-s", "--", w.sha, str(w.shared / "repos" / "repo"), str(tree)], input=(LANDING / "tree_export.sh").read_text(),
                       capture_output=True, text=True, env=w.env)
    assert r.returncode == 0 and "TREE exported" in r.stdout, r.stdout + r.stderr
    plan = B.make_plan(w.sha, "direct", ["tests/test_probe_ok.py", "tests/test_probe_ok_too.py"], {}, tasks=2, task_flags=[])
    plan_p = tmp_path / "direct.plan.json"
    plan_p.write_text(json.dumps(plan))
    results = w.shared / "gates"

    def task(i, label="direct", **env):
        e = dict(w.env, SLURM_ARRAY_TASK_ID=str(i), SLURM_ARRAY_JOB_ID="7100", SLURM_JOB_ID=f"7100_{i}", **env)
        return subprocess.run(["bash", str(LANDING / "battery.sbatch"), str(tree), str(w.venv), str(results), label, str(plan_p)],
                              capture_output=True, text=True, env=e, timeout=180)

    r = task(1)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    rec = json.loads((results / w.sha / "reruns" / "direct" / "t1" / "summary.json").read_text())
    assert rec["shard"] == {"index": 1, "count": 2, "n_files": 1, "files": ["tests/test_probe_ok_too.py"], "rerun": True, "label": "t1"}
    assert rec["status"] == "PASS" and rec["counts"]["tests"] == 2 and rec["sha"] == w.sha
    assert rec["slurm_array_job_id"] == "7100" and rec["slurm_array_task_id"] == "1" and "[battery] record names array=7100 task=1 (this job)" in r.stdout
    assert not (results / w.sha / "reruns" / "direct" / "t0").exists(), "task 1 wrote nothing under task 0's path"
    r = task(2)
    assert r.returncode == 2 and "task 2 is outside the plan's 2 tasks" in r.stdout, r.stdout
    r = task(0, label="other")
    assert r.returncode == 2 and "plan label 'direct' is not 'other'" in r.stdout, r.stdout
    e = dict(w.env)
    r = subprocess.run(["bash", str(LANDING / "battery.sbatch"), str(tree), str(w.venv), str(results), "direct", str(plan_p)], capture_output=True, text=True, env=e)
    assert r.returncode == 2 and "SLURM_ARRAY_TASK_ID is unset" in r.stdout
    assert sorted(os.listdir(results / w.sha / "reruns" / "direct")) == ["t1"], "the refusals wrote no record"


def test_battery_replans_with_one_more_task_when_a_task_runs_out_of_its_walltime(tmp_path):
    """R-battery-plan: a TIMEOUT record is not a result; the driver re-plans the whole list with K+1 under <label>-r2 (at most
    --max-replans times), and the final verdict is copied to <label>.* naming the attempts."""
    w = _battery_world(tmp_path)
    (w.mock / "timeout_task_1").write_text("")
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_ok_too.py", "tests/test_probe_bad.py", "--label", "rp", "--tasks", "2")
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-2000:]
    assert "re-planning the whole list with K=3 under label rp-r2 (re-plan 1 of 1)" in r.stdout and "VERDICT status=FAIL label=rp final_label=rp-r2" in r.stdout
    first = json.loads((w.out / "rp-r2.verdict.json").read_text())
    assert first["status"] == "FAIL" and first["tasks"] == 3 and first["array_job_id"] == "7002" and [f["node"] for f in first["failures"]] == ["tests.test_probe_bad::test_broken"]
    final = json.loads((w.out / "rp.verdict.json").read_text())
    assert final["status"] == "FAIL" and final["final_label"] == "rp-r2" and final["replans"] == 1 and final["tasks"] == 3
    assert [a["status"] for a in final["attempts"]] == ["INCOMPLETE", "FAIL"] and final["attempts"][0]["refusal_kind"] == "timeout" and final["attempts"][0]["label"] == "rp"
    assert "task 1: status TIMEOUT is not a result" in final["attempts"][0]["refusal"]
    calls = _calls(w.env, "sbatch")
    assert len(calls) == 2 and "--array=0-1" in calls[0] and "--array=0-2" in calls[1] and "-J battery-" in calls[1] and "-rp-r2-tester" in calls[1]
    assert sorted(os.listdir(w.shared / "gates" / w.sha / "reruns")) == ["rp", "rp-r2", "rp-r2.plan.json", "rp.plan.json"]
    # with no re-plan allowed the timeout is the (incomplete) outcome
    w2 = _battery_world(tmp_path / "second")
    (w2.mock / "timeout_task_0").write_text("")
    r = _battery(w2, w2.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_bad.py", "--label", "once", "--max-replans", "0")
    assert r.returncode == 4 and "VERDICT status=INCOMPLETE label=once final_label=once" in r.stdout and len(_calls(w2.env, "sbatch")) == 1, r.stdout
    assert json.loads((w2.out / "once.verdict.json").read_text())["refusal_kind"] == "timeout"


def test_battery_recollection_after_a_replan_walks_the_chain_from_the_root_and_rewrites_no_attempt(tmp_path):
    """After a re-plan every attempt keeps its OWN plan, jobs and records — <label>.plan.json stays the first
    attempt's and the re-plan's .jobs names its root — and --collect-only on the root (or on the re-plan's label, redirected to the
    root) verifies each attempt against its own files and reproduces the chain's verdict, instead of judging the root's K records by
    the final K+1 plan and overwriting a PASS/FAIL with INCOMPLETE."""
    w = _battery_world(tmp_path)
    (w.mock / "timeout_task_1").write_text("")
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_ok_too.py", "tests/test_probe_bad.py", "--label", "rp", "--tasks", "2")
    assert r.returncode == 1 and "final_label=rp-r2" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    root_plan = json.loads((w.out / "rp.plan.json").read_text())
    assert root_plan["count"] == 2 and root_plan["label"] == "rp", "the root's plan is the first attempt's, never the re-plan's"
    assert json.loads((w.out / "rp-r2.plan.json").read_text())["count"] == 3
    root_jobs = dict(l.split("=", 1) for l in (w.out / "rp.jobs").read_text().splitlines())
    r2_jobs = dict(l.split("=", 1) for l in (w.out / "rp-r2.jobs").read_text().splitlines())
    assert root_jobs["ARRAY"] == "7001" and root_jobs["TASKS"] == "2" and root_jobs["MAX_REPLANS"] == "1" and "REPLAN_OF" not in root_jobs
    assert r2_jobs["ARRAY"] == "7002" and r2_jobs["TASKS"] == "3" and r2_jobs["REPLAN_OF"] == "rp" and r2_jobs["ATTEMPT"] == "2" and "MAX_REPLANS" not in r2_jobs
    assert sorted(os.listdir(w.out / "rp.records")) == ["t0", "t1"] and sorted(os.listdir(w.out / "rp-r2.records")) == ["t0", "t1", "t2"]
    chain = json.loads((w.out / "rp.verdict.json").read_text())
    assert chain["status"] == "FAIL" and chain["final_label"] == "rp-r2" and chain["final_plan"] == str(w.out / "rp-r2.plan.json") and chain["tasks"] == 3
    assert chain["final_records"] == str(w.out / "rp-r2.records") and chain["final_jobs"] == str(w.out / "rp-r2.jobs")
    assert [a["label"] for a in chain["attempts"]] == ["rp", "rp-r2"] and chain["attempts"][0]["plan"] == str(w.out / "rp.plan.json")
    assert chain["attempts"][0]["records"] == str(w.out / "rp.records") and chain["attempts"][1]["jobs"] == str(w.out / "rp-r2.jobs")
    frozen = {p.name: p.read_bytes() for p in w.out.iterdir() if p.is_file() and (p.suffix == ".jobs" or p.name.endswith(".plan.json"))}
    assert set(frozen) == {"rp.jobs", "rp-r2.jobs", "rp.plan.json", "rp-r2.plan.json"}
    n_ssh = len(_calls(w.env, "ssh"))
    # --collect-only on the root: both attempts re-read with their own plan, jobs and records; the chain's verdict reproduced; nothing
    # submitted, no plan or submission record rewritten, the cluster side (mirror, code, tree) not touched
    r = _battery(w, w.sha, "--collect-only", "--label", "rp")
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-2000:]
    assert "re-plan 1 of rp was already submitted as rp-r2 (array 7002): collecting it" in r.stdout and "VERDICT status=FAIL label=rp final_label=rp-r2" in r.stdout
    again = json.loads((w.out / "rp.verdict.json").read_text())
    assert again["status"] == "FAIL" and again["final_label"] == "rp-r2" and again["replans"] == 1 and again["tasks"] == 3
    assert [a["status"] for a in again["attempts"]] == ["INCOMPLETE", "FAIL"] and again["attempts"][0]["refusal_kind"] == "timeout"
    assert [f["node"] for f in again["failures"]] == ["tests.test_probe_bad::test_broken"]
    assert json.loads((w.out / "rp-r2.verdict.json").read_text())["status"] == "FAIL" and "final_label" not in json.loads((w.out / "rp-r2.verdict.json").read_text())
    assert {p.name: p.read_bytes() for p in w.out.iterdir() if p.is_file() and p.name in frozen} == frozen, "a plan or a submission record was rewritten"
    assert len(_calls(w.env, "sbatch")) == 2 and len(_calls(w.env, "watch")) == 2
    new_ssh = _calls(w.env, "ssh")[n_ssh:]
    assert len(new_ssh) == 2 and all("sacct " in c for c in new_ssh), new_ssh
    # the re-plan's own label is redirected to the root
    r = _battery(w, w.sha, "--collect-only", "--label", "rp-r2")
    assert r.returncode == 1 and "rp-r2 is a re-plan of rp: a chain is collected from its root" in r.stdout and "VERDICT status=FAIL label=rp final_label=rp-r2" in r.stdout
    assert json.loads((w.out / "rp.verdict.json").read_text())["final_label"] == "rp-r2" and len(_calls(w.env, "sbatch")) == 2
    # a fresh submission under the root's label is refused by the root's record; under a label whose re-plan exists, by that re-plan's record
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "rp")
    assert r.returncode == 2 and "already submitted from here" in r.stderr and len(_calls(w.env, "sbatch")) == 2
    (w.out / "rp.jobs").rename(w.out / "rp.jobs.aside")
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "rp")
    (w.out / "rp.jobs.aside").rename(w.out / "rp.jobs")
    assert r.returncode == 2 and "already has a re-plan recorded here" in r.stderr and len(_calls(w.env, "sbatch")) == 2, r.stderr


def test_battery_names_the_venv_builds_log_and_the_arrays_under_the_sites_shared_root(tmp_path):
    """A tree whose gate venv is not ready: the battery submits the tree's gate_venv.sbatch first and the array after it, each with --output
    under the shared root the site names (the job scripts name no log path of their own)."""
    w = _battery_world(tmp_path)
    (w.venv / ".gate-ready.json").unlink()
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "vb", "--tasks", "1", "--no-wait")
    assert r.returncode == 0 and "VENV-JOB submitted 7001" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]
    calls = _calls(w.env, "sbatch")
    assert len(calls) == 2, calls
    assert calls[0] == (f"sbatch --parsable -J gate-venv-{w.venv.name.rsplit('-', 1)[1]} --output={w.shared}/exp/logs/%x_%j.out "
                        f"{w.shared}/gates/trees/{w.sha}/scripts/registry/landing/gate_venv.sbatch {w.shared}/gates/trees/{w.sha} {w.venv}"), calls[0]
    assert "--dependency=afterok:7001" in calls[1] and f" --output={w.shared}/exp/logs/%x_%A_%a.out {w.shared}/gates/battery/bin/battery.sbatch " in calls[1], calls[1]
    assert (w.shared / "exp" / "logs").is_dir()


def test_battery_no_wait_then_collect_only_replans_a_timed_out_task_from_the_saved_request(tmp_path):
    """A battery submitted with --no-wait is finished by --collect-only, which restores the file list, the branch
    and the task flags from the saved plan and the re-plan budget from the saved submission record, so a task that ran out of its
    walltime gets the same K+1 re-plan a waiting run would have — submitted under <label>-r2 naming its root, watched, collected into
    the chain's verdict at the root; a budget of 0 saved at submission holds until --max-replans is given again."""
    w = _battery_world(tmp_path)
    (w.mock / "timeout_task_0").write_text("")
    files = ["tests/test_probe_ok.py", "tests/test_probe_ok_too.py", "tests/test_probe_bad.py"]
    r = _battery(w, w.sha, "--files", *files, "--label", "nw", "--tasks", "2", "--max-replans", "0", "--no-wait")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "--no-wait: finish with:" in r.stdout and f" {w.sha} --collect-only --label nw --out {w.out}" in r.stdout and "VERDICT" not in r.stdout
    assert not (w.out / "nw.verdict.json").exists() and not _calls(w.env, "watch") and len(_calls(w.env, "sbatch")) == 1
    jobs = dict(l.split("=", 1) for l in (w.out / "nw.jobs").read_text().splitlines())
    assert jobs["MAX_REPLANS"] == "0" and jobs["TASKS"] == "2" and jobs["ARRAY"] == "7001"
    # the saved budget of 0: the timeout is collected and stands, nothing is submitted
    r = _battery(w, w.sha, "--collect-only", "--label", "nw")
    assert r.returncode == 4, r.stdout[-3000:] + r.stderr[-2000:]
    assert "re-plan budget is spent (0 of 0 used)" in r.stdout and "VERDICT status=INCOMPLETE label=nw final_label=nw" in r.stdout
    v = json.loads((w.out / "nw.verdict.json").read_text())
    assert v["refusal_kind"] == "timeout" and "task 0: status TIMEOUT is not a result" in v["refusal"] and len(_calls(w.env, "sbatch")) == 1
    # the request restored from the plan (no --files here; the plan's task flags even though the environment now says otherwise), the
    # budget given again: the K+1 re-plan is submitted under nw-r2 naming its root, watched, collected -> the chain's verdict at the root
    saved_flags = w.env["BATTERY_TASK_FLAGS"]
    w.env["BATTERY_TASK_FLAGS"] = "--hide-slurm"
    r = _battery(w, w.sha, "--collect-only", "--label", "nw", "--max-replans", "1")
    w.env["BATTERY_TASK_FLAGS"] = saved_flags
    assert r.returncode == 1, r.stdout[-3000:] + r.stderr[-2000:]
    assert "re-planning the whole list with K=3 under label nw-r2 (re-plan 1 of 1)" in r.stdout and "VERDICT status=FAIL label=nw final_label=nw-r2" in r.stdout
    assert "[battery:cluster] TREE existing" in r.stdout, "the cluster side is prepared before the re-plan is submitted"
    calls = _calls(w.env, "sbatch")
    assert len(calls) == 2 and "--array=0-2" in calls[1] and f"-J battery-{w.sha[:7]}-nw-r2-tester" in calls[1], calls
    assert calls[1].rstrip().endswith(f"nw-r2 {w.shared}/gates/{w.sha}/reruns/nw-r2.plan.json"), calls[1]
    assert _calls(w.env, "watch") == ["watch --slurm 7002 --host TESTHOST --poll 30 --pending-min 20 --stall-min 0 --max-hours 3"]
    replan = json.loads((w.out / "nw-r2.plan.json").read_text())
    assert replan["count"] == 3 and replan["requested_files"] == sorted(files) and replan["task_flags"] == [] and replan["label"] == "nw-r2" and replan["sha"] == w.sha
    assert json.loads((w.out / "nw.plan.json").read_text())["count"] == 2, "the root's plan is untouched"
    r2_jobs = dict(l.split("=", 1) for l in (w.out / "nw-r2.jobs").read_text().splitlines())
    assert r2_jobs["REPLAN_OF"] == "nw" and r2_jobs["ATTEMPT"] == "2" and r2_jobs["ARRAY"] == "7002" and r2_jobs["TASKS"] == "3"
    chain = json.loads((w.out / "nw.verdict.json").read_text())
    assert chain["status"] == "FAIL" and chain["final_label"] == "nw-r2" and chain["replans"] == 1 and chain["tasks"] == 3 and chain["array_job_id"] == "7002"
    assert [a["status"] for a in chain["attempts"]] == ["INCOMPLETE", "FAIL"] and [f["node"] for f in chain["failures"]] == ["tests.test_probe_bad::test_broken"]
    assert chain["counts"] == {"tests": 6, "failures": 1, "errors": 0, "skipped": 0}
    assert (w.out / "nw.junit.xml").is_file() and (w.out / "nw.log").read_text().count("==== task t") == 3
    assert sorted(os.listdir(w.shared / "gates" / w.sha / "reruns")) == ["nw", "nw-r2", "nw-r2.plan.json", "nw.plan.json"]
    assert sorted(os.listdir(w.out / "nw.records")) == ["t0", "t1"] and sorted(os.listdir(w.out / "nw-r2.records")) == ["t0", "t1", "t2"]
    # collected once more: the same chain, nothing submitted
    r = _battery(w, w.sha, "--collect-only", "--label", "nw")
    assert r.returncode == 1 and "VERDICT status=FAIL label=nw final_label=nw-r2" in r.stdout and len(_calls(w.env, "sbatch")) == 2
    # what --collect-only refuses, before any ssh: a file list or a task count of its own, no label, an unknown label, another sha
    n_ssh = len(_calls(w.env, "ssh"))
    for args, needle in (((w.sha, "--collect-only", "--label", "nw", "--files", "tests/test_probe_ok.py"), "takes no --files"),
                         ((w.sha, "--collect-only", "--label", "nw", "--tasks", "2"), "takes no --tasks"),
                         ((w.sha, "--collect-only"), "needs --label"),
                         ((w.sha, "--collect-only", "--label", "never"), "was not submitted from here")):
        r = _battery(w, *args)
        assert r.returncode == 2 and needle in r.stderr, (args, r.stdout, r.stderr)
    _git(w.repo, "commit", "-q", "--allow-empty", "-m", "another commit")
    other = _git(w.repo, "rev-parse", "HEAD")
    r = _battery(w, other, "--collect-only", "--label", "nw")
    assert r.returncode == 2 and f"label nw was submitted for sha {w.sha}" in r.stderr, r.stderr
    assert len(_calls(w.env, "ssh")) == n_ssh and len(_calls(w.env, "sbatch")) == 2


def _collector(w, *args, env=None):
    """A driver started in the background, for the races (both outputs captured)."""
    return subprocess.Popen([str(a) for a in (LANDING / "battery_slurm.sh", *args)], env=env or w.env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _race(w, first_env, second_env, *args, offset_s=1.5):
    """Two drivers on the same arguments, the second started offset_s later; returns {returncode: (stdout, stderr)} — the
    returncodes must differ (one submits, one refuses)."""
    p1 = _collector(w, *args, env=first_env)
    time.sleep(offset_s)
    p2 = _collector(w, *args, env=second_env)
    o1, e1 = p1.communicate(timeout=240)
    o2, e2 = p2.communicate(timeout=240)
    assert p1.returncode != p2.returncode, (p1.returncode, o1[-1500:], e1[-1500:], p2.returncode, o2[-1500:], e2[-1500:])
    return {p1.returncode: (o1, e1), p2.returncode: (o2, e2)}


def test_battery_two_collectors_on_one_timed_out_chain_submit_its_replan_once_and_the_other_refuses_by_name(tmp_path):
    """Two drivers collecting the same asynchronous, timed-out chain could both find <label>-r2 unsubmitted and
    both submit it (two arrays on the same task paths). Two --collect-only runs race on one state dir here: the chain lock lets
    exactly one through — it re-plans and submits <label>-r2 ONCE — and the other refuses at once, naming the holder (host:pid,
    since, mode). The lock is flock(1)'s own and its fd is closed before the script runs, so a child that outlives the run (an ssh
    ControlMaster — the shim leaves a detached 20-s child behind every ssh) cannot keep it: a later run on the chain proceeds."""
    w = _battery_world(tmp_path)
    (w.mock / "timeout_task_0").write_text("")
    files = ["tests/test_probe_ok.py", "tests/test_probe_ok_too.py", "tests/test_probe_bad.py"]
    r = _battery(w, w.sha, "--files", *files, "--label", "race", "--tasks", "2", "--max-replans", "1", "--no-wait")
    assert r.returncode == 0 and len(_calls(w.env, "sbatch")) == 1, r.stdout[-2000:] + r.stderr[-2000:]
    (w.mock / "sbatch_delay").write_text("6")     # the winner's re-plan sbatch takes 6 s: the other driver arrives while the lock is held
    (w.mock / "linger").write_text("")
    res = _race(w, w.env, w.env, w.sha, "--collect-only", "--label", "race")
    (w.mock / "sbatch_delay").unlink()
    (w.mock / "linger").unlink()
    assert set(res) == {1, 2}, {k: (v[0][-800:], v[1][-800:]) for k, v in res.items()}
    win_out, win_err = res[1]
    lose_out, lose_err = res[2]
    assert "re-planning the whole list with K=3 under label race-r2 (re-plan 1 of 1)" in win_out and "VERDICT status=FAIL label=race final_label=race-r2" in win_out, win_out[-2000:]
    m = re.search(rf"\[battery\] ERROR: another collector holds race for {w.sha[:7]}: HOLDER=(\S+):(\d+) SINCE_UTC=\S+ MODE=collect LABEL=race SHA={w.sha} — one driver per \(sha, label\) at a time", lose_err)
    assert m, lose_err
    assert m.group(1) == socket.gethostname() and "VERDICT" not in lose_out and "re-planning" not in lose_out and "[battery] ref=" not in lose_out, lose_out
    assert f"holder={m.group(1)}:{m.group(2)}" in win_out, "the loser names the holder the winner wrote"
    calls = _calls(w.env, "sbatch")
    assert len(calls) == 2 and "--array=0-2" in calls[1] and f"-J battery-{w.sha[:7]}-race-r2-tester" in calls[1], calls
    assert sorted(os.listdir(w.shared / "gates" / w.sha / "reruns")) == ["race", "race-r2", "race-r2.plan.json", "race.plan.json"]
    resv = w.shared / "gates" / w.sha / "reservations"
    assert sorted(os.listdir(resv)) == ["race", "race-r2"] and all((resv / a / f).is_file() for a in ("race", "race-r2") for f in ("holder", "array"))
    assert (resv / "race-r2" / "array").read_text().startswith("ARRAY=7002\n") and f"HOLDER={m.group(1)}:{m.group(2)}\n" in (resv / "race-r2" / "holder").read_text()
    r2_jobs = dict(l.split("=", 1) for l in (w.out / "race-r2.jobs").read_text().splitlines())
    assert r2_jobs["ARRAY"] == "7002" and r2_jobs["SUBMITTER"] == f"{m.group(1)}:{m.group(2)}" and r2_jobs["RESERVATION"] == str(resv / "race-r2")
    chain = json.loads((w.out / "race.verdict.json").read_text())
    assert chain["status"] == "FAIL" and chain["final_label"] == "race-r2" and chain["array_job_id"] == "7002"
    assert [(t["slurm_array_job_id"], t["slurm_array_task_id"]) for t in chain["task_records"]] == [("7002", "0"), ("7002", "1"), ("7002", "2")]
    # the lock is released with the run — the detached children the shim left behind (still alive for ~20 s) hold no fd on it
    lock = w.state / "battery" / "locks" / f"{w.sha}.race.lock"
    assert f"HOLDER={m.group(1)}:{m.group(2)}\n" in lock.read_text() and "MODE=collect\n" in lock.read_text()
    r = _battery(w, w.sha, "--collect-only", "--label", "race")
    assert r.returncode == 1 and "VERDICT status=FAIL label=race final_label=race-r2" in r.stdout and len(_calls(w.env, "sbatch")) == 2, r.stderr[-1500:]
    # a fresh submission under the same label while a collector holds the chain (its two sacct reads take 4 s each): the same
    # refusal, before any ssh
    (w.mock / "sacct_delay").write_text("4")
    n_ssh = len(_calls(w.env, "ssh"))
    p1 = _collector(w, w.sha, "--collect-only", "--label", "race")
    time.sleep(1.5)
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "race")
    o1, e1 = p1.communicate(timeout=240)
    (w.mock / "sacct_delay").unlink()
    assert p1.returncode == 1 and r.returncode == 2 and "another collector holds race for" in r.stderr and "MODE=collect" in r.stderr, (o1[-800:], e1[-800:], r.stderr)
    assert len(_calls(w.env, "sbatch")) == 2
    assert not [c for c in _calls(w.env, "ssh")[n_ssh:] if "mkdir" in c and "reservations" in c], "a refused driver never reached the cluster's reservation step"


def test_battery_cluster_reservation_makes_an_attempt_single_owner_across_state_dirs_and_is_never_stolen(tmp_path):
    """The cluster side of the same finding: two drivers that share no state dir (no lock between them) race on one timed-out chain.
    The atomic reservation of <label>-r2 on the cluster lets exactly one submit; the other refuses naming the holder. A reservation
    left behind is reported, never stolen: fresh (wait for its holder), STALE after BATTERY_RESERVATION_STALE_S with no array (the
    operator confirms the queue and removes it), or finalized with an array this state dir does not know (record it, never resubmit)."""
    w = _battery_world(tmp_path)
    files = ["tests/test_probe_ok.py", "tests/test_probe_ok_too.py", "tests/test_probe_bad.py"]
    (w.mock / "timeout_task_0").write_text("")
    r = _battery(w, w.sha, "--files", *files, "--label", "xs", "--tasks", "2", "--max-replans", "1", "--no-wait")
    assert r.returncode == 0 and len(_calls(w.env, "sbatch")) == 1, r.stdout[-2000:] + r.stderr[-2000:]
    # a second state dir and out dir that know the root attempt (its submission record and plan copied over): no local lock in common
    state2, out2 = tmp_path / "state2", tmp_path / "out2"
    state2.mkdir()
    out2.mkdir()
    for n in ("xs.jobs", "xs.plan.json"):
        shutil.copyfile(w.out / n, out2 / n)
    env2 = dict(w.env, REGISTRY_LANDING_STATE=str(state2), BATTERY_OUT=str(out2))
    (w.mock / "sbatch_delay").write_text("10")      # the loser must meet the reservation while it is still unfinalized
    res = _race(w, w.env, env2, w.sha, "--collect-only", "--label", "xs")
    (w.mock / "sbatch_delay").unlink()
    assert set(res) == {1, 2}, {k: (v[0][-800:], v[1][-800:]) for k, v in res.items()}
    win_out, lose_out, lose_err = res[1][0], res[2][0], res[2][1]
    assert "VERDICT status=FAIL label=xs final_label=xs-r2" in win_out
    assert "[battery] ref=" in lose_out and "re-planning the whole list with K=3 under label xs-r2" in lose_out, "the loser got as far as the cluster"
    m = re.search(r"\[battery:cluster\] HELD label=xs-r2 array=(\d*) age_s=(\d+) holder=(\S+):(\d+) since=(\S+) path=(\S+)", lose_out)
    assert m, lose_out[-2500:]
    assert m.group(3) == socket.gethostname() and m.group(6) == str(w.shared / "gates" / w.sha / "reservations" / "xs-r2")
    assert re.search(r"\[battery\] ERROR: another collector holds xs-r2: reservation on TESTHOST by \S+:\d+ since \S+ \(\d+s ago, no array recorded yet; holder process alive on this host\) — one submitter per attempt", lose_err), lose_err
    assert m.group(1) == "", "the loser met the reservation before it was finalized"
    assert "VERDICT" not in lose_out and len(_calls(w.env, "sbatch")) == 2 and "--array=0-2" in _calls(w.env, "sbatch")[1]
    assert not (out2 / "xs-r2.jobs").exists() and not (out2 / "xs-r2.plan.json").exists() and not (out2 / "xs-r2.plan.json.partial").exists(), "a refused submission leaves no attempt behind"
    assert (w.out / "xs-r2.jobs").is_file() and json.loads((w.out / "xs.verdict.json").read_text())["final_label"] == "xs-r2"
    holder = dict(l.split("=", 1) for l in (w.shared / "gates" / w.sha / "reservations" / "xs-r2" / "holder").read_text().splitlines())
    assert holder["HOLDER"] == f"{m.group(3)}:{m.group(4)}" and holder["LABEL"] == "xs-r2" and holder["RUN_NAME"] == f"battery-{w.sha[:7]}-xs-r2-tester"
    # the plan the tasks read is the WINNER's (the second state dir has no duration table, so its LPT allocation differs): a refused
    # submitter's plan never reaches <label>.plan.json — it travels under a submitter-unique name and is removed with the refusal
    reruns = w.shared / "gates" / w.sha / "reruns"
    assert sorted(os.listdir(reruns)) == ["xs", "xs-r2", "xs-r2.plan.json", "xs.plan.json"], "no partial plan left behind"
    remote_plan = json.loads((reruns / "xs-r2.plan.json").read_text())
    assert remote_plan == json.loads((w.out / "xs-r2.plan.json").read_text()) and remote_plan["durations_sources"][0]["state"] == "read"
    import hashlib
    assert holder["PLAN_SHA256"] == hashlib.sha256((reruns / "xs-r2.plan.json").read_bytes()).hexdigest()
    # a reservation left behind on a fresh chain: reported by name in every state, never stolen
    (w.mock / "timeout_task_0").write_text("")
    r = _battery(w, w.sha, "--files", *files, "--label", "st", "--tasks", "2", "--max-replans", "1", "--no-wait")
    assert r.returncode == 0 and len(_calls(w.env, "sbatch")) == 3
    resv = w.shared / "gates" / w.sha / "reservations" / "st-r2"
    resv.mkdir(parents=True)
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    resv.joinpath("holder").write_text(f"LABEL=st-r2\nSHA={w.sha}\nHOLDER=otherhost:424242\nSINCE_UTC={old}\nRUN_NAME=battery-{w.sha[:7]}-st-r2-tester\n")
    r = _battery(w, w.sha, "--collect-only", "--label", "st")
    assert r.returncode == 2, r.stdout[-1500:] + r.stderr
    assert re.search(rf"another collector holds st-r2: a STALE reservation on TESTHOST by otherhost:424242 since {old} \(\d{{4,}}s with no array recorded; holder on another host\) — not stolen: "
                     rf"confirm nothing is queued \(ssh TESTHOST squeue -u tester -n battery-{w.sha[:7]}-st-r2-tester\), remove {re.escape(str(resv))} by hand, then run again", r.stderr), r.stderr
    assert resv.is_dir() and not (w.out / "st-r2.jobs").exists() and not (w.out / "st-r2.plan.json").exists() and not (w.out / "st-r2.plan.json.partial").exists()
    assert len(_calls(w.env, "sbatch")) == 3, "a stale reservation is never stolen: nothing submitted"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    resv.joinpath("holder").write_text(f"LABEL=st-r2\nSHA={w.sha}\nHOLDER=otherhost:424242\nSINCE_UTC={now}\nRUN_NAME=x\n")
    r = _battery(w, w.sha, "--collect-only", "--label", "st")
    assert r.returncode == 2 and re.search(r"another collector holds st-r2: reservation on TESTHOST by otherhost:424242 since \S+ \(\d+s ago, no array recorded yet; holder on another host\) — one submitter per attempt: wait for it to finish", r.stderr), r.stderr
    saved = w.env.get("BATTERY_RESERVATION_STALE_S")
    w.env["BATTERY_RESERVATION_STALE_S"] = "1"
    time.sleep(1.2)
    r = _battery(w, w.sha, "--collect-only", "--label", "st")
    w.env.pop("BATTERY_RESERVATION_STALE_S", None) if saved is None else w.env.__setitem__("BATTERY_RESERVATION_STALE_S", saved)
    assert r.returncode == 2 and "a STALE reservation on TESTHOST by otherhost:424242" in r.stderr, "the stale bound is the knob's"
    resv.joinpath("array").write_text("ARRAY=4242\nVENV_JOB=none\nSUBMITTED_UTC=x\n")
    r = _battery(w, w.sha, "--collect-only", "--label", "st")
    assert r.returncode == 2 and re.search(rf"another collector finalized st-r2 on TESTHOST \(reservation {re.escape(str(resv))}\) but it cannot be reconciled here: "
                                           rf"no plan at TESTHOST:{re.escape(str(w.shared))}/gates/{w.sha}/reruns/st-r2\.plan\.json — nothing submitted, nothing written", r.stderr), r.stderr
    assert len(_calls(w.env, "sbatch")) == 3 and resv.is_dir() and not (w.out / "st-r2.jobs").exists() and not (w.out / "st-r2.plan.json").exists()
    # a reservation without a holder file (a submitter that died between mkdir and its write) is still a refusal
    shutil.rmtree(resv)
    resv.mkdir()
    r = _battery(w, w.sha, "--collect-only", "--label", "st")
    assert r.returncode == 2 and "another collector holds st-r2: reservation on TESTHOST by ? since unknown (age unknown, no array recorded yet; holder unknown)" in r.stderr, r.stderr
    # the operator removed it: the re-plan is submitted, once
    shutil.rmtree(resv)
    r = _battery(w, w.sha, "--collect-only", "--label", "st")
    assert r.returncode == 1 and "VERDICT status=FAIL label=st final_label=st-r2" in r.stdout and len(_calls(w.env, "sbatch")) == 4, r.stdout[-1500:] + r.stderr
    assert (resv / "array").read_text().startswith("ARRAY=7004\n")


def test_battery_collection_refuses_a_record_another_array_wrote_under_the_label(tmp_path):
    """Had two arrays been submitted under one label, their tasks would write the same paths; the collector binds every record to
    the array the submission recorded, so a record from any other array is INCOMPLETE (identity) — never a PASS, never a re-plan."""
    w = _battery_world(tmp_path)
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_ok_too.py", "--label", "mix", "--tasks", "2", "--no-wait")
    assert r.returncode == 0 and len(_calls(w.env, "sbatch")) == 1, r.stdout[-2000:] + r.stderr[-2000:]
    rec1 = w.shared / "gates" / w.sha / "reruns" / "mix" / "t1" / "summary.json"
    d = json.loads(rec1.read_text())
    assert d["slurm_array_job_id"] == "7001" and d["slurm_array_task_id"] == "1" and d["status"] == "PASS"
    rec1.write_text(json.dumps({**d, "slurm_array_job_id": "7777", "slurm_job_id": "7777_1"}))       # task 1 of a second array wrote the same path
    r = _battery(w, w.sha, "--collect-only", "--label", "mix")
    assert r.returncode == 4 and "VERDICT status=INCOMPLETE label=mix final_label=mix" in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    v = json.loads((w.out / "mix.verdict.json").read_text())
    assert v["status"] == "INCOMPLETE" and v["refusal_kind"] == "identity" and v["array_job_id"] == "7001"
    assert "task 1: record was written by array '7777' (slurm_array_job_id), not this attempt's array 7001 — another submission of label mix?" in v["refusal"]
    assert len(_calls(w.env, "sbatch")) == 1, "an identity refusal is not a timeout: nothing is re-planned"
    rec1.write_text(json.dumps({**d, "slurm_array_task_id": "0"}))                                   # this array's record, another task's id
    r = _battery(w, w.sha, "--collect-only", "--label", "mix")
    assert r.returncode == 4 and "task 1: record's slurm_array_task_id '0' is not its task index 1" in json.loads((w.out / "mix.verdict.json").read_text())["refusal"]
    rec1.write_text(json.dumps(d))
    r = _battery(w, w.sha, "--collect-only", "--label", "mix")
    assert r.returncode == 0 and "VERDICT status=PASS label=mix final_label=mix" in r.stdout and len(_calls(w.env, "sbatch")) == 1, r.stdout[-1500:] + r.stderr[-1000:]


def test_battery_reconciles_a_finalized_reservation_the_grading_host_lost_and_submits_nothing(tmp_path):
    """The transport drops (or the driver dies) AFTER sbatch returned — the reservation is finalized, the plan is
    in place, the array runs — but BEFORE <label>.plan.json and <label>.jobs are written here, so this state dir holds only the
    .partial plan. A restarted --collect-only, and a restarted plain run of the same label, submit NOTHING: the finalized reservation
    is read back (plan verified against its PLAN_SHA256, label and sha), the local plan and jobs restored, the finished array's
    records collected to a verdict. The same for a re-plan lost mid-chain. A reservation whose plan hashes differently, or that
    names another sha, is refused by name with nothing written or submitted."""
    w = _battery_world(tmp_path)
    two = ["tests/test_probe_ok.py", "tests/test_probe_bad.py"]
    three = two + ["tests/test_probe_ok_too.py"]
    reruns = w.shared / "gates" / w.sha / "reruns"
    # 1. the drop: the cluster side completed (the shim's sbatch ran both tasks; array 7001; reservation finalized), the grading host lost it
    (w.mock / "drop_after_submit").write_text("0")
    r = _battery(w, w.sha, "--files", *two, "--label", "lost", "--tasks", "2")
    assert r.returncode == 2 and "cluster step failed" in r.stderr and "[battery:cluster] JOBS venv=none array=7001" in r.stdout, r.stdout[-1500:] + r.stderr
    assert (w.out / "lost.plan.json.partial").is_file() and not (w.out / "lost.jobs").exists() and not (w.out / "lost.plan.json").exists()
    resv = w.shared / "gates" / w.sha / "reservations" / "lost"
    assert (resv / "array").read_text().startswith("ARRAY=7001\n") and sorted(os.listdir(reruns / "lost")) == ["t0", "t1"]
    assert not _calls(w.env, "watch") and len(_calls(w.env, "sbatch")) == 1 and "ssh-drop TESTHOST" in _calls(w.env, "ssh")[-1]
    # 2. --collect-only: reconciled under the chain lock, nothing submitted, the finished array collected
    r = _battery(w, w.sha, "--collect-only", "--label", "lost")
    assert r.returncode == 1, r.stdout[-2500:] + r.stderr[-1500:]
    assert "[battery] RECONCILED label=lost array=7001 tasks=2 submitter=" in r.stdout and "VERDICT status=FAIL label=lost final_label=lost" in r.stdout
    assert len(_calls(w.env, "sbatch")) == 1 and not _calls(w.env, "watch")
    jobs = dict(l.split("=", 1) for l in (w.out / "lost.jobs").read_text().splitlines())
    holder = dict(l.split("=", 1) for l in (resv / "holder").read_text().splitlines())
    arr = dict(l.split("=", 1) for l in (resv / "array").read_text().splitlines())
    assert jobs["ARRAY"] == "7001" and jobs["TASKS"] == "2" and jobs["SUBMITTER"] == holder["HOLDER"] and jobs["RESERVATION"] == str(resv) and jobs["SHA"] == w.sha and jobs["LABEL"] == "lost"
    assert jobs["SUBMITTED_UTC"] == arr["SUBMITTED_UTC"] and jobs["VENV_JOB"] == "none" and jobs["MAX_REPLANS"] == "1" and jobs["RUN_NAME"] == holder["RUN_NAME"] and jobs["WALLTIME"] == holder["WALLTIME"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", jobs["RECONCILED_UTC"]) and "REPLAN_OF" not in jobs and holder["REPLAN_OF"] == "" and holder["MAX_REPLANS"] == "1"
    assert (w.out / "lost.plan.json").read_bytes() == (reruns / "lost.plan.json").read_bytes() and not (w.out / "lost.plan.json.partial").exists()
    v = json.loads((w.out / "lost.verdict.json").read_text())
    assert v["status"] == "FAIL" and v["array_job_id"] == "7001" and [f["node"] for f in v["failures"]] == ["tests.test_probe_bad::test_broken"]
    assert [(t["slurm_array_job_id"], t["slurm_array_task_id"]) for t in v["task_records"]] == [("7001", "0"), ("7001", "1")]
    assert f"battery=lost reconciled_from=TESTHOST:{resv} array=7001" in (w.state / "jobs.log").read_text()
    # 3. a restarted plain run of a lost label: the finalized reservation answers the submission — reconciled, watched, collected
    (w.mock / "drop_after_submit").write_text("0")
    r = _battery(w, w.sha, "--files", *two, "--label", "lost2", "--tasks", "2")
    assert r.returncode == 2 and (w.out / "lost2.plan.json.partial").is_file() and len(_calls(w.env, "sbatch")) == 2
    r = _battery(w, w.sha, "--files", *two, "--label", "lost2", "--tasks", "2")
    assert r.returncode == 1, r.stdout[-2500:] + r.stderr[-1500:]
    assert "[battery:cluster] HELD label=lost2 array=7002" in r.stdout and "[battery] RECONCILED label=lost2 array=7002" in r.stdout and "VERDICT status=FAIL label=lost2" in r.stdout
    assert "(reconciled from the cluster, not submitted by this run)" in r.stdout and "[battery] submitted:" not in r.stdout
    assert len(_calls(w.env, "sbatch")) == 2 and _calls(w.env, "watch") == ["watch --slurm 7002 --host TESTHOST --poll 30 --pending-min 20 --stall-min 0 --max-hours 3"]
    assert sorted(p for p in os.listdir(reruns) if p.startswith("lost2")) == ["lost2", "lost2.plan.json"], "the restarted run's own plan never reached the cluster"
    assert dict(l.split("=", 1) for l in (w.out / "lost2.jobs").read_text().splitlines())["ARRAY"] == "7002" and not (w.out / "lost2.plan.json.partial").exists()
    # 4. the drop on a RE-PLAN's submission: collecting the root reaches the reserved label, reconciles it and finishes the chain
    (w.mock / "timeout_task_0").write_text("")
    (w.mock / "drop_after_submit").write_text("1")                                  # the root's sbatch step passes; the re-plan's drops
    r = _battery(w, w.sha, "--files", *three, "--label", "rpl", "--tasks", "2", "--max-replans", "1")
    assert r.returncode == 2 and (w.out / "rpl.jobs").is_file() and (w.out / "rpl-r2.plan.json.partial").is_file() and not (w.out / "rpl-r2.jobs").exists(), r.stdout[-1500:] + r.stderr
    assert len(_calls(w.env, "sbatch")) == 4 and (w.shared / "gates" / w.sha / "reservations" / "rpl-r2" / "array").read_text().startswith("ARRAY=7004\n")
    r = _battery(w, w.sha, "--collect-only", "--label", "rpl")
    assert r.returncode == 1 and "[battery] RECONCILED label=rpl-r2 array=7004 tasks=3" in r.stdout and "VERDICT status=FAIL label=rpl final_label=rpl-r2" in r.stdout, r.stdout[-2500:] + r.stderr
    r2 = dict(l.split("=", 1) for l in (w.out / "rpl-r2.jobs").read_text().splitlines())
    assert r2["REPLAN_OF"] == "rpl" and r2["ATTEMPT"] == "2" and r2["ARRAY"] == "7004" and "MAX_REPLANS" not in r2 and "RECONCILED_UTC" in r2
    assert len(_calls(w.env, "sbatch")) == 4 and json.loads((w.out / "rpl.verdict.json").read_text())["replans"] == 1
    # --collect-only on the lost re-plan's OWN label finds the chain's root through the reservation and collects from the root
    (w.out / "rpl-r2.jobs").unlink()
    (w.out / "rpl-r2.plan.json").rename(w.out / "rpl-r2.plan.json.partial")
    r = _battery(w, w.sha, "--collect-only", "--label", "rpl-r2")
    assert r.returncode == 1 and "rpl-r2 is a re-plan of rpl: a chain is collected from its root" in r.stdout and "RECONCILED label=rpl-r2 array=7004" in r.stdout, r.stdout[-2500:] + r.stderr
    assert "VERDICT status=FAIL label=rpl final_label=rpl-r2" in r.stdout and len(_calls(w.env, "sbatch")) == 4 and (w.out / "rpl-r2.jobs").is_file()
    # 5. negatives: the remote plan hashes differently, or the reservation names another sha -> refused by name; nothing submitted,
    #    nothing restored, the .partial trace kept for a later --collect-only
    (w.mock / "drop_after_submit").write_text("0")
    r = _battery(w, w.sha, "--files", *two, "--label", "tamper", "--tasks", "2")
    assert r.returncode == 2 and len(_calls(w.env, "sbatch")) == 5
    remote_plan = reruns / "tamper.plan.json"
    original = remote_plan.read_bytes()
    remote_plan.write_bytes(original + b"\n")
    for args in ((w.sha, "--collect-only", "--label", "tamper"), (w.sha, "--files", *two, "--label", "tamper", "--tasks", "2")):
        r = _battery(w, *args)
        assert r.returncode == 2, (args, r.stdout[-1500:], r.stderr)
        assert re.search(rf"\[battery\] ERROR: another collector finalized tamper on TESTHOST \(reservation {re.escape(str(w.shared))}/gates/{w.sha}/reservations/tamper\) but it cannot be reconciled here: "
                         r"the remote plan hashes to [0-9a-f]{64}, the reservation recorded [0-9a-f]{64} — nothing submitted, nothing written", r.stderr), r.stderr
        assert not (w.out / "tamper.jobs").exists() and not (w.out / "tamper.plan.json").exists() and (w.out / "tamper.plan.json.partial").is_file()
        assert len(_calls(w.env, "sbatch")) == 5 and "RECONCILED" not in r.stdout
    remote_plan.write_bytes(original)
    hp = w.shared / "gates" / w.sha / "reservations" / "tamper" / "holder"
    htxt = hp.read_text()
    hp.write_text(htxt.replace(f"SHA={w.sha}", "SHA=" + "f" * 40))
    r = _battery(w, w.sha, "--collect-only", "--label", "tamper")
    assert r.returncode == 2 and f"cannot be reconciled here: the reservation names sha {'f' * 40}, not {w.sha} — nothing submitted" in r.stderr and len(_calls(w.env, "sbatch")) == 5, r.stderr
    hp.write_text(htxt)
    r = _battery(w, w.sha, "--collect-only", "--label", "tamper")
    assert r.returncode == 1 and "RECONCILED label=tamper array=7005" in r.stdout and len(_calls(w.env, "sbatch")) == 5, r.stdout[-1500:] + r.stderr
    # a .partial with no reservation behind it (a submission cut BEFORE its sbatch) has nothing to collect: still "not submitted from here"
    (w.out / "never2.plan.json.partial").write_text("{}")
    r = _battery(w, w.sha, "--collect-only", "--label", "never2")
    assert r.returncode == 2 and "label never2 was not submitted from here, and the cluster holds no finalized reservation for it" in r.stderr and len(_calls(w.env, "sbatch")) == 5, r.stderr
    assert not (w.out / "never2.jobs").exists()


def test_battery_a_finalized_label_met_with_a_different_request_is_refused_by_name_never_inherited(tmp_path):
    """A plain submission from another state dir with the same sha and label but a DIFFERENT request (files,
    task flags, or a forced task count) must not inherit the finalized array's records — an array that tested one passing file
    would otherwise certify a request that adds a failing one. Refused by name, nothing written or submitted; the same request
    reconciles; --collect-only collects the original; a new label is a normal submission."""
    w = _battery_world(tmp_path)
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "--label", "same")
    assert r.returncode == 0 and "VERDICT status=PASS label=same" in r.stdout and len(_calls(w.env, "sbatch")) == 1, r.stdout[-1500:] + r.stderr
    out2, state2 = tmp_path / "out2", tmp_path / "state2"
    env2 = dict(w.env, BATTERY_OUT=str(out2), REGISTRY_LANDING_STATE=str(state2))
    # the files differ: the one-file PASS array must not certify a request that adds the failing file
    r = _run([LANDING / "battery_slurm.sh", w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_bad.py", "--label", "same"], env2)
    assert r.returncode == 2 and "HELD label=same array=7001" in r.stdout and "RECONCILED" not in r.stdout and "VERDICT" not in r.stdout, r.stdout[-1500:] + r.stderr
    assert re.search(r"\[battery\] ERROR: label same is finalized for a different request: 1 files vs 2 requested \(only in the finalized plan: \[\]; only in this request: \['tests/test_probe_bad.py'\]\); "
                     r"use a new label, or `--collect-only --label same` to collect the original request — nothing written, nothing submitted", r.stderr), r.stderr
    assert len(_calls(w.env, "sbatch")) == 1 and not (out2 / "same.jobs").exists() and not (out2 / "same.plan.json").exists() and not (out2 / "same.verdict.json").exists()
    assert sorted(p for p in os.listdir(w.shared / "gates" / w.sha / "reruns") if p.startswith("same")) == ["same", "same.plan.json"], "the refused request's plan never reached the cluster"
    # the same file, a forced task count the original did not have
    r = _run([LANDING / "battery_slurm.sh", w.sha, "--files", "tests/test_probe_ok.py", "--label", "same", "--tasks", "1"], env2)
    assert r.returncode == 2 and "label same is finalized for a different request: tasks not forced vs forced to 1 requested;" in r.stderr and len(_calls(w.env, "sbatch")) == 1, r.stderr
    # the same file, other task flags
    r = _run([LANDING / "battery_slurm.sh", w.sha, "--files", "tests/test_probe_ok.py", "--label", "same"], dict(env2, BATTERY_TASK_FLAGS="--hide-slurm"))
    assert r.returncode == 2 and "label same is finalized for a different request: task flags [] vs ['--hide-slurm'] requested;" in r.stderr and len(_calls(w.env, "sbatch")) == 1, r.stderr
    # the SAME request: reconciled (the original's plan and jobs), collected, nothing submitted
    r = _run([LANDING / "battery_slurm.sh", w.sha, "--files", "tests/test_probe_ok.py", "--label", "same"], env2)
    assert r.returncode == 0 and "[battery] RECONCILED label=same array=7001 tasks=1" in r.stdout and "VERDICT status=PASS label=same" in r.stdout and len(_calls(w.env, "sbatch")) == 1, r.stdout[-1500:] + r.stderr
    assert json.loads((out2 / "same.plan.json").read_text()) == json.loads((w.out / "same.plan.json").read_text())
    # --collect-only after a refused different request collects the original (the refused request's .partial is the trace it reads)
    shutil.rmtree(out2)
    out2.mkdir()
    r = _run([LANDING / "battery_slurm.sh", w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_bad.py", "--label", "same"], env2)
    assert r.returncode == 2 and (out2 / "same.plan.json.partial").is_file()
    r = _run([LANDING / "battery_slurm.sh", w.sha, "--collect-only", "--label", "same"], env2)
    assert r.returncode == 0 and "RECONCILED label=same array=7001 tasks=1" in r.stdout and "VERDICT status=PASS label=same" in r.stdout and len(_calls(w.env, "sbatch")) == 1, r.stdout[-1500:] + r.stderr
    assert json.loads((out2 / "same.verdict.json").read_text())["requested_files"] == ["tests/test_probe_ok.py"], "the original request, never the refused one"
    # the two files under a NEW label: a normal submission
    r = _run([LANDING / "battery_slurm.sh", w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_bad.py", "--label", "same2"], env2)
    assert r.returncode == 1 and "[battery] submitted: array=7002" in r.stdout and "VERDICT status=FAIL label=same2" in r.stdout and len(_calls(w.env, "sbatch")) == 2, r.stdout[-1500:] + r.stderr


def test_battery_a_cut_between_the_two_local_renames_leaves_a_final_plan_without_jobs_that_collect_only_reconciles(tmp_path):
    """The local record is two renames (the plan into place, then .jobs); a kill between them leaves a final
    plan, no .jobs, no .partial, and a finalized reservation. That is a recoverable trace like the .partial: --collect-only probes
    the reservation under the chain lock, verifies the local plan's hash against the reservation's PLAN_SHA256, restores .jobs and
    collects with no sbatch; a local plan that is not the submission's is refused by name; plan + .jobs stays the normal path.
    The restored record carries the submission's stamp BYTE FOR BYTE: a submission is stamped once, on the cluster, into the
    reservation's array file, and the JOBS line hands that stamp to the local record (the local record used to read the
    grading host's clock a round trip later, and the two disagreed whenever a second boundary fell in between)."""
    w = _battery_world(tmp_path)
    r = _battery(w, w.sha, "--files", "tests/test_probe_ok.py", "tests/test_probe_bad.py", "--label", "cut", "--tasks", "2")
    assert r.returncode == 1 and len(_calls(w.env, "sbatch")) == 1, r.stdout[-1500:] + r.stderr
    plan_bytes = (w.out / "cut.plan.json").read_bytes()
    jobs0 = dict(l.split("=", 1) for l in (w.out / "cut.jobs").read_text().splitlines())
    arr0 = dict(l.split("=", 1) for l in (w.shared / "gates" / w.sha / "reservations" / "cut" / "array").read_text().splitlines())
    assert jobs0["SUBMITTED_UTC"] == arr0["SUBMITTED_UTC"] and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", jobs0["SUBMITTED_UTC"]), "one stamp per submission: the local record carries the reservation's"
    assert re.search(r"\[battery:cluster\] JOBS venv=none array=7001 reservation=\S+ submitted=" + re.escape(jobs0["SUBMITTED_UTC"]) + r"$", r.stdout, re.M), r.stdout[-1500:]
    assert not (w.out / "cut.jobs.partial").exists() and not (w.out / "cut.plan.json.partial").exists()
    # the cut: the plan is final, the record never landed (and the verdict is gone too, so the collection must happen again)
    (w.out / "cut.jobs").unlink()
    (w.out / "cut.verdict.json").unlink()
    n_ssh, n_watch = len(_calls(w.env, "ssh")), len(_calls(w.env, "watch"))
    r = _battery(w, w.sha, "--collect-only", "--label", "cut")
    assert r.returncode == 1, r.stdout[-2000:] + r.stderr
    assert "[battery] RECONCILED label=cut array=7001 tasks=2" in r.stdout and "local_plan=verified" in r.stdout and "VERDICT status=FAIL label=cut final_label=cut" in r.stdout
    assert len(_calls(w.env, "sbatch")) == 1 and len(_calls(w.env, "watch")) == n_watch == 1, "nothing submitted, nothing new watched"
    jobs = dict(l.split("=", 1) for l in (w.out / "cut.jobs").read_text().splitlines())
    assert jobs["ARRAY"] == "7001" and jobs["SUBMITTER"] == jobs0["SUBMITTER"] and jobs["SUBMITTED_UTC"] == jobs0["SUBMITTED_UTC"] and jobs["TASKS"] == "2" and "RECONCILED_UTC" in jobs
    assert (w.out / "cut.plan.json").read_bytes() == plan_bytes and json.loads((w.out / "cut.verdict.json").read_text())["array_job_id"] == "7001"
    assert any("reservations/cut/" in c for c in _calls(w.env, "ssh")[n_ssh:]), "the reservation was probed"
    # a local final plan that is NOT the submission's (no .jobs): refused by name, nothing written
    (w.out / "cut.jobs").unlink()
    (w.out / "cut.plan.json").write_bytes(plan_bytes + b"\n")
    r = _battery(w, w.sha, "--collect-only", "--label", "cut")
    assert r.returncode == 2 and re.search(rf"cannot be reconciled here: a final plan already at {re.escape(str(w.out))}/cut\.plan\.json \(with no \.jobs\) hashes to [0-9a-f]{{64}}, the reservation recorded [0-9a-f]{{64}}: not this submission's plan — move it aside", r.stderr), r.stderr
    assert not (w.out / "cut.jobs").exists() and len(_calls(w.env, "sbatch")) == 1 and "RECONCILED" not in r.stdout
    (w.out / "cut.plan.json").write_bytes(plan_bytes)
    r = _battery(w, w.sha, "--collect-only", "--label", "cut")
    assert r.returncode == 1 and "RECONCILED label=cut array=7001" in r.stdout and (w.out / "cut.jobs").is_file()
    # plan + .jobs present (the normal state): the normal path, no reconciliation, the reservation not probed
    n_ssh = len(_calls(w.env, "ssh"))
    r = _battery(w, w.sha, "--collect-only", "--label", "cut")
    assert r.returncode == 1 and "RECONCILED" not in r.stdout and "VERDICT status=FAIL label=cut" in r.stdout and len(_calls(w.env, "sbatch")) == 1
    assert not any("reservations" in c for c in _calls(w.env, "ssh")[n_ssh:])
