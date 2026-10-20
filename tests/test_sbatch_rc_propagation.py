"""The instrument must not lie: a payload that exits non-zero MUST make the job
script exit non-zero — the headline of the masked-failure incident. The sweep
sbatch backgrounds the workload and `wait`s on it; the
original tail `wait $!; log "...rc=$?"` let the final `log` (exit 0) become the
script's exit status, so a python that exited 1 was reported COMPLETED by sacct
and every watcher's DONE event lied. The fix is `wait $!; rc=$?; log; exit $rc`.

This test cannot be trusted to stay working by inspection — a future edit that
reintroduces a trailing command after `wait` silently breaks it again — so it is
pinned here: run the exact tail pattern with a failing and a succeeding payload
and assert the script's exit code tracks the payload's, not the log's."""

from __future__ import annotations

import subprocess
import textwrap


def _run_tail(payload_exit: int) -> int:
    # the FIXED sbatch tail, reduced to its rc-carrying essence
    script = textwrap.dedent(f"""
        set -uo pipefail
        log(){{ echo "[ts] $*"; }}   # a trailing command that always succeeds
        ( exit {payload_exit} ) &    # the backgrounded workload
        wait $!; rc=$?
        log "task done rc=$rc"       # must NOT become the script's exit status
        exit $rc
    """)
    return subprocess.run(["bash", "-c", script]).returncode


def test_failing_payload_makes_script_fail():
    # the exact bug: payload exits 1 -> script MUST exit non-zero (sacct FAILED)
    assert _run_tail(1) == 1
    assert _run_tail(2) == 2


def test_succeeding_payload_makes_script_succeed():
    assert _run_tail(0) == 0


def test_the_broken_pattern_would_have_masked_it():
    # documents WHY: the OLD tail (log after wait, no exit rc) masks failures —
    # this asserts the broken pattern is indeed broken, so the fix is meaningful.
    broken = textwrap.dedent("""
        set -uo pipefail
        log(){ echo "[ts] $*"; }
        ( exit 1 ) &
        wait $!
        log "task done rc=$?"        # log succeeds -> script exits 0 (the lie)
    """)
    assert subprocess.run(["bash", "-c", broken]).returncode == 0  # the masked-COMPLETED bug
