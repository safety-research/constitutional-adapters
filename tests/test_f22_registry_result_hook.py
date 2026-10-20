"""common.sh `f22_registry_result` (the job's RESULT hook): a RESULT line carrying `rc=<non-zero>` files `failed`; a clean exit runs the
job-side closer (`f22_registry_close_stores` → scripts/sdfbase/full22/registry_close_stores.py) BEFORE the final push; FAILED / PREEMPTED /
CANCELLED words file their terminal state as before.

Before the hook: target_attacks.sbatch's "attacks <tag> <group> rc=N" took the clean branch, so the FAILED 2:0 targets' 36 multi-turn runs
and 808553's 12 static runs stayed `running`; and no jailbreak_run family filed `generated` itself. Executed under bash with the
real function bodies and recording stubs for `f22_registry_state`, the closer's python and the pusher's python — never string-matched."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

COMMON = REPO / "scripts/sdfbase/full22/common.sh"


def _run(tmp: Path, result_line: str, *, with_pending: bool = True, stub_state: bool = True, extra_env: str = "", call: str = "f22_registry_result",
         python_stub: str | None = None) -> dict:
    """Source common.sh, stub the state filer and python (both the closer and the pusher go through it), call f22_registry_result.
    `stub_state=False` keeps the REAL f22_registry_state (its array routing is under test; python is still the recording stub); `extra_env` is
    exported before the call (an array task's SLURM_ARRAY_* variables); `call` names the function called with `result_line` as its arguments; `python_stub` replaces the recording python's body."""
    import shutil
    shared = tmp / "shared"; shutil.rmtree(shared, ignore_errors=True)                      # each call starts clean (a test calls _run more than once)
    (shared / "registry" / "pending").mkdir(parents=True); (shared / "scripts" / "registry").mkdir(parents=True)
    if with_pending:
        (shared / "registry" / "pending" / "4242.json").write_text("{}")
    bin_dir = tmp / "bin"; bin_dir.mkdir(exist_ok=True)
    (tmp / "hook.log").unlink(missing_ok=True)
    (bin_dir / "python3").write_text(python_stub or "#!/bin/bash\nprintf '%s\\n' \"PYTHON $*\" >> \"$HOOK_LOG\"\nexit 0\n"); (bin_dir / "python3").chmod(0o755)
    log = tmp / "hook.log"
    script = textwrap.dedent(f"""
        set -u
        export SHARED={shared} SISTEER_REGISTRY_SPOOL={shared}/registry/records SLURM_JOB_ID=4242 SLURM_JOB_NAME=f22-att-test WAVE_SHA={'a' * 40}
        export REPO_DIR={REPO} FIGS_DIR={tmp}/figs HOOK_LOG={log} TAG=t NSHARDS=9 WAVE_ROOT={shared}/waves/g SLURMD_NODENAME=TESTHOST
        {extra_env}
        mkdir -p "$FIGS_DIR"
        source {COMMON}
        {"f22_registry_state(){ printf '%s\\n' \"STATE $*\" >> \"$HOOK_LOG\"; }" if stub_state else ""}
        {call} {result_line}
        echo "RC=$?"
    """)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp), "HOOK_LOG": str(log)}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    lines = log.read_text().splitlines() if log.exists() else []
    return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "log": lines}


def test_a_result_line_with_a_non_zero_rc_files_failed_and_runs_no_closer_or_push(tmp_path):
    out = _run(tmp_path, "'attacks qwen35_4b_demo_battery mt-u0-a rc=2'")
    assert out["rc"] == 0 and "RC=0" in out["stdout"], out
    assert out["log"] == ["STATE failed FAILED:rc=2 final"], out


def test_the_failed_and_cancelled_words_file_their_terminal_state_as_before(tmp_path):
    assert _run(tmp_path, "'FAILED exit 1'")["log"] == ["STATE failed FAILED final"]
    assert _run(tmp_path, "'PREEMPTED by slurm'")["log"] == ["STATE cancelled PREEMPTED final"]
    assert _run(tmp_path, "'CANCELLED'")["log"] == ["STATE cancelled CANCELLED final"]


def test_a_clean_exit_runs_the_closer_before_the_final_push(tmp_path):
    out = _run(tmp_path, "'attacks qwen35_4b_demo_battery mt-u0-a rc=0'")
    assert out["rc"] == 0, out
    py = [l for l in out["log"] if l.startswith("PYTHON ")]
    assert len(py) == 2, out["log"]
    closer, push = py
    assert "registry_close_stores.py" in closer and "--pending " in closer and "/registry/pending/4242.json" in closer and "--spool " in closer \
        and f"--code-sha {'a' * 40}" in closer and "--nshards 9" in closer and "--tag t" in closer, closer
    assert "sisteer.registry.emit_cli push" in push and "--job 4242" in push and "--final" in push, push
    assert not [l for l in out["log"] if l.startswith("STATE ")]                       # the store hook / closer file generated, not the RESULT hook


def test_a_clean_exit_without_a_pending_file_skips_the_closer_and_still_pushes(tmp_path):
    out = _run(tmp_path, "'attacks tag grp'", with_pending=False)
    py = [l for l in out["log"] if l.startswith("PYTHON ")]
    assert len(py) == 1 and "emit_cli push" in py[0], out["log"]
    assert "no pending file" in out["stderr"] or "nothing registered to close" in out["stderr"], out["stderr"]


def test_the_standalone_rc_token_is_taken_never_a_family_rc_suffix(tmp_path):
    """misalign_proxy.sbatch's RESULT: "misalign <tag> rc=$RC agentic_rc=$RC_A em8q_rc=$RC_E" — agentic failed (rc=1) while em8q passed
    (em8q_rc=0): the hook files FAILED:rc=1, never the LAST `rc=` on the line; a trailing `na` family rc changes nothing."""
    out = _run(tmp_path, "'misalign qwen35_4b_demo_battery rc=1 agentic_rc=1 em8q_rc=0'")
    assert out["log"] == ["STATE failed FAILED:rc=1 final"], out
    out = _run(tmp_path, "'misalign qwen35_4b_demo_battery rc=1 agentic_rc=1 em8q_rc=na'")
    assert out["log"] == ["STATE failed FAILED:rc=1 final"], out
    # a clean job whose family rcs read 0 is clean: the closer then the push, nothing filed failed
    out = _run(tmp_path, "'misalign qwen35_4b_demo_battery rc=0 agentic_rc=0 em8q_rc=0'")
    assert not [l for l in out["log"] if l.startswith("STATE ")] and len([l for l in out["log"] if l.startswith("PYTHON ")]) == 2, out["log"]
    # a family rc alone (no standalone rc=) is not a failure signal for this hook: em8q_rc=1 with rc=0 stays clean (the family's own RESULT word decides elsewhere)
    out = _run(tmp_path, "'misalign qwen35_4b_demo_battery rc=0 agentic_rc=0 em8q_rc=1'")
    assert not [l for l in out["log"] if l.startswith("STATE ")], out["log"]


# --- an ARRAY TASK's events ------------------------------------------------------------------------------------------------------------------
# The climb array's tasks register ONE run per cell (the companion key); a task's `rc=N` filed as the RUN's `failed` reversed a peer's close.
# For a task with SLURM_ARRAY_TASK_ID, f22_registry_state routes `running` / `failed` / `cancelled` through the closer's array-task mode
# (registry_close_stores.py --array-task-event, which never files over a close and records the exit on the task), and the clean exit records
# `completed` after the closer and the push; a job that is not an array is byte for byte as before (the tests above).

def _array_env(task="3", job="4200", lo="0", hi="3", step="1"):
    return f"export SLURM_ARRAY_TASK_ID={task} SLURM_ARRAY_JOB_ID={job} SLURM_ARRAY_TASK_MIN={lo} SLURM_ARRAY_TASK_MAX={hi} SLURM_ARRAY_TASK_STEP={step}"


def _py(out):
    return [l for l in out["log"] if l.startswith("PYTHON ")]


def test_an_array_tasks_failure_goes_through_the_closers_array_task_mode_never_the_bare_per_task_failed(tmp_path):
    out = _run(tmp_path, "'bon-climb t shard=3/4 rc=2 failed_drivers=1'", stub_state=False, extra_env=_array_env())
    assert out["rc"] == 0 and "RC=0" in out["stdout"], out
    py = _py(out); assert len(py) == 1, out["log"]
    for want in ("registry_close_stores.py", "--pending ", "/registry/pending/4242.json", "--spool ", f"--code-sha {'a' * 40}", "--array-task-event failed", "--array-job 4200",
                 "--array-task 3", "--array-tasks 0-3", "--node TESTHOST", "--scheduler-state FAILED:rc=2", "--wave-root ", "--tag t", "--job-name f22-att-test"):
        assert want in py[0], (want, py[0])
    assert "emit_cli" not in py[0] and not [l for l in out["log"] if "emit_cli state" in l], out["log"]      # never the bare per-task `failed`
    # the FAILED / PREEMPTED / CANCELLED words route the same way, with their state
    for line, ev, sched in (("'FAILED exit 1'", "failed", "FAILED"), ("'PREEMPTED by slurm'", "cancelled", "PREEMPTED"), ("'CANCELLED'", "cancelled", "CANCELLED")):
        py = _py(_run(tmp_path, line, stub_state=False, extra_env=_array_env()))
        assert len(py) == 1 and f"--array-task-event {ev}" in py[0] and f"--scheduler-state {sched}" in py[0] and "emit_cli" not in py[0], (line, py)
    # a stepped array spells its step; a task set the scheduler did not export is passed EMPTY (the mode refuses by name — nothing is guessed)
    py = _py(_run(tmp_path, "'bon-climb t rc=2'", stub_state=False, extra_env=_array_env(step="2", hi="6")))
    assert "--array-tasks 0-6:2" in py[0], py
    py = _py(_run(tmp_path, "'bon-climb t rc=2'", stub_state=False, extra_env="export SLURM_ARRAY_TASK_ID=3 SLURM_ARRAY_JOB_ID=4200"))
    assert len(py) == 1 and "--array-tasks  --node" in py[0] and "--array-tasks 0" not in py[0], py      # `$*` keeps the empty word: two spaces
    # the mode's refusal is loud on stdout (the RESULT hook drops stderr) and the hook still returns 0: the job's own rc stands
    refusing = "#!/bin/bash\nprintf '%s\\n' \"PYTHON $*\" >> \"$HOOK_LOG\"\necho 'REFUSED: array task set names no task'\nexit 2\n"
    out = _run(tmp_path, "'bon-climb t rc=2'", stub_state=False, extra_env=_array_env(), python_stub=refusing)
    assert out["rc"] == 0 and "RC=0" in out["stdout"] and "[registry] REFUSED: array task set names no task" in out["stdout"], out
    assert "array task 3 of 4200: event failed NOT settled (rc 2 under python3) — the shared runs' lifecycle was not touched; nothing filed over a peer's close" in out["stdout"], out


def test_an_array_tasks_clean_exit_runs_the_closer_the_push_then_records_the_exit_on_the_task(tmp_path):
    out = _run(tmp_path, "'bon-climb t shard=0/4 rc=0 failed_drivers=0'", stub_state=False, extra_env=_array_env(task="0"))
    assert out["rc"] == 0 and "RC=0" in out["stdout"], out
    py = _py(out); assert len(py) == 3, out["log"]
    closer, push, exit_ = py
    assert "registry_close_stores.py" in closer and "--array-task-event" not in closer and "--pending " in closer, closer
    assert "emit_cli push" in push and "--job 4242" in push and "--final" in push, push
    assert "registry_close_stores.py" in exit_ and "--array-task-event completed" in exit_ and "--scheduler-state COMPLETED" in exit_ and "--array-task 0" in exit_ \
        and "--array-job 4200" in exit_ and "--array-tasks 0-3" in exit_, exit_
    # without a pending file: the closer is skipped, the push still runs, and the exit has no run to attach to (a notice, no python)
    out = _run(tmp_path, "'bon-climb t rc=0'", stub_state=False, extra_env=_array_env(task="0"), with_pending=False)
    py = _py(out); assert len(py) == 1 and "emit_cli push" in py[0], out["log"]
    assert "array task 0 of 4200: no pending file" in out["stdout"] and "the task's completed has no run to attach to" in out["stdout"], out["stdout"]


def test_an_array_tasks_job_start_running_is_routed_as_started_and_a_plain_jobs_state_is_byte_for_byte_as_before(tmp_path):
    # f22_registry_state running (f22_init and f22_registry_register call it): an array task's goes through the mode as `started`
    out = _run(tmp_path, "running RUNNING", stub_state=False, extra_env=_array_env(task="2"), call="f22_registry_state")
    py = _py(out); assert len(py) == 1 and "--array-task-event started" in py[0] and "--scheduler-state RUNNING" in py[0] and "--array-task 2" in py[0] and "emit_cli" not in py[0], out["log"]
    # a plain job (no SLURM_ARRAY_TASK_ID): the state emitter over the job's pending file, exactly as before
    out = _run(tmp_path, "running RUNNING", stub_state=False, call="f22_registry_state")
    py = _py(out); assert len(py) == 1 and py[0].startswith("PYTHON -S -m sisteer.registry.emit_cli state --state running --pending-dir ") and "--job 4242" in py[0] \
        and "--node TESTHOST" in py[0] and "--scheduler-state RUNNING" in py[0] and "registry_close_stores" not in py[0], py
    out = _run(tmp_path, "failed FAILED:rc=2 final", stub_state=False, call="f22_registry_state")
    py = _py(out); assert len(py) == 1 and "emit_cli state --state failed" in py[0] and "--final" in py[0] and "registry_close_stores" not in py[0], py
    # an array task before its registration (f22_init's `running`, no pending file yet): a notice, nothing run
    out = _run(tmp_path, "running RUNNING", stub_state=False, extra_env=_array_env(task="2"), with_pending=False, call="f22_registry_state")
    assert _py(out) == [] and "array task 2 of 4200: no pending file" in out["stdout"] and "the task's started has no run to attach to" in out["stdout"], out
