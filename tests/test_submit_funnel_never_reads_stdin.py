"""The submit funnel never reads its caller's stdin (paper-companion-repo stage E, 2026-09-30).

06_eval_waves.sh feeds its unit plan to a `while read` loop over a here-string and calls lib/slurm.sh slurm_submit ->
lib/submit_twin.py once per unit. The first REAL submission ran `ssh` with the loop's stdin inherited: ssh forwarded the
remaining 31 plan lines to the remote, the loop ended after its first unit and the stage reported `06 done: 1 unit(s)
placed` (the dry run, which runs no ssh, had placed all 37). common.sh repro_ssh always carried the guard (`ssh -n`, the
test seam `</dev/null`); these tests pin the same guard on the funnel's Remote.sh (both branches), on the detached watcher
and election children it forks and on lib/watch_job.sh's own ssh — each by EXECUTING
the form against a fake that reads its stdin (a string match on `-n` would not notice a caller that drops the redirect).
The loop-level property itself is pinned by the fake sbatch in front_door_testlib.py, which drains its stdin the way a
real sbatch-behind-ssh does, so test_front_door_wave_integration's 33-unit wave fails again if any path lets stdin through.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "reproduce" / "lib"
sys.path.insert(0, str(LIB))
import submit_twin as ST  # noqa: E402

MARK = "STDIN-LEAKED-INTO-THE-CHILD"

#: a fake `ssh` that behaves like the real one on the point under test: `-n` means "my stdin is /dev/null"; otherwise it
#: forwards (here: reads) the caller's stdin. It reports a leak to $FAKE_WATCH_DIR/leak and answers the sacct / squeue
#: forms lib/watch_job.sh issues with one COMPLETED job so the watcher reaches DONE.
FAKE_SSH = r'''#!/bin/bash
case " $* " in *" -n "*) exec </dev/null; touch "${FAKE_WATCH_DIR:?}/has_n";; esac
if read -t 1 line; then echo "LEAK:$line" >> "${FAKE_WATCH_DIR:?}/leak"; echo "LEAK:$line"; else echo NOLEAK; fi >&2
cmd="$*"
case "$cmd" in
  *"-o State,ExitCode,Elapsed,NodeList,JobID"*) echo "COMPLETED|0:0|00:00:01|TESTHOST|4242";;
  *"-o JobID,JobName,State,ExitCode,Elapsed,NodeList"*) echo "4242|jobname|COMPLETED|0:0|00:00:01|TESTHOST";;
  *"-o JobName,State"*) echo "jobname|COMPLETED";;
  *) ;;
esac
'''


def _exe(path: Path, text: str) -> Path:
    path.write_text(text); path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH); return path


def _child(prog: str, *argv: str, env=None) -> subprocess.CompletedProcess:
    """Run `prog` in a child interpreter whose stdin CARRIES the marker, so what the funnel does with its stdin is observable
    whatever pytest's own fd 0 is."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(prog), *argv], input=MARK + "\n", capture_output=True, text=True,
                          timeout=120, env=env)


def test_remote_sh_local_seam_does_not_pass_the_callers_stdin_through(tmp_path):
    prog = f"""
        import sys; sys.path.insert(0, {str(LIB)!r})
        from types import SimpleNamespace
        import submit_twin as ST
        site = SimpleNamespace(dry=False, clusters={{"cluster_a": {{"CLUSTER_SSH": "local:" + sys.argv[1], "CLUSTER_SHARED_ROOT": sys.argv[1]}}}})
        rc, out, err = ST.Remote(site, "cluster_a").sh("cat; echo REMOTE-DONE")
        print(out, end="")
    """
    p = _child(prog, str(tmp_path))
    assert p.returncode == 0, p.stderr
    assert "REMOTE-DONE" in p.stdout
    assert MARK not in p.stdout, "the seam handed the caller's stdin to the remote command"


def test_remote_sh_ssh_branch_never_reads_stdin_and_says_so_to_ssh(tmp_path):
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(); _exe(fake_bin / "ssh", FAKE_SSH)
    watch = tmp_path / "watch"; watch.mkdir()
    prog = f"""
        import sys; sys.path.insert(0, {str(LIB)!r})
        from types import SimpleNamespace
        import submit_twin as ST
        site = SimpleNamespace(dry=False, clusters={{"c": {{"CLUSTER_SSH": "somehost", "CLUSTER_SHARED_ROOT": "/x"}}}})
        rc, out, err = ST.Remote(site, "c").sh("true")
        print(err, end="")
    """
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", FAKE_WATCH_DIR=str(watch))
    p = _child(prog, env=env)
    assert p.returncode == 0, p.stderr
    assert not (watch / "leak").exists(), (watch / "leak").read_text() if (watch / "leak").exists() else ""
    assert "NOLEAK" in p.stdout
    assert (watch / "has_n").exists(), "ssh is invoked without -n: a real ssh would drain the caller's stdin"


def test_detached_watcher_and_election_children_are_forked_with_devnull_stdin(tmp_path, monkeypatch):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            calls.append((cmd, kw))

    monkeypatch.setattr(ST.subprocess, "Popen", FakePopen)
    monkeypatch.delenv("SISTEER_NO_WATCHER", raising=False)
    co = ST.Coordinator.__new__(ST.Coordinator)
    co.site = SimpleNamespace(dry=False, cluster_root=lambda label, model: str(tmp_path / "root"))
    co.remotes = {"c": SimpleNamespace(local_dir=None, host="somehost", shared="/x")}
    co.state = str(tmp_path / "state"); co.model = "m"
    co.launch_watcher("c", "4242", "static.t0")
    assert len(calls) == 1
    cmd, kw = calls[0]
    assert cmd[0].endswith("watch_job.sh") and kw["start_new_session"] is True
    assert kw.get("stdin") is subprocess.DEVNULL, "a detached watcher inherits the submitter's stdin and its ssh drains the plan"
    # the election child (`submit --twin` detaches `elect`) takes the same guard: pinned on the source form, since reaching that
    # Popen needs a full twin election (test_submit_twin.py covers the election itself)
    src = (LIB / "submit_twin.py").read_text()
    elect = [ln for ln in src.splitlines() if "elect.log" in ln and "Popen" in ln]
    assert elect and all("stdin=subprocess.DEVNULL" in ln for ln in elect), elect


def test_watch_job_sh_ssh_never_reads_the_watchers_stdin(tmp_path):
    fake_bin = tmp_path / "bin"; fake_bin.mkdir(); _exe(fake_bin / "ssh", FAKE_SSH)
    watch = tmp_path / "watch"; watch.mkdir()
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}", FAKE_WATCH_DIR=str(watch))
    p = subprocess.run([str(LIB / "watch_job.sh"), "--slurm", "4242", "--host", "fakehost", "--shared-root", "/x",
                        "--events", str(tmp_path / "events.jsonl"), "--poll", "30"],
                       input=MARK + "\n", capture_output=True, text=True, timeout=120, env=env)
    assert p.returncode == 0, p.stderr
    assert "DONE job=4242" in p.stdout, p.stdout
    assert not (watch / "leak").exists(), (watch / "leak").read_text()
    assert (watch / "has_n").exists(), "watch_job.sh calls ssh without -n"

