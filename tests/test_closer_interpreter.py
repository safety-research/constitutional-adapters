"""The closer's interpreter: since a4a304a38 the job-side closer (scripts/sdfbase/full22/registry_close_stores.py)
recomputes the eval goal set in-process through jailbreak_run, and an interpreter that cannot import it refuses by name as UNTRUSTED. No sbatch
assigned F22_PY, so every EXIT trap ran the compute node's bare python3 and common.sh's `|| true` swallowed the refusal: Slurm reported success
while the runs never advanced to generated or the grading worklist.

Now `f22_driver_venv` hands the confirmed driver venv to the trap as F22_PY, and `f22_registry_close_stores` keeps the job's rc but names a
refusal three times — the job log (`[closer] REFUSED rc=… under …`), the RESULT surface (a CLOSER_REFUSED line) and the registry (one `running`
state event PER REFUSED RUN from the closer's refused-run list <pending>.refused_run_ids.json, each note carrying the closer's first REFUSED
line — through the closer's refusal-note mode, which files each under its store lock and never over a
close; no list → nothing stated and a loud line, never `--pending-dir … --job` over the job's whole run set). And the
final push still follows. Executed under bash with the real function
bodies and recording stubs for the two interpreters (the bare `python3` on PATH — which hands the recipe writer's `-S -` program to the real interpreter — and a driver-venv stand-in); the REAL closer under the bare
python and under the driver venv is the grading path's test."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

COMMON = REPO / "scripts/sdfbase/full22/common.sh"
FULL22 = REPO / "scripts/sdfbase/full22"
UNTRUSTED = "REFUSED: UNTRUSTED: the eval goal set cannot be recomputed by this interpreter from cfg.yaml (ModuleNotFoundError: No module named 'jailbreak_run')"


def _run(tmp: Path, call: str, *, f22_py: str | None, closer_rc: int, with_pending: bool = True, refused_ids: list[str] | None = None, note_rc: int = 0) -> dict:
    """Source common.sh with a recording `python3` on PATH (the bare interpreter: refuses the closer with `closer_rc`, answers the closer's
    refusal-note mode (`--refusal-note`, the exit path's statement) with `note_rc`, hands `-S -` and `-c`
    programs — the recipe writer and the exit path's refused-run-list counter — to the real interpreter, answers everything else with 0) and a
    recording driver-venv stand-in (`venvpy`, always 0); run `call`. `refused_ids`: the FULL run ids the stub closer leaves in
    <pending>.refused_run_ids.json beside the pending file as it refuses — None leaves no list, which is what a closer that died before
    judging (or one from before the sidecar existed) leaves."""
    import shutil
    shared = tmp / "shared"; shutil.rmtree(shared, ignore_errors=True)
    (shared / "registry" / "pending").mkdir(parents=True)
    if with_pending:
        (shared / "registry" / "pending" / "4242.json").write_text('{"runs": [{"run_id": "r2v4:m:ev:obj:sha", "campaign": "r2"}]}')
    figs = tmp / "figs"; shutil.rmtree(figs, ignore_errors=True); figs.mkdir()
    bin_dir = tmp / "bin"; bin_dir.mkdir(exist_ok=True)
    log = tmp / "hook.log"; log.unlink(missing_ok=True)
    # the refused-run list the stub closer leaves beside the pending file (staged here, copied into place by the closer stub as it refuses)
    side = shared / "registry" / "pending" / "4242.json.refused_run_ids.json"; staged = tmp / "refused_run_ids.json"; staged.unlink(missing_ok=True)
    if refused_ids is not None:
        staged.write_text(json.dumps({"job_id": "4242", "refused_run_ids": list(refused_ids), "decision": "D2100"}))
    leave_list = f"cp {staged} {side}; " if refused_ids is not None else ""
    (bin_dir / "python3").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        printf '%s\\n' "PYTHON $*" >> "$HOOK_LOG"
        case "$*" in *--refusal-note*) {'echo "REFUSED: synthetic refusal of the note mode"; ' if note_rc else ''}exit {note_rc};; *registry_close_stores.py*) {leave_list}echo "{UNTRUSTED}" >&2; exit {closer_rc};; "-S -") exec {sys.executable} -S -;; "-c "*) exec {sys.executable} "$@";; esac
        exit 0
        """)); (bin_dir / "python3").chmod(0o755)
    (bin_dir / "venvpy").write_text("#!/bin/bash\nprintf '%s\\n' \"VENVPY $*\" >> \"$HOOK_LOG\"\nexit 0\n"); (bin_dir / "venvpy").chmod(0o755)
    script = textwrap.dedent(f"""
        set -u
        # closer-fix-5 (D779): only what production exports is exported — the job env (SISTEER_*, SLURM_*, FIGS_DIR, HOOK_LOG); SHARED, WAVE_SHA,
        # REPO_DIR, TAG, NSHARDS and WAVE_ROOT are PLAIN shell variables, exactly as f22_init and the sbatches assign them (the old fixture exported
        # them all and so never saw the recipe writer read os.environ for values that were never there)
        export SISTEER_REGISTRY_SPOOL={shared}/registry/records SLURM_JOB_ID=4242 SLURM_JOB_NAME=f22-att-test FIGS_DIR={figs} HOOK_LOG={log} SLURMD_NODENAME=TESTHOST
        SHARED={shared}; WAVE_SHA={'a' * 40}; REPO_DIR={REPO}; TAG=t; NSHARDS=9; WAVE_ROOT={shared}/waves/g
        {"F22_PY=" + f22_py if f22_py else "unset F22_PY"}
        source {COMMON}
        f22_registry_state(){{ printf '%s\\n' "STATE $*" >> "$HOOK_LOG"; }}
        {call}
        echo "RC=$?"
    """)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp), "HOOK_LOG": str(log)}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    lines = log.read_text().splitlines() if log.exists() else []
    result = (figs / "RESULT").read_text() if (figs / "RESULT").exists() else ""
    return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "log": lines, "result": result}


def test_a_refusing_closer_under_the_bare_interpreter_is_loud_on_the_record_and_returns_the_closers_rc(tmp_path):
    out = _run(tmp_path, "f22_registry_close_stores", f22_py=None, closer_rc=2)
    assert out["rc"] == 0 and "RC=2" in out["stdout"], out                                              # the function RETURNS the closer's rc (closer-fix-2): f22_result makes it the job's exit
    closer = [l for l in out["log"] if "registry_close_stores.py" in l]
    assert len(closer) == 1 and closer[0].startswith("PYTHON ") and "--pending " in closer[0] and "/registry/pending/4242.json" in closer[0], out["log"]
    assert "[registry] " + UNTRUSTED in out["stdout"], out["stdout"]                                   # the closer's own words still reach the log
    loud = [l for l in out["stderr"].splitlines() if l.startswith("[closer] REFUSED")]
    assert len(loud) == 1 and loud[0].startswith("[closer] REFUSED rc=2 under python3 (F22_PY unset — the bare interpreter, not the driver venv): job 4242's registered runs stay as they were — REFUSED: UNTRUSTED"), out["stderr"]
    assert re.search(r"job=4242 name=f22-att-test CLOSER_REFUSED rc=2 interpreter=python3$", out["result"], re.M), out["result"]   # the RESULT surface
    notice = [l for l in out["log"] if "sisteer.registry.emit_cli state" in l]
    # the stub closer left no refused-run list, so the exit path states NOTHING — before, ONE `--pending-dir … --job` notice restated
    # `running` over the job's whole registered set (the twins the same closer had just superseded included); the job log says why, by name
    assert notice == [], notice
    side = tmp_path / "shared" / "registry" / "pending" / "4242.json.refused_run_ids.json"
    assert f"[registry] closer refused (rc 2) but no refused-run list at {side} — stating nothing (D2100 (b): never the job's whole run set)" in out["stdout"], out["stdout"]
    assert not [l for l in out["log"] if l.startswith("STATE ")]                                        # no terminal state is filed for it


def test_a_closer_that_exits_clean_under_the_driver_venv_is_quiet_and_runs_under_f22_py(tmp_path):
    out = _run(tmp_path, "f22_registry_close_stores", f22_py="venvpy", closer_rc=2)                      # the bare python3 WOULD refuse — it is not called
    assert out["rc"] == 0 and "RC=0" in out["stdout"], out
    closer = [l for l in out["log"] if "registry_close_stores.py" in l]
    assert len(closer) == 1 and closer[0].startswith("VENVPY "), out["log"]                               # F22_PY is the interpreter
    assert "[closer]" not in out["stderr"] and out["result"] == "" and not [l for l in out["log"] if "emit_cli state" in l], out


def test_a_refused_closer_that_left_a_refused_run_list_hands_exactly_those_runs_to_the_closers_refusal_note_mode(tmp_path):
    """The closer writes the FULL ids of the runs it refused to <pending>.refused_run_ids.json; the exit path states `running` for
    exactly those — through the closer's REFUSAL-NOTE MODE (one call: `--refusal-note` carrying the
    closer's first REFUSED line, the pending file, the spool, the wave root and tag the store lock is named from, the node; the mode files
    per run under its store lock and never over a close of the run), never the bare state emitter by `--run-id` (the unfenced statement
    that undid a peer's supersession) and never `--pending-dir … --job` over the job's whole registered set (which restated `running` over the
    twins the same closer had just superseded). The mode's refusal is loud and the closer's rc still stands. Synthetic ids."""
    rid = "sdfbase_full22:static:1111111111111111:2222222222222222"
    out = _run(tmp_path, "f22_registry_close_stores", f22_py=None, closer_rc=2, refused_ids=[rid])
    assert out["rc"] == 0 and "RC=2" in out["stdout"], out                                              # the closer's rc is still returned
    assert not [l for l in out["log"] if "emit_cli state" in l], out["log"]                             # the bare state emitter is never called
    calls = [l for l in out["log"] if "registry_close_stores.py" in l]
    assert len(calls) == 2 and all(l.startswith("PYTHON ") for l in calls) and "--refusal-note" not in calls[0], out["log"]   # the closer, then the note
    note = calls[1]
    for want in ("--refusal-note closer REFUSED rc=2 under python3: REFUSED: UNTRUSTED", "--pending ", "/registry/pending/4242.json", "--spool ", f"--code-sha {'a' * 40}",
                 "--node TESTHOST", "--wave-root ", "/waves/g", "--tag t", "--job-name f22-att-test"):
        assert want in note, (want, note)
    assert "--pending-dir" not in note and "--run-id" not in note and "--array-task-event" not in note, note   # never the job's whole run set; the note's own mode
    assert ("[registry] closer refusal note: 1 listed refused run(s) of job 4242 decided by the closer's refusal-note mode under python3 (the refused-run list; D2100 (b)) — "
            "`running` filed per run under its store lock, never over a close") in out["stdout"] and "no refused-run list" not in out["stdout"], out["stdout"]
    assert not [l for l in out["log"] if l.startswith("STATE ")]                                        # no terminal state is filed for it
    assert re.search(r"job=4242 name=f22-att-test CLOSER_REFUSED rc=2 interpreter=python3$", out["result"], re.M), out["result"]   # the RESULT surface is unchanged
    # two refused runs: still ONE call — the mode reads the list itself; the hook counts what it hands over
    rid2 = "sdfbase_full22:static:3333333333333333:4444444444444444"
    out = _run(tmp_path, "f22_registry_close_stores", f22_py=None, closer_rc=2, refused_ids=[rid, rid2])
    calls = [l for l in out["log"] if "registry_close_stores.py" in l]
    assert len(calls) == 2 and "--refusal-note" in calls[1] and not [l for l in out["log"] if "emit_cli state" in l], out["log"]
    assert "closer refusal note: 2 listed refused run(s) of job 4242 decided by the closer's refusal-note mode" in out["stdout"], out["stdout"]
    # the mode's refusal: loud on stdout, nothing stated for the runs, the closer's rc unchanged
    out = _run(tmp_path, "f22_registry_close_stores", f22_py=None, closer_rc=2, refused_ids=[rid], note_rc=2)
    assert out["rc"] == 0 and "RC=2" in out["stdout"] and "[registry] REFUSED: synthetic refusal of the note mode" in out["stdout"], out["stdout"]
    assert ("[registry] closer refusal note NOT filed for job 4242 (rc 2 under python3): the refused runs' lifecycle was not touched; the refusal is on the job log, the RESULT surface "
            "and the reclose recipe") in out["stdout"] and "closer refusal note: 1 listed" not in out["stdout"], out["stdout"]
    assert not [l for l in out["log"] if "emit_cli state" in l], out["log"]


def test_the_clean_exit_branch_still_pushes_after_a_refused_closer_and_a_missing_pending_file_is_a_notice(tmp_path):
    out = _run(tmp_path, "f22_registry_result 'attacks t g rc=0'", f22_py=None, closer_rc=2)
    kinds = [("closer" if "registry_close_stores.py" in l else "notice" if "emit_cli state" in l else "recipe" if l == "PYTHON -S -" else "push" if "emit_cli push" in l else l) for l in out["log"]]
    # refusal → the reclose recipe → the final push; no `notice` — without the closer's refused-run list nothing is stated (a `notice`
    # here would be the old `--pending-dir … --job` restatement over the job's whole run set)
    assert kinds == ["closer", "recipe", "push"], out["log"]
    assert "no refused-run list at" in out["stdout"] and "stating nothing (D2100 (b)" in out["stdout"], out["stdout"]
    assert "[closer] REFUSED rc=2" in out["stderr"] and "--final" in [l for l in out["log"] if "emit_cli push" in l][0], out
    out = _run(tmp_path, "f22_registry_close_stores", f22_py=None, closer_rc=2, with_pending=False)
    assert out["rc"] == 0 and "RC=0" in out["stdout"] and "nothing registered to close" in out["stderr"] and out["log"] == [] and out["result"] == "", out


def test_the_driver_venv_hands_itself_to_the_trap_and_the_multi_turn_sbatches_confirm_it_before_arming_the_trap():
    src = COMMON.read_text()
    body = re.search(r"^f22_driver_venv\(\)\{.*?^\}", src, re.S | re.M)
    assert body and 'F22_PY="${F22_PY:-$MPY}"' in body.group(0), "f22_driver_venv must hand the confirmed venv to the EXIT trap as F22_PY"
    assert body.group(0).index('f22_fatal "no driver venv at $MPY"') < body.group(0).index('F22_PY="${F22_PY:-$MPY}"')      # after the venv's guards
    for name in ("target_attacks.sbatch", "target_attacks_goalset.sbatch"):
        sb = (FULL22 / name).read_text()
        assert sb.index("f22_driver_venv") < sb.index("trap cleanup EXIT"), f"{name}: the venv must be confirmed before the trap is armed"
    for name in ("bon_climb.sbatch", "target_static.sbatch", "mmlu_full.sbatch", "smoke_full22.sbatch", "petri_proxy.sbatch"):
        assert "f22_driver_venv" in (FULL22 / name).read_text(), name                                     # every other job type that closes confirms the venv too
