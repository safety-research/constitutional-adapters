"""31e-fix-3: the resume reservation's reconciliation fails CLOSED. Before it, `sacct_by_name` discarded sacct's exit
code and the reconciliation read the resulting EMPTY rows as proof that sbatch never ran — so a retry during a slurmdbd failure, or inside
accounting's visibility lag after sbatch accepted, destroyed the reservation while the first job ran, and a second resume drove the same
runs and stores. Now a nonzero sacct (or squeue) is AccountingUnavailable → refused with the token standing; an empty sacct is corroborated
in live squeue by the RESERVED JOB NAME (a hit is adopted); and non-launch is declared only on sacct rc 0 + empty, squeue empty by name, and a
token at least TRANSITION_MIN_AGE_S old. Executed against the real funnel with sacct/squeue wrappers ahead of the fake Slurm."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_submit_twin import World  # noqa: E402
from test_resume_from_job import LW, SHA, TARGET, _register  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402


def _wrap_bin(w: World) -> Path:
    """sacct/squeue wrappers ahead of the fake: `<fake>/sacct_fail` → sacct errors (rc 1); `<fake>/jobs/<id>.hidden` → sacct omits that job
    (accounting lag) while squeue still shows it; squeue gains `-n NAME` (the controller's live view by job name; `<fake>/squeue_fail` → rc 1)."""
    d = w.tmp / "wrap"; d.mkdir(exist_ok=True)
    (d / "sacct").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        d="${{FAKE_SLURM_DIR:?}}"
        [ -f "$d/sacct_fail" ] && {{ echo "sacct: error: Problem talking to the database: Connection refused" >&2; exit 1; }}
        {w.bin}/sacct "$@" | while IFS= read -r ln; do id="${{ln%%|*}}"; id="${{id%%_*}}"; [ -f "$d/jobs/$id.hidden" ] && continue; printf '%s\\n' "$ln"; done
        exit "${{PIPESTATUS[0]}}"
        """)); (d / "sacct").chmod(0o755)
    (d / "squeue").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        d="${{FAKE_SLURM_DIR:?}}"; [ -f "$d/outage" ] && exit 255
        name=""; prev=""; for a in "$@"; do [ "$prev" = "-n" ] && name="$a"; case "$a" in --name=*) name="${{a#--name=}}";; esac; prev="$a"; done
        if [ -n "$name" ]; then
          [ -f "$d/squeue_fail" ] && {{ echo "squeue: error: slurm_load_jobs error: Unable to contact slurm controller (connect failure)" >&2; exit 1; }}
          for f in "$d"/jobs/*.name; do [ -f "$f" ] || continue; id=$(basename "$f" .name); [ "$(cat "$f")" = "$name" ] || continue
            comment=$(grep -oE -- '--comment=[^ ]+' "$d/jobs/$id.argv" 2>/dev/null | head -1 | sed 's/^--comment=//'); [ -n "$comment" ] || comment="(null)"   # the nonce the job carries (31e-fix-6)
            st=$(cat "$d/jobs/$id.state"); case "$st" in PENDING|RUNNING|CONFIGURING|COMPLETING) echo "$id|$st|$(cat "$d/jobs/$id.submit" 2>/dev/null)|$USER|$comment";; esac; done
          exit 0
        fi
        exec {w.bin}/squeue "$@"
        """)); (d / "squeue").chmod(0o755)
    return d


def _submit(w: World, wrap: Path, *extra, **env):
    e = w.env(**env); e["PATH"] = f"{wrap}:{e['PATH']}"
    return subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m", "submit", "--unit", "multiturn.actorattack.g012",
                           "--lwid", LW, "--script", TARGET, "--name", "f22-att-m-g012-actorattack", "--args", f"m u0 actorattack {SHA} 9 0,1,2", "--clusters", "cluster_a",
                           "--group", "mt-u0-cluster_a", "--wave-sha", SHA, *extra], capture_output=True, text=True, env=e, timeout=120)


def _world(tmp_path):
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n"); wrap = _wrap_bin(w)
    r = _submit(w, wrap); assert r.returncode == 0, r.stdout + r.stderr
    j1 = str(job_ids(w.fake["cluster_a"])[-1]); aid1 = [e for e in w.ledger() if e["event"] == "attempt"][-1]["attempt_id"]
    _register(w, j1, aid1); set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00")
    return w, wrap, j1


def _token(w: World):
    return ST.read_transition(str(w.state / "submissions.jsonl"), LW)


def _attempts(w: World):
    return [e for e in w.ledger() if e["event"] == "attempt"]


def test_an_accounting_error_after_a_post_sbatch_crash_refuses_and_keeps_the_reservation_and_a_visibility_lag_adopts_from_squeue(tmp_path):
    w, wrap, j1 = _world(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, wrap, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="submit_after_sbatch"); assert r.returncode == 97, r.stdout + r.stderr
    jobs = [str(j) for j in job_ids(fake)]; j2 = jobs[-1]; assert len(jobs) == 2 and _token(w)["resume"] is True and len(_attempts(w)) == 1
    # slurmdbd down: sacct exits 1 — before the fix its empty output read as "never launched" and the reservation was destroyed under a running job
    (fake / "sacct_fail").write_text("")
    r = _submit(w, wrap, "--resume-from-job", j1)
    assert r.returncode == 3 and "accounting unavailable" in r.stderr and "STANDS" in r.stderr and "Connection refused" in r.stderr, r.stdout + r.stderr
    assert _token(w) is not None and _token(w)["resume"] is True and [str(j) for j in job_ids(fake)] == jobs and len(_attempts(w)) == 1     # intact, nothing launched
    (fake / "sacct_fail").unlink()
    # accounting lag: sacct has no row for j2 yet while the controller shows it under the reserved name — adopted as the resume attempt, nothing launched twice
    (fake / "jobs" / f"{j2}.hidden").write_text("")
    r = _submit(w, wrap, "--resume-from-job", j1)
    out = r.stdout + r.stderr
    assert r.returncode == 3 and "accounting has no row yet" in out and "ADOPTED" in out and "already IN FLIGHT" in r.stderr and "not in sacct yet" in r.stderr, out
    att = _attempts(w); assert len(att) == 2 and att[-1]["job_id"] == j2 and att[-1]["spec"]["resume_of"]["job_id"] == j1
    assert _token(w) is None and [str(j) for j in job_ids(fake)] == jobs
    b = ST.binding_state(str(w.state / "submissions.jsonl"), LW); assert b["state"] == "registered" and str(b["job_id"]) == j1         # no binding for the adopted resume
    (fake / "jobs" / f"{j2}.hidden").unlink()
    # the row appears: the same in-flight answer, from sacct this time
    r = _submit(w, wrap, "--resume-from-job", j1); assert r.returncode == 3 and "already IN FLIGHT" in r.stderr and "not in sacct yet" not in r.stderr, r.stderr


def test_non_launch_needs_sacct_empty_squeue_empty_by_the_reserved_name_and_an_aged_token(tmp_path):
    w, wrap, j1 = _world(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, wrap, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="resume_after_reservation"); assert r.returncode == 97, r.stdout + r.stderr   # token written, sbatch never ran
    jobs = [str(j) for j in job_ids(fake)]; assert len(jobs) == 1 and _token(w)["resume"] is True
    # the controller cannot be asked: absence is not proven
    (fake / "squeue_fail").write_text("")
    r = _submit(w, wrap, "--resume-from-job", j1)
    assert r.returncode == 3 and "controller cannot be asked" in r.stderr and "STANDS" in r.stderr and _token(w) is not None and [str(j) for j in job_ids(fake)] == jobs, r.stdout + r.stderr
    (fake / "squeue_fail").unlink()
    # sacct rc 0 and empty, squeue empty by name, but the token is seconds old: 'not yet', never 'never'
    r = _submit(w, wrap, "--resume-from-job", j1)
    assert r.returncode == 3 and "accounting-visibility floor" in r.stderr and "not yet" in r.stderr and _token(w) is not None and [str(j) for j in job_ids(fake)] == jobs, r.stdout + r.stderr
    assert len(_attempts(w)) == 1
    # the token ages past the floor: non-launch is proven, the token clears, and the resume launches exactly once
    p = ST.transition_path(str(w.state / "submissions.jsonl"), LW); tok = json.loads(Path(p).read_text()); tok["utc"] = "2026-01-01T00:00:00Z"; Path(p).write_text(json.dumps(tok))
    r = _submit(w, wrap, "--resume-from-job", j1)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "the sbatch never happened" in out and "sacct rc 0 and empty, squeue empty by name" in out and "token cleared" in out, out
    assert len(job_ids(fake)) == 2 and _token(w) is None and len(_attempts(w)) == 2
