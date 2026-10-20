"""closer-fix-4. HIGH: `reclose` called sbatch BEFORE appending the ledger line, with no reservation — a
launcher that died after sbatch accepted left a reclose job no ledger line knew of; if that job then died in its own pre-trap window it left no
receipt either, the live-only same-name check never saw the terminal orphan, and the cap permitted one more real reclose than it says. Now a
reclose is RESERVED before sbatch with the SAME primitive the resume path uses (the snapshot of every same-name job, a nonce the sbatch line
carries as --comment, the reservation's clock) and the next call reconciles the reservation by positive identity before any further reclose.
MEDIUM: the sbatch's STOP exit returned 0 before the trap, so the terminal receipt-less job counted as a failed reclose on every walk; now the
funnel refuses to submit while the wave's STOP stands, and a reclose that leaves under STOP records a `stopped` outcome the cap, the closed rule
and the same-refusal rule all ignore."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_closer_refusal_outcome import UNTRUSTED, _argv, _need_closer_history, _reclose_run, _world  # noqa: E402
from test_submit_resubmit_proof import MMLU  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402

LW_SLUG = "m__s1__mmlu__mmlu"


def _token(w):
    p = w.state / "bindings" / f"{LW_SLUG}.reclose.json"
    return json.loads(p.read_text()) if p.exists() else None


def _reclose_events(w):
    return [e for e in w.ledger() if e["event"] == "reclose"]


def test_a_launcher_that_dies_after_sbatch_leaves_a_reservation_the_next_walk_adopts_and_the_cap_still_counts_that_job(tmp_path):
    _need_closer_history()                                                       # the crash point sits past the closer-tree guard
    w = _world(tmp_path); fake = w.fake["cluster_a"]
    r = w.run(*MMLU, "--resubmit", SISTEER_TWIN_CRASH_AT="reclose_after_sbatch")
    assert r.returncode == 97 and "CRASH INJECTED at reclose_after_sbatch" in r.stderr, r.stdout + r.stderr
    assert job_ids(fake) == [1001, 1002] and _reclose_events(w) == [] and _token(w) is not None                   # the job exists, the ledger never heard of it, the reservation stands
    tok = _token(w); assert tok["job_id"] == "1001" and tok["job_name"] == "f22-reclose-1001" and re.fullmatch(r"[0-9a-f]{12}", tok["nonce"]) and tok["snapshot_job_ids"] == [] and tok["attempt"] == 1
    assert f"--comment=nonce:{tok['nonce']}" in _argv(w, "1002")
    # the next walk reconciles the reservation first: the job is ADOPTED into the ledger (by its nonce), then it is in flight — nothing launched
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "the reclose reservation's job exists" in out and "ADOPTED" in out and "positively identified by its nonce" in out and "in flight" in r.stdout, out
    ev = _reclose_events(w); assert len(ev) == 1 and ev[0]["reclose_job_id"] == "1002" and ev[0]["adopted"] is True and ev[0]["attempt"] == 1
    assert _token(w) is None and job_ids(fake) == [1001, 1002]
    # that reclose dies in its own pre-trap window (no receipt): the adopted ledger line makes it a counted, named attempt — the second reclose is 2/3
    set_job_state(fake, "1002", "FAILED", end="2026-01-01T00:20:00")
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1003 mmlu" in r.stdout and "reclose 2/3" in r.stdout and "('1002', ['FAILED'])" in out, out
    assert _token(w) is None and len(_reclose_events(w)) == 2 and f"--comment=nonce:" in _argv(w, "1003")


def test_a_launcher_that_dies_after_the_reservation_launched_nothing_the_floor_holds_and_the_aged_reservation_clears(tmp_path):
    _need_closer_history()                                                       # the reservation is written past the closer-tree guard
    w = _world(tmp_path); fake = w.fake["cluster_a"]
    r = w.run(*MMLU, "--resubmit", SISTEER_TWIN_CRASH_AT="reclose_after_reservation")
    assert r.returncode == 97 and job_ids(fake) == [1001] and _token(w) is not None and _reclose_events(w) == []
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "a reclose reservation of job 1001 stands" in r.stderr and "not yet" in r.stderr and job_ids(fake) == [1001] and _token(w) is not None, r.stdout + r.stderr
    p = w.state / "bindings" / f"{LW_SLUG}.reclose.json"; d = json.loads(p.read_text()); d["utc"] = "2026-01-01T00:00:00Z"; p.write_text(json.dumps(d))
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "launched nothing" in out and "cleared" in out and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "reclose 1/3" in r.stdout, out
    assert _token(w) is None and len(_reclose_events(w)) == 1


def test_a_reclose_is_refused_while_the_waves_stop_stands(tmp_path):
    # the STOP fence comes before the closer-tree guard: refused by name everywhere, the published repository included
    w = _world(tmp_path); fake = w.fake["cluster_a"]
    recipe = w.shared["cluster_a"] / "registry/pending/1001.closer_refused.json"; wave_root = Path(json.loads(recipe.read_text())["wave_root"])
    wave_root.mkdir(parents=True, exist_ok=True); (wave_root / "STOP").write_text("")
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "the wave's STOP stands" in r.stderr and "clear the STOP first" in r.stderr and job_ids(fake) == [1001] and _reclose_events(w) == [], r.stdout + r.stderr


def test_a_stopped_receipt_counts_toward_nothing_once_the_stop_is_cleared(tmp_path):
    _need_closer_history()
    w = _world(tmp_path)
    recipe = w.shared["cluster_a"] / "registry/pending/1001.closer_refused.json"; wave_root = Path(json.loads(recipe.read_text())["wave_root"])
    wave_root.mkdir(parents=True, exist_ok=True); (wave_root / "STOP").write_text(""); (wave_root / "STOP").unlink()        # the STOP stood, and was cleared
    # a receipt left under STOP (rc 0, stopped): not `closed`, not the same refusal, not an attempt — the first real reclose is 1/3
    d = json.loads(recipe.read_text()); d["recloses"] = [{"job_id": "71", "rc": 0, "why": "wave STOP stands", "stopped": True, "utc": "2026-01-01T00:30:00Z"}]; recipe.write_text(json.dumps(d))
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "reclose 1/3" in r.stdout and "closed by reclose" not in out, out


def test_the_reclose_job_honours_stop_after_its_trap_and_records_a_stopped_outcome_without_running_the_closer(tmp_path):
    out = _reclose_run(tmp_path, closer_rc=0, stop=True)
    assert out["rc"] == 0 and "not reclosing; recorded as stopped" in out["stdout"], out
    assert not [l for l in out["log"] if "registry_close_stores.py" in l] and [l for l in out["log"] if l.startswith("VENV ")] == []     # neither the venv nor the closer
    rec = out["recipe"]["recloses"]; assert len(rec) == 1 and rec[0]["stopped"] is True and rec[0]["rc"] == 0 and rec[0]["why"] == "wave STOP stands" and rec[0]["job_id"] == "5005"
    assert "RESULT reclose orig=1001 rc=0" in out["log"]                                                       # the trap ran: the outcome line exists
    # and the ordinary path still records rc / why without the flag
    out = _reclose_run(tmp_path / "plain", closer_rc=2)
    assert out["rc"] == 2 and out["recipe"]["recloses"][0].get("stopped") is None and out["recipe"]["recloses"][0]["rc"] == 2
