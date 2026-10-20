"""closer-fix-7: no reclose record ever has two writers.

An earlier design appended each reclose job's receipt onto the recipe and had the controller append its `--retry-why` note onto the same recipe: two
read/replace writers with no shared lock, so when both read before either replaced, the loser's write erased the other's record — a lost
`preflight` marker made the cap count a failure that must count toward nothing. Now the reclose job writes its receipt ONCE to its own file
(`<orig>.reclose.<job>.receipt.json`, atomic rename, never replaced), the controller appends its notes to `<orig>.reclose_retries.jsonl`, and
the recipe is immutable after the eval job writes it; the funnel reads all three in one pass. A LATER retry is a different job submitted after
the receipt's (job id and submission order), never a ledger line newer than the receipt — an adopted reservation job is appended at
reconciliation time, possibly after its own receipt. Every retry records who (user@host:pid). Everything executed: the real sbatch under bash,
the real funnel over the fake Slurm."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_closer_refusal_outcome import _need_closer_history, _reclose_run, _world  # noqa: E402
from test_reclose_preflight_recovery import _append_receipt, _now_input, _preflight, _receipt_files, _recipe, _retries  # noqa: E402
from test_submit_resubmit_proof import MMLU  # noqa: E402


def test_the_reclose_job_writes_its_own_receipt_file_and_leaves_the_recipe_byte_identical(tmp_path):
    for sub, kw in (("pend", dict(closer_rc=0, pending=False)), ("ok", dict(closer_rc=0)), ("again", dict(closer_rc=2)), ("stop", dict(closer_rc=0, stop=True))):
        out = _reclose_run(tmp_path / sub, **kw)
        assert out["recipe_unchanged"], (sub, out)                                                       # the recipe is never rewritten
        assert out["receipt_files"] == ["1001.reclose.5005.receipt.json"], (sub, out["receipt_files"])   # one receipt, this job's, in its own file
        rec = out["recipe"]["recloses"][0]; assert rec["job_id"] == "5005" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rec["utc"]), rec
    assert _reclose_run(tmp_path / "pend2", closer_rc=0, pending=False)["recipe"]["recloses"][0].get("preflight") is True
    assert _reclose_run(tmp_path / "stop2", closer_rc=0, stop=True)["recipe"]["recloses"][0].get("stopped") is True


def test_a_receipt_written_while_the_controller_records_its_retry_survives_and_the_cap_does_not_count_it(tmp_path):
    """The interleaving that lost the marker: the controller submits the retry and records its note while the new job's preflight receipt lands.
    With separate files the order cannot matter: the receipt file is pre-created the instant the job would write it (before the controller's
    note) and must survive the controller's write untouched; the recipe's bytes never change; the next walks judge the retry live, then — a
    preflight that counts toward nothing — refuse by name (not at the cap) while the input is unchanged."""
    _need_closer_history()
    w = _world(tmp_path); now = _now_input(w); recipe_before = _recipe(w).read_bytes()
    _append_receipt(w, _preflight("71", "2026-01-01T00:30:00Z", now))
    # the retry job the controller is about to submit will be 1002: its preflight receipt is already on disk when the controller writes its note
    _append_receipt(w, _preflight("1002", None, now))
    r = w.run("reclose", "--job", "1001", "--retry-why", "restored the checkout"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, out
    assert _recipe(w).read_bytes() == recipe_before                                                                  # the controller never touched the recipe
    assert sorted(f.name for f in _receipt_files(w)) == ["1001.reclose.1002.receipt.json", "1001.reclose.71.receipt.json"]
    assert json.loads((_receipt_files(w)[0]).read_text()).get("preflight") is True                                   # 1002's marker survived the controller's write
    rt = _retries(w); assert len(rt) == 1 and rt[0]["reclose_job_id"] == "1002" and re.fullmatch(r"[^@\s]+@[^:\s]+:\d+", rt[0]["who"]), rt
    # 1002 is live: in flight (the receipt it already wrote does not pre-empt the live check)
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "reclose f22-reclose-1001 1002 is RUNNING" in r.stdout and job_ids(w.fake["cluster_a"]) == [1001, 1002], r.stdout + r.stderr
    # 1002 ended (its preflight stands): counted toward nothing — refused by name for the unchanged input, never "cap"
    set_job_state(w.fake["cluster_a"], "1002", "FAILED", end="2026-01-01T00:50:00")
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "reclose job 1002 of job 1001 refused in PREFLIGHT" in r.stderr and "cap" not in r.stderr and job_ids(w.fake["cluster_a"]) == [1001, 1002], r.stderr
    # the input repaired: one more reclose, and it is `reclose 1/3` — neither preflight counted
    f = _receipt_files(w)[0]; d = json.loads(f.read_text()); d["input"] = {**now, "pending": "absent"}; f.write_text(json.dumps(d))
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "RECLOSE cluster_a 1003 mmlu" in r.stdout and "reclose 1/3" in r.stdout, r.stdout + r.stderr


def test_an_adopted_reclose_job_appended_after_its_own_receipt_is_not_a_later_retry(tmp_path):
    """MEDIUM: the ledger line of job 1002 is newer than 1002's own preflight receipt (an adopted reservation job is appended at reconciliation
    time) — by utc it looked like a later retry and the unchanged-input fence was skipped. By job id it is the same job: the fence applies."""
    _need_closer_history()
    w = _world(tmp_path); now = _now_input(w)
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, r.stdout + r.stderr   # the ledger line: now
    set_job_state(w.fake["cluster_a"], "1002", "FAILED", end="2026-01-01T00:50:00")
    _append_receipt(w, _preflight("1002", "2026-01-01T00:40:00Z", now))                                              # its receipt, dated BEFORE the ledger line
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "reclose job 1002 of job 1001 refused in PREFLIGHT" in r.stderr and "unchanged" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001, 1002], r.stdout + r.stderr
    # a genuinely later job (a higher id) IS a later retry: judged live / by outcome, not by the fence
    _append_receipt(w, _preflight("1002", "2026-01-01T00:40:00Z", now))
    r = w.run("reclose", "--job", "1001", "--retry-why", "fixed the pending file"); assert r.returncode == 0 and "RECLOSE cluster_a 1003 mmlu" in r.stdout, r.stdout + r.stderr
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "1003 is RUNNING" in r.stdout and "in flight" in r.stdout, r.stdout + r.stderr


def test_who_is_recorded_on_every_retry_record(tmp_path):
    _need_closer_history()
    w = _world(tmp_path); now = _now_input(w)
    _append_receipt(w, _preflight("71", "2026-01-01T00:30:00Z", {**now, "pending": "absent"}))
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, r.stdout + r.stderr
    ev = [e for e in w.ledger() if e["event"] == "reclose"][0]["after_preflight"]
    assert ev["changed"] == ["pending"] and re.fullmatch(r"[^@\s]+@[^:\s]+:\d+", ev["who"]), ev
