"""closer-fix-6: a reclose that refused in PREFLIGHT counts toward nothing and is recoverable.

closer-fix-5 made the reclose job record a `preflight` receipt when the recipe or the environment refused before the closer ran, and made the
funnel stop on it. But a preflight failure can be TRANSIENT (the pending file missing and then restored; the checkout restored later): the funnel
refused forever, offered no recovery, and counted the preflight job toward the 3-attempt cap, so a completed eval's stores could never be closed
by the supported path even after repair. Now the receipt names the INPUT it saw (recipe identity, pending file sha256, checkout presence); the
funnel accepts one more reclose when that input has CHANGED (the repair is new evidence, recorded on the ledger event), or when the operator records
a deliberate retry (`reclose --job J --retry-why …`, kept beside the receipt in the recipe's `retries`); the refusal otherwise names both recoveries
and forbids deleting the receipt; preflight jobs are excluded from the cap exactly like `stopped` ones. Everything executed: the real funnel over
the fake Slurm, the real reclose sbatch under bash."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_closer_refusal_outcome import _argv, _need_closer_history, _reclose_run, _world  # noqa: E402
from test_submit_resubmit_proof import MMLU  # noqa: E402
from test_submit_twin import SHA  # noqa: E402

sys.path.insert(0, str(LIB))
import submit_twin as TW  # noqa: E402


def _recipe(w) -> Path:
    return w.shared["cluster_a"] / "registry/pending/1001.closer_refused.json"


def _now_input(w) -> dict:
    """What the funnel sees: the recipe's identity, the pending file's sha256, the checkout's presence at the recipe's sha."""
    shared = w.shared["cluster_a"]; rec = json.loads(_recipe(w).read_text())
    pend = shared / "registry/pending/1001.json"
    return {"recipe_sha256": TW.recipe_identity(rec), "pending": hashlib.sha256(pend.read_bytes()).hexdigest() if pend.exists() else "absent",
            "checkout": "present" if (shared / "sisteer-eval-repo" / SHA / "src/sisteer").is_dir() else "absent"}


def _preflight(job: str, utc: str | None, inp: dict, why: str = "reclose 1001: the recipe's pending file /x/registry/pending/1001.json is gone") -> dict:
    """A preflight receipt as the reclose job's trap writes it; `utc` None = now (a receipt is always written AFTER the ledger line that submitted
    its job — a ledger line newer than the receipt means a later retry, which the funnel judges by the live / outcome rules instead)."""
    return {"job_id": job, "rc": 1, "why": why, "utc": utc or TW.utc_now(), "preflight": True, "input": inp}


def _receipts_dir(w) -> Path:
    return w.shared["cluster_a"] / "registry/pending"


def _append_receipt(w, rec: dict) -> None:
    """A receipt as the reclose job leaves it (closer-fix-7): its OWN file beside the recipe, never a line in the recipe."""
    (_receipts_dir(w) / f"1001.reclose.{rec['job_id']}.receipt.json").write_text(json.dumps(rec))


def _receipt_files(w) -> list:
    return sorted(_receipts_dir(w).glob("1001.reclose.*.receipt.json"))


def _retries(w) -> list:
    p = _receipts_dir(w) / "1001.reclose_retries.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def test_a_repaired_input_after_a_preflight_refusal_is_new_evidence_for_one_more_reclose_and_the_receipt_stays(tmp_path):
    _need_closer_history()
    w = _world(tmp_path); now = _now_input(w)
    # the preflight saw the pending file ABSENT; it is present now: changed → one more reclose, the change named, the receipt untouched
    _append_receipt(w, _preflight("71", "2026-01-01T00:30:00Z", {**now, "pending": "absent"}))
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "refused in preflight" in out and "the input it saw has since changed" in out and "pending: absent ->" in out \
        and "the repaired input is new evidence" in out and "reclose 1/3" in r.stdout, out                              # 1/3: the preflight job counted toward nothing
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    assert len(ev) == 1 and ev[0]["after_preflight"]["job_id"] == "71" and ev[0]["after_preflight"]["changed"] == ["pending"] and ev[0]["after_preflight"]["input_now"] == now, ev
    assert [json.loads(f.read_text()).get("preflight") for f in _receipt_files(w)] == [True] and "recloses" not in json.loads(_recipe(w).read_text()) or json.loads(_recipe(w).read_text()).get("recloses") == []   # never deleted
    assert f"reclose_stores.sbatch 1001 {SHA}" in _argv(w, "1002")


def test_an_unchanged_input_is_refused_by_name_with_both_recoveries_and_never_by_deleting_the_receipt(tmp_path):
    w = _world(tmp_path); now = _now_input(w)
    _append_receipt(w, _preflight("71", "2026-01-01T00:30:00Z", now))
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "reclose job 71 of job 1001 refused in PREFLIGHT" in r.stderr and "the input it saw is unchanged (recipe, pending file and checkout identical)" in r.stderr \
        and "Recovery: repair the input" in r.stderr and "--retry-why" in r.stderr and "never by deleting the receipt" in r.stderr and "preflight refusals count toward nothing" in r.stderr \
        and job_ids(w.fake["cluster_a"]) == [1001] and "reclose" not in [e["event"] for e in w.ledger()], r.stdout + r.stderr
    # a receipt that names no input (unknowable): refused the same way, saying so
    f = _receipt_files(w)[-1]; d = json.loads(f.read_text()); del d["input"]; f.write_text(json.dumps(d))
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 3 and "is unknown (the receipt names no input)" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stderr


def test_an_operator_retry_is_recorded_on_the_ledger_and_beside_the_receipt_and_a_retry_in_flight_is_judged_live(tmp_path):
    _need_closer_history()
    w = _world(tmp_path); now = _now_input(w)
    _append_receipt(w, _preflight("71", "2026-01-01T00:30:00Z", now))
    r = w.run("reclose", "--job", "1001", "--retry-why", "restored the pending file by hand; the recipe is right"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "the operator records a deliberate retry: restored the pending file by hand" in out and "reclose 1/3" in r.stdout, out
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    ap = ev[0]["after_preflight"]
    assert len(ev) == 1 and {k: ap[k] for k in ("job_id", "retry_why", "input")} == {"job_id": "71", "retry_why": "restored the pending file by hand; the recipe is right", "input": now} \
        and re.fullmatch(r"[^@\s]+@[^:\s]+:\d+", ap["who"]), ev                                                        # who retried (user@host:pid)
    d = json.loads(_recipe(w).read_text()); rt = _retries(w)
    assert [json.loads(f.read_text()).get("preflight") for f in _receipt_files(w)] == [True] and "retries" not in d and len(rt) == 1 and rt[0]["after_job"] == "71" \
        and rt[0]["reclose_job_id"] == "1002" and rt[0]["why"].startswith("restored the pending file") and rt[0]["input"] == now and rt[0]["who"] == ap["who"], (d, rt)
    assert TW.recipe_identity(d) == now["recipe_sha256"]                                                                 # the recipe is untouched: its identity stands
    # the retry 1002 is live (start_on_submit → RUNNING): a walk judges it in flight — not the preflight refusal
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "reclose f22-reclose-1001 1002 is RUNNING" in r.stdout and "in flight" in r.stdout and job_ids(w.fake["cluster_a"]) == [1001, 1002], r.stdout + r.stderr
    # the retry filed (rc 0 receipt): closed
    set_job_state(w.fake["cluster_a"], "1002", "COMPLETED", end="2026-01-01T00:40:00")
    _append_receipt(w, {"job_id": "1002", "rc": 0, "why": "", "utc": "2026-01-01T00:40:00Z"})
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "closed by reclose job 1002" in (r.stdout + r.stderr) and job_ids(w.fake["cluster_a"]) == [1001, 1002], r.stdout + r.stderr
    # --retry-why with no preflight receipt standing is inert (nothing to retry past): the ordinary rules decide
    w2 = _world(tmp_path / "plain")
    r = w2.run("reclose", "--job", "1001", "--retry-why", "nothing to see"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "deliberate retry" not in r.stdout, r.stdout + r.stderr
    assert "after_preflight" not in [e for e in w2.ledger() if e["event"] == "reclose"][0]


def test_three_preflight_refusals_do_not_exhaust_the_cap(tmp_path):
    """The cap boundary: three recloses that each refused in preflight (pending absent) and were each repaired — the fourth is `reclose 1/3`,
    because a preflight job is not an attempt; under the old counting the fourth would have been refused at the cap."""
    _need_closer_history()
    w = _world(tmp_path); fake = w.fake["cluster_a"]; now = _now_input(w)
    for i, (job, state) in enumerate([("1002", "FAILED"), ("1003", "FAILED"), ("1004", "NODE_FAIL")], start=1):
        r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
        assert r.returncode == 0 and f"RECLOSE cluster_a {job} mmlu" in r.stdout and "reclose 1/3" in r.stdout, (i, out)
        set_job_state(fake, job, state, end=f"2026-01-01T0{i}:00:00")
        _append_receipt(w, _preflight(job, None, {**now, "pending": "absent"}))                         # it refused in preflight (now): the pending file was gone; it is back
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1005 mmlu" in r.stdout and "reclose 1/3" in r.stdout and "the repaired input is new evidence" in out, out
    assert job_ids(fake) == [1001, 1002, 1003, 1004, 1005] and len([e for e in w.ledger() if e["event"] == "reclose"]) == 4
    assert [json.loads(f.read_text())["job_id"] for f in _receipt_files(w)] == ["1002", "1003", "1004"]                  # the history stands
    # a closer OUTCOME still counts: 1005 refuses in the closer (rc 2, a different why) → the next is reclose 2/3
    set_job_state(fake, "1005", "FAILED", end="2026-01-01T04:00:00")
    _append_receipt(w, {"job_id": "1005", "rc": 2, "why": "REFUSED: shard03 incomplete", "utc": "2026-01-01T04:00:00Z"})
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "RECLOSE cluster_a 1006 mmlu" in r.stdout and "reclose 2/3" in r.stdout, r.stdout + r.stderr


def test_the_reclose_jobs_preflight_receipt_names_the_input_it_saw_with_the_funnels_identity(tmp_path):
    out = _reclose_run(tmp_path / "pend", closer_rc=0, pending=False)
    assert out["rc"] == 1 and "is gone" in out["stderr"], out
    rec = out["recipe"]["recloses"][0]; assert rec["preflight"] is True and rec["rc"] == 1, rec
    recipe_on_disk = {k: v for k, v in out["recipe"].items() if k != "recloses"}
    assert rec["input"]["recipe_sha256"] == TW.recipe_identity(out["recipe"]) == TW.recipe_identity(recipe_on_disk)      # the sbatch's twin of the funnel's identity
    assert rec["input"]["pending"] == "absent" and rec["input"]["checkout"] == "absent", rec["input"]                   # the stub checkout has no src/sisteer
    # the pending file present: the input names its sha256 (a bad sha refuses in preflight before the pending check, so the receipt comes from the sha case)
    out = _reclose_run(tmp_path / "sha", closer_rc=0, recipe_sha="b" * 40)
    rec = out["recipe"]["recloses"][0]; pend = tmp_path / "sha" / "shared" / "registry/pending/1001.json"
    assert rec["preflight"] is True and rec["input"]["pending"] == hashlib.sha256(pend.read_bytes()).hexdigest() and rec["input"]["recipe_sha256"] == TW.recipe_identity(out["recipe"]), rec
