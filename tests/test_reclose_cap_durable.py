"""closer-fix-3: the reclose cap counted only the recipe's receipt entries and LIVE same-name jobs — a reclose that died before
appending its receipt (f22_init / f22_driver_venv refusing before the trap, a node failure, walltime) left a terminal sacct row and a durable
ledger submission but no receipt, so every `06 --resubmit` walk launched another identical CPU job, past RECLOSE_MAX, the failure hidden behind
a fresh launch. Now the cap is enforced against the DURABLE record: every ledger `reclose` submission of the job is read on sacct; a terminal one
counts, receipted or not; a receipt-less one is named with its sacct state; the refusal at the cap names them all."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_closer_refusal_outcome import _argv, _need_closer_history, _world  # noqa: E402
from test_submit_resubmit_proof import MMLU  # noqa: E402


def test_a_reclose_that_died_before_its_receipt_counts_toward_the_cap_and_the_cap_refuses_naming_it(tmp_path):
    _need_closer_history()                                                       # the cap is reached only past three SUBMITTED recloses
    w = _world(tmp_path); fake = w.fake["cluster_a"]
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "reclose 1/3" in r.stdout, r.stdout + r.stderr
    # 1002 dies before its trap (no receipt on the recipe): the next walk counts it, names it, and submits the SECOND reclose
    set_job_state(fake, "1002", "FAILED", end="2026-01-01T00:20:00")
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1003 mmlu" in r.stdout and "reclose 2/3" in r.stdout and "1 earlier reclose(s) of job 1001 left no receipt" in out and "('1002', ['FAILED'])" in out, out
    assert "reclose_stores.sbatch 1001" in _argv(w, "1003")
    # 1003 dies the same way: the THIRD and last
    set_job_state(fake, "1003", "NODE_FAIL", end="2026-01-01T00:30:00")
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1004 mmlu" in r.stdout and "reclose 3/3" in r.stdout and "2 earlier reclose(s)" in out, out
    # 1004 dies too: the cap holds against the DURABLE submissions — refused by name, every receipt-less attempt and its state listed, nothing launched
    set_job_state(fake, "1004", "TIMEOUT", end="2026-01-01T00:40:00")
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "already had 3 reclose(s) (cap 3) — 3 durable submission(s) in the ledger, 0 receipt(s)" in r.stderr and "receipt-less — died before their trap wrote one" in r.stderr \
        and "('1002', ['FAILED'])" in r.stderr and "('1003', ['NODE_FAIL'])" in r.stderr and "('1004', ['TIMEOUT'])" in r.stderr and "last receipt: none" in r.stderr, r.stderr
    assert job_ids(fake) == [1001, 1002, 1003, 1004]
    assert len([e for e in w.ledger() if e["event"] == "reclose"]) == 3
    # a prior reclose still LIVE on sacct is in flight, whatever the recipe says: skipped, nothing launched
    w2 = _world(tmp_path / "live"); fake2 = w2.fake["cluster_a"]
    r = w2.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout
    r = w2.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "reclose f22-reclose-1001 1002 is RUNNING" in r.stdout and "in flight" in r.stdout and job_ids(fake2) == [1001, 1002], r.stdout + r.stderr
