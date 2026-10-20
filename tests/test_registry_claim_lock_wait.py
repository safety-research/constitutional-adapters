"""A grader's registry claim WAITS for the harvester's write lock instead of dying at the first held poll (2026-09-30: every grader the
daemon launched died `LockHeld` at its first claim because the harvester's back-to-back cycles held the lock ~100 % of the time and the
claim gave up after 30 s). The wait is `pending.LOCK_TIMEOUT_S` (30 s by default, `SISTEER_REGISTRY_LOCK_TIMEOUT_S` raises it) and the
claim is a real write under that lock — its record is spooled and folded at once — so it is the harvester's lock it needs, not a lock of
its own. The knob carries the registry owner's semantics: the wait defaults to 30 s and ``SISTEER_REGISTRY_LOCK_TIMEOUT_S`` RAISES it."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth

HOLDER = r'''
import fcntl, sys, time
fh = open(sys.argv[1], "a+"); fcntl.flock(fh.fileno(), fcntl.LOCK_EX); print("held", flush=True); time.sleep(float(sys.argv[2])); fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
'''


def _root(tmp_path):
    root = str(tmp_path / "reg"); H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = synth.run_records("job7@none", 100, through="generated")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    return root, rid


def _hold(root: str, seconds: float) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", HOLDER, os.path.join(root, "harvester.lock"), str(seconds)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def test_the_default_wait_outlasts_a_harvester_cycle_and_the_env_overrides_it(monkeypatch):
    assert P.LOCK_TIMEOUT_S == float(os.environ.get("SISTEER_REGISTRY_LOCK_TIMEOUT_S") or 30.0)
    monkeypatch.setenv("SISTEER_REGISTRY_LOCK_TIMEOUT_S", "7.5")
    mod = importlib.reload(P)
    try:
        assert mod.LOCK_TIMEOUT_S == 7.5
    finally:
        monkeypatch.delenv("SISTEER_REGISTRY_LOCK_TIMEOUT_S"); importlib.reload(P)


def test_a_claim_waits_out_a_held_lock_within_the_bound_and_refuses_past_it(tmp_path, monkeypatch):
    root, rid = _root(tmp_path)
    monkeypatch.setattr(P, "LOCK_TIMEOUT_S", 6.0)
    holder = _hold(root, 2.0)                                     # a "harvester cycle" shorter than the bound: the claim lands after it
    t0 = time.monotonic()
    jid = P.claim(root, rid, synth.JUDGE_ID, "grader-1")
    waited = time.monotonic() - t0
    holder.wait(timeout=10)
    assert jid and 1.0 <= waited < 6.0
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "in_progress"
    monkeypatch.setattr(P, "LOCK_TIMEOUT_S", 1.0)
    holder = _hold(root, 3.0)                                     # held past the bound: refused by name, nothing written
    try:
        with pytest.raises(DB.LockHeld):
            P.claim(root, rid, synth.JUDGE_ID_2, "grader-2")
    finally:
        holder.wait(timeout=10)
