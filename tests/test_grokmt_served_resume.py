"""The grokmt Modal driver's resume path (scripts/modal/grokmt_served.py) and
the analyzer's handling of a resumed run (scripts/grokmt_analyze.py).

Background: attempt 1 of the smoke was stopped from the CLI at 53/348 goals
with the run tree synced to the volume. A resumed attempt must (a) seed the
local run dir from that copy so the drivers skip covered goals, (b) retry
``error:`` goals (the harness's resume rule), (c) set aside a torn PyRIT
``memory.db`` instead of crashing PyRIT on it, (d) file the previous attempt's
standup/driver logs where this attempt's cannot overwrite them, and (e) give
the analyzer the attempt windows so per-driver wall-clock never bridges the gap
between attempts. The driver module is imported with a stub ``modal`` (the test
venv has none) — the helpers under test are pure.

Synthetic data only: placeholder goal ids, TESTHOST-free paths under tmp_path.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "modal" / "grokmt_served.py"
sys.path.insert(0, str(REPO / "scripts"))


class _Chain:
    """Any attribute is a method returning self (Image builder chain, App decorators)."""
    name = "grokmt-served"

    def __getattr__(self, _name):
        return lambda *a, **k: (self if not (a and callable(a[0]) and len(a) == 1 and not k) else a[0])

    def __call__(self, *a, **k):
        if a and callable(a[0]) and len(a) == 1 and not k:
            return a[0]
        return self


def _stub_modal() -> types.ModuleType:
    m = types.ModuleType("modal")
    m.App = lambda name=None, **k: type("App", (), {"name": name or "grokmt-served",
                                                    "function": lambda self, **kw: (lambda f: f),
                                                    "local_entrypoint": lambda self, **kw: (lambda f: f)})()
    m.Volume = types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(commit=lambda: None, reload=lambda: None))
    m.Secret = types.SimpleNamespace(from_dict=lambda d: d)
    m.Image = types.SimpleNamespace(from_registry=lambda *a, **k: _Chain())
    m.is_local = lambda: True
    return m


@pytest.fixture(scope="module")
def drv():
    sys.modules.setdefault("modal", _stub_modal())
    spec = importlib.util.spec_from_file_location("grokmt_served_under_test", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ana():
    import grokmt_analyze
    return grokmt_analyze


def _rows(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def _valid_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("create table t(x)"); con.execute("insert into t values (1)"); con.commit(); con.close()


# --------------------------------------------------------------------------- #
# progress_counts: goals, not rows                                              #
# --------------------------------------------------------------------------- #

def test_progress_counts_last_row_per_goal_and_errors(drv, tmp_path):
    pj = tmp_path / "progress.jsonl"
    _rows(pj, [
        {"goal_id": "g1", "outcome": "AttackOutcome.SUCCESS"},
        {"goal_id": "g2", "outcome": "error:ActorDiscoveryError"},
        {"goal_id": "g2", "outcome": "error:ActorDiscoveryError"},   # retried, errored again
        {"goal_id": "g3", "outcome": "error:ActorDiscoveryError"},
        {"goal_id": "g3", "outcome": "failure"},                      # retried, then ran
    ])
    pj.write_text(pj.read_text() + '{"goal_id": "g4", "outc')          # torn tail
    c = drv.progress_counts(pj)
    assert c == {"done": 3, "errored": 1, "rows": 5}


def test_progress_counts_missing_file(drv, tmp_path):
    assert drv.progress_counts(tmp_path / "nope.jsonl") == {"done": 0, "errored": 0, "rows": 0}


# --------------------------------------------------------------------------- #
# seed_from_previous                                                           #
# --------------------------------------------------------------------------- #

def _prior_run(dest: Path, drv) -> None:
    pr = drv.PRIVATE_RUN
    # grok pair/none shard00: two covered goals + PyRIT memory
    d = dest / "grok" / pr / "pair" / "none" / "shard00"
    _rows(d / "progress.jsonl", [{"goal_id": "g1", "outcome": "AttackOutcome.SUCCESS",
                                  "n_target_forwards": 5},
                                 {"goal_id": "g2", "outcome": "AttackOutcome.FAILURE",
                                  "n_target_forwards": 15}])
    _rows(d / "responses.jsonl", [{"goal_id": "g1", "response": "r"}])
    _rows(d / "attacker_calls.jsonl", [{"t_start": 1000.0, "reply": "x"},
                                       {"t_start": 1500.5, "reply": "y"}])
    _valid_db(d / "memory.db")
    # grok actorattack/none shard00: one errored goal (retry on resume) + torn db
    e = dest / "grok" / pr / "actorattack" / "none" / "shard00"
    _rows(e / "progress.jsonl", [{"goal_id": "g1", "outcome": "error:ActorDiscoveryError",
                                  "n_target_forwards": 0}])
    _rows(e / "target_timing.jsonl", [{"t_start": 1700.0, "latency_s": 3.0}])
    (e / "memory.db").write_bytes(b"not a sqlite file at all" * 10)
    # qwen arm: a shard with nothing terminal yet
    q = dest / "qwen" / pr / "crescendo" / "none" / "shard01"
    _rows(q / "responses.jsonl", [{"goal_id": "g9", "response": "partial"}])
    # attempt-level artifacts
    (dest / "srv").mkdir(); (dest / "srv" / "standup.log").write_text("old standup\n")
    (dest / "driver_logs").mkdir(); (dest / "driver_logs" / "grok__pair__none__shard00.log").write_text("Retry attempt 1\n")
    (dest / "run_provenance.json").write_text(json.dumps({"run_id": "r", "started_at": "2026-01-01T00:00:00Z"}))
    (dest / "served_timing_probe.json").write_text("{}")
    (dest / "hf_timing_probe.json").write_text(json.dumps({"engine": "hf", "cells": {}}))


def test_seed_from_previous_seeds_files_counts_and_tornness(drv, tmp_path):
    dest, local = tmp_path / "dest", tmp_path / "local"
    _prior_run(dest, drv)
    rep = drv.seed_from_previous(dest, local, ["grok", "qwen"])
    pr = drv.PRIVATE_RUN
    # driver trees came over — progress + responses + the intact memory.db
    assert (local / "grok" / pr / "pair" / "none" / "shard00" / "progress.jsonl").exists()
    assert (local / "grok" / pr / "pair" / "none" / "shard00" / "memory.db").exists()
    assert (local / "qwen" / pr / "crescendo" / "none" / "shard01" / "responses.jsonl").exists()
    # torn memory.db set aside, never handed to PyRIT
    aa = local / "grok" / pr / "actorattack" / "none" / "shard00"
    assert not (aa / "memory.db").exists()
    assert (aa / "memory.db.torn-attempt01").exists()
    assert rep["torn_memory_dbs"] == [f"grok/{pr}/actorattack/none/shard00/memory.db"]
    # counts: 2 covered, 1 errored-to-retry
    assert rep["attempt"] == 2 and rep["previous_attempt"] == 1
    assert rep["covered_total"] == 2 and rep["errored_to_retry_total"] == 1
    assert rep["seeded"]["grok/pair/none/shard00"] == {"done": 2, "errored": 0, "rows": 2}
    assert rep["seeded"]["grok/actorattack/none/shard00"] == {"done": 1, "errored": 1, "rows": 1}
    # previous attempt filed, not left where this attempt's standup would overwrite it
    assert (local / "attempts" / "01" / "srv" / "standup.log").read_text() == "old standup\n"
    assert (local / "attempts" / "01" / "driver_logs" / "grok__pair__none__shard00.log").exists()
    assert (local / "attempts" / "01" / "run_provenance.json").exists()
    assert (local / "attempts" / "01" / "served_timing_probe.json").exists()
    assert not (local / "srv").exists() and not (local / "driver_logs").exists()
    # the HF probe is reused at the root
    assert json.loads((local / "hf_timing_probe.json").read_text())["engine"] == "hf"
    # attempt window: started_at from provenance, ended at the last logged t_start
    (a1,) = rep["attempts"]
    assert a1["attempt"] == 1 and a1["started_at"] == "2026-01-01T00:00:00Z"
    assert a1["started_at_epoch"] == pytest.approx(1767225600.0)
    assert a1["ended_at_epoch"] == 1700.0


def test_seed_from_previous_third_attempt_numbers_from_filed_attempts(drv, tmp_path):
    dest, local = tmp_path / "dest", tmp_path / "local"
    _prior_run(dest, drv)
    (dest / "attempts" / "01").mkdir(parents=True)
    (dest / "run_provenance.json").write_text(json.dumps({
        "run_id": "r", "started_at": "2026-01-01T00:00:00Z",
        "attempts": [{"attempt": 1, "started_at_epoch": 1.0, "ended_at_epoch": 2.0},
                     {"attempt": 2, "started_at_epoch": 3.0}]}))
    rep = drv.seed_from_previous(dest, local, ["grok"])
    assert rep["attempt"] == 3 and rep["previous_attempt"] == 2
    assert (local / "attempts" / "02" / "srv").is_dir()
    assert [a["attempt"] for a in rep["attempts"]] == [1, 2]
    assert rep["attempts"][1]["ended_at_epoch"] == 1700.0


def test_seed_from_previous_refuses_without_a_prior_run(drv, tmp_path):
    with pytest.raises(RuntimeError, match="no prior driver tree"):
        drv.seed_from_previous(tmp_path / "empty", tmp_path / "local", ["grok", "qwen"])


# --------------------------------------------------------------------------- #
# RunSyncer: ship only what changed                                             #
# --------------------------------------------------------------------------- #

def test_run_syncer_copies_changed_files_only(drv, tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    (src / "a").mkdir(parents=True)
    (src / "a" / "progress.jsonl").write_text("1\n")
    (src / "memory.db").write_bytes(b"x" * 10)
    sy = drv.RunSyncer(src, str(dest))
    assert sy.copy_changed() == 2
    assert (dest / "a" / "progress.jsonl").read_text() == "1\n"
    assert sy.copy_changed() == 0                      # nothing changed
    (src / "a" / "progress.jsonl").write_text("1\n2\n")  # size changed
    assert sy.copy_changed() == 1
    assert (dest / "a" / "progress.jsonl").read_text() == "1\n2\n"


def test_app_name_env_override(tmp_path):
    """A resumed attempt runs under its own app name via GROKMT_APP_NAME."""
    import os
    import subprocess
    stub = REPO / "tests" / "test_grokmt_served_resume.py"
    code = (
        "import importlib.util, sys, types\n"
        f"sys.path.insert(0, {str(REPO / 'tests')!r}); import test_grokmt_served_resume as t\n"
        "sys.modules['modal'] = t._stub_modal()\n"
        f"spec = importlib.util.spec_from_file_location('g', {str(DRIVER)!r}); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(m.app.name)\n")
    env = {**os.environ, "GROKMT_APP_NAME": "grokmt-served-r2", "PYTHONPATH": str(REPO / "scripts")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "grokmt-served-r2", out


# --------------------------------------------------------------------------- #
# analyzer: a resumed run's duplicate rows and attempt windows                  #
# --------------------------------------------------------------------------- #

def test_analyzer_last_progress_dedupes_by_goal(ana):
    rows = [{"goal_id": "g1", "outcome": "error:X"}, {"goal_id": "g2", "outcome": "success"},
            {"goal_id": "g1", "outcome": "failure"}]
    last = ana._last_progress(rows)
    assert sorted((r["goal_id"], r["outcome"]) for r in last) == [("g1", "failure"), ("g2", "success")]


def test_analyzer_span_within_windows_skips_the_gap(ana):
    # attempt 1: 1000..1500 ; killed ; attempt 2: 9000..9300  -> 500 + 300, not 8300
    ts = [1000.0, 1200.0, 1500.0, 9000.0, 9100.0, 9300.0]
    assert ana._span_within_windows(ts, [(900.0, 1600.0), (8900.0, float("inf"))]) == pytest.approx(800.0)
    assert ana._span_within_windows(ts, []) == pytest.approx(8300.0)   # single attempt: unchanged
    assert ana._span_within_windows([], [(0.0, 1.0)]) == 0.0


def test_analyzer_attempt_windows_from_provenance(ana, tmp_path):
    (tmp_path / "run_provenance.json").write_text(json.dumps({"attempts": [
        {"attempt": 1, "started_at_epoch": 10.0, "ended_at_epoch": 20.0},
        {"attempt": 2, "started_at_epoch": 30.0}]}))
    assert ana._attempt_windows(tmp_path) == [(10.0, 20.0), (30.0, float("inf"))]
    assert ana._attempt_windows(tmp_path / "none") == []


def test_analyzer_driver_logs_span_attempts(ana, tmp_path):
    (tmp_path / "driver_logs").mkdir(); (tmp_path / "driver_logs" / "grok__pair__none__shard00.log").write_text("")
    (tmp_path / "attempts" / "01" / "driver_logs").mkdir(parents=True)
    (tmp_path / "attempts" / "01" / "driver_logs" / "grok__pair__none__shard00.log").write_text("")
    assert len(ana._driver_logs(tmp_path, "grok__pair__none__*.log")) == 2
