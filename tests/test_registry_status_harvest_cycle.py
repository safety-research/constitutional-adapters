"""The status page's harvester row: the age of the last COMPLETED harvester cycle, red past 3x the harvest cadence (2026-09-29).

``state/last_run/ingest.json`` stamps every exit of the command; a cycle its unit kills at TimeoutStartSec stamps nothing there
either, but a schema refusal (exit 3) does, so that row cannot tell "folding" from "dying". The harvester row reads
``state/harvest_cycle.json``, which only a cycle that ran to completion writes (``harvest.write_cycle_state``), and turns LATE
when that stamp is older than ``[harvest] cadence_s`` x 3 -- four killed cycles in a row froze the paper frame for four hours with
every timer row green.
"""
from __future__ import annotations

import json
import os
import time

from sisteer.registry import config as CFG
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import status
from sisteer.registry import synth

from test_registry_status_ops import _render, _setup  # noqa: E402


def _utc_ago(now_s: float, seconds: float) -> str:
    ms = int((now_s - seconds) * 1000)
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


def _write_cycle(root, now_s, age_s, **extra):
    payload = {"last_completed_cycle_utc": _utc_ago(now_s, age_s), "cycle_utc": _utc_ago(now_s, age_s + 30), "elapsed_s": 30.0, "stopped": None,
               "budget_s": None, "files": 12, "ok": 12, "refused": 0, "blocked": 0, "revisions_applied": 2, "revisions_held": 0,
               "revisions_deferred_budget": 0, "plain_files_deferred_budget": 0, "transactions": 9}
    payload.update(extra)
    with open(os.path.join(root, "state", H.CYCLE_STATE_NAME), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def test_the_harvester_row_ages_the_last_completed_cycle_and_turns_late_past_three_cadences(tmp_path):
    root = _setup(tmp_path)
    now_s = time.time()
    # the fixture's own ingest_once wrote the state: a completed cycle a moment ago -> ok
    sec = status.harvest_cycle_section(root, now_s)
    assert sec["state"] == "ok" and sec["late"] is False and sec["age_s"] is not None and 0 <= sec["age_s"] < 60
    assert sec["cadence_s"] == CFG.DEFAULTS["harvest"]["cadence_s"] == 1200 and sec["late_after_s"] == 3600
    # no state file at all: unknown (a fresh install, or a registry no cycle has completed under this code yet)
    os.remove(os.path.join(root, "state", H.CYCLE_STATE_NAME))
    sec = status.harvest_cycle_section(root, now_s)
    assert sec["state"] == "unknown" and sec["last_completed_cycle_utc"] is None and sec["age_s"] is None and sec["late"] is False
    # 50 minutes old: within 3 x 20 min -> ok; 61 minutes: late (the live shape after two killed 45-minute cycles)
    _write_cycle(root, now_s, 50 * 60)
    assert status.harvest_cycle_section(root, now_s)["state"] == "ok"
    _write_cycle(root, now_s, 61 * 60, stopped="budget", budget_s=2400.0, revisions_deferred_budget=7)
    sec = status.harvest_cycle_section(root, now_s)
    assert sec["state"] == "late" and sec["late"] is True and sec["age_s"] >= 3660 and sec["stopped"] == "budget" and sec["revisions_deferred_budget"] == 7
    # the cadence is the operator's ([harvest] cadence_s in config.toml): at 100 s the row is late past 300 s
    with open(CFG.config_path(root), "w", encoding="utf-8") as fh:
        fh.write("[harvest]\ncadence_s = 100\n")
    _write_cycle(root, now_s, 400)
    sec = status.harvest_cycle_section(root, now_s)
    assert sec["cadence_s"] == 100 and sec["late_after_s"] == 300 and sec["state"] == "late"
    _write_cycle(root, now_s, 200)
    assert status.harvest_cycle_section(root, now_s)["state"] == "ok"
    # the default config renders the knob
    assert "cadence_s = 1200" in CFG.render_default(root)


def test_the_page_renders_the_harvester_row_red_when_late_and_plain_when_fresh(tmp_path):
    root = _setup(tmp_path)
    now_s = time.time()
    _write_cycle(root, now_s, 20, revisions_applied=1)
    ctx = status.status_json(root)
    assert ctx["harvest_cycle"]["state"] == "ok"
    html = _render(root, tmp_path, "fresh.html")
    assert 'id="harvester-cycle"' in html and 'class="note" id="harvester-cycle"' in html and "LATE" not in html
    assert "last completed cycle" in html and "1 revision directory / bundle(s)" in html
    _write_cycle(root, now_s, 4 * 3600, stopped="budget", budget_s=2400.0, revisions_deferred_budget=5, plain_files_deferred_budget=3)
    html = _render(root, tmp_path, "late.html")
    assert 'class="late" id="harvester-cycle"' in html and "LATE" in html and "older than 3600 s" in html and "TimeoutStartSec" in html
    assert "stopped on its <code>budget</code>" in html and "left 5 revision directories / bundle(s) and 3 plain file(s)" in html
    os.remove(os.path.join(root, "state", H.CYCLE_STATE_NAME))
    html = _render(root, tmp_path, "unknown.html")
    assert 'class="stale" id="harvester-cycle"' in html and "none recorded yet" in html


def test_a_completed_cycle_writes_the_state_the_row_reads_and_a_paused_one_does_not(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    assert H.read_cycle_state(root) is None
    monkeypatch.setattr(H, "free_bytes", lambda path: 1024 ** 3)      # a controlled reading: 1 GB free, whatever the host has (a container reports ~9.2e18)
    ing = H.ingest_once(root, workers=0, min_free=0, now=synth.t(100))
    assert not ing.paused and ing.free_bytes == 1024 ** 3
    state = H.read_cycle_state(root)
    assert state["last_completed_cycle_utc"] == ing.completed_utc and state["cycle_utc"] == synth.t(100) and state["files"] == ing.files
    # the free-space guard pausing the cycle (a 2 GB floor over the 1 GB reading) completes nothing: the stamp does not move
    paused = H.ingest_once(root, workers=0, min_free=2 * 1024 ** 3, now=synth.t(200))
    assert paused.paused and paused.free_bytes == 1024 ** 3 and paused.completed_utc is None and H.read_cycle_state(root) == state


def test_the_wal_bytes_render_beside_the_harvester_row(tmp_path):
    """The size of registry.sqlite-wal is on the page beside the last completed cycle, read from the
    file (0 when absent), so checkpoint starvation -- a WAL that only grows between completed cycles -- is visible without a shell."""
    root = _setup(tmp_path)
    # the pure reader: a root with a WAL file of a known size, a root without one, no root at all
    fake = str(tmp_path / "fake_root")
    os.makedirs(fake)
    with open(os.path.join(fake, "registry.sqlite-wal"), "wb") as fh:
        fh.write(b"\0" * 4096 * 3)
    assert status.wal_size_bytes(fake) == 12288 and status.wal_size_bytes(str(tmp_path / "absent")) == 0 and status.wal_size_bytes(None) == 0
    ctx = status.status_json(root)
    assert isinstance(ctx["wal_bytes"], int) and ctx["wal_bytes"] >= 0 and ctx["wal_gb"] == round(ctx["wal_bytes"] / 1e9, 2)
    html = _render(root, tmp_path)
    assert f"WAL: <code>{ctx['wal_bytes']}</code> bytes" in html and "registry.sqlite-wal" in html and "checkpoint starvation" in html
