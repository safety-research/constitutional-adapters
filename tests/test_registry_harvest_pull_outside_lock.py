"""The cluster pull backstops run BEFORE ``harvester.lock`` is taken (2026-09-30).

Measured on the live registry: ``ingest_once`` ran ``pull_clusters`` (an rsync per cluster, ``--timeout=60`` per transfer, a 600-s
subprocess bound) INSIDE the write lock, after the inbox merge -- so the lock covered the network and a pulled file folded one cycle
later. ``ingest_under_lock(pull=True)`` now pulls first, with the lock free (the pull writes ``inbox/<host>/`` only and reads no
database), then takes the lock and runs the cycle, which merges and folds what the pull landed in the SAME cycle. Pinned here: the
lock is free while the pull runs; the pulled file folds in the same report; the free-space floor still gates the pull (read-only, before
the lock); ``ingest_once`` no longer accepts ``pull`` and never pulls (``pending.ingest_now`` included); the CLI pulls exactly once with
``--pull``, stamps the whole command, and refuses by name with exit 4 when another writer holds the lock past ``--lock-timeout``.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _land_in_inbox(root, host="cluster_a"):
    """What a pull lands: one record file under ``inbox/<host>/<campaign>/<run_dir>/``."""
    dims = synth.seed_dimensions()
    path = os.path.join(root, "inbox", host, "dims", "seed", "000001-dims.jsonl")
    R.write_records(path, dims)
    return path, len(dims)


def test_the_pull_runs_with_the_lock_free_and_what_it_lands_folds_in_the_same_cycle(tmp_path, monkeypatch):
    root = _root(tmp_path)
    calls = []

    def stub(r, config=None):
        assert r == root
        with DB.write_lock(root, timeout_s=0):          # try once: the lock is FREE while the pull runs (LockHeld otherwise)
            pass
        calls.append(_land_in_inbox(root))
        return ["cluster_a"]
    monkeypatch.setattr(H, "pull_clusters", stub)
    rep = H.ingest_under_lock(root, workers=0, min_free=0, pull=True)
    assert len(calls) == 1
    path, n = calls[0]
    # merged by this cycle's inbox scan and folded by its fold: the same report counts the records (plus the arrival facts)
    assert rep.ok - rep.arrival_records == n and rep.conflicts == 0 and not rep.paused
    assert not os.path.exists(path) and os.path.exists(os.path.join(root, "records", "dims", "seed", "000001-dims.jsonl"))
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT status FROM record_delivery WHERE file_uri = ?", ("records/dims/seed/000001-dims.jsonl",)).fetchone()[0] == "ok"
    # a second cycle with the pull landing nothing new changes nothing
    monkeypatch.setattr(H, "pull_clusters", lambda r, config=None: ["cluster_a"])
    rep2 = H.ingest_under_lock(root, workers=0, min_free=0, pull=True)
    assert rep2.ok == 0 and rep2.files == 0


def test_the_free_space_floor_gates_the_pull_before_the_lock(tmp_path, monkeypatch):
    """A paused registry (below the floor) must not keep rsyncing into the inbox; the read-only check before the lock uses the same
    reading the cycle's guard uses, and the guard (inside the lock) is still what writes ``state/paused.json``."""
    root = _root(tmp_path)
    monkeypatch.setattr(H, "free_bytes", lambda path: 1024 ** 3)                          # 1 GB free
    monkeypatch.setattr(H, "pull_clusters", lambda r, config=None: pytest.fail("pulled below the free-space floor"))
    rep = H.ingest_under_lock(root, workers=0, min_free=2 * 1024 ** 3, pull=True)         # a 2 GB floor
    assert rep.paused and rep.ok == 0 and H.paused(root) is not None
    # above the floor the pull runs again and the flag clears
    pulled = []
    monkeypatch.setattr(H, "pull_clusters", lambda r, config=None: pulled.append(r) or [])
    rep2 = H.ingest_under_lock(root, workers=0, min_free=0, pull=True)
    assert pulled == [root] and not rep2.paused and H.paused(root) is None


def test_ingest_once_never_pulls_and_no_longer_takes_a_pull_argument(tmp_path, monkeypatch):
    root = _root(tmp_path)
    monkeypatch.setattr(H, "pull_clusters", lambda r, config=None: pytest.fail("the cycle under the lock must not pull"))
    H.ingest_once(root, workers=0, min_free=0)
    P.ingest_now(root)                                       # the explicit fold under the lock: no pull either
    H.ingest_under_lock(root, workers=0, min_free=0)         # without pull=True: no pull
    with pytest.raises(TypeError):
        H.ingest_once(root, workers=0, min_free=0, pull=True)


def test_the_cli_pulls_exactly_once_with_the_lock_free_and_stamps_the_whole_command(tmp_path, monkeypatch):
    """``registry ingest --pull``: one pull, the lock free while it runs; ``state/last_run/ingest.json`` still records the whole
    command (argv with ``--pull``, rc 0) -- what ``mirror_status.py`` reads as "the pull covered every cluster"."""
    root = _root(tmp_path)
    monkeypatch.setattr(H, "free_bytes", lambda path: 100 * H.GB)                       # the floor reads a controlled figure, never the host's disk
    seen = []

    def stub(r, config=None):
        with DB.write_lock(root, timeout_s=0):
            pass
        seen.append(r)
        return []
    monkeypatch.setattr(H, "pull_clusters", stub)
    assert cli.main(["--root", root, "ingest", "--workers", "0", "--pull"]) == 0
    assert seen == [root]
    stamp = json.load(open(os.path.join(root, "state", "last_run", "ingest.json")))
    assert stamp["rc"] == 0 and "--pull" in stamp["argv"] and "ingest" in stamp["argv"]
    assert cli.main(["--root", root, "ingest", "--workers", "0"]) == 0                    # without --pull: no pull
    assert seen == [root]


def test_the_cli_refuses_by_name_with_exit_4_when_another_writer_holds_the_lock(tmp_path, monkeypatch, capsys):
    """A cycle that cannot take the lock inside ``--lock-timeout`` refuses (exit 4, the ``migrate`` convention) instead of dying on a
    traceback (the 2026-09-30 journal: ``status=1/FAILURE`` twice in a row on an uncaught LockHeld); the timer re-fires."""
    root = _root(tmp_path)
    monkeypatch.setattr(H, "free_bytes", lambda path: 100 * H.GB)
    with DB.write_lock(root):
        assert cli.main(["--root", root, "ingest", "--workers", "0", "--lock-timeout", "0.2"]) == 4
    err = capsys.readouterr().err
    assert "lock is held" in err and "harvester.lock" in err and "nothing was folded" in err
    assert cli.main(["--root", root, "ingest", "--workers", "0", "--lock-timeout", "0.2"]) == 0      # the lock is free again
