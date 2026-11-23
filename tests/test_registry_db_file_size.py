"""A SQLite database's size is read race-free against a concurrent checkpoint (db.file_size / db.path_bytes and their callers).

A checkpoint deletes the ``-wal``/``-shm`` twins at any moment; ``os.path.exists`` followed by ``os.path.getsize`` raised
``FileNotFoundError`` when a twin went between the two calls (tests.test_registry_snapshot::
test_concurrent_ingest_during_build_does_not_block_or_tear failed once on Slurm on exactly that). Each test below deletes the file
right after the first ``stat`` of it returns -- the existence check of the old shape -- so the old code's second ``stat`` (the size
read) raises, and the fixed code, which stats each file once, does not."""
from __future__ import annotations

import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import replica as RP
from sisteer.registry import snapshot as SN


def _vanish_after_first_stat(monkeypatch, target: str) -> dict:
    """Patch ``os.stat`` so the first stat of ``target`` returns its real result and then deletes the file (a checkpoint landing between
    an existence check and a size read); every other stat is the real one. Returns a dict whose ``calls`` counts the stats of ``target``."""
    real_stat, real_remove = os.stat, os.remove
    seen = {"calls": 0}

    def stat(path, *args, **kwargs):
        res = real_stat(path, *args, **kwargs)
        if not isinstance(path, int) and os.fspath(path) == target:
            seen["calls"] += 1
            if seen["calls"] == 1:
                real_remove(target)
        return res

    monkeypatch.setattr(os, "stat", stat)
    return seen


def _write(path: str, n: int) -> None:
    with open(path, "wb") as f:
        f.write(b"x" * n)


def _db_with_twins(tmp_path, main=4096, wal=100, shm=32) -> str:
    db = str(tmp_path / "registry.sqlite")
    _write(db, main)
    if wal is not None:
        _write(db + "-wal", wal)
    if shm is not None:
        _write(db + "-shm", shm)
    return db


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_file_size_survives_a_side_file_deleted_between_check_and_read(tmp_path, monkeypatch, suffix):
    db = _db_with_twins(tmp_path)
    seen = _vanish_after_first_stat(monkeypatch, db + suffix)
    # the twin is counted at the size its one stat saw; the old exists-then-getsize pair raised FileNotFoundError here
    assert DB.file_size(db) == 4096 + 100 + 32
    assert seen["calls"] == 1 and not os.path.exists(db + suffix)


def test_file_size_counts_an_absent_side_file_as_zero(tmp_path, monkeypatch):
    db = _db_with_twins(tmp_path, wal=None, shm=None)
    assert DB.file_size(db) == 4096
    assert DB.file_size(str(tmp_path / "absent.sqlite")) == 0
    # a twin a checkpoint deleted before its stat: FileNotFoundError from the one stat counts 0
    _write(db + "-wal", 100)
    real_stat = os.stat

    def stat(path, *args, **kwargs):
        if not isinstance(path, int) and os.fspath(path) == db + "-wal":
            raise FileNotFoundError(2, "No such file or directory", db + "-wal")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", stat)
    assert DB.file_size(db) == 4096
    assert DB.path_bytes(db + "-wal") == 0


def test_file_size_of_a_live_wal_database_is_the_sum_of_its_files(tmp_path):
    db = str(tmp_path / "live.sqlite")
    conn = sqlite3.connect(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t (x BLOB)")
        conn.execute("INSERT INTO t VALUES (?)", (b"y" * 50_000,))
        conn.commit()
        assert os.path.getsize(db + "-wal") > 0
        want = sum(os.path.getsize(db + s) for s in ("", "-wal", "-shm") if os.path.exists(db + s))
        assert DB.file_size(db) == want
    finally:
        conn.close()


@pytest.mark.parametrize("building", [False, True])
def test_footprint_sampler_survives_a_side_file_deleted_between_check_and_read(tmp_path, monkeypatch, building):
    db = str(tmp_path / "registry.sqlite")
    base = db + DB.REBUILD_TMP_SUFFIX if building else db
    _db_with_twins(tmp_path, main=4096, wal=100, shm=32)
    if building:
        for s in ("", "-wal", "-shm"):
            os.rename(db + s, base + s)
    watcher = SN.FootprintWatcher([db])
    _vanish_after_first_stat(monkeypatch, base + "-wal")
    watcher.sample()
    assert watcher.peak_breakdown == {db: 4096 + 100 + 32}


def test_stage_input_survives_an_empty_wal_deleted_between_check_and_read(tmp_path, monkeypatch):
    db = _db_with_twins(tmp_path, wal=0, shm=None)
    _vanish_after_first_stat(monkeypatch, db + "-wal")
    out = RP.stage_input(db, str(tmp_path / "work"), log=lambda msg: None)
    assert out["wal_bytes"] == 0 and out["bytes"] == 4096


def test_copy_db_hashed_survives_an_empty_wal_deleted_between_check_and_read(tmp_path, monkeypatch):
    db = _db_with_twins(tmp_path, wal=0, shm=None)
    _vanish_after_first_stat(monkeypatch, db + "-wal")
    out = RP._copy_db_hashed(db, str(tmp_path / "out" / "copy.sqlite"))
    assert out["bytes"] == 4096


def test_a_wal_holding_bytes_when_measured_is_still_refused(tmp_path, monkeypatch):
    db = _db_with_twins(tmp_path, wal=100, shm=None)
    _vanish_after_first_stat(monkeypatch, db + "-wal")
    with pytest.raises(RP.RefoldInputRefused, match="holds 100 bytes"):
        RP.stage_input(db, str(tmp_path / "work"), log=lambda msg: None)
