"""The compaction swap (``sisteer.registry.compact``, ``scripts/registry/landing/compact_swap.sh``) and ``db.open_or_create``'s
refusal to create a database in a root that holds harvest state.

The swap is a state machine. A failure is injected after EVERY step -- as an exception in-process and as SIGKILL of a subprocess -- and
after each: the live path exists and opens to a valid database (old or new, never absent); ``--validate-live`` refuses while the swap is
in flight; ``--recover`` completes it forward (the rename had happened) or rolls it back consistently (it had not); ``--validate-live``
then passes. The restart gate in ``compact_swap.sh`` is rehearsed with ``systemctl`` and ``fuser`` shims: units come back only after the
gate passes, and stay stopped (``swap_blocked.json``, the maintenance flag kept) when it refuses.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import traceback

import pytest

import sisteer
from sisteer.registry import cli
from sisteer.registry import compact as C
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import schema_def as S
from sisteer.registry import synth

SRC = os.path.dirname(os.path.dirname(os.path.abspath(sisteer.__file__)))
REPO = os.path.dirname(SRC)
SWAP_SH = os.path.join(REPO, "scripts", "registry", "landing", "compact_swap.sh")
BEFORE_RENAME = C.STEPS[:C.STEPS.index("rename")]


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A small registry built by the harvester (runs, items, the cycle's state files), copied per test."""
    root = str(tmp_path_factory.mktemp("compact-template") / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=2, n_items=5)
    H.ingest_once(root, workers=0, min_free=0)
    assert os.path.exists(os.path.join(root, "state", "harvest_cycle.json"))
    assert not C.companions_present(DB.db_path(root))
    return root


@pytest.fixture
def reg(template, tmp_path):
    root = str(tmp_path / "reg")
    shutil.copytree(template, root)
    return root


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _cli(*argv):
    return subprocess.run([sys.executable, "-m", "sisteer.registry.compact", *argv], capture_output=True, text=True, env=_env())


def _kill_at(root, attic, step):
    """``compact --swap`` in a subprocess that SIGKILLs itself at ``step``."""
    code = ("import os, signal, sys\n"
            "from sisteer.registry import compact as C\n"
            f"def hook(name):\n    if name == {step!r}:\n        os.kill(os.getpid(), signal.SIGKILL)\n"
            f"sys.exit(C.main(['--swap', '--root', {root!r}, '--attic', {attic!r}, '--jobs', '2'], on_step=hook))\n")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env())


class Injected(RuntimeError):
    pass


def _raise_at(step):
    def hook(name):
        if name == step:
            raise Injected(step)
    return hook


def _assert_live_valid(root, pre_tables):
    """The live path exists, has the code's shape, and holds the pre identity's rows (old and new files both do)."""
    live = DB.db_path(root)
    assert os.path.isfile(live), "the live path is absent"
    assert C.shape_lines(live) == []
    assert C.table_counts(live, 2) == pre_tables


def _delete_one_row(path):
    """A writer's change to ``path``: one row of a populated table deleted (plain sqlite3, no registry function needed)."""
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        for t in ("record_delivery", "observation", "item_result", "record_ingest"):
            pk = S.TABLE_BY_NAME[t].pk
            if conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]:
                cols = ", ".join(pk)
                conn.execute(f"DELETE FROM {t} WHERE ({cols}) IN (SELECT {cols} FROM {t} LIMIT 1)")
                return t
        raise AssertionError("no populated table to change")
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# the swap
# --------------------------------------------------------------------------------------------

def test_swap_installs_a_verified_compacted_file_and_keeps_the_old_inode_in_the_attic(reg, tmp_path):
    live = DB.db_path(reg)
    old = C.inode(live)
    attic = str(tmp_path / "attic")
    st = C.swap(reg, attic, jobs=2)
    assert st["phase"] == "validated" and [h["phase"] for h in st["history"]] == ["preconditions", "vacuumed", "verified", "linked", "renamed", "validated"]
    assert C.inode(live) == (st["new"]["dev"], st["new"]["ino"]) != old
    assert C.inode(os.path.join(attic, DB.DB_NAME)) == old                    # the attic shares the old inode
    assert not os.path.exists(st["output"]) and not C.companions_present(live)
    assert C.wal_header(live) == (2, 2)                                       # VACUUM INTO writes rollback mode; the swap made it WAL
    with DB.reading(live) as c:
        assert int(c.execute("PRAGMA freelist_count").fetchone()[0]) == 0
    assert C.identity_diff(st["pre"], C.identity(live, 2)) == []
    on_disk = json.load(open(os.path.join(attic, C.STATE_NAME)))
    assert on_disk["phase"] == "validated" and on_disk["pre"]["ino"] == old[1]
    assert C.validate_live(reg, attic)["ok"] is True
    # the writers' open works on it, and the harvester folds into it
    DB.open_or_create(live).close()
    H.ingest_once(reg, workers=0, min_free=0)


def test_the_swap_names_its_output_by_the_clock_and_syncs_directories_with_the_db_helper():
    """The output is always ``registry.sqlite.compact-<utc>`` (no caller-chosen stamp), and one directory-sync implementation
    serves the swap: ``compact.fsync_dir`` is ``db.fsync_dir``, with no second file-sync helper beside ``db.fsync_file_and_dir``."""
    import inspect
    assert "stamp" not in inspect.signature(C.swap).parameters
    assert C.fsync_dir is DB.fsync_dir
    assert not hasattr(C, "fsync_file")


@pytest.mark.parametrize("mode", ["exception", "sigkill"])
@pytest.mark.parametrize("step", C.STEPS)
def test_a_failure_after_every_step_leaves_a_valid_live_path_and_recover_completes_or_rolls_back(reg, tmp_path, step, mode):
    live = DB.db_path(reg)
    old = C.inode(live)
    pre_tables = C.table_counts(live, 2)
    attic = str(tmp_path / "attic")
    if mode == "exception":
        with pytest.raises(Injected):
            C.swap(reg, attic, jobs=2, on_step=_raise_at(step))
    else:
        r = _kill_at(reg, attic, step)
        assert r.returncode == -signal.SIGKILL, (r.returncode, r.stderr[-2000:])
    st = C.read_state(attic)
    assert st is not None and st["phase"] not in C.TERMINAL               # every hook fires before its phase is recorded
    _assert_live_valid(reg, pre_tables)                                     # old or new, never absent
    # the restart gate refuses a swap in flight
    gate = _cli("--validate-live", "--root", reg, "--attic", attic)
    assert gate.returncode == C.ValidationFailed.exit_code, gate.stdout + gate.stderr
    assert "--recover" in json.loads(gate.stdout)["problems"][0]
    rec = _cli("--recover", "--attic", attic)
    assert rec.returncode == 0, rec.stdout + rec.stderr
    st = C.read_state(attic)
    if step in BEFORE_RENAME:
        assert st["phase"] == "rolled_back"
        assert C.inode(live) == old
        assert not os.path.exists(os.path.join(attic, DB.DB_NAME))         # the duplicate name of the live file is gone
    else:
        assert st["phase"] == "validated"
        assert C.inode(live) == (st["new"]["dev"], st["new"]["ino"]) != old
        assert C.inode(os.path.join(attic, DB.DB_NAME)) == old
    assert not any(os.path.exists(st["output"] + s) for s in ("",) + C.COMPANIONS)
    _assert_live_valid(reg, pre_tables)
    gate = _cli("--validate-live", "--root", reg, "--attic", attic)
    assert gate.returncode == 0, gate.stdout + gate.stderr
    # recovery is idempotent, and a finished swap is never re-decided
    again = _cli("--recover", "--attic", attic)
    assert again.returncode == 0 and C.read_state(attic)["phase"] == st["phase"]
    DB.open_or_create(live).close()


def test_recover_rolls_back_a_compacted_file_that_fails_validation(reg, tmp_path):
    live = DB.db_path(reg)
    old = C.inode(live)
    pre_tables = C.table_counts(live, 2)
    attic = str(tmp_path / "attic")
    with pytest.raises(Injected):
        C.swap(reg, attic, jobs=2, on_step=_raise_at("rename"))
    assert C.inode(live) != old
    _delete_one_row(live)                                                  # the installed file no longer has the pre identity
    st = C.recover(attic, jobs=2)
    assert st["phase"] == "rolled_back" and C.inode(live) == old
    assert any("renamed back" in n for n in st["recovery"]["notes"])
    _assert_live_valid(reg, pre_tables)
    assert C.validate_live(reg, attic)["ok"] is True


def test_recover_refuses_an_unknown_file_at_the_live_path_and_changes_nothing(reg, tmp_path):
    live = DB.db_path(reg)
    attic = str(tmp_path / "attic")
    with pytest.raises(Injected):
        C.swap(reg, attic, jobs=2, on_step=_raise_at("link"))
    tmp = live + ".foreign"
    shutil.copy2(live, tmp)
    os.rename(tmp, live)                                                    # a third inode at the live path
    foreign = C.inode(live)
    out = C.read_state(attic)["output"]
    r = _cli("--recover", "--attic", attic)
    assert r.returncode == C.RecoveryRefused.exit_code and "neither the old file" in r.stderr
    assert C.inode(live) == foreign and os.path.exists(out) and os.path.exists(os.path.join(attic, DB.DB_NAME))
    assert C.read_state(attic)["phase"] == "verified"
    assert _cli("--validate-live", "--root", reg, "--attic", attic).returncode == C.ValidationFailed.exit_code


def test_recover_relinks_an_absent_live_path_from_the_attic(reg, tmp_path):
    live = DB.db_path(reg)
    old = C.inode(live)
    pre_tables = C.table_counts(live, 2)
    attic = str(tmp_path / "attic")
    with pytest.raises(Injected):
        C.swap(reg, attic, jobs=2, on_step=_raise_at("link_fsync"))
    os.unlink(live)                                                         # never by construction; the attic link keeps the inode
    st = C.recover(attic, jobs=2)
    assert st["phase"] == "rolled_back" and C.inode(live) == old
    assert any("re-linked" in n for n in st["recovery"]["notes"])
    _assert_live_valid(reg, pre_tables)


def test_recover_never_redecides_a_finished_swap_after_the_writers_ran(reg, tmp_path):
    live = DB.db_path(reg)
    attic = str(tmp_path / "attic")
    st = C.swap(reg, attic, jobs=2)
    new = C.inode(live)
    _delete_one_row(live)                                                   # a writer ran on the installed file
    counts = C.table_counts(live, 2)
    st2 = C.recover(attic, jobs=2)
    assert st2["phase"] == "validated" and "already validated" in st2["note"]
    assert C.inode(live) == new and C.table_counts(live, 2) == counts      # its writes kept
    assert st2["history"] == st["history"]


def test_the_preconditions_refuse_a_holder_the_lock_and_low_free_space_with_nothing_changed(reg, tmp_path, monkeypatch):
    live = DB.db_path(reg)
    old = C.inode(live)

    def nothing_changed(attic):
        assert C.inode(live) == old and not os.path.exists(os.path.join(attic, C.STATE_NAME))
        assert not [f for f in os.listdir(reg) if C.COMPACT_INFIX in f]
    # another process holds the database open
    holder = subprocess.Popen([sys.executable, "-c", "import sqlite3, sys, time\nc = sqlite3.connect(sys.argv[1])\n"
                               "c.execute('SELECT COUNT(*) FROM schema_migration').fetchone()\nprint('ready', flush=True)\ntime.sleep(60)\n", live],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "ready"
        with pytest.raises(C.PreconditionFailed, match=f"'pid': {holder.pid}"):
            C.swap(reg, str(tmp_path / "a1"), jobs=2)
        nothing_changed(str(tmp_path / "a1"))
    finally:
        holder.kill()
        holder.wait()
    # the registry's write lock is held (a harvester cycle): exit 2 by name
    with DB.write_lock(reg):
        r = _cli("--swap", "--root", reg, "--attic", str(tmp_path / "a2"), "--lock-timeout", "0")
    assert r.returncode == C.PreconditionFailed.exit_code and "write lock is held" in r.stderr
    nothing_changed(str(tmp_path / "a2"))
    # too little free space for 1.5x the used pages
    monkeypatch.setattr(DB, "free_bytes", lambda path: 1024)
    with pytest.raises(C.PreconditionFailed, match="the swap needs"):
        C.swap(reg, str(tmp_path / "a3"), jobs=2)
    nothing_changed(str(tmp_path / "a3"))
    # an attic already used by a swap
    monkeypatch.undo()
    C.swap(reg, str(tmp_path / "a4"), jobs=2)
    with pytest.raises(C.PreconditionFailed, match="one attic per swap"):
        C.swap(reg, str(tmp_path / "a4"), jobs=2)


def test_vacuum_into_stops_at_the_free_space_floor_and_removes_the_partial_output(tmp_path, monkeypatch):
    src = str(tmp_path / "big.sqlite")
    c = sqlite3.connect(src, isolation_level=None)
    c.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)")
    c.execute("BEGIN")
    c.executemany("INSERT INTO t (b) VALUES (?)", (("x" * 400,) for _ in range(150000)))
    c.execute("COMMIT")
    c.close()
    out = str(tmp_path / "out.sqlite")
    monkeypatch.setattr(C, "FLOOR_POLL_S", 0.001)
    monkeypatch.setattr(DB, "free_bytes", lambda path: 0)
    with pytest.raises(C.PreconditionFailed, match="free-space floor"):
        C.vacuum_into(src, out, floor_bytes=1)
    assert not os.path.exists(out)
    monkeypatch.undo()
    open(out, "w").close()                                                 # SQLite would VACUUM INTO an empty file: refused instead
    with pytest.raises(C.PreconditionFailed, match="exists"):
        C.vacuum_into(src, out)


def test_vacuum_into_cli_makes_a_verified_wal_copy_with_the_sources_identity(reg, tmp_path):
    live = DB.db_path(reg)
    out = str(tmp_path / "copy.sqlite")
    r = _cli("--vacuum-into", out, "--db", live, "--compare-source", "--jobs", "2", "--json", str(tmp_path / "v.json"))
    assert r.returncode == 0, r.stderr
    doc = json.load(open(tmp_path / "v.json"))
    assert doc["result"] == "VACUUMED" and C.identity_diff(doc["source_identity"], doc["output_identity"]) == []
    assert C.wal_header(out) == (2, 2) and not C.companions_present(out)
    again = _cli("--vacuum-into", out, "--db", live)
    assert again.returncode == C.PreconditionFailed.exit_code


def test_validate_live_without_a_swap_checks_the_installed_file(reg, tmp_path):
    assert C.validate_live(reg)["ok"] is True
    assert C.validate_live(reg, str(tmp_path / "no-such-attic"))["ok"] is True
    away = DB.db_path(reg) + ".away"
    os.rename(DB.db_path(reg), away)
    res = C.validate_live(reg)
    assert res["ok"] is False and "absent" in res["problems"][0]
    os.rename(away, DB.db_path(reg))
    conn = sqlite3.connect(DB.db_path(reg), isolation_level=None)
    conn.execute("DROP VIEW v_paper_long")
    conn.close()
    res = C.validate_live(reg)
    assert res["ok"] is False and any("v_paper_long" in p for p in res["problems"])


def test_rollback_after_a_validated_swap_restores_the_old_file_and_keeps_the_new_one(reg, tmp_path):
    live = DB.db_path(reg)
    old = C.inode(live)
    attic = str(tmp_path / "attic")
    st = C.swap(reg, attic, jobs=2)
    new = C.inode(live)
    r = _cli("--rollback", "--attic", attic)
    assert r.returncode == 0, r.stderr
    assert C.inode(live) == old and C.inode(os.path.join(attic, DB.DB_NAME + ".post-swap")) == new
    st2 = C.read_state(attic)
    assert st2["phase"] == "rolled_back" and C.shape_lines(live) == []
    assert [h["phase"] for h in st2["history"]][-2:] == ["rolling_back", "rolled_back"]       # recorded before any name moved
    assert st2["rollback"]["outcome"] == "rolled_back"
    assert not os.path.exists(os.path.join(attic, DB.DB_NAME))                                 # no second name of the live file
    assert C.table_counts(live, 2) == st["pre"]["tables"]
    gate = _cli("--validate-live", "--root", reg, "--attic", attic)                            # the restart gate passes on the old file
    assert gate.returncode == 0, gate.stdout + gate.stderr
    assert json.loads(gate.stdout)["expected"] == "the old file (rolled back)"


# --------------------------------------------------------------------------------------------
# an interrupted rollback: recorded before any name moves, finished (or undone) by --recover
# --------------------------------------------------------------------------------------------

KEEP = DB.DB_NAME + C.KEEP_SUFFIX


def _kill_rollback_at(attic, step):
    """``compact --rollback`` in a subprocess that SIGKILLs itself at ``step``."""
    code = ("import os, signal, sys\n"
            "from sisteer.registry import compact as C\n"
            f"def hook(name):\n    if name == {step!r}:\n        os.kill(os.getpid(), signal.SIGKILL)\n"
            f"sys.exit(C.main(['--rollback', '--attic', {attic!r}, '--jobs', '2'], on_step=hook))\n")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=_env())


def _swapped_then_written(reg, tmp_path):
    """A validated swap, then a writer's change to the installed file (the rollback comes later, writers stopped again): the
    compacted file no longer has the pre identity, the old one in the attic still does."""
    live = DB.db_path(reg)
    old = C.inode(live)
    attic = str(tmp_path / "attic")
    st = C.swap(reg, attic, jobs=2)
    new = C.inode(live)
    _delete_one_row(live)
    assert C.table_counts(live, 2) != st["pre"]["tables"]
    return live, attic, st, old, new


def _interrupt_rollback(attic, step, mode):
    if mode == "exception":
        with pytest.raises(Injected):
            C.rollback(attic, jobs=2, on_step=_raise_at(step))
    else:
        r = _kill_rollback_at(attic, step)
        assert r.returncode == -signal.SIGKILL, (r.returncode, r.stderr[-2000:])


def _assert_in_flight(reg, attic, old, new):
    """A rollback stopped mid-way: recorded as such, the live path a valid database (old or new, never absent), the restart gate and a
    second --rollback both refusing with --recover named."""
    live = DB.db_path(reg)
    st = C.read_state(attic)
    assert st["phase"] == "rolling_back"
    assert st["rollback"]["old"] == list(old) and st["rollback"]["new"] == list(new) and st["rollback"]["keep"].endswith(KEEP)
    assert C.inode(live) in (old, new) and C.shape_lines(live) == []
    gate = _cli("--validate-live", "--root", reg, "--attic", attic)
    assert gate.returncode == C.ValidationFailed.exit_code, gate.stdout + gate.stderr
    assert "rollback stopped mid-way" in json.loads(gate.stdout)["problems"][0] and "--recover" in gate.stdout
    again = _cli("--rollback", "--attic", attic)
    assert again.returncode == C.RecoveryRefused.exit_code and "--recover" in again.stderr, again.stderr


@pytest.mark.parametrize("mode", ["exception", "sigkill"])
@pytest.mark.parametrize("step", C.ROLLBACK_STEPS)
def test_a_rollback_interrupted_after_every_step_is_finished_forward_by_recover(reg, tmp_path, step, mode):
    """The rollback's link (the compacted file kept as .post-swap) and its rename (the old file over the live path) are each a
    crash boundary; whichever was reached, --recover finishes the rollback from the inodes on disk and the restart gate then passes on
    the old file with the pre identity -- though the writers changed the compacted file after the swap."""
    live, attic, st, old, new = _swapped_then_written(reg, tmp_path)
    _interrupt_rollback(attic, step, mode)
    _assert_in_flight(reg, attic, old, new)
    rec = _cli("--recover", "--attic", attic)
    assert rec.returncode == 0, rec.stdout + rec.stderr
    st2 = C.read_state(attic)
    assert st2["phase"] == "rolled_back" and st2["rollback"]["outcome"] == "rolled_back"
    assert C.inode(live) == old                                                      # the old file is live
    assert C.inode(os.path.join(attic, KEEP)) == new                                 # the compacted one kept, by its one other name
    assert not os.path.exists(os.path.join(attic, DB.DB_NAME))                       # no second name of the live file
    assert not C.companions_present(live)
    _assert_live_valid(reg, st["pre"]["tables"])
    gate = _cli("--validate-live", "--root", reg, "--attic", attic)
    assert gate.returncode == 0, gate.stdout + gate.stderr
    again = _cli("--recover", "--attic", attic)                                      # idempotent; a finished rollback is not re-decided
    assert again.returncode == 0 and C.read_state(attic)["phase"] == "rolled_back" and C.inode(live) == old
    DB.open_or_create(live).close()


def _damage_old(attic):
    """The old file in the attic no longer has the pre identity (one row gone through its attic name): it keeps the code's shape and
    a WAL header, so the rollback starts, and fails only at the old file's validation."""
    attic_db = os.path.join(attic, DB.DB_NAME)
    _delete_one_row(attic_db)
    assert not C.companions_present(attic_db) and C.wal_header(attic_db) == (2, 2)


def _assert_kept_new(reg, attic, old, new):
    live = DB.db_path(reg)
    st = C.read_state(attic)
    assert st["phase"] == "validated" and st["rollback"]["outcome"] == "kept_new" and st["rollback"]["old_failed"]["problems"]
    assert C.inode(live) == new                                                      # the compacted file is live
    assert C.inode(os.path.join(attic, DB.DB_NAME)) == old                           # the old file kept under its attic name
    assert not os.path.exists(os.path.join(attic, KEEP))                             # no second name of the live file
    assert not C.companions_present(live)                                            # (before any read-only open makes them)
    assert C.shape_lines(live) == []
    # the gate validates the compacted file against the identity recorded when the rollback began (the writers' change included),
    # not against the pre identity, which no longer describes it
    assert C.table_counts(live, 2) == st["rollback"]["new_identity"]["tables"] != st["pre"]["tables"]
    gate = _cli("--validate-live", "--root", reg, "--attic", attic)
    assert gate.returncode == 0, gate.stdout + gate.stderr
    assert json.loads(gate.stdout)["expected"] == "the compacted file (its rollback undone)"
    assert _cli("--recover", "--attic", attic).returncode == 0 and C.read_state(attic)["phase"] == "validated"


def test_a_rollback_whose_old_file_does_not_validate_is_undone_and_the_compacted_file_stays_live(reg, tmp_path):
    live, attic, st, old, new = _swapped_then_written(reg, tmp_path)
    _damage_old(attic)
    r = _cli("--rollback", "--attic", attic)
    assert r.returncode == C.ValidationFailed.exit_code and "the rollback was undone" in r.stderr, r.stderr
    _assert_kept_new(reg, attic, old, new)
    phases = [h["phase"] for h in C.read_state(attic)["history"]]
    assert phases[-3:] == ["rolling_back", "rolling_back", "validated"]            # the verdict recorded before the undo moved anything


@pytest.mark.parametrize("step", C.ROLLBACK_UNDO_STEPS + ("rollback_rename_fsync",))
def test_an_interrupted_undo_of_a_rollback_is_resumed_by_recover(reg, tmp_path, step):
    """Killed while undoing (or before the old file's validation decided to undo): --recover reaches the same consistent end -- the
    compacted file live and validated, the old file under its attic name, no .post-swap name left."""
    live, attic, st, old, new = _swapped_then_written(reg, tmp_path)
    _damage_old(attic)
    r = _kill_rollback_at(attic, step)
    assert r.returncode == -signal.SIGKILL, (r.returncode, r.stderr[-2000:])
    _assert_in_flight(reg, attic, old, new)
    if step != "rollback_rename_fsync":
        assert C.read_state(attic)["rollback"]["old_failed"]                         # the verdict was on disk before the undo began
    rec = _cli("--recover", "--attic", attic)
    assert rec.returncode == 0, rec.stdout + rec.stderr
    _assert_kept_new(reg, attic, old, new)


def test_a_rollback_refuses_an_old_file_without_the_codes_shape_and_changes_nothing(reg, tmp_path):
    live, attic, st, old, new = _swapped_then_written(reg, tmp_path)
    conn = sqlite3.connect(os.path.join(attic, DB.DB_NAME), isolation_level=None)  # e.g. a migration the live file got since the swap
    conn.execute("DROP VIEW v_paper_long")
    conn.close()
    before = C.read_state(attic)
    r = _cli("--rollback", "--attic", attic)
    assert r.returncode == C.PreconditionFailed.exit_code and "code's shape" in r.stderr, r.stderr
    assert C.read_state(attic) == before                                            # not even rolling_back recorded
    assert C.inode(live) == new and C.inode(os.path.join(attic, DB.DB_NAME)) == old
    assert not os.path.exists(os.path.join(attic, KEEP))


def test_validate_live_follows_the_rollback_from_refused_in_flight_to_passing_on_the_old_file(reg, tmp_path):
    """The restart gate in each state of a rollback: passing on the compacted file before it, refusing while it is in flight (its
    state names neither terminal file), passing on the old file once --recover finished it."""
    live, attic, st, old, new = _swapped_then_written(reg, tmp_path)
    # the compacted file was written since the swap: the gate's pre identity no longer matches, as before this change
    assert C.validate_live(reg, attic)["ok"] is False
    with pytest.raises(Injected):
        C.rollback(attic, jobs=2, on_step=_raise_at("rollback_rename"))
    res = C.validate_live(reg, attic)
    assert res["ok"] is False and res["phase"] == "rolling_back" and C.inode(live) == old
    C.recover(attic, jobs=2)
    res = C.validate_live(reg, attic)
    assert res["ok"] is True and res["phase"] == "rolled_back" and res["inode"] == list(old)


# --------------------------------------------------------------------------------------------
# db.open_or_create: no fresh database in a root that holds harvest state
# --------------------------------------------------------------------------------------------

def test_open_or_create_refuses_to_create_a_database_in_a_root_with_harvest_state(reg, tmp_path):
    live = DB.db_path(reg)
    os.rename(live, live + ".away")
    with pytest.raises(DB.SchemaDrift) as ei:
        DB.open_or_create(live)
    exc = ei.value
    assert type(exc) is DB.SchemaDrift                                     # a plain SchemaDrift: every writer's refusal path handles it
    assert "harvest state" in str(exc) and "--recover" in str(exc) and exc.have == 0 and exc.lines
    assert not os.path.exists(live)                                        # nothing created
    # the grade daemon holds a claim refused this way as TRANSIENT, by the terminal line's qualified name
    last = traceback.format_exception_only(type(exc), exc)[-1]
    assert last.startswith("sisteer.registry.db.SchemaDrift:") or last.startswith("SchemaDrift:")
    assert f"{type(exc).__module__}.{type(exc).__qualname__}" == "sisteer.registry.db.SchemaDrift"
    # the harvester's own open refuses the same way, before anything is written
    with pytest.raises(DB.SchemaDrift):
        H.ingest_once(reg, workers=0, min_free=0)
    assert not os.path.exists(live)
    # a schema-less file at the path is not migrated into a fresh registry either
    sqlite3.connect(live).close()
    with pytest.raises(DB.SchemaDrift, match="holds no migrated schema"):
        DB.open_or_create(live)
    os.unlink(live)
    # an explicit creation still works
    DB.open_or_create(live, allow_create=True).close()
    assert os.path.exists(live)
    # a root with no harvest state creates as before (the first cycle of a new registry)
    fresh = str(tmp_path / "fresh")
    os.makedirs(os.path.join(fresh, "state"))
    DB.open_or_create(DB.db_path(fresh)).close()
    assert os.path.exists(DB.db_path(fresh))


def test_registry_migrate_never_creates_a_database(reg, tmp_path, capsys):
    """``registry migrate`` upgrades an existing database and never creates one: an absent path is refused before any connection (a
    live file gone mid-swap must not become a fresh, empty registry), and so is a schema-less file in a root with harvest state."""
    live = DB.db_path(reg)
    os.rename(live, live + ".away")
    assert cli.main(["--root", reg, "migrate"]) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err and "does not exist" in err and "--recover" in err
    assert not os.path.exists(live) and not os.path.exists(live + "-wal")
    # a --db path outside any root: refused the same way, and its directory is not even created
    nowhere = str(tmp_path / "nowhere" / "registry.sqlite")
    assert cli.main(["migrate", "--db", nowhere]) == 1
    assert "does not exist" in capsys.readouterr().err and not os.path.exists(os.path.dirname(nowhere))
    # a schema-less file in a root with harvest state: refused by the writers' rule, the file left exactly as it was
    sqlite3.connect(live).close()
    assert os.path.getsize(live) == 0
    assert cli.main(["--root", reg, "migrate"]) == 1
    assert "holds no migrated schema" in capsys.readouterr().err
    assert os.path.getsize(live) == 0 and not os.path.exists(live + "-wal")
    # the real file back: the migrate is the no-op it always was
    os.unlink(live)
    os.rename(live + ".away", live)
    assert cli.main(["--root", reg, "migrate"]) == 0
    assert "applied nothing" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# compact_swap.sh: the restart gate
# --------------------------------------------------------------------------------------------

WRAPPER = r'''
import os, shutil, signal, sys
from sisteer.registry import compact as C
fault = os.environ.get("CS_FAULT", "")
hook = None
if "--swap" in sys.argv[1:] and fault:
    kind, step = fault.split(":")
    root = sys.argv[sys.argv.index("--root") + 1]
    def hook(name):
        if name != step:
            return
        if kind == "foreign":                                  # a third inode at the live path: recovery must refuse
            live = os.path.join(root, "registry.sqlite")
            shutil.copy2(live, live + ".foreign")
            os.rename(live + ".foreign", live)
        os.kill(os.getpid(), signal.SIGKILL)
sys.exit(C.main(sys.argv[1:], on_step=hook))
'''


def _shim(d, name, body):
    p = d / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def _swap_world(tmp_path, reg, active, fault="", on_start="", pidfile=None):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    calls = tmp_path / "calls.log"
    act = tmp_path / "active"
    act.write_text(" ".join(active))
    show = f'cat "{pidfile}" 2>/dev/null || echo 0' if pidfile else "echo 0"
    _shim(bin_, "systemctl", f'''echo "systemctl $*" >> "{calls}"
[ "$1" = "--user" ] && shift
cmd="$1"; shift
a=" $(cat "{act}") "
case "$cmd" in
  is-active) q=0; [ "$1" = "--quiet" ] && {{ q=1; shift; }}; rc=0
    for u in "$@"; do case "$a" in *" $u "*) [ $q = 1 ] || echo active ;; *) [ $q = 1 ] || echo inactive; rc=3 ;; esac; done; exit $rc ;;
  stop) for u in "$@"; do a="${{a// $u / }}"; done; echo "$a" > "{act}" ;;
  start) for u in "$@"; do a="$a $u "; done; echo "$a" > "{act}"
{on_start} ;;
  show) {show} ;;
esac
exit 0
''')
    _shim(bin_, "fuser", "exit 1\n")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(WRAPPER)
    env = _env()
    env.update({"PATH": f"{bin_}:{env['PATH']}", "HOME": str(tmp_path / "home"), "CS_FAULT": fault,
                "COMPACT_SWAP_PY": sys.executable, "COMPACT_SWAP_COMPACT": f"{sys.executable} {wrapper}",
                "COMPACT_SWAP_UNITS": "registry-harvest.timer registry-status.timer registry-datasette.service paper-autobuild.timer",
                "COMPACT_SWAP_WINDOW_LOCK": str(tmp_path / "window.lock"), "COMPACT_SWAP_QUIET_TRIES": "3", "COMPACT_SWAP_QUIET_SLEEP_S": "0"})
    return env, {"calls": calls, "active": act, "attic_root": tmp_path / "attic-root"}


def _run_swap(env, P, reg, *extra):
    return subprocess.run(["bash", SWAP_SH, "--root", reg, "--attic-root", str(P["attic_root"]), "--no-post-checks", *extra],
                          capture_output=True, text=True, env=env)


def _attic(P):
    (d,) = list(P["attic_root"].iterdir())
    return str(d)


ACTIVE = ["registry-harvest.timer", "registry-status.timer", "registry-datasette.service"]


def test_compact_swap_restores_exactly_the_active_units_after_a_validated_swap(reg, tmp_path):
    env, P = _swap_world(tmp_path, reg, ACTIVE)
    old = C.inode(DB.db_path(reg))
    r = _run_swap(env, P, reg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "result=validated rc=0" in r.stdout
    calls = P["calls"].read_text().splitlines()
    assert "systemctl --user stop registry-harvest.timer registry-status.timer registry-datasette.service" in calls
    starts = [c for c in calls if c.startswith("systemctl --user start")]
    assert starts == ["systemctl --user start registry-harvest.timer registry-status.timer registry-datasette.service"]
    assert calls.index(starts[0]) > max(i for i, c in enumerate(calls) if "stop" in c)
    assert sorted(P["active"].read_text().split()) == sorted(ACTIVE)             # paper-autobuild was inactive: never started
    attic = _attic(P)
    assert C.read_state(attic)["phase"] == "validated" and C.inode(DB.db_path(reg)) != old
    assert C.inode(os.path.join(attic, DB.DB_NAME)) == old
    assert open(os.path.join(attic, "units.before")).read().split("\n")[:4] == [
        "registry-harvest.timer active", "registry-status.timer active", "registry-datasette.service active", "paper-autobuild.timer inactive"]
    assert not os.path.exists(os.path.join(reg, "state", "maintenance.json"))


def test_compact_swap_killed_at_the_rename_is_completed_forward_by_the_trap(reg, tmp_path):
    env, P = _swap_world(tmp_path, reg, ACTIVE, fault="kill:rename")
    r = _run_swap(env, P, reg)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "-- recover" in r.stdout and C.read_state(_attic(P))["phase"] == "validated"
    assert sorted(P["active"].read_text().split()) == sorted(ACTIVE)


def test_compact_swap_refused_restores_the_units_on_the_old_file(reg, tmp_path):
    env, P = _swap_world(tmp_path, reg, ACTIVE)
    old = C.inode(DB.db_path(reg))
    r = _run_swap(env, P, reg, "--min-free-bytes", str(1 << 62))            # the preconditions refuse: nothing swapped
    assert r.returncode == 3, r.stdout + r.stderr
    assert "result=not-swapped rc=3" in r.stdout
    assert C.inode(DB.db_path(reg)) == old
    assert sorted(P["active"].read_text().split()) == sorted(ACTIVE)
    assert not os.path.exists(os.path.join(reg, "state", "maintenance.json"))


def test_compact_swap_leaves_every_unit_stopped_when_the_restart_gate_refuses(reg, tmp_path):
    env, P = _swap_world(tmp_path, reg, ACTIVE, fault="foreign:link")
    r = _run_swap(env, P, reg)
    assert r.returncode == 5, r.stdout + r.stderr
    assert "ALERT: the restart gate REFUSED" in r.stdout and "result=BLOCKED rc=5" in r.stdout
    assert P["active"].read_text().split() == []                                 # nothing restarted
    assert not [c for c in P["calls"].read_text().splitlines() if c.startswith("systemctl --user start")]
    blocked = json.load(open(os.path.join(reg, "state", "swap_blocked.json")))
    assert blocked["attic"] == _attic(P) and "registry-harvest.timer active" in blocked["units_before"]
    assert os.path.exists(os.path.join(reg, "state", "maintenance.json"))       # the harvester stays at the door too
    # the operator restores the attic copy by hand, then --finish passes the gate and restarts the recorded units
    attic = _attic(P)
    os.rename(os.path.join(attic, DB.DB_NAME), DB.db_path(reg))
    r2 = subprocess.run(["bash", SWAP_SH, "--finish", attic, "--root", reg, "--no-post-checks"], capture_output=True, text=True, env=env)
    assert r2.returncode == 3, r2.stdout + r2.stderr
    assert sorted(P["active"].read_text().split()) == sorted(ACTIVE)
    assert not os.path.exists(os.path.join(reg, "state", "swap_blocked.json"))


HOLDER = ("import sqlite3, sys, time\nc = sqlite3.connect(sys.argv[1])\nc.execute('SELECT COUNT(*) FROM schema_migration').fetchone()\n"
          "open(sys.argv[2], 'w').write('ready')\ntime.sleep(300)\n")


def _start_holder(live, ready):
    p = subprocess.Popen([sys.executable, "-c", HOLDER, live, str(ready)])
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.1)
    return p


def test_compact_swap_post_checks_pass_when_the_restarted_datasette_holds_the_new_inode(reg, tmp_path):
    """The post checks for real: the systemctl shim's `start registry-datasette.service` launches a process holding
    the live file open, its pid answering `show -p MainPID`; curl answers 200, journalctl shows a finished harvest cycle, fuser finds no
    attic holder. The check reads the holder's descriptors from /proc and requires the NEW inode."""
    live, pidf, ready = DB.db_path(reg), tmp_path / "ds.pid", tmp_path / "ds.ready"
    (tmp_path / "holder.py").write_text(HOLDER)
    on_start = (f'    case " $* " in *" registry-datasette.service "*)\n'
                f'      "{sys.executable}" "{tmp_path / "holder.py"}" "{live}" "{ready}" >/dev/null 2>&1 & echo $! > "{pidf}"\n'
                f'      for i in $(seq 1 100); do [ -s "{ready}" ] && break; sleep 0.1; done ;; esac')
    units = ["registry-harvest.timer", "registry-datasette.service"]
    env, P = _swap_world(tmp_path, reg, units, on_start=on_start, pidfile=pidf)
    env["COMPACT_SWAP_UNITS"] = " ".join(units)
    _shim(tmp_path / "bin", "curl", "echo -n 200\n")
    _shim(tmp_path / "bin", "journalctl", 'echo "Finished registry-harvest.service - one harvester cycle."\n')
    try:
        r = subprocess.run(["bash", SWAP_SH, "--root", reg, "--attic-root", str(P["attic_root"]), "--post-wait-s", "30"],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "post check: Datasette answers 200" in r.stdout
        assert f"all on inode {os.stat(live).st_ino}" in r.stdout and "holds 0 descriptor" not in r.stdout
        assert "nobody holds" in r.stdout and "one harvest cycle after the restart: Finished" in r.stdout
        assert "result=validated rc=0" in r.stdout
    finally:
        if pidf.exists() and pidf.read_text().strip().isdigit():
            try:
                os.kill(int(pidf.read_text().strip()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_compact_swap_refuses_while_a_reader_outside_the_stop_set_holds_the_live_file(reg, tmp_path):
    """A reader the window does not stop (here a stand-in for Datasette left out of the unit set) holds the live file: the swap's
    preconditions refuse, nothing is swapped, and the units come back on the old file (exit 3)."""
    holder = _start_holder(DB.db_path(reg), tmp_path / "r.ready")
    try:
        env, P = _swap_world(tmp_path, reg, ["registry-harvest.timer"])
        env["COMPACT_SWAP_UNITS"] = "registry-harvest.timer"
        old = C.inode(DB.db_path(reg))
        r = _run_swap(env, P, reg)
        assert r.returncode == 3, r.stdout + r.stderr
        assert "held open" in r.stdout + r.stderr and "result=not-swapped rc=3" in r.stdout
        assert C.inode(DB.db_path(reg)) == old and P["active"].read_text().split() == ["registry-harvest.timer"]
    finally:
        holder.kill()
        holder.wait()


def test_compact_swap_dry_run_stops_nothing_and_writes_nothing(reg, tmp_path):
    env, P = _swap_world(tmp_path, reg, ACTIVE)
    r = _run_swap(env, P, reg, "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "dry run: would stop registry-harvest.timer registry-status.timer registry-datasette.service" in r.stdout
    assert not [c for c in P["calls"].read_text().splitlines() if " stop " in c or " start " in c]
    assert list(P["attic_root"].iterdir()) == []
    assert not os.path.exists(os.path.join(reg, "state", "maintenance.json"))
    bad = _run_swap(env, P, reg, "--require-file", str(tmp_path / "dispatch.hold"))
    assert bad.returncode == 2 and "required file absent" in bad.stdout
