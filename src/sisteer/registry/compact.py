"""Compaction of the registry database, and the swap of the compacted file into the live path.

``auto_vacuum`` is off, so pages freed by deleting rows return to the filesystem only when the file is rewritten. This module rewrites
it with ``VACUUM INTO`` and puts the copy in place without the live pathname ever being absent.

**The swap** (``--swap``; the writers stopped and drained by ``scripts/registry/landing/compact_swap.sh``). Under the registry's write
lock (``<root>/harvester.lock``), as a state machine whose every step is recorded, fsynced, in ``<attic>/SWAP_STATE.json``:

1. **preconditions**: no process holds ``registry.sqlite``, ``-wal``, ``-shm`` or ``-journal`` open (a ``/proc/*/fd`` scan);
   ``wal_checkpoint(TRUNCATE)`` reports busy 0 and leaves an empty WAL; free space is at least 1.5 x the larger of the database's
   used pages and the caller's expected compacted size; the pre identity is recorded (``st_dev:st_ino``, size, schema version, fold
   order, paper head, ``frame_projection/1``, every table's row count); the attic is on the database's filesystem.
2. **vacuum**: ``VACUUM INTO <root>/registry.sqlite.compact-<utc>`` from a read-only connection, under a free-space floor watch that
   interrupts it and removes the partial file.
3. **verify**: on the output, ``journal_mode=WAL`` (``VACUUM INTO`` writes a rollback-journal file), the header reads WAL,
   ``freelist_count`` 0, ``quick_check`` ok, the shape is the code's (``db.require_shape``), the fold order row present, and the
   identity equal to the pre identity; then the output is closed with no ``-wal``/``-shm`` left, and it and its directory fsynced.
4. **link**: every connection closed, the live companions absent (stale empty ones with no holder moved into the attic), then
   ``os.link(registry.sqlite, <attic>/registry.sqlite)`` -- the attic shares the old inode -- and the attic directory fsynced.
5. **rename**: ``os.rename(<output>, registry.sqlite)`` replaces the live name atomically; the root directory fsynced.
6. **validate**: the live inode is the output's, its shape the code's and its identity the pre identity.

A crash before step 5 leaves the old file at the live name, after it the new one; the name is never absent. **Recovery**
(``--recover``) reads the state file and the FILESYSTEM -- the live inode against the recorded old and new inodes, never the
recorded phase alone (a kill between the rename and its state record leaves the phase one step behind): live = old -> the leftover
output is removed and the attic's duplicate link of the live file unlinked (a consistent rollback); live = new -> validated, and on a
failed validation the attic link is renamed back over the live path (atomic; the old inode is intact with an empty WAL).
**The restart gate** (``--validate-live``) exits 0 only when the installed database validates: with a state file, its phase must be
terminal (``validated`` or ``rolled_back``) and the live inode the one that phase names, with that file's identity (the pre identity,
or -- for a compacted file kept when its rollback was undone -- the identity recorded when the rollback began); without one (no swap
ran), the shape alone. ``compact_swap.sh`` restores the writers only after it exits 0.

**Rollback** (``--rollback``, after a validated swap, the writers stopped again) puts the old file back through the same kind of
steps, and is recoverable the same way. Before any name moves: no holder, the old file (the attic link) has the code's shape, the live
file's WAL is folded in, and phase ``rolling_back`` is recorded with the old, new and kept inodes and the live file's identity (the
writers may have run since the swap, so the pre identity no longer describes it). Then: (r1) ``os.link(registry.sqlite,
<attic>/registry.sqlite.post-swap)`` keeps the compacted file; (r2) ``os.rename(<attic>/registry.sqlite, registry.sqlite)`` puts the
old file back atomically; (r3) the old file validates against the pre identity -> ``rolled_back``. A crash anywhere in between
leaves ``rolling_back``: the restart gate refuses and ``--recover`` finishes the rollback FORWARD from the inodes on disk (the
operator asked for it, and the old file is intact: nothing writes it while it is in the attic). If the old file does not validate,
the rollback is undone instead -- the old file kept under its attic name, the kept link renamed back over the live path (atomic) --
and the compacted file, validated against the identity recorded at the rollback's start, stays live (phase ``validated``, the
rollback's ``outcome`` ``kept_new``); that verdict is recorded before the undo moves anything, so a crash during the undo resumes it
rather than retrying the old file. Either way the live name is never absent and no second name of the live file is left behind.

``--vacuum-into OUT --db SRC`` is steps 2-3 alone, on any database (the rehearsal's compaction, the replica carry, the copy of the
live database taken with the writers up: ``VACUUM INTO`` from a read-only connection is a consistent snapshot).

``db.open_or_create`` refuses to create a fresh database in a root that holds harvest state, so a writer that opens the live path at
the wrong moment fails loudly instead of creating an empty registry.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sqlite3
import stat
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from . import db as DB

STATE_FORMAT = "compact-swap/1"
STATE_NAME = "SWAP_STATE.json"
#: ``registry.sqlite.compact-<utc>``: the output, on the live database's filesystem.
COMPACT_INFIX = ".compact-"
#: SQLite's companion files of a database path.
COMPANIONS: Tuple[str, ...] = ("-wal", "-shm", "-journal")
#: The recorded phases, in order; ``validated`` and ``rolled_back`` are terminal, ``rolling_back`` (a rollback in progress) is not.
PHASES: Tuple[str, ...] = ("preconditions", "vacuumed", "verified", "linked", "renamed", "validated", "rolling_back", "rolled_back")
TERMINAL: Tuple[str, ...] = ("validated", "rolled_back")
#: The step boundaries a test can fail at (``on_step``): each is called right after its action and BEFORE its phase is recorded,
#: so the state file lags the filesystem (the worst case recovery must handle) -- except ``preconditions``, recorded before its
#: boundary because the state file must exist before any output does.
STEPS: Tuple[str, ...] = ("preconditions", "vacuum", "verify", "link", "link_fsync", "rename", "rename_fsync", "validate")
#: The rollback's boundaries (``--rollback``), the same way: ``rollback_preconditions`` right after ``rolling_back`` is recorded and
#: before any name moves; then each action's; ``rollback_validate`` after the old file validated, before ``rolled_back`` is recorded.
ROLLBACK_STEPS: Tuple[str, ...] = ("rollback_preconditions", "rollback_link", "rollback_link_fsync", "rollback_rename",
                                   "rollback_rename_fsync", "rollback_validate")
#: The boundaries of undoing a rollback whose old file does not validate: the old file re-linked under its attic name, then the
#: kept compacted file renamed back over the live path.
ROLLBACK_UNDO_STEPS: Tuple[str, ...] = ("rollback_undo_link", "rollback_undo_rename")
#: The compacted file's second name while the old one is live again (``<attic>/registry.sqlite.post-swap``).
KEEP_SUFFIX = ".post-swap"
FREE_FACTOR = 1.5
DEFAULT_FLOOR_BYTES = 10 * 1024 ** 3
DEFAULT_JOBS = 4
FLOOR_POLL_S = 2.0

StepHook = Optional[Callable[[str], None]]


class CompactError(RuntimeError):
    exit_code = 1


class PreconditionFailed(CompactError):
    """Nothing was changed: the live database is as it was."""
    exit_code = 2


class VerifyFailed(CompactError):
    """The compacted output does not verify; the live database is untouched (``--recover`` removes the output)."""
    exit_code = 3


class RecoveryRefused(CompactError):
    """The filesystem is in a state the recorded swap cannot explain; nothing was changed by recovery."""
    exit_code = 4


class ValidationFailed(CompactError):
    exit_code = 5


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def utc_now() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"


def _log(msg: str) -> None:
    print(f"[compact {time.strftime('%H:%M:%S', time.gmtime())}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------------------------

def inode(path: str) -> Optional[Tuple[int, int]]:
    """``(st_dev, st_ino)`` of a regular file at ``path`` (not following a symlink), None when absent."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        raise RecoveryRefused(f"{path} is not a regular file")
    return int(st.st_dev), int(st.st_ino)


#: A directory's fsync (a rename or a new entry in it is durable only once the directory is): the database module's one implementation.
fsync_dir = DB.fsync_dir


def companions_present(path: str) -> List[str]:
    return [path + s for s in COMPANIONS if os.path.exists(path + s)]


def holders(paths: Sequence[str]) -> List[Dict[str, object]]:
    """Every OTHER process holding one of ``paths`` open: a scan of ``/proc/*/fd`` (processes this user cannot read are skipped --
    the registry's writers and readers all run as this user). ``[{pid, fd, path, cmd}]``."""
    want = {os.path.realpath(p) for p in paths}
    me = os.getpid()
    out: List[Dict[str, object]] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        d = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(d)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(d, fd))
            except OSError:
                continue
            if target.endswith(" (deleted)"):
                continue
            if target in want:
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as fh:
                        cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()[:200]
                except OSError:
                    cmd = ""
                out.append({"pid": int(pid), "fd": int(fd), "path": target, "cmd": cmd})
    return out


def _ro(path: str) -> sqlite3.Connection:
    return DB.connect(path, readonly=True, busy_timeout_ms=5000)


@contextlib.contextmanager
def anchored(path: str) -> Iterator[sqlite3.Connection]:
    """A read-write connection held around read-only work on ``path`` and closed LAST: SQLite's last connection to close checkpoints
    the WAL and removes ``-wal`` and ``-shm``, which a read-only connection never does. Writes nothing to the database."""
    conn = DB.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def wal_header(path: str) -> Tuple[int, int]:
    """Bytes 18 and 19 of the database header: ``(2, 2)`` for a WAL database, ``(1, 1)`` for a rollback-journal one."""
    with open(path, "rb") as fh:
        h = fh.read(100)
    if len(h) < 100 or not h.startswith(b"SQLite format 3\0"):
        raise VerifyFailed(f"{path} is not an SQLite database")
    return h[18], h[19]


# --------------------------------------------------------------------------------------------
# identity: what must be equal across a compaction
# --------------------------------------------------------------------------------------------

def _tables(conn: sqlite3.Connection) -> List[str]:
    return [str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                                            "ORDER BY name").fetchall()]


def table_counts(path: str, jobs: int = DEFAULT_JOBS) -> Dict[str, int]:
    """``{table: COUNT(*)}`` for every user table: ``jobs`` workers, one read-only connection each, taking the next table from a shared
    queue (SQLite releases the GIL while it scans, and one large table never holds the others behind it)."""
    import queue
    with contextlib.closing(_ro(path)) as c:
        names = _tables(c)
    todo: "queue.Queue[str]" = queue.Queue()
    for n in names:
        todo.put(n)

    def worker() -> List[Tuple[str, int]]:
        got: List[Tuple[str, int]] = []
        with contextlib.closing(_ro(path)) as c2:
            while True:
                try:
                    name = todo.get_nowait()
                except queue.Empty:
                    return got
                got.append((name, int(c2.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])))
    n = max(1, min(int(jobs), len(names) or 1))
    out: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=n) as ex:
        for part in [ex.submit(worker) for _ in range(n)]:
            out.update(part.result())
    return dict(sorted(out.items()))


def identity(path: str, jobs: int = DEFAULT_JOBS, projection: bool = True) -> Dict[str, object]:
    """The identity a compaction keeps: schema version, fold order, paper head, ``frame_projection/1`` (sha and counts), every table's
    row count; plus the file's ``st_dev:st_ino``, size and page figures, which it does not."""
    from .migrations import db_version
    st = os.stat(path)
    out: Dict[str, object] = {"path": os.path.abspath(path), "dev": int(st.st_dev), "ino": int(st.st_ino), "size": int(st.st_size)}
    with contextlib.closing(_ro(path)) as c:
        out["schema_version"] = int(db_version(c))
        out["fold_order"] = DB.fold_order_version(c)
        out["page_size"] = int(c.execute("PRAGMA page_size").fetchone()[0])
        out["page_count"] = int(c.execute("PRAGMA page_count").fetchone()[0])
        out["freelist_count"] = int(c.execute("PRAGMA freelist_count").fetchone()[0])
        try:
            row = c.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()
            out["paper_head"] = None if row is None else row[0]
        except sqlite3.Error as exc:
            out["paper_head"] = f"unreadable: {exc}"
        if projection:
            from .replica import frame_projection
            p = frame_projection(c, with_rows=False)
            out["projection"] = {"version": p["version"], "sha256": p["sha256"], "counts": p["counts"]}
    out["tables"] = table_counts(path, jobs)
    return out


#: The identity keys a compaction must keep equal.
KEPT_KEYS: Tuple[str, ...] = ("schema_version", "fold_order", "paper_head", "projection", "tables")


def identity_diff(a: Mapping[str, object], b: Mapping[str, object]) -> List[str]:
    """What differs between two identities on the kept keys (empty when the compaction kept them all)."""
    out: List[str] = []
    for k in KEPT_KEYS:
        if k == "tables":
            ta, tb = dict(a.get("tables") or {}), dict(b.get("tables") or {})
            for t in sorted(set(ta) | set(tb)):
                if ta.get(t) != tb.get(t):
                    out.append(f"table {t}: {ta.get(t)} rows before, {tb.get(t)} after")
        elif a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k)!r} before, {b.get(k)!r} after")
    return out


def shape_lines(path: str) -> List[str]:
    """``db.require_shape`` over a read-only connection: the version, the shape and the fold order are the code's (empty), else the
    refusal's lines (its message when it names none)."""
    with contextlib.closing(_ro(path)) as c:
        try:
            DB.require_shape(c, path)
        except DB.SchemaDrift as exc:
            return list(exc.lines) or [str(exc)]
    return []


def quick_check(path: str) -> List[str]:
    with contextlib.closing(_ro(path)) as c:
        rows = [str(r[0]) for r in c.execute("PRAGMA quick_check").fetchall()]
    return [] if rows == ["ok"] else rows[:20]


# --------------------------------------------------------------------------------------------
# VACUUM INTO and the output's checks
# --------------------------------------------------------------------------------------------

def _remove_output(out: str) -> List[str]:
    gone = []
    for p in [out] + [out + s for s in COMPANIONS]:
        if os.path.exists(p):
            os.unlink(p)
            gone.append(p)
    return gone


def vacuum_into(src: str, out: str, floor_bytes: int = DEFAULT_FLOOR_BYTES) -> Dict[str, object]:
    """``VACUUM INTO out`` from a read-only connection to ``src`` (a consistent snapshot even with writers up). ``out`` must not exist
    (SQLite would overwrite an empty file). A thread watches the free space of ``out``'s filesystem and interrupts the statement below
    ``floor_bytes``; on any failure the partial output is removed and the error raised."""
    for p in [out] + [out + s for s in COMPANIONS]:
        if os.path.exists(p):
            raise PreconditionFailed(f"VACUUM INTO target {p} exists")
    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    conn = DB.connect(src, readonly=True)
    stop = threading.Event()
    tripped: Dict[str, int] = {}

    def watch() -> None:
        # interrupts again at every poll while below the floor: sqlite3_interrupt is a no-op when no statement is running yet
        while not stop.wait(FLOOR_POLL_S):
            free = DB.free_bytes(out_dir)
            if free < floor_bytes:
                tripped["free"] = free
                conn.interrupt()
    w = threading.Thread(target=watch, daemon=True)
    t0 = time.monotonic()
    w.start()
    failure: Optional[BaseException] = None
    try:
        conn.execute("VACUUM INTO ?", (out,))
    except BaseException as exc:          # noqa: BLE001 -- re-raised below, after the watcher stopped and the partial file went
        failure = exc
    finally:
        stop.set()
        w.join()                          # the watcher never touches the connection after this
        conn.close()
    if failure is not None:
        removed = _remove_output(out)
        if tripped:
            raise PreconditionFailed(f"VACUUM INTO {out} stopped at the free-space floor ({tripped['free']} B free < {floor_bytes} B); "
                                     f"partial output removed: {removed}") from failure
        raise failure
    return {"seconds": round(time.monotonic() - t0, 3), "bytes": os.path.getsize(out), "min_free_floor": int(floor_bytes)}


def finish_output(out: str, expect: Optional[Mapping[str, object]] = None, quick: bool = True, shape: bool = True,
                  jobs: int = DEFAULT_JOBS) -> Dict[str, object]:
    """Make ``out`` (a fresh ``VACUUM INTO`` copy) a WAL database and verify it: the header reads WAL, ``freelist_count`` 0,
    ``quick_check`` ok (``quick``), the shape the code's (``shape``), and -- with ``expect`` -- the kept identity equal. Closed with no
    companion left, then it and its directory are fsynced. Raises :class:`VerifyFailed` naming every failed check."""
    t0 = time.monotonic()
    bad: List[str] = []
    with anchored(out) as a:
        mode = str(a.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode != "wal":
            bad.append(f"journal_mode reads {mode!r}, not 'wal'")
        freelist = int(a.execute("PRAGMA freelist_count").fetchone()[0])
        if freelist != 0:
            bad.append(f"freelist_count {freelist}, not 0")
        qc = quick_check(out) if quick else []
        if qc:
            bad.append("quick_check: " + "; ".join(qc))
        drift = shape_lines(out) if shape else []
        if drift:
            bad.append("shape: " + "; ".join(drift[:8]))
        ident = identity(out, jobs, projection=expect is not None and "projection" in expect)
        if ident.get("fold_order") is None and shape:
            bad.append("fold-order row 0 absent")
        diff = identity_diff(expect, ident) if expect is not None else []
        if diff:
            bad.append("identity: " + "; ".join(diff[:20]))
    left = companions_present(out)
    if left:
        bad.append(f"companion file(s) left after close: {left}")
    if wal_header(out) != (2, 2):
        bad.append(f"header bytes 18-19 read {wal_header(out)}, not (2, 2)")
    if bad:
        raise VerifyFailed(f"{out} does not verify: " + " | ".join(bad))
    DB.fsync_file_and_dir(out)
    ident["seconds"] = round(time.monotonic() - t0, 3)
    ident["quick_check"] = "ok" if quick else "skipped"
    return ident


# --------------------------------------------------------------------------------------------
# the swap state
# --------------------------------------------------------------------------------------------

def state_path(attic: str) -> str:
    return os.path.join(attic, STATE_NAME)


def read_state(attic: str) -> Optional[Dict[str, object]]:
    p = state_path(attic)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        st = json.load(fh)
    if st.get("format") != STATE_FORMAT:
        raise RecoveryRefused(f"{p} is not a {STATE_FORMAT} file")
    return st


def write_state(attic: str, st: Dict[str, object]) -> None:
    p = state_path(attic)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    fsync_dir(attic)


def record(attic: str, st: Dict[str, object], phase: str, **extra) -> None:
    if phase not in PHASES:
        raise ValueError(phase)
    st["phase"] = phase
    st.setdefault("history", []).append({"phase": phase, "utc": utc_now()})
    st.update(extra)
    write_state(attic, st)
    _log(f"phase {phase}")


def _hook(on_step: StepHook, name: str) -> None:
    if on_step is not None:
        on_step(name)


def _ino_of(d: Mapping[str, object]) -> Tuple[int, int]:
    return int(d["dev"]), int(d["ino"])


# --------------------------------------------------------------------------------------------
# the swap
# --------------------------------------------------------------------------------------------

def _clear_live_companions(live: str, attic: str) -> List[str]:
    """Step 4's precondition: no ``-wal``/``-shm``/``-journal`` at the live path (or at any other name of a file about to be renamed:
    the rollback calls it for the attic and kept names too). One read-write open and close folds a WAL into the file at that path and
    removes what read-only connections left; anything still there with no holder and no content (a ``-shm``, an empty ``-wal``) is
    moved into the attic; a non-empty ``-wal`` or ``-journal``, or a holder, refuses."""
    if companions_present(live):
        c = DB.connect(live)
        c.close()
    moved = []
    for p in companions_present(live):
        if holders([p]):
            raise PreconditionFailed(f"{p} is held open by {holders([p])}")
        if not p.endswith("-shm") and os.path.getsize(p) > 0:
            raise PreconditionFailed(f"{p} is not empty ({os.path.getsize(p)} B): un-checkpointed changes")
        dest = os.path.join(attic, os.path.basename(p) + ".stale")
        os.rename(p, dest)
        moved.append(dest)
    if moved:
        fsync_dir(attic)
        fsync_dir(os.path.dirname(live))
    return moved


def swap(root: str, attic: str, expect_bytes: int = 0, min_free_bytes: int = 0, floor_bytes: int = DEFAULT_FLOOR_BYTES,
         lock_timeout_s: float = 30.0, quick: bool = True, jobs: int = DEFAULT_JOBS, on_step: StepHook = None) -> Dict[str, object]:
    """Compact ``<root>/registry.sqlite`` and swap the copy in (module doc, steps 1-6). Returns the final state (phase
    ``validated``). Raises :class:`PreconditionFailed` with nothing changed, or another error with the state file naming how far it
    got (``--recover`` finishes or rolls back)."""
    root = os.path.abspath(root)
    attic = os.path.abspath(attic)
    live = DB.db_path(root)
    if not os.path.isfile(live):
        raise PreconditionFailed(f"{live} is not a file")
    for p in (state_path(attic), os.path.join(attic, DB.DB_NAME)):
        if os.path.exists(p):
            raise PreconditionFailed(f"{p} exists: one attic per swap (a new directory for a new attempt)")
    os.makedirs(attic, exist_ok=True)
    if os.stat(attic).st_dev != os.stat(root).st_dev:
        raise PreconditionFailed(f"attic {attic} is not on the database's filesystem (a hard link needs one)")
    out = os.path.join(root, DB.DB_NAME + COMPACT_INFIX + utc_stamp())
    attic_db = os.path.join(attic, DB.DB_NAME)
    st: Dict[str, object] = {"format": STATE_FORMAT, "root": root, "live": live, "attic": attic, "attic_db": attic_db, "output": out,
                             "host": socket.gethostname(), "pid": os.getpid(), "code": os.path.abspath(__file__),
                             "started_utc": utc_now()}
    with DB.write_lock(root, timeout_s=lock_timeout_s):
        held = holders([live] + [live + s for s in COMPANIONS])
        if held:
            raise PreconditionFailed(f"the live database is held open: {held}")
        for p in [out] + [out + s for s in COMPANIONS] + [attic_db]:
            if os.path.exists(p):
                raise PreconditionFailed(f"{p} exists")
        with anchored(live) as a:
            busy, log_frames, done = DB.wal_checkpoint_truncate(a, busy_timeout_ms=5000)
            wal = os.path.getsize(live + "-wal") if os.path.exists(live + "-wal") else 0
            if busy or wal:
                raise PreconditionFailed(f"wal_checkpoint(TRUNCATE) busy={busy} log={log_frames} checkpointed={done}; WAL {wal} B after")
            page_size = int(a.execute("PRAGMA page_size").fetchone()[0])
            used = (int(a.execute("PRAGMA page_count").fetchone()[0]) - int(a.execute("PRAGMA freelist_count").fetchone()[0])) * page_size
            need = max(int(FREE_FACTOR * max(used, int(expect_bytes or 0))), int(min_free_bytes or 0))
            free = DB.free_bytes(root)
            if free < need:
                raise PreconditionFailed(f"{free} B free on the database's filesystem; the swap needs {need} B "
                                         f"(1.5 x max(used pages {used} B, expected {expect_bytes} B), floor {min_free_bytes} B)")
            _log(f"preconditions hold: no holder, WAL empty, {free} B free >= {need} B; recording the pre identity")
            pre = identity(live, jobs)
            st.update({"pre": pre, "free_bytes": free, "need_bytes": need, "used_bytes": used})
            record(attic, st, "preconditions")
            _hook(on_step, "preconditions")
            _log(f"VACUUM INTO {out}")
            vac = vacuum_into(live, out, floor_bytes)
            _hook(on_step, "vacuum")
            record(attic, st, "vacuumed", vacuum=vac)
            new = finish_output(out, expect=pre, quick=quick, jobs=jobs)
            _hook(on_step, "verify")
            record(attic, st, "verified", new=new)
        # every connection of ours is closed (the anchor last): the swap itself
        moved = _clear_live_companions(live, attic)
        held = holders([live] + [live + s for s in COMPANIONS] + [out])
        if held:
            raise PreconditionFailed(f"held open at the swap: {held}")
        if inode(live) != _ino_of(pre) or inode(out) != _ino_of(new):
            raise PreconditionFailed(f"an inode moved under the swap: live {inode(live)} (pre {_ino_of(pre)}), output {inode(out)} "
                                     f"(verified {_ino_of(new)})")
        os.link(live, attic_db)
        _hook(on_step, "link")
        fsync_dir(attic)
        _hook(on_step, "link_fsync")
        record(attic, st, "linked", stale_companions=moved)
        os.rename(out, live)
        _hook(on_step, "rename")
        fsync_dir(root)
        _hook(on_step, "rename_fsync")
        record(attic, st, "renamed")
        res = validate_installed(st, _ino_of(new), jobs=jobs)
        if res["problems"]:
            raise ValidationFailed(f"the installed compacted file does not validate: {res['problems']} (run --recover: it rolls back)")
        _hook(on_step, "validate")
        record(attic, st, "validated", validation=res, finished_utc=utc_now())
    return st


def validate_installed(st: Mapping[str, object], expect_ino: Tuple[int, int], jobs: int = DEFAULT_JOBS,
                       quick: bool = False, expect_identity: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    """The live file is the expected inode, its header WAL, its shape the code's and its identity ``expect_identity`` (default: the
    pre identity)."""
    live = str(st["live"])
    want = st["pre"] if expect_identity is None else expect_identity
    problems: List[str] = []
    got = inode(live)
    if got is None:
        return {"problems": [f"{live} is absent"], "inode": None}
    if got != tuple(expect_ino):
        problems.append(f"live inode {got}, expected {tuple(expect_ino)}")
    if wal_header(live) != (2, 2):
        problems.append(f"header bytes 18-19 read {wal_header(live)}, not WAL")
    with anchored(live):
        problems += ["shape: " + x for x in shape_lines(live)]
        if quick:
            problems += ["quick_check: " + x for x in quick_check(live)]
        ident = identity(live, jobs)
    problems += ["identity: " + x for x in identity_diff(want, ident)]
    return {"problems": problems, "inode": list(got), "identity": ident, "utc": utc_now()}


def gate_target(st: Mapping[str, object]) -> Tuple[Tuple[int, int], Mapping[str, object], str]:
    """What a TERMINAL phase says is live: ``(inode, identity to validate against, a label)``. ``rolled_back``: the old file, with the
    pre identity (nothing writes it in the attic). ``validated``: the compacted file with the pre identity -- or, when a rollback of
    it was undone (``rollback.outcome == "kept_new"``), with the identity recorded at that rollback's start, since the writers ran on
    it between the swap and the rollback."""
    phase = st.get("phase")
    if phase == "rolled_back":
        return _ino_of(st["pre"]), st["pre"], "the old file (rolled back)"
    if phase == "validated":
        rb = st.get("rollback") or {}
        if isinstance(rb, Mapping) and rb.get("outcome") == "kept_new":
            return _ino_of(st["new"]), rb["new_identity"], "the compacted file (its rollback undone)"
        return _ino_of(st["new"]), st["pre"], "the compacted file"
    raise ValueError(f"phase {phase!r} is not terminal")


# --------------------------------------------------------------------------------------------
# recovery, the restart gate, rollback
# --------------------------------------------------------------------------------------------

def recover(attic: str, lock_timeout_s: float = 30.0, jobs: int = DEFAULT_JOBS) -> Dict[str, object]:
    """Complete the swap the state file records, or roll it back, from the inodes on disk (module doc). A ``--rollback`` stopped
    mid-way (phase ``rolling_back``) is finished forward, or undone when the old file does not validate (:func:`_advance_rollback`).
    Returns the final state. Raises :class:`RecoveryRefused` when the live path holds neither recorded inode, or the attic copy needed
    to roll back is not the old file."""
    attic = os.path.abspath(attic)
    st = read_state(attic)
    if st is None:
        return {"phase": None, "note": f"no {STATE_NAME} under {attic}: no swap recorded there, nothing to recover"}
    if "pre" not in st:
        # the preconditions were never recorded: nothing was made but the attic directory
        return {"phase": st.get("phase"), "note": "the swap stopped before its preconditions were recorded: nothing to recover"}
    if st.get("phase") in TERMINAL:
        # finished, either way: never re-decided -- the writers may have run on the installed file since, and a re-validation against
        # the pre identity would then fail and roll their writes back (an after-the-fact rollback is --rollback, deliberately)
        st["note"] = f"the swap is already {st['phase']}: nothing to recover"
        return st
    if st.get("phase") == "rolling_back":
        # a --rollback stopped mid-way: finished forward (or undone when the old file does not validate), from the inodes on disk
        with DB.write_lock(str(st["root"]), timeout_s=lock_timeout_s):
            _refuse_rollback_holders(st)
            return _advance_rollback(attic, st, jobs=jobs, notes=["recovery of a rollback stopped mid-way"])
    live, out, attic_db, root = str(st["live"]), str(st["output"]), str(st["attic_db"]), str(st["root"])
    old = _ino_of(st["pre"])
    new = _ino_of(st["new"]) if st.get("new") else None
    with DB.write_lock(root, timeout_s=lock_timeout_s):
        got = inode(live)
        notes: List[str] = []
        if got is None:
            # never by construction (link, then rename over): put the old file back from the attic
            if inode(attic_db) != old:
                raise RecoveryRefused(f"{live} is absent and {attic_db} is not the old file ({inode(attic_db)} vs {old}): restore by hand")
            os.link(attic_db, live)
            fsync_dir(root)
            got = old
            notes.append(f"the live path was absent: re-linked from {attic_db}")
        if got == old:
            removed = []
            if os.path.exists(out) and inode(out) != old:
                removed = _remove_output(out)
            if inode(attic_db) == old:
                os.unlink(attic_db)
                fsync_dir(attic)
                notes.append(f"removed {attic_db}, a second name of the live file")
            res = validate_installed(st, old, jobs=jobs)
            notes.append(f"the rename never happened: rolled back; removed {removed or 'no output'}")
            if res["problems"]:
                record(attic, st, st["phase"] if st["phase"] in PHASES else "preconditions", recovery={"notes": notes, "validation": res})
                raise RecoveryRefused(f"rolled back to the old file, but it does not validate: {res['problems']}")
            record(attic, st, "rolled_back", recovery={"notes": notes, "validation": res, "utc": utc_now()})
            return st
        if new is not None and got == new:
            res = validate_installed(st, new, jobs=jobs)
            if not res["problems"]:
                notes.append("the rename had happened: validated, completed forward")
                record(attic, st, "validated", validation=res, recovery={"notes": notes, "utc": utc_now()}, finished_utc=utc_now())
                return st
            if inode(attic_db) != old:
                raise RecoveryRefused(f"the new file does not validate ({res['problems']}) and {attic_db} is not the old file: restore by hand")
            os.rename(attic_db, live)
            fsync_dir(root)
            fsync_dir(attic)
            res2 = validate_installed(st, old, jobs=jobs)
            notes.append(f"the new file did not validate ({res['problems']}): the attic link renamed back over the live path")
            if res2["problems"]:
                record(attic, st, "renamed", recovery={"notes": notes, "validation": res2})
                raise RecoveryRefused(f"rolled back to the old file, but it does not validate: {res2['problems']}")
            record(attic, st, "rolled_back", recovery={"notes": notes, "validation": res2, "utc": utc_now()})
            return st
        raise RecoveryRefused(f"{live} is inode {got}: neither the old file {old} nor the compacted one {new}; nothing changed")


def validate_live(root: str, attic: Optional[str] = None, quick: bool = False, jobs: int = DEFAULT_JOBS) -> Dict[str, object]:
    """The writers' restart gate: ``ok`` only when the installed database validates. With ``attic``'s state file, its phase must be
    terminal and the live inode the one it names, with that file's identity (:func:`gate_target`); a swap or a rollback still in
    flight is refused (``--recover`` first). Without one, the live file must exist with the code's shape."""
    live = DB.db_path(os.path.abspath(root))
    st = read_state(os.path.abspath(attic)) if attic else None
    if st is None or "pre" not in st:
        problems: List[str] = []
        if inode(live) is None:
            problems.append(f"{live} is absent")
        else:
            with anchored(live):
                problems += ["shape: " + x for x in shape_lines(live)]
                if quick:
                    problems += ["quick_check: " + x for x in quick_check(live)]
        return {"ok": not problems, "phase": None if st is None else st.get("phase"), "problems": problems, "live": live}
    if os.path.abspath(str(st["live"])) != live:
        return {"ok": False, "phase": st.get("phase"), "problems": [f"the state names {st['live']}, not {live}"], "live": live}
    phase = st.get("phase")
    if phase not in TERMINAL:
        what = "a rollback stopped mid-way" if phase == "rolling_back" else f"the swap stopped at phase {phase!r}"
        return {"ok": False, "phase": phase, "live": live, "problems": [f"{what}: run --recover --attic {attic} first"]}
    expect, ident, label = gate_target(st)
    res = validate_installed(st, expect, jobs=jobs, quick=quick, expect_identity=ident)
    return {"ok": not res["problems"], "phase": phase, "problems": res["problems"], "live": live, "inode": res["inode"],
            "expected": label}


def _rollback_paths(st: Mapping[str, object]) -> Tuple[str, str, str, str]:
    """``(live, attic_db, keep, root)`` of a recorded rollback."""
    rb = st["rollback"]
    assert isinstance(rb, Mapping)
    return str(st["live"]), str(st["attic_db"]), str(rb["keep"]), str(st["root"])


def _refuse_rollback_holders(st: Mapping[str, object]) -> None:
    live, attic_db, keep, _ = _rollback_paths(st)
    held = holders([live] + [live + s for s in COMPANIONS] + [attic_db] + [attic_db + s for s in COMPANIONS] + [keep])
    if held:
        raise PreconditionFailed(f"held open (the writers must be stopped for a rollback): {held}")


def rollback(attic: str, lock_timeout_s: float = 30.0, jobs: int = DEFAULT_JOBS, on_step: StepHook = None) -> Dict[str, object]:
    """After a VALIDATED swap, with the writers stopped again: the old file back at the live path, the compacted file kept in the
    attic as ``registry.sqlite.post-swap`` (module doc, "Rollback"). Writes made since the swap are not in the old file; the harvester
    re-folds what it had delivered since from the record log. Everything a crash could interrupt is recorded first (phase
    ``rolling_back`` with the old, new and kept inodes and the live file's identity), so ``--recover`` finishes it. Raises
    :class:`PreconditionFailed` / :class:`RecoveryRefused` with nothing changed when it cannot start, and :class:`ValidationFailed`
    when the old file did not validate and the rollback was undone (the compacted file live and validated, phase ``validated``)."""
    attic = os.path.abspath(attic)
    st = read_state(attic)
    if st is not None and st.get("phase") == "rolling_back":
        raise RecoveryRefused(f"a rollback under {attic} stopped mid-way: run --recover --attic {attic} (it finishes it)")
    if st is None or st.get("phase") != "validated":
        raise RecoveryRefused(f"rollback needs a validated swap under {attic} (phase {None if st is None else st.get('phase')!r})")
    live, attic_db, root = str(st["live"]), str(st["attic_db"]), str(st["root"])
    old, new = _ino_of(st["pre"]), _ino_of(st["new"])
    keep = os.path.join(attic, DB.DB_NAME + KEEP_SUFFIX)
    with DB.write_lock(root, timeout_s=lock_timeout_s):
        held = holders([live] + [live + s for s in COMPANIONS] + [attic_db] + [attic_db + s for s in COMPANIONS] + [keep])
        if held:
            raise PreconditionFailed(f"held open: {held}")
        if inode(live) != new or inode(attic_db) != old or os.path.exists(keep):
            raise RecoveryRefused(f"live {inode(live)} / attic {inode(attic_db)} are not the swap's new {new} / old {old}, or {keep} exists")
        # the old file must open with the code's shape: a migration applied to the live file since the swap leaves it behind, and
        # rolling back would install a database the writers refuse
        if wal_header(attic_db) != (2, 2):
            raise PreconditionFailed(f"the old file {attic_db} is not a WAL database (header {wal_header(attic_db)}); nothing changed")
        with anchored(attic_db):
            drift = shape_lines(attic_db)
        if drift or companions_present(attic_db):
            raise PreconditionFailed(f"the old file {attic_db} does not open cleanly with the code's shape ({drift[:5]}, companions "
                                     f"{companions_present(attic_db)}): nothing changed")
        # the live (compacted) file's WAL folded in -- the writers ran on it since the swap -- and its identity recorded: the restart
        # gate's reference should the rollback have to be undone
        with anchored(live) as a:
            busy, log_frames, done = DB.wal_checkpoint_truncate(a, busy_timeout_ms=5000)
            wal = os.path.getsize(live + "-wal") if os.path.exists(live + "-wal") else 0
            if busy or wal:
                raise PreconditionFailed(f"wal_checkpoint(TRUNCATE) busy={busy} log={log_frames} checkpointed={done}; WAL {wal} B after")
            new_ident = identity(live, jobs)
        left = companions_present(live)
        if left or inode(live) != new:
            raise PreconditionFailed(f"after the checkpoint: companions {left}, live inode {inode(live)} (expected {new}); nothing changed")
        if st.get("rollback"):
            st.setdefault("rollbacks_before", []).append(st.pop("rollback"))   # an earlier rollback that was undone
        record(attic, st, "rolling_back", rollback={"started_utc": utc_now(), "host": socket.gethostname(), "pid": os.getpid(),
                                                    "old": list(old), "new": list(new), "keep": keep, "new_identity": new_ident})
        _hook(on_step, "rollback_preconditions")
        st = _advance_rollback(attic, st, jobs=jobs, on_step=on_step)
    rb = st["rollback"]
    if rb.get("outcome") == "kept_new":
        raise ValidationFailed(f"the old file does not validate ({rb['old_failed']['problems'][:5]}): the rollback was undone, the "
                               f"compacted file is live and validated (phase {st['phase']}); --validate-live passes")
    return st


def _advance_rollback(attic: str, st: Dict[str, object], jobs: int = DEFAULT_JOBS, on_step: StepHook = None,
                      notes: Optional[List[str]] = None) -> Dict[str, object]:
    """Drive a recorded rollback (phase ``rolling_back``) to a terminal phase, deciding every move from the inodes on disk -- the live
    path (``L``), the attic name of the old file (``A``) and the kept name of the compacted one (``K``) against the recorded ``old`` and
    ``new`` -- never from the recorded step. Forward (the default): ``L = new, A = old`` -> ``K`` linked if absent, ``A`` renamed over
    ``L``; ``L = old, K = new`` -> the old file validated against the pre identity, any second name of it (``A``) removed, phase
    ``rolled_back``. When the old file does not validate, that verdict is recorded (``rollback.old_failed``) BEFORE anything moves and
    the rollback is undone (:func:`_undo_rollback`). The live path is given a name first if it is ever found absent (never by
    construction). The caller holds the registry's write lock."""
    rb = st["rollback"]
    assert isinstance(rb, dict)
    notes = list(notes or [])
    if rb.get("old_failed"):
        return _undo_rollback(attic, st, jobs=jobs, on_step=on_step, notes=notes)
    live, attic_db, keep, root = _rollback_paths(st)
    old, new = _ino_of(st["pre"]), _ino_of(st["new"])
    L, A, K = inode(live), inode(attic_db), inode(keep)
    if L is None:
        if A == old:
            os.link(attic_db, live)
            fsync_dir(root)
            L = old
            notes.append(f"the live path was absent: re-linked from {attic_db}")
        elif K == new:
            os.link(keep, live)
            fsync_dir(root)
            L = new
            notes.append(f"the live path was absent: re-linked from {keep}")
        else:
            raise RecoveryRefused(f"{live} is absent, {attic_db} is {A} (old {old}) and {keep} is {K} (new {new}): restore by hand")
    if L == new:
        if A != old:
            raise RecoveryRefused(f"the compacted file is live but {attic_db} is {A}, not the old file {old}: nothing to roll back to")
        if K is None:
            os.link(live, keep)
            _hook(on_step, "rollback_link")
            fsync_dir(attic)
            _hook(on_step, "rollback_link_fsync")
            K = new
            notes.append(f"the compacted file linked as {keep}")
        elif K != new:
            raise RecoveryRefused(f"{keep} is inode {K}, not the compacted file {new}")
        # the companions named after either path belong to the inode there now: folded into it and cleared before a name moves
        # (a stale -wal of one file at the live path would be replayed into the other)
        _clear_live_companions(attic_db, attic)
        _clear_live_companions(live, attic)
        os.rename(attic_db, live)
        _hook(on_step, "rollback_rename")
        fsync_dir(root)
        fsync_dir(attic)
        _hook(on_step, "rollback_rename_fsync")
        L, A = old, None
        notes.append(f"{attic_db} renamed over the live path: the old file is live")
    if L != old:
        raise RecoveryRefused(f"{live} is inode {L}: neither the old file {old} nor the compacted one {new}")
    if K != new:
        raise RecoveryRefused(f"the old file is live but {keep} is {K}, not the compacted file {new}: the rollback can be neither "
                              "confirmed nor undone as recorded; restore by hand")
    res = validate_installed(st, old, jobs=jobs)
    if not res["problems"]:
        _hook(on_step, "rollback_validate")
        if A == old:
            os.unlink(attic_db)
            fsync_dir(attic)
            notes.append(f"removed {attic_db}, a second name of the live file")
        rb.update({"outcome": "rolled_back", "validation": res, "notes": notes, "finished_utc": utc_now()})
        record(attic, st, "rolled_back")
        return st
    # the old file does not validate: the verdict first, then the way back
    notes.append(f"the old file does not validate ({res['problems'][:5]}): the rollback is undone")
    rb.update({"old_failed": {"utc": utc_now(), "problems": res["problems"]}, "notes": notes})
    record(attic, st, "rolling_back")
    return _undo_rollback(attic, st, jobs=jobs, on_step=on_step, notes=notes)


def _undo_rollback(attic: str, st: Dict[str, object], jobs: int = DEFAULT_JOBS, on_step: StepHook = None,
                   notes: Optional[List[str]] = None) -> Dict[str, object]:
    """The way back from a rollback whose old file does not validate (``rollback.old_failed`` recorded), from the inodes on disk: the
    target is ``L = new`` (the compacted file live), ``A = old`` (the old file kept under its attic name) and no ``K`` (no second name
    of the live file). ``L = old`` -> ``A`` linked if absent, then ``K`` renamed over ``L`` (atomic). The compacted file is validated
    against the identity recorded when the rollback began -> phase ``validated`` with ``rollback.outcome = "kept_new"``; if it does not
    validate either, the phase stays ``rolling_back`` (the restart gate refuses) and :class:`RecoveryRefused` names both failures."""
    rb = st["rollback"]
    assert isinstance(rb, dict)
    notes = list(notes or [])
    live, attic_db, keep, root = _rollback_paths(st)
    old, new = _ino_of(st["pre"]), _ino_of(st["new"])
    L, A, K = inode(live), inode(attic_db), inode(keep)
    if L is None:
        if K == new:
            os.link(keep, live)
            fsync_dir(root)
            L = new
            notes.append(f"the live path was absent: re-linked from {keep}")
        elif A == old:
            os.link(attic_db, live)
            fsync_dir(root)
            L = old
            notes.append(f"the live path was absent: re-linked from {attic_db}")
        else:
            raise RecoveryRefused(f"{live} is absent, {attic_db} is {A} (old {old}) and {keep} is {K} (new {new}): restore by hand")
    if L == old:
        if A is None:
            os.link(live, attic_db)
            _hook(on_step, "rollback_undo_link")
            fsync_dir(attic)
            A = old
            notes.append(f"the old file kept as {attic_db}")
        elif A != old:
            raise RecoveryRefused(f"{attic_db} is {A}, not the old file {old}: the undo would leave the old file without a name")
        if K != new:
            raise RecoveryRefused(f"{keep} is {K}, not the compacted file {new}: nothing to undo the rollback with; restore by hand")
        _clear_live_companions(keep, attic)
        _clear_live_companions(live, attic)
        os.rename(keep, live)
        _hook(on_step, "rollback_undo_rename")
        fsync_dir(root)
        fsync_dir(attic)
        L, K = new, None
        notes.append(f"{keep} renamed back over the live path: the compacted file is live")
    if L != new:
        raise RecoveryRefused(f"{live} is inode {L}: neither the old file {old} nor the compacted one {new}")
    if A != old:
        raise RecoveryRefused(f"the compacted file is live but {attic_db} is {A}, not the old file {old}: restore by hand")
    if K == new:
        os.unlink(keep)
        fsync_dir(attic)
        notes.append(f"removed {keep}, a second name of the live file")
    elif K is not None:
        raise RecoveryRefused(f"{keep} is {K}, neither absent nor the compacted file {new}")
    res = validate_installed(st, new, jobs=jobs, expect_identity=rb["new_identity"])  # type: ignore[arg-type]
    if res["problems"]:
        rb.update({"notes": notes, "undo_validation": res})
        record(attic, st, "rolling_back")
        raise RecoveryRefused(f"the rollback was undone (the old file does not validate: {rb['old_failed']['problems'][:5]}), but the "
                              f"compacted file does not validate either: {res['problems'][:5]}; the writers stay stopped -- restore by hand")
    rb.update({"outcome": "kept_new", "validation": res, "notes": notes, "finished_utc": utc_now()})
    record(attic, st, "validated")
    return st


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sisteer.registry.compact", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    act = p.add_mutually_exclusive_group(required=True)
    act.add_argument("--swap", action="store_true", help="compact <root>/registry.sqlite and swap the copy in (writers stopped)")
    act.add_argument("--recover", action="store_true", help="complete or roll back the swap recorded under --attic")
    act.add_argument("--validate-live", action="store_true", help="the restart gate: exit 0 only when the installed database validates")
    act.add_argument("--rollback", action="store_true", help="after a validated swap (writers stopped): the old file back at the live "
                                                               "path; a rollback stopped mid-way is finished by --recover")
    act.add_argument("--vacuum-into", metavar="OUT", help="VACUUM INTO OUT from --db, made WAL and verified; no swap")
    act.add_argument("--show", action="store_true", help="print the state file under --attic")
    p.add_argument("--root", help="the registry root (holds registry.sqlite and harvester.lock)")
    p.add_argument("--attic", help="the swap's attic directory (new and on the database's filesystem for --swap)")
    p.add_argument("--db", help="--vacuum-into: the source database")
    p.add_argument("--expect-bytes", type=int, default=0, help="--swap: the expected compacted size (the rehearsal's); free space >= 1.5x")
    p.add_argument("--min-free-bytes", type=int, default=0, help="--swap: an absolute free-space floor for the preconditions")
    p.add_argument("--floor-bytes", type=int, default=DEFAULT_FLOOR_BYTES, help="interrupt VACUUM INTO below this free space (default 10 GiB)")
    p.add_argument("--lock-timeout", type=float, default=30.0, help="seconds to wait for the registry's write lock")
    p.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="read-only connections for the row counts")
    p.add_argument("--no-quick-check", action="store_true", help="skip PRAGMA quick_check on the output")
    p.add_argument("--quick-check", action="store_true", help="--validate-live: also run PRAGMA quick_check on the live file")
    p.add_argument("--no-shape-check", action="store_true", help="--vacuum-into: skip the shape check (a database at another schema)")
    p.add_argument("--compare-source", action="store_true", help="--vacuum-into: require the output's identity to equal the source's "
                                                                  "(only for a source nothing writes)")
    p.add_argument("--json", dest="json_out", help="also write the result document to this file")
    return p


def _emit(doc: Mapping[str, object], json_out: Optional[str]) -> None:
    text = json.dumps(doc, indent=1, sort_keys=True, default=str)
    print(text)
    if json_out:
        tmp = json_out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        os.replace(tmp, json_out)


def main(argv: Optional[Sequence[str]] = None, on_step: StepHook = None) -> int:
    """The CLI. ``on_step`` is a test seam (a callable run at each :data:`STEPS` boundary of ``--swap`` and each
    :data:`ROLLBACK_STEPS` / :data:`ROLLBACK_UNDO_STEPS` boundary of ``--rollback``); the command line never sets it."""
    args = _parser().parse_args(argv)
    need = {"swap": ("root", "attic"), "recover": ("attic",), "validate_live": ("root",), "rollback": ("attic",),
            "vacuum_into": ("db",), "show": ("attic",)}
    action = next(a for a in need if getattr(args, a))
    missing = [f"--{k}" for k in need[action] if not getattr(args, k)]
    if missing:
        print(f"compact: --{action.replace('_', '-')} needs {' '.join(missing)}", file=sys.stderr)
        return 2
    try:
        if action == "show":
            _emit(read_state(args.attic) or {"phase": None}, None)
            return 0
        if action == "vacuum_into":
            src = os.path.abspath(args.db)
            pre = identity(src, args.jobs) if args.compare_source else None
            vac = vacuum_into(src, args.vacuum_into, args.floor_bytes)
            ident = finish_output(args.vacuum_into, expect=pre, quick=not args.no_quick_check, shape=not args.no_shape_check, jobs=args.jobs)
            doc = {"result": "VACUUMED", "source": src, "source_identity": pre, "output": os.path.abspath(args.vacuum_into),
                   "vacuum": vac, "output_identity": ident}
            _emit(doc, args.json_out)
            print(f"RESULT vacuum-into ok bytes={vac['bytes']} seconds={vac['seconds']}", file=sys.stderr)
            return 0
        if action == "swap":
            st = swap(args.root, args.attic, expect_bytes=args.expect_bytes, min_free_bytes=args.min_free_bytes,
                      floor_bytes=args.floor_bytes, lock_timeout_s=args.lock_timeout, quick=not args.no_quick_check, jobs=args.jobs,
                      on_step=on_step)
            _emit(st, args.json_out)
            print(f"RESULT swap {st['phase']} live_ino={st['new']['ino']} size_before={st['pre']['size']} size_after={st['new']['size']}",
                  file=sys.stderr)
            return 0
        if action == "recover":
            st = recover(args.attic, lock_timeout_s=args.lock_timeout, jobs=args.jobs)
            _emit(st, args.json_out)
            print(f"RESULT recover {st.get('phase')}", file=sys.stderr)
            return 0
        if action == "validate_live":
            res = validate_live(args.root, args.attic, quick=args.quick_check, jobs=args.jobs)
            _emit(res, args.json_out)
            print(f"RESULT validate-live {'ok' if res['ok'] else 'REFUSED'} phase={res.get('phase')}", file=sys.stderr)
            return 0 if res["ok"] else ValidationFailed.exit_code
        if action == "rollback":
            st = rollback(args.attic, lock_timeout_s=args.lock_timeout, jobs=args.jobs, on_step=on_step)
            _emit(st, args.json_out)
            print(f"RESULT rollback {st['phase']}", file=sys.stderr)
            return 0
    except CompactError as exc:
        print(f"compact --{action.replace('_', '-')} REFUSED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return exc.exit_code
    except DB.LockHeld as exc:
        print(f"compact --{action.replace('_', '-')} REFUSED: the registry's write lock is held ({exc}); nothing changed", file=sys.stderr)
        return PreconditionFailed.exit_code
    return 1


if __name__ == "__main__":
    sys.exit(main())
