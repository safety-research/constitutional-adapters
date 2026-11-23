"""The run directory's SQLite ledger (``state.sqlite``, WAL): the submission state of every batch and the verdict cache of every
request (stdlib only). One writer at a time -- the harvester, or a one-shot ``submitter`` / ``harvester --once`` invocation -- under
an exclusive ``fcntl`` lock on ``<run_dir>/.writer.lock``; readers open read-only.

Tables
  batches       one row per manifest: ``state`` is one of
                  pending          not yet accepted (also after a retryable refusal; ``next_attempt_ts`` gates the next try)
                  balance_queued   refused for balance / spending limit (402, or 403 naming the limit): queued, retried with backoff
                  submitted        accepted (202); ``remote_status`` tracks validating / in_progress / finalizing
                  finished         a final remote status (completed / failed / expired / cancelled) seen, results not yet parsed
                  harvested        results parsed into ``requests`` (or, for a failed batch, its requests re-queued)
                  split            413: replaced by two child manifests
                  refused          set aside for good: a single request too large for any batch, or an operator's ``--set-aside``;
                                   the reason is kept. NOT finished: a run holding one never reports done (harvester exit 5), and
                                   the filing refuses it without ``--allow-partial``
  requests      the VERDICT CACHE, one row per custom_id: the latest outcome, whether it is terminal, the parsed verdict, the error,
                the usage and cost of the attempt that produced it
  attempts      every outcome of every attempt (custom_id, attempt, batch, outcome, error kind): retries never overwrite history
  events        a log of notable events (submits, refusals, splits, balance queueing) that progress.json summarizes
"""
from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  batch_key TEXT PRIMARY KEY, attempt INTEGER NOT NULL DEFAULT 0, parent TEXT, n_requests INTEGER NOT NULL,
  body_bytes INTEGER NOT NULL, est_cost REAL, state TEXT NOT NULL DEFAULT 'pending', batch_id TEXT, submit_attempts INTEGER NOT NULL DEFAULT 0,
  last_error_kind TEXT, last_error TEXT, next_attempt_ts REAL NOT NULL DEFAULT 0, submitted_utc TEXT, provider TEXT, remote_status TEXT,
  request_counts_json TEXT, usage_json TEXT, cost REAL, finished_utc TEXT, harvested_utc TEXT, created_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_batches_state ON batches(state);
CREATE TABLE IF NOT EXISTS requests (
  custom_id TEXT PRIMARY KEY, batch_key TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, outcome TEXT, terminal INTEGER NOT NULL DEFAULT 0,
  verdict_json TEXT, error_json TEXT, status_code INTEGER, finish_reason TEXT, usage_json TEXT, cost REAL, batch_id TEXT, provider TEXT,
  harvested_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_batch ON requests(batch_key);
CREATE INDEX IF NOT EXISTS idx_requests_outcome ON requests(outcome);
CREATE TABLE IF NOT EXISTS attempts (
  custom_id TEXT NOT NULL, attempt INTEGER NOT NULL, batch_key TEXT NOT NULL, batch_id TEXT, outcome TEXT NOT NULL, error_kind TEXT,
  harvested_utc TEXT NOT NULL, PRIMARY KEY (custom_id, attempt)
);
CREATE TABLE IF NOT EXISTS events (ts TEXT NOT NULL, kind TEXT NOT NULL, batch_key TEXT, message TEXT);
"""

PENDING_STATES = ("pending", "balance_queued")
INFLIGHT_STATES = ("submitted",)
#: batch states with work still to move (a harvester can advance them)
LIVE_STATES = ("pending", "balance_queued", "submitted", "submit_unknown")
PAUSE_FILE = "PAUSED"
#: signatures of identical-error floods a person has accepted as real per-item outcomes (one per line; ``#`` comments), :mod:`harvester`
ACK_FILE = "ACKNOWLEDGED"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class WriterLocked(RuntimeError):
    pass


class Ledger:
    """``Ledger(run_dir)`` opens (creating) ``state.sqlite``; ``Ledger(run_dir, readonly=True)`` opens it read-only.
    ``writer()`` takes the run directory's exclusive writer lock (non-blocking; :class:`WriterLocked` when another writer holds it)."""

    def __init__(self, run_dir: str, readonly: bool = False):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, "state.sqlite")
        self._lock_fh = None
        if readonly:
            self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=60)
        else:
            os.makedirs(run_dir, exist_ok=True)
            self.conn = sqlite3.connect(self.path, timeout=60)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        self.conn.row_factory = sqlite3.Row

    def writer(self) -> "Ledger":
        fh = open(os.path.join(self.run_dir, ".writer.lock"), "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.close()
            raise WriterLocked(f"another writer holds {self.run_dir}/.writer.lock (a harvester running?)")
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} since={utc_now()}\n")
        fh.flush()
        self._lock_fh = fh
        return self

    def close(self) -> None:
        self.conn.close()
        if self._lock_fh is not None:
            self._lock_fh.close()
            self._lock_fh = None

    # -- registration ----------------------------------------------------------------------------
    def register_manifest(self, man: Mapping[str, Any]) -> bool:
        """Insert a batch row and its request rows (a request already registered under another batch is re-pointed only when it has
        no terminal outcome: a retry batch takes over a request whose last outcome was retryable). Idempotent; returns True when new."""
        cur = self.conn.execute("SELECT 1 FROM batches WHERE batch_key = ?", (man["batch_key"],)).fetchone()
        if cur:
            return False
        est = (man.get("est") or {}).get("cost_usd_with_fee")
        self.conn.execute("INSERT INTO batches(batch_key, attempt, parent, n_requests, body_bytes, est_cost, state, created_utc) VALUES (?,?,?,?,?,?,?,?)",
                          (man["batch_key"], int(man.get("attempt") or 0), man.get("parent_batch_key"), int(man["n_requests"]),
                           int(man["body_bytes"]), est, "pending", utc_now()))
        for cid in man["custom_ids"]:
            self.conn.execute("INSERT INTO requests(custom_id, batch_key) VALUES (?, ?) ON CONFLICT(custom_id) DO UPDATE SET "
                              "batch_key = excluded.batch_key WHERE requests.terminal = 0", (cid, man["batch_key"]))
        self.conn.commit()
        return True

    def event(self, kind: str, message: str, batch_key: Optional[str] = None) -> None:
        self.conn.execute("INSERT INTO events(ts, kind, batch_key, message) VALUES (?,?,?,?)", (utc_now(), kind, batch_key, message[:4000]))
        self.conn.commit()

    # -- batches -------------------------------------------------------------------------------------
    def batches(self, states: Optional[Sequence[str]] = None) -> List[sqlite3.Row]:
        if states:
            q = f"SELECT * FROM batches WHERE state IN ({','.join('?' * len(states))}) ORDER BY attempt, batch_key"
            return self.conn.execute(q, tuple(states)).fetchall()
        return self.conn.execute("SELECT * FROM batches ORDER BY attempt, batch_key").fetchall()

    def update_batch(self, batch_key: str, **cols: Any) -> None:
        if not cols:
            return
        sets = ", ".join(f"{k} = ?" for k in cols)
        vals = [json.dumps(v) if isinstance(v, (dict, list)) else v for v in cols.values()]
        self.conn.execute(f"UPDATE batches SET {sets} WHERE batch_key = ?", (*vals, batch_key))
        self.conn.commit()

    # -- requests / verdict cache -----------------------------------------------------------------
    def record_outcome(self, custom_id: str, batch_key: str, batch_id: Optional[str], outcome: str, terminal: bool,
                       verdict: Optional[Mapping[str, Any]], error: Optional[Mapping[str, Any]], status_code: Optional[int],
                       finish_reason: Optional[str], usage: Optional[Mapping[str, Any]], cost: Optional[float],
                       provider: Optional[str], commit: bool = True) -> int:
        """Store one attempt's outcome; returns the attempt number (1-based)."""
        row = self.conn.execute("SELECT attempts FROM requests WHERE custom_id = ?", (custom_id,)).fetchone()
        attempt = (row["attempts"] if row else 0) + 1
        now = utc_now()
        self.conn.execute(
            "INSERT INTO requests(custom_id, batch_key, attempts, outcome, terminal, verdict_json, error_json, status_code, finish_reason, "
            "usage_json, cost, batch_id, provider, harvested_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(custom_id) DO UPDATE SET "
            "batch_key=excluded.batch_key, attempts=excluded.attempts, outcome=excluded.outcome, terminal=excluded.terminal, "
            "verdict_json=excluded.verdict_json, error_json=excluded.error_json, status_code=excluded.status_code, "
            "finish_reason=excluded.finish_reason, usage_json=excluded.usage_json, cost=excluded.cost, batch_id=excluded.batch_id, "
            "provider=excluded.provider, harvested_utc=excluded.harvested_utc",
            (custom_id, batch_key, attempt, outcome, 1 if terminal else 0, json.dumps(verdict) if verdict is not None else None,
             json.dumps(error) if error is not None else None, status_code, finish_reason, json.dumps(usage) if usage is not None else None,
             cost, batch_id, provider, now))
        self.conn.execute("INSERT OR REPLACE INTO attempts(custom_id, attempt, batch_key, batch_id, outcome, error_kind, harvested_utc) "
                          "VALUES (?,?,?,?,?,?,?)", (custom_id, attempt, batch_key, batch_id, outcome,
                                                     (error or {}).get("kind") if isinstance(error, Mapping) else None, now))
        if commit:
            self.conn.commit()
        return attempt

    def get(self, custom_id: str) -> Optional[Dict[str, Any]]:
        r = self.conn.execute("SELECT * FROM requests WHERE custom_id = ?", (custom_id,)).fetchone()
        return _decode(r) if r else None

    def iter_requests(self, outcome: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        q, a = "SELECT * FROM requests", ()
        if outcome:
            q, a = q + " WHERE outcome = ?", (outcome,)
        for r in self.conn.execute(q, a):
            yield _decode(r)

    def retryable(self) -> List[Tuple[str, str, int, str]]:
        """``(custom_id, batch_key, attempts, outcome)`` of every request whose latest outcome is not terminal (never harvested, or retryable)."""
        return [(r["custom_id"], r["batch_key"], r["attempts"], r["outcome"]) for r in
                self.conn.execute("SELECT custom_id, batch_key, attempts, outcome FROM requests WHERE terminal = 0 AND outcome IS NOT NULL")]

    def attempt_outcomes(self, custom_id: str) -> List[str]:
        """Every recorded outcome of one request, oldest first (the budget counters read them)."""
        return [str(r[0]) for r in self.conn.execute("SELECT outcome FROM attempts WHERE custom_id = ? ORDER BY attempt", (custom_id,))]

    def finalize(self, custom_id: str, note: str, commit: bool = True) -> None:
        """Make a request's CURRENT outcome terminal without a new attempt (a re-ask that cannot help, e.g. a truncated call whose cap
        cannot be raised); the note goes into its error and an event."""
        row = self.conn.execute("SELECT error_json FROM requests WHERE custom_id = ?", (custom_id,)).fetchone()
        err = json.loads(row["error_json"]) if row and row["error_json"] else {}
        err["finalized"] = note
        self.conn.execute("UPDATE requests SET terminal = 1, error_json = ? WHERE custom_id = ?", (json.dumps(err), custom_id))
        if commit:
            self.conn.commit()

    def outcome_counts(self) -> Dict[str, int]:
        out = {str(r[0]): int(r[1]) for r in self.conn.execute("SELECT COALESCE(outcome, 'not_harvested'), COUNT(*) FROM requests GROUP BY 1")}
        return out

    def state_counts(self) -> Dict[str, int]:
        return {str(r[0]): int(r[1]) for r in self.conn.execute("SELECT state, COUNT(*) FROM batches GROUP BY 1")}


def pause(run_dir: str, ledger: Optional[Ledger], reason: str, log=None) -> bool:
    """Write ``<run_dir>/PAUSED`` (first reason wins; returns True when this call wrote it): nothing is submitted or re-asked until a
    person removes the file. The harvester keeps polling and harvesting what is already in flight."""
    path = os.path.join(run_dir, PAUSE_FILE)
    if os.path.exists(path):
        return False
    with open(path, "w") as fh:
        fh.write(f"{utc_now()} {reason}\nRemove this file to resume submitting (the harvester keeps polling and harvesting).\n")
    if ledger is not None:
        ledger.event("PAUSED", reason)
    if log is not None:
        log(f"[harvester] PAUSED: {reason}")
    return True


def paused(run_dir: str) -> Optional[str]:
    path = os.path.join(run_dir, PAUSE_FILE)
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return None


def acknowledged(run_dir: str, extra: Iterable[str] = ()) -> set:
    """The error signatures a person accepted (``<run_dir>/ACKNOWLEDGED``, plus ``track.json`` ``health.acknowledged``)."""
    out = {str(x).strip() for x in extra if str(x).strip()}
    path = os.path.join(run_dir, ACK_FILE)
    if os.path.exists(path):
        with open(path) as fh:
            out |= {l.split("#", 1)[0].strip() for l in fh if l.split("#", 1)[0].strip()}
    return out


def readiness(run_dir: str) -> Dict[str, Any]:
    """Whether a run directory has STOPPED MOVING and can be filed (read-only): ``ready`` is true only when it is not paused, no batch
    is pending / queued / in flight / unknown, and every registered request has a terminal outcome. ``refused`` batches are reported
    separately (``ready_if_partial`` ignores them and the requests they leave unjudged). ``fingerprint`` changes whenever an outcome
    does (a filing staged earlier checks it before delivery)."""
    led = Ledger(run_dir, readonly=True)
    try:
        c = led.conn
        states = led.state_counts()
        n_req, n_term, n_none = c.execute("SELECT COUNT(*), COALESCE(SUM(terminal), 0), COALESCE(SUM(outcome IS NULL), 0) FROM requests").fetchone()
        nonterm = {str(r[0]): int(r[1]) for r in c.execute("SELECT COALESCE(outcome, 'not_harvested'), COUNT(*) FROM requests WHERE terminal = 0 GROUP BY 1")}
        refused_keys = [r["batch_key"] for r in led.batches(("refused",))]
        in_refused = 0
        if refused_keys:
            q = f"SELECT COUNT(*) FROM requests WHERE terminal = 0 AND batch_key IN ({','.join('?' * len(refused_keys))})"
            in_refused = int(c.execute(q, tuple(refused_keys)).fetchone()[0])
        last = c.execute("SELECT COALESCE(MAX(harvested_utc), ''), COUNT(*) FROM attempts").fetchone()
    finally:
        led.close()
    live = {k: v for k, v in states.items() if k in LIVE_STATES and v}
    p = paused(run_dir)
    nonterm_total = int(n_req) - int(n_term)
    ready = not p and not live and nonterm_total == 0 and not refused_keys
    ready_if_partial = not p and not live and nonterm_total == in_refused
    return {"run_dir": run_dir, "ready": ready, "ready_if_partial": ready_if_partial, "paused": p, "live_batches": live,
            "refused_batches": refused_keys, "requests": int(n_req), "terminal": int(n_term), "non_terminal": nonterm,
            "non_terminal_in_refused": in_refused, "never_harvested": int(n_none),
            "fingerprint": f"{n_req}:{n_term}:{last[1]}:{last[0]}"}


def _decode(r: sqlite3.Row) -> Dict[str, Any]:
    d = dict(r)
    for k in ("verdict_json", "error_json", "usage_json"):
        if d.get(k):
            d[k[:-5]] = json.loads(d[k])
        else:
            d[k[:-5]] = None
    return d
