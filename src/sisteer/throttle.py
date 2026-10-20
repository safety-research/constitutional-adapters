"""Shared 429 policy: throttle DOWN, never switch org.

A standing rule (2026-08-21, floor amended 2026-08-27) replaced the older
"429 -> High-Prio fallback" behaviour and said not to reinstate it. At the
widths this project runs (200-500 concurrent) a saturated Low-Prio pool is
most likely OUR doing; moving that load onto a second shared pool relocates
the contention onto the pool's other users instead of easing it. The remedy is to
take up less room: back off on the SAME org and cut how many requests we
have in flight.

This module exists because the policy has to hold at EVERY call site, not
just the one that was being read when the rule came up. There are two live
transports in this repo -- ``capeval.judging.ClaudeJudge`` and
``eval.attacks.grading_live.LiveStrongREJECTGrader`` -- and only the first
one implemented any org policy at all. A footgun fixed at one site and
documented in prose recurs; a shared gate plus a test does not.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from sisteer import vault

# Never throttle below this. Below ~50 concurrent the run stops being able to
# finish in reasonable wall-clock, and a 429 means the pool is busy, not that
# we must grind to a halt.
DEFAULT_FLOOR = 50

_SLEEP_S = 5
_MAX_SLEEP_S = 120
_MAX_ATTEMPTS = 8


class AdaptiveGate:
    """A concurrency ceiling that only ever ratchets DOWN, on 429s.

    Backoff alone spaces out ONE worker's retries; it does not reduce how
    many workers are hitting the pool. Cutting the ceiling is the part that
    actually shrinks our footprint.
    """

    def __init__(self, initial: int, floor: int = DEFAULT_FLOOR) -> None:
        initial = max(1, int(initial))
        # A caller asking for less than the floor gets what it asked for --
        # the floor bounds THROTTLING, it does not raise a deliberate choice.
        self.floor = max(1, min(int(floor), initial))
        self.live = initial
        self.n_rate_limited = 0
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(initial)
        self._debt = 0

    def acquire(self) -> None:
        self._sem.acquire()

    def release(self) -> None:
        """Hand the permit back, unless we owe it to a throttle-down.

        Swallowing the permit is what makes a cut PERMANENT. Merely acquiring
        permits at throttle time would not work: a permit held by a running
        worker is handed straight back when that worker finishes, and the
        ceiling silently climbs again.
        """
        with self._lock:
            if self._debt > 0:
                self._debt -= 1
                return
        self._sem.release()

    def note_rate_limit(self) -> None:
        with self._lock:
            self.n_rate_limited += 1
            new = max(self.floor, self.live // 2)
            if new < self.live:
                self._debt += self.live - new
                self.live = new
                print(f"[throttle] 429: cutting in-flight to {new} "
                      f"(floor {self.floor}); staying on the same org",
                      flush=True)


class _ThrottledMessages:
    def __init__(self, inner: Any, gate: AdaptiveGate, latch: "vault.CredentialLatch") -> None:
        self._inner, self._gate, self._latch = inner, gate, latch

    def create(self, **params):
        for attempt in range(_MAX_ATTEMPTS):
            # the run's credential latch, BEFORE EVERY attempt: the first, and each wake from the
            # 429 backoff below -- a worker asleep here when a sibling's request met the run's first 401 / 403 used to wake
            # into another request on the rejected credential; now it wakes into the typed stop, no request
            self._latch.check()
            try:
                return self._inner.create(**params)
            except Exception as e:                        # noqa: BLE001
                held = self._latch.rejection(e)           # the credential FIRST: typed, held for every sibling layer, raised
                if held is not None:
                    raise held from e
                if getattr(e, "status_code", None) != 429:
                    raise
                self._gate.note_rate_limit()
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise
                time.sleep(min(_SLEEP_S * (2 ** attempt), _MAX_SLEEP_S))
        raise AssertionError("unreachable")


class ThrottledClient:
    """Wrap an Anthropic client so 429s throttle the gate and get retried.

    Wrapping the CLIENT rather than each call site means the grading scripts
    need no change: they already call ``client.messages.create``.

    The wrapper also carries the run's credential latch:
    ``latch`` when the caller hands one (a grader's, so a later wrapper swap
    over the same inner client keeps it), else the inner client's own, else a
    new one attached to the inner client (:func:`sisteer.vault.client_credential_latch`).
    Every ``create`` checks it before EVERY attempt, the first rejection the
    SDK raises is classified here and held as the typed error, and the same
    latch is what ``grade_bon_live._forced_tool_grade`` reads off the client
    before each of ITS attempts -- so no layer of the retry stack issues a
    request after the run's first 401 / 403. The SDK's own retries must be
    OFF on the client (they sleep and re-send below any latch).
    """

    def __init__(self, inner: Any, gate: AdaptiveGate, *, latch: "vault.CredentialLatch | None" = None) -> None:
        self._inner = inner
        self.gate = gate
        self._messages: Any = None
        if latch is None:
            latch = vault.client_credential_latch(inner, create=True)
        else:
            vault.attach_credential_latch(inner, latch)
        self.credential_latch = latch
        setattr(self, vault.CLIENT_LATCH_ATTR, latch)     # the wrapper's own, whatever the inner accepts: client_credential_latch(wrapper)

    @property
    def messages(self) -> Any:
        # Resolved LAZILY, not in __init__. Test doubles substitute a bare
        # stub for the client and only grow a `.messages` when a call is
        # actually made, so touching it at construction turned every such
        # test into an AttributeError at setup — a wrapper should not impose
        # requirements the thing it wraps only meets at call time.
        if self._messages is None:
            self._messages = _ThrottledMessages(self._inner.messages, self.gate, self.credential_latch)
        return self._messages

    def __getattr__(self, name: str) -> Any:
        # Private names never forward: __getattr__ fires before _inner exists
        # during unpickling/early access and would recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_inner"], name)


# --------------------------------------------------------------------------- #
# FloorWatch: the ONE thing left to decide once the gate is at its floor       #
# --------------------------------------------------------------------------- #

#: seconds without a completed judgment, with the gate at its floor, before the
#: run stops and reports. 15 min: long enough that a slow-but-serving pool is
#: not mistaken for a dead one; short enough that a dead one is noticed inside
#: one polling cycle of the person who launched it.
DEFAULT_STALL_AT_FLOOR_S = 900.0
#: the exit status a stopped run reports (the driver resumes later from its
#: append-only cache; the org is NOT switched)
FLOOR_STALL_EXIT = 3


class FloorWatch(threading.Thread):
    """Watch a grader's :class:`AdaptiveGate` and stop the run when it is
    stalled at the floor. The gate already halves the in-flight ceiling on
    every 429 down to its floor, on the SAME org. This thread adds the one
    decision that remains: if the ceiling sits at the floor and NO judgment
    has completed for ``stall_s`` seconds, stop the run and report (exit
    :data:`FLOOR_STALL_EXIT`) — never swap the key. It replaces the
    org-switching escalator the full22 graders used to carry (deleted, not
    disabled: the rule is throttle down on the same org, never switch
    pools), and it was copied verbatim into six lane graders before it lived
    here (2026-09-09).

    ``progress`` is a zero-arg callable returning a monotone count of
    completed judgments — the append-only judge cache's row count is the one
    the graders use, because it is the same record the run resumes from.
    ``org`` is a zero-arg callable naming the pool, for the event log only.
    ``stop`` is what fires at the stall (default ``os._exit(exit_code)``
    from the watch thread: the run is wedged behind a pool that is not
    serving, and a cooperative stop would wait on it); tests pass a stub.
    The trigger is a CLOCK on completions, not a count of 429s — a pool
    that sheds 529-shaped, or simply stops answering, stalls the clock just
    the same. Every event is appended to ``events`` and printed.
    """

    def __init__(self, gate: AdaptiveGate, progress, *, initial: int | None = None,
                 org=lambda: "?", stall_s: float = DEFAULT_STALL_AT_FLOOR_S,
                 poll_s: float = 5.0, stop=None, clock=time.time,
                 exit_code: int = FLOOR_STALL_EXIT, now_iso=None) -> None:
        super().__init__(daemon=True)
        self.gate, self.progress, self.org = gate, progress, org
        self.initial = int(initial if initial is not None else gate.live)
        self.stall_s, self.poll_s, self.clock = float(stall_s), float(poll_s), clock
        self.exit_code = int(exit_code)
        self._stop_fn = stop if stop is not None else self._hard_exit
        self._now_iso = now_iso or _utc_now_iso
        self.events: list = [{"t": self._now_iso(), "event": "start", "org": self.org(),
                              "concurrency": self.initial,
                              "rule": (f"429 -> halve on the same org to floor; at floor with no "
                                       f"completion for {self.stall_s:.0f}s -> stop and report")}]
        self.escalated = False    # kept for the provenance schema; always False — there is no escalator
        self.stalled = False
        self._halt = threading.Event()

    def _hard_exit(self) -> None:
        import os
        import sys
        sys.stdout.flush()
        os._exit(self.exit_code)

    def stop(self) -> None:
        """End the watch (the run finished on its own)."""
        self._halt.set()

    def check(self, *, state: dict) -> bool:
        """ONE poll: record a 429 count change, reset the stall clock on
        progress, and fire ``stop`` when the gate is at the floor with no
        completion for ``stall_s``. Returns True when it fired. Exposed so a
        test can drive the clock instead of sleeping."""
        gate = self.gate
        n, rows, now = gate.n_rate_limited, int(self.progress()), self.clock()
        if n != state["n429"]:
            self.events.append({"t": self._now_iso(), "event": "429", "org": self.org(), "n_429": n,
                                "ceiling": gate.live, "floor": gate.floor})
            print(f"[org] 429s={n} on {self.org()}; in-flight ceiling now {gate.live} "
                  f"(floor {gate.floor}) — same org, no switch", flush=True)
            state["n429"] = n
        if rows != state["rows"]:
            state["rows"], state["progress_t"] = rows, now
        idle = now - state["progress_t"]
        if gate.live <= gate.floor and idle > self.stall_s:
            ev = {"t": self._now_iso(), "event": "STALLED_AT_FLOOR", "org": self.org(), "n_429": n,
                  "ceiling": gate.live, "floor": gate.floor,
                  "seconds_without_completion": round(idle), "cache_rows": rows,
                  "action": (f"stop and report (exit {self.exit_code}); resume later from the "
                             "append-only cache; the org is NOT switched")}
            self.events.append(ev)
            self.stalled = True
            print(f"[org] STALLED AT FLOOR: ceiling {gate.live} = floor, no judgment completed for "
                  f"{idle:.0f}s, 429s={n}; stopping and reporting rather than switching org "
                  f"(user rule). {ev}", flush=True)
            self._stop_fn()
            return True
        return False

    def run(self) -> None:
        state = {"n429": self.gate.n_rate_limited, "rows": int(self.progress()),
                 "progress_t": self.clock()}
        while not self._halt.wait(self.poll_s):
            if self.check(state=state):
                return


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
