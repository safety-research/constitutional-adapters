"""429 -> throttle down on the SAME org, never a hop, never below the floor.

The policy replaced "429 -> High-Prio fallback" and must not be reinstated;
a floor was added later: we should not generally need to drop below 50
concurrent. These tests pin BOTH live transports to that policy, because the
repo had two and only one of them implemented any org policy at all.
"""
import pathlib
import sys
import threading

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sisteer.throttle import (                            # noqa: E402
    DEFAULT_FLOOR, AdaptiveGate, ThrottledClient)


class _Boom(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status_code = status


def test_floor_is_fifty():
    assert DEFAULT_FLOOR == 50, (
        "the user set the throttle floor at 50 on 2026-08-27; a lower floor "
        "makes a busy pool grind the run to a halt")


def test_halves_then_stops_at_the_floor():
    g = AdaptiveGate(500)
    g.note_rate_limit(); assert g.live == 250
    g.note_rate_limit(); assert g.live == 125
    g.note_rate_limit(); assert g.live == 62
    g.note_rate_limit(); assert g.live == 50      # 31 would breach the floor
    g.note_rate_limit(); assert g.live == 50      # and it stays there
    assert g.n_rate_limited == 5


def test_a_deliberate_small_concurrency_is_not_raised_to_the_floor():
    """The floor bounds THROTTLING; it must not override an explicit choice."""
    g = AdaptiveGate(8)
    assert g.live == 8
    g.note_rate_limit()
    assert g.live == 8


def test_throttle_down_is_permanent():
    g = AdaptiveGate(100)
    g.acquire()
    g.note_rate_limit()          # 100 -> 50, owing 50 permits
    g.release()                  # swallowed, not handed back
    assert g.live == 50 and g._debt == 49


def test_gate_actually_bounds_in_flight():
    g = AdaptiveGate(2, floor=1)
    g.acquire(); g.acquire()
    blocked = threading.Event()

    def third():
        g.acquire()
        blocked.set()

    t = threading.Thread(target=third, daemon=True); t.start()
    assert not blocked.wait(0.3), "a third worker got in past a ceiling of 2"
    g.release()
    assert blocked.wait(2), "releasing a permit did not admit the waiter"


def test_throttled_client_retries_429_and_feeds_the_gate(monkeypatch):
    import sisteer.throttle as th
    monkeypatch.setattr(th, "_SLEEP_S", 0)
    monkeypatch.setattr(th, "_MAX_SLEEP_S", 0)
    state = {"n": 0}

    class _M:
        def create(self, **params):
            state["n"] += 1
            if state["n"] < 3:
                raise _Boom(429)
            return "ok"

    class _C:
        messages = _M()

    g = AdaptiveGate(200)
    out = ThrottledClient(_C(), g).messages.create(model="m")
    assert out == "ok"
    assert g.n_rate_limited == 2
    assert g.live == 50          # 200 -> 100 -> 50


def test_throttled_client_reraises_non_429():
    class _M:
        def create(self, **params):
            raise _Boom(400)

    class _C:
        messages = _M()

    g = AdaptiveGate(200)
    with pytest.raises(_Boom):
        ThrottledClient(_C(), g).messages.create(model="m")
    assert g.n_rate_limited == 0


def test_live_grader_wraps_its_client_and_caps_sdk_retries():
    """The grading path is the one that actually runs; it must be gated too.

    Source-level, because constructing the grader reads a real key file.
    """
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src/sisteer/eval/attacks/grading_live.py").read_text()
    assert "ThrottledClient(" in src, "live grader is not gated"
    assert "self._gate.acquire()" in src, "live grader never holds a permit"
    assert "self._sdk_max_retries = min(int(max_retries), 2)" in src, (
        "SDK max_retries is not capped: a retrying client put thousands of attempts in flight during a rate-limit event and hid "
        "every 429 from the throttle (the base's clamp to 2; WIF round 12 on fd06740ae restored it in every mode after round 11 "
        "had set 0 in op / env too -- under wif the VAULT turns the SDK's own retries off and carries the count in its latched "
        "layer, tests/test_wif_federated_destination_and_sdk_retries.py)")
    assert "make_client(self.org, max_retries=self._sdk_max_retries)" in src, "the client is built with the clamp, as the base did"
    assert "max_retries: int = 8" in src, "the base's default"


# -- FloorWatch: stop-and-report at the floor, never a pool switch -------------


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _watch(gate, rows, clock, **kw):
    from sisteer.throttle import FloorWatch
    fired = []
    w = FloorWatch(gate, lambda: rows[0], org=lambda: "lowprio", stall_s=900.0,
                   poll_s=0.01, stop=lambda: fired.append(True), clock=clock, **kw)
    return w, fired


def test_floorwatch_fires_at_the_floor_on_a_completion_clock_not_a_429_count():
    """Independent of 429 cadence: the trigger is the gate AT ITS FLOOR with no
    completed judgment for stall_s. No 429 needs to arrive during the stall
    (a pool that sheds 529-shaped or goes silent stalls the clock just the
    same), and a burst of 429s does NOT fire it while completions keep coming."""
    from sisteer.throttle import FLOOR_STALL_EXIT, FloorWatch
    g = AdaptiveGate(200, floor=50)
    for _ in range(3):
        g.note_rate_limit()
    assert g.live == 50
    rows, clock = [10], _Clock()
    w, fired = _watch(g, rows, clock)
    state = {"n429": g.n_rate_limited, "rows": rows[0], "progress_t": clock()}
    # at the floor, but completions keep landing: never fires, however long
    for _ in range(50):
        clock.t += 600; rows[0] += 1
        assert w.check(state=state) is False
    assert not fired
    # completions stop; no further 429 arrives; the clock alone decides
    clock.t += 899
    assert w.check(state=state) is False
    clock.t += 2
    assert w.check(state=state) is True
    assert fired == [True] and w.stalled and w.escalated is False
    ev = w.events[-1]
    assert ev["event"] == "STALLED_AT_FLOOR" and ev["org"] == "lowprio"
    assert ev["ceiling"] == 50 == ev["floor"] and ev["seconds_without_completion"] == 901
    assert f"exit {FLOOR_STALL_EXIT}" in ev["action"] and "NOT switched" in ev["action"]
    assert FLOOR_STALL_EXIT == 3
    assert FloorWatch.__mro__[1] is threading.Thread and w.daemon


def test_floorwatch_does_not_fire_above_the_floor_and_records_429s_as_events():
    """Silence ABOVE the floor is not a stall (the pool is answering slowly or
    the run is between phases); the watch only records 429 changes there."""
    g = AdaptiveGate(200, floor=50)
    rows, clock = [0], _Clock()
    w, fired = _watch(g, rows, clock)
    state = {"n429": 0, "rows": 0, "progress_t": clock()}
    clock.t += 5000
    assert w.check(state=state) is False and not fired
    g.note_rate_limit()                                  # 200 -> 100: still above the floor
    clock.t += 5000
    assert w.check(state=state) is False and not fired
    assert [e["event"] for e in w.events] == ["start", "429"]
    assert w.events[-1]["ceiling"] == 100 and w.events[-1]["n_429"] == 1
    # the start event carries the rule in words and the initial width
    assert w.events[0]["concurrency"] == 200 and "same org" in w.events[0]["rule"]


def test_floorwatch_thread_runs_and_stops_cooperatively_when_the_run_ends():
    g = AdaptiveGate(8, floor=8)
    rows = [0]
    w, fired = _watch(g, rows, _Clock())
    w.start()
    w.stop()
    w.join(timeout=2)
    assert not w.is_alive() and not fired
