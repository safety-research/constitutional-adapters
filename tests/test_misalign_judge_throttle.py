"""The misalign judge's pool policy: on sustained 429s THROTTLE DOWN on
the same Low-Prio pool -- halve the in-flight width (floor 2), restart the clock, and at the floor surface a PROBLEM event when no
progress is made -- never a hop to High Prio. A sustained 529 episode is provider capacity: no shrink, a problem event. Synthetic
clients and judges; no API call."""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

from sisteer.eval.api_keys import AdaptiveLimiter

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_judge.py"


def _load():
    spec = importlib.util.spec_from_file_location("misalign_judge_throttle_under_test", SRC)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def _client(pool="lowprio"):
    return SimpleNamespace(lock=threading.Lock(), pool=pool, episode_start=None, episode_kind=None, windows={}, events=[], n_429=0, n_529=0)


def _episode(client, kind, start, last=None):
    """Open a sustained window of ``kind`` on the synthetic client: first error at ``start``, most recent at ``last`` (= start)."""
    client.episode_start = start if client.episode_start is None else min(client.episode_start, start)
    client.episode_kind = kind
    client.windows[kind] = {"start": start, "last": start if last is None else last}


def test_a_sustained_429_episode_halves_the_width_to_the_floor_then_surfaces_a_problem_and_never_switches_pools():
    mj = _load()
    judge = SimpleNamespace(limiter=AdaptiveLimiter(20, min_width=mj.THROTTLE_FLOOR))
    original_limiter = judge.limiter
    client = _client()
    thr = mj.Throttler(judge, client, after_s=300.0, floor=mj.THROTTLE_FLOOR)
    assert thr.step(10.0) is None                                             # no episode: nothing
    _episode(client, "429", 0.0, last=250.0)                                  # 429s keep arriving (fresh evidence)
    assert thr.step(100.0) is None                                            # not sustained yet
    ev = thr.step(301.0)
    assert ev["event"] == mj.THROTTLED_DOWN and (ev["from"], ev["to"]) == (20, 10) and judge.limiter.width == 10 and client.windows["429"]["start"] == 301.0
    assert judge.limiter is original_limiter and judge.limiter.launch_width == 20     # g8-fix: shrunk IN PLACE, the launch ceiling kept
    assert thr.step(500.0) is None                                            # the clock restarted: another full after_s is needed
    client.windows["429"]["last"] = 600.0; assert thr.step(602.0)["to"] == 5 and judge.limiter.width == 5
    client.windows["429"]["last"] = 900.0; assert thr.step(903.0)["to"] == 2 and judge.limiter.width == 2            # the floor
    client.windows["429"]["last"] = 1200.0; problem = thr.step(1204.0)
    assert problem["event"] == mj.NO_PROGRESS_AT_FLOOR and problem["problem"] is True and problem["width"] == 2 and judge.limiter.width == 2
    client.windows["429"]["last"] = 1500.0
    assert thr.step(1300.0) is None and thr.step(1505.0)["event"] == mj.NO_PROGRESS_AT_FLOOR      # once per sustained window, bounded
    assert client.pool == "lowprio" and thr.halvings == 3 and len(thr.problems) == 2
    assert [e["event"] for e in client.events] == [mj.THROTTLED_DOWN] * 3 + [mj.NO_PROGRESS_AT_FLOOR] * 2
    # the judge's own limiter still halves per 429 burst on the same pool, flooring at 2
    judge.limiter.on_rate_limit(); assert judge.limiter.width == 2
    # the episode ends on a success: nothing more fires
    client.episode_start, client.episode_kind, client.windows = None, None, {}
    assert thr.step(2000.0) is None


def test_a_sustained_529_episode_is_provider_capacity_no_shrink_one_problem_per_window():
    mj = _load()
    judge = SimpleNamespace(limiter=AdaptiveLimiter(20, min_width=mj.THROTTLE_FLOOR))
    client = _client(); _episode(client, "529", 0.0, last=290.0)
    thr = mj.Throttler(judge, client, after_s=300.0)
    ev = thr.step(301.0)
    assert ev["event"] == mj.PROVIDER_OVERLOADED and ev["problem"] is True and judge.limiter.width == 20 and thr.halvings == 0
    client.windows["529"]["last"] = 600.0
    assert thr.step(400.0) is None and thr.step(602.0)["event"] == mj.PROVIDER_OVERLOADED and client.pool == "lowprio"


def test_a_run_started_on_high_prio_by_explicit_request_is_left_alone():
    mj = _load()
    judge = SimpleNamespace(limiter=AdaptiveLimiter(8, min_width=mj.THROTTLE_FLOOR))
    client = _client(pool="highprio"); _episode(client, "429", 0.0, last=290.0)
    thr = mj.Throttler(judge, client, after_s=300.0)
    assert thr.step(301.0) is None and judge.limiter.width == 8                 # the policy governs Low Prio; the operator chose this pool


def test_the_parser_keeps_the_old_flags_as_inert_aliases_and_the_module_never_switches_to_high_prio():
    mj = _load()
    ap = mj.build_parser()
    a = ap.parse_args(["--results-root", "/synthetic/root", "--target", "synthetic_target", "--no-escalate", "--escalate-after-s", "12", "--escalate-cap", "7"])
    assert a.throttle_after_s == 12.0 and a.no_escalate is True
    b = ap.parse_args(["--results-root", "/synthetic/root", "--target", "synthetic_target", "--throttle-after-s", "45"])
    assert b.throttle_after_s == 45.0 and b.pool == "lowprio"
    src = SRC.read_text(encoding="utf-8")
    assert 'switch("highprio")' not in src and "ESCALATED_TO_HIGHPRIO" not in src and "class Escalator" not in src
    assert mj.THROTTLE_FLOOR == 2
    prov = mj.throttle_provenance(mj.Throttler(SimpleNamespace(limiter=AdaptiveLimiter(4)), _client(), after_s=300.0), b)
    assert prov["throttle"]["floor"] == 2 and prov["throttle"]["halvings"] == 0 and prov["throttle"]["problems"] == []
    assert prov["escalation"] == {"enabled": False, "escalated": False, "retired": mj.ESCALATION_RETIRED}


def test_the_in_place_shrink_keeps_the_launch_ceiling_so_recovery_is_slow_and_additive(monkeypatch):
    """Replacing the limiter set the recovery ceiling to the reduced width for good (20 -> 10, then 100 clean
    admissions, width 10 forever). The in-place shrink keeps launch_width 20: after the cooldown, one full window of clean completions
    at the reduced width earns +1, and the width climbs back toward 20 -- slowly, never in one jump."""
    mj = _load()
    lim = AdaptiveLimiter(20, min_width=mj.THROTTLE_FLOOR, cooldown=0.0, recover_floor=1)
    judge = SimpleNamespace(limiter=lim); client = _client(); _episode(client, "429", 0.0, last=290.0)
    thr = mj.Throttler(judge, client, after_s=300.0)
    assert thr.step(301.0)["to"] == 10 and lim.width == 10 and lim.launch_width == 20 and judge.limiter is lim
    client.episode_start = None; client.windows = {}                            # the pool recovered: successes again
    widths = []
    for _ in range(10 * 20):                                                   # clean completions AT the reduced width
        with lim:
            pass
        widths.append(lim.width)
    assert widths[0] == 10 and max(widths) == 20 and widths == sorted(widths)   # climbs back one at a time, never above 20
    assert widths.index(11) == 9                                               # +1 exactly after a full window (10 completions) at width 10
    # a fresh limiter (the old behaviour) would have been pinned: its launch width IS the reduced width
    pinned = AdaptiveLimiter(10, min_width=2, cooldown=0.0, recover_floor=1)
    for _ in range(200):
        with pinned:
            pass
    assert pinned.width == 10


def test_queued_workers_span_the_shrink_on_one_gate():
    """The workers blocked on the limiter before the shrink stay on the SAME gate: after an in-place shrink to 2, at most 2 run at once
    -- a fresh limiter would have let the old gate's queue drain alongside the new one."""
    import threading, time as _t
    mj = _load()
    lim = AdaptiveLimiter(8, min_width=mj.THROTTLE_FLOOR)
    judge = SimpleNamespace(limiter=lim); client = _client(); _episode(client, "429", 0.0, last=900.0)
    thr = mj.Throttler(judge, client, after_s=300.0)
    guard = threading.Lock(); state = {"inside": 0, "peak_after_shrink": 0, "shrunk": False}
    release = threading.Event()

    def worker():
        with lim:
            with guard:
                state["inside"] += 1
                if state["shrunk"]:
                    state["peak_after_shrink"] = max(state["peak_after_shrink"], state["inside"])
            release.wait(2.0); _t.sleep(0.01)
            with guard:
                state["inside"] -= 1
    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    _t.sleep(0.2)                                                              # 8 inside, 8 queued on the gate
    assert thr.step(301.0)["to"] == 4 and lim.width == 4 and thr.step(602.0)["to"] == 2 and lim.width == 2
    with guard:
        state["shrunk"] = True
    release.set()
    for t in threads:
        t.join(5.0)
    assert state["peak_after_shrink"] <= 2, state                               # the queued workers admitted under the reduced width only


def test_an_error_kind_transition_restarts_the_sustained_window():
    """PoolClient latched the episode kind to the FIRST error, so one 429 followed by only 529s read as a
    'continuous 429' at 300 s and the throttler halved again. Now a kind transition ends the episode and starts one of the new kind:
    429 -> 529s is a 529 episode (no shrink, a PROVIDER_OVERLOADED problem after its own 300 s); 529 -> 429 starts the 429 clock at the
    first 429; pure 529 never shrinks."""
    mj = _load()
    import time as _t
    client = mj.PoolClient.__new__(mj.PoolClient)                               # the record() logic without an API client
    import threading
    client.lock = threading.Lock(); client.pool = "lowprio"; client.n_429 = client.n_529 = client.n_ok = 0
    client.episode_start = None; client.episode_kind = None; client.windows = {}; client.window_s = 300.0; client.events = []
    judge = SimpleNamespace(limiter=AdaptiveLimiter(20, min_width=mj.THROTTLE_FLOOR))
    thr = mj.Throttler(judge, client, after_s=300.0)
    base = _t.monotonic()
    client.record("429")
    for _ in range(20):
        client.record("529")
    assert client.episode_kind == "529" and client.n_429 == 1 and client.n_529 == 20
    assert [e["event"] for e in client.events] == ["episode_start", "episode_kind_change", "episode_start"]
    assert client.events[1]["from"] == "429" and client.events[1]["to"] == "529"
    t0 = client.windows["429"]["start"]
    client.windows["529"]["last"] = t0 + 290.0                                 # the 529s keep coming; the lone 429 does not
    ev = thr.step(t0 + 301.0)
    assert ev["event"] == mj.PROVIDER_OVERLOADED and judge.limiter.width == 20 and thr.halvings == 0   # no 429 shrink: no fresh 429 evidence
    # 529 -> 429: the 429 clock starts at the first 429 (its own window), the 529 window keeps its own start
    client.record("ok"); client.events.clear()
    client.record("529"); t529 = client.windows["529"]["start"]
    client.record("429"); t429 = client.windows["429"]["start"]
    assert client.episode_kind == "429" and t429 >= t529 and [e["event"] for e in client.events][-2:] == ["episode_kind_change", "episode_start"]
    client.windows["429"]["last"] = t429 + 290.0; client.windows["529"]["last"] = t529
    assert thr.step(t429 + 100.0) is None and thr.step(t429 + 301.0)["event"] == mj.THROTTLED_DOWN and judge.limiter.width == 10
    # pure 529: never a shrink
    client.record("ok"); judge2 = SimpleNamespace(limiter=AdaptiveLimiter(20, min_width=2)); thr2 = mj.Throttler(judge2, client, after_s=300.0)
    for _ in range(5):
        client.record("529")
    client.windows["529"]["last"] = client.windows["529"]["start"] + 290.0
    assert thr2.step(client.windows["529"]["start"] + 301.0)["event"] == mj.PROVIDER_OVERLOADED and judge2.limiter.width == 20


def test_alternating_kinds_with_no_success_still_reach_the_window_for_both_kinds():
    """Every 429<->529 transition discarded the previous kind's window start, so a saturated pool returning 429s
    with 529s interspersed (no successes) never reached after_s for either kind -- the limiter fell to the floor by per-burst halving
    while halvings = problems = 0 and the judge sat stranded without NO_PROGRESS_AT_FLOOR / PROVIDER_OVERLOADED. Per-kind windows now
    survive the switches until a success: 429@0, 529@250, 429@251, 529@500, 429@501, 529@750, 429@751 -> at 751 both kinds are
    sustained and fresh, so the 429 window shrinks (or files the floor problem) AND the 529 window files PROVIDER_OVERLOADED."""
    mj = _load()
    import threading
    client = mj.PoolClient.__new__(mj.PoolClient)
    client.lock = threading.Lock(); client.pool = "lowprio"; client.n_429 = client.n_529 = client.n_ok = 0
    client.episode_start = None; client.episode_kind = None; client.windows = {}; client.window_s = 300.0; client.events = []
    judge = SimpleNamespace(limiter=AdaptiveLimiter(20, min_width=mj.THROTTLE_FLOOR))
    thr = mj.Throttler(judge, client, after_s=300.0)
    import time as _t
    base = _t.monotonic()
    schedule = [("429", 0.0), ("529", 250.0), ("429", 251.0), ("529", 500.0), ("429", 501.0), ("529", 750.0), ("429", 751.0)]
    for kind, at in schedule:
        opened = client.windows.get(kind, {}).get("start") is None             # this error opens the kind's window
        client.record(kind)
        if opened:
            client.windows[kind]["start"] = base + at                           # the synthetic clock: the first error of the kind
        client.windows[kind]["last"] = base + at                                # ... and its most recent one
    assert client.windows["429"]["start"] == base + 0.0 and client.windows["529"]["start"] == base + 250.0
    assert thr.step(base + 299.0) is None                                       # neither kind sustained yet
    ev = thr.step(base + 751.0)
    assert ev["event"] == mj.THROTTLED_DOWN and (ev["from"], ev["to"]) == (20, 10) and thr.halvings == 1
    kinds = [e["event"] for e in client.events if e["event"] in (mj.THROTTLED_DOWN, mj.PROVIDER_OVERLOADED, mj.NO_PROGRESS_AT_FLOOR)]
    assert kinds == [mj.THROTTLED_DOWN, mj.PROVIDER_OVERLOADED] and len(thr.problems) == 1      # BOTH kinds acted in the one step
    assert client.windows["429"]["start"] == base + 751.0 and client.windows["529"]["start"] == base + 751.0   # both windows restarted
    client.record("ok")
    assert client.windows == {} and thr.step(base + 2000.0) is None             # a success closes every window


def test_a_kinds_window_expires_on_a_same_kind_gap_longer_than_the_throttle_window():
    """A per-kind window start was never reset after its OWN error gap exceeded after_s -- 429@0, 529@100/400/700/999,
    429@1000: one fresh 429 at t1000 fired THROTTLED_DOWN at once, claiming 1,000 s of continuous 429 pressure. Now the kind's window
    expires when its next error follows the last same-kind error by more than window_s (= the throttle's after_s): the 429 at 1000 opens a
    NEW window (no action at 1000; an action only after another sustained, fresh 300 s), while 429 -> 529 -> 429 within the window keeps
    the first start."""
    mj = _load()
    import threading, time as _t
    client = mj.PoolClient.__new__(mj.PoolClient)
    client.lock = threading.Lock(); client.pool = "lowprio"; client.n_429 = client.n_529 = client.n_ok = 0
    client.episode_start = None; client.episode_kind = None; client.windows = {}; client.window_s = 300.0; client.events = []
    judge = SimpleNamespace(limiter=AdaptiveLimiter(20, min_width=mj.THROTTLE_FLOOR))
    thr = mj.Throttler(judge, client, after_s=300.0)
    base = 1_000_000.0                                                           # exactly representable: base + t carries every offset
    real_monotonic = _t.monotonic                                                # exactly, where a host's monotonic() reading can round a
    clock = {"now": base}                                                        # 300 s same-kind gap above window_s and expire a window
    mj.time.monotonic = lambda: clock["now"]                                      # the module's clock, driven by the test
    try:
        def at(t, kind):
            clock["now"] = base + t; client.record(kind)
        at(0.0, "429"); at(100.0, "529"); at(400.0, "529"); at(700.0, "529"); at(999.0, "529")
        ev999 = thr.step(base + 999.0)                                           # 529s only: a PROVIDER_OVERLOADED problem at most, never a shrink
        assert ev999 is None or ev999["event"] == mj.PROVIDER_OVERLOADED
        assert thr.halvings == 0
        at(1000.0, "429")                                                        # 1,000 s after the only other 429: a NEW window
        expired = [e for e in client.events if e["event"] == "episode_window_expired"]
        assert len(expired) == 1 and expired[0]["kind"] == "429" and expired[0]["gap_s"] == 1000.0 and client.windows["429"]["start"] == base + 1000.0
        assert thr.step(base + 1000.0) is None and thr.halvings == 0             # no action on one fresh 429
        at(1150.0, "429"); at(1300.0, "429")
        ev = thr.step(base + 1301.0)
        assert ev["event"] == mj.THROTTLED_DOWN and (ev["from"], ev["to"]) == (20, 10) and thr.halvings == 1    # sustained AND fresh from 1000
        # 429 -> 529 -> 429 within the window keeps the first start (no expiry)
        client.record("ok"); client.events.clear()
        at(2000.0, "429"); at(2100.0, "529"); at(2200.0, "429")
        assert not [e for e in client.events if e["event"] == "episode_window_expired"] and client.windows["429"]["start"] == base + 2000.0
    finally:
        mj.time.monotonic = real_monotonic
