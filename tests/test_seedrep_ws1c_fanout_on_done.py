"""The completion-order ``on_done`` callback on ``proxy_engine.bounded_gather`` and its
pass-through in the driver's fan-out.

  * ``on_done(key, outcome)`` fires ONCE per thunk in COMPLETION order (the test completes the
    thunks deliberately out of submission order), after the result is recorded; a failed thunk
    fires with its exception AND is collected in ``errors`` exactly as before; the returned maps
    are identical with and without the callback; a callback that raises is never collected as a
    request error — it cancels the remaining thunks and propagates.
  * ``served_capability_run._fanout_responses`` / ``_fanout`` pass ``on_done`` through, and
    ``_fanout`` reads the controls publication sink (``_CONTROLS_EVENT_SINK``) when no explicit
    callback is given — the seam the shared ``sampled_controls._run_judged_served`` body calls.
  * proof-set gate 4, second half: the FIRST ``controls_events.jsonl`` event is on disk while a
    later thunk is still deliberately blocked — publication does not wait for the fan-out.

Offline; no GPU, no network."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402
from sisteer.capeval.serving import proxy_engine as pe  # noqa: E402
from sisteer.eval import controls_events as cev  # noqa: E402


def _thunks(delays: dict, fail=()):
    async def make(key):
        await asyncio.sleep(delays[key])
        if key in fail:
            raise ValueError(f"refused {key}")
        return {"key": key, "payload": f"result-{key}"}
    return [(k, (lambda k=k: make(k))) for k in delays]


def test_on_done_fires_once_per_thunk_in_completion_order_with_the_maps_unchanged():
    delays = {"a": 0.06, "b": 0.0, "c": 0.03, "d": 0.045}
    seen = []

    async def run(on_done):
        return await pe.bounded_gather(_thunks(delays, fail=("d",)), concurrency=8, on_done=on_done)

    with_hook = asyncio.run(run(lambda k, o: seen.append((k, o))))
    without = asyncio.run(run(None))
    assert with_hook == without                                     # the maps are the same either way
    results, errors = with_hook
    assert results == {"a": {"key": "a", "payload": "result-a"}, "b": {"key": "b", "payload": "result-b"},
                       "c": {"key": "c", "payload": "result-c"}}
    assert errors == [("d", "ValueError: refused d")]               # collected exactly as before
    # completion order, not submission order: b (0 ms), c (30), d (45, the failure), a (60)
    assert [k for k, _ in seen] == ["b", "c", "d", "a"]
    assert seen[0][1] == results["b"] and seen[1][1] == results["c"] and seen[3][1] == results["a"]
    assert isinstance(seen[2][1], ValueError) and str(seen[2][1]) == "refused d"
    assert len(seen) == 4                                            # once per thunk, the failure included


def test_on_done_sees_the_recorded_result_and_a_concurrency_of_one_keeps_submission_order():
    delays = {"x": 0.02, "y": 0.0, "z": 0.01}
    order = []

    async def run():
        return await pe.bounded_gather(_thunks(delays), concurrency=1,
                                       on_done=lambda k, o: order.append(k))
    results, errors = asyncio.run(run())
    assert order == ["x", "y", "z"] and errors == [] and set(results) == {"x", "y", "z"}


def test_a_raising_callback_is_not_a_request_error_it_propagates_and_cancels_the_rest():
    delays = {"first": 0.0, "slow": 0.5}
    started = []

    async def run():
        def boom(key, outcome):
            raise OSError("disk full while publishing")
        return await pe.bounded_gather(_thunks(delays), concurrency=8, on_done=boom)

    async def outer():
        t0 = asyncio.get_running_loop().time()
        with pytest.raises(OSError, match="disk full while publishing"):
            await run()
        return asyncio.get_running_loop().time() - t0
    elapsed = asyncio.run(outer())
    assert elapsed < 0.4, elapsed    # the slow sibling was cancelled, not awaited to completion
    del started


def test_no_callback_is_byte_for_byte_the_old_fan_out():
    delays = {"a": 0.0, "b": 0.01}
    r, e = asyncio.run(pe.bounded_gather(_thunks(delays, fail=("b",)), concurrency=2))
    assert r == {"a": {"key": "a", "payload": "result-a"}} and e == [("b", "ValueError: refused b")]


# --------------------------------------------------------------------------- the driver seam
def _resp(text, fr="stop", cid=None, ct=7):
    r = {"choices": [{"message": {"content": text}, "finish_reason": fr}],
         "usage": {"prompt_tokens": 3, "completion_tokens": ct}}
    if cid is not None:
        r["id"] = cid
    return r


def test_driver_fanout_passes_on_done_through_and_reads_the_sink_only_when_no_callback_is_given(tmp_path):
    work = [("i1", {"model": "petri__base", "messages": [], "max_tokens": 64, "seed": 11}),
            ("i2", {"model": "petri__base", "messages": [], "max_tokens": 64, "seed": 22})]

    async def post(payload):
        await asyncio.sleep(0.03 if payload["seed"] == 11 else 0.0)
        return _resp(f"text-{payload['seed']}", cid=f"cmpl-{payload['seed']}")

    seen = []
    results, errors, finish, meta = asyncio.run(
        srun._fanout(post, work, concurrency=4, on_done=lambda k, o: seen.append(k)))
    assert seen == ["i2", "i1"] and results == {"i1": "text-11", "i2": "text-22"} and errors == []
    assert finish == {"stop": 2} and meta["i1"]["max_tokens"] == 64
    responses, errors2 = asyncio.run(
        srun._fanout_responses(post, work, concurrency=4, on_done=lambda k, o: seen.append(("r", k))))
    assert seen[2:] == [("r", "i2"), ("r", "i1")] and responses["i2"]["id"] == "cmpl-22" and errors2 == []
    # the sink route: no explicit callback, a sink set in the context -> events land; an explicit
    # callback wins over the sink (the sink sees nothing)
    writer = cev.ControlsEventWriter(tmp_path / cev.CONTROLS_EVENTS_FILENAME, cell_id="none")
    sink = writer.sink(control="xstest", cap=64)
    token = srun._CONTROLS_EVENT_SINK.set(sink)
    try:
        asyncio.run(srun._fanout(post, work, concurrency=4))
        events = cev.read_controls_events(writer.path)
        assert [e["item_id"] for e in events] == ["i2", "i1"]
        assert events[0]["sampling_seed"] == 22 and events[0]["completion_id"] == "cmpl-22"
        assert events[0]["text_sha256"] == srun._text_sha("text-22") and events[0]["at_cap"] is False
        asyncio.run(srun._fanout(post, work, concurrency=4, on_done=lambda k, o: None))
        assert len(cev.read_controls_events(writer.path)) == 2       # the explicit callback wins
        # _fanout_responses (the MMLU leg's entry) never reads the sink
        asyncio.run(srun._fanout_responses(post, work, concurrency=4))
        assert len(cev.read_controls_events(writer.path)) == 2
    finally:
        srun._CONTROLS_EVENT_SINK.reset(token)
    assert srun._CONTROLS_EVENT_SINK.get() is None
    asyncio.run(srun._fanout(post, work, concurrency=4))
    assert len(cev.read_controls_events(writer.path)) == 2           # unset sink: nothing published


def test_the_first_event_is_on_disk_while_a_later_thunk_is_still_blocked(tmp_path):
    """Proof-set gate 4 (second half): publication is per completion, not per fan-out."""
    work = [("fast", {"model": "m", "messages": [], "max_tokens": 64, "seed": 1}),
            ("held", {"model": "m", "messages": [], "max_tokens": 64, "seed": 2})]
    path = tmp_path / cev.CONTROLS_EVENTS_FILENAME
    writer = cev.ControlsEventWriter(path, cell_id="none")

    async def scenario():
        gate = asyncio.Event()

        async def post(payload):
            if payload["seed"] == 2:
                await gate.wait()
                return _resp("held text", fr="length", ct=64)
            return _resp("fast text", cid="cmpl-fast")

        token = srun._CONTROLS_EVENT_SINK.set(writer.sink(control="or_bench_hard_1k", cap=64))
        try:
            task = asyncio.ensure_future(srun._fanout(post, work, concurrency=4))
            for _ in range(200):                                    # until the fast draw's event lands
                await asyncio.sleep(0.005)
                if path.exists() and path.read_text().count("\n") >= 1:
                    break
            assert not task.done()                                  # the fan-out is still waiting on `held`
            on_disk = cev.read_controls_events(path)
            assert [e["item_id"] for e in on_disk] == ["fast"]
            assert on_disk[0]["at_cap"] is False and on_disk[0]["completion_id"] == "cmpl-fast"
            gate.set()
            results, errors, finish, meta = await task
        finally:
            srun._CONTROLS_EVENT_SINK.reset(token)
        return results, errors, finish, meta

    results, errors, finish, meta = asyncio.run(scenario())
    assert results == {"fast": "fast text", "held": "held text"} and errors == []
    events = cev.read_controls_events(path)
    assert [(e["item_id"], e["at_cap"], e["seq"]) for e in events] == [("fast", False, 0), ("held", True, 1)]
    assert events[1]["finish_reason"] == "length" and events[1]["completion_tokens"] == 64
    assert finish == {"stop": 1, "length": 1}
    # a failed request is an event too, with the hole named
    async def failing(payload):
        raise RuntimeError("proxy 409")
    token = srun._CONTROLS_EVENT_SINK.set(writer.sink(control="or_bench_hard_1k", cap=64))
    try:
        results, errors, *_ = asyncio.run(srun._fanout(failing, work[:1], concurrency=1))
    finally:
        srun._CONTROLS_EVENT_SINK.reset(token)
    assert results == {} and errors == [("fast", "RuntimeError: proxy 409")]
    last = cev.read_controls_events(path)[-1]
    assert last["error"] == "RuntimeError: proxy 409" and last["at_cap"] is None and last["text_sha256"] is None
    assert json.loads(path.read_text().splitlines()[-1]) == last
