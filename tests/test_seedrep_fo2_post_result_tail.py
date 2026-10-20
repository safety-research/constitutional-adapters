"""F-AM (fixer fo2) -- a controls BASE row's container does not exit after its RESULT.

THE DEFECT. Three controls rows from two prior trees (S3 controls DONE 17:38:31Z, S3 controls DONE 18:42:49Z, S5 controls FULL DONE
18:49:10Z) were all still RUNNING 50-110 min after `[sync final] N file(s) committed`, the last log line -- an idle H200 each until
the launch agent ended them (16 h to the production container timeout; 16 controls containers per wave). Every app thread is a daemon,
so the row FUNCTION was the thing not returning, and only for controls rows.

THE BLOCKING CALL. `StreamTailer.drain()` -> `_enqueue()` -> `ControlsStreamAdapter.records()` -> PER at-cap event ->
`prompt_of(control, item_id, arm)` (the `ladder_adapter_for_row` controls closure) -> `controls_prompts_via_helper(..., [item_id])` = ONE
`subprocess.run(<driver venv> seedrep_ladder_helpers.py controls-prompts ..., timeout=1800)` per DISTINCT at-cap (control, item) (the
prompt cache was filled one asked id at a time), each importing `sisteer.capeval.tiers.judged` (~6 s warm on the grading host, more cold
in the container) and loading the whole corpus -- plus `system_of(arm)`, ONE `si-text` helper subprocess PER EVENT for the two prompted arms
(no cache at all). Measured with this module's world: 120 events over 3 streams -> 80 single-id `controls-prompts`
launches + 120 `si-text` launches (the fix: 6 + 1). Controls is `live = False`: every record is built at drain, AFTER `teardown_standup`, so the H200 idles while 8 streams
x up to 1,769 draws worth of at-cap events each launch their own interpreter. Not a deadlock: an O(events) serial subprocess fan-out that
nothing bounded. Static (`live = True`) builds its records while the drivers still run; misalign resolves ONE `messages_of(wanted)` per
stream.

THE FIX. (1) `ControlsStreamAdapter` takes batch resolvers -- `prompts_of(control, item_ids)` ONCE per (stream, control) and
`system_of(arm)` once per arm per stream; `ladder_adapter_for_row` wires them (a cross-stream prompt cache: only the ids not yet resolved
go to the helper; one `si-text` per arm for the whole row) and keeps the per-item `prompt_of` for an injected test resolver only. (2) The
belt: `_run_standup_row` runs the whole post-RESULT tail (drain -> seal -> inventory -> `finish_attempt`) through `run_post_result_tail`
under `POST_RESULT_EXIT_S` (spawn_ctx `post_result_exit_s` overrides); past the deadline the row logs `POST_RESULT_HANG <phase>` with the
tail thread's blocking frames and a dump of every thread, records the hang under `modal_evidence.json["post_result"]` and RETURNS -- the
RESULT already written is never touched (no acceptance is published, so the pull refuses by name; the H200 is freed).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts" / "modal", REPO / "tests"):
    sys.path.insert(0, str(_p))
import seedrep_launch_plan as lp  # noqa: E402
import seedrep_steward as st  # noqa: E402
import seedrep_watch as sw  # noqa: E402
import test_seedrep_review_fixes as RF  # noqa: E402
from test_seedrep_review_fixes import L, draft, inputs, srv  # noqa: E402,F401  (module fixtures register by import; `srv` = the app under a modal stub)
from test_seedrep_fa2_controls_events import _at_cap_event  # noqa: E402
from sisteer.eval import controls_events as cev  # noqa: E402

SRC = (REPO / "scripts/modal/seedrep_served.py").read_text(encoding="utf-8")
ARM = lp.ARDITI_CELL                                                  # the arditi controls row's ONE arm (cell == ladder arm)
ARDITI_STORE_DIR = "arditi_allpos__controls__t0__a0__92b66dce__6a1274f3"
CONTROLS = ("xstest", "or_bench_hard_1k")
BOUND_S = 30.0                                                        # the fixture's deadline: a drain that has not returned by then IS the hang


def _bounded(work, *, bound_s: float = BOUND_S, srv=None):
    """Run ``work`` in a thread; fail BY NAME (the blocking frames) when it has not returned within ``bound_s``."""
    out = {}

    def run():
        try:
            out["result"] = work()
        except BaseException as e:  # noqa: BLE001
            out["error"] = e
    t = threading.Thread(target=run, daemon=True, name="fo2-bounded")
    t0 = time.time()
    t.start()
    t.join(bound_s)
    if t.is_alive():
        frames = srv.thread_stack(t) if srv is not None else []
        pytest.fail(f"did not return within {bound_s} s -- the blocking call:\n{''.join(frames[-4:])}")
    if "error" in out:
        raise out["error"]
    return out["result"], time.time() - t0


def _events(stream_no: int, n_per_control: int, shared: int) -> list:
    """``n_per_control`` at-cap events per control for stream ``stream_no``: ``shared`` ids common to every stream (the cross-stream cache
    must not ask them twice), the rest unique to this stream (every stream still needs its own batch)."""
    evs = []
    seq = 0
    for control in CONTROLS:
        for i in range(n_per_control):
            item = f"{control[:3]}-{i}" if i < shared else f"{control[:3]}-{stream_no}-{i}"
            seq += 1
            evs.append({**_at_cap_event(17 + stream_no), "seq": seq, "cell_id": ARM, "control": control, "item_id": item})
    return evs


def _controls_world(srv, L, draft, tmp_path, *, n_streams: int, n_per_control: int, shared: int):
    row = RF._row(draft, "controls", "arditi")
    volroot, root, attempt_id, arv, _vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
    for s in range(1, n_streams + 1):
        d = root / f"exp/sampled/controls_cap512__s{s}/cells/{ARDITI_STORE_DIR}"
        d.mkdir(parents=True)
        (d / cev.CONTROLS_EVENTS_FILENAME).write_text("".join(json.dumps(e) + "\n" for e in _events(s, n_per_control, shared)))
    return row, volroot, root, attempt_id, arv, prod


class _Helpers:
    """Counting doubles for the two driver-venv helpers the controls drain runs, each costing ``cost_s`` of wall clock (the real ones import
    `sisteer.capeval.tiers.judged` / `misalign.runner` in a fresh interpreter: ~6 s warm on the grading host)."""

    def __init__(self, cost_s: float) -> None:
        self.cost_s = cost_s
        self.prompt_calls: list = []                       # (control, tuple(ids))
        self.si_calls: list = []                           # argv

    def controls_prompts_via_helper(self, driver_python, tree, control, item_ids, *, env):
        self.prompt_calls.append((control, tuple(item_ids)))
        time.sleep(self.cost_s)
        return {str(i): f"prompt-{control}-{i}" for i in item_ids}

    def run(self, argv, **kw):
        assert "si-text" in argv, argv                    # the ONLY subprocess the wiring may still run here (sampling is injected)
        self.si_calls.append(list(argv))
        time.sleep(self.cost_s)
        return subprocess.CompletedProcess(argv, 0, stdout="SI TEXT\n", stderr="")


def _wire(srv, L, row, prod, root, arv, attempt_id, monkeypatch, helpers, *, ctx_extra=None):
    """The app's REAL controls wiring (`ladder_adapter_for_row`) over the counting helpers: the sampling row injected (no helper), the arm
    PROMPTED (`si_by_harness` names an SI id) so the `si-text` path runs, the two helpers replaced by the doubles."""
    monkeypatch.setattr(srv, "controls_prompts_via_helper", helpers.controls_prompts_via_helper)
    monkeypatch.setattr(srv, "subprocess", types.SimpleNamespace(run=helpers.run, Popen=subprocess.Popen, TimeoutExpired=subprocess.TimeoutExpired,
                                                                  STDOUT=subprocess.STDOUT, CompletedProcess=subprocess.CompletedProcess))
    ctx = {"controls_sampling": dict(RF.SIX_SAMPLER_KEYS), **(ctx_extra or {})}
    h = lp.harness_id(ARM)
    return srv.ladder_adapter_for_row(L, row, ctx, producer=prod, root=root, attempt_root_vol=arv, attempt_id=attempt_id,
                                      vmodel_by_harness={h: f"petri__{h}"}, si_by_harness={h: "si-fake"}, driver_python=sys.executable, tree=REPO,
                                      env={**os.environ, "PYTHONPATH": str(REPO / "src")})


# --------------------------------------------------------------------------- the drain: one helper per (stream, control), never per event -- #
class TestControlsDrainIsBatched:
    N_STREAMS, N_PER_CONTROL, SHARED, COST = 3, 20, 10, 0.1          # 120 at-cap events; the per-event wiring = 120 prompt + 120 si-text helpers = 24 s

    def test_the_real_wiring_drains_120_events_with_one_helper_per_stream_and_control_and_one_si_text_per_arm(self, srv, L, draft, tmp_path, monkeypatch):
        row, volroot, root, attempt_id, arv, prod = _controls_world(srv, L, draft, tmp_path, n_streams=self.N_STREAMS, n_per_control=self.N_PER_CONTROL, shared=self.SHARED)
        helpers = _Helpers(self.COST)
        adapter = _wire(srv, L, row, prod, root, arv, attempt_id, monkeypatch, helpers)
        assert adapter.prompts_of is not None and adapter.system_of is not None and adapter.live is False       # production: the batch path
        total = self.N_STREAMS * len(CONTROLS) * self.N_PER_CONTROL
        summary, elapsed = _bounded(lambda: srv.StreamTailer(root, adapter, prod, poll_s=10).drain(), srv=srv)
        assert summary["complete"] and summary["problems"] == [] and summary["enqueued"] == total == 120
        # ONE helper per (stream, control), each carrying a BATCH of ids; no id is asked twice (the cross-stream cache) ...
        assert len(helpers.prompt_calls) == self.N_STREAMS * len(CONTROLS) == 6
        assert all(len(ids) > 1 for _, ids in helpers.prompt_calls) and all(ids == tuple(sorted(set(ids))) for _, ids in helpers.prompt_calls)
        asked = [(c, i) for c, ids in helpers.prompt_calls for i in ids]
        assert len(asked) == len(set(asked)) == len(CONTROLS) * (self.SHARED + self.N_STREAMS * (self.N_PER_CONTROL - self.SHARED)) == 80
        # ... and ONE `si-text` for the arm over the whole row (the per-event wiring ran 120)
        assert len(helpers.si_calls) == 1
        # the known-bad's shape, stated as the bound the fix must clear: strictly less than one helper's cost per event
        assert elapsed < self.COST * total / 2, f"drain took {elapsed:.1f} s over {total} events: the per-event fan-out is back"
        entries = prod.seal()
        assert len(entries) == total
        rec = json.loads((volroot / entries[0]["path"]).read_text())
        assert rec["kind"] == "controls" and rec["arm"] == ARM and rec["redraw"]["sampling"] == RF.SIX_SAMPLER_KEYS
        msgs = rec["redraw"]["messages"]
        assert msgs[0] == {"role": "system", "content": "SI TEXT"} and msgs[-1]["content"].startswith("prompt-")       # the batch's prompt + the cached system text

    def test_a_missing_item_in_the_batch_refuses_by_name_and_the_drain_fails(self, srv, L, draft, tmp_path, monkeypatch):
        row, volroot, root, attempt_id, arv, prod = _controls_world(srv, L, draft, tmp_path, n_streams=1, n_per_control=3, shared=3)
        helpers = _Helpers(0.0)
        monkeypatch.setattr(helpers, "controls_prompts_via_helper",
                            lambda dp, tree, control, ids, *, env: {i: f"p-{i}" for i in ids if not i.endswith("-1")})     # the helper drops one id
        adapter = _wire(srv, L, row, prod, root, arv, attempt_id, monkeypatch, helpers)
        with pytest.raises(srv.LadderUnavailable, match=r"controls: no pinned xstest row carries item xst-1"):
            srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert prod.entries == []                                                                           # nothing partial is enqueued

    def test_an_injected_per_item_resolver_keeps_the_test_path_and_disables_the_batch_path(self, srv, L, draft, tmp_path, monkeypatch):
        row, volroot, root, attempt_id, arv, prod = _controls_world(srv, L, draft, tmp_path, n_streams=1, n_per_control=2, shared=2)
        helpers = _Helpers(0.0)
        seen = []
        adapter = _wire(srv, L, row, prod, root, arv, attempt_id, monkeypatch, helpers,
                        ctx_extra={"controls_prompt_resolver": lambda c, i, a: seen.append((c, i, a)) or (f"inj-{i}", None)})
        assert adapter.prompts_of is None and adapter.system_of is None                                   # the fa2 tests' contract, unchanged
        summary = srv.StreamTailer(root, adapter, prod, poll_s=10).drain()
        assert summary["enqueued"] == 4 and len(seen) == 4 and helpers.prompt_calls == [] and helpers.si_calls == []


class TestAdapterResolver:
    """`ControlsStreamAdapter._resolver` on its own: the batch contract and the fallback."""

    def _rows(self, n):
        return [{**_at_cap_event(), "cell_id": ARM, "control": CONTROLS[i % 2], "item_id": f"it-{i}"} for i in range(n)]

    def test_batch_resolvers_are_called_once_per_control_and_once_per_arm(self, srv, L):
        calls, sys_calls = [], []
        ad = srv.ControlsStreamAdapter(L, None, None, "ta-1", container_root=Path("/x"), attempt_root_vol="/vol/x", seed_of_store=lambda p: 1,
                                       vmodel_of=lambda a: "vm", prompts_of=lambda c, ids: calls.append((c, tuple(ids))) or {i: f"p-{i}" for i in ids},
                                       system_of=lambda a: sys_calls.append(a) or "SYS")
        resolve = ad._resolver(self._rows(10))
        assert sorted(c for c, _ in calls) == sorted(CONTROLS) and all(len(ids) == 5 for _, ids in calls)
        for i in range(10):
            assert resolve(CONTROLS[i % 2], f"it-{i}", ARM) == (f"p-it-{i}", "SYS")
        assert sys_calls == [ARM]                                                                            # once per arm, not per event
        with pytest.raises(srv.LadderUnavailable, match="no pinned xstest row carries item nope"):
            resolve("xstest", "nope", ARM)

    def test_without_any_resolver_the_adapter_refuses_by_name_and_the_per_item_fallback_is_called_per_event(self, srv, L):
        ad = srv.ControlsStreamAdapter(L, None, None, "ta-1", container_root=Path("/x"), attempt_root_vol="/vol/x", seed_of_store=lambda p: 1, vmodel_of=lambda a: "vm")
        with pytest.raises(srv.LadderUnavailable, match="no prompt resolver was wired"):
            ad._resolver(self._rows(1))
        seen = []
        ad2 = srv.ControlsStreamAdapter(L, None, None, "ta-1", container_root=Path("/x"), attempt_root_vol="/vol/x", seed_of_store=lambda p: 1,
                                        vmodel_of=lambda a: "vm", prompt_of=lambda c, i, a: seen.append(i) or (f"p-{i}", None))
        resolve = ad2._resolver(self._rows(3))
        for i in range(3):
            resolve(CONTROLS[i % 2], f"it-{i}", ARM)
        assert seen == ["it-0", "it-1", "it-2"]


# --------------------------------------------------------------------------- the belt: the post-RESULT tail is bounded ------------------- #
class TestPostResultBelt:
    def test_known_bad_a_tail_that_never_returns_is_cut_at_the_deadline_naming_the_blocking_call(self, srv):
        never = threading.Event()
        logged, dumped = [], []
        (rec, elapsed) = _bounded(lambda: srv.run_post_result_tail(never.wait, deadline_s=0.5, phase_of=lambda: "ladder-drain", log=logged.append,
                                                                   dump=lambda: dumped.append(True)), bound_s=10.0, srv=srv)
        assert rec["hang"] is True and rec["phase"] == "ladder-drain" and rec["deadline_s"] == 0.5 and 0.5 <= rec["elapsed_s"] <= 5 and rec["at"].endswith("Z")
        assert elapsed < 5, "the belt itself must return at the deadline"
        assert any("wait" in fr for fr in rec["blocking"]), rec["blocking"]                                  # the frames name the blocking call
        assert len(logged) == 1 and logged[0].startswith(f"[row] {srv.POST_RESULT_HANG} phase=ladder-drain deadline_s=0.5") and "Blocking call" in logged[0]
        assert dumped == [True]
        never.set()

    def test_a_tail_that_returns_passes_its_result_through_and_re_raises_its_exception(self, srv):
        rec = srv.run_post_result_tail(lambda: {"state": "closed"}, deadline_s=5.0, log=lambda m: pytest.fail(m))
        assert rec == {"hang": False, "elapsed_s": rec["elapsed_s"], "result": {"state": "closed"}} and rec["elapsed_s"] < 5

        def refuse():
            raise srv.LadderInventoryIncomplete("short")
        with pytest.raises(srv.LadderInventoryIncomplete, match="short"):                              # the caller's fail(...) classification holds
            srv.run_post_result_tail(refuse, deadline_s=5.0, log=lambda m: pytest.fail(m))

    def test_dump_all_threads_names_every_live_thread(self, srv, capfd):
        gate = threading.Event()
        t = threading.Thread(target=gate.wait, daemon=True, name="fo2-parked")
        t.start()
        lines = []
        try:
            srv.dump_all_threads(log=lines.append)
        finally:
            gate.set()
        joined = "\n".join(lines)
        assert "[row] thread fo2-parked" in joined and "wait" in joined
        assert any(f"[row] thread {threading.current_thread().name}" in ln for ln in lines)
        capfd.readouterr()                                                                                # faulthandler's fd-2 dump, if any, is drained

    def test_the_deadline_is_sized_between_the_batched_drain_and_the_container_timeout(self, srv, draft):
        rows = [r for r in draft["standups"] if r["leg"] == "controls" and r["kind"] == "base"]
        assert rows and all(int(r["modal"]["timeout_s"]) >= 10 * srv.POST_RESULT_EXIT_S for r in rows), "the belt must fire well before the container timeout"
        assert 600 <= srv.POST_RESULT_EXIT_S <= 3600                    # minutes above the batched drain's few-minute cost, never hours
        assert srv.POST_RESULT_EXIT_CTX == "post_result_exit_s" and srv.POST_RESULT_HANG == "POST_RESULT_HANG"

    def test_the_row_runs_its_whole_acceptance_tail_under_the_belt_and_never_rewrites_RESULT(self):
        """Source-shape pin (no harness drives `_run_standup_row` end to end): the drain/seal/inventory/finish_attempt block is the body of
        `acceptance_tail`, run through `run_post_result_tail` with the spawn_ctx override; the hang lands in evidence["post_result"]; the
        RESULT file is written exactly once on the DONE path, BEFORE the tail, and never after it."""
        body = SRC[SRC.index("def _run_standup_row("):SRC.index("def ladder_adapter_for_row(")]
        tail = body[body.index("def acceptance_tail()"):]
        assert "tailer.drain()" in tail and "producer.seal()" in tail and "write_ladder_enqueued(" in tail and "return finish_attempt(" in tail
        assert "run_post_result_tail(acceptance_tail, deadline_s=float(ctx.get(POST_RESULT_EXIT_CTX, POST_RESULT_EXIT_S))" in tail
        assert 'evidence["post_result"]' in tail and "if tail[\"hang\"]:" in tail
        # fo3: the verdict comes from `post_result_outcome` over the accepted Dict's holder -- never a bare DONE with no acceptance
        assert "holder = accepted.get(akey)" in tail and "post_result_outcome(tail, attempt_id=attempt_id, accepted_holder=holder)" in tail
        assert '"status": verdict["status"]' in tail and '"acceptance": verdict["acceptance"]' in tail and '"status": "DONE"' not in tail
        assert tail.count("(root / RESULT_FILE).write_text(") == 0 and body.index("(root / RESULT_FILE).write_text(") < body.index('_sync_all(syncers, lock, "final")') < body.index("def acceptance_tail()")
        for status in ("LADDER_INVENTORY_INCOMPLETE", "LADDER_TAILER_ERROR", "PUBLICATION_FAILED"):
            assert f'return fail("{status}"' in tail                                                     # the three refusals still classify by name
        assert 'hb_state["phase"] = "ladder-drain"' in tail and 'hb_state["phase"] = "ladder-seal"' in tail and 'hb_state["phase"] = "acceptance"' in tail


# --------------------------------------------------------------------------- the other ladder producers return promptly through the same tail #
class TestOtherLaddersReturnPromptly:
    def test_static_builds_records_live_and_resolves_the_arm_system_text_once_for_the_row(self, srv, L, draft, tmp_path, monkeypatch):
        row = RF._row(draft, "pasttense")
        volroot, root, attempt_id, arv, vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
        helpers = _Helpers(0.0)
        monkeypatch.setattr(srv, "subprocess", types.SimpleNamespace(run=helpers.run, Popen=subprocess.Popen, TimeoutExpired=subprocess.TimeoutExpired,
                                                                      STDOUT=subprocess.STDOUT, CompletedProcess=subprocess.CompletedProcess))
        h = lp.harness_id("none")
        adapter = srv.ladder_adapter_for_row(L, row, {}, producer=prod, root=root, attempt_root_vol=arv, attempt_id=attempt_id,
                                             vmodel_by_harness={h: f"petri__{h}"}, si_by_harness={h: "si-fake"}, driver_python=sys.executable, tree=REPO, env=dict(os.environ))
        assert adapter.kind == "static" and adapter.live is True                                           # records during the run, not at drain
        stream = root / "exp/transcripts_results/run/pasttense/none/shard00/responses.jsonl"
        stream.parent.mkdir(parents=True)
        stream.write_text("".join(json.dumps(RF._static_row(f"g{i}", 0, at_cap=True)) + "\n" for i in range(30)))
        rows = [json.loads(ln) for ln in stream.read_text().splitlines()]
        recs = adapter.records(stream, rows)
        assert len(recs) == 30 and all(r["redraw"]["messages"][0] == {"role": "system", "content": "SI TEXT"} for r in recs)
        assert len(helpers.si_calls) == 1                                                                   # one `si-text` per arm, not 30

    def test_misalign_resolves_one_messages_batch_per_stream(self, srv, L, draft, tmp_path):
        row = RF._row(draft, "misalign", "s2")
        volroot, root, attempt_id, arv, vol, prod = RF.TestP1_5_ladder_tailer()._setup(srv, L, tmp_path, row)
        cell_h = "si_constv2_mix_full__lora_r4_ce"
        rng = random.Random(11)
        for beh in ("leaking", "sabotage"):
            d = root / f"exp/misalign/agentic__s2__{beh}" / cell_h
            d.mkdir(parents=True)
            (d / "meta.json").write_text(json.dumps({"sampling": dict(RF.SIX_SAMPLER_KEYS)}))
            rows = [{"family": "agentic", "cell_id": cell_h, "virtual_model": f"petri__{cell_h}", "item_id": f"{beh}-it{i}", "sample_idx": 0, "behavior": beh,
                     "prompt_sha": "p", "response": "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ,.") for _ in range(900)), "finish_reason": "length",
                     "usage": {}, "model_reported": "m", "seed": 7 + i} for i in range(25)]
            (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        calls = []

        def messages_of(reqs):
            calls.append(len(reqs))
            return {srv.MisalignStreamAdapter.request_key(r["cell_id"], r["behavior"], r["item_id"]): [{"role": "user", "content": r["item_id"]}] for r in reqs}
        adapter = srv.MisalignStreamAdapter(L, prod.spec, prod.ref, attempt_id, container_root=root, attempt_root_vol=arv, messages_of=messages_of)
        summary, _ = _bounded(lambda: srv.StreamTailer(root, adapter, prod, poll_s=10).drain(), srv=srv)
        assert summary["enqueued"] == 50 and calls == [25, 25]                                              # one batch per stream

    def test_the_rung_tail_is_a_volume_enumeration_with_no_helper_subprocess(self, srv, L, draft, tmp_path, monkeypatch):
        def no_subprocess(*a, **k):
            raise AssertionError("the rung's producer side must not run a subprocess")
        monkeypatch.setattr(srv, "subprocess", types.SimpleNamespace(run=no_subprocess, Popen=no_subprocess))
        row = RF._row(draft, "pasttense", "s1", "8192")
        volroot = tmp_path / "vol"
        next_root = L.ladder_root_rel("2222222", RF.WAVE, "pasttense", "s1", 16384)
        for i in range(3):
            q = volroot / next_root / "queue" / "none__s1" / f"static__pasttense__g{i}__0" / "ta-9.json"
            q.parent.mkdir(parents=True)
            q.write_text(json.dumps({"item_key": f"static__pasttense__g{i}__0"}) + "\n")
        entries = srv.rung_entries_from_volume(volroot, next_root, "ta-9")
        (prod, _) = _bounded(lambda: srv.rung_next_producer(L, row, {"W": RF.W, "wave_uuid": RF.WAVE}, vol=L.LocalVolume(volroot), attempt_id="ta-9",
                                                            attempt_root_vol="/vol/seedrep/2222222/exp/seedrep_battery_modal/pasttense/s1/rungs/8192/shard00/attempts/ta-9",
                                                            entries=entries), srv=srv)
        assert len(prod.seal()) == 3 and prod.spec.cap_next == 16384


# --------------------------------------------------------------------------- fo3: a pre-acceptance timeout is an EXPLICIT failure #
def _hang_tail(phase="ladder-drain"):
    return {"hang": True, "phase": phase, "deadline_s": 1800.0, "elapsed_s": 1800.3, "at": "2026-09-19T20:00:00Z",
            "blocking": ['  File "seedrep_served.py", line 1, in drain\n    r = subprocess.run(argv, timeout=1800)\n']}


def _call(sid="controls__s1__base__shard00", cid="fc-1"):
    return {"call_id": cid, "standup_id": sid, "leg": "controls", "engine": "s1", "stage": "base", "shard": 0, "kind": "base",
            "control_key": "W/wave/controls/s1/base/0", "accepted_key": "W|wave|controls|s1|base|0", "wave_uuid": "wave"}


def _returned(srv, verdict, attempt_id="ta-1"):
    """The dict `_run_standup_row` returns from a `post_result_outcome` verdict = the watcher snapshot's `summary` (FunctionCall.get)."""
    return {"status": verdict["status"], "attempt_id": attempt_id, "attempt_root": "/vol/x", "standup_id": _call()["standup_id"], "acceptance": verdict["acceptance"],
            "elapsed_s": 900.0, "S": 70, "petri": None, "post_result": verdict["post_result"], **({"error": verdict["error"]} if verdict.get("error") else {})}


def _watch_events(calls, snap, required):
    ev = sw.classify(now=1_800_000_000.0, leg="controls", calls=calls, snapshot=snap, state=sw.WatchState(), stall_min=20, rows_flat_min=40, progress_min=30,
                     required_keys=required)
    return [e for e in ev if not e.startswith(("WAVE_LAUNCHED ", "WAVE_DONE ", "HOLD "))]


class TestPostResultTimeoutClassifies:
    RESULT_LINE = "2026-09-19T19:00:00Z status=DONE rc=0 attempt=ta-1 standup=controls__s1__base__shard00 children=8 failed=0 S=70 elapsed_s=900.0"

    def test_known_bad_shape_a_hang_before_acceptance_returned_as_DONE_passes_every_classifier(self, srv):
        """What fo2 returned: `status DONE, acceptance None` -> call_failure (False, DONE, None), no FAILED event, the steward
        invokes nothing -- the missing attempt never enters recovery. Pinned as the shape the fix must NOT produce."""
        fo2 = {"status": "DONE", "acceptance": None, "post_result": {**_hang_tail()}}
        rec = {"done": True, "summary": fo2, "error": None, "result_line": self.RESULT_LINE, "accepted": None}
        assert srv.call_failure(rec) == (False, "DONE", None)
        calls = [_call()]
        req = {calls[0]["accepted_key"]: calls[0]["standup_id"]}
        assert not any(e.startswith("FAILED") for e in _watch_events(calls, {"fc-1": rec}, req))
        assert st.steward_tick(plan={}, calls=calls, snapshot={"fc-1": rec}, holds={}, state=st.StewardState(), required=req)["invoke"] == []

    def test_a_hang_BEFORE_the_acceptance_is_POST_RESULT_TIMEOUT_and_every_consumer_reads_it_as_a_failure(self, srv):
        v = srv.post_result_outcome(_hang_tail(), attempt_id="ta-1", accepted_holder=None)
        assert v["status"] == srv.POST_RESULT_TIMEOUT == "POST_RESULT_TIMEOUT" and v["acceptance"] is None
        assert v["post_result"]["terminal_status"] == srv.POST_RESULT_TIMEOUT and v["post_result"]["hang_after_acceptance"] is False
        assert v["post_result"]["phase"] == "ladder-drain" and v["post_result"]["elapsed_s"] == 1800.3 and "subprocess.run" in v["post_result"]["blocking"][0]
        assert v["error"].startswith("POST_RESULT_TIMEOUT: the post-RESULT tail did not return within 1800.0 s (phase ladder-drain)")
        # another attempt holds the key: THIS attempt still timed out un-accepted (its call is a failure; the key's coverage is the holder's)
        v2 = srv.post_result_outcome(_hang_tail(), attempt_id="ta-1", accepted_holder="ta-0")
        assert v2["status"] == srv.POST_RESULT_TIMEOUT and v2["post_result"]["accepted_holder"] == "ta-0"
        # the ACTUAL consumers over the returned dict as the snapshot summary
        summary = _returned(srv, v)
        rec = {"done": True, "summary": summary, "error": None, "result_line": self.RESULT_LINE, "accepted": None}      # RESULT still says DONE rc=0
        assert srv.call_failure(rec) == (True, "POST_RESULT_TIMEOUT", None)
        calls = [_call()]
        req = {calls[0]["accepted_key"]: calls[0]["standup_id"]}
        ev = _watch_events(calls, {"fc-1": rec}, req)
        assert any(e.startswith("FAILED controls__s1__base__shard00 call=fc-1 POST_RESULT_TIMEOUT") for e in ev), ev
        assert any(e.startswith("PRODUCER_MISSING controls/s1/base/shard0") for e in ev) and not any(e.startswith("DONE") for e in ev)
        tick = st.steward_tick(plan={}, calls=calls, snapshot={"fc-1": rec}, holds={}, state=st.StewardState(), required=req)
        assert tick == {"stop": None, "invoke": ["fc-1"], "covered": 0, "required": 1, "held": [], "carried": 0}
        # infra class -> ONE automatic re-attempt, then HOLD
        assert srv.POST_RESULT_TIMEOUT in lp.D73_INFRA_STATUSES and lp.failure_class(srv.POST_RESULT_TIMEOUT) == "infra"
        d1 = lp.reattempt_decision(None, status=summary["status"], error=summary.get("error"))
        assert d1["action"] == "reattempt" and d1["class"] == "infra" and d1["reattempts"] == 1
        d2 = lp.reattempt_decision({"reattempts": 1}, status=summary["status"], error=summary.get("error"))
        assert d2["action"] == "hold" and "second infra failure" in d2["reason"]

    def test_a_hang_AFTER_the_acceptance_stays_DONE_covered_and_triggers_no_recovery(self, srv):
        v = srv.post_result_outcome(_hang_tail("acceptance"), attempt_id="ta-1", accepted_holder="ta-1")
        assert v["status"] == "DONE" and "error" not in v and v["acceptance"]["state"] == srv.AttemptState.ACCEPTED == "accepted"
        assert v["acceptance"]["accepted_attempt"] == "ta-1" and v["post_result"]["hang_after_acceptance"] is True and "terminal_status" not in v["post_result"]
        assert v["post_result"]["hang"] is True and v["post_result"]["phase"] == "acceptance"                       # the diagnostics are kept
        rec = {"done": True, "summary": _returned(srv, v), "error": None, "result_line": self.RESULT_LINE, "accepted": "ta-1"}
        assert srv.call_failure(rec) == (False, "DONE", None)
        calls = [_call()]
        req = {calls[0]["accepted_key"]: calls[0]["standup_id"]}
        ev = _watch_events(calls, {"fc-1": rec}, req)
        assert not any(e.startswith(("FAILED", "PRODUCER_MISSING")) for e in ev) and ev[-1] == "DONE controls scope=base required=1 accepted=1", ev
        tick = st.steward_tick(plan={}, calls=calls, snapshot={"fc-1": rec}, holds={}, state=st.StewardState(), required=req)
        assert tick["stop"] == "done" and tick["invoke"] == [] and tick["covered"] == 1                             # completed work is never re-run

    def test_a_tail_that_returned_passes_its_acceptance_through_unchanged(self, srv):
        v = srv.post_result_outcome({"hang": False, "elapsed_s": 12.5, "result": {"state": "closed", "accepted_attempt": "ta-1"}}, attempt_id="ta-1", accepted_holder=None)
        assert v == {"status": "DONE", "acceptance": {"state": "closed", "accepted_attempt": "ta-1"}, "post_result": {"hang": False, "elapsed_s": 12.5}}

    def test_an_expired_call_resolves_the_timeout_from_the_durable_evidence_never_from_RESULT(self, srv, draft):
        """The expired path reads RESULT (`status=DONE rc=0`) -- the same hole through the other door: without the durable
        post_result the un-accepted attempt would read DONE for as long as the queue stays open."""
        c = _call()
        root = lp.attempt_root_for(draft["results_root"], leg="controls", view="s1", stage="base", shard=0, attempt_id="ta-1", kind="base")
        v = srv.post_result_outcome(_hang_tail(), attempt_id="ta-1", accepted_holder=None)
        files = {f"{root}/RESULT": self.RESULT_LINE + "\n",
                 f"{root}/{srv.EVIDENCE_NAME}": json.dumps({"schema": "seedrep_served/modal_evidence/1", "post_result": v["post_result"]})}
        out = sw.resolve_expired_call(files.get, draft["results_root"], c, {"attempt_id": "ta-1"})
        assert out["summary"]["status"] == "POST_RESULT_TIMEOUT" and out["summary"]["post_result"]["phase"] == "ladder-drain"
        assert out["error"].startswith("POST_RESULT_TIMEOUT: the post-RESULT tail did not return (phase ladder-drain, 1800.3 s)")
        assert out["resolved_from"] == f"{root}/modal_evidence.json" and out["result_line"] == self.RESULT_LINE
        assert srv.call_failure({**out, "done": True}) == (True, "POST_RESULT_TIMEOUT", out["error"])
        # after the acceptance: the evidence carries the hang but no terminal status -> DONE, as before
        v_ok = srv.post_result_outcome(_hang_tail("acceptance"), attempt_id="ta-1", accepted_holder="ta-1")
        files[f"{root}/{srv.EVIDENCE_NAME}"] = json.dumps({"post_result": v_ok["post_result"]})
        out = sw.resolve_expired_call(files.get, draft["results_root"], c, {"attempt_id": "ta-1"})
        assert out["summary"]["status"] == "DONE" and out["error"] is None and out["resolved_from"] == f"{root}/RESULT"
        # no evidence / unreadable evidence / a returned tail: RESULT decides
        for ev_text in (None, "{not json", json.dumps({"post_result": {"hang": False, "elapsed_s": 3.0}})):
            files[f"{root}/{srv.EVIDENCE_NAME}"] = ev_text
            out = sw.resolve_expired_call(lambda p: files.get(p), draft["results_root"], c, {"attempt_id": "ta-1"})
            assert out["summary"]["status"] == "DONE" and out["error"] is None and out["resolved_from"] == f"{root}/RESULT"
        assert sw.post_result_of(None) == {} and sw.post_result_of("[1]") == {} and sw.EVIDENCE_NAME == srv.EVIDENCE_NAME
