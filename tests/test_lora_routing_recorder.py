"""The per-request LoRA ROUTING RECORDER (gemma canaries 489308 and
489533): ``worker_reports.LoraRoutingRecorder`` over V1- AND V2-shaped fake
GPU model runners (classes named exactly as the pinned vllm 0.25.1 ones, so
the kind dispatch is the real one), and the three ``sisteer_`` RPCs of
``native_worker.NativeWorkerExtension``.

Canary #4 (job 489533) ran the V2 runner, which has no ``input_batch``; the
recorder's PRIMARY evidence is now the ``scheduler_output`` both runners
consume (new requests bound BEFORE the original call — a request that
finishes inside the step is still observed; each step's scheduled set
appended to the epoch's step list) and the runner's own state is a SECONDARY
cross-check by kind (V1 ``input_batch``; V2 ``req_states`` + ``lora_state``):
agreement records nothing new, disagreement goes under ``inconsistent``, an
unknown runner class gets no secondary and is recorded for the check to
refuse. EPOCHS: arm mints one and resets everything; a second arm while armed
refuses naming the live epoch; observations/disarm on another epoch refuse
(``epoch_mismatch``); a second session's steps never include the first's.
Also: the wrapper's return value untouched; ``None`` V1 slots skipped; the
first binding never overwritten; the id cap and the steps cap; read failures
recorded, never raised into the engine step; disarm restores a bound class
method or a pre-existing instance attribute. No vllm; everything synthetic.
The CLIENT mints the epoch and ``arm(epoch)`` adopts it (idempotent for the
live one, ``already_armed`` naming the live one otherwise; an empty or
malformed epoch refuses) — a per-worker mint made a multi-worker launch's arm
payloads disagree and left every worker armed."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import worker_reports as wr

V1_CLASS = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
V2_CLASS = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
NAMES = {1: "zero", 2: "genuine"}


class _LoraReq:
    def __init__(self, lora_int_id):
        self.lora_int_id = lora_int_id
        self.lora_name = NAMES[lora_int_id]


class _NewReq:
    def __init__(self, req_id, lora_id):
        self.req_id = req_id
        self.lora_request = _LoraReq(lora_id) if lora_id else None


class _SchedOut:
    """A SchedulerOutput stand-in: new requests, the scheduled set, finished."""

    def __init__(self, new=(), scheduled=None, finished=()):
        self.scheduled_new_reqs = [_NewReq(r, l) for r, l in new]
        sched = list(scheduled) if scheduled is not None else [r for r, _ in new]
        self.num_scheduled_tokens = {r: 1 for r in sched}
        self.finished_req_ids = set(finished)
        self.preempted_req_ids = None


class _V1Batch:
    """A V1 InputBatch stand-in built from the live requests."""

    def __init__(self, live, *, max_num_reqs=8, none_slot=False):
        rows = list(live.items())
        if none_slot:
            rows.insert(1, (None, 0))
        self._req_ids = [r for r, _ in rows]
        self.request_lora_mapping = np.zeros((max_num_reqs,), dtype=np.int64)
        for i, (_r, lid) in enumerate(rows):
            self.request_lora_mapping[i] = lid
        self.lora_id_to_lora_request = {lid: _LoraReq(lid) for _r, lid in rows if lid}
        self.req_id_to_index = {r: i for i, (r, _) in enumerate(rows)}   # None counted: num_reqs spans it

    @property
    def req_ids(self):
        return self._req_ids

    @property
    def num_reqs(self):
        return len(self.req_id_to_index)


class _V1Runner:
    """The V1 model runner: applies each step's scheduler output to its
    ``input_batch`` (re-created every step, as gpu_model_runner.py:7056 may),
    with an optional bookkeeping DISAGREEMENT (a rid bound elsewhere)."""

    def __init__(self, *, disagree=None, none_slot=False):
        self.live = {}
        self.calls = []
        self.disagree = dict(disagree or {})
        self.none_slot = none_slot
        self.input_batch = _V1Batch({})

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self.calls.append(scheduler_output)
        for r in scheduler_output.finished_req_ids:
            self.live.pop(r, None)
        for nr in scheduler_output.scheduled_new_reqs:
            lid = nr.lora_request.lora_int_id if nr.lora_request else 0
            self.live[nr.req_id] = self.disagree.get(nr.req_id, lid)
        self.input_batch = _V1Batch(self.live, none_slot=self.none_slot)
        return {"step": len(self.calls)}


_V1Runner.__module__, _V1Runner.__qualname__ = V1_CLASS.rsplit(".", 1)


class _ReqStates:
    def __init__(self):
        self.req_id_to_index = {}
        self.index_to_req_id = {}

    def num_reqs(self):
        return len(self.req_id_to_index)


class _LoraState:
    def __init__(self, max_num_reqs=8):
        self.lora_ids = np.zeros(max_num_reqs, dtype=np.int32)
        self.lora_requests = {}


class _V2Runner:
    """The V2 model runner (gpu/model_runner.py @0.25.1): finish_requests then
    add_requests INSIDE execute_model; state in req_states + lora_state; no
    input_batch."""

    def __init__(self, *, disagree=None):
        self.req_states = _ReqStates()
        self.lora_state = _LoraState()
        self.calls = []
        self.disagree = dict(disagree or {})

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self.calls.append(scheduler_output)
        for r in scheduler_output.finished_req_ids:
            idx = self.req_states.req_id_to_index.pop(r, None)
            if idx is not None:
                self.req_states.index_to_req_id.pop(idx, None)
            self.lora_state.lora_requests.pop(r, None)
        for nr in scheduler_output.scheduled_new_reqs:
            idx = next(i for i in range(8) if i not in self.req_states.index_to_req_id)
            self.req_states.req_id_to_index[nr.req_id] = idx
            self.req_states.index_to_req_id[idx] = nr.req_id
            lreq = nr.lora_request
            if nr.req_id in self.disagree:
                lreq = _LoraReq(self.disagree[nr.req_id]) if self.disagree[nr.req_id] else None
            if lreq is not None:
                self.lora_state.lora_requests[nr.req_id] = lreq
                self.lora_state.lora_ids[idx] = lreq.lora_int_id
            else:
                self.lora_state.lora_ids[idx] = 0
        return {"step": len(self.calls)}


_V2Runner.__module__, _V2Runner.__qualname__ = V2_CLASS.rsplit(".", 1)


class _OtherRunner:
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        return "x"


E = {"base": "chatcmpl-b-0a1b2c3d", "zero": "chatcmpl-z-0a1b2c3d", "gen": "chatcmpl-g-0a1b2c3d"}


@pytest.mark.parametrize("make", [_V1Runner, _V2Runner], ids=["v1", "v2"])
def test_primary_binds_from_the_scheduler_output_and_the_secondary_agrees(make):
    runner = make()
    rec = wr.LoraRoutingRecorder(runner)
    assert rec.runner_class in (V1_CLASS, V2_CLASS) and rec.runner_kind in ("v1", "v2")
    epoch = "c" * 32                                                   # the CLIENT's id
    armed = rec.arm(epoch)
    assert armed == {"ok": True, "armed": True, "epoch": epoch, "already_armed": False,
                     "runner_class": rec.runner_class, "runner_kind": rec.runner_kind}
    assert rec.arm(epoch) == {**armed, "already_armed": True}          # idempotent for the live epoch
    # step 1: three new requests, all scheduled — one finishes INSIDE the same step
    out = runner.execute_model(_SchedOut(new=[(E["base"], 0), (E["zero"], 1), (E["gen"], 2)],
                                         finished=[E["gen"]]))
    assert out == {"step": 1}                                          # the return value untouched
    # step 2: decode for the two survivors
    runner.execute_model(_SchedOut(scheduled=[E["base"], E["zero"]]))
    p = rec.observations_payload(epoch)
    assert p["ok"] is True and p["epoch"] == epoch and p["armed"] is True and p["runner_kind"] == rec.runner_kind
    # the finished-in-step request was bound by the PRIMARY (pre-call) read
    assert p["observations"] == {E["base"]: {"lora_int_id": 0, "lora_name": None},
                                 E["zero"]: {"lora_int_id": 1, "lora_name": "zero"},
                                 E["gen"]: {"lora_int_id": 2, "lora_name": "genuine"}}
    assert p["inconsistent"] == {} and p["snapshot_errors"] == 0 and p["errors"] == []
    assert p["steps"] == [sorted(E.values()), sorted([E["base"], E["zero"]])]
    assert p["n_steps"] == 2 and p["overflow"] is False and p["steps_overflow"] is False
    json.dumps(p)
    d = rec.disarm(epoch)
    assert d["disarmed"] is True and d["was_armed"] is True and d["restored"] is True
    assert d["armed"] is False and "observations" not in d and d["epoch"] == epoch
    assert "execute_model" not in vars(runner)                         # the class method is back
    runner.execute_model(_SchedOut(new=[("late", 0)]))
    assert rec.n_steps == 2                                            # no longer recording


@pytest.mark.parametrize("make", [_V1Runner, _V2Runner], ids=["v1", "v2"])
def test_secondary_disagreement_is_recorded_as_inconsistent(make):
    """The runner's own state binds a request to another adapter than the
    scheduler output said: the first (primary) binding stands, the
    disagreement goes under inconsistent — the check refuses it by name."""
    runner = make(disagree={E["zero"]: 2})
    rec = wr.LoraRoutingRecorder(runner)
    epoch = "d" * 32
    rec.arm(epoch)
    runner.execute_model(_SchedOut(new=[(E["base"], 0), (E["zero"], 1)]))
    p = rec.observations_payload(epoch)
    assert p["observations"][E["zero"]] == {"lora_int_id": 1, "lora_name": "zero"}
    assert p["inconsistent"] == {E["zero"]: [{"lora_int_id": 1, "lora_name": "zero"},
                                             {"lora_int_id": 2, "lora_name": "genuine"}]}
    assert p["n_inconsistent"] == 1 and p["snapshot_errors"] == 0


def test_unknown_runner_class_is_recorded_without_a_secondary():
    runner = _OtherRunner()
    rec = wr.LoraRoutingRecorder(runner)
    assert rec.runner_kind == "unknown" and rec.runner_class.endswith("._OtherRunner")
    epoch = "f" * 32
    rec.arm(epoch)
    assert runner.execute_model(_SchedOut(new=[(E["base"], 0)])) == "x"
    p = rec.observations_payload(epoch)
    assert p["runner_kind"] == "unknown" and p["observations"][E["base"]]["lora_int_id"] == 0
    assert p["snapshot_errors"] == 0                                   # no secondary attempted


def test_v1_none_slots_are_skipped_and_the_batch_is_re_read_every_step():
    runner = _V1Runner(none_slot=True)
    rec = wr.LoraRoutingRecorder(runner)
    epoch = "a" * 32
    rec.arm(epoch)
    runner.execute_model(_SchedOut(new=[(E["base"], 0), (E["gen"], 2)]))
    first_batch = runner.input_batch
    runner.execute_model(_SchedOut(new=[(E["zero"], 1)]))
    assert runner.input_batch is not first_batch                       # re-created, re-read
    p = rec.observations_payload(epoch)
    assert set(p["observations"]) == set(E.values()) and p["inconsistent"] == {}
    assert p["snapshot_errors"] == 0


def test_epochs_reset_and_refuse_mismatches_and_double_arming():
    runner = _V2Runner()
    rec = wr.LoraRoutingRecorder(runner)
    e1 = "1" * 32
    assert rec.arm(e1)["epoch"] == e1
    runner.execute_model(_SchedOut(new=[(E["base"], 0), (E["gen"], 2)]))
    with pytest.raises(wr.LoraRoutingRefusal, match=f"already_armed.*{e1}") as ei:
        rec.arm("2" * 32)                                                # a DIFFERENT epoch
    assert (ei.value.refusal, ei.value.live_epoch) == ("already_armed", e1)   # typed, a LensWorkerError
    assert isinstance(ei.value, wr.LensWorkerError)
    assert ei.value.payload(runner_kind="v2")["refusal"] == "already_armed"
    assert rec.arm(e1)["already_armed"] is True and rec.n_steps == 1      # the live one: idempotent, no reset
    for bad in ("", None, 7):
        with pytest.raises(wr.LensWorkerError, match="epoch must be a non-empty string"):
            rec.arm(bad)
    with pytest.raises(wr.LoraRoutingRefusal, match="epoch_mismatch: observations") as ei:
        rec.observations_payload("someone-elses")
    assert ei.value.refusal == "epoch_mismatch" and ei.value.live_epoch == e1
    with pytest.raises(wr.LensWorkerError, match="epoch_mismatch: disarm"):
        rec.disarm("someone-elses")
    assert rec.observations_payload(e1)["steps"] == [sorted([E["base"], E["gen"]])]
    rec.disarm(e1)
    # the second session: a fresh epoch, NOTHING inherited (the regression's
    # recorder half: the mixed step above is not in this epoch's list)
    e2 = "2" * 32
    assert rec.arm(e2)["epoch"] == e2 and rec.arm(e2)["already_armed"] is True
    runner.execute_model(_SchedOut(new=[(E["zero"], 1)], finished=[E["base"], E["gen"]]))
    p = rec.observations_payload(e2)
    assert p["epoch"] == e2 and p["steps"] == [[E["zero"]]] and p["n_steps"] == 1
    assert set(p["observations"]) == {E["zero"]} and p["inconsistent"] == {}
    with pytest.raises(wr.LensWorkerError, match="epoch_mismatch"):
        rec.observations_payload(e1)                                   # the old epoch is gone
    d = rec.disarm(e2)
    assert d["was_armed"] is True
    # disarm of the same epoch again: idempotent, nothing to restore
    d2 = rec.disarm(e2)
    assert d2["was_armed"] is False and d2["restored"] is False


def test_the_caps_set_overflow_flags_and_counting_continues():
    runner = _V2Runner()
    rec = wr.LoraRoutingRecorder(runner, cap=2, steps_cap=2)
    epoch = "b" * 32
    rec.arm(epoch)
    runner.execute_model(_SchedOut(new=[("a", 0), ("b", 1)]))
    runner.execute_model(_SchedOut(new=[("c", 2)], scheduled=["a", "b", "c"]))
    runner.execute_model(_SchedOut(scheduled=["a"]))
    p = rec.observations_payload(epoch)
    assert p["overflow"] is True and set(p["observations"]) == {"a", "b"}
    assert p["steps_overflow"] is True and len(p["steps"]) == 2 and p["n_steps"] == 3


def test_read_failures_are_recorded_not_raised_into_the_engine_step():
    class _BadSched:
        scheduled_new_reqs = []
        num_scheduled_tokens = None                                    # the recorder's read breaks here
        finished_req_ids = set()
    runner = _V2Runner()
    rec = wr.LoraRoutingRecorder(runner)
    epoch = "e" * 32
    rec.arm(epoch)
    assert runner.execute_model(_BadSched()) == {"step": 1}            # the step completed
    p = rec.observations_payload(epoch)
    assert p["snapshot_errors"] == 1 and p["errors"][0].startswith("pre-step: AttributeError")
    # a secondary read failure (the V2 lora ids unreadable) is recorded too — the fake
    # runner only WRITES lora_ids, the recorder reads them
    class _Unreadable:
        def __setitem__(self, k, v):
            pass

        def __getitem__(self, k):
            raise RuntimeError("torn")
    runner.lora_state.lora_ids = _Unreadable()
    runner.execute_model(_SchedOut(new=[("r", 1)]))
    p = rec.observations_payload(epoch)
    assert p["snapshot_errors"] == 2 and "post-step (v2)" in p["errors"][1]
    assert p["observations"]["r"] == {"lora_int_id": 1, "lora_name": "zero"}   # primary still bound it


def test_disarm_puts_back_a_pre_existing_instance_attribute_and_leaves_a_foreign_rebinding():
    runner = _V1Runner()
    marker = lambda *a, **k: "prior"                                    # noqa: E731
    runner.execute_model = marker
    rec = wr.LoraRoutingRecorder(runner)
    epoch = "9" * 32
    rec.arm(epoch)
    assert runner.execute_model(_SchedOut()) == "prior" and rec.n_steps == 1
    rec.disarm(epoch)
    assert vars(runner)["execute_model"] is marker
    runner2 = _V1Runner()
    rec2 = wr.LoraRoutingRecorder(runner2)
    e = "8" * 32
    rec2.arm(e)
    foreign = lambda *a, **k: "foreign"                                 # noqa: E731
    runner2.execute_model = foreign
    d = rec2.disarm(e)
    assert d["restored"] is False and vars(runner2)["execute_model"] is foreign
    assert d["snapshot_errors"] == 1 and "no longer this recorder's wrapper" in d["errors"][0]


def test_recorder_refuses_a_missing_runner_or_execute_model_and_decodes_epoch_args():
    with pytest.raises(wr.LensWorkerError, match="no model_runner"):
        wr.LoraRoutingRecorder(None)

    class _NoExec:
        pass
    with pytest.raises(wr.LensWorkerError, match="no callable execute_model"):
        wr.LoraRoutingRecorder(_NoExec()).arm("e" * 32)
    assert wr.decode_epoch_arg("abc") == "abc"
    for bad in (None, "", 3, ["abc"]):
        with pytest.raises(wr.LensWorkerError, match="epoch must be a non-empty string"):
            wr.decode_epoch_arg(bad)


def test_model_runner_report_names_the_pinned_classes_and_the_rest_unknown():
    assert wr.model_runner_report(_V1Runner()) == {"model_runner_class": V1_CLASS, "model_runner_kind": "v1"}
    assert wr.model_runner_report(_V2Runner()) == {"model_runner_class": V2_CLASS, "model_runner_kind": "v2"}
    other = wr.model_runner_report(_OtherRunner())
    assert other["model_runner_kind"] == "unknown" and other["model_runner_class"].endswith("._OtherRunner")
    with pytest.raises(wr.LensWorkerError, match="no model_runner"):
        wr.model_runner_report(None)
    assert sa.MODEL_RUNNER_CLASSES == {V1_CLASS: "v1", V2_CLASS: "v2"} and sa.MODEL_RUNNER_KINDS == ("v1", "v2")


# --------------------------------------------------------------------------
# the extension's RPCs on a fake worker
# --------------------------------------------------------------------------


def test_extension_rpcs_arm_observe_disarm_with_epochs_on_the_worker_instance():
    class _Self(nw.NativeWorkerExtension):
        pass
    worker = _Self()
    worker.model_runner = _V2Runner()
    # never armed: observations / disarm answer the not_armed PAYLOAD (no silence-as-agreement)
    r = worker.sisteer_lora_routing_observations("e")
    assert (r["ok"], r["refusal"], r["runner_kind"]) == (False, "not_armed", None) and "never armed" in r["detail"]
    r = worker.sisteer_disarm_lora_routing_recorder("e")
    assert (r["ok"], r["refusal"]) == (False, "not_armed")
    # REFUSALS ARE PAYLOADS (transport): never an exception out of the method,
    # so the pinned executor's per-worker reply queues stay aligned on a TP>1 launch
    r = worker.sisteer_arm_lora_routing_recorder()                          # arm takes the epoch
    assert r["ok"] is False and r["refusal"] == "bad_epoch" and "non-empty string" in r["detail"]
    assert r["live_epoch"] is None and r["runner_kind"] == "v2"
    epoch = "c" * 32
    armed = worker.sisteer_arm_lora_routing_recorder(epoch)
    assert armed["ok"] is True and armed["armed"] is True and armed["runner_kind"] == "v2"
    assert armed["epoch"] == epoch and armed["already_armed"] is False
    assert isinstance(worker.sisteer_lora_routing_recorder, wr.LoraRoutingRecorder)   # instance state
    assert worker.sisteer_arm_lora_routing_recorder(epoch)["already_armed"] is True   # idempotent
    r = worker.sisteer_arm_lora_routing_recorder("d" * 32)
    assert r == {"ok": False, "refusal": "already_armed", "detail": r["detail"], "live_epoch": epoch,
                 "runner_kind": "v2", "runner_class": V2_CLASS}
    assert "armed by epoch " + epoch in r["detail"]
    worker.model_runner.execute_model(_SchedOut(new=[(E["base"], 0), (E["gen"], 2)]))
    p = worker.sisteer_lora_routing_observations(epoch)
    assert p["ok"] is True
    assert set(p["observations"]) == {E["base"], E["gen"]} and p["steps"] == [sorted([E["base"], E["gen"]])]
    r = worker.sisteer_lora_routing_observations("other")
    assert (r["ok"], r["refusal"], r["live_epoch"]) == (False, "epoch_mismatch", epoch)
    r = worker.sisteer_lora_routing_observations(None)
    assert (r["ok"], r["refusal"]) == (False, "bad_epoch")
    r = worker.sisteer_disarm_lora_routing_recorder("other")
    assert (r["ok"], r["refusal"], r["live_epoch"]) == (False, "epoch_mismatch", epoch)
    assert worker.sisteer_lora_routing_recorder.armed is True               # a refused disarm changed nothing
    d = worker.sisteer_disarm_lora_routing_recorder(epoch)
    assert d["ok"] is True and d["disarmed"] is True and d["was_armed"] is True and d["restored"] is True
    assert "execute_model" not in vars(worker.model_runner)
    assert all(k in wr.LORA_ROUTING_REFUSALS for k in ("already_armed", "epoch_mismatch", "bad_epoch", "not_armed"))
    # the readiness report carries the runner
    assert wr.model_runner_report(worker.model_runner) == {"model_runner_class": V2_CLASS,
                                                           "model_runner_kind": "v2"}


def test_extension_arm_refuses_without_a_model_runner():
    class _Self(nw.NativeWorkerExtension):
        pass
    with pytest.raises(wr.LensWorkerError, match="no model_runner"):
        _Self().sisteer_arm_lora_routing_recorder("e" * 32)


def test_extension_defines_exactly_the_four_rpcs_and_no_class_state():
    """vLLM's injection (worker_base.py @0.25.1) asserts NO non-dunder
    attribute of the extension exists on the Worker: the recorder is
    INSTANCE state set at arm time, so the class carries only its methods.
    Five since native serving became the default: the Nemotron backend
    report ``sisteer_mamba_backend_report`` was PORTED from the lens
    worker so a native Nemotron standup attests its fused kernels; seven
    since native PSR serving: the residual slot's report and the
    per-request failure claim (the test's name keeps its historical "four"
    so the suite's baseline ids stay stable)."""
    attrs = [a for a in dir(nw.NativeWorkerExtension) if not a.startswith("__")]
    assert attrs == sorted(["sisteer_scheduler_config", "sisteer_arm_lora_routing_recorder",
                            "sisteer_lora_routing_observations",
                            "sisteer_disarm_lora_routing_recorder",
                            "sisteer_mamba_backend_report",
                            "sisteer_residual_slot_report",
                            "sisteer_take_request_failure"])
    assert all(a.startswith("sisteer_") for a in attrs)
    assert "sisteer_lora_routing_recorder" not in attrs
