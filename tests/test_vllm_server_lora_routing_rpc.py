"""The launcher's LoRA routing recorder RPC wrappers: ``arm_lora_routing_
recorder(epoch)`` / ``lora_routing_observations(epoch)`` / ``disarm_lora_
routing_recorder(epoch)`` over a FAKE ``_post_collective_rpc`` — the method
names, the CLIENT-minted epoch as the one string arg of all three, the
per-worker result reader (empty / non-mapping / disagreeing workers refuse;
agreeing workers collapse to one payload with ``n_workers``), refusals
propagating, the non-lora-native refusal (a lens launch has no recorder) —
``validate_model_runner_reports`` (every worker names one of the two pinned
GPU model runners and they agree; canary #4 ran V2) — and THE MULTI-WORKER
REGRESSION (serving core): two independently
instantiated recorders behind one RPC adopt the client's epoch, agree, and
disarm together; the pre-fix shape (each worker minting its own) is pinned as
the refusal it was. No server."""
from __future__ import annotations

from dataclasses import replace

import pytest

from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import vllm_server as vs

V1 = "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
V2 = "vllm.v1.worker.gpu.model_runner.GPUModelRunner"


def _launch(lora_native=True):
    return vs.ServerLaunch(cmd=("x",), env={}, host="127.0.0.1", port=1, model="m",
                           lora_native=lora_native, api_key="k")


def test_method_names_are_the_leafs_and_the_extension_implements_them():
    from sisteer.capeval.serving import native_worker as nw
    assert vs.LORA_ROUTING_RPC_METHODS == sa.LORA_ROUTING_RPC_METHODS == (
        "sisteer_arm_lora_routing_recorder", "sisteer_lora_routing_observations",
        "sisteer_disarm_lora_routing_recorder")
    for m in vs.LORA_ROUTING_RPC_METHODS:
        assert callable(getattr(nw.NativeWorkerExtension, m))


def test_epoch_args_are_the_one_string_arg():
    assert vs.lora_routing_epoch_args("abc123") == ["abc123"]
    for bad in (None, "", 3, ["a"]):
        with pytest.raises(vs.VllmServerError, match="epoch must be the non-empty string"):
            vs.lora_routing_epoch_args(bad)


def test_rpc_result_reader_collapses_agreeing_workers_and_refuses_the_rest():
    one = {"ok": True, "armed": True, "epoch": "e"}
    assert vs.lora_routing_rpc_result({"results": [one]}, what="w") == {**one, "n_workers": 1}
    assert vs.lora_routing_rpc_result({"results": [one, dict(one)]}, what="w")["n_workers"] == 2
    for bad in ({}, {"results": []}, {"results": None}, [], None, "x"):
        with pytest.raises(vs.VllmServerError, match="expected one recorder payload PER WORKER"):
            vs.lora_routing_rpc_result(bad, what="w")
    with pytest.raises(vs.VllmServerError, match="worker 1 answered a non-mapping payload"):
        vs.lora_routing_rpc_result({"results": [one, "oops"]}, what="w")
    with pytest.raises(vs.VllmServerError, match="worker 1's recorder payload disagrees"):
        vs.lora_routing_rpc_result({"results": [one, {**one, "epoch": "f"}]}, what="w")
    # REFUSAL PAYLOADS (transport): collected from every worker, then raised naming
    # the kind and the workers; a mixed accepted/refused set is itself a refusal
    ref = {"ok": False, "refusal": "already_armed", "detail": "armed by epoch x", "live_epoch": "x"}
    with pytest.raises(vs.VllmServerError, match=r"worker\(s\) \[0, 1\] refused — already_armed: 'armed by epoch x' \(live epoch 'x'\)") as ei:
        vs.lora_routing_rpc_result({"results": [ref, dict(ref)]}, what="w")
    assert "mixed" not in str(ei.value)
    with pytest.raises(vs.VllmServerError, match=r"worker\(s\) \[1\] refused — epoch_mismatch.*worker\(s\) \[0\] accepted — a mixed set is itself a refusal"):
        vs.lora_routing_rpc_result({"results": [one, {**ref, "refusal": "epoch_mismatch"}]}, what="w")
    with pytest.raises(vs.VllmServerError, match="refused — already_armed, epoch_mismatch"):
        vs.lora_routing_rpc_result({"results": [ref, {**ref, "refusal": "epoch_mismatch"}]}, what="w")
    # a success MUST say ok: True — a payload without the field is not the recorder's
    with pytest.raises(vs.VllmServerError, match="payload without an 'ok' field"):
        vs.lora_routing_rpc_result({"results": [{"armed": True, "epoch": "e"}]}, what="w")


def test_wrappers_post_the_right_method_and_the_epoch_and_read_one_payload(monkeypatch):
    calls = []
    obs = {"ok": True, "armed": True, "epoch": "e1",
           "observations": {"r-0a1b2c3d": {"lora_int_id": 1, "lora_name": "z"}},
           "inconsistent": {}, "steps": [["r-0a1b2c3d"]], "n_steps": 1, "overflow": False,
           "steps_overflow": False, "runner_kind": "v2", "runner_class": V2}

    def fake_post(launch, method, args, *, timeout_s, what):
        calls.append((method, list(args), timeout_s, what))
        if method == vs.LORA_ROUTING_OBSERVE_RPC_METHOD:
            return {"results": [obs]}
        if method == vs.LORA_ROUTING_ARM_RPC_METHOD:
            return {"results": [{"ok": True, "armed": True, "epoch": "e1", "runner_kind": "v2"}]}
        return {"results": [{"ok": True, "armed": False, "disarmed": True, "was_armed": True, "epoch": "e1"}]}
    monkeypatch.setattr(vs, "_post_collective_rpc", fake_post)
    launch = _launch()
    armed = vs.arm_lora_routing_recorder(launch, "e1")
    assert armed == {"ok": True, "armed": True, "epoch": "e1", "runner_kind": "v2", "n_workers": 1}
    got = vs.lora_routing_observations(launch, armed["epoch"], timeout_s=5.0)
    assert got == {**obs, "n_workers": 1}
    assert vs.disarm_lora_routing_recorder(launch, "e1")["disarmed"] is True
    assert [c[0] for c in calls] == list(vs.LORA_ROUTING_RPC_METHODS)
    assert calls[0][1] == ["e1"] and calls[1][1] == ["e1"] and calls[2][1] == ["e1"]   # the CLIENT's epoch
    assert calls[1][2] == 5.0 and calls[0][2] == vs.LORA_ROUTING_RPC_TIMEOUT_S
    with pytest.raises(vs.VllmServerError, match="epoch must be the non-empty string"):
        vs.arm_lora_routing_recorder(launch, "")


def test_wrapper_refusals_propagate_and_a_lens_launch_refuses_before_posting(monkeypatch):
    def failing(*a, **k):
        raise vs.VllmServerError("HTTP 500 — a worker raised")
    monkeypatch.setattr(vs, "_post_collective_rpc", failing)
    with pytest.raises(vs.VllmServerError, match="a worker raised"):
        vs.arm_lora_routing_recorder(_launch(), "e")
    posted = []
    monkeypatch.setattr(vs, "_post_collective_rpc", lambda *a, **k: posted.append(1) or {"results": [{"ok": True}]})
    lens = replace(_launch(), lora_native=False)
    with pytest.raises(vs.VllmServerError, match="not lora-native"):
        vs.arm_lora_routing_recorder(lens, "e")
    with pytest.raises(vs.VllmServerError, match="not lora-native"):
        vs.lora_routing_observations(lens, "e")
    with pytest.raises(vs.VllmServerError, match="not lora-native"):
        vs.disarm_lora_routing_recorder(lens, "e")
    assert posted == []


def test_model_runner_reports_are_held_to_the_two_pinned_classes():
    ok = {"model_runner_class": V2, "model_runner_kind": "v2"}
    assert vs.validate_model_runner_reports([ok]) == {"class": V2, "kind": "v2", "n_workers": 1}
    v1 = {"model_runner_class": V1, "model_runner_kind": "v1"}
    assert vs.validate_model_runner_reports([v1, dict(v1)])["n_workers"] == 2
    with pytest.raises(vs.VllmServerError, match="no worker reports"):
        vs.validate_model_runner_reports([])
    with pytest.raises(vs.VllmServerError, match="reports no model_runner_class"):
        vs.validate_model_runner_reports([{"enforce_eager": False}])
    with pytest.raises(vs.VllmServerError, match="not one of the pinned GPU model runners") as ei:
        vs.validate_model_runner_reports([{"model_runner_class": "tests.Fake", "model_runner_kind": "unknown"}])
    assert "unpinned engine" in str(ei.value)
    with pytest.raises(vs.VllmServerError, match="not one of the pinned"):
        vs.validate_model_runner_reports([{"model_runner_class": V2, "model_runner_kind": "v1"}])   # kind lies
    with pytest.raises(vs.VllmServerError, match="workers disagree"):
        vs.validate_model_runner_reports([ok, v1])
    # the attestation carries it as an additive, validated key
    body = {"class": V2, "kind": "v2"}
    sa._validate_model_runner_key(body, where="t")
    with pytest.raises(sa.ServingAttestationError, match="not one of the pinned GPU model runners"):
        sa._validate_model_runner_key({"class": "x.Y", "kind": "v2"}, where="t")
    with pytest.raises(sa.ServingAttestationError):
        sa._validate_model_runner_key({"class": V2, "kind": "v1"}, where="t")


# --------------------------------------------------------------------------
# THE MULTI-WORKER REGRESSION (serving core)
# --------------------------------------------------------------------------


class _Sched:
    def __init__(self, new=()):
        self.scheduled_new_reqs = []
        self.num_scheduled_tokens = {r: 1 for r in new}
        self.finished_req_ids = set()
        self.preempted_req_ids = None


class _Runner:
    """A model runner shaped like the pinned V2 class (the recorder's kind
    dispatch is by class name), one per worker."""

    def __init__(self):
        class _RS:
            req_id_to_index = {}
            index_to_req_id = {}
        class _LS:
            lora_ids = [0] * 8
            lora_requests = {}
        self.req_states, self.lora_state = _RS(), _LS()

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        return "ok"


_Runner.__module__, _Runner.__qualname__ = V2.rsplit(".", 1)


def _two_worker_group(monkeypatch):
    """Two REAL extension instances over two runners behind one fake
    collective_rpc: the route calls every worker's method with the same args
    and returns one result per worker, as the dev route does."""
    from sisteer.capeval.serving import native_worker as nw

    class _Worker(nw.NativeWorkerExtension):
        pass
    workers = []
    for _ in range(2):
        w = _Worker()
        w.model_runner = _Runner()
        workers.append(w)

    def fake_post(launch, method, args, *, timeout_s, what):
        results = []
        for w in workers:
            try:
                results.append(getattr(w, method)(*args))
            except Exception as e:  # the dev route surfaces a worker exception as a 5xx
                raise vs.VllmServerError(f"{what} RPC failed with HTTP 500 — a worker raised: {e}") from e
        return {"results": results}
    monkeypatch.setattr(vs, "_post_collective_rpc", fake_post)
    return workers


class _PinnedExecutor:
    """The pinned 0.25.1 ``MultiprocExecutor.collective_rpc`` transport, modelled
    here: every worker runs the method and pushes
    ``("SUCCESS", result)`` or ``("FAILURE", exc)`` onto ITS OWN response
    queue; the client reads the queues SEQUENTIALLY and raises on the FIRST
    FAILURE without draining the later workers' replies; the queues carry no
    RPC identifier, so an undrained reply is read by the NEXT call."""

    def __init__(self, workers):
        import collections
        self.workers = list(workers)
        self.queues = [collections.deque() for _ in workers]

    def collective_rpc(self, method, args):
        for w, q in zip(self.workers, self.queues):
            try:
                q.append(("SUCCESS", getattr(w, method)(*args)))
            except Exception as e:  # noqa: BLE001 — the worker's FAILURE reply
                q.append(("FAILURE", e))
        results = []
        for q in self.queues:
            status, out = q.popleft()
            if status != "SUCCESS":
                raise vs.VllmServerError(f"RPC failed with HTTP 500 — a worker raised: {out}")
            results.append(out)
        return {"results": results}

    def post(self, launch, method, args, *, timeout_s, what):
        return self.collective_rpc(method, args)

    def queued(self):
        return [len(q) for q in self.queues]


def _pinned_group(monkeypatch, *, pre_fix=False):
    """Two REAL extension instances behind the pinned transport. ``pre_fix``
    re-creates 114cb343's shape — refusal payloads RAISED out of the method
    (the executor then sees a FAILURE)."""
    from sisteer.capeval.serving import native_worker as nw
    from sisteer.capeval.serving import worker_reports as wr

    class _Worker(nw.NativeWorkerExtension):
        pass
    if pre_fix:
        def raising(method_name):
            real = getattr(nw.NativeWorkerExtension, method_name)

            def m(self, *a):
                out = real(self, *a)
                if isinstance(out, dict) and out.get("ok") is False:
                    raise wr.LensWorkerError(f"{out['refusal']}: {out['detail']}")
                return out
            return m
        for name in vs.LORA_ROUTING_RPC_METHODS:
            setattr(_Worker, name, raising(name))
    workers = []
    for _ in range(2):
        w = _Worker()
        w.model_runner = _Runner()
        workers.append(w)
    ex = _PinnedExecutor(workers)
    monkeypatch.setattr(vs, "_post_collective_rpc", ex.post)
    return workers, ex


def test_a_rejected_competing_control_leaves_the_owner_usable_and_every_queue_drained(monkeypatch):
    """Transport: refusals are payloads, so the competing
    control's arm and its best-effort disarm are SUCCESS replies on every
    worker's queue, read together and refused by the client; the owner's next
    observe and disarm read their own replies; every queue is empty after
    every call."""
    workers, ex = _pinned_group(monkeypatch)
    launch = _launch()
    a, b = "a" * 32, "b" * 32
    assert vs.arm_lora_routing_recorder(launch, a)["n_workers"] == 2 and ex.queued() == [0, 0]
    # control B competes: refused by the CLIENT after both workers answered
    with pytest.raises(vs.VllmServerError, match=r"worker\(s\) \[0, 1\] refused — already_armed") as ei:
        vs.arm_lora_routing_recorder(launch, b)
    assert f"live epoch '{a}'" in str(ei.value) and ex.queued() == [0, 0]
    # B's best-effort disarm with its own epoch: refused, queues still aligned
    with pytest.raises(vs.VllmServerError, match="refused — epoch_mismatch"):
        vs.disarm_lora_routing_recorder(launch, b)
    assert ex.queued() == [0, 0]
    # the OWNER goes on: its observe and disarm read THEIR OWN replies
    for w in workers:
        w.model_runner.execute_model(_Sched(new=["r-0a1b2c3d"]))
    obs = vs.lora_routing_observations(launch, a)
    assert obs["ok"] is True and obs["epoch"] == a and obs["steps"] == [["r-0a1b2c3d"]] and obs["n_workers"] == 2
    d = vs.disarm_lora_routing_recorder(launch, a)
    assert d["ok"] is True and d["disarmed"] is True and d["n_workers"] == 2 and ex.queued() == [0, 0]
    assert all(not w.sisteer_lora_routing_recorder.armed for w in workers)
    # after the disarm the owner may still read ITS epoch's final evidence (armed False);
    # any other epoch is refused — and every refusal still drains the queues
    final = vs.lora_routing_observations(launch, a)
    assert final["ok"] is True and final["armed"] is False and final["epoch"] == a
    with pytest.raises(vs.VllmServerError, match="refused — epoch_mismatch"):
        vs.lora_routing_observations(launch, "z" * 32)
    assert ex.queued() == [0, 0]


def test_the_pre_fix_shape_raised_refusals_desynced_the_owners_later_calls(monkeypatch):
    """114cb343 raised expected refusals out of the extension methods. Under
    the pinned transport: B's competing arm makes both workers FAIL, the
    first failure is consumed and the second stays queued; B's best-effort
    disarm queues two more failures; the OWNER's next valid observe then
    reads B's stale already_armed failure as its own. Pinned as the desync it
    was."""
    workers, ex = _pinned_group(monkeypatch, pre_fix=True)
    launch = _launch()
    a, b = "a" * 32, "b" * 32
    vs.arm_lora_routing_recorder(launch, a)
    with pytest.raises(vs.VllmServerError, match="already_armed"):
        vs.arm_lora_routing_recorder(launch, b)
    assert ex.queued() == [0, 1]                                      # worker 1's failure left queued
    with pytest.raises(vs.VllmServerError, match="epoch_mismatch"):
        vs.disarm_lora_routing_recorder(launch, b)
    assert ex.queued() == [0, 2]
    for w in workers:
        w.model_runner.execute_model(_Sched(new=["r-0a1b2c3d"]))
    # the owner's VALID observation is answered with B's stale already_armed failure
    with pytest.raises(vs.VllmServerError, match="already_armed"):
        vs.lora_routing_observations(launch, a)
    assert ex.queued() == [0, 2]                                      # and the desync persists


def test_two_workers_adopt_the_clients_epoch_agree_and_disarm_together(monkeypatch):
    workers = _two_worker_group(monkeypatch)
    launch = _launch()
    epoch = "c" * 32
    armed = vs.arm_lora_routing_recorder(launch, epoch)
    assert armed["ok"] is True and armed["epoch"] == epoch and armed["n_workers"] == 2
    assert armed["already_armed"] is False
    assert all(w.sisteer_lora_routing_recorder.epoch == epoch for w in workers)
    # re-arming the live epoch is idempotent on every worker; a different epoch is refused by
    # every worker AS A PAYLOAD and raised by the client naming both workers
    assert vs.arm_lora_routing_recorder(launch, epoch)["already_armed"] is True
    with pytest.raises(vs.VllmServerError, match=r"worker\(s\) \[0, 1\] refused — already_armed"):
        vs.arm_lora_routing_recorder(launch, "d" * 32)
    for w in workers:
        w.model_runner.execute_model(_Sched(new=["r-0a1b2c3d"]))
    obs = vs.lora_routing_observations(launch, epoch)
    assert obs["epoch"] == epoch and obs["steps"] == [["r-0a1b2c3d"]] and obs["n_workers"] == 2
    d = vs.disarm_lora_routing_recorder(launch, epoch)
    assert d["disarmed"] is True and d["was_armed"] is True and d["n_workers"] == 2
    assert all(not w.sisteer_lora_routing_recorder.armed for w in workers)          # disarmed TOGETHER
    assert all("execute_model" not in vars(w.model_runner) for w in workers)


def test_the_pre_fix_shape_per_worker_minted_epochs_refused_and_left_every_worker_armed(monkeypatch):
    """ceba8874's arm() minted uuid4 PER WORKER: the two payloads disagree, the
    reader refuses before any try/finally, and no single epoch can disarm
    them. Pinned as the refusal it was by giving each worker its own id."""
    import uuid
    workers = _two_worker_group(monkeypatch)

    def per_worker_mint(launch, method, args, *, timeout_s, what):
        return {"results": [w.sisteer_arm_lora_routing_recorder(uuid.uuid4().hex) for w in workers]}
    monkeypatch.setattr(vs, "_post_collective_rpc", per_worker_mint)
    with pytest.raises(vs.VllmServerError, match="worker 1's recorder payload disagrees"):
        vs.arm_lora_routing_recorder(_launch(), "ignored")
    epochs = {w.sisteer_lora_routing_recorder.epoch for w in workers}
    assert len(epochs) == 2 and all(w.sisteer_lora_routing_recorder.armed for w in workers)
    for e in epochs:                                   # no ONE epoch disarms both
        replies = [w.sisteer_disarm_lora_routing_recorder(e) for w in workers]
        assert sorted(r["ok"] for r in replies) == [False, True]
        assert [r["refusal"] for r in replies if r["ok"] is False] == ["epoch_mismatch"]
        for w in workers:                              # restore for the next epoch's attempt
            if not w.sisteer_lora_routing_recorder.armed:
                w.sisteer_arm_lora_routing_recorder(w.sisteer_lora_routing_recorder.epoch)

