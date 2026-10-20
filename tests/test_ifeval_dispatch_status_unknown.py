"""ifeval-canonical-battery: a failure to OBTAIN a call's status is not a status — and TERMINAL REQUIRES
REMOTE EVIDENCE, EVERYTHING ELSE IS UNKNOWN.

``FunctionCall.get(timeout=0)`` raises client-side errors while the remote function is still running: a transport error (a
``ConnectionError`` after Modal's own retries, a grpclib stream error, a closed client) or a REJECTED poll RPC, which Modal
1.5.4's ``grpc_error_converter`` translates into a ``modal.exception`` class — ``ResourceExhaustedError`` for a server-side
rate limit, ``PermissionDeniedError``, ``AuthError``, ``ServiceError``, ``InternalError``, ``NotFoundError``. ``_poll_active``
used to treat every exception other than the bare timeout as a TERMINAL failure: the entry moved to ``failed``, left
``active`` / ``outstanding()``, ``_fill_slots`` refilled the slot (8 unfinished calls, one failing poll -> 9 calls at Modal
while the dispatcher counted 8), and the forgotten detached call could be cancelled by the clean-completion cleanup or
duplicated by the FAILED re-queue. The first fix listed the transport classes and kept ``terminal`` as the default; the
re-check reproduced the same defect with the installed SDK's real converter — ``ResourceExhaustedError`` /
``PermissionDeniedError`` matched neither the names nor the modules and read as terminal. Now ``poll_outcome`` files
``terminal`` ONLY on affirmative evidence of the remote function's own result — Modal's remote-traceback mark on a
deserialised remote exception (``__line_cache__`` / ``<ta-…>`` frames), or one of the SDK's own result classes raised from a
decoded ``GenericResult`` (``FunctionTimeoutError`` / ``OutputExpiredError`` / ``InternalFailure`` / ``RemoteError`` from the
``modal`` package) — and ``unknown`` for everything else, known class or not. On an UNKNOWN outcome the entry stays active
with its slot, persists ``status_unknown_since`` / ``_error`` / ``_polls``, backs off by doubling ticks, and clears the marks
on the first status obtained; past ``--status-unknown-max-s`` the wave exits FAILED with nothing cancelled and the reservation
path's remedies (``--release`` / ``--adopt``) apply to the entry. Adoption (``FunctionCall.from_id`` opens the client and
decodes no result) is guarded by the same rule: every exception there, a ``NotFoundError`` included, is a status not obtained.

The exception classes below model Modal 1.5.4's own hierarchy (``modal/exception.py``, ``modal/_grpc_client.py``) by name,
module and bases, so the classifier meets the shapes the installed SDK raises without importing it (the pipeline's test venv
has no Modal); ``test_ifeval_dispatch_status_unknown_sdk.py`` repeats the table on the REAL classes, the real converter and
the real remote-traceback marker wherever the SDK is importable.
"""
from __future__ import annotations

import json
import shutil
from typing import Dict, List

import pytest

from test_ifeval_dispatch import CID, D, FakeClock, FakeRegistry, LAUNCH, MATRIX, ROW_BY_CID, Wave, remote_result


# --------------------------------------------------------------------------- #
# Modal 1.5.4's exception hierarchy, modelled by name / module / bases (nothing here imports Modal)                     #
# --------------------------------------------------------------------------- #
def _modal_cls(name: str, *bases: type) -> type:
    cls = type(name, bases or (Exception,), {})
    cls.__module__ = "modal.exception"
    return cls


class GRPCError(Exception):
    """``grpclib.GRPCError``: what the channel raises for a rejected RPC before Modal translates it."""


class StreamTerminatedError(Exception):
    """``grpclib.exceptions.StreamTerminatedError``: the stream died under the poll."""


GRPCError.__module__ = StreamTerminatedError.__module__ = "grpclib.exceptions"
Error = _modal_cls("Error")                                             # modal.exception.Error(Exception)
_GRPCErrorWrapper = _modal_cls("_GRPCErrorWrapper", GRPCError)         # the transitional grpclib.GRPCError subclass
#: grpc_error_converter's table (modal/_grpc_client.py): gRPC status -> a modal.exception class(Error, _GRPCErrorWrapper)
ResourceExhaustedError = _modal_cls("ResourceExhaustedError", Error, _GRPCErrorWrapper)   # RESOURCE_EXHAUSTED: a rate limit
PermissionDeniedError = _modal_cls("PermissionDeniedError", Error, _GRPCErrorWrapper)     # PERMISSION_DENIED
NotFoundError = _modal_cls("NotFoundError", Error, _GRPCErrorWrapper)                     # NOT_FOUND
ServiceError = _modal_cls("ServiceError", Error, _GRPCErrorWrapper)                       # UNAVAILABLE / CANCELLED / DEADLINE_EXCEEDED / UNKNOWN
AuthError = _modal_cls("AuthError", Error, _GRPCErrorWrapper)                             # UNAUTHENTICATED
InternalError = _modal_cls("InternalError", Error, _GRPCErrorWrapper)                     # INTERNAL
InvalidError = _modal_cls("InvalidError", Error, _GRPCErrorWrapper)                       # INVALID_ARGUMENT / OUT_OF_RANGE
#: the client's own
ModalConnectionError = _modal_cls("ConnectionError", Error)             # NOT the builtin: not an OSError
ClientClosed = _modal_cls("ClientClosed", Error)
ExecutionError = _modal_cls("ExecutionError", Error)                    # a dozen client-side raise sites; one is _process_result
ModalTimeoutError = _modal_cls("TimeoutError", Error)                   # Modal's own; the bare poll raises the BUILTIN one
#: the SDK's RESULT classes — raised from a decoded GenericResult of the call (modal/_utils/function_utils.py
#: _process_result, modal/_functions.py poll_function): the server's verdict on THIS call
FunctionTimeoutError = _modal_cls("FunctionTimeoutError", ModalTimeoutError)
OutputExpiredError = _modal_cls("OutputExpiredError", ModalTimeoutError)
InternalFailure = _modal_cls("InternalFailure", Error)
RemoteError = _modal_cls("RemoteError", Error)


def _inject(fc, errors: List[BaseException]) -> List[int]:
    """Make the next ``len(errors)`` polls of ``fc`` raise them in order, RAW — the way the client raises what it could not
    turn into a status (then behave as before); returns the list the wrapper appends the dispatcher's tick to on every
    poll (the backoff schedule under test)."""
    orig, queue, polled_at = fc.get, list(errors), []
    holder: Dict[str, object] = {}

    def get(timeout=None):
        polled_at.append(holder["d"].tick if "d" in holder else -1)
        if queue:
            raise queue.pop(0)
        return orig(timeout=timeout)

    fc.get = get
    fc._holder = holder
    return polled_at


def _both_functions_rows() -> List[str]:
    """12 configuration ids in matrix order: 4 default-image (gemma) rows and 4 Nemotron rows fill the eight slots, 4
    llama rows (later in the matrix) stay queued — so BOTH image functions are among the active calls."""
    gemma = [r for r in LAUNCH if r["model_key"] == "gemma4e4b"][:4]
    nemo = [r for r in LAUNCH if r["image"] == "nemotron"][:4]
    llama = [r for r in LAUNCH if r["model_key"] == "llama8b"][:4]
    assert len(gemma) == len(nemo) == len(llama) == 4
    return [r["configuration_ids"]["2048"] for r in gemma + nemo + llama]


def _record(w, cid: str) -> dict:
    return json.loads((w.local_root / ROW_BY_CID[cid]["dispatch_paths"]["2048"]).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# the reproduction, fixed: a status not obtained keeps the call, its slot, and spawns nothing                            #
# --------------------------------------------------------------------------- #
def test_a_status_not_obtained_while_polling_keeps_the_call_its_slot_and_spawns_nothing_across_both_functions(tmp_path):
    seen: Dict[str, object] = {}

    def on_tick(w, st, tick):
        if tick == 1:
            assert len(st["active"]) == 8 and len(w.registry.order) == 8 and len(st["queued"]) == 4
            live = w.registry.unfinished()
            assert {fc.fn_name for fc in live} == {"run_configuration", "run_configuration_nemotron"}
            default = [fc for fc in live if fc.fn_name == "run_configuration"]
            nemo = [fc for fc in live if fc.fn_name == "run_configuration_nemotron"]
            victims = {
                default[0]: ResourceExhaustedError("rate limit exceeded: too many requests"),  # the re-check's P1: a rejected poll RPC
                nemo[0]: PermissionDeniedError("permission denied"),                             # the other translated class it named
                default[1]: ConnectionError("[Errno 104] Connection reset by peer"),             # builtin: an OSError
                nemo[1]: StreamTerminatedError("stream closed by the server"),                   # grpclib
                default[2]: ClientClosed("the client was closed"),                               # Modal's client
                nemo[2]: RuntimeError("raised by the client, no remote traceback"),              # the inverted default: not evidence
                default[3]: ExecutionError("Could not deserialize remote exception due to local error"),  # an undecodable result
            }
            for fc, err in victims.items():
                _inject(fc, [err])
            seen["victims"] = {fc.cfg["configuration_id"]: type(err).__name__ for fc, err in victims.items()}
        elif tick == 2:
            seen["st2"], seen["outstanding2"], seen["spawned2"] = st, w.d.outstanding(), len(w.registry.order)
            seen["registry2"] = w.registry.outstanding
            seen["records2"] = {cid: _record(w, cid) for cid in seen["victims"]}
        elif tick == 3:
            seen["st3"] = st
            w.complete_per_tick = 3

    w = Wave(tmp_path, complete_per_tick=0, on_tick=on_tick, rows=_both_functions_rows())
    rc = w.run()
    assert rc == D.EXIT_OK
    victims: Dict[str, str] = seen["victims"]
    assert len(victims) == 7 and {"ResourceExhaustedError", "PermissionDeniedError", "RuntimeError", "ExecutionError"} <= set(victims.values())
    st2 = seen["st2"]
    # after the failing polls: eight calls at Modal, eight counted, none failed, nothing spawned into a "freed" slot
    assert seen["spawned2"] == 8 and seen["outstanding2"] == 8 and seen["registry2"] == 8
    assert st2["failed"] == {} and len(st2["active"]) == 8 and len(st2["queued"]) == 4
    for cid, name in victims.items():
        ent = st2["active"][cid]
        assert ent["status_unknown_since"] is not None and ent["status_unknown_polls"] == 1
        assert ent["status_unknown_error"].startswith("poll: " + name), ent["status_unknown_error"]
        rec = seen["records2"][cid]
        assert rec["state"] == "active" and rec["status_unknown_since"] == ent["status_unknown_since"]
        assert rec["status_unknown_error"] == ent["status_unknown_error"] and rec["status_unknown_polls"] == 1
        assert any(f"{cid}: status UNKNOWN (poll: {name}" in s and "keeps its slot" in s and "failed poll 1" in s for s in w.log)
    healthy = [cid for cid in st2["active"] if cid not in victims]
    assert len(healthy) == 1 and "status_unknown_since" not in st2["active"][healthy[0]]
    # the next poll obtains a status (still running): the marks clear, the recovery is logged
    st3 = seen["st3"]
    for cid in victims:
        assert cid in st3["active"] and "status_unknown_since" not in st3["active"][cid] and "status_unknown_error" not in st3["active"][cid]
        assert any(f"{cid}: status re-obtained after 1 failed poll(s)" in s for s in w.log)
    # the wave completes with ONE spawn per row and never more than eight calls at Modal
    final = w.state()
    assert len(final["done"]) == 12 and final["failed"] == {} and final["active"] == {} and final["queued"] == []
    assert set(w.registry.spawns_by_cid.values()) == {1} and len(w.registry.order) == 12 and w.registry.peak == 8
    assert not any("-> failed" in s for s in w.log)
    assert all("status_unknown_since" not in e for e in final["done"].values())


def test_a_persistent_unknown_status_backs_off_by_doubling_ticks_while_remote_evidence_still_frees_the_slot(tmp_path):
    seen: Dict[str, object] = {}

    def on_tick(w, st, tick):
        if tick == 1:
            live = w.registry.unfinished()
            victim, crashed, witness, timed_out = live[0], live[1], live[2], live[3]
            seen["victim"], seen["crashed"], seen["timed_out"] = (fc.cfg["configuration_id"] for fc in (victim, crashed, timed_out))
            seen["victim_polls"] = _inject(victim, [ResourceExhaustedError(f"rate limit exceeded (attempt {i})") for i in range(5)])
            victim._holder["d"] = w.d
            seen["witness_polls"] = _inject(witness, [])
            witness._holder["d"] = w.d
            # remote evidence (1): the container's own exception handed back as the call's result — the fixture marks it
            # the way Modal's _process_result does (the remote traceback)
            w.registry.finish(crashed, error=RuntimeError("container crashed"))
            # remote evidence (2): the server's verdict as an SDK result class — no remote traceback, terminal by class +
            # package (the call finished at Modal; the poll raises it raw, as _process_result does)
            w.registry.finish(timed_out)
            _inject(timed_out, [FunctionTimeoutError("Function exceeded maximum execution time")])
        elif tick == 2:
            seen["st2"], seen["spawned2"] = st, len(w.registry.order)
        elif tick == 30:
            w.complete_per_tick = 3

    w = Wave(tmp_path, complete_per_tick=0, on_tick=on_tick, rows=_both_functions_rows())
    rc = w.run()
    assert rc == D.EXIT_FAILED                                    # the two confirmed failures are FAILED records
    victim, crashed, timed_out = seen["victim"], seen["crashed"], seen["timed_out"]
    st2 = seen["st2"]
    # the two confirmed failures freed THEIR slots (refilled: 10 spawned); the unknown one did not
    assert set(st2["failed"]) == {crashed, timed_out} and seen["spawned2"] == 10
    assert "RuntimeError" in st2["failed"][crashed]["error"] and "FunctionTimeoutError" in st2["failed"][timed_out]["error"]
    assert victim in st2["active"] and st2["active"][victim]["status_unknown_polls"] == 1
    assert "ResourceExhaustedError" in st2["active"][victim]["status_unknown_error"]
    # the backoff: polls at ticks 1, 2, 4, 8, 16 fail; the sixth attempt at 24 (8-tick cap) obtains the status
    polls = seen["victim_polls"]
    assert polls[:6] == [1, 2, 4, 8, 16, 24], polls
    assert polls[6:] == list(range(25, polls[-1] + 1)), "once the status is back the call is polled every tick again"
    assert seen["witness_polls"][:10] == list(range(1, 11)), "the other calls are polled every tick throughout"
    assert any(f"{victim}: status re-obtained after 5 failed poll(s)" in s for s in w.log)
    assert sum(1 for s in w.log if f"{victim}: status UNKNOWN" in s) == 5
    assert any("next poll in 8 tick(s)" in s for s in w.log) and any("next poll in 1 tick(s)" in s for s in w.log)
    final = w.state()
    assert set(final["failed"]) == {crashed, timed_out} and victim in final["done"] and "status_unknown_since" not in final["done"][victim]
    assert w.registry.peak == 8 and set(w.registry.spawns_by_cid.values()) == {1}


def test_poll_outcome_terminal_requires_remote_evidence_and_everything_else_is_unknown():
    U, T, R = D.OUTCOME_UNKNOWN, D.OUTCOME_TERMINAL, D.OUTCOME_RUNNING
    # running: the bare timeout — the BUILTIN TimeoutError modal/_functions.py poll_function raises — and Modal's own
    # TimeoutError by name (an Error, not the builtin); its result subclasses below differ by name
    assert D.poll_outcome(TimeoutError("still running")) == R and D.poll_outcome(ModalTimeoutError()) == R
    # unknown: EVERYTHING the client raises without remote evidence — grpc_error_converter's whole table (the two the first
    # fix missed first), the transport, the client's own, a cancelled task, and an unmarked exception of ANY class
    for exc in (ResourceExhaustedError("rate limit exceeded"), PermissionDeniedError("permission denied"),
                ConnectionError("reset"), ConnectionResetError(104, "reset"), BrokenPipeError(), OSError("socket"),
                ModalConnectionError("Could not connect to the Modal server."), ClientClosed(1),
                StreamTerminatedError("closed"), GRPCError("UNAVAILABLE"), AuthError("token"), ServiceError("unavailable"),
                InternalError("internal"), NotFoundError("FunctionCall fc-x not found"), InvalidError("bad request"),
                type("CancelledError", (BaseException,), {})(),
                ExecutionError("Could not deserialize remote exception due to local error"),
                RuntimeError("container crashed?"), ValueError("bad"), SystemExit(1),
                type("RemoteError", (Exception,), {})("a RemoteError from some other package")):
        assert D.poll_outcome(exc) == U, exc
        assert not D.remote_origin(exc) and not D.sdk_result(exc), exc
    # terminal (1): Modal's remote-traceback mark makes ANY class the call's own result — the attribute, or a <ta-…> frame
    marked = ConnectionError("the container could not reach the proxy")
    setattr(marked, D.REMOTE_TRACEBACK_ATTR, {})
    assert D.remote_origin(marked) and D.poll_outcome(marked) == T
    try:
        exec(compile("raise ConnectionError('raised remotely')", "<ta-01ABCDEFGHIJKLMNOPQRSTUVWXYZ>:/root/driver.py", "exec"), {})
    except ConnectionError as e:
        assert D.remote_origin(e) and D.poll_outcome(e) == T
    for exc in (RuntimeError("container crashed"), SystemExit(1), ResourceExhaustedError("raised inside the container")):
        assert D.poll_outcome(remote_result(exc)) == T, exc
    assert not D.remote_origin(ConnectionError("local"))
    # terminal (2): the SDK's result classes, by name AND the modal package
    for exc in (FunctionTimeoutError("timed out remotely"), OutputExpiredError(), InternalFailure("internal failure"), RemoteError("remote")):
        assert D.sdk_result(exc) and D.poll_outcome(exc) == T, exc
    assert D.RESULT_EXCEPTION_NAMES == ("FunctionTimeoutError", "OutputExpiredError", "InternalFailure", "RemoteError")
    assert D.RESULT_EXCEPTION_PACKAGE == "modal"
    # no allow-list of transport classes survives: the default is unknown, not a list
    assert not hasattr(D, "STATUS_UNKNOWN_EXCEPTION_NAMES") and not hasattr(D, "STATUS_UNKNOWN_EXCEPTION_MODULES")
    assert "status_unknown_since" in D.ENTRY_KEYS and "status_unknown_error" in D.ENTRY_KEYS and "status_unknown_polls" in D.ENTRY_KEYS


# --------------------------------------------------------------------------- #
# the bounded budget: fail closed, cancel nothing, the reservation path's remedies                                       #
# --------------------------------------------------------------------------- #
def _budget_wave(tmp_path, *, budget_s: float):
    seen: Dict[str, object] = {}

    def on_tick(w, st, tick):
        if tick == 1:
            victim = w.registry.unfinished()[0]
            seen["victim"] = victim.cfg["configuration_id"]
            # the re-check's P1 end to end: a poll RPC rate-limited on every attempt
            _inject(victim, [ResourceExhaustedError(f"rate limit exceeded (attempt {i})") for i in range(1000)])

    w = Wave(tmp_path, complete_per_tick=0, on_tick=on_tick, rows=_both_functions_rows(), status_unknown_max_s=budget_s,
             stop_app=lambda app_id: pytest.fail(f"cancelled app {app_id} under an unknown status"))
    return w, seen


def test_status_unknown_past_the_budget_fails_closed_without_cancelling_and_the_operator_remedies_settle_it(tmp_path):
    w, seen = _budget_wave(tmp_path / "a", budget_s=25.0)      # the fake clock advances 1 s per call: a few ticks
    rc = w.run()
    assert rc == D.EXIT_FAILED
    victim = seen["victim"]
    st = w.state()
    # nothing cancelled, nothing finished, nothing spawned past the eight; the entry is still ACTIVE with its marks
    assert w.registry.outstanding == 8 and len(w.registry.order) == 8 and all(not fc.finished for fc in w.registry.calls.values())
    assert len(st["active"]) == 8 and len(st["queued"]) == 4 and st["failed"] == {} and st.get("apps_stopped") is None
    ent = st["active"][victim]
    assert ent["status_unknown_since"] is not None and ent["status_unknown_polls"] >= 1
    assert "ResourceExhaustedError" in ent["status_unknown_error"]
    assert ent["fc_id"] in w.registry.calls
    assert any(f"{victim}: status UNKNOWN for" in s and "budget" in s and "NOT re-queued, NOT cancelled" in s
               and f"--release {victim}" in s and f"--adopt {victim}=<fc_id>" in s and "re-run" in s and "same --state" in s for s in w.log)
    assert any("status unknown past the budget" in s and "exits FAILED" in s and "nothing spawned further" in s for s in w.log)
    assert any("status UNKNOWN since" in s and "failed polls" in s for s in w.d.summary_lines())
    # (a) the rate limit lifts (the injected failures removed from the call): a restart adopts the call by id, the first
    # poll clears the marks, the wave completes
    del w.registry.calls[ent["fc_id"]].get
    b = tmp_path / "b"
    shutil.copytree(tmp_path / "a", b)
    w2 = Wave(b, registry=w.registry, adopt=lambda fc_id: w.registry.calls[fc_id], rows=_both_functions_rows())
    assert w2.d.state["active"][victim]["status_unknown_since"] == ent["status_unknown_since"]   # the budget clock rode along
    assert "status_unknown_retry_tick" not in w2.d.state["active"][victim]                        # the old tick did not
    assert w2.run() == D.EXIT_OK
    assert any(f"{victim}: status re-obtained after" in s for s in w2.log)
    assert set(w.registry.spawns_by_cid.values()) == {1} and len(w2.state()["done"]) == 12
    # (b) the operator VERIFIED the call is gone: --release re-queues it on the next start (one deliberate re-spawn)
    logs: List[str] = []
    mpath = tmp_path / "matrix.json"
    mpath.write_text(json.dumps(MATRIX), encoding="utf-8")
    rc = D.main(["--matrix", str(mpath), "--sha", w.d.sha, "--state", str(w.state_path), "--app-name", w.d.app_name, "--release", victim],
                launcher=w.launcher, log=logs.append)
    assert rc == D.EXIT_OK and any("operator --release" in s and "status unknown since" in s for s in logs)
    st_r = json.loads(w.state_path.read_text(encoding="utf-8"))
    assert victim not in st_r["active"] and victim in st_r["released"] and "status_unknown_since" not in st_r["released"][victim]
    assert st_r["reconciled"][victim]["how"].startswith("operator --release") and len(st_r["active"]) == 7
    reg2 = w.registry
    w3 = Wave(tmp_path / "a", registry=reg2, adopt=lambda fc_id: reg2.calls[fc_id], rows=_both_functions_rows())
    assert victim in w3.d.state["queued"]
    assert w3.run() == D.EXIT_OK and reg2.spawns_by_cid[victim] == 2
    # (c) --adopt re-points an unknown-status active entry (marks cleared); (d) an active entry WITH a status refuses
    w4, seen4 = _budget_wave(tmp_path / "c", budget_s=25.0)
    assert w4.run() == D.EXIT_FAILED
    v4 = seen4["victim"]
    other = next(fc_id for fc_id in w4.registry.order if fc_id != w4.state()["active"][v4]["fc_id"])
    assert D.reconcile_reservation(w4.state_path, v4, fc_id="fc-relocated", log=logs.append) == D.EXIT_OK
    st4 = json.loads(w4.state_path.read_text(encoding="utf-8"))
    assert st4["active"][v4]["fc_id"] == "fc-relocated" and "status_unknown_since" not in st4["active"][v4]
    assert st4["reconciled"][v4]["how"].startswith("operator --adopt fc-relocated")
    healthy = next(c for c in st4["active"] if c != v4)
    assert D.reconcile_reservation(w4.state_path, healthy, release=True, log=logs.append) == D.EXIT_FAILED
    assert any("neither a pending reservation" in s and "nor an active call with status unknown" in s for s in logs)
    assert other in w4.registry.calls                                         # (nothing about the registry moved)


def test_a_status_not_obtained_at_adoption_keeps_the_entry_active_and_the_loop_retries_it(tmp_path):
    w1 = Wave(tmp_path, max_ticks=1, rows=_both_functions_rows())
    assert w1.run() == D.EXIT_FAILED                                          # the guard: 8 active, 4 queued
    st1 = w1.state()
    victim = next(iter(st1["active"]))
    victim_fc = st1["active"][victim]["fc_id"]
    attempts: Dict[str, int] = {}

    def adopt(fc_id):
        attempts[fc_id] = attempts.get(fc_id, 0) + 1
        if fc_id == victim_fc and attempts[fc_id] == 1:
            raise ConnectionError("could not connect to the Modal server")
        return w1.registry.calls[fc_id]

    w2 = Wave(tmp_path, registry=w1.registry, adopt=adopt, rows=_both_functions_rows())
    assert w2.run() == D.EXIT_OK
    assert attempts[victim_fc] == 2 and all(n == 1 for k, n in attempts.items() if k != victim_fc)
    assert any(f"{victim}: status UNKNOWN (adopt {victim_fc}: ConnectionError" in s for s in w2.log)
    assert any(f"{victim}: adopted FunctionCall {victim_fc}" in s for s in w2.log)
    assert not any("-> failed" in s for s in w2.log) and set(w1.registry.spawns_by_cid.values()) == {1}
    assert len(w2.state()["done"]) == 12 and w1.registry.peak == 8
    # a call id the server does not know (`NotFoundError`, grpc_error_converter's NOT_FOUND) is a REJECTED RPC, not the
    # call's result: unknown too — the entry stays active with its slot, the adoption is retried under the backoff, nothing
    # is filed failed and nothing re-spawned; a wrong id is the operator's to settle with --release after looking it up
    w3 = Wave(tmp_path / "b", max_ticks=1, rows=_both_functions_rows())
    assert w3.run() == D.EXIT_FAILED
    bad = next(iter(w3.state()["active"]))
    bad_fc = w3.state()["active"][bad]["fc_id"]
    attempts3: Dict[str, int] = {}

    def adopt3(fc_id):
        attempts3[fc_id] = attempts3.get(fc_id, 0) + 1
        if fc_id == bad_fc:
            raise NotFoundError(f"FunctionCall {fc_id} not found")
        return w3.registry.calls[fc_id]

    w4 = Wave(tmp_path / "b", registry=w3.registry, adopt=adopt3, rows=_both_functions_rows(), max_ticks=40)
    assert w4.run() == D.EXIT_FAILED                                          # never settled: the guard (or the budget) exits
    st4 = w4.state()
    assert st4["failed"] == {} and bad in st4["active"] and st4["queued"] == [] and len(st4["done"]) == 11
    ent = st4["active"][bad]
    assert ent["status_unknown_error"].startswith(f"adopt {bad_fc}: NotFoundError") and ent["status_unknown_since"] is not None
    assert attempts3[bad_fc] == ent["status_unknown_polls"] >= 3, "retried under the backoff, never filed"
    assert all(n == 1 for k, n in attempts3.items() if k != bad_fc)
    assert any(f"{bad}: status UNKNOWN (adopt {bad_fc}: NotFoundError" in s for s in w4.log)
    assert not any("-> failed" in s for s in w4.log) and set(w3.registry.spawns_by_cid.values()) == {1} and w3.registry.peak == 8
    assert st4.get("apps_stopped") is None                                    # nothing cancelled: the call is still adoptable
    assert bad_fc in w3.registry.calls and ent["fc_id"] == bad_fc              # the entry still names the call it reserved


def test_source_order_poll_then_budget_then_fill_the_fail_closed_exit_cancels_nothing_and_the_default_is_unknown():
    src = open(D.__file__, encoding="utf-8").read()
    run = src[src.index("    def run(self) -> int:"):src.index("    # -- reporting")]
    assert run.index("self._poll_active()") < run.index("expired = self.status_unknown_expired()") \
        < run.index("return self._fail_closed_status_unknown(expired)") < run.index("self._fill_slots()")
    i_fail = src.index("    def _fail_closed_status_unknown(")
    fail = src[i_fail:src.index("\n    def ", i_fail + 10)]
    body = fail.split('"""', 2)[2]                                       # the code after the docstring
    assert "cleanup_apps" not in body and "stop_app" not in body and "_transition" not in body
    poll = src[src.index("    def _poll_active("):src.index("    def _fill_slots(")]
    unknown = poll[poll.index("elif outcome == OUTCOME_UNKNOWN:"):poll.index("else:", poll.index("elif outcome == OUTCOME_UNKNOWN:"))]
    assert "_status_unknown(" in unknown and "_transition" not in unknown
    # the classifier's rule, structurally: terminal has ONE return, behind the two forms of remote evidence; the LAST
    # statement is the unknown default — no allow-list of transport classes anywhere in the body
    i_po = src.index("def poll_outcome(")
    po_body = src[i_po:src.index("\ndef ", i_po + 10)].split('"""', 2)[2]
    assert po_body.rstrip().endswith("return OUTCOME_UNKNOWN")
    assert po_body.count("return OUTCOME_TERMINAL") == 1 and "if remote_origin(exc) or sdk_result(exc):" in po_body
    assert "isinstance(exc, OSError)" not in po_body and "__module__" not in po_body
    # adoption files a failure only on the same evidence
    adopt = src[src.index("    def _adopt_one("):src.index("    def _adopt_active(")]
    assert "if poll_outcome(e) == OUTCOME_TERMINAL:" in adopt and "OUTCOME_UNKNOWN" not in adopt
    assert "--status-unknown-max-s" in src and D.DEFAULT_STATUS_UNKNOWN_MAX_S == 1800.0 and D.STATUS_UNKNOWN_MAX_BACKOFF_TICKS == 8
    assert str(CID) and FakeClock and FakeRegistry                             # (fixtures untouched)
