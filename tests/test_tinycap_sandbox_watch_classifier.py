"""The FunctionCall classifier of the bcb-bfcl-200 sandbox watcher
(``scripts/tinycap_sandbox_watch.py``: ``classify_functioncall``) against the
REAL Modal SDK classes.

``modal`` is in no repo venv, so this file ``importorskip``s it — a SKIP, never a
false pass, in the Modal CPU suite — and is run locally in an ephemeral pinned
env::

    PYTHONPATH=$PWD/src nice -n 10 uv run --no-project --python 3.12 --with pytest \\
        --with modal==1.5.4 --exclude-newer 2026-09-02T00:00:00Z \\
        python -m pytest -q -p no:cacheprovider tests/test_tinycap_sandbox_watch_classifier.py

Covered: the classifier's exact order with real instances (observation failures
FIRST — ``ConnectionError`` / ``AuthError`` / ``ServiceError`` / ``InternalError``
/ ``NotFoundError`` / grpclib / ``OSError`` → UNAVAILABLE, with the not-ready
builtin ``TimeoutError`` (SDK 1.5.4's "no output yet", an ``OSError`` subclass)
→ PENDING ahead of them, pinned to the installed SDK's source;
``FunctionTimeoutError`` / ``OutputExpiredError`` → TERMINAL before
``modal.exception.TimeoutError`` → PENDING; ``RemoteError`` / ``ExecutionError``
→ TERMINAL; ``InternalFailure`` through the REAL decoder
``modal._utils.function_utils._process_result`` → TERMINAL; a user
``RuntimeError`` → TERMINAL(raised); a returned dict → TERMINAL(returned); a
remotely decoded ``SystemExit`` → TERMINAL; a local ``KeyboardInterrupt``
propagates. Carried over from the retired Terminal-Bench watcher's tests when
that harness left ``main``.
"""
from __future__ import annotations

import asyncio
import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

# these tests need an INSTALLED modal distribution (the pinned ephemeral env). Inside any
# Modal container the runtime's own client at /pkg makes `import modal` succeed WITHOUT its
# dependencies (grpclib, modal_proto), so a bare importorskip("modal") is a false positive there —
# the Modal CPU suite hit exactly that (collection error on grpclib). Guard on the dist-info first.
import importlib.metadata as _md  # noqa: E402

try:
    _md.version("modal")
except _md.PackageNotFoundError:
    pytest.skip("the classifier tests need an INSTALLED modal distribution; a Modal container's "
                "/pkg runtime client is not one", allow_module_level=True)
modal = pytest.importorskip("modal", reason="the classifier tests need the REAL Modal exception hierarchy")
GE = pytest.importorskip("grpclib.exceptions", reason="grpclib ships with an installed modal")
ME = pytest.importorskip("modal.exception")
Status = pytest.importorskip("grpclib.const").Status
api_pb2 = pytest.importorskip("modal_proto.api_pb2")

REPO = Path(__file__).resolve().parents[1]
WATCHER = REPO / "scripts/tinycap_sandbox_watch.py"


def _load():
    spec = importlib.util.spec_from_file_location("tinycap_sandbox_watch_classifier_under_test", WATCHER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def W():
    return _load()


class FC:
    """A fake FunctionCall whose ``get`` raises a REAL instance or returns."""

    def __init__(self, raises=None, returns=None):
        self.raises, self.returns, self.calls = raises, returns, 0

    def get(self, timeout=None, *, index=0):
        self.calls += 1
        assert timeout == 0
        if self.raises is not None:
            raise self.raises
        return self.returns


def _internal_failure_via_real_decoder():
    """The exception ``FunctionCall.get`` raises for a RETURNED
    GENERIC_STATUS_INTERNAL_FAILURE result, produced by the real decoder."""
    result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_INTERNAL_FAILURE,
                                   exception="synthetic internal failure")
    from modal._utils.function_utils import _process_result
    try:
        asyncio.run(_process_result(result, api_pb2.DATA_FORMAT_PICKLE, None))
    except BaseException as e:                    # noqa: BLE001 — the decoder's own raise is the object under test
        return e
    raise AssertionError("the real decoder did not raise on an INTERNAL_FAILURE result")


def _function_timeout_via_real_decoder():
    result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_TIMEOUT, exception="timed out")
    from modal._utils.function_utils import _process_result
    try:
        asyncio.run(_process_result(result, api_pb2.DATA_FORMAT_PICKLE, None))
    except BaseException as e:                    # noqa: BLE001
        return e
    raise AssertionError("the real decoder did not raise on a TIMEOUT result")


def _system_exit_via_real_decoder(code=3):
    """What ``FunctionCall.get`` raises for a function that let ``SystemExit``
    escape: the container runtime serializes every ``BaseException`` except
    ``KeyboardInterrupt`` / ``GeneratorExit`` (``handle_input_exception``) into a
    GENERIC_STATUS_FAILURE result, the real decoder re-raises the decoded object
    AS IS, and synchronicity passes a ``BaseException`` through to the sync caller
    unchanged (``wrap_coro_exception`` → ``UserCodeException`` → unwrapped)."""
    from modal._serialization import serialize
    from modal._utils.function_utils import _process_result
    result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, data=serialize(SystemExit(code)),
                                   exception=f"SystemExit: {code}", traceback="Traceback (most recent call last): ...")
    try:
        asyncio.run(_process_result(result, api_pb2.DATA_FORMAT_PICKLE, None))
    except BaseException as e:                    # noqa: BLE001 — the decoder's own raise is the object under test
        return e
    raise AssertionError("the real decoder did not raise on a FAILURE result carrying a SystemExit")


# --------------------------------------------------------------------------- #
# the classifier                                                               #
# --------------------------------------------------------------------------- #

CASES = [
    ("ConnectionError", lambda: ME.ConnectionError("blip"), "OBSERVATION_UNAVAILABLE", None),
    ("AuthError", lambda: ME.AuthError("token"), "OBSERVATION_UNAVAILABLE", None),
    ("ServiceError", lambda: ME.ServiceError("svc"), "OBSERVATION_UNAVAILABLE", None),
    ("InternalError", lambda: ME.InternalError("rpc"), "OBSERVATION_UNAVAILABLE", None),
    ("NotFoundError", lambda: ME.NotFoundError("nf"), "OBSERVATION_UNAVAILABLE", None),
    ("grpclib.GRPCError", lambda: GE.GRPCError(Status.UNAVAILABLE, "unavailable"), "OBSERVATION_UNAVAILABLE", None),
    ("grpclib.StreamTerminatedError", lambda: GE.StreamTerminatedError("stream"), "OBSERVATION_UNAVAILABLE", None),
    ("OSError", lambda: OSError("socket"), "OBSERVATION_UNAVAILABLE", None),
    ("builtins.ConnectionError", lambda: builtins.ConnectionError("reset"), "OBSERVATION_UNAVAILABLE", None),
    # SDK 1.5.4's not-ready signal is the BUILTIN TimeoutError (an OSError subclass; asyncio.TimeoutError is
    # the same class on 3.11+) — PENDING, never an observation failure
    ("builtins.TimeoutError", lambda: builtins.TimeoutError(), "PENDING", None),
    ("asyncio.TimeoutError", lambda: asyncio.TimeoutError(), "PENDING", None),
    ("other modal.Error", lambda: ME.VersionError("ambiguous"), "OBSERVATION_UNAVAILABLE", None),
    ("FunctionTimeoutError", lambda: ME.FunctionTimeoutError("12h"), "CONFIRMED_TERMINAL", "function_timeout"),
    ("FunctionTimeoutError(decoder)", _function_timeout_via_real_decoder, "CONFIRMED_TERMINAL", "function_timeout"),
    ("OutputExpiredError", lambda: ME.OutputExpiredError("gone"), "CONFIRMED_TERMINAL", "output_expired"),
    ("modal.TimeoutError", lambda: ME.TimeoutError("no result yet"), "PENDING", None),
    ("RemoteError", lambda: ME.RemoteError("server"), "CONFIRMED_TERMINAL", "failed"),
    ("ExecutionError", lambda: ME.ExecutionError("exec"), "CONFIRMED_TERMINAL", "failed"),
    ("InternalFailure(decoder)", _internal_failure_via_real_decoder, "CONFIRMED_TERMINAL", "infra_failure"),
    ("InternalFailure", lambda: ME.InternalFailure("direct"), "CONFIRMED_TERMINAL", "infra_failure"),
    ("RuntimeError(user)", lambda: RuntimeError("user"), "CONFIRMED_TERMINAL", "raised"),
    # BaseExceptions OUT of fc.get: a remotely decoded SystemExit is the function's own raise (terminal);
    # a local asyncio cancellation is the loop machinery failing (unavailable); KeyboardInterrupt propagates
    ("SystemExit(decoder)", _system_exit_via_real_decoder, "CONFIRMED_TERMINAL", "raised"),
    ("SystemExit", lambda: SystemExit(2), "CONFIRMED_TERMINAL", "raised"),
    ("asyncio.CancelledError", lambda: asyncio.CancelledError(), "OBSERVATION_UNAVAILABLE", None),
]


@pytest.mark.parametrize("name,make,state,kind", CASES, ids=[c[0] for c in CASES])
def test_classifier_with_real_instances(W, name, make, state, kind):
    exc = make()
    fc = FC(raises=exc)
    ob = W.classify_functioncall(fc)
    assert fc.calls == 1
    assert ob.state == state, ob.describe()
    assert ob.kind == kind
    if state == "OBSERVATION_UNAVAILABLE":
        assert ob.reason == type(exc).__name__ and not ob.terminal
    if state == "CONFIRMED_TERMINAL":
        assert ob.error is exc and ob.terminal
    if state == "PENDING":
        assert not ob.terminal


def test_returned_value_is_terminal_and_the_mro_facts_the_order_relies_on_hold(W):
    ob = W.classify_functioncall(FC(returns={"status": "OK"}))
    assert ob.state == "CONFIRMED_TERMINAL" and ob.kind == "returned" and ob.rec == {"status": "OK"}
    # the facts the ordering rests on, re-verified against the installed SDK
    assert issubclass(ME.FunctionTimeoutError, ME.TimeoutError) and issubclass(ME.OutputExpiredError, ME.TimeoutError)
    assert not issubclass(ME.TimeoutError, builtins.TimeoutError)
    for cls in (ME.ConnectionError, ME.AuthError, ME.ServiceError, ME.InternalError, ME.NotFoundError,
                ME.RemoteError, ME.ExecutionError, ME.InternalFailure, ME.TimeoutError):
        assert issubclass(cls, ME.Error), cls
    assert issubclass(ME.AuthError, GE.GRPCError) and not issubclass(ME.InternalFailure, ME.InternalError)
    # the fact the not-ready branch's POSITION rests on: the builtin TimeoutError IS an OSError, so PENDING
    # must be tested before the OSError observation branch — and no observation class is a builtin TimeoutError
    assert issubclass(builtins.TimeoutError, OSError) and asyncio.TimeoutError is builtins.TimeoutError
    for cls in (ME.ConnectionError, ME.AuthError, ME.ServiceError, ME.InternalError, ME.TimeoutError,
                GE.GRPCError, GE.StreamTerminatedError, GE.ProtocolError, builtins.ConnectionError):
        assert not issubclass(cls, builtins.TimeoutError), cls


def test_sdk_1_5_4_signals_not_ready_with_the_builtin_timeout_error(W):
    """The facts the PENDING branch rests on, read from the INSTALLED SDK: the
    not-ready raise in ``_Invocation.poll_function`` (what ``FunctionCall.get``
    delegates to) is a bare ``raise TimeoutError()`` in a module that binds no
    ``TimeoutError`` of its own — so the name is the builtin — and modal's own
    synchronizer (the blocking bridge ``FunctionCall.get`` crosses) re-raises it
    unchanged. The only other timeout on this path, a transport deadline, is
    re-raised by modal's RPC retry layer as ``modal.exception.ConnectionError``
    (grpclib's client-side ``asyncio.TimeoutError``) or left as the
    ``GRPCError`` (a server ``DEADLINE_EXCEEDED``): never a builtin
    ``TimeoutError``, so the builtin is unambiguously "no output yet"."""
    import inspect
    import grpclib.utils
    import modal._functions as MF
    from modal._utils import grpc_utils
    from modal._utils.async_utils import synchronizer
    assert "TimeoutError" not in vars(MF) and "OutputExpiredError" in vars(MF)
    assert "raise TimeoutError()" in inspect.getsource(MF._Invocation.poll_function)
    assert "poll_function(timeout=timeout" in inspect.getsource(MF._FunctionCall.get)
    assert synchronizer._translate_in(modal.FunctionCall) is MF._FunctionCall

    async def _not_ready():
        raise TimeoutError()

    with pytest.raises(builtins.TimeoutError) as info:
        synchronizer.create_blocking(_not_ready)()
    assert type(info.value) is builtins.TimeoutError and str(info.value) == ""
    assert W.classify_functioncall(FC(raises=info.value)).pending
    # a transport deadline never reaches the caller as a builtin TimeoutError
    assert "raise asyncio.TimeoutError('Deadline exceeded')" in inspect.getsource(grpclib.utils.DeadlineWrapper.start)
    assert grpc_utils.ConnectionError is ME.ConnectionError
    src = inspect.getsource(grpc_utils.process_exception_before_retry)
    assert "isinstance(exc, asyncio.TimeoutError)" in src and "raise ConnectionError(str(exc))" in src
    assert Status.DEADLINE_EXCEEDED in grpc_utils.RETRYABLE_GRPC_STATUS_CODES


def test_internal_failure_is_terminal_while_rpc_failures_are_not(W):
    infra = W.classify_functioncall(FC(raises=_internal_failure_via_real_decoder()))
    assert infra.state == "CONFIRMED_TERMINAL" and infra.kind == "infra_failure"
    assert isinstance(infra.error, ME.InternalFailure) and infra.terminal
    for exc in (ME.InternalError("rpc"), ME.ConnectionError("blip"), ME.AuthError("token")):
        ob = W.classify_functioncall(FC(raises=exc))
        assert ob.state == "OBSERVATION_UNAVAILABLE" and not ob.terminal


def test_a_local_keyboard_interrupt_propagates_out_of_the_classifier(W):
    """A Ctrl-C in the watcher process during ``fc.get`` is LOCAL by construction
    — the container runtime re-raises a ``KeyboardInterrupt`` instead of
    serializing it — and must still stop the watcher: never terminal (a false
    sweep), never unavailable (a swallowed interrupt)."""
    fc = FC(raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        W.classify_functioncall(fc)
    assert fc.calls == 1
    # the container-side fact the rule rests on, read from the installed SDK
    import inspect
    from modal._runtime import container_io_manager as cim
    src = inspect.getsource(cim)
    assert "except (KeyboardInterrupt, GeneratorExit):" in src and "except BaseException as exc:" in src
