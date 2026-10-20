"""The status boundary against the INSTALLED Modal SDK (Z18): the classification table of
``test_ifeval_dispatch_status_unknown.py`` on the SDK's REAL classes, its real ``grpc_error_converter`` (the reproduction of
the re-check's defect: a poll RPC rejected with RESOURCE_EXHAUSTED becomes ``modal.exception.ResourceExhaustedError``, which
the first fix read as terminal) and its real remote-traceback marker (``modal._traceback.extract_traceback`` in the container,
``append_modal_tb`` at the client — the producer of the evidence the dispatcher requires before it files a call as finished).

Skipped where the SDK is not importable (the pipeline's test venv carries no Modal); the production dispatcher venv runs it
as a script — ``<modal venv python> tests/test_ifeval_dispatch_status_unknown_sdk.py`` — with no pytest needed there.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "tinycap_ifeval_dispatch.py"


def _dispatcher():
    mod = sys.modules.get("tinycap_ifeval_dispatch")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("tinycap_ifeval_dispatch", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _sdk():
    try:
        import grpclib
        import grpclib.exceptions
        import modal
        import modal.exception
        from modal._grpc_client import _STATUS_TO_EXCEPTION, grpc_error_converter
        from modal._traceback import append_modal_tb, extract_traceback
    except ImportError as e:  # the shape-modelled table in the sibling module stands in where the SDK is absent
        raise unittest.SkipTest(f"the Modal SDK is not importable here: {e}")
    return modal, grpclib, _STATUS_TO_EXCEPTION, grpc_error_converter, extract_traceback, append_modal_tb


def test_the_installed_sdk_s_rejected_rpcs_read_unknown_and_only_its_result_classes_and_marked_remote_exceptions_terminal():
    D = _dispatcher()
    modal, grpclib, table, grpc_error_converter, extract_traceback, append_modal_tb = _sdk()
    E = modal.exception
    U, T, R = D.OUTCOME_UNKNOWN, D.OUTCOME_TERMINAL, D.OUTCOME_RUNNING
    # every class grpc_error_converter translates a rejected RPC into (its table, modal/_grpc_client.py), produced by the
    # converter itself from the grpclib error the channel raises — none is the call's result: unknown, every one
    translated = {}
    for status, cls in table.items():
        try:
            with grpc_error_converter():
                raise grpclib.GRPCError(status, f"synthetic {status.name}")
        except grpclib.GRPCError as e:      # the transitional wrapper still subclasses grpclib.GRPCError
            translated[status] = e
        assert type(translated[status]) is cls and isinstance(translated[status], E.Error), (status, type(translated[status]))
        assert D.poll_outcome(translated[status]) == U, (status, translated[status])
    assert type(translated[grpclib.Status.RESOURCE_EXHAUSTED]) is E.ResourceExhaustedError     # the re-check's P1
    assert type(translated[grpclib.Status.PERMISSION_DENIED]) is E.PermissionDeniedError
    assert type(translated[grpclib.Status.NOT_FOUND]) is E.NotFoundError
    assert E.ResourceExhaustedError.__module__ == "modal.exception"                          # why the module list never matched
    # the client's own classes and the transport: unknown
    for exc in (E.ConnectionError("Could not connect to the Modal server."), E.ClientClosed(1), E.ExecutionError("undecodable"),
                grpclib.GRPCError(grpclib.Status.UNAVAILABLE, "unavailable"), grpclib.exceptions.StreamTerminatedError(),
                ConnectionError("reset"), ConnectionResetError(104, "reset"), OSError("socket"),
                RuntimeError("raised by the client, no remote traceback"), SystemExit(1)):
        assert D.poll_outcome(exc) == U, exc
    # running: the BUILTIN TimeoutError modal/_functions.py poll_function raises for an unfinished call, and Modal's own by name
    assert D.poll_outcome(TimeoutError()) == R and D.poll_outcome(E.TimeoutError()) == R
    # terminal (2): the SDK's result classes — raised from a decoded GenericResult of the call
    for exc in (E.FunctionTimeoutError("timed out"), E.OutputExpiredError(), E.InternalFailure("internal"), E.RemoteError("remote")):
        assert D.sdk_result(exc) and D.poll_outcome(exc) == T, exc
    # terminal (1): the real marker — the traceback extracted in the "container", appended at the client onto the fresh
    # instance deserialisation yields (no traceback of its own until then)
    def container_side():
        raise RuntimeError("the container's own failure")

    try:
        container_side()
    except RuntimeError as e:
        tb_dict, line_cache = extract_traceback(e, "ta-01SYNTHETICTASKID000000000")
    client_side = RuntimeError("the container's own failure")
    assert D.poll_outcome(client_side) == U and not D.remote_origin(client_side)
    append_modal_tb(client_side, tb_dict, line_cache)
    assert D.remote_origin(client_side) and D.poll_outcome(client_side) == T
    # the mark makes even a rate-limit class the call's own result (the container raised it)
    marked = E.ResourceExhaustedError("raised inside the container")
    append_modal_tb(marked, tb_dict, line_cache)
    assert D.poll_outcome(marked) == T


if __name__ == "__main__":
    test_the_installed_sdk_s_rejected_rpcs_read_unknown_and_only_its_result_classes_and_marked_remote_exceptions_terminal()
    print("ok: the installed SDK's translated exceptions read unknown; its result classes and marked remote exceptions terminal")
