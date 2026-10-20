"""The live grader must set an explicit per-request timeout.

The SDK default is 600s. At 500 concurrent that is a trap: on 2026-08-28 a
grade wedged with every worker blocked in a network read, 100 sockets open,
CPU flat at futex_do_wait, no progress for 40+ minutes, and neither a 429 nor
an exception to show for it. The run looked slow, not broken.
"""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def test_make_client_passes_a_timeout():
    import grade_bon_live as g
    src = inspect.getsource(g.make_client)
    assert "timeout=" in src, "make_client does not pass a timeout to the client"
    assert g.REQUEST_TIMEOUT_S <= 300, (
        f"timeout {g.REQUEST_TIMEOUT_S}s is too close to the 600s default to "
        "protect a wide run")
    assert g.REQUEST_TIMEOUT_S >= 30, "timeout too tight for a slow-but-healthy call"


def test_default_signature_carries_it():
    import grade_bon_live as g
    sig = inspect.signature(g.make_client)
    assert "timeout" in sig.parameters
    assert sig.parameters["timeout"].default == g.REQUEST_TIMEOUT_S
