"""ifeval-canonical-battery (P2): IFEval is a LEDGER-REQUIRED control at the producer.

Before this ``contracts.LEDGER_REQUIRED_CONTROLS == ("math500",)`` while the IFEval request-evidence reconciliation
at the producer refuses a run whose usage ledger holds no rows — so an identity-version-2
IFEval driver invocation WITHOUT ``--ledger-url`` generated every draw (541 x 2 at pass 1) and only then refused the
empty ledger. Now the IFEval atom is in the tuple and the driver refuses the omission BEFORE any request, any marker
and any store directory, naming the flag and the version — exactly the MATH-500 behaviour.
"""
from __future__ import annotations

import pytest

from sisteer.capeval.tinycap import contracts as tc

import served_capability_run as srun
from test_served_capability import _never_runs, _percell_argv, stub_framework_versions, world  # noqa: F401


def test_ifeval_is_ledger_required_and_the_omission_refuses_before_any_request_marker_or_store(world, monkeypatch):
    assert tc.CONTROL_IFEVAL in tc.LEDGER_REQUIRED_CONTROLS and tc.CONTROL_MATH500 in tc.LEDGER_REQUIRED_CONTROLS
    stub_framework_versions(monkeypatch)
    _never_runs(monkeypatch)                                    # generation is never reached
    out_root = world.out_root.parent / "ifeval_no_ledger"
    argv = _percell_argv(world, tc.CONTROL_IFEVAL, out_root=out_root)
    argv.remove("--greedy")
    argv += [tc.FLAG_THINKING, tc.THINKING_OFF, tc.FLAG_SAMPLING, tc.SAMPLING_VENDOR_PRESET, tc.FLAG_SAMPLES, "2",
             tc.FLAG_TINYCAP_IDENTITY_VERSION, "2"]
    assert tc.FLAG_LEDGER_URL not in argv
    with pytest.raises(SystemExit) as e:
        srun.main(argv)
    msg = str(e.value)
    assert f"{tc.FLAG_LEDGER_URL} is REQUIRED for ['{tc.CONTROL_IFEVAL}']" in msg
    assert f"{tc.FLAG_TINYCAP_IDENTITY_VERSION} 2" in msg and "before any request" in msg
    # nothing was touched: no store, no marker, no chat request on either fake client
    assert not out_root.exists() and not (out_root / "cells").exists()
    assert world.posted == [] and not getattr(world.chat, "calls", [])
    # the same invocation WITH a ledger URL gets past this door (it then fails later on the world's dead proxy /
    # the fixture's `_never_runs` guard, which is the point: the ledger check is the one refusing here)
    with pytest.raises((SystemExit, AssertionError)) as e2:
        srun.main(argv + [tc.FLAG_LEDGER_URL, "http://127.0.0.1:1/v1"])
    assert f"{tc.FLAG_LEDGER_URL} is REQUIRED" not in str(e2.value)
