"""The stage-2 adapter check's verdict logic
(scripts/petri_stage2_adapter_check.py) — CPU-only.

The load-bearing case: ``equivalence_bit_exact`` GATES. The script's
claim 1 is that the served intervention IS the fit intervention; an
adapter that steers-but-differently used to report ``status: "pass"``
and exit 0 with the mismatch buried in ``equivalence_max_abs_delta`` —
exactly the silent-wrong-serve failure this check exists to refuse.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "_petri_stage2_adapter_check",
    ROOT / "scripts" / "petri_stage2_adapter_check.py")
s2c = importlib.util.module_from_spec(_spec)
sys.modules["_petri_stage2_adapter_check"] = s2c
_spec.loader.exec_module(s2c)


def _report(**over):
    base = {"equivalence_bit_exact": True, "steering_bites": True,
            "scale0_is_noop": True}
    base.update(over)
    return base


def test_all_three_gating_booleans_pass():
    assert s2c.overall_status(_report()) == "pass"


@pytest.mark.parametrize("broken", [
    "equivalence_bit_exact",  # THE regression: this one never gated
    "steering_bites",
    "scale0_is_noop",
])
def test_each_gating_boolean_fails_the_verdict_alone(broken):
    assert s2c.overall_status(_report(**{broken: False})) == "fail"


def test_a_close_but_not_equal_equivalence_is_a_fail_not_a_diagnostic():
    """'close' is a different finding from 'equal': a tiny max-abs
    deviation rides along as evidence but can never rescue the
    verdict."""
    report = _report(equivalence_bit_exact=False)
    report["equivalence_max_abs_delta"] = 1e-7
    assert s2c.overall_status(report) == "fail"
