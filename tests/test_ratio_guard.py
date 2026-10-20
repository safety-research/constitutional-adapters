"""The residual-ratio hard-stop: a distillation
object whose probed KL ratio vs the unsteered student exceeds 1.5 on two
CONSECUTIVE probe evals is worse than no-op — terminate, don't burn the
budget. Mirrors tests/test_divergence_guard.py's structure: guard
semantics first, then the wiring seams."""
import pytest

from sisteer.extraction.fitmon import (FitMonitor, ProbeCadence,
                                       RatioDivergenceGuard,
                                       TrainingDiverged)


def test_healthy_run_never_triggers():
    g = RatioDivergenceGuard()
    for _ in range(50):
        g.update(0.4)   # healthy: well under 1
    assert g.strikes == 0


def test_single_spike_does_not_trigger():
    g = RatioDivergenceGuard()
    g.update(0.3)          # skip_probes swallows the first eval
    g.update(0.5)
    g.update(9.0)          # one transient spike
    g.update(0.6)          # recovers -> strikes reset
    for _ in range(10):
        g.update(0.7)
    assert g.strikes == 0


def test_two_consecutive_breaches_terminate_with_the_prefix():
    g = RatioDivergenceGuard()
    g.update(0.5)          # warmup skip
    g.update(1.6)          # strike 1
    with pytest.raises(TrainingDiverged) as ei:
        g.update(2.0)      # strike 2 -> verdict
    assert str(ei.value).startswith(RatioDivergenceGuard.PREFIX)
    assert "2.000" in str(ei.value)


def test_first_probe_is_warmup_skipped():
    g = RatioDivergenceGuard()
    g.update(50.0)         # first eval NEVER counts (early-ramp transient)
    assert g.strikes == 0
    g.update(1.6)
    with pytest.raises(TrainingDiverged):
        g.update(1.6)


def test_boundary_is_strict():
    g = RatioDivergenceGuard()
    g.update(0.5)
    for _ in range(10):
        g.update(1.5)      # AT the threshold: not a breach
    assert g.strikes == 0


def test_nonfinite_is_ignored_here():
    g = RatioDivergenceGuard()
    g.update(0.5)
    g.update(float("nan"))
    g.update(float("inf"))
    assert g.strikes == 0  # the callers' non-finite aborts own that case


class _StubProbe:
    """A KLRatioProbe stand-in returning a scripted ratio sequence."""

    def __init__(self, ratios):
        self._ratios = list(ratios)
        self.denom = 1.0

    def measure(self, model, obj=None):
        r = self._ratios.pop(0)
        return {"probe/residual_kl": r, "probe/residual_ratio": r,
                "probe/noop_floor": 1.0}


def test_fit_monitor_probe_seam_terminates():
    mon = FitMonitor("psr_kl_all", model=None,
                     probe=_StubProbe([0.5, 1.7, 1.8]),
                     cadence=ProbeCadence(), emit=lambda *a, **k: None)
    assert isinstance(mon.ratio_guard, RatioDivergenceGuard)  # default ON
    mon._probe(obj=None)   # warmup skip
    mon._probe(obj=None)   # strike 1
    with pytest.raises(TrainingDiverged) as ei:
        mon._probe(obj=None)
    assert str(ei.value).startswith(RatioDivergenceGuard.PREFIX)


def test_fit_monitor_guard_can_be_disabled():
    mon = FitMonitor("psr_kl_all", model=None,
                     probe=_StubProbe([9.0] * 6),
                     cadence=ProbeCadence(), emit=lambda *a, **k: None,
                     ratio_guard=False)
    for _ in range(6):
        mon._probe(obj=None)   # never raises


def test_trial_status_classification():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import tune_arm

    assert "ratio_diverged" in tune_arm._TERMINAL_STATUSES
    e = TrainingDiverged(f"{RatioDivergenceGuard.PREFIX}: probe/"
                         "residual_ratio=2.000 > 1.5 on 2 consecutive ...")
    status = ("ratio_diverged"
              if str(e).startswith(RatioDivergenceGuard.PREFIX)
              else "nan_diverged")
    assert status == "ratio_diverged"
