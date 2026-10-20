"""The benign control set must not silently diverge from the config.

The config (configs/two_si.yaml ``controls``) declares FOUR controls (the
AlpacaEval and WMDP slots are left unfilled). The benign
runner (scripts/cell_controls_run.py) WIRES three of them; the fourth (gsm8k) is
the served standard's. Before 2026-08-24 the sweep's BENIGN_CONTROLS silently
dropped one —
it had dropped xstest, a DECLARED, WIRED over-refusal control, halving the
over-refusal measurement to a single instrument with nothing failing loud.

These tests are the durable guard: they pin BENIGN_CONTROLS
to the config AND to what the runner actually wires, so a future silent drop (or
a config control added without wiring it into the sweep) fails here instead of
in a quietly-narrowed result. RED paths included."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from sisteer.eval import sweep  # noqa: E402
import cell_controls_run as ccr  # noqa: E402


def _real_config() -> dict:
    return yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())


# --------------------------------------------------------------------------- #
# GREEN — the real config, the real runner, the real constant agree            #
# --------------------------------------------------------------------------- #

def test_real_config_passes_the_guard():
    """The shipped config validates against BENIGN_CONTROLS with no divergence."""
    sweep.assert_benign_controls_match_config(_real_config())


def test_benign_controls_equal_the_wired_controls():
    """The sweep runs EXACTLY the controls the benign runner wires — no more
    (would crash on an unwired control), no fewer (the xstest-drop bug)."""
    assert set(sweep.BENIGN_CONTROLS) == set(ccr.WIRED_CONTROLS), (
        f"BENIGN_CONTROLS={sorted(sweep.BENIGN_CONTROLS)} != "
        f"WIRED_CONTROLS={sorted(ccr.WIRED_CONTROLS)}")


def test_xstest_is_restored():
    """The exact control that was silently dropped is present again."""
    assert "xstest" in sweep.BENIGN_CONTROLS


def test_nonbenign_is_exactly_the_config_controls_not_wired():
    """The documented-excluded set is precisely the config controls the benign
    runner does not wire (today: {gsm8k} — the served STANDARD's since PW3,
    the bespoke HF probe being retired). If the config gains/loses a control,
    this forces a deliberate update."""
    declared = set(_real_config()["controls"])
    not_wired = declared - set(ccr.WIRED_CONTROLS)
    assert set(sweep.BENIGN_CONTROLS_NONBENIGN) == not_wired
    assert set(sweep.BENIGN_CONTROLS_NONBENIGN) == {"gsm8k"}
    assert not ({"wmdp", "alpaca_eval"} & declared)          # retired 2026-09-11


def test_gsm8k_benign_member_refuses_with_the_pointer():
    """PW3: a sweep condition naming gsm8k in its benign leg is refused with
    the pointer at the served standard, not a bare 'unknown member'."""
    with pytest.raises(ValueError) as ei:
        sweep.Condition(label="c", cell="none", static=("bon",), benign=("gsm8k",))
    msg = str(ei.value)
    assert "RETIRED" in msg and "inspect/gsm8k" in msg and "--controls gsm8k" in msg
    assert "gsm8k" not in sweep.BENIGN_CONTROLS


def test_every_nonbenign_control_is_actually_refused_by_the_runner():
    """A control we claim is 'ported elsewhere' must in fact be refused by the
    benign runner — otherwise 'excluded' is a fiction and it should just run."""
    for ctl in sweep.BENIGN_CONTROLS_NONBENIGN:
        with pytest.raises(SystemExit):
            ccr.parse_controls(ctl)


# --------------------------------------------------------------------------- #
# RED — each way the config and code can diverge must fail loud                 #
# --------------------------------------------------------------------------- #

def test_red_declared_control_neither_run_nor_excluded():
    """A config that declares a control the benign leg neither runs nor
    documents as excluded is the xstest-drop bug — it must raise."""
    cfg = {"controls": list(sweep.BENIGN_CONTROLS)
           + list(sweep.BENIGN_CONTROLS_NONBENIGN) + ["phtest"]}
    with pytest.raises(ValueError, match="silently not honoured"):
        sweep.assert_benign_controls_match_config(cfg)


def test_red_stale_nonbenign_exclusion():
    """If the config no longer declares an 'excluded' control, the stale
    exclusion must fail loud rather than mask a shrunk config."""
    cfg = {"controls": list(sweep.BENIGN_CONTROLS)}  # drops gsm8k
    with pytest.raises(ValueError, match="stale exclusion"):
        sweep.assert_benign_controls_match_config(cfg)


def test_red_benign_control_not_declared(monkeypatch):
    """A benign control the config does not name (typo / removed upstream) must
    fail loud — the sweep cannot run a control the config does not declare."""
    monkeypatch.setattr(
        sweep, "BENIGN_CONTROLS",
        tuple(sweep.BENIGN_CONTROLS) + ("not_a_real_control",))
    with pytest.raises(ValueError, match="does not declare"):
        sweep.assert_benign_controls_match_config(_real_config())


def test_red_empty_controls():
    """A config with no controls is refused, not silently run as empty."""
    with pytest.raises(ValueError, match="no `controls`"):
        sweep.assert_benign_controls_match_config({"controls": []})
