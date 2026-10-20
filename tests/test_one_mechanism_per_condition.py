"""ONE decode mechanism per condition.

The bug this pins: a mask='generated' condition's JB/held-out legs served through
the POSITION MASK (token 1 steered) while its benign leg served through the
capeval ENGINE-PHASE GATE (token 1 unsteered — the retired mechanism), and the
stamp said engine_phase_gate for all of them. Capability and robustness would
present as measured on the same condition when they came from DIFFERENT
interventions. Now: the benign leg REFUSES for cross-phase conditions (the engine
cannot express the position mask — verified, see the trap test in
test_phase_probe_gpu.py), the stamp says position_mask (the only mechanism a
valid condition can mean), and the driver asserts the BUILT object's live state
backs the stamp before generating.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sisteer.eval import sweep


def _driver():
    drv_path = Path(__file__).resolve().parents[1] / "scripts" / "unified_sweep_run.py"
    spec = importlib.util.spec_from_file_location("unified_sweep_run", drv_path)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    return drv


def test_generated_mask_with_benign_members_refuses_at_construction():
    with pytest.raises(ValueError) as ei:
        sweep.Condition(label="c", cell="safety_refusal__vector",
                        mask="generated", benign=("mmlu",), adaptive=())
    msg = str(ei.value)
    assert "engine-phase gate" in msg and "DIFFERENT intervention" in msg


def test_generated_mask_without_benign_is_fine():
    c = sweep.Condition(label="c", cell="safety_refusal__vector",
                        mask="generated", benign=(), adaptive=())
    assert c.benign == ()


def test_load_conditions_refuses_generated_with_default_benign(tmp_path):
    """A TSV row with mask=generated and the DEFAULT benign set (missing column)
    refuses at parse time — the author must state benign='-' deliberately."""
    p = tmp_path / "conds.tsv"
    p.write_text("dec\tsafety_refusal__vector\t\tgenerated\n")
    with pytest.raises(ValueError):
        sweep.load_conditions(p)
    p.write_text("dec\tsafety_refusal__vector\t\tgenerated\t*\t-\t-\thf\tvllm\thf\t\t*\n")
    (cond,) = sweep.load_conditions(p)
    assert cond.mask == "generated" and cond.benign == ()


def test_all_legs_of_a_generated_condition_stamp_position_mask():
    """The stamp is position_mask on EVERY leg of a decode-only condition — one
    mechanism end to end (the benign leg cannot exist on such a condition)."""
    c = sweep.Condition(label="c", cell="safety_refusal__vector",
                        mask="generated", benign=(), adaptive=())
    mechs = {c.static_stamp()["decode_mechanism"],
             c.held_out_stamp()["decode_mechanism"]}
    assert mechs == {"position_mask"}
    # all-position rows carry no mechanism.
    c_all = sweep.Condition(label="a", cell="safety_refusal__vector", adaptive=())
    assert c_all.static_stamp()["decode_mechanism"] is None


def test_position_mask_and_engine_gate_rows_refuse_to_pair():
    """The mechanism stays a pairing axis: a position_mask row must never be
    contrasted against an engine_phase_gate row."""
    a = sweep.engine_stamp(target_engine="hf", served_template_hash="t",
                           decode_mechanism="position_mask")
    b = sweep.engine_stamp(target_engine="hf", served_template_hash="t",
                           decode_mechanism="engine_phase_gate")
    with pytest.raises(Exception):
        sweep.assert_pairable(a, b, what="mechanism mismatch")


def test_driver_asserts_live_object_backs_the_stamp():
    """_assert_position_mask_live: passes when the built object's LIVE state
    expresses the response mask through either real seam; raises when it does not
    (a declared stamp with no measured backing), and for a baseline."""
    drv = _driver()
    cond = sweep.Condition(label="c", cell="safety_refusal__vector",
                           mask="generated", benign=(), adaptive=())

    class _Defense:
        def __init__(self, obj):
            self.obj = obj

    class _Deploy:      # the vector deployment seam
        positions = "response"
        response_header_len = 4

    class _Fit:         # the PositionMaskedFit seam
        fit_positions = "response"
        fit_response_header_len = 4

    class _Unmasked:    # hydrated all-position: the stamp would be a lie
        positions = "all"
        response_header_len = None

    assert drv._assert_position_mask_live(_Defense(_Deploy()), cond) == "position_mask"
    assert drv._assert_position_mask_live(_Defense(_Fit()), cond) == "position_mask"
    with pytest.raises(ValueError, match="live state expresses no"):
        drv._assert_position_mask_live(_Defense(_Unmasked()), cond)
    with pytest.raises(ValueError, match="no intervention object"):
        drv._assert_position_mask_live(_Defense(None), cond)


def test_conditions_focused_tsv_loads_and_respects_the_cuts():
    """The LAUNCH config must actually load (audit 2026-08-21: it did not — row
    12's mask='generated' + benign='*' tripped the one-mechanism guard and, since
    load_conditions constructs rows in file order, the WHOLE file failed, not
    just that row) and must respect both standing cuts: (a) NO adaptive members
    anywhere — every row carried an explicit adaptive='*', which OVERRIDES the
    loader's adaptive-off default and would have revived the ~1e4-GPU-hour
    attacker-in-the-loop leg that was cut; (b) no decode-only condition carries
    benign members (the guard that refused row 12 is correct and stays). Nothing
    exercised this file before a launch attempt — exactly the class of bug a
    test must catch first."""
    from pathlib import Path
    tsv = Path(__file__).resolve().parents[1] / "configs" / "swp" / \
        "conditions_focused.tsv"
    conds = sweep.load_conditions(tsv)          # (a) parses without raising
    assert len(conds) == 16
    assert all(c.adaptive == () for c in conds), (
        "adaptive members present — an explicit '*' overrides the scope cut")
    assert all(not (c.mask == "generated" and c.benign) for c in conds), (
        "a decode-only condition carries benign members — one mechanism per "
        "condition")
