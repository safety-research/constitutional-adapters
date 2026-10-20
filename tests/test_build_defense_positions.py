"""Change 1 — positions is MANDATORY-EXPLICIT on every defense-construction path.

The footgun: ``_set_response_positions`` now ACTIVELY applies "all" (resetting a
persisted response-fit object) instead of early-returning, so an OMITTED
``positions`` no longer no-ops — it silently re-serves a matched decode-only
(Class C) fit as the reverse cross-phase cell. ``build_defense`` (the artifact
path) has NO default: a path that forgets positions fails loudly (TypeError
naming the footgun). ``build_defense_cell`` (the cell-keyed path) instead
DEFAULTS to positions="fit" since the phase-C join — serve auto-matches the
artifact's recorded fit leg, and an explicit mismatch raises — so an omitted
positions there serves the artifact as what it is, never a mask reset. The
sweep's construction path still passes the correct leg explicitly for a
matched Class C condition.
"""

from __future__ import annotations

import pytest

from sisteer.eval import jailbreak as jb


def test_build_defense_requires_explicit_positions():
    """A path that FORGETS positions fails loudly (the guard fires before any
    model use, so model/tok can be None)."""
    with pytest.raises(TypeError) as ei:
        jb.build_defense(None, None, mode="none")
    msg = str(ei.value)
    assert "positions" in msg
    assert "resets" in msg.lower() or "reset" in msg.lower()


def test_build_defense_cell_defaults_to_the_fit_leg():
    # Phase-C join: build_defense_cell defaults positions="fit" (a baseline
    # has no artifact and no fit leg, so "fit" resolves to "all") — the
    # mandatory-explicit contract now lives on build_defense alone, and the
    # fit==serve mismatch guard on compiled cells (see test_defense_cell).
    class _Cell:
        id = "none"
        arm = None
        si_id = None
    d = jb.build_defense_cell(None, None, _Cell())
    assert d.mode == "none"
    assert d.positions_served == "all"


def test_build_defense_none_all_position_ok():
    """The undefended baseline builds with positions explicitly stated."""
    d = jb.build_defense(None, None, mode="none", positions="all")
    assert d.mode == "none"


def test_build_defense_none_rejects_response():
    """A baseline has no maskable intervention — 'response' refuses loudly (not
    silently coerced to all)."""
    with pytest.raises(ValueError):
        jb.build_defense(None, None, mode="none", positions="response")


def test_matched_class_c_condition_passes_response_through_the_sweep():
    """A matched Class C condition (a decode-only fit served decode-only,
    mask='generated') passes positions='response' through the sweep's construction
    path — never defaulting to 'all', which would silently reset the mask. Verified
    by capturing the positions arg build_defense_cell receives."""
    import importlib.util
    from pathlib import Path
    from sisteer.eval import sweep
    drv_path = Path(__file__).resolve().parents[1] / "scripts" / "unified_sweep_run.py"
    spec = importlib.util.spec_from_file_location("unified_sweep_run", drv_path)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)

    captured = {}

    class _MaskedObj:
        # the live serve-mask state the driver's mechanism assertion requires
        positions = "response"
        response_header_len = 4

    class _FakeDefense:
        obj = _MaskedObj()
        scale = 1.0

    def fake_build_defense_cell(model, tok, cell, *, scale, positions,
                               batch_size=None, generate_fn=None):
        captured["positions"] = positions
        return _FakeDefense()

    # a Class C cell id (a real compiled arm that supports a decode-only fit).
    cond = sweep.Condition(label="cc", cell="safety_refusal__psr_kl_all",
                           mask="generated", static=("bon",), adaptive=(),
                           benign=(), held_out=())
    # verify the decode-phase gate default-denies (no provenance), so we stub it
    # out here — this test is only about the positions arg being threaded, not the
    # probe.
    import sisteer.eval.jailbreak as jbmod
    orig = jbmod.build_defense_cell
    jbmod.build_defense_cell = fake_build_defense_cell
    drv_verify = drv._verify_decode_phase
    drv._verify_decode_phase = lambda *a, **k: "matched"
    # by_id / registry lookups aren't needed because cell not in DEFENSES routes to
    # build_defense_cell; stub the registry resolution too.
    import sisteer.eval.defense_cell as dc
    orig_by_id, orig_reg = dc.by_id, dc.registry_from_cfg

    class _C:
        id = "safety_refusal__psr_kl_all"
        arm = "psr_kl_all"
    dc.registry_from_cfg = lambda cfg, repo_root=None: {}
    dc.by_id = lambda reg: {"safety_refusal__psr_kl_all": _C()}
    try:
        cfg = {"scale": 1.0}
        drv._build_condition_defense(model=object(), tok=object(), cond=cond,
                                     cfg=cfg)
    finally:
        jbmod.build_defense_cell = orig
        drv._verify_decode_phase = drv_verify
        dc.by_id, dc.registry_from_cfg = orig_by_id, orig_reg

    assert captured["positions"] == "response"
