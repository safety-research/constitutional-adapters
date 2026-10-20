"""Empirical serving-phase verification: measure which positions an object
actually steers, and the three-way requested/fit/measured agreement check.

The four pinned behaviours:
- a MISLABELLED cell refuses (persisted response object requested all-position),
  naming requested vs measured — the footgun a two-way check misses
- a Class A/B object fit all-position, served decode-only, RUNS labelled
  cross_phase (measured != fit is the axis's defining cross-phase, not a defect)
- a MATCHED cell runs labelled matched
- a NON-PERSISTING Class C artifact (fit response, re-hydrates all) refuses.
"""

from __future__ import annotations

import pytest

from sisteer.eval.phase_probe import (
    ALL,
    CROSS_PHASE,
    MATCHED,
    NONE,
    PROMPT,
    RESPONSE,
    PhaseVerificationError,
    measure_applied_phase,
    phase_of_mask,
    verify_phase,
)


# -- measure_applied_phase: which positions actually changed ------------------

def _hidden(changed_positions, seq=4, d=3):
    base = [[0.0] * d for _ in range(seq)]
    steer = [list(row) for row in base]
    for i in changed_positions:
        steer[i] = [1.0] * d
    return base, steer


def test_measure_all_positions_changed():
    base, steer = _hidden([0, 1, 2, 3])
    assert measure_applied_phase(base, steer, prompt_len=2) == ALL


def test_measure_response_only():
    base, steer = _hidden([2, 3])          # prompt_len=2 -> only generated
    assert measure_applied_phase(base, steer, prompt_len=2) == RESPONSE


def test_measure_prompt_only():
    base, steer = _hidden([0, 1])
    assert measure_applied_phase(base, steer, prompt_len=2) == PROMPT


def test_measure_no_op():
    base, steer = _hidden([])
    assert measure_applied_phase(base, steer, prompt_len=2) == NONE


def test_phase_of_mask_maps_generated_to_response():
    assert phase_of_mask("all") == ALL
    assert phase_of_mask("generated") == RESPONSE


# -- verify_phase: the three-way agreement + phase_relation label -------------

def test_mislabelled_cell_refuses_naming_requested_vs_measured():
    """THE FOOTGUN: a persisted response-mode object requested as all-position
    must refuse (a two-way measured==fit check passes it: fit=response,
    measured=response). The refusal names requested vs measured."""
    with pytest.raises(PhaseVerificationError) as ei:
        verify_phase(requested=ALL, fit=RESPONSE, measured=RESPONSE)
    msg = str(ei.value)
    assert "MISLABELLED" in msg
    assert "requested all-position" in msg and "measured response" in msg


def test_class_a_b_fit_all_served_decode_only_runs_cross_phase():
    """A Class A/B object fit all-position, served decode-only, is a legitimate
    TRANSFER test — runs, labelled cross_phase (measured != fit is the point)."""
    rel = verify_phase(requested=RESPONSE, fit=ALL, measured=RESPONSE)
    assert rel == CROSS_PHASE


def test_matched_cell_runs_labelled_matched():
    assert verify_phase(requested=ALL, fit=ALL, measured=ALL) == MATCHED
    assert verify_phase(requested=RESPONSE, fit=RESPONSE, measured=RESPONSE) == MATCHED


def test_non_persisting_class_c_refuses():
    """A decode-only (Class C) artifact fit response that re-hydrates
    all-position probes ALL; requested response -> mislabelled refuse (the
    serialisation footgun this catches, without a per-arm table)."""
    with pytest.raises(PhaseVerificationError) as ei:
        verify_phase(requested=RESPONSE, fit=RESPONSE, measured=ALL)
    assert "MISLABELLED" in str(ei.value)


def test_class_c_requested_all_position_refuses_cross_phase():
    """The other face of the footgun: a response-fit (Class C) artifact asked
    for all-position and probing all-position is NOT a valid cross-phase (only
    fit-all may go cross-phase) — refuse."""
    with pytest.raises(PhaseVerificationError) as ei:
        verify_phase(requested=ALL, fit=RESPONSE, measured=ALL)
    assert "cross-phase refused" in str(ei.value)


def test_no_op_probe_default_denies():
    with pytest.raises(PhaseVerificationError):
        verify_phase(requested=RESPONSE, fit=ALL, measured=NONE)


def test_unverifiable_fit_default_denies():
    """No positions_at_fit provenance -> default-deny."""
    with pytest.raises(PhaseVerificationError) as ei:
        verify_phase(requested=RESPONSE, fit=None, measured=RESPONSE)
    assert "unverifiable" in str(ei.value)


def test_max_new_tokens_one_is_refused():
    """A decode-only cell with max_new_tokens <= 1 never enters the ctx -> 100%
    baseline regardless of arm; the configuration is refused outright."""
    from sisteer.eval.phase_probe import assert_decode_tokens_reachable
    for n in (0, 1):
        with pytest.raises(PhaseVerificationError):
            assert_decode_tokens_reachable(n)
    assert_decode_tokens_reachable(2)  # >= 2 is fine (no raise)


def test_probe_object_phase_uses_the_capture_pair():
    """probe_object_phase diffs the with/without-ctx captures -> measured phase."""
    from sisteer.eval.phase_probe import probe_object_phase
    def capture(applied):
        # prompt_len=2, seq=4; steered changes only generated positions 2,3
        base = [[0.0] * 3 for _ in range(4)]
        if not applied:
            return base
        steer = [list(r) for r in base]
        steer[2] = steer[3] = [1.0, 1.0, 1.0]
        return steer
    assert probe_object_phase(capture, prompt_len=2, seq_len=4) == RESPONSE


# -- FOOTPRINT comparison (Change 3): position SETS, not coarse labels --------- #

from sisteer.eval.phase_probe import (  # noqa: E402
    MIXED,
    Footprint,
    expand_footprint,
    footprints_equal,
    measure_footprint,
    probe_object_footprint,
    verify_footprint,
)


def _hidden_at(changed, seq=14, d=3):
    base = [[0.0] * d for _ in range(seq)]
    steer = [list(r) for r in base]
    for i in changed:
        steer[i] = [1.0] * d
    return base, steer


# region layout for these tests: prompt_len(content boundary)=10, header_len(H)=4
# so content=[0,6), header=[6,10), generated=[10,14). BOS at 0 ignored.
_P, _N, _H = 10, 14, 4


def test_footprint_regions_all_response_prompt():
    # decode-only serve (position_mask): header 6..9 + generated 10..13
    base, steer = _hidden_at([6, 7, 8, 9, 10, 11, 12, 13])
    fp = measure_footprint(base, steer, prompt_len=_P, header_len=_H,
                           ignore_positions=(0,))
    assert fp == Footprint("none", "all", "all") and fp.label() == RESPONSE
    # all-position serve (skip BOS): content 1..5 + header + generated
    base, steer = _hidden_at(range(1, _N))
    fp = measure_footprint(base, steer, prompt_len=_P, header_len=_H,
                           ignore_positions=(0,))
    assert fp == Footprint("all", "all", "all") and fp.label() == ALL
    # prompt-only serve (loreft identity): content + header, no generated
    base, steer = _hidden_at(range(1, _P))
    fp = measure_footprint(base, steer, prompt_len=_P, header_len=_H,
                           ignore_positions=(0,))
    assert fp == Footprint("all", "all", "none") and fp.label() == PROMPT


def test_antidote_footprint_is_mixed_not_a_clean_phase():
    """THE TRAP the footprint catches: antidote steers ONLY the last prompt token
    (index 9, the final header token) + generated. Its header coverage is PARTIAL,
    so it is MIXED — a coarse 'any prompt + any gen -> ALL' check would miss it."""
    base, steer = _hidden_at([9, 10, 11, 12, 13])
    fp = measure_footprint(base, steer, prompt_len=_P, header_len=_H,
                           ignore_positions=(0,))
    assert fp.header == "partial"
    assert fp.label() == MIXED


def test_expand_footprint_descriptors():
    assert expand_footprint("all", prompt_len=_P, seq_len=_N, header_len=_H) == \
        Footprint("all", "all", "all")
    # the decode-only request INCLUDES the header (it shapes token 1).
    assert expand_footprint("response", prompt_len=_P, seq_len=_N, header_len=_H) == \
        Footprint("none", "all", "all")
    assert expand_footprint("prompt", prompt_len=_P, seq_len=_N, header_len=_H) == \
        Footprint("all", "all", "none")
    # antidote: last header token (partial when H>1) + generated.
    assert expand_footprint("last_prompt+decode", prompt_len=_P, seq_len=_N,
                            header_len=_H) == Footprint("none", "partial", "all")


def test_footprint_subset_semantics():
    allfp = Footprint("all", "all", "all")
    resp = Footprint("none", "all", "all")
    prompt = Footprint("all", "all", "none")
    assert resp <= allfp                    # decode ⊆ all: a restriction
    assert not (resp <= prompt)             # decode ⊄ prompt: a relocation
    assert Footprint("none", "partial", "all") <= allfp  # antidote ⊆ all


def test_verify_footprint_cross_phase_and_matched():
    P, N, H = _P, _N, _H
    req = expand_footprint("response", prompt_len=P, seq_len=N, header_len=H)
    defd = expand_footprint("all", prompt_len=P, seq_len=N, header_len=H)
    fit_all = expand_footprint("all", prompt_len=P, seq_len=N, header_len=H)
    # a fit-all object served decode-only: cross_phase.
    assert verify_footprint(requested=req, defined=defd, measured=req,
                            fit=fit_all) == CROSS_PHASE
    # all==all==all: matched.
    assert verify_footprint(requested=defd, defined=defd, measured=defd,
                            fit=fit_all) == MATCHED


def test_verify_footprint_antidote_mislabelled():
    """antidote's measured footprint (partial header) equals neither a clean
    all nor a clean decode request -> MISLABELLED, both directions."""
    P, N, H = _P, _N, _H
    measured = Footprint("none", "partial", "all")   # antidote
    defd_anti = expand_footprint("last_prompt+decode", prompt_len=P, seq_len=N,
                                 header_len=H)
    for req_desc in ("response", "all"):
        req = expand_footprint(req_desc, prompt_len=P, seq_len=N, header_len=H)
        with pytest.raises(PhaseVerificationError) as ei:
            verify_footprint(requested=req, defined=defd_anti, measured=measured,
                             fit=measured)
        assert "MISLABELLED" in str(ei.value)


def test_verify_footprint_relocation_refused():
    """A decode-only serve of a prompt-only method (loreft, had it reached the
    probe): measured==requested (decode) but decode ⊄ prompt -> RELOCATION."""
    P, N, H = _P, _N, _H
    req = expand_footprint("response", prompt_len=P, seq_len=N, header_len=H)
    defd_prompt = expand_footprint("prompt", prompt_len=P, seq_len=N, header_len=H)
    with pytest.raises(PhaseVerificationError) as ei:
        verify_footprint(requested=req, defined=defd_prompt, measured=req,
                         fit=req)
    assert "RELOCATION" in str(ei.value)


def test_verify_footprint_noop_and_unverifiable():
    P, N, H = _P, _N, _H
    allfp = expand_footprint("all", prompt_len=P, seq_len=N, header_len=H)
    nofp = Footprint("none", "none", "none")
    with pytest.raises(PhaseVerificationError):     # no-op probe
        verify_footprint(requested=allfp, defined=allfp, measured=nofp, fit=allfp)
    req = expand_footprint("response", prompt_len=P, seq_len=N, header_len=H)
    with pytest.raises(PhaseVerificationError):     # no fit provenance
        verify_footprint(requested=req, defined=allfp, measured=req, fit=None)


def test_probe_object_footprint_ignores_bos():
    """probe_object_footprint diffs the capture pair and honours ignore_positions
    (the true BOS) so an all-position skip-BOS serve reads full content, not
    partial."""
    def capture(applied):
        base = [[0.0] * 3 for _ in range(_N)]
        if not applied:
            return base
        steer = [list(r) for r in base]
        for i in range(1, _N):   # everything except BOS at 0
            steer[i] = [1.0] * 3
        return steer
    fp = probe_object_footprint(capture, prompt_len=_P, seq_len=_N, header_len=_H,
                                ignore_positions=(0,))
    assert fp == Footprint("all", "all", "all")


def test_token1_position_is_last_prompt_position():
    """The position whose logits produce token 1 is the last prompt token — under
    position-mask serving it is the final header token and IS steered."""
    from sisteer.eval.phase_probe_gpu import token1_position
    assert token1_position(prompt_width=10, header_len=4) == 9
