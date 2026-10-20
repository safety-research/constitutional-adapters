"""The decode-only (serving-PHASE) capability registry + the meaningfulness RULE.

Three DISTINCT refusal reasons are asserted (never one bucket):
- ``not_expressible`` — the intervention cannot apply a decode mask at all
  (a prompt-embedded sentinel; a prefill-only closed-form object with no seam);
- ``phase_invariant`` — one hardcoded footprint regardless of request (antidote);
- ``meaningless_serve`` — CAN express the mask, but a decode-only serve is a
  RELOCATION onto positions the method is defined to avoid (loreft).
The SUPPORTED arms serve decode-only as a meaningful RESTRICTION; an unregistered
arm is DEFAULT-DENY. The underlying rule (:func:`classify_phase_cell`) is checked
against all four known cases so a NEW arm classifies automatically.
"""

from __future__ import annotations

import pytest

from sisteer.eval.decode_capability import (
    DECODE_ONLY_CAPABILITY,
    FOOTPRINT_ALL,
    FOOTPRINT_LAST_PLUS_DECODE,
    FOOTPRINT_PROMPT,
    MEANINGLESS_SERVE,
    NOT_EXPRESSIBLE,
    PHASE_INVARIANT,
    REFUSAL_REASONS,
    SUPPORTED,
    NoDecodeAnalogueError,
    assert_decode_capable,
    classify_phase_cell,
    decode_capability,
    defined_footprint,
    refusal_reason_code,
    supports_decode_only,
)

_SUPPORTED = ["alphakl_null", "alphakl_soft", "psr_kl_all", "psr_mse_all",
              "vector", "diffmean", "bidpo", "lora_r1", "lofit", "bidpo_lora",
              "lora_r1_all", "lora_r4"]
#: refused arms, each with its DISTINCT reason code.
_REFUSED = {
    "token_bos": NOT_EXPRESSIBLE,
    "token_faithful": NOT_EXPRESSIBLE,
    "alphasteer": NOT_EXPRESSIBLE,
    "caa_static_r": NOT_EXPRESSIBLE,
    "antidote": PHASE_INVARIANT,
    "loreft": MEANINGLESS_SERVE,
}


@pytest.mark.parametrize("arm", _SUPPORTED)
def test_supported_arms_serve_decode_only(arm):
    assert decode_capability(arm)[0] == SUPPORTED
    assert supports_decode_only(arm)
    assert assert_decode_capable(arm) == SUPPORTED
    # a supported arm is DEFINED all-position, so decode ⊆ all is a restriction.
    assert defined_footprint(arm) == FOOTPRINT_ALL


@pytest.mark.parametrize("arm,reason", sorted(_REFUSED.items()))
def test_refused_arms_carry_their_distinct_reason(arm, reason):
    assert refusal_reason_code(arm) == reason
    assert reason in REFUSAL_REASONS
    assert not supports_decode_only(arm)
    with pytest.raises(NoDecodeAnalogueError) as ei:
        assert_decode_capable(arm)
    # the refusal names the arm, the reason code, and a reason string.
    msg = str(ei.value)
    assert arm in msg and reason in msg
    assert decode_capability(arm)[1] in msg


def test_the_three_reasons_are_all_distinct_and_present():
    """All three refusal reasons must appear in the registry — a reader must be
    able to tell mechanical incapacity from phase-invariance from a relocation."""
    present = {refusal_reason_code(a) for a in _REFUSED}
    assert present == set(REFUSAL_REASONS)


def test_antidote_is_phase_invariant_not_partial():
    """antidote is reclassified from PARTIAL to phase_invariant: its
    positions='last' footprint is hardcoded (varies_with_request=False), so it has
    no phase contrast and is NOT a servable phase cell."""
    status, reason = decode_capability("antidote")
    assert status == PHASE_INVARIANT
    assert not supports_decode_only("antidote")
    assert defined_footprint("antidote") == FOOTPRINT_LAST_PLUS_DECODE
    # the docstring precision: adjacent to, not identical with, a cross-phase serve.
    assert "adjacent" in reason.lower()
    assert "header" in reason.lower()
    # the surprise consequence is recorded: an all-position antidote cell refuses.
    assert "all-position" in reason.lower() or "all ⊄" in reason


def test_loreft_is_meaningless_serve_a_relocation():
    """loreft CAN express the mask (prompt-only fit supported) but a decode serve
    is a relocation onto the positions it is defined to avoid — meaningless_serve,
    refused at construction (declared-reason fallback)."""
    status, reason = decode_capability("loreft")
    assert status == MEANINGLESS_SERVE
    assert defined_footprint("loreft") == FOOTPRINT_PROMPT
    assert "relocat" in reason.lower() or "complement" in reason.lower()


def test_unregistered_arm_is_default_deny():
    status, reason = decode_capability("some_unregistered_arm")
    assert status == NOT_EXPRESSIBLE
    assert "UNVERIFIED" in reason.upper()
    # an unknown arm's defined footprint is the most restrictive (prompt), so any
    # generated serve fails the subset test besides.
    assert defined_footprint("some_unregistered_arm") == FOOTPRINT_PROMPT
    with pytest.raises(NoDecodeAnalogueError):
        assert_decode_capable("some_unregistered_arm")


# -- the RULE: classify_phase_cell against the four known cases ---------------- #

class _FP:
    """A tiny footprint stub supporting ``<=`` for the rule's subset test."""

    def __init__(self, s):
        self.s = frozenset(s)

    def __le__(self, other):
        return self.s <= other.s


def test_rule_supported_arm_decode_is_a_restriction():
    """psr/vector/... defined all-position: decode ⊆ all → SUPPORTED restriction."""
    verdict = classify_phase_cell(
        expressible=True, footprint_varies=True,
        served_footprint=_FP({"decode"}),
        defined_footprint=_FP({"content", "header", "decode"}))
    assert verdict == SUPPORTED


def test_rule_loreft_decode_is_a_relocation():
    """loreft defined prompt-only: decode ⊄ prompt → MEANINGLESS_SERVE."""
    verdict = classify_phase_cell(
        expressible=True, footprint_varies=True,
        served_footprint=_FP({"decode"}),
        defined_footprint=_FP({"content", "header"}))
    assert verdict == MEANINGLESS_SERVE


def test_rule_antidote_hardcoded_is_phase_invariant():
    """antidote's footprint does NOT vary with request → PHASE_INVARIANT (before
    the subset test is even reached)."""
    verdict = classify_phase_cell(
        expressible=True, footprint_varies=False,
        served_footprint=_FP({"last", "decode"}),
        defined_footprint=_FP({"last", "decode"}))
    assert verdict == PHASE_INVARIANT


def test_rule_antidote_all_position_would_also_fail_subset():
    """The independent reason antidote is a novel arm, not a phase cell: an
    all-position request (content+header+decode) is NOT a subset of its defined
    last+decode footprint. (Here footprint_varies=True isolates the subset test.)"""
    verdict = classify_phase_cell(
        expressible=True, footprint_varies=True,
        served_footprint=_FP({"content", "header", "decode"}),
        defined_footprint=_FP({"last", "decode"}))
    assert verdict == MEANINGLESS_SERVE


def test_rule_soft_token_is_not_expressible():
    """token_bos/token_faithful cannot apply a decode mask at all → NOT_EXPRESSIBLE
    (gated first, before phase-invariance or subset)."""
    verdict = classify_phase_cell(
        expressible=False, footprint_varies=False,
        served_footprint=_FP(set()),
        defined_footprint=_FP({"content", "header"}))
    assert verdict == NOT_EXPRESSIBLE


def test_registry_statuses_reproduce_the_rule():
    """Each registry entry's stored status must equal what the rule yields from
    its stored (expressible, varies, defined) inputs against a canonical decode
    request — so the table and the rule can never silently disagree."""
    from sisteer.eval import phase_probe as pp
    P, N, H = 10, 14, 4
    req_decode = pp.expand_footprint("response", prompt_len=P, seq_len=N,
                                     header_len=H)
    for arm, (status, defn, expr, varies, _reason) in DECODE_ONLY_CAPABILITY.items():
        defined = pp.expand_footprint(defn, prompt_len=P, seq_len=N, header_len=H)
        verdict = classify_phase_cell(expressible=expr, footprint_varies=varies,
                                      served_footprint=req_decode,
                                      defined_footprint=defined)
        assert verdict == status, f"{arm}: rule {verdict} != registry {status}"


def test_every_registered_arm_is_a_real_arm():
    """The registry keys must be real methods.ARMS names — a typo/drift would
    silently default-deny a live arm (or leave a dead entry)."""
    from sisteer.methods import ARMS
    unknown = [a for a in DECODE_ONLY_CAPABILITY if a not in ARMS]
    assert unknown == [], f"registry names not in methods.ARMS: {unknown}"


def test_sweep_serve_gate_is_the_empirical_probe_default_deny(tmp_path):
    """The sweep's serve gate is the EMPIRICAL footprint probe, NOT this per-arm
    table. With no positions_at_fit provenance the check is unverifiable ->
    default-deny, refusing before any GPU probe forward."""
    import importlib.util
    from pathlib import Path
    from sisteer.eval import sweep
    from sisteer.eval.phase_probe import PhaseVerificationError
    drv_path = Path(__file__).resolve().parents[1] / "scripts" / "unified_sweep_run.py"
    spec = importlib.util.spec_from_file_location("unified_sweep_run", drv_path)
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    cond = sweep.Condition(label="c", cell="safety_refusal__alphasteer",
                           mask="generated", static=("bon",), adaptive=(),
                           benign=(), held_out=())
    with pytest.raises(PhaseVerificationError):
        drv._verify_decode_phase(model=None, tok=None, defense=None,
                                 cond=cond, cfg={})


def test_max_new_tokens_one_is_refused():
    """A decode-only cell with max_new_tokens <= 1 never enters the ctx -> 100%
    baseline regardless of arm; the configuration is refused outright."""
    from sisteer.eval.phase_probe import (PhaseVerificationError,
                                          assert_decode_tokens_reachable)
    for n in (0, 1):
        with pytest.raises(PhaseVerificationError):
            assert_decode_tokens_reachable(n)
    assert_decode_tokens_reachable(2)  # >= 2 is fine (no raise)


def test_rule_derivation_locality_is_the_second_half():
    """An edit computed at one position and applied at others is a
    RELOCATION OF THE FUNCTION even when the served footprint is inside the
    defined set — the positional subset test alone would pass it
    (alphasteer's compute-once-broadcast shape). derivation_local=False
    refuses as meaningless_serve; the default (True) changes nothing for the
    current arms, all of which compute per position from the local activation."""
    served = _FP({"decode"})
    defined = _FP({"content", "header", "decode"})
    assert classify_phase_cell(expressible=True, footprint_varies=True,
                               served_footprint=served, defined_footprint=defined,
                               derivation_local=False) == MEANINGLESS_SERVE
    assert classify_phase_cell(expressible=True, footprint_varies=True,
                               served_footprint=served,
                               defined_footprint=defined) == SUPPORTED
