"""The axis-agnostic engine-identity primitive (engine_identity.py).

Two observations produced on different engines must never be compared;
the primitive is that rule as a callable invariant (an invariant that
refuses beats a sidecar artifact someone has to remember to check).
Covers: identical identities accepted; ANY difference refused with a
reason naming the axis, both identities, and the differing components;
malformed identities refused loudly; and the petri delegation —
``check_pairing`` CALLS the primitive (its own accept/refuse set stays
pinned, unmodified, in test_petri_cells.py)."""

import pytest

from sisteer.capeval import engine_identity as ei
from sisteer.capeval.petri import defense_cells as dc

AXIS = "attacker engine"


# --- the consuming case: opaque identities on an arbitrary axis -------------


def test_differing_engines_refused_on_arbitrary_axis():
    """Two observations differing ONLY in engine identity, on an axis the
    petri vocabulary knows nothing about, refuse — with both identities
    and the axis named in the reason."""
    with pytest.raises(ei.EngineIdentityError) as ex:
        ei.check_same_identity("vllm-0.11.0", "hf-transformers", axis=AXIS)
    msg = str(ex.value)
    assert AXIS in msg
    assert "vllm-0.11.0" in msg and "hf-transformers" in msg
    assert ex.value.axis == AXIS
    assert ex.value.identity_a == "vllm-0.11.0"
    assert ex.value.identity_b == "hf-transformers"
    assert ex.value.differing == (ei.OPAQUE,)


def test_identical_engines_accepted_on_arbitrary_axis():
    ok = ei.check_same_identity("vllm-0.11.0", "vllm-0.11.0", axis=AXIS)
    assert isinstance(ok, ei.IdentityMatch)
    assert ok.axis == AXIS
    assert ok.identity_a == ok.identity_b == "vllm-0.11.0"


def test_refusal_is_a_value_error():
    # repo convention: refusals are ValueError subclasses, so existing
    # catch-and-report plumbing handles the primitive unchanged
    assert issubclass(ei.EngineIdentityError, ValueError)


def test_comparison_is_exact_no_normalization():
    # an identity is a label of record: any differing byte refuses
    with pytest.raises(ei.EngineIdentityError):
        ei.check_same_identity("vLLM", "vllm", axis=AXIS)
    with pytest.raises(ei.EngineIdentityError):
        ei.check_same_identity("vllm ", "vllm", axis=AXIS)


# --- structured (component-mapping) identities -------------------------------


def test_structured_identity_names_the_differing_component():
    a = {"engine": "vllm", "version": "0.11.0"}
    b = {"engine": "vllm", "version": "0.6.3"}
    with pytest.raises(ei.EngineIdentityError) as ex:
        ei.check_same_identity(a, b, axis=AXIS)
    msg = str(ex.value)
    assert ex.value.differing == ("version",)
    assert "version" in msg
    assert "0.11.0" in msg and "0.6.3" in msg


def test_structured_identity_equal_accepted():
    a = {"engine": "vllm", "version": "0.11.0"}
    ok = ei.check_same_identity(a, dict(a), axis=AXIS)
    assert ok.identity_a == ok.identity_b == "engine='vllm', version='0.11.0'"


def test_multiple_differing_components_all_named_in_order():
    a = {"backend": "proxy_vllm", "path": "lens", "mode": "interleaved"}
    b = {"backend": "hf_shim", "path": "lens", "mode": "exact_replay"}
    assert ei.identity_diff(a, b) == ("backend", "mode")


def test_component_present_on_one_side_only_differs():
    assert ei.identity_diff({"engine": "vllm"},
                            {"engine": "vllm", "gpu": "H200"}) == ("gpu",)
    assert ei.identity_diff({"engine": "vllm", "gpu": "H200"},
                            {"engine": "vllm"}) == ("gpu",)


def test_identity_diff_pure_results():
    assert ei.identity_diff("x", "x") == ()
    assert ei.identity_diff("x", "y") == (ei.OPAQUE,)
    assert ei.identity_diff({"a": "1"}, {"a": "1"}) == ()


# --- malformed identities refuse loudly (never a silent verdict) ------------


@pytest.mark.parametrize("bad", ["", {}, None, 3, ["vllm"]])
def test_malformed_identity_refused(bad):
    with pytest.raises(ei.EngineIdentityError):
        ei.check_same_identity(bad, "vllm", axis=AXIS)
    with pytest.raises(ei.EngineIdentityError):
        ei.check_same_identity("vllm", bad, axis=AXIS)


@pytest.mark.parametrize("bad", [{"engine": 1}, {"engine": ""},
                                 {1: "vllm"}, {"": "vllm"}])
def test_non_string_or_empty_component_refused(bad):
    with pytest.raises(ei.EngineIdentityError):
        ei.check_same_identity(bad, bad, axis=AXIS)


def test_mixed_representations_refused_even_when_plausibly_equal():
    with pytest.raises(ei.EngineIdentityError, match="mixed"):
        ei.check_same_identity("vllm", {"engine": "vllm"}, axis=AXIS)


@pytest.mark.parametrize("bad_axis", ["", None, 3])
def test_empty_or_non_string_axis_refused(bad_axis):
    with pytest.raises(ei.EngineIdentityError, match="axis"):
        ei.check_same_identity("vllm", "vllm", axis=bad_axis)


def test_malformed_refusal_carries_no_differing_components():
    # .differing distinguishes "the identities differ" from "the metadata
    # is malformed" for callers that map refusals onto their own prose
    with pytest.raises(ei.EngineIdentityError) as ex:
        ei.check_same_identity("", "vllm", axis=AXIS)
    assert ex.value.differing == ()


# --- petri serving identities consume as opaque strings ----------------------


def test_petri_serving_identity_strings_work_as_opaque_identities():
    """The petri `<backend>+<path>+<mode>` strings are valid opaque
    identities for a caller that wants the blunt all-or-nothing check
    rather than petri's per-component tier rules."""
    lens = "proxy_vllm+lens+exact_replay"
    hf = "hf_shim+apply+exact_replay"
    assert ei.check_same_identity(lens, lens,
                                  axis="target serving").identity_a == lens
    with pytest.raises(ei.EngineIdentityError, match="target serving"):
        ei.check_same_identity(lens, hf, axis="target serving")


# --- the petri pairing rules are a CALLER of the primitive -------------------


LENS = "proxy_vllm+lens+exact_replay"
HF = "hf_shim+apply+exact_replay"
LENS_INTERLEAVED = "proxy_vllm+lens+interleaved"


def _leg(cell_id, identity, control_of=None):
    return dc.PairingLeg(cell_id=cell_id, serving_identity=identity,
                         control_of=control_of)


def test_check_pairing_delegates_the_identity_comparison(monkeypatch):
    """Pin the delegation: check_pairing's engine comparison IS the
    primitive, so a future edit cannot quietly fork the logic. (What the
    pairing accepts/refuses stays pinned in test_petri_cells.py.)"""
    calls = []
    real = dc.identity_diff

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(dc, "identity_diff", spy)
    ok = dc.check_pairing(_leg("si_safety__cast", LENS),
                          _leg("none__lens0", LENS, control_of="none"),
                          tier=1)
    assert ok.cross_path is False
    (a, b), = calls
    assert a == b == {"backend": "proxy_vllm", "path": "lens",
                      "mode": "exact_replay"}


def test_check_pairing_still_refuses_cross_engine_rows():
    # the original invariant, now routed through the primitive: proxy
    # rows never pair against HF rows, and modes never mix
    with pytest.raises(dc.PairingError, match="backend"):
        dc.check_pairing(_leg("si_safety__cast", LENS),
                         _leg("prompted__safety", HF), tier=1)
    with pytest.raises(dc.PairingError, match="mode"):
        dc.check_pairing(_leg("si_safety__cast", LENS_INTERLEAVED),
                         _leg("none__lens0", LENS, control_of="none"),
                         tier=1)
