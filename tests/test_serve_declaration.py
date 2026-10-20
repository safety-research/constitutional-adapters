"""The hashed serve-declaration vocabulary and the ``@<variant>`` cell-id
grammar (``sisteer.capeval.serving.serve_declaration``).

Covers: the CLOSED vocabulary (two named experiments, one out-of-support
token, three variants — an unknown name refuses everywhere); ``split_variant``
/ ``join_variant`` round trips and refusals (at most one separator, an empty
base, an unknown variant); ``apply_variant`` writing exactly one declaration
per variant, REPLACING the artifact-derived ``positions`` default, refusing a
double overlay (a spec already carrying a declaration key) and never mutating
the base spec; ``declarations_of`` accepting the two closed shapes and
refusing every malformed value; and — on a real-shaped PSR blob — that the
overlay touches the SPEC only: the artifact bytes, their sha256 and the fit
regime ``recorded_fit_positions`` reads from those bytes are untouched.

CPU-offline. The module under test is torch-free (the proxy imports it); the
two tests that need torch (the lens_adapters position literals, the PSR blob)
import it inside the test."""

import copy
import hashlib
import io

import pytest

from sisteer.capeval.serving import serve_declaration as sd

BASE_PSR_SPEC = {"kind": "psr", "arm": "psr_kl_all", "si_id": "safety",
                 "layers": [0, 2], "positions": "response",
                 "response_header_len": 7, "train_direction": False}


# --------------------------------------------------------------------------
# the closed vocabulary
# --------------------------------------------------------------------------


def test_vocabulary_is_closed_and_pinned():
    assert sd.SERVE_OUT_OF_SUPPORT_KEY == "serve_out_of_support"
    assert sd.SERVE_EXPERIMENT_KEY == "serve_experiment"
    assert sd.DECLARATION_KEYS == ("serve_out_of_support", "serve_experiment")
    assert sd.OUT_OF_SUPPORT_TOKENS == ("thinking",)
    assert sd.SERVE_EXPERIMENTS == ("fit_response_serve_all_turns",
                                    "fit_response_serve_all_positions")
    assert sd.VARIANT_SEP == "@"
    assert set(sd.SERVE_VARIANTS) == {"thinking", "all_turns", "all_positions"}
    assert sd.SERVE_VARIANTS["thinking"] == {
        "serve_out_of_support": ["thinking"]}
    assert sd.SERVE_VARIANTS["all_turns"] == {
        "positions": "response_all_turns",
        "serve_experiment": "fit_response_serve_all_turns"}
    assert sd.SERVE_VARIANTS["all_positions"] == {
        "positions": "all",
        "serve_experiment": "fit_response_serve_all_positions"}


def test_every_variant_carries_exactly_one_declaration():
    """No variant is attested by its ``positions``
    alone — every variant writes exactly one of the two declaration keys,
    and ``declarations_of`` on the overlay is that key alone."""
    for variant, overlay in sd.SERVE_VARIANTS.items():
        decl_keys = [k for k in sd.DECLARATION_KEYS if k in overlay]
        assert len(decl_keys) == 1, variant
        assert sd.declarations_of(overlay) == sd.variant_declarations(variant)
        assert sd.declarations_of(overlay) == {
            k: overlay[k] for k in decl_keys}
        assert sd.variant_overlay(variant) == overlay
        # the accessors hand out COPIES — a caller cannot edit the vocabulary
        sd.variant_overlay(variant)["positions"] = "edited"
        sd.variant_declarations(variant).clear()
        assert sd.SERVE_VARIANTS[variant] == overlay


def test_each_experiment_names_exactly_one_fit_serve_pair():
    """A name describes ONE (recorded fit, served positions)
    pair, so the guard can check that the name describes the deployment."""
    assert set(sd.EXPERIMENT_FIT_SERVE_PAIRS) == set(sd.SERVE_EXPERIMENTS)
    assert sd.EXPERIMENT_FIT_SERVE_PAIRS[
        sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS] == (
            "response", "response_all_turns")
    assert sd.EXPERIMENT_FIT_SERVE_PAIRS[
        sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_POSITIONS] == ("response", "all")
    # and each variant that names an experiment SERVES that experiment's
    # positions — the vocabulary cannot contradict itself
    for variant, overlay in sd.SERVE_VARIANTS.items():
        name = overlay.get(sd.SERVE_EXPERIMENT_KEY)
        if name is not None:
            assert overlay[sd.POSITIONS_KEY] == \
                sd.EXPERIMENT_FIT_SERVE_PAIRS[name][1], variant


def test_position_literals_match_lens_adapters():
    """This module is torch-free so it cannot import lens_adapters' POSITIONS_*
    constants; pin the literals against them here, where torch is loaded."""
    from sisteer.capeval.serving import lens_adapters as la

    assert sd.SERVE_VARIANTS["all_turns"]["positions"] == \
        la.POSITIONS_RESPONSE_ALL_TURNS
    assert sd.SERVE_VARIANTS["all_positions"]["positions"] == la.POSITIONS_ALL
    for recorded, served in sd.EXPERIMENT_FIT_SERVE_PAIRS.values():
        assert recorded == la.POSITIONS_RESPONSE
        assert served in la._SERVED_POSITION_MODES
    assert sd.POSITIONS_KEY == "positions"


def test_unknown_names_refuse_everywhere():
    for bad in ("bogus", "THINKING", "", None, "thinking "):
        with pytest.raises(sd.ServeDeclarationError):
            sd.variant_overlay(bad)
        with pytest.raises(sd.ServeDeclarationError):
            sd.variant_declarations(bad)
        with pytest.raises(sd.ServeDeclarationError):
            sd.apply_variant(dict(BASE_PSR_SPEC), bad)
    with pytest.raises(sd.ServeDeclarationError):
        sd.join_variant("si_safety__psr_kl_all", "bogus")


# --------------------------------------------------------------------------
# split / join
# --------------------------------------------------------------------------


def test_split_variant_round_trips():
    assert sd.split_variant("si_safety__psr_kl_all") == (
        "si_safety__psr_kl_all", None)
    for variant in sd.SERVE_VARIANTS:
        full = f"si_safety__psr_kl_all@{variant}"
        assert sd.split_variant(full) == ("si_safety__psr_kl_all", variant)
        assert sd.join_variant("si_safety__psr_kl_all", variant) == full
    assert sd.join_variant("none", None) == "none"
    # ids without the separator are untouched — '@' never occurred in a
    # cell id before, so no existing id changes meaning
    for legacy in ("none", "prompted__safety", "antidote", "none__lens0",
                   "stacked__safety__psr_kl_all",
                   "safety_refusal__psr_kl_all"):
        assert sd.split_variant(legacy) == (legacy, None)


@pytest.mark.parametrize("bad", [
    "", None, 7,
    "@thinking",                                   # empty base
    "si_safety__psr_kl_all@",                      # empty variant
    "si_safety__psr_kl_all@bogus",                 # unknown variant
    "si_safety__psr_kl_all@THINKING",              # case matters
    "si_safety__psr_kl_all@thinking@thinking",     # at most one separator
    "si_safety__psr_kl_all@all_turns@all_positions",
])
def test_split_variant_refuses(bad):
    with pytest.raises(sd.ServeDeclarationError):
        sd.split_variant(bad)


def test_join_variant_refuses_a_base_that_already_carries_the_separator():
    with pytest.raises(sd.ServeDeclarationError):
        sd.join_variant("si_safety__psr_kl_all@thinking", "all_turns")
    with pytest.raises(sd.ServeDeclarationError):
        sd.join_variant("si_safety__psr_kl_all@thinking", None)
    with pytest.raises(sd.ServeDeclarationError):
        sd.join_variant("", "thinking")


# --------------------------------------------------------------------------
# apply_variant — the ONE overlay
# --------------------------------------------------------------------------


def test_apply_variant_writes_the_declaration_and_replaces_positions():
    base = dict(BASE_PSR_SPEC)
    snapshot = copy.deepcopy(base)

    thinking = sd.apply_variant(base, "thinking")
    assert thinking["positions"] == "response"          # untouched
    assert thinking["serve_out_of_support"] == ["thinking"]
    assert "serve_experiment" not in thinking

    all_turns = sd.apply_variant(base, "all_turns")
    assert all_turns["positions"] == "response_all_turns"  # REPLACED
    assert all_turns["serve_experiment"] == "fit_response_serve_all_turns"
    assert "serve_out_of_support" not in all_turns

    all_pos = sd.apply_variant(base, "all_positions")
    assert all_pos["positions"] == "all"                   # REPLACED
    assert all_pos["serve_experiment"] == "fit_response_serve_all_positions"

    # every other field rides through unchanged, and the base is not mutated
    for out in (thinking, all_turns, all_pos):
        assert out is not base
        for key, value in BASE_PSR_SPEC.items():
            if key != "positions":
                assert out[key] == value
    assert base == snapshot
    # the overlay's list is a fresh object — editing the result never
    # reaches the vocabulary
    thinking["serve_out_of_support"].append("edited")
    assert sd.SERVE_VARIANTS["thinking"]["serve_out_of_support"] == ["thinking"]


def test_apply_variant_on_a_spec_without_positions_still_declares():
    """A base spec need not carry ``positions`` (not every kind persists a
    default) — the variant then SETS it (position variants) or leaves it
    absent (thinking)."""
    base = {"kind": "vector_all", "arm": "vector_all"}
    assert "positions" not in sd.apply_variant(base, "thinking")
    assert sd.apply_variant(base, "all_turns")["positions"] == \
        "response_all_turns"


def test_apply_variant_refuses_a_double_overlay():
    """The positive markers a variant writes are the refusal trigger — a
    second application, or a base artifact someone hand-edited. ``positions``
    is NEVER the trigger (the campaign's PSR specs already carry
    positions='response')."""
    once = sd.apply_variant(dict(BASE_PSR_SPEC), "all_turns")
    for again in sd.SERVE_VARIANTS:
        with pytest.raises(sd.ServeDeclarationError, match="already"):
            sd.apply_variant(once, again)
    hand_edited = dict(BASE_PSR_SPEC, serve_out_of_support=["thinking"])
    with pytest.raises(sd.ServeDeclarationError, match="already"):
        sd.apply_variant(hand_edited, "all_positions")
    # even a MALFORMED pre-existing declaration key refuses (the key's
    # presence is the signal, not its validity)
    with pytest.raises(sd.ServeDeclarationError, match="already"):
        sd.apply_variant(dict(BASE_PSR_SPEC, serve_experiment=None),
                         "thinking")


def test_apply_variant_refuses_a_non_mapping_spec():
    for bad in (None, "psr", ["kind"], 3):
        with pytest.raises(sd.ServeDeclarationError):
            sd.apply_variant(bad, "thinking")


# --------------------------------------------------------------------------
# declarations_of — the strict reader
# --------------------------------------------------------------------------


def test_declarations_of_accepts_the_two_closed_shapes_only():
    assert sd.declarations_of(BASE_PSR_SPEC) == {}
    assert sd.declarations_of({"kind": "none"}) == {}
    assert sd.declarations_of(
        sd.apply_variant(BASE_PSR_SPEC, "thinking")) == {
            "serve_out_of_support": ["thinking"]}
    assert sd.declarations_of(
        sd.apply_variant(BASE_PSR_SPEC, "all_turns")) == {
            "serve_experiment": "fit_response_serve_all_turns"}
    assert sd.declarations_of(
        sd.apply_variant(BASE_PSR_SPEC, "all_positions")) == {
            "serve_experiment": "fit_response_serve_all_positions"}
    # the returned list is a copy
    spec = sd.apply_variant(BASE_PSR_SPEC, "thinking")
    sd.declarations_of(spec)["serve_out_of_support"].append("x")
    assert spec["serve_out_of_support"] == ["thinking"]


@pytest.mark.parametrize("bad", [
    {"serve_out_of_support": "thinking"},            # not a list
    {"serve_out_of_support": ("thinking",)},         # a tuple is not a list
    {"serve_out_of_support": []},                    # declares nothing
    {"serve_out_of_support": ["multi_turn"]},        # unknown token
    {"serve_out_of_support": ["thinking", "thinking"]},  # duplicate
    {"serve_out_of_support": [None]},
    {"serve_out_of_support": None},
    {"serve_experiment": "fit_all_serve_response"},  # outside the vocabulary
    {"serve_experiment": ["fit_response_serve_all_turns"]},
    {"serve_experiment": None},
    {"serve_experiment": ""},
    # the two closed shapes never combine
    {"serve_out_of_support": ["thinking"],
     "serve_experiment": "fit_response_serve_all_turns"},
])
def test_declarations_of_refuses_malformed_values(bad):
    spec = dict(BASE_PSR_SPEC, **bad)
    with pytest.raises(sd.ServeDeclarationError):
        sd.declarations_of(spec)


def test_declarations_of_refuses_a_non_mapping():
    for bad in (None, "thinking", ["serve_out_of_support"]):
        with pytest.raises(sd.ServeDeclarationError):
            sd.declarations_of(bad)


# --------------------------------------------------------------------------
# the overlay never touches the artifact (revision 1)
# --------------------------------------------------------------------------


def _psr_blob_bytes(**extra) -> bytes:
    """A real-shaped PSR blob (the PSRObject.save schema: kind, layers,
    train_direction, the fit leg ``positions``, ``response_header_len``,
    and the per-layer direction/gate tensors)."""
    import torch

    layers = (0, 2)
    g = torch.Generator().manual_seed(9)
    blob = {
        "kind": "psr", "layers": list(layers), "train_direction": False,
        "positions": "response", "response_header_len": 7,
        "directions": {str(l): torch.randn(8, generator=g) * 0.05
                       for l in layers},
        "gate_w": {str(l): torch.randn(8, generator=g) * 0.3
                   for l in layers},
        "gate_b": {str(l): torch.randn(1, generator=g) * 0.5
                   for l in layers},
    }
    blob.update(extra)
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _spec_from_blob(blob_bytes: bytes) -> dict:
    """The adapter's construction (registry_adapter._resolve_torch_blob):
    every non-tensor stored field, plus kind/arm/si_id."""
    import torch

    blob = torch.load(io.BytesIO(blob_bytes), map_location="cpu",
                      weights_only=True)
    spec = {k: v for k, v in blob.items()
            if k != "kind" and not isinstance(v, dict)}
    spec.update(kind="psr", arm="psr_kl_all", si_id="safety")
    return spec


def test_psr_blob_overlay_touches_the_spec_only():
    """Apply ``@all_turns`` and ``@all_positions`` to a spec built from a
    real-shaped response-fit PSR blob: the served ``positions`` becomes the
    variant's, while the artifact bytes, their sha256 and the fit regime
    the guard reads FROM THOSE BYTES stay exactly what they were — the
    overlay cannot reach the fit regime by construction (it reads bytes,
    never the spec)."""
    from sisteer.capeval.serving.lens_adapters import recorded_fit_positions

    blob_bytes = _psr_blob_bytes()
    sha_before = hashlib.sha256(blob_bytes).hexdigest()
    base = _spec_from_blob(blob_bytes)
    assert base["positions"] == "response"  # the campaign's PSR base specs
    snapshot = copy.deepcopy(base)

    served = {}
    for variant in ("all_turns", "all_positions", "thinking"):
        out = sd.apply_variant(base, variant)
        served[variant] = out["positions"]
        assert sd.declarations_of(out) == sd.variant_declarations(variant)
    assert served == {"all_turns": "response_all_turns",
                      "all_positions": "all", "thinking": "response"}

    # the base spec object is unmodified; the bytes are the same bytes
    assert base == snapshot
    assert hashlib.sha256(blob_bytes).hexdigest() == sha_before
    # and the recorded FIT regime — bytes only — is still 'response'
    assert recorded_fit_positions("psr", blob_bytes, b"") == "response"
