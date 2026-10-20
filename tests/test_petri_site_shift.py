"""The resid_pre site shift, and the mis-keyed-lookup trap it creates.

An object fit at resid_pre is served one block EARLIER, because
resid_pre of layer L is the block output of layer L-1. Only the SERVING
LOCATION shifts — the fit layer stays the lookup key into the artifact's
per-layer mapping.

Getting that backwards is the dangerous bug, and it is dangerous
precisely because it mostly succeeds. The released multi-layer objects
are fit at layers [8,9,10,11,12,13,14,16,18,19]; shifted, that is
[7,8,9,10,11,12,13,15,17,18]. SEVEN OF TEN OVERLAP, so keying the
lookup by the shifted index finds a real, valid tensor for most layers
and serves it one block off — fluent text, plausible steering, wrong
everywhere, and nothing downstream would notice.

What catches it is the per-layer norm recorded alongside the tensors
at fit time: on the released matched object those norms are strictly
monotone in depth, so an off-by-one in either direction always lands on
a different value and can never coincidentally agree.
"""

import io

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la

#: the released multi-layer footprint (AlphaSteer's layer set), and what
#: it becomes once each resid_pre fit is served at the preceding block.
FIT_LAYERS = (8, 9, 10, 11, 12, 13, 14, 16, 18, 19)
SERVED_LAYERS = (7, 8, 9, 10, 11, 12, 13, 15, 17, 18)
#: the overlap that makes a mis-keyed lookup succeed instead of crashing
OVERLAP = (8, 9, 10, 11, 12, 13, 18)
D = 6


def _blob(payload) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _dir_with_norm(norm: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(D, generator=g, dtype=torch.float32)
    return v * (norm / float(v.norm()))


#: a distinct, strictly increasing norm per fit layer — the property
#: that makes the cross-check able to localize an off-by-one
NORMS = {l: 1.0 + 0.5 * i for i, l in enumerate(FIT_LAYERS)}


def _matched_static_r(**over) -> bytes:
    rs = {l: _dir_with_norm(NORMS[l], seed=l) for l in FIT_LAYERS}
    payload = {"kind": "static_r", "layer": None, "r": None,
               "layers": list(FIT_LAYERS), "rs": rs,
               "steer_positions": "prefill_all", "site": "pre",
               "base_scale": 1.0,
               "metadata": {"per_layer_r_norm": dict(NORMS)}}
    payload.update(over)
    return _blob(payload)


def _official_alphasteer(**over) -> bytes:
    g = torch.Generator().manual_seed(4)
    Ms = {l: torch.randn(D, D, generator=g, dtype=torch.float32)
          for l in FIT_LAYERS}
    payload = {"kind": "alphasteer",
               # the legacy singular fields are PRESENT but NULL on a
               # released object — a presence check passes, the value
               # then fails
               "layer": None, "M": None, "refusal": None, "refusals": None,
               "layers": list(FIT_LAYERS), "Ms": Ms,
               "steer_positions": "prefill", "site": "pre",
               "base_scale": 1.0, "metadata": {}}
    payload.update(over)
    return _blob(payload)


# --- the shift itself ---------------------------------------------------------


def test_the_overlap_that_makes_this_trap_live():
    """Pins the premise: a mis-keyed lookup would SUCCEED, not crash."""
    assert tuple(l - 1 for l in FIT_LAYERS) == SERVED_LAYERS
    assert tuple(sorted(set(FIT_LAYERS) & set(SERVED_LAYERS))) == OVERLAP
    assert len(OVERLAP) == 7


def test_static_r_matched_serves_one_block_earlier():
    ops = la.get_adapter("static_r")({"kind": "static_r"},
                                     _matched_static_r(), b"")
    assert tuple(op.layer for op in ops) == SERVED_LAYERS


def test_each_served_op_carries_its_own_fit_layers_tensor():
    """The load-bearing assertion: the op at served layer L-1 must carry
    the tensor fit at L, not the tensor fit at L-1 (which also exists
    for seven of these layers)."""
    ops = la.get_adapter("static_r")({"kind": "static_r"},
                                     _matched_static_r(), b"")
    for fit_layer, op in zip(FIT_LAYERS, ops):
        assert op.layer == fit_layer - 1
        assert float(op.vector.norm()) == pytest.approx(NORMS[fit_layer],
                                                        abs=1e-4)


def test_a_mis_keyed_lookup_is_caught_by_the_recorded_norms():
    """Simulates the bug by rotating the tensors one layer against their
    recorded norms — exactly what keying by the shifted index would
    produce for the overlapping layers."""
    rs = {l: _dir_with_norm(NORMS[l], seed=l) for l in FIT_LAYERS}
    rotated = {l: rs[FIT_LAYERS[(i + 1) % len(FIT_LAYERS)]]
               for i, l in enumerate(FIT_LAYERS)}
    art = _matched_static_r(rs=rotated)
    with pytest.raises(la.AdapterError, match="records"):
        la.get_adapter("static_r")({"kind": "static_r"}, art, b"")


def test_norms_are_distinct_enough_to_localize_an_off_by_one():
    gaps = [abs(NORMS[b] - NORMS[a])
            for a, b in zip(FIT_LAYERS, FIT_LAYERS[1:])]
    assert min(gaps) > la._NORM_ATOL * 100


def test_an_artifact_without_recorded_norms_still_serves():
    """Older artifacts predate the field; absence must not become a
    refusal, only a missing cross-check."""
    art = _matched_static_r(metadata={})
    ops = la.get_adapter("static_r")({"kind": "static_r"}, art, b"")
    assert tuple(op.layer for op in ops) == SERVED_LAYERS


def test_a_malformed_norm_record_refuses_rather_than_skipping_the_check():
    art = _matched_static_r(metadata={"per_layer_r_norm": {8: "not a number"}})
    with pytest.raises(la.AdapterError, match="integrity check"):
        la.get_adapter("static_r")({"kind": "static_r"}, art, b"")


# --- alphasteer: the null-singular trap ---------------------------------------


def test_alphasteer_reads_the_plural_fields_not_the_null_singular_ones():
    ops = la.get_adapter("alphasteer")({"kind": "alphasteer"},
                                       _official_alphasteer(), b"")
    assert tuple(op.layer for op in ops) == SERVED_LAYERS
    assert all(op.op == la.OP_PREFILL_LAST_MATMUL for op in ops)
    assert all(op.phase == la.PHASE_PREFILL for op in ops)


def test_alphasteer_legacy_single_layer_blob_still_serves():
    g = torch.Generator().manual_seed(9)
    art = _blob({"kind": "alphasteer", "layer": 12,
                 "M": torch.randn(D, D, generator=g), "base_scale": 2.0,
                 "metadata": {}})
    (op,) = la.get_adapter("alphasteer")({"kind": "alphasteer"}, art, b"")
    assert op.layer == 12          # site defaults to post: no shift
    assert op.coeff == 2.0


def test_alphasteer_with_neither_plural_nor_legacy_refuses():
    art = _blob({"kind": "alphasteer", "layer": None, "M": None,
                 "base_scale": 1.0, "metadata": {}})
    with pytest.raises(la.AdapterError, match="no matrix to serve"):
        la.get_adapter("alphasteer")({"kind": "alphasteer"}, art, b"")


def test_alphasteer_refuses_layers_disagreeing_with_its_matrices():
    art = _official_alphasteer(layers=[8, 9])
    with pytest.raises(la.AdapterError, match="inconsistent artifact"):
        la.get_adapter("alphasteer")({"kind": "alphasteer"}, art, b"")


def test_alphasteer_refuses_an_unexpressed_position_mode():
    art = _official_alphasteer(steer_positions="prompt_hold")
    with pytest.raises(la.AdapterError, match="not the prefill-only"):
        la.get_adapter("alphasteer")({"kind": "alphasteer"}, art, b"")


def test_a_resid_pre_object_touching_layer_zero_refuses():
    """resid_pre of layer 0 is the embedding output — no preceding block
    can carry the edit, and serving it one block later would be a
    different intervention."""
    art = _official_alphasteer(layers=[0, 8],
                               Ms={0: torch.eye(D), 8: torch.eye(D)})
    with pytest.raises(la.AdapterError, match="embedding output"):
        la.get_adapter("alphasteer")({"kind": "alphasteer"}, art, b"")


def test_non_discriminating_recorded_norms_refuse_rather_than_pass_vacuously():
    """A sibling arm's directions are all exactly unit-norm and its
    metadata records precisely those norms. A cross-check against such a
    record passes for EVERY permutation of the layers — it would report
    success while verifying nothing, which is worse than no check
    because the false assurance is invisible."""
    flat = {l: 1.0 for l in FIT_LAYERS}
    rs = {l: _dir_with_norm(1.0, seed=l) for l in FIT_LAYERS}
    art = _matched_static_r(rs=rs, metadata={"per_layer_r_norm": flat})
    with pytest.raises(la.AdapterError, match="do not .*discriminate"):
        la.get_adapter("static_r")({"kind": "static_r"}, art, b"")


def test_norms_too_close_to_localize_an_off_by_one_also_refuse():
    near = {l: 1.0 + 0.0001 * i for i, l in enumerate(FIT_LAYERS)}
    rs = {l: _dir_with_norm(near[l], seed=l) for l in FIT_LAYERS}
    art = _matched_static_r(rs=rs, metadata={"per_layer_r_norm": near})
    with pytest.raises(la.AdapterError, match="do not .*discriminate"):
        la.get_adapter("static_r")({"kind": "static_r"}, art, b"")


def test_a_single_layer_record_is_exempt():
    """One layer cannot be confused with another, so a lone recorded
    norm is not vacuous — it still catches gross corruption."""
    rs = {16: _dir_with_norm(2.5, seed=16)}
    art = _blob({"kind": "static_r", "layer": None, "r": None,
                 "layers": [16], "rs": rs, "steer_positions": "prompt_last",
                 "site": "post", "base_scale": 1.0,
                 "metadata": {"per_layer_r_norm": {16: 2.5}}})
    (op,) = la.get_adapter("static_r")({"kind": "static_r"}, art, b"")
    assert op.layer == 16


# --- the cross-phase cell: fit-all, served response ---------------------------
#
# A cross-phase cell IS an all-position artifact served under the
# response mask. The mode therefore belongs to the CELL, not the object,
# and the resolved spec overrides the artifact. Before this, the adapter
# took the mask from the artifact and cross-checked the spec against it,
# so a cross-phase cell was a hard refusal — the axis was unservable by
# construction.

HEADER = (128006, 78191, 128007, 271)   # Llama-3.1 assistant header


def test_a_fit_all_vector_object_serves_under_the_response_mask():
    buf = io.BytesIO()
    torch.save({"kind": "vector_all", "vectors": {"3": torch.ones(D)},
                "positions": "all", "skip_bos": True}, buf)
    ops = la.get_adapter("vector_all")(
        {"kind": "vector_all", "positions": "response"},
        buf.getvalue(), b"", header_ids=HEADER)
    assert [o.positions for o in ops] == [la.POSITIONS_RESPONSE]
    assert ops[0].header_ids == HEADER


def test_the_artifacts_own_mode_remains_the_default():
    buf = io.BytesIO()
    torch.save({"kind": "vector_all", "vectors": {"3": torch.ones(D)},
                "positions": "all", "skip_bos": True}, buf)
    (op,) = la.get_adapter("vector_all")({"kind": "vector_all"},
                                         buf.getvalue(), b"")
    assert op.positions == la.POSITIONS_ALL


def test_static_r_served_response_also_flips_the_phase():
    """The object's own deployment is prefill-only. Served cross-phase
    the region spans the header at prefill AND every generated
    position, so the op must be invoked in both phases — a
    prefill-only op would leave generation unsteered."""
    ops = la.get_adapter("static_r")(
        {"kind": "static_r", "positions": "response"},
        _matched_static_r(), b"", header_ids=HEADER)
    assert {o.positions for o in ops} == {la.POSITIONS_RESPONSE}
    assert {o.phase for o in ops} == {la.PHASE_BOTH}
    assert tuple(o.layer for o in ops) == SERVED_LAYERS   # shift preserved


def test_static_r_without_the_override_stays_prefill_only():
    ops = la.get_adapter("static_r")({"kind": "static_r"},
                                     _matched_static_r(), b"")
    assert {o.phase for o in ops} == {la.PHASE_PREFILL}
    assert all(o.header_ids is None for o in ops)


def test_response_override_still_needs_a_header():
    with pytest.raises(la.AdapterError, match="no assistant header"):
        la.get_adapter("static_r")(
            {"kind": "static_r", "positions": "response"},
            _matched_static_r(), b"")


def test_alphasteer_refuses_a_response_override_on_derivation_locality():
    """Its edit is computed ONCE from the last prompt token and
    broadcast, so masking to header-plus-generated applies a
    prompt-derived vector to generated positions — a different
    function, not the same one on fewer positions."""
    with pytest.raises(la.AdapterError, match="DERIVATION LOCALITY"):
        la.get_adapter("alphasteer")(
            {"kind": "alphasteer", "positions": "response"},
            _official_alphasteer(), b"", header_ids=HEADER)


def test_cosine_gated_r_refuses_a_response_override_too():
    g = torch.Generator().manual_seed(11)
    art = _blob({"kind": "cosine_gated_r", "layer": 16,
                 "r": torch.randn(D, generator=g),
                 "d_cond": torch.randn(D, generator=g),
                 "theta": 0.1, "base_scale": 1.0, "metadata": {}})
    with pytest.raises(la.AdapterError, match="DERIVATION LOCALITY"):
        la.get_adapter("cosine_gated_r")(
            {"kind": "cosine_gated_r", "positions": "response"},
            art, b"", header_ids=HEADER)
