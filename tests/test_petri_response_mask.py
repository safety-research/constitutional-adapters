"""The cross-phase response mask (``positions="response"``) and the
linear-gated op — CPU-only, synthetic tensors and invented token ids.

The load-bearing property here is the one the cross-phase ruling exists
to enforce: THE FORWARD THAT EMITS THE FIRST GENERATED TOKEN IS STEERED.
Token 1 is sampled from the prefill forward, so a mechanism that leaves
the prefill unsteered and starts at the first decode step cannot shape
it. ``test_the_last_prefill_row_is_steered`` pins exactly that, and the
assistant-prefill tests pin the failure mode that would silently break
it while passing every width check.
"""

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la

D = 8
#: invented ids standing in for a 4-token assistant header
HEADER = (900, 901, 902, 903)


def _vec(seed: int, dim: int = D) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g, dtype=torch.float32)


def _hidden(rows: int, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, D, generator=g, dtype=torch.float32)


def _response_op(**kw) -> la.HookOp:
    kw.setdefault("vector", _vec(1))
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                     positions=la.POSITIONS_RESPONSE, header_ids=HEADER, **kw)


def _prefill_ctx(prompt_ids):
    ids = tuple(prompt_ids)
    return la.HookContext(phase=la.PHASE_PREFILL,
                          positions=tuple(range(len(ids))), token_ids=ids)


def _ordinary_prompt(n_content: int = 6):
    """A render ending at the assistant header, as every ordinary
    template does: content tokens, then the header."""
    return tuple(range(10, 10 + n_content)) + HEADER


# --- the mask itself ----------------------------------------------------------


def test_prefill_steers_exactly_the_header_rows():
    ids = _ordinary_prompt()
    P, H = len(ids), len(HEADER)
    op = _response_op()
    h = _hidden(P)
    out = la.apply_hook_ops(h, (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))
    touched = (out != h).any(dim=-1)
    assert touched.tolist() == [False] * (P - H) + [True] * H


def test_the_last_prefill_row_is_steered():
    """The whole point of the mechanism: the row whose logits produce the
    first generated token must be inside the mask."""
    ids = _ordinary_prompt()
    h = _hidden(len(ids))
    out = la.apply_hook_ops(h, (_response_op(),), 1.0, layer=0,
                            ctx=_prefill_ctx(ids))
    assert not torch.equal(out[-1], h[-1])


def test_every_decode_row_is_steered():
    op = _response_op()
    h = _hidden(1)
    ctx = la.HookContext(phase=la.PHASE_DECODE, positions=(12,),
                         token_ids=(4242,))
    out = la.apply_hook_ops(h, (op,), 1.0, layer=0, ctx=ctx)
    assert not torch.equal(out, h)


def test_a_header_only_prompt_steers_every_row():
    ids = HEADER
    h = _hidden(len(ids))
    out = la.apply_hook_ops(h, (_response_op(),), 1.0, layer=0,
                            ctx=_prefill_ctx(ids))
    assert bool((out != h).any(dim=-1).all())


def test_mask_arithmetic_matches_an_explicit_masked_add():
    ids = _ordinary_prompt()
    P, H = len(ids), len(HEADER)
    vec = _vec(3)
    op = _response_op(vector=vec, coeff=2.0)
    h = _hidden(P)
    out = la.apply_hook_ops(h, (op,), 0.5, layer=0, ctx=_prefill_ctx(ids))
    mask = torch.zeros(P, dtype=torch.float32)
    mask[P - H:] = 1.0
    assert torch.equal(out, h + (vec * (0.5 * 2.0)) * mask.unsqueeze(-1))


def test_scale_zero_is_bit_identical_under_the_response_mask():
    ids = _ordinary_prompt()
    h = _hidden(len(ids))
    out = la.apply_hook_ops(h, (_response_op(),), 0.0, layer=0,
                            ctx=_prefill_ctx(ids))
    assert out is h


def test_skip_bos_composes_with_the_response_mask():
    """BOS-skip is a position mask WITHIN the region, never a phase gate
    — so on a prompt that is nothing but the header it carves row 0 out
    of an otherwise fully-steered region."""
    ids = HEADER
    op = _response_op(skip_bos=True)
    h = _hidden(len(ids))
    out = la.apply_hook_ops(h, (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))
    touched = (out != h).any(dim=-1)
    assert touched.tolist() == [False, True, True, True]


# --- the refusals that keep the mask honest -----------------------------------


def test_assistant_prefill_refuses_rather_than_steering_the_seed():
    """Prefix forcing appends seeded assistant tokens AFTER the header.
    A width-only mask would steer the seed and leave the header — and so
    the token-1 forward — unsteered, while passing every width check."""
    ids = _ordinary_prompt() + (555, 556)
    with pytest.raises(la.AdapterError, match="not the assistant header"):
        la.apply_hook_ops(_hidden(len(ids)), (_response_op(),), 1.0, layer=0,
                          ctx=_prefill_ctx(ids))


def test_a_width_only_mask_would_have_silently_accepted_the_seeded_render():
    """Pins that the refusal above is doing real work: the seeded render
    has the same H and satisfies H < P, so every count-based check
    passes. Only matching the ids catches it."""
    ids = _ordinary_prompt() + (555, 556)
    P, H = len(ids), len(HEADER)
    assert 0 < H < P                      # a width check sees nothing wrong
    assert tuple(ids[P - H:]) != HEADER   # the ids do


def test_a_different_chat_template_refuses():
    ids = tuple(range(10, 16)) + (901, 902, 903, 904)
    with pytest.raises(la.AdapterError, match="not the assistant header"):
        la.apply_hook_ops(_hidden(len(ids)), (_response_op(),), 1.0, layer=0,
                          ctx=_prefill_ctx(ids))


def test_missing_token_ids_refuses_instead_of_falling_back():
    ids = _ordinary_prompt()
    ctx = la.HookContext(phase=la.PHASE_PREFILL,
                         positions=tuple(range(len(ids))))
    with pytest.raises(la.AdapterError, match="no token ids"):
        la.apply_hook_ops(_hidden(len(ids)), (_response_op(),), 1.0, layer=0,
                          ctx=ctx)


def test_a_prompt_shorter_than_its_own_header_refuses():
    ids = HEADER[:2]
    with pytest.raises(la.AdapterError, match="shorter than its own header"):
        la.apply_hook_ops(_hidden(len(ids)), (_response_op(),), 1.0, layer=0,
                          ctx=_prefill_ctx(ids))


def test_response_mode_requires_header_ids():
    with pytest.raises(la.AdapterError, match="requires header_ids"):
        la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1),
                  positions=la.POSITIONS_RESPONSE)


def test_a_collapsed_header_boundary_refuses():
    with pytest.raises(la.AdapterError, match="collapsed assistant-header"):
        la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1),
                  positions=la.POSITIONS_RESPONSE, header_ids=())


def test_header_ids_are_refused_outside_response_mode():
    with pytest.raises(la.AdapterError, match="does not take header_ids"):
        la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1),
                  positions=la.POSITIONS_ALL, header_ids=HEADER)


def test_header_ids_must_be_non_negative_ints():
    with pytest.raises(la.AdapterError, match="non-negative ints"):
        la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=_vec(1),
                  positions=la.POSITIONS_RESPONSE, header_ids=(1, -2))


def test_response_ops_need_a_context():
    assert la.op_needs_context(_response_op())
    with pytest.raises(la.AdapterError, match="needs a HookContext"):
        la.apply_hook_ops(_hidden(4), (_response_op(),), 1.0, layer=0)


def test_context_refuses_token_ids_misaligned_with_its_positions():
    with pytest.raises(la.AdapterError, match="token ids for"):
        la.HookContext(phase=la.PHASE_DECODE, positions=(5,),
                       token_ids=(1, 2))


# --- the linear-gated op (clamp_gate=False) -----------------------------------


def _gated_pair(**kw):
    """A relu-gated and a linear-gated op over identical tensors."""
    shared = dict(layer=0, vector=_vec(11), gate_w=_vec(12),
                  gate_b=torch.zeros(1, dtype=torch.float32), **kw)
    return (la.HookOp(op=la.OP_RELU_GATED_VECTOR, **shared),
            la.HookOp(op=la.OP_LINEAR_GATED_VECTOR, **shared))


def test_linear_gate_matches_the_hand_written_arithmetic():
    _, lin = _gated_pair()
    h = _hidden(5)
    out = la.apply_hook_ops(h, (lin,), 1.0, layer=0)
    g = h.float() @ lin.gate_w + lin.gate_b
    assert torch.equal(out, h + (g.unsqueeze(-1) * lin.vector).to(h.dtype))


def test_linear_gate_keeps_the_negative_half_the_relu_discards():
    """The whole reason this is a separate op rather than a flag: on a
    row whose projection is negative the two ops disagree, so serving one
    as the other is a different intervention, not a tolerance."""
    relu, lin = _gated_pair()
    h = _hidden(9)
    negative = (h.float() @ lin.gate_w + lin.gate_b) < 0
    assert bool(negative.any()), "fixture must exercise the negative half"
    out_relu = la.apply_hook_ops(h, (relu,), 1.0, layer=0)
    out_lin = la.apply_hook_ops(h, (lin,), 1.0, layer=0)
    assert torch.equal(out_relu[~negative], out_lin[~negative])
    assert not torch.equal(out_relu[negative], out_lin[negative])


def test_relu_gate_is_unchanged_by_the_refactor():
    relu, _ = _gated_pair()
    h = _hidden(5)
    out = la.apply_hook_ops(h, (relu,), 1.0, layer=0)
    g = torch.relu(h.float() @ relu.gate_w + relu.gate_b)
    assert torch.equal(out, h + (g.unsqueeze(-1) * relu.vector).to(h.dtype))


def test_linear_gate_is_bit_identical_at_scale_zero():
    _, lin = _gated_pair()
    h = _hidden(5)
    assert la.apply_hook_ops(h, (lin,), 0.0, layer=0) is h


def test_linear_gate_honors_the_response_mask():
    ids = _ordinary_prompt()
    P, H = len(ids), len(HEADER)
    _, lin = _gated_pair(positions=la.POSITIONS_RESPONSE, header_ids=HEADER)
    h = _hidden(P)
    out = la.apply_hook_ops(h, (lin,), 1.0, layer=0, ctx=_prefill_ctx(ids))
    touched = (out != h).any(dim=-1)
    assert touched.tolist() == [False] * (P - H) + [True] * H


def test_linear_gate_materializes_its_tensors_as_fp32():
    _, lin = _gated_pair()
    moved, = la.materialize_ops((lin,), device="cpu", dtype=torch.bfloat16)
    for name in ("vector", "gate_w", "gate_b"):
        assert getattr(moved, name).dtype is torch.float32


def test_linear_gate_shape_validation_mirrors_the_relu_op():
    with pytest.raises(la.AdapterError, match="gate_w must be 1-D"):
        la.HookOp(op=la.OP_LINEAR_GATED_VECTOR, layer=0, vector=_vec(1),
                  gate_w=_vec(1, dim=D + 1),
                  gate_b=torch.zeros(1, dtype=torch.float32))
