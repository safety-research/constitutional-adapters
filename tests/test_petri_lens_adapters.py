"""Real per-kind lens adapters: equivalence
against the AUTHORITATIVE object classes, CPU-only.

The load-bearing tests here are the per-kind EQUIVALENCE tests: each
builds a synthetic artifact through the REAL object class's own ``save``
(never a hand-rolled dict), adapts the exact bytes into hook ops, runs a
REAL tiny-model forward (prefill + one KV-cached decode step) under the
object's own ``apply(model, scale)``, captures every decoder layer's
hidden states immediately BEFORE and AFTER the object's hooks, and
asserts ``apply_hook_ops`` maps each pre-tensor to the post-tensor
BIT-EXACTLY. Ops that only check shapes prove nothing; this proves the
translation IS the object's transformation.

Also pinned here, per the shared decode-only contract with the Evals
steer-generated-only lane: a phase-restricted op is NOT INVOKED outside
its phase — under a decode-only GATED op the prefill's hidden states
(and hence the prompt KV) are byte-identical to the unhooked base
forward. The gated case is the one that matters: a zero-scale-but-hooked
prefill coincides with unhooked for pure additive ops, so an additive
test would pass either way and prove nothing.

All fixtures are synthetic: a from-config tiny Llama (no hub access),
generator-seeded random tensors, invented ids.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from sisteer.capeval import manifest as cm
from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls

ROOT = Path(__file__).parents[1]

HIDDEN = 32
N_LAYERS = 4
PROMPT_LEN = 7


# --------------------------------------------------------------------------
# tiny model + the pre/post capture harness
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=97, max_position_embeddings=64,
        tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def _prompt_batch(vocab=97, seed=7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, vocab, (1, PROMPT_LEN), generator=g)
    mask = torch.ones(1, PROMPT_LEN, dtype=torch.long)
    next_id = torch.randint(3, vocab, (1, 1), generator=g)
    next_mask = torch.ones(1, PROMPT_LEN + 1, dtype=torch.long)
    return ids, mask, next_id, next_mask


@contextmanager
def _null_ctx():
    yield


def _run_pre_post(model, obj_apply_ctx):
    """Run prefill + one KV-cached decode step; capture every decoder
    layer's output hidden states immediately BEFORE (pre) and AFTER
    (post) the object's hooks: forward hooks fire in registration order,
    so a hook registered before the object's sees the block output the
    object's hook consumes, and one registered after sees what it
    returned. Returns ({layer: [prefill, decode]} x2)."""
    from sisteer.models import get_decoder_layers

    decoder = get_decoder_layers(model)
    pre = {l: [] for l in range(len(decoder))}
    post = {l: [] for l in range(len(decoder))}

    def mk(store, l):
        def hook(_mod, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[l].append(h.detach().clone())
        return hook

    ids, mask, next_id, next_mask = _prompt_batch()
    pre_handles = [decoder[l].register_forward_hook(mk(pre, l))
                   for l in range(len(decoder))]
    try:
        with obj_apply_ctx:
            post_handles = [decoder[l].register_forward_hook(mk(post, l))
                            for l in range(len(decoder))]
            try:
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=mask,
                                use_cache=True)
                    model(input_ids=next_id, attention_mask=next_mask,
                          past_key_values=out.past_key_values,
                          use_cache=True)
            finally:
                for h in post_handles:
                    h.remove()
    finally:
        for h in pre_handles:
            h.remove()
    return pre, post


PREFILL_CTX = la.HookContext(la.PHASE_PREFILL, tuple(range(PROMPT_LEN)))
DECODE_CTX = la.HookContext(la.PHASE_DECODE, (PROMPT_LEN,))


def _assert_equivalent(pre, post, ops, op_layers):
    """apply_hook_ops on each captured pre-tensor must equal the object's
    own post-tensor BIT-EXACTLY, at every op layer, both phases; layers
    without ops must be untouched by the object."""
    for l in range(N_LAYERS):
        for phase_idx, ctx in ((0, PREFILL_CTX), (1, DECODE_CTX)):
            mine = la.apply_hook_ops(pre[l][phase_idx], ops, 1.0, layer=l,
                                     ctx=ctx)
            assert torch.equal(mine, post[l][phase_idx]), (
                f"layer {l} phase {ctx.phase}: adapter ops do not "
                "reproduce the object's own transformation")
            if l not in op_layers:
                assert mine is pre[l][phase_idx]  # untouched, same tensor


# --------------------------------------------------------------------------
# per-kind equivalence: psr / vector_all / antidote / alphasteer / loreft
# --------------------------------------------------------------------------


def test_psr_adapter_reproduces_psrobject_apply(tiny_model, tmp_path):
    """psr: relu-gated per-layer vector, fp32 gate on the pre-steering
    hidden state, EVERY position (BOS included), prefill AND decode —
    PSRObject's own deployment, multi-layer (the psr_*_all arms)."""
    from sisteer.extraction.psr import PSRObject

    layers = [0, 2, 3]
    obj = PSRObject(tiny_model, layers, train_direction=True)
    g = torch.Generator().manual_seed(11)
    for l in layers:
        obj.directions[l].data = torch.randn(HIDDEN, generator=g) * 0.05
        obj.gate_w[l].data = torch.randn(HIDDEN, generator=g) * 0.3
        # a mix of dead and live relu gates across tokens
        obj.gate_b[l].data = torch.randn(1, generator=g) * 0.5
    path = tmp_path / "object.pt"
    obj.save(path)

    ops = la.get_adapter("psr")(
        {"kind": "psr", "layers": layers, "train_direction": True},
        path.read_bytes(), b"")
    assert [op.layer for op in ops] == layers
    assert all(op.op == la.OP_RELU_GATED_VECTOR
               and op.phase == la.PHASE_BOTH and not op.skip_bos
               for op in ops)

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=1.0))
    # non-vacuous: the gate actually fired somewhere
    assert any(not torch.equal(pre[l][0], post[l][0]) for l in layers)
    _assert_equivalent(pre, post, ops, set(layers))


def test_vector_all_adapter_reproduces_multilayer_vector_apply(
        tiny_model, tmp_path):
    """vector_all: one additive vector per layer through hooks.steering —
    positions 'all', skip_bos True (the stored default), both phases.
    The BOS row must come back untouched during the prefill."""
    from sisteer.extraction.multilayer import MultiLayerVectorObject

    obj = MultiLayerVectorObject(tiny_model)  # all layers, skip_bos=True
    g = torch.Generator().manual_seed(13)
    for l in obj.layers:
        obj.params[l].data = torch.randn(HIDDEN, generator=g) * 0.05
    path = tmp_path / "object.pt"
    obj.save(path)

    ops = la.get_adapter("vector_all")(
        {"kind": "vector_all", "positions": "all", "skip_bos": True},
        path.read_bytes(), b"")
    assert [op.layer for op in ops] == list(range(N_LAYERS))
    assert all(op.op == la.OP_ADD_VECTOR and op.skip_bos for op in ops)

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=1.0))
    assert any(not torch.equal(pre[l][0], post[l][0]) for l in obj.layers)
    _assert_equivalent(pre, post, ops, set(obj.layers))
    # the BOS carve-out did real work: row 0 untouched, row 1 steered
    l = obj.layers[0]
    assert torch.equal(pre[l][0][:, 0, :], post[l][0][:, 0, :])
    assert not torch.equal(pre[l][0][:, 1, :], post[l][0][:, 1, :])


def test_antidote_adapter_reproduces_antidote_apply(tiny_model, tmp_path):
    """antidote: positions='last' (final prompt token at prefill, every
    decode step's single position), ABSOLUTE base_alpha baked as the op
    coeff — AntidoteObject.apply passes base_alpha * scale to
    hooks.steering."""
    from sisteer.extraction.antidote import AntidoteObject
    from sisteer.vectors import SteeringVector

    g = torch.Generator().manual_seed(17)
    sv = SteeringVector(
        vectors={2: torch.randn(HIDDEN, generator=g) * 0.05},
        method="antidote", model_name="TESTMODEL",
        instruction_id="synthetic",
        metadata={"base_alpha": 3.5})
    obj = AntidoteObject(sv)
    path = tmp_path / "object.safetensors"
    sv.save(path)
    artifact = path.read_bytes()
    sidecar = path.with_suffix(".json").read_bytes()

    ops = la.get_adapter("antidote")(
        {"kind": "antidote", "method": "antidote"}, artifact, sidecar)
    (op,) = ops
    assert op.op == la.OP_ADD_VECTOR and op.layer == 2
    assert op.positions == la.POSITIONS_LAST and not op.skip_bos
    assert op.coeff == 3.5  # base_alpha baked into the op, not the selector

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=1.0))
    assert not torch.equal(pre[2][0], post[2][0])
    _assert_equivalent(pre, post, ops, {2})
    # 'last' means LAST: at prefill only the final prompt row moved
    assert torch.equal(pre[2][0][:, :-1, :], post[2][0][:, :-1, :])
    assert not torch.equal(pre[2][0][:, -1, :], post[2][0][:, -1, :])
    # ...and the decode step's single position IS steered
    assert not torch.equal(pre[2][1], post[2][1])


def test_alphasteer_adapter_reproduces_alphasteer_apply(
        tiny_model, tmp_path):
    """alphasteer: PREFILL ONLY — base_scale * (h_last @ M) broadcast to
    every prompt position in the model dtype; decode steps untouched
    (influenced only through the steered prompt KV)."""
    from sisteer.extraction.alphasteer import AlphaSteerObject

    g = torch.Generator().manual_seed(19)
    obj = AlphaSteerObject(
        tiny_model, 1, torch.randn(HIDDEN, HIDDEN, generator=g) * 0.02,
        base_scale=0.7, refusal=torch.randn(HIDDEN, generator=g),
        metadata={"synthetic": True})
    path = tmp_path / "object.pt"
    obj.save(path)

    ops = la.get_adapter("alphasteer")(
        {"kind": "alphasteer", "layer": 1, "base_scale": 0.7},
        path.read_bytes(), b"")
    (op,) = ops
    assert op.op == la.OP_PREFILL_LAST_MATMUL and op.layer == 1
    assert op.phase == la.PHASE_PREFILL and op.coeff == 0.7

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=1.0))
    assert not torch.equal(pre[1][0], post[1][0])   # prefill steered
    assert torch.equal(pre[1][1], post[1][1])       # decode untouched
    _assert_equivalent(pre, post, ops, {1})
    # decode-phase application is NOT INVOKED: the same tensor comes back
    out = la.apply_hook_ops(pre[1][1], ops, 1.0, layer=1, ctx=DECODE_CTX)
    assert out is pre[1][1]


def test_loreft_adapter_reproduces_loreft_apply(tiny_model, tmp_path):
    """loreft: the fp32 low-rank affine subspace edit at the stored
    layer, PROMPT POSITIONS ONLY (decode passes through) — R materialized
    from the saved orthogonal-parametrization state must equal the
    object's own rot.weight bit-exactly."""
    from sisteer.extraction.reft import LoReFTObject

    torch.manual_seed(23)
    obj = LoReFTObject(tiny_model, [2], rank=4)
    g = torch.Generator().manual_seed(29)
    with torch.no_grad():  # perturb: at init W=R, b=0 makes a no-op object
        obj.learned[2].weight.add_(
            torch.randn(4, HIDDEN, generator=g) * 0.2)
        obj.learned[2].bias.add_(torch.randn(4, generator=g) * 0.2)
    path = tmp_path / "object.pt"
    obj.save(path)

    ops = la.get_adapter("loreft")(
        {"kind": "loreft", "layers": [2], "rank": 4},
        path.read_bytes(), b"")
    (op,) = ops
    assert op.op == la.OP_AFFINE_SUBSPACE and op.layer == 2
    assert op.phase == la.PHASE_PREFILL
    # the parametrization rebuild IS the object's materialized R
    assert torch.equal(op.rotate, obj.rotate[2].weight.detach())

    pre, post = _run_pre_post(tiny_model, obj.apply(tiny_model, scale=1.0))
    assert not torch.equal(pre[2][0], post[2][0])   # prefill steered
    assert torch.equal(pre[2][1], post[2][1])       # decode untouched
    _assert_equivalent(pre, post, ops, {2})


def test_loreft_adapter_is_rng_pure(tiny_model, tmp_path):
    """The orthogonal-parametrization rebuild constructs an nn.Linear,
    whose init draws RNG — the adapter must fork it: same inputs, same
    ops, and NO global generator side effect (the adapter contract)."""
    from sisteer.extraction.reft import LoReFTObject

    torch.manual_seed(31)
    obj = LoReFTObject(tiny_model, [1], rank=2)
    path = tmp_path / "object.pt"
    obj.save(path)
    blob = path.read_bytes()
    spec = {"kind": "loreft", "layers": [1], "rank": 2}

    state_before = torch.get_rng_state()
    ops1 = la.get_adapter("loreft")(spec, blob, b"")
    assert torch.equal(state_before, torch.get_rng_state())
    ops2 = la.get_adapter("loreft")(spec, blob, b"")
    for a, b in zip(ops1, ops2):
        assert torch.equal(a.rotate, b.rotate)
        assert torch.equal(a.affine_w, b.affine_w)
        assert torch.equal(a.affine_b, b.affine_b)


# --------------------------------------------------------------------------
# decode-only semantics (shared with the Evals steer-generated-only lane):
# NOT INVOKED at prompt positions — never a zero-scale hooked application.
# Pinned on a GATED op: the additive case coincides with unhooked either
# way and would prove nothing.
# --------------------------------------------------------------------------


def _decode_only_gated_op(seed=37):
    g = torch.Generator().manual_seed(seed)
    return la.HookOp(
        op=la.OP_RELU_GATED_VECTOR, layer=1,
        vector=torch.randn(HIDDEN, generator=g) * 0.1,
        # a small nonzero w keeps the gate READING h (the property the
        # decode-only pin needs) while the +5 bias keeps relu(w.h + b)
        # strictly positive for any realistic h, so the decode-fires
        # assertion below is deterministic, never a lucky draw
        gate_w=torch.randn(HIDDEN, generator=g) * 0.01,
        gate_b=torch.full((1,), 5.0),
        phase=la.PHASE_DECODE)


def test_decode_only_gated_op_never_touches_the_prefill():
    op = _decode_only_gated_op()
    g = torch.Generator().manual_seed(41)
    hidden = torch.randn(1, PROMPT_LEN, HIDDEN, generator=g)
    # prefill: the SAME tensor comes back — not a numerically-zero copy
    out = la.apply_hook_ops(hidden, (op,), 1.0, layer=1, ctx=PREFILL_CTX)
    assert out is hidden
    # decode: the op fires (the gate is live), so this is non-vacuous
    dec = torch.randn(1, 1, HIDDEN, generator=g)
    out_dec = la.apply_hook_ops(dec, (op,), 1.0, layer=1, ctx=DECODE_CTX)
    assert not torch.equal(out_dec, dec)


def test_decode_only_gated_op_prompt_kv_is_the_base_models(tiny_model):
    """Full-forward pin: with a decode-only GATED op hooked through a
    real forward, every prompt-position hidden state and the prefill
    logits are BYTE-IDENTICAL to the unhooked base forward, and a decode
    step over the hooked prefill's KV cache (op disabled at decode)
    equals the fully-unhooked decode — i.e. the prompt KV is the BASE
    model's bytes. With the op live at decode, the decode step differs
    (steered decode over unsteered prompt KV)."""
    from sisteer.models import get_decoder_layers

    op = _decode_only_gated_op()
    decoder = get_decoder_layers(tiny_model)
    ids, mask, next_id, next_mask = _prompt_batch()

    def run(hook_prefill: bool, hook_decode: bool):
        captured = {l: [] for l in range(N_LAYERS)}
        state = {"seen": 0, "arm": True}

        def cap(l):
            def hook(_mod, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[l].append(h.detach().clone())
                if l != op.layer or not state["arm"]:
                    return None
                S = h.shape[1]
                if S > 1:
                    ctx = la.HookContext(la.PHASE_PREFILL, tuple(range(S)))
                else:
                    ctx = la.HookContext(la.PHASE_DECODE, (state["seen"],))
                new = la.apply_hook_ops(h, (op,), 1.0, layer=l, ctx=ctx)
                if new is h:
                    return None
                if isinstance(output, tuple):
                    return (new,) + output[1:]
                return new
            return hook

        handles = [decoder[l].register_forward_hook(cap(l))
                   for l in range(N_LAYERS)]
        try:
            with torch.no_grad():
                state["arm"] = hook_prefill
                out = tiny_model(input_ids=ids, attention_mask=mask,
                                 use_cache=True)
                state["seen"] = ids.shape[1]
                state["arm"] = hook_decode
                dec = tiny_model(input_ids=next_id,
                                 attention_mask=next_mask,
                                 past_key_values=out.past_key_values,
                                 use_cache=True)
        finally:
            for h in handles:
                h.remove()
        return captured, out.logits.detach().clone(), \
            dec.logits.detach().clone()

    base_cap, base_logits, base_dec = run(False, False)
    hook_cap, hook_logits, unhooked_dec_over_hooked_kv = run(True, False)
    # prompt-position hidden states: byte-identical to the unhooked base
    for l in range(N_LAYERS):
        assert torch.equal(base_cap[l][0], hook_cap[l][0])
    assert torch.equal(base_logits, hook_logits)
    # decode over the hooked-prefill KV, op off: byte-identical decode —
    # the prompt KV holds the base model's bytes
    assert torch.equal(base_dec, unhooked_dec_over_hooked_kv)
    # with the op live at decode, decode diverges (non-vacuous evidence)
    _, _, steered_dec = run(True, True)
    assert not torch.equal(base_dec, steered_dec)


# --------------------------------------------------------------------------
# vocabulary + context validation, materialization policy, refusals
# --------------------------------------------------------------------------


def _vec_op(**over):
    kw = dict(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN))
    kw.update(over)
    return la.HookOp(**kw)


def test_hookop_field_validation():
    with pytest.raises(la.AdapterError, match="unknown hook op"):
        la.HookOp(op="rewrite_everything", layer=0,
                  vector=torch.ones(4))
    with pytest.raises(la.AdapterError, match="does not take field"):
        _vec_op(gate_w=torch.ones(HIDDEN))  # add_vector has no gate
    with pytest.raises(la.AdapterError, match="requires tensor field"):
        la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0,
                  vector=torch.ones(4), gate_w=torch.ones(4))  # no gate_b
    with pytest.raises(la.AdapterError, match="prefill only"):
        la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=0,
                  matrix=torch.eye(4), phase=la.PHASE_BOTH)
    with pytest.raises(la.AdapterError, match="square"):
        la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=0,
                  matrix=torch.ones(2, 4), phase=la.PHASE_PREFILL)
    with pytest.raises(la.AdapterError, match="finite"):
        _vec_op(coeff=float("inf"))
    with pytest.raises(la.AdapterError, match="phase"):
        _vec_op(phase="sometimes")
    with pytest.raises(la.AdapterError, match="positions"):
        _vec_op(positions="every_other")


def test_hookcontext_validation():
    with pytest.raises(la.AdapterError, match="COMPLETE prompt"):
        la.HookContext(la.PHASE_PREFILL, (0, 2, 3))  # a chunk, not a prompt
    with pytest.raises(la.AdapterError, match="COMPLETE prompt"):
        la.HookContext(la.PHASE_PREFILL, (1, 2))     # does not start at 0
    with pytest.raises(la.AdapterError, match="exactly ONE"):
        la.HookContext(la.PHASE_DECODE, (7, 8))
    with pytest.raises(la.AdapterError, match=">= 1"):
        la.HookContext(la.PHASE_DECODE, (0,))
    with pytest.raises(la.AdapterError, match="non-empty"):
        la.HookContext(la.PHASE_PREFILL, ())
    with pytest.raises(la.AdapterError, match="phase"):
        la.HookContext(la.PHASE_BOTH, (0,))  # a context has a real phase


def test_position_dependent_ops_refuse_without_context():
    hidden = torch.randn(1, 4, HIDDEN)
    for op in (_vec_op(skip_bos=True),
               _vec_op(positions=la.POSITIONS_LAST),
               _vec_op(phase=la.PHASE_DECODE)):
        with pytest.raises(la.AdapterError, match="HookContext"):
            la.apply_hook_ops(hidden, (op,), 1.0, layer=1)


def test_context_row_count_must_match_hidden():
    hidden = torch.randn(1, 4, HIDDEN)
    ctx = la.HookContext(la.PHASE_PREFILL, tuple(range(5)))
    with pytest.raises(la.AdapterError, match="rows"):
        la.apply_hook_ops(hidden, (_vec_op(skip_bos=True),), 1.0,
                          layer=1, ctx=ctx)


def test_scale_zero_and_phase_skip_return_the_same_tensor():
    g = torch.Generator().manual_seed(43)
    hidden = torch.randn(1, 3, HIDDEN, generator=g)
    ctx = la.HookContext(la.PHASE_PREFILL, (0, 1, 2))
    gated = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0,
                      vector=torch.ones(HIDDEN),
                      gate_w=torch.ones(HIDDEN), gate_b=torch.ones(1))
    affine = la.HookOp(op=la.OP_AFFINE_SUBSPACE, layer=0,
                       rotate=torch.randn(2, HIDDEN, generator=g),
                       affine_w=torch.randn(2, HIDDEN, generator=g),
                       affine_b=torch.zeros(2), phase=la.PHASE_DECODE)
    matmul = la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=0,
                       matrix=torch.eye(HIDDEN), phase=la.PHASE_PREFILL)
    for ops in ((_vec_op(layer=0),), (gated,), (affine,), (matmul,)):
        assert la.apply_hook_ops(hidden, ops, 0.0, layer=0,
                                 ctx=ctx) is hidden
    # a decode-only op at prefill / a prefill-only op at decode: NOT
    # INVOKED — the same tensor, never a numerically-zero application
    assert la.apply_hook_ops(hidden, (affine,), 1.0, layer=0,
                             ctx=ctx) is hidden
    dec = torch.randn(1, 1, HIDDEN, generator=g)
    dctx = la.HookContext(la.PHASE_DECODE, (3,))
    assert la.apply_hook_ops(dec, (matmul,), 1.0, layer=0,
                             ctx=dctx) is dec


def test_fully_masked_op_returns_the_same_tensor():
    # a 1-token prompt whose only row is the BOS: skip_bos empties the
    # mask; untouched means the SAME tensor, not an added zero
    hidden = torch.randn(1, 1, HIDDEN)
    ctx = la.HookContext(la.PHASE_PREFILL, (0,))
    out = la.apply_hook_ops(hidden, (_vec_op(skip_bos=True),), 1.0,
                            layer=1, ctx=ctx)
    assert out is hidden


def test_bos_skip_is_orthogonal_to_phase():
    """BOS-skip masks positions WITHIN the hooked region; phase decides
    invocation — the flags never collapse (the shared Evals contract)."""
    g = torch.Generator().manual_seed(47)
    hidden = torch.randn(1, 3, HIDDEN, generator=g)
    ctx = la.HookContext(la.PHASE_PREFILL, (0, 1, 2))
    both = _vec_op(skip_bos=True, phase=la.PHASE_BOTH)
    out = la.apply_hook_ops(hidden, (both,), 1.0, layer=1, ctx=ctx)
    assert torch.equal(out[:, 0, :], hidden[:, 0, :])       # BOS masked
    assert not torch.equal(out[:, 1:, :], hidden[:, 1:, :])  # rest hooked
    decode_only = _vec_op(skip_bos=True, phase=la.PHASE_DECODE)
    assert la.apply_hook_ops(hidden, (decode_only,), 1.0, layer=1,
                             ctx=ctx) is hidden               # not invoked


def test_materialize_dtype_policy_keeps_fp32_math_fp32():
    g = torch.Generator().manual_seed(53)
    ops = (
        _vec_op(),
        la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0,
                  vector=torch.randn(HIDDEN, generator=g),
                  gate_w=torch.randn(HIDDEN, generator=g),
                  gate_b=torch.ones(1)),
        la.HookOp(op=la.OP_AFFINE_SUBSPACE, layer=0,
                  rotate=torch.randn(2, HIDDEN, generator=g),
                  affine_w=torch.randn(2, HIDDEN, generator=g),
                  affine_b=torch.zeros(2), phase=la.PHASE_PREFILL),
        la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=0,
                  matrix=torch.eye(HIDDEN), phase=la.PHASE_PREFILL),
    )
    mat = la.materialize_ops(ops, device="cpu", dtype=torch.bfloat16)
    assert mat[0].vector.dtype == torch.bfloat16          # model-dtype add
    assert mat[1].vector.dtype == torch.float32           # fp32 gate math
    assert mat[1].gate_w.dtype == torch.float32
    assert mat[2].rotate.dtype == torch.float32           # fp32 subspace
    assert mat[3].matrix.dtype == torch.bfloat16          # model-dtype mm
    # and the fp32-math ops ACCEPT a bf16 hidden state after materialize
    hidden = torch.randn(1, 2, HIDDEN, generator=g).to(torch.bfloat16)
    ctx = la.HookContext(la.PHASE_PREFILL, (0, 1))
    out = la.apply_hook_ops(hidden, (mat[1],), 1.0, layer=0, ctx=ctx)
    assert out.dtype == torch.bfloat16 and not torch.equal(out, hidden)
    # while an UN-materialized model-dtype op refuses a bf16 hidden
    with pytest.raises(la.AdapterError, match="materialize"):
        la.apply_hook_ops(hidden, (ops[0],), 1.0, layer=ops[0].layer)


def test_unservable_kinds_refuse_with_the_exact_reason():
    # "lora" left this table 2026-08-28: the o_proj module site
    # (HOOK_SITE_OPROJ) landed and the kind is served — see
    # tests/test_lora_lens_serving.py for its adapter round-trip.
    for kind, phrase in (("circuit_breaker", "seven"),
                         ("token", "input-embedding"),
                         ("lofit", "o_proj"),
                         ("cast", "LATCH")):
        with pytest.raises(la.AdapterError, match="no lens adapter") as err:
            la.get_adapter(kind)
        assert "UNSERVABLE" in str(err.value)
        assert phrase in str(err.value), (kind, str(err.value))
    # and registering an adapter for a documented-unservable kind refuses
    with pytest.raises(la.AdapterError, match="UNSERVABLE"):
        la.register_adapter("cast", lambda s, a, b: ())


def test_every_petri_matrix_kind_is_adapted_or_documented_unservable():
    from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS

    covered = set(la.registered_kinds()) | set(la.UNSERVABLE_KIND_REASONS)
    # PENDING (2026-08-28, cross-branch seam — remove at the merge):
    # "lora_all" (the lora_r4 cells) joined the petri matrix vocabulary on
    # THIS branch while its lens adapter lands concurrently on the
    # lora-serving lane (branch loralens, "Serve LoRA on the vLLM lens
    # path via the o_proj module site", 764a3d0 + hardening). Until the
    # two merge, lora_all is exempted BY NAME here — every other matrix
    # kind stays under the guard. SELF-RETIRING: once the adapter
    # registers the kind, the subtraction below makes this exemption dead
    # code — delete these lines at the merge rather than leaving them.
    pending = {"lora_all"} - covered
    missing = set(ARM_OBJECT_KINDS.values()) - covered - pending
    assert not missing, (
        f"petri matrix kinds with neither an adapter nor a documented "
        f"unservable reason: {sorted(missing)}")
    # no kind is both adapted and documented-unservable
    assert not set(la.registered_kinds()) & set(la.UNSERVABLE_KIND_REASONS)


def test_adapter_kind_strings_match_the_real_classes():
    from sisteer.extraction.alphasteer import AlphaSteerObject
    from sisteer.extraction.antidote import AntidoteObject
    from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                               MultiLayerLoRAObject,
                                               MultiLayerVectorObject)
    from sisteer.extraction.psr import PSRObject
    from sisteer.extraction.reft import LoReFTObject

    for cls in (PSRObject, MultiLayerVectorObject, AntidoteObject,
                AlphaSteerObject, LoReFTObject, MultiLayerLoRAObject,
                # the sdf-base-steering DERIVED difference object
                # (2026-09-06): its own kind string, its own adapter
                LoRASubtractionObject):
        assert cls.kind in la.registered_kinds(), cls.kind


def test_torch_blob_kinds_refuse_sidecars_and_wrong_kinds(tmp_path):
    import io as _io

    buf = _io.BytesIO()
    torch.save({"kind": "psr", "layers": [0],
                "directions": {"0": torch.ones(4)},
                "gate_w": {"0": torch.zeros(4)},
                "gate_b": {"0": torch.ones(1)}}, buf)
    blob = buf.getvalue()
    with pytest.raises(la.AdapterError, match="sidecar"):
        la.get_adapter("psr")({"kind": "psr"}, blob, b"stray")
    with pytest.raises(la.AdapterError, match="wrong adapter"):
        la.get_adapter("vector_all")({"kind": "vector_all"}, blob, b"")
    with pytest.raises(la.AdapterError, match="cross-check"):
        la.get_adapter("psr")({"kind": "psr", "layers": [5]}, blob, b"")


def _vector_all_blob(positions):
    import io as _io

    buf = _io.BytesIO()
    torch.save({"kind": "vector_all", "vectors": {"0": torch.ones(4)},
                "positions": positions, "skip_bos": True}, buf)
    return buf.getvalue()


def test_vector_all_refuses_unservable_position_modes():
    """The hooks.py modes with no op-vocabulary translation still ride
    the reference path — a silent 'all' substitution would serve a WIDER
    intervention than the one that was fit."""
    for mode in ("prompt", "front"):
        with pytest.raises(la.AdapterError, match="not lens-served"):
            la.get_adapter("vector_all")({"kind": "vector_all"},
                                         _vector_all_blob(mode), b"")


def test_vector_all_serves_the_response_mask_when_given_a_header():
    ops = la.get_adapter("vector_all")({"kind": "vector_all"},
                                       _vector_all_blob("response"), b"",
                                       header_ids=(900, 901, 902, 903))
    assert [o.positions for o in ops] == [la.POSITIONS_RESPONSE]
    assert ops[0].header_ids == (900, 901, 902, 903)


def test_response_mode_without_a_header_refuses_rather_than_widening():
    """A response-fit object served at all positions is a different,
    wider intervention — so a registry that supplied no header refuses
    instead of falling back."""
    with pytest.raises(la.AdapterError, match="no assistant header"):
        la.get_adapter("vector_all")({"kind": "vector_all"},
                                     _vector_all_blob("response"), b"")


# --------------------------------------------------------------------------
# psr: the served mask is the RECORDED mask (the scale-12 regression)
# --------------------------------------------------------------------------

PSR_HEADER = (900, 901, 902, 903)


def _psr_blob(tiny_model, tmp_path, fit_positions="response", header_len=4):
    """A REAL PSRObject.save blob (never a hand-rolled dict), fit-leg
    recorded per ``fit_positions``."""
    from sisteer.extraction.psr import PSRObject

    obj = PSRObject(tiny_model, [0, 2], train_direction=True)
    g = torch.Generator().manual_seed(23)
    for l in obj.layers:
        obj.directions[l].data = torch.randn(HIDDEN, generator=g) * 0.05
        obj.gate_w[l].data = torch.randn(HIDDEN, generator=g) * 0.3
        obj.gate_b[l].data = torch.randn(1, generator=g) * 0.5
    if fit_positions == "response":
        obj.set_fit_positions("response", header_len)
    path = tmp_path / "object.pt"
    obj.save(path)
    return path.read_bytes()


def test_psr_serves_the_recorded_response_mask(tiny_model, tmp_path):
    """THE scale-12 psr regression test. The wave's registries recorded
    positions="response" + response_header_len and supplied header ids;
    the adapter ignored both and emitted every-position ops — the
    measured fit=response/serve=all model-destroyer corner — for every
    fitted psr_kl condition of a completed campaign. The served ops must
    carry the recorded mask, exactly as recorded."""
    blob = _psr_blob(tiny_model, tmp_path)
    ops = la.get_adapter("psr")(
        {"kind": "psr", "layers": [0, 2], "positions": "response",
         "response_header_len": 4, "train_direction": True},
        blob, b"", header_ids=PSR_HEADER)
    assert [op.layer for op in ops] == [0, 2]
    assert all(op.positions == la.POSITIONS_RESPONSE for op in ops), (
        [op.positions for op in ops])
    assert all(op.header_ids == PSR_HEADER for op in ops)
    assert all(op.op == la.OP_RELU_GATED_VECTOR
               and op.phase == la.PHASE_BOTH and not op.skip_bos
               for op in ops)


def test_psr_fit_leg_is_the_default_when_the_spec_is_silent(
        tiny_model, tmp_path):
    """The blob's own recorded fit mask serves even when the resolved
    spec omits ``positions`` — the artifact is the default, never a
    silent 'all'."""
    blob = _psr_blob(tiny_model, tmp_path)
    ops = la.get_adapter("psr")(
        {"kind": "psr", "layers": [0, 2], "train_direction": True},
        blob, b"", header_ids=PSR_HEADER)
    assert all(op.positions == la.POSITIONS_RESPONSE for op in ops)
    assert all(op.header_ids == PSR_HEADER for op in ops)


def test_psr_cross_phase_cell_spec_overrides_the_all_fit(
        tiny_model, tmp_path):
    """A cross-phase cell IS an all-position artifact served under the
    response mask — the spec's positions overrides the fit default,
    exactly as for vector_all/static_r/lora (derivation locality holds:
    the gate reads each position's own hidden state)."""
    blob = _psr_blob(tiny_model, tmp_path, fit_positions="all")
    ops = la.get_adapter("psr")(
        {"kind": "psr", "layers": [0, 2], "positions": "response",
         "train_direction": True},
        blob, b"", header_ids=PSR_HEADER)
    assert all(op.positions == la.POSITIONS_RESPONSE for op in ops)
    assert all(op.header_ids == PSR_HEADER for op in ops)


def test_psr_response_mode_without_a_header_refuses(tiny_model, tmp_path):
    """No header ids -> REFUSE, never a silent widening to all
    positions. The silent widening is exactly what the campaign served;
    on the pre-fix adapter this test fails by returning all-position
    ops instead of raising."""
    blob = _psr_blob(tiny_model, tmp_path)
    with pytest.raises(la.AdapterError, match="no assistant header"):
        la.get_adapter("psr")(
            {"kind": "psr", "layers": [0, 2], "positions": "response",
             "response_header_len": 4, "train_direction": True},
            blob, b"")


def test_psr_response_header_width_mismatch_refuses(tiny_model, tmp_path):
    """The fit's recorded width and the served header must agree — a
    drifted chat template is a refusal, not a shrug (the lora kinds'
    rule, shared through _fit_leg_positions)."""
    blob = _psr_blob(tiny_model, tmp_path, header_len=3)
    with pytest.raises(la.AdapterError, match="DIFFERENT regions"):
        la.get_adapter("psr")(
            {"kind": "psr", "layers": [0, 2], "positions": "response",
             "train_direction": True},
            blob, b"", header_ids=PSR_HEADER)


def test_loreft_refuses_a_response_fit_blob_rather_than_dropping_the_mask(
        tiny_model, tmp_path):
    """loreft persists its fit leg too, and its lens translation is the
    class's prompt-positions (prefill-only) deployment — a response-fit
    blob must refuse loudly, never serve prefill-only under a
    decode-only label (the psr bug's shape on another kind)."""
    from sisteer.extraction.reft import LoReFTObject

    obj = LoReFTObject(tiny_model, [0], rank=2)
    obj.set_fit_positions("response", 4)
    path = tmp_path / "object.pt"
    obj.save(path)
    with pytest.raises(la.AdapterError, match="response-mask fit leg"):
        la.get_adapter("loreft")(
            {"kind": "loreft", "layers": [0], "rank": 2},
            path.read_bytes(), b"")


def test_registry_load_serves_psr_response_and_refuses_a_dropped_mask(
        tiny_model, tmp_path, monkeypatch):
    """End to end through the EXACT production path
    (export_worker_registry -> WorkerObjectRegistry.load): a psr cell
    whose spec records positions='response' loads into response-masked
    ops carrying the registry's header ids; and if an adapter ever drops
    the recorded mask again (simulated by rewiring the psr adapter to
    the pre-fix behavior), load REFUSES at startup instead of serving
    the widened intervention. This assertion exists because nothing
    stood between the recorded spec and the served ops while a whole
    campaign served the drop."""
    import dataclasses as _dc

    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.extraction.psr import PSRObject

    obj = PSRObject(tiny_model, [0, 2], train_direction=True)
    g = torch.Generator().manual_seed(31)
    for l in obj.layers:
        obj.directions[l].data = torch.randn(HIDDEN, generator=g) * 0.05
        obj.gate_w[l].data = torch.randn(HIDDEN, generator=g) * 0.3
        obj.gate_b[l].data = torch.randn(1, generator=g) * 0.5
    obj.set_fit_positions("response", len(PSR_HEADER))
    art = tmp_path / "object.pt"
    obj.save(art)

    spec = cm.PetriSpec(
        audit="petri-twosi-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=["none", "si_safety__psr_kl_all"],
        scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5",
        judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model="meta-llama/Llama-3.1-8B-Instruct",
        target_revision="0123456789abcdef0123456789abcdef01234567",
        serving_backend="proxy_vllm",
        serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enable-lora", "--enforce-eager"],
        chat_template_hash="12" * 8,
        tool_parser="llama3_json",
        serving_impl_commit="fedcba9876" * 4,
        prompted_prefix_hashes={
            "safety": cm.sha256_hex16(b"synthetic safety prefix"),
            "constitution": cm.sha256_hex16(b"synthetic constitution"),
        },
        bloom_options=cm.BloomOptions(
            max_turns=15, enable_prefill=False, enable_rollback=False,
            compaction=False, realism_filter=False, epochs=1, seed=0))
    entries = {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "si_safety__psr_kl_all": dc.RegistryEntry(
            cell_id="si_safety__psr_kl_all", kind="psr",
            si_id="safety", object_path=str(art), serving_path="lens",
            resolved_spec={"kind": "psr", "si_id": "safety",
                           "layers": [0, 2], "train_direction": True,
                           "positions": "response",
                           "response_header_len": len(PSR_HEADER)}),
    }
    source = dc.MappingCellSource(entries)
    cells = dc.enumerate_cells(
        spec, source,
        behavior_files={k: f"# behavior {k}\n".encode()
                        for k in cm.petri_expected_behavior_keys(spec)},
        prompt_files={"auditor": b"# synthetic auditor prompt bytes\n",
                      "judge": b"# synthetic judge prompt bytes\n"})
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir,
                              assistant_header_ids=PSR_HEADER)
    # a RE-EXPORT into the same dir is idempotent: the blobs are left 0440 and
    # a bare rewrite used to EACCES (a re-submitted proxy could never come up)
    before = {q.relative_to(reg_dir): q.read_bytes() for q in reg_dir.rglob("*") if q.is_file()}
    ls.export_worker_registry(cells, source, reg_dir,
                              assistant_header_ids=PSR_HEADER)
    after = {q.relative_to(reg_dir): q.read_bytes() for q in reg_dir.rglob("*") if q.is_file()}
    assert after == before

    reg = ls.WorkerObjectRegistry.load(reg_dir)
    loaded = {o.cell_id: o for o in reg.objects()}["si_safety__psr_kl_all"]
    assert loaded.ops and all(
        op.positions == la.POSITIONS_RESPONSE for op in loaded.ops), (
        [op.positions for op in loaded.ops])
    assert all(op.header_ids == PSR_HEADER for op in loaded.ops)

    # the invariant CAN fail: the pre-fix adapter (recorded mask dropped,
    # ops widened to every position) must be refused at STARTUP.
    real = la.get_adapter("psr")

    def dropped(rspec, artifact, sidecar=b"", *, header_ids=None,
                turn_end_ids=None):
        # the adapter contract's serving-side kwargs (header_ids, and since
        # the all-turns mask landed, turn_end_ids) — the fake mirrors the
        # real signature because the registry load passes both
        ops = real(rspec, artifact, sidecar, header_ids=header_ids,
                   turn_end_ids=turn_end_ids)
        return tuple(_dc.replace(op, positions=la.POSITIONS_ALL,
                                 header_ids=None) for op in ops)

    monkeypatch.setitem(la._ADAPTERS, "psr", dropped)
    with pytest.raises(ls.LensSteerError, match="dropped the recorded mask"):
        ls.WorkerObjectRegistry.load(reg_dir)


def test_antidote_refuses_uncalibrated_or_sidecarless_artifacts(tmp_path):
    from sisteer.vectors import SteeringVector

    sv = SteeringVector(vectors={0: torch.ones(4)}, method="antidote",
                        model_name="TESTMODEL", instruction_id="synthetic",
                        metadata={})  # NO base_alpha
    path = tmp_path / "object.safetensors"
    sv.save(path)
    artifact = path.read_bytes()
    sidecar = path.with_suffix(".json").read_bytes()
    with pytest.raises(la.AdapterError, match="base_alpha"):
        la.get_adapter("antidote")({"kind": "antidote"}, artifact, sidecar)
    with pytest.raises(la.AdapterError, match="sidecar"):
        la.get_adapter("antidote")({"kind": "antidote"}, artifact, b"")
    # a sidecar claiming another method refuses (load_object dispatch)
    other = json.loads(sidecar.decode("utf-8"))
    other["method"] = "vector"
    with pytest.raises(la.AdapterError, match="method"):
        la.get_adapter("antidote")(
            {"kind": "antidote"}, artifact,
            json.dumps(other).encode("utf-8"))


def test_registry_apply_threads_the_context(tmp_path, tiny_model):
    """WorkerObjectRegistry.apply passes ctx through to apply_hook_ops —
    a REAL position-dependent kind (vector_all, BOS-skipping) exported
    and preloaded through the registry refuses without a context and
    applies correctly with one."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.extraction.multilayer import MultiLayerVectorObject

    obj = MultiLayerVectorObject(tiny_model, layers=[0])
    g = torch.Generator().manual_seed(59)
    obj.params[0].data = torch.randn(HIDDEN, generator=g) * 0.1
    art = tmp_path / "object.pt"
    obj.save(art)

    spec = cm.PetriSpec(
        audit="petri-twosi-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=["none", "si_safety__vector_all"],
        scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5",
        judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model="meta-llama/Llama-3.1-8B-Instruct",
        target_revision="0123456789abcdef0123456789abcdef01234567",
        serving_backend="proxy_vllm",
        serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enable-lora", "--enforce-eager"],
        chat_template_hash="12" * 8,
        tool_parser="llama3_json",
        serving_impl_commit="fedcba9876" * 4,
        prompted_prefix_hashes={
            "safety": cm.sha256_hex16(b"synthetic safety prefix"),
            "constitution": cm.sha256_hex16(b"synthetic constitution"),
        },
        bloom_options=cm.BloomOptions(
            max_turns=15, enable_prefill=False, enable_rollback=False,
            compaction=False, realism_filter=False, epochs=1, seed=0))
    entries = {
        "none": dc.RegistryEntry(cell_id="none", kind="none"),
        "si_safety__vector_all": dc.RegistryEntry(
            cell_id="si_safety__vector_all", kind="vector_all",
            si_id="safety", object_path=str(art), serving_path="lens",
            resolved_spec={"kind": "vector_all", "si_id": "safety",
                           "positions": "all", "skip_bos": True}),
    }
    source = dc.MappingCellSource(entries)
    cells = dc.enumerate_cells(
        spec, source,
        behavior_files={k: f"# behavior {k}\n".encode()
                        for k in cm.petri_expected_behavior_keys(spec)},
        prompt_files={"auditor": b"# synthetic auditor prompt bytes\n",
                      "judge": b"# synthetic judge prompt bytes\n"})
    reg_dir = tmp_path / "registry"
    ls.export_worker_registry(cells, source, reg_dir)
    reg = ls.WorkerObjectRegistry.load(reg_dir)
    ihash = {c.cell_id: c for c in cells.canonical}[
        "si_safety__vector_all"].intervention_hash

    hidden = torch.randn(1, 3, HIDDEN, generator=g)
    with pytest.raises(la.AdapterError, match="HookContext"):
        reg.apply("si_safety__vector_all", ihash, hidden, layer=0)
    ctx = la.HookContext(la.PHASE_PREFILL, (0, 1, 2))
    out = reg.apply("si_safety__vector_all", ihash, hidden, layer=0,
                    ctx=ctx)
    assert torch.equal(out[:, 0, :], hidden[:, 0, :])  # BOS skipped
    assert not torch.equal(out[:, 1:, :], hidden[:, 1:, :])


# --------------------------------------------------------------------------
# the frozen chat-template hash (gap 3)
# --------------------------------------------------------------------------

MANIFEST = ROOT / "configs" / "capeval" / "manifests" / "petri_twosi.yaml"


def test_chat_template_hash_is_frozen():
    spec = cm.load_petri_spec(MANIFEST)
    assert spec.chat_template_hash != cm.PETRI_UNFROZEN
    assert spec.chat_template_hash == "b48c47f644389271"  # GEN-2 /
    # NousResearch @ d10aef79 (repointed 2026-08-21 on the confirmation
    # that the Class A re-pool is fitted on that mirror). meta-llama's
    # e10ca381b1ccc5cf was the GEN-1 pin and must NOT reappear here.
    assert spec.chat_template_hash != "e10ca381b1ccc5cf"


def test_chat_template_hash_matches_the_pinned_tokenizer_when_cached():
    """Provenance re-check wherever the pinned tokenizer snapshot is in
    the local HF cache (the gated repo cannot be fetched here): the
    frozen hash is sha256_hex16 over tokenizer_config.json's
    chat_template string bytes at target_revision — the exact bytes the
    launcher must write to the --chat-template file
    (build_server_launch hashes that file's bytes against the spec)."""
    spec = cm.load_petri_spec(MANIFEST)
    hub = Path(os.environ.get(
        "HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
    snap = (hub / f"models--{spec.target_model.replace('/', '--')}"
            / "snapshots" / spec.target_revision / "tokenizer_config.json")
    if not snap.is_file():
        pytest.skip("pinned tokenizer snapshot not in the local HF cache")
    template = json.loads(snap.read_text(encoding="utf-8"))["chat_template"]
    assert isinstance(template, str) and template
    got = hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]
    assert got == spec.chat_template_hash


# ---------------------------------------------------------------------------
# The gate must read the PRE-steering activation (2026-08-20)
# ---------------------------------------------------------------------------


def test_gated_op_gate_reads_the_pre_steering_hidden_not_the_accumulator():
    """A gated op's gate MUST be computed on the unmodified activation,
    even when another op at the same layer has already contributed.

    Correct in the code today (``_apply_one`` takes both ``out`` and
    ``hidden`` and gates on ``hidden``), but nothing pinned it — so a
    refactor that "simplified" the two-tensor signature to gate on the
    accumulator would silently change every stacked object's arithmetic.
    The peer lane whose object is exactly this shape (a per-layer,
    per-token learned gate across 32 layers) flagged the consequence: a
    gate fed an already-modified tensor makes a zero-scale hooked run
    diverge from an unhooked one, i.e. a silently different object.
    """
    torch.manual_seed(20260820)
    d, T = HIDDEN, 5
    hidden = torch.randn(T, d)

    # op A: a plain additive contribution at layer 0
    vec_a = torch.randn(d)
    op_a = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=vec_a, coeff=1.0)
    # op B: a gated contribution at the SAME layer
    gate_w, gate_b = torch.randn(d), torch.randn(1)
    vec_b = torch.randn(d)
    op_b = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=vec_b,
                     gate_w=gate_w, gate_b=gate_b, coeff=1.0)

    got = la.apply_hook_ops(hidden, (op_a, op_b), 1.0, layer=0)

    # expected: BOTH contributions, with B's gate read off the ORIGINAL
    g_pre = torch.relu(hidden.float() @ gate_w + gate_b)
    want = (hidden + vec_a
            + (g_pre.unsqueeze(-1) * vec_b.float()).to(hidden.dtype))
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    # and it must NOT equal the wrong reading (gate on the accumulated
    # tensor), which is what a collapsed signature would produce
    g_post = torch.relu((hidden + vec_a).float() @ gate_w + gate_b)
    wrong = (hidden + vec_a
             + (g_post.unsqueeze(-1) * vec_b.float()).to(hidden.dtype))
    assert not torch.equal(want, wrong), "fixture too weak to distinguish"
    assert not torch.equal(got, wrong)


# ---------------------------------------------------------------------------
# Degenerate shapes (2026-08-21): a guard or op derived from
# ONE reproduction must be checked against the degenerate cases of the same
# function before it is trusted. One tier0 bug had three shapes and the
# specials-only guard passed the observed one while missing an
# EMPTY assistant turn, whose scaffold carried no special id at all. These ops
# handle the degenerate shapes correctly today; nothing pinned that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [0, 1, 2])
def test_gated_op_on_degenerate_row_counts(rows):
    """T=0 (empty forward), T=1 (every decode step) and T=2 must all apply the
    gate elementwise without special-casing. T=1 is not exotic: it is the shape
    of EVERY KV-cached decode step, so a regression here would corrupt
    generation while leaving the prefill-shaped tests green."""
    torch.manual_seed(20260821)
    d = HIDDEN
    hidden = torch.randn(rows, d)
    gate_w, gate_b, vec = torch.randn(d), torch.randn(1), torch.randn(d)
    op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=vec,
                   gate_w=gate_w, gate_b=gate_b, coeff=1.0)

    got = la.apply_hook_ops(hidden, (op,), 1.0, layer=0)
    g = torch.relu(hidden.float() @ gate_w + gate_b)
    want = hidden + (g.unsqueeze(-1) * vec.float()).to(hidden.dtype)
    assert got.shape == hidden.shape == (rows, d)
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    # scale 0 stays bit-identical (the same tensor) at every row count —
    # Phase-0 assertion 8 needs exactness, not float-zero arithmetic
    assert la.apply_hook_ops(hidden, (op,), 0.0, layer=0) is hidden


def test_all_dead_gate_contributes_exactly_zero_not_approximately():
    """A gate that is closed on every row must leave the stream BIT-identical,
    not merely close. Real trained objects have such layers — the peer's psr
    artifacts are dead-closed at 26/28/30 — so this is the arithmetic the
    real-artifact gate relies on when it reports those layers as reproduced."""
    torch.manual_seed(7)
    d, T = HIDDEN, 5
    hidden = torch.randn(T, d)
    # a bias driving relu(.) to 0 for every row => dead gate
    op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=torch.randn(d),
                   gate_w=torch.zeros(d), gate_b=torch.tensor([-1.0]),
                   coeff=1.0)
    got = la.apply_hook_ops(hidden, (op,), 1.0, layer=0)
    assert torch.equal(got, hidden), "a dead gate must be an exact no-op"
