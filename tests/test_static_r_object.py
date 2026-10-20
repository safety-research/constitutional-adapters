"""native-psr-serving step 7 — ``extraction.static_r.StaticRObject``, the
vendored object class behind the served-only ``arditi`` arm, and its identity
with the native residual slot's CONSTANT write.

Pinned here:

- the class is registered in ``methods._KIND_LOADERS`` under kind ``static_r``,
  so ``load_object`` hydrates a real Arditi-shaped blob (before this the fp32
  serving oracle's ``arm_for_object`` raised "unknown intervention kind"), and
  ``save`` -> ``load_blob`` round-trips every deployment field;
- **the four ``steer_positions`` x ``site`` combinations**: what
  ``StaticRObject.apply`` writes into the residual stream is BIT-IDENTICAL to
  what ``native_residual.residual_slot_forward`` writes for the same object,
  at the boundary and position mode the exporter derives for it
  (``lora_export._residual_static_r``). That identity is the whole reason the
  class is vendored: it is the oracle's reference for a natively served Arditi
  cell, so the two paths must agree by construction and not by inspection;
- ``site='pre'`` at block 0 refuses — the embedding output has no
  preceding block to carry the edit — and the refusal comes from the export's
  kind helper, the same rule the lens applies;
- ``scale=0`` is an exact no-op (the arm's own contract), and a decode-shaped
  forward (seq_len 1) is never edited in either mode.

CPU-only, fp32, one tiny Llama; no snapshot, no HF cache, no vLLM."""

from __future__ import annotations

import pytest
import torch

from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_residual as nr
from sisteer.extraction.static_r import STATIC_R_STEER_POSITIONS, StaticRObject

HIDDEN = 32
N_LAYERS = 3
VOCAB = 97


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
                      max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


def _direction(seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(HIDDEN, generator=g)


def _obj(model, *, layer: int, steer: str, site: str, base_scale: float = 0.05,
         r: torch.Tensor | None = None) -> StaticRObject:
    return StaticRObject(model, layers=[layer], rs={layer: _direction() if r is None else r},
                         base_scale=base_scale, steer_positions=steer, site=site,
                         metadata={"per_layer_r_norm": {layer: float(_direction().norm())}})


def _blob(*, layer: int, steer: str, site: str, base_scale: float = 0.05) -> dict:
    r = _direction()
    return {"kind": "static_r", "layers": [layer], "rs": {layer: r}, "base_scale": base_scale,
            "steer_positions": steer, "site": site,
            "metadata": {"per_layer_r_norm": {layer: float(r.norm())}}}


def _run(model, ids: torch.Tensor):
    """``(every block's OUTPUT hidden state, the logits)`` for one forward.

    The capture hooks are registered LAST, so a steering hook registered before
    them has already replaced the output PyTorch hands on (``Module._call_impl``
    passes each hook's non-None result to the next)."""
    out: list = [None] * N_LAYERS
    handles = []

    def mk(i):
        def hook(_m, _a, output):
            h = output[0] if isinstance(output, tuple) else output
            out[i] = h.detach().clone()
        return hook

    for i, block in enumerate(model.model.layers):
        handles.append(block.register_forward_hook(mk(i)))
    try:
        with torch.no_grad():
            logits = model(ids).logits.detach().clone()
    finally:
        for h in handles:
            h.remove()
    return out, logits


def _slot_buffers(*, boundary: int, vector: torch.Tensor, mode: str, n_tokens: int,
                  prompt_len: int, response_start: int) -> nr.SlotBuffers:
    """ONE slot holding the object's constant write, in the worker's own buffer
    shapes — the exporter's (boundary, mode, coeff-folded vector), nothing else."""
    zeros_sl = torch.zeros(1, N_LAYERS, HIDDEN, dtype=torch.float32)
    vec = torch.zeros(1, N_LAYERS, HIDDEN, dtype=torch.float32)
    vec[0, boundary] = vector
    active = torch.zeros(1, N_LAYERS, dtype=torch.bool)
    active[0, boundary] = True
    return nr.SlotBuffers(
        slot_by_token=torch.zeros(n_tokens, dtype=torch.int64),
        active=active,
        gated=torch.zeros(1, dtype=torch.bool),
        skip_bos=torch.zeros(1, dtype=torch.bool),
        gate_w=zeros_sl, gate_b=torch.zeros(1, N_LAYERS, dtype=torch.float32),
        direction=zeros_sl.clone(), vector=vec,
        mode=torch.tensor([nr.MODE_CODES[mode]], dtype=torch.int8),
        prompt_len_by_token=torch.full((n_tokens,), prompt_len, dtype=torch.int64),
        response_start_by_token=torch.full((n_tokens,), response_start, dtype=torch.int64))


@pytest.mark.parametrize("steer", STATIC_R_STEER_POSITIONS)
@pytest.mark.parametrize("site", ["post", "pre"])
def test_apply_equals_the_constant_slots_write_on_every_combination(tiny_model, steer, site):
    """The object's hook and the slot's op write the SAME residual stream.

    The slot is configured from the EXPORTER's reading of the very blob the
    object saves — the boundary (one block earlier at ``site='pre'``), the
    position mode, and ``coeff = base_scale`` folded into the vector — so this
    is the full path an Arditi cell takes onto the native engine, compared
    against the class whose hook the lens and the oracle both reproduce."""
    layer = 2                               # >= 1 so the `pre` shift has a block to land on
    blob = _blob(layer=layer, steer=steer, site=site)
    parsed = lx._residual_static_r({"kind": "static_r"}, blob, hidden_size=HIDDEN)
    (boundary,) = parsed["boundaries"]
    assert boundary == (layer - 1 if site == "pre" else layer)   # the pre-site shift, or none
    assert parsed["mode"] == steer and parsed["coeff"] == pytest.approx(blob["base_scale"])
    payload_vec = parsed["tensors"][f"{nr.PAYLOAD_PREFIX}layers.{layer}.vector"]

    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (1, 7))
    prompt_len = ids.shape[1]

    obj = _obj(tiny_model, layer=layer, steer=steer, site=site, r=blob["rs"][layer],
               base_scale=blob["base_scale"])
    with obj.apply(tiny_model, scale=1.0):
        steered, steered_logits = _run(tiny_model, ids)

    # the slot's write, at the exporter's boundary, through the worker's own op. The
    # worker sees vLLM's flat [T, d] rows; one sequence keeps the comparison exact.
    # `response_start` is LOCATED (>= 0) even though neither static_r mode reads it: the op
    # fails closed on every RESTRICTED mode whose request has no located boundary — an
    # unlocated request is ledgered and left unsteered rather than guessed at
    # (native_residual.position_class_selection), and this forward is a located one.
    buffers = _slot_buffers(boundary=boundary, vector=payload_vec * parsed["coeff"],
                            mode=steer, n_tokens=prompt_len, prompt_len=prompt_len,
                            response_start=prompt_len)
    positions = torch.arange(prompt_len, dtype=torch.int64)

    def slot_hook(_m, _a, output):
        tup = isinstance(output, tuple)
        hidden = output[0] if tup else output
        written = nr.residual_slot_forward(hidden[0], None, positions, boundary,
                                           buffers).unsqueeze(0)
        return ((written,) + tuple(output[1:])) if tup else written

    handle = tiny_model.model.layers[boundary].register_forward_hook(slot_hook)
    try:
        slotted, slotted_logits = _run(tiny_model, ids)
    finally:
        handle.remove()

    # the two paths write at DIFFERENT sites of the same seam when site='pre' (the object
    # edits block l's INPUT, the slot block l-1's OUTPUT — the same tensor, one hook
    # earlier), so the comparison is everything DOWNSTREAM of the boundary plus the logits
    for i in range(boundary + 1, N_LAYERS):
        assert torch.equal(steered[i], slotted[i]), (
            f"steer={steer} site={site}: block {i} differs "
            f"(max |diff| {float((steered[i] - slotted[i]).abs().max())})")
    assert torch.equal(steered_logits, slotted_logits), (
        f"steer={steer} site={site}: logits differ "
        f"(max |diff| {float((steered_logits - slotted_logits).abs().max())})")
    if site == "post":      # same site: the boundary's own output must match too
        assert torch.equal(steered[boundary], slotted[boundary])
    # and the edit is REAL — a test that passed by writing nothing would be worthless
    bare, bare_logits = _run(tiny_model, ids)
    assert not torch.equal(bare_logits, steered_logits)


def test_pre_at_block_zero_refuses_and_post_at_block_zero_does_not(tiny_model):
    """``resid_pre`` of block 0 is the embedding output — there is no preceding
    block whose output could carry the edit, so the export refuses it by name (the
    lens's own rule). ``site='post'`` at block 0 is an ordinary block output."""
    with pytest.raises(lx.LoraExportError, match="resid_pre of layer 0"):
        lx._residual_static_r({"kind": "static_r"},
                              _blob(layer=0, steer="prefill_all", site="pre"), hidden_size=HIDDEN)
    parsed = lx._residual_static_r({"kind": "static_r"},
                                   _blob(layer=0, steer="prompt_last", site="post"),
                                   hidden_size=HIDDEN)
    assert parsed["boundaries"] == [0]


def test_scale_zero_is_an_exact_no_op_and_decode_forwards_are_never_edited(tiny_model):
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB, (1, 5))
    bare, _ = _run(tiny_model, ids)
    obj = _obj(tiny_model, layer=1, steer="prefill_all", site="pre")
    with obj.apply(tiny_model, scale=0.0):
        assert all(torch.equal(a, b) for a, b in zip(bare, _run(tiny_model, ids)[0]))
    # a decode-shaped forward (seq_len 1) is untouched in either mode: the edit reaches
    # generation through the steered prompt KV alone
    one = torch.randint(0, VOCAB, (1, 1))
    bare1, _ = _run(tiny_model, one)
    for steer, site in (("prefill_all", "pre"), ("prompt_last", "post")):
        with _obj(tiny_model, layer=1, steer=steer, site=site).apply(tiny_model, scale=1.0):
            assert all(torch.equal(a, b) for a, b in zip(bare1, _run(tiny_model, one)[0]))


def test_the_kind_is_registered_and_a_real_arditi_shaped_blob_hydrates(tiny_model, tmp_path):
    """``methods._KIND_LOADERS['static_r']`` is the vendored class, and the blob
    shape staged for arditi (int-keyed ``rs``, ``layers``, no legacy
    ``layer``/``r`` pair, ``prefill_all`` at ``pre``) hydrates through
    ``load_object`` — the path the fp32 oracle takes."""
    from sisteer.methods import _KIND_LOADERS, load_object

    assert _KIND_LOADERS["static_r"].__self__ is StaticRObject
    path = tmp_path / "object.pt"
    torch.save(_blob(layer=2, steer="prefill_all", site="pre", base_scale=1.0), path)
    obj = load_object(path, model=tiny_model, tok=None)
    assert isinstance(obj, StaticRObject)
    assert obj.layers == [2] and obj.steer_positions == "prefill_all" and obj.site == "pre"
    assert obj.base_scale == 1.0 and obj.position_convention == "prefill_all"
    assert obj.layer == 2 and obj.r is not None and obj.n_params() == HIDDEN
    # save -> load_blob round-trips every deployment field, legacy keys included
    round_trip = tmp_path / "again.pt"
    obj.save(round_trip)
    blob = torch.load(round_trip, map_location="cpu", weights_only=True)
    assert blob["kind"] == "static_r" and blob["steer_positions"] == "prefill_all"
    assert blob["site"] == "pre" and blob["layers"] == [2] and blob["layer"] == 2
    again = StaticRObject.load_blob(blob, tiny_model, None)
    assert again.layers == obj.layers and torch.equal(again.rs[2], obj.rs[2])


@pytest.mark.parametrize("kw,needle", [
    ({"steer_positions": "prefill"}, "steer_positions must be one of"),
    ({"site": "mid"}, "site must be one of"),
])
def test_the_constructor_refuses_a_deployment_it_does_not_know(tiny_model, kw, needle):
    with pytest.raises(ValueError, match=needle):
        StaticRObject(tiny_model, layers=[1], rs={1: _direction()}, **kw)
    with pytest.raises(ValueError, match="EITHER"):
        StaticRObject(tiny_model, layer=1, r=_direction(), layers=[1], rs={1: _direction()})
