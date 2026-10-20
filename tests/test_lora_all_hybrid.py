"""lora_r1_all on hybrid-attention models.

Qwen3.5's ``qwen3_5`` stack interleaves full-attention layers (which have
``self_attn.o_proj``) with linear-attention blocks (which have NO
``self_attn`` at all): on Qwen3.5-27B only 16 of 64 layers are
full_attention. The all-layer rank-1 o_proj LoRA (MultiLayerLoRAObject,
arm ``lora_r1_all``) used to default to ``range(num_layers)`` and died
with an AttributeError before training. This file pins the
capability-filtered contract:

- the DEFAULT layer set is exactly the layers that HAVE self_attn.o_proj
  (a capability probe — models.oproj_layers — never a model_type table),
  and apply() hooks only those layers;
- BACKWARD COMPAT: on an all-full-attention model (a synthetic all-full
  stack AND the real tiny-Llama fixture) the default set is every layer,
  unchanged from the historical behavior;
- the targeted set is RECORDED: the save blob carries an explicit
  ``layers`` key, and the artifact round-trips (save -> load_object)
  with the same set and the same parameters;
- loudness (repo rule — fail loudly, never degrade silently): an
  EXPLICIT layer without o_proj refuses with ValueError (never a silent
  narrowing of what was asked for); a model with no capable layer at all
  refuses too (never a silent no-op object); and the single-layer
  RankOneLoRAObject names the problem instead of a bare AttributeError.

CPU-only: synthetic modules + the tiny_llama fixture; no network, no GPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.objects import RankOneLoRAObject
from sisteer.methods import load_object
from sisteer.models import num_layers, oproj_layers

D_IN, D_OUT = 6, 4  # deliberately non-square (the gemma-2 o_proj shape trap)


class _FullAttnBlock(nn.Module):
    """A decoder layer with an attention output projection."""

    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(D_IN, D_OUT, bias=False)


class _LinearAttnBlock(nn.Module):
    """The hybrid sibling: a linear-attention block with NO self_attn at
    all (the qwen3_5 shape — GatedDeltaNet lives under ``linear_attn``)."""

    def __init__(self):
        super().__init__()
        self.linear_attn = nn.Linear(D_OUT, D_OUT, bias=False)


def _stack(pattern: list[bool]) -> nn.Module:
    """A synthetic model whose decoder layers follow ``pattern`` (True =
    full attention), discoverable by models.get_decoder_layers via the
    ``model.layers`` path."""
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList(
        [_FullAttnBlock() if full else _LinearAttnBlock()
         for full in pattern])
    return m


HYBRID = [True, False, False, True, False, True]  # capable: 0, 3, 5


def test_default_layers_are_capability_filtered_on_hybrid():
    model = _stack(HYBRID)
    assert oproj_layers(model) == [0, 3, 5]
    obj = MultiLayerLoRAObject(model)
    assert obj.layers == [0, 3, 5]
    assert sorted(obj.A) == [0, 3, 5] and sorted(obj.B) == [0, 3, 5]
    # Real-module dims, per targeted layer (the non-square o_proj rule).
    for l in obj.layers:
        assert obj.A[l].shape == (D_IN,) and obj.B[l].shape == (D_OUT,)


def test_apply_hooks_only_the_targeted_layers():
    model = _stack(HYBRID)
    obj = MultiLayerLoRAObject(model)
    layers = model.model.layers
    with obj.apply(model):
        for i, full in enumerate(HYBRID):
            if full:
                assert layers[i].self_attn.o_proj._forward_hooks, (
                    f"targeted layer {i} has no live hook")
            else:
                assert not layers[i].linear_attn._forward_hooks, (
                    f"linear-attention layer {i} was hooked")
    # apply() is a context manager: everything unhooked on exit.
    for i, full in enumerate(HYBRID):
        mod = layers[i].self_attn.o_proj if full else layers[i].linear_attn
        assert not mod._forward_hooks


def test_all_full_attention_default_is_every_layer_backward_compat():
    # Synthetic all-full stack: the filtered default IS range(n).
    model = _stack([True] * 5)
    assert MultiLayerLoRAObject(model).layers == list(range(5))


def test_tiny_llama_default_is_every_layer_backward_compat(tiny_llama):
    # The real Llama-shaped fixture (all layers full attention): the
    # capability filter changes NOTHING — the historical all-layer set.
    model, _tok = tiny_llama
    n = num_layers(model)
    assert oproj_layers(model) == list(range(n))
    assert MultiLayerLoRAObject(model).layers == list(range(n))


def test_roundtrip_records_and_restores_the_targeted_set(tmp_path):
    model = _stack(HYBRID)
    obj = MultiLayerLoRAObject(model)
    # Perturb so the round-trip check is on real values, not zero-init.
    for l in obj.layers:
        obj.B[l].data = torch.randn_like(obj.B[l])
    p = tmp_path / "object.pt"
    obj.save(p)

    blob = torch.load(p, map_location="cpu", weights_only=True)
    # The explicit audit record: 3 targeted layers, not len(pattern).
    assert blob["layers"] == [0, 3, 5]
    assert sorted(blob["A"]) == ["0", "3", "5"]

    loaded = load_object(p, model=model, tok=None)
    assert isinstance(loaded, MultiLayerLoRAObject)
    assert loaded.layers == [0, 3, 5]
    for l in loaded.layers:
        assert torch.equal(loaded.A[l].detach(), obj.A[l].detach())
        assert torch.equal(loaded.B[l].detach(), obj.B[l].detach())


def test_explicit_incapable_layer_refuses_loudly():
    model = _stack(HYBRID)
    with pytest.raises(ValueError, match="no self_attn.o_proj"):
        MultiLayerLoRAObject(model, layers=[0, 1])


def test_no_capable_layer_refuses_the_silent_noop():
    model = _stack([False, False])
    with pytest.raises(ValueError, match="nothing to target"):
        MultiLayerLoRAObject(model)


def test_single_layer_lora_names_the_incapable_layer():
    model = _stack(HYBRID)
    # Valid layer: unchanged behavior.
    assert RankOneLoRAObject(model, layer=3).A.shape == (D_IN,)
    # Incapable layer: a named refusal, not a bare AttributeError.
    with pytest.raises(ValueError, match="no self_attn.o_proj"):
        RankOneLoRAObject(model, layer=1)
