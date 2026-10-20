"""Trainable-parameter counts: the count the table
and the registry carry is read off the built object's ACTUAL trainable
tensors under the joint gate/up 3-D expert layout, agrees with the closed
form on tiny dims, and the closed form reproduces the hand counts on the
verified Qwen3.5-4B (worked example) dims. Zero companions (the
``down`` set's frozen gate_up pair the export ships) are not trainable
tensors and are therefore never counted."""

from __future__ import annotations

import pytest
import torch

from sisteer.extraction import lora_modules as lm
from sisteer.extraction.object_factory import build_object, describe_object

TEXT = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, intermediate_size=64,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2, vocab_size=128,
            full_attention_interval=4, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4)
TINY = lm.ArchDims(hidden=32, attention_layers=1, q_out=64, kv_out=16, o_in=32,
                   moe_layers=4, num_experts=4, active_experts=2,
                   moe_intermediate=16, shared_intermediate=16)


@pytest.fixture(scope="module")
def moe():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration)
    torch.manual_seed(0)
    return Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=TEXT)).eval()


def test_tiny_moe_counts_by_hand_decomposition():
    """The closed form's arithmetic spelled out on TINY (d = 32, I = 16, E = 4,
    2 active -> r_e = max(1, r // 2)): routed experts per block
    E·r_e·(2d + 3I); shared expert 3·r·(d + I) on the mlp set, r·(d + I) on
    down; attention per full-attention block r·(d + q_out) + 2·r·(d + kv_out)
    + r·(o_in + d) = r·256 (qkvo) or r·64 (wo)."""
    d, I, E = 32, 16, 4
    assert lm.formula_parameter_count(TINY, "mlp", 4) == 4 * (3 * 4 * (d + I) + E * 2 * (2 * d + 3 * I))
    assert lm.formula_parameter_count(TINY, "attn_mlp", 64) == 4 * (3 * 64 * (d + I) + E * 32 * (2 * d + 3 * I)) + 64 * 256
    assert lm.formula_parameter_count(TINY, "wo", 4) == 4 * 64
    assert lm.formula_parameter_count(TINY, "qkvo", 4) == 4 * 256
    assert lm.formula_parameter_count(TINY, "down", 4) == 4 * (4 * (d + I) + E * 2 * (I + d))
    assert lm.formula_parameter_count(TINY, "mlp", 1) == 4 * (3 * 1 * (d + I) + E * 1 * (2 * d + 3 * I))   # r_e floors at 1


def test_qwen35_4b_demo_dims_are_dense_linear_counts():
    d = lm.QWEN35_4B_DIMS            # the worked example's Qwen3.5-4B: 8 attention blocks, 32 dense MLPs, no experts
    assert lm.formula_parameter_count(d, "wo", 4) == 8 * 4 * (4096 + 2560)
    assert lm.formula_parameter_count(d, "down", 4) == 32 * 4 * (2560 + 9_216)
    assert lm.formula_parameter_count(d, "mlp", 4) == 3 * lm.formula_parameter_count(d, "down", 4)


@pytest.mark.parametrize("sid", sorted(lm.MODULE_SETS))
@pytest.mark.parametrize("rank", [4, 16, 64])
def test_built_object_tensors_equal_the_formula(moe, sid, rank):
    obj = build_object(moe, object_kind="lora", lora_rank=rank, lora_modules=sid)
    want = lm.formula_parameter_count(TINY, sid, rank)
    assert lm.parameter_count(obj) == want
    assert obj.n_params() == want
    assert describe_object(obj)["trainable_params"] == want
    # the count is over the ACTUAL trainable tensors, no companions
    assert sum(p.numel() for p in obj.trainable) == want
    if sid == "down":
        # no gate_up factor is trainable on the down set (the export's zero
        # companion is not an object tensor)
        assert not any(k.endswith(lm.EXPERTS_GATE_UP) for k in obj.A)
        assert any(k.endswith(lm.EXPERTS_DOWN) for k in obj.A)


def test_set_order_is_derived_from_counts_not_a_list(moe):
    counts = {s: lm.parameter_count(build_object(moe, object_kind="lora", lora_rank=4,
                                                 lora_modules=s))
              for s in lm.MODULE_SETS}
    order = lm.set_order_by_parameter_count(counts)
    assert order[0] == "wo" and order[-1] == "attn_mlp"
    assert counts[order[0]] < counts[order[1]] < counts[order[-1]]
    assert order == ["wo", "qkvo", "down", "mlp", "attn_mlp"]            # the tiny MoE: routed experts make down > qkvo
    real = {s: lm.formula_parameter_count(lm.QWEN35_4B_DIMS, s, 4) for s in lm.MODULE_SETS}
    assert lm.set_order_by_parameter_count(real) == ["wo", "qkvo", "down", "mlp", "attn_mlp"]   # the hybrid: 8 attention blocks' four projections < 32 down_projs


def test_per_expert_rank_floors_at_one_and_the_difference_doubles(moe):
    small = build_object(moe, object_kind="lora", lora_rank=1, lora_modules="mlp")
    assert small.per_expert_rank == {"shared_expert": 1, "routed_expert": 1}
    from sisteer.extraction.multilayer import LoRASubtractionObject
    a = build_object(moe, object_kind="lora", lora_rank=4, lora_modules="mlp", seed=1)
    b = build_object(moe, object_kind="lora", lora_rank=4, lora_modules="mlp", seed=2)
    sub = LoRASubtractionObject(a, b)
    assert sub.n_params() == 0  # derived: nothing trainable
    assert describe_object(sub)["parent_trainable_params"] == a.n_params()
    blob = LoRASubtractionObject.compose_blob(a.to_blob(), b.to_blob())
    stacked = sum(blob[f"{t}_{side}"][k].numel()
                  for side in ("pos", "neg") for t in ("A", "B") for k in blob["A_pos"])
    assert stacked == 2 * a.n_params()
