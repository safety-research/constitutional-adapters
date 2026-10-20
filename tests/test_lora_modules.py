"""LoRA module sets: the vocabulary, per-model target
resolution on a tiny sparse-MoE hybrid and a tiny dense gemma-4, the
per-expert rank rule, the general ``lora_all`` blob round trip, the
routed-expert forward (grouped GEMM path == per-expert loop == a hand
reference), hook equivalence on a 2-block toy, the rank-2r difference as
the sum of the parents' deltas, and the legacy o_proj object's
bit-identity."""

from __future__ import annotations

import io

import pytest
import torch

from sisteer.extraction import lora_modules as lm
from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                           MultiLayerLoRAObject,
                                           _experts_lora_forward,
                                           _ExpertLoRATerm)

TEXT = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, intermediate_size=64,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2, vocab_size=128,
            full_attention_interval=4, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4)
TINY_DIMS = lm.ArchDims(hidden=32, attention_layers=1, q_out=64, kv_out=16,
                        o_in=32, moe_layers=4, num_experts=4, active_experts=2,
                        moe_intermediate=16, shared_intermediate=16)


@pytest.fixture(scope="module")
def moe():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration)
    torch.manual_seed(0)
    return Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=TEXT)).eval()


@pytest.fixture(scope="module")
def gemma():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, intermediate_size=64,
                           vocab_size=128, sliding_window=8,
                           layer_types=["sliding_attention", "sliding_attention",
                                        "full_attention"])
    return Gemma4ForCausalLM(cfg).eval()


# --------------------------------------------------------------------------
# vocabulary + rule
# --------------------------------------------------------------------------

def test_vocabulary_and_set_ids():
    assert lm.normalise_modules("mlp") == ("down_proj", "gate_proj", "up_proj")
    assert lm.normalise_modules("o_proj,q_proj") == ("o_proj", "q_proj")
    assert lm.normalise_modules(["v_proj", "q_proj", "q_proj"]) == ("q_proj", "v_proj")
    for sid, members in lm.MODULE_SETS.items():
        assert lm.set_id_for(members) == sid
        assert lm.set_id_for(sid) == sid
    assert lm.set_id_for(["o_proj", "q_proj"]) == "o_proj+q_proj"
    with pytest.raises(lm.LoRAModuleError, match="unknown"):
        lm.normalise_modules(["experts"])
    with pytest.raises(lm.LoRAModuleError, match="empty"):
        lm.normalise_modules([])


def test_per_expert_rank_rule():
    assert lm.per_expert_rank(4, 8) == 1
    assert lm.per_expert_rank(16, 8) == 2
    assert lm.per_expert_rank(64, 8) == 8
    assert lm.per_expert_rank(1, 8) == 1
    assert lm.per_expert_rank_map(64, 8) == {"shared_expert": 64, "routed_expert": 8}
    with pytest.raises(lm.LoRAModuleError):
        lm.per_expert_rank(0, 8)


def test_formula_matches_hand_counts_on_the_tiny_moe_dims():
    """TINY_DIMS by hand (d = 32, I = 16, E = 4, 2 active -> r_e = max(1, r // 2); 4 MoE blocks, 1 attention block,
    q_out 64, kv_out 16, o_in 32): attention per block r*(d + q_out) + 2*r*(d + kv_out) + r*(o_in + d); shared expert
    r*(d + I) per adapted projection; routed E*r_e*(d + 2I) for gate_up and E*r_e*(I + d) for down."""
    d = TINY_DIMS
    assert lm.formula_parameter_count(d, "wo", 4) == 4 * (32 + 32)
    assert lm.formula_parameter_count(d, "qkvo", 4) == 4 * (32 + 64) + 2 * 4 * (32 + 16) + 4 * (32 + 32)
    assert lm.formula_parameter_count(d, "down", 4) == 4 * (4 * (32 + 16) + 4 * 2 * (16 + 32))
    assert lm.formula_parameter_count(d, "mlp", 4) == 4 * (3 * 4 * (32 + 16) + 4 * 2 * (32 + 32) + 4 * 2 * (16 + 32))
    assert lm.formula_parameter_count(d, "attn_mlp", 64) == (64 * (96 + 48 + 48 + 64)
                                                              + 4 * (3 * 64 * 48 + 4 * 32 * 64 + 4 * 32 * 48))
    counts = {s: lm.formula_parameter_count(d, s, 4) for s in lm.MODULE_SETS}
    assert lm.set_order_by_parameter_count(counts) == ["wo", "qkvo", "down", "mlp", "attn_mlp"]


def test_formula_matches_hand_counts_on_the_qwen35_4b_demo_dims():
    """The verified hybrid entry (the worked example's Qwen3.5-4B): 8 full-attention blocks carry the attention projections, every
    one of the 32 blocks a dense MLP; d = 2560, I = 9,216, q_out = 8192 (gated attention), kv_out = 1024, o_in = 4096 — linear counts."""
    d = lm.QWEN35_4B_DIMS
    assert lm.formula_parameter_count(d, "wo", 4) == 8 * 4 * (4096 + 2560)
    assert lm.formula_parameter_count(d, "qkvo", 4) == 8 * 4 * ((2560 + 8192) + 2 * (2560 + 1024) + (4096 + 2560))
    assert lm.formula_parameter_count(d, "down", 4) == 32 * 4 * (2560 + 9_216)
    assert lm.formula_parameter_count(d, "mlp", 4) == 3 * lm.formula_parameter_count(d, "down", 4)
    assert lm.formula_parameter_count(d, "attn_mlp", 4) == (lm.formula_parameter_count(d, "qkvo", 4)
                                                            + lm.formula_parameter_count(d, "mlp", 4))
    counts = {s: lm.formula_parameter_count(d, s, 4) for s in lm.MODULE_SETS}
    # the hybrid's few attention blocks put qkvo UNDER down (the dense 60-block case ordered them the other way round)
    assert lm.set_order_by_parameter_count(counts) == ["wo", "qkvo", "down", "mlp", "attn_mlp"]


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def test_resolution_on_the_tiny_moe_hybrid(moe):
    assert lm.active_experts_of(moe) == 2
    wo = lm.resolve_targets(moe, "wo", rank=4)
    assert [(t.layer, t.site) for t in wo] == [(3, "self_attn.o_proj")]
    qkvo = lm.resolve_targets(moe, "qkvo", rank=4)
    assert sorted(t.site for t in qkvo) == ["self_attn.k_proj", "self_attn.o_proj",
                                            "self_attn.q_proj", "self_attn.v_proj"]
    q = next(t for t in qkvo if t.module == "q_proj")
    assert (q.in_features, q.out_features) == (32, 64)  # query + output gate doubled
    mlp = lm.resolve_targets(moe, "mlp", rank=16)
    summary = lm.targets_summary(mlp)
    assert summary["gate_proj"]["site"] == "mlp.shared_expert.gate_proj"
    assert summary["gate_proj"]["layers"] == [0, 1, 2, 3] and summary["gate_proj"]["rank"] == 16
    gu = summary[lm.EXPERTS_GATE_UP]
    assert gu == {"site": "mlp.experts.gate_up_proj", "layers": [0, 1, 2, 3],
                  "rank": 8, "kind": "experts", "num_experts": 4}
    dn = summary[lm.EXPERTS_DOWN]
    assert dn["site"] == "mlp.experts.down_proj" and dn["rank"] == 8
    down = lm.resolve_targets(moe, "down", rank=4)
    assert sorted({t.module for t in down}) == ["down_proj", lm.EXPERTS_DOWN]
    # active_experts = 2 on the tiny model: r_e = max(1, 4 // 2) = 2
    assert all(t.rank == (2 if t.kind == "experts" else 4) for t in down)
    assert {t.rank for t in lm.resolve_targets(moe, "down", rank=1) if t.kind == "experts"} == {1}
    # gate-only refuses (the fused factor cannot express it)
    with pytest.raises(lm.LoRAModuleError, match="gate_proj/up_proj"):
        lm.resolve_targets(moe, ["gate_proj"], rank=4)
    # an explicit block without the set's modules refuses by name
    with pytest.raises(lm.LoRAModuleError, match="carries none"):
        lm.resolve_targets(moe, "wo", rank=4, layers=[0])
    # per-expert rank given explicitly is honoured
    t = lm.resolve_targets(moe, "down", rank=64, per_expert_rank=3)
    assert {x.rank for x in t if x.kind == "experts"} == {3}


def test_ffn_parent_vocabulary_is_the_attestations(moe):
    """``_ffn_layout`` admits exactly the FFN parents
    ``serving_attestation.FFN_PARENTS`` names (ONE tuple, imported, not a
    copy), so a block whose FFN is called ``feed_forward`` or
    ``block_sparse_moe`` resolves here AND its canonical keys are admitted
    by the readiness fingerprint — never fitted and then refused at serve
    time. Checked on synthetic blocks carrying each parent name (dense;
    shared expert + 3-D experts) and on a stranger (``ffn``)."""
    from sisteer.capeval.serving import serving_attestation as sa
    assert lm.FFN_PARENTS is sa.FFN_PARENTS
    d, i, e, r_e = 8, 6, 4, 2

    def dense():
        m = torch.nn.Module()
        m.gate_proj, m.up_proj, m.down_proj = torch.nn.Linear(d, i), torch.nn.Linear(d, i), torch.nn.Linear(i, d)
        return m

    def sparse():
        m = torch.nn.Module()
        m.shared_expert = dense()
        m.experts = torch.nn.Module()
        m.experts.gate_up_proj = torch.nn.Parameter(torch.zeros(e, 2 * i, d))
        m.experts.down_proj = torch.nn.Parameter(torch.zeros(e, d, i))
        return m

    for parent in lm.FFN_PARENTS:
        block = torch.nn.Module()
        setattr(block, parent, dense())
        lay = lm._ffn_layout(block)
        assert lay["dense_prefix"] == parent and lay["shared"] is None and lay["experts"] is None, parent
        assert sa.module_key_of(f"model.layers.0.{lay['dense_prefix']}.gate_proj") == f"L0.{parent}.gate_proj"
        block = torch.nn.Module()
        setattr(block, parent, sparse())
        lay = lm._ffn_layout(block)
        assert lay["shared_prefix"] == f"{parent}.shared_expert" and lay["experts_prefix"] == f"{parent}.experts", parent
        # the fitter's target prefixes are exactly the fingerprint's admitted spellings (file + engine side)
        assert sa.module_key_of(f"base_model.model.model.layers.0.{lay['shared_prefix']}.up_proj") == f"L0.{parent}.shared_expert.up_proj"
        assert sa.module_key_of(f"base_model.model.model.layers.0.{lay['experts_prefix']}.base_layer") == f"L0.{parent}.experts.gate_up_proj"
        assert sa.module_key_of(f"model.layers.0.{lay['experts_prefix']}") == f"L0.{parent}.experts"
    stranger = torch.nn.Module()
    stranger.ffn = dense()
    lay = lm._ffn_layout(stranger)
    assert lay["dense"] is None and lay["shared"] is None and lay["experts"] is None
    assert sa.module_key_of("model.layers.0.ffn.gate_proj") is None
    # the real tiny MoE hybrid's blocks still resolve under `mlp`
    block = lm.get_decoder_layers(moe)[0]
    assert lm._ffn_layout(block)["shared_prefix"] == "mlp.shared_expert"


def test_resolution_on_the_tiny_dense_gemma(gemma):
    assert lm.active_experts_of(gemma) is None
    for sid in lm.MODULE_SETS:
        ts = lm.resolve_targets(gemma, sid, rank=4)
        assert {t.layer for t in ts} == {0, 1, 2}
        assert all(t.kind == "linear" for t in ts)
    mlp = lm.targets_summary(lm.resolve_targets(gemma, "mlp", rank=4))
    assert mlp["gate_proj"]["site"] == "mlp.gate_proj"


@pytest.mark.parametrize("sid", sorted(lm.MODULE_SETS))
@pytest.mark.parametrize("rank", [4, 16, 64])
def test_target_counts_equal_the_formula(moe, sid, rank):
    ts = lm.resolve_targets(moe, sid, rank=rank)
    assert sum(t.parameter_count for t in ts) == lm.formula_parameter_count(TINY_DIMS, sid, rank)


# --------------------------------------------------------------------------
# the object
# --------------------------------------------------------------------------

def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_legacy_o_proj_object_is_bit_identical(moe):
    a = MultiLayerLoRAObject(moe, rank=4, seed=5)
    b = MultiLayerLoRAObject(moe, rank=4, seed=5, modules=("o_proj",))
    assert a.legacy and b.legacy
    assert a.layers == [3]
    assert sorted(a.A) == [3]  # int keys — the legacy shape
    assert torch.equal(a.A[3], b.A[3]) and torch.equal(a.B[3], b.B[3])
    assert torch.count_nonzero(a.B[3]) == 0
    blob = a.to_blob()
    assert set(blob) == {"kind", "layers", "rank", "A", "B", "positions", "response_header_len"}
    g = torch.Generator().manual_seed(5 + 3)
    assert torch.equal(a.A[3], torch.randn(32, 4, generator=g) / 32 ** 0.5)
    assert a.describe()["lora_set_id"] == "wo"


def test_legacy_object_random_init_only_adds_the_init_key(moe):
    a = MultiLayerLoRAObject(moe, rank=4, seed=5, init_scheme="random_lecun_1e-3")
    assert a.legacy
    assert torch.count_nonzero(a.B[3]) > 0
    blob = a.to_blob()
    assert blob["init_scheme"] == "random_lecun_1e-3"
    # A is the same draw as the zero-init object (B follows on the stream)
    z = MultiLayerLoRAObject(moe, rank=4, seed=5)
    assert torch.equal(a.A[3], z.A[3])
    sigma = 1e-3 / 4 ** 0.5
    assert abs(a.B[3].std().item() - sigma) < sigma  # right scale, tiny sample
    back = MultiLayerLoRAObject.load_blob(blob, moe, None)
    assert back.init_scheme == "random_lecun_1e-3" and torch.equal(back.B[3], a.B[3])


@pytest.mark.parametrize("sid", ["qkvo", "attn_mlp", "mlp", "down"])
def test_general_object_zero_init_is_the_unsteered_model_and_restores(moe, sid):
    ids = torch.randint(0, 128, (2, 7))
    base = _logits(moe, ids)
    obj = MultiLayerLoRAObject(moe, rank=16, seed=1, modules=sid)
    assert not obj.legacy
    with obj.apply(moe, scale=1.0):
        out = _logits(moe, ids)
    assert torch.allclose(out, base, atol=1e-5)
    assert torch.equal(_logits(moe, ids), base)  # forward override removed
    assert obj.n_params() == lm.formula_parameter_count(TINY_DIMS, sid, 16)
    assert obj.n_params() == lm.parameter_count(obj)


@pytest.mark.parametrize("sid", ["qkvo", "mlp", "down", "attn_mlp"])
def test_general_blob_round_trip_and_twins(moe, sid):
    obj = MultiLayerLoRAObject(moe, rank=16, seed=3, modules=sid,
                               init_scheme="random_lecun_1e-3")
    blob = obj.to_blob()
    for k in ("lora_modules", "lora_set_id", "per_expert_rank", "init_scheme", "targets"):
        assert k in blob
    assert blob["lora_set_id"] == sid
    assert all("." in k for k in blob["A"])  # "<layer>.<module>" keys
    if sid != "qkvo":
        assert blob["per_expert_rank"] == {"shared_expert": 16, "routed_expert": 8}
        gu = f"0.{lm.EXPERTS_GATE_UP}" if sid != "down" else f"0.{lm.EXPERTS_DOWN}"
        assert blob["A"][gu].dim() == 3
    buf = io.BytesIO()
    torch.save(blob, buf)
    buf.seek(0)
    back = MultiLayerLoRAObject.load_blob(torch.load(buf, weights_only=True), moe, None)
    assert back.modules == obj.modules and back.rank == 16
    assert all(torch.equal(back.A[k], obj.A[k]) and torch.equal(back.B[k], obj.B[k])
               for k in obj.A)
    twin = MultiLayerLoRAObject(moe, rank=16, seed=3, modules=sid,
                                init_scheme="random_lecun_1e-3")
    assert all(torch.equal(twin.A[k], obj.A[k]) and torch.equal(twin.B[k], obj.B[k])
               for k in obj.A)
    other = MultiLayerLoRAObject(moe, rank=16, seed=4, modules=sid,
                                 init_scheme="random_lecun_1e-3")
    assert not all(torch.equal(other.A[k], obj.A[k]) for k in obj.A)


def test_general_blob_refuses_a_different_architecture(moe):
    obj = MultiLayerLoRAObject(moe, rank=4, seed=0, modules="mlp")
    blob = obj.to_blob()
    bad = dict(blob)
    bad["A"] = dict(blob["A"])
    k = next(iter(bad["A"]))
    bad["A"][k] = torch.zeros(bad["A"][k].shape[0] + 1, *bad["A"][k].shape[1:])
    with pytest.raises(ValueError, match="refusing"):
        MultiLayerLoRAObject.load_blob(bad, moe, None)
    wrong = dict(blob)
    wrong["lora_modules"] = ["down_proj"]
    with pytest.raises(ValueError, match="refusing"):
        MultiLayerLoRAObject.load_blob(wrong, moe, None)
    mixed = dict(blob)
    mixed["lora_modules"] = ["o_proj"]
    with pytest.raises(ValueError, match="legacy shape|no meaning"):
        MultiLayerLoRAObject.load_blob(mixed, moe, None)


def test_experts_forward_grouped_equals_loop_and_hand_reference(moe):
    experts = moe.model.language_model.layers[0].mlp.experts
    E, two_i, d = experts.gate_up_proj.shape
    inter = two_i // 2
    r_e = 3  # deliberately not a multiple of the grouped_mm pad
    g = torch.Generator().manual_seed(11)
    A_gu = torch.randn(E, r_e, d, generator=g) * 0.3
    B_gu = torch.randn(E, two_i, r_e, generator=g) * 0.3
    A_d = torch.randn(E, r_e, inter, generator=g) * 0.3
    B_d = torch.randn(E, d, r_e, generator=g) * 0.3
    term = _ExpertLoRATerm(A_gu, B_gu, A_d, B_d, scale=0.7)
    x = torch.randn(9, d, generator=g)
    idx = torch.randint(0, E, (9, 2), generator=g)
    w = torch.softmax(torch.randn(9, 2, generator=g), -1)
    grouped = _experts_lora_forward(experts, [term])
    import sisteer.extraction.multilayer as ml
    orig = ml._grouped_linear
    try:
        ml._grouped_linear = lambda: None
        loop = _experts_lora_forward(experts, [term])
    finally:
        ml._grouped_linear = orig
    assert grouped.__name__ == "forward_grouped" and loop.__name__ == "forward_loop"
    y_g = grouped(x, idx, w)
    y_l = loop(x, idx, w)
    # hand reference: per token, per selected expert
    ref = torch.zeros_like(x)
    for t in range(9):
        for j in range(2):
            e = int(idx[t, j])
            gu = x[t] @ experts.gate_up_proj[e].T + 0.7 * ((x[t] @ A_gu[e].T) @ B_gu[e].T)
            gate, up = gu.chunk(2)
            h = torch.nn.functional.silu(gate) * up
            y = h @ experts.down_proj[e].T + 0.7 * ((h @ A_d[e].T) @ B_d[e].T)
            ref[t] += w[t, j] * y
    assert torch.allclose(y_g, ref, atol=1e-5)
    assert torch.allclose(y_l, ref, atol=1e-5)
    # zero LoRA == the module's own forward
    zero = _ExpertLoRATerm(A_gu, torch.zeros_like(B_gu), A_d, torch.zeros_like(B_d), 1.0)
    with torch.no_grad():
        assert torch.allclose(_experts_lora_forward(experts, [zero])(x, idx, w),
                              experts(x, idx, w), atol=1e-5)
    # a gate_up-only term (no down factors) and a down-only term both work
    gu_only = _ExpertLoRATerm(A_gu, B_gu, None, None, 1.0)
    dn_only = _ExpertLoRATerm(None, None, A_d, B_d, 1.0)
    for tm in (gu_only, dn_only):
        assert _experts_lora_forward(experts, [tm])(x, idx, w).shape == x.shape


def test_linear_hook_equivalence_on_the_shared_expert(moe):
    """The linear-target hook is exactly ΔW = B Aᵀ applied to the module's
    input (scaling 1), checked against a hand matmul on gemma-style dense
    projections and the MoE's shared expert."""
    obj = MultiLayerLoRAObject(moe, rank=4, seed=2, modules="mlp",
                               init_scheme="random_lecun_1e-3")
    key = "1.up_proj"
    mod = obj._linear[key]
    x = torch.randn(3, 5, 32)
    captured = {}
    h = mod.register_forward_hook(lambda m, i, o: captured.setdefault("out", o.clone()))
    with torch.no_grad():
        mod(x)
    h.remove()
    base = captured["out"]
    with obj.apply(moe, scale=2.0), torch.no_grad():
        steered = mod(x)
    delta = 2.0 * ((x @ obj.A[key]) @ obj.B[key].T)
    assert torch.allclose(steered, base + delta, atol=1e-6)


@pytest.mark.parametrize("sid", ["wo", "qkvo", "mlp", "down"])
def test_difference_object_is_the_sum_of_the_parents_deltas(moe, sid):
    ids = torch.randint(0, 128, (2, 6))
    base = _logits(moe, ids)
    pos = MultiLayerLoRAObject(moe, rank=8, seed=1, modules=sid, init_scheme="random_lecun_1e-3")
    neg = MultiLayerLoRAObject(moe, rank=8, seed=2, modules=sid, init_scheme="random_lecun_1e-3")
    # make the deltas visible above fp32 noise
    for o in (pos, neg):
        for k in o.B:
            o.B[k].data *= 40.0
    sub = LoRASubtractionObject(pos, neg, provenance={"test": True})
    with sub.apply(moe, scale=1.0):
        out_sub = _logits(moe, ids)
    assert not torch.allclose(out_sub, base)
    if sid in ("wo", "qkvo"):
        # STRUCTURAL identity: the difference IS the two parents' own hooks
        # registered together (+1 / −1) — bit-identical to that composition,
        # and dropping the negative side recovers the positive parent's
        # forward bitwise.
        with pos.apply(moe, scale=1.0), neg.apply(moe, scale=-1.0):
            out_both = _logits(moe, ids)
        assert torch.equal(out_sub, out_both)
        with pos.apply(moe, scale=1.0):
            out_pos = _logits(moe, ids)
        zero = MultiLayerLoRAObject(moe, rank=8, seed=2, modules=sid,
                                    init_scheme="random_lecun_1e-3")
        for k in zero.B:
            zero.B[k].data.zero_()
        with LoRASubtractionObject(pos, zero, provenance={}).apply(moe, scale=1.0):
            out_pos_minus_zero = _logits(moe, ids)
        assert torch.equal(out_pos_minus_zero, out_pos)
    else:
        # routed experts compose both terms inside ONE experts forward; the
        # hand reference is the combined-term forward on a token batch
        experts = moe.model.language_model.layers[0].mlp.experts
        x = torch.randn(7, 32)
        idx = torch.randint(0, 4, (7, 2))
        w = torch.softmax(torch.randn(7, 2), -1)
        pt = pos.expert_terms(1.0)[0]
        nt = neg.expert_terms(-1.0)[0]
        both = _experts_lora_forward(experts, [pt, nt])(x, idx, w)
        with torch.no_grad():
            base_e = experts(x, idx, w)
        assert not torch.allclose(both, base_e)
        zero_neg = MultiLayerLoRAObject(moe, rank=8, seed=9, modules=sid)  # B = 0
        only_pos = _experts_lora_forward(experts, [pt, zero_neg.expert_terms(-1.0)[0]])(x, idx, w)
        assert torch.allclose(only_pos, _experts_lora_forward(experts, [pt])(x, idx, w), atol=1e-6)
    blob = LoRASubtractionObject.compose_blob(pos.to_blob(), neg.to_blob(), provenance={"x": 1})
    assert blob["kind"] == "lora_subtraction" and blob["rank"] == 8
    if sid != "wo":
        assert blob["lora_set_id"] == sid and "targets" in blob
    else:
        assert "targets" not in blob  # legacy artifact byte-for-byte
    back = LoRASubtractionObject.load_blob(blob, moe, None)
    with back.apply(moe, scale=1.0):
        out_back = _logits(moe, ids)
    assert torch.equal(out_back, out_sub)
    # mismatched twins refuse by name
    with pytest.raises(ValueError, match="differ"):
        LoRASubtractionObject.compose_blob(
            pos.to_blob(), MultiLayerLoRAObject(moe, rank=8, seed=2, modules=sid).to_blob())


def test_difference_refuses_parents_of_different_sets(moe):
    a = MultiLayerLoRAObject(moe, rank=4, modules="mlp")
    b = MultiLayerLoRAObject(moe, rank=4, modules="down")
    with pytest.raises(ValueError, match="not twins|differ"):
        LoRASubtractionObject.compose_blob(a.to_blob(), b.to_blob())
    c = MultiLayerLoRAObject(moe, rank=4)
    with pytest.raises(ValueError, match="not twins"):
        LoRASubtractionObject.compose_blob(a.to_blob(), c.to_blob())


def test_masked_fit_refused_for_expert_targets(moe):
    obj = MultiLayerLoRAObject(moe, rank=4, modules="mlp")
    obj.set_fit_positions("response", 3)
    with pytest.raises(NotImplementedError, match="routed-expert"):
        with obj.apply(moe, scale=1.0):
            pass
    # the override was not left behind
    experts = moe.model.language_model.layers[0].mlp.experts
    assert "forward" not in experts.__dict__


def test_gemma_dense_sets_train_a_step(gemma):
    """A gradient flows into every factor of a dense set on gemma-4 (the
    31B path is ordinary linear LoRA on every block)."""
    obj = MultiLayerLoRAObject(gemma, rank=4, seed=0, modules="attn_mlp")
    ids = torch.randint(0, 128, (1, 8))
    with obj.apply(gemma, scale=1.0):
        loss = gemma(input_ids=ids, labels=ids).loss
    loss.backward()
    assert all(p.grad is not None for p in obj.trainable)
    assert obj.n_params() == sum(t.parameter_count for t in obj.targets)
    assert {t.layer for t in obj.targets} == {0, 1, 2}


def test_the_vocabulary_is_one_object_shared_with_its_stdlib_leaf():
    """lora_modules RE-EXPORTS the module-set vocabulary from sisteer.extraction.lora_module_sets
    (the stdlib leaf the steered-padding preflight reads without torch or transformers) — the same objects, not copies, so a
    caller of either module sees one table, one error class, one rule."""
    from sisteer.extraction import lora_module_sets as leaf
    for name in ("MODULE_SETS", "ATTENTION_MODULES", "MLP_MODULES", "KNOWN_MODULES", "LoRAModuleError", "normalise_modules", "set_id_for"):
        assert getattr(lm, name) is getattr(leaf, name), name
    assert lm.PAPER_LORA_MODULES == leaf.MODULE_SETS["wo"] == ("o_proj",)
    assert lm.set_id_for("k_proj,o_proj,q_proj,v_proj") == "qkvo" and leaf.set_id_for(["o_proj", "down_proj"]) == "down_proj+o_proj"
    with pytest.raises(lm.LoRAModuleError):
        leaf.normalise_modules("gate")
