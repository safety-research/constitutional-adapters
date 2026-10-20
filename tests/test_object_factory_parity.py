"""The shared object factory: identical initial
tensors for the same seed + architecture whether built for a search
trial, a production fit or the constitution/control twins of a seed;
LeCun scales under FD 7's random scheme (σ = 1e-3/√r for LoRA B,
1e-3/√d for the ReLU steer's v and w); byte-identical zero-init defaults;
unknown kwargs refused; the production trainers build through it."""

from __future__ import annotations

import inspect

import pytest
import torch

from sisteer.extraction import object_factory as of
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.psr import PSRObject

TEXT = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, intermediate_size=64,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            num_experts=4, num_experts_per_tok=2, vocab_size=128,
            full_attention_interval=4, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4)


@pytest.fixture(scope="module")
def moe():
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration)
    torch.manual_seed(0)
    return Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=TEXT)).eval()


def _same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.trainable, b.trainable)) \
        and len(a.trainable) == len(b.trainable)


def test_factory_signature_is_closed():
    sig = inspect.signature(of.build_object)
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert [p.kind for n, p in sig.parameters.items() if n != "model"] == \
        [inspect.Parameter.KEYWORD_ONLY] * (len(sig.parameters) - 1)


def test_unknown_kwarg_and_unknown_kind_refuse(moe):
    with pytest.raises(TypeError):
        of.build_object(moe, object_kind="lora", temperature=2.0)  # noqa
    with pytest.raises(of.ObjectFactoryError, match="unknown object_kind"):
        of.build_object(moe, object_kind="vector_all")
    with pytest.raises(of.ObjectFactoryError, match="init_scheme"):
        of.build_object(moe, object_kind="lora", init_scheme="xavier")
    with pytest.raises(of.ObjectFactoryError, match="no LoRA axes"):
        of.build_object(moe, object_kind="relu_steer", lora_rank=4)
    with pytest.raises(of.ObjectFactoryError, match="no meaning"):
        of.build_object(moe, object_kind="lora", per_expert_rank=1)
    with pytest.raises(of.ObjectFactoryError, match="contradicts"):
        of.build_object(moe, object_kind="lora", lora_rank=16, lora_modules="mlp",
                        per_expert_rank=3)
    with pytest.raises(of.ObjectFactoryError, match="shared_expert"):
        of.build_object(moe, object_kind="lora", lora_rank=16, lora_modules="mlp",
                        per_expert_rank={"shared_expert": 4, "routed_expert": 8})


def test_aliases_route_to_the_two_kinds():
    assert of.canonical_kind("lora_r4_ce") == "lora"
    assert of.canonical_kind("lora_all") == "lora"
    assert of.canonical_kind("psr_ce_all") == "relu_steer"
    assert of.canonical_kind("psr") == "relu_steer"


def test_zero_init_defaults_are_byte_identical_to_direct_construction(moe):
    """The paper's objects: the factory's default LoRA is exactly
    ``MultiLayerLoRAObject(model, rank=4, seed=s)`` and its ReLU steer
    exactly ``PSRObject(model, all layers, train_direction=True)``."""
    for seed in (0, 7):
        f = of.build_object(moe, object_kind="lora", seed=seed)
        d = MultiLayerLoRAObject(moe, rank=4, seed=seed)
        assert f.legacy and _same(f, d)
        assert f.to_blob().keys() == d.to_blob().keys()
    f = of.build_object(moe, object_kind="relu_steer")
    d = PSRObject(moe, list(range(4)), train_direction=True)
    assert _same(f, d)
    assert all(torch.count_nonzero(p) == 0 for p in f.directions.values())
    assert all(torch.count_nonzero(p) == 0 for p in f.gate_w.values())
    assert all(torch.equal(p, torch.ones(1)) for p in f.gate_b.values())


@pytest.mark.parametrize("modules", ["wo", "qkvo", "mlp", "down", "attn_mlp"])
@pytest.mark.parametrize("init", ["zero", "random_lecun_1e-3"])
def test_search_and_production_builds_agree_and_twins_share_init(moe, modules, init):
    """The trial runner's build (arm spelling, per_expert_rank as the map)
    and the production trainer's build (kind spelling, per_expert_rank
    omitted) give identical tensors; the constitution and control twins
    of a seed share the identical initialisation; another seed differs."""
    rank = 16
    per = ({"shared_expert": 16, "routed_expert": 8}
           if modules in ("mlp", "down", "attn_mlp") else None)
    trial = of.build_object(moe, object_kind="lora_r4_ce", lora_rank=rank,
                            lora_modules=modules, per_expert_rank=per,
                            init_scheme=init, seed=3)
    prod = of.build_object(moe, object_kind="lora", lora_rank=rank,
                           lora_modules=list(of.normalise_modules(modules)),
                           init_scheme=init, seed=3)
    assert _same(trial, prod)
    constitution = of.build_object(moe, object_kind="lora", lora_rank=rank,
                                   lora_modules=modules, init_scheme=init, seed=3)
    control = of.build_object(moe, object_kind="lora", lora_rank=rank,
                              lora_modules=modules, init_scheme=init, seed=3)
    assert _same(constitution, control)
    other = of.build_object(moe, object_kind="lora", lora_rank=rank,
                            lora_modules=modules, init_scheme=init, seed=4)
    assert not _same(constitution, other)
    if init == "zero":
        assert all(torch.count_nonzero(p) == 0 for k, p in trial.B.items())
    else:
        assert all(torch.count_nonzero(p) > 0 for k, p in trial.B.items())


def test_lecun_scales(moe):
    """FD 7: B ~ N(0, (1e-3/√r)²) per factor; the routed experts use their
    own factor rank r_e; A keeps N(0, 1/d_in). Sample stds are within a
    generous band of σ (the tiny tensors are small samples)."""
    obj = of.build_object(moe, object_kind="lora", lora_rank=64, lora_modules="attn_mlp",
                          init_scheme="random_lecun_1e-3", seed=0)
    for t in obj.targets:
        B = obj.B[t.key]
        sigma = 1e-3 / t.rank ** 0.5
        std = B.std().item()
        assert 0.5 * sigma < std < 1.5 * sigma, (t.key, std, sigma)
        A = obj.A[t.key]
        a_std = A.std().item()
        assert 0.5 / t.in_features ** 0.5 < a_std < 1.5 / t.in_features ** 0.5
    # ReLU steer: v, w ~ N(0, (1e-3/√d)²), b = 1
    p = of.build_object(moe, object_kind="relu_steer", init_scheme="random_lecun_1e-3", seed=1)
    d = 32
    sigma = 1e-3 / d ** 0.5
    for l in p.layers:
        for v in (p.directions[l], p.gate_w[l]):
            assert 0.4 * sigma < v.std().item() < 1.6 * sigma
        assert torch.equal(p.gate_b[l], torch.ones(1))
    twin = of.build_object(moe, object_kind="relu_steer", init_scheme="random_lecun_1e-3", seed=1)
    assert _same(p, twin)
    assert p.to_blob() if hasattr(p, "to_blob") else True


def test_psr_random_init_blob_key_and_load(moe, tmp_path):
    p = of.build_object(moe, object_kind="relu_steer", init_scheme="random_lecun_1e-3", seed=2)
    path = tmp_path / "object.pt"
    p.save(path)
    blob = torch.load(path, weights_only=True)
    assert blob["init_scheme"] == "random_lecun_1e-3"
    z = of.build_object(moe, object_kind="relu_steer")
    z.save(tmp_path / "zero.pt")
    assert "init_scheme" not in torch.load(tmp_path / "zero.pt", weights_only=True)
    back = PSRObject.load_blob(blob, moe, None)
    assert back.init_scheme == "random_lecun_1e-3"
    assert all(torch.equal(back.directions[l], p.directions[l]) for l in p.layers)


def test_describe_object_and_architecture_params(moe):
    obj = of.build_object(moe, object_kind="lora", lora_rank=16, lora_modules="mlp",
                          init_scheme="random_lecun_1e-3", seed=0)
    d = of.describe_object(obj)
    assert d["object_kind"] == "lora" and d["lora_rank"] == 16
    assert d["lora_modules"] == ["down_proj", "gate_proj", "up_proj"]
    assert d["lora_set_id"] == "mlp"
    assert d["per_expert_rank"] == {"shared_expert": 16, "routed_expert": 8}
    assert d["init_scheme"] == "random_lecun_1e-3"
    assert d["trainable_params"] == obj.n_params()
    assert set(d["layers_by_module"]) == {"gate_proj", "up_proj", "down_proj",
                                          "experts.gate_up_proj", "experts.down_proj"}
    p = of.build_object(moe, object_kind="relu_steer")
    dp = of.describe_object(p)
    assert dp["object_kind"] == "relu_steer" and dp["layers_by_module"] == {"residual": [0, 1, 2, 3]}
    ap = of.architecture_params(object_kind="lora", lora_rank=64, lora_modules="down",
                                active_experts=8)
    assert ap == {"object_kind": "lora", "lora_rank": 64, "lora_modules": ["down_proj"],
                  "lora_set_id": "down", "per_expert_rank": {"shared_expert": 64, "routed_expert": 8},
                  "init_scheme": "zero"}
    assert of.architecture_params(object_kind="lora", lora_rank=4, lora_modules="wo",
                                  active_experts=8)["per_expert_rank"] is None


def test_production_trainers_build_through_the_factory(monkeypatch, moe):
    """``methods._train_lora_ce`` / ``_train_psr_ce`` construct their
    objects via ``object_factory.build_object`` with the arm's axes."""
    from sisteer import methods
    import sisteer.extraction.object_factory as factory

    calls = []
    real = factory.build_object

    def spy(model, **kw):
        calls.append(kw)
        return real(model, **kw)

    monkeypatch.setattr(factory, "build_object", spy)
    trained = {}

    def fake_train(model, tok, obj, exs, **kw):
        trained["obj"] = obj
        return [0.0]

    monkeypatch.setattr(methods, "train_object_ce", fake_train)
    monkeypatch.setattr(methods, "docs_to_ce_examples", lambda tok, pairs: [])
    methods._train_lora_ce(moe, None, "constv2-mix-1k", [], lora_rank=16,
                           lora_modules="mlp", init_scheme="random_lecun_1e-3", seed=5)
    assert calls[-1]["object_kind"] == "lora" and calls[-1]["lora_rank"] == 16
    assert calls[-1]["lora_modules"] == "mlp" and calls[-1]["seed"] == 5
    assert trained["obj"].set_id == "mlp"
    monkeypatch.setattr(methods, "check_psr_gates", lambda *a, **k: {0: 1.0}, raising=False)
    import sisteer.extraction.psr as psr_mod
    monkeypatch.setattr(psr_mod, "check_psr_gates", lambda *a, **k: {l: 1.0 for l in range(4)})
    methods._train_psr_ce(moe, None, "constv2-mix-1k", [], init_scheme="random_lecun_1e-3", seed=5)
    assert calls[-1]["object_kind"] == "relu_steer" and calls[-1]["seed"] == 5
    assert trained["obj"].init_scheme == "random_lecun_1e-3"
