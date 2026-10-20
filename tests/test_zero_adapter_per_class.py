"""The CLASS-AWARE zero adapter: ``lora_export.export_zero_adapter_for_class``
builds one all-zero adapter per module-set class, so the mandatory no-op leg
of a native standup adapts EXACTLY the class its genuine adapters adapt (the
launch restricts vLLM's wrappers to one class — an o_proj-only zero beside
``mlp`` cells would refuse the launch, or certify a mislabelled control).

On the tiny in-code Qwen3.5-MoE snapshot, one zero adapter per class of
``lora_modules.MODULE_SETS`` × rank {1, 4}:

- every tensor is exactly zero and ``verify_zero_adapter`` accepts;
- ``adapter_target_class`` equals a genuine adapter's of the same set, and
  the file carries the SAME PEFT module names and shapes (the two
  obligatory expert pairs included — the ``down`` class ships the gate_up
  companion too);
- the ``wo`` class IS ``export_or_reuse_zero``: the same dir, byte for byte;
- the content address differs across classes and across ranks, reuse is
  verified and rewrites nothing, and ``zero_adapter_key`` gains its
  experts triples / ``per_expert_rank`` only for experts classes (the
  o_proj key is exactly what it was);
- the generalised class check refuses an exporter that names another class;
- ``vllm_server._adapted_module_class`` admits the zero beside a genuine
  adapter of its class.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import vllm_server as vs
from sisteer.extraction import lora_modules as lm

MOE_TEXT = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, intermediate_size=64,
                moe_intermediate_size=16, shared_expert_intermediate_size=16,
                num_experts=4, num_experts_per_tok=2, vocab_size=128,
                full_attention_interval=4, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_conv_kernel_dim=4)
D, I, E, TOPK = 32, 16, 4, 2
PREFIX = "model.language_model.layers"
SETS = tuple(lm.MODULE_SETS)
RANKS = (1, 4)


@pytest.fixture(scope="module")
def moe_model():
    from transformers import Qwen3_5MoeForConditionalGeneration
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=dict(MOE_TEXT)))
    model.eval()
    return model


@pytest.fixture(scope="module")
def moe_snapshot(moe_model, tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("tiny-qwen35-moe-snapshot")
    moe_model.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def gemma_snapshot(tmp_path_factory) -> str:
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(
        hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, vocab_size=128, sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "full_attention"]))
    d = tmp_path_factory.mktemp("tiny-gemma4-snapshot")
    model.save_pretrained(d)
    return str(d)


def _load(adapter: Path):
    return (load_file(str(Path(adapter) / lx.ADAPTER_WEIGHTS)),
            json.loads((Path(adapter) / lx.ADAPTER_CONFIG).read_text()))


def _shas(adapter: Path):
    return {n: hashlib.sha256((Path(adapter) / n).read_bytes()).hexdigest()
            for n in (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS)}


def _snapshot(d: Path):
    return {p.name: p.read_bytes() for p in Path(d).iterdir() if p.is_file()}


def _genuine(model, snapshot, sid, rank, out):
    return lx.export_module_set(out, base_model=snapshot,
                                blob=lx.synthetic_module_set_blob(model, sid, rank, seed=1))


def test_one_zero_adapter_per_class_and_rank(moe_model, moe_snapshot, tmp_path):
    bank = tmp_path / "bank"
    dirs = {}
    for sid in SETS:
        mods = lm.MODULE_SETS[sid]
        experts = bool(set(mods) & set(lm.MLP_MODULES))
        for rank in RANKS:
            zd, dig, reused = lx.export_zero_adapter_for_class(
                bank, base_model=moe_snapshot, rank=rank, lora_modules=sid)
            assert reused is False and zd.parent == bank.resolve()
            assert zd.name.startswith(f"{lx.ZERO_ADAPTER_NAME}__")
            dirs[(sid, rank)] = zd
            lx.verify_zero_adapter(zd)
            tensors, cfg = _load(zd)
            assert tensors and all(t.dtype is torch.float32 and torch.count_nonzero(t) == 0
                                   for t in tensors.values())
            # the class the file implies == a genuine adapter's of the same set, same names+shapes
            genuine = _genuine(moe_model, moe_snapshot, sid, rank, tmp_path / f"g_{sid}_{rank}")
            want_class = tuple(sorted(set(mods) | ({"experts"} if experts else set())))
            assert lx.adapter_target_class(zd) == lx.adapter_target_class(genuine) == want_class
            gt, gcfg = _load(genuine)
            assert set(tensors) == set(gt)
            assert {k: tuple(t.shape) for k, t in tensors.items()} == {k: tuple(t.shape) for k, t in gt.items()}
            # config: the class's real HF names (deduplicated for a non-o_proj class), scaling 1,
            # the experts keys only where the class reaches routed experts, the no-op marker
            assert cfg["petri_zero_adapter"] is True and "petri_genuine_adapter" not in cfg
            assert cfg["r"] == rank and cfg["lora_alpha"] == float(rank)
            assert set(cfg["target_modules"]) == set(mods) and "experts" not in cfg["target_modules"]
            if sid == "wo":
                assert cfg["target_modules"] == ["o_proj"]   # ONE o_proj layer on this base
            else:
                assert cfg["target_modules"] == list(mods)
                assert cfg["petri_lora_modules"] == list(mods) and cfg["petri_lora_set_id"] == sid
                assert cfg["petri_targets"] == gcfg["petri_targets"]
            if experts:
                r_e = lm.per_expert_rank(rank, TOPK)
                assert cfg["target_parameters"] == ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
                regex = r"mlp\.experts\.(gate_up_proj|down_proj)"
                assert cfg["rank_pattern"] == {regex: r_e} == cfg["alpha_pattern"]
                assert cfg["petri_per_expert_rank"] == {"shared_expert": rank, "routed_expert": r_e}
                for l in range(4):
                    base = f"base_model.model.{PREFIX}.{l}.mlp.experts"
                    assert tensors[f"{base}.base_layer.lora_A.weight"].shape == (r_e * E, D)
                    assert tensors[f"{base}.base_layer.lora_B.weight"].shape == (2 * I, r_e * E)
                    assert tensors[f"{base}.lora_A.weight"].shape == (r_e * E, I)
                    assert tensors[f"{base}.lora_B.weight"].shape == (D, r_e * E)
            else:
                assert not ({"target_parameters", "rank_pattern", "alpha_pattern"} & set(cfg))
            # the bank record
            assert dig.petri_zero_adapter is True and dig.r == rank and dig.name_hint == lx.ZERO_ADAPTER_NAME
            assert dig.config_sha256 == _shas(zd)[lx.ADAPTER_CONFIG]
            # the launch admits the zero beside a genuine adapter of its class
            assert vs._adapted_module_class({"g": str(genuine), "z": str(zd)}) == want_class
    # content-addressed: 5 classes × 2 ranks -> 10 distinct dirs
    assert len({d.name for d in dirs.values()}) == len(SETS) * len(RANKS) == 10
    # verified reuse rewrites nothing
    for (sid, rank), zd in dirs.items():
        before = _snapshot(zd)
        zd2, dig2, reused = lx.export_zero_adapter_for_class(
            bank, base_model=moe_snapshot, rank=rank, lora_modules=lm.MODULE_SETS[sid])
        assert zd2 == zd and reused is True and _snapshot(zd) == before
        assert dig2.config_sha256 == _shas(zd)[lx.ADAPTER_CONFIG]
    assert not [p for p in bank.iterdir() if p.name.startswith(".tmp-")]


def test_the_wo_class_is_export_or_reuse_zero_byte_for_byte(moe_snapshot, tmp_path):
    bank = tmp_path / "bank"
    for rank in RANKS:
        zd, dig, reused = lx.export_or_reuse_zero(bank, base_model=moe_snapshot, rank=rank)
        assert reused is False
        cd, cdig, creused = lx.export_zero_adapter_for_class(
            bank, base_model=moe_snapshot, rank=rank, lora_modules=["o_proj"])
        assert cd == zd and creused is True
        assert _shas(cd) == _shas(zd) and cdig == dig._replace(name_hint=cdig.name_hint)
        # today's un-deduplicated short names (one o_proj layer here), today's key
        modules, _ = lx.checkpoint_target_modules(moe_snapshot)
        assert zd.name == f"{lx.ZERO_ADAPTER_NAME}__" + lx.zero_adapter_key(
            base_model=moe_snapshot, base_revision=None, target_modules=modules, rank=rank)
        assert _load(zd)[1]["target_modules"] == ["o_proj"] * len(modules)
    # the other order — the class generator first — lands the same entry
    bank2 = tmp_path / "bank2"
    cd, _, _ = lx.export_zero_adapter_for_class(bank2, base_model=moe_snapshot, rank=1, lora_modules="wo")
    zd, _, reused = lx.export_or_reuse_zero(bank2, base_model=moe_snapshot, rank=1)
    assert zd == cd and reused is True


def test_zero_adapter_key_gains_experts_triples_only_for_experts_classes(moe_snapshot):
    linear = {"m.layers.3.self_attn.o_proj": (32, 32)}
    # the o_proj key ignores the new keyword when it is None: exactly what it was
    assert lx.zero_adapter_key(base_model="b", base_revision=None, target_modules=linear, rank=1) == \
        lx.zero_adapter_key(base_model="b", base_revision=None, target_modules=linear, rank=1,
                            per_expert_rank=None)
    assert lx.zero_adapter_key(base_model="b", base_revision=None, target_modules=linear, rank=1) != \
        lx.zero_adapter_key(base_model="b", base_revision=None, target_modules=linear, rank=1,
                            per_expert_rank=1)
    experts = {"m.layers.0.mlp.experts.gate_up_proj": (4, 32, 32), "m.layers.0.mlp.experts.down_proj": (4, 16, 32)}
    k1 = lx.zero_adapter_key(base_model="b", base_revision=None, target_modules=experts, rank=4, per_expert_rank=2)
    k2 = lx.zero_adapter_key(base_model="b", base_revision=None, target_modules=experts, rank=4, per_expert_rank=1)
    k3 = lx.zero_adapter_key(base_model="b", base_revision=None,
                             target_modules=dict(experts, **{"m.layers.0.mlp.experts.down_proj": (8, 16, 32)}),
                             rank=4, per_expert_rank=2)
    assert len({k1, k2, k3}) == 3
    for bad in ((4,), (4, 32, 32, 1), (0, 32, 32), ("4", 32, 32)):
        with pytest.raises(lx.LoraExportError, match="must map a non-empty module name to"):
            lx.zero_adapter_key(base_model="b", base_revision=None,
                                target_modules={"m": bad}, rank=1)
    with pytest.raises(lx.LoraExportError, match="per_expert_rank must be None or a positive int"):
        lx.zero_adapter_key(base_model="b", base_revision=None, target_modules=linear, rank=1, per_expert_rank=0)
    # the resolved class carries the triples, the companion included for `down`
    resolved = lx.checkpoint_target_modules_for_class(moe_snapshot, None, "down", 4)
    triples = {k: v for k, v in resolved.target_modules.items() if len(v) == 3}
    assert set(triples) == {f"{PREFIX}.{l}.mlp.experts.{p}" for l in range(4)
                            for p in ("gate_up_proj", "down_proj")}
    assert triples[f"{PREFIX}.0.mlp.experts.gate_up_proj"] == (E, D, 2 * I)
    assert triples[f"{PREFIX}.0.mlp.experts.down_proj"] == (E, I, D)
    assert {k: v for k, v in resolved.target_modules.items() if len(v) == 2} == {
        f"{PREFIX}.{l}.mlp.shared_expert.down_proj": (I, D) for l in range(4)}
    assert resolved.shape.routed_rank == 2 and resolved.shape.set_id == "down"
    assert set(resolved.shape.targets) == {"down_proj", "experts.down_proj"}   # no companion here
    assert resolved.namespace.checkpoint == PREFIX


def test_the_class_check_refuses_an_exporter_that_names_another_class(moe_snapshot, tmp_path, monkeypatch):
    bank = tmp_path / "bank"
    real_class_export = lx._export_zero_adapter_for_class
    real_oproj_export = lx.export_zero_adapter

    def wrong(out_dir, **kw):
        d = real_class_export(out_dir, **kw)
        cfg = json.loads((d / lx.ADAPTER_CONFIG).read_text())
        cfg["target_modules"] = ["q_proj"]
        (d / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg, indent=2, sort_keys=True))
        return d

    monkeypatch.setattr(lx, "_export_zero_adapter_for_class", wrong)
    with pytest.raises(lx.LoraExportError,
                       match=r"target_modules=\['q_proj'\] .* class asked for \(\['down_proj', 'gate_proj', 'up_proj'\]\)"):
        lx.export_zero_adapter_for_class(bank, base_model=moe_snapshot, rank=1, lora_modules="mlp")
    assert not [p for p in bank.iterdir() if p.name.startswith(".tmp-")] if bank.exists() else True
    # the same check on the o_proj branch (export_or_reuse_zero) is now the class asked for
    monkeypatch.undo()

    def wrong_oproj(out_dir, **kw):
        d = real_oproj_export(out_dir, **kw)
        cfg = json.loads((d / lx.ADAPTER_CONFIG).read_text())
        cfg["target_modules"] = ["o_proj", "q_proj"]
        (d / lx.ADAPTER_CONFIG).write_text(json.dumps(cfg, indent=2, sort_keys=True))
        return d

    monkeypatch.setattr(lx, "export_zero_adapter", wrong_oproj)
    with pytest.raises(lx.LoraExportError, match=r"class asked for \(\['o_proj'\]\)"):
        lx.export_or_reuse_zero(bank, base_model=moe_snapshot)
    # a class outside the vocabulary, a bad rank
    monkeypatch.undo()
    with pytest.raises(lx.LoraExportError, match="unknown LoRA target module"):
        lx.export_zero_adapter_for_class(bank, base_model=moe_snapshot, rank=1, lora_modules=["conv1d"])
    with pytest.raises(lx.LoraExportError, match="rank must be a positive int"):
        lx.export_zero_adapter_for_class(bank, base_model=moe_snapshot, rank=0, lora_modules="mlp")


def test_dense_gemma_classes_carry_no_experts(gemma_snapshot, tmp_path):
    bank = tmp_path / "bank"
    names = set()
    for sid in SETS:
        zd, _, _ = lx.export_zero_adapter_for_class(bank, base_model=gemma_snapshot, rank=4, lora_modules=sid)
        lx.verify_zero_adapter(zd)
        names.add(zd.name)
        assert lx.adapter_target_class(zd) == lm.MODULE_SETS[sid]
        tensors, cfg = _load(zd)
        assert not any(".experts" in k for k in tensors)
        assert not ({"target_parameters", "rank_pattern", "alpha_pattern"} & set(cfg))
        assert {int(k.split(".layers.")[1].split(".")[0]) for k in tensors} == {0, 1, 2}
    assert len(names) == len(SETS)
