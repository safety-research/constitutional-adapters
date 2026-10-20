"""The GENERAL module-set export:
``lora_export._export_multilayer_lora`` / ``_export_lora_subtraction`` on a
blob that adapts anything but the paper's ``("o_proj",)``, the bare-blob
convenience ``export_module_set``, the synthetic builder
``synthetic_module_set_blob`` and the file-derived ``adapter_target_class``.

Load-bearing here, on a tiny in-code Qwen3.5-MoE (4 blocks, attention on
block 3 only, 4 routed experts, top-2) and a tiny gemma-4 (3 dense blocks,
the global-attention blocks twice as wide on the query side):

- every set of ``lora_modules.MODULE_SETS`` × rank {4, 16} exports; the
  PEFT keys are ``base_model.model.{checkpoint prefix}.{layer}.{site}.lora_
  {A,B}.weight`` for a linear module and PEFT ``ParamWrapper``'s TWO pairs
  per MoE block for the routed experts — ``...mlp.experts.base_layer.*`` =
  gate_up ``[(r_e·E), d]`` / ``[2I, (r_e·E)]`` and ``...mlp.experts.*`` =
  down ``[(r_e·E), I]`` / ``[d, (r_e·E)]``;
- the file -> ``reshape(E, -1, in)`` / ``reshape(out, -1, E).permute(2, 0,
  1)`` round trip (vLLM 0.25.1 ``model_manager.py:857-880``, spelled inline
  here, not through the exporter's own helper) recovers the blob's 3-D
  tensors BIT-EXACTLY;
- the ``down`` set ships an all-zero gate_up companion at the same r_e;
- the difference adapter's factors equal the concatenation rule (linear
  ``cat([A_pos.T, A_neg.T])`` / ``cat([B_pos, -B_neg])``; experts along the
  per-expert rank axis BEFORE flattening), and its per-module ΔW equals
  ``ΔW_pos - ΔW_neg`` to fp32 rounding — checked WITHOUT peft by applying
  ``ΔW = B @ A`` by hand per linear module and the per-expert delta on a
  token batch, then a whole-model forward;
- config: ``target_modules`` are the REAL HF short names (never
  ``experts``), ``target_parameters`` + ``rank_pattern``/``alpha_pattern``
  only where routed experts are adapted, ``lora_alpha == r``, the
  ``petri_*`` module-set keys only on general exports;
- the LEGACY o_proj blob (no ``targets``) exports BYTE-IDENTICALLY to
  goldens captured from the exporter before this change (sha256 of both
  PEFT files on the stated-prefix path; the weights sha and the config
  minus its tmp-path key on the derived path over a tiny llama snapshot);
- ``adapter_target_class`` returns the expected class for every set.

Blobs are built BY HAND as dicts per the contract (fp32 CPU tensors over
``lora_modules.resolve_targets``' shapes) so nothing here depends on the
object classes in ``extraction/multilayer.py``.
"""

import copy
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import vllm_server as vs
from sisteer.extraction import lora_modules as lm

sys.path.insert(0, str(Path(__file__).parent))
from test_petri_serving_gate import _chat_template, _tpl_spec  # noqa: E402

# --------------------------------------------------------------------------
# the tiny bases
# --------------------------------------------------------------------------

#: the tiny Qwen3.5-MoE text config: d = 32, I = 16 (so 2I = 32 = d — the
#: fused gate_up is SQUARE here; the round-trip and forward checks below
#: are value checks, which a transposition fails regardless), E = 4, top-2
#: (r_e = max(1, r // 2)), full attention every 4th block => block 3 only
MOE_TEXT = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=8, intermediate_size=64,
                moe_intermediate_size=16, shared_expert_intermediate_size=16,
                num_experts=4, num_experts_per_tok=2, vocab_size=128,
                full_attention_interval=4, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_conv_kernel_dim=4)
D, I, E, TOPK = 32, 16, 4, 2
MOE_PREFIX = "model.language_model.layers"
MOE_ATTN_LAYERS = [3]
MOE_ALL_LAYERS = [0, 1, 2, 3]
GEMMA_PREFIX = "model.layers"
SETS = tuple(lm.MODULE_SETS)
RANKS = (4, 16)


def _r_e(rank):
    return lm.per_expert_rank(rank, TOPK)


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
def gemma_model():
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(
        hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, intermediate_size=64, vocab_size=128, sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "full_attention"]))
    model.eval()
    return model


@pytest.fixture(scope="module")
def gemma_snapshot(gemma_model, tmp_path_factory) -> str:
    d = tmp_path_factory.mktemp("tiny-gemma4-snapshot")
    gemma_model.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def llama_snapshot(tmp_path_factory) -> str:
    """The tiny llama of tests/test_adapter_bank.py, saved as a local
    snapshot — the legacy o_proj byte-identity check's base."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=97,
                      max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    model.eval()
    d = tmp_path_factory.mktemp("tiny-llama-snapshot")
    model.save_pretrained(d)
    return str(d)


# --------------------------------------------------------------------------
# hand-built blobs (D_contract §2) and the by-hand ΔW application
# --------------------------------------------------------------------------


def _hand_blob(model, modules, rank, *, seed, scale=0.1):
    """A general-shape ``lora_all`` blob written BY HAND from
    ``resolve_targets``' shapes (never through the object class): linear
    A ``[d_in, r]`` / B ``[d_out, r]``, experts A ``[E, r_e, in]`` / B
    ``[E, out, r_e]``, fp32 CPU, keyed ``"<layer>.<module>"``."""
    mods = lm.normalise_modules(modules)
    targets = lm.resolve_targets(model, mods, rank=rank)
    g = torch.Generator().manual_seed(seed)
    A, B, routed = {}, {}, None
    for t in targets:
        if t.kind == "linear":
            A[t.key] = torch.randn(t.in_features, t.rank, generator=g)
            B[t.key] = torch.randn(t.out_features, t.rank, generator=g) * scale
        else:
            routed = t.rank
            A[t.key] = torch.randn(t.num_experts, t.rank, t.in_features, generator=g)
            B[t.key] = torch.randn(t.num_experts, t.out_features, t.rank, generator=g) * scale
    return {
        "kind": "lora_all", "layers": sorted({t.layer for t in targets}), "rank": rank,
        "A": A, "B": B, "positions": "all", "response_header_len": None,
        "lora_modules": list(mods), "lora_set_id": lm.set_id_for(mods),
        "per_expert_rank": ({"shared_expert": rank, "routed_expert": routed}
                            if routed is not None else None),
        "init_scheme": "random_lecun_1e-3",
        "targets": lm.targets_summary(targets),
    }


def _difference_blob(pos, neg):
    """``LoRASubtractionObject.compose_blob``'s shape, written by hand."""
    out = {k: pos[k] for k in ("layers", "rank", "positions", "response_header_len",
                              "lora_modules", "lora_set_id", "per_expert_rank", "targets")}
    out.update({"kind": "lora_subtraction", "A_pos": pos["A"], "B_pos": pos["B"],
                "A_neg": neg["A"], "B_neg": neg["B"],
                "provenance": {"parent_pos": {"cell_id": "mix", "object_sha256": "a" * 64},
                               "parent_neg": {"cell_id": "ctl", "object_sha256": "b" * 64}}})
    return out


def _load(adapter: Path):
    return (load_file(str(Path(adapter) / lx.ADAPTER_WEIGHTS)),
            json.loads((Path(adapter) / lx.ADAPTER_CONFIG).read_text()))


def _shas(adapter: Path):
    return {n: hashlib.sha256((Path(adapter) / n).read_bytes()).hexdigest()
            for n in (lx.ADAPTER_CONFIG, lx.ADAPTER_WEIGHTS)}


def _vllm_unflatten(lora_A, lora_B, num_experts):
    """vLLM 0.25.1's read of the file, spelled INLINE (model_manager.py:
    857-880) so the round trip does not trust the exporter's own helper."""
    A3 = lora_A.reshape(num_experts, -1, lora_A.shape[-1])
    B3 = lora_B.reshape(lora_B.shape[0], -1, num_experts).permute(2, 0, 1)
    return A3, B3


def _expected_keys(prefix, blob):
    """Every PEFT key the export must write for ``blob``: the linear pairs
    at their sites and, per MoE block, BOTH experts pairs (the companion
    included when the blob carries one half)."""
    keys, experts_blocks = set(), set()
    for module, rec in blob["targets"].items():
        for l in rec["layers"]:
            if rec["kind"] == "linear":
                m = f"base_model.model.{prefix}.{l}.{rec['site']}"
                keys |= {f"{m}.lora_A.weight", f"{m}.lora_B.weight"}
            else:
                experts_blocks.add((l, rec["site"].rsplit(".", 1)[0]))
    for l, path in experts_blocks:
        for suffix in ("", ".base_layer"):
            m = f"base_model.model.{prefix}.{l}.{path}{suffix}"
            keys |= {f"{m}.lora_A.weight", f"{m}.lora_B.weight"}
    return keys


def _deltas_from_file(tensors, prefix, num_experts):
    """ΔW per adapted module, from the FILE: linear ``lora_B @ lora_A``
    ``[out, in]`` keyed by the module path; experts keyed ``(experts path,
    param)`` -> ``[E, out, in]`` of per-expert ``B3[e] @ A3[e]``."""
    out = {}
    modules = sorted({k[len("base_model.model."):-len(".lora_A.weight")]
                      for k in tensors if k.endswith(".lora_A.weight")})
    for m in modules:
        a = tensors[f"base_model.model.{m}.lora_A.weight"]
        b = tensors[f"base_model.model.{m}.lora_B.weight"]
        if m.endswith(".experts") or m.endswith(".experts.base_layer"):
            inner = m.endswith(".base_layer")
            path = m[:-len(".base_layer")] if inner else m
            A3, B3 = _vllm_unflatten(a, b, num_experts)
            out[(path, "gate_up_proj" if inner else "down_proj")] = torch.einsum(
                "eor,eri->eoi", B3, A3)
        else:
            out[m] = b @ a
    return out


def _deltas_from_blob(blob, prefix, *, a="A", b="B", sign=1.0):
    """ΔW per adapted module, from the BLOB's raw factors: linear
    ``B @ A.T``; experts per-expert ``B3[e] @ A3[e]``."""
    out = {}
    for key, A in blob[a].items():
        B = blob[b][key]
        layer, module = key.split(".", 1)
        rec = blob["targets"][module]
        if rec["kind"] == "linear":
            out[f"{prefix}.{layer}.{rec['site']}"] = sign * (B @ A.t())
        else:
            path, param = rec["site"].rsplit(".", 1)
            out[(f"{prefix}.{layer}.{path}", param)] = sign * torch.einsum("eor,eri->eoi", B, A)
    return out


def _sum_deltas(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out[k] + v if k in out else v.clone()
    return out


def _apply_deltas(model, deltas):
    """A deep copy of ``model`` with every ΔW ADDED to the base weight by
    hand (scaling exactly 1): ``W += ΔW`` for a linear module,
    ``P[e] += ΔW_e`` for an experts parameter. No peft anywhere."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        for k, dw in deltas.items():
            if isinstance(k, tuple):
                path, param = k
                p = getattr(m.get_submodule(path), param)
                assert p.shape == dw.shape, (k, tuple(p.shape), tuple(dw.shape))
                p.add_(dw)
            else:
                w = m.get_submodule(k).weight
                assert w.shape == dw.shape, (k, tuple(w.shape), tuple(dw.shape))
                w.add_(dw)
    return m


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


GENERAL_ONLY_KEYS = {"petri_lora_modules", "petri_lora_set_id", "petri_per_expert_rank",
                     "petri_targets"}
EXPERTS_ONLY_KEYS = {"target_parameters", "rank_pattern", "alpha_pattern"}
COMMON_KEYS = {"peft_type", "base_model_name_or_path", "lora_dropout", "bias", "fan_in_fan_out",
               "task_type", "petri_genuine_adapter", "petri_decoder_prefix", "r", "lora_alpha",
               "target_modules", "layers_to_transform", "petri_source_kind",
               "petri_source_layers", "petri_fit_positions"}
#: the legacy o_proj lora_all config's key set, as the golden export writes it
LEGACY_KEYS = COMMON_KEYS | {"petri_intervention_hash"}


# --------------------------------------------------------------------------
# every set × rank on the tiny MoE
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("sid", SETS)
def test_every_set_and_rank_exports_on_the_tiny_moe(moe_model, moe_snapshot, tmp_path, sid, rank):
    blob = _hand_blob(moe_model, sid, rank, seed=rank * 10 + len(sid))
    mods = lm.MODULE_SETS[sid]
    reaches_experts = bool(set(mods) & set(lm.MLP_MODULES))
    out = lx.export_module_set(tmp_path / f"{sid}_r{rank}", base_model=moe_snapshot, blob=blob)
    tensors, cfg = _load(out)
    r_e = _r_e(rank)

    if sid == "wo":
        # the paper's set: the legacy blob shape (no targets) — built BY HAND below;
        # a general-shape blob for ("o_proj",) is also admitted and names the same module
        assert set(tensors) == {f"base_model.model.{MOE_PREFIX}.3.self_attn.o_proj.lora_{w}.weight"
                                for w in ("A", "B")}
    assert set(tensors) == _expected_keys(MOE_PREFIX, blob)
    for k, t in tensors.items():
        assert t.dtype is torch.float32

    # linear pairs: lora_A = A.T [r, in], lora_B = B [out, r], at the blob's site
    for key, A in blob["A"].items():
        layer, module = key.split(".", 1)
        rec = blob["targets"][module]
        if rec["kind"] != "linear":
            continue
        m = f"base_model.model.{MOE_PREFIX}.{layer}.{rec['site']}"
        assert torch.equal(tensors[f"{m}.lora_A.weight"], A.t())
        assert torch.equal(tensors[f"{m}.lora_B.weight"], blob["B"][key])
        assert tensors[f"{m}.lora_A.weight"].shape == (rank, A.shape[0])
    # attention modules only on block 3, FFN modules on every block
    for module in mods:
        want = MOE_ATTN_LAYERS if module in lm.ATTENTION_MODULES else MOE_ALL_LAYERS
        assert blob["targets"][module]["layers"] == want
    # experts: PEFT ParamWrapper's two pairs per MoE block, the exact shapes, bit-exact round trip
    if reaches_experts:
        for l in MOE_ALL_LAYERS:
            base = f"base_model.model.{MOE_PREFIX}.{l}.mlp.experts"
            gu_A, gu_B = tensors[f"{base}.base_layer.lora_A.weight"], tensors[f"{base}.base_layer.lora_B.weight"]
            dn_A, dn_B = tensors[f"{base}.lora_A.weight"], tensors[f"{base}.lora_B.weight"]
            assert gu_A.shape == (r_e * E, D) and gu_B.shape == (2 * I, r_e * E)
            assert dn_A.shape == (r_e * E, I) and dn_B.shape == (D, r_e * E)
            for half, (fa, fb) in (("experts.gate_up_proj", (gu_A, gu_B)),
                                   ("experts.down_proj", (dn_A, dn_B))):
                A3, B3 = _vllm_unflatten(fa, fb, E)
                key = f"{l}.{half}"
                if key in blob["A"]:
                    assert torch.equal(A3, blob["A"][key]) and torch.equal(B3, blob["B"][key])
                    # the exporter's own inverse agrees with the inline vLLM spelling
                    A3b, B3b = lx.experts_unflatten(fa, fb, E)
                    assert torch.equal(A3b, A3) and torch.equal(B3b, B3)
                else:
                    # the half the set does not adapt is the ZERO companion at the same r_e
                    assert sid == "down" and half == "experts.gate_up_proj"
                    assert torch.count_nonzero(fa) == 0 and torch.count_nonzero(fb) == 0
        assert cfg["target_parameters"] == ["mlp.experts.down_proj", "mlp.experts.gate_up_proj"]
        regex = r"mlp\.experts\.(gate_up_proj|down_proj)"
        assert cfg["rank_pattern"] == {regex: r_e} and cfg["alpha_pattern"] == {regex: r_e}
        assert cfg["petri_per_expert_rank"] == {"shared_expert": rank, "routed_expert": r_e}
        # PEFT's matching rule (utils/other.get_pattern_key) selects the experts parameters and
        # nothing else
        import re
        for name in (f"{MOE_PREFIX}.0.mlp.experts.gate_up_proj", f"{MOE_PREFIX}.3.mlp.experts.down_proj"):
            assert re.match(rf"(.*\.)?({regex})$", name)
        for name in (f"{MOE_PREFIX}.0.mlp.shared_expert.down_proj", f"{MOE_PREFIX}.3.self_attn.o_proj"):
            assert not re.match(rf"(.*\.)?({regex})$", name)
    else:
        assert not (EXPERTS_ONLY_KEYS & set(cfg))
        assert cfg["petri_per_expert_rank"] is None

    # config: real HF short names, never the wrapper word; scaling exactly 1
    assert cfg["target_modules"] == list(mods) and "experts" not in cfg["target_modules"]
    assert cfg["r"] == rank and cfg["lora_alpha"] == float(rank)
    assert cfg["layers_to_transform"] == blob["layers"] == cfg["petri_source_layers"]
    assert cfg["petri_lora_modules"] == list(mods) and cfg["petri_lora_set_id"] == sid
    assert cfg["petri_targets"] == blob["targets"]
    assert cfg["petri_decoder_prefix"] == MOE_PREFIX and cfg["petri_genuine_adapter"] is True
    assert "petri_zero_adapter" not in cfg and cfg["petri_source_kind"] == "lora_all"
    want_keys = COMMON_KEYS | GENERAL_ONLY_KEYS | (EXPERTS_ONLY_KEYS if reaches_experts else set())
    assert set(cfg) == want_keys
    # the class the file implies
    want_class = tuple(sorted(set(mods) | ({"experts"} if reaches_experts else set())))
    assert lx.adapter_target_class(out) == want_class
    # a genuine adapter is never certified as the no-op leg
    with pytest.raises(lx.LoraExportError, match="not marked petri_zero_adapter"):
        lx.verify_zero_adapter(out)


def test_general_export_is_deterministic_and_the_stated_prefix_matches_the_derived(
        moe_model, moe_snapshot, tmp_path):
    blob = _hand_blob(moe_model, "mlp", 4, seed=3)
    a = lx.export_module_set(tmp_path / "a", base_model=moe_snapshot, blob=blob,
                             intervention_hash="ab" * 8)
    b = lx.export_module_set(tmp_path / "b", base_model=moe_snapshot, blob=blob,
                             intervention_hash="ab" * 8)
    assert _shas(a) == _shas(b)
    assert json.loads((a / lx.ADAPTER_CONFIG).read_text())["petri_intervention_hash"] == "ab" * 8
    # the stated-prefix path (no skeleton, the blob's sites trusted) spells the same names
    c = lx.export_module_set(tmp_path / "c", base_model="org/tiny-moe", blob=blob,
                             decoder_prefix=MOE_PREFIX, intervention_hash="ab" * 8)
    assert _shas(c)[lx.ADAPTER_WEIGHTS] == _shas(a)[lx.ADAPTER_WEIGHTS]
    ca, cc = _load(a)[1], _load(c)[1]
    ca.pop("base_model_name_or_path"), cc.pop("base_model_name_or_path")
    assert ca == cc


def test_single_adapter_forward_equivalence_by_hand(moe_model, moe_snapshot, tmp_path):
    """The FILE's factors, applied by hand (linear ``lora_B @ lora_A``;
    experts unflattened the vLLM way, ``B3[e] @ A3[e]`` per expert), change
    the model exactly as the BLOB's raw factors do (``B @ A.T``) — the same
    ops on the same numbers, so bit-identical deltas and logits."""
    blob = _hand_blob(moe_model, "attn_mlp", 4, seed=5)
    out = lx.export_module_set(tmp_path / "s", base_model=moe_snapshot, blob=blob)
    tensors, _ = _load(out)
    from_file = _deltas_from_file(tensors, MOE_PREFIX, E)
    from_blob = _deltas_from_blob(blob, MOE_PREFIX)
    assert set(from_file) == set(from_blob)
    for k in from_blob:
        assert torch.equal(from_file[k], from_blob[k]), k
    ids = torch.randint(0, 128, (2, 6), generator=torch.Generator().manual_seed(0))
    base = _logits(moe_model, ids)
    steered = _logits(_apply_deltas(moe_model, from_file), ids)
    assert torch.equal(steered, _logits(_apply_deltas(moe_model, from_blob), ids))
    # a KNOWN NONZERO effect: the adapter moves the logits
    assert (steered - base).abs().max() > 1e-3


def test_difference_adapter_concatenates_and_equals_the_parents_delta_sum(
        moe_model, moe_snapshot, tmp_path):
    rank, r_e = 4, _r_e(4)
    pos = _hand_blob(moe_model, "mlp", rank, seed=11)
    neg = _hand_blob(moe_model, "mlp", rank, seed=12)
    diff = _difference_blob(pos, neg)
    out = lx.export_module_set(tmp_path / "d", base_model=moe_snapshot, blob=diff)
    tensors, cfg = _load(out)
    assert set(tensors) == _expected_keys(MOE_PREFIX, pos)
    # the concatenation rule, per target
    for key in pos["A"]:
        layer, module = key.split(".", 1)
        rec = pos["targets"][module]
        if rec["kind"] == "linear":
            m = f"base_model.model.{MOE_PREFIX}.{layer}.{rec['site']}"
            assert torch.equal(tensors[f"{m}.lora_A.weight"],
                               torch.cat([pos["A"][key].t(), neg["A"][key].t()], 0))
            assert torch.equal(tensors[f"{m}.lora_B.weight"],
                               torch.cat([pos["B"][key], -neg["B"][key]], 1))
            assert tensors[f"{m}.lora_A.weight"].shape[0] == 2 * rank
        else:
            path, param = rec["site"].rsplit(".", 1)
            m = f"base_model.model.{MOE_PREFIX}.{layer}.{path}" + lx.EXPERTS_PEFT_SUFFIX[param]
            fa, fb = tensors[f"{m}.lora_A.weight"], tensors[f"{m}.lora_B.weight"]
            assert fa.shape[0] == 2 * r_e * E and fb.shape[1] == 2 * r_e * E
            A3, B3 = _vllm_unflatten(fa, fb, E)
            assert torch.equal(A3, torch.cat([pos["A"][key], neg["A"][key]], dim=1))
            assert torch.equal(B3, torch.cat([pos["B"][key], -neg["B"][key]], dim=2))
            # the negated block, read per expert: rows [r_e, 2r_e) of the rank axis
            assert torch.equal(B3[:, :, r_e:], -neg["B"][key])
    # config: rank-2r at scaling 1, the experts pattern doubled too, the parents recorded
    assert cfg["r"] == 2 * rank and cfg["lora_alpha"] == float(2 * rank)
    regex = r"mlp\.experts\.(gate_up_proj|down_proj)"
    assert cfg["rank_pattern"] == {regex: 2 * r_e} == cfg["alpha_pattern"]
    assert cfg["petri_lora_subtraction_parent_rank"] == rank
    assert cfg["petri_lora_subtraction_negated_block"] == lx.SUBTRACTION_NEGATED_BLOCK
    assert cfg["petri_lora_subtraction_provenance"] == {
        "parent_pos": {"cell_id": "mix", "object_sha256": "a" * 64},
        "parent_neg": {"cell_id": "ctl", "object_sha256": "b" * 64}}
    assert cfg["petri_source_kind"] == "lora_subtraction"
    assert cfg["petri_per_expert_rank"] == {"shared_expert": rank, "routed_expert": r_e}
    assert cfg["petri_targets"] == pos["targets"] and cfg["target_modules"] == list(lm.MODULE_SETS["mlp"])
    assert set(cfg) == (COMMON_KEYS | GENERAL_ONLY_KEYS | EXPERTS_ONLY_KEYS
                        | {"petri_lora_subtraction_parent_rank",
                           "petri_lora_subtraction_negated_block",
                           "petri_lora_subtraction_provenance"})
    assert lx.adapter_target_class(out) == ("down_proj", "experts", "gate_proj", "up_proj")

    # THE FORWARD EQUIVALENCE, WITHOUT PEFT: the file's ΔW per module equals ΔW_pos - ΔW_neg
    # to fp32 rounding (one rank-2r GEMM vs two rank-r GEMMs and a subtraction) ...
    from_file = _deltas_from_file(tensors, MOE_PREFIX, E)
    parents = _sum_deltas(_deltas_from_blob(pos, MOE_PREFIX),
                          _deltas_from_blob(neg, MOE_PREFIX, sign=-1.0))
    assert set(from_file) == set(parents)
    for k in parents:
        torch.testing.assert_close(from_file[k], parents[k], atol=1e-5, rtol=1e-5)
    # ... the per-expert delta on a token batch agrees expert by expert ...
    x = torch.randn(7, D, generator=torch.Generator().manual_seed(1))
    h = torch.randn(7, I, generator=torch.Generator().manual_seed(2))
    experts_keys = [k for k in from_file if isinstance(k, tuple)]
    assert len(experts_keys) == 2 * len(MOE_ALL_LAYERS)
    for key in experts_keys:
        inp = x if key[1] == "gate_up_proj" else h
        for e in range(E):
            torch.testing.assert_close(inp @ from_file[key][e].t(), inp @ parents[key][e].t(),
                                       atol=1e-5, rtol=1e-5)
    # ... and the whole model, the deltas added to its weights by hand, agrees on the logits
    ids = torch.randint(0, 128, (2, 6), generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(_logits(_apply_deltas(moe_model, from_file), ids),
                               _logits(_apply_deltas(moe_model, parents), ids),
                               atol=1e-3, rtol=1e-3)
    # the difference is a genuine kind, never the zero adapter
    with pytest.raises(lx.LoraExportError, match="not marked petri_zero_adapter"):
        lx.verify_zero_adapter(out)


# --------------------------------------------------------------------------
# gemma-4: dense, mixed-width attention
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("sid", SETS)
def test_gemma4_dense_sets_export_per_layer_shapes(gemma_model, gemma_snapshot, tmp_path, sid, rank):
    blob = _hand_blob(gemma_model, sid, rank, seed=rank + len(sid))
    mods = lm.MODULE_SETS[sid]
    out = lx.export_module_set(tmp_path / f"g_{sid}_r{rank}", base_model=gemma_snapshot, blob=blob)
    tensors, cfg = _load(out)
    assert set(tensors) == _expected_keys(GEMMA_PREFIX, blob)
    assert not any(".experts" in k for k in tensors)
    blocks = gemma_model.model.layers
    for module in mods:
        assert blob["targets"][module]["layers"] == [0, 1, 2]
        site = ("self_attn" if module in lm.ATTENTION_MODULES else "mlp") + "." + module
        assert blob["targets"][module]["site"] == site
        for l in range(3):
            w = blocks[l].get_submodule(site).weight
            m = f"base_model.model.{GEMMA_PREFIX}.{l}.{site}"
            assert tensors[f"{m}.lora_A.weight"].shape == (rank, w.shape[1])
            assert tensors[f"{m}.lora_B.weight"].shape == (w.shape[0], rank)
    if "o_proj" in mods:
        # the global-attention blocks (1, 2) are 2048 wide on the query side, block 0 is 32
        assert tensors[f"base_model.model.{GEMMA_PREFIX}.0.self_attn.o_proj.lora_A.weight"].shape == (rank, 32)
        assert tensors[f"base_model.model.{GEMMA_PREFIX}.1.self_attn.o_proj.lora_A.weight"].shape == (rank, 2048)
    assert cfg["target_modules"] == list(mods)
    assert not (EXPERTS_ONLY_KEYS & set(cfg)) and cfg["petri_per_expert_rank"] is None
    assert set(cfg) == COMMON_KEYS | GENERAL_ONLY_KEYS
    assert lx.adapter_target_class(out) == tuple(mods)
    # the by-hand forward: file == blob, bitwise
    from_file = _deltas_from_file(tensors, GEMMA_PREFIX, 1)
    from_blob = _deltas_from_blob(blob, GEMMA_PREFIX)
    assert set(from_file) == set(from_blob) and all(
        torch.equal(from_file[k], from_blob[k]) for k in from_blob)


# --------------------------------------------------------------------------
# the LEGACY o_proj blob is byte-identical to the pre-change exporter
# --------------------------------------------------------------------------

#: sha256 of both PEFT files, captured from the exporter BEFORE the module-set
#: change (2026-09-29) on the STATED-prefix path (no tmp path in the config),
#: over the seeded legacy blobs below
GOLDEN_STATED = {
    ("lora_all", 4): {"adapter_config.json": "0c8d644c4241b28d240d444dc26eed6e3aa92ed06bcde2c7976ce5bf2455d7c3",
                      "adapter_model.safetensors": "3806340c191f9df24c824a90095731d0e203fa1bef8f4312d006a4917f7e38bb"},
    ("lora_all", 1): {"adapter_config.json": "d01100884de0868529f6c5bbeafb4e012b4bdff870da899e8df69512fcbbdefe",
                      "adapter_model.safetensors": "08be7ee08dd9e29ec210c855e0407a2152ead16acc5c33a87db9281367b0dde6"},
    ("lora_subtraction", 4): {"adapter_config.json": "14031ae887e08af113687cf7280fc77714d2fc3f15f0dbe5b1f04725993242f6",
                              "adapter_model.safetensors": "937ade6bc71ac6813629d1cad66857fd4d84070acda3ae26f565c3f03fddddf8"},
}
#: the DERIVED path over the tiny llama snapshot: the weights sha, and sha256
#: over the config with its snapshot-path key removed (the path is per run)
GOLDEN_DERIVED = {
    "lora_all": ("3806340c191f9df24c824a90095731d0e203fa1bef8f4312d006a4917f7e38bb",
                 "d9ea9f2358bc8d72568c9a01421574b6a5c3435fa6c301371dcc86eda56c484d"),
    "lora_subtraction": ("937ade6bc71ac6813629d1cad66857fd4d84070acda3ae26f565c3f03fddddf8",
                         "b3b96462720642765afe7d0dab33687d0fca99140aef3ff7de2cb75db6209f44"),
}
STATED = dict(base_model="meta-llama/Llama-3.1-8B-Instruct", decoder_prefix="model.layers")


def _legacy_blob(seed, rank=4, d_in=32, d_out=32, layers=(0, 1)):
    """The paper's o_proj ``lora_all`` blob, written by hand: ``str(layer)``
    keys, 1-D factors at rank 1, no ``targets``."""
    g = torch.Generator().manual_seed(seed)
    if rank == 1:
        A = {str(l): torch.randn(d_in, generator=g) for l in layers}
        B = {str(l): torch.randn(d_out, generator=g) * 0.1 for l in layers}
    else:
        A = {str(l): torch.randn(d_in, rank, generator=g) for l in layers}
        B = {str(l): torch.randn(d_out, rank, generator=g) * 0.1 for l in layers}
    return {"kind": "lora_all", "layers": list(layers), "rank": rank, "A": A, "B": B,
            "positions": "all", "response_header_len": None}


def _legacy_subtraction(seeds=(11, 12)):
    pos, neg = _legacy_blob(seeds[0]), _legacy_blob(seeds[1])
    return {"kind": "lora_subtraction", "layers": [0, 1], "rank": 4,
            "A_pos": pos["A"], "B_pos": pos["B"], "A_neg": neg["A"], "B_neg": neg["B"],
            "provenance": {"parent_pos": {"cell_id": "mix", "object_sha256": "a" * 64},
                           "parent_neg": {"cell_id": "ctl", "object_sha256": "b" * 64}},
            "positions": "all", "response_header_len": None}


def _bytes(blob) -> bytes:
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def test_legacy_oproj_blobs_export_byte_identically_to_the_pre_change_goldens(llama_snapshot, tmp_path):
    a = lx.export_genuine_adapter({"kind": "lora_all", "intervention_hash": "ab" * 8},
                                  _bytes(_legacy_blob(7)), tmp_path / "a", **STATED)
    assert _shas(a) == GOLDEN_STATED[("lora_all", 4)]
    a1 = lx.export_genuine_adapter({"kind": "lora_all", "intervention_hash": "ab" * 8},
                                   _bytes(_legacy_blob(9, rank=1)), tmp_path / "a1", **STATED)
    assert _shas(a1) == GOLDEN_STATED[("lora_all", 1)]
    s = lx.export_genuine_adapter({"kind": "lora_subtraction", "intervention_hash": "cd" * 8},
                                  _bytes(_legacy_subtraction()), tmp_path / "s", **STATED)
    assert _shas(s) == GOLDEN_STATED[("lora_subtraction", 4)]
    assert set(_load(a)[1]) == LEGACY_KEYS
    # a legacy blob carrying the FD 7 init_scheme key exports the SAME bytes
    with_scheme = dict(_legacy_blob(7), init_scheme="random_lecun_1e-3")
    a2 = lx.export_genuine_adapter({"kind": "lora_all", "intervention_hash": "ab" * 8},
                                   _bytes(with_scheme), tmp_path / "a2", **STATED)
    assert _shas(a2) == GOLDEN_STATED[("lora_all", 4)]
    # the derived path over the tiny llama: the same weights, the same config but for the path
    for kind, blob, ih in (("lora_all", _legacy_blob(7), "ab" * 8),
                           ("lora_subtraction", _legacy_subtraction(), "cd" * 8)):
        d = lx.export_genuine_adapter({"kind": kind, "intervention_hash": ih}, _bytes(blob),
                                      tmp_path / f"d_{kind}", base_model=llama_snapshot)
        weights_sha, config_sha = GOLDEN_DERIVED[kind]
        assert _shas(d)[lx.ADAPTER_WEIGHTS] == weights_sha
        cfg = _load(d)[1]
        assert cfg.pop("base_model_name_or_path") == llama_snapshot
        assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == config_sha
    # export_module_set on a legacy blob is the same exporter
    e = lx.export_module_set(tmp_path / "e", blob=_legacy_blob(7), intervention_hash="ab" * 8, **STATED)
    assert _shas(e) == GOLDEN_STATED[("lora_all", 4)]
    assert lx.adapter_target_class(e) == ("o_proj",)
    # the general-shape keys without a targets block are neither shape: refused, never guessed
    with pytest.raises(lx.LoraExportError, match="general-shape keys .* but no 'targets'"):
        lx.export_module_set(tmp_path / "x", blob=dict(_legacy_blob(7), lora_modules=["o_proj"]),
                             **STATED)
    with pytest.raises(lx.LoraExportError, match="init_scheme='bogus' is not one of"):
        lx.export_module_set(tmp_path / "y", blob=dict(_legacy_blob(7), init_scheme="bogus"), **STATED)


# --------------------------------------------------------------------------
# the synthetic builder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sid", SETS)
def test_synthetic_builder_matches_the_hand_built_shape(moe_model, moe_snapshot, tmp_path, sid):
    rank = 4
    hand = _hand_blob(moe_model, sid, rank, seed=1)
    syn = lx.synthetic_module_set_blob(moe_model, sid, rank, seed=1)
    zero = lx.synthetic_module_set_blob(moe_model, sid, rank, seed=1, zero=True)
    if sid == "wo":
        # the paper's set is the LEGACY shape: str(layer) keys, no targets
        assert set(syn) == {"kind", "layers", "rank", "A", "B", "positions", "response_header_len"}
        assert set(syn["A"]) == {"3"} and syn["A"]["3"].shape == (D, rank)
        r1 = lx.synthetic_module_set_blob(moe_model, sid, 1, seed=1)
        assert r1["A"]["3"].shape == (D,) and r1["B"]["3"].shape == (D,)
    else:
        assert set(syn) == set(hand) - {"init_scheme"}
        assert set(syn["A"]) == set(hand["A"]) and set(syn["B"]) == set(hand["B"])
        for k in hand["A"]:
            assert syn["A"][k].shape == hand["A"][k].shape and syn["B"][k].shape == hand["B"][k].shape
            assert syn["A"][k].dtype is torch.float32
        for k in ("lora_modules", "lora_set_id", "per_expert_rank", "targets", "layers", "rank"):
            assert syn[k] == hand[k]
    # the zero twin: A identical, every B zero; both export and the zero one has zero factors
    for k in syn["A"]:
        assert torch.equal(zero["A"][k], syn["A"][k]) and torch.count_nonzero(zero["B"][k]) == 0
        assert torch.count_nonzero(syn["B"][k]) > 0
    out = lx.export_module_set(tmp_path / "syn", base_model=moe_snapshot, blob=syn)
    zout = lx.export_module_set(tmp_path / "zero", base_model=moe_snapshot, blob=zero)
    assert lx.adapter_target_class(out) == lx.adapter_target_class(zout)
    zt, _ = _load(zout)
    assert all(torch.count_nonzero(t) == 0 for k, t in zt.items() if ".lora_B." in k)
    # scale sets the effect size
    big = lx.synthetic_module_set_blob(moe_model, sid, rank, seed=1, scale=10.0)
    k = next(iter(syn["B"]))
    torch.testing.assert_close(big["B"][k], 10.0 * syn["B"][k])


# --------------------------------------------------------------------------
# adapter_target_class reads the FILE
# --------------------------------------------------------------------------


def test_adapter_target_class_reads_the_file_not_the_config(tmp_path):
    def write(d, tensors, target_modules=("o_proj",)):
        d.mkdir()
        # safetensors refuses tensors that share storage: one fresh copy per key
        save_file({k: v.clone().contiguous() for k, v in tensors.items()},
                  str(d / lx.ADAPTER_WEIGHTS))
        (d / lx.ADAPTER_CONFIG).write_text(json.dumps({"r": 4, "lora_alpha": 4.0,
                                                       "target_modules": list(target_modules)}))
        return d

    z = torch.zeros(4, 8)
    m = "base_model.model.model.layers.0"
    # a residual carrier: the o_proj pair plus a payload key the class ignores
    d = write(tmp_path / "carrier", {f"{m}.self_attn.o_proj.lora_A.weight": z,
                                     f"{m}.self_attn.o_proj.lora_B.weight": z.t().contiguous(),
                                     "sisteer_residual.layers.0.vector": torch.zeros(8)})
    assert lx.adapter_target_class(d) == ("o_proj",)
    # the config's target_modules is NOT consulted
    d = write(tmp_path / "cfg", {f"{m}.self_attn.q_proj.lora_A.weight": z,
                                 f"{m}.self_attn.q_proj.lora_B.weight": z.t().contiguous()},
              target_modules=("o_proj",))
    assert lx.adapter_target_class(d) == ("q_proj",)
    # both experts pairs -> "experts"; a lone half refuses (vLLM asserts both at load)
    both = {f"{m}.mlp.experts.base_layer.lora_A.weight": z, f"{m}.mlp.experts.base_layer.lora_B.weight": z,
            f"{m}.mlp.experts.lora_A.weight": z, f"{m}.mlp.experts.lora_B.weight": z,
            f"{m}.mlp.shared_expert.down_proj.lora_A.weight": z,
            f"{m}.mlp.shared_expert.down_proj.lora_B.weight": z}
    assert lx.adapter_target_class(write(tmp_path / "both", both)) == ("down_proj", "experts")
    lone = {k: v for k, v in both.items() if "base_layer" in k}
    with pytest.raises(lx.LoraExportError, match=r"carries only \['gate_up_proj'\]"):
        lx.adapter_target_class(write(tmp_path / "lone", lone))
    # a suffix outside the vocabulary, a foreign key, an empty file, a missing file
    with pytest.raises(lx.LoraExportError, match="adapts 'lm_head', outside the module vocabulary"):
        lx.adapter_target_class(write(tmp_path / "head", {"base_model.model.lm_head.lora_A.weight": z,
                                                          "base_model.model.lm_head.lora_B.weight": z}))
    with pytest.raises(lx.LoraExportError, match="neither a lora_A/lora_B factor nor a"):
        lx.adapter_target_class(write(tmp_path / "foreign", {f"{m}.self_attn.o_proj.weight": z}))
    with pytest.raises(lx.LoraExportError, match="carries no LoRA factor"):
        lx.adapter_target_class(write(tmp_path / "empty", {"sisteer_residual.layers.0.vector": z}))
    with pytest.raises(lx.LoraExportError, match="has no adapter_model.safetensors"):
        lx.adapter_target_class(tmp_path / "nowhere")


# --------------------------------------------------------------------------
# refusals on the general shape
# --------------------------------------------------------------------------


def test_general_shape_refusals_name_the_reason(moe_model, moe_snapshot, tmp_path):
    good = _hand_blob(moe_model, "mlp", 4, seed=2)

    def refuses(mutate, match):
        blob = copy.deepcopy(good)
        mutate(blob)
        with pytest.raises(lx.LoraExportError, match=match):
            lx.export_module_set(tmp_path / "r", base_model=moe_snapshot, blob=blob)

    refuses(lambda b: b.__setitem__("lora_modules", ["up_proj", "gate_proj", "down_proj"]),
            "is not the sorted, de-duplicated list")
    refuses(lambda b: b.__setitem__("lora_modules", ["down_proj", "gate_proj", "up_proj", "conv1d"]),
            "unknown LoRA target module")
    refuses(lambda b: b.__setitem__("lora_set_id", "down"), "lora_set_id='down' does not name")
    refuses(lambda b: b["per_expert_rank"].__setitem__("shared_expert", 2),
            "per_expert_rank.shared_expert=2 != the object's rank 4")
    refuses(lambda b: b["per_expert_rank"].__setitem__("routed_expert", 1),
            r"rank=2 != per_expert_rank.routed_expert=1")
    refuses(lambda b: b.__setitem__("init_scheme", "xavier"), "init_scheme='xavier' is not one of")
    refuses(lambda b: b.pop("rank"), "must declare its rank")
    refuses(lambda b: b["A"].pop("0.gate_proj"), r"A's keys do not match the targets block \(missing \['0.gate_proj'\]")
    refuses(lambda b: b["B"].__setitem__("9.gate_proj", torch.zeros(D, 4)), r"stray \['9.gate_proj'\]")
    refuses(lambda b: b["targets"].__setitem__("conv1d", {"site": "x.conv1d", "layers": [0], "rank": 4, "kind": "linear"}),
            "names module 'conv1d', outside the vocabulary")
    refuses(lambda b: b["targets"]["gate_proj"].__setitem__("kind", "experts"),
            "kind='experts', but module 'gate_proj' is 'linear'")
    refuses(lambda b: b["targets"]["gate_proj"].__setitem__("rank", 2),
            r"targets\['gate_proj'\].rank=2 != the object's rank 4")
    refuses(lambda b: b["targets"]["experts.down_proj"].pop("num_experts"), "lacks num_experts")
    refuses(lambda b: b["targets"]["experts.down_proj"].__setitem__("num_experts", 8),
            "cover different blocks or expert counts")
    refuses(lambda b: b.__setitem__("layers", [0, 1, 2]), "is not the sorted union")
    # a target the blob names on a block the base does not carry (a linear-attention block's
    # o_proj), with the factor keys kept consistent, is refused by name off the skeleton
    def add_oproj_on_block_0(b):
        b["lora_modules"] = ["down_proj", "gate_proj", "o_proj", "up_proj"]
        b["lora_set_id"] = "down_proj+gate_proj+o_proj+up_proj"
        b["targets"]["o_proj"] = {"site": "self_attn.o_proj", "layers": [0], "rank": 4, "kind": "linear"}
        b["A"]["0.o_proj"] = torch.zeros(D, 4)
        b["B"]["0.o_proj"] = torch.zeros(D, 4)
    refuses(add_oproj_on_block_0, r"layer 0 carries no 'o_proj' on this base \(blocks that do: \[3\]\)")
    # a site the base spells differently, and a shape the module does not have
    refuses(lambda b: b["targets"]["gate_proj"].__setitem__("site", "mlp.gate_proj"),
            "the blob says site 'mlp.gate_proj' but the base's block 0 carries it at 'mlp.shared_expert.gate_proj'")
    refuses(lambda b: b["A"].__setitem__("0.gate_proj", torch.zeros(D + 1, 4)),
            r"0.gate_proj has linear dims \(33, 16\) but the base's own module")
    refuses(lambda b: b["A"].__setitem__("0.experts.down_proj", torch.zeros(E, 2, I + 1)),
            r"0.experts.down_proj has experts dims \(4, 17, 32\) but the base's own module")
    refuses(lambda b: b["A"].__setitem__("0.experts.down_proj", torch.zeros(E, 3, I)),
            r"but the target is 4 experts at per-expert rank 2")
    refuses(lambda b: b["A"].__setitem__("0.gate_proj", torch.zeros(D, 4, dtype=torch.float16)),
            "is torch.float16, but the object's save path writes float32")
    refuses(lambda b: b.__setitem__("positions", "response"), "lens o_proj")
    # the F9-style spec cross-checks
    with pytest.raises(lx.LoraExportError, match="resolved_spec lora_modules=.* != artifact lora_modules"):
        lx._export_multilayer_lora({"kind": "lora_all", "lora_modules": ["o_proj"]}, good,
                                   decoder_prefix=MOE_PREFIX)
    with pytest.raises(lx.LoraExportError, match="resolved_spec per_expert_rank"):
        lx._export_multilayer_lora({"kind": "lora_all", "per_expert_rank": None}, good,
                                   decoder_prefix=MOE_PREFIX)
    # a difference whose parents disagree in shape
    pos, neg = _hand_blob(moe_model, "mlp", 4, seed=1), _hand_blob(moe_model, "mlp", 4, seed=2)
    neg["A"]["0.gate_proj"] = torch.zeros(D + 1, 4)
    with pytest.raises(lx.LoraExportError, match="parent shapes differ"):
        lx._export_lora_subtraction({"kind": "lora_subtraction"}, _difference_blob(pos, neg),
                                    decoder_prefix=MOE_PREFIX)
    # a bare blob of another kind, or not a mapping
    with pytest.raises(lx.LoraExportError, match="must be a lora_all / lora_subtraction blob dict"):
        lx.export_module_set(tmp_path / "k", base_model=moe_snapshot, blob={"kind": "psr"})


# --------------------------------------------------------------------------
# vllm_server: the class off the files, and --moe-backend triton
# --------------------------------------------------------------------------


def test_adapted_module_class_infers_experts_and_the_launch_composes_the_moe_backend(
        moe_model, moe_snapshot, tmp_path):
    """``_adapted_module_class`` is the FILE's class (``adapter_target_class``)
    and nothing else — the file is the authoritative record of what the
    engine executes: an experts
    adapter's class carries ``experts`` though its config never names it,
    a config ``target_modules`` entry with NO factor behind it does NOT
    widen the class (the earlier config∪file union did, silently allocating
    LoRA buffers and running the kernels at wrappers no adapter adapts),
    and the launch composes ``--lora-target-modules <class>`` +
    ``--moe-backend triton`` + the routed-rank env knob for an experts class
    (the serving library's routed-expert OPTION); an o_proj launch composes
    no MoE flag and no knob. The launch is composed through the REAL
    kernel-ceiling read (``serving_attestation.adapter_module_dims``,
    generalised to the experts pairs by the attestation workstream)."""
    mlp = lx.export_module_set(tmp_path / "mlp", base_model=moe_snapshot,
                               blob=_hand_blob(moe_model, "mlp", 4, seed=1), intervention_hash="ab" * 8)
    zero_mlp, _, _ = lx.export_zero_adapter_for_class(tmp_path / "bank", base_model=moe_snapshot,
                                                     rank=1, lora_modules="mlp")
    wo = lx.export_module_set(tmp_path / "wo", base_model=moe_snapshot,
                              blob=_legacy_blob(7, layers=(3,)), intervention_hash="cd" * 8)
    assert vs._adapted_module_class({"m": str(mlp), "z": str(zero_mlp)}) == (
        "down_proj", "experts", "gate_proj", "up_proj")
    assert vs._adapted_module_class({"w": str(wo)}) == ("o_proj",)
    with pytest.raises(vs.VllmServerError, match="registered adapters disagree on their adapted module class"):
        vs._adapted_module_class({"m": str(mlp), "w": str(wo)})
    # the config declares q_proj too, the file carries o_proj factors only -> the class is the FILE's
    # ("o_proj",), never the union ("o_proj", "q_proj"); and it still registers beside a plain o_proj adapter
    wide = tmp_path / "wide"
    wide.mkdir()
    (wide / lx.ADAPTER_WEIGHTS).write_bytes((wo / lx.ADAPTER_WEIGHTS).read_bytes())
    wcfg = json.loads((wo / lx.ADAPTER_CONFIG).read_text())
    assert set(wcfg["target_modules"]) == {"o_proj"}
    wcfg["target_modules"] = ["o_proj", "q_proj"]
    (wide / lx.ADAPTER_CONFIG).write_text(json.dumps(wcfg, sort_keys=True))
    assert vs._adapted_module_class({"x": str(wide)}) == ("o_proj",)
    assert vs._adapted_module_class({"x": str(wide), "w": str(wo)}) == ("o_proj",)
    # the config is still read and still has to be a well-formed PEFT target list (vLLM's PEFTHelper reads it)
    bad = tmp_path / "badcfg"
    bad.mkdir()
    (bad / lx.ADAPTER_WEIGHTS).write_bytes((wo / lx.ADAPTER_WEIGHTS).read_bytes())
    (bad / lx.ADAPTER_CONFIG).write_text(json.dumps({**wcfg, "target_modules": []}, sort_keys=True))
    with pytest.raises(vs.VllmServerError, match="not a non-empty list of module suffixes"):
        vs._adapted_module_class({"b": str(bad)})
    # a file whose factors carry one experts half refuses at the class read
    from safetensors.torch import load_file as _lf
    half = tmp_path / "half"
    half.mkdir()
    t = _lf(str(mlp / lx.ADAPTER_WEIGHTS))
    save_file({k: v for k, v in t.items() if ".experts.lora_" not in k}, str(half / lx.ADAPTER_WEIGHTS))
    (half / lx.ADAPTER_CONFIG).write_bytes((mlp / lx.ADAPTER_CONFIG).read_bytes())
    with pytest.raises(vs.VllmServerError, match="cannot infer the adapted module class from its factors"):
        vs._adapted_module_class({"h": str(half)})

    def native(adapters):
        return vs.build_server_launch(_tpl_spec(serving_launch_flags=[]), port=8801,
                                      lora_modules=adapters, exact_replay_strategy="serialize",
                                      chat_template_path=_chat_template(tmp_path), lora_native=True)

    launch = native({"si_mlp__lora": mlp, "none__lora0": zero_mlp})
    cmd = list(launch.cmd)
    j = cmd.index("--lora-target-modules")
    assert cmd[j + 1:j + 7] == ["down_proj", "experts", "gate_proj", "up_proj", "--moe-backend", "triton"]
    assert cmd.count("--moe-backend") == 1 and cmd.count("--lora-target-modules") == 1
    assert launch.lora_target_modules == ("down_proj", "experts", "gate_proj", "up_proj")
    # the routed-expert buffer rank the worker allocates at — the adapters' own rank_pattern (r_e = max(1, 4 // 2) = 2
    # on the tiny top-2 MoE; the class zero adapter at rank 1 declares 1) — rides the launch and its env knob
    assert launch.moe_lora_rank == _r_e(4) == 2 and launch.env[vs.MOE_LORA_RANK_ENV] == "2"
    plain = native({"si_wo__lora": wo})
    assert "--moe-backend" not in plain.cmd and plain.lora_target_modules == ("o_proj",)
    assert plain.moe_lora_rank is None and vs.MOE_LORA_RANK_ENV not in plain.env
