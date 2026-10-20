"""The native adapter fingerprint over EVERY module class of the companion
repo's module sets, on CPU,
over vLLM-SHAPED fixtures: the packed q/k/v ``PackedLoRALayerWeights`` entry
(``MergedQKVParallelLinearWithLoRA``, output slices ``(q, kv, kv)``), the
packed gate/up entry on the shared expert (``MergedColumnParallelLinearWithLoRA``),
the routed experts' list-valued 3-D entry vLLM's ``_stack_moe_lora_weights``
leaves (``[A_gu [E, r_e, d], A_d [E, r_e, I]]`` / ``[B_gu [E, 2I, r_e], B_d
[E, d, r_e]]``) with its ``...experts.base_layer`` alias, a ``FusedMoE3DWithLoRA``-
shaped wrapper exposing the four stacked buffers, mixed ranks (attention R,
experts r_e), a missing expert binding (refused by name), the rank-doubled
difference; AND file-side adapters written by hand per contract §3 (fp32
safetensors + ``adapter_config.json`` with ``target_parameters``,
``rank_pattern``, ``petri_per_expert_rank``, ``petri_targets``) proving
``expected_adapter_fingerprint_bf16(dir) == adapter_fingerprint_bf16(loaded
fake)`` for every class. The o_proj fixtures' digests are pinned to the exact
shas computed on the tree BEFORE this generalisation (the paper's standups
are certified on them).

Synthetic geometry throughout: hidden d = 16, attention q_out = 24 (the
output gate doubles q), kv_out = 10 (packed qkv stride 44), o_proj 24 -> 16,
shared-expert intermediate I = 6 (gate/up 16 -> 6, down 6 -> 16), E = 4 routed
experts with intermediate 6 (gate_up [E, 12, r_e] / down [E, 16, r_e]).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import worker_reports as wr

# --------------------------------------------------------------------------
# the o_proj fixtures' digests on the tree BEFORE this generalisation
# --------------------------------------------------------------------------

#: ``tests/test_native_worker_reports.py::_fake_model(1, seed=1)`` /
#: ``_fake_model(2, seed=2)`` (GEMMA_VLLM spelling, layers (0, 1), r 4)
OPROJ_LOADED_SHA = {
    1: "9e15767742836d69875ae3fc6dbac9ec23e46095aa5ee015f10d8378cc21c5cb",
    2: "4767f0854b5c9cbf73d9730939c2d9be1482e938496b76c8c4c01c487e01b53f",
}
#: ``tests/test_petri_serving_gate.py::_tiny_peft_adapter`` with its defaults
#: (r 4, layers (0, 1), 24 -> 32, seed 0) and with ``r=4, layers=(0, 1, 3), seed=9``
OPROJ_FILE_SHA_DEFAULT = "42d91a5f8e631b96de13f157fab09e65cac4d1c7535f599a6f5e5726bcfcd598"
OPROJ_FILE_SHA_L013_S9 = "e74474e80d05a475573050ee891856819906b91da70d47a3bb345c48ee662e23"

GEMMA_VLLM_OPROJ = "language_model.model.layers.{}.self_attn.o_proj"


def _bf16(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(torch.bfloat16)


def _oproj_loaded_modules(seed, *, layers=(0, 1), rank=4):
    """``_fake_model`` of the pinned test, as ModuleWeights."""
    out = {}
    for i, layer in enumerate(layers):
        a = _bf16(rank, 24, seed=seed * 10 + i)
        b = _bf16(32, rank, seed=seed * 10 + 5 + i)
        out[GEMMA_VLLM_OPROJ.format(layer)] = wr.ModuleWeights(a, b, rank, rank, False)
    return out


def _tiny_peft_adapter(adapter_dir, *, r=4, layers=(0, 1), d_in=24, d_out=32, seed=0):
    """``tests/test_petri_serving_gate.py::_tiny_peft_adapter`` with the
    knobs that reach the fingerprint (the same generator draw order)."""
    from safetensors.torch import save_file

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in layers:
        m = f"base_model.model.model.layers.{layer}.self_attn.o_proj"
        tensors[f"{m}.lora_A.weight"] = torch.randn(r, d_in, generator=g)
        tensors[f"{m}.lora_B.weight"] = torch.randn(d_out, r, generator=g) * 0.1
    save_file(tensors, str(adapter_dir / sa.ADAPTER_WEIGHTS_FILE))
    (adapter_dir / sa.ADAPTER_CONFIG_FILE).write_text(json.dumps({
        "peft_type": "LORA", "r": r, "lora_alpha": float(r), "target_modules": ["o_proj"],
    }, indent=2, sort_keys=True))
    return adapter_dir


def test_oproj_fingerprints_are_byte_identical_to_the_certified_recipe(tmp_path):
    for seed in (1, 2):
        fp, keys = wr.adapter_fingerprint_bf16(_oproj_loaded_modules(seed))
        assert fp == OPROJ_LOADED_SHA[seed]
        assert keys == ("L0.self_attn.o_proj", "L1.self_attn.o_proj")
    assert wr.expected_adapter_fingerprint_bf16(_tiny_peft_adapter(tmp_path / "a")) == (
        OPROJ_FILE_SHA_DEFAULT, ("L0.self_attn.o_proj", "L1.self_attn.o_proj"))
    assert wr.expected_adapter_fingerprint_bf16(
        _tiny_peft_adapter(tmp_path / "b", r=4, layers=(0, 1, 3), seed=9)) == (
        OPROJ_FILE_SHA_L013_S9, ("L0.self_attn.o_proj", "L1.self_attn.o_proj", "L3.self_attn.o_proj"))
    # the zero adapter's un-deduplicated target_modules stays admitted
    d = _tiny_peft_adapter(tmp_path / "zero")
    cfg = json.loads((d / sa.ADAPTER_CONFIG_FILE).read_text())
    cfg["target_modules"] = ["o_proj", "o_proj"]
    (d / sa.ADAPTER_CONFIG_FILE).write_text(json.dumps(cfg))
    assert wr.expected_adapter_fingerprint_bf16(d)[0] == OPROJ_FILE_SHA_DEFAULT


# --------------------------------------------------------------------------
# the synthetic MoE geometry and the two sides' fixtures
# --------------------------------------------------------------------------

D, Q_OUT, KV_OUT, O_IN = 16, 24, 10, 24       # hidden, packed q / kv widths, o_proj input (no dim equals r 4 / 8)
I_SHARED, E, I_EXPERT = 6, 4, 6               # shared-expert intermediate, experts, expert intermediate
LAYERS = (3, 7)

FILE = "base_model.model.model.language_model.layers.{}.{}"       # HF / PEFT
VLLM = "language_model.model.layers.{}.{}"                       # post-mapper


def _f32(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _draw(*, r, r_e, seed, layers=LAYERS, attn=True, shared=True, experts=True):
    """Every factor of one synthetic module-set adapter, fp32, keyed by
    ``(layer, module)`` with the tensors in the BLOB / 3-D form: linear
    ``A [r, in]`` / ``B [out, r]``, experts ``A [E, r_e, in]`` / ``B [E, out, r_e]``."""
    out = {}
    k = seed * 1000
    for layer in layers:
        if attn:
            for name, d_out in (("q_proj", Q_OUT), ("k_proj", KV_OUT), ("v_proj", KV_OUT)):
                out[(layer, f"self_attn.{name}")] = (_f32(r, D, seed=k), _f32(d_out, r, seed=k + 1) * 0.1)
                k += 2
            out[(layer, "self_attn.o_proj")] = (_f32(r, O_IN, seed=k), _f32(D, r, seed=k + 1) * 0.1)
            k += 2
        if shared:
            for name, d_in, d_out in (("gate_proj", D, I_SHARED), ("up_proj", D, I_SHARED),
                                      ("down_proj", I_SHARED, D)):
                out[(layer, f"mlp.shared_expert.{name}")] = (
                    _f32(r, d_in, seed=k), _f32(d_out, r, seed=k + 1) * 0.1)
                k += 2
        if experts:
            out[(layer, "mlp.experts.gate_up_proj")] = (
                _f32(E, r_e, D, seed=k), _f32(E, 2 * I_EXPERT, r_e, seed=k + 1) * 0.1)
            out[(layer, "mlp.experts.down_proj")] = (
                _f32(E, r_e, I_EXPERT, seed=k + 2), _f32(E, D, r_e, seed=k + 3) * 0.1)
            k += 4
    return out


def _write_adapter(adapter_dir, factors, *, r, r_e=None, config_extra=None,
                   drop_config_keys=(), drop_modules=()):
    """Contract §3 on disk: fp32 safetensors, ``target_modules`` = the linear
    short names, ``target_parameters`` + ``rank_pattern`` / ``alpha_pattern``
    + ``petri_per_expert_rank`` + ``petri_targets`` when experts are adapted;
    experts pairs flattened exactly as the exporter does (``A.reshape(E*r_e,
    d)``, ``B.permute(1, 2, 0).reshape(out, r_e*E)``) under
    ``...mlp.experts.base_layer`` (gate_up) and ``...mlp.experts`` (down)."""
    from safetensors.torch import save_file

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tensors, linear_names, has_experts = {}, set(), False
    layers_by_module = {}
    for (layer, module), (a, b) in factors.items():
        if module in drop_modules:
            continue
        layers_by_module.setdefault(module, []).append(layer)
        if module.startswith("mlp.experts."):
            has_experts = True
            E_, re_, d_in = a.shape
            flat_a = a.reshape(E_ * re_, d_in)
            flat_b = b.permute(1, 2, 0).reshape(b.shape[1], re_ * E_)
            site = "mlp.experts.base_layer" if module.endswith("gate_up_proj") else "mlp.experts"
            tensors[FILE.format(layer, site) + ".lora_A.weight"] = flat_a.contiguous()
            tensors[FILE.format(layer, site) + ".lora_B.weight"] = flat_b.contiguous()
        else:
            linear_names.add(module.rsplit(".", 1)[-1])
            tensors[FILE.format(layer, module) + ".lora_A.weight"] = a
            tensors[FILE.format(layer, module) + ".lora_B.weight"] = b
    save_file(tensors, str(adapter_dir / sa.ADAPTER_WEIGHTS_FILE))
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": float(r), "lora_dropout": 0.0,
           "bias": "none", "target_modules": sorted(linear_names), "task_type": "CAUSAL_LM"}
    if has_experts:
        cfg["target_parameters"] = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
        cfg["rank_pattern"] = {r"mlp\.experts\.(gate_up|down)_proj": r_e}
        cfg["alpha_pattern"] = {r"mlp\.experts\.(gate_up|down)_proj": r_e}
        cfg["petri_per_expert_rank"] = {"shared_expert": r, "routed_expert": r_e}
        cfg["petri_targets"] = {
            "experts.gate_up_proj": {"site": "mlp.experts.gate_up_proj", "kind": "experts",
                                     "layers": sorted(layers_by_module.get("mlp.experts.gate_up_proj", [])),
                                     "rank": r_e, "num_experts": E},
            "experts.down_proj": {"site": "mlp.experts.down_proj", "kind": "experts",
                                  "layers": sorted(layers_by_module.get("mlp.experts.down_proj", [])),
                                  "rank": r_e, "num_experts": E},
        }
    for key in drop_config_keys:
        cfg.pop(key, None)
    if config_extra:
        cfg.update(config_extra)
    (adapter_dir / sa.ADAPTER_CONFIG_FILE).write_text(json.dumps(cfg, indent=2, sort_keys=True))
    return adapter_dir


class _W:
    """``LoRALayerWeights`` @0.25.1 (a stacked experts entry keeps this class
    with LIST-valued 3-D factors and ``is_packed=False``)."""

    def __init__(self, a, b, rank, alpha=None):
        self.lora_a, self.lora_b, self.rank = a, b, rank
        self.lora_alpha = rank if alpha is None else alpha
        self.is_packed = False


class _PackedW:
    """``PackedLoRALayerWeights`` @0.25.1: list-valued factors (``None`` in an
    unadapted slot), ``lora_alpha=0`` at the top, the per-component alphas in
    ``lora_alphas``, ``is_packed=True``."""

    def __init__(self, a_list, b_list, rank, alphas=None):
        self.lora_a, self.lora_b, self.rank = list(a_list), list(b_list), rank
        self.lora_alpha = 0
        self.lora_alphas = list(alphas) if alphas is not None else [
            (rank if a is not None else None) for a in a_list]
        self.is_packed = True


class _LoRAModel:
    def __init__(self, lora_id, rank, loras):
        self.id, self.rank, self.loras = lora_id, rank, loras


def _loaded_fake(lora_id, factors, *, r, spelling=VLLM, pack=True, alias=True):
    """The worker's ``LoRAModel.loras`` for one synthetic adapter AFTER vLLM's
    ``_create_merged_loras_inplace`` + ``_stack_moe_lora_weights``: q/k/v
    packed under ``qkv_proj`` (slice order q, k, v), gate/up under the shared
    expert's ``gate_up_proj``, the experts under ``...mlp.experts`` as
    ``[gate_up, down]`` 3-D lists with the ``...mlp.experts.base_layer`` alias
    still present (the stacker mutates it in place and never pops it); one
    fp32 -> bf16 cast on every tensor. ``pack=False`` leaves q/k/v and
    gate/up as plain per-component entries (an engine that does not pack)."""
    bf = {k: (a.to(torch.bfloat16), b.to(torch.bfloat16)) for k, (a, b) in factors.items()}
    loras = {}
    layers = sorted({layer for layer, _ in factors})
    for layer in layers:
        def has(module):
            return (layer, module) in bf
        if pack and any(has(f"self_attn.{c}") for c in ("q_proj", "k_proj", "v_proj")):
            comps = [bf.get((layer, f"self_attn.{c}")) for c in ("q_proj", "k_proj", "v_proj")]
            loras[spelling.format(layer, "self_attn.qkv_proj")] = _PackedW(
                [c[0] if c else None for c in comps], [c[1] if c else None for c in comps], r)
        else:
            for c in ("q_proj", "k_proj", "v_proj"):
                if has(f"self_attn.{c}"):
                    loras[spelling.format(layer, f"self_attn.{c}")] = _W(*bf[(layer, f"self_attn.{c}")], r)
        if has("self_attn.o_proj"):
            loras[spelling.format(layer, "self_attn.o_proj")] = _W(*bf[(layer, "self_attn.o_proj")], r)
        if pack and any(has(f"mlp.shared_expert.{c}") for c in ("gate_proj", "up_proj")):
            comps = [bf.get((layer, f"mlp.shared_expert.{c}")) for c in ("gate_proj", "up_proj")]
            loras[spelling.format(layer, "mlp.shared_expert.gate_up_proj")] = _PackedW(
                [c[0] if c else None for c in comps], [c[1] if c else None for c in comps], r)
        else:
            for c in ("gate_proj", "up_proj"):
                if has(f"mlp.shared_expert.{c}"):
                    loras[spelling.format(layer, f"mlp.shared_expert.{c}")] = _W(
                        *bf[(layer, f"mlp.shared_expert.{c}")], r)
        if has("mlp.shared_expert.down_proj"):
            loras[spelling.format(layer, "mlp.shared_expert.down_proj")] = _W(
                *bf[(layer, "mlp.shared_expert.down_proj")], r)
        if has("mlp.experts.gate_up_proj") or has("mlp.experts.down_proj"):
            gu = bf.get((layer, "mlp.experts.gate_up_proj"))
            dn = bf.get((layer, "mlp.experts.down_proj"))
            loras[spelling.format(layer, "mlp.experts")] = _W(
                [gu[0] if gu else None, dn[0] if dn else None],
                [gu[1] if gu else None, dn[1] if dn else None], r)
            if alias and gu:
                loras[spelling.format(layer, "mlp.experts.base_layer")] = _W(gu[0], gu[1], r)
    return _LoRAModel(lora_id, r, loras)


def _as_module_weights(model):
    return {n: wr.ModuleWeights(w.lora_a, w.lora_b, w.rank, w.lora_alpha, w.is_packed,
                                getattr(w, "lora_alphas", None)) for n, w in model.loras.items()}


class _Wrap:
    """``BaseLinearLayerWithLoRA`` @0.25.1 with its stacked buffers: ``input_size``,
    ``output_slices``, ``lora_a_stacked`` (one ``[max_loras, 1, R, in]`` per slice),
    ``lora_b_stacked`` (one ``[max_loras, 1, out_i, R]`` per slice)."""

    def __init__(self, input_size, output_slices, *, max_loras=2, max_lora_rank=8, buffers=True):
        self.input_size = input_size
        self.output_slices = tuple(output_slices)
        if buffers:
            self.lora_a_stacked = tuple(torch.zeros(max_loras, 1, max_lora_rank, input_size,
                                                    dtype=torch.bfloat16) for _ in output_slices)
            self.lora_b_stacked = tuple(torch.zeros(max_loras, 1, o, max_lora_rank,
                                                    dtype=torch.bfloat16) for o in output_slices)


class _MoEWrap:
    """``FusedMoE3DWithLoRA`` @0.25.1: NO ``input_size`` / ``output_slices``; the
    four stacked buffers as tuples of ONE tensor — ``w13_lora_a [n, E, R, d]``,
    ``w13_lora_b [n, E, 2I, R]``, ``w2_lora_a [n, E, R, I]``, ``w2_lora_b [n, E, d, R]``
    — plus ``hidden_size`` / ``intermediate_size_per_partition`` / ``local_num_experts``."""

    def __init__(self, *, max_loras=2, num_experts=E, max_lora_rank=8, hidden=D,
                 intermediate=I_EXPERT, attrs=True, shapes=None):
        n, R = max_loras, max_lora_rank
        sh = shapes or {
            "w13_lora_a_stacked": (n, num_experts, R, hidden),
            "w13_lora_b_stacked": (n, num_experts, 2 * intermediate, R),
            "w2_lora_a_stacked": (n, num_experts, R, intermediate),
            "w2_lora_b_stacked": (n, num_experts, hidden, R),
        }
        for attr, shape in sh.items():
            setattr(self, attr, (torch.zeros(*shape, dtype=torch.bfloat16),))
        if attrs:
            self.hidden_size = hidden
            self.intermediate_size_per_partition = intermediate
            self.local_num_experts = num_experts


def _registry(*, layers=LAYERS, spelling=VLLM, max_loras=2, max_lora_rank=8, drop=(), buffers=True):
    """The execution registry of a healthy Qwen3.5-shaped engine restricted to
    the attn_mlp class: per layer the packed qkv wrapper, o_proj, the shared
    expert's packed gate_up and down, and the MoE wrapper."""
    mods = {}
    for layer in layers:
        mods[spelling.format(layer, "self_attn.qkv_proj")] = _Wrap(
            D, (Q_OUT, KV_OUT, KV_OUT), max_loras=max_loras, max_lora_rank=max_lora_rank, buffers=buffers)
        mods[spelling.format(layer, "self_attn.o_proj")] = _Wrap(
            O_IN, (D,), max_loras=max_loras, max_lora_rank=max_lora_rank, buffers=buffers)
        mods[spelling.format(layer, "mlp.shared_expert.gate_up_proj")] = _Wrap(
            D, (I_SHARED, I_SHARED), max_loras=max_loras, max_lora_rank=max_lora_rank, buffers=buffers)
        mods[spelling.format(layer, "mlp.shared_expert.down_proj")] = _Wrap(
            I_SHARED, (D,), max_loras=max_loras, max_lora_rank=max_lora_rank, buffers=buffers)
        mods[spelling.format(layer, "mlp.experts")] = _MoEWrap(max_loras=max_loras, max_lora_rank=max_lora_rank)
    for name in drop:
        mods.pop(name)
    return mods


class _AdapterManager:
    def __init__(self, models, modules):
        self._models = {m.id: m for m in models}
        self.modules = modules

    def get_adapter(self, lora_id):
        return self._models.get(lora_id)

    def _get_lora_layer_weights(self, lora_model, module_name):
        return lora_model.loras.get(module_name)


class _WorkerLoRAManager:
    def __init__(self, models, modules):
        self._adapter_manager = _AdapterManager(models, modules)

    def list_adapters(self):
        return set(self._adapter_manager._models)


class _Runner:
    def __init__(self, models, modules):
        self.lora_manager = _WorkerLoRAManager(models, modules)


def _keys(layers, *, attn=True, shared=True, experts=True):
    out = []
    for layer in layers:
        if attn:
            out += [f"L{layer}.self_attn.{c}" for c in ("k_proj", "o_proj", "q_proj", "v_proj")]
        if shared:
            out += [f"L{layer}.mlp.shared_expert.{c}" for c in ("down_proj", "gate_proj", "up_proj")]
        if experts:
            out += [f"L{layer}.mlp.experts.down_proj", f"L{layer}.mlp.experts.gate_up_proj"]
    return tuple(sorted(out))


# --------------------------------------------------------------------------
# the vocabulary
# --------------------------------------------------------------------------


def test_module_vocabulary_keys_kinds_and_carriers():
    for name, key in (
        (FILE.format(3, "self_attn.q_proj"), "L3.self_attn.q_proj"),
        (VLLM.format(3, "self_attn.qkv_proj"), "L3.self_attn.qkv_proj"),
        ("model.layers.3.self_attn.o_proj", "L3.self_attn.o_proj"),
        ("model.layers.24.mixer.o_proj", "L24.mixer.o_proj"),
        ("model.layers.24.mixer.qkv_proj", "L24.mixer.qkv_proj"),
        ("model.layers.24.mixer.up_proj", "L24.mixer.up_proj"),
        ("model.layers.5.mlp.gate_proj", "L5.mlp.gate_proj"),
        ("model.layers.5.mlp.gate_up_proj", "L5.mlp.gate_up_proj"),
        (FILE.format(3, "mlp.shared_expert.down_proj"), "L3.mlp.shared_expert.down_proj"),
        (VLLM.format(3, "mlp.shared_expert.gate_up_proj"), "L3.mlp.shared_expert.gate_up_proj"),
        (FILE.format(3, "mlp.experts.base_layer"), "L3.mlp.experts.gate_up_proj"),
        (FILE.format(3, "mlp.experts"), "L3.mlp.experts"),
        (VLLM.format(3, "mlp.experts"), "L3.mlp.experts"),
    ):
        assert sa.module_key_of(name) == key, name
        assert wr.canonical_module_key(name) == key
    for bad in ("model.layers.3.mlp.gate", "model.layers.3.mlp.shared_expert_gate",
                "model.layers.3.linear_attn.in_proj_qkvz", "model.layers.3.linear_attn.conv1d",
                "model.layers.3.mlp.experts.0.gate_proj", "layers.3.mlp.experts.gate_up_proj",
                "model.layers.x.self_attn.q_proj", "", None,
                "model.layers.3.feed_forward.gate", "model.layers.3.block_sparse_moe.router",
                "model.layers.3.ffn.gate_proj"):                       # `ffn` is NOT in the vocabulary
        assert sa.module_key_of(bad) is None, bad
        with pytest.raises(wr.FingerprintError, match="not a recognised adapted module|not a non-empty"):
            wr.canonical_module_key(bad)
    # ONE FFN-parent vocabulary: every parent the fitter's capability probe admits
    # (extraction.lora_modules._ffn_layout reads sa.FFN_PARENTS) keys here too — dense,
    # shared-expert and both routed-experts spellings — so a set the fitter adapts on a `feed_forward` /
    # `block_sparse_moe` block is never refused at readiness by a fingerprint that only knew `mlp`
    from sisteer.extraction import lora_modules as lm
    assert lm.FFN_PARENTS is sa.FFN_PARENTS
    assert sa.FFN_PARENTS == ("mlp", "feed_forward", "block_sparse_moe", "mixer")
    for parent in sa.FFN_PARENTS:
        assert sa.module_key_of(f"model.layers.2.{parent}.gate_proj") == f"L2.{parent}.gate_proj"
        assert sa.module_key_of(FILE.format(2, f"{parent}.shared_expert.down_proj")) == f"L2.{parent}.shared_expert.down_proj"
        assert sa.module_key_of(VLLM.format(2, f"{parent}.shared_expert.gate_up_proj")) == f"L2.{parent}.shared_expert.gate_up_proj"
        assert sa.module_key_of(FILE.format(2, f"{parent}.experts.base_layer")) == f"L2.{parent}.experts.gate_up_proj"
        assert sa.module_key_of(VLLM.format(2, f"{parent}.experts")) == f"L2.{parent}.experts"
        assert sa.module_kind_of_key(f"L2.{parent}.experts") == sa.MODULE_KIND_EXPERTS
        assert sa.module_kind_of_key(f"L2.{parent}.experts.down_proj") == sa.MODULE_KIND_EXPERTS_PROJECTION
        assert sa.module_kind_of_key(f"L2.{parent}.shared_expert.gate_up_proj") == sa.MODULE_KIND_PACKED
        assert sa.execution_carrier_key(f"L2.{parent}.experts.down_proj") == f"L2.{parent}.experts"
        assert sa.execution_carrier_key(f"L2.{parent}.shared_expert.up_proj") == f"L2.{parent}.shared_expert.gate_up_proj"
    kinds = sa.module_kind_of_key
    assert kinds("L3.self_attn.o_proj") == sa.MODULE_KIND_LINEAR
    assert kinds("L3.self_attn.q_proj") == sa.MODULE_KIND_LINEAR
    assert kinds("L3.self_attn.qkv_proj") == sa.MODULE_KIND_PACKED
    assert kinds("L3.mlp.shared_expert.gate_up_proj") == sa.MODULE_KIND_PACKED
    assert kinds("L3.mlp.experts") == sa.MODULE_KIND_EXPERTS
    assert kinds("L3.mlp.experts.gate_up_proj") == sa.MODULE_KIND_EXPERTS_PROJECTION
    assert kinds("L3.mlp.experts.down_proj") == sa.MODULE_KIND_EXPERTS_PROJECTION
    assert sa.packed_component_keys("L3.self_attn.qkv_proj") == (
        "L3.self_attn.q_proj", "L3.self_attn.k_proj", "L3.self_attn.v_proj")
    assert sa.packed_component_keys("L3.mlp.shared_expert.gate_up_proj") == (
        "L3.mlp.shared_expert.gate_proj", "L3.mlp.shared_expert.up_proj")
    assert sa.experts_projection_keys("L3.mlp.experts") == (
        "L3.mlp.experts.gate_up_proj", "L3.mlp.experts.down_proj")
    with pytest.raises(sa.ServingAttestationError):
        sa.packed_component_keys("L3.self_attn.o_proj")
    carrier = sa.execution_carrier_key
    assert carrier("L3.self_attn.k_proj") == "L3.self_attn.qkv_proj"
    assert carrier("L3.mlp.shared_expert.up_proj") == "L3.mlp.shared_expert.gate_up_proj"
    assert carrier("L5.mlp.gate_proj") == "L5.mlp.gate_up_proj"
    assert carrier("L3.mlp.experts.down_proj") == carrier("L3.mlp.experts.gate_up_proj") == "L3.mlp.experts"
    for own in ("L3.self_attn.o_proj", "L3.mlp.shared_expert.down_proj", "L3.self_attn.qkv_proj",
                "L3.mlp.experts", "L24.mixer.o_proj"):
        assert carrier(own) == own
    keys = _keys((3,))
    assert sa.carried_keys("L3.self_attn.qkv_proj", keys) == [
        "L3.self_attn.k_proj", "L3.self_attn.q_proj", "L3.self_attn.v_proj"]
    assert sa.carried_keys("L3.mlp.experts", keys) == ["L3.mlp.experts.down_proj", "L3.mlp.experts.gate_up_proj"]
    assert sa.carried_keys("L3.self_attn.o_proj", keys) == ["L3.self_attn.o_proj"]
    assert sa.carried_keys("L3.self_attn.q_proj", keys) == ["L3.self_attn.q_proj"]     # an unpacked engine
    assert sa.carried_keys("L7.self_attn.qkv_proj", keys) == [] and sa.carried_keys("L0", keys) == []
    # vLLM's is_in_target_modules over the composed class
    cls = ("k_proj", "o_proj", "q_proj", "v_proj", "down_proj", "gate_proj", "up_proj", "experts")
    assert sa.is_in_target_class(VLLM.format(3, "self_attn.qkv_proj"), cls)
    assert sa.is_in_target_class(VLLM.format(3, "mlp.shared_expert.gate_up_proj"), cls)
    assert sa.is_in_target_class(VLLM.format(3, "mlp.experts"), cls)
    assert not sa.is_in_target_class(VLLM.format(3, "mlp.experts"), cls[:-1])
    assert not sa.is_in_target_class(VLLM.format(3, "self_attn.qkv_proj"), ("o_proj",))
    assert not sa.is_in_target_class(VLLM.format(3, "linear_attn.in_proj_qkvz"), cls)
    assert sa.is_in_target_class("model.layers.0.self_attn.q_proj", ("qkv_proj",))    # child by parent


# --------------------------------------------------------------------------
# the fingerprint, class by class: loaded (vLLM-shaped) == file (contract §3)
# --------------------------------------------------------------------------


def test_packed_qkv_unpacks_into_component_keys_in_slice_order(tmp_path):
    fac = _draw(r=4, r_e=2, seed=1, shared=False, experts=False)
    packed = _as_module_weights(_loaded_fake(1, fac, r=4))
    assert set(packed) == {VLLM.format(l, "self_attn.qkv_proj") for l in LAYERS} | {
        VLLM.format(l, "self_attn.o_proj") for l in LAYERS}
    fp, keys = wr.adapter_fingerprint_bf16(packed)
    assert keys == _keys(LAYERS, shared=False, experts=False)
    # an engine that does not pack (per-component entries) hashes the same projections
    unpacked = _as_module_weights(_loaded_fake(1, fac, r=4, pack=False))
    assert wr.adapter_fingerprint_bf16(unpacked) == (fp, keys)
    # the file side agrees (HF/PEFT spelling, fp32 -> one bf16 cast)
    d = _write_adapter(tmp_path / "qkvo", fac, r=4)
    cfg = json.loads((d / sa.ADAPTER_CONFIG_FILE).read_text())
    assert cfg["target_modules"] == ["k_proj", "o_proj", "q_proj", "v_proj"]
    assert "target_parameters" not in cfg and "rank_pattern" not in cfg
    assert wr.expected_adapter_fingerprint_bf16(d) == (fp, keys)
    # slice ORDER is inside the fingerprint: swapping k and v in the packed lists changes it
    swapped = dict(packed)
    w = swapped[VLLM.format(3, "self_attn.qkv_proj")]
    swapped[VLLM.format(3, "self_attn.qkv_proj")] = w._replace(
        lora_a=[w.lora_a[0], w.lora_a[2], w.lora_a[1]], lora_b=[w.lora_b[0], w.lora_b[2], w.lora_b[1]])
    assert wr.adapter_fingerprint_bf16(swapped)[0] != fp
    # an absent component (None in both slots) is fine: a q-only packed entry
    q_only = {VLLM.format(3, "self_attn.qkv_proj"): wr.ModuleWeights(
        [w.lora_a[0], None, None], [w.lora_b[0], None, None], 4, 0, True, [4, None, None])}
    assert wr.adapter_fingerprint_bf16(q_only)[1] == ("L3.self_attn.q_proj",)


def test_packed_gate_up_on_the_shared_expert_and_dense_mlp(tmp_path):
    fac = _draw(r=4, r_e=2, seed=2, attn=False, experts=False)
    loaded = _as_module_weights(_loaded_fake(1, fac, r=4))
    assert set(loaded) == {VLLM.format(l, "mlp.shared_expert.gate_up_proj") for l in LAYERS} | {
        VLLM.format(l, "mlp.shared_expert.down_proj") for l in LAYERS}
    fp, keys = wr.adapter_fingerprint_bf16(loaded)
    assert keys == _keys(LAYERS, attn=False, experts=False)
    d = _write_adapter(tmp_path / "mlp", fac, r=4)
    assert json.loads((d / sa.ADAPTER_CONFIG_FILE).read_text())["target_modules"] == [
        "down_proj", "gate_proj", "up_proj"]
    assert wr.expected_adapter_fingerprint_bf16(d) == (fp, keys)
    # the dense spelling (gemma: mlp.gate_up_proj over mlp.gate_proj / mlp.up_proj) too
    a_g, b_g = _bf16(4, D, seed=11), _bf16(I_SHARED, 4, seed=12)
    a_u, b_u = _bf16(4, D, seed=13), _bf16(I_SHARED, 4, seed=14)
    dense_packed = {"language_model.model.layers.0.mlp.gate_up_proj": wr.ModuleWeights(
        [a_g, a_u], [b_g, b_u], 4, 0, True, [4, 4])}
    dense_plain = {"base_model.model.model.layers.0.mlp.gate_proj": wr.ModuleWeights(a_g, b_g, 4, 4, False),
                   "base_model.model.model.layers.0.mlp.up_proj": wr.ModuleWeights(a_u, b_u, 4, 4, False)}
    assert wr.adapter_fingerprint_bf16(dense_packed) == wr.adapter_fingerprint_bf16(dense_plain)
    assert wr.adapter_fingerprint_bf16(dense_packed)[1] == ("L0.mlp.gate_proj", "L0.mlp.up_proj")


def test_experts_3d_list_entry_with_its_base_layer_alias_equals_the_flat_file(tmp_path):
    fac = _draw(r=4, r_e=2, seed=3, attn=False, shared=False)
    model = _loaded_fake(1, fac, r=4)
    # the worker's loras after _stack_moe_lora_weights: the wrapper entry's [gate_up, down]
    # 3-D lists AND the .base_layer alias pointing at the same reshaped tensors
    assert set(model.loras) == {VLLM.format(l, "mlp.experts") for l in LAYERS} | {
        VLLM.format(l, "mlp.experts.base_layer") for l in LAYERS}
    w = model.loras[VLLM.format(3, "mlp.experts")]
    assert [tuple(t.shape) for t in w.lora_a] == [(E, 2, D), (E, 2, I_EXPERT)]
    assert [tuple(t.shape) for t in w.lora_b] == [(E, 2 * I_EXPERT, 2), (E, D, 2)]
    fp, keys = wr.adapter_fingerprint_bf16(_as_module_weights(model))
    assert keys == _keys(LAYERS, attn=False, shared=False)
    # without the alias: identical (the alias adds no projection)
    no_alias = _as_module_weights(_loaded_fake(1, fac, r=4, alias=False))
    assert wr.adapter_fingerprint_bf16(no_alias) == (fp, keys)
    # the file: flat [E*r_e, d] / [out, r_e*E] pairs under experts.base_layer / experts
    d = _write_adapter(tmp_path / "experts", fac, r=4, r_e=2,
                       config_extra={"target_modules": ["down_proj"]})   # a `down`-style config
    from safetensors import safe_open
    with safe_open(str(d / sa.ADAPTER_WEIGHTS_FILE), framework="pt") as fh:
        shapes = {k: tuple(fh.get_slice(k).get_shape()) for k in fh.keys()}
    assert shapes[FILE.format(3, "mlp.experts.base_layer") + ".lora_A.weight"] == (E * 2, D)
    assert shapes[FILE.format(3, "mlp.experts.base_layer") + ".lora_B.weight"] == (2 * I_EXPERT, 2 * E)
    assert shapes[FILE.format(3, "mlp.experts") + ".lora_A.weight"] == (E * 2, I_EXPERT)
    assert shapes[FILE.format(3, "mlp.experts") + ".lora_B.weight"] == (D, 2 * E)
    # ... but the file's target_modules must name what its linear factors adapt: none here
    with pytest.raises(wr.FingerprintError, match=r"target_modules=\['down_proj'\].*\(\[\]\)"):
        wr.expected_adapter_fingerprint_bf16(d)
    # with a shared-expert down_proj beside the experts the `down` set's file is admitted
    fac_down = {k: v for k, v in _draw(r=4, r_e=2, seed=3, attn=False).items()
                if k[1] in ("mlp.shared_expert.down_proj", "mlp.experts.gate_up_proj", "mlp.experts.down_proj")}
    d2 = _write_adapter(tmp_path / "down", fac_down, r=4, r_e=2)
    fp2, keys2 = wr.expected_adapter_fingerprint_bf16(d2)
    assert keys2 == tuple(sorted(_keys(LAYERS, attn=False, shared=False)
                                 + tuple(f"L{l}.mlp.shared_expert.down_proj" for l in LAYERS)))
    assert (fp2, keys2) == wr.adapter_fingerprint_bf16(_as_module_weights(_loaded_fake(1, fac_down, r=4)))
    # the alias with DIFFERENT tensors is a genuine collision
    bad = _as_module_weights(model)
    alias = VLLM.format(3, "mlp.experts.base_layer")
    bad[alias] = bad[alias]._replace(lora_a=bad[alias].lora_a.clone() * 2)
    with pytest.raises(wr.FingerprintError, match="two modules canonicalise to 'L3.mlp.experts.gate_up_proj'"):
        wr.adapter_fingerprint_bf16(bad)
    # the 3-D digest carries E, r_e and both widths in its header: a different E with the same bytes differs
    w3 = no_alias[VLLM.format(3, "mlp.experts")]
    a_gu, a_d = w3.lora_a
    b_gu, b_d = w3.lora_b
    reshaped = dict(no_alias)
    reshaped[VLLM.format(3, "mlp.experts")] = w3._replace(
        lora_a=[a_gu.reshape(2, E, D), a_d.reshape(2, E, I_EXPERT)],
        lora_b=[b_gu.reshape(2, 2 * I_EXPERT, E), b_d.reshape(2, D, E)])
    assert wr.adapter_fingerprint_bf16(reshaped)[0] != fp


def test_mixed_ranks_attention_R_experts_r_e_agree_on_both_sides(tmp_path):
    """The full attn_mlp shape at r = 4 with per-expert rank 2: the file's top-level
    ``r`` sizes every linear factor, the experts' physical rank comes from the shapes
    (rows // E), and E resolves from any one of the three config sources."""
    fac = _draw(r=4, r_e=2, seed=4)
    model = _loaded_fake(1, fac, r=4)
    fp, keys = wr.adapter_fingerprint_bf16(_as_module_weights(model))
    assert keys == _keys(LAYERS) and len(keys) == 2 * 9
    d = _write_adapter(tmp_path / "attn_mlp", fac, r=4, r_e=2)
    cfg = json.loads((d / sa.ADAPTER_CONFIG_FILE).read_text())
    assert cfg["target_modules"] == ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
    assert cfg["target_parameters"] == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    assert wr.expected_adapter_fingerprint_bf16(d) == (fp, keys)
    # E from rank_pattern alone (PEFT's get_pattern_key rule), from petri_per_expert_rank alone
    d_rp = _write_adapter(tmp_path / "rank_pattern_only", fac, r=4, r_e=2,
                          drop_config_keys=("petri_targets", "petri_per_expert_rank"))
    assert wr.expected_adapter_fingerprint_bf16(d_rp) == (fp, keys)
    d_pe = _write_adapter(tmp_path / "per_expert_only", fac, r=4, r_e=2,
                          drop_config_keys=("petri_targets", "rank_pattern", "alpha_pattern"))
    assert wr.expected_adapter_fingerprint_bf16(d_pe) == (fp, keys)
    d_pe_int = _write_adapter(tmp_path / "per_expert_int", fac, r=4, r_e=2,
                              drop_config_keys=("petri_targets", "rank_pattern", "alpha_pattern"),
                              config_extra={"petri_per_expert_rank": 2})
    assert wr.expected_adapter_fingerprint_bf16(d_pe_int) == (fp, keys)
    # no source pins E: refused by name (never guessed)
    d_none = _write_adapter(tmp_path / "no_E", fac, r=4, r_e=2,
                            drop_config_keys=("petri_targets", "rank_pattern", "alpha_pattern",
                                              "petri_per_expert_rank"))
    with pytest.raises(wr.FingerprintError, match="cannot pin the expert count E"):
        wr.expected_adapter_fingerprint_bf16(d_none)
    # a wrong E (petri_targets says 8 over 8 rows at r_e 2 -> r_e 1): the digest header differs
    d_wrong = _write_adapter(tmp_path / "wrong_E", fac, r=4, r_e=2)
    cfg = json.loads((d_wrong / sa.ADAPTER_CONFIG_FILE).read_text())
    for t in cfg["petri_targets"].values():
        t["num_experts"] = 8
    (d_wrong / sa.ADAPTER_CONFIG_FILE).write_text(json.dumps(cfg))
    assert wr.expected_adapter_fingerprint_bf16(d_wrong)[0] != fp
    # ... and an E that does not divide the rows refuses
    for t in cfg["petri_targets"].values():
        t["num_experts"] = 3
    (d_wrong / sa.ADAPTER_CONFIG_FILE).write_text(json.dumps(cfg))
    with pytest.raises(wr.FingerprintError, match="not a multiple of num_experts=3"):
        wr.expected_adapter_fingerprint_bf16(d_wrong)
    # the loaded report over the same fake: keys, n_modules, dtype, raw shapes (lists ride along)
    rep = wr.loaded_adapters_report(_Runner([model], _registry()))
    entry = rep["lora_adapters"][0]
    assert entry["fingerprint_bf16"] == fp and entry["module_keys"] == list(keys)
    assert entry["n_modules"] == 18 and entry["tensor_dtype"] == "torch.bfloat16" and entry["rank"] == 4
    assert entry["raw_shapes"][VLLM.format(3, "mlp.experts")] == {
        "lora_a": [[E, 2, D], [E, 2, I_EXPERT]], "lora_b": [[E, 2 * I_EXPERT, 2], [E, D, 2]]}
    assert entry["raw_shapes"][VLLM.format(3, "self_attn.qkv_proj")]["lora_b"] == [[Q_OUT, 4], [KV_OUT, 4], [KV_OUT, 4]]


def test_rank_doubled_difference_2R_attention_2r_e_experts(tmp_path):
    """The difference adapter: top-level r = 2R, linear factors 2R, experts 2 r_e
    (concatenated along the per-expert rank axis before flattening, contract §3)."""
    pos, neg = _draw(r=4, r_e=2, seed=5), _draw(r=4, r_e=2, seed=6)
    diff = {}
    for k, (a_p, b_p) in pos.items():
        a_n, b_n = neg[k]
        if k[1].startswith("mlp.experts."):
            diff[k] = (torch.cat([a_p, a_n], dim=1), torch.cat([b_p, -b_n], dim=2))
        else:
            diff[k] = (torch.cat([a_p, a_n], dim=0), torch.cat([b_p, -b_n], dim=1))
    model = _loaded_fake(1, diff, r=8)
    fp, keys = wr.adapter_fingerprint_bf16(_as_module_weights(model))
    assert keys == _keys(LAYERS)
    d = _write_adapter(tmp_path / "diff", diff, r=8, r_e=4)
    cfg = json.loads((d / sa.ADAPTER_CONFIG_FILE).read_text())
    assert cfg["r"] == 8 and cfg["lora_alpha"] == 8.0
    assert wr.expected_adapter_fingerprint_bf16(d) == (fp, keys)
    # the parents' fingerprints differ from the difference's and from each other
    fps = {wr.adapter_fingerprint_bf16(_as_module_weights(_loaded_fake(1, f, r=4)))[0] for f in (pos, neg)}
    assert len(fps) == 2 and fp not in fps
    # a rank-8 file over rank-4 linear factors: the linear physical rank must equal r
    bad = _write_adapter(tmp_path / "r8_over_r4", pos, r=8, r_e=2)
    with pytest.raises(wr.FingerprintError, match="neither axis equals r=8"):
        wr.expected_adapter_fingerprint_bf16(bad)


# --------------------------------------------------------------------------
# refusals, by name
# --------------------------------------------------------------------------


def test_missing_expert_binding_refuses_by_name(tmp_path):
    fac = _draw(r=4, r_e=2, seed=7, attn=False, shared=False)
    # worker: a None slot in the stacked list (down missing)
    m = _as_module_weights(_loaded_fake(1, fac, r=4, alias=False))
    w = m[VLLM.format(3, "mlp.experts")]
    m[VLLM.format(3, "mlp.experts")] = w._replace(lora_a=[w.lora_a[0], None], lora_b=[w.lora_b[0], None])
    with pytest.raises(wr.FingerprintError, match="mlp.experts.*down_proj slot .* is None — a missing expert binding"):
        wr.adapter_fingerprint_bf16(m)
    # worker: only the gate_up (base_layer) entry — no stacked wrapper entry at all
    gu_only = {VLLM.format(3, "mlp.experts.base_layer"): wr.ModuleWeights(
        w.lora_a[0], w.lora_b[0], 4, 4, False)}
    with pytest.raises(wr.FingerprintError, match=r"L3.mlp.experts has \['gate_up_proj'\].*no \['down_proj'\] factor — a missing expert binding"):
        wr.adapter_fingerprint_bf16(gu_only)
    # worker: only a 3-D down pair under the wrapper name (a launcher-shaped entry without gate_up)
    dn_only = {VLLM.format(3, "mlp.experts"): wr.ModuleWeights(w.lora_a[1], w.lora_b[1], 4, 4, False)}
    with pytest.raises(wr.FingerprintError, match=r"has \['down_proj'\].*no \['gate_up_proj'\]"):
        wr.adapter_fingerprint_bf16(dn_only)
    # file: one pair only per layer, either way
    for drop, missing in (("mlp.experts.gate_up_proj", "gate_up_proj"), ("mlp.experts.down_proj", "down_proj")):
        d = _write_adapter(tmp_path / f"drop_{missing}", _draw(r=4, r_e=2, seed=7, attn=False), r=4, r_e=2,
                           drop_modules=(drop,))
        with pytest.raises(wr.FingerprintError, match=f"no \\['{missing}'\\] factor — a missing expert binding"):
            wr.expected_adapter_fingerprint_bf16(d)
    # the two projections of one block disagree on the hidden size
    m = _as_module_weights(_loaded_fake(1, fac, r=4, alias=False))
    w = m[VLLM.format(3, "mlp.experts")]
    m[VLLM.format(3, "mlp.experts")] = w._replace(lora_b=[w.lora_b[0], _bf16(E, D + 1, 2, seed=1)])
    with pytest.raises(wr.FingerprintError, match="input width 16 != down .* output width 17"):
        wr.adapter_fingerprint_bf16(m)


def test_fingerprint_refusals_over_the_new_classes(tmp_path):
    a, b = _bf16(4, 24, seed=1), _bf16(32, 4, seed=2)
    o = "language_model.model.layers.0.self_attn.o_proj"
    # the o_proj refusals stand: packed / list-valued on a site vLLM does not pack
    with pytest.raises(wr.FingerprintError, match="packed module .*PackedLoRALayerWeights"):
        wr.adapter_fingerprint_bf16({o: wr.ModuleWeights(a, b, 4, 4, True)})
    with pytest.raises(wr.FingerprintError, match="packed module .*a site vLLM does not pack"):
        wr.adapter_fingerprint_bf16({o: wr.ModuleWeights([a, None, a], [b, None, b], 4, 4, False)})
    qkv = "language_model.model.layers.0.self_attn.qkv_proj"
    aq, bq = _bf16(4, D, seed=3), _bf16(Q_OUT, 4, seed=4)
    # a plain tensor pair under a packed site
    with pytest.raises(wr.FingerprintError, match="qkv_proj is a packed wrapper site"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights(aq, bq, 4, 4, False)})
    # list-valued but is_packed False
    with pytest.raises(wr.FingerprintError, match="is_packed=False — not PackedLoRALayerWeights"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights([aq, None, None], [bq, None, None], 4, 0, False, [4, None, None])})
    # wrong slot count
    with pytest.raises(wr.FingerprintError, match="2 lora_a / 2 lora_b slots, expected 3"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights([aq, None], [bq, None], 4, 0, True, [4, None])})
    # no per-component alphas
    with pytest.raises(wr.FingerprintError, match="carries no per-component lora_alphas"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights([aq, None, None], [bq, None, None], 4, 0, True)})
    # a component's OWN alpha != rank refuses (the packed top-level alpha 0 is ignored)
    with pytest.raises(wr.FingerprintError, match="non-unit LoRA scaling alpha=8 r=4 on packed component 'q_proj'"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights([aq, None, None], [bq, None, None], 4, 0, True, [8, None, None])})
    # half a component
    with pytest.raises(wr.FingerprintError, match="component 'q_proj' \\(slot 0\\): lora_a is set but lora_b is None"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights([aq, None, None], [None, None, None], 4, 0, True, [4, None, None])})
    # every slot None
    with pytest.raises(wr.FingerprintError, match="every component slot is None"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights([None] * 3, [None] * 3, 4, 0, True, [None] * 3)})
    # a packed component whose physical rank is not r refuses like any linear factor
    with pytest.raises(wr.FingerprintError, match="neither axis equals r=4"):
        wr.adapter_fingerprint_bf16({qkv: wr.ModuleWeights([_bf16(6, D, seed=5), None, None], [bq, None, None], 4, 0, True, [4, None, None])})
    # experts: flat 2-D on the worker = unstacked (no FusedMoE3DWithLoRA wrapper claimed it)
    ex = "language_model.model.layers.0.mlp.experts"
    with pytest.raises(wr.FingerprintError, match="FLAT 2-D tensor .* not stacked into vLLM's 3-D"):
        wr.adapter_fingerprint_bf16({ex: wr.ModuleWeights(_bf16(E * 2, I_EXPERT, seed=6), _bf16(D, E * 2, seed=7), 4, 4, False)})
    a3, b3 = _bf16(E, 2, I_EXPERT, seed=8), _bf16(E, D, 2, seed=9)
    # experts: top-level scaling must be unit
    with pytest.raises(wr.FingerprintError, match="non-unit LoRA scaling alpha=8 r=4"):
        wr.adapter_fingerprint_bf16({ex: wr.ModuleWeights(a3, b3, 4, 8, False)})
    # experts: list of the wrong length / a list under the base_layer alias
    with pytest.raises(wr.FingerprintError, match="exactly 2 slots"):
        wr.adapter_fingerprint_bf16({ex: wr.ModuleWeights([a3], [b3], 4, 4, False)})
    with pytest.raises(wr.FingerprintError, match="list-valued factors under the gate_up"):
        wr.adapter_fingerprint_bf16({ex + ".base_layer": wr.ModuleWeights([a3, a3], [b3, b3], 4, 4, False)})
    # experts: A and B disagree on E / r_e
    with pytest.raises(wr.FingerprintError, match="disagree on the expert count"):
        wr.adapter_fingerprint_bf16({ex: wr.ModuleWeights(a3, _bf16(E + 1, D, 2, seed=9), 4, 4, False)})
    with pytest.raises(wr.FingerprintError, match="disagree on the per-expert rank"):
        wr.adapter_fingerprint_bf16({ex: wr.ModuleWeights(a3, _bf16(E, D, 3, seed=9), 4, 4, False)})
    # file: target_modules must equal the linear short names; target_parameters must name the experts
    fac = _draw(r=4, r_e=2, seed=8)
    d = _write_adapter(tmp_path / "targets", fac, r=4, r_e=2, config_extra={"target_modules": ["o_proj"]})
    with pytest.raises(wr.FingerprintError, match=r"target_modules=\['o_proj'\]; its unique contents must equal"):
        wr.expected_adapter_fingerprint_bf16(d)
    d = _write_adapter(tmp_path / "no_params", fac, r=4, r_e=2, drop_config_keys=("target_parameters",))
    with pytest.raises(wr.FingerprintError, match="target_parameters=None does not name both"):
        wr.expected_adapter_fingerprint_bf16(d)
    d = _write_adapter(tmp_path / "params_no_experts", _draw(r=4, r_e=2, seed=8, experts=False), r=4,
                       config_extra={"target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]})
    with pytest.raises(wr.FingerprintError, match="declares routed-expert parameters but the file carries no experts"):
        wr.expected_adapter_fingerprint_bf16(d)
    # file: a packed wrapper name in a PEFT file is not a thing
    from safetensors.torch import save_file
    d = tmp_path / "packed_in_file"
    d.mkdir()
    save_file({FILE.format(0, "self_attn.qkv_proj") + ".lora_A.weight": torch.randn(4, D),
               FILE.format(0, "self_attn.qkv_proj") + ".lora_B.weight": torch.randn(Q_OUT, 4)},
              str(d / sa.ADAPTER_WEIGHTS_FILE))
    (d / sa.ADAPTER_CONFIG_FILE).write_text(json.dumps({"r": 4, "lora_alpha": 4.0, "target_modules": ["qkv_proj"]}))
    with pytest.raises(wr.FingerprintError, match="names vLLM's packed wrapper qkv_proj"):
        wr.expected_adapter_fingerprint_bf16(d)
    # the launcher's reshape is vLLM's
    A = torch.arange(E * 2 * D, dtype=torch.float32).reshape(E * 2, D)
    B = torch.arange(D * 2 * E, dtype=torch.float32).reshape(D, 2 * E)
    a3, b3 = wr.reshape_flat_experts(A, B, E)
    assert torch.equal(a3, A.reshape(E, -1, D)) and torch.equal(b3, B.reshape(D, -1, E).permute(2, 0, 1))
    assert a3.is_contiguous() and b3.is_contiguous()


# --------------------------------------------------------------------------
# the execution registry: packed wrappers, the FusedMoE3DWithLoRA wrapper, the buffer law
# --------------------------------------------------------------------------


def _expected_buffer_bytes(*, layers=LAYERS, max_loras=2, R=8):
    """Hand count, bf16 (2 bytes): per layer qkv A 3×[n,1,R,D] + B [n,1,q,R]+2×[n,1,kv,R];
    o_proj A [n,1,R,O_IN] + B [n,1,D,R]; gate_up A 2×[n,1,R,D] + B 2×[n,1,I,R]; down
    A [n,1,R,I] + B [n,1,D,R]; experts w13_a [n,E,R,D] + w13_b [n,E,2I,R] + w2_a [n,E,R,I]
    + w2_b [n,E,D,R]."""
    n = max_loras
    per_layer = (
        3 * n * R * D + n * (Q_OUT + 2 * KV_OUT) * R            # qkv
        + n * R * O_IN + n * D * R                              # o_proj
        + 2 * n * R * D + 2 * n * I_SHARED * R                  # shared gate_up
        + n * R * I_SHARED + n * D * R                          # shared down
        + n * E * R * D + n * E * 2 * I_EXPERT * R + n * E * R * I_EXPERT + n * E * D * R   # experts
    )
    return 2 * per_layer * len(layers)


def test_lora_execution_modules_report_reads_packed_and_moe_wrappers():
    fac = _draw(r=4, r_e=2, seed=9)
    model = _loaded_fake(1, fac, r=4)
    rep = wr.lora_execution_modules_report(_Runner([model], _registry()))
    assert rep["lora_execution_module_keys"] == sorted(
        f"L{l}.{s}" for l in LAYERS for s in ("self_attn.qkv_proj", "self_attn.o_proj",
                                               "mlp.shared_expert.gate_up_proj",
                                               "mlp.shared_expert.down_proj", "mlp.experts"))
    assert rep["lora_execution_module_names"] == sorted(_registry())
    assert rep["lora_execution_modules_other"] == [] and rep["lora_execution_modules_n"] == 10
    st = rep["lora_execution_module_strides"]
    assert st[VLLM.format(3, "self_attn.qkv_proj")] == {"in": D, "out": Q_OUT + 2 * KV_OUT}   # 44: the packed stride
    assert st[VLLM.format(3, "self_attn.o_proj")] == {"in": O_IN, "out": D}
    assert st[VLLM.format(3, "mlp.shared_expert.gate_up_proj")] == {"in": D, "out": 2 * I_SHARED}
    assert st[VLLM.format(3, "mlp.experts")] == {"in": max(D, I_EXPERT), "out": max(2 * I_EXPERT, D)}
    assert rep["lora_execution_module_stride_max"] == Q_OUT + 2 * KV_OUT
    moe = rep["lora_execution_moe_wrappers"][VLLM.format(3, "mlp.experts")]
    assert moe == {"num_experts_local": E, "max_lora_rank": 8, "max_loras": 2, "hidden_size": D,
                   "intermediate_size_per_partition": I_EXPERT,
                   "w13_in": D, "w13_out": 2 * I_EXPERT, "w2_in": I_EXPERT, "w2_out": D,
                   "w13_lora_a_stacked": [2, E, 8, D], "w13_lora_b_stacked": [2, E, 2 * I_EXPERT, 8],
                   "w2_lora_a_stacked": [2, E, 8, I_EXPERT], "w2_lora_b_stacked": [2, E, D, 8]}
    # the expected-bytes law: Σ numel × element_size over every wrapper's stacked buffers
    assert rep["lora_execution_module_lora_buffer_bytes"] == _expected_buffer_bytes()
    by = rep["lora_execution_module_lora_buffer_bytes_by_module"]
    assert set(by) == set(_registry()) and sum(by.values()) == rep["lora_execution_module_lora_buffer_bytes"]
    assert by[VLLM.format(3, "mlp.experts")] == 2 * 2 * E * (8 * D + 2 * I_EXPERT * 8 + 8 * I_EXPERT + D * 8)
    assert rep["lora_execution_module_lora_buffer_unread"] == []
    # the MoE part scales with R (the binding scaling fact): R 16 doubles it
    rep16 = wr.lora_execution_modules_report(_Runner([model], _registry(max_lora_rank=16)))
    assert rep16["lora_execution_module_lora_buffer_bytes_by_module"][VLLM.format(3, "mlp.experts")] == \
        2 * by[VLLM.format(3, "mlp.experts")]
    # a linear wrapper without stacked buffers is named in `unread`, never silently summed
    bare = wr.lora_execution_modules_report(_Runner([model], _registry(buffers=False)))
    assert bare["lora_execution_module_lora_buffer_unread"] == sorted(
        n for n in _registry() if not n.endswith(".mlp.experts"))
    assert bare["lora_execution_module_lora_buffer_bytes"] == sum(
        v for k, v in by.items() if k.endswith(".mlp.experts"))
    # MoE wrapper drift refuses by name: a missing buffer, disagreeing shapes, a lying attribute
    mods = _registry(layers=(3,))
    name = VLLM.format(3, "mlp.experts")
    del mods[name].w2_lora_b_stacked
    with pytest.raises(wr.LensWorkerError, match="has no w2_lora_b_stacked"):
        wr.lora_execution_modules_report(_Runner([model], mods))
    mods = _registry(layers=(3,))
    mods[name] = _MoEWrap(shapes={"w13_lora_a_stacked": (2, E, 8, D), "w13_lora_b_stacked": (2, E, 2 * I_EXPERT, 8),
                                  "w2_lora_a_stacked": (2, E, 8, I_EXPERT), "w2_lora_b_stacked": (2, E, D + 1, 8)})
    with pytest.raises(wr.LensWorkerError, match="MoE LoRA buffers disagree"):
        wr.lora_execution_modules_report(_Runner([model], mods))
    mods = _registry(layers=(3,))
    mods[name].hidden_size = D + 1
    with pytest.raises(wr.LensWorkerError, match="wrapper.hidden_size=17 but its stacked LoRA buffers say 16"):
        wr.lora_execution_modules_report(_Runner([model], mods))
    mods = _registry(layers=(3,))
    mods[name].w13_lora_a_stacked = mods[name].w13_lora_a_stacked * 2      # two slices: the 2-D class
    with pytest.raises(wr.LensWorkerError, match="exactly one w13 slice"):
        wr.lora_execution_modules_report(_Runner([model], mods))
    # the 2-D FusedMoEWithLoRA parent ALSO exposes lora_a_stacked views: never double counted
    mods = _registry(layers=(3,))
    mods[name].lora_a_stacked = [mods[name].w13_lora_a_stacked[0][0][0]]
    mods[name].lora_b_stacked = [mods[name].w13_lora_b_stacked[0][0][0]]
    assert wr.lora_execution_modules_report(_Runner([model], mods))[
        "lora_execution_module_lora_buffer_bytes_by_module"][name] == by[name]
    # an unrecognised wrapper rides in `other`, still with its stride
    mods = _registry(layers=(3,))
    mods[VLLM.format(3, "linear_attn.in_proj_qkvz")] = _Wrap(D, (32, 32, 64, 64))
    rep = wr.lora_execution_modules_report(_Runner([model], mods))
    assert rep["lora_execution_modules_other"] == [VLLM.format(3, "linear_attn.in_proj_qkvz")]
    assert rep["lora_execution_module_stride_max"] == 192


def test_execution_bindings_cover_every_recognised_wrapper():
    fac = _draw(r=4, r_e=2, seed=10)
    model = _loaded_fake(1, fac, r=4)
    rep = wr.loaded_adapters_report(_Runner([model], _registry()))
    bindings = rep["lora_adapters"][0]["execution_bindings"]
    assert set(bindings) == set(_registry())
    # every wrapper binds ITS packed / stacked factor by exact name (the base_layer alias binds nowhere)
    assert all(bindings[n] == n for n in _registry())
    assert VLLM.format(3, "mlp.experts.base_layer") in rep["lora_adapters"][0]["raw_module_names"]
    assert VLLM.format(3, "mlp.experts.base_layer") not in bindings.values()
    # an adapter of another class binds None on the wrappers it does not adapt
    wo = _loaded_fake(2, _draw(r=4, r_e=2, seed=11, shared=False, experts=False), r=4)
    for k in list(wo.loras):
        if not k.endswith("o_proj"):
            del wo.loras[k]
    rep = wr.loaded_adapters_report(_Runner([model, wo], _registry()))
    b2 = rep["lora_adapters"][1]["execution_bindings"]
    assert {n for n, f in b2.items() if f is not None} == {VLLM.format(l, "self_attn.o_proj") for l in LAYERS}


# --------------------------------------------------------------------------
# the launcher's problem checkers over a worker report built from the fakes
# --------------------------------------------------------------------------


def _worker_report(models, modules):
    runner = _Runner(models, modules)
    rep = {}
    rep.update(wr.loaded_adapters_report(runner))
    rep.update(wr.lora_execution_modules_report(runner))
    return rep


def test_native_lora_report_problems_hold_every_class_to_the_execution_set(tmp_path):
    fac = _draw(r=4, r_e=2, seed=12)
    d = _write_adapter(tmp_path / "cell", fac, r=4, r_e=2)
    exp = sa.expected_native_lora_state_for(["cell"], {"cell": str(d)})
    model = _loaded_fake(1, fac, r=4)
    healthy = _worker_report([model], _registry())
    assert sa.native_lora_report_problems([healthy], exp) == []
    # the experts wrapper is missing (a --lora-target-modules class without `experts`):
    # both expert projections of every layer have no execution layer
    no_moe = _worker_report([model], _registry(drop=tuple(VLLM.format(l, "mlp.experts") for l in LAYERS)))
    problems = sa.native_lora_report_problems([no_moe], exp)
    assert len(problems) == 1 and "4 of 18 fingerprinted module(s)" in problems[0] \
        and "have NO LoRA execution layer" in problems[0] and "L3.mlp.experts.down_proj" in problems[0]
    # the qkv wrapper of one layer is missing: its q/k/v keys are uncovered, o_proj stays covered
    no_qkv = _worker_report([model], _registry(drop=(VLLM.format(7, "self_attn.qkv_proj"),)))
    problems = sa.native_lora_report_problems([no_qkv], exp)
    assert len(problems) == 1 and "3 of 18" in problems[0] and "L7.self_attn.k_proj" in problems[0] \
        and "L7.self_attn.o_proj" not in problems[0]
    # the EXACT-NAME finding across classes: a mis-prefixed adapter canonicalises and
    # fingerprints identically but binds nothing on the packed / MoE wrappers
    prefixed = _loaded_fake(1, fac, r=4, spelling="language_model.model.model.layers.{}.{}")
    rep = _worker_report([prefixed], _registry())
    assert rep["lora_adapters"][0]["fingerprint_bf16"] == exp["adapters"]["cell"]["fingerprint_bf16"]
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 1 and "18 of 18 fingerprinted module(s) have a LoRA execution layer at the same " \
        "canonical key but NO exact-name binding" in problems[0]
    assert "L3.mlp.experts.down_proj: factor ['language_model.model.model.layers.3.mlp.experts'] vs execution " \
        "['language_model.model.layers.3.mlp.experts']" in problems[0]
    # a wrapper binding a factor whose components are outside the fingerprinted set (the
    # `wo` adapter's report shows q/k/v bound): refused as an applied set exceeding the attested one
    wo_fac = {k: v for k, v in fac.items() if k[1] == "self_attn.o_proj"}
    d_wo = _write_adapter(tmp_path / "wo", wo_fac, r=4)
    exp_wo = sa.expected_native_lora_state_for(["wo"], {"wo": str(d_wo)})
    wo = _loaded_fake(1, wo_fac, r=4)
    assert sa.native_lora_report_problems([_worker_report([wo], _registry())], exp_wo) == []
    rep = _worker_report([wo], _registry())
    rep["lora_adapters"][0]["execution_bindings"][VLLM.format(3, "self_attn.qkv_proj")] = \
        VLLM.format(3, "self_attn.o_proj")
    problems = sa.native_lora_report_problems([rep], exp_wo)
    assert any("execution module(s) at ['L3.self_attn.qkv_proj'] bind a factor of this adapter" in p
               for p in problems)
    # the wording: an unrecognised execution name refuses as such
    rep = _worker_report([wo], _registry())
    rep["lora_execution_module_names"] = list(rep["lora_execution_module_names"]) + ["model.layers.0.mlp.gate"]
    assert any("not a recognised adapted module" in p for p in sa.native_lora_report_problems([rep], exp_wo))


def test_execution_stride_problems_admit_the_packed_and_moe_wrappers_in_class():
    fac = _draw(r=4, r_e=2, seed=13)
    rep = _worker_report([_loaded_fake(1, fac, r=4)], _registry())
    cls = ("down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj", "experts")
    D_exec = Q_OUT + 2 * KV_OUT
    assert sa.execution_stride_problems([rep], target_modules=cls, max_module_dim=D_exec) == []
    # the packed qkv stride exceeds the widest exported tensor (24): sized from the tensor, refused
    problems = sa.execution_stride_problems([rep], target_modules=cls, max_module_dim=Q_OUT)
    assert len(problems) == 1 and f"self_attn.qkv_proj' carries a LoRA wrapper with stride {D_exec}" in problems[0]
    # without `experts` in the class the MoE wrappers are outside it
    problems = sa.execution_stride_problems([rep], target_modules=cls[:-1], max_module_dim=D_exec)
    assert len(problems) == 1 and "2 execution module(s) carry a LoRA wrapper whose suffix is NOT in" in problems[0] \
        and VLLM.format(3, "mlp.experts") in problems[0]
    # the o_proj-only class refuses every packed wrapper (their children are not in it)
    problems = sa.execution_stride_problems([rep], target_modules=("o_proj",), max_module_dim=D_exec)
    assert len(problems) == 1 and "8 execution module(s)" in problems[0]


def test_adapter_module_dims_and_execution_widths(tmp_path):
    fac = _draw(r=4, r_e=2, seed=14)
    d = _write_adapter(tmp_path / "cell", fac, r=4, r_e=2)
    dims = sa.adapter_module_dims(d)
    assert dims[FILE.format(3, "self_attn.q_proj")] == (D, Q_OUT)
    assert dims[FILE.format(3, "self_attn.k_proj")] == (D, KV_OUT)
    assert dims[FILE.format(3, "self_attn.o_proj")] == (O_IN, D)
    assert dims[FILE.format(3, "mlp.shared_expert.gate_proj")] == (D, I_SHARED)
    assert dims[FILE.format(3, "mlp.shared_expert.down_proj")] == (I_SHARED, D)
    # the experts' flat pairs size by their own shape (rank axis E*r_e, not r): (d, 2I) and (I, d)
    assert dims[FILE.format(3, "mlp.experts.base_layer")] == (D, 2 * I_EXPERT)
    assert dims[FILE.format(3, "mlp.experts")] == (I_EXPERT, D)
    widths = sa.execution_widths(dims)
    assert widths[FILE.format(3, "self_attn.qkv_proj")] == (D, Q_OUT + 2 * KV_OUT)
    assert widths[FILE.format(3, "mlp.shared_expert.gate_up_proj")] == (D, 2 * I_SHARED)
    assert widths[FILE.format(3, "self_attn.o_proj")] == (O_IN, D)
    assert widths[FILE.format(3, "mlp.experts.base_layer")] == (D, 2 * I_EXPERT)
    assert widths[FILE.format(3, "mlp.experts")] == (I_EXPERT, D)
    assert FILE.format(3, "self_attn.q_proj") not in widths
    assert sa.adapter_execution_widths(d) == widths
    # the ceiling D is the EXECUTING stride: the packed qkv width, labelled by the wrapper
    assert sa.max_module_dim({"cell": dims}) == (Q_OUT + 2 * KV_OUT, "cell:" + FILE.format(3, "self_attn.qkv_proj"))
    # an o_proj-only adapter keeps today's label and value
    d_o = _tiny_peft_adapter(tmp_path / "o")
    assert sa.max_module_dim({"o": sa.adapter_module_dims(d_o)}) == (
        32, "o:base_model.model.model.layers.0.self_attn.o_proj")
    # components disagreeing on the input width refuse; an unrecognised name packs with
    # nothing and stands at its own dims (the geometry tests fake bare {"o_proj": (D, D)})
    with pytest.raises(sa.ServingAttestationError, match="disagree on the input width"):
        sa.execution_widths({FILE.format(3, "self_attn.q_proj"): (D, Q_OUT), FILE.format(3, "self_attn.k_proj"): (D + 1, KV_OUT)})
    assert sa.execution_widths({"o_proj": (4096, 4096), "model.layers.3.mlp.gate": (D, E)}) == {
        "model.layers.3.mlp.gate": (D, E), "o_proj": (4096, 4096)}
    assert sa.max_module_dim({"fake": {"o_proj": (4096, 4096)}}) == (4096, "fake:o_proj")
    # a flat experts pair whose rank axes disagree refuses at the header
    from safetensors.torch import save_file
    bad = tmp_path / "bad_experts"
    bad.mkdir()
    save_file({FILE.format(0, "mlp.experts") + ".lora_A.weight": torch.randn(8, I_EXPERT),
               FILE.format(0, "mlp.experts") + ".lora_B.weight": torch.randn(D, 6)},
              str(bad / sa.ADAPTER_WEIGHTS_FILE))
    (bad / sa.ADAPTER_CONFIG_FILE).write_text(json.dumps({"r": 4, "lora_alpha": 4.0, "target_modules": ["down_proj"]}))
    with pytest.raises(sa.ServingAttestationError, match="do not share the E\\*r_e rank axis"):
        sa.adapter_module_dims(bad)
