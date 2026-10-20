"""NVIDIA-Nemotron-3-Nano-4B (nemotron_h) on the vLLM serving lane: the
hybrid Mamba2/MLP/attention seams and the ChatML markers, pinned on CPU.

NemotronH is the first target whose decoder is NOT a stack of
``self_attn``-bearing blocks: 42 blocks, each wrapping ONE ``mixer`` —
21 Mamba2 (``mixer.out_proj``), 17 MLP (``mixer.down_proj``) and 4
attention (``mixer.o_proj``, blocks 12/17/24/32 per the config's
``hybrid_override_pattern``) — under a stack the HF class names
``backbone``. Everything the lora kinds need (the o_proj MODULE site) is
therefore a strict subset of the layers, and a Mamba ``out_proj`` sits one
attribute-name away from being mistaken for it. Pinned here:

- ``models._LAYER_PATHS`` reaches ``backbone.layers``; ``models.
  oproj_layers`` yields exactly [12, 17, 24, 32] on a NemotronH-shaped tree
  (built from the REAL pattern string) and every layer on a Llama;
  ``attention_oproj`` never returns a Mamba ``out_proj``;
- ``lens_worker._resolve_oproj`` resolves the attention mixer's o_proj and
  REFUSES a Mamba/MLP block — and the worker install on a NemotronH-shaped
  view hooks ``mixer.o_proj`` while a lora op at an MLP block refuses at
  preflight;
- ``lora_export`` names adapter tensors from ONE tree-derived seam — the
  decoder prefix (``models.decoder_layers_path``) plus the attention site
  (``models.attention_oproj_site``) off a meta-device skeleton of the
  cached config: ``model.layers.N.mixer.o_proj`` for NemotronH on
  transformers 5.14 (its live tree; the checkpoint's ``backbone.`` form
  maps to the same vLLM module — resolve_decoder_layout's docstring
  carries the 0.25.1 source evidence), the llama/GLM exports byte-
  identical, a fit at a Mamba2/MLP layer refused, a fit whose factors do
  not fit the attention mixer's o_proj (in 5120, out 3136 at the pinned
  revision) refused by name, the circuit-breaker grid refused on a mixer
  site;
- ``vllm_server.hybrid_recurrent_model`` classifies it hybrid (the real
  pinned config via ``layer_types``; the name fallback for CPU synthetics),
  so a launch composes prefix caching OFF + the thinking-off pin; the
  opt-in ``mamba_ssm_cache_dtype`` composes only on a hybrid target.

Real-tokenizer tests (offline-gated like test_glm_chat_markers) pin the
measured template facts: generation header ``<|im_start|>assistant\\n
<think></think>`` = (10, 1503, 19464, 1010, 12, 13) (H=6 — "assistant" is
TWO tokens here), terminator ``<|im_end|>`` = 11 == the tokenizer's eos
while config.json/generation_config.json say eos 2 (``</s>``) — the eos
TRAP, which the HF leg's ``_eos_ids`` union covers; no BOS in any render;
an EMPTY system block is emitted unconditionally; A-D are single tokens.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import sisteer.chat_markers as cmk
import sisteer.models as sm
from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import lora_export as lx
from lora_snapshot_fixtures import snapshot_like
from sisteer.capeval.serving import vllm_server as vs

NEMO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
NEMO_REV = "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"

#: config.json ``hybrid_override_pattern`` at the pinned revision: M =
#: Mamba2, - = MLP, * = attention. 42 chars; '*' at 12, 17, 24, 32.
PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
ATTN_LAYERS = (12, 17, 24, 32)

# measured ground truth on the pinned tokenizer (2026-09-02)
IM_START, IM_END, S_CLOSE, THINK, THINK_CLOSE, NEWLINE = 10, 11, 2, 12, 13, 1010
#: the generation-prompt header under the thinking-off pin:
#: <|im_start|> ass istant \n <think> </think>
NEMO_HEADER_IDS = (10, 1503, 19464, 1010, 12, 13)
#: the LOCATED header is anchored after the previous turn's <|im_end|>,
#: so it carries the template's trailing "\n" (gemma-4 style)
NEMO_LOCATED_HEADER = (1010,) + NEMO_HEADER_IDS
NEMO_CLOSING = (11, 1010)               # <|im_end|> \n
EMPTY_SYSTEM_BLOCK = (10, 25708, 1010, 11, 1010)   # <|im_start|>system\n<|im_end|>\n
NEMO_TEMPLATE_HASH = "ab7813c3abdd9cb6"
#: the sentinel choice (saved objects carry it): the LAST of
#: the <SPECIAL_N> reserved block (ids 18..999, id == N), never emitted by
#: the template
NEMO_SENTINEL = ("<SPECIAL_999>", 999)
MMLU_LETTER_IDS = {"A": 1065, "B": 1066, "C": 1067, "D": 1068,
                   " A": 1349, " B": 1398, " C": 1359, " D": 1407}


# ---------------------------------------------------------------------------
# a NemotronH-shaped module tree (real nn.Modules, tiny dims, offline)
# ---------------------------------------------------------------------------

D_MODEL = 8
D_ATTN_IN = 12   # non-square o_proj, like the real [3136, 5120]


class _MambaMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)   # NOT an o_proj


class _MlpMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(D_MODEL, 2 * D_MODEL)
        self.down_proj = nn.Linear(2 * D_MODEL, D_MODEL)


class _AttnMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_ATTN_IN)
        self.k_proj = nn.Linear(D_MODEL, 4)
        self.v_proj = nn.Linear(D_MODEL, 4)
        self.o_proj = nn.Linear(D_ATTN_IN, D_MODEL)


_MIXERS = {"M": _MambaMixer, "-": _MlpMixer, "*": _AttnMixer}


class _Block(nn.Module):
    def __init__(self, code: str):
        super().__init__()
        self.mixer = _MIXERS[code]()


class _Backbone(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.layers = nn.ModuleList([_Block(c) for c in pattern])


class FakeNemotronH(nn.Module):
    def __init__(self, pattern: str = PATTERN):
        super().__init__()
        self.backbone = _Backbone(pattern)
        self.config = types.SimpleNamespace(hidden_size=D_MODEL,
                                            model_type="nemotron_h")


@pytest.fixture(scope="module")
def nemo():
    torch.manual_seed(0)
    return FakeNemotronH()


@pytest.fixture(scope="module")
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, tie_word_embeddings=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------
# models.py: the layer table and the o_proj capability probe
# ---------------------------------------------------------------------------


def test_pattern_is_the_pinned_config_shape():
    assert len(PATTERN) == 42
    assert tuple(i for i, c in enumerate(PATTERN) if c == "*") == ATTN_LAYERS
    assert (PATTERN.count("M"), PATTERN.count("-"), PATTERN.count("*")) \
        == (21, 17, 4)


def test_layer_paths_reach_backbone_layers_additively(nemo):
    layers = sm.get_decoder_layers(nemo)
    assert layers is nemo.backbone.layers and len(layers) == 42
    assert sm.num_layers(nemo) == 42
    # additive: the historical entries keep their order, the new path is
    # appended, so every existing model resolves exactly as before
    assert sm._LAYER_PATHS[0] == "model.layers"
    assert sm._LAYER_PATHS[-1] == "backbone.layers"


def test_oproj_layers_on_nemotron_is_the_four_attention_blocks(nemo):
    assert sm.oproj_layers(nemo) == list(ATTN_LAYERS)


def test_attention_oproj_never_matches_a_mamba_out_proj(nemo):
    for i, block in enumerate(nemo.backbone.layers):
        got = sm.attention_oproj(block)
        if i in ATTN_LAYERS:
            assert got is block.mixer.o_proj
        else:
            # Mamba blocks DO carry mixer.out_proj (and in_proj); MLP
            # blocks carry down_proj — none is an attention o_proj
            assert got is None
            assert not hasattr(block.mixer, "o_proj")


def test_oproj_layers_on_llama_is_every_layer(tiny_llama):
    """The capability-probe contract: all-full-attention models keep the
    historical all-layer behavior byte-for-byte."""
    assert sm.oproj_layers(tiny_llama) == [0, 1, 2]
    block = sm.get_decoder_layers(tiny_llama)[1]
    assert sm.attention_oproj(block) is block.self_attn.o_proj


def test_oproj_sites_are_exact_attribute_names():
    """No substring matching anywhere: a block whose only projection is
    named ``o_proj_out`` or ``out_proj`` must not resolve."""
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = nn.Module()
            self.mixer.out_proj = nn.Linear(2, 2)
            self.mixer.o_proj_out = nn.Linear(2, 2)
    assert sm.attention_oproj(Odd()) is None
    assert sm._OPROJ_SITES == (("self_attn", "o_proj"), ("mixer", "o_proj"))


# ---------------------------------------------------------------------------
# lens_worker: the o_proj hook-site resolution on the served tree
# ---------------------------------------------------------------------------


def test_resolve_oproj_finds_the_attention_mixer(nemo):
    for l in ATTN_LAYERS:
        assert lw._resolve_oproj(nemo.backbone.layers[l], l) \
            is nemo.backbone.layers[l].mixer.o_proj


@pytest.mark.parametrize("layer", [0, 15, 41])   # Mamba, MLP, MLP
def test_resolve_oproj_refuses_non_attention_blocks(nemo, layer):
    assert PATTERN[layer] != "*"
    with pytest.raises(lw.LensWorkerError,
                       match="no hookable attention o_proj"):
        lw._resolve_oproj(nemo.backbone.layers[layer], layer)


def test_resolve_oproj_still_serves_the_llama_layout():
    from test_lora_lens_serving import OprojLayer
    layer = OprojLayer()
    assert lw._resolve_oproj(layer, 0) is layer.self_attn.o_proj


# -- the worker install on a NemotronH-shaped view --------------------------

from test_lora_lens_serving import OprojModule, _lora_obj, _registry  # noqa: E402
from test_petri_lens_worker import FakeLayer, FakeView, _ok_header  # noqa: E402


class _HookableMixer:
    """A vLLM-shaped NemotronH mixer surface: the attention mixer exposes
    a hookable ``o_proj``; the Mamba mixer exposes a hookable ``out_proj``
    ONLY — hookable on purpose, so a lookup that matched the wrong name
    would install a hook instead of refusing."""

    def __init__(self, attention: bool):
        if attention:
            self.qkv_proj = object()   # vLLM packs q/k/v; o_proj stays alone
            self.o_proj = OprojModule()
        else:
            self.in_proj = object()
            self.out_proj = OprojModule()


class NemoLayer(FakeLayer):
    def __init__(self, code: str):
        super().__init__()
        self.pre_hooks = []
        self.mixer = _HookableMixer(attention=(code == "*"))

    def register_forward_pre_hook(self, fn, *, with_kwargs=False):
        self.pre_hooks.append(fn)
        return object()


def _nemo_steering(registry):
    view = FakeView(layers=[NemoLayer(c) for c in PATTERN])
    view.tensor_parallel_size = lambda: 1
    steering = lw.LensWorkerSteering(view=view, registry=registry)
    steering.preflight(object(), header_verifier=_ok_header)
    steering.install()
    return steering, view


def test_worker_installs_oproj_hooks_on_attention_mixers():
    arm = _lora_obj(layers=(12, 32))
    steering, view = _nemo_steering(_registry(arm))
    rep = steering._report
    assert rep.layers == () and rep.oproj_layers == (12, 32)
    assert rep.plan_layer == 12
    for i, layer in enumerate(view.layers):
        if i in (12, 32):
            assert len(layer.mixer.o_proj.hooks) == 1
            assert len(layer.pre_hooks) == 1      # the positions stash
        else:
            assert len(layer.pre_hooks) == 0
            if hasattr(layer.mixer, "out_proj"):
                assert layer.mixer.out_proj.hooks == []   # never touched
        assert layer.hooks == []                  # no block hooks needed


@pytest.mark.parametrize("layer", [0, 15])   # Mamba block, MLP block
def test_worker_refuses_a_lora_op_on_a_non_attention_block(layer):
    with pytest.raises(lw.LensWorkerError,
                       match="no hookable attention o_proj"):
        _nemo_steering(_registry(_lora_obj(layers=(12, layer))))


def test_split_output_shapes_are_unchanged():
    """Deliberately NOT widened for NemotronH: the decoder-layer output
    shape of vLLM's nemotron_h Mamba/MLP/attention blocks is on the
    GPU-verify list; until measured, the accepted forms stay the
    (hidden, residual) pair and the plain tensor, everything else
    refuses."""
    h = torch.zeros(3, 4)
    assert lw.LensWorkerSteering._split_output((h, None)) == (h, None, True)
    assert lw.LensWorkerSteering._split_output(h) == (h, None, False)
    with pytest.raises(lw.LensWorkerError, match="refusing to guess"):
        lw.LensWorkerSteering._split_output((h, h, h))
    with pytest.raises(lw.LensWorkerError, match="refusing to guess"):
        lw.LensWorkerSteering._split_output({"hidden_states": h})


# ---------------------------------------------------------------------------
# lora_export: ONE tree-derived seam — decoder prefix + attention site
# ---------------------------------------------------------------------------

LLAMA_BASE = "meta-llama/Llama-3.1-8B-Instruct"
NEMO_LAYOUT = lx.DecoderLayout(prefix="model.layers", oproj_site="mixer.o_proj",
                               oproj_layers=ATTN_LAYERS)


def _rank1_blob_bytes(layer, d_in=D_ATTN_IN, d_out=D_MODEL, seed=23):
    import io
    g = torch.Generator().manual_seed(seed)
    blob = {"kind": "lora", "A": torch.randn(d_in, generator=g),
            "B": torch.randn(d_out, generator=g), "layer": layer,
            "positions": "all"}
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def test_default_oproj_site_is_the_llama_family_site():
    assert lx.DEFAULT_OPROJ_SITE == "self_attn.o_proj"
    assert lx.DEFAULT_OPROJ_SITE in {".".join(s) for s in sm._OPROJ_SITES}
    assert lx._validated_oproj_site("mixer.o_proj") == "mixer.o_proj"
    for bogus in ("mixer.out_proj", "o_proj", "self_attn", "", None, 3):
        with pytest.raises(lx.LoraExportError, match="not a known attention"):
            lx._validated_oproj_site(bogus)


def test_stated_nemotron_layout_names_the_attention_mixer(tmp_path):
    """A caller holding the model states prefix + site (the tree-derived
    values for NemotronH on transformers 5.14: model.layers + mixer.o_proj)
    and the export spells PEFT's HF path — the one vLLM's nemotron_h
    mapper passes through unchanged to module model.layers.12.mixer.o_proj
    (resolve_decoder_layout's docstring carries the source evidence)."""
    from safetensors.torch import load_file

    adapter = lx.export_genuine_adapter(
        {"kind": "lora", "layer": 12, "positions": "all"},
        _rank1_blob_bytes(12), tmp_path / "adapter", base_model=NEMO_MODEL,
        decoder_prefix="model.layers", oproj_site="mixer.o_proj")
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    prefix = "base_model.model.model.layers.12.mixer.o_proj"
    assert sorted(tensors) == [f"{prefix}.lora_A.weight",
                               f"{prefix}.lora_B.weight"]
    assert tuple(tensors[f"{prefix}.lora_A.weight"].shape) == (1, D_ATTN_IN)
    assert tuple(tensors[f"{prefix}.lora_B.weight"].shape) == (D_MODEL, 1)
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    # the PEFT short name stays o_proj — a Mamba mixer's projection is
    # out_proj, so the short name cannot collide with it
    assert cfg["target_modules"] == ["o_proj"]
    assert cfg["layers_to_transform"] == [12]
    assert cfg["base_model_name_or_path"] == NEMO_MODEL
    assert cfg["petri_decoder_prefix"] == "model.layers"
    assert cfg["petri_oproj_site"] == "mixer.o_proj"
    assert cfg["petri_genuine_adapter"] is True


def test_llama_export_is_byte_identical_with_and_without_the_site(tmp_path):
    """The default site adds NO config key and changes NO name: a stated
    llama prefix exports exactly as the GLM lineage's export did."""
    from safetensors.torch import load_file

    spec = {"kind": "lora", "layer": 3}
    art = _rank1_blob_bytes(3)
    a = lx.export_genuine_adapter(spec, art, tmp_path / "a",
                                  base_model=LLAMA_BASE,
                                  decoder_prefix=lx.LLAMA_DECODER_PREFIX)
    b = lx.export_genuine_adapter(spec, art, tmp_path / "b",
                                  base_model=LLAMA_BASE,
                                  decoder_prefix=lx.LLAMA_DECODER_PREFIX,
                                  oproj_site=lx.DEFAULT_OPROJ_SITE)
    assert (a / lx.ADAPTER_CONFIG).read_bytes() \
        == (b / lx.ADAPTER_CONFIG).read_bytes()
    assert (a / lx.ADAPTER_WEIGHTS).read_bytes() \
        == (b / lx.ADAPTER_WEIGHTS).read_bytes()
    cfg = json.loads((a / lx.ADAPTER_CONFIG).read_text())
    assert "petri_oproj_site" not in cfg
    assert cfg["petri_decoder_prefix"] == lx.LLAMA_DECODER_PREFIX
    prefix = "base_model.model.model.layers.3.self_attn.o_proj"
    assert sorted(load_file(str(a / lx.ADAPTER_WEIGHTS))) \
        == [f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"]


def test_free_form_site_refuses(tmp_path):
    with pytest.raises(lx.LoraExportError, match="not a known attention"):
        lx.export_genuine_adapter(
            {"kind": "lora", "layer": 0}, _rank1_blob_bytes(0),
            tmp_path / "adapter", base_model="x/y",
            decoder_prefix="model.layers", oproj_site="mixer.out_proj")


def test_circuit_breaker_export_refuses_on_a_mixer_site(tmp_path):
    import io
    buf = io.BytesIO()
    torch.save({"kind": "circuit_breaker"}, buf)
    with pytest.raises(lx.LoraExportError, match="self_attn/mlp layout"):
        lx.export_genuine_adapter(
            {"kind": "circuit_breaker"}, buf.getvalue(),
            tmp_path / "adapter", base_model=NEMO_MODEL,
            decoder_prefix="model.layers", oproj_site="mixer.o_proj")


def test_rank1_export_refuses_a_non_attention_layer_when_the_tree_is_known():
    """With the layout resolved from the tree, a fit at a Mamba2/MLP block
    names a module the base does not have — refused, never emitted."""
    import io
    blob = torch.load(io.BytesIO(_rank1_blob_bytes(15)), weights_only=True)
    with pytest.raises(lx.LoraExportError, match="carries no attention o_proj"):
        lx._export_rank1_lora({"kind": "lora"}, blob,
                              decoder_prefix=NEMO_LAYOUT.prefix,
                              oproj_site=NEMO_LAYOUT.oproj_site,
                              oproj_layers=NEMO_LAYOUT.oproj_layers)
    # an attention layer under the same layout exports
    blob12 = torch.load(io.BytesIO(_rank1_blob_bytes(12)), weights_only=True)
    tensors, cfg = lx._export_rank1_lora(
        {"kind": "lora"}, blob12, decoder_prefix=NEMO_LAYOUT.prefix,
        oproj_site=NEMO_LAYOUT.oproj_site,
        oproj_layers=NEMO_LAYOUT.oproj_layers)
    assert set(tensors) == {
        "base_model.model.model.layers.12.mixer.o_proj.lora_A.weight",
        "base_model.model.model.layers.12.mixer.o_proj.lora_B.weight"}
    assert cfg["petri_oproj_site"] == "mixer.o_proj"


def test_resolve_decoder_layout_refuses_an_unresolvable_base():
    with pytest.raises(lx.LoraExportError, match="not in the local HF cache"):
        lx.resolve_decoder_layout("not-an-org/not-a-cached-model-11111111")


# -- the real pinned config (offline-gated) ---------------------------------


def test_real_layout_resolves_from_the_pinned_config():
    """resolve_decoder_layout on the cached NemotronH config: transformers
    5.14.1's live tree is model.layers (base_model_prefix "model"; the
    checkpoint's backbone.* keys are renamed on load), the attention site
    is mixer.o_proj, and exactly the four attention blocks carry it."""
    _pinned_nemotron_cfg()
    assert lx.resolve_decoder_layout(NEMO_MODEL, NEMO_REV) == NEMO_LAYOUT
    assert lx.resolve_decoder_prefix(NEMO_MODEL, NEMO_REV) == "model.layers"


def test_real_export_derives_the_nemotron_layout_from_the_config(tmp_path):
    """No stated layout: base_model + base_revision resolve the tree (the
    skeleton's model.layers + mixer.o_proj), the export spells the name as
    the CHECKPOINT does — backbone.layers.12.mixer.o_proj — and holds
    the fit's factors to THAT module's real (in_features, out_features) — the blob is built with
    the resolved module's actual dimensions, the same source the export
    reads (a synthetic blob at the file's tiny dims is exactly the mis-sized
    artifact the check exists to refuse, and vLLM's set_lora would zero-pad
    it at load rather than refuse it) — refuses a Mamba2 layer, and refuses
    a stated site that disagrees with the tree."""
    from safetensors.torch import load_file

    _pinned_nemotron_cfg()
    layout, shapes = lx._derived_layout(NEMO_MODEL, NEMO_REV)
    assert layout == NEMO_LAYOUT
    d_in, d_out = shapes[12]
    # the pinned config's attention mixer o_proj: weight [3136, 5120]
    assert (d_in, d_out) == (5120, 3136)
    assert lx.oproj_target_modules(NEMO_MODEL, NEMO_REV)[
        "model.layers.12.mixer.o_proj"] == (d_in, d_out)         # the skeleton's spelling
    # The names are spelled as the CHECKPOINT spells them — Nemotron's
    # model.safetensors carries backbone.layers.{12,17,24,32}.mixer.o_proj.weight
    # (measured on the cluster cache, 2026-09-12); this cache holds the config
    # only, so a snapshot shaped like it stands in. vLLM's nemotron_h mapper
    # (orig_to_new_prefix backbone -> model) lands both spellings on the one
    # execution module model.layers.12.mixer.o_proj, which is why the skeleton
    # spelling bound on nemotron where it did not on qwen3.5.
    snap = snapshot_like(NEMO_MODEL, tmp_path / "snapshot", layout=layout)
    modules, ns = lx.checkpoint_target_modules(NEMO_MODEL, NEMO_REV, snapshot_dir=snap)
    assert modules["backbone.layers.12.mixer.o_proj"] == (d_in, d_out)
    assert sorted(modules) == [f"backbone.layers.{l}.mixer.o_proj" for l in (12, 17, 24, 32)]
    assert (ns.skeleton, ns.checkpoint, ns.source, ns.n_layers) \
        == ("model.layers", "backbone.layers", "safetensors-header", 4)
    adapter = lx.export_genuine_adapter(
        {"kind": "lora", "layer": 12}, _rank1_blob_bytes(12, d_in=d_in, d_out=d_out),
        tmp_path / "derived", base_model=NEMO_MODEL, base_revision=NEMO_REV,
        snapshot_dir=snap)
    prefix = "base_model.model.backbone.layers.12.mixer.o_proj"
    tensors = load_file(str(adapter / lx.ADAPTER_WEIGHTS))
    assert sorted(tensors) == [f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"]
    assert tuple(tensors[f"{prefix}.lora_A.weight"].shape) == (1, d_in)
    assert tuple(tensors[f"{prefix}.lora_B.weight"].shape) == (d_out, 1)
    cfg = json.loads((adapter / lx.ADAPTER_CONFIG).read_text())
    assert (cfg["petri_decoder_prefix"], cfg["petri_oproj_site"]) \
        == ("backbone.layers", "mixer.o_proj")
    # the file's tiny synthetic vectors under the REAL layout: factors that do
    # not fit the module they name, refused naming the module and both sizes
    # (an earlier change inverted this regression by dropping the check
    # instead of fixing this fixture)
    with pytest.raises(lx.LoraExportError,
                       match=r"kind 'lora': layer 12 has \(d_in, d_out\)=\(12, 8\) but the "
                             r"base's own o_proj at that layer is \(in_features, "
                             r"out_features\)=\(5120, 3136\).*\(backbone\.layers\.12\.mixer\.o_proj\)"):
        lx.export_genuine_adapter(
            {"kind": "lora", "layer": 12}, _rank1_blob_bytes(12),
            tmp_path / "tiny", base_model=NEMO_MODEL, base_revision=NEMO_REV,
            snapshot_dir=snap)
    assert not (tmp_path / "tiny").exists()
    with pytest.raises(lx.LoraExportError, match="carries no attention o_proj"):
        lx.export_genuine_adapter(
            {"kind": "lora", "layer": 15}, _rank1_blob_bytes(15),
            tmp_path / "mamba", base_model=NEMO_MODEL, base_revision=NEMO_REV,
            snapshot_dir=snap)
    with pytest.raises(lx.LoraExportError, match="module tree carries"):
        lx.export_genuine_adapter(
            {"kind": "lora", "layer": 12}, _rank1_blob_bytes(12),
            tmp_path / "wrong", base_model=NEMO_MODEL, base_revision=NEMO_REV,
            oproj_site="self_attn.o_proj", snapshot_dir=snap)


# ---------------------------------------------------------------------------
# vllm_server: hybrid classification + the opt-in Mamba cache dtype
# ---------------------------------------------------------------------------

from test_petri_serving_gate import (  # noqa: E402
    THINK_TEMPLATE_HASH, _chat_template, _sched_report, _spec,
    _think_template, _tpl_spec)


def test_hybrid_name_fallback_covers_nemotron_synthetics():
    """CPU-only synthetic ids (no cached config) classify by name — the
    fallback exists for tests, never for a live serve."""
    assert vs.hybrid_recurrent_model(
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-synthetic-uncached") is True
    assert vs.hybrid_recurrent_model("synthetic/nemotron_h-tiny") is True
    assert vs.hybrid_recurrent_model("Qwen/Qwen3.5-9B-synthetic") is True
    assert vs.hybrid_recurrent_model("meta-llama/Llama-3.1-8B-synthetic") \
        is False


def _hybrid_launch(tmp_path, **kw):
    return vs.build_server_launch(
        _spec(target_model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-synthetic",
              chat_template_hash=THINK_TEMPLATE_HASH,
              serving_launch_flags=["--enforce-eager"]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=_think_template(tmp_path), **kw)


def test_nemotron_launch_composes_caching_off_and_thinking_off(tmp_path):
    launch = _hybrid_launch(tmp_path)
    cmd = list(launch.cmd)
    assert "--no-enable-prefix-caching" in cmd
    assert "--enable-prefix-caching" not in cmd
    assert launch.prefix_caching is False
    assert cmd[cmd.index("--default-chat-template-kwargs") + 1] \
        == vs.THINKING_OFF_TEMPLATE_KWARGS
    # generation modality: no parser, and no Mamba dtype unless asked
    assert "--tool-call-parser" not in cmd
    assert "--mamba-ssm-cache-dtype" not in cmd
    assert launch.mamba_ssm_cache_dtype is None


def test_mamba_ssm_cache_dtype_opt_in_composes_on_a_hybrid(tmp_path):
    launch = _hybrid_launch(tmp_path, mamba_ssm_cache_dtype="float32")
    cmd = list(launch.cmd)
    assert cmd[cmd.index("--mamba-ssm-cache-dtype") + 1] == "float32"
    assert launch.mamba_ssm_cache_dtype == "float32"
    # the rest of the launch identity is untouched by the opt-in
    base = list(_hybrid_launch(tmp_path).cmd)
    i = cmd.index("--mamba-ssm-cache-dtype")
    assert cmd[:i] + cmd[i + 2:] == base


def test_mamba_ssm_cache_dtype_refuses_on_a_dense_target(tmp_path):
    with pytest.raises(vs.VllmServerError, match="non-hybrid target"):
        vs.build_server_launch(
            _tpl_spec(serving_launch_flags=["--enforce-eager"]),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=_chat_template(tmp_path),
            mamba_ssm_cache_dtype="float32")


def test_mamba_ssm_cache_dtype_rejects_junk(tmp_path):
    with pytest.raises(vs.VllmServerError, match="non-empty dtype name"):
        _hybrid_launch(tmp_path, mamba_ssm_cache_dtype="")


def _dense_launch(tmp_path):
    return vs.build_server_launch(
        _tpl_spec(serving_launch_flags=["--enforce-eager"]),
        port=8801, exact_replay_strategy="serialize",
        chat_template_path=_chat_template(tmp_path))


def test_expected_scheduler_state_attests_the_composed_mamba_dtype_only(tmp_path):
    """The composed opt-in is ATTESTED at readiness — the launcher expects
    the RESOLVED cache_config.mamba_ssm_cache_dtype to equal it — and
    nothing else changes: a hybrid launch that composed no dtype and every
    dense launch expect exactly what they expected before (no new key),
    so dense targets attest nothing new."""
    exp = vs.expected_scheduler_state(
        _hybrid_launch(tmp_path, mamba_ssm_cache_dtype="float32"))
    assert exp["mamba_ssm_cache_dtype"] == "float32"
    for launch in (_hybrid_launch(tmp_path), _dense_launch(tmp_path)):
        exp0 = vs.expected_scheduler_state(launch)
        assert "mamba_ssm_cache_dtype" not in exp0
    # the key is the ONLY difference the opt-in makes to the expectation
    a = vs.expected_scheduler_state(_hybrid_launch(tmp_path))
    b = dict(exp)
    del b["mamba_ssm_cache_dtype"]
    assert a == b


def test_validate_scheduler_reports_holds_the_resolved_mamba_dtype(tmp_path):
    """A hybrid launch that composed float32 passes readiness only when
    every worker RESOLVED float32 (GPU-verified 2026-09-02 on
    Nemotron-3-Nano-4B); a worker that resolved the engine default
    ("auto" — the flag ignored) or cannot report the field refuses, naming
    the field. A dense launch's expectation has no such key, so a report
    carrying any resolved value passes untouched."""
    exp = vs.expected_scheduler_state(
        _hybrid_launch(tmp_path, mamba_ssm_cache_dtype="float32"))
    ok = dict(_sched_report(), **exp)          # a worker that resolved float32
    assert ok["mamba_ssm_cache_dtype"] == "float32"
    assert vs.validate_scheduler_reports([ok], exp) == [ok]
    with pytest.raises(vs.VllmServerError, match="mamba_ssm_cache_dtype"):
        vs.validate_scheduler_reports(
            [dict(ok, mamba_ssm_cache_dtype="auto")], exp)
    with pytest.raises(vs.VllmServerError, match="mamba_ssm_cache_dtype.*MISSING"):
        missing = dict(ok)
        del missing["mamba_ssm_cache_dtype"]
        vs.validate_scheduler_reports([missing], exp)
    with pytest.raises(vs.VllmServerError, match="mamba_ssm_cache_dtype"):
        vs.validate_scheduler_reports(
            [dict(ok, mamba_ssm_cache_dtype=None)], exp)
    # dense: nothing new attested — the resolved value rides the report
    # unasserted, exactly as before this key existed
    exp_dense = vs.expected_scheduler_state(_dense_launch(tmp_path))
    rep_dense = dict(_sched_report(), **exp_dense, mamba_ssm_cache_dtype="auto")
    assert vs.validate_scheduler_reports([rep_dense], exp_dense) == [rep_dense]


def test_mamba_ssm_cache_dtype_is_wrapper_owned(tmp_path):
    """A manifest cannot smuggle the flag in: it is a composed opt-in."""
    assert "--mamba-ssm-cache-dtype" in vs._WRAPPER_OWNED_OPTIONS
    with pytest.raises(vs.VllmServerError):
        vs.build_server_launch(
            _spec(target_model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16-synthetic",
                  chat_template_hash=THINK_TEMPLATE_HASH,
                  serving_launch_flags=["--enforce-eager",
                                        "--mamba-ssm-cache-dtype", "float32"]),
            port=8801, exact_replay_strategy="serialize",
            chat_template_path=_think_template(tmp_path))


# ---------------------------------------------------------------------------
# the real pinned tokenizer / config (offline-gated, same pattern as GLM)
# ---------------------------------------------------------------------------


def _pinned_nemotron_tok():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            NEMO_MODEL, revision=NEMO_REV, local_files_only=True)
    except Exception as e:  # cache miss, offline resolution failure, ...
        pytest.skip(f"pinned {NEMO_MODEL} tokenizer @ {NEMO_REV[:8]} not in "
                    f"the local HF cache ({type(e).__name__})")
    # models.load_model_and_tokenizer's thinking-off pin, replicated so
    # this test needs no network: the template consults enable_thinking
    assert sm.pin_reasoning_template_off(tok) is True
    return tok


def _pinned_nemotron_cfg():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoConfig.from_pretrained(
            NEMO_MODEL, revision=NEMO_REV, local_files_only=True)
    except Exception as e:
        pytest.skip(f"pinned {NEMO_MODEL} config @ {NEMO_REV[:8]} not in "
                    f"the local HF cache ({type(e).__name__})")


def test_real_template_hash_and_control_ids():
    tok = _pinned_nemotron_tok()
    assert sm.chat_template_hash(tok) == NEMO_TEMPLATE_HASH
    assert sm.check_chat_template(tok) == NEMO_TEMPLATE_HASH  # not meta-llama
    ids = [tok.convert_tokens_to_ids(t) for t in
           ("<|im_start|>", "<|im_end|>", "</s>", "<think>", "</think>")]
    assert ids == [IM_START, IM_END, S_CLOSE, THINK, THINK_CLOSE]
    assert len(tok) == 131072


def test_real_markers_and_generation_header():
    tok = _pinned_nemotron_tok()
    m = cmk.derive_assistant_markers(tok)
    # terminator-anchored (NOT successor-bounded like GLM): the turn
    # closes on <|im_end|>, which IS the tokenizer's eos
    assert m.terminator == IM_END == tok.eos_token_id
    assert m.closing == NEMO_CLOSING and m.glue == 0
    assert m.header == NEMO_LOCATED_HEADER
    assert cmk.generation_header_ids(tok, m) == NEMO_HEADER_IDS
    assert tok.decode(list(NEMO_HEADER_IDS)) \
        == "<|im_start|>assistant\n<think></think>"
    # fit-time width (hooks.response_header_len) and the id derivation
    # agree — the invariant hf_engine asserts at arm time; H=6 because
    # "assistant" tokenizes as two pieces on this vocab
    from sisteer.hooks import response_header_len
    assert response_header_len(tok) == len(NEMO_HEADER_IDS) == 6
    # the located header differs from the appended one by exactly the
    # template's post-terminator "\n" (the gemma-4 shape) — a serve
    # boundary must use the appended form, never the located one
    assert m.header[1:] == NEMO_HEADER_IDS and m.header[0] == NEWLINE


def test_real_thinking_off_pin_moves_only_the_last_header_token():
    """Thinking ON renders ``<think>\\n``; OFF renders ``<think></think>``.
    Same width, different final token — a thinking-ON serve would steer
    from inside an open <think> the fit never saw."""
    tok = _pinned_nemotron_tok()
    msgs = [{"role": "user", "content": "Hi"}]
    off = cmk.render_ids(tok, msgs, add_generation_prompt=True)
    tok._sisteer_template_kwargs = {}
    on = cmk.render_ids(tok, msgs, add_generation_prompt=True)
    tok._sisteer_template_kwargs = {"enable_thinking": False}
    assert on[:-1] == off[:-1]
    assert (on[-1], off[-1]) == (NEWLINE, THINK_CLOSE)
    assert tuple(off[-len(NEMO_HEADER_IDS):]) == NEMO_HEADER_IDS


def test_real_spans_multi_turn():
    tok = _pinned_nemotron_tok()
    msgs = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "Six."}]
    full, spans = cmk.assistant_content_spans(tok, msgs)
    assert [tok.decode(full[lo:hi]) for lo, hi in spans] == ["Four.", "Six."]
    ctrl = cmk.control_token_ids(tok)
    assert not [p for lo, hi in spans for p in range(lo, hi)
                if full[p] in ctrl]
    # every span is closed by the terminator in the full render
    assert all(full[hi] == IM_END for _, hi in spans)
    # history assistant turns keep the empty <think></think> block, so
    # the header before every span is the same 6 ids the fit keyed on
    assert all(tuple(full[lo - 6:lo]) == NEMO_HEADER_IDS for lo, _ in spans)


def test_real_eos_trap_config_says_2_template_stops_on_11():
    """config.json / generation_config.json declare eos_token_id 2
    (``</s>``) while every chat turn closes on <|im_end|> = 11, which is
    the TOKENIZER's eos. The HF leg unions both (hf_engine._eos_ids), so
    a generate() stops on 11; the served vLLM path stops on the
    tokenizer's eos by construction — end-to-end stop on 11 through the
    chat API is on the GPU-verify list."""
    tok = _pinned_nemotron_tok()
    cfg = _pinned_nemotron_cfg()
    assert cfg.eos_token_id == S_CLOSE == 2
    assert tok.eos_token_id == IM_END == 11
    assert tok.eos_token != "</s>"
    # generation_config.json at the pinned revision (a tokenizer-sized
    # metadata file; cached alongside the tokenizer): HF generate()'s
    # DEFAULT eos would be 2 — the trap the union below covers
    from huggingface_hub import try_to_load_from_cache
    gen_cfg = try_to_load_from_cache(NEMO_MODEL, "generation_config.json",
                                     revision=NEMO_REV)
    if isinstance(gen_cfg, str) and Path(gen_cfg).is_file():
        assert json.loads(Path(gen_cfg).read_text())["eos_token_id"] == 2
    # (not cached -> config.json's eos_token_id == 2 above already pins
    # the trap; the union check below runs either way)
    from sisteer.capeval.serving.hf_engine import _eos_ids
    fake_model = types.SimpleNamespace(
        generation_config=types.SimpleNamespace(eos_token_id=cfg.eos_token_id))
    ids = _eos_ids(tok, fake_model)
    assert IM_END in ids and S_CLOSE in ids
    # the marker derivation's terminator is exactly the effective stop
    assert cmk.derive_assistant_markers(tok).terminator in ids


def test_real_no_bos_in_any_render():
    """No BOS: the template renders none and add_bos_token is false, so a
    chat render encodes identically with or without special tokens —
    encode_kwargs has no double-BOS to dodge."""
    from sisteer.generate import encode_kwargs
    tok = _pinned_nemotron_tok()
    assert tok("")["input_ids"] == []
    assert tok.bos_token_id == 1 and tok("Hi")["input_ids"] == [37133]
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Hi"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    assert not rendered.startswith(tok.bos_token)
    assert encode_kwargs(tok, rendered) == {}
    a = tok(rendered)["input_ids"]
    b = tok(rendered, add_special_tokens=False)["input_ids"]
    assert a == b and a[0] == IM_START


def test_real_empty_system_block_is_unconditional():
    """The template opens EVERY render with a system block — EMPTY when no
    system message is given. Constant scaffold, so the `none` and
    `prompted` arms differ by the SI content alone (unlike the meta-llama
    default-system-block trap, nothing is injected)."""
    tok = _pinned_nemotron_tok()
    none = cmk.render_ids(tok, [{"role": "user", "content": "Hi"}],
                          add_generation_prompt=True)
    assert tuple(none[:len(EMPTY_SYSTEM_BLOCK)]) == EMPTY_SYSTEM_BLOCK
    prompted = cmk.render_ids(
        tok, [{"role": "system", "content": "Be terse."},
              {"role": "user", "content": "Hi"}], add_generation_prompt=True)
    assert tuple(prompted[:3]) == EMPTY_SYSTEM_BLOCK[:3]   # <|im_start|>system\n
    assert prompted[3] != IM_END                            # content, not empty
    # from the user turn on, the two renders are identical
    u_none = none.index(IM_START, 1)
    u_prompted = prompted.index(IM_START, 1)
    assert none[u_none:] == prompted[u_prompted:]
    assert prompted[u_prompted - 2:u_prompted] == [IM_END, NEWLINE]


def test_real_mmlu_letters_are_single_tokens():
    tok = _pinned_nemotron_tok()
    for text, tid in MMLU_LETTER_IDS.items():
        assert tok(text, add_special_tokens=False)["input_ids"] == [tid], text


def test_real_sentinel():
    from sisteer.extraction.objects import find_sentinel
    tok = _pinned_nemotron_tok()
    assert find_sentinel(tok) == NEMO_SENTINEL
    assert NEMO_SENTINEL[1] in cmk.control_token_ids(tok)
    assert NEMO_SENTINEL[0] not in (tok.chat_template or "")


def test_real_config_is_hybrid_with_the_pinned_pattern():
    cfg = _pinned_nemotron_cfg()
    assert cfg.model_type == "nemotron_h"
    assert cfg.hybrid_override_pattern == PATTERN
    assert (cfg.num_hidden_layers, cfg.hidden_size) == (42, 3136)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) \
        == (40, 8, 128)
    # transformers 5.14 maps layer_types -> layers_block_type: Mamba2
    # blocks report linear_attention, so hybrid_recurrent_model's
    # architecture-authoritative read fires with no model_type table
    lt = list(cfg.layer_types)
    assert [i for i, t in enumerate(lt) if t == "full_attention"] \
        == list(ATTN_LAYERS)
    assert lt.count("linear_attention") == 21 and lt.count("mlp") == 17
    assert vs.hybrid_recurrent_model(NEMO_MODEL, NEMO_REV) is True
    assert sm.auto_model_class(NEMO_MODEL, revision=NEMO_REV).__name__ \
        == "AutoModelForCausalLM"
    assert lx.resolve_decoder_layout(NEMO_MODEL, NEMO_REV) == NEMO_LAYOUT


def test_real_template_launch_composes_the_pins(tmp_path):
    """build_server_launch on the REAL pinned template bytes + the real
    cached config: thinking-off composed (the template has the switch),
    prefix caching OFF (hybrid), no parser (generation modality)."""
    import sisteer.capeval.manifest as manifest
    tok = _pinned_nemotron_tok()
    _pinned_nemotron_cfg()
    tpl = tmp_path / "nemotron_chat_template.jinja"
    tpl.write_bytes(tok.chat_template.encode())
    launch = vs.build_server_launch(
        _spec(target_model=NEMO_MODEL, target_revision=NEMO_REV,
              chat_template_hash=manifest.sha256_hex16(tpl.read_bytes()),
              serving_launch_flags=["--enforce-eager"]),
        port=8801, exact_replay_strategy="serialize", chat_template_path=tpl)
    cmd = list(launch.cmd)
    assert cmd[:3] == ["vllm", "serve", NEMO_MODEL]
    assert cmd[cmd.index("--revision") + 1] == NEMO_REV
    assert launch.prefix_caching is False
    assert "--no-enable-prefix-caching" in cmd
    assert launch.default_chat_template_kwargs \
        == vs.THINKING_OFF_TEMPLATE_KWARGS
    assert "--tool-call-parser" not in cmd
    assert vs.expected_scheduler_state(launch)["enable_prefix_caching"] \
        is False


def test_real_model_family_is_nemotron_h_and_hybrid():
    """model_family reads nemotron_h off the cached config; the family's
    TOOL_PARSER_BY_FAMILY pin is `qwen3_xml` since the parser probe PASSED
    both halves on the staged tree 8365992 (2026-09-23): the template's
    <tool_call><function=...> XML format pairs with vLLM's qwen3_xml parser
    (the table's rule: add a family only with its verified pairing — this one
    is). The generation modality never composes one."""
    _pinned_nemotron_cfg()
    assert vs.model_family(NEMO_MODEL, NEMO_REV) == "nemotron_h"
    assert vs.hybrid_recurrent_model(NEMO_MODEL, NEMO_REV) is True
    # the S5 live parser probe PASSED on the staged tree 8365992 (2026-09-23; configs/capeval/tasks/parser_probe/nemotron_h_verdict_8365992.json): the pairing is VERIFIED and pinned
    assert vs.TOOL_PARSER_BY_FAMILY["nemotron_h"] == "qwen3_xml"


# ---------------------------------------------------------------------------
# capctrl (served-MMLU) registration: the CPU preflight's per-model checks
# mirrored on the real pinned tokenizer, plus text pins on the scripts (the
# Modal app is not importable in the CPU venv) — test_glm_serving_registration
# is the template
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]


def test_capctrl_preflight_seeded_render_parity_for_nemotron():
    """What capctrl_served.preflight's mmlu_render_parity predicts for
    Nemotron, run on the pinned tokenizer: the server-side seeded render
    (continue_final_message, under the composed thinking-off kwargs)
    equals the HF instrument's build_chat_prompt + "Answer:" byte for
    byte; the 6-id header is located at the same boundary by the served
    mask's rfind and parity's; the seed tokens follow it; the letters are
    single tokens. Measured difference from GLM: the UNPINNED seeded
    render is identical too (a completed assistant turn renders
    <think></think> whatever enable_thinking says), so on this template
    only the header / guard prediction depends on the pin — pinned here so
    a template change that makes parity pin-dependent is noticed."""
    import sys

    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.tiers import parity
    from sisteer.eval.capability_ext import _letter_token_ids, _letters
    from sisteer.generate import encode_kwargs

    sys.path.insert(0, str(REPO / "scripts"))
    import served_capability_run as srun

    tok = _pinned_nemotron_tok()
    render_kw = dict(tok._sisteer_template_kwargs)
    assert render_kw == {"enable_thinking": False}
    row = {"item_id": "synthetic-0",
           "question": "Which letter comes first in the alphabet?",
           "choices": ["b", "a", "d", "c"]}
    want = parity._mmlu_prompt(tok, row, system=None)
    msgs = [{"role": "user", "content": srun._mmlu_user_text(row)},
            {"role": "assistant", "content": srun.MMLU_ANSWER_SEED}]
    got = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        continue_final_message=True, **render_kw)
    assert got == want
    raw = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        continue_final_message=True)
    assert raw == want                       # pin-independent on this template
    assert want.endswith("<|im_start|>assistant\n<think></think>Answer:")
    header = [int(t) for t in cmk.generation_header_ids(tok)]
    assert header == list(NEMO_HEADER_IDS)
    ids = list(tok(want, **encode_kwargs(tok, want))["input_ids"])
    start = la._rfind_header(ids, header)
    assert start >= 0 and start == parity._rfind_subseq(ids, header)
    assert ids[start:start + len(header)] == header
    assert ids[start + len(header):]                 # the "Answer:" seed
    letters = _letter_token_ids(tok, _letters(4))
    assert len(letters) == 4 and len(set(letters)) == 4
    assert [int(t) for t in letters] == [MMLU_LETTER_IDS[" " + c]
                                         for c in "ABCD"]


def test_capctrl_scripts_register_nemotron_under_the_pinned_regime():
    """Text pins on the two scripts (the Modal app is not importable here):
    the MODELS entry carries the pinned revision and the GENERATED config
    path; the generator's MODEL_SPECS carries the same pin; the generated
    config exists, pins model + revision, and is the FIT arm set (no
    lora_r4 — a served-wave config follows the fits, as for GLM/gemma)."""
    import yaml

    app = (REPO / "scripts/modal/capctrl_served.py").read_text()
    assert ('"nemotron3nano4b": dict(hf="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"'
            in app)
    assert f'rev="{NEMO_REV}"' in app
    assert 'cfg="configs/two_si_nemotron3nano4b.yaml"' in app
    assert "pin_reasoning_template_off(tok)" in app
    gen = (REPO / "scripts/derive_model_config.py").read_text()
    assert '"nemotron3nano4b": ModelSpec(' in gen and NEMO_REV in gen
    cfg = yaml.safe_load(
        (REPO / "configs/two_si_nemotron3nano4b.yaml").read_text())
    assert cfg["model"] == NEMO_MODEL and cfg["model_revision"] == NEMO_REV
    assert cfg["layer"] == 20
    assert cfg["max_len_by_si"] == {"safety_refusal": 2048,
                                    "si_constitution": 8192,
                                    "si_irrelevant": 16384}
    assert "lora_r4" not in cfg["class_a_arms"]
