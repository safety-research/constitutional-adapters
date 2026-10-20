"""Nemotron-3-Nano-4B (NemotronH) architecture seams (2026-09-02).

NemotronHBlock hangs whichever mixer the layer is — Mamba2 / MLP / attention
— under ONE attribute, ``mixer``; there is no ``self_attn`` anywhere, and
transformers 5.14.1's native class exposes the stack at ``model.layers``
(the checkpoint's ``backbone.layers`` prefix is remapped on load). Pinned:

* ``get_decoder_layers`` finds all 42 blocks through either spelling;
* ``models.attention_oproj`` / ``oproj_layers`` resolve exactly the four
  attention blocks (12, 17, 24, 32 per the real hybrid_override_pattern) —
  a Mamba mixer's ``out_proj`` and an MLP mixer never match;
* MultiLayerLoRAObject / RankOneLoRAObject attach to the attention mixer's
  o_proj, sized from the module (non-square: heads*head_dim -> hidden), and
  refuse the non-attention depth-fraction layer 20 by naming the four;
* on a tiny random-weight instance of the REAL class (from_config, CPU, no
  downloads): the cached forward raises KeyError('mlp') under 5.14.1 and
  ``models.pin_kv_cache_off_if_unallocatable`` rescues default forwards and
  cache-free generate(); block-output hooks at every mixer kind receive
  gradient through the pure-torch SSD scan (the Modal image's path).

Fakes follow the real pattern with tiny dims; the real class's module tree
and shapes are pinned on the tiny instance here and, cache-gated, on the
real config in tests/test_nemotron_chat_markers.py.
"""

import warnings

import pytest
import torch
from torch import nn

from sisteer.models import (attention_oproj, get_decoder_layers, hidden_size,
                            oproj_layers)

NEMO_PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
NEMO_ATTN = [12, 17, 24, 32]
_D, _ATTN_D = 8, 12  # hidden, heads*head_dim — deliberately NON-square o_proj


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _MambaMixer(nn.Module):
    """NemotronHMamba2Mixer's projection names: in_proj / out_proj /
    conv1d — its ``out_proj`` is an SSM projection, not attention's."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(_D, 32, bias=False)
        self.out_proj = nn.Linear(16, _D, bias=False)
        self.conv1d = nn.Conv1d(16, 16, 4, groups=16)
        self.A_log = nn.Parameter(torch.zeros(2))


class _MlpMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(_D, 16, bias=False)
        self.down_proj = nn.Linear(16, _D, bias=False)


class _AttnMixer(nn.Module):
    """NemotronHAttention: q/k/v/o under the MIXER; o_proj maps
    heads*head_dim -> hidden (the real model: 5120 -> 3136)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(_D, _ATTN_D, bias=False)
        self.k_proj = nn.Linear(_D, 4, bias=False)
        self.v_proj = nn.Linear(_D, 4, bias=False)
        self.o_proj = nn.Linear(_ATTN_D, _D, bias=False)


class _NemoBlock(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.norm = nn.LayerNorm(_D)
        self.mixer = {"M": _MambaMixer, "-": _MlpMixer, "*": _AttnMixer}[kind]()


def _nemo_stack(pattern: str = NEMO_PATTERN, path: str = "model") -> nn.Module:
    m = nn.Module()
    layers = nn.ModuleList([_NemoBlock(c) for c in pattern])
    holder = nn.Module()
    holder.layers = layers
    setattr(m, path, holder)
    m.config = _Cfg(hidden_size=_D)
    return m


# ---- the pattern + fake-stack dispatch -----------------------------------------


def test_nemotron_pattern_facts():
    """The pattern the fakes follow: 42 blocks, 21 Mamba2 + 17 MLP + 4
    attention; the depth-fraction layer round(0.469 x 42) = 20 is an MLP
    block — block-output hooks (psr_*_all, vector, and bidpo, which trains
    a steering VECTOR) do not care; a single-layer o_proj arm (lora_r1 /
    bidpo_lora) would have nothing to target there
    (tests/test_two_si_layer_targets.py)."""
    assert len(NEMO_PATTERN) == 42
    assert [i for i, c in enumerate(NEMO_PATTERN) if c == "*"] == NEMO_ATTN
    assert NEMO_PATTERN.count("M") == 21 and NEMO_PATTERN.count("-") == 17
    assert round(0.469 * 42) == 20 and NEMO_PATTERN[20] == "-"


def test_nemotron_shaped_stack_dispatch():
    """transformers 5.14.1's native class exposes the stack at model.layers
    (base_model_prefix "model"); every block is hookable, only the four
    attention blocks carry an o_proj."""
    m = _nemo_stack()
    assert len(get_decoder_layers(m)) == 42
    assert hidden_size(m) == _D
    assert oproj_layers(m) == NEMO_ATTN


def test_nemotron_backbone_path_dispatch():
    """The checkpoint's own naming (backbone.layers — NVIDIA's remote-code
    modeling and vLLM keep it) dispatches too; inert on the native class,
    a guard for loaders that keep the checkpoint prefix."""
    m = _nemo_stack(path="backbone")
    assert len(get_decoder_layers(m)) == 42
    assert oproj_layers(m) == NEMO_ATTN


def test_attention_oproj_probe_is_shape_keyed():
    """models.attention_oproj accepts a mixer ONLY when it is attention-
    shaped (q/k/v/o all present): a Mamba mixer's out_proj and an MLP mixer
    never match, a lone o_proj without q/k/v never matches, and the classic
    self_attn.o_proj shape still resolves."""
    assert attention_oproj(_NemoBlock("M")) is None
    assert attention_oproj(_NemoBlock("-")) is None
    attn = _NemoBlock("*")
    assert attention_oproj(attn) is attn.mixer.o_proj
    lone = nn.Module()
    lone.mixer = nn.Module()
    lone.mixer.o_proj = nn.Linear(4, 4)
    assert attention_oproj(lone) is None
    classic = nn.Module()
    classic.self_attn = nn.Module()
    classic.self_attn.o_proj = nn.Linear(4, 4)
    assert attention_oproj(classic) is classic.self_attn.o_proj
    assert attention_oproj(nn.Linear(2, 2)) is None


def test_multilayer_lora_targets_the_attention_mixers():
    """MultiLayerLoRAObject on the NemotronH shape: the default layer set is
    the four attention blocks, each hook attaches to THAT block's
    mixer.o_proj, and A/B are sized from the module (in = heads*head_dim,
    out = hidden — non-square), at rank 1 and rank 4."""
    from sisteer.extraction.multilayer import MultiLayerLoRAObject

    m = _nemo_stack()
    layers = get_decoder_layers(m)
    obj = MultiLayerLoRAObject(m)
    assert obj.layers == NEMO_ATTN
    for l in NEMO_ATTN:
        assert obj._o_projs[l] is layers[l].mixer.o_proj
        assert tuple(obj.A[l].shape) == (_ATTN_D,)
        assert tuple(obj.B[l].shape) == (_D,)
    r4 = MultiLayerLoRAObject(m, rank=4)
    for l in NEMO_ATTN:
        assert tuple(r4.A[l].shape) == (_ATTN_D, 4)
        assert tuple(r4.B[l].shape) == (_D, 4)
    with obj.apply(m):
        for l in NEMO_ATTN:
            assert layers[l].mixer.o_proj._forward_hooks
        assert not layers[20].mixer.down_proj._forward_hooks
    for l in NEMO_ATTN:
        assert not layers[l].mixer.o_proj._forward_hooks
    # an explicit list naming a non-attention block still refuses loudly
    with pytest.raises(ValueError, match="no self_attn.o_proj"):
        MultiLayerLoRAObject(m, layers=[12, 20])


def test_rank_one_lora_refuses_the_mlp_depth_fraction_layer():
    """A single-layer o_proj arm (lora_r1 / bidpo_lora — NOT bidpo, whose
    object is a block-output vector) at Nemotron's layer 20 (an MLP block)
    names the attention layers it CAN target; at 12 it attaches to the
    attention mixer's o_proj with module-sized A/B."""
    from sisteer.extraction.objects import RankOneLoRAObject

    m = _nemo_stack()
    with pytest.raises(ValueError, match=r"\[12, 17, 24, 32\]"):
        RankOneLoRAObject(m, layer=20)
    obj = RankOneLoRAObject(m, layer=12)
    assert obj._module is get_decoder_layers(m)[12].mixer.o_proj
    assert tuple(obj.A.shape) == (_ATTN_D,) and tuple(obj.B.shape) == (_D,)


# ---- the REAL class, tiny (from_config, CPU, no downloads) ----------------------


def _tiny_nemotron():
    """A 4-block hybrid with every mixer kind (M - * M), tiny dims, kernels
    off (the pure-torch SSD path — the one the Modal image runs too, since
    mamba_ssm is absent there)."""
    from transformers import AutoModelForCausalLM
    from transformers.models.nemotron_h import NemotronHConfig

    cfg = NemotronHConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        hybrid_override_pattern="M-*M", num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, mamba_num_heads=8,
        mamba_head_dim=8, ssm_state_size=8, n_groups=2, conv_kernel=4,
        chunk_size=8, vocab_size=128, max_position_embeddings=512,
        use_mamba_kernels=False)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m, cfg


def test_real_class_accessors_tiny_nemotron():
    m, cfg = _tiny_nemotron()
    assert type(m).__name__ == "NemotronHForCausalLM"
    assert not hasattr(m, "backbone")  # 5.14.1 remaps the prefix to model.
    layers = get_decoder_layers(m)
    assert len(layers) == 4 and hidden_size(m) == 32
    assert [type(b.mixer).__name__ for b in layers] == [
        "NemotronHMamba2Mixer", "NemotronHMLP", "NemotronHAttention",
        "NemotronHMamba2Mixer"]
    assert oproj_layers(m) == [2]
    assert attention_oproj(layers[2]) is layers[2].mixer.o_proj
    assert list(cfg.layer_types) == ["linear_attention", "mlp",
                                     "full_attention", "linear_attention"]


def test_nemotron_default_forward_needs_the_kv_cache_pin():
    """transformers 5.14.1: config.layer_types carry "mlp", which
    DynamicCache has no layer for, so the DEFAULT forward (use_cache=None
    -> config.use_cache True) and generate() raise KeyError('mlp') before
    the first block. models.pin_kv_cache_off_if_unallocatable is what
    load_model_and_tokenizer applies; after it, default forwards run and
    generate() takes the cache-free path. If a future transformers maps
    "mlp", the probe returns [] and this test still holds."""
    from sisteer.models import pin_kv_cache_off_if_unallocatable

    m, _ = _tiny_nemotron()
    ids = torch.randint(5, 128, (2, 12))
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)  # the one shape that always works
    try:
        with torch.no_grad():
            m(input_ids=ids)
        raised = False
    except KeyError as e:
        raised = True
        assert "mlp" in str(e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the pin's own notice
        bad = pin_kv_cache_off_if_unallocatable(m)
    assert raised == (bad == ["mlp"])
    with torch.no_grad():
        out = m(input_ids=ids)
    assert tuple(out.logits.shape) == (2, 12, 128)
    gen = m.generate(ids[:, :5], attention_mask=torch.ones_like(ids[:, :5]),
                     max_new_tokens=2, do_sample=False, pad_token_id=0)
    assert tuple(gen.shape) == (2, 7)


def test_nemotron_block_output_hooks_get_gradients_on_torch_ssm():
    """The PSR premise on this stack: an additive block-OUTPUT hook at every
    block — Mamba2, MLP, attention alike — receives gradient through the
    pure-torch SSD scan (the path the Modal image runs). NemotronHBlock
    returns a plain tensor, the shape hooks.steering already handles."""
    m, _ = _tiny_nemotron()
    m.train()
    layers = get_decoder_layers(m)
    params = {i: nn.Parameter(torch.zeros(32)) for i in range(4)}

    def make(i):
        def hook(_mod, _inp, out):
            assert torch.is_tensor(out)
            return out + params[i]
        return hook

    handles = [layers[i].register_forward_hook(make(i)) for i in range(4)]
    try:
        ids = torch.randint(5, 128, (2, 12))
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids),
                use_cache=False)
        out.logits.float().sum().backward()
    finally:
        for h in handles:
            h.remove()
    for i in range(4):
        assert params[i].grad is not None and params[i].grad.norm() > 0, i


# ---- fused Mamba kernels: REQUIRED, and "used" is read off the live closures ----
#
# transformers 5.15.0 binds each Mamba2 kernel wrapper at IMPORT time to the
# package kernel (mamba_ssm / causal_conv1d) when it imports, else to the
# module's pure-torch function. Nothing re-checks at call time and the config's
# use_mamba_kernels is never read, so `import mamba_ssm` succeeding proves
# nothing about the forward: the resolver reads the closures, the loader
# refuses the fallback.

from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[1]

#: the wrappers the NemotronH modeling module binds (transformers 5.15.0): the
#: causal_conv1d pair and the Mamba2 trio. models.MAMBA_KERNEL_WRAPPERS also
#: carries the Mamba-1 trio, which jamba / mamba / falcon_mamba / zamba bind
#: and nemotron_h does not (review 2026-10-12), so the table is a superset of
#: what any one module resolves.
NEMO_WRAPPERS = frozenset({"causal_conv1d_fn", "causal_conv1d_update", "mamba2_chunk_scan",
                           "mamba2_selective_state_update", "mamba2_split_conv1d_scan_combined"})


def _fake_fused(name: str, module: str):
    """A wrapper shaped like use_kernel_func_from_hub_with_fallback's: closes
    over ``implementation`` (a function claiming ``module``) and the
    applicable-params tuple."""
    def implementation(*a, **k):
        return None
    implementation.__module__ = module
    implementation.__name__ = name
    applicable_params = ("a",)

    def wrapped(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}
        return implementation(*args, **kwargs)
    return wrapped


def test_resolved_mamba_kernels_reads_the_live_closures(monkeypatch, tiny_llama):
    import transformers.models.nemotron_h.modeling_nemotron_h as mod

    from sisteer.models import (MAMBA_KERNEL_WRAPPERS, assert_fused_mamba_kernels,
                                resolved_mamba_kernels)

    import sisteer.models as sm

    m, _ = _tiny_nemotron()
    # the subject is the instantiated mixer (models.mamba_mixer_classes), whose
    # methods call these five wrappers; the norm class NemotronH imports from
    # the zamba2 module calls none and is not one
    assert [c.__name__ for c in sm.mamba_mixer_classes(m)] == ["NemotronHMamba2Mixer"]
    got = resolved_mamba_kernels(m)
    assert set(got) == NEMO_WRAPPERS                   # the mixer's own five, not the table's eight
    assert all(MAMBA_KERNEL_WRAPPERS[n] in ("causal_conv1d", "mamba_ssm") for n in got)
    # this CPU env has neither package: every wrapper IS the torch fallback
    assert all(v == mod.__name__ for v in got.values()), got
    # On cpu (and mps) the fused packages cannot exist, so the guard WARNS once
    # and returns the resolution instead of refusing (2026-10-12 scope; the
    # device-reading cases, and Qwen3.5 -- a gated-deltanet hybrid the guard
    # does not apply to -- are tests/test_models_device_defaults.py's) ...
    with pytest.warns(RuntimeWarning, match="is on cpu") as rec:
        assert assert_fused_mamba_kernels(m) == got
    assert sum(w.category is RuntimeWarning for w in rec) == 1
    assert "['causal_conv1d', 'mamba_ssm']" in str(rec[0].message)
    # ... and on a CUDA device it is the hard refusal it always was: the
    # directive's purpose is that a GPU job never crawls on the fallback
    # (model_devices is the guard's reader: every execution device of the model)
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
    with pytest.raises(RuntimeError, match="REQUIRED") as ei:
        assert_fused_mamba_kernels(m)
    assert "['causal_conv1d', 'mamba_ssm']" in str(ei.value)
    assert "--extra mamba" in str(ei.value)
    # (model_devices stays patched to cuda below: the fused case is device-
    # independent, and the one-wrapper-back refusal is the cuda contract)
    # a non-Mamba architecture: nothing to resolve, nothing to refuse
    llama, _ = tiny_llama
    assert resolved_mamba_kernels(llama) is None and assert_fused_mamba_kernels(llama) is None
    # the packages present: each wrapper closes over a package function
    for n in NEMO_WRAPPERS:
        monkeypatch.setattr(mod, n, _fake_fused(n, MAMBA_KERNEL_WRAPPERS[n] + ".ops.fake"))
    want = {n: MAMBA_KERNEL_WRAPPERS[n] + ".ops.fake" for n in NEMO_WRAPPERS}
    assert resolved_mamba_kernels(m) == want
    assert assert_fused_mamba_kernels(m) == want
    # ONE wrapper back on the fallback: the refusal names it and its package only
    monkeypatch.setattr(mod, "mamba2_chunk_scan", _fake_fused("mamba2_chunk_scan", mod.__name__))
    with pytest.raises(RuntimeError) as ei:
        assert_fused_mamba_kernels(m)
    msg = str(ei.value)
    assert f"mamba2_chunk_scan -> {mod.__name__}" in msg and "['mamba_ssm']" in msg
    assert "causal_conv1d" not in msg


def test_loader_applies_the_fused_kernel_guard(monkeypatch):
    """load_model_and_tokenizer calls assert_fused_mamba_kernels on the LOADED
    model (a no-op on non-Mamba architectures); a refusal there propagates."""
    import sisteer.models as M

    seen = []

    def spy(model):
        seen.append(type(model).__name__)
        raise RuntimeError("fallback refused (test)")
    monkeypatch.setattr(M, "assert_fused_mamba_kernels", spy)
    with pytest.raises(RuntimeError, match="fallback refused"):
        M.load_model_and_tokenizer("hf-internal-testing/tiny-random-LlamaForCausalLM",
                                   device="cpu", dtype=torch.float32)
    assert seen == ["LlamaForCausalLM"]


def test_mamba_kernels_check_blocks_the_packages_before_transformers_binds():
    """scripts/mamba_kernels_check.py --fallback reproduces 470950's path in
    a fused env: with the packages blocked in sys.modules BEFORE transformers
    imports, use_kernel_func_from_hub_with_fallback's import_module raises
    and every wrapper binds to the torch function. --help parses without
    torch."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import mamba_kernels_check as c\n"
        "c._block_fused_imports()\n"
        "import importlib\n"
        "for n in ('mamba_ssm', 'causal_conv1d'):\n"
        "    try:\n"
        "        importlib.import_module(n); print('IMPORTED', n)\n"
        "    except ImportError:\n"
        "        print('BLOCKED', n)\n"
        "import transformers.models.nemotron_h.modeling_nemotron_h as mod\n"
        "from sisteer.models import MAMBA_KERNEL_WRAPPERS, _kernel_implementation\n"
        "names = [n for n in MAMBA_KERNEL_WRAPPERS if hasattr(mod, n)]\n"
        "print(len(names), all(_kernel_implementation(getattr(mod, n)).__module__ == mod.__name__"
        " for n in names))\n"
    ) % str(_REPO / "scripts")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["BLOCKED", "mamba_ssm", "BLOCKED", "causal_conv1d",
                           str(len(NEMO_WRAPPERS)), "True"], out
    h = subprocess.run([sys.executable, str(_REPO / "scripts" / "mamba_kernels_check.py"), "--help"],
                       capture_output=True, text=True)
    assert h.returncode == 0 and "--fallback" in h.stdout and "--rows" in h.stdout


def test_mamba_tiny_proof_refuses_without_the_kernels_and_builds_the_hybrid():
    """scripts/mamba_tiny_proof.py (mamba_build.sbatch's USED-not-importable
    proof after its cache-warming sync) builds a tiny random-init NemotronH
    hybrid with every mixer kind at kernel-friendly Mamba2 shapes and runs the
    loader's own guard on it. This CPU env has neither package, so the proof
    must REFUSE through assert_fused_mamba_kernels (naming the install hint),
    exactly as a job on the fallback would; its pins are pyproject's."""
    import importlib
    import subprocess
    import sys
    import tomllib

    sys.path.insert(0, str(_REPO / "scripts"))
    try:
        proof = importlib.import_module("mamba_tiny_proof")
    finally:
        sys.path.pop(0)
    pins = {d.split(" ;")[0].split("==")[0]: d.split(" ;")[0].split("==")[1]
            for d in tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]["mamba"]
            if "==" in d and not d.startswith("torch")}
    assert proof.KERNEL_PINS == pins == {"mamba-ssm": "2.3.1", "causal-conv1d": "1.6.2.post1"}
    cfg = proof.tiny_config()
    assert cfg.use_mamba_kernels is True                      # kernels ON, unlike _tiny_nemotron
    assert list(cfg.layer_types) == ["linear_attention", "mlp", "full_attention", "linear_attention"]
    assert (cfg.mamba_head_dim, cfg.ssm_state_size, cfg.n_groups, cfg.chunk_size) == (64, 128, 8, 128)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_config(cfg)
    assert [type(b.mixer).__name__ for b in get_decoder_layers(m)] == [
        "NemotronHMamba2Mixer", "NemotronHMLP", "NemotronHAttention", "NemotronHMamba2Mixer"]
    with pytest.raises(RuntimeError, match="REQUIRED") as ei:
        proof.main(["--device", "cpu"])
    assert "--extra mamba" in str(ei.value) and "['causal_conv1d', 'mamba_ssm']" in str(ei.value)
    h = subprocess.run([sys.executable, str(_REPO / "scripts" / "mamba_tiny_proof.py"), "--help"],
                       capture_output=True, text=True)
    assert h.returncode == 0 and "--device" in h.stdout and "--json" in h.stdout
