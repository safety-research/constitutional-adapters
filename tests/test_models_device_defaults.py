"""The README's worked example off CUDA: an Apple-Silicon GPU (torch ``mps``) or a
plain CPU, installed with ``uv sync --frozen --extra cpu`` (2026-10-12).

CPU-only, availability MOCKED (this box has neither CUDA nor MPS), no weights, no
network:

- ``models.default_device`` orders cuda > mps > cpu off ``torch.cuda.is_available``
  / ``torch.backends.mps.is_available``;
- ``models.default_dtype``: bf16 on the bare ``"cuda"`` spelling (verbatim the old
  rule, an indexed ``"cuda:0"`` still fp32), bf16 on mps when torch's own macOS-14
  gate (``torch.backends.mps.is_macos_or_newer(14, 0)``, the question
  ``MpsInterface.is_bf16_supported`` asks) says yes, fp16 when it says no or cannot
  say, fp32 on cpu;
- ``load_model_and_tokenizer`` applies both defaults and an explicit ``device=`` /
  ``dtype=`` wins over either;
- ``assert_fused_mamba_kernels`` on a model with Mamba mixers (a tiny real
  NemotronH, ``test_nemotron_arch._tiny_nemotron``, whose modeling module binds
  the CUDA-only ``causal_conv1d`` and ``mamba_ssm`` wrappers) RAISES when the model
  can execute on cuda — read through ``models.model_devices`` (every parameter and
  buffer, Accelerate's per-module execution devices through every
  ``SequentialHook`` nested to any depth — ``models.hook_execution_devices``, since
  a repeated ``add_hook_to_module(..., append=True)`` nests one container in
  another — and ``hf_device_map``), so a partitioned model whose first parameter
  sits on cpu or meta while a block executes on cuda is refused too, with nothing
  patched, and so is one whose only cuda marker is an ``AlignDevicesHook`` two
  containers down with no ``hf_device_map`` at all — and
  WARNS once — no raise — on mps and on cpu, naming the device and the missing
  packages; the loader surfaces that warning and still returns the model;
- the guard's SUBJECT is a model that INSTANTIATES a Mamba mixer and nothing else
  (``models.mamba_mixer_classes``: a submodule class whose own methods, decorators
  unwrapped, call a ``mamba_ssm`` wrapper bound in their globals): a tiny Qwen3.5
  (the README's own architecture, a gated-deltanet hybrid whose
  ``Qwen3_5GatedDeltaNet`` calls only the ``causal_conv1d`` pair, for its short
  convolution) gets ``None`` from ``resolved_mamba_kernels`` and from the guard,
  with no warning, on cpu and with ``model_devices`` reporting cuda, and the loader
  emits no fused-kernel warning for it (Qwen3.5 is
  not a Mamba hybrid; only Nemotron is — the earlier reading, any wrapper as the
  marker, refused it on a GPU node synced without the mamba extra); and so does a
  Jamba configured with attention layers only (``attn_layer_period=1``,
  ``attn_layer_offset=0``: no ``JambaMambaMixer`` is built), although its modeling
  module binds every wrapper — the reading that took the model's class module as
  the marker refused that model on cuda;
- the subject includes the Mamba-1 families: tiny real Jamba, Zamba, Mamba and
  FalconMamba models, whose mixers call the Mamba-1 ``mamba_ssm`` names
  (``mamba_inner_fn`` / ``mamba_selective_state_update`` / ``mamba_selective_scan``
  for Mamba and FalconMamba; the last two for Jamba, whose module binds
  ``mamba_inner_fn`` without its mixer calling it, and for Zamba) rather than the
  Mamba2 trio, warn once on cpu and are refused on cuda like NemotronH, through the
  guard and through the loader; the reading that took the Mamba2 trio as the only
  marker let every one of them through on cuda with each wrapper on the fallback
  and no warning. The wrapper table itself is pinned to the
  installed transformers' source, since the wrapper records no package and the name
  is all that is left of one, and the ten mixer classes to the installed modules;
- ``load_object`` moves an object's tensors to the model's device
  (``models.model_device``) and ``apply`` serves a model in the model's own dtype
  (fp32 and bf16 on CPU), the object's fp32 math cast per call as on CUDA.
"""

from __future__ import annotations

import copy
import sys
import types
import warnings

import pytest
import torch

import sisteer.models as sm
from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                           MultiLayerLoRAObject,
                                           MultiLayerVectorObject)
from sisteer.generate import generate_with_object
from sisteer.methods import load_object
from test_nemotron_arch import NEMO_WRAPPERS, _fake_fused, _tiny_nemotron

#: every wrapper the NemotronH modeling module binds: the causal_conv1d pair
#: and the Mamba2 trio, all five on the torch fallback in this env (the
#: table's Mamba-1 trio is the Mamba-1 modules' and nemotron_h binds none)
_NEMO_WRAPPERS = set(NEMO_WRAPPERS)
#: the wrappers the Mamba-1 modeling modules bind (transformers 5.15.0) ...
_CONV_PAIR = frozenset({"causal_conv1d_fn", "causal_conv1d_update"})
_MAMBA1_TRIO = frozenset({"mamba_inner_fn", "mamba_selective_state_update", "mamba_selective_scan"})
#: ... and the two of them the Jamba and Zamba mixers actually call
_MAMBA1_PAIR = frozenset({"mamba_selective_state_update", "mamba_selective_scan"})


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #


def _availability(monkeypatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def _macos_gate(monkeypatch, answer):
    """``torch.backends.mps.is_macos_or_newer`` returning ``answer`` — or, when
    ``answer`` is an exception instance, raising it (what a torch built without
    the MPS backend does: this Linux CPU wheel raises RuntimeError)."""
    def gate(major, minor):
        assert (major, minor) == (14, 0), (major, minor)   # the bf16 gate torch itself uses
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", gate)


def _tiny_qwen35():
    """A 4-block Qwen3.5 hybrid (3 gated-deltanet blocks + 1 full attention),
    tiny dims — the README's architecture. Its ``Qwen3_5GatedDeltaNet``
    calls ``causal_conv1d_fn`` / ``causal_conv1d_update`` (a short causal
    convolution) and NO ``mamba_ssm`` wrapper, and its module binds none:
    not a Mamba hybrid, so the fused-kernel guard has nothing to say about
    it on any device."""
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5 import Qwen3_5Config

    text_cls = type(Qwen3_5Config().get_text_config())
    cfg = text_cls(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                   num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                   max_position_embeddings=512,
                   layer_types=["linear_attention", "linear_attention",
                                "linear_attention", "full_attention"])
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m


def _tiny_mamba1(family: str, *, attention_only: bool = False):
    """A tiny real Mamba-1 model built from config on CPU, no weights, no
    network: ``jamba`` (a Mamba block and an attention block, one expert so
    no router), ``zamba`` (a Mamba block and two hybrid blocks -- its shared
    transformer ties across hybrid blocks and needs two), ``mamba`` and
    ``falcon_mamba`` (two Mamba blocks). Each one's modeling module binds its
    kernel wrappers to transformers' pure-torch fallback in this env, and
    none of them says so (5.15.0). ``attention_only`` (jamba) places an
    attention layer at every depth (``attn_layer_period=1``,
    ``attn_layer_offset=0``): a valid Jamba with no ``JambaMambaMixer`` in
    it, whose module still binds every wrapper."""
    from transformers import AutoModelForCausalLM

    if family == "jamba":
        from transformers.models.jamba import JambaConfig
        period, offset = (1, 0) if attention_only else (2, 1)
        cfg = JambaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                          attn_layer_period=period, attn_layer_offset=offset, expert_layer_period=2,
                          expert_layer_offset=1, num_experts=1, num_experts_per_tok=1,
                          mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=4,
                          use_mamba_kernels=False)
    elif family == "zamba":
        from transformers.models.zamba import ZambaConfig
        cfg = ZambaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=4, vocab_size=128,
                          mamba_d_state=8, mamba_d_conv=4, mamba_expand=2, mamba_dt_rank=4,
                          n_mamba_heads=2, layers_block_type=["linear_attention", "hybrid", "hybrid"],
                          use_mamba_kernels=False)
    elif family == "mamba":
        from transformers.models.mamba import MambaConfig
        cfg = MambaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, vocab_size=128,
                          state_size=8, conv_kernel=4, expand=2, time_step_rank=4, use_mambapy=False)
    elif family == "falcon_mamba":
        from transformers.models.falcon_mamba import FalconMambaConfig
        cfg = FalconMambaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                vocab_size=128, state_size=8, conv_kernel=4, expand=2,
                                time_step_rank=4, use_mambapy=False)
    else:
        raise ValueError(family)
    assert not attention_only or family == "jamba", family
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg)
    m.eval()
    return m


class _FakeTok:
    eos_token_id = 1
    chat_template = None
    pad_token = "<pad>"


class _FakeModel:
    """What the loader needs of a model, recording ``.to()``: one CPU
    parameter (so ``model_device`` answers), no config / generation_config
    (prepare_kv_cache and pin_generation_stop_ids return early)."""

    config = None
    generation_config = None

    def __init__(self):
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.moved_to = []

    def parameters(self):
        yield self._p

    def buffers(self):
        return iter(())

    def to(self, device):
        self.moved_to.append(device)
        return self

    def eval(self):
        return self


def _stub_loader(monkeypatch, model):
    """Route ``load_model_and_tokenizer`` around the hub: a fake tokenizer, a
    fake auto-class whose ``from_pretrained`` records its kwargs and returns
    ``model``. Returns the record."""
    seen: dict = {}

    def from_pretrained(name, **kw):
        seen["name"] = name
        seen.update(kw)
        return model

    monkeypatch.setattr(sm, "load_tokenizer", lambda name, revision=None, **kw: _FakeTok())
    monkeypatch.setattr(sm, "auto_model_class",
                        lambda name, revision=None, **kw: types.SimpleNamespace(
                            from_pretrained=from_pretrained))
    monkeypatch.setattr(sm, "check_chat_template", lambda tok, **kw: None)
    monkeypatch.setattr(sm, "_enforce_llama31_mirror", lambda name: None)
    return seen


# --------------------------------------------------------------------------- #
# default_device / default_dtype                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cuda, mps, want", [
    (True, True, "cuda"),      # cuda first, even when mps is also there
    (True, False, "cuda"),
    (False, True, "mps"),      # the Apple-Silicon case
    (False, False, "cpu"),
])
def test_default_device_order_is_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, want):
    _availability(monkeypatch, cuda=cuda, mps=mps)
    assert sm.default_device() == want


def test_default_dtype_cuda_and_cpu_are_the_historical_rule():
    assert sm.default_dtype("cuda") is torch.bfloat16
    assert sm.default_dtype("cpu") is torch.float32
    assert sm.default_dtype(torch.device("cpu")) is torch.float32
    # the old rule was a string compare on the bare spelling; an indexed cuda
    # device got fp32 then and gets fp32 now (byte-identical CUDA behaviour)
    assert sm.default_dtype("cuda:0") is torch.float32


def test_default_dtype_mps_follows_torchs_macos14_bf16_gate(monkeypatch):
    _macos_gate(monkeypatch, True)
    assert sm.default_dtype("mps") is torch.bfloat16
    assert sm.default_dtype(torch.device("mps")) is torch.bfloat16
    _macos_gate(monkeypatch, False)
    assert sm.default_dtype("mps") is torch.float16
    # a torch that cannot answer (no MPS backend built) gets the dtype every
    # MPS build runs, never a crash and never fp32
    _macos_gate(monkeypatch, RuntimeError("Cannot execute isOnMacOSorNewer() without MPS backend."))
    assert sm.default_dtype("mps") is torch.float16


# --------------------------------------------------------------------------- #
# load_model_and_tokenizer applies the defaults; explicit args win              #
# --------------------------------------------------------------------------- #


def test_loader_defaults_to_mps_bf16_on_an_apple_gpu(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=True)
    _macos_gate(monkeypatch, True)
    fake = _FakeModel()
    seen = _stub_loader(monkeypatch, fake)
    model, tok = sm.load_model_and_tokenizer("fake/qwen")
    assert model is fake and isinstance(tok, _FakeTok)
    assert seen["torch_dtype"] is torch.bfloat16
    assert fake.moved_to == ["mps"]


def test_loader_defaults_to_mps_fp16_below_macos14(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=True)
    _macos_gate(monkeypatch, False)
    fake = _FakeModel()
    seen = _stub_loader(monkeypatch, fake)
    sm.load_model_and_tokenizer("fake/qwen")
    assert seen["torch_dtype"] is torch.float16
    assert fake.moved_to == ["mps"]


def test_loader_defaults_to_cuda_bf16_when_cuda_is_there(monkeypatch):
    """CUDA wins over mps and keeps bf16 — the cluster's load, unchanged."""
    _availability(monkeypatch, cuda=True, mps=True)
    fake = _FakeModel()
    seen = _stub_loader(monkeypatch, fake)
    sm.load_model_and_tokenizer("fake/qwen")
    assert seen["torch_dtype"] is torch.bfloat16
    assert fake.moved_to == ["cuda"]


def test_loader_defaults_to_cpu_fp32_with_no_accelerator(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=False)
    fake = _FakeModel()
    seen = _stub_loader(monkeypatch, fake)
    sm.load_model_and_tokenizer("fake/qwen")
    assert seen["torch_dtype"] is torch.float32
    assert fake.moved_to == ["cpu"]


def test_explicit_device_and_dtype_win_over_the_defaults(monkeypatch):
    _availability(monkeypatch, cuda=False, mps=True)
    _macos_gate(monkeypatch, True)
    fake = _FakeModel()
    seen = _stub_loader(monkeypatch, fake)
    sm.load_model_and_tokenizer("fake/qwen", device="cpu", dtype="float16")
    assert seen["torch_dtype"] is torch.float16          # the string spelling resolves
    assert fake.moved_to == ["cpu"]
    fake2 = _FakeModel()
    seen2 = _stub_loader(monkeypatch, fake2)
    sm.load_model_and_tokenizer("fake/qwen", device="mps", dtype=torch.float32)
    assert seen2["torch_dtype"] is torch.float32         # a torch.dtype is taken as is
    assert fake2.moved_to == ["mps"]


# --------------------------------------------------------------------------- #
# the fused-kernel guard: hard on cuda, one warning on mps / cpu                #
# --------------------------------------------------------------------------- #


def test_guard_warns_on_cpu_and_mps_but_raises_on_cuda(monkeypatch):
    m, _ = _tiny_nemotron()
    assert [c.__name__ for c in sm.mamba_mixer_classes(m)] == ["NemotronHMamba2Mixer"]
    assert sm.mamba_modeling_module(m) is sys.modules[type(m).__module__]
    got = sm.resolved_mamba_kernels(m)
    # a Mamba2 hybrid's mixer calls the causal_conv1d pair AND the Mamba2 mamba_ssm
    # trio, all five on the torch fallback in this env (neither CUDA-only package)
    assert set(got) == _NEMO_WRAPPERS
    assert all(v == type(m).__module__ for v in got.values()), got
    assert sm.model_device(m).type == "cpu"
    assert sm.model_devices(m) == {torch.device("cpu")}

    # cpu: ONE RuntimeWarning, no raise, the resolution returned for the record
    with pytest.warns(RuntimeWarning) as rec:
        assert sm.assert_fused_mamba_kernels(m) == got
    msgs = [str(w.message) for w in rec if w.category is RuntimeWarning]
    assert len(msgs) == 1, msgs
    msg = msgs[0]
    assert "NemotronHForCausalLM is on cpu" in msg
    assert "['causal_conv1d', 'mamba_ssm']" in msg and "CUDA-only" in msg
    assert "its Mamba mixers run on transformers' pure-torch fallback" in msg and "SLOW" in msg
    assert "expected path on cpu" in msg
    assert "cuda only" in msg

    # mps: the same warning, naming mps (model_devices is the guard's reader)
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("mps")})
    with pytest.warns(RuntimeWarning, match="is on mps") as rec:
        assert sm.assert_fused_mamba_kernels(m) == got
    assert sum(w.category is RuntimeWarning for w in rec) == 1
    assert "expected path on mps" in str(rec[0].message)

    # cuda: the hard refusal, the same message as before, nothing warned
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeError, match="REQUIRED") as ei:
            sm.assert_fused_mamba_kernels(m)
    text = str(ei.value)
    assert "NemotronHForCausalLM would run its Mamba mixers on transformers' pure-torch fallback" in text
    assert "(NemotronHMamba2Mixer: causal_conv1d_fn -> " in text       # the mixer, then its wrappers
    assert "['causal_conv1d', 'mamba_ssm']" in text and "--extra mamba" in text
    assert "never proceed on the fallback" in text


def test_guard_is_silent_on_cuda_with_the_kernels_bound(monkeypatch):
    """A cuda model WITH the fused kernels: the resolution, no warning, no
    raise — the fused case is device-independent."""
    m, _ = _tiny_nemotron()
    mod = sys.modules[type(m).__module__]
    for n in _NEMO_WRAPPERS:
        monkeypatch.setattr(mod, n, _fake_fused(n, sm.MAMBA_KERNEL_WRAPPERS[n] + ".ops.fake"))
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sm.assert_fused_mamba_kernels(m) == {
            n: sm.MAMBA_KERNEL_WRAPPERS[n] + ".ops.fake" for n in _NEMO_WRAPPERS}


def test_guard_is_a_no_op_off_the_mamba_families(tiny_llama):
    llama, _ = tiny_llama
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sm.mamba_mixer_classes(llama) == ()
        assert sm.mamba_modeling_module(llama) is None
        assert sm.resolved_mamba_kernels(llama) is None
        assert sm.assert_fused_mamba_kernels(llama) is None


def test_qwen35_is_not_a_mamba_hybrid_so_the_guard_says_nothing(monkeypatch):
    """Qwen3.5 is NOT a Mamba hybrid; only
    Nemotron is. Its modeling module binds ``causal_conv1d_fn`` /
    ``causal_conv1d_update`` for the gated-deltanet short convolution and NO
    ``mamba_ssm`` wrapper, and a reading that took any wrapper as the Mamba2
    marker refused a GPU node synced with ``--extra cu128`` alone. The
    subject test is an instantiated class calling a ``mamba_ssm`` wrapper:
    its ``Qwen3_5GatedDeltaNet`` calls the ``causal_conv1d`` pair and nothing
    else, so ``resolved_mamba_kernels`` and the guard return ``None`` with no
    warning, on cpu and when the model reports cuda — the device reader is
    never even consulted."""
    m = _tiny_qwen35()
    mod = sys.modules[type(m).__module__]
    # the facts about the architecture the rule rests on, pinned
    assert type(m).__name__ == "Qwen3_5ForCausalLM"
    assert all(callable(getattr(mod, n, None)) for n in ("causal_conv1d_fn", "causal_conv1d_update"))
    assert not any(hasattr(mod, n) for n in sm.MAMBA_SSM_WRAPPERS)
    assert set(sm.MAMBA_SSM_WRAPPERS) == {"mamba2_chunk_scan", "mamba2_selective_state_update",
                                          "mamba2_split_conv1d_scan_combined"} | _MAMBA1_TRIO
    gdn = {type(sub) for sub in m.modules() if type(sub).__name__ == "Qwen3_5GatedDeltaNet"}
    assert len(gdn) == 1
    assert dict(sm.kernel_wrappers_called(gdn.pop())) == {n: mod.__name__ for n in _CONV_PAIR}
    assert sm.mamba_mixer_classes(m) == ()
    assert sm.mamba_modeling_module(m) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # any warning fails
        assert sm.resolved_mamba_kernels(m) is None
        assert sm.assert_fused_mamba_kernels(m) is None        # cpu

        def never(model):
            raise AssertionError("model_devices consulted for a non-Mamba model")
        monkeypatch.setattr(sm, "model_devices", never)
        assert sm.assert_fused_mamba_kernels(m) is None
        monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
        assert sm.assert_fused_mamba_kernels(m) is None        # "on cuda": still nothing


def test_loader_emits_no_fused_kernel_warning_for_qwen35(monkeypatch):
    """The README's load (Qwen3.5-4B) through ``load_model_and_tokenizer``:
    no fused-kernel warning and no refusal, on cpu and with the model
    reporting a cuda execution device."""
    m = _tiny_qwen35()
    _stub_loader(monkeypatch, m)
    monkeypatch.setattr(sm, "pin_generation_stop_ids", lambda model, tok: [])

    def kernel_talk(rec):
        return [str(w.message) for w in rec
                if any(s in str(w.message) for s in ("fused", "pure-torch", "mamba", "Mamba"))]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model, _tok = sm.load_model_and_tokenizer("fake/qwen", device="cpu", dtype="float32")
    assert model is m
    assert kernel_talk(rec) == []
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model, _tok = sm.load_model_and_tokenizer("fake/qwen", device="cpu", dtype="float32")
    assert model is m
    assert kernel_talk(rec) == []


def test_guard_ignores_a_mamba_family_model_built_without_a_mixer(monkeypatch):
    """Review 2026-10-12: a valid Jamba with ``attn_layer_period=1`` and
    ``attn_layer_offset=0`` instantiates ``JambaAttentionDecoderLayer`` only
    — no ``JambaMambaMixer``, so no Mamba kernel can run — while its
    modeling module binds all five wrappers; the reading that took the
    model's class module as the marker refused it on cuda. The marker is the
    INSTANTIATED mixer: this model is no subject (``None``, no warning, on
    cpu and reporting cuda, the device reader never consulted, through the
    guard and through the loader), and the hybrid Jamba from the same module
    is one (``test_guard_covers_the_mamba1_families``)."""
    m = _tiny_mamba1("jamba", attention_only=True)
    mod = sys.modules[type(m).__module__]
    classes = {type(sub).__name__ for sub in m.modules()}
    assert "JambaAttentionDecoderLayer" in classes
    assert not any("Mamba" in c for c in classes), sorted(classes)
    # the module-level facts the wrong reading rested on: every wrapper bound, on the fallback
    assert all(callable(getattr(mod, n, None)) for n in _CONV_PAIR | _MAMBA1_TRIO)
    assert all(sm._kernel_implementation(getattr(mod, n)).__module__ == mod.__name__
               for n in _CONV_PAIR | _MAMBA1_TRIO)
    assert all(sm.kernel_wrappers_called(type(sub)) == () for sub in m.modules())
    assert sm.mamba_mixer_classes(m) == ()
    assert sm.mamba_modeling_module(m) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # any warning fails
        assert sm.resolved_mamba_kernels(m) is None
        assert sm.assert_fused_mamba_kernels(m) is None        # cpu

        def never(model):
            raise AssertionError("model_devices consulted for a model with no Mamba mixer")
        monkeypatch.setattr(sm, "model_devices", never)
        assert sm.assert_fused_mamba_kernels(m) is None
        monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
        assert sm.assert_fused_mamba_kernels(m) is None        # "on cuda": still nothing
        with torch.no_grad():
            assert torch.isfinite(m(torch.tensor([[1, 2, 3, 4, 5]])).logits).all()
    # through the loader, cpu and "cuda" alike: the model comes back, nothing said
    _stub_loader(monkeypatch, m)
    monkeypatch.setattr(sm, "pin_generation_stop_ids", lambda model, tok: [])
    for devices in ({torch.device("cpu")}, {torch.device("cuda", 0)}):
        monkeypatch.setattr(sm, "model_devices", lambda model, d=devices: d)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            model, _tok = sm.load_model_and_tokenizer("fake/jamba-attn", device="cpu", dtype="float32")
        assert model is m
        assert not [str(w.message) for w in rec
                    if any(s in str(w.message) for s in ("fused", "pure-torch", "mamba", "Mamba"))]


# --------------------------------------------------------------------------- #
# the Mamba-1 families are subjects too (review 2026-10-12)                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family, cls_name, mixer, wrappers", [
    ("jamba", "JambaForCausalLM", "JambaMambaMixer", _CONV_PAIR | _MAMBA1_PAIR),
    ("zamba", "ZambaForCausalLM", "ZambaMambaMixer", _CONV_PAIR | _MAMBA1_PAIR),
    ("mamba", "MambaForCausalLM", "MambaMixer", _CONV_PAIR | _MAMBA1_TRIO),
    ("falcon_mamba", "FalconMambaForCausalLM", "FalconMambaMixer", _CONV_PAIR | _MAMBA1_TRIO),
])
def test_guard_covers_the_mamba1_families(monkeypatch, family, cls_name, mixer, wrappers):
    """Review 2026-10-12: with the Mamba2 trio as the only marker, a real
    two-layer Jamba (a ``JambaMambaDecoderLayer`` and all) had every wrapper
    on the fallback and the guard returned ``None`` even with
    ``model_devices`` reporting cuda:0 -- no refusal, no warning -- and so
    did Mamba, FalconMamba and Zamba, whose mixers call the Mamba-1 names.
    The marker is an instantiated class calling any ``mamba_ssm`` wrapper:
    each family is a subject, its resolution is the wrappers its mixer CALLS
    (all on the fallback here; Jamba's module binds ``mamba_inner_fn`` too,
    which its mixer never calls, so the resolution leaves it out), the
    fallback forward itself says nothing (the silence the guard is for), cpu
    warns once, cuda refuses, and with the packages bound cuda is silent."""
    m = _tiny_mamba1(family)
    assert type(m).__name__ == cls_name
    assert [c.__name__ for c in sm.mamba_mixer_classes(m)] == [mixer]
    mod = sm.mamba_modeling_module(m)
    assert mod is not None and mod is sys.modules[type(m).__module__]
    assert set(getattr(sm, "MAMBA_SSM_WRAPPERS")) >= wrappers - _CONV_PAIR
    assert all(callable(getattr(mod, n, None)) for n in _CONV_PAIR | _MAMBA1_TRIO) == (family != "zamba")
    got = sm.resolved_mamba_kernels(m)
    assert set(got) == wrappers
    assert all(v == mod.__name__ for v in got.values()), got    # every wrapper on the fallback
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with torch.no_grad():
            logits = m(torch.tensor([[1, 2, 3, 4, 5]])).logits
    assert torch.isfinite(logits).all()
    assert not [str(w.message) for w in rec
                if any(s in str(w.message).lower() for s in ("kernel", "fallback", "mamba", "fast path"))]
    # cpu: ONE warning naming the class, both packages and the Mamba mixers
    with pytest.warns(RuntimeWarning) as rec:
        assert sm.assert_fused_mamba_kernels(m) == got
    msgs = [str(w.message) for w in rec if w.category is RuntimeWarning]
    assert len(msgs) == 1, msgs
    assert f"{cls_name} is on cpu" in msgs[0] and "['causal_conv1d', 'mamba_ssm']" in msgs[0]
    assert "its Mamba mixers run on transformers' pure-torch fallback" in msgs[0]
    # cuda: the refusal, no warning
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeError, match="REQUIRED") as ei:
            sm.assert_fused_mamba_kernels(m)
    text = str(ei.value)
    assert f"{cls_name} would run its Mamba mixers on transformers' pure-torch fallback ({mixer}: " in text
    assert "['causal_conv1d', 'mamba_ssm']" in text and "never proceed on the fallback" in text
    # the packages bound: silence on cuda, the resolution for the record
    for n in wrappers:
        monkeypatch.setattr(mod, n, _fake_fused(n, sm.MAMBA_KERNEL_WRAPPERS[n] + ".ops.fake"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sm.assert_fused_mamba_kernels(m) == {
            n: sm.MAMBA_KERNEL_WRAPPERS[n] + ".ops.fake" for n in wrappers}


def test_loader_warns_for_a_mamba1_hybrid_on_cpu_and_refuses_it_on_cuda(monkeypatch):
    """Through ``load_model_and_tokenizer``: a tiny Jamba on cpu warns once
    and comes back; reporting a cuda execution device it is refused at
    load, as a NemotronH is."""
    m = _tiny_mamba1("jamba")
    _stub_loader(monkeypatch, m)
    monkeypatch.setattr(sm, "pin_generation_stop_ids", lambda model, tok: [])
    with pytest.warns(RuntimeWarning, match="JambaForCausalLM is on cpu") as rec:
        model, _tok = sm.load_model_and_tokenizer("fake/jamba", device="cpu", dtype="float32")
    assert model is m
    assert sum("pure-torch fallback" in str(w.message) for w in rec) == 1
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
    with pytest.raises(RuntimeError, match="REQUIRED"):
        sm.load_model_and_tokenizer("fake/jamba", device="cpu", dtype="float32")


def test_wrapper_table_is_the_installed_transformers_bindings():
    """``MAMBA_KERNEL_WRAPPERS`` is a name list because the wrapper records
    no package (its closure holds the implementation and a parameter tuple),
    so this pins the list to the installed transformers' SOURCE: every
    ``@use_kernel_func_from_hub_with_fallback(<kernel>, <pkg>)`` with ``pkg``
    in {mamba_ssm, causal_conv1d} names a ``def`` the table carries under
    that package and the table carries nothing else; the modules binding a
    ``mamba_ssm`` wrapper include the ten Mamba families and never qwen3_5;
    each family binds the trio the guard expects of it. A transformers bump
    that renames or adds a wrapper fails here instead of narrowing the
    guard."""
    import re
    from pathlib import Path

    import transformers

    pat = re.compile(
        r'@use_kernel_func_from_hub_with_fallback\(\s*"(?P<kernel>\w+)"\s*,\s*"(?P<pkg>\w+)"\s*,?\s*\)'
        r'\s*(?:@[^\n]*\s*)*def\s+(?P<name>\w+)\(')
    bound: dict[str, set[str]] = {}            # def name -> packages it is bound to
    by_module: dict[str, set[str]] = {}        # model dir -> its mamba_ssm def names
    for f in sorted((Path(transformers.__file__).parent / "models").glob("*/modeling_*.py")):
        text = f.read_text(encoding="utf-8")
        if "use_kernel_func_from_hub_with_fallback" not in text:
            continue
        for mm in pat.finditer(text):
            if mm["pkg"] not in ("mamba_ssm", "causal_conv1d"):
                continue
            bound.setdefault(mm["name"], set()).add(mm["pkg"])
            if mm["pkg"] == "mamba_ssm":
                by_module.setdefault(f.parent.name, set()).add(mm["name"])
    assert bound, "no fallback-wrapped mamba_ssm / causal_conv1d kernel in transformers' models"
    assert all(len(pkgs) == 1 for pkgs in bound.values()), bound
    assert {n: next(iter(pkgs)) for n, pkgs in bound.items()} == sm.MAMBA_KERNEL_WRAPPERS
    assert set(sm.MAMBA_SSM_WRAPPERS) == {n for n, pkgs in bound.items() if "mamba_ssm" in pkgs}
    mamba2 = {"nemotron_h", "mamba2", "bamba", "falcon_h1", "zamba2", "granitemoehybrid"}
    mamba1 = {"mamba", "falcon_mamba", "jamba", "zamba"}
    assert (mamba2 | mamba1) <= set(by_module), sorted(by_module)
    assert not {"qwen3_5", "qwen3_5_moe", "qwen3_next", "olmo_hybrid", "lfm2"} & set(by_module)
    for d in mamba2:
        assert by_module[d] == set(_NEMO_WRAPPERS) - _CONV_PAIR, d
    for d in ("mamba", "falcon_mamba", "jamba"):
        assert by_module[d] == _MAMBA1_TRIO, d
    assert by_module["zamba"] == _MAMBA1_PAIR


def test_mamba_mixer_classes_are_the_ten_of_the_installed_transformers():
    """The structural rule against the installed transformers: in every
    modeling module that binds a ``mamba_ssm`` / ``causal_conv1d`` wrapper
    (the files ``test_wrapper_table_is_the_installed_transformers_bindings``
    scans), the ``nn.Module`` classes whose methods call a ``mamba_ssm``
    wrapper are exactly the ten mixers, one per Mamba family, each calling
    the names its family's module binds (Jamba's and Zamba's two of the
    Mamba-1 trio); the classes calling only the ``causal_conv1d`` pair are
    the short-convolution / gated-deltanet classes of the non-Mamba modules,
    Qwen3.5's ``Qwen3_5GatedDeltaNet`` among them; and no class names a
    wrapper its own module does not bind. A transformers bump that moves a
    kernel call fails here instead of silently changing the guard's
    subject."""
    import importlib
    from pathlib import Path

    import transformers

    mixers: dict[str, set[str]] = {}
    conv_only: set[str] = set()
    for f in sorted((Path(transformers.__file__).parent / "models").glob("*/modeling_*.py")):
        text = f.read_text(encoding="utf-8")
        if "use_kernel_func_from_hub_with_fallback" not in text:
            continue
        if not any(f'"{pkg}"' in text for pkg in ("mamba_ssm", "causal_conv1d")):
            continue
        mod = importlib.import_module(f"transformers.models.{f.parent.name}.{f.stem}")
        for cname, cls in vars(mod).items():
            if not (isinstance(cls, type) and issubclass(cls, torch.nn.Module) and cls.__module__ == mod.__name__):
                continue
            called = dict(sm.kernel_wrappers_called(cls))
            if not called:
                continue
            assert set(called.values()) == {mod.__name__}, (cname, called)
            if set(called) & set(sm.MAMBA_SSM_WRAPPERS):
                mixers[cname] = set(called)
            else:
                assert set(called) <= _CONV_PAIR, (cname, called)
                conv_only.add(cname)
    mamba2 = _NEMO_WRAPPERS
    assert mixers == {
        "NemotronHMamba2Mixer": mamba2, "Mamba2Mixer": mamba2, "BambaMixer": mamba2,
        "FalconH1Mixer": mamba2, "Zamba2MambaMixer": mamba2, "GraniteMoeHybridMambaLayer": mamba2,
        "MambaMixer": _CONV_PAIR | _MAMBA1_TRIO, "FalconMambaMixer": _CONV_PAIR | _MAMBA1_TRIO,
        "JambaMambaMixer": _CONV_PAIR | _MAMBA1_PAIR, "ZambaMambaMixer": _CONV_PAIR | _MAMBA1_PAIR,
    }, mixers
    assert {"Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet", "Qwen3NextGatedDeltaNet",
            "OlmoHybridGatedDeltaNet", "Lfm2ShortConv"} <= conv_only, sorted(conv_only)
    assert not {c for c in conv_only if "Mamba" in c}, sorted(conv_only)


def test_model_device_reads_parameters_then_buffers_then_cpu():
    class _Buf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("b", torch.zeros(2))
    assert sm.model_device(_tiny_qwen35()) == torch.device("cpu")
    assert sm.model_device(_Buf()) == torch.device("cpu")
    assert sm.model_device(torch.nn.Module()) == torch.device("cpu")


def test_model_devices_reads_every_weight_hook_and_device_map_entry():
    """``model_devices`` is the set of EXECUTION devices: every parameter and
    buffer (``meta`` skipped), each Accelerate hook's ``execution_device`` (a
    ``SequentialHook``'s members too; an int is a cuda ordinal, as Accelerate
    writes it) and every ``hf_device_map`` entry (``disk`` is storage, not a
    device). No accelerator is touched: the cuda values are the markers a
    ``device_map=`` load leaves on a model, which is all the reader needs."""
    cpu = torch.device("cpu")

    def cuda(i):
        return torch.device("cuda", i)

    class _Leaf(torch.nn.Module):
        def __init__(self, device="cpu"):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2, device=device))
            self.register_buffer("b", torch.zeros(1, device=device))

    root = torch.nn.Sequential(_Leaf(), _Leaf("meta"))
    assert sm.model_devices(root) == {cpu}                    # meta is a placeholder
    root[1]._hf_hook = types.SimpleNamespace(execution_device=cuda(1))
    assert sm.model_devices(root) == {cpu, cuda(1)}           # the offloaded leaf runs on cuda:1
    root[0]._hf_hook = types.SimpleNamespace(hooks=(
        types.SimpleNamespace(execution_device=0),             # an int ordinal
        types.SimpleNamespace(execution_device=None)))         # a hook with no placement
    assert sm.model_devices(root) == {cpu, cuda(0), cuda(1)}
    # a container INSIDE a container (what a second ``append=True`` builds):
    # the inner members are read too, at any depth, and a container that
    # reaches itself is walked once (identity-keyed), not forever
    inner = types.SimpleNamespace(hooks=[
        types.SimpleNamespace(execution_device="cuda:4"),
        types.SimpleNamespace(hooks=(types.SimpleNamespace(execution_device=cuda(5)),))])
    outer = types.SimpleNamespace(execution_device=None, hooks=(inner, root[0]._hf_hook))
    inner.hooks.append(outer)                                  # a cycle
    root[0]._hf_hook = outer
    assert sm.model_devices(root) == {cpu, cuda(0), cuda(1), cuda(4), cuda(5)}
    # ... raw as the hooks hold them, in the hooks' own order, each once
    assert [d for d in sm.hook_execution_devices(outer) if d is not None] == ["cuda:4", cuda(5), 0]
    # a hook whose ``hooks`` is not a sequence (a string, a non-iterable) is
    # a leaf: its own execution_device and nothing else
    assert sm.hook_execution_devices(types.SimpleNamespace(execution_device=2, hooks="x")) == [2]
    assert sm.hook_execution_devices(types.SimpleNamespace(execution_device=3, hooks=7)) == [3]
    assert sm.hook_execution_devices(None) == []
    root.hf_device_map = {"0": "cpu", "1": "disk", "2": "cuda:2", "3": 3}
    assert sm.model_devices(root) == {cpu, cuda(0), cuda(1), cuda(2), cuda(3), cuda(4), cuda(5)}
    # the singular reader still answers from the first parameter alone
    assert sm.model_device(root) == cpu
    # nothing placed anywhere: an empty set (the guard then says cpu); the
    # loader's parameters-only fakes and a real tiny model read as cpu
    assert sm.model_devices(torch.nn.Module()) == set()
    assert sm.model_devices(_FakeModel()) == {cpu}
    assert sm.model_devices(_tiny_qwen35()) == {cpu}


def test_guard_refuses_a_partitioned_model_whose_mixer_can_execute_on_cuda(monkeypatch):
    """A model whose FIRST parameter is on cpu — or on ``meta``, where Accelerate
    parks an offloaded weight — while a later part executes on cuda is a GPU
    job and is refused like a uniform cuda model, with ``model_device`` left
    unpatched: the guard reads the model's structure (review 2026-10-12: the
    first-parameter read let each of these through with a warning)."""
    def refused(model):
        with warnings.catch_warnings():
            warnings.simplefilter("error")                     # and no warning either
            with pytest.raises(RuntimeError, match="REQUIRED") as ei:
                sm.assert_fused_mamba_kernels(model)
        assert "never proceed on the fallback" in str(ei.value)

    m, _ = _tiny_nemotron()

    # (a) the review's reproduction: a real tiny NemotronH whose PARAMETER
    # walk yields a cpu weight first and a cuda-placed one later (the walk
    # is replaced on the instance; its modules, and so its mixer, are its own)
    p, _ = _tiny_nemotron()
    cpu_first = [torch.nn.Parameter(torch.zeros(1)), types.SimpleNamespace(device=torch.device("cuda", 0))]
    p.parameters = lambda recurse=True: iter(cpu_first)
    assert [c.__name__ for c in sm.mamba_mixer_classes(p)] == ["NemotronHMamba2Mixer"]
    assert sm.model_device(p).type == "cpu"
    assert sm.model_devices(p) == {torch.device("cpu"), torch.device("cuda", 0)}
    refused(p)

    # (b) a real tiny NemotronH with every weight on cpu and ONE block carrying
    # the execution-device hook a device_map= load leaves behind
    got = sm.resolved_mamba_kernels(m)
    m.model.layers[1]._hf_hook = types.SimpleNamespace(execution_device=torch.device("cuda", 0))
    assert sm.model_device(m).type == "cpu"
    refused(m)
    # ... and through the loader, whose model.to("cpu") moves no weight off cuda
    _stub_loader(monkeypatch, m)
    with pytest.raises(RuntimeError, match="REQUIRED"):
        sm.load_model_and_tokenizer("fake/nemo", device="cpu", dtype="float32")
    del m.model.layers[1]._hf_hook

    # (c) an hf_device_map placing a block on cuda ordinal 0 (an int, as
    # Accelerate writes it) refuses; one naming only cpu and disk warns as before
    m.hf_device_map = {"model.embeddings": "cpu", "model.layers.0": "cpu",
                       "model.layers.1": 0, "lm_head": "disk"}
    refused(m)
    m.hf_device_map = {"model.embeddings": "cpu", "model.layers.1": "disk"}
    with pytest.warns(RuntimeWarning, match="is on cpu"):
        assert sm.assert_fused_mamba_kernels(m) == got
    del m.hf_device_map

    # (d) offload proper: the first parameter parked on meta, its module
    # executing on cuda — the first-parameter read says "meta", not cuda
    m.model.embeddings.weight = torch.nn.Parameter(
        torch.empty_like(m.model.embeddings.weight, device="meta"))
    m.model.embeddings._hf_hook = types.SimpleNamespace(execution_device=0)
    assert sm.model_device(m).type == "meta"
    refused(m)


def test_guard_refuses_a_cuda_hook_nested_by_repeated_accelerate_appends(monkeypatch):
    """Accelerate's REAL ``add_hook_to_module(..., append=True)`` applied twice
    to one block of a tiny NemotronH — every weight on cpu, NO ``hf_device_map``:
    each append wraps the module's existing hook and the new one in a fresh
    ``SequentialHook``, so after the second the
    ``AlignDevicesHook(execution_device=cuda:0)`` sits inside a
    ``SequentialHook`` inside the ``SequentialHook`` the module carries, and
    neither container has an ``execution_device`` of its own. A walk over the
    outer container's immediate members read this model as ``{cpu}`` and the
    guard warned instead of raising (review 2026-10-12, reproduced here before
    the fix); the nested walk reads cuda:0 and refuses. ``offload=True`` on a
    decoder layer, whose tensors all belong to its children, moves nothing, so
    the hook's init runs on this CPU-only torch."""
    from accelerate.hooks import (AlignDevicesHook, ModelHook, SequentialHook,
                                  add_hook_to_module)

    cuda0 = torch.device("cuda", 0)
    m, _ = _tiny_nemotron()
    assert not hasattr(m, "hf_device_map")
    layer = m.model.layers[1]
    add_hook_to_module(layer, ModelHook())
    add_hook_to_module(layer, AlignDevicesHook(execution_device=cuda0, offload=True), append=True)
    add_hook_to_module(layer, ModelHook(), append=True)
    hook = layer._hf_hook
    # the shape accelerate 1.14.0 builds: SequentialHook(SequentialHook(h1, h2), h3)
    assert isinstance(hook, SequentialHook) and isinstance(hook.hooks[0], SequentialHook)
    assert isinstance(hook.hooks[0].hooks[1], AlignDevicesHook)
    assert not hasattr(hook, "execution_device") and not hasattr(hook.hooks[0], "execution_device")
    # the immediate members alone carry no device: the reading that missed it
    assert all(getattr(h, "execution_device", None) is None for h in hook.hooks)
    assert sm.hook_execution_devices(hook) == [None, None, None, cuda0, None]
    assert sm.model_device(m).type == "cpu"
    assert sm.model_devices(m) == {torch.device("cpu"), cuda0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # and no warning either
        with pytest.raises(RuntimeError, match="REQUIRED") as ei:
            sm.assert_fused_mamba_kernels(m)
    assert "never proceed on the fallback" in str(ei.value)
    # a third append nests one level deeper still; any depth is read
    add_hook_to_module(layer, ModelHook(), append=True)
    assert isinstance(layer._hf_hook.hooks[0].hooks[0], SequentialHook)
    assert sm.model_devices(m) == {torch.device("cpu"), cuda0}
    # ... and the loader's cpu load refuses the same way
    _stub_loader(monkeypatch, m)
    with pytest.raises(RuntimeError, match="REQUIRED"):
        sm.load_model_and_tokenizer("fake/nemo", device="cpu", dtype="float32")


def test_loader_surfaces_the_warning_and_returns_the_model_on_cpu(monkeypatch):
    """A Mamba2 hybrid's load on a CPU: the loader WARNS about the fallback
    and hands the model back, instead of refusing."""
    m, _ = _tiny_nemotron()
    _stub_loader(monkeypatch, m)
    monkeypatch.setattr(sm, "pin_generation_stop_ids", lambda model, tok: [])
    with pytest.warns(RuntimeWarning, match="is on cpu") as rec:
        model, _tok = sm.load_model_and_tokenizer("fake/nemo", device="cpu", dtype="float32")
    assert model is m
    assert sum("pure-torch fallback" in str(w.message) for w in rec) == 1


def test_loader_still_refuses_the_fallback_on_cuda(monkeypatch):
    """The cluster's protection: a Mamba2 hybrid whose weights land on cuda
    without the kernels is refused at load, as before."""
    m, _ = _tiny_nemotron()
    _stub_loader(monkeypatch, m)
    monkeypatch.setattr(sm, "model_devices", lambda model: {torch.device("cuda", 0)})
    with pytest.raises(RuntimeError, match="REQUIRED"):
        sm.load_model_and_tokenizer("fake/nemo", device="cpu", dtype="float32")


# --------------------------------------------------------------------------- #
# objects follow the model's device; apply serves the model's dtype           #
# --------------------------------------------------------------------------- #


def _lora(model, seed: int, rank: int = 2) -> MultiLayerLoRAObject:
    obj = MultiLayerLoRAObject(model, layers=[0, 1], rank=rank)
    g = torch.Generator().manual_seed(seed)
    for l in obj.layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g) * 0.05
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.05
    return obj


def _tensors(obj) -> list[torch.Tensor]:
    if isinstance(obj, LoRASubtractionObject):
        return _tensors(obj.pos) + _tensors(obj.neg)
    if isinstance(obj, MultiLayerLoRAObject):
        return [p.data for g in (obj.A, obj.B) for p in g.values()]
    if isinstance(obj, MultiLayerVectorObject):
        return [p.data for p in obj.params.values()]
    raise TypeError(type(obj))


def _artifacts(model, tmp_path) -> dict:
    """One committed-demo kind each: lora (lora_all), vector_all (the
    refusal direction's kind) and lora_subtraction (difference_lora)."""
    out = {}
    lora = _lora(model, seed=1)
    lora.save(tmp_path / "lora.pt")
    out["lora_all"] = tmp_path / "lora.pt"
    vec = MultiLayerVectorObject(model, layers=[0, 1])    # the tiny Llama has 2 blocks
    g = torch.Generator().manual_seed(7)
    for l in vec.layers:
        vec.params[l].data = torch.randn(vec.params[l].shape, generator=g) * 0.1
    vec.save(tmp_path / "vec.pt")
    out["vector_all"] = tmp_path / "vec.pt"
    sub = LoRASubtractionObject(_lora(model, seed=1), _lora(model, seed=2))
    sub.save(tmp_path / "sub.pt")
    out["lora_subtraction"] = tmp_path / "sub.pt"
    return out


def test_load_object_moves_every_tensor_to_the_models_device(tiny_llama, tmp_path, monkeypatch):
    model, tok = tiny_llama
    want = sm.model_device(model)
    moved: list = []
    for cls in (MultiLayerLoRAObject, MultiLayerVectorObject, LoRASubtractionObject):
        real = cls.to_

        def spy(self, device, _real=real):
            moved.append((type(self).__name__, torch.device(device)))
            return _real(self, device)
        monkeypatch.setattr(cls, "to_", spy)
    for kind, path in _artifacts(model, tmp_path).items():
        obj = load_object(path, model=model, tok=tok)
        assert obj.kind == kind
        assert all(t.device == want for t in _tensors(obj)), kind
        # the stored float32 is kept: the hooks cast per call (CUDA parity)
        assert all(t.dtype is torch.float32 for t in _tensors(obj)), kind
    # the move went through to_ with the MODEL's device, once per top-level
    # object (the subtraction forwards to its two parents)
    assert {d for _, d in moved} == {want}
    assert [n for n, _ in moved] == ["MultiLayerLoRAObject", "MultiLayerVectorObject",
                                     "LoRASubtractionObject", "MultiLayerLoRAObject",
                                     "MultiLayerLoRAObject"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_serves_the_model_in_its_own_dtype(tiny_llama, tmp_path, dtype):
    """A 16-bit model (what mps and cuda run) is steered by fp32 objects: the
    activations come back in the MODEL's dtype, finite, and different from
    the unsteered pass — on CPU, so the dtype seam is exercised without a
    GPU. generate_with_object runs end to end under each object."""
    base, tok = tiny_llama
    model = copy.deepcopy(base).to(dtype) if dtype is not torch.float32 else base
    assert next(model.parameters()).dtype is dtype
    enc = tok(["one two three four"], return_tensors="pt")
    with torch.no_grad():
        plain = model(**enc).logits
    assert plain.dtype is dtype
    for kind, path in _artifacts(model, tmp_path).items():
        obj = load_object(path, model=model, tok=tok)
        with torch.no_grad(), obj.apply(model, scale=1.0):
            steered = model(**enc).logits
        assert steered.dtype is dtype, kind
        assert torch.isfinite(steered.float()).all(), kind
        assert not torch.equal(steered, plain), kind      # the object acted
        out = generate_with_object(model, tok, ["one two three"], obj=obj,
                                   scale=1.0, max_new_tokens=3)
        assert len(out) == 1 and isinstance(out[0], str), kind
    with torch.no_grad():
        again = model(**enc).logits
    assert torch.equal(again, plain)                      # every hook removed
