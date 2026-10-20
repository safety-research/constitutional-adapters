"""vllm_patches: the non-neox M-RoPE routing (GLM-4V's interleaved layout hit
vLLM's neox-only triton_mrope kernel) and the shared engine hook. Dispatch
tests need vLLM importable (petri venv); the hook tests run anywhere."""
import types

import pytest
import torch

from sisteer.capeval.serving import vllm_patches as vp
from sisteer.sampling.engine import vllm_engine_kwargs


@pytest.fixture
def vllm_mrope():
    return pytest.importorskip(
        "vllm.model_executor.layers.rotary_embedding.mrope",
        reason="dispatch tests need the pinned vLLM (petri venv)")


def _rot(mod, is_neox_style: bool):
    # a CustomOp binds its dispatch at __init__ from the CURRENT vLLM config,
    # and refuses to be built outside one — exactly the moment the patch has
    # to precede (see vllm_patches)
    from vllm.config import VllmConfig, set_current_vllm_config
    with set_current_vllm_config(VllmConfig()):
        return mod.MRotaryEmbedding(
            head_size=128, rotary_dim=64, max_position_embeddings=512,
            base=500000, is_neox_style=is_neox_style, dtype=torch.float32,
            mrope_section=[8, 12, 12])


def test_patch_is_idempotent_and_marks_the_class(vllm_mrope):
    # importing the module already applied it (the worker-extension route)
    assert vp.MROPE_PATCH_APPLIED is True
    assert vp.patch_mrope_interleaved_layout() is True
    first = vllm_mrope.MRotaryEmbedding.forward_cuda
    assert vp.patch_mrope_interleaved_layout() is True
    assert vllm_mrope.MRotaryEmbedding.forward_cuda is first
    assert hasattr(first, "_sisteer_orig")


@pytest.mark.parametrize("neox,ndim,expect", [
    (False, 2, "native"),   # the bug: non-neox + M-RoPE positions
    (True, 2, "kernel"),    # Qwen-VL family: untouched
    (False, 1, "kernel"),   # text-only 1-D positions never hit triton_mrope
])
def test_dispatch_routes_only_the_buggy_case(vllm_mrope, monkeypatch, neox,
                                             ndim, expect):
    vp.patch_mrope_interleaved_layout()
    rot = _rot(vllm_mrope, neox)
    calls = []
    monkeypatch.setattr(rot, "forward_native",
                        lambda *a, **k: calls.append("native") or ("q", "k"))
    patched = vllm_mrope.MRotaryEmbedding.forward_cuda
    cells = [c for c in patched.__closure__ if callable(c.cell_contents)]
    assert cells, "wrapper must close over the original forward_cuda"
    monkeypatch.setattr(cells[0], "cell_contents",
                        lambda self, *a, **k: calls.append("kernel") or ("q", "k"))
    pos = torch.arange(6) if ndim == 1 else torch.arange(6)[None].expand(3, -1)
    rot.forward_cuda(pos, torch.zeros(6, 128), torch.zeros(6, 128))
    assert calls == [expect]
    # the routed branch leaves evidence a post-forward report can show
    assert getattr(rot, "_sisteer_routed_native", 0) == (1 if expect == "native" else 0)


def test_text_only_mrope_collapses_to_plain_interleaved_rope(vllm_mrope):
    """For text tokens all three M-RoPE axes carry the same position, so the
    corrected path must equal plain non-neox RoPE with 1-D positions."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    rot = _rot(vllm_mrope, False)
    with set_current_vllm_config(VllmConfig()):
        plain = RotaryEmbedding(head_size=128, rotary_dim=64,
                                max_position_embeddings=512, base=500000,
                                is_neox_style=False, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(24, 32 * 128, generator=g)
    k = torch.randn(24, 2 * 128, generator=g)
    pos1 = torch.arange(24)
    q3, k3 = rot.forward_native(pos1[None].expand(3, -1), q.clone(), k.clone())
    q1, k1 = plain.forward_native(pos1, q.clone(), k.clone())
    torch.testing.assert_close(q3, q1, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k3, k1, atol=1e-5, rtol=1e-5)


def _model_with(rot):
    lm = types.SimpleNamespace(layers=[types.SimpleNamespace(
        self_attn=types.SimpleNamespace(rotary_emb=rot))])
    return types.SimpleNamespace(language_model=types.SimpleNamespace(model=lm))


def test_report_names_the_bound_dispatch(vllm_mrope):
    rot = _rot(vllm_mrope, False)
    rep = vp.mrope_patch_report(_model_with(rot))
    assert rep["rotary_cls"] == "MRotaryEmbedding" and rep["is_neox_style"] is False
    assert rep["mrope_section"] == [8, 12, 12] and rep["rotary_dim"] == 64
    # whatever CustomOp bound here (CPU test process: never forward_cuda), the
    # report says so and needs_patch follows reachability, not the model alone
    assert rep["dispatch"] == rot._forward_method.__name__
    assert rep["kernel_reachable"] == (rep["dispatch"] == "forward_cuda")
    assert rep["needs_patch"] == rep["kernel_reachable"]
    assert rep["routed_native_calls"] == 0


class _FakeRot:
    """A non-neox M-RoPE rotary with a chosen bound dispatch (no vLLM needed)."""
    is_neox_style = False
    mrope_section = [8, 12, 12]
    rotary_dim = 64

    def __init__(self, dispatch: str, routed: int = 0):
        def forward_cuda(): pass
        def forward_native(): pass
        self._forward_method = {"forward_cuda": forward_cuda,
                                "forward_native": forward_native}[dispatch]
        self._sisteer_routed_native = routed


_FakeRot.__name__ = "MRotaryEmbedding"


def test_report_needs_patch_only_when_the_kernel_is_reachable():
    eager = vp.mrope_patch_report(_model_with(_FakeRot("forward_cuda", routed=3)))
    assert eager["dispatch"] == "forward_cuda" and eager["kernel_reachable"] is True
    assert eager["needs_patch"] is True and eager["routed_native_calls"] == 3
    compiled = vp.mrope_patch_report(_model_with(_FakeRot("forward_native")))
    assert compiled["dispatch"] == "forward_native"
    assert compiled["kernel_reachable"] is False and compiled["needs_patch"] is False


def test_attest_passes_a_compiled_engine_where_the_kernel_is_unreachable():
    rep = {"needs_patch": False, "patch_applied": False, "rotary_cls": "MRotaryEmbedding",
           "is_neox_style": False, "dispatch": "forward_native", "kernel_reachable": False}
    assert vp.attest_engine(_FakeLLM(rep)) == rep


# ---- the shared hook: no vLLM needed ---------------------------------------

def test_engine_kwargs_carry_the_patch_extension():
    kw = vllm_engine_kwargs("some/model", revision="abc")
    assert kw["worker_extension_cls"] == vp.PATCH_EXTENSION_CLS
    assert kw["model"] == "some/model" and kw["revision"] == "abc"


def test_engine_kwargs_keep_a_callers_extension():
    kw = vp.engine_kwargs_with_patch({"model": "m", "worker_extension_cls": "x.Y"})
    assert kw["worker_extension_cls"] == "x.Y"


class _FakeLLM:
    def __init__(self, rep):
        self.rep = rep

    def collective_rpc(self, name):
        assert name == "sisteer_mrope_report"
        return [self.rep]


def test_attest_raises_when_a_non_neox_mrope_model_is_unpatched():
    with pytest.raises(vp.UnpatchedEngineError):
        vp.attest_engine(_FakeLLM({"needs_patch": True, "patch_applied": False,
                                   "rotary_cls": "MRotaryEmbedding"}))


@pytest.mark.parametrize("rep", [
    {"needs_patch": True, "patch_applied": True},      # GLM-4V, patched
    {"needs_patch": False, "patch_applied": True},     # neox M-RoPE model
    {"needs_patch": False, "patch_applied": False, "rotary_cls": "RotaryEmbedding"},
])
def test_attest_passes_otherwise_and_returns_the_report(rep):
    assert vp.attest_engine(_FakeLLM(rep)) == rep


def test_mrope_report_reads_the_first_attention_layer_of_a_hybrid_model():
    """Qwen3.5 interleaves gated-deltanet layers (``linear_attn``, no rotary) with attention layers; layer 0 of
    the 4B is recurrent, so a layer-0 read errored and the standup refused a correct rotary (canary 84787,
    2026-09-09). The report must read the FIRST layer that carries ``self_attn`` and say which one."""
    from types import SimpleNamespace as NS
    from sisteer.capeval.serving import vllm_patches as vp

    class Rot:
        is_neox_style = True
        mrope_section = [24, 20, 20]
        rotary_dim = 128
        def _forward_method(self):  # bound method named like vLLM's CustomOp dispatch
            pass
    Rot._forward_method.__name__ = "forward_cuda"
    recurrent = NS(linear_attn=object())            # no self_attn attribute at all
    attn = NS(self_attn=NS(rotary_emb=Rot()))
    model = NS(model=NS(layers=[recurrent, recurrent, attn, recurrent]))
    rep = vp.mrope_patch_report(model)
    assert "error" not in rep, rep
    assert rep["rotary_layer_index"] == 2 and rep["n_layers"] == 4 and rep["n_layers_with_rotary"] == 1 and rep["rotary_attr"] == "self_attn"
    assert rep["rotary_cls"] == "Rot" and rep["is_neox_style"] is True and rep["needs_patch"] is False
    # Nemotron-H: attention behind `mixer`, and NO rotary anywhere — a report saying so (needs_patch False),
    # never an error: "no rotary" is exactly the proof the attestation wants (jobs 84834/84835, 2026-09-09)
    mamba = NS(mixer=NS(conv1d=object()))
    attn_no_rope = NS(mixer=NS(attn=object()))
    rep2 = vp.mrope_patch_report(NS(model=NS(layers=[mamba, attn_no_rope, mamba])))
    assert "error" not in rep2, rep2
    assert rep2["has_rotary"] is False and rep2["needs_patch"] is False and rep2["n_layers_with_rotary"] == 0
    # an attention module behind `mixer` WITH a rotary is read like self_attn
    rep3 = vp.mrope_patch_report(NS(model=NS(layers=[mamba, NS(mixer=NS(rotary_emb=Rot()))])))
    assert rep3["has_rotary"] is True and rep3["rotary_layer_index"] == 1 and rep3["rotary_attr"] == "mixer"
    # an EMPTY layer list is still an error (never a silent pass)
    rep4 = vp.mrope_patch_report(NS(model=NS(layers=[])))
    assert "error" in rep4
