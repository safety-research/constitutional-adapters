"""The rotary attestation must survive a HYBRID decoder stack (2026-09-11).

`mrope_patch_report` used to read `lm.layers[0].self_attn.rotary_emb`. Layer 0 is not an attention
layer in the hybrid families -- Qwen3.5's is a gated-delta-net, NemotronH's is a Mamba mixer -- so the
probe raised AttributeError, and `mt_serving_up.mrope_attestation_problems` turns an errored report
into a refusal for EVERY model: two qwen35_4b standups died on it with a rotary never in question
(jobs 482100 / 482101). main's fix reads the FIRST attention layer that carries a `rotary_emb` under
any of the family-specific attribute names (`self_attn` / `mixer` / `attn`) and reports which one it
read (`has_rotary`, `rotary_layer_index`, `rotary_attr`, `n_layers_with_rotary`). These tests pin
both that and the gate it feeds: GLM (the one non-neox M-RoPE key) must still be refused by every
shape check, and a fully recurrent stack is a truthful report, not an error. CPU only, no vLLM.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from sisteer.capeval.serving.vllm_patches import mrope_patch_report

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_mt_up", REPO / "scripts/sdfbase/mt_serving_up.py")


def _mt_up():
    m = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(m)
    return m


def _Rotary(cls_name="RotaryEmbedding", neox=True, dispatch="forward_cuda"):
    """A rotary stub whose TYPE NAME is what the report reads (type(rot).__name__)."""
    r = type(cls_name, (), {})()
    r.is_neox_style = neox
    r.rotary_dim = 128
    r.mrope_section = []
    r._sisteer_routed_native = 0
    r._forward_method = type("f", (), {"__name__": dispatch})()
    return r


class _Attn:
    def __init__(self, rot):
        self.rotary_emb = rot


class _Recurrent:
    """a gated-delta-net / Mamba block: no rotary anywhere on it"""
    def __init__(self):
        self.conv1d = object()
        self.in_proj = object()


class _Layer:
    def __init__(self, **children):
        for k, v in children.items():
            setattr(self, k, v)


class _LM:
    def __init__(self, layers):
        self.layers = layers


def test_pure_attention_stack_reports_layer_zero_exactly_as_before():
    lm = _LM([_Layer(self_attn=_Attn(_Rotary())), _Layer(self_attn=_Attn(_Rotary()))])
    rep = mrope_patch_report(lm)
    assert "error" not in rep and rep["has_rotary"] is True
    assert (rep["rotary_layer_index"], rep["rotary_attr"]) == (0, "self_attn")
    assert (rep["n_layers"], rep["n_layers_with_rotary"]) == (2, 2)
    assert rep["rotary_cls"] == "RotaryEmbedding" and rep["kernel_reachable"] is True


def test_hybrid_stack_skips_the_recurrent_layers_instead_of_erroring():
    """Qwen3.5: layers 0-2 gated-delta-net (`linear_attn`, no rotary), layer 3 full attention."""
    lm = _LM([_Layer(linear_attn=_Recurrent()), _Layer(linear_attn=_Recurrent()),
              _Layer(linear_attn=_Recurrent()), _Layer(self_attn=_Attn(_Rotary()))])
    rep = mrope_patch_report(lm)
    assert "error" not in rep, rep          # the exact failure that killed jobs 482100 / 482101
    assert (rep["rotary_layer_index"], rep["rotary_attr"]) == (3, "self_attn")
    assert rep["n_layers_with_rotary"] == 1 and rep["rotary_cls"] == "RotaryEmbedding"
    assert _mt_up().mrope_attestation_problems([rep], "qwen35_4b") == []


def test_an_attention_module_named_mixer_is_found_too():
    """NemotronH-style: the attention layers sit behind the same `mixer` attribute as the Mamba ones."""
    lm = _LM([_Layer(mixer=_Recurrent()), _Layer(mixer=_Attn(_Rotary()))])
    rep = mrope_patch_report(lm)
    assert "error" not in rep and (rep["rotary_layer_index"], rep["rotary_attr"]) == (1, "mixer")


def test_a_fully_recurrent_stack_is_a_truthful_report_not_an_error_and_still_refuses_glm():
    lm = _LM([_Layer(mixer=_Recurrent()), _Layer(mixer=_Recurrent())])
    rep = mrope_patch_report(lm)
    assert "error" not in rep and rep["has_rotary"] is False
    assert rep["rotary_cls"] is None and rep["needs_patch"] is False
    assert (rep["n_layers"], rep["n_layers_with_rotary"]) == (2, 0)
    mt = _mt_up()
    assert mt.mrope_attestation_problems([rep], "nemotron3nano4b") == []
    # THE GATE IS NOT WEAKENED: the one non-neox M-RoPE key still fails every shape check
    probs = mt.mrope_attestation_problems([rep], "glm46vflash")
    assert probs and any("not MRotaryEmbedding" in p for p in probs)


def test_an_errored_report_is_still_a_refusal_for_every_model():
    class _Explodes:
        @property
        def layers(self):
            raise RuntimeError("boom")
    rep = mrope_patch_report(_Explodes())
    assert rep.get("error")
    mt = _mt_up()
    for key in ("qwen35_4b", "glm46vflash"):
        assert any("errored" in p for p in mt.mrope_attestation_problems([rep], key))
