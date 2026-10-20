"""A hybrid Mamba/attention model needs max_num_seqs lowered, or vLLM refuses.

Qwen3.8-27B is hybrid. Its weights leave ~15 GiB of cache on an 80GB H100,
which is 310 Mamba cache blocks -- and EACH DECODE SEQUENCE NEEDS ONE. Against
vLLM's default of 1024 the engine will not start:

  ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (310).
  Each decode sequence requires one Mamba cache block, so CUDA graph capture
  cannot proceed.

A DENSE model of the same parameter count does not hit this, so it cannot be
predicted from size alone -- which is why the knob exists rather than a
heuristic.
"""
from sisteer.sampling.engine import VLLM_ENGINE_DEFAULTS, vllm_engine_kwargs


def test_absent_by_default_so_vllm_chooses():
    """Passing max_num_seqs=None must OMIT the key, not send None."""
    assert "max_num_seqs" not in vllm_engine_kwargs("m")
    assert VLLM_ENGINE_DEFAULTS["max_num_seqs"] is None


def test_explicit_value_reaches_the_engine():
    assert vllm_engine_kwargs("m", max_num_seqs=256)["max_num_seqs"] == 256


def test_it_is_a_known_setting_not_a_typo_passthrough():
    """vllm_engine_kwargs refuses unknown keys; max_num_seqs must be known."""
    import pytest
    with pytest.raises(TypeError):
        vllm_engine_kwargs("m", max_num_seq=256)      # typo, singular


def test_both_pool_halves_still_agree():
    """Adding a setting must not re-split the two halves of the pool."""
    from sisteer.sampling.generate_fn import VllmGenerateFn
    assert vllm_engine_kwargs("m") == VllmGenerateFn.for_cell("none", "m")._cfg
