"""Both halves of the pool must come from ONE engine configuration.

VllmGenerateFn exists because the pool's pos side ran on vLLM while its
none/neg side ran on HF, and the measured acceptance gap between engines
(+25.8pp) made that split a confound. The class's docstring asserted its
settings mirrored VllmSampler's -- an invariant kept by hand in two places,
which is to say not kept: on 2026-08-26 a fix moved the neg side to
gpu_memory_utilization 0.85 while the sampler stayed at 0.90, and the two
halves silently stopped sharing a configuration. This pins it structurally.
"""
import inspect

import pytest

from sisteer.sampling.engine import (VLLM_ENGINE_DEFAULTS, VllmSampler,
                                     vllm_engine_kwargs)
from sisteer.sampling.generate_fn import VllmGenerateFn

SETTINGS = ("dtype", "max_model_len", "gpu_memory_utilization",
            "enforce_eager", "enable_prefix_caching")


def test_both_halves_build_identical_engine_kwargs():
    assert (vllm_engine_kwargs("m", seed=3)
            == VllmGenerateFn.for_cell("none", "m", seed=3)._cfg)


@pytest.mark.parametrize("setting", SETTINGS)
def test_every_setting_has_exactly_one_definition(setting):
    """No engine setting may be restated as a signature default: that is the
    exact mechanism by which the two sides drifted apart."""
    for fn in (VllmSampler.__init__, VllmGenerateFn.__init__):
        params = inspect.signature(fn).parameters
        if setting in params:
            assert params[setting].default is None, (
                f"{fn.__qualname__} restates {setting}="
                f"{params[setting].default!r}; it belongs in "
                f"VLLM_ENGINE_DEFAULTS only")
    assert setting in VLLM_ENGINE_DEFAULTS


def test_explicit_value_still_overrides():
    kw = vllm_engine_kwargs("m", gpu_memory_utilization=0.90)
    assert kw["gpu_memory_utilization"] == 0.90


def test_unknown_setting_is_refused_not_ignored():
    """A typo'd kwarg must not silently fall through to the defaults."""
    with pytest.raises(TypeError, match="unknown vLLM engine setting"):
        vllm_engine_kwargs("m", gpu_mem_util=0.9)


def test_unhooked_only_settings_are_still_the_unhooked_ones():
    """Guardrail: a steered path needs the OPPOSITE of both, because
    CUDA-graph replay bypasses Python forward hooks. If someone flips these
    for a compiled cell, the neg side must not follow silently."""
    assert VLLM_ENGINE_DEFAULTS["enable_prefix_caching"] is True
    assert VLLM_ENGINE_DEFAULTS["enforce_eager"] is False
    with pytest.raises(Exception):
        VllmGenerateFn.for_cell("compiled", "m")
