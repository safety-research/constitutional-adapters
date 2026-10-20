"""petri_lens_probe.probe_geometry names the DECODER, on every config shape.

The probe sizes itself (probe layer = n_layers // 2, the synthetic steer
vector's width) from the pinned model config. Composite VLM configs —
``Glm4vConfig`` for GLM-4.6V-Flash under transformers 5.14.1 — carry
``num_hidden_layers`` / ``hidden_size`` ONLY on ``text_config``; the old
top-level ``getattr`` raised AttributeError there (measured 2026-09-02),
so the probe could not run on the model at all. No downloads: the real
class is instantiated from its in-code defaults.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "_petri_lens_probe_geometry", ROOT / "scripts" / "petri_lens_probe.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


class _Text:
    num_hidden_layers = 40
    hidden_size = 4096


class _Composite:
    """Glm4vConfig-shaped: nothing decoder-related at the top level."""
    text_config = _Text()

    def get_text_config(self):
        return self.text_config


class _Plain:
    """A causal-LM config: get_text_config() returns itself."""
    num_hidden_layers = 32
    hidden_size = 4096

    def get_text_config(self):
        return self


class _Legacy:
    """A config predating get_text_config."""
    num_hidden_layers = 12
    hidden_size = 768


def test_composite_config_reads_the_text_config():
    assert probe.probe_geometry(_Composite()) == (40, 4096)


def test_plain_config_returns_itself():
    assert probe.probe_geometry(_Plain()) == (32, 4096)


def test_legacy_config_without_get_text_config():
    assert probe.probe_geometry(_Legacy()) == (12, 768)


def test_top_level_read_fails_on_the_composite_shape():
    # the failure mode the helper exists to avoid, pinned so a "tidy"
    # revert back to getattr(hf_cfg, ...) breaks here first
    with pytest.raises(AttributeError):
        _ = _Composite().hidden_size


def test_real_glm4v_config_class():
    transformers = pytest.importorskip("transformers")
    Glm4vConfig = getattr(transformers, "Glm4vConfig", None)
    if Glm4vConfig is None:
        pytest.skip("Glm4vConfig not in this transformers")
    cfg = Glm4vConfig()   # in-code defaults, no download
    text = cfg.get_text_config()
    assert probe.probe_geometry(cfg) == (int(text.num_hidden_layers),
                                         int(text.hidden_size))
    assert not hasattr(cfg, "hidden_size") and not hasattr(
        cfg, "num_hidden_layers"), (
        "Glm4vConfig now exposes decoder dims at the top level — the "
        "helper is still right, but this pin should be re-read")
