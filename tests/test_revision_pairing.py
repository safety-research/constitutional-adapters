"""A revision pin belongs to the model it pins.

--model and --revision defaulted as an unlinked pair (Llama's id and Llama's
sha). Overriding --model alone therefore kept Llama's sha, and the run died
with "Revision Not Found for url: .../google/gemma-4-E4B-it/tree/d10aef79..."
-- after all eight engines had already loaded. Cost: a full pod-minutes-long
startup on 8 idle H100s before anything failed.
"""
import importlib.util as ilu
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ps():
    if "_ps_rev" in sys.modules:
        return sys.modules["_ps_rev"]
    spec = ilu.spec_from_file_location("_ps_rev", ROOT / "scripts" / "pool_sample.py")
    m = ilu.module_from_spec(spec); sys.modules["_ps_rev"] = m
    spec.loader.exec_module(m)
    return m


def test_pinned_model_keeps_its_pin():
    m = _ps()
    assert m.resolve_revision(m.DEFAULT_MODEL, None) == m.DEFAULT_REVISION


def test_other_model_does_NOT_inherit_the_pin():
    """The whole bug: gemma must not be asked for Llama's sha."""
    m = _ps()
    assert m.resolve_revision("google/gemma-4-E4B-it", None) is None
    assert m.resolve_revision("Qwen/Qwen3.8-27B", None) is None


def test_explicit_revision_is_honoured():
    m = _ps()
    assert m.resolve_revision("google/gemma-4-E4B-it", "abc123") == "abc123"


def test_pinning_one_models_sha_onto_another_is_refused():
    """Passing it explicitly is the same mistake made deliberately -- refuse
    with the reason rather than reproducing the Revision Not Found death."""
    m = _ps()
    with pytest.raises(SystemExit, match="belongs to the model it pins"):
        m.resolve_revision("Qwen/Qwen3.8-27B", m.DEFAULT_REVISION)
