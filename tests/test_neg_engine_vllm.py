"""The none/prompted side's engine: declared, and refused where unsafe."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import two_si_run as t  # noqa: E402

from sisteer.sampling.generate_fn import (VllmGenerateFn,  # noqa: E402
                                          VllmGenerateUnsupported)


def test_config_declares_both_engines_as_vllm():
    """Both halves of a pool must name the same engine. The +25.8pp HF/vLLM
    acceptance gap is larger than most effects measured with these pools, so
    a split pool carries it as a confound."""
    pool = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())["pool"]
    assert pool["pos_generation_engine"] == "vllm"
    assert pool["neg_generation_engine"] == "vllm"


def test_neg_engine_defaults_to_hf_when_unset():
    """Every pool before 2026-08-26 was HF on this side; an absent key must
    keep meaning that rather than silently switching engines."""
    assert t._neg_engine({"pool": {}}) == "hf"
    assert t._neg_engine({}) == "hf"
    assert t._neg_engine({"pool": {"neg_generation_engine": "vllm"}}) == "vllm"


def test_a_compiled_cell_is_refused_not_silently_unhooked():
    """The failure this prevents: vLLM cannot run the object's forward
    hooks, so a compiled cell would emit UNDEFENDED text and store it as a
    defended generation -- an undefended baseline wearing a defense's name,
    which looks like a working defense that does nothing."""
    for mode in ("lora_r1", "psr_kl_all", "vector", "diffmean"):
        with pytest.raises(VllmGenerateUnsupported, match="forward hooks"):
            VllmGenerateFn.for_cell(mode, "some/model")


def test_the_two_safe_modes_are_exactly_none_and_prompted():
    """Pinned as a closed set: a new mode must be considered explicitly
    rather than inheriting vLLM by being spelled differently."""
    import inspect
    src = inspect.getsource(VllmGenerateFn.for_cell)
    assert 'cell_mode not in ("none", "prompted")' in src


def test_compiled_cells_fall_back_to_hf_visibly_not_by_crashing(capsys):
    """The config flag means "vLLM where it is safe". A compiled cell must
    take the HF path and SAY so -- crashing would make the flag unusable for
    eval sweeps, and falling back silently would recreate the unstated
    per-cell engine split this work exists to remove."""
    from types import SimpleNamespace

    cfg = {"model": "m", "pool": {"neg_generation_engine": "vllm"}}
    assert t._vllm_generate_fn(cfg, SimpleNamespace(arm="lora_r1")) is None
    out = capsys.readouterr().out
    assert "engine=hf" in out and "lora_r1" in out and "forward hooks" in out


def test_the_engine_is_announced_for_the_safe_cells_too():
    """Both branches print. 'Which engine produced this row' was
    unanswerable from the logs for the entire life of the HF/vLLM problem."""
    import inspect
    src = inspect.getsource(t._vllm_generate_fn)
    assert src.count("[generate] neg-side engine=") == 2


def test_the_cuda_graph_hook_guardrail_is_recorded():
    """enforce_eager=False + prefix caching are safe ONLY because these
    cells carry no object. CUDA-graph replay bypasses Python forward hooks,
    so widening for_cell without flipping them would run and emit unsteered
    text rather than raising. The reason must survive in the file."""
    from pathlib import Path
    import sisteer.sampling.generate_fn as g

    src = Path(g.__file__).read_text()
    assert "CUDA-graph replay bypasses" in src
    assert "steered-vLLM backend" in src


# --- The config value must REACH the engine, not merely be declared --------
# Job 212618: cfg["pool"]["neg_generation_engine"] resolved correctly and
# turned vLLM on, while cfg["pool"]["vllm_gpu_memory_utilization"] was read
# from the TOP level by the adjacent function, missed, and fell back to 0.90.
# vLLM then refused to start. The declared value was present and correct the
# whole time. Asserting the config FILE says 0.85 would have stayed green.

import importlib.util as _ilu
import pathlib as _pl

import yaml as _yaml

_SPEC = _ilu.spec_from_file_location(
    "_two_si_run_cfg", _pl.Path(__file__).resolve().parents[1]
    / "scripts" / "two_si_run.py")


def _run_mod():
    mod = _ilu.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def _cfg():
    return _yaml.safe_load(
        (_pl.Path(__file__).resolve().parents[1]
         / "configs" / "two_si.yaml").read_text())


def test_declared_settings_reach_the_engine_kwargs():
    from sisteer.sampling.generate_fn import VllmGenerateFn
    mod, cfg = _run_mod(), _cfg()
    declared = (cfg.get("pool") or {}).get("vllm_gpu_memory_utilization")
    assert declared is not None, "config no longer declares the setting"
    built = VllmGenerateFn.for_cell(
        "none", "m", **mod.neg_vllm_settings(cfg))._cfg
    assert built["gpu_memory_utilization"] == declared


def test_settings_come_from_the_pool_block_not_the_top_level():
    mod = _run_mod()
    nested = {"pool": {"vllm_gpu_memory_utilization": 0.42}}
    assert mod.neg_vllm_settings(nested)["gpu_memory_utilization"] == 0.42
    # A key at the top level is NOT the declaration site and must not be read;
    # silently honouring both scopes is how the two functions disagreed.
    assert mod.neg_vllm_settings({"vllm_gpu_memory_utilization": 0.42}) == {}


def test_engine_switch_and_its_settings_share_a_scope():
    """Whatever scope turns the engine ON must be the scope its settings are
    read from -- that mismatch is the entire bug."""
    mod = _run_mod()
    cfg = {"pool": {"neg_generation_engine": "vllm",
                    "vllm_gpu_memory_utilization": 0.5}}
    assert mod._neg_engine(cfg) == "vllm"
    assert mod.neg_vllm_settings(cfg)["gpu_memory_utilization"] == 0.5


def test_unset_keys_fall_through_to_the_shared_defaults():
    from sisteer.sampling.engine import VLLM_ENGINE_DEFAULTS
    from sisteer.sampling.generate_fn import VllmGenerateFn
    mod = _run_mod()
    assert mod.neg_vllm_settings({"pool": {}}) == {}
    built = VllmGenerateFn.for_cell("none", "m")._cfg
    assert (built["gpu_memory_utilization"]
            == VLLM_ENGINE_DEFAULTS["gpu_memory_utilization"])
