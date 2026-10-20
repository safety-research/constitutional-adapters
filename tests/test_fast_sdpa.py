"""The cuDNN-SDPA footgun is prevented in CODE, not just documented.

cuDNN's SDPA backend re-plans on CPU for every new (batch, seqlen) shape,
starving the GPU on ragged training batches and on autoregressive decode
(measured twice on this project: 33-45% and ~20% GPU duty). It was
documented in the gpu-util skill and STILL recurred, because a training-only
fix didn't reach the eval generation path. These tests lock the enforcement
so it cannot silently regress or be bypassed by a new model-load site.
"""

import ast
import os
from pathlib import Path

import torch

from sisteer.models import configure_fast_sdpa

REPO = Path(__file__).resolve().parent.parent


def test_configure_fast_sdpa_disables_cudnn():
    """The invariant: after the shared helper, cuDNN SDPA is OFF. The flag is
    a global preference (settable without CUDA), so this runs on a CPU box."""
    torch.backends.cuda.enable_cudnn_sdp(True)  # simulate the default-on state
    configure_fast_sdpa()
    assert torch.backends.cuda.cudnn_sdp_enabled() is False


def test_env_override_keeps_cudnn(monkeypatch):
    """SISTEER_KEEP_CUDNN_SDP=1 reproduces a cuDNN-baseline run."""
    monkeypatch.setenv("SISTEER_KEEP_CUDNN_SDP", "1")
    torch.backends.cuda.enable_cudnn_sdp(True)
    configure_fast_sdpa()
    assert torch.backends.cuda.cudnn_sdp_enabled() is True
    torch.backends.cuda.enable_cudnn_sdp(False)  # restore for other tests


def test_loader_calls_configure_fast_sdpa():
    """load_model_and_tokenizer must call the helper (no GPU/model needed —
    assert on the source so the wiring can't silently drop)."""
    src = (REPO / "src/sisteer/models.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "load_model_and_tokenizer")
    calls = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "configure_fast_sdpa" in calls


def test_every_model_load_site_is_guarded():
    """No new bypass: every direct AutoModelForCausalLM.from_pretrained in the
    package/scripts must sit in a file that also calls configure_fast_sdpa (or
    inlines enable_cudnn_sdp for the sub-env worker). This is the check that
    turns 'remember the footgun' into 'the suite fails if you forget'."""
    roots = [REPO / "src/sisteer", REPO / "scripts"]
    offenders = []
    for root in roots:
        for p in root.rglob("*.py"):
            text = p.read_text()
            if "AutoModelForCausalLM.from_pretrained" not in text:
                continue
            guarded = ("configure_fast_sdpa" in text
                       or "enable_cudnn_sdp" in text)
            if not guarded:
                offenders.append(str(p.relative_to(REPO)))
    assert not offenders, (
        "model-load sites with no cuDNN-SDPA guard (call configure_fast_sdpa "
        f"before from_pretrained): {offenders}")
