"""The default-'all' phase-reset hazard, pinned. A response-fit diffmean
artifact hydrates MASKED (DiffMeanObject.load_blob re-arms it — diffmean is
a kind-loaded torch blob, NOT on the SteeringVector safetensors path that
skips the fit leg). But _set_response_positions ALWAYS applies its argument,
'all' included, so a build that passes the default 'all' silently RESETS the
masked artifact to all-position — a C cell serving like an A cell while still
labelled response-phase. The sweep is safe only because it passes 'response'
explicitly for every mask='generated' cell; this test fails if that guarantee
ever erodes (a default flip, a lost explicit arg). Tests the HAZARD's presence
(the reset), not just the happy path."""

from __future__ import annotations

import torch

from sisteer.eval.jailbreak import _set_response_positions
from sisteer.extraction.diffmean import DiffMeanObject
from sisteer.methods import load_object


def _save_response_fit_diffmean(model, tmp_path, layer=1, H=4):
    d = model.config.hidden_size
    obj = DiffMeanObject(model, {layer: torch.randn(d)})
    obj.set_fit_positions("response", H)          # a response-fit artifact
    path = tmp_path / "object.pt"
    obj.save(path)
    return path


def test_diffmean_artifact_hydrates_masked(tiny_llama, tmp_path):
    # STAGE 1 — what the ARTIFACT carries: load_object re-arms the fit leg.
    model, tok = tiny_llama
    path = _save_response_fit_diffmean(model, tmp_path)
    obj = load_object(path, model=model, tok=tok)
    assert obj.fit_positions == "response"        # hydrates MASKED
    assert obj.fit_response_header_len == 4


def test_default_all_build_RESETS_a_masked_artifact(tiny_llama, tmp_path):
    # STAGE 2 — the HAZARD: a default 'all' serve resets the masked artifact.
    model, tok = tiny_llama
    obj = load_object(_save_response_fit_diffmean(model, tmp_path),
                      model=model, tok=tok)
    assert obj.fit_positions == "response"        # masked as loaded
    _set_response_positions(obj, "all", tok)      # what a default build does
    assert obj.fit_positions == "all"             # SILENTLY UNMASKED -> the hazard
    assert obj.fit_response_header_len is None


def test_explicit_response_build_preserves_the_mask(tiny_llama, tmp_path):
    # the correct call the sweep makes for every mask='generated' cell.
    model, tok = tiny_llama
    from sisteer.eval.jailbreak import response_header_len
    if response_header_len(tok) <= 0:
        import pytest
        pytest.skip("tiny_llama chat template exposes no assistant header; "
                    "the response-mask serve is covered on the real tokenizer "
                    "by the live run + smoke")
    obj = load_object(_save_response_fit_diffmean(model, tmp_path),
                      model=model, tok=tok)
    _set_response_positions(obj, "response", tok)
    assert obj.fit_positions == "response"        # explicit 'response' keeps it masked
