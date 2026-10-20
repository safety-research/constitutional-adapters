"""Circuit Breakers implementation (arXiv 2406.04313): the layer-resolution
helper, the RR training loop (retain-L2 + reroute-ReLU(cos) under the
scheduled coeffs), the frozen-base LoRA-hook object protocol (trainable
params, scale-0 bit-exact no-op, save/load round trip), the explicit-kwargs
vs recorded pairs-fallback data paths, and the registry-removal wiring.
circuit_breaker left methods.ARMS ENTIRELY on 2026-08-20:
nothing trains or enumerates it, but the implementation and — load-bearing
for a peer lane's serving/kind-coverage canary — the load_object hydration
of already-fitted artifacts survive. CPU-only, tiny-llama fixture.

All fixture text is SYNTHETIC placeholder — the reroute "harmful
continuation" is an invented benign string; the RR loss is content-agnostic
for a smoke, and no real harmful text belongs in the repo.
"""

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair
from sisteer.extraction.circuit_breaker import (
    PROJECTIONS,
    CircuitBreakerObject,
    resolve_layers,
    train_circuit_breaker,
)
from sisteer.methods import ARMS, _KIND_LOADERS, load_object
from sisteer.models import num_layers


def _pairs():
    return [
        Pair(prompt="a benign request", pos="a refusal or safe reply",
             neg="an unsafe continuation placeholder", instruction_id="safety"),
        Pair(prompt="another benign request", pos="another safe reply",
             neg="another unsafe continuation placeholder",
             instruction_id="safety"),
    ]


def _retain():
    return ["RETAIN PROMPT one\nRETAIN CONTINUATION one",
            "RETAIN PROMPT two\nRETAIN CONTINUATION two",
            "RETAIN PROMPT three\nRETAIN CONTINUATION three",
            "RETAIN PROMPT four\nRETAIN CONTINUATION four"]


def _reroute():
    # Synthetic stand-ins for "harmful prompt + harmful CONTINUATION".
    return ["REROUTE PROMPT one\nREROUTE CONTINUATION one",
            "REROUTE PROMPT two\nREROUTE CONTINUATION two",
            "REROUTE PROMPT three\nREROUTE CONTINUATION three",
            "REROUTE PROMPT four\nREROUTE CONTINUATION four"]


# ---------------------------------------------------------------- layer resolve


def test_resolve_layers_clamps_to_depth(tiny_llama):
    model, _ = tiny_llama
    nl = num_layers(model)
    tgt, transform = resolve_layers(model, (10, 20))  # far beyond a tiny model
    assert tgt and all(0 <= l < nl for l in tgt)
    assert transform == list(range(0, max(tgt) + 1))


# ---------------------------------------------------------------- object protocol


def _constructed_object(model, scale_up=0.05):
    """A CircuitBreakerObject with a moderate (non-tiny) B so apply(scale=1)
    unambiguously moves the logits and scale-0 unambiguously does not."""
    nl = num_layers(model)
    tgt, transform = resolve_layers(model, (nl - 1,))
    obj = CircuitBreakerObject(model, transform, rank=2, lora_alpha=2.0,
                               target_layers=tgt, seed=1)
    for k in obj.B:
        obj.B[k].data = scale_up * torch.randn_like(obj.B[k])
    return obj


def test_cb_object_protocol_scale0_and_params(tiny_llama):
    model, tok = tiny_llama
    obj = _constructed_object(model)
    # trainable = every (layer, projection) A and B present on this model
    assert len(obj.trainable) == 2 * len(obj._modules)
    assert obj.n_params() == sum(p.numel() for p in obj.trainable)
    assert obj.n_params() > 0
    assert obj.inject("prompt text") == "prompt text"
    # every touched module is one of the seven projections
    assert all(name in PROJECTIONS for (_, name) in obj._modules)
    enc = tok("the quick brown fox jumps over", return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits
        with obj.apply(model, scale=0.0):
            zeroed = model(**enc).logits
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
    assert torch.equal(base, zeroed)             # scale-0 is bit-exact unsteered
    assert not torch.allclose(base, steered)     # ... and non-vacuously so


def test_cb_save_load_roundtrip(tiny_llama, tmp_path):
    model, tok = tiny_llama
    obj = _constructed_object(model)
    obj.metadata = {"fit_path": "explicit_kwargs", "target_layers": obj.target_layers}
    path = tmp_path / "cb.pt"
    obj.save(path)
    loaded = load_object(path, model, tok)  # _KIND_LOADERS dispatch
    assert isinstance(loaded, CircuitBreakerObject)
    assert loaded.kind == "circuit_breaker"
    assert loaded.transform_layers == obj.transform_layers
    assert loaded.target_layers == obj.target_layers
    assert loaded.rank == obj.rank
    assert set(loaded._modules) == set(obj._modules)
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad():
        with obj.apply(model, scale=1.0):
            orig = model(**enc).logits
        with loaded.apply(model, scale=1.0):
            rehydrated = model(**enc).logits
    assert torch.allclose(orig, rehydrated, atol=1e-6)


# ---------------------------------------------------------------- train smoke


def test_cb_train_smoke_explicit_texts(tiny_llama):
    # Called directly (train_circuit_breaker): no registry entry any more.
    model, tok = tiny_llama
    nl = num_layers(model)
    obj = train_circuit_breaker(
        model, tok, BY_ID["french"], _pairs(), layer=1,
        retain_texts=_retain(), reroute_texts=_reroute(),
        target_layers=(nl - 1,), rank=2, lora_alpha=2.0,
        max_steps=3, batch_size=2, max_len=24, alpha=10.0)
    assert isinstance(obj, CircuitBreakerObject)
    assert obj.kind == "circuit_breaker"
    assert obj.metadata["fit_path"] == "explicit_kwargs"
    assert obj.metadata["final_loss"] is not None
    assert obj.metadata["max_steps"] == 3
    assert obj.n_params() > 0
    # training moved the adapter off its init: the trained object steers.
    enc = tok("the quick brown fox jumps", return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
    assert not torch.allclose(base, steered)


def test_cb_refuses_without_data_or_fallback(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="retain_texts"):
        train_circuit_breaker(model, tok, BY_ID["french"], _pairs(),
                              layer=1, target_layers=(num_layers(model) - 1,),
                              rank=2, max_steps=1, batch_size=2, max_len=24)


def test_cb_pairs_fallback_recorded(tiny_llama):
    model, tok = tiny_llama
    obj = train_circuit_breaker(
        model, tok, BY_ID["french"], _pairs(), layer=1,
        allow_pairs_fallback=True, target_layers=(num_layers(model) - 1,),
        rank=2, lora_alpha=2.0, max_steps=2, batch_size=2, max_len=24)
    assert obj.metadata["fit_path"] == "pairs_fallback_si_derived"
    assert obj.metadata["n_retain"] == len(_pairs())
    assert obj.metadata["n_reroute"] == len(_pairs())


def test_cb_micro_batching_is_the_same_optimization(tiny_llama):
    """The 156649 OOM fix: micro-chunked training (backward per chunk against
    the FULL batch's denominator) must be the SAME optimization as the
    single-pass full batch — same rng draws, same samples, gradients that sum
    to the full-batch loss gradient — so the trained params must match the
    unchunked run to fp-summation-order noise."""
    model, tok = tiny_llama
    nl = num_layers(model)
    kw = dict(layer=1, retain_texts=_retain(), reroute_texts=_reroute(),
              target_layers=(nl - 1,), rank=2, lora_alpha=2.0,
              max_steps=3, batch_size=4, max_len=24, alpha=10.0, seed=0)
    full = train_circuit_breaker(
        model, tok, BY_ID["french"], _pairs(), micro_batch_size=4, **kw)
    chunked = train_circuit_breaker(
        model, tok, BY_ID["french"], _pairs(), micro_batch_size=1, **kw)
    assert full.metadata["micro_batch_size"] == 4
    assert chunked.metadata["micro_batch_size"] == 1
    assert full.metadata["batch_size"] == chunked.metadata["batch_size"] == 4
    for key in full._modules:
        assert torch.allclose(full.A[key], chunked.A[key],
                              atol=1e-5, rtol=1e-4), key
        assert torch.allclose(full.B[key], chunked.B[key],
                              atol=1e-5, rtol=1e-4), key
    assert abs(full.metadata["final_loss"]
               - chunked.metadata["final_loss"]) < 1e-4


# ---------------------------------------------------------------- wiring


def test_circuit_breaker_removed_from_registry_but_loader_kept():
    """circuit_breaker left the registry ENTIRELY (2026-08-20, user
    instruction): no ArmSpec, no COMPILED defense, no Class B cell — but the
    KIND LOADER stays, because the already-fitted artifact exists on disk
    and a peer lane's serving/kind-coverage canary hydrates it through
    load_object. (The round-trip itself is pinned by
    test_cb_save_load_roundtrip above.)"""
    assert "circuit_breaker" not in ARMS
    from sisteer.eval.defense_cell import CLASS_B_ARMS
    from sisteer.eval.jailbreak import COMPILED

    assert "circuit_breaker" not in COMPILED
    assert "circuit_breaker" not in CLASS_B_ARMS
    assert "circuit_breaker" in _KIND_LOADERS  # hydration survives
    # the stage2-evals arms survive the removal, appended in registry
    # order (did/adasteer, then the cheap-estimator caa/actadd arms) — a positional
    # pin, no longer a TAIL pin: the sdf-base-steering CE arms
    # (psr_ce_all/lora_r4_ce, 2026-09-01) are appended after them.
    assert COMPILED[27:32] == ("did", "adasteer", "caa", "actadd",
                               "actadd_front")
    # the stage2-evals arms survive the removal, appended at the tail in
    # registry order (did/adasteer, then the cheap-estimator caa/actadd arms)
    # the tail in RELATIVE order (a literal tail slice moves at every
    # append; appending is the rule) — did/adasteer, the cheap-estimator arms, the
    # transferred psr_xfer_* arms (merged 2026-09-02), the sdf-base DERIVED
    # LoRA arm (2026-09-10), then the UNDISTILLED paired write as the FINAL
    # registered arm (relu-differenced-native-qwen, 2026-09-17)
    _tail = ("did", "adasteer", "caa", "actadd", "actadd_front",
             "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
             "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1",
             "lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl")
    _idx = [COMPILED.index(a) for a in _tail]
    assert _idx == sorted(_idx) and len(set(_idx)) == len(_idx)
    assert COMPILED[-2:] == ("lora_r4_ce_minus_ctl", "psr_ce_all_minus_ctl")
