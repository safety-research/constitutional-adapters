"""Seed threading into the LoRA CE arm's object init (2026-09-15).

``methods._train_lora_ce`` passes the config seed (``train_kwargs.default.
seed`` -> ``candidate_kw["seed"]`` -> ``kw["seed"]``) into
``MultiLayerLoRAObject(seed=)``: two replicates now differ in their
A-factor initialisation as well as in data order. Before this the init
generator was ``manual_seed(0 + l)`` for EVERY config. Pinned here through
the production constructor path (the trainer stubbed — object construction
is the subject), on the tiny model:

- seed 1 vs seed 2 -> a different ``A`` on EVERY layer, identical zero
  ``B`` (the untrained object is the unsteered model at any seed);
- the same seed twice -> identical ``A``;
- seed absent == seed 0 == the frozen constructor call (no seed kwarg),
  bit for bit — every existing config's object is unchanged;
- the seed still reaches the trainer (read, never popped);
- the PSR arm's object takes no seed (zero directions, gate w 0 / b 1 by
  design): byte-identical parameters under any seed — for ReLU the seed
  governs data order and the held-out split only.

CPU-only: tiny model (the shared ``tiny_llama`` fixture), synthetic
doc-shaped pairs — never staged pools.
"""

from __future__ import annotations

import pytest
import torch

from sisteer import methods
from sisteer.data.pairs import Pair
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.psr import PSRObject

SI = "constv2-mix-1k"
LAYERS = [0, 1]


def _docs():
    rows = [("Sys scaffold\n\n", "A doc about trees and rivers."),
            ("", "Story body with no masked prefix at all here."),
            ("Second sys\n\n", "Another advice body follows the user turn."),
            ("", "Yet another tiny story body.")]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=SI)
            for p, pos in rows]


@pytest.fixture
def stubbed_trainer(monkeypatch):
    """Replace train_object_ce with a recorder: the object comes back
    exactly as constructed, and the kwargs the trainer WOULD have received
    are kept (the seed must still be among them)."""
    calls: list[dict] = []

    def fake_train_object_ce(model, tok, obj, exs, **kw):
        calls.append(dict(kw))
        return None

    monkeypatch.setattr(methods, "train_object_ce", fake_train_object_ce)
    return calls


def _lora(model, tok, **kw) -> MultiLayerLoRAObject:
    obj = methods._train_lora_ce(model, tok, SI, _docs(), layers=LAYERS,
                                 lora_rank=4, **kw)
    assert isinstance(obj, MultiLayerLoRAObject)
    return obj


def _same(a: MultiLayerLoRAObject, b: MultiLayerLoRAObject) -> None:
    assert a.layers == b.layers and a.rank == b.rank
    for l in a.layers:
        assert torch.equal(a.A[l], b.A[l]), l
        assert torch.equal(a.B[l], b.B[l]), l


def test_different_seeds_give_different_A_on_every_layer_and_zero_B(
        tiny_llama, stubbed_trainer):
    model, tok = tiny_llama
    o1, o2 = _lora(model, tok, seed=1), _lora(model, tok, seed=2)
    assert o1.layers == o2.layers == LAYERS and o1.rank == o2.rank == 4
    for l in LAYERS:
        assert o1.A[l].shape == o2.A[l].shape and o1.A[l].ndim == 2
        assert not torch.equal(o1.A[l], o2.A[l]), l
        # B is zero-init at every seed: the untrained object is the
        # unsteered model whatever the replicate
        assert not o1.B[l].any() and not o2.B[l].any()
        assert torch.equal(o1.B[l], o2.B[l])
    # the seed is READ, never popped: the trainer saw both
    assert [c["seed"] for c in stubbed_trainer] == [1, 2]


def test_same_seed_is_reproducible(tiny_llama, stubbed_trainer):
    model, tok = tiny_llama
    _same(_lora(model, tok, seed=2), _lora(model, tok, seed=2))
    _same(_lora(model, tok, seed=1), _lora(model, tok, seed=1))


def test_seed_absent_equals_seed_zero_equals_the_frozen_constructor(
        tiny_llama, stubbed_trainer):
    """The frozen behaviour, bit for bit: a config with no seed builds the
    object every batch-8 cell was built with — MultiLayerLoRAObject(model,
    layers=, rank=) with the constructor's own default seed 0 — and so does
    an explicit seed 0."""
    model, tok = tiny_llama
    absent = _lora(model, tok)
    zero = _lora(model, tok, seed=0)
    frozen = MultiLayerLoRAObject(model, layers=LAYERS, rank=4)
    _same(absent, zero)
    _same(zero, frozen)
    # an absent seed stays absent for the trainer (its own default rules);
    # an explicit 0 reaches it as 0
    assert "seed" not in stubbed_trainer[0]
    assert stubbed_trainer[1]["seed"] == 0


def test_seed_none_is_read_as_zero(tiny_llama, stubbed_trainer):
    """A YAML ``seed: null`` must not crash the constructor path: it builds
    the seed-0 object (the trainer's handling of None is its own)."""
    model, tok = tiny_llama
    _same(_lora(model, tok, seed=None), _lora(model, tok, seed=0))


def test_all_layer_default_follows_the_seed_too(tiny_llama, stubbed_trainer):
    """The production configs pass no explicit layer list (the constructor's
    capability-filtered default): seeds 1 and 2 differ on every layer there
    too, and the layer set itself is seed-independent."""
    model, tok = tiny_llama
    o1 = methods._train_lora_ce(model, tok, SI, _docs(), lora_rank=4, seed=1)
    o2 = methods._train_lora_ce(model, tok, SI, _docs(), lora_rank=4, seed=2)
    assert o1.layers == o2.layers and len(o1.layers) >= 1
    for l in o1.layers:
        assert not torch.equal(o1.A[l], o2.A[l]), l
        assert torch.equal(o1.B[l], o2.B[l]) and not o1.B[l].any()


def test_psr_object_is_seed_invariant(tiny_llama, monkeypatch):
    """The PSR arm's object has no seed parameter and none is threaded:
    zero directions, zero gate w, gate b = 1 by design — byte-identical
    parameters under no seed, seed 0, 1 and 2: the seed
    governs the ReLU arm's data order and split only."""
    import sisteer.extraction.psr as psr_mod

    model, tok = tiny_llama
    seen: list[dict] = []
    monkeypatch.setattr(methods, "train_object_ce",
                        lambda m, t, obj, exs, **kw: seen.append(dict(kw)))
    # the post-training gate guard measures liveness on the trained
    # weights; with training stubbed every gate is "live" for this test
    monkeypatch.setattr(psr_mod, "check_psr_gates",
                        lambda model, tok, **kw: {l: 1.0 for l in kw["obj"].layers})

    def _psr(seed=None) -> PSRObject:
        kw = {} if seed is None else {"seed": seed}
        obj = methods._train_psr_ce(model, tok, SI, _docs(), layers=LAYERS, **kw)
        assert isinstance(obj, PSRObject)
        return obj

    objs = [_psr(s) for s in (None, 0, 1, 2)]
    ref = objs[0]
    for obj in objs:
        assert obj.layers == LAYERS and obj.train_direction is True
        for l in LAYERS:
            assert torch.equal(obj.directions[l], ref.directions[l]), l
            assert torch.equal(obj.gate_w[l], ref.gate_w[l]), l
            assert torch.equal(obj.gate_b[l], ref.gate_b[l]), l
            # the by-design init, at every seed
            assert not obj.directions[l].any()
            assert not obj.gate_w[l].any()
            assert torch.equal(obj.gate_b[l], torch.ones(1))
    # the seed still reaches the trainer for the epoch shuffle
    assert [c.get("seed", "absent") for c in seen] == ["absent", 0, 1, 2]
