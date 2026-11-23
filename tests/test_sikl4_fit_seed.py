"""The fit seed reaches the multi-layer LoRA's initialisation on the lockstep path (the sikl4 refits).

``train_two_si_objects.train_si_objects`` reads ``train_kwargs.default.seed`` and, since the sikl4 change, passes it to
``lockstep.build_lockstep_object(seed=)``, which hands it to ``MultiLayerLoRAObject(seed=)`` (A ~ N(0, 1/d_in) drawn from
``manual_seed(seed + layer)``, B = 0). Before, the lockstep builder took no seed and every lockstep LoRA drew its A factor from
``manual_seed(0 + layer)``. Pinned on the tiny model:

- seed absent == seed 0 == the frozen constructor call, bit for bit: every fit that states no seed builds the object it always did;
- seeds 1, 2 and 3 give a different A on every layer and the same zero B (the untrained object is the unsteered model);
- the psr_kl_all object stays zero-init: byte-identical parameters under any seed (for the ReLU steer the seed varies data order only);
- two lockstep fits that differ only in seed see their batches in a different order (the same batches, permuted).

CPU-only: the shared ``tiny_llama`` fixture, synthetic pairs.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.lockstep import (LockstepArm, build_lockstep_object, default_lockstep_params,
                                         train_lockstep)
from sisteer.extraction.multilayer import MultiLayerLoRAObject
from sisteer.extraction.psr import PSRObject

INSTRUCTION = BY_ID["french"]
MAX_LEN = 128


def _pairs():
    data = [("Say something nice.", "Bonjour le monde", "Hello world"),
            ("Describe a tree.", "C'est un grand arbre qui pousse", "Tall tree"),
            ("Greet the reader.", "Salut a tous", "Hi everyone"),
            ("Name a color.", "Le bleu est joli et tres calme", "Blue is nice"),
            ("Count to two.", "Un deux", "One two"),
            ("Name a fruit.", "La pomme rouge est sucree", "Apple"),
            ("Say goodbye.", "Au revoir mes amis", "Goodbye")]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french") for p, pos, neg in data]


def _lora(model, tok, arm="lora_r4", **kw) -> MultiLayerLoRAObject:
    rank = {"lora_r4": 4}.get(arm)
    obj = build_lockstep_object(arm, model, tok, INSTRUCTION, _pairs(), layer=1,
                                **({"lora_rank": rank} if rank else {}), **kw)
    assert isinstance(obj, MultiLayerLoRAObject)
    return obj


def _same(a: MultiLayerLoRAObject, b: MultiLayerLoRAObject) -> None:
    assert a.layers == b.layers and a.rank == b.rank
    for l in a.layers:
        assert torch.equal(a.A[l], b.A[l]), l
        assert torch.equal(a.B[l], b.B[l]), l


def test_seed_absent_equals_seed_zero_equals_the_frozen_constructor(tiny_llama):
    model, tok = tiny_llama
    absent, zero = _lora(model, tok), _lora(model, tok, seed=0)
    _same(absent, zero)
    _same(zero, MultiLayerLoRAObject(model, rank=4))
    # the default-rank arm too (lora_r1_all: the constructor's own rank 1, legacy shapes)
    _same(_lora(model, tok, arm="lora_r1_all"), MultiLayerLoRAObject(model))
    _same(_lora(model, tok, arm="lora_r1_all", seed=0), MultiLayerLoRAObject(model))


def test_seeds_one_two_three_give_distinct_A_on_every_layer_and_zero_B(tiny_llama):
    model, tok = tiny_llama
    objs = {s: _lora(model, tok, seed=s) for s in (0, 1, 2, 3)}
    layers = objs[1].layers
    assert layers and all(o.layers == layers and o.rank == 4 for o in objs.values())
    for l in layers:
        for s in (1, 2, 3):
            assert objs[s].A[l].shape == objs[0].A[l].shape and objs[s].A[l].ndim == 2
            assert not objs[s].B[l].any() and torch.equal(objs[s].B[l], objs[0].B[l])
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            assert not torch.equal(objs[a].A[l], objs[b].A[l]), (l, a, b)
    # the draw IS the constructor's for that seed (manual_seed(seed + layer))
    _same(objs[2], MultiLayerLoRAObject(model, rank=4, seed=2))


def test_the_same_seed_is_reproducible(tiny_llama):
    model, tok = tiny_llama
    _same(_lora(model, tok, seed=3), _lora(model, tok, seed=3))


def test_psr_kl_all_stays_zero_init_under_any_seed(tiny_llama):
    model, tok = tiny_llama
    objs = [build_lockstep_object("psr_kl_all", model, tok, INSTRUCTION, _pairs(), layer=1, **kw)
            for kw in ({}, {"seed": 0}, {"seed": 1}, {"seed": 2}, {"seed": 3})]
    assert all(isinstance(o, PSRObject) for o in objs)
    ref = [t.detach().clone() for t in objs[0].trainable]
    for o in objs[1:]:
        got = list(o.trainable)
        assert len(got) == len(ref)
        for a, b in zip(ref, got):
            assert torch.equal(a, b)


def _loss_order(model, tok, seed: int) -> list[float]:
    """One lockstep fit at lr 0 (the object never moves, and B = 0 makes the LoRA's output the unsteered model's whatever its
    seed): its per-step losses are the per-batch losses in the order the fit drew the batches."""
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    obj = build_lockstep_object("lora_r4", model, tok, INSTRUCTION, _pairs(), layer=1, lora_rank=4, seed=seed)
    params = replace(default_lockstep_params("lora_r4"), lr=0.0)
    hist = train_lockstep(model, tok, [LockstepArm("lora_r4", obj, params)], examples, epochs=2, batch_size=1,
                          max_len=MAX_LEN, seed=seed, n_probe=2)
    return [float(x) for x in hist["lora_r4"]]


def test_fits_differing_only_in_seed_see_the_same_batches_in_another_order(tiny_llama):
    model, tok = tiny_llama
    h1, h2 = _loss_order(model, tok, 1), _loss_order(model, tok, 2)
    assert len(h1) == len(h2) == 2 * len(_pairs())
    assert sorted(h1) == pytest.approx(sorted(h2), rel=0, abs=1e-6)   # the same batches ...
    assert h1 != h2                                                    # ... in another order
    assert _loss_order(model, tok, 1) == h1                            # and the order is the seed's, reproducibly
