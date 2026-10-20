"""train_steered_lm / train_distill guard probes ride the SHARED estimator.

distill.py kept a second, parallel guard implementation that still read
``batches[0]`` — so ``train_distill`` / ``train_steered_lm`` called OUTSIDE
the lockstep driver retained exactly the 2-4-sample-floor hazard commit
e4aa8d3 removed from the lockstep path. The probe is now
:func:`distill.probe_steered_lm`: sha-rank selection over the whole pool,
per-example units, uncertainty and the recorded population — and for the
soft losses it literally CALLS ``lockstep.arm_probe_losses`` through a
wrapper object, pinned here against the shared estimator. ``loss='hard'``
keeps a small parallel leg by necessity (CE against teacher TOKENS has no
teacher-divergence counterpart); its chunk invariance is pinned too.

CPU-only: the tiny_llama fixture, synthetic invented prompts only.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.distill import (
    DistillExample,
    FloorRatioError,
    probe_steered_lm,
    train_steered_lm,
)
from sisteer.extraction.lockstep import arm_probe_losses, build_lockstep_object
from sisteer.extraction.subsample import sha_hex, sha_rank_indices
from sisteer.models import hidden_size

INSTRUCTION = BY_ID["french"]


def _pairs():
    data = [
        ("alpha question 0 about the sky.", "Le ciel est bleu", "x"),
        ("alpha question 1 about the sky.", "Le ciel est grand", "x"),
        ("alpha question 2 about the sky.", "Le ciel est clair", "x"),
        ("alpha question 3 about the sky.", "Le ciel est beau", "x"),
        ("beta question 0 about the sea.", "La mer est calme", "x"),
        ("beta question 1 about the sea.", "La mer est froide", "x"),
        ("gamma question 0 about a tree.", "Un grand arbre vert", "x"),
        ("gamma question 1 about a tree.", "Un petit arbre", "x"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _vec(model, fill=0.05):
    v = nn.Parameter(torch.zeros(hidden_size(model)))
    with torch.no_grad():
        v.fill_(fill)
    return v


def test_probe_steered_lm_equals_the_shared_estimator(tiny_llama):
    """The soft path IS arm_probe_losses: the same vector probed through
    probe_steered_lm (vector_provider) and through the lockstep wrapper
    object gives the same per-example values, means, sems and recorded
    selection — one estimator, two entry points."""
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    vec = _vec(model)

    via_distill = probe_steered_lm(
        model, tok, examples, 1, lambda idxs: vec,
        loss="soft", max_len=96, batch_size=2, n_probe=None)
    obj = build_lockstep_object("vector", model, tok, INSTRUCTION, pairs,
                                layer=1)
    with torch.no_grad():
        obj.trainable[0].copy_(vec)
    via_lockstep = arm_probe_losses(model, tok, obj, examples, max_len=96,
                                    batch_size=2, n_probe=None)

    assert via_distill.n_examples == via_lockstep.n_examples == 8
    assert list(via_distill.active_values) == pytest.approx(
        list(via_lockstep.active_values), abs=1e-9)
    assert list(via_distill.noop_values) == pytest.approx(
        list(via_lockstep.noop_values), abs=1e-9)
    assert via_distill.active == pytest.approx(via_lockstep.active, abs=1e-9)
    assert via_distill.noop == pytest.approx(via_lockstep.noop, abs=1e-9)
    assert via_distill.selected_sha8 == via_lockstep.selected_sha8


def test_probe_steered_lm_selects_by_sha_rank_and_records_it(tiny_llama):
    """n_probe below the pool: the sha-rank selection, recorded — never
    the pool prefix (which, on the family-sorted fixture, is all-alpha)."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    vec = _vec(model)

    est = probe_steered_lm(model, tok, examples, 1, lambda idxs: vec,
                           loss="soft", max_len=96, batch_size=2, n_probe=4)
    keys = [ex.prompt_without for ex in examples]
    expect = [sha_hex(keys[i])[:8] for i in sha_rank_indices(keys, 4)]
    assert est.n_examples == 4
    assert list(est.selected_sha8) == expect
    assert list(est.selected_sha8) != [sha_hex(k)[:8] for k in keys[:4]]


def test_hard_loss_probe_is_chunk_invariant(tiny_llama):
    """The hard (CE) leg is a parallel implementation by necessity — same
    selection, same per-example unit, and the same batch_size-is-memory-
    only contract, pinned here."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    vec = _vec(model)

    ests = [probe_steered_lm(model, tok, examples, 1, lambda idxs: vec,
                             loss="hard", max_len=96, batch_size=bs,
                             n_probe=None)
            for bs in (2, 3, 8)]
    ref = ests[0]
    assert ref.n_examples == 8
    for est in ests[1:]:
        assert est.active == pytest.approx(ref.active, abs=1e-6)
        assert est.noop == pytest.approx(ref.noop, abs=1e-6)
        assert list(est.active_values) == pytest.approx(
            list(ref.active_values), abs=1e-6)


def test_probe_groups_by_instruction_for_hypernet_pools(tiny_llama):
    """A multi-instruction pool (the hypersteer case) probes each group
    under ITS OWN vector — the wrapper fetches from vector_provider at
    probe time — and the estimate covers every group."""
    model, tok = tiny_llama
    pairs = _pairs()
    base = pairs_to_distill_examples(tok, pairs[:4], INSTRUCTION)
    other = [DistillExample(ex.prompt_with_si, ex.prompt_without,
                            ex.completion, instruction_idx=1)
             for ex in pairs_to_distill_examples(tok, pairs[4:],
                                                 INSTRUCTION)]
    examples = base + other
    vecs = {0: _vec(model, 0.05), 1: _vec(model, -0.05)}
    seen: list[int] = []

    def provider(idxs):
        seen.extend(idxs)
        return vecs[idxs[0]]

    est = probe_steered_lm(model, tok, examples, 1, provider,
                           loss="soft", max_len=96, batch_size=2,
                           n_probe=None)
    assert est.n_examples == 8
    assert set(seen) == {0, 1}


def test_train_steered_lm_guard_rules_on_the_pool_not_batch0(tiny_llama):
    """End to end: the trainer's diagnostics carry the probe's realised
    sample size (the POOL, not the training batch), its uncertainty and
    its recorded selection; the floor-ratio guard still rules — on the
    shared estimator's numbers now."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    vec = _vec(model, 0.0)
    diag: dict = {}
    train_steered_lm(model, tok, examples, 1, lambda idxs: vec, [vec],
                     loss="soft", epochs=1, batch_size=2, max_len=96,
                     diagnostics=diag)
    assert diag["probe_n"] == 8            # the pool, not batch 0's 2
    assert diag["probe_selection"] == "sha256_rank(prompt_without)"
    assert len(diag["probe_selected_sha8"]) == 8
    for k in ("noop_floor_sem", "final_probe_sem", "floor_ratio_sem"):
        assert isinstance(diag[k], float), k
    assert diag["floor_ratio"] == pytest.approx(
        diag["final_probe_loss"] / max(diag["noop_floor"], 1e-6))

    with pytest.raises(FloorRatioError):
        v2 = _vec(model, 0.0)
        train_steered_lm(model, tok, examples, 1, lambda idxs: v2, [v2],
                         loss="soft", epochs=1, batch_size=2, max_len=96,
                         floor_ratio_max=1e-9)
