"""Lockstep teacher sharing (efficiency optimisation 4).

The load-bearing property: a lockstep-trained object is BITWISE identical to
the same arm trained solo through its registry trainer, given the same seed,
data order and batch composition. These tests prove that against the REAL
``methods.ARMS[arm].train`` path (not a reimplementation), for every
lockstep-compatible arm, and prove that incompatible arms are excluded rather
than silently coerced.

CPU-only: the tiny_llama fixture, synthetic French pairs (never real store
data).
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.lockstep import (
    LockstepArm,
    ProbeEstimate,
    arm_probe_losses,
    build_lockstep_object,
    default_lockstep_params,
    lockstep_compatible,
    lockstep_incompatible_reason,
    train_lockstep,
    _LOCKSTEP_COMPATIBLE,
    _lr_at_factory,
)
from sisteer.methods import ARMS

# A registered instruction + synthetic pairs — invented text only, never
# real store data.
INSTRUCTION = BY_ID["french"]


def _pairs():
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


# The lockstep-compatible arms that are also OBJECT arms trained through
# train_object_kl — the ones the two-SI matrix actually uses. Comparing these
# against ARMS[arm].train exercises the production path end to end.
OBJECT_ARMS = ["vector_all", "lora_r1", "lora_r1_all", "token_bos",
               "psr_kl", "psr_kl_all"]

TRAIN_KW = dict(epochs=2, batch_size=2, max_len=96, seed=0)

# Soft-token arms need a single-token sentinel; the tiny tokenizer has none to
# auto-find, so pass its unk token (the convention the other suites use). The
# real Llama-3.1 tokenizer auto-finds a reserved token, so production needs no
# such kwarg.
_TOKEN_ARMS = {"token_bos", "token_faithful"}


def _build_kw(arm, tok):
    if arm in _TOKEN_ARMS:
        return {"sentinel": (tok.unk_token, tok.unk_token_id)}
    return {}


def _trainable_snapshot(obj):
    return [p.detach().clone() for p in obj.trainable]


def _assert_bitwise(a, b, *, arm):
    ta, tb = _trainable_snapshot(a), _trainable_snapshot(b)
    assert len(ta) == len(tb), f"{arm}: param count differs"
    for i, (x, y) in enumerate(zip(ta, tb)):
        assert x.shape == y.shape, f"{arm}: param {i} shape {x.shape} vs {y.shape}"
        assert torch.equal(x, y), (
            f"{arm}: param {i} not bitwise identical "
            f"(max abs diff {float((x - y).abs().max())})")


@pytest.mark.parametrize("arm", OBJECT_ARMS)
def test_lockstep_matches_solo_registry_trainer_bitwise(tiny_llama, arm):
    """Train ``arm`` SOLO via its registry trainer, and in a lockstep group
    with a DIFFERENT compatible arm; assert its trained parameters are bitwise
    identical. If exact equality were unattainable this test would fail loudly
    (never silently relaxed to allclose)."""
    model, tok = tiny_llama
    pairs = _pairs()
    layer = 1

    solo = ARMS[arm].train(model, tok, INSTRUCTION, pairs, layer=layer,
                           **_build_kw(arm, tok), **TRAIN_KW)

    # A second compatible arm sharing the group — its presence must not
    # perturb `arm`'s trajectory. vector_all needs no sentinel and is a valid
    # partner for every arm except itself (then lora_r1).
    other = "lora_r1" if arm == "vector_all" else "vector_all"
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    group = [
        LockstepArm(arm, build_lockstep_object(arm, model, tok, INSTRUCTION,
                                               pairs, layer=layer,
                                               **_build_kw(arm, tok)),
                    default_lockstep_params(arm)),
        LockstepArm(other, build_lockstep_object(other, model, tok, INSTRUCTION,
                                                 pairs, layer=layer,
                                                 **_build_kw(other, tok)),
                    default_lockstep_params(other)),
    ]
    train_lockstep(model, tok, group, examples,
                   epochs=TRAIN_KW["epochs"], batch_size=TRAIN_KW["batch_size"],
                   max_len=TRAIN_KW["max_len"], seed=TRAIN_KW["seed"])

    _assert_bitwise(solo, group[0].obj, arm=arm)


def test_lockstep_group_order_does_not_matter(tiny_llama):
    """The SAME arm trained in a group is bitwise identical whether it is
    listed first or second — the group is order-agnostic for each arm."""
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    def run(order):
        objs = {a: build_lockstep_object(a, model, tok, INSTRUCTION, pairs,
                                         layer=1, **_build_kw(a, tok))
                for a in order}
        group = [LockstepArm(a, objs[a], default_lockstep_params(a))
                 for a in order]
        train_lockstep(model, tok, group, examples, **TRAIN_KW)
        return objs

    first = run(["lora_r1", "token_bos"])
    second = run(["token_bos", "lora_r1"])
    _assert_bitwise(first["lora_r1"], second["lora_r1"], arm="lora_r1")
    _assert_bitwise(first["token_bos"], second["token_bos"], arm="token_bos")


def test_lockstep_vector_arm_matches_train_steered_lm(tiny_llama):
    """The additive-vector arm (train_steered_lm path) is bitwise identical
    solo vs lockstep too — the second engine the shared step reproduces."""
    model, tok = tiny_llama
    pairs = _pairs()
    layer = 1
    solo = ARMS["vector"].train(model, tok, INSTRUCTION, pairs, layer=layer,
                                **TRAIN_KW)  # a VectorObject
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    group = [
        LockstepArm("vector",
                    build_lockstep_object("vector", model, tok, INSTRUCTION,
                                          pairs, layer=layer),
                    default_lockstep_params("vector")),
        LockstepArm("lora_r1",
                    build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                          pairs, layer=layer),
                    default_lockstep_params("lora_r1")),
    ]
    train_lockstep(model, tok, group, examples, **TRAIN_KW)
    # solo is a VectorObject; compare its single trained vector to the lockstep
    # vector arm's bare Parameter.
    solo_vec = solo.sv.vectors[layer]
    lock_vec = group[0].obj.trainable[0].detach().float().cpu()
    assert torch.equal(solo_vec, lock_vec), (
        f"vector arm not bitwise identical (max abs diff "
        f"{float((solo_vec - lock_vec).abs().max())})")


def test_divergence_ablation_arms_are_compatible_and_faithful(tiny_llama):
    """The rKL / JS divergence-ablation vectors share teachers too, at their
    own divergence — bitwise identical solo vs lockstep."""
    model, tok = tiny_llama
    pairs = _pairs()
    layer = 1
    for arm in ("vector_rkl", "vector_js"):
        solo = ARMS[arm].train(model, tok, INSTRUCTION, pairs, layer=layer,
                               **TRAIN_KW)
        examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
        group = [LockstepArm(arm, build_lockstep_object(arm, model, tok,
                                                        INSTRUCTION, pairs,
                                                        layer=layer),
                             default_lockstep_params(arm))]
        train_lockstep(model, tok, group, examples, **TRAIN_KW)
        solo_vec = solo.sv.vectors[layer]
        lock_vec = group[0].obj.trainable[0].detach().float().cpu()
        assert torch.equal(solo_vec, lock_vec), f"{arm} not bitwise identical"


# -- compatibility predicate + exclusion --------------------------------------


def test_lockstep_compatible_predicate_matches_registry():
    """Every compatible arm is registered (or is a known vector arm), and the
    exclusion reasons cover the incompatible registered arms — so a new arm
    cannot silently drift the compatibility set."""
    for arm in _LOCKSTEP_COMPATIBLE:
        assert lockstep_compatible(arm)
        assert lockstep_incompatible_reason(arm) is None

    # The incompatible arms that ARE registered must all report a reason.
    incompatible_registered = [a for a in ARMS if not lockstep_compatible(a)]
    assert incompatible_registered, "expected some incompatible registered arms"
    for arm in incompatible_registered:
        reason = lockstep_incompatible_reason(arm)
        assert reason, f"{arm}: incompatible but no reason given"

    # The specific exclusions the task names.
    for arm in ("psr_mse_all", "loreft", "lofit", "alphakl_fisher", "bidpo",
                "psr_kl_all_hard"):
        assert not lockstep_compatible(arm), f"{arm} must be excluded"


def test_train_lockstep_refuses_incompatible_arm(tiny_llama):
    """A group containing an incompatible arm raises rather than coercing it."""
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    class _Dummy:
        trainable = [torch.nn.Parameter(torch.zeros(2))]

        def inject(self, s):
            return s

    with pytest.raises(ValueError, match="psr_mse_all"):
        train_lockstep(model, tok,
                       [LockstepArm("psr_mse_all", _Dummy())], examples,
                       **TRAIN_KW)


def test_build_lockstep_object_refuses_incompatible_arm(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="not lockstep-compatible"):
        build_lockstep_object("loreft", model, tok, INSTRUCTION, _pairs(),
                              layer=1)


def test_distill_teacher_logits_hook_is_transparent(tiny_llama):
    """The additive ``teacher_logits_hook`` on ``train_steered_lm`` is a pure
    interposition: feeding it exactly the logits the local teacher forward
    would compute yields bitwise-identical training to leaving it unset."""
    import torch.nn as nn
    from sisteer.extraction.distill import (
        _gather_completion_logits, train_steered_lm)
    from sisteer.models import hidden_size

    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    kw = dict(loss="soft", epochs=2, batch_size=2, max_len=96, seed=0)

    def run(use_hook):
        vec = nn.Parameter(torch.zeros(hidden_size(model)))

        def hook(_bi, teacher_b):
            with torch.no_grad():
                logits = model(input_ids=teacher_b.input_ids,
                               attention_mask=teacher_b.attention_mask).logits
            return _gather_completion_logits(logits, teacher_b)

        hist = train_steered_lm(
            model, tok, examples, 1,
            vector_provider=lambda idxs: vec, trainable_params=[vec],
            teacher_logits_hook=(hook if use_hook else None), **kw)
        return vec.detach().clone(), hist

    v_off, h_off = run(False)
    v_on, h_on = run(True)
    assert torch.equal(v_off, v_on), "hook changed the trained vector"
    assert h_off == h_on, "hook changed the loss history"


# -- per-arm optimiser schedule (2026-08-24): opt-in, partner-inert ------------
#
# psr_kl_all's fixed-lr KL fit ran away in the wave while its mse sibling got
# warmup/cosine/clipping inside train_psr; the shared lockstep loop now takes
# the SAME treatment per arm, through LockstepArmParams. The load-bearing
# property is the default: an arm that does not opt in — including the wave's
# working lora_r1 / vector / token_faithful — trains bit-identically to the
# historical loop, even when a group partner IS scheduled.


def test_scheduled_arm_departs_while_partner_stays_bitwise(tiny_llama):
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    def run(schedule_vector: bool):
        objs = {a: build_lockstep_object(a, model, tok, INSTRUCTION, pairs,
                                         layer=1) for a in ("vector",
                                                            "lora_r1")}
        v_params = default_lockstep_params("vector")
        if schedule_vector:
            v_params = replace(v_params, lr_schedule="warmup_cosine",
                               warmup_steps=1, max_grad_norm=1.0)
        group = [LockstepArm("vector", objs["vector"], v_params),
                 LockstepArm("lora_r1", objs["lora_r1"],
                             default_lockstep_params("lora_r1"))]
        train_lockstep(model, tok, group, examples, **TRAIN_KW)
        return objs

    plain = run(schedule_vector=False)
    mixed = run(schedule_vector=True)
    # the unscheduled partner is bit-identical to today — a group member's
    # schedule cannot perturb it
    _assert_bitwise(plain["lora_r1"], mixed["lora_r1"], arm="lora_r1")
    # ...and the schedule really engaged: the scheduled arm departed
    v_plain = plain["vector"].trainable[0].detach()
    v_mixed = mixed["vector"].trainable[0].detach()
    assert not torch.equal(v_plain, v_mixed), \
        "schedule opt-in changed nothing — it never engaged"


def test_schedule_params_validate_loudly(tiny_llama):
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    def group_with(**kw):
        obj = build_lockstep_object("vector", model, tok, INSTRUCTION, pairs,
                                    layer=1)
        return [LockstepArm("vector", obj,
                            replace(default_lockstep_params("vector"), **kw))]

    with pytest.raises(ValueError, match="lr_schedule"):
        train_lockstep(model, tok, group_with(lr_schedule="linear"),
                       examples, **TRAIN_KW)
    with pytest.raises(ValueError, match="warmup_steps"):
        train_lockstep(model, tok, group_with(warmup_steps=3),
                       examples, **TRAIN_KW)


def test_lr_at_factory_is_train_psrs_formula():
    """Warmup ramp lr*(i+1)/warmup, cosine to 0 over the remaining budget,
    'constant' = the historical fixed lr — train_psr's mse-loop schedule,
    value for value, so the vocabulary means one thing across trainers."""
    lr_at = _lr_at_factory(1.0, "warmup_cosine", 2, 6)
    assert lr_at(0) == pytest.approx(0.5)      # warmup step 1 of 2
    assert lr_at(1) == pytest.approx(1.0)      # warmup step 2 of 2
    assert lr_at(2) == pytest.approx(1.0)      # cosine frac 0
    assert lr_at(4) == pytest.approx(0.5)      # cosine midpoint
    assert lr_at(5) == pytest.approx(
        0.5 * (1.0 + math.cos(math.pi * 3 / 4)))
    const = _lr_at_factory(0.7, "constant", 2, 6)
    assert [const(i) for i in range(6)] == [0.7] * 6


# -- the floor probe: sample size decoupled from batch size --------------------
#
# The wave-wide measurement failure: arm_probe_losses read BATCH 0 ONLY at
# the arm's training batch size, so every recorded floor and ratio was a
# 2-4 sample estimate of a KL — within one SI the batch-0 heldout floor
# read 0.01510 / 0.03816 / 0.04650 depending only on which arm's probe
# batch it landed in, and si_irrelevant's floor read 2x the truth measured
# over 256 prompts (job 187947). These tests pin the fix: the estimate
# averages over n_probe examples, reports its own uncertainty, clamps to a
# tiny pool, and stays a 2-tuple for every existing caller.


def _many_pairs():
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
        ("Offer some tea.", "Voulez-vous du the", "Would you like tea"),
        ("Count to two.", "Un puis deux", "One then two"),
        ("Name a planet.", "Mars est rouge", "Mars is red"),
        ("Describe the sea.", "La mer est calme", "The sea is calm"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _probe_obj(model, tok, pairs):
    """A live vector object (nonzero, so active != no-op)."""
    obj = build_lockstep_object("vector", model, tok, INSTRUCTION, pairs,
                                layer=1)
    with torch.no_grad():
        obj.trainable[0].fill_(0.05)
    return obj


def test_arm_probe_losses_averages_over_n_probe_not_batch0(tiny_llama):
    """The estimate is the mean over n_probe examples — batch 0 (the OLD
    default, silently the arm's training batch size) is one term of it,
    not the whole measurement."""
    model, tok = tiny_llama
    pairs = _many_pairs()
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    est = arm_probe_losses(model, tok, obj, examples, max_len=96,
                           batch_size=2, n_probe=None)
    assert isinstance(est, ProbeEstimate)
    assert est.n_examples == 8 and est.n_batches == 4 and est.batch_size == 2

    # Per-slice reference values: the probe on each 2-slice in isolation
    # carries the SAME per-example values the pooled estimate averages.
    slices = [arm_probe_losses(model, tok, obj, examples[i:i + 2],
                               max_len=96, batch_size=2)
              for i in range(0, 8, 2)]
    assert est.active == pytest.approx(sum(s.active for s in slices) / 4)
    assert est.noop == pytest.approx(sum(s.noop for s in slices) / 4)
    flat = [v for s in slices for v in s.active_values]
    assert list(est.active_values) == pytest.approx(flat)
    # ...and batch 0 alone — the old estimate in full — differs materially
    # from the mean (the fixture's batches are distinct texts on purpose).
    assert est.active != pytest.approx(slices[0].active)
    assert est.noop != pytest.approx(slices[0].noop)
    # the reported sd/sem are exactly the PER-EXAMPLE spread of the mean
    m = sum(flat) / 8
    sd = math.sqrt(sum((v - m) ** 2 for v in flat) / 7)
    assert est.active_sd == pytest.approx(sd)
    assert est.active_sem == pytest.approx(sd / math.sqrt(8.0))


def test_probe_sem_is_the_per_example_spread(tiny_llama):
    """The estimate's unit is the EXAMPLE: mean/sd/sem are computed over
    per-example token-mean divergences, so the uncertainty tightens with
    n_probe and never moves with the memory chunking. A smaller probe over
    the same pool is a SUBSET of a larger one (selection nests), so the
    two stay comparable."""
    model, tok = tiny_llama
    pairs = _many_pairs()
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    est8 = arm_probe_losses(model, tok, obj, examples, max_len=96,
                            batch_size=3, n_probe=None)   # ragged chunks
    assert est8.n_examples == 8
    vals = list(est8.active_values)
    assert est8.active == pytest.approx(sum(vals) / 8)
    m = sum(vals) / 8
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / 7)
    assert est8.active_sem == pytest.approx(sd / math.sqrt(8.0))

    est4 = arm_probe_losses(model, tok, obj, examples, max_len=96,
                            batch_size=3, n_probe=4)
    assert est4.n_examples == 4
    # nesting: the 4-example probe's values are a subset of the 8's
    for v in est4.active_values:
        assert any(abs(v - w) < 1e-6 for w in est8.active_values)


def test_probe_estimate_invariant_to_batch_size(tiny_llama):
    """THE estimator contract: batch_size chunks the no-grad forwards
    for MEMORY and changes nothing else. The same examples chunked at
    2, 3, 4 and 8 give the same mean, ratio and sem (within float
    tolerance) — before the fix, batchmeans over ragged chunks meant
    an object near the 1.0 bar could pass or fail on batch_size alone."""
    model, tok = tiny_llama
    pairs = _many_pairs()   # 8 distinct texts, varied completion lengths
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    ests = [arm_probe_losses(model, tok, obj, examples, max_len=96,
                             batch_size=bs, n_probe=None)
            for bs in (2, 3, 4, 8)]
    ref = ests[0]
    assert ref.n_examples == 8
    for est in ests[1:]:
        assert est.n_examples == 8
        assert est.active == pytest.approx(ref.active, abs=1e-6)
        assert est.noop == pytest.approx(ref.noop, abs=1e-6)
        # ratio and its sem divide by a near-zero fixture floor, which
        # amplifies benign float noise from padding-width changes by 1/f^2
        # — relative tolerance there, absolute everywhere else.
        assert est.ratio == pytest.approx(ref.ratio, rel=1e-3)
        assert est.active_sem == pytest.approx(ref.active_sem, abs=1e-6)
        assert est.noop_sem == pytest.approx(ref.noop_sem, abs=1e-6)
        assert est.ratio_sem == pytest.approx(ref.ratio_sem, rel=1e-3)
        assert list(est.active_values) == pytest.approx(
            list(ref.active_values), abs=1e-6)


def test_soft_divergence_per_token_matches_batchmean():
    """The per-token divergence is the UNREDUCED soft_divergence:
    mean over tokens == batchmean, for every loss — the identity that
    keeps the per-example estimator the same quantity the training step
    optimises."""
    from sisteer.extraction.distill import (SOFT_LOSSES, soft_divergence,
                                            soft_divergence_per_token)
    g = torch.Generator().manual_seed(7)
    t_logp = torch.log_softmax(torch.randn(11, 13, generator=g), -1)
    s_logp = torch.log_softmax(torch.randn(11, 13, generator=g), -1)
    for loss in SOFT_LOSSES:
        per_tok = soft_divergence_per_token(loss, t_logp, s_logp)
        assert per_tok.shape == (11,)
        assert float(per_tok.mean()) == pytest.approx(
            float(soft_divergence(loss, t_logp, s_logp)), abs=1e-6)


def test_per_example_means_groups_row_major():
    from sisteer.extraction.distill import per_example_means
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert per_example_means(vals, [2, 1, 3]) == pytest.approx(
        [1.5, 3.0, 5.0])
    assert per_example_means(vals, torch.tensor([2, 1, 0, 3])) == \
        pytest.approx([1.5, 3.0, 5.0])  # 0-count rows are skipped
    with pytest.raises(ValueError, match="disagree"):
        per_example_means(vals, [2, 2])


def test_arm_probe_losses_clamps_to_a_tiny_pool(tiny_llama):
    """n_probe far beyond the pool uses the WHOLE pool — no crash, no
    over-sampling, the realised count recorded. A single chunk still
    resolves the spread: the unit is the example, not the batch."""
    model, tok = tiny_llama
    pairs = _pairs()[:3]
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    est = arm_probe_losses(model, tok, obj, examples, max_len=96,
                           batch_size=2, n_probe=256)
    assert est.n_examples == 3 and est.n_batches == 2
    assert math.isfinite(est.active) and math.isfinite(est.noop)
    assert est.active_sd is not None and est.ratio_sem is not None

    # batch_size larger than the pool: ONE forward, the SAME estimate —
    # the spread is over the 3 examples either way (the batchmean-of-
    # ragged-chunks estimator this replaced reported None here, and a
    # DIFFERENT mean at every batch_size).
    one = arm_probe_losses(model, tok, obj, examples, max_len=96,
                           batch_size=8, n_probe=256)
    assert one.n_batches == 1 and one.batch_size == 3
    assert one.active == pytest.approx(est.active, abs=1e-6)
    assert one.noop == pytest.approx(est.noop, abs=1e-6)
    assert one.active_sd == pytest.approx(est.active_sd, abs=1e-6)
    # relative: the ratio sem divides by the fixture's near-zero floor,
    # which amplifies benign padding-width float noise by 1/f^2
    assert one.ratio_sem == pytest.approx(est.ratio_sem, rel=1e-3)

    with pytest.raises(ValueError, match="n_probe"):
        arm_probe_losses(model, tok, obj, examples, max_len=96, n_probe=0)
    with pytest.raises(ValueError, match="at least one example"):
        arm_probe_losses(model, tok, obj, [], max_len=96)


def test_probe_estimate_stays_a_two_tuple_for_existing_callers(tiny_llama):
    """Backward compatibility is load-bearing: lofit.heldout_probe and the
    fit driver unpack ``act, floor = arm_probe_losses(...)`` — the richer
    return must keep that working, uncertainty riding as attributes."""
    model, tok = tiny_llama
    pairs = _pairs()
    obj = _probe_obj(model, tok, pairs)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    est = arm_probe_losses(model, tok, obj, examples, max_len=96,
                           batch_size=2)
    act, floor = est                     # the historical unpack
    assert isinstance(est, tuple) and len(est) == 2
    assert (act, floor) == (est.active, est.noop)
    assert isinstance(act, float) and isinstance(floor, float)
    # the ratio convention matches the recorded floor_ratio fields
    assert est.ratio == pytest.approx(est.active / max(est.noop, 1e-6))


def test_probe_widening_cannot_perturb_training(tiny_llama):
    """The probes are no-grad, no-RNG, no-write: training the same arm with
    a 2-example probe and an every-example probe yields bitwise identical
    parameters — widening the measurement can never move the measured."""
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)

    def run(n_probe):
        obj = build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                    pairs, layer=1)
        group = [LockstepArm("lora_r1", obj,
                             default_lockstep_params("lora_r1"))]
        train_lockstep(model, tok, group, examples, epochs=2, batch_size=2,
                       max_len=96, seed=0, n_probe=n_probe)
        return obj

    _assert_bitwise(run(2), run(None), arm="lora_r1")


def test_lockstep_diagnostics_carry_probe_uncertainty(tiny_llama):
    """The group's diagnostics (-> each arm's fit_record sidecar) carry the
    probe means WITH sd/sem, the ratio's own sem, and the realised sample
    counts; the per-epoch heldout stream logs its mean AND sem so a rising
    probe is visible live, not post-hoc."""
    model, tok = tiny_llama
    pairs = _pairs()
    h_data = [("Offer some tea.", "Voulez-vous du the", "Would you like tea"),
              ("Count to two.", "Un puis deux", "One then two"),
              ("Name a planet.", "Mars est rouge", "Mars is red"),
              ("Describe the sea.", "La mer est calme", "The sea is calm")]
    h_pairs = [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
               for p, pos, neg in h_data]
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    heldout = pairs_to_distill_examples(tok, h_pairs, INSTRUCTION)

    logged: list = []
    diag: dict = {}
    group = [LockstepArm("vector",
                         build_lockstep_object("vector", model, tok,
                                               INSTRUCTION, pairs, layer=1),
                         default_lockstep_params("vector"))]
    train_lockstep(model, tok, group, examples, epochs=1, batch_size=2,
                   max_len=96, seed=0, heldout=heldout, diagnostics=diag,
                   log_fn=lambda d, *a, **k: logged.append(d),
                   n_probe=None)

    d = diag["vector"]
    assert d["probe_n"] == 4 and d["probe_n_batches"] == 2
    for k in ("noop_floor_sd", "noop_floor_sem", "final_probe_sd",
              "final_probe_sem", "floor_ratio_sem"):
        assert d[k] is not None and d[k] >= 0.0, k
    assert d["floor_ratio"] == pytest.approx(
        d["final_probe_loss"] / max(d["noop_floor"], 1e-6))
    # the diagnostics' means ARE arm_probe_losses' estimator on the trained
    # object over the same examples — one metric, two entry points
    est = arm_probe_losses(model, tok, group[0].obj, examples, max_len=96,
                           batch_size=2, n_probe=None)
    assert d["final_probe_loss"] == pytest.approx(est.active)
    assert d["noop_floor"] == pytest.approx(est.noop)

    keys = set().union(*[set(r) for r in logged if isinstance(r, dict)])
    assert "lockstep/vector/heldout_loss" in keys
    assert "lockstep/vector/heldout_sem" in keys       # uncertainty, live
    assert "lockstep/vector/noop_floor" in keys
    assert "lockstep/vector/noop_floor_sem" in keys
    assert "lockstep/vector/final_probe_loss" in keys
    assert "lockstep/vector/floor_ratio" in keys
    assert "lockstep/vector/floor_ratio_sem" in keys
    assert "lockstep/vector/probe_n" in keys


def test_partition_by_compatibility_is_the_intended_use(tiny_llama):
    """A mixed arm list partitions cleanly: compatible ones share a teacher,
    incompatible ones fall to solo (the entry point's contract)."""
    arms = ["psr_kl_all", "psr_mse_all", "lora_r1", "loreft", "lofit",
            "token_bos", "alphakl_fisher"]
    compat = [a for a in arms if lockstep_compatible(a)]
    solo = [a for a in arms if not lockstep_compatible(a)]
    assert compat == ["psr_kl_all", "lora_r1", "token_bos"]
    assert solo == ["psr_mse_all", "loreft", "lofit", "alphakl_fisher"]


# --------------------------------------------------------------------------- #
# Production monitoring seam: probe_cb is purely observational, captures      #
# pre-clip vs post-clip, and its non-finite abort stops ONE arm while the     #
# group trains on. warmup_frac is the fraction-form warmup, resolved against  #
# THIS fit's own step budget.                                                 #
# --------------------------------------------------------------------------- #


def _fitmon_monitor(tok, examples, emit, arm_obj, *, arm="lora_r1_all",
                    model=None, cadence=None):
    from sisteer.extraction import fitmon

    probe = fitmon.KLRatioProbe(tok, examples, max_len=96, batch_size=2,
                                n=4, inject=arm_obj.inject)
    return fitmon.FitMonitor(
        arm, model, probe,
        cadence or fitmon.ProbeCadence(probe_every=1000, dense_every=1,
                                       dense_until=2, log_every=1),
        emit).attach(arm_obj)


def test_probe_cb_monitors_do_not_perturb_and_capture_clipping(tiny_llama):
    """Monitors ON vs OFF trains bitwise-identical objects (the monitors
    are observational: no RNG, no writes, hooks return None) — the
    'bitwise-untouched when monitors are off' contract holds in BOTH
    directions. And the emitted records separate pre-clip from post-clip
    when clipping bites (T9)."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    params = replace(default_lockstep_params("lora_r1_all"),
                     lr_schedule="warmup_cosine", warmup_steps=1,
                     max_grad_norm=1e-6)  # clip hard so pre != post

    def _train(probe_cb_on: bool):
        obj = build_lockstep_object("lora_r1_all", model, tok, INSTRUCTION,
                                    _pairs(), layer=1)
        emitted = []
        mon = _fitmon_monitor(tok, examples, 
                              lambda rec, step: emitted.append((step, rec)),
                              obj, model=model)

        def cb(step, arm_name, arm_obj, loss):
            mon.on_step(step, arm_obj, loss)

        train_lockstep(model, tok, [LockstepArm("lora_r1_all", obj, params)],
                       examples, probe_cb=(cb if probe_cb_on else None),
                       **TRAIN_KW)
        mon.close()
        return obj, emitted

    obj_on, emitted = _train(True)
    obj_off, off_emitted = _train(False)
    _assert_bitwise(obj_on, obj_off, arm="lora_r1_all")
    assert off_emitted == []          # monitors off: nothing observed
    recs = [r for _s, r in emitted if "train/grad_norm" in r]
    assert recs, "the monitor never captured a gradient step"
    for r in recs:
        assert r["train/grad_norm_postclip"] <= 1e-5 < r["train/grad_norm"], (
            "pre-clip and post-clip must separate when clipping bites")
    assert any("train/loss" in r for _s, r in emitted)
    assert any("probe/residual_ratio" in r for _s, r in emitted)
    assert any("norms/B_total" in r for _s, r in emitted)


def test_probe_cb_divergence_aborts_only_that_arm(tiny_llama):
    """A TrainingDiverged raise from the monitor stops THAT arm — recorded
    in its diagnostics as ``diverged`` with no final probe — while its
    group partner trains to the full budget, bitwise-identical to a run
    where the diverged arm never joined (per-arm optimiser/RNG isolation,
    the early-stopping argument)."""
    from sisteer.extraction.fitmon import TrainingDiverged

    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)

    def make(arm):
        return build_lockstep_object(arm, model, tok, INSTRUCTION, _pairs(),
                                     layer=1)

    def cb(step, arm_name, arm_obj, loss):
        if arm_name == "lora_r1" and step >= 1:
            raise TrainingDiverged(f"synthetic divergence at step {step}")

    diag: dict = {}
    doomed, partner = make("lora_r1"), make("lora_r1_all")
    hist = train_lockstep(
        model, tok,
        [LockstepArm("lora_r1", doomed,
                     default_lockstep_params("lora_r1")),
         LockstepArm("lora_r1_all", partner,
                     default_lockstep_params("lora_r1_all"))],
        examples, probe_cb=cb, diagnostics=diag, **TRAIN_KW)

    assert "synthetic divergence" in diag["lora_r1"]["diverged"]
    assert diag["lora_r1"]["final_probe_loss"] is None   # never probed final
    assert "diverged" not in diag["lora_r1_all"]
    # the doomed arm stopped after its step-1 abort; the partner ran on
    assert len(hist["lora_r1"]) < len(hist["lora_r1_all"])

    solo_partner = make("lora_r1_all")
    train_lockstep(model, tok,
                   [LockstepArm("lora_r1_all", solo_partner,
                                default_lockstep_params("lora_r1_all"))],
                   examples, **TRAIN_KW)
    _assert_bitwise(partner, solo_partner, arm="lora_r1_all")


def test_warmup_frac_resolves_against_this_fits_budget(tiny_llama):
    """warmup_frac -> steps happens INSIDE train_lockstep where total steps
    are known (winner fractions re-derive on every budget), and the
    resolution is recorded in the diagnostics."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    obj = build_lockstep_object("lora_r1_all", model, tok, INSTRUCTION,
                                _pairs(), layer=1)
    params = replace(default_lockstep_params("lora_r1_all"),
                     lr_schedule="warmup_cosine", warmup_frac=0.5)
    diag: dict = {}
    train_lockstep(model, tok, [LockstepArm("lora_r1_all", obj, params)],
                   examples, diagnostics=diag, **TRAIN_KW)
    # 4 examples, bs 2, 2 epochs -> 4 total steps; 0.5 -> 2 warmup steps
    assert diag["lora_r1_all"]["resolved_warmup_steps"] == 2
    # an unscheduled arm records None (no schedule ran)
    obj2 = build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                 _pairs(), layer=1)
    diag2: dict = {}
    train_lockstep(model, tok,
                   [LockstepArm("lora_r1", obj2,
                                default_lockstep_params("lora_r1"))],
                   examples, diagnostics=diag2, **TRAIN_KW)
    assert diag2["lora_r1"]["resolved_warmup_steps"] is None


def test_warmup_frac_validates_loudly(tiny_llama):
    """The 3dd9e36 mirror: a fraction without a schedule, next to a step
    count, or out of range refuses BEFORE any training."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    obj = build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                _pairs(), layer=1)

    def run(**over):
        params = replace(default_lockstep_params("lora_r1"), **over)
        train_lockstep(model, tok, [LockstepArm("lora_r1", obj, params)],
                       examples, **TRAIN_KW)

    with pytest.raises(ValueError, match="without an lr_schedule"):
        run(warmup_frac=0.1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        run(lr_schedule="warmup_cosine", warmup_frac=0.1, warmup_steps=3)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        run(lr_schedule="warmup_cosine", warmup_frac=1.5)
