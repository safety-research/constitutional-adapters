"""The mask→loss wiring: the SAME mask object
drives the steering hook and the loss positions.

The steering hook has supported a header-steered decode-only mask since
``hooks.response_header_len`` landed, but the distillation loss used to
gather EVERY completion position regardless of what the hook steered. If
the hook steers a subset while the loss sums over all completion
positions, the decode-only leg of the train x test 2x2 measures an
implementation artifact rather than a generalization gap — silently.

Pinned here, on fixed CPU fixtures (tiny_llama + synthetic French pairs,
never real store data):

1. loss positions == mask output for BOTH legs: the all-token leg's gather
   is exactly the completion mask, and the decode-only legs' gathers are
   exactly completion ∩ the handle's own ``_position_mask`` output —
   asserted end-to-end DURING training via instrumented recorders, so the
   equality covers the real call path, not a reimplementation;
2. header-steered (boundary = prompt_len - H) keeps every completion
   position by construction; PLAIN decode-only (boundary = prompt_len)
   drops exactly the first completion logit per row, on the student AND
   the aligned teacher gather;
3. the fail-loudly guards: a boundary misderived from the padded batch
   width raises, and an empty supervised∩steered row raises — a wrong mask
   must never silently shrink the loss;
4. the lockstep shared-teacher path honors the same mask: a decode-only
   vector arm trained in a lockstep group is BITWISE identical to the same
   leg trained solo through train_steered_lm (both header-steered and
   plain), and a mask-less arm (lora_r1) sharing the group is unperturbed;
5. arms whose objects expose no position mask REFUSE a decode-only fit
   (build_lockstep_object raises) instead of silently fitting all-token
   under a decode-only label.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import sisteer.extraction.distill as distill_mod
import sisteer.hooks as hooks_mod
from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.distill import (
    _encode_aligned,
    loss_gather_masks,
    steer_prompt_boundaries,
    train_steered_lm,
)
from sisteer.extraction.lockstep import (
    LockstepArm,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.extraction.traindata import collate
from sisteer.hooks import steering
from sisteer.methods import ARMS
from sisteer.models import hidden_size

INSTRUCTION = BY_ID["french"]
H = 2  # synthetic header allowance — the mask machinery is convention-agnostic
MAX_LEN = 96
# Steer layer 0 (NOT the tiny model's last layer): with no attention above
# the steered layer, prompt-position steering could never reach completion
# logits and the legs would coincide by architecture, not by bug.
LAYER = 0


def _pairs():
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _batches(tok):
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    encoded = [_encode_aligned(tok, ex, MAX_LEN) for ex in examples]
    teacher_b = collate([t for t, _ in encoded], tok.pad_token_id)
    student_b = collate([s for _, s in encoded], tok.pad_token_id)
    return examples, teacher_b, student_b


def _completion_mask(batch):
    return batch.labels[:, 1:] != -100


# ---------------------------------------------------------------------------
# the helper, against a LIVE handle (mask consumed from hooks._position_mask)
# ---------------------------------------------------------------------------


def _live_masks(model, student_b, teacher_b, *, positions, rhl,
                boundaries=None):
    vec = torch.zeros(hidden_size(model))
    with steering(model, {LAYER: vec}, positions=positions, skip_bos=True,
                  response_header_len=rhl) as handle:
        if positions in ("response", "prompt"):
            bounds = (boundaries if boundaries is not None
                      else steer_prompt_boundaries(student_b, rhl))
            # peer-pinned width <= H trap: a collapsed (0) boundary steers
            # every position and the leg silently tests nothing.
            assert bool((bounds > 0).all()), (
                "fixture prompt at or below H — boundary collapsed to 0")
            handle.set_prompt_lens(bounds)
        with torch.no_grad():
            model(input_ids=student_b.input_ids,
                  attention_mask=student_b.attention_mask)
        return loss_gather_masks(handle, student_b, teacher_b)


def test_all_token_leg_gathers_every_completion_position(tiny_llama):
    model, tok = tiny_llama
    _, teacher_b, student_b = _batches(tok)
    s_gather, t_gather, keep = _live_masks(
        model, student_b, teacher_b, positions="all", rhl=None)
    assert torch.equal(s_gather, _completion_mask(student_b))
    assert torch.equal(t_gather, _completion_mask(teacher_b))
    assert keep is None  # nothing dropped -> bit-identical default gathers


def test_header_steered_leg_keeps_every_completion_position(tiny_llama):
    """boundary = prompt_len - H covers the position that predicts the
    FIRST completion token (P-1 >= P-H), so header-steered supervision is
    the full completion set — while the hook mask itself turns the last H
    prompt positions ON and the rest of the prompt OFF."""
    model, tok = tiny_llama
    _, teacher_b, student_b = _batches(tok)
    s_gather, t_gather, keep = _live_masks(
        model, student_b, teacher_b, positions="response", rhl=H)
    assert torch.equal(s_gather, _completion_mask(student_b))
    assert torch.equal(t_gather, _completion_mask(teacher_b))
    assert keep is None


def test_plain_decode_only_drops_exactly_the_first_completion_logit(
        tiny_llama):
    """boundary = prompt_len leaves the logit at P-1 (the one that picks
    the first completion token) UNSTEERED, so the loss must not read it —
    on the student and, index-aligned, on the teacher."""
    model, tok = tiny_llama
    _, teacher_b, student_b = _batches(tok)
    s_gather, t_gather, keep = _live_masks(
        model, student_b, teacher_b, positions="response", rhl=None)
    s_keep = _completion_mask(student_b)
    t_keep = _completion_mask(teacher_b)
    assert keep is not None and not bool(keep.all())
    for i in range(student_b.input_ids.shape[0]):
        p_s = int(student_b.prompt_lens[i])
        p_t = int(teacher_b.prompt_lens[i])
        # exactly one position dropped per row: the first completion logit
        assert int(s_gather[i].sum()) == int(s_keep[i].sum()) - 1
        assert bool(s_keep[i, p_s - 1]) and not bool(s_gather[i, p_s - 1])
        assert int(t_gather[i].sum()) == int(t_keep[i].sum()) - 1
        assert bool(t_keep[i, p_t - 1]) and not bool(t_gather[i, p_t - 1])
    # aligned counts: student and teacher keep the SAME completion indices
    assert int(s_gather.sum()) == int(t_gather.sum())


def test_misderived_boundary_raises_loudly(tiny_llama):
    """The padded-batch-width trap: deriving the boundary from the batch
    width (seq_len - H) instead of each row's prompt_len - H steers only
    the last columns of the padded batch. The wiring must raise, never
    train on the wrong support."""
    model, tok = tiny_llama
    _, teacher_b, student_b = _batches(tok)
    full_width = student_b.input_ids.shape[1]
    bad = torch.full_like(student_b.prompt_lens, full_width - H)
    with pytest.raises(ValueError, match="mask→loss wiring"):
        _live_masks(model, student_b, teacher_b, positions="response",
                    rhl=H, boundaries=bad)


# ---------------------------------------------------------------------------
# end-to-end: loss positions == mask output DURING real training, both legs
# ---------------------------------------------------------------------------


def _instrument(monkeypatch):
    """Record every mask the hook computes and every (s_gather, keep) the
    loss wiring derives, so the equality is asserted on the REAL training
    call path."""
    hook_masks: list = []
    orig_pm = hooks_mod.SteeringHandle._position_mask

    def rec_pm(self, batch, seq_len, device):
        m = orig_pm(self, batch, seq_len, device)
        hook_masks.append(None if m is None else m.detach().clone())
        return m

    monkeypatch.setattr(hooks_mod.SteeringHandle, "_position_mask", rec_pm)

    records: list = []
    orig_lgm = distill_mod.loss_gather_masks

    def rec_lgm(handle, student_b, teacher_b=None):
        out = orig_lgm(handle, student_b, teacher_b)
        # hook_masks[-1] is the loss wiring's own _position_mask call (made
        # inside orig_lgm); hook_masks[-2] is the mask the steering hook
        # used during the forward that just ran, on the same handle state.
        records.append((hook_masks[-2], hook_masks[-1], out[0],
                        _completion_mask(student_b)))
        return out

    monkeypatch.setattr(distill_mod, "loss_gather_masks", rec_lgm)
    return records


@pytest.mark.parametrize("positions,rhl", [("all", None), ("response", H)])
def test_training_loss_positions_equal_hook_mask_both_legs(
        tiny_llama, monkeypatch, positions, rhl):
    """The task's gate: for the all-token AND the decode-only leg, every
    training step's loss gather equals (hook mask ∩ completion mask), with
    the hook-side and loss-side masks bit-identical."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    records = _instrument(monkeypatch)
    vec = nn.Parameter(torch.zeros(hidden_size(model)))
    train_steered_lm(
        model, tok, examples, LAYER,
        vector_provider=lambda idxs: vec, trainable_params=[vec],
        loss="soft", epochs=1, batch_size=2, max_len=MAX_LEN, seed=0,
        positions=positions, response_header_len=rhl)
    assert records, "the instrumented wiring never ran"
    for hook_mask, loss_mask, s_gather, completion in records:
        assert hook_mask is not None  # skip_bos=True -> a mask exists
        assert torch.equal(hook_mask, loss_mask), (
            "hook-side and loss-side masks diverged on the same handle")
        expected = completion & hook_mask[:, :-1].bool()
        assert torch.equal(s_gather, expected), (
            "loss positions != mask output — the loss is reading positions "
            "the hook did not steer (or vice versa)")


def test_decode_only_training_masks_the_prompt_and_differs_from_all_token(
        tiny_llama, monkeypatch):
    """Semantic sanity for the leg itself: during a decode-only fit the
    hook mask turns each row's prompt (before prompt_len - H) OFF — the
    boundary comes from the per-row prompt lengths, never the padded batch
    width — and the trained vector differs from the all-token leg's."""
    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    records = _instrument(monkeypatch)

    def run(positions, rhl):
        vec = nn.Parameter(torch.zeros(hidden_size(model)))
        train_steered_lm(
            model, tok, examples, LAYER,
            vector_provider=lambda idxs: vec, trainable_params=[vec],
            loss="soft", epochs=2, batch_size=2, max_len=MAX_LEN, seed=0,
            positions=positions, response_header_len=rhl)
        return vec.detach().clone()

    v_hdr = run("response", H)
    hdr_records = list(records)
    v_all = run("all", None)
    assert not torch.equal(v_hdr, v_all)
    # per-row boundary check on the recorded decode-only masks: everything
    # before prompt_len - H is off, everything from it on is on
    examples_enc = [_encode_aligned(tok, ex, MAX_LEN) for ex in examples]
    prompt_lens = [e.prompt_len for _, e in examples_enc]
    checked = 0
    for hook_mask, _lm, _sg, _cm in hdr_records:
        b, s = hook_mask.shape
        # match this batch's rows to their prompt lens by batch width
        for i in range(b):
            row = hook_mask[i]
            on = row.nonzero().flatten()
            assert len(on) > 0
            boundary = int(on[0])
            assert boundary in [max(p - H, 0) for p in prompt_lens]
            assert row[boundary:].eq(1).all()
            assert row[:boundary].eq(0).all()
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# lockstep: the shared-teacher path under the same mask, bitwise vs solo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rhl", [H, None])
def test_lockstep_decode_only_vector_bitwise_vs_solo(tiny_llama, rhl):
    """A decode-only (header-steered AND plain) vector arm trained in a
    lockstep group — shared teacher forward, mask-aligned sub-selection —
    is bitwise identical to the same leg trained solo, and the mask-less
    lora_r1 sharing the group is bitwise identical to its own solo
    registry fit (a masked partner cannot perturb it)."""
    model, tok = tiny_llama
    pairs = _pairs()
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    kw = dict(epochs=2, batch_size=2, max_len=MAX_LEN, seed=0)

    vec = nn.Parameter(torch.zeros(hidden_size(model)))
    train_steered_lm(
        model, tok, examples, 1,
        vector_provider=lambda idxs: vec, trainable_params=[vec],
        loss="soft", positions="response", response_header_len=rhl, **kw)

    group = [
        LockstepArm("vector",
                    build_lockstep_object("vector", model, tok, INSTRUCTION,
                                          pairs, layer=1,
                                          positions="response",
                                          response_header_len=rhl),
                    default_lockstep_params("vector")),
        LockstepArm("lora_r1",
                    build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                          pairs, layer=1),
                    default_lockstep_params("lora_r1")),
    ]
    train_lockstep(model, tok, group, examples, **kw)
    assert torch.equal(group[0].obj.trainable[0].detach(), vec.detach()), (
        "decode-only lockstep vector != decode-only solo vector")

    solo_lora = ARMS["lora_r1"].train(model, tok, INSTRUCTION, pairs,
                                      layer=1, **kw)
    for i, (a, b) in enumerate(zip(group[1].obj.trainable,
                                   solo_lora.trainable)):
        assert torch.equal(a.detach(), b.detach()), (
            f"lora_r1 param {i} perturbed by a masked group partner")


def test_maskless_arms_refuse_decode_only_fits(tiny_llama):
    """No position mask, no decode-only fit: a silent all-token fit under a
    decode-only label is exactly the artifact the wiring prevents. (lora_r1
    and the psr_kl arms LEFT this list when they gained PositionMaskedFit
    support, and vector_all / lora_r1_all left it on 2026-08-20 the same
    way — tests/test_arm_position_masks.py covers all their masked fits.
    The soft-token arms refuse FOREVER: a soft PROMPT token has no
    generated-position notion.)"""
    model, tok = tiny_llama
    for arm in ("token_bos", "token_faithful"):
        with pytest.raises(ValueError, match="cannot honor"):
            build_lockstep_object(
                arm, model, tok, INSTRUCTION, _pairs(), layer=1,
                positions="response", response_header_len=H,
                sentinel=(tok.unk_token, tok.unk_token_id))
