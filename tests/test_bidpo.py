"""bidpo (the BiPO steering-vector arm): train/deploy parity + masked fits.

Pins the two 2026-08-21 defect fixes (synthetic placeholder data ONLY —
invented prompts/completions, never recorded session or pool text):

1. TRAIN/DEPLOY BOS PARITY: methods.load_object serves every plain
   (non-antidote) SteeringVector as a VectorObject, whose default is
   skip_bos=True — so train_bidpo's steered forwards must skip the BOS
   attention sink too. Before the fix, bidpo alone among the vector arms
   optimised its vector while perturbing BOS (very high norm in Llama-3)
   and then deployed where that perturbation is absent (the raw
   hooks.steering default, skip_bos=False; distill/ladder plumb the flag,
   lockstep hardcodes it, every other vector arm fits through
   VectorObject).
2. THE positions KWARG WAS INERT: train_bidpo accepted positions="response"
   but never called set_prompt_lens, so the handle's boundary stayed 0 and
   "response" silently meant ALL positions — and it did not accept
   response_header_len at all, so the two-SI driver (which passes both as
   PROTECTED kwargs) failed the arm with a TypeError. Now wired exactly
   like bidpo_lora's engine (train_object_kl's shape): per-row boundaries
   on the live handle before each policy forward, and BOTH
   completion-logprob sums of each DPO margin term — policy and frozen
   reference — gathered from that same mask.

Also pinned: the default all-token leg keeps the full completion loss
support (every gather None) and stays deterministic; its step-for-step
equivalence to bidpo_lora's engine (same losses, same artifact) is the
drift alarm in test_bidpo_lora.py. The FIT leg travels in the artifact's
sidecar metadata as provenance, while the DEPLOY leg stays the eval cell's
choice (the train x test 2x2; jailbreak._set_response_positions) — LoReFT
alone persists positions INTO the object, because there they are part of
what the method IS.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import pytest
import torch

from sisteer.data.instructions import Instruction
from sisteer.data.pairs import Pair
from sisteer.extraction.bidpo import PrefExample, train_bidpo
from sisteer.extraction.traindata import collate, encode_prompt_completion
from sisteer.hooks import steering
from sisteer.methods import ARMS, VectorObject, load_object, object_artifact_name
from sisteer.models import get_decoder_layers, hidden_size

LAYER = 1
H = 2  # synthetic header allowance — the machinery is convention-agnostic

INSTRUCTION = Instruction(
    id="testsi", category="style", text="Always answer in French.",
    anti_text=None, checker="contains")


def _pairs():
    return [
        Pair(prompt="Say something nice.", pos="Bonjour le monde",
             neg="Hello world", instruction_id="testsi"),
        Pair(prompt="Describe a tree.", pos="C'est un grand arbre",
             neg="It is a tall tree", instruction_id="testsi"),
    ]


def _prefs():
    return [
        PrefExample(prompt="Say something nice.\n",
                    chosen="Bonjour le monde", rejected="Hello world"),
        PrefExample(prompt="Describe a tree.\n",
                    chosen="C'est un grand arbre", rejected="It is a tall tree"),
        PrefExample(prompt="Name a color.\n",
                    chosen="Le ciel est bleu", rejected="The sky is blue"),
    ]


# -- 1. the training forward runs under the artifact's deploy convention ---------


def test_training_forward_skips_bos_matching_deployment(tiny_llama, monkeypatch):
    """Every policy forward's steering mask excludes exactly the BOS
    position and steers everything else — the SAME exclusion set the
    deployed artifact (VectorObject, skip_bos=True) applies. Against the
    pre-fix code every recorded mask is None (skip_bos never passed, BOS
    steered during training, unsteered at deploy)."""
    import sisteer.extraction.bidpo as bp

    model, tok = tiny_llama
    masks = []
    real = bp.steering

    @contextmanager
    def spy(mdl, sv, **kw):
        with real(mdl, sv, **kw) as h:
            yield h
            am = h._attn_mask  # set by the handle's tracker during the forward
            if am is None:
                masks.append(None)  # no tracker ran: BOS was steered
            else:
                masks.append(h._position_mask(am.shape[0], am.shape[1],
                                              am.device))

    monkeypatch.setattr(bp, "steering", spy)
    sv = train_bidpo(model, tok, _prefs(), LAYER,
                     epochs=1, batch_size=2, max_len=64, seed=0)
    # 3 examples at batch 2 -> 2 steps x 4 policy forwards (chosen/rejected
    # under +1/-1)
    assert len(masks) == 8
    for m in masks:
        assert m is not None, (
            "the training forward steered EVERY position, BOS included — "
            "deployment (VectorObject) runs skip_bos=True, so the vector "
            "was optimised against a BOS-sink perturbation that is absent "
            "at deploy (the pre-2026-08-21 train/test mismatch)")
        mb = m.bool()
        assert not mb[:, 0].any()  # BOS (right-padded col 0) unsteered
        assert mb[:, 1:].all()     # every other position steered
    assert sv.metadata["skip_bos"] is True

    # ...and the deployed artifact's own forward computes the SAME mask
    obj = VectorObject(sv)  # what load_object hydrates for method != antidote
    seqs = [encode_prompt_completion(tok, p.prompt, p.chosen, 64)
            for p in _prefs()[:2]]
    batch = collate(seqs, tok.pad_token_id)
    with torch.no_grad():
        with obj.apply(model) as h:
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
            dm = h._position_mask(batch.input_ids.shape[0],
                                  batch.input_ids.shape[1],
                                  batch.input_ids.device)
    assert dm is not None
    dmb = dm.bool()
    assert not dmb[:, 0].any() and dmb[:, 1:].all()


# -- 2. positions="response" genuinely restricts the edit ------------------------


def test_masked_fit_restricts_the_edit_during_training(tiny_llama, monkeypatch):
    """During a REAL positions='response' training run, every policy
    forward's edit is active at exactly the handle's mask: each row keeps
    its prompt prefix unsteered (NOT silently all-positions — the pre-fix
    degeneration this test fails on: no set_prompt_lens call ever ran, the
    boundary stayed 0 and the mask covered everything), masked-out
    positions are bit-identical to the unsteered model, and every steered
    real position is edited once the vector is nonzero. The base/steered
    pair comes from the SAME forward: a capture hook registered before the
    steering hook sees the pre-addition layer output, one registered after
    sees the post-addition output."""
    import sisteer.extraction.bidpo as bp

    model, tok = tiny_llama
    layer_mod = get_decoder_layers(model)[LAYER]
    records = []
    real = bp.steering

    @contextmanager
    def spy(mdl, sv, **kw):
        rec = {}

        def grab(_m, args, kwargs):
            rec["input_ids"] = kwargs.get("input_ids",
                                          args[0] if args else None)
            rec["attn"] = kwargs.get("attention_mask")
            return None

        def cap(key):
            def hook(_m, _a, out):
                h_ = out[0] if isinstance(out, tuple) else out
                rec[key] = h_.detach().clone()
            return hook

        ph = model.register_forward_pre_hook(grab, with_kwargs=True)
        h_base = layer_mod.register_forward_hook(cap("base"))
        try:
            with real(mdl, sv, **kw) as h:
                h_steer = layer_mod.register_forward_hook(cap("steered"))
                try:
                    yield h
                finally:
                    h_steer.remove()
                b, s = rec["input_ids"].shape
                m = h._position_mask(b, s, rec["input_ids"].device)
                records.append(
                    (m, rec["attn"], rec["base"], rec["steered"],
                     float(next(iter(sv.values())).detach().norm())))
        finally:
            h_base.remove()
            ph.remove()

    monkeypatch.setattr(bp, "steering", spy)
    sv = train_bidpo(model, tok, _prefs(), LAYER, epochs=1, lr=1e-2,
                     batch_size=2, max_len=64, seed=0, positions="response")
    assert sv.metadata["positions"] == "response"
    assert len(records) == 8
    nonzero_forwards = 0
    for m, attn, base, steered, vnorm in records:
        assert m is not None
        mb = m.bool()
        real_pos = attn.bool()
        off_real = ~mb & real_pos
        assert bool(off_real.any(dim=1).all()), (
            "positions='response' steered every real position — the per-row "
            "boundary was never set on the handle and the mask silently "
            "degenerated to all-positions (the pre-2026-08-21 inert-kwarg "
            "defect)")
        # masked-out positions: bit-identical to the unsteered model
        assert torch.equal(steered[~mb], base[~mb]), (
            "a masked-out position was not identical to the unsteered model")
        if vnorm > 0:
            diff = (steered != base).any(dim=-1)
            assert bool(diff[mb & real_pos].all()), (
                "a steered position was left unedited")
            nonzero_forwards += 1
    assert nonzero_forwards  # non-vacuous: forwards ran with a trained vector


def test_response_fit_differs_from_all_token(tiny_llama):
    """The mask actually reaches the training forwards: a decode-only fit
    trains a DIFFERENT vector from the all-token fit at the same seed.

    Fitted at layer 0, NOT the usual LAYER=1, and necessarily so (the same
    rationale as bidpo_lora's twin test): layer 1 is the tiny fixture's
    FINAL decoder layer, where a prompt-position residual addition is
    position-local (no downstream attention follows) and can never reach a
    completion logit — and under the header-steered convention both fits
    read the identical completion support — so the masked and all-token
    fits coincide EXACTLY there and the test would pass while testing
    nothing. Layer 0's prompt-position additions propagate through layer
    1's attention into the completion logits, which is precisely the
    leakage channel decode-only masking exists to close."""
    model, tok = tiny_llama
    prefs = _prefs()
    kw = dict(epochs=1, batch_size=2, max_len=64, seed=0)
    masked = train_bidpo(model, tok, prefs, 0,
                         positions="response", response_header_len=H, **kw)
    plain = train_bidpo(model, tok, prefs, 0, **kw)
    assert masked.vectors[0].norm() > 0 and plain.vectors[0].norm() > 0
    assert not torch.equal(masked.vectors[0], plain.vectors[0]), (
        "the masked bidpo fit is indistinguishable from all-token — the "
        "mask never reached the training forward")


def test_non_fit_position_modes_are_refused(tiny_llama):
    """The fit vocabulary is hooks.FIT_POSITION_MODES ('all'/'response') —
    the serving-only modes must refuse loudly up front, never pass through
    to an inert (boundary-less) steering call."""
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="fit positions"):
        train_bidpo(model, tok, _prefs(), LAYER, positions="prompt",
                    epochs=1, batch_size=2, max_len=64)


# -- 3. response_header_len is accepted and threaded (mask→loss wiring) ----------


def test_masked_fit_ties_dpo_loss_to_the_hook_mask(tiny_llama, monkeypatch):
    """Every completion-logprob gather of a masked fit — the frozen
    reference's (via the transient vectorless tracker) AND the policy's
    (via the live handle) — equals (handle mask ∩ completion mask), and the
    mask is the per-row RESPONSE mask with the header-steered boundary
    (prompt_len - H): response_header_len is threaded, not just accepted.
    The unset-boundary trap (prompt_len left at 0) degenerates to an
    all-positions mask, which the expected-mask equality catches."""
    import sisteer.extraction.bidpo as bp

    model, tok = tiny_llama
    records = []
    orig = bp.loss_gather_masks

    def rec(handle, batch, teacher_b=None):
        out = orig(handle, batch, teacher_b)
        b, s = batch.input_ids.shape
        m = handle._position_mask(b, s, batch.input_ids.device).bool()
        records.append((m, out[0].clone(), batch))
        return out

    monkeypatch.setattr(bp, "loss_gather_masks", rec)
    train_bidpo(model, tok, _prefs(), LAYER, epochs=1, batch_size=2,
                max_len=64, seed=0, positions="response",
                response_header_len=H)
    # 3 examples at batch 2 -> 2 batches: 4 reference gathers (chosen +
    # rejected per batch) + 2 steps x 4 policy forwards (chosen/rejected
    # under +1/-1) = 12 — BOTH ends of the margin are tied to the mask.
    assert len(records) == 12
    for m, s_gather, batch in records:
        completion = batch.labels[:, 1:] != -100
        assert torch.equal(s_gather, completion & m[:, :-1]), (
            "loss positions != hook mask ∩ completion — the loss read "
            "positions the hook did not steer (or vice versa)")
        assert bool((batch.prompt_lens > H).all())  # non-vacuous boundary
        local = torch.arange(batch.input_ids.shape[1])[None, :]
        expected = local >= (batch.prompt_lens - H).clamp(min=0)[:, None]
        assert torch.equal(m, expected), (
            "the handle's mask is not the per-row header-steered response "
            "mask — the boundary was never set (or misderived)")


def test_masked_fit_trains_through_the_registry(tiny_llama, tmp_path):
    """The two-SI driver hands positions/response_header_len to the solo
    trainer as PROTECTED kwargs (never dropped in the TypeError-retry) —
    the registry trainer must accept them (it used to TypeError, failing
    the arm) and record the fit leg in the artifact's metadata. The
    round-trip pins the deploy convention: the fit leg travels as sidecar
    provenance while hydration stays all-position/skip_bos=True like every
    plain vector — the DEPLOY leg is the eval cell's choice
    (jailbreak._set_response_positions), not the artifact's."""
    model, tok = tiny_llama
    obj = ARMS["bidpo"].train(
        model, tok, INSTRUCTION, _pairs(), layer=LAYER, epochs=1,
        batch_size=2, max_len=64, seed=0, positions="response",
        response_header_len=H)
    assert type(obj) is VectorObject
    sv = obj.sv
    assert sv.method == "bidpo"
    assert sv.metadata["positions"] == "response"
    assert sv.metadata["response_header_len"] == H
    assert sv.metadata["skip_bos"] is True
    assert sv.vectors[LAYER].norm() > 0

    path = tmp_path / object_artifact_name("bidpo")  # object.safetensors
    obj.save(path)
    back = load_object(path, model=model, tok=tok)
    assert type(back) is VectorObject
    assert torch.equal(back.sv.vectors[LAYER], sv.vectors[LAYER])
    assert back.sv.metadata["positions"] == "response"
    assert back.sv.metadata["response_header_len"] == H
    assert back.positions == "all" and back.skip_bos is True


# -- 4. the default all-token path stays pinned -----------------------------------


def test_default_fit_keeps_the_full_completion_loss_support(
        tiny_llama, monkeypatch):
    """The unmasked path: no gather is ever derived (None = the historical
    completion sum, bit-for-bit) and the fit is deterministic under a fixed
    seed. Its step-for-step equivalence to bidpo_lora's engine — identical
    per-step losses and artifact — is the drift alarm in
    test_bidpo_lora.py."""
    import sisteer.extraction.bidpo as bp

    model, tok = tiny_llama
    calls = []
    orig = bp.loss_gather_masks

    def rec(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(bp, "loss_gather_masks", rec)
    kw = dict(epochs=1, batch_size=2, max_len=64, seed=5)
    sv1 = train_bidpo(model, tok, _prefs(), LAYER, **kw)
    assert not calls, (
        "the default all-token fit derived a loss gather — its loss "
        "support must stay the historical full completion sum")
    sv2 = train_bidpo(model, tok, _prefs(), LAYER, **kw)
    assert torch.equal(sv1.vectors[LAYER], sv2.vectors[LAYER])


# -- 5. saturation-keyed early stopping (the 2026-08-24 wave post-mortem fix) ----
#
# On the 6-epoch wave the small preference pools saturated the DPO
# preference (epoch-mean beta*margin ~3) by epoch 1-2, spent the remaining
# budget inflating the margin to ~6, and failed floor_ratio_check at
# 14-15x; the one pool that only reached saturation at the final epoch
# passed at 2.7x. stop_pref_prob keys the stop on the exact event "the
# modeled preference reached probability p in both directions".


def test_stop_pref_prob_stops_at_preference_saturation(tiny_llama):
    """The rule TRUNCATES the trajectory, never alters it: with the stop
    armed, the per-step losses are a strict prefix of the fixed-budget
    run's, metadata records the truncation point, and the artifact is
    bit-equal to training exactly that many epochs. beta=0.5 / p=0.51 puts
    the margin target at logit(0.51)/0.5 ~ 0.08, which the tiny fixture's
    margins cross within a few epochs — the machinery is under test here;
    the wave-calibrated threshold (p ~ 0.95) is the config's decision."""
    model, tok = tiny_llama
    prefs = _prefs()
    kw = dict(epochs=8, lr=1e-2, beta=0.5, batch_size=2, max_len=64, seed=0)

    full_losses: list[float] = []
    full = train_bidpo(
        model, tok, prefs, LAYER,
        log_fn=lambda m, s: full_losses.append(m["bidpo/loss"])
        if "bidpo/loss" in m else None, **kw)
    assert len(full_losses) == 16  # 3 examples at batch 2 -> 2 steps/epoch
    assert full.metadata["epochs_run"] == 8
    assert full.metadata["stop_pref_prob"] is None

    stopped_losses: list[float] = []
    sv = train_bidpo(
        model, tok, prefs, LAYER, stop_pref_prob=0.51,
        log_fn=lambda m, s: stopped_losses.append(m["bidpo/loss"])
        if "bidpo/loss" in m else None, **kw)
    k = sv.metadata["epochs_run"]
    assert sv.metadata["stop_pref_prob"] == 0.51
    assert 0 < k < 8, "the stop never fired — the test is vacuous"
    assert len(stopped_losses) == 2 * k
    assert stopped_losses == full_losses[: 2 * k], (
        "the stop rule changed the training trajectory — it may only cut "
        "it short")

    exact = train_bidpo(model, tok, prefs, LAYER, **{**kw, "epochs": k})
    assert torch.equal(sv.vectors[LAYER], exact.vectors[LAYER]), (
        "the stopped artifact is not the k-epoch artifact — stopping must "
        "keep the stopping epoch's weights, nothing else")


def test_stop_pref_prob_requires_both_directions_saturated(tiny_llama):
    """BiPO's objective is bidirectional, so one saturated direction alone
    is not 'learned': at p=0.6 the tiny fixture's REVERSE margin clears the
    target while the forward one never does, and the run must use its full
    budget. Non-vacuity is asserted from the logged margins, not assumed."""
    model, tok = tiny_llama
    rows: list[dict] = []
    sv = train_bidpo(model, tok, _prefs(), LAYER, stop_pref_prob=0.6,
                     epochs=8, lr=1e-2, beta=0.5, batch_size=2, max_len=64,
                     seed=0, log_fn=lambda m, s: rows.append(dict(m)))
    target = math.log(0.6 / 0.4) / 0.5
    by_epoch: dict[int, list[dict]] = {}
    for r in rows:
        if "bidpo/fwd_margin" in r:
            by_epoch.setdefault(r["epoch"], []).append(r)
    means = {
        e: (sum(x["bidpo/fwd_margin"] for x in xs) / len(xs),
            sum(x["bidpo/rev_margin"] for x in xs) / len(xs))
        for e, xs in by_epoch.items()
    }
    one_sided = [e for e, (f, r) in means.items()
                 if (f >= target) != (r >= target)]
    assert one_sided, (
        "no epoch saturated exactly one direction — the fixture's dynamics "
        "moved and this test no longer exercises the both-directions gate")
    assert sv.metadata["epochs_run"] == 8, (
        "a single saturated direction stopped the run — the gate must "
        "require BOTH")


def test_stop_pref_prob_validated(tiny_llama):
    """p outside (0.5, 1) is meaningless (<=0.5 stops before anything is
    learned; 1.0 is an infinite margin target) and must refuse loudly,
    before any training compute."""
    model, tok = tiny_llama
    for bad in (0.0, 0.5, 1.0, 1.5):
        with pytest.raises(ValueError, match="stop_pref_prob"):
            train_bidpo(model, tok, _prefs(), LAYER, stop_pref_prob=bad,
                        epochs=1, batch_size=2, max_len=64)


# -- 6. weight decay actually brakes the vector norm (2026-08-24 post-mortem) ----
#
# The wave's failing runs optimized the DPO loss monotonically (0.577 ->
# 0.123, epoch means) while bidpo/vec_norm grew 1.11 -> 3.44 with no
# plateau, and the artifact failed floor_ratio_check at 14.3x the no-op
# floor — pure over-optimization of an UNBOUNDED parameter norm, not
# divergence. The weight_decay knob predates this (BiPO's recipe, 0.05,
# always explicit in the signature) but at lr 5e-4 its decay time constant
# is 1/(lr*wd) = 40,000 steps, ~35x those runs, i.e. inert; the default is
# now 2.0 (train_bidpo's docstring derives the value from the wave's
# measured norm velocity). These tests pin the MECHANISM, not the wave's
# numbers: decay genuinely bounds the norm, its target (v = 0) is verified
# to be the EXACT no-op, and the historical recipe stays reachable at
# weight_decay=0.05.


def test_weight_decay_bounds_the_vector_norm(tiny_llama):
    """Same fit, same seed, only the decay differs: the decayed vector must
    end at a strictly smaller norm while still training off the no-op
    (both norms nonzero — the brake shrinks the object, it does not zero
    the fit). The per-step bidpo/vec_norm log is the instrument the next
    wave reads to SEE the bound, so it must carry the same verdict as the
    artifact."""
    model, tok = tiny_llama
    prefs = _prefs()
    kw = dict(epochs=6, lr=1e-2, batch_size=2, max_len=64, seed=0)

    def fit(wd):
        traj: list[float] = []
        sv = train_bidpo(model, tok, prefs, LAYER, weight_decay=wd,
                         log_fn=lambda m, s: traj.append(m["bidpo/vec_norm"])
                         if "bidpo/vec_norm" in m else None, **kw)
        assert sv.metadata["weight_decay"] == wd
        assert traj, "bidpo/vec_norm was not logged — the next wave is blind"
        return sv.vectors[LAYER].norm(), traj

    free_norm, free_traj = fit(0.0)
    decayed_norm, decayed_traj = fit(5.0)
    assert free_norm > 0 and decayed_norm > 0
    assert decayed_norm < free_norm, (
        "weight decay did not bound the vector norm — the wave's runaway "
        "(1.11 -> 3.44, no plateau) would recur")
    assert decayed_traj[-1] < free_traj[-1]


def test_weight_decay_pulls_toward_the_exact_noop(tiny_llama):
    """What decay shrinks toward must be "do nothing": the ZERO vector's
    steered forward is BIT-identical to the unsteered model (same
    positions/skip_bos flags the training forward runs under), so v = 0 —
    decay's fixed point — is exactly the no-op. That fact is what makes
    decay the right brake for THIS arm; loreft's no-op is the nonzero
    rotation R, which is why reft.py pins wd=0 there (an anti-identity
    pull, not a brake)."""
    model, tok = tiny_llama
    seqs = [encode_prompt_completion(tok, p.prompt, p.chosen, 64)
            for p in _prefs()]
    batch = collate(seqs, tok.pad_token_id)
    zero = torch.zeros(hidden_size(model))
    with torch.no_grad():
        base = model(input_ids=batch.input_ids,
                     attention_mask=batch.attention_mask).logits
        with steering(model, {LAYER: zero}, positions="all", skip_bos=True):
            steered = model(input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask).logits
    assert torch.equal(steered, base), (
        "v = 0 is not the exact no-op — decay would pull toward something "
        "other than the unsteered model (the loreft trap)")


def test_weight_decay_default_changed_loudly_old_recipe_pinned(tiny_llama):
    """The effective default moved 0.05 (BiPO's recipe — explicit in this
    signature since the arm landed, NOT torch's silent 0.01) -> 2.0
    (2026-08-24). Pins: the bare default IS explicit 2.0 bitwise, the
    historical recipe stays reachable at weight_decay=0.05, and the change
    is real (0.05 and 2.0 diverge at the same seed). Bitwise agreement of
    the explicit-0.05 run with the PRE-change trainer was verified at
    change time: sha256 over the trained vector + per-step losses + the
    LoRA twin's A/B, identical across the default edit."""
    model, tok = tiny_llama
    prefs = _prefs()
    kw = dict(epochs=2, lr=1e-2, batch_size=2, max_len=64, seed=3)
    bare = train_bidpo(model, tok, prefs, LAYER, **kw)
    explicit = train_bidpo(model, tok, prefs, LAYER, weight_decay=2.0, **kw)
    old = train_bidpo(model, tok, prefs, LAYER, weight_decay=0.05, **kw)
    assert bare.metadata["weight_decay"] == 2.0
    assert torch.equal(bare.vectors[LAYER], explicit.vectors[LAYER])
    assert not torch.equal(bare.vectors[LAYER], old.vectors[LAYER]), (
        "0.05 and 2.0 trained identical vectors — the weight_decay kwarg "
        "is not reaching the optimizer")
