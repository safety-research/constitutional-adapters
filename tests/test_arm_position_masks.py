"""Position-masked (decode-only) fits for the Class A compilation arms.

The two-SI decode-only comparison leg ("steer only the generated tokens")
used to be honorable only by the hooks.steering-backed vector arms; every
other Class A arm refused. This file pins the extension of that support
(hooks.PositionMaskedFit — a vectorless SteeringHandle as the ONE mask
source) to the arms where a mask is exact, and the refusals where it is
not:

MASKABLE (per-position hook interventions; masked positions must be
BIT-identical to the unsteered model, steered positions exactly the
handle's mask):
- PSRObject (psr_kl_all / psr_mse_all / psr_kl):  h + g(h)·d at the
  decoder-layer output;
- RankOneLoRAObject (lora_r1): (x·A)·B on the o_proj OUTPUT — o_proj is
  position-wise linear, so masking the delta == the weight edit active at
  those positions;
- LoFiTObject (lofit): per-head bias on the PRE-o_proj attention output —
  post-attention-mix, so position t's addition reaches only position t's
  o_proj output; phase-A scalers mask the same way;
- LoReFTObject (loreft): per-position residual affine edit. Its default
  policy is PROMPT-only (the method's identity); the decode-only leg
  deliberately inverts it, and — uniquely among these arms — the mode is
  PERSISTED in the artifact, because re-hydrating a response-fit LoReFT
  as prompt-only would silently run the wrong method;
- MultiLayerLoRAObject (lora_r1_all, 2026-08-20): RankOneLoRAObject's
  wiring per layer — one shared vectorless tracker, torch.where gating —
  so the same exactness holds at every steered layer;
- MultiLayerVectorObject (vector_all, 2026-08-20): the steering()-backed
  twin of the `vector` arm — its fit leg routes into hooks.steering, whose
  live handle is both the mask (SteeringHandle._position_mask) and the
  thing yielded for the loss tie.

REFUSING (loudly, with the reason in the message):
- token_bos / token_faithful: a soft PROMPT token — every position it can
  act on is a prompt position; no generated-position notion exists;
- alphakl_null / alphakl_soft / alphakl_fisher: closed-form probe fits on
  UNSTEERED activations at spec-pinned positions — no steered training
  forward exists for a mask to act on, and a "masked" fit would relabel a
  bit-identical artifact while invalidating the stage-3 KL certificate.

Also pinned: the bitwise lockstep-vs-solo contract holds for the MASKED
leg of psr_kl_all and lora_r1 (the existing tests pin the unmasked leg),
and the loss gather of a masked fit equals hook-mask ∩ completion during
real training (the mask→loss wiring).

CPU, conftest tiny_llama, synthetic French pairs — never real store data.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

import sisteer.extraction.objects as objects_mod
from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.distill import (
    _encode_aligned,
    steer_prompt_boundaries,
)
from sisteer.extraction.lockstep import (
    LockstepArm,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.extraction.lofit import LoFiTObject, scaler_context
from sisteer.extraction.multilayer import (
    MultiLayerLoRAObject,
    MultiLayerVectorObject,
)
from sisteer.extraction.objects import RankOneLoRAObject, train_object_kl
from sisteer.extraction.psr import PSRObject
from sisteer.extraction.reft import LoReFTObject
from sisteer.extraction.traindata import collate
from sisteer.hooks import SteeringHandle
from sisteer.methods import ARMS, load_object
from sisteer.models import get_decoder_layers, hidden_size

INSTRUCTION = BY_ID["french"]
H = 2  # synthetic header allowance — the machinery is convention-agnostic
MAX_LEN = 96
TRAIN_KW = dict(epochs=2, batch_size=2, max_len=MAX_LEN, seed=0)


def _pairs():
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _student_batch(tok):
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    encoded = [_encode_aligned(tok, ex, MAX_LEN) for ex in examples]
    return collate([s for _, s in encoded], tok.pad_token_id)


@contextlib.contextmanager
def _capture(model, layers):
    """Post-hook decoder-layer outputs (registered LAST, so it sees the
    object's edits — later-registered forward hooks receive earlier hooks'
    modified output)."""
    captured: dict[int, torch.Tensor] = {}
    dec = get_decoder_layers(model)

    def mk(i):
        def hook(_m, _a, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[i] = h.detach().clone()

        return hook

    handles = [dec[i].register_forward_hook(mk(i)) for i in layers]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _base_and_masked(model, batch, obj, layers):
    """(base capture, masked capture, [B, S] bool mask the handle used)."""
    with torch.no_grad(), _capture(model, layers) as base:
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    with torch.no_grad(), obj.apply(model, scale=1.0) as handle:
        assert isinstance(handle, SteeringHandle), (
            "a response-mode object's apply() must yield its mask handle")
        bounds = steer_prompt_boundaries(batch, handle.response_header_len)
        assert bool((bounds > 0).all()), (
            "fixture prompt at or below H — that row's boundary collapsed "
            "to 0, so 'response' steers EVERY position and the test would "
            "pass while testing nothing (the peer-pinned width <= H trap)")
        handle.set_prompt_lens(bounds)
        with _capture(model, layers) as steered:
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
        b, s = batch.input_ids.shape
        mask = handle._position_mask(b, s, batch.input_ids.device).bool()
    return base, steered, mask


def _assert_masked_exact(base, steered, mask, batch, layers):
    """Masked-out positions BIT-identical; every steered real position
    changed. The response mask is a per-row suffix under right padding, so
    causal attention keeps the off-prefix clean at EVERY layer."""
    on_real = mask & batch.attention_mask.bool()
    assert bool((~mask).any()), "fixture left nothing masked"
    assert bool(on_real.any()), "fixture left nothing steered"
    for l in layers:
        assert torch.equal(steered[l][~mask], base[l][~mask]), (
            f"layer {l}: a masked-out position was not bit-identical")
        diff = (steered[l] != base[l]).any(dim=-1)
        assert bool(diff[on_real].all()), (
            f"layer {l}: a steered position was left unedited")


# ---------------------------------------------------------------------------
# object-level exactness: steered set == the handle's mask, off == bitwise
# ---------------------------------------------------------------------------


def test_psr_masked_hook_steers_exactly_the_mask(tiny_llama):
    model, tok = tiny_llama
    batch = _student_batch(tok)
    obj = PSRObject(model, [0, 1])  # gate init w=0, b=1 -> g == 1 everywhere
    d = hidden_size(model)
    obj.set_directions({0: 5.0 * torch.ones(d), 1: 5.0 * torch.ones(d)})
    # default leg: apply() yields None and stays the historical all-token fit
    with obj.apply(model, scale=1.0) as ctx:
        assert ctx is None
    obj.set_fit_positions("response", H)
    base, steered, mask = _base_and_masked(model, batch, obj, [0, 1])
    _assert_masked_exact(base, steered, mask, batch, [0, 1])


def test_lora_r1_masked_hook_steers_exactly_the_mask(tiny_llama):
    model, tok = tiny_llama
    batch = _student_batch(tok)
    obj = RankOneLoRAObject(model, layer=1)
    obj.B.data = 10.0 * torch.ones_like(obj.B.data)
    with obj.apply(model, scale=1.0) as ctx:
        assert ctx is None
    obj.set_fit_positions("response", H)
    base, steered, mask = _base_and_masked(model, batch, obj, [1])
    _assert_masked_exact(base, steered, mask, batch, [1])


def test_lora_r1_all_masked_hook_steers_exactly_the_mask(tiny_llama):
    model, tok = tiny_llama
    batch = _student_batch(tok)
    obj = MultiLayerLoRAObject(model, layers=[0, 1])
    for l in (0, 1):
        obj.B[l].data = 10.0 * torch.ones_like(obj.B[l].data)
    # default leg: apply() yields None and stays the historical all-token fit
    with obj.apply(model, scale=1.0) as ctx:
        assert ctx is None
    obj.set_fit_positions("response", H)
    base, steered, mask = _base_and_masked(model, batch, obj, [0, 1])
    _assert_masked_exact(base, steered, mask, batch, [0, 1])


def test_vector_all_masked_hook_steers_exactly_the_mask(tiny_llama):
    model, tok = tiny_llama
    batch = _student_batch(tok)
    obj = MultiLayerVectorObject(model, layers=[0, 1])
    for l in (0, 1):
        obj.params[l].data = 5.0 * torch.ones_like(obj.params[l].data)
    # default leg: steering()-backed, so apply() ALWAYS yields the live
    # steering handle — on the default leg it is an all-positions handle
    # (positions="all"), the historical fit.
    with obj.apply(model, scale=1.0) as ctx:
        assert isinstance(ctx, SteeringHandle) and ctx.positions == "all"
    obj.set_fit_positions("response", H)
    base, steered, mask = _base_and_masked(model, batch, obj, [0, 1])
    _assert_masked_exact(base, steered, mask, batch, [0, 1])


def test_lofit_masked_hook_steers_exactly_the_mask(tiny_llama):
    model, tok = tiny_llama
    batch = _student_batch(tok)
    obj = LoFiTObject(model, heads=[(1, 0)])
    obj.v.data = 10.0 * torch.ones_like(obj.v.data)
    with obj.apply(model, scale=1.0) as ctx:
        assert ctx is None
    obj.set_fit_positions("response", H)
    base, steered, mask = _base_and_masked(model, batch, obj, [1])
    _assert_masked_exact(base, steered, mask, batch, [1])


def test_lofit_scaler_context_honors_the_mask(tiny_llama):
    """Phase-A head selection under the decode-only leg measures head
    importance on the same position support as phase v: the scalers mask
    exactly like the biases."""
    model, tok = tiny_llama
    batch = _student_batch(tok)
    n_layers = len(get_decoder_layers(model))
    from sisteer.extraction.lofit import head_geometry

    _, n_heads, head_dim = head_geometry(model)
    A = 0.5 * torch.ones(n_layers, n_heads, head_dim)
    with torch.no_grad(), _capture(model, [0, 1]) as base:
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    handle = SteeringHandle(model, {}, positions="response",
                            response_header_len=H)
    try:
        handle.set_prompt_lens(steer_prompt_boundaries(batch, H))
        with torch.no_grad(), scaler_context(model, A, mask_handle=handle):
            with _capture(model, [0, 1]) as steered:
                model(input_ids=batch.input_ids,
                      attention_mask=batch.attention_mask)
        b, s = batch.input_ids.shape
        mask = handle._position_mask(b, s, batch.input_ids.device).bool()
    finally:
        handle.remove()
    _assert_masked_exact(base, steered, mask, batch, [0, 1])


def test_loreft_masked_hook_steers_exactly_the_mask(tiny_llama):
    model, tok = tiny_llama
    batch = _student_batch(tok)
    obj = LoReFTObject(model, [1])
    # W=R, b=0 is an exact identity at init — give the affine map a bias so
    # the edit is visibly nonzero: delta = (W h + b - R h) @ R = b @ R.
    with torch.no_grad():
        obj.learned[1].bias.fill_(1.0)
    obj.set_fit_positions("response", H)
    base, steered, mask = _base_and_masked(model, batch, obj, [1])
    _assert_masked_exact(base, steered, mask, batch, [1])


def test_loreft_response_mode_edits_decode_steps_prompt_mode_passes_through(
        tiny_llama):
    """The generated-only serving shape (ctx entered for the decode forward
    only, prefill outside): a response-fit LoReFT EDITS the decode step; the
    default prompt-only object passes it through bit-identically — the
    S==1 pass-through that would have silently no-opped a decode-only cell."""
    model, tok = tiny_llama
    enc = tok("one two three four five six", return_tensors="pt")
    nxt = enc["input_ids"][:, -1:]

    def fresh_past():
        with torch.no_grad():
            return model(**enc, use_cache=True).past_key_values

    with torch.no_grad():
        ref = model(input_ids=nxt, past_key_values=fresh_past()).logits

    def decode_under(obj):
        with torch.no_grad(), obj.apply(model, scale=1.0):
            return model(input_ids=nxt, past_key_values=fresh_past()).logits

    resp = LoReFTObject(model, [1])
    with torch.no_grad():
        resp.learned[1].bias.fill_(1.0)
    resp.set_fit_positions("response", H)
    assert not torch.equal(decode_under(resp), ref), (
        "response-mode LoReFT left a decode step unedited")

    prompt_only = LoReFTObject(model, [1])
    with torch.no_grad():
        prompt_only.learned[1].bias.fill_(1.0)
    assert torch.equal(decode_under(prompt_only), ref), (
        "prompt-only LoReFT must pass decode steps through untouched")


def test_loreft_response_mode_persists_through_save_load(tiny_llama,
                                                         tmp_path):
    """A response-fit LoReFT artifact must re-hydrate as response-mode (or
    it silently runs the wrong method — and no-ops entirely under
    generated-only serving); a default artifact stays prompt-only."""
    model, tok = tiny_llama
    obj = LoReFTObject(model, [1])
    obj.set_fit_positions("response", H)
    p = tmp_path / "object.pt"
    obj.save(p)
    blob = torch.load(p, weights_only=True)
    assert blob["positions"] == "response"
    assert blob["response_header_len"] == H
    loaded = load_object(p, model=model, tok=tok)
    assert loaded.fit_positions == "response"
    assert loaded.fit_response_header_len == H

    default = LoReFTObject(model, [1])
    p2 = tmp_path / "object_default.pt"
    default.save(p2)
    blob2 = torch.load(p2, weights_only=True)
    assert blob2["positions"] == "prompt"  # the method's own identity
    loaded2 = load_object(p2, model=model, tok=tok)
    assert loaded2.fit_positions == "all"

    # prompt-mode machinery is refused on a response-mode object: the two
    # position sources must never be mixed silently.
    with pytest.raises(RuntimeError, match="prompt-mode"):
        loaded.set_prompt_lens(torch.tensor([3]))


# ---------------------------------------------------------------------------
# mask→loss wiring during REAL masked training (the seam, not a re-derivation)
# ---------------------------------------------------------------------------


def test_masked_lora_training_loss_positions_equal_hook_mask(
        tiny_llama, monkeypatch):
    """Every training-step loss gather of a masked lora_r1 fit equals
    (handle mask ∩ completion mask), read from the SAME handle the hook
    consumed — asserted on the real train_object_kl call path."""
    model, tok = tiny_llama
    records = []
    orig = objects_mod.loss_gather_masks

    def rec(handle, student_b, teacher_b=None):
        out = orig(handle, student_b, teacher_b)
        b, s = student_b.input_ids.shape
        m = handle._position_mask(b, s, student_b.input_ids.device)
        records.append((m.detach().clone(), out[0],
                        student_b.labels[:, 1:] != -100))
        return out

    monkeypatch.setattr(objects_mod, "loss_gather_masks", rec)
    obj = RankOneLoRAObject(model, layer=1)
    train_object_kl(model, tok, obj,
                    pairs_to_distill_examples(tok, _pairs(), INSTRUCTION),
                    positions="response", response_header_len=H, **TRAIN_KW)
    assert records, "the masked wiring never ran"
    for mask, s_gather, completion in records:
        assert torch.equal(s_gather, completion & mask[:, :-1].bool()), (
            "loss positions != hook mask ∩ completion — the loss read "
            "positions the hook did not steer (or vice versa)")


# ---------------------------------------------------------------------------
# bitwise lockstep-vs-solo for the newly maskable KL arms (masked leg)
# ---------------------------------------------------------------------------


def _assert_bitwise(a, b, *, arm):
    ta = [p.detach().clone() for p in a.trainable]
    tb = [p.detach().clone() for p in b.trainable]
    assert len(ta) == len(tb), f"{arm}: param count differs"
    for i, (x, y) in enumerate(zip(ta, tb)):
        assert torch.equal(x, y), (
            f"{arm}: param {i} not bitwise identical "
            f"(max abs diff {float((x - y).abs().max())})")


@pytest.mark.parametrize("arm", ["psr_kl_all", "lora_r1",
                                 "vector_all", "lora_r1_all"])
@pytest.mark.parametrize("rhl", [H, None])
def test_masked_arm_bitwise_lockstep_vs_solo(tiny_llama, arm, rhl):
    """The lockstep numerical contract extends to the masked leg: a
    decode-only (header-steered AND plain) psr_kl_all / lora_r1 — and,
    since 2026-08-20, vector_all / lora_r1_all — trained in a lockstep
    group is bitwise identical to the same leg trained solo through its
    registry trainer."""
    model, tok = tiny_llama
    pairs = _pairs()
    layer = 1
    solo = ARMS[arm].train(model, tok, INSTRUCTION, pairs, layer=layer,
                           positions="response", response_header_len=rhl,
                           **TRAIN_KW)
    examples = pairs_to_distill_examples(tok, pairs, INSTRUCTION)
    group = [
        LockstepArm(arm,
                    build_lockstep_object(arm, model, tok, INSTRUCTION,
                                          pairs, layer=layer,
                                          positions="response",
                                          response_header_len=rhl),
                    default_lockstep_params(arm)),
        LockstepArm("vector",
                    build_lockstep_object("vector", model, tok, INSTRUCTION,
                                          pairs, layer=layer),
                    default_lockstep_params("vector")),
    ]
    train_lockstep(model, tok, group, examples, **TRAIN_KW)
    _assert_bitwise(solo, group[0].obj, arm=arm)


# ---------------------------------------------------------------------------
# solo trainers: the masked leg trains, and trains DIFFERENTLY from all-token
# ---------------------------------------------------------------------------


def test_psr_mse_all_masked_fit_differs_from_all_token(tiny_llama):
    model, tok = tiny_llama
    pairs = _pairs()

    def run(**mask_kw):
        obj = ARMS["psr_mse_all"].train(model, tok, INSTRUCTION, pairs,
                                        layer=1, **mask_kw, **TRAIN_KW)
        return torch.cat([p.detach().flatten() for p in obj.trainable])

    masked = run(positions="response", response_header_len=H)
    plain = run()
    assert not torch.equal(masked, plain), (
        "the masked psr_mse_all fit is indistinguishable from all-token — "
        "the mask never reached the training forward")


def test_psr_mse_masked_misderived_boundary_raises(tiny_llama, monkeypatch):
    """The padded-batch-width trap on the mse (activation-loss) side: a
    boundary derived from the batch width leaves supervised completion
    positions outside the mask — refuse, never train on the wrong
    support."""
    import sisteer.extraction.psr as psr_mod

    model, tok = tiny_llama

    def bad_boundaries(batch, rhl):
        return torch.full_like(batch.prompt_lens, batch.input_ids.shape[1])

    monkeypatch.setattr(psr_mod, "steer_prompt_boundaries", bad_boundaries)
    with pytest.raises(ValueError, match="mask→loss wiring \\(psr mse\\)"):
        ARMS["psr_mse_all"].train(model, tok, INSTRUCTION, _pairs(), layer=1,
                                  positions="response",
                                  response_header_len=H, **TRAIN_KW)


def test_loreft_masked_fit_differs_from_prompt_only(tiny_llama):
    model, tok = tiny_llama
    pairs = _pairs()

    def run(**mask_kw):
        obj = ARMS["loreft"].train(model, tok, INSTRUCTION, pairs, layer=1,
                                   **mask_kw, **TRAIN_KW)
        return obj, torch.cat([p.detach().flatten() for p in obj.trainable])

    masked_obj, masked = run(positions="response", response_header_len=H)
    _plain_obj, plain = run()
    assert masked_obj.fit_positions == "response"
    assert not torch.equal(masked, plain), (
        "the masked loreft fit is indistinguishable from the prompt-only "
        "fit — the mask never reached the training forward")


def test_lofit_masked_fit_wires_both_phases(tiny_llama, monkeypatch):
    """The decode-only lofit fit masks BOTH phases: every phase-A and
    phase-v training step derives its CE gather from the live mask handle
    (loss_gather_masks — recorded on the real call path), and the phase-A
    head-ranking signal diverges from the all-token leg's. (The trained v
    itself can coincide bitwise on this tiny fixture: v has 4 elements and
    AdamW's first-step update is exactly ±lr, so the assertion pins the
    wiring and the ranking, not v.)"""
    import sisteer.extraction.lofit as lofit_mod
    from sisteer.extraction.lofit import train_lofit

    model, tok = tiny_llama
    examples = pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)
    records = []
    orig = lofit_mod.loss_gather_masks

    def rec(handle, student_b, teacher_b=None):
        out = orig(handle, student_b, teacher_b)
        b, s = student_b.input_ids.shape
        m = handle._position_mask(b, s, student_b.input_ids.device)
        records.append((m.detach().clone(), out[0],
                        student_b.labels[:, 1:] != -100))
        return out

    monkeypatch.setattr(lofit_mod, "loss_gather_masks", rec)
    masked = train_lofit(model, tok, examples, layer=1, epochs=1,
                         batch_size=2, max_len=MAX_LEN, seed=0,
                         positions="response", response_header_len=H)
    assert masked.fit_positions == "response"
    assert masked.metadata["fit_positions"] == "response"
    assert masked.metadata["fit_response_header_len"] == H
    # 2 batches x 1 epoch x 2 phases: the mask reached BOTH phases.
    assert len(records) == 4, (
        f"expected 4 masked steps (2 per phase), saw {len(records)} — a "
        "phase trained without the mask")
    for mask, s_gather, completion in records:
        assert torch.equal(s_gather, completion & mask[:, :-1].bool()), (
            "a lofit training step's CE gather != hook mask ∩ completion")

    plain = train_lofit(model, tok, examples, layer=1, epochs=1,
                        batch_size=2, max_len=MAX_LEN, seed=0)
    assert plain.metadata["fit_positions"] == "all"
    assert len(records) == 4  # the all-token leg never touches the wiring
    # Same seed, same data — the masked leg's selection signal must differ.
    assert not torch.equal(plain.scaler_magnitudes,
                           masked.scaler_magnitudes), (
        "the masked lofit fit's phase-A ranking signal is indistinguishable "
        "from all-token — the mask never reached the selection forward")


# ---------------------------------------------------------------------------
# the refusing arms: loud, reasoned, and BEFORE any training work
# ---------------------------------------------------------------------------


def test_token_bos_refuses_masked_fit_solo(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="cannot honor.*soft PROMPT token"):
        ARMS["token_bos"].train(
            model, tok, INSTRUCTION, _pairs(), layer=1,
            sentinel=(tok.unk_token, tok.unk_token_id),
            positions="response", response_header_len=H, **TRAIN_KW)


@pytest.mark.parametrize("arm", ["alphakl_null", "alphakl_soft",
                                 "alphakl_fisher"])
def test_alphakl_arms_refuse_masked_fit_with_the_reason(tiny_llama, arm):
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="cannot honor.*closed-form"):
        ARMS[arm].train(model, tok, INSTRUCTION, _pairs(), layer=1,
                        positions="response", response_header_len=H)


# ---------------------------------------------------------------------------
# the production driver end to end: masked cells train, refusing cells fail
# loudly and are CONTAINED (recorded per arm, never fatal to the run)
# ---------------------------------------------------------------------------


def test_two_si_driver_masked_fit_trains_maskable_and_contains_refusals(
        tiny_llama, tmp_path):
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import train_two_si_objects as t2

    model, tok = tiny_llama
    si = "safety_refusal"
    pairs = [Pair(prompt=p, pos=pos, neg=neg, instruction_id=si)
             for p, pos, neg in [
                 ("Say something nice.", "Bonjour le monde", "Hello world"),
                 ("Describe a tree.", "C'est un grand arbre", "Tall tree"),
                 ("Greet the reader.", "Salut a tous", "Hi everyone")]]
    cfg = {
        "layer": 1,
        "completion_tokens": 16,
        "max_len": 256,
        "train_kwargs": {"default": {
            "epochs": 1, "seed": 0,
            # The fit's mask leg, as a config would request it. H is given
            # explicitly: the tiny tokenizer cannot derive a header length
            # and the driver refuses H == 0 rather than silently collapsing
            # header-steered to plain decode-only.
            "positions": "response", "response_header_len": H,
            # Not under test here; the tiny fixture's floors are noise.
            "floor_ratio_max": None,
        }},
        "si_ids": [si],
    }
    arms = ["lora_r1", "psr_kl_all", "loreft", "token_bos", "alphakl_null"]
    outcome = t2.train_si_objects(
        model, tok, si, pairs, arms, cfg=cfg, save_dir=tmp_path,
        sentinel=(tok.unk_token, tok.unk_token_id))

    for arm in ("lora_r1", "psr_kl_all", "loreft"):
        assert outcome.arms[arm].status == "trained", (
            f"{arm}: {outcome.arms[arm].error}")
    # loreft trained through the driver really is in response mode
    assert outcome.objects["loreft"].fit_positions == "response"

    # the refusing arms fail LOUDLY, with the reason recorded, and their
    # failure is contained — the maskable arms above still trained + saved
    assert outcome.arms["token_bos"].status == "failed"
    assert "soft PROMPT token" in outcome.arms["token_bos"].error
    assert outcome.arms["alphakl_null"].status == "failed"
    assert "closed-form" in outcome.arms["alphakl_null"].error

    from sisteer.eval.defense_cell import object_path_for
    import json
    for arm in ("lora_r1", "psr_kl_all", "loreft"):
        cell = object_path_for(tmp_path, f"{si}__{arm}")
        assert cell.exists()
        meta = json.loads(
            (cell.parent / "train_budget_meta.json").read_text())
        assert meta["positions"] == "response"
        assert meta["response_header_len"] == H
    for arm in ("token_bos", "alphakl_null"):
        cell = object_path_for(tmp_path, f"{si}__{arm}")
        assert cell is None or not cell.exists()


# ---------------------------------------------------------------------------
# the FIRST-response-token anchor, against INDEPENDENT ground truth
# ---------------------------------------------------------------------------
# Every exactness test above compares the steered set to the HANDLE'S OWN
# mask — sound for "hook == mask", but blind to a boundary that is itself
# off by one (handle and comparison would drift together). These tests
# anchor the boundary to a ground truth the mask machinery never touches:
# the LABELS (-100 on prompt & padding, real ids on completion tokens).
# The Class C contract is "the intervention is present from the very first
# response token": with c0 = each row's first completion-token index, that
# requires
#   - position c0 - 1 steered — the logit computed THERE is what picks the
#     first response token (an unsteered c0 - 1 means token 1 is sampled
#     from unsteered logits, silently disabling a refusal defense)
#   - position c0 steered — the first response token's own hidden state
#   - position c0 - H - 1 (the last real prompt position before the
#     assistant header) BIT-identical to the unsteered forward
#   - the masked fit's loss gather including the c0-predicting logit.
# An off-by-one in EITHER direction fails at least one of these.


def _labels_first_completion(batch):
    """[B] index of each row's first completion token, from the labels —
    ground truth independent of prompt_lens / steer_prompt_boundaries."""
    comp = batch.labels != -100
    assert bool(comp.any(dim=1).all()), "fixture row with no completion"
    return comp.float().argmax(dim=1)


def _assert_first_response_token_anchor(model, tok, obj, layers):
    from sisteer.extraction.distill import loss_gather_masks

    batch = _student_batch(tok)
    c0 = _labels_first_completion(batch)
    # Cross-check the two independent encodings of the same boundary: the
    # first completion token must sit exactly at prompt_lens — if these
    # ever disagree, steer_prompt_boundaries and the loss masks are built
    # from different worlds and every downstream assertion is void.
    assert torch.equal(c0, batch.prompt_lens), (
        "labels' first completion index != prompt_lens — the two encodings "
        "of the response boundary disagree")
    assert bool((c0 > H).all()), "fixture prompt must be longer than H"

    base, steered, mask = _base_and_masked(model, batch, obj, layers)
    rows = range(batch.input_ids.shape[0])
    for l in layers:
        for r in rows:
            b = int(c0[r])
            # steered: the logit source of response token 1, the token
            # itself, and every header position in between
            for pos in range(b - H, b + 1):
                assert bool((steered[l][r, pos] != base[l][r, pos]).any()), (
                    f"layer {l} row {r}: position {pos} (boundary {b}, "
                    f"H={H}) must be steered — the intervention is not "
                    "live from the very first response token")
            # untouched: the last real prompt position before the header
            assert torch.equal(steered[l][r, b - H - 1],
                               base[l][r, b - H - 1]), (
                f"layer {l} row {r}: prompt position {b - H - 1} was "
                "edited — the mask leaks into the prompt")
    # the handle's mask agrees with the labels-derived boundary everywhere
    local = torch.arange(batch.input_ids.shape[1])[None, :]
    expect = local >= (c0 - H)[:, None]
    assert torch.equal(mask, expect), (
        "handle mask != labels-derived header-steered mask")

    # mask→loss wiring reaches token 1: the gather keeps c0 - 1
    fresh = obj._fit_mask_handle(model)
    try:
        fresh.set_prompt_lens(steer_prompt_boundaries(
            batch, fresh.response_header_len))
        s_gather, _tg, keep = loss_gather_masks(fresh, batch)
    finally:
        fresh.remove()
    for r in rows:
        assert bool(s_gather[r, int(c0[r]) - 1]), (
            f"row {r}: the first-response-token logit position "
            f"{int(c0[r]) - 1} is outside the loss gather")
    assert keep is None or bool(keep.all()), (
        "header-steered gather dropped a completion position")


def test_psr_response_fit_steers_from_the_first_response_token(tiny_llama):
    model, tok = tiny_llama
    obj = PSRObject(model, [0, 1])
    d = hidden_size(model)
    obj.set_directions({0: 5.0 * torch.ones(d), 1: 5.0 * torch.ones(d)})
    obj.set_fit_positions("response", H)
    _assert_first_response_token_anchor(model, tok, obj, [0, 1])


def test_vector_all_response_fit_steers_from_the_first_response_token(
        tiny_llama):
    model, tok = tiny_llama
    obj = MultiLayerVectorObject(model, layers=[0, 1])
    for l in (0, 1):
        obj.params[l].data = 5.0 * torch.ones_like(obj.params[l].data)
    obj.set_fit_positions("response", H)
    _assert_first_response_token_anchor(model, tok, obj, [0, 1])
