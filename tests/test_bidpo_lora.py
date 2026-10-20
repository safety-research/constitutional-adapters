"""bidpo_lora: bidpo's bidirectional-DPO objective on lora_r1's substrate.

Pins the four claims the arm exists on (synthetic placeholder data ONLY —
invented prompts/completions, never recorded session or pool text):

1. the SUBSTRATE is lora_r1's exactly (same class, same layer, same target
   module, same init) — anything else re-confounds the lora_r1 comparison;
2. the OBJECTIVE is bidirectional and the -object term applies the NEGATED
   low-rank update (scale=-1 == -B A^T), in the hook and in training;
3. the objective IS train_bidpo's: bidpo.py holds its loss inline (and a
   peer branch owns that file, so no shared helper could be factored out),
   so the engine is a transcription — and THIS file's equivalence test is
   the drift alarm that keeps "transcription" from decaying into
   "divergent second copy": both trainers run on the same vector
   substrate/data/seed and must produce identical per-step losses and
   artifacts. If a deliberate bidpo.py objective change lands, re-align
   extraction/bidpo_lora.py to it, then update this test's expectation —
   never delete the comparison;
4. the arm resolves through the registry and the fit wave's plan (solo, at
   the DPO width), deterministically under a fixed seed, producing
   lora_r1's artifact type;
5. position-masked (decode-only) fits are WIRED (2026-08-21): a
   positions="response" fit is accepted, restricts the edit to exactly the
   handle's mask (masked positions bit-identical to the unsteered model),
   ties BOTH completion-logprob sums of the DPO margin to that same mask,
   and refuses loudly for an object exposing no fit mask — while the
   default all-token leg stays bit-identical (claim 3's drift alarm holds
   it to the untouched train_bidpo).
"""

from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from sisteer.data.instructions import Instruction
from sisteer.data.pairs import Pair, pairs_to_pref_examples
from sisteer.extraction.bidpo import PrefExample, train_bidpo
from sisteer.extraction.bidpo_lora import train_bidpo_lora, train_bidpo_object
from sisteer.extraction.distill import steer_prompt_boundaries
from sisteer.extraction.lockstep import lockstep_compatible
from sisteer.extraction.objects import RankOneLoRAObject
from sisteer.extraction.traindata import (
    collate,
    completion_logprobs,
    encode_prompt_completion,
)
from sisteer.hooks import SteeringHandle, steering
from sisteer.methods import ARMS, S_TWO_SI, load_object, object_artifact_name
from sisteer.models import get_decoder_layers, hidden_size

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

LAYER = 1

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


# -- 1. the substrate is lora_r1's, exactly --------------------------------------


def test_substrate_matches_lora_r1_exactly(tiny_llama):
    model, tok = tiny_llama
    a = ARMS["lora_r1"].build(model, tok, LAYER)
    b = ARMS["bidpo_lora"].build(model, tok, LAYER)
    # same CLASS (not a subclass), same kind, same pinned layer
    assert type(b) is type(a) is RankOneLoRAObject
    assert b.kind == a.kind == "lora"
    assert b.layer == a.layer == LAYER
    # same TARGET MODULE — the identical o_proj module object
    assert b._module is a._module
    assert b._module is get_decoder_layers(model)[LAYER].self_attn.o_proj
    # same rank-1 shapes and the same INIT (seed-0 A, zero B) — bit-equal
    assert a.A.shape == b.A.shape and a.B.shape == b.B.shape
    assert torch.equal(a.A, b.A)
    assert torch.equal(a.B, b.B) and not b.B.any()
    # the TRAINED object is still that exact substrate
    obj = train_bidpo_lora(model, tok, _prefs(), LAYER,
                           epochs=1, batch_size=2, max_len=64)
    assert type(obj) is RankOneLoRAObject
    assert obj.layer == LAYER and obj._module is a._module
    # and the artifact contract is lora_r1's (torch blob, kind-dispatched)
    assert (object_artifact_name("bidpo_lora")
            == object_artifact_name("lora_r1") == "object.pt")


def test_artifact_round_trips_like_lora_r1s(tiny_llama, tmp_path):
    model, tok = tiny_llama
    obj = train_bidpo_lora(model, tok, _prefs(), LAYER,
                           epochs=1, batch_size=2, max_len=64)
    path = tmp_path / object_artifact_name("bidpo_lora")
    obj.save(path)
    back = load_object(path, model=model, tok=tok)
    assert type(back) is RankOneLoRAObject
    assert back.layer == LAYER
    assert torch.equal(back.A, obj.A.detach().float().cpu())
    assert torch.equal(back.B, obj.B.detach().float().cpu())


# -- 2. bidirectional, with the negation actually applied ------------------------


def test_scale_minus_one_is_the_negated_low_rank_update(tiny_llama):
    """The -object leg exists for LoRA because the edit is additive:
    apply(scale=-1) must shift the target module's output by exactly MINUS
    the scale=+1 shift (the negated update -B A^T), not by the same one."""
    model, tok = tiny_llama
    obj = RankOneLoRAObject(model, layer=LAYER)
    g = torch.Generator().manual_seed(11)
    obj.B.data = 0.1 * torch.randn(obj.B.shape, generator=g)
    module = get_decoder_layers(model)[LAYER].self_attn.o_proj
    x = torch.randn(2, 5, module.weight.shape[1], generator=g)
    with torch.no_grad():
        y0 = module(x)
        with obj.apply(model, scale=+1.0):
            y_pos = module(x)
        with obj.apply(model, scale=-1.0):
            y_neg = module(x)
    delta_pos, delta_neg = y_pos - y0, y_neg - y0
    assert delta_pos.abs().max() > 0  # non-vacuous
    assert torch.allclose(delta_neg, -delta_pos, atol=1e-6)
    assert not torch.allclose(delta_neg, delta_pos, atol=1e-6)


def test_training_applies_the_object_at_both_signs(tiny_llama):
    """direction='both' must evaluate the policy under the object AND its
    negation every step: exactly [+1, +1, -1, -1] applies per optimization
    step (chosen/rejected under +v, then chosen/rejected under -v), and the
    reference forwards never apply the object at all."""
    model, tok = tiny_llama
    obj = RankOneLoRAObject(model, layer=LAYER)
    scales: list[float] = []
    orig_apply = obj.apply

    def spy(mdl, scale=1.0):
        scales.append(scale)
        return orig_apply(mdl, scale=scale)

    obj.apply = spy
    prefs = _prefs()
    history = train_bidpo_object(model, tok, obj, prefs,
                                 epochs=1, batch_size=2, max_len=64, seed=0)
    n_steps = len(history)
    assert n_steps == 2  # 3 examples at batch 2 -> 2 batches x 1 epoch
    assert scales == [+1.0, +1.0, -1.0, -1.0] * n_steps


def test_trained_object_moves_preferences_bidirectionally(tiny_llama):
    """After training: +object prefers chosen over rejected MORE than the
    unsteered reference does, and -object prefers it LESS (i.e. prefers
    rejected) — BiPO's two legs, on the LoRA substrate."""
    model, tok = tiny_llama
    prefs = _prefs()
    obj = train_bidpo_lora(model, tok, prefs, LAYER,
                           epochs=12, lr=1e-2, batch_size=3, max_len=64,
                           seed=0)

    seqs_c = [encode_prompt_completion(tok, p.prompt, p.chosen, 64)
              for p in prefs]
    seqs_r = [encode_prompt_completion(tok, p.prompt, p.rejected, 64)
              for p in prefs]
    bc = collate(seqs_c, tok.pad_token_id)
    br = collate(seqs_r, tok.pad_token_id)

    def pref_gap(scale):
        with torch.no_grad():
            with (obj.apply(model, scale=scale) if scale is not None
                  else _null()):
                lc = completion_logprobs(model(
                    input_ids=bc.input_ids,
                    attention_mask=bc.attention_mask).logits, bc)
                lr_ = completion_logprobs(model(
                    input_ids=br.input_ids,
                    attention_mask=br.attention_mask).logits, br)
        return (lc - lr_).mean()

    ref = pref_gap(None)
    fwd = pref_gap(+1.0)
    rev = pref_gap(-1.0)
    assert fwd > ref, (fwd, ref)   # +object: prefer chosen
    assert rev < ref, (rev, ref)   # -object: prefer rejected
    assert fwd > rev


@contextmanager
def _null():
    yield


# -- 3. the objective IS train_bidpo's (the drift alarm) --------------------------


class _VectorSubstrate:
    """train_bidpo's exact substrate behind the object protocol: a zero-init
    d_model vector applied through the same hooks.steering call train_bidpo
    makes. Lets the generic engine run the VECTOR arm, so its loss can be
    compared to train_bidpo's step for step.

    skip_bos=True is part of "the same hooks.steering call" (2026-08-21):
    train_bidpo now trains under its artifact's DEPLOY convention —
    methods.load_object serves every plain SteeringVector as VectorObject,
    whose default is skip_bos=True — so the substrate carries the same flag
    (exactly like lockstep's _VectorArmObject, the tree's other
    vector-behind-the-object-protocol adapter). The BOS exclusion is a
    property of the VECTOR substrate's forward, NOT of the DPO objective
    this alarm guards: bidpo_lora's own o_proj substrate has no BOS notion
    (its weight edit applies at every position at train and deploy alike,
    consistently), so the engine needed no change — only this adapter's
    forward was re-derived to keep the two trainers bit-comparable."""

    def __init__(self, model, layer: int):
        self.layer = layer
        self.vec = nn.Parameter(torch.zeros(
            hidden_size(model), device=next(model.parameters()).device))

    @property
    def trainable(self):
        return [self.vec]

    def to_(self, device):
        self.vec.data = self.vec.data.to(device)

    @contextmanager
    def apply(self, model, scale: float = 1.0):
        with steering(model, {self.layer: self.vec}, scale=scale,
                      positions="all", skip_bos=True):
            yield


@pytest.mark.parametrize("direction", ["both", "sampled"])
def test_engine_matches_train_bidpo_on_the_vector_substrate(tiny_llama,
                                                            direction):
    """THE drift alarm (see the module docstring): on the vector substrate
    the generic engine and train_bidpo are the same ops in the same order,
    so their per-step losses and final artifacts must match EXACTLY — a
    transcription that drifts (either side) fails here, loudly."""
    model, tok = tiny_llama
    prefs = _prefs()
    hp = dict(beta=0.1, lr=5e-4, weight_decay=0.05, epochs=2, batch_size=2,
              max_len=64, direction=direction, seed=3)

    bidpo_losses: list[float] = []
    sv = train_bidpo(model, tok, prefs, LAYER,
                     log_fn=lambda m, step: bidpo_losses.append(
                         m["bidpo/loss"]),
                     **hp)

    adapter = _VectorSubstrate(model, LAYER)
    history = train_bidpo_object(model, tok, adapter, prefs, **hp)

    assert history == bidpo_losses
    assert torch.equal(sv.vectors[LAYER],
                       adapter.vec.detach().float().cpu())


# -- 4. registry, plan, determinism ----------------------------------------------


def test_arm_wiring_registry_and_suites():
    spec = ARMS["bidpo_lora"]
    # S_TWO_SI ONLY — its cells derive from configs/two_si.yaml's
    # class_a_arms, and it must stay out of every stage-2 positional decode
    assert spec.suites == frozenset({S_TWO_SI})
    # appended after diffmean (caa_static_r followed on 2026-08-21), so
    # existing decode indices are unmoved
    # the stage2-evals/cheap-estimator arms were appended after these at the merge;
    # indices +1 since the lora_r4 alias joined at index 4 (2026-08-28)
    assert list(ARMS)[24:27] == ["diffmean", "bidpo_lora", "caa_static_r"]
    # a preference arm shares no teacher forward: SOLO, never lockstepped
    assert not lockstep_compatible("bidpo_lora")


def test_arm_trains_through_the_registry(tiny_llama):
    model, tok = tiny_llama
    obj = ARMS["bidpo_lora"].train(
        model, tok, INSTRUCTION, _pairs(), layer=LAYER,
        epochs=1, batch_size=2, max_len=64, seed=0)
    assert type(obj) is RankOneLoRAObject
    # training moved the object off the exact no-op init
    assert obj.B.detach().abs().max() > 0


def test_arm_receives_bidpos_pair_routing_name():
    """The data-routing contract: pairs_for_arm special-cases the DPO
    preference arms by NAME — the registry name must match that frozenset
    exactly or bidpo_lora would silently train on the full pool."""
    import train_two_si_objects as tt

    assert "bidpo_lora" in tt.DPO_PREF_ARMS


def test_resolves_through_the_fit_wave_plan():
    """plan_si on the SHIPPED config: bidpo_lora plans SOLO at the DPO width
    (2 — four differentiable graphs per step, not lora_r1's one), and the
    newly added `vector` joins the lockstep group WITHOUT dragging its
    shared width below the measured 4."""
    import train_two_si_objects as tt

    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    arms = tt.default_class_a_arms(cfg)
    assert arms[-4:] == ["vector", "bidpo_lora", "token_faithful", "lora_r4"]
    for si in cfg["si_ids"]:
        plan = tt.plan_si(cfg, si, arms)
        assert "bidpo_lora" in plan["solo"]
        assert plan["solo"]["bidpo_lora"] == 2
        assert "vector" in plan["lockstep_group"]
        assert plan["lockstep_batch_size"] == 4


def test_deterministic_under_a_fixed_seed(tiny_llama):
    model, tok = tiny_llama
    prefs = _prefs()
    obj1 = train_bidpo_lora(model, tok, prefs, LAYER,
                            epochs=2, batch_size=2, max_len=64, seed=5)
    obj2 = train_bidpo_lora(model, tok, prefs, LAYER,
                            epochs=2, batch_size=2, max_len=64, seed=5)
    assert torch.equal(obj1.A.detach(), obj2.A.detach())
    assert torch.equal(obj1.B.detach(), obj2.B.detach())


# -- 5. position-masked (decode-only) fits ----------------------------------------

H = 2  # synthetic header allowance — the machinery is convention-agnostic


def test_masked_fit_is_accepted_and_restricts_the_edit(tiny_llama):
    """positions='response' trains without refusal, records the fit leg on
    the object, and the TRAINED object's edit is active at exactly the
    handle's mask: masked-out positions BIT-identical to the unsteered
    model, every steered real position changed (the PositionMaskedFit
    exactness guarantee, held with the trained weights)."""
    model, tok = tiny_llama
    prefs = _prefs()
    obj = train_bidpo_lora(model, tok, prefs, LAYER, epochs=1, lr=1e-2,
                           batch_size=2, max_len=64, seed=0,
                           positions="response", response_header_len=H)
    assert obj.fit_positions == "response"
    assert obj.fit_response_header_len == H
    assert obj.B.detach().abs().max() > 0  # trained off the no-op init

    seqs = [encode_prompt_completion(tok, p.prompt, p.chosen, 64)
            for p in prefs]
    batch = collate(seqs, tok.pad_token_id)
    layer_mod = get_decoder_layers(model)[LAYER]
    captured: dict[str, torch.Tensor] = {}

    def cap(key):
        def hook(_m, _a, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[key] = h.detach().clone()
        return hook

    with torch.no_grad():
        h1 = layer_mod.register_forward_hook(cap("base"))
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        h1.remove()
        with obj.apply(model, scale=1.0) as handle:
            # the masked leg yields its live tracker (the mask->loss seam)
            assert isinstance(handle, SteeringHandle)
            bounds = steer_prompt_boundaries(batch, handle.response_header_len)
            assert bool((bounds > 0).all())  # non-vacuous: a prompt prefix stays masked
            handle.set_prompt_lens(bounds)
            h2 = layer_mod.register_forward_hook(cap("steered"))
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
            h2.remove()
            b, s = batch.input_ids.shape
            mask = handle._position_mask(b, s, batch.input_ids.device).bool()
    off = ~mask
    on_real = mask & batch.attention_mask.bool()
    assert bool(off.any()) and bool(on_real.any())
    assert torch.equal(captured["steered"][off], captured["base"][off]), (
        "a masked-out position was not bit-identical to the unsteered model")
    diff = (captured["steered"] != captured["base"]).any(dim=-1)
    assert bool(diff[on_real].all()), "a steered position was left unedited"


def test_masked_fit_ties_dpo_loss_to_the_hook_mask(tiny_llama, monkeypatch):
    """Every completion-logprob gather of a masked fit — the frozen
    reference's (via the transient tracker) AND the policy's (via the live
    handle) — equals (handle mask ∩ completion mask), and the mask is the
    per-row RESPONSE mask (boundary = prompt_len - H). The unset-boundary
    trap (prompt_len left at 0) degenerates to an all-positions mask, which
    the expected-mask equality catches."""
    import sisteer.extraction.bidpo_lora as bl

    model, tok = tiny_llama
    records = []
    orig = bl.loss_gather_masks

    def rec(handle, batch, teacher_b=None):
        out = orig(handle, batch, teacher_b)
        b, s = batch.input_ids.shape
        m = handle._position_mask(b, s, batch.input_ids.device).bool()
        records.append((m, out[0].clone(), batch))
        return out

    monkeypatch.setattr(bl, "loss_gather_masks", rec)
    obj = RankOneLoRAObject(model, layer=LAYER)
    train_bidpo_object(model, tok, obj, _prefs(), epochs=1, batch_size=2,
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
        local = torch.arange(batch.input_ids.shape[1])[None, :]
        expected = local >= (batch.prompt_lens - H).clamp(min=0)[:, None]
        assert torch.equal(m, expected), (
            "the handle's mask is not the per-row response mask — the "
            "per-row boundary was never set (or misderived)")


def test_masked_fit_refused_for_an_object_without_a_mask(tiny_llama):
    """The engine keeps the refusal for objects exposing no fit mask
    (no set_fit_positions): never fit all-token under a decode-only
    label. The raise happens BEFORE any training compute."""
    model, tok = tiny_llama
    adapter = _VectorSubstrate(model, LAYER)
    with pytest.raises(ValueError, match="cannot honor"):
        train_bidpo_object(model, tok, adapter, _prefs(), epochs=1,
                           batch_size=2, max_len=64,
                           positions="response", response_header_len=H)


def test_masked_fit_differs_from_all_token(tiny_llama):
    """The mask actually reaches the training forwards: a decode-only fit
    trains DIFFERENT parameters from the all-token fit at the same seed.
    (The converse guarantee — the all-token leg unchanged by the mask
    wiring — is claim 3's drift alarm, which holds the default path to the
    untouched train_bidpo step for step.)

    Fitted at layer 0, NOT the usual LAYER=1, and necessarily so: layer 1
    is the tiny fixture's FINAL decoder layer, where an o_proj edit at a
    prompt position is position-local (no downstream attention follows) and
    can never reach a completion logit — so the masked and all-token fits
    coincide EXACTLY there (verified: max param delta 0.0), and the test
    would pass while testing nothing. Layer 0's prompt-position edits
    propagate through layer 1's attention into the completion logits, which
    is precisely the leakage channel decode-only masking exists to close —
    and what makes the two fits differ."""
    model, tok = tiny_llama
    prefs = _prefs()
    kw = dict(epochs=1, batch_size=2, max_len=64, seed=0)

    def flat(obj):
        return torch.cat([p.detach().flatten() for p in obj.trainable])

    masked = train_bidpo_lora(model, tok, prefs, 0,
                              positions="response", response_header_len=H,
                              **kw)
    plain = train_bidpo_lora(model, tok, prefs, 0, **kw)
    assert not torch.equal(flat(masked), flat(plain)), (
        "the masked bidpo_lora fit is indistinguishable from all-token — "
        "the mask never reached the training forward")


def test_masked_fit_trains_through_the_registry(tiny_llama):
    """The driver hands positions/response_header_len to the solo trainer
    as PROTECTED kwargs (train_two_si_objects never drops them in the
    TypeError-retry) — the registry trainer must accept them and produce a
    masked-leg artifact of lora_r1's type."""
    model, tok = tiny_llama
    obj = ARMS["bidpo_lora"].train(
        model, tok, INSTRUCTION, _pairs(), layer=LAYER, epochs=1,
        batch_size=2, max_len=64, seed=0, positions="response",
        response_header_len=H)
    assert type(obj) is RankOneLoRAObject
    assert obj.fit_positions == "response"
    assert obj.fit_response_header_len == H
    assert obj.B.detach().abs().max() > 0


# -- 6. saturation-keyed early stopping (the 2026-08-24 wave post-mortem fix) ----
#
# bidpo_lora was the arm the wave's floor guard failed at EVERY pool size
# (its margins saturate ~2x faster than the vector's), so the stop rule
# matters most on this substrate. The rule itself is train_bidpo's — these
# tests pin that the transcription carries it identically (claim 3's drift
# discipline) and that it works on the LoRA substrate the arm ships on.


def test_stop_rule_is_transcribed_not_divergent(tiny_llama):
    """Drift-alarm extension: with stop_pref_prob armed, both engines stop
    at the SAME step with identical per-step losses and bit-equal
    artifacts on the vector substrate — the stop rule is part of the
    transcribed objective, not a second diverging copy."""
    model, tok = tiny_llama
    prefs = _prefs()
    hp = dict(beta=0.5, lr=1e-2, weight_decay=0.05, epochs=8, batch_size=2,
              max_len=64, direction="both", seed=0, stop_pref_prob=0.51)

    bidpo_losses: list[float] = []
    sv = train_bidpo(
        model, tok, prefs, LAYER,
        log_fn=lambda m, step: bidpo_losses.append(m["bidpo/loss"])
        if "bidpo/loss" in m else None, **hp)

    adapter = _VectorSubstrate(model, LAYER)
    history = train_bidpo_object(model, tok, adapter, prefs, **hp)

    assert 0 < len(history) < 16, "the stop never fired — vacuous"
    assert history == bidpo_losses
    assert torch.equal(sv.vectors[LAYER],
                       adapter.vec.detach().float().cpu())


def test_stop_pref_prob_stops_on_the_lora_substrate(tiny_llama):
    """On the shipped substrate the rule truncates, never alters: the
    stopped run's history is a strict prefix of the fixed-budget run's,
    and the stopped object is bit-equal to training exactly the stopping
    epoch count (stopping keeps the stopping epoch's weights)."""
    model, tok = tiny_llama
    prefs = _prefs()
    # lr 5e-2, not the vector tests' 1e-2: the tiny fixture's rank-1 o_proj
    # LoRA moves margins ~5x slower than the vector substrate, and the stop
    # must actually fire inside the 8-epoch budget for the test to bite.
    kw = dict(epochs=8, lr=5e-2, beta=0.5, batch_size=2, max_len=64, seed=0)

    obj_full = RankOneLoRAObject(model, layer=LAYER)
    hist_full = train_bidpo_object(model, tok, obj_full, prefs, **kw)
    assert len(hist_full) == 16  # 3 examples at batch 2 -> 2 steps/epoch

    obj_stop = RankOneLoRAObject(model, layer=LAYER)
    hist_stop = train_bidpo_object(model, tok, obj_stop, prefs,
                                   stop_pref_prob=0.51, **kw)
    assert 0 < len(hist_stop) < 16, "the stop never fired — vacuous"
    assert len(hist_stop) % 2 == 0  # epoch-end is the event: whole epochs only
    assert hist_stop == hist_full[: len(hist_stop)], (
        "the stop rule changed the training trajectory — it may only cut "
        "it short")

    k = len(hist_stop) // 2
    exact = train_bidpo_lora(model, tok, prefs, LAYER, **{**kw, "epochs": k})
    assert torch.equal(obj_stop.A.detach(), exact.A.detach())
    assert torch.equal(obj_stop.B.detach(), exact.B.detach())


def test_stop_pref_prob_validated_by_the_engine(tiny_llama):
    """The engine carries train_bidpo's own (0.5, 1) refusal — before any
    training compute, on any substrate."""
    model, tok = tiny_llama
    obj = RankOneLoRAObject(model, layer=LAYER)
    with pytest.raises(ValueError, match="stop_pref_prob"):
        train_bidpo_object(model, tok, obj, _prefs(), stop_pref_prob=1.0,
                           epochs=1, batch_size=2, max_len=64)


# -- 7. weight decay on the LoRA substrate (2026-08-24 post-mortem) ---------------
#
# bidpo_lora was the arm that failed the wave's floor guard at EVERY pool
# size (9-52x) — the same unbounded-norm over-optimization as bidpo's, on
# the substrate that saturates ~2x faster. The default weight_decay moved
# 0.05 -> 2.0 (rationale and wave evidence on train_bidpo's docstring, the
# objective's home); the drift discipline (claim 3) requires both engines
# to carry the SAME optimizer recipe, so these tests pin the brake on this
# substrate AND the symmetry of the default itself.


def test_weight_decay_bounds_the_param_norm(tiny_llama):
    """Same fit, same seed, only the decay differs: the decayed object's
    trainable-parameter norm (what bidpo_lora/param_norm logs, what AdamW
    decays) must end strictly smaller, while both runs still train off the
    no-op (B nonzero — the brake shrinks the object, it does not zero the
    fit)."""
    model, tok = tiny_llama
    prefs = _prefs()
    kw = dict(epochs=6, lr=1e-2, batch_size=2, max_len=64, seed=0)

    def flat_norm(obj):
        return torch.cat(
            [p.detach().flatten() for p in obj.trainable]).norm()

    free = train_bidpo_lora(model, tok, prefs, LAYER, weight_decay=0.0, **kw)
    decayed = train_bidpo_lora(model, tok, prefs, LAYER, weight_decay=5.0,
                               **kw)
    assert free.B.detach().abs().max() > 0
    assert decayed.B.detach().abs().max() > 0
    assert flat_norm(decayed) < flat_norm(free), (
        "weight decay did not bound the adapter norm — the wave's runaway "
        "(floor failures at every pool size, 9-52x) would recur")


def test_weight_decay_pulls_toward_the_exact_noop_on_lora(tiny_llama):
    """Decay's fixed point (A = B = 0) lies inside this substrate's no-op
    set: with the zero-init B the low-rank update B A^T vanishes and
    apply()'s forward is BIT-identical to the unsteered model — so decay
    pulls toward "do nothing" here exactly as on the vector substrate
    (and unlike loreft, whose no-op is the nonzero rotation R and which
    pins wd=0 for that reason)."""
    model, tok = tiny_llama
    obj = RankOneLoRAObject(model, layer=LAYER)
    assert not obj.B.detach().any()  # the zero-init no-op, pre-verified
    seqs = [encode_prompt_completion(tok, p.prompt, p.chosen, 64)
            for p in _prefs()]
    batch = collate(seqs, tok.pad_token_id)
    with torch.no_grad():
        base = model(input_ids=batch.input_ids,
                     attention_mask=batch.attention_mask).logits
        with obj.apply(model, scale=1.0):
            steered = model(input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask).logits
    assert torch.equal(steered, base), (
        "B = 0 is not the exact no-op — decay would pull toward something "
        "other than the unsteered model (the loreft trap)")


def test_weight_decay_defaults_are_symmetric_across_the_engines():
    """The optimizer recipe travels with the OBJECTIVE (module docstring),
    so the 2026-08-24 default change must land on both trainers at once —
    a one-sided default would silently split the recipe UNDER the drift
    alarm's radar (the alarm passes its kwargs explicitly)."""
    tb = inspect.signature(train_bidpo).parameters
    assert tb["weight_decay"].default == 2.0
    for fn in (train_bidpo_object, train_bidpo_lora):
        sig = inspect.signature(fn).parameters
        for knob in ("weight_decay", "lr", "beta"):
            assert sig[knob].default == tb[knob].default, (
                f"{fn.__name__}.{knob} default drifted from train_bidpo's")


def test_engines_match_at_the_default_weight_decay(tiny_llama):
    """Drift-alarm extension for the default itself: run both engines with
    NO explicit weight_decay and require identical per-step losses and a
    bit-equal artifact on the vector substrate — the default is symmetric
    in EFFECT, not just in signature."""
    model, tok = tiny_llama
    prefs = _prefs()
    kw = dict(epochs=2, lr=1e-2, batch_size=2, max_len=64, seed=3)

    bidpo_losses: list[float] = []
    sv = train_bidpo(
        model, tok, prefs, LAYER,
        log_fn=lambda m, step: bidpo_losses.append(m["bidpo/loss"])
        if "bidpo/loss" in m else None, **kw)

    adapter = _VectorSubstrate(model, LAYER)
    history = train_bidpo_object(model, tok, adapter, prefs, **kw)

    assert history == bidpo_losses
    assert torch.equal(sv.vectors[LAYER],
                       adapter.vec.detach().float().cpu())
