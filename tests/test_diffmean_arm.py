"""The diffmean registry arm: the diff-in-means / CAA-family floor —
Stolfo (2410.12877) instruction-difference steering, equivalently the
static actuation direction r with the gate deleted.

Pins, per the task's test list:
- registration: in methods.ARMS (S_TWO_SI only, appended LAST), in
  defense_cell.CLASS_A_ARMS and the shipped config's class_a_arms, torch-blob
  artifact, lockstep-incompatible (solo) with a stated reason;
- the r REUSE: the trained vector is bitwise the alphakl recipe's raw r —
  the same build_chat_prompt(system=SI) vs system=None renderings and the
  same collect_pooled_activations(position="last") capture train_alphakl
  builds its r from (before alphakl's unit normalization, which this arm
  deliberately omits);
- save/load round-trip through methods.load_object, including the persisted
  response-mode deployment (the LoReFT persistence precedent);
- scale-0 bit-identity (and scale-1 non-vacuity);
- planted-direction recovery through diff_of_means, the arm's fit primitive
  (mirroring test_antidote's synthetic-clusters pattern);
- the masked (decode-only) leg: the object steers EXACTLY the handle's mask
  with masked-out positions bit-identical — SteeringHandle._position_mask is
  the one mask source, a plain addition being the exact case for it.

CPU, conftest tiny_llama, synthetic French pairs — never real store data.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.capture import (ActivationBatch,
                                        collect_pooled_activations)
from sisteer.extraction.diffmean import (DiffMeanObject, diff_of_means,
                                         train_diffmean)
from sisteer.extraction.distill import _encode_aligned, steer_prompt_boundaries
from sisteer.extraction.lockstep import (lockstep_compatible,
                                         lockstep_incompatible_reason)
from sisteer.extraction.traindata import collate
from sisteer.hooks import SteeringHandle
from sisteer.methods import ARMS, S_TWO_SI, load_object, object_artifact_name
from sisteer.models import build_chat_prompt, get_decoder_layers, hidden_size

INSTRUCTION = BY_ID["french"]
H = 2  # synthetic header allowance — the machinery is convention-agnostic
MAX_LEN = 96


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
    object's edits)."""
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


def _nonzero_object(model, scale_fill=5.0, layers=(1,)):
    d = hidden_size(model)
    return DiffMeanObject(model, {l: scale_fill * torch.ones(d)
                                  for l in layers})


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_diffmean_arm_wiring():
    """Registered, S_TWO_SI only (never a core suite — its cells derive from
    configs/two_si.yaml's class_a_arms), appended LAST so every existing
    positional decode keeps its meaning, torch-blob artifact, and a SOLO
    (lockstep-incompatible, closed-form) fit."""
    assert "diffmean" in ARMS
    assert ARMS["diffmean"].suites == frozenset({S_TWO_SI})
    # bidpo_lora, then the caa_static_r Class B floor (both 2026-08-21)
    # were appended AFTER diffmean, the same append-only discipline —
    # diffmean's own index is unmoved.
    # the stage2-evals/cheap-estimator arms were appended after these at the merge;
    # indices +1 since the 2026-08-28 lora_r4 alias joined at index 4 next
    # to its lora_r1_all sibling (the one audited non-append insertion,
    # see test_methods_s2's order test)
    assert list(ARMS)[24:27] == ["diffmean", "bidpo_lora", "caa_static_r"]
    assert object_artifact_name("diffmean") == "object.pt"
    assert not lockstep_compatible("diffmean")
    assert "diff-in-means" in lockstep_incompatible_reason("diffmean")

    from sisteer.eval.defense_cell import CLASS_A_ARMS
    from sisteer.eval.jailbreak import COMPILED

    # Append-only discipline: diffmean, then bidpo (2026-08-20),
    # then vector + bidpo_lora, then token_faithful (all 2026-08-21), then
    # lora_r4 (2026-08-28, the staged rank-4 objects' canonical cell id) —
    # each appended in sequence, so every earlier arm's index is unmoved.
    assert CLASS_A_ARMS[-10:] == ("diffmean", "bidpo", "vector", "bidpo_lora",
                                  "token_faithful", "lora_r4",
                                  # psr_xfer_* appended LAST (merged 2026-09-02)
                                  "psr_xfer_from_e2b", "psr_xfer_from_e2b_rw1",
                                  "psr_xfer_from_e4b", "psr_xfer_from_e4b_rw1")
    assert "diffmean" in COMPILED


def test_diffmean_in_the_shipped_config_matrix():
    """The shipped two_si.yaml enumerates diffmean as a Class A arm (appended
    last, then bidpo after it), and registry_from_cfg derives one
    cell per SI from it."""
    import yaml
    from pathlib import Path

    from sisteer.eval.defense_cell import registry_from_cfg

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / "configs" / "two_si.yaml").read_text())
    # vector + bidpo_lora, then token_faithful, were appended after
    # diffmean/bidpo on 2026-08-21 (and alphakl_fisher left the trained list
    # the same day — a mid-list removal, safe only because nothing in flight
    # depended on the indices).
    assert cfg["class_a_arms"][-6:] == ["diffmean", "bidpo", "vector",
                                        "bidpo_lora", "token_faithful",
                                        "lora_r4"]
    reg = registry_from_cfg(cfg, repo_root=repo)
    ids = {c.id for c in reg}
    for si in cfg["si_ids"]:
        assert f"{si}__diffmean" in ids


# ---------------------------------------------------------------------------
# the r reuse: bitwise the alphakl recipe's raw r
# ---------------------------------------------------------------------------


def test_trained_vector_is_alphakl_raw_r_bitwise(tiny_llama):
    """train_diffmean's vector == the exact quantity train_alphakl builds its
    r from — the same two build_chat_prompt renderings (system=SI vs
    system=None) and the same collect_pooled_activations(position='last')
    capture, mean-differenced — BEFORE alphakl's unit normalization (this arm
    keeps r raw so scale 1.0 adds exactly the mean SI-presence displacement).
    Same ops, same batching: bitwise, not approximately."""
    model, tok = tiny_llama
    pairs = _pairs()
    obj = train_diffmean(model, tok, INSTRUCTION, pairs, layer=1)

    prompts = [p.prompt for p in pairs]
    with_si = [build_chat_prompt(tok, p, system=INSTRUCTION.text)
               for p in prompts]                        # alphakl's benign side
    without = [build_chat_prompt(tok, p, system=None)
               for p in prompts]                        # alphakl's malicious side
    b_last = collect_pooled_activations(model, tok, with_si, [1],
                                        position="last").acts
    m_last = collect_pooled_activations(model, tok, without, [1],
                                        position="last").acts
    raw = b_last[1].mean(0) - m_last[1].mean(0)         # alphakl's raw r
    assert torch.equal(obj.vectors[1], raw)
    # and the direction is exactly alphakl's unit r
    unit = raw / raw.norm().clamp(min=1e-12)
    got = obj.vectors[1] / obj.vectors[1].norm().clamp(min=1e-12)
    assert torch.allclose(got, unit)
    # reliability diagnostics ride along from the tested primitive
    assert "1" in obj.metadata["diagnostics"]
    assert obj.metadata["n_pairs"] == len(pairs)
    assert obj.metadata["r_norms"][1] > 0


# ---------------------------------------------------------------------------
# planted-direction recovery through the fit primitive
# ---------------------------------------------------------------------------


def test_diff_of_means_recovers_planted_axis():
    """Two clusters separated along a known axis (test_antidote's synthetic
    pattern): the mean-difference vector — the arm's entire fit — recovers
    the planted axis, and the Braun diagnostics certify it."""
    g = torch.Generator().manual_seed(0)
    d, n, sep, noise = 64, 200, 8.0, 0.1
    axis = torch.zeros(d)
    axis[3] = 1.0
    base = torch.randn(n, d, generator=g) * noise
    pos = ActivationBatch({1: base + sep / 2 * axis})
    neg = ActivationBatch({1: torch.randn(n, d, generator=g) * noise
                           - sep / 2 * axis})
    sv = diff_of_means(pos, neg, method="instr_diff", model_name="m",
                       instruction_id="t")
    v = sv.vectors[1]
    cos = float(v @ axis / (v.norm() * axis.norm()))
    assert cos > 0.99
    diag = sv.metadata["diagnostics"]["1"]
    assert diag["d_prime"] > 10           # the planted separation dominates
    assert diag["directional_agreement"] > 0.9


# ---------------------------------------------------------------------------
# deployment: scale-0 bit-identity, exact addition, BOS exclusion
# ---------------------------------------------------------------------------


def test_scale_zero_is_bit_identical_and_scale_one_is_not(tiny_llama):
    model, tok = tiny_llama
    obj = _nonzero_object(model)
    enc = tok("the quick brown fox jumps", return_tensors="pt")
    with torch.no_grad():
        base = model(**enc).logits
        with obj.apply(model, scale=0.0):
            zeroed = model(**enc).logits
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
    assert torch.equal(base, zeroed), "scale-0 is not a no-op"
    assert not torch.allclose(base, steered), "scale-1 changed nothing"


def test_addition_is_exactly_the_vector_with_bos_excluded(tiny_llama):
    """The plain-vector deployment convention: h + scale*v at every real
    position EXCEPT the BOS (an ungated vector on the BOS outlier is the
    known blowup; this arm has no gate to down-weight it)."""
    model, tok = tiny_llama
    obj = _nonzero_object(model)
    enc = tok("alpha beta gamma delta", return_tensors="pt")
    with torch.no_grad(), _capture(model, [1]) as base:
        model(**enc)
    with torch.no_grad(), obj.apply(model, scale=2.0), \
            _capture(model, [1]) as steered:
        model(**enc)
    delta = steered[1] - base[1]
    v = 2.0 * obj.vectors[1]
    assert torch.equal(delta[0, 0], torch.zeros_like(delta[0, 0]))  # BOS
    for pos in range(1, delta.shape[1]):
        assert torch.allclose(delta[0, pos], v, atol=1e-5)


# ---------------------------------------------------------------------------
# save/load round-trip (incl. the persisted response-mode deployment)
# ---------------------------------------------------------------------------


def test_save_load_round_trip_through_load_object(tiny_llama, tmp_path):
    model, tok = tiny_llama
    d = hidden_size(model)
    g = torch.Generator().manual_seed(7)
    obj = DiffMeanObject(model, {1: 0.5 * torch.randn(d, generator=g)},
                         metadata={"n_pairs": 4})
    path = tmp_path / "object.pt"
    obj.save(path)
    loaded = load_object(path, model=model, tok=tok)
    assert isinstance(loaded, DiffMeanObject)
    assert loaded.kind == "diffmean"
    assert torch.equal(loaded.vectors[1], obj.vectors[1])
    assert loaded.metadata["n_pairs"] == 4
    assert loaded.fit_positions == "all"
    enc = tok(loaded.inject("the quick brown fox"), return_tensors="pt")
    with torch.no_grad():
        with obj.apply(model, scale=1.0):
            a = model(**enc).logits
        with loaded.apply(model, scale=1.0):
            b = model(**enc).logits
    assert torch.equal(a, b)


def test_response_mode_persists_through_save_load(tiny_llama, tmp_path):
    """A response-mode diffmean artifact must re-hydrate as response-mode
    (the LoReFT precedent: the deployment leg is part of the artifact's
    identity — re-hydrating it all-token would silently run the wrong leg)."""
    model, tok = tiny_llama
    obj = _nonzero_object(model)
    obj.set_fit_positions("response", H)
    p = tmp_path / "object.pt"
    obj.save(p)
    blob = torch.load(p, weights_only=True)
    assert blob["positions"] == "response"
    assert blob["response_header_len"] == H
    loaded = load_object(p, model=model, tok=tok)
    assert loaded.fit_positions == "response"
    assert loaded.fit_response_header_len == H

    default = _nonzero_object(model)
    p2 = tmp_path / "object_default.pt"
    default.save(p2)
    assert torch.load(p2, weights_only=True)["positions"] == "all"
    assert load_object(p2, model=model, tok=tok).fit_positions == "all"


# ---------------------------------------------------------------------------
# the masked (decode-only) leg: exactly the mask, off bit-identical
# ---------------------------------------------------------------------------


def test_masked_object_steers_exactly_the_mask(tiny_llama):
    """Response-mode diffmean steers the handle's mask positions and NOTHING
    else — masked-out positions bit-identical, every steered real position
    changed. The mask comes from SteeringHandle._position_mask (the one
    place a mask is computed), a plain addition being its exact case."""
    model, tok = tiny_llama
    batch = _student_batch(tok)
    obj = _nonzero_object(model)
    # default leg: an all-token vector arm (handle still yielded)
    with obj.apply(model, scale=1.0) as ctx:
        assert isinstance(ctx, SteeringHandle)
    obj.set_fit_positions("response", H)
    with torch.no_grad(), _capture(model, [1]) as base:
        model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    with torch.no_grad(), obj.apply(model, scale=1.0) as handle:
        assert isinstance(handle, SteeringHandle), (
            "a response-mode object's apply() must yield its mask handle")
        assert handle.positions == "response"
        bounds = steer_prompt_boundaries(batch, handle.response_header_len)
        assert bool((bounds > 0).all()), (
            "fixture prompt at or below H — that row's boundary collapsed "
            "to 0, so 'response' steers EVERY position and the test would "
            "pass while testing nothing (the peer-pinned width <= H trap)")
        handle.set_prompt_lens(bounds)
        with _capture(model, [1]) as steered:
            model(input_ids=batch.input_ids,
                  attention_mask=batch.attention_mask)
        b, s = batch.input_ids.shape
        mask = handle._position_mask(b, s, batch.input_ids.device).bool()
    on_real = mask & batch.attention_mask.bool()
    assert bool((~mask).any()), "fixture left nothing masked"
    assert bool(on_real.any()), "fixture left nothing steered"
    assert torch.equal(steered[1][~mask], base[1][~mask]), (
        "a masked-out position was not bit-identical")
    diff = (steered[1] != base[1]).any(dim=-1)
    assert bool(diff[on_real].all()), "a steered position was left unedited"


def test_response_mode_edits_decode_steps(tiny_llama):
    """The generated-only serving shape (ctx entered for the decode forward
    only): a response-mode diffmean EDITS a KV-cached decode step."""
    model, tok = tiny_llama
    enc = tok("one two three four five six", return_tensors="pt")
    nxt = enc["input_ids"][:, -1:]

    def fresh_past():
        with torch.no_grad():
            return model(**enc, use_cache=True).past_key_values

    with torch.no_grad():
        ref = model(input_ids=nxt, past_key_values=fresh_past()).logits
    obj = _nonzero_object(model)
    obj.set_fit_positions("response", H)
    with torch.no_grad(), obj.apply(model, scale=1.0):
        out = model(input_ids=nxt, past_key_values=fresh_past()).logits
    assert not torch.equal(out, ref), (
        "response-mode diffmean left a decode step unedited")


# ---------------------------------------------------------------------------
# the registry trainer end to end (the fit is position-independent; the
# response leg changes the DEPLOYMENT and is persisted + labelled honestly)
# ---------------------------------------------------------------------------


def test_registry_trainer_smoke_and_driver_kwargs(tiny_llama):
    """ARMS['diffmean'].train runs end-to-end with the two-SI driver's
    candidate kwargs (epochs/batch_size/max_len/seed are documented no-ops
    of the closed form) and yields a usable object at the pinned layer."""
    model, tok = tiny_llama
    obj = ARMS["diffmean"].train(
        model, tok, INSTRUCTION, _pairs(), layer=1,
        epochs=1, batch_size=2, max_len=MAX_LEN, seed=0)
    assert isinstance(obj, DiffMeanObject)
    assert obj.layers == [1]
    assert obj.n_params() == hidden_size(model)
    assert obj.trainable == []            # closed form: nothing trains
    enc = tok(obj.inject("the quick brown fox"), return_tensors="pt")
    with torch.no_grad(), obj.apply(model, scale=0.5):
        model(**enc)


def test_masked_fit_persists_mode_and_vector_is_the_all_token_one(tiny_llama):
    """positions='response' through the trainer persists the decode-only
    DEPLOYMENT — while the fitted vector is BITWISE the all-token leg's
    (closed form on unsteered activations), and the metadata says so, so
    the two legs' artifacts can never be misread as different fits."""
    model, tok = tiny_llama
    pairs = _pairs()
    plain = train_diffmean(model, tok, INSTRUCTION, pairs, layer=1)
    masked = train_diffmean(model, tok, INSTRUCTION, pairs, layer=1,
                            positions="response", response_header_len=H)
    assert masked.fit_positions == "response"
    assert masked.fit_response_header_len == H
    assert plain.fit_positions == "all"
    assert torch.equal(masked.vectors[1], plain.vectors[1])
    assert masked.metadata["fit_position_independent"] is True
    assert masked.metadata["fit_positions"] == "response"

    with pytest.raises(ValueError, match="fit positions"):
        train_diffmean(model, tok, INSTRUCTION, pairs, layer=1,
                       positions="front")
    with pytest.raises(ValueError, match="empty pairs"):
        train_diffmean(model, tok, INSTRUCTION, [], layer=1)
