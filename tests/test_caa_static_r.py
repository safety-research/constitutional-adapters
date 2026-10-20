"""caa_static_r — the CAA / diff-in-means Class B floor, now the peer's
MATCHED implementation (second round): the fit
internals, resid_pre harvest and per-layer fitted-sign assertion are
vendored VERBATIM from an earlier development branch @ 9c63c0d
(baseline_arms.train_static_r_matched + StaticRObject's prefill_all/site=pre
path + alphasteer._collect_last_acts); this lane keeps the arm id, artifact
kind, registry/config membership, Class B sample-floor records, and the
SI-agnosticism contract.

Pins, per the task's contract: the fitted-sign assertion (INVERTED vs
degenerate, raising — never a silently backwards arm), capture AND apply
both at resid_pre THROUGH THE SHARED VENDORED HELPER (never an l−1 index
map), prefill-only (no decode-step edit), broadcast to every prompt
position, the RAW un-normalised harmful−benign direction, the paper
ten-layer set, the deployed operating point lam = DEPLOYED_LAM = 0.05
(deliberately below the peer's suggested grid), scale-0
bit-identity, and SI-agnosticism (system=None on every render — a property
of THIS lane's cell that survives the swap).

Behavioural tests are ported from the peer's tests/test_static_r_matched.py
@ 9c63c0d where they exist, and kept from this lane's pre-swap file where
they still apply. CPU-only, tiny-llama fixture (2 layers, d=16) plus an
in-test 20-layer random Llama for the real ten-paper-layer fit; synthetic
data throughout.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fit_class_b as fcb  # noqa: E402

from sisteer.data.instructions import BY_ID  # noqa: E402
from sisteer.data.pairs import Pair  # noqa: E402
from sisteer.extraction import caa_static_r as caa  # noqa: E402
from sisteer.extraction.caa_static_r import (  # noqa: E402
    DEPLOYED_LAM,
    PAPER_LLAMA31_LAYERS,
    RECOMMENDED_LAM_GRID,
    CaaStaticRObject,
    _collect_last_acts,
    train_caa_static_r_content,
)
from sisteer.extraction.alphasteer import refusal_direction  # noqa: E402
from sisteer.methods import ARMS, load_object, object_artifact_name  # noqa: E402
from sisteer.models import (  # noqa: E402
    build_chat_prompt,
    get_decoder_layers,
    hidden_size,
)

BENIGN = ["Write a haiku about the sea.", "Explain photosynthesis.",
          "Name three cats.", "Summarize Hamlet."]
HARMFUL = ["Do a harmful thing.", "Do another harmful thing."]

# The tiny fixture has 2 decoder layers; the vendored fit harvests resid_pre
# of the steered layer DIRECTLY (site="pre"), no l-1 index map anywhere.
TINY_LAYERS = (1,)


@pytest.fixture(scope="module")
def deep_tiny_llama(tiny_llama):
    """A 20-layer random Llama (tiny dims, tiny_llama's tokenizer) so the
    fit can run at the REAL ten paper layers [8..19]. Ported from the
    peer's test file @ 9c63c0d."""
    from transformers import LlamaConfig, LlamaForCausalLM

    _, tok = tiny_llama
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=max(len(tok), tok.vocab_size),
                      hidden_size=16, intermediate_size=32,
                      num_hidden_layers=20, num_attention_heads=4,
                      num_key_value_heads=4)
    model = LlamaForCausalLM(cfg).eval()
    return model, tok


def _base_logits(model, enc):
    with torch.no_grad():
        return model(**enc).logits


# ------------------------------------------ the fit: raw, signed, resid_pre


def test_ten_layer_fit_per_layer_directions_and_metadata(deep_tiny_llama):
    """The campaign shape (ported from the peer @ 9c63c0d): ten paper
    layers, each with its OWN raw r_l; per-layer norms and separations
    recorded in metadata and equal to the live tensors'; plus this lane's
    metadata fields and the deployed base_scale."""
    model, tok = deep_tiny_llama
    layers = list(PAPER_LLAMA31_LAYERS)
    obj = train_caa_static_r_content(model, tok, BENIGN, HARMFUL,
                                     layers=layers, batch_size=2)
    assert obj.layers == layers
    assert sorted(obj.rs) == sorted(layers)
    md = obj.metadata
    assert md["site"] == "pre"
    assert md["arm"] == "caa_static_r"
    assert md["layers"] == layers
    assert md["n_benign"] == len(BENIGN)
    assert md["n_harmful"] == len(HARMFUL)
    assert md["direction"] == "raw_diff_in_means_harmful_minus_benign"
    assert md["position_convention"] == "prefill_all"
    assert sorted(md["per_layer_r_norm"]) == sorted(layers)
    for l in layers:
        assert md["per_layer_r_norm"][l] == float(obj.rs[l].norm())
        assert md["per_layer_mean_separation"][l] > 0.0
    # each layer really got its OWN direction
    assert not torch.equal(obj.rs[8], obj.rs[19])
    # this lane's fields survive the swap
    assert md["deployment"] == caa.DEPLOYMENT
    assert md["deployment_matched_to"] == "alphasteer_paper_10layer"
    assert set(md["sample_floor"]) == {"benign", "harmful"}
    # the deployed operating point is baked into the object
    assert obj.base_scale == DEPLOYED_LAM


def test_direction_is_raw_diff_in_means_not_unit(tiny_llama):
    """r_l must be alphasteer.refusal_direction's RAW output — bitwise —
    over the SAME harvest the trainer uses (the vendored _collect_last_acts
    at site='pre': capture == apply site, matched by shared code), and NOT
    unit-normalised. Ported from the peer @ 9c63c0d."""
    model, tok = tiny_llama
    obj = train_caa_static_r_content(model, tok, BENIGN, HARMFUL,
                                     layers=[1], batch_size=2)
    benign_texts = [build_chat_prompt(tok, p, system=None) for p in BENIGN]
    harmful_texts = [build_chat_prompt(tok, p, system=None) for p in HARMFUL]
    ben = _collect_last_acts(model, tok, benign_texts, [1], "pre", 2)
    har = _collect_last_acts(model, tok, harmful_texts, [1], "pre", 2)
    want = refusal_direction(har[1], ben[1])
    assert torch.equal(obj.rs[1], want)                 # RAW, bit-for-bit
    norm = float(obj.rs[1].norm())
    assert abs(norm - 1.0) > 1e-3                       # not unit-normalised
    assert not torch.allclose(obj.rs[1], want / want.norm())


def test_trainer_harvests_steered_layers_at_pre_site_via_shared_helper(
        tiny_llama, monkeypatch):
    """THE regression the swap exists to kill: the trainer's harvest goes
    through the vendored shared helper at site='pre' with the STEERED
    layer list — never the replaced implementation's decoder-output-(l-1)
    map, whose capture/apply matching was by argument, not by code."""
    model, tok = tiny_llama
    calls = []
    real = caa._collect_last_acts

    def spy(model_, tok_, texts, layers, site, batch_size):
        calls.append({"layers": list(layers), "site": site})
        return real(model_, tok_, texts, layers, site, batch_size)

    monkeypatch.setattr(caa, "_collect_last_acts", spy)
    train_caa_static_r_content(model, tok, BENIGN, HARMFUL,
                               layers=TINY_LAYERS, batch_size=2)
    assert len(calls) == 2                        # one benign, one harmful
    for c in calls:
        assert c["site"] == "pre"                 # the DEPLOY site
        assert c["layers"] == list(TINY_LAYERS)   # steered layers, NOT l-1


def test_sign_guard_fails_loudly_on_inverted_or_degenerate_direction(
        tiny_llama, monkeypatch):
    """Positive lam must push toward refusal (r = harmful − benign). A
    direction that comes out inverted — a sign-convention regression in
    refusal_direction or swapped arguments — or degenerate must raise, not
    ship a silently backwards arm. Ported verbatim-in-spirit from the peer
    @ 9c63c0d (the replaced implementation had NO inverted-direction guard
    and would have shipped it silently)."""
    model, tok = tiny_llama
    real = caa.refusal_direction
    monkeypatch.setattr(caa, "refusal_direction",
                        lambda h, b: -real(h, b))
    with pytest.raises(ValueError, match="INVERTED"):
        train_caa_static_r_content(model, tok, BENIGN, HARMFUL, layers=[1],
                                   batch_size=2)
    monkeypatch.setattr(caa, "refusal_direction",
                        lambda h, b: torch.zeros_like(real(h, b)))
    with pytest.raises(ValueError, match="degenerate"):
        train_caa_static_r_content(model, tok, BENIGN, HARMFUL, layers=[1],
                                   batch_size=2)
    # and the degeneracy guard fires FOR REAL at layer 0: resid_pre there
    # is the raw embedding, and the shared final template token makes the
    # harmful and benign last-token means coincide exactly
    monkeypatch.setattr(caa, "refusal_direction", real)
    with pytest.raises(ValueError, match="degenerate"):
        train_caa_static_r_content(model, tok, BENIGN, HARMFUL, layers=[0],
                                   batch_size=2)


def test_trainer_validation(tiny_llama):
    """The vendored validation matrix @ 9c63c0d. NOTE layer 0 is now VALID
    input (resid_pre of block 0 is the embedding stream) — the replaced
    implementation refused it structurally (its l-1 capture map needed
    l >= 1); the vendored fit instead range-checks against the model depth
    and lets the degeneracy guard catch the embedding case (test above)."""
    model, tok = tiny_llama
    with pytest.raises(ValueError, match="non-empty"):
        train_caa_static_r_content(model, tok, [], HARMFUL, layers=[1])
    with pytest.raises(ValueError, match="non-empty"):
        train_caa_static_r_content(model, tok, BENIGN, [], layers=[1])
    with pytest.raises(ValueError, match="non-empty list"):
        train_caa_static_r_content(model, tok, BENIGN, HARMFUL, layers=[])
    with pytest.raises(ValueError, match="duplicate"):
        train_caa_static_r_content(model, tok, BENIGN, HARMFUL,
                                   layers=[1, 1])
    with pytest.raises(ValueError, match="out of range"):
        train_caa_static_r_content(model, tok, BENIGN, HARMFUL,
                                   layers=[0, 99])


def test_default_layers_are_the_paper_set():
    """The matched deployment's identity: the AlphaSteer paper's ten-layer
    Llama-3.1-8B set, not one layer, and not including 15 (the campaign's
    single-layer pin does not apply to this arm — its exemption is
    documented in fit_class_b.class_b_layer and the config)."""
    assert PAPER_LLAMA31_LAYERS == (8, 9, 10, 11, 12, 13, 14, 16, 18, 19)
    sig = inspect.signature(train_caa_static_r_content)
    assert sig.parameters["layers"].default == PAPER_LLAMA31_LAYERS


# ------------------------------------------- the deployed operating point


def test_deployed_lam_is_user_specified_0p05(tiny_llama):
    """lam = 0.05 is the DEPLOYED operating point:
    the trainer default bakes it into base_scale, so serving at the
    pre-registered dose (apply scale=1.0) writes exactly 0.05·r_l. It is
    DELIBERATELY below the peer's suggested grid — a user-specified
    operating point, not a swept default; do not 'correct' it into the
    grid or back to the old default of 1.0."""
    assert DEPLOYED_LAM == 0.05
    sig = inspect.signature(train_caa_static_r_content)
    assert sig.parameters["base_scale"].default == DEPLOYED_LAM
    assert DEPLOYED_LAM < min(RECOMMENDED_LAM_GRID)   # below the grid, on purpose

    model, tok = tiny_llama
    obj = train_caa_static_r_content(model, tok, BENIGN, HARMFUL,
                                     layers=TINY_LAYERS, batch_size=2)
    assert obj.base_scale == DEPLOYED_LAM
    assert obj.metadata["deployed_lam"] == DEPLOYED_LAM
    # the live edit at the default dose is exactly 0.05 * r
    layer = TINY_LAYERS[0]
    enc = tok("the quick brown fox jumps over the lazy dog",
              return_tensors="pt")
    base, h1 = _resid_pre_probe(model, layer)
    with torch.no_grad():
        model(**enc)
    h1.remove()
    with obj.apply(model, scale=1.0):
        steered, h2 = _resid_pre_probe(model, layer)
        with torch.no_grad():
            model(**enc)
        h2.remove()
    delta = steered["h"] - base["h"]
    expected = DEPLOYED_LAM * obj.rs[layer]
    assert torch.allclose(delta, expected.view(1, 1, -1).expand_as(delta),
                          atol=1e-6)
    # and the roundtripped artifact carries the operating point
    assert float(obj.rs[layer].norm()) > 0            # non-vacuous


# ------------------------------------------------------------ SI-agnosticism


def test_fit_renders_si_free_no_system_prompt(tiny_llama, monkeypatch):
    """The Class B contract (THIS lane's cell property — it survives the
    swap; the vendored fit renders system=None too): every prompt is
    rendered with system=None — no SI text can reach the fit through any
    path."""
    model, tok = tiny_llama
    seen_systems = []
    real = caa.build_chat_prompt

    def spy(tok_, user, system=None):
        seen_systems.append(system)
        return real(tok_, user, system=system)

    monkeypatch.setattr(caa, "build_chat_prompt", spy)
    train_caa_static_r_content(model, tok, BENIGN, HARMFUL,
                               layers=TINY_LAYERS, batch_size=4)
    assert len(seen_systems) == len(BENIGN) + len(HARMFUL)
    assert all(s is None for s in seen_systems)


def test_registry_shim_ignores_instruction_and_layer_pin(
        tiny_llama, monkeypatch):
    """ARMS['caa_static_r'].train takes the uniform (instruction, pairs,
    layer) surface but DELIBERATELY ignores all three — the render stays
    system-free even when an SI is handed in, and the single-layer pin
    never reaches the ten-layer fit (the lofit precedent)."""
    model, tok = tiny_llama
    seen_systems = []
    real = caa.build_chat_prompt

    def spy(tok_, user, system=None):
        seen_systems.append(system)
        return real(tok_, user, system=system)

    monkeypatch.setattr(caa, "build_chat_prompt", spy)
    pairs = [Pair(prompt="Say something nice.", pos="a", neg="b",
                  instruction_id="french")]
    obj = ARMS["caa_static_r"].train(
        model, tok, BY_ID["french"], pairs, layer=0,   # pin ignored
        benign_prompts=BENIGN, harmful_prompts=HARMFUL,
        layers=TINY_LAYERS, batch_size=4)
    assert isinstance(obj, CaaStaticRObject)
    assert obj.layers == list(TINY_LAYERS)             # not the layer=0 pin
    assert all(s is None for s in seen_systems)
    # the pairs' prompts never entered the harvest — only the content lists
    assert len(seen_systems) == len(BENIGN) + len(HARMFUL)


def test_registry_shim_accepts_prompt_files_and_records_shas(
        tiny_llama, tmp_path):
    """The peer's registry-shim contract: benign_prompts_path /
    harmful_prompts_path point the arm straight at the two JSON files, and
    the consumed files' sha256s land in the object's provenance."""
    model, tok = tiny_llama
    bp = tmp_path / "benign.json"
    hp = tmp_path / "harmful.json"
    bp.write_text(json.dumps(BENIGN))
    hp.write_text(json.dumps(HARMFUL))
    obj = ARMS["caa_static_r"].train(
        model, tok, BY_ID["french"], [], layer=1,
        benign_prompts_path=str(bp), harmful_prompts_path=str(hp),
        min_prompts=1, layers=TINY_LAYERS, batch_size=4)
    meta = obj.metadata["content_contrast"]
    assert meta["n_benign"] == len(BENIGN)
    assert meta["n_harmful"] == len(HARMFUL)
    assert meta["benign_sha256"] == hashlib.sha256(bp.read_bytes()).hexdigest()
    assert meta["harmful_sha256"] == hashlib.sha256(hp.read_bytes()).hexdigest()


def test_registry_shim_refuses_without_content():
    """No lists and no files -> a loud refusal naming the two surfaces (a
    Class B arm must never silently fit on the (instruction, pairs) pool)."""
    with pytest.raises(ValueError, match="Class B content fit"):
        ARMS["caa_static_r"].train(None, None, BY_ID["french"], [], layer=1)


# ---------------------------------------------------- deployment (the point)


def _resid_pre_probe(model, layer):
    """Observer PRE-hook capturing the LAYER's INPUT hidden states
    (resid_pre). Registered AFTER the object's pre-hook (inside apply()),
    it sees the edited stream — pre-hooks chain in registration order."""
    captured = {}

    def probe(_mod, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        captured["h"] = hidden.detach().clone()

    handle = get_decoder_layers(model)[layer].register_forward_pre_hook(
        probe, with_kwargs=True)
    return captured, handle


def test_prefill_broadcasts_raw_r_to_all_prompt_positions_at_resid_pre(
        tiny_llama):
    """THE deployment pin (matched to AlphaSteer, diverging from the CAA
    paper's last-token write on purpose): during prefill, EVERY prompt
    position of the steered layer's INPUT (resid_pre, before
    input_layernorm) moves by exactly base_scale·scale·r — not just the
    last token, and r's RAW magnitude is used as-is."""
    model, tok = tiny_llama
    d = hidden_size(model)
    g = torch.Generator().manual_seed(7)
    r = 3.7 * torch.randn(d, generator=g)           # deliberately NOT unit
    layer = 1
    obj = CaaStaticRObject(model, {layer: r}, base_scale=2.0)
    enc = tok("the quick brown fox jumps over the lazy dog",
              return_tensors="pt")
    seq_len = enc["input_ids"].shape[1]
    assert seq_len > 2  # a real multi-token prefill

    base, h1 = _resid_pre_probe(model, layer)
    with torch.no_grad():
        model(**enc)
    h1.remove()
    with obj.apply(model, scale=1.5):
        steered, h2 = _resid_pre_probe(model, layer)  # after the object's hook
        with torch.no_grad():
            model(**enc)
        h2.remove()

    delta = steered["h"] - base["h"]                 # [1, S, d]
    expected = (2.0 * 1.5) * r                       # base_scale·scale·r, RAW
    for pos in range(seq_len):                       # ALL positions
        assert torch.allclose(delta[0, pos], expected, atol=1e-4), pos
    # and identically across a left-padded batch (generation convention)
    prev_side, prev_pad = tok.padding_side, tok.pad_token
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        batch = tok(["short prompt", "a much longer prompt with many words"],
                    return_tensors="pt", padding=True)
        b_base, h3 = _resid_pre_probe(model, layer)
        with torch.no_grad():
            model(**batch)
        h3.remove()
        with obj.apply(model, scale=1.0):
            b_steer, h4 = _resid_pre_probe(model, layer)
            with torch.no_grad():
                model(**batch)
            h4.remove()
    finally:
        tok.padding_side, tok.pad_token = prev_side, prev_pad
    b_delta = b_steer["h"] - b_base["h"]
    assert torch.allclose(
        b_delta, (2.0 * r).view(1, 1, -1).expand_as(b_delta), atol=1e-4)


def test_multi_layer_hooks_every_steered_layer_at_resid_pre(tiny_llama):
    """A multi-layer object steers every listed layer: the FIRST steered
    layer's input moves by exactly mag·r_first (nothing upstream), each
    steered layer wears exactly one forward PRE-hook (resid_pre, never a
    forward/output hook — that would be resid_post, the site this arm
    deliberately left), and the logits move."""
    model, tok = tiny_llama
    d = hidden_size(model)
    g = torch.Generator().manual_seed(9)
    rs = {0: torch.randn(d, generator=g), 1: torch.randn(d, generator=g)}
    obj = CaaStaticRObject(model, rs, base_scale=1.0)
    assert obj.layers == [0, 1]
    assert obj.n_params() == 2 * d
    enc = tok("the quick brown fox", return_tensors="pt")
    base = _base_logits(model, enc)

    layers = get_decoder_layers(model)
    pre_before = [len(l._forward_pre_hooks) for l in layers]
    fwd_before = [len(l._forward_hooks) for l in layers]
    first, h1 = _resid_pre_probe(model, 0)
    with torch.no_grad():
        model(**enc)
    h1.remove()
    with obj.apply(model, scale=1.0):
        # one PRE-hook per steered layer; no forward (output) hooks added
        assert [len(l._forward_pre_hooks) for l in layers] == [
            n + 1 for n in pre_before]
        assert [len(l._forward_hooks) for l in layers] == fwd_before
        steered_first, h2 = _resid_pre_probe(model, 0)
        with torch.no_grad():
            steered_logits = model(**enc).logits
        h2.remove()
    # hooks removed on exit
    assert [len(l._forward_pre_hooks) for l in layers] == pre_before
    delta = steered_first["h"] - first["h"]
    assert torch.allclose(delta, rs[0].view(1, 1, -1).expand_as(delta),
                          atol=1e-4)
    assert not torch.allclose(base, steered_logits)


def test_decode_step_is_untouched_prefill_only(tiny_llama):
    """seq_len == 1 forwards (KV-cached decode steps) are returned untouched
    — bit-identical logits — exactly AlphaSteer's prefill-only gate."""
    model, tok = tiny_llama
    d = hidden_size(model)
    r = torch.zeros(d)
    r[0] = 5.0
    obj = CaaStaticRObject(model, {0: r, 1: r}, base_scale=5.0)
    one = tok("hello", return_tensors="pt")
    one = {k: v[:, :1] for k, v in one.items()}
    base = _base_logits(model, one)
    with torch.no_grad(), obj.apply(model, scale=1.0):
        decode = model(**one).logits
    assert torch.equal(base, decode), "caa_static_r must not steer decode steps"


def test_decode_inside_apply_matches_outside_with_steered_cache(tiny_llama):
    """Ported from the peer @ 9c63c0d: a decode step run INSIDE apply()
    (hook installed) off a steered cache is bit-identical to the same
    decode step run OUTSIDE apply() off an identically steered cache — the
    hook must not touch seq-len-1 forwards; the edit reaches decode only
    through the KV cache."""
    model, tok = tiny_llama
    d = hidden_size(model)
    g = torch.Generator().manual_seed(5)
    obj = CaaStaticRObject(model, {0: torch.randn(d, generator=g)},
                           base_scale=3.0)
    ids = tok("The quick brown fox jumps over", return_tensors="pt")[
        "input_ids"]
    with torch.no_grad():
        with obj.apply(model, scale=1.0):
            steered = model(input_ids=ids, use_cache=True)
        nxt = steered.logits[:, -1].argmax(-1, keepdim=True)
        with obj.apply(model, scale=1.0):
            inside = model(input_ids=nxt,
                           past_key_values=steered.past_key_values,
                           use_cache=True)
        # same decode step, hook not installed: must be bit-identical
        with obj.apply(model, scale=1.0):
            steered2 = model(input_ids=ids, use_cache=True)
        outside = model(input_ids=nxt,
                        past_key_values=steered2.past_key_values,
                        use_cache=True)
    assert torch.equal(inside.logits, outside.logits)


def test_prefill_kv_persistence_matches_manual_reforward(tiny_llama):
    """Ported from the peer @ 9c63c0d: the prefill edit must reach decode
    steps through the KV cache, matching a manual re-forward that edits
    every PROMPT position (and only prompt positions) of the full
    [prompt + next-token] sequence at the same resid_pre site."""
    model, tok = tiny_llama
    d = hidden_size(model)
    lam = 3.0
    g = torch.Generator().manual_seed(9)
    r = torch.randn(d, generator=g)
    # steer layer 0, NOT the top layer (an edit above every attention
    # block never enters the cache)
    obj = CaaStaticRObject(model, {0: r}, base_scale=lam)
    ids = tok("The quick brown fox jumps over", return_tensors="pt")[
        "input_ids"]
    T = ids.shape[1]
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=True)
        with obj.apply(model, scale=1.0):
            steered = model(input_ids=ids, use_cache=True)
        nxt = steered.logits[:, -1].argmax(-1, keepdim=True)
        with obj.apply(model, scale=1.0):
            step = model(input_ids=nxt,
                         past_key_values=steered.past_key_values,
                         use_cache=True)
        base_step = model(input_ids=nxt,
                          past_key_values=base.past_key_values,
                          use_cache=True)

        # manual re-forward: the same resid_pre edit at positions [0, T) —
        # the prompt — of the full sequence, and NOT at decode position T
        def ref_hook(_mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            edited = hidden.clone()
            edited[:, :T, :] = (hidden[:, :T, :].float()
                                + lam * r.float()).to(hidden.dtype)
            if "hidden_states" in kwargs:
                return (args, {**kwargs, "hidden_states": edited})
            return ((edited,) + tuple(args[1:]), kwargs)

        h = get_decoder_layers(model)[0].register_forward_pre_hook(
            ref_hook, with_kwargs=True)
        try:
            full = model(input_ids=torch.cat([ids, nxt], dim=1),
                         use_cache=False)
        finally:
            h.remove()
    assert torch.allclose(step.logits[:, -1], full.logits[:, -1],
                          atol=1e-4, rtol=1e-4)
    # persistence is non-vacuous: the steered cache changes decode logits
    assert not torch.allclose(step.logits[:, -1], base_step.logits[:, -1],
                              atol=1e-6)


def test_scale0_exact_noop_and_object_protocol(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    g = torch.Generator().manual_seed(11)
    rs = {1: torch.randn(d, generator=g)}
    obj = CaaStaticRObject(model, rs, base_scale=1.0)
    assert obj.kind == "caa_static_r"
    assert obj.trainable == []          # closed-form: nothing gradient-trains
    assert obj.n_params() == d
    assert obj.inject("prompt text") == "prompt text"
    enc = tok("the quick brown fox jumps", return_tensors="pt")
    base = _base_logits(model, enc)
    with torch.no_grad():
        with obj.apply(model, scale=0.0):
            zeroed = model(**enc).logits
        with obj.apply(model, scale=1.0):
            steered = model(**enc).logits
    assert torch.equal(base, zeroed)    # scale-0 is bit-exact unsteered
    assert not torch.allclose(base, steered)  # ... and not vacuously so


def test_save_load_blob_roundtrip(tiny_llama, tmp_path):
    model, tok = tiny_llama
    d = hidden_size(model)
    g = torch.Generator().manual_seed(13)
    rs = {0: torch.randn(d, generator=g), 1: torch.randn(d, generator=g)}
    obj = CaaStaticRObject(model, rs, base_scale=DEPLOYED_LAM,
                           metadata={"per_layer_r_norm": {0: 1.1, 1: 4.2},
                                     "deployment": caa.DEPLOYMENT})
    path = tmp_path / "object.pt"
    obj.save(path)
    loaded = load_object(path, model, tok)   # _KIND_LOADERS dispatch
    assert isinstance(loaded, CaaStaticRObject)
    assert loaded.layers == [0, 1]
    assert loaded.base_scale == DEPLOYED_LAM   # the operating point survives
    for l in (0, 1):
        assert torch.allclose(loaded.rs[l].cpu(), obj.rs[l].cpu())
    assert loaded.metadata["deployment"] == caa.DEPLOYMENT
    enc = tok("the quick brown fox", return_tensors="pt")
    with torch.no_grad():
        with obj.apply(model, scale=1.0):
            orig = model(**enc).logits
        with loaded.apply(model, scale=1.0):
            rehydrated = model(**enc).logits
    assert torch.allclose(orig, rehydrated, atol=1e-6)


# ----------------------------------------------------- Class B fit contract


def test_sha_pins_enforced_on_the_fit_path(tiny_llama, tmp_path):
    """The Class B sha256 pins gate THIS arm's fit set exactly as they gate
    the other two: matching pins assemble; a mismatched pin refuses LOUDLY
    before any arm trains."""
    _, tok = tiny_llama
    bp = tmp_path / "benign_alphasteer.json"
    hp = tmp_path / "harmful_alphasteer.json"
    bp.write_text(json.dumps(BENIGN))
    hp.write_text(json.dumps(HARMFUL))
    cfg = {
        "defenses_dir": str(tmp_path / "defenses"),
        "layer": 1,
        "class_b_arms": ["antidote", "alphasteer", "caa_static_r"],
        "class_b": {
            "benign_prompts_path": str(bp),
            "harmful_prompts_path": str(hp),
            "benign_sha256": hashlib.sha256(bp.read_bytes()).hexdigest(),
            "harmful_sha256": hashlib.sha256(hp.read_bytes()).hexdigest(),
            "min_prompts": 1, "batch_size": 4,
        },
    }
    data = fcb.build_class_b_data(cfg, tok)   # pins match -> assembles
    assert data["benign_prompts"] == BENIGN
    # a wrong pin refuses loudly, naming the side
    bad = json.loads(json.dumps(cfg))
    bad["class_b"]["harmful_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="harmful_sha256 pin mismatch"):
        fcb.build_class_b_data(bad, tok)


def test_fit_class_b_fits_and_saves_caa_static_r(tiny_llama, tmp_path,
                                                 monkeypatch):
    """The production driver path end-to-end on synthetic content: the
    default caa_static_r closure IGNORES the driver's single-layer pin
    (fitting the module's layer set — shrunk here to the tiny model), and
    the saved object re-hydrates with the recorded deployment, the vendored
    per-layer norms, AND the deployed base_scale of DEPLOYED_LAM = 0.05;
    meta.json records the REALISED layers from the object, not just the
    driver pin."""
    model, tok = tiny_llama
    cfg = {"defenses_dir": str(tmp_path / "defenses"), "layer": 1,
           "class_b": {"batch_size": 4, "min_prompts": 1}}
    from sisteer.data.content_sets import ContentContrast
    import sisteer.extraction.caa_static_r as caa_mod

    # the ten-layer paper default does not fit a 2-layer tiny model; shrink
    # the default via the module seam (the driver closure imports the
    # trainer at call time, and passes no layers, by design)
    real = caa_mod.train_caa_static_r_content

    def tiny_default(*args, **kw):
        kw.setdefault("layers", TINY_LAYERS)
        return real(*args, **kw)

    monkeypatch.setattr(caa_mod, "train_caa_static_r_content", tiny_default)

    data = fcb.build_class_b_data(
        cfg, tok, content=ContentContrast(benign=list(BENIGN),
                                          harmful=list(HARMFUL), meta={}))
    status = fcb.fit_and_save(cfg, model, tok, data, arms=["caa_static_r"])
    assert status["caa_static_r"]["state"] == "fit"
    path = fcb._object_path(cfg, "caa_static_r")
    obj = load_object(path, model, tok)
    assert isinstance(obj, CaaStaticRObject)
    assert obj.layers == list(TINY_LAYERS)
    assert obj.base_scale == DEPLOYED_LAM     # lam=0.05 reaches the artifact
    for l in obj.layers:
        assert obj.metadata["per_layer_r_norm"][l] > 0
        assert obj.metadata["per_layer_mean_separation"][l] > 0
    assert obj.metadata["deployment"] == caa.DEPLOYMENT
    meta = json.loads((path.parent / "meta.json").read_text())
    assert meta["class"] == "B" and meta["si_id"] is None
    assert meta["layers"] == list(TINY_LAYERS)   # the realised layer set


def test_shipped_config_appends_caa_last_and_keeps_the_others_pin():
    """The shipped two-SI config: caa_static_r is a Class B arm, APPENDED
    LAST (order is load-bearing); the antidote/alphasteer single-layer pin
    stays 15 and caa_static_r's exemption is the module's paper set — the
    peer's layer-16 artifact is deliberately not copied, and the withdrawn
    single-layer form is deliberately not built."""
    import yaml

    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    assert cfg["class_b_arms"] == ["antidote", "alphasteer", "caa_static_r"]
    assert fcb.class_b_layer(cfg) == 15          # the OTHER two arms' pin
    assert fcb.default_class_b_arms(cfg)[-1] == "caa_static_r"


# ------------------------------------------------------------------- wiring


def test_registry_wiring_appended_last_two_si_only():
    """Appended LAST (after diffmean / bidpo_lora), tagged S_TWO_SI ONLY so
    it enters no stage-2 positional array decode; artifact is the torch-blob
    default; the kind hydrates through _KIND_LOADERS; and the Class B tuples
    (defense_cell + fit driver) agree, caa_static_r LAST in both."""
    from sisteer.eval.defense_cell import CLASS_B_ARMS
    from sisteer.methods import _KIND_LOADERS

    # the stage2-evals/cheap-estimator arms were appended after it at the merge;
    # index +1 since the lora_r4 alias joined at index 4 (2026-08-28)
    assert list(ARMS)[26] == "caa_static_r"
    assert ARMS["caa_static_r"].suites == frozenset({"two_si"})
    assert object_artifact_name("caa_static_r") == "object.pt"
    assert "caa_static_r" in _KIND_LOADERS
    assert CLASS_B_ARMS == ("antidote", "alphasteer", "caa_static_r")
    assert tuple(fcb.CLASS_B_ARMS) == CLASS_B_ARMS
    # NOT named `static_r`: on the earlier development branch that name is
    # the preserved LAST-TOKEN form; the deployment difference is
    # load-bearing, so the matched arm wears its own name.
    assert "static_r" not in ARMS
