"""The probe FORWARD-PAIR (phase_probe_gpu) end to end on the tiny CPU model.

Two real bugs this file exists to pin (both caught 2026-08-21, one by the first
GPU validation run, one by its CPU repro):

1. REGION CONVENTION: ``decode_probe_capture`` must return the FULL prefill width
   as ``prompt_len`` (measure_footprint's convention — the header is the prompt's
   LAST H positions). Returning the content boundary slid the header window onto
   unsteered content and mislabelled a clean position-mask serve as MIXED
   (measured (none, none, all)), refusing a correct serve.
2. CAPTURE-HOOK ORDER: the capture hook must register AFTER the steering ctx's
   hooks — forward hooks run in registration order, so a capture registered first
   at a STEERED layer records PRE-steering hidden and the probe reads a no-op.

The tiny fixture tokenizer gets real control tokens + a Llama-3-style template so
chat_markers and hooks.response_header_len both resolve a real header.
"""

from __future__ import annotations

import pytest
import torch

_TEMPLATE = (
    "{% for m in messages %}<|start_header_id|>{{ m['role'] }}"
    "<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant"
    "<|end_header_id|>\n\n{% endif %}"
)


@pytest.fixture()
def chat_tiny(tiny_llama):
    """tiny_llama with REAL control tokens + a header-bearing template (module-
    scoped fixtures must not be mutated — copy the additions per test)."""
    model, tok = tiny_llama
    added = tok.add_special_tokens(
        {"additional_special_tokens": ["<|start_header_id|>",
                                       "<|end_header_id|>", "<|eot_id|>"]})
    if added:
        model.resize_token_embeddings(len(tok))
    old_template = tok.chat_template
    tok.chat_template = _TEMPLATE
    yield model, tok
    tok.chat_template = old_template


def _vector_obj(model, layer, tok, positions):
    from sisteer.hooks import response_header_len
    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    v = torch.randn(d, generator=g)
    v = v / v.norm()
    sv = SteeringVector(vectors={layer: v}, method="synth", model_name="",
                        instruction_id="")
    H = response_header_len(tok)
    return VectorObject(sv, positions=positions, skip_bos=True,
                        response_header_len=(H if positions == "response" else None))


def _run(model, tok, obj):
    from sisteer.eval.phase_probe import changed_positions, measure_footprint
    from sisteer.eval.phase_probe_gpu import decode_probe_capture
    prompt_len, seq_len, header_len, ignore, capture = decode_probe_capture(
        model, tok, obj, scale=8.0)
    base, steer = capture(False), capture(True)
    changed = changed_positions(base, steer)
    fp = measure_footprint(base, steer, prompt_len=prompt_len, seq_len=seq_len,
                           header_len=header_len, ignore_positions=ignore)
    return prompt_len, seq_len, header_len, changed, fp


def test_position_mask_serve_measures_response_and_token1_steered(chat_tiny):
    """A position-mask (response) serve steers the assistant header + every
    decode step; the measured footprint is RESPONSE (content untouched) and
    the position producing token 1 (the last prompt/header token) is in the
    changed set."""
    from sisteer.eval.phase_probe import RESPONSE
    from sisteer.eval.phase_probe_gpu import token1_position
    model, tok = chat_tiny
    obj = _vector_obj(model, 0, tok, "response")
    prompt_len, seq_len, header_len, changed, fp = _run(model, tok, obj)
    assert fp.label() == RESPONSE, fp
    assert fp.content in ("none", "n/a")           # content untouched
    assert fp.header == "all" and fp.generated == "all"
    assert token1_position(prompt_len, header_len) in changed   # token 1 steered
    # regression (bug 1): prompt_len is the FULL width — the header region
    # [prompt_len - H, prompt_len) is inside the prompt, entirely changed.
    hdr = set(range(prompt_len - header_len, prompt_len))
    assert hdr <= changed


def test_all_position_serve_measures_all(chat_tiny):
    from sisteer.eval.phase_probe import ALL
    model, tok = chat_tiny
    obj = _vector_obj(model, 0, tok, "all")
    _, _, _, changed, fp = _run(model, tok, obj)
    assert fp.label() == ALL, fp
    assert fp.content == "all"                     # content genuinely steered
    assert 0 not in changed                        # skip_bos: BOS untouched


def test_capture_at_the_steered_layer_still_sees_the_steering(chat_tiny):
    """Regression (bug 2): steering the LAST layer — the capture layer — must
    still measure the footprint. A capture hook registered before the ctx runs
    first and records PRE-steering hidden, reading a no-op."""
    from sisteer.eval.phase_probe import RESPONSE
    model, tok = chat_tiny
    last = model.config.num_hidden_layers - 1
    obj = _vector_obj(model, last, tok, "response")
    _, _, _, changed, fp = _run(model, tok, obj)
    assert fp.label() == RESPONSE, fp
    assert changed, "capture at the steered layer read a no-op (hook order bug)"


def test_probe_agrees_with_hookfree_ground_truth(chat_tiny):
    """CORROBORATION between two differently-shaped instruments (the pattern that
    caught every wrong reading on 2026-08-21): the probe's capture-hook
    measurement must agree, position for position over the prefill, with a
    HOOK-FREE ground truth (``output_hidden_states`` + logits — the model's own
    outputs, sharing no code with the capture machinery). Also pins the stale-diag
    shape: a capture registered BEFORE the ctx at the steered layer reads [] even
    though the ground truth shows the positions steered."""
    import torch
    from sisteer.eval.phase_probe import changed_positions
    from sisteer.eval.phase_probe_gpu import (build_probe_sequence,
                                              decode_probe_capture)
    from sisteer.models import get_decoder_layers
    model, tok = chat_tiny
    obj = _vector_obj(model, 0, tok, "response")
    prompt_ids, _cont, pw, hl, _b, _s = build_probe_sequence(tok)
    prompt_t = torch.tensor([prompt_ids])

    # the probe's measurement (capture hooks, last layer, inside ctx)
    prompt_len, _sl, _hl, _ig, capture = decode_probe_capture(model, tok, obj,
                                                              scale=8.0)
    probe_prefill = {i for i in changed_positions(capture(False), capture(True))
                     if i < pw}

    # hook-free ground truth: the model's own final hidden + logits
    with torch.no_grad():
        base = model(input_ids=prompt_t, output_hidden_states=True)
        with obj.apply(model, scale=8.0) as h:
            h.reset(prompt_len=pw)
            steer = model(input_ids=prompt_t, output_hidden_states=True)
    gt = {i for i in range(pw) if float(
        (steer.hidden_states[-1][0, i] - base.hidden_states[-1][0, i])
        .abs().max()) > 1e-5}
    gt_logits = {i for i in range(pw) if float(
        (steer.logits[0, i] - base.logits[0, i]).abs().max()) > 1e-5}

    assert probe_prefill == gt == gt_logits          # instruments agree
    assert (pw - 1) in gt_logits                     # token 1 steered, by logits

    # the STALE diag shape (capture registered BEFORE the ctx at the steered
    # layer) reads [] on the same known-steered serve — never trust it.
    cap = []
    hk = get_decoder_layers(model)[0].register_forward_hook(
        lambda m, i, o: cap.append(
            (o[0] if isinstance(o, tuple) else o).detach().float()))
    try:
        with torch.no_grad():
            model(input_ids=prompt_t)
            base_h = cap[-1][0]
            with obj.apply(model, scale=8.0) as h:
                h.reset(prompt_len=pw)
                model(input_ids=prompt_t)
            steer_h = cap[-1][0]
    finally:
        hk.remove()
    stale = changed_positions([base_h[i] for i in range(pw)],
                              [steer_h[i] for i in range(pw)])
    assert stale == frozenset(), "the pre-ctx capture should read PRE-steering"


def test_engine_style_cross_phase_serve_hits_the_prompt_len_zero_trap(chat_tiny):
    """WHY the benign (capeval-engine) leg is REFUSED for cross-phase conditions
    rather than converted to the position mask: the engine's vector ``positions``
    override sets ``obj.positions='response'`` with NO response-boundary wiring
    (no ``response_header_len``, no ``reset(prompt_len=...)`` — hf_engine.ctx() is
    bare ``obj.apply``). Measured by footprint probe, that serve steers EVERY
    position (the prompt_len-0 trap): not the position mask, not even the engine
    gate — a third, unintended intervention. Refusal is the only correct benign
    behaviour for cross-phase conditions until the capeval engine grows real
    boundary wiring."""
    import torch
    from sisteer.eval.phase_probe import changed_positions
    from sisteer.eval.phase_probe_gpu import build_probe_sequence
    from sisteer.models import get_decoder_layers
    model, tok = chat_tiny
    from sisteer.methods import VectorObject
    from sisteer.vectors import SteeringVector
    d = model.config.hidden_size
    g = torch.Generator().manual_seed(0)
    v = torch.randn(d, generator=g)
    v = v / v.norm()
    sv = SteeringVector(vectors={0: v}, method="synth", model_name="",
                        instruction_id="")
    # EXACTLY the engine's construction for spec.positions='response': the mask
    # mode set, the boundary NOT wired, the ctx entered with no reset.
    obj = VectorObject(sv, positions="response", skip_bos=True,
                       response_header_len=None)
    prompt_ids, _c, pw, _h, _b, _s = build_probe_sequence(tok)
    prompt_t = torch.tensor([prompt_ids])

    def forward(applied):
        cap = []
        hook = (lambda m, i, o: cap.append(
            (o[0] if isinstance(o, tuple) else o).detach().float()))
        with torch.no_grad():
            if applied:
                with obj.apply(model, scale=8.0):
                    hk = get_decoder_layers(model)[-1].register_forward_hook(hook)
                    model(input_ids=prompt_t)
                    hk.remove()
            else:
                hk = get_decoder_layers(model)[-1].register_forward_hook(hook)
                model(input_ids=prompt_t)
                hk.remove()
        return [cap[0][0, i] for i in range(pw)]

    changed = changed_positions(forward(False), forward(True))
    # the trap: EVERYTHING except the BOS is steered — content included.
    assert changed == frozenset(range(1, pw)), (
        "the unwired engine-style serve was expected to steer every non-BOS "
        f"position (the prompt_len-0 trap); changed={sorted(changed)}")


def test_mixed_length_batch_masks_each_rows_header_under_left_padding(chat_tiny):
    """THE BATCHING HAZARD: batched generation left-pads, so prompt_width is
    the PADDED width — if the response boundary used each row's UNPADDED
    length, a short row's mask would land in padding and the steering would
    silently be a no-op (generation succeeds, numbers are baseline). Verified
    per row with a deliberately MIXED-LENGTH batch: LEFT padding right-aligns
    every row, so each row's assistant header IS its last H columns and the
    scalar boundary (padded_width - H) is exact per row — asserting for BOTH
    rows that exactly the header columns are steered, the content is
    untouched, and the SHORT row's padding columns are untouched."""
    import torch
    from sisteer.generate import encode_kwargs
    from sisteer.hooks import response_header_len
    from sisteer.models import get_decoder_layers
    model, tok = chat_tiny
    H = response_header_len(tok)
    obj = _vector_obj(model, 0, tok, "response")

    def render(user_text):
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False,
            add_generation_prompt=True)

    rendered = [render("Hi"), render(
        "Please explain in detail how rainbows form after a storm passes")]
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(rendered, return_tensors="pt", padding=True,
                  **encode_kwargs(tok, rendered[0]))
    finally:
        tok.padding_side = prev
    W = enc["input_ids"].shape[1]
    real_lens = enc["attention_mask"].sum(dim=1).tolist()
    assert real_lens[0] < real_lens[1], "the batch must be genuinely mixed-length"

    def final_hidden(applied):
        cap = []
        hook = (lambda m, i, o: cap.append(
            (o[0] if isinstance(o, tuple) else o).detach().float()))
        with torch.no_grad():
            if applied:
                with obj.apply(model, scale=8.0) as h:
                    hk = get_decoder_layers(model)[-1].register_forward_hook(hook)
                    h.reset(prompt_len=W)
                    model(**enc)
                    hk.remove()
            else:
                hk = get_decoder_layers(model)[-1].register_forward_hook(hook)
                model(**enc)
                hk.remove()
        return cap[0]  # [2, W, d]

    from sisteer.eval.phase_probe import classify_batched_row_mask
    diff = (final_hidden(True) - final_hidden(False)).abs().amax(dim=-1)  # [2, W]
    header_cols = set(range(W - H, W))
    for row, real_len in enumerate(real_lens):
        pad_cols = set(range(W - int(real_len)))
        cls = classify_batched_row_mask(diff[row].tolist(), header_cols=header_cols,
                                        pad_cols=pad_cols)
        assert cls["header_steered_exactly"], (
            f"row {row} (real_len={real_len}): header not steered exactly {cls}")
        assert not cls["mask_in_padding"], (
            f"row {row}: mask landed in padding — silent-baseline hazard {cls}")
        # the separation must be real: header steered at scale, padding untouched.
        assert cls["min_header_diff"] > cls["max_pad_diff"]


def test_classify_batched_row_mask_separates_steering_from_kernel_noise():
    """The magnitude-aware classifier (finding, slurm 173568): a binary atol
    conflates real steering with H200 bf16 batch-non-invariance noise on
    causally-independent columns. Sub-threshold padding noise must PASS (not a
    mis-mask); supra-threshold padding steering must FAIL (a real mis-mask)."""
    from sisteer.eval.phase_probe import classify_batched_row_mask
    W, H = 10, 3
    header = set(range(W - H, W))            # {7,8,9}
    pad = set(range(4))                      # {0,1,2,3} left padding
    # header steered at scale ~8; padding carries only bf16 kernel NOISE (~1e-3).
    diff = [0.0] * W
    for c in header:
        diff[c] = 8.0
    for c in pad:
        diff[c] = 1e-3
    ok = classify_batched_row_mask(diff, header_cols=header, pad_cols=pad)
    assert ok["header_steered_exactly"] and not ok["mask_in_padding"]
    assert ok["max_pad_diff"] < ok["min_header_diff"]
    # a REAL mis-mask: the boundary landed in padding, steering it at scale.
    bad = [0.0] * W
    for c in (2, 3, *header):
        bad[c] = 8.0
    res = classify_batched_row_mask(bad, header_cols=header, pad_cols=pad)
    assert res["mask_in_padding"] and not res["header_steered_exactly"]


def test_left_padding_coupling_is_enforced_at_the_batch_build_point(chat_tiny):
    """The left-padding coupling made ENFORCED: the response-mask boundary
    (prompt_width - H) is correct only under left padding; right/mixed padding
    would land a short row's mask in padding and silently serve baseline.
    assert_left_padded (called by generate_batch at the batch-build point)
    turns that implicit coupling into an enforced one. Measured on a REAL
    mixed-length batch: flipping the tokenizer to RIGHT padding produces a
    mask that RAISES; LEFT padding does not; and the actual batched serving
    path (defense.generate) stays left-padded so it never trips its own
    tripwire."""
    import pytest
    import torch
    from sisteer.generate import assert_left_padded, encode_kwargs
    from sisteer.eval.jailbreak import build_defense
    model, tok = chat_tiny

    def render(t):
        return tok.apply_chat_template([{"role": "user", "content": t}],
                                       tokenize=False, add_generation_prompt=True)
    rendered = [render("Hi"), render("A considerably longer user request here")]

    def mask_for(side):
        prev = tok.padding_side
        tok.padding_side = side
        try:
            enc = tok(rendered, return_tensors="pt", padding=True,
                      **encode_kwargs(tok, rendered[0]))
        finally:
            tok.padding_side = prev
        return enc["attention_mask"]

    # a genuinely mixed-length batch, so right padding really does trail a short
    # row (a same-length batch would be trivially safe either way).
    assert mask_for("left").shape[1] > 0
    with pytest.raises(RuntimeError, match="LEFT-padded"):
        assert_left_padded(mask_for("right"), what="test")
    assert_left_padded(mask_for("left"), what="test")   # must not raise
    # a single-row batch is unambiguous — never raises regardless of side.
    one = tok([render("Hi")], return_tensors="pt",
              **encode_kwargs(tok, render("Hi")))["attention_mask"]
    assert_left_padded(one, what="test")

    # the real batched serving path forces left padding, so it never trips the
    # tripwire on a mixed-length batch (this would raise if generate_batch's
    # padding_side coupling were ever broken).
    d = build_defense(model, tok, mode="none", positions="all", batch_size=8)
    out = d.generate(["Hi", "A considerably longer user request here"],
                     max_new_tokens=2)
    assert len(out) == 2


# --------------------------------------------------------------------------
# gemma-pattern header: located != generation-prompt header (pt829, 2026-08-28)
# --------------------------------------------------------------------------

_GEMMA_PATTERN_TEMPLATE = (
    # a user turn whose CLOSE carries post-terminator scaffold (<|scaf|>) —
    # the role gemma-4's trailing '\n' plays after '<turn|>'. derive_assistant_
    # markers anchors the located header at the terminator and ABSORBS that
    # scaffold; the generation-prompt header (what add_generation_prompt
    # appends) does NOT. So located_header = generation_header + 1, exactly the
    # gemma disagreement — reproduced with a special token so nothing
    # retokenizes across the boundary on the tiny tokenizer.
    "{% for m in messages %}<|hs|>{{ m['role'] }}<|he|>{{ m['content'] }}"
    "<|eot|><|scaf|>{% endfor %}"
    "{% if add_generation_prompt %}<|hs|>assistant<|he|>{% endif %}"
)


@pytest.fixture()
def chat_gemma_pattern(tiny_llama):
    """tiny_llama with a template whose LOCATED assistant header is one token
    wider than the GENERATION-PROMPT header (post-terminator scaffold), the
    gemma-4 shape."""
    model, tok = tiny_llama
    added = tok.add_special_tokens(
        {"additional_special_tokens": ["<|hs|>", "<|he|>", "<|eot|>",
                                       "<|scaf|>"]})
    if added:
        model.resize_token_embeddings(len(tok))
    old = tok.chat_template
    tok.chat_template = _GEMMA_PATTERN_TEMPLATE
    yield model, tok
    tok.chat_template = old


def test_gemma_pattern_located_header_is_wider_than_generation(chat_gemma_pattern):
    """Setup validity: on the gemma pattern the two header definitions DIVERGE
    (the whole reason the probe must not use the located one)."""
    from sisteer.chat_markers import (derive_assistant_markers,
                                      generation_header_ids)
    _model, tok = chat_gemma_pattern
    located = derive_assistant_markers(tok)
    gen = generation_header_ids(tok, located)
    assert len(gen) < len(located.header), (list(gen), list(located.header))


def test_probe_header_region_uses_the_generation_header_not_the_locator(
        chat_gemma_pattern):
    """The fix (pt829): build_probe_sequence must size the HEADER region by the
    GENERATION-PROMPT header (== the serve-time mask boundary,
    hooks.response_header_len), never derive_assistant_markers' located header.
    Before the fix it used the located header — one token wider on gemma — so a
    clean response serve left that extra prompt-side token unsteered and the
    measured footprint read header='partial', refusing every gemma
    fitted/transferred cell as MISLABELLED."""
    from sisteer.chat_markers import generation_header_ids
    from sisteer.eval.phase_probe_gpu import build_probe_sequence
    _model, tok = chat_gemma_pattern
    _pids, _cont, _pw, header_len, _cb, _sl = build_probe_sequence(tok)
    assert header_len == len(generation_header_ids(tok))


def test_gemma_pattern_response_serve_measures_all_header_not_partial(
        chat_gemma_pattern):
    """End to end: a response-masked object on the gemma pattern measures a
    CLEAN RESPONSE footprint (header='all', content untouched) — the object's
    mask (response_header_len) and the probe's header region now agree, so the
    leading scaffold token sits in CONTENT (unsteered) and the header is fully
    covered. Pre-fix this measured header='partial' and refused."""
    from sisteer.eval.phase_probe import RESPONSE
    model, tok = chat_gemma_pattern
    obj = _vector_obj(model, 0, tok, "response")
    _pl, _sl, header_len, changed, fp = _run(model, tok, obj)
    assert fp.header == "all", fp
    assert fp.content in ("none", "n/a"), fp
    assert fp.generated == "all", fp
    assert fp.label() == RESPONSE, fp
