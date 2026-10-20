"""Response-only steering: the inference-time ``positions="response"`` option.

Steers ONLY the model's generated tokens plus the assistant-header scaffolding
after the user's turn, leaving the system+user prompt untouched. This is a
DELIBERATE off-distribution train/test mismatch: training applied the
intervention at ALL positions and only restricted the LOSS to completion
tokens, so the objects were fit assuming the prompt is steered too.

CPU only (tiny-llama for the mechanics; the real Llama-3.1 chat template, from
the local cache, for the scaffolding-boundary test). The tests are deliberately
NON-VACUOUS: the trap is that ``prompt_len`` defaults to 0, so a mis-wired
``positions="response"`` silently steers EVERYTHING and looks like it worked.
Each mechanics test asserts a REAL behavioural split (prompt bit-identical,
response changed) AND that the same config with ``prompt_len=0`` would steer the
prompt too — i.e. proves the test would catch the default-0 trap.

Coverage:
 (a) vector arm (SteeringHandle): prompt bit-identical, response changed; the
     prompt_len=0 trap is caught.
 (b) PSR hook (new masking code): same, for a MULTI-LAYER PSRObject.
 (c) decode-step continuity: prefill + several KV-cached decode forwards; every
     decode position is steered under response mode (a broken running offset
     shows up here).
 (d) positions="all" is BYTE-IDENTICAL to unconditional steering (regression
     guard) for BOTH the vector and PSR paths.
 (e) the scaffolding boundary on the real Llama-3.1 template: the assistant
     header IS steered and the user text is NOT.
"""

import pytest
import torch

from sisteer.hooks import steering
from sisteer.models import get_decoder_layers, hidden_size, num_layers

# A minimal chat template with a real assistant-header suffix. The tiny test
# tokenizer ships NO template, so response_header_len() correctly returns 0 and
# the H==0 guard refuses -- which is the guard working, not a test bug. Give the
# fixture a header so the test exercises the real path instead of the refusal.
_TINY_CHAT_TEMPLATE = (
    "{% for m in messages %}<|start_header_id|>{{ m['role'] }}"
    "<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant"
    "<|end_header_id|>\n\n{% endif %}"
)


def _with_header(monkeypatch):
    """Make response_header_len report a real header WITHOUT mutating the
    shared tiny_llama tokenizer. Setting tok.chat_template directly leaked
    into unrelated tests (the fixture is shared), so patch the function
    instead and let monkeypatch undo it."""
    monkeypatch.setattr(
        "sisteer.eval.jailbreak.response_header_len", lambda tok: 4
    )



def _final_hidden(model, enc, ctx=None):
    """Last-layer residual stream [B, S, d] under an optional steering context."""
    from contextlib import nullcontext

    with (ctx or nullcontext()):
        return model(**enc, output_hidden_states=True).hidden_states[-1]


def _psr(model, layers, seed=0):
    from sisteer.extraction.psr import PSRObject

    g = torch.Generator().manual_seed(seed)
    d = hidden_size(model)
    obj = PSRObject(model, layers, train_direction=False)
    obj.set_directions({l: torch.randn(d, generator=g) for l in layers})
    for l in layers:
        obj.gate_w[l].data = torch.randn(d, generator=g)
        obj.gate_b[l].data = torch.ones(1)  # gate starts >0 so masking is visible
    return obj


# -- (a) anti-vacuity: vector arm ---------------------------------------------


def test_anti_vacuity_response_vector(tiny_llama):
    """positions='response' with a NONZERO prompt_len leaves prompt hidden
    states BIT-IDENTICAL and changes response positions; prompt_len=0 steers
    everything — proving the test catches the default-0 trap."""
    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(0)
    vec = torch.randn(d) * 5.0
    enc = tok("one two three four five six", return_tensors="pt")
    n = enc["input_ids"].shape[1]
    K = 3  # nonzero boundary strictly inside the sequence
    assert 0 < K < n

    base = _final_hidden(model, enc)
    with steering(model, {1: vec}, scale=3.0, positions="response",
                  prompt_len=K) as h:
        h.reset(prompt_len=K)
        steered = _final_hidden(model, enc)

    # prompt positions [:K] are BIT-IDENTICAL (masked add is + exactly 0, and
    # causal attention means p<K depends only on unsteered positions <= p).
    assert torch.equal(base[:, :K], steered[:, :K])
    # response positions [K:] genuinely change.
    assert not torch.allclose(base[:, K:], steered[:, K:])

    # THE TRAP: prompt_len=0 makes response steer EVERYTHING, so even position 0
    # differs. This is exactly what a vacuous test would miss.
    with steering(model, {1: vec}, scale=3.0, positions="response",
                  prompt_len=0) as h:
        h.reset(prompt_len=0)
        trap = _final_hidden(model, enc)
    assert not torch.equal(base[:, :K], trap[:, :K])
    assert not torch.allclose(base[:, 0], trap[:, 0])


def test_response_header_len_derives_boundary_vector(tiny_llama):
    """The generation path sets response_header_len (H), not prompt_len: the
    boundary is DERIVED per prefill as prefill_width - H. With H set, the last H
    columns (the assistant-header suffix) plus generation are steered and the
    rest is bit-identical."""
    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(7)
    vec = torch.randn(d) * 5.0
    enc = tok("alpha beta gamma delta epsilon", return_tensors="pt")
    n = enc["input_ids"].shape[1]
    H = 2
    base = _final_hidden(model, enc)
    with steering(model, {1: vec}, scale=3.0, positions="response",
                  response_header_len=H) as h:
        steered = _final_hidden(model, enc)  # no reset needed: prefill derives it
    assert torch.equal(base[:, : n - H], steered[:, : n - H])
    assert not torch.allclose(base[:, n - H:], steered[:, n - H:])


# -- (b) anti-vacuity: PSR hook (new masking code) -----------------------------


def test_anti_vacuity_response_psr(tiny_llama):
    """Same non-vacuous split for the PSR hook, MULTI-LAYER: prompt positions
    bit-identical, response positions changed; prompt_len=0 steers the prompt
    too."""
    model, tok = tiny_llama
    layers = list(range(num_layers(model)))
    assert len(layers) >= 2  # multi-layer
    obj = _psr(model, layers, seed=1)
    enc = tok("alpha beta gamma delta", return_tensors="pt")
    n = enc["input_ids"].shape[1]
    K = 2
    assert 0 < K < n

    base = _final_hidden(model, enc)
    obj.positions = "response"
    obj.prompt_len = K
    steered = _final_hidden(model, enc, ctx=obj.apply(model, scale=3.0))
    assert torch.equal(base[:, :K], steered[:, :K])
    assert not torch.allclose(base[:, K:], steered[:, K:])

    # trap: prompt_len=0 -> everything steered (position 0 changes).
    obj.prompt_len = 0
    trap = _final_hidden(model, enc, ctx=obj.apply(model, scale=3.0))
    assert not torch.equal(base[:, :K], trap[:, :K])
    assert not torch.allclose(base[:, 0], trap[:, 0])


def test_response_header_len_derives_boundary_psr(tiny_llama):
    """PSR generation path: response_header_len (H) derives the boundary from
    each prefill's width, exactly like the vector arm."""
    model, tok = tiny_llama
    layers = list(range(num_layers(model)))
    obj = _psr(model, layers, seed=2)
    enc = tok("alpha beta gamma delta epsilon", return_tensors="pt")
    n = enc["input_ids"].shape[1]
    H = 2
    base = _final_hidden(model, enc)
    obj.positions = "response"
    obj.response_header_len = H
    steered = _final_hidden(model, enc, ctx=obj.apply(model, scale=3.0))
    assert torch.equal(base[:, : n - H], steered[:, : n - H])
    assert not torch.allclose(base[:, n - H:], steered[:, n - H:])


# -- (c) decode-step continuity ------------------------------------------------


def _decode_run(model, enc, fixed_tokens, ctx=None):
    """Prefill then several KV-cached S==1 decode forwards over a FIXED token
    sequence (identical inputs on the steered and unsteered runs, so a logits
    difference is purely the steering). Returns [prefill_logits, step0, ...]."""
    from contextlib import nullcontext

    with torch.no_grad(), (ctx or nullcontext()):
        out = model(**enc, use_cache=True)
        outs = [out.logits]
        past = out.past_key_values
        for t in fixed_tokens:
            step = model(input_ids=torch.tensor([[t]]), past_key_values=past)
            past = step.past_key_values
            outs.append(step.logits)
    return outs


def test_decode_continuity_vector(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(3)
    vec = torch.randn(d) * 5.0
    enc = tok("a b c d", return_tensors="pt")
    n = enc["input_ids"].shape[1]
    fixed = [enc["input_ids"][0, -1].item(), 5, 7, 9]

    base = _decode_run(model, enc, fixed)
    ctx = steering(model, {1: vec}, scale=4.0, positions="response", prompt_len=n)
    steered = _decode_run(model, enc, fixed, ctx=ctx)

    assert torch.allclose(base[0], steered[0])  # prefill unsteered (prompt_len=n)
    for i in range(1, len(base)):
        assert not torch.allclose(base[i], steered[i]), f"decode step {i-1} unsteered"


def test_decode_continuity_psr(tiny_llama):
    model, tok = tiny_llama
    layers = list(range(num_layers(model)))
    obj = _psr(model, layers, seed=4)
    enc = tok("a b c d", return_tensors="pt")
    n = enc["input_ids"].shape[1]
    fixed = [enc["input_ids"][0, -1].item(), 5, 7, 9]

    base = _decode_run(model, enc, fixed)
    obj.positions = "response"
    obj.prompt_len = n
    steered = _decode_run(model, enc, fixed, ctx=obj.apply(model, scale=5.0))

    assert torch.allclose(base[0], steered[0])  # prefill unsteered
    for i in range(1, len(base)):
        assert not torch.allclose(base[i], steered[i]), f"decode step {i-1} unsteered"


# -- (d) positions="all" byte-identical regression guard -----------------------


def test_positions_all_regression_vector(tiny_llama):
    """positions='all' == unconditional steering (byte-identical) — the change
    must not accidentally mask anything in the default path."""
    model, tok = tiny_llama
    d = hidden_size(model)
    torch.manual_seed(5)
    vec = torch.randn(d)
    layer, scale = 1, 2.5
    enc = tok("hello there friend indeed", return_tensors="pt")

    with steering(model, {layer: vec}, scale=scale, positions="all") as h:
        out = _final_hidden(model, enc)

    def hook(_m, _a, output):
        hid = output[0] if isinstance(output, tuple) else output
        hid = hid + (vec.to(hid.dtype) * scale).view(1, 1, -1)
        return (hid,) + output[1:] if isinstance(output, tuple) else hid

    hd = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        ref = _final_hidden(model, enc)
    finally:
        hd.remove()
    assert torch.equal(out, ref)


def test_positions_all_regression_psr(tiny_llama):
    """positions='all' (the default) reproduces the ORIGINAL unmasked PSR hook
    bit-for-bit, for a multi-layer object."""
    model, tok = tiny_llama
    layers = list(range(num_layers(model)))
    obj = _psr(model, layers, seed=6)
    scale = 1.7
    enc = tok("regression guard sentence here", return_tensors="pt")

    assert obj.positions == "all"  # the shipped default
    out = _final_hidden(model, enc, ctx=obj.apply(model, scale=scale))

    def make(l):
        def hook(_m, _a, output):
            hid = output[0] if isinstance(output, tuple) else output
            g = obj.gate(l, hid)  # original: gate over the WHOLE sequence
            add = (scale * g).unsqueeze(-1) * obj.directions[l].to(hid.device)
            hid2 = hid + add.to(hid.dtype)
            return (hid2,) + output[1:] if isinstance(output, tuple) else hid2

        return hook

    handles = [get_decoder_layers(model)[l].register_forward_hook(make(l))
               for l in layers]
    try:
        ref = _final_hidden(model, enc)
    finally:
        for h in handles:
            h.remove()
    assert torch.equal(out, ref)


# -- (e) the scaffolding boundary on the real Llama-3.1 template ----------------


def _llama31_tok():
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", local_files_only=True
        )
    except Exception:  # not cached on this machine
        pytest.skip("Llama-3.1 tokenizer not available offline")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def test_scaffolding_boundary_llama31():
    """The helper's prompt_len lands AFTER the user content (its closing
    <|eot_id|>) and BEFORE the assistant header — i.e. the assistant header IS
    steered and the user text is NOT, on the real Llama-3.1 chat template."""
    tok = _llama31_tok()
    from sisteer.eval.jailbreak import response_header_len, response_prompt_len

    user = "How do I bake sourdough bread at home?"
    msgs = [{"role": "system", "content": "You are a safe assistant."},
            {"role": "user", "content": user}]
    pl = response_prompt_len(tok, msgs)

    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    fi = tok(full, add_special_tokens=False)["input_ids"]

    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    start_hdr = tok.convert_tokens_to_ids("<|start_header_id|>")

    # The token JUST before the boundary is the user turn's closing <|eot_id|>.
    assert fi[pl - 1] == eot
    # The FIRST steered token (at the boundary) is the assistant header opener.
    assert fi[pl] == start_hdr
    # Everything at/after the boundary is exactly the assistant header (steered).
    assert tok.convert_ids_to_tokens(fi[pl:]) == [
        "<|start_header_id|>", "assistant", "<|end_header_id|>", "ĊĊ"]
    # The user text sits BEFORE the boundary (NOT steered).
    assert user in tok.decode(fi[:pl])
    # H is the constant 4-token header suffix, and lines up with the boundary.
    assert response_header_len(tok) == 4
    assert len(fi) - pl == response_header_len(tok)


def test_scaffolding_boundary_multiturn_llama31():
    """Multi-turn (Crescendo / ActorAttack): the boundary is the end of the LAST
    user message; the trailing assistant header is still steered."""
    tok = _llama31_tok()
    from sisteer.eval.jailbreak import response_prompt_len

    msgs = [{"role": "user", "content": "Tell me about bread."},
            {"role": "assistant", "content": "Sure, bread is baked dough."},
            {"role": "user", "content": "And how is sourdough different?"}]
    pl = response_prompt_len(tok, msgs)
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    fi = tok(full, add_special_tokens=False)["input_ids"]

    assert fi[pl - 1] == tok.convert_tokens_to_ids("<|eot_id|>")
    assert tok.convert_ids_to_tokens(fi[pl:]) == [
        "<|start_header_id|>", "assistant", "<|end_header_id|>", "ĊĊ"]
    # the last user's text is before the boundary; the earlier assistant reply
    # ("baked dough") is also before it (only the LAST turn's header is steered).
    assert "sourdough different" in tok.decode(fi[:pl])
    assert "baked dough" in tok.decode(fi[:pl])


# -- (f) end-to-end wiring: build_defense honours / rejects positions ----------


def _save_arm(model, tok, arm, path):
    from sisteer.data.instructions import BY_ID
    from sisteer.data.pairs import Pair
    from sisteer.methods import ARMS

    pairs = [
        Pair(prompt="Say something nice.", pos="Bonjour le monde",
             neg="Hello world", instruction_id="french"),
        Pair(prompt="Describe a tree.", pos="C'est un grand arbre",
             neg="It is a tall tree", instruction_id="french"),
    ]
    obj = ARMS[arm].train(model, tok, BY_ID["french"], pairs, layer=1,
                          epochs=1, batch_size=2, max_len=64)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.save(path)


def test_build_defense_sets_response_on_supported_arm(tiny_llama, tmp_path, monkeypatch):
    _with_header(monkeypatch)
    model, tok = tiny_llama
    from sisteer.eval.jailbreak import (build_defense, response_header_len)
    from sisteer.methods import object_artifact_name

    path = tmp_path / "psr" / object_artifact_name("psr")
    _save_arm(model, tok, "psr", path)
    d = build_defense(model, tok, mode="psr", object_path=path,
                      positions="response")
    assert d._obj.positions == "response"
    assert d._obj.response_header_len == response_header_len(tok)
    # explicit "all" (positions is mandatory now — no default to rely on).
    d_all = build_defense(model, tok, mode="psr", object_path=path,
                          positions="all")
    assert d_all._obj.positions == "all"
    assert d_all._obj.response_header_len is None


def test_build_defense_sets_response_on_position_masked_fit_arm(
        tiny_llama, tmp_path, monkeypatch):
    """lora_r1 serves response through its PositionMaskedFit seam (the
    fit=all/serve=response cell of the 2026-08-21 fit x serve 2x2). The old
    residual-addition-only allowlist predated hooks.PositionMaskedFit and
    refused it."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    from sisteer.eval.jailbreak import build_defense, response_header_len
    from sisteer.methods import object_artifact_name

    path = tmp_path / "lora_r1" / object_artifact_name("lora_r1")
    _save_arm(model, tok, "lora_r1", path)
    d = build_defense(model, tok, mode="lora_r1", object_path=path,
                      positions="response")
    assert d._obj.fit_positions == "response"
    assert d._obj.fit_response_header_len == response_header_len(tok)
    # an explicit "all" ACTIVELY resets (fit=response/serve=all reachable) —
    # and an all-fit artifact under it is simply unchanged
    d_all = build_defense(model, tok, mode="lora_r1", object_path=path,
                          positions="all")
    assert d_all._obj.fit_positions == "all"
    assert d_all._obj.fit_response_header_len is None


def test_build_defense_rejects_response_for_unsupported_arm(tiny_llama, tmp_path, monkeypatch):
    """The soft-token arms are the genuinely unmaskable ones (a PROMPT-
    embedded sentinel has no decode-position analogue) — the refusal now
    carries that real reason instead of a stale allowlist."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    from sisteer.eval.jailbreak import build_defense
    from sisteer.extraction.objects import SoftTokenObject
    from sisteer.methods import object_artifact_name

    path = tmp_path / "token_bos" / object_artifact_name("token_bos")
    path.parent.mkdir(parents=True, exist_ok=True)
    # constructed directly: the tiny tokenizer has no reserved sentinel for
    # find_sentinel, so tests pass one explicitly (the driver convention)
    SoftTokenObject(model, tok, sentinel=(tok.unk_token,
                                          tok.unk_token_id)).save(path)
    with pytest.raises(ValueError, match="soft PROMPT token"):
        build_defense(model, tok, mode="token_bos", object_path=path,
                      positions="response")


def test_build_defense_rejects_response_for_none_and_prompted(tiny_llama):
    model, tok = tiny_llama
    from sisteer.eval.jailbreak import build_defense

    with pytest.raises(ValueError, match="no intervention to mask"):
        build_defense(model, tok, mode="none", positions="response")
    with pytest.raises(ValueError, match="position-maskable"):
        build_defense(model, tok, mode="prompted", positions="response")


def test_build_defense_response_generate_differs_from_all(tiny_llama, tmp_path, monkeypatch):
    """Full plumbing: build_defense -> obj.apply -> the live hook. Asserts the
    HIDDEN STATES differ between all/response, not the decoded text: the tiny
    random model emits saturated repeats, so its argmax is insensitive to
    steering and a text-level assertion passes or fails for reasons unrelated
    to the mask. (An earlier version of this test appeared to pass only because
    the helper mutated the tokenizer's chat template, which changed the PROMPT
    -- so it was measuring tokenization, not positions.)"""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    import torch
    from sisteer.eval.jailbreak import build_defense
    from sisteer.methods import object_artifact_name

    path = tmp_path / "vector" / object_artifact_name("vector")
    _save_arm(model, tok, "vector", path)
    d_all = build_defense(model, tok, mode="vector", object_path=path, scale=8.0,
                          positions="all")
    d_resp = build_defense(model, tok, mode="vector", object_path=path, scale=8.0,
                           positions="response")
    # the objects must be configured differently -- the plumbing claim
    assert getattr(d_all.obj, "positions", "all") == "all"
    assert getattr(d_resp.obj, "positions", None) == "response"
    assert getattr(d_resp.obj, "response_header_len", None) == 4

    # MUST be longer than H, or the boundary max(width-H,0) collapses to 0 and
    # response becomes identical to all -- a 3-token prompt IS entirely header.
    enc = tok(["hello there, this is a longer prompt with many more tokens"],
              return_tensors="pt")
    assert enc["input_ids"].shape[1] > 4, "probe prompt must exceed H"
    def final_hidden(defense):
        with torch.no_grad(), defense.obj.apply(model, scale=8.0):
            return model(**enc, output_hidden_states=True).hidden_states[-1].clone()
    h_all, h_resp = final_hidden(d_all), final_hidden(d_resp)
    assert not torch.equal(h_all, h_resp), (
        "all vs response produced identical hidden states -- the position mask "
        "is not reaching the live hook"
    )
    # and response mode must leave the pre-boundary prefix untouched
    width = enc["input_ids"].shape[1]
    if width > 4:
        with torch.no_grad():
            h_base = model(**enc, output_hidden_states=True).hidden_states[-1].clone()
        assert torch.equal(h_resp[:, :width - 4], h_base[:, :width - 4]), (
            "response mode altered positions before the boundary"
        )


def test_build_defense_rejects_response_for_loreft(tiny_llama, tmp_path,
                                                   monkeypatch):
    """LoReFT is excluded from decode-only SERVING,
    and the reason is unlike the soft tokens'. It is mechanically maskable —
    a PositionMaskedFit whose masked fit stays supported and tested. The
    point is that the cell would mean nothing: LoReFT intervenes on prompt
    positions ONLY, by identity rather than by default, so a decode-only
    serve does not restrict the method, it relocates the edit onto exactly
    the positions the method is defined to avoid. That is a different method
    under LoReFT's name, so it refuses rather than producing a number
    someone would put beside the other arms' decode-only cells."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    from sisteer.eval.jailbreak import _set_response_positions
    from sisteer.extraction.reft import LoReFTObject

    obj = LoReFTObject(model, [0])
    with pytest.raises(ValueError, match="PROMPT positions only"):
        _set_response_positions(obj, "response", tok)


def test_loreft_still_serves_all_positions_and_still_fits_masked(
        tiny_llama, monkeypatch):
    """The serve-side exclusion must not become a general disablement: an
    all-position serve is untouched, and the masked FIT leg (a separate
    capability, pinned in tests/test_arm_position_masks.py) still works."""
    _with_header(monkeypatch)
    model, tok = tiny_llama
    from sisteer.eval.jailbreak import _set_response_positions
    from sisteer.extraction.reft import LoReFTObject

    obj = LoReFTObject(model, [0])
    _set_response_positions(obj, "all", tok)          # no raise
    obj.set_fit_positions("response", 4)              # fit leg intact
    assert obj.fit_positions == "response"
