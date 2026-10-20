"""Mask parity: the ONE position-mask primitive, pinned semantics.

This is the agreed gate for the train x test 2x2 (train with {all, decode-only}
x test with {all, decode-only}). If the train-time mask and the test-time mask
are separate implementations, the off-diagonal cells measure an IMPLEMENTATION
DIFFERENCE rather than a generalization gap and the experiment is void. So the
mask lives in ONE place (sisteer.hooks.SteeringHandle._position_mask) and these
tests pin its semantics against drift.

PINNED SEMANTICS (all three of the ambiguities that "decode-only" leaves open):
  1. The mask is over INPUT positions; the boundary is the end of the LAST user
     message's content. Everything at index >= boundary is steered. The first
     generated token needs no special case: it is produced BY the forward at the
     prefill's final position, which is >= boundary and therefore steered.
  2. Chat-header / special tokens after the user turn are steered (masked IN).
     boundary = prompt_width - H, H = assistant-header token count.
  3. On the prefill the hook RUNS and WRITES ZERO at masked positions -- it is
     never an early return, so a gate reading pre-steering activations still
     sees every position and a gate regularizer's input set is unchanged.

NOTE the two natural boundary conventions differ by exactly H, and that is the
divergence this file exists to make explicit rather than discover later:
  header-steered (ours):  boundary = prompt_width - H
  decode-only:            boundary = prompt_width      (header NOT steered)

Two sections. The first drives _position_mask directly (the shared JB-lane
gate, adopted verbatim from the pushed primitive @ 0ea34bd). The second runs
the LIVE path — real forwards on the conftest tiny_llama, masks recorded from
the handle the hook actually consumed — and adds this lane's pins:
  * H == 4 on the real NousResearch mirror tokenizer (the marker fallback IS
    the live path there; H == 0 means the diff-only derivation regressed);
  * the flag-ignoring-template regression on a stub (no download needed);
  * header-steered vs plain decode-only differ at EXACTLY H on live forwards;
  * the unset-boundary trap and the width <= H collapse trap, visible in
    logits, not just in the mask;
  * train/eval mask identity on a fixed batch (the 2x2 gate, live).
"""

import re
import pathlib

import pytest
import torch

from sisteer.chat_markers import derive_assistant_markers
from sisteer.generate import encode_kwargs
from sisteer.hooks import (
    POSITION_MODES,
    SteeringHandle,
    response_header_len,
    steering,
)
from sisteer.models import apply_chat_template_folding, hidden_size

# =============================================================================
# Section 1 — the shared primitive's own gate (JB lane, adopted @ 0ea34bd)
# =============================================================================



class _Tiny(torch.nn.Module):
    """Minimal stand-in exposing the decoder-layer list the handle hooks."""

    def __init__(self, n_layers=2, d=8):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Identity() for _ in range(n_layers)
        )
        self.config = type("C", (), {"num_hidden_layers": n_layers, "hidden_size": d})()

    def get_decoder(self):
        return self


def _mask(positions, *, prompt_len, seq_len, batch=1, skip_bos=False,
          response_header_len=None, offset=0):
    """Drive _position_mask directly — no model forward needed."""
    h = SteeringHandle.__new__(SteeringHandle)
    h.positions = positions
    h.prompt_len = prompt_len
    h.skip_bos = skip_bos
    h.response_header_len = response_header_len
    h._prompt_lens = None
    h._attn_mask = None
    h._cur_offset = offset
    h._seen = offset + seq_len
    h._cache_len = None
    m = h._position_mask(batch, seq_len, torch.device("cpu"))
    return None if m is None else m[0].bool()


def test_positions_all_is_the_unmasked_default():
    """'all' with no skip_bos returns None (no mask) — the regression guard."""
    assert _mask("all", prompt_len=0, seq_len=6) is None


@pytest.mark.parametrize("mode", POSITION_MODES)
def test_every_declared_mode_is_constructible(mode):
    """A mode in POSITION_MODES must not raise — catches a mode added to the
    tuple without a branch in _position_mask (which would silently fall through
    to steering everything)."""
    m = _mask(mode, prompt_len=3, seq_len=6)
    assert m is None or m.shape[-1] == 6


def test_response_boundary_splits_prompt_from_generation():
    """The core semantic: idx >= boundary steered, idx < boundary untouched."""
    m = _mask("response", prompt_len=4, seq_len=7)
    assert m.tolist() == [False, False, False, False, True, True, True]


def test_prompt_len_zero_steers_everything_THE_TRAP():
    """prompt_len defaults to 0, so 'response' with an unset boundary steers
    EVERY position while looking like it worked. Any consumer that forgets to
    set the boundary gets full-tensor steering, not decode-only. This test
    exists so that failure mode is documented and cannot be mistaken for a
    passing decode-only run."""
    m = _mask("response", prompt_len=0, seq_len=5)
    assert m.all(), "prompt_len=0 must steer all positions — the silent trap"


def test_header_steered_vs_decode_only_differ_by_exactly_H():
    """THE PARITY POINT. The two boundary conventions are not interchangeable:
    they differ at precisely the H assistant-header positions. A 2x2 built from
    one convention at train time and the other at test time would attribute
    those H tokens to a generalization gap."""
    width, H = 10, 4
    ours = _mask("response", prompt_len=width - H, seq_len=width)
    decode_only = _mask("response", prompt_len=width, seq_len=width)
    assert ours.sum().item() == H
    assert decode_only.sum().item() == 0          # nothing in the prefill
    diff = ours ^ decode_only
    assert diff.sum().item() == H
    assert diff[width - H:].all(), "the difference is exactly the header tail"


def test_identical_boundary_gives_identical_mask():
    """Given the SAME boundary, the primitive is deterministic and
    convention-independent — so parity holds once the boundary is agreed."""
    a = _mask("response", prompt_len=6, seq_len=9)
    b = _mask("response", prompt_len=6, seq_len=9)
    assert torch.equal(a, b)


def test_decode_steps_are_steered_via_running_offset():
    """Each KV-cached decode forward is seq_len==1 at a global offset past the
    boundary, so it must be steered. A broken offset shows up here."""
    for step in range(4):
        m = _mask("response", prompt_len=5, seq_len=1, offset=5 + step)
        assert m.all(), f"decode step {step} must be steered"


def test_prefill_positions_before_boundary_are_never_steered():
    """Complement of the above: inside the prefill, everything below the
    boundary stays untouched, which is what makes the user's turn undisturbed."""
    m = _mask("response", prompt_len=8, seq_len=8)
    assert not m.any()


def test_boundary_collapses_when_prompt_shorter_than_header():
    """A prompt shorter than H makes max(width-H, 0) == 0, so 'response'
    degenerates to 'all'. That is CORRECT (a prompt that short is entirely
    header) but it is indistinguishable from the prompt_len=0 trap, so it is
    pinned here: any probe or fixture must use a prompt LONGER than H, or it
    silently tests nothing. This cost a real debugging cycle."""
    H = 4
    for width in (1, 2, 3, 4):
        m = _mask("response", prompt_len=max(width - H, 0), seq_len=width)
        assert m.all(), f"width={width} <= H={H} must steer everything"
    m = _mask("response", prompt_len=max(8 - H, 0), seq_len=8)
    assert m.sum().item() == H, "width > H must steer exactly the H-token tail"


# =============================================================================
# Section 2 — the live path: real forwards, recorded masks, real tokenizer H
# =============================================================================



NOUS = "NousResearch/Meta-Llama-3.1-8B-Instruct"
NOUS_REV = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"

# The Llama-3 assistant header, as GPT-2-style byte-level tokens
# ("ĊĊ" is "\n\n"): the H positions the header-steered convention turns on.
HEADER_TAIL = ["<|start_header_id|>", "assistant", "<|end_header_id|>", "ĊĊ"]
H = 4  # pinned by test_header_len_is_4_on_the_real_tokenizer


def _nous_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(NOUS, revision=NOUS_REV)
    except Exception as e:  # loud skip: names what must be fetched
        pytest.skip(
            f"NousResearch mirror tokenizer not loadable ({type(e).__name__})"
            f" — hf_hub_download {NOUS}@{NOUS_REV[:8]} on this machine first;"
            " the H golden requires the real tokenizer"
        )


# ---------------------------------------------------------------------------
# 1. H on the real tokenizer
# ---------------------------------------------------------------------------


def test_header_len_is_4_on_the_real_tokenizer():
    tok = _nous_tokenizer()
    h = response_header_len(tok)
    assert h != 0, (
        "H == 0 on the NousResearch mirror: the diff-only path is live. The"
        " mirror's stripped chat template ignores add_generation_prompt, so"
        " the True/False renders come back identical, the diff is 0, and the"
        " boundary silently collapses to prompt_width — the whole assistant"
        " header lands on the prompt side and the FIRST GENERATED TOKEN is"
        " sampled from UNSTEERED logits. The marker fallback in"
        " response_header_len must be exercised."
    )
    assert h == 4
    msgs = [{"role": "user", "content": "x"}]
    full = apply_chat_template_folding(tok, msgs, add_generation_prompt=True)
    ids = tok(full, **encode_kwargs(tok, full))["input_ids"]
    assert tok.convert_ids_to_tokens(ids[-h:]) == HEADER_TAIL


# ---------------------------------------------------------------------------
# 2. the flag-ignoring-template regression (stub — always runs)
# ---------------------------------------------------------------------------

# one special marker, one "\n\n", one non-space run, or one space per token —
# matches the real tokenizer's count (11) on the single-user-turn render.
_STUB_TOKEN_RE = re.compile(r"<\|[^<>|]+\|>|\n\n|[^\s<]+|\s")


class _FlagIgnoringTokenizer:
    """Stand-in for the NousResearch mirror's stripped template: the
    assistant header is appended UNCONDITIONALLY and
    ``add_generation_prompt`` is deliberately unread — the exact template
    shape that silently neutered a diff-only H derivation."""

    bos_token = "<|begin_of_text|>"
    chat_template = "stripped-llama3"  # truthy; the render below ignores flags

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        assert not tokenize
        out = self.bos_token
        for m in messages:
            out += (
                f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
                f"{m['content']}<|eot_id|>"
            )
        # UNCONDITIONAL: add_generation_prompt is ignored, like the mirror.
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out

    def __call__(self, text, **kwargs):
        return {"input_ids": _STUB_TOKEN_RE.findall(text)}


def test_flag_ignoring_template_still_yields_h_4():
    tok = _FlagIgnoringTokenizer()
    msgs = [{"role": "user", "content": "x"}]
    full = tok.apply_chat_template(msgs, add_generation_prompt=True)
    head = tok.apply_chat_template(msgs, add_generation_prompt=False)
    assert full == head  # the diff-only signal really is dead on this shape
    assert response_header_len(tok) == 4, (
        "the marker fallback must recover H = 4 when the rendered diff is 0;"
        " 0 here means the fallback regressed and the boundary silently"
        " collapses to prompt_width on the mirror this programme runs"
    )


# ---------------------------------------------------------------------------
# 2b. H on the real Gemma-4-E4B tokenizer — the two-SI Gemma lane's pin
# ---------------------------------------------------------------------------


GEMMA = "google/gemma-4-E4B-it"
# The two-SI Gemma pin (configs/two_si_gemma4e4b_reduced.yaml
# ``model_revision``; every staged gemma4e4b object's train_budget_meta.json
# records this same sha). The shared cluster cache's refs/main resolves to a
# TOKENIZER-ONLY snapshot, so every load must pass this revision explicitly
# or trigger a gated fetch.
GEMMA_REV = "ee0ef6023621cff504d758262d4e04895a5af4a2"

# The Gemma-4 assistant header as the pinned template renders it:
# add_generation_prompt=True appends exactly these three tokens.
GEMMA_HEADER_TAIL = ["<|turn>", "model", "\n"]
GEMMA_H = 3


def _gemma_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(GEMMA, revision=GEMMA_REV)
    except Exception as e:  # loud skip: names what must be fetched
        pytest.skip(
            f"Gemma-4-E4B tokenizer not loadable ({type(e).__name__}) — "
            f"hf_hub_download {GEMMA}@{GEMMA_REV[:8]} on this machine first"
            " (gated repo: accepted licence + token); the Gemma H golden"
            " requires the real tokenizer"
        )


def test_header_len_is_3_on_the_real_gemma_tokenizer():
    """The gemma4e4b generation cells serve fit=response objects, so
    positions='response' is the live leg and _set_response_positions
    hard-errors on H <= 0. This golden pins H > 0 (the gate passes) and
    H == 3 exactly: the boundary sits at the start of ``<|turn>model\n``,
    so the final prefill row — the forward that emits the first generated
    token, the decisive one for a refusal defense — is inside the mask.

    NOTE the rendered-DIFF path is the live derivation here: the pinned
    gemma-4 template honors add_generation_prompt (unlike the NousResearch
    mirror above). That matters because response_header_len's marker
    FALLBACK list knows ``<start_of_turn>`` (gemma-2/3) but NOT gemma-4's
    ``<|turn>`` — on a flag-ignoring gemma-4 template the fallback would
    miss and H would silently be 0. The renders-differ assertion is the
    tripwire that says which path must be alive.
    """
    tok = _gemma_tokenizer()
    msgs = [{"role": "user", "content": "x"}]
    full = apply_chat_template_folding(tok, msgs, add_generation_prompt=True)
    head = apply_chat_template_folding(tok, msgs, add_generation_prompt=False)
    assert full != head, (
        "the gemma-4 template stopped honoring add_generation_prompt: the"
        " diff measures 0 and the marker fallback does not know gemma-4's"
        " <|turn> header token, so H collapses to 0 — every fit=response"
        " gemma cell then hard-errors at _set_response_positions"
    )
    h = response_header_len(tok)
    assert h > 0, (
        "H == 0 on the pinned gemma-4-E4B template — positions='response'"
        " hard-errors and every gemma generation cell is blocked"
    )
    assert h == GEMMA_H
    ids = tok(full, **encode_kwargs(tok, full))["input_ids"]
    assert tok.convert_ids_to_tokens(ids[-h:]) == GEMMA_HEADER_TAIL


def test_assistant_markers_locate_the_gemma_header():
    """derive_assistant_markers must locate a correct, non-empty header on
    the gemma-4 template (any structural surprise raises ChatMarkerError
    inside the probe). Pinned relationships:

    * ``markers.header`` is the exact token TAIL of the generation-prompt
      render — the property the serving hook verifies before masking
      (HookOp.header_ids matched against the request's own tokens);
    * the marker header is ONE token longer than response_header_len's H:
      the header is located from the previous turn's terminator
      (``<turn|>``), so it absorbs the inter-turn ``\n`` that the closing
      scaffold also carries. Both conventions keep the final prefill row
      inside the mask and neither reaches user content — the extra token
      is inter-turn scaffold. Pinned explicitly so nobody "fixes" one
      derivation to agree with the other and silently moves a boundary.
    """
    tok = _gemma_tokenizer()
    m = derive_assistant_markers(tok)
    assert m.header, "no assistant header located on the gemma-4 template"
    assert tok.convert_ids_to_tokens(list(m.header)) == ["\n"] + GEMMA_HEADER_TAIL
    assert tok.convert_ids_to_tokens([m.terminator]) == ["<turn|>"]
    assert tok.convert_ids_to_tokens(list(m.closing)) == ["<turn|>", "\n"]
    assert m.glue == 0  # the terminator opens the closing scaffold directly
    msgs = [{"role": "user", "content": "x"}]
    full = apply_chat_template_folding(tok, msgs, add_generation_prompt=True)
    ids = tok(full, **encode_kwargs(tok, full))["input_ids"]
    assert tuple(ids[-len(m.header):]) == m.header  # exact render tail
    assert len(m.header) == response_header_len(tok) + 1


# ---------------------------------------------------------------------------
# mask plumbing shared by 3–5
# ---------------------------------------------------------------------------


def _record_masks(handle):
    """Wrap handle._position_mask so every mask handed to the steering hook
    is captured (one call per forward here: exactly one steered layer)."""
    rec = []
    orig = handle._position_mask

    def wrapped(batch, seq_len, device):
        m = orig(batch, seq_len, device)
        rec.append(m)
        return m

    handle._position_mask = wrapped
    return rec


def _prefill_then_decode(model, input_ids, attention_mask, handle, gen_steps):
    """Run the eval-time shape: one prefill forward, then ``gen_steps``
    KV-cached single-token decode forwards; return the [B, P+gen_steps]
    concatenation of the masks the hook actually used."""
    rec = _record_masks(handle)
    handle.reset()
    with torch.no_grad():
        out = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=True
        )
        past = out.past_key_values
        nxt = input_ids[:, -1:]
        for _ in range(gen_steps):
            out = model(input_ids=nxt, past_key_values=past)
            past = out.past_key_values
    return torch.cat(rec, dim=1)


# ---------------------------------------------------------------------------
# 3. parity: header-steered vs plain decode-only differ at EXACTLY H
# ---------------------------------------------------------------------------


def test_header_steered_and_plain_masks_differ_at_exactly_h(tiny_llama):
    model, tok = tiny_llama
    d = hidden_size(model)
    vec = torch.randn(d)
    enc = tok("one two three four five six seven", return_tensors="pt")
    P = enc["input_ids"].shape[1]
    assert P > H
    gen_steps = 3

    with steering(
        model, {1: vec}, positions="response", response_header_len=H
    ) as h_hdr:
        m_hdr = _prefill_then_decode(
            model, enc["input_ids"], enc["attention_mask"], h_hdr, gen_steps
        )
        # the boundary was DERIVED from the prefill width, never supplied
        assert h_hdr.prompt_len == P - H

    with steering(model, {1: vec}, positions="response", prompt_len=P) as h_pl:
        m_pl = _prefill_then_decode(
            model, enc["input_ids"], enc["attention_mask"], h_pl, gen_steps
        )

    assert m_hdr.shape == m_pl.shape == (1, P + gen_steps)
    diff = m_hdr != m_pl
    assert int(diff.sum()) == H  # EXACTLY the H header positions
    assert diff[0, P - H : P].all()
    # header-steered: header + every generated position on, user prompt off
    assert m_hdr[0, P - H :].eq(1).all()
    assert m_hdr[0, : P - H].eq(0).all()
    # plain decode-only: nothing before the boundary, everything after
    assert m_pl[0, :P].eq(0).all()
    assert m_pl[0, P:].eq(1).all()


# ---------------------------------------------------------------------------
# 4. the trap: an unset boundary steers EVERY position
# ---------------------------------------------------------------------------


def test_unset_boundary_steers_every_position(tiny_llama):
    """positions='response' with NO boundary at all (prompt_len left at its
    default 0, no response_header_len) degenerates to steering every
    position, prompt included. Pinned explicitly so this failure mode reads
    as what it is — a forgotten boundary — and can never be mistaken for a
    working decode-only run."""
    model, tok = tiny_llama
    d = hidden_size(model)
    vec = torch.randn(d) * 3
    enc = tok("alpha beta gamma delta", return_tensors="pt")
    P = enc["input_ids"].shape[1]
    gen_steps = 2

    base = model(**enc).logits.detach()
    with steering(model, {1: vec}, positions="response") as h:
        masks = _prefill_then_decode(
            model, enc["input_ids"], enc["attention_mask"], h, gen_steps
        )
    assert masks.shape == (1, P + gen_steps)
    assert masks.eq(1).all()  # every position, prompt included

    # and it is visible in the logits: position 0 (a PROMPT position, which
    # a real decode-only run leaves untouched) is steered
    with steering(model, {1: vec}, positions="response") as h:
        h.reset()
        steered = model(**enc).logits.detach()
    assert not torch.allclose(base[:, 0], steered[:, 0])


# ---------------------------------------------------------------------------
# 4b. the second trap: a prompt no longer than H collapses the boundary to 0
# ---------------------------------------------------------------------------


def test_prompt_no_longer_than_h_steers_every_position(tiny_llama):
    """A peer-reported collapse trap (2026-08-20): with the header-steered
    convention the boundary is max(prompt_width - H, 0), so a prompt of
    width <= H collapses it to 0 and positions='response' steers EVERY
    position — prompt included — exactly like the forgotten-boundary trap
    above. A probe, fixture, or smoke test whose prompt is not LONGER
    than H therefore silently tests nothing (all-token and decode-only
    become indistinguishable). Pinned so the failure mode reads as what
    it is; every real-prompt test in this file asserts P > H for the same
    reason."""
    model, tok = tiny_llama
    d = hidden_size(model)
    vec = torch.randn(d) * 3
    enc = tok("hi", return_tensors="pt")
    P = enc["input_ids"].shape[1]
    assert P <= H, "fixture must be at most H tokens to exercise the trap"
    gen_steps = 2

    with steering(
        model, {1: vec}, positions="response", response_header_len=H
    ) as h:
        masks = _prefill_then_decode(
            model, enc["input_ids"], enc["attention_mask"], h, gen_steps
        )
        assert h.prompt_len == 0  # max(P - H, 0) collapsed
    assert masks.shape == (1, P + gen_steps)
    assert masks.eq(1).all()  # every position, prompt included

    # and the collapse is visible in the logits: position 0 (a PROMPT
    # position a working decode-only run leaves untouched) is steered
    base = model(**enc).logits.detach()
    with steering(
        model, {1: vec}, positions="response", response_header_len=H
    ) as h:
        h.reset()
        steered = model(**enc).logits.detach()
    assert not torch.allclose(base[:, 0], steered[:, 0])


# ---------------------------------------------------------------------------
# 5. train/eval mask identity (the 2x2 gate)
# ---------------------------------------------------------------------------


def test_train_and_eval_paths_produce_identical_masks(tiny_llama):
    """The train-time path (ONE full-sequence forward with per-sample
    boundaries via set_prompt_lens) and the eval-time path (prefill +
    KV-cached decode, boundary derived per prefill) must select IDENTICAL
    position sets on the same fixed batch — the gate that makes a
    train x test 2x2 trustworthy; without it the off-diagonal cells
    measure an implementation delta."""
    model, tok = tiny_llama
    d = hidden_size(model)
    vec = torch.randn(d)
    enc = tok(
        "one two three four five six seven eight nine", return_tensors="pt"
    )
    ids = enc["input_ids"].repeat(2, 1)  # fixed fixture batch, B=2
    attn = torch.ones_like(ids)
    B, T = ids.shape
    gen_steps = 3
    P = T - gen_steps
    assert P > H

    # eval-time: prefill of width P, then the "generated" tokens replayed
    # as decode steps; the boundary is derived from the prefill width.
    with steering(
        model, {1: vec}, positions="response", response_header_len=H
    ) as h_ev:
        rec = _record_masks(h_ev)
        h_ev.reset()
        with torch.no_grad():
            out = model(
                input_ids=ids[:, :P],
                attention_mask=attn[:, :P],
                use_cache=True,
            )
            past = out.past_key_values
            for i in range(gen_steps):
                out = model(
                    input_ids=ids[:, P + i : P + i + 1], past_key_values=past
                )
                past = out.past_key_values
        assert h_ev.prompt_len == P - H  # derived, not caller-supplied
        m_eval = torch.cat(rec, dim=1)

    # train-time: one full-sequence forward over the SAME tokens with the
    # per-sample header-steered boundary (prompt width minus H).
    with steering(
        model, {1: vec}, positions="response", response_header_len=H
    ) as h_tr:
        rec = _record_masks(h_tr)
        h_tr.set_prompt_lens(torch.full((B,), P - H, dtype=torch.long))
        with torch.no_grad():
            model(input_ids=ids, attention_mask=attn)
        m_train = torch.cat(rec, dim=1)

    assert m_eval.shape == m_train.shape == (B, T)
    assert torch.equal(m_eval, m_train)



def test_psr_tracker_has_cache_offset_field():
    """DEFECT 2 regression (adopted from psr-hard-mse-arms @ afe3e10):
    PSRObject's position tracker must read the authoritative KV-cache length,
    like SteeringHandle. Without it an apply() entered MID-GENERATION restarts
    positions at 0, so the first decode step looks like global position 0 and
    skip_bos masks it -- silently dropping steering on the first generated
    token, which for a refusal defense is the decisive one. Latent in the
    jailbreak lane (prefill+decode share one context) but live for any
    generated-only engine that enters mid-stream."""
    import inspect
    from sisteer.extraction import psr as psr_mod
    src = inspect.getsource(psr_mod)
    assert "_cache_len" in src, "PSR tracker lacks the KV-cache offset"
    assert "get_seq_length" in src, "PSR tracker never reads past_key_values length"
    assert "_track_model_inputs" in src, (
        "PSR needs a model-boundary pre-hook: past_key_values is only visible there"
    )


def test_chat_template_hash_guard_catches_the_injecting_template():
    """The template HASH is the real invariant, not the model id.

    A name check (which is what this repo had first) passes any mirror under a
    different name, including one carrying the template that injects a
    ~25-token default system block. That block makes a no-system-prompt arm
    not-unprompted, and mixing the two templates inside one comparison already
    cost a headline result in this lane. The petri lane found the same defect
    from the opposite direction by freezing chat_template_hash, so the two
    hashes below are independently confirmed.
    """
    from sisteer.models import (_TEMPLATE_HASH_METALLAMA,
                                _TEMPLATE_HASH_NOUSRESEARCH,
                                chat_template_hash, check_chat_template)
    assert _TEMPLATE_HASH_NOUSRESEARCH != _TEMPLATE_HASH_METALLAMA

    class _Tok:
        def __init__(self, tmpl): self.chat_template = tmpl

    assert chat_template_hash(_Tok(None)) is None, "no template -> no hash"
    # a tokenizer with an unknown template is not flagged (only the known-bad one is)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_chat_template(_Tok("{{ messages }}"))
    assert not w, "an unrelated template must not warn"


def test_mixed_attacker_engine_root_is_refused(tmp_path):
    """A private root mixing HF- and vLLM-generated transcripts must REFUSE.

    The two engines cannot produce identical attacker text, so such a root
    holds two different attack distributions and its ASR is a meaningless
    blend. vLLM is now the DEFAULT engine, so a pre/post-cutover mix is the
    likely accident rather than an exotic one -- which is why this is checked
    at read time rather than left to a sidecar someone has to remember to
    inspect. Delegates to the shared axis-agnostic identity primitive.
    """
    import json
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
    import jailbreak_run
    from sisteer.capeval.engine_identity import EngineIdentityError

    root = tmp_path / "actorattack" / "psr_kl_all"
    for shard, engine in (("shard00", "hf"), ("shard01", "vllm")):
        d = root / shard
        d.mkdir(parents=True)
        (d / "attacker_engine.json").write_text(json.dumps({
            "attacker_engine": engine,
            "attacker_model": "huihui-ai/Huihui-Qwen3.6-27B-abliterated",
        }))
    try:
        jailbreak_run._assert_uniform_attacker_engine(root)
    except EngineIdentityError as e:
        assert "attacker engine" in str(e)
        assert "attacker_engine" in e.differing
        return
    raise AssertionError("a mixed-engine root must be refused")


def test_uniform_attacker_engine_root_is_accepted(tmp_path):
    """The complement: a root where every shard agrees must pass, so the guard
    cannot pass vacuously by refusing everything."""
    import json, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
    import jailbreak_run

    root = tmp_path / "actorattack" / "psr_kl_all"
    for shard in ("shard00", "shard01"):
        d = root / shard
        d.mkdir(parents=True)
        (d / "attacker_engine.json").write_text(json.dumps({
            "attacker_engine": "vllm",
            "attacker_model": "huihui-ai/Huihui-Qwen3.6-27B-abliterated",
            # a KNOWN identity: the guard also refuses an aggregate that is
            # uniformly unrecorded (unknown is not uniform), so the complement
            # records its precision and cache setting
            "attacker_quantization": "none",
            "attacker_prefix_caching": True,
        }))
    jailbreak_run._assert_uniform_attacker_engine(root)  # must not raise
