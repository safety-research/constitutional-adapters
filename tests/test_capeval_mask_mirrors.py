"""Assistant-content mask on BOTH Llama-3.1 mirrors, and the measured
chat-template ground truth behind the NousResearch mask corruption
(sisteer.chat_markers).

THE BUG THIS PINS DOWN (found 2026-08-21): under
``NousResearch/Meta-Llama-3.1-8B-Instruct`` @ d10aef79 the old
``assistant_content_mask`` returned 7 positions for
``[user: "What is 2+2?", assistant: "Four."]`` — ``['Four', '.',
'<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>',
'ĊĊ']`` — silently averaging teacher-forced NLL over deterministic
scaffold (and the coarsened KL rode on the same mask). An
empty-content assistant turn was worse: 4 pure-scaffold positions with
ZERO special ids among them (the header tokens are added CONTROL tokens,
not ``all_special_ids``). Multi-turn conversations were loudly BLOCKED
instead (the unconditional trailing header diverges from the next user
turn's header in the full render). meta-llama was correct throughout.

FIVE-CELL GROUND TRUTH, re-measured on this machine 2026-08-21
(transformers 5.14.1, via the tier0/chat_markers render path; counts
include the leading BOS; user turn = "hello"; SI = the 66-word
SAFETY_SI). Two peer lanes measured pieces of this table independently;
one lane's earlier counts were a len()-of-a-mapping artifact and were
withdrawn, so THIS re-measurement is the authoritative record:

    render token counts        agp=False   agp=True   True-False
    meta-llama, bare user turn      32         36          4
    NousResearch, bare user turn    11         11          0
    meta-llama, SAFETY_SI+user     120        124          4
    NousResearch, SAFETY_SI+user   104        104          0

    mirror difference (agp=True):  bare 36-11 = 25;  SI 124-104 = 20

MECHANISM, confirmed against the raw templates (not just the counts):

* The NousResearch template appends
  ``<|start_header_id|>assistant<|end_header_id|>\\n\\n`` UNCONDITIONALLY
  at the end of every render — the string ``add_generation_prompt``
  does not appear in its 348-char template. So the flag is neither
  honored nor "ignored with no header": the header is ALWAYS present,
  and a probe that diffs the True/False renders measures 0 added tokens
  exactly where the header always is. That diff artifact is why an
  earlier probe concluded "the flag is ignored" on both mirrors.
* The old mask took the suffix of ``render(msgs[:k+1])`` after the
  ``render(msgs[:k], agp=True)`` prefix. The PREFIX boundary was right
  (content starts exactly there, under both mirrors); the SUFFIX
  boundary was wrong — the render's trailing ``<|eot_id|>`` + the
  unconditional header landed inside the span, and the trailing
  special-token strip halted at ``ĊĊ`` (a plain newline token).
* The 25-vs-20 mirror difference is the meta-llama template's injected
  "Cutting Knowledge Date / Today Date" block: with NO system message
  it forms its OWN system block (~20 content + 5 block-scaffold tokens
  = 25); with an SI present it is MERGED into the SAME system block as
  the SI, sharing the scaffold (net 20). Both figures are correct — for
  different cases — so neither should be "corrected" into the other.
* Cross-check: the meta-llama True-False diff (4) equals the
  marker-located assistant header length H=4 on BOTH mirrors, so marker
  location reproduces what a conventional template reports.

The generalizable rule (all three lanes converged on it): derive
template structure by LOCATING it in a render, never by reading it off
a flag's effect; where two renders must be compared, assert the
difference is non-zero and verify the extracted structure round-trips.

meta-llama absolute counts include the date-dependent "Today Date"
string; the day/month/year each tokenize to one token so the counts are
stable in practice (the tier0 goldens pin absolutes the same way). The
diffs (4/0/25/20) are date-independent.
"""

import pytest

from sisteer import chat_markers as cm
from sisteer.capeval.tiers import tier0 as t0
from sisteer.data.instructions import SAFETY_SI

META_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
META_REV = "0e9e39f249a16976918f6564b8830bc894c89659"
NSR_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
NSR_REV = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # conftest pin

MIRRORS = [pytest.param(META_MODEL, META_REV, id="meta-llama"),
           pytest.param(NSR_MODEL, NSR_REV, id="nousresearch")]

TWO_TURN = [{"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "Four."}]
MULTI_TURN = [{"role": "user", "content": "Name a color."},
              {"role": "assistant", "content": "Blue."},
              {"role": "user", "content": "Another?"},
              {"role": "assistant", "content": "Green and gold."}]
EMPTY_TURN = [{"role": "user", "content": "Say nothing."},
              {"role": "assistant", "content": ""},
              {"role": "user", "content": "Ok now speak."},
              {"role": "assistant", "content": "Words."}]

CONVS = [pytest.param(TWO_TURN, "Four.", id="two-turn"),
         pytest.param(MULTI_TURN, "Blue.Green and gold.", id="multi-turn"),
         pytest.param(EMPTY_TURN, "Words.", id="empty-assistant-turn")]


def _pinned_tok(model, revision):
    """Same offline gating as test_template_guard: pinned revision from
    the local cache only, loud skip naming what is missing otherwise —
    so CI without the caches skips while this machine really runs."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=True)
    except Exception as e:  # cache miss, offline resolution failure, ...
        pytest.skip(f"pinned {model} tokenizer @ {revision[:8]} not in the "
                    f"local HF cache ({type(e).__name__})")


# ---------------------------------------------------------------------------
# the fix: content-only masks on BOTH mirrors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model,revision", MIRRORS)
@pytest.mark.parametrize("conv,expected", CONVS)
def test_mask_is_content_only_on_both_mirrors(model, revision, conv, expected):
    """The regression this pins down: on NousResearch the
    two-turn mask must be exactly the content (2 positions, was 7 with
    EOT+header+ĊĊ), multi-turn must WORK (was a spurious MaskError), and
    the empty assistant turn must contribute nothing (was 4 scaffold
    positions). meta-llama must be unchanged."""
    tok = _pinned_tok(model, revision)
    full, pos = t0.assistant_content_mask(tok, conv)
    assert tok.decode([full[p] for p in pos]) == expected
    ctrl = cm.control_token_ids(tok)
    assert not [p for p in pos if full[p] in ctrl], \
        "scaffold/control tokens leaked into the assistant content mask"
    # the terminal EOT sits immediately AFTER the mask, never inside it
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    assert full[pos[-1] + 1] == eot
    assert eot not in [full[p] for p in pos]


def test_mask_counts_match_between_mirrors():
    """Identical content must be scored over the same number of positions
    under either mirror (the mirrors share the vocab; only the scaffold
    around the content differs). This is the invariant whose violation
    silently diluted the NousResearch NLL."""
    meta = _pinned_tok(META_MODEL, META_REV)
    nsr = _pinned_tok(NSR_MODEL, NSR_REV)
    for conv in (TWO_TURN, MULTI_TURN, EMPTY_TURN):
        full_m, pos_m = t0.assistant_content_mask(meta, conv)
        full_n, pos_n = t0.assistant_content_mask(nsr, conv)
        assert len(pos_m) == len(pos_n)
        assert ([full_m[p] for p in pos_m] == [full_n[p] for p in pos_n]), \
            "the two mirrors must mask the SAME content token ids"


@pytest.mark.parametrize("model,revision", MIRRORS)
def test_markers_locate_the_llama31_scaffold(model, revision):
    """Marker derivation is by LOCATION, and reproduces the conventional
    template's own numbers: H=4 header on both mirrors (equal to the
    meta-llama add_generation_prompt render diff), <|eot_id|> terminator,
    and a closing scaffold that carries NousResearch's unconditional
    trailing header."""
    tok = _pinned_tok(model, revision)
    m = cm.derive_assistant_markers(tok)
    ids = tok.convert_tokens_to_ids
    header = (ids("<|start_header_id|>"), ids("assistant"),
              ids("<|end_header_id|>"), ids("ĊĊ"))
    assert m.header == header and len(m.header) == 4
    assert m.terminator == ids("<|eot_id|>") == tok.eos_token_id == 128009
    assert m.glue == 0
    if model == META_MODEL:
        assert m.closing == (m.terminator,)
    else:
        assert m.closing == (m.terminator,) + header


# ---------------------------------------------------------------------------
# the measured template ground truth (docstring table, asserted)
# ---------------------------------------------------------------------------


def test_five_cell_template_ground_truth():
    """The docstring's table, asserted, so nobody re-derives this story a
    fourth time. Primary invariants are the DIFFS (date-independent):
    meta True-False = 4 = H; NousResearch True-False = 0 with the header
    present in BOTH renders; mirror difference 25 bare vs 20 with the SI
    (the injected date block merges into an existing system block)."""
    meta = _pinned_tok(META_MODEL, META_REV)
    nsr = _pinned_tok(NSR_MODEL, NSR_REV)
    bare = [{"role": "user", "content": "hello"}]
    si = [{"role": "system", "content": SAFETY_SI.text},
          {"role": "user", "content": "hello"}]

    r = {(t, c, agp): cm.render_ids(tok, msgs, add_generation_prompt=agp)
         for t, tok in (("meta", meta), ("nsr", nsr))
         for c, msgs in (("bare", bare), ("si", si))
         for agp in (False, True)}

    # NousResearch: add_generation_prompt is never consulted — the string
    # does not appear in the template, and the True/False renders are
    # IDENTICAL (so a header-by-diff probe measures 0 where the header is
    # always present). Both renders end with the unconditional header.
    assert "add_generation_prompt" not in nsr.chat_template
    header = list(cm.derive_assistant_markers(nsr).header)
    for c in ("bare", "si"):
        assert r[("nsr", c, True)] == r[("nsr", c, False)]
        assert r[("nsr", c, False)][-4:] == header
    # meta-llama: the flag genuinely appends H=4 header tokens.
    for c in ("bare", "si"):
        assert r[("meta", c, True)] == r[("meta", c, False)] + header
    # absolute counts (meta includes the injected date block; day/month/
    # year are one token each so these are stable in practice)
    counts = {k: len(v) for k, v in r.items()}
    assert counts == {
        ("meta", "bare", False): 32, ("meta", "bare", True): 36,
        ("nsr", "bare", False): 11, ("nsr", "bare", True): 11,
        ("meta", "si", False): 120, ("meta", "si", True): 124,
        ("nsr", "si", False): 104, ("nsr", "si", True): 104,
    }
    # the mirror differences BOTH peers' figures reduce to: 25 bare
    # (date block forms its own system block: ~20 content + 5 scaffold),
    # 20 with an SI (merged into the SI's block, scaffold shared).
    assert counts[("meta", "bare", True)] - counts[("nsr", "bare", True)] == 25
    assert counts[("meta", "si", True)] - counts[("nsr", "si", True)] == 20
    # the merge, shown directly: the date block is present either way,
    # but with an SI there is still only ONE system header in the render.
    sys_id = meta.convert_tokens_to_ids("system")
    for c in ("bare", "si"):
        assert "Cutting Knowledge Date" in meta.decode(r[("meta", c, False)])
        assert r[("meta", c, False)].count(sys_id) == 1
    assert "Cutting Knowledge Date" not in nsr.decode(r[("nsr", "si", False)])


# ---------------------------------------------------------------------------
# synthetic templates on the tiny tokenizer (always run, no gated download)
# ---------------------------------------------------------------------------


def _tiny_tok_with(template):
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(TINY_LLAMA)
    tok.add_special_tokens({"additional_special_tokens": [
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]})
    tok.chat_template = template
    return tok


_TURNS = ("{% for message in messages %}"
          "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}"
          "{{ message['content'] }}{{ '<|eot_id|>' }}"
          "{% endfor %}")

# the NousResearch SHAPE: the assistant header is appended UNCONDITIONALLY
# and add_generation_prompt is never consulted — the exact structure that
# defeated the old suffix arithmetic, reproduced with no gated download.
NSR_SHAPE_TEMPLATE = (
    _TURNS + "{{ '<|start_header_id|>assistant<|end_header_id|>' }}")

# a template that defeats the marker logic too: add_generation_prompt is
# ignored AND no assistant header is appended, so no boundary marks where
# assistant content begins. The old code would return a span starting at
# the assistant HEADER (scaffold-contaminated); the only correct answer
# is a loud refusal.
DEFEAT_TEMPLATE = _TURNS


def test_nousresearch_shape_replica_masks_content_only():
    tok = _tiny_tok_with(NSR_SHAPE_TEMPLATE)
    full, pos = t0.assistant_content_mask(tok, TWO_TURN)
    ctrl = cm.control_token_ids(tok)
    assert tok.decode([full[p] for p in pos]).strip() == "Four."
    assert not [p for p in pos if full[p] in ctrl]
    # the terminal EOT is excluded, and the trailing header never leaks in
    assert full[pos[-1] + 1] == tok.convert_tokens_to_ids("<|eot_id|>")
    # multi-turn WORKS (the real NousResearch template spuriously errored
    # under the old chained full-render prefix check) ...
    full2, pos2 = t0.assistant_content_mask(tok, MULTI_TURN)
    assert tok.decode([full2[p] for p in pos2]) == "Blue.Green and gold."
    # ... and the empty assistant turn contributes no positions (it used
    # to contribute FOUR pure-scaffold positions, none of them special ids)
    full3, pos3 = t0.assistant_content_mask(tok, EMPTY_TURN)
    assert tok.decode([full3[p] for p in pos3]).strip() == "Words."


# the GEMMA-4 SHAPE: post-terminator scaffold ("\n" after the eot)
# lands inside derive_assistant_markers' terminator-anchored header but
# NOT inside the add_generation_prompt append — the located header is
# one token wider than the generation-prompt header. The pinned
# google/gemma-4-E4B-it @ ee0ef602 measures appended 3 vs located 4;
# this tiny replica measures appended 3 vs located 4 the same way.
GEMMA_SHAPE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}"
    "{{ message['content'] }}{{ '<|eot_id|>' }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>' }}{% endif %}")


def test_generation_header_ids_exclude_prompt_side_scaffold():
    """generation_header_ids is the id-form of the FIT-TIME width
    (hooks.response_header_len): on templates whose turns trail scaffold
    after the terminator, the located header absorbs that scaffold (it
    is PROMPT content, present in both renders) while the generation
    ids do not — serving the located header would steer outside the fit
    support (the scale-12 gemma finding, 2026-08-28)."""
    tok = _tiny_tok_with(GEMMA_SHAPE_TEMPLATE)
    markers = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, markers)
    # the absorbed scaffold is exactly the closing's post-terminator
    # tail (the "\n" that follows every turn's eot in this template):
    scaffold = tuple(markers.closing[markers.glue + 1:])
    assert scaffold  # non-empty: this template really has the gemma shape
    assert tuple(markers.header) == scaffold + tuple(gen)
    # ... and the count derivation agrees with the id derivation
    from sisteer.hooks import response_header_len
    assert response_header_len(tok) == len(gen)


def test_generation_header_ids_flag_ignoring_template_uses_the_header():
    """The NousResearch shape: add_generation_prompt is never consulted
    and the header is appended unconditionally, so the render diff is
    empty exactly where the header is always present — there the located
    header IS the appended header, matching response_header_len's
    marker fallback."""
    tok = _tiny_tok_with(NSR_SHAPE_TEMPLATE)
    markers = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, markers)
    assert tuple(gen) == tuple(markers.header)


def test_generation_header_ids_headerless_template_refuses():
    tok = _tiny_tok_with(DEFEAT_TEMPLATE)
    with pytest.raises(cm.ChatMarkerError):
        cm.generation_header_ids(tok)


@pytest.mark.parametrize("model,revision", MIRRORS)
def test_generation_header_ids_on_the_pinned_mirrors(model, revision):
    """Both Llama-3.1 mirrors: no post-terminator scaffold, so the
    generation-prompt header and the located header coincide (H=4) —
    which is why the located-vs-width disagreement first surfaced on
    gemma and not here."""
    tok = _pinned_tok(model, revision)
    markers = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, markers)
    assert tuple(gen) == tuple(markers.header)
    assert len(gen) == 4


def test_defeating_template_raises_instead_of_returning_scaffold():
    tok = _tiny_tok_with(DEFEAT_TEMPLATE)
    with pytest.raises(t0.MaskError,
                       match="does not sit at the content start"):
        t0.assistant_content_mask(tok, TWO_TURN)


def test_surviving_control_token_raises_naming_the_tokens(monkeypatch):
    """The defense-in-depth half: even if span derivation regresses and
    hands back a scaffold-contaminated span, the mask refuses — naming
    the offending tokens — rather than returning it (F11)."""
    tok = _tiny_tok_with(NSR_SHAPE_TEMPLATE)
    full = cm.render_ids(tok, TWO_TURN)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    bad_hi = full.index(eot, 1) + 1  # a span that swallows an EOT
    monkeypatch.setattr(t0, "derive_assistant_markers",
                        lambda _tok: cm.AssistantMarkers((), eot, (eot,)))
    monkeypatch.setattr(t0, "assistant_content_spans",
                        lambda *a, **k: (full, [(1, bad_hi)]))
    with pytest.raises(t0.MaskError, match=r"<\|eot_id\|>"):
        t0.assistant_content_mask(tok, TWO_TURN)


def test_glue_scaffold_is_excluded_on_the_default_llama2_template():
    """The tiny tokenizer's stock template is Llama-2-shaped
    (``[INST] Hi [/INST] Hello there </s>``): the closing scaffold is a
    glue space token then ``</s>``, and the user turn has NO terminator
    (so no header marker is locatable — the probe boundary check still
    guards the content start). The old strip-specials logic kept that
    glue token in the mask; the closing-scaffold arithmetic excludes it."""
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(TINY_LLAMA)
    m = cm.derive_assistant_markers(tok)
    assert m.header == ()
    assert m.terminator == tok.eos_token_id
    assert m.glue == 1 and len(m.closing) == 2  # ['▁', '</s>']
    conv = [{"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there"}]
    full, pos = t0.assistant_content_mask(tok, conv)
    assert tok.decode([full[p] for p in pos]).strip() == "Hello there"
    assert full[pos[-1] + 1 :] == list(m.closing)


# the QWEN THINKING SHAPE (job 255399, 2026-08-30): completed assistant
# turns carry the CLOSED empty think block, while the generation prompt
# carries the OPEN '<think>\n' under the template default (thinking-ON)
# and the closed block only under enable_thinking=False. Unpinned, the
# two probe renders chat_markers compares therefore diverge right after
# '<think>' — read as boundary retokenization, refused. Pinned, both
# carry the closed block and the header derives exactly.
QWEN_THINK_SHAPE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<think>\n\n</think>\n\n' }}{% endif %}"
    "{{ message['content'] }}{{ '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>' }}"
    "{% if enable_thinking is defined and not enable_thinking %}"
    "{{ '<think>\n\n</think>\n\n' }}"
    "{% else %}{{ '<think>\n' }}{% endif %}"
    "{% endif %}")


def _qwen_shape_tok():
    tok = _tiny_tok_with(QWEN_THINK_SHAPE_TEMPLATE)
    tok.add_special_tokens(
        {"additional_special_tokens": ["<think>", "</think>"]})
    return tok


def test_worker_header_verify_needs_the_thinking_pin():
    """The install-time preflight loads a FRESH tokenizer
    (lens_worker._sisteer_load_tokenizer); unpinned, a reasoning
    template renders the generation prompt thinking-ON while probe
    renders carry the closed think block, and the derivation refuses
    (job 255399). Pinned — the fit's regime — the header derives, the
    registry's recorded header verifies, the count equals the fit-side
    derivation, and a genuinely wrong recorded header STILL refuses:
    the pin changes the render regime, never the comparison."""
    from sisteer.capeval.serving import lens_steer as ls
    from sisteer.models import pin_reasoning_template_off

    unpinned = _qwen_shape_tok()
    with pytest.raises(cm.ChatMarkerError):
        cm.generation_header_ids(unpinned)

    pinned = _qwen_shape_tok()
    assert pin_reasoning_template_off(pinned) is True
    gen = cm.generation_header_ids(pinned)
    assert gen  # a real, non-empty header
    # the closed think block IS part of the header the mask keys on
    think = pinned.convert_tokens_to_ids("<think>")
    close = pinned.convert_tokens_to_ids("</think>")
    assert think in gen and close in gen
    # the id-form agrees with the fit-time width derivation
    from sisteer.hooks import response_header_len
    assert response_header_len(pinned) == len(gen)
    # the registry's recorded header verifies against the served render
    ls.verify_assistant_header(pinned, tuple(gen))
    # ... and a genuine mismatch refuses — pinning did not loosen it
    with pytest.raises(ls.LensSteerError):
        ls.verify_assistant_header(pinned, tuple(gen[:-1]))
