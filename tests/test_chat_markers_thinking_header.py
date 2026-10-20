"""The assistant header of a REASONING template, per render regime.

The bug (tinycap O27, 2026-09-06): the served-capability driver loaded the
target's tokenizer with a bare ``from_pretrained`` — no thinking pin — and
``chat_markers.generation_header_ids`` refused it on Qwen3.5 with

    the add_generation_prompt render is not a token prefix of the probe
    render — boundary retokenization at token 10 ([198] vs [271, 248069,
    271])

so the cell could not start. The merge is real but incidental: Qwen3.5's
user turn ends in ``\\n`` (198) and the header opens ``\\n`` again, which BPE
joins into ``\\n\\n`` (271). What actually differs is the WIDTH. Under the
template's default (thinking-ON) the generation prompt leaves the reasoning
block OPEN — ``<|im_start|>assistant\\n<think>\\n``, 5 tokens — while a
COMPLETED assistant turn renders the closed empty block,
``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``, 7 tokens. Both are
derivable; neither is "the" header until someone says which render regime
is being served.

Two halves, both pinned here:

* the derivation states its rule on TEXT and anchored at the end of the
  no-generation-prompt user render, so the merge no longer decides. Pinned
  OFF it derives H=7 (the width every Qwen fitted object recorded), pinned
  ON it derives H=5, and UNPINNED it still refuses — naming the pin, which
  is the fix, instead of the merge, which is the symptom;
* gemma-4 (H=3 both arms — it signals thinking with a LEADING system turn)
  and Llama-3.1 (no switch at all, H=4) are unmoved, which is why this was
  invisible until a Qwen cell was launched.

The same refusal had already been diagnosed once, at the lens worker's
install preflight (job 255399, 2026-08-30), and fixed the same way: pin the
render regime. ``tests/test_capeval_mask_mirrors.py`` pins that the unpinned
refusal survives on a synthetic Qwen-shaped template; this module pins the
widths on the real tokenizers.
"""

import pytest

from sisteer import chat_markers as cm

# pinned exactly as the campaign configs pin them (configs/two_si_*.yaml)
QWEN_9B = ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
QWEN_4B = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
GEMMA_E4B = ("google/gemma-4-E4B-it", "ee0ef6023621cff504d758262d4e04895a5af4a2")
GEMMA_E2B = ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7")
LLAMA_8B = ("meta-llama/Llama-3.1-8B-Instruct",
            "0e9e39f249a16976918f6564b8830bc894c89659")

#: (model, revision, thinking arm, expected generation-header width). ``None``
#: for the arm means "as loaded" — no pin at all, the driver's old behaviour.
HEADER_WIDTHS = [
    pytest.param(*QWEN_9B, False, 7, id="qwen35_9b-off"),
    pytest.param(*QWEN_9B, True, 5, id="qwen35_9b-on"),
    pytest.param(*QWEN_4B, False, 7, id="qwen35_4b-off"),
    pytest.param(*QWEN_4B, True, 5, id="qwen35_4b-on"),
    pytest.param(*GEMMA_E4B, False, 3, id="gemma4e4b-off"),
    pytest.param(*GEMMA_E4B, True, 3, id="gemma4e4b-on"),
    pytest.param(*GEMMA_E4B, None, 3, id="gemma4e4b-unpinned"),
    pytest.param(*GEMMA_E2B, False, 3, id="gemma4e2b-off"),
    pytest.param(*GEMMA_E2B, True, 3, id="gemma4e2b-on"),
    pytest.param(*GEMMA_E2B, None, 3, id="gemma4e2b-unpinned"),
    pytest.param(*LLAMA_8B, None, 4, id="llama31_8b-unpinned"),
    pytest.param(*LLAMA_8B, False, 4, id="llama31_8b-off"),
]


def _cached_tok(model, revision):
    """The pinned tokenizer from the LOCAL cache only, with a loud skip
    naming what is missing — the idiom test_capeval_mask_mirrors uses, so
    a clone without these caches skips instead of reaching the network."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=True)
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"pinned {model} tokenizer @ {revision[:8]} not in the "
                    f"local HF cache ({type(e).__name__})")


def _armed(model, revision, thinking):
    """``_cached_tok`` with the render regime pinned to ``thinking``
    (``None`` = left undeclared, as a bare ``from_pretrained`` leaves it).
    The OFF pin goes through ``models.pin_reasoning_template_off`` — the
    production seam — rather than writing the kwargs by hand."""
    from sisteer.models import pin_reasoning_template_off

    tok = _cached_tok(model, revision)
    if thinking is False:
        pin_reasoning_template_off(tok)
    elif thinking is True:
        if not cm._reasoning_template(tok):
            pytest.skip(f"{model} has no thinking switch to pin ON")
        tok._sisteer_template_kwargs = {"enable_thinking": True}
    return tok


@pytest.mark.parametrize("model,revision,thinking,width", HEADER_WIDTHS)
def test_generation_header_width_per_render_regime(model, revision,
                                                   thinking, width):
    """The one number the serve-time response boundary is built from."""
    tok = _armed(model, revision, thinking)
    assert len(cm.generation_header_ids(tok)) == width


@pytest.mark.parametrize("model,revision,thinking,width", HEADER_WIDTHS)
def test_generation_header_agrees_with_the_fit_side_width(model, revision,
                                                          thinking, width):
    """``hooks.response_header_len`` is the fit-time mask's width home and
    ``generation_header_ids`` is its id form; the two are separate
    derivations of ONE convention and must not drift. They disagreed on
    Qwen3.5 before this: hooks answered 5 by bare width arithmetic while
    chat_markers refused outright."""
    from sisteer.hooks import response_header_len

    tok = _armed(model, revision, thinking)
    assert response_header_len(tok) == width


@pytest.mark.parametrize("model,revision",
                         [pytest.param(*QWEN_9B, id="qwen35_9b"),
                          pytest.param(*QWEN_4B, id="qwen35_4b")])
def test_qwen_thinking_off_header_is_the_width_the_fits_recorded(model,
                                                                 revision):
    """H=7, and the ids are the closed empty reasoning block — the render
    every Qwen fitted object's ``response_header_len`` was derived under
    and the one the server pins
    (``vllm_server.THINKING_OFF_TEMPLATE_KWARGS``). A header that came back
    5 here would be refused by the fit/serve header guard on every Qwen
    cell; one that came back 7 by luck on a thinking-ON render would steer
    two of the model's own reasoning tokens as response."""
    tok = _armed(model, revision, False)
    ids = cm.generation_header_ids(tok)
    assert len(ids) == 7
    assert tok.decode(list(ids)) == \
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@pytest.mark.parametrize("model,revision",
                         [pytest.param(*QWEN_9B, id="qwen35_9b"),
                          pytest.param(*QWEN_4B, id="qwen35_4b")])
def test_qwen_thinking_on_header_leaves_the_reasoning_block_open(model,
                                                                 revision):
    """H=5: the model itself emits ``</think>`` and closes the block, so
    the served boundary stops at the open ``<think>``."""
    tok = _armed(model, revision, True)
    ids = cm.generation_header_ids(tok)
    assert len(ids) == 5
    assert tok.decode(list(ids)) == "<|im_start|>assistant\n<think>\n"


@pytest.mark.parametrize("model,revision",
                         [pytest.param(*QWEN_9B, id="qwen35_9b"),
                          pytest.param(*QWEN_4B, id="qwen35_4b")])
def test_unpinned_reasoning_tokenizer_refuses_and_names_the_pin(model,
                                                                revision):
    """The driver's failure, reproduced: with no regime declared BOTH
    widths render, so the derivation refuses. What changed is the message
    — it names the pin (the fix) rather than the token merge (the symptom
    that sent O27's first diagnosis to the wrong layer)."""
    tok = _armed(model, revision, None)
    with pytest.raises(cm.ChatMarkerError) as excinfo:
        cm.generation_header_ids(tok)
    msg = str(excinfo.value)
    assert "pin_reasoning_template_off" in msg
    assert "5 tokens" in msg and "7 tokens" in msg


@pytest.mark.parametrize("model,revision",
                         [pytest.param(*QWEN_9B, id="qwen35_9b"),
                          pytest.param(*QWEN_4B, id="qwen35_4b")])
def test_thinking_on_spans_still_refuse(model, revision):
    """Deriving the ON header does NOT license locating recorded assistant
    CONTENT under it: the generation prompt stops two tokens short of where
    a completed turn's content begins, so ``assistant_content_spans`` has
    no boundary to trust and refuses per message — the invariant the
    relaxed derivation must not have quietly widened."""
    tok = _armed(model, revision, True)
    with pytest.raises(cm.ChatMarkerError):
        cm.assistant_content_spans(
            tok, [{"role": "user", "content": "What is 2+2?"},
                  {"role": "assistant", "content": "Four."}])


@pytest.mark.parametrize("model,revision,thinking,width", HEADER_WIDTHS)
def test_markers_carry_the_generation_header_they_derived(model, revision,
                                                          thinking, width):
    """ONE derivation: ``generation_header_ids`` reads the width off the
    markers the content-substitution probe already produced rather than
    re-rendering and re-deriving it (the second copy this consolidates)."""
    tok = _armed(model, revision, thinking)
    markers = cm.derive_assistant_markers(tok)
    assert markers.generation_header is not None
    assert cm.generation_header_ids(tok, markers) == markers.generation_header
    assert len(markers.generation_header) == width


@pytest.mark.parametrize("model,revision,thinking,width", HEADER_WIDTHS)
def test_located_header_ends_with_the_generation_header(model, revision,
                                                        thinking, width):
    """The terminator-anchored ``header`` absorbs inter-turn glue (gemma-4's
    post-``<turn|>`` newline, Qwen3.5's post-``<|im_end|>`` newline), so it
    is >= the generation header and ends with it — EXCEPT thinking-ON,
    where the generation prompt stops short inside the header and the
    located header instead STARTS with it (after the glue). Pinned so the
    two anchors can never be silently swapped at a serve site."""
    tok = _armed(model, revision, thinking)
    markers = cm.derive_assistant_markers(tok)
    located = tok.decode(list(markers.header))
    generation = tok.decode(list(markers.generation_header))
    assert generation in located
    if thinking is True and cm._reasoning_template(tok) and width == 5:
        assert not located.endswith(generation)   # the open-block case
    else:
        assert located.endswith(generation)
