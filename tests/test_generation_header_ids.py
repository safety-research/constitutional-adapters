"""generation_header_ids: the id-form of the serve-time response boundary.

Ported subset of the stage-3 lineage's tests for commit 03184be ("The
response boundary is the generation-prompt header, id-located") — the
tests that exercise code present on main (chat_markers +
the lens serving seam). The lineage's full suite for that commit lives
in tests/test_capeval_mask_mirrors.py / test_response_mask_capabilities.py,
which depend on the HF-engine response-mask leg main does not have; when
that lineage merges, its copies supersede these.

THE BUG THESE PIN (scale-12 gemma wave, 2026-08-28): the located header
(``derive_assistant_markers().header``) is anchored at the PREVIOUS
turn's terminator and absorbs post-terminator scaffold — on gemma-4 the
user turn closes ``<eot>\n``, so the located header carries a leading
``\n`` that is PROMPT content. The fit-time mask counts only the tokens
``add_generation_prompt`` appends, so a serve boundary built from the
located header steers one prompt-side token OUTSIDE the fit support.
``generation_header_ids`` derives the appended header directly and is
the ONE id vocabulary for the serve-time boundary.
"""

import pytest

from sisteer import chat_markers as cm

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # conftest pin


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
# and add_generation_prompt is never consulted.
NSR_SHAPE_TEMPLATE = (
    _TURNS + "{{ '<|start_header_id|>assistant<|end_header_id|>' }}")

# no assistant header at all: no boundary can be located; the only
# correct answer is a loud refusal.
DEFEAT_TEMPLATE = _TURNS

# the GEMMA-4 SHAPE: post-terminator scaffold ("\n" after the eot) lands
# inside derive_assistant_markers' terminator-anchored header but NOT
# inside the add_generation_prompt append — the located header is one
# token wider than the generation-prompt header (located 4 vs appended 3
# on the pinned google/gemma-4-E4B-it; same shape replicated here).
GEMMA_SHAPE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}"
    "{{ message['content'] }}{{ '<|eot_id|>' }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>' }}{% endif %}")


def test_generation_header_ids_exclude_prompt_side_scaffold():
    """On templates whose turns trail scaffold after the terminator, the
    located header absorbs that scaffold (it is PROMPT content, present
    in both renders) while the generation ids do not — serving the
    located header would steer outside the fit support."""
    tok = _tiny_tok_with(GEMMA_SHAPE_TEMPLATE)
    markers = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, markers)
    # the absorbed scaffold is exactly the closing's post-terminator
    # tail (the "\n" that follows every turn's eot in this template):
    scaffold = tuple(markers.closing[markers.glue + 1:])
    assert scaffold  # non-empty: this template really has the gemma shape
    assert tuple(markers.header) == scaffold + tuple(gen)


def test_generation_header_ids_flag_ignoring_template_uses_the_header():
    """The NousResearch shape: add_generation_prompt is never consulted
    and the header is appended unconditionally, so the render diff is
    empty exactly where the header is always present — there the located
    header IS the appended header."""
    tok = _tiny_tok_with(NSR_SHAPE_TEMPLATE)
    markers = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, markers)
    assert tuple(gen) == tuple(markers.header)


def test_generation_header_ids_headerless_template_refuses():
    tok = _tiny_tok_with(DEFEAT_TEMPLATE)
    with pytest.raises(cm.ChatMarkerError):
        cm.generation_header_ids(tok)


def test_verify_assistant_header_speaks_generation_ids():
    """The worker-startup cross-check compares the registry's recorded
    header against the GENERATION-PROMPT ids: a registry recorded with
    generation ids verifies, and one recorded with the (wider) located
    header refuses at startup instead of steering a prompt-side token."""
    from sisteer.capeval.serving.lens_steer import (LensSteerError,
                                                    verify_assistant_header)
    tok = _tiny_tok_with(GEMMA_SHAPE_TEMPLATE)
    markers = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, markers)
    assert tuple(markers.header) != tuple(gen)  # the gemma disagreement
    verify_assistant_header(tok, tuple(gen))
    with pytest.raises(LensSteerError, match="assistant header mismatch"):
        verify_assistant_header(tok, tuple(markers.header))


def test_resolve_header_derives_generation_ids():
    """registry_shim's header fallback (no recorded ids, tokenizer given)
    derives the generation-prompt header, not the located one."""
    from sisteer.capeval.serving.registry_shim import _resolve_header
    tok = _tiny_tok_with(GEMMA_SHAPE_TEMPLATE)
    markers = cm.derive_assistant_markers(tok)
    gen = cm.generation_header_ids(tok, markers)
    got = _resolve_header([], None, tok)
    assert got == tuple(gen)
    assert got != tuple(markers.header)
