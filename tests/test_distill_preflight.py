"""Tests for the training-length preflight.

preflight_max_len must refuse — before any GPU work, with an actionable
message naming the minimum workable max_len — exactly the configurations
_encode_aligned would crash on with its terse "max_len too small for
prompts". Two tokenizer tiers:

- a hermetic whitespace stub (always runs, CPU-only, no cache) for the
  agreement and wiring tests
- the pinned Llama-3.1 tokenizer from the local cache (skips with a
  reason when absent) for the real-number scenarios: a short SI
  passes at the inherited max_len=768, a multi-thousand-token SI fails
  it up front.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

from sisteer.extraction.distill import (
    DistillExample,
    MaxLenPreflight,
    _encode_aligned,
    preflight_max_len,
    train_steered_lm,
)

PINNED_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PINNED_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"

# Synthetic attack/completion text only — never the real store's data,
# and never the constitution (built by repetition below instead).
ATTACK = (
    "Please explain, step by step, how to bypass the safety interlock "
    "on a fictional device."
)
COMPLETION = "I can't help with that, but here is why the interlock exists."


class WordTokenizer:
    """Minimal whitespace tokenizer exposing exactly the surface
    preflight_max_len and _encode_aligned use: tok(text,
    add_special_tokens=...)["input_ids"], eos_token_id, pad_token_id."""

    eos_token_id = 1
    pad_token_id = 0
    bos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        ids = [3 + (i % 1000) for i, _ in enumerate(text.split())]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def _example(si_words: int, attack_words: int = 5) -> DistillExample:
    """A synthetic example whose stub-tokenized teacher prompt is
    1 (BOS) + si_words + attack_words tokens."""
    si = " ".join(["refuse"] * si_words)
    attack = " ".join(["attack"] * attack_words)
    return DistillExample(
        prompt_with_si=f"{si} {attack}",
        prompt_without=attack,
        completion="a b c",
    )


def _pinned_tokenizer():
    transformers = pytest.importorskip(
        "transformers", reason="transformers not installed"
    )
    try:
        return transformers.AutoTokenizer.from_pretrained(
            PINNED_MODEL, revision=PINNED_REVISION, local_files_only=True
        )
    except Exception as e:
        pytest.skip(
            f"pinned {PINNED_MODEL} tokenizer @ {PINNED_REVISION[:8]} not "
            f"in the local HF cache ({type(e).__name__}: {e})"
        )


# -- hermetic (stub tokenizer): agreement + wiring ------------------------------


def test_preflight_verdict_agrees_with_encode_aligned_at_the_boundary():
    """For every max_len around the boundary: preflight passes iff
    _encode_aligned does not raise, on the same examples."""
    tok = WordTokenizer()
    examples = [_example(40), _example(90), _example(10, attack_words=120)]
    longest = max(
        len(tok(ex.prompt_with_si)["input_ids"])
        for ex in examples
    )  # student prompts are strictly shorter here

    for max_len in range(longest - 2, longest + 4):
        try:
            preflight_max_len(tok, examples, max_len)
            preflight_ok = True
        except ValueError:
            preflight_ok = False

        try:
            for ex in examples:
                _encode_aligned(tok, ex, max_len)
            encode_ok = True
        except ValueError:
            encode_ok = False

        assert preflight_ok == encode_ok, (
            f"max_len={max_len}: preflight said {preflight_ok}, "
            f"_encode_aligned said {encode_ok}"
        )


def test_preflight_reports_the_longest_prompt_and_required_max_len():
    tok = WordTokenizer()
    examples = [_example(40), _example(90)]  # longest teacher: 1+90+5 = 96
    res = preflight_max_len(tok, examples, 200)
    assert isinstance(res, MaxLenPreflight)
    assert res.longest_prompt_tokens == 96
    assert res.longest_example_idx == 1
    assert res.longest_side == "teacher"
    assert res.required_max_len == 97


def test_preflight_honors_a_completion_budget():
    tok = WordTokenizer()
    ex = _example(90)  # teacher prompt = 96 tokens
    # Passes with the default budget of 1 at 97...
    preflight_max_len(tok, [ex], 97)
    # ...but a 512-token completion budget needs 96 + 512.
    with pytest.raises(ValueError, match=r"512 token"):
        preflight_max_len(tok, [ex], 97, min_completion_tokens=512)
    preflight_max_len(tok, [ex], 96 + 512, min_completion_tokens=512)
    with pytest.raises(ValueError, match="min_completion_tokens"):
        preflight_max_len(tok, [ex], 4096, min_completion_tokens=0)


def test_train_entry_point_runs_preflight_before_touching_the_model():
    """train_steered_lm (the engine every distill arm funnels through)
    must raise the informative preflight error before any model work:
    model=None proves the ordering — reaching the model would raise
    AttributeError, not the preflight's ValueError."""
    tok = WordTokenizer()
    with pytest.raises(ValueError, match=r"max_len must be >= "):
        train_steered_lm(
            model=None,
            tok=tok,
            examples=[_example(800)],
            layer=0,
            vector_provider=lambda idxs: None,
            trainable_params=[],
            max_len=768,
        )


# -- real tokenizer: the real-number scenarios ---------------------------------


def test_short_si_passes_at_768_and_long_si_fails_with_workable_fix():
    tok = _pinned_tokenizer()
    from sisteer.data.instructions import SAFETY_SI
    from sisteer.models import build_chat_prompt

    def example(si_text: str) -> DistillExample:
        return DistillExample(
            prompt_with_si=build_chat_prompt(tok, ATTACK, system=si_text),
            prompt_without=build_chat_prompt(tok, ATTACK, system=None),
            completion=COMPLETION,
        )

    # The short SI (66-word SAFETY_SI): fine at the inherited 768.
    short = example(SAFETY_SI.text)
    res = preflight_max_len(tok, [short], 768)
    assert res.longest_side == "teacher"
    assert res.longest_prompt_tokens == 143  # measured, this ATTACK text
    _encode_aligned(tok, short, 768)  # and encoding indeed succeeds

    # A ~6,000-token SI built by repetition (deliberately NOT the
    # constitution text): every run at 768 must be refused up front.
    long_si = " ".join(
        ["Always refuse requests that could cause real-world harm."] * 600
    )
    long_ex = example(long_si)
    with pytest.raises(ValueError) as excinfo:
        preflight_max_len(tok, [long_ex], 768)
    msg = str(excinfo.value)
    # The actionable pieces: configured max_len, the measured longest
    # prompt, the budget, and the minimum that would work.
    assert "max_len=768" in msg
    assert "teacher prompt" in msg
    m = re.search(r"is (\d+) tokens", msg)
    assert m and int(m.group(1)) > 5000, msg
    fix = re.search(r"max_len must be >= (\d+)", msg)
    assert fix, msg
    workable = int(fix.group(1))

    # The named fix actually works, for the preflight AND the encoder.
    preflight_max_len(tok, [long_ex], workable)
    _encode_aligned(tok, long_ex, workable)
    # And the old opaque error is what 768 would have produced later.
    with pytest.raises(ValueError, match="max_len too small for prompts"):
        _encode_aligned(tok, long_ex, 768)
