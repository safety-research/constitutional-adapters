"""Fidelity tests for the stage-3 IRRELEVANT_PROMPT: the verbose,
task-irrelevant control SI, length-matched to CONSTITUTION_PROMPT so a
constitution-vs-irrelevant gap separates safety CONTENT from the mere
off-policy push of any long instruction.

The normative source is the vendored copy
tests/fixtures/specs/irrelevant_si.md (point IRRELEVANT_SI_SPEC at another
copy to check against that instead). The embedded IRRELEVANT_PROMPT must be
byte-identical to that file (sans its single trailing newline); these tests
READ the source file rather than carrying a second copy of the text in
Python, because two copies drift — the same convention as
tests/test_constitution_si.py.

CPU-only, no network: the token-count checks use the locally cached pinned
tokenizer under HF_HUB_OFFLINE=1 and skip — with a reason — when the cache is
absent. The byte-identity check never skips.
"""

from __future__ import annotations

import os
import pathlib

# Must precede any transformers/huggingface_hub import (they read it at
# import time): these tests never touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

from sisteer.data.safety_texts import (
    CONSTITUTION_SI_WORDS,
    IRRELEVANT_PROMPT,
    IRRELEVANT_SI_TOKENS,
    IRRELEVANT_SI_WORDS,
    PROVENANCE,
)

# The vendored normative source ships with the repository, so the check runs
# in every checkout (a plain CPU container included); the override is for
# checking against a copy held elsewhere.
SPEC_PATH = pathlib.Path(
    os.environ.get(
        "IRRELEVANT_SI_SPEC",
        pathlib.Path(__file__).resolve().parent
        / "fixtures/specs/irrelevant_si.md",
    )
)

# The lane's target pin since 2026-08-20 (configs/two_si.yaml): the
# NousResearch mirror. meta-llama @ 0e9e39f2 is grandfathered for the runs
# completed before the switch; the raw SI token count is identical on both
# mirrors (same vocabulary, different chat template).
PINNED_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
PINNED_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
EXPECTED_WORDS = 4930    # measured: len(IRRELEVANT_PROMPT.split())
EXPECTED_TOKENS = 6380   # measured: pinned tokenizer, add_special_tokens=False
# Rendered as a teacher chat prompt (system=IRRELEVANT_PROMPT, empty user
# turn, add_special_tokens=True): SI tokens + 16 of template overhead on the
# NousResearch pin (measured 2026-08-20; the grandfathered meta-llama pin
# rendered 6,416 = +36 overhead). The LONGEST SI in the matrix — it drives
# the config's max_len arithmetic.
EXPECTED_TEACHER_RENDER_TOKENS = 6396
WORD_LIMIT = 5000        # the same hard budget the constitution obeys

TWO_SI_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs/two_si.yaml"


def _spec_text() -> str:
    """The normative source file's text, sans its single trailing newline.

    A missing source FAILS (never skips): byte-identity is the only guard
    against the served prompt drifting from the approved text, and silently
    skipping it would let exactly that drift through.
    """
    if not SPEC_PATH.exists():
        pytest.fail(
            f"normative source {SPEC_PATH} not found — the byte-identity "
            "check cannot run without it. Set IRRELEVANT_SI_SPEC to a copy "
            "of irrelevant_si.md; do not skip this check."
        )
    md = SPEC_PATH.read_text(encoding="utf-8")
    return md[:-1] if md.endswith("\n") else md


def _pinned_tokenizer():
    """The pinned Llama-3.1 tokenizer from the local cache, else skip
    (ONLY the token-count assertions — byte-identity ran regardless)."""
    transformers = pytest.importorskip(
        "transformers", reason="transformers not installed"
    )
    try:
        return transformers.AutoTokenizer.from_pretrained(
            PINNED_MODEL, revision=PINNED_REVISION, local_files_only=True
        )
    except Exception as e:  # cache miss, offline resolution failure, ...
        pytest.skip(
            f"pinned {PINNED_MODEL} tokenizer @ {PINNED_REVISION[:8]} not "
            f"in the local HF cache ({type(e).__name__}: {e}) — skipping "
            "the token-count assertion only; the byte-identity test does "
            "not depend on it"
        )


def test_prompt_byte_identical_to_spec_file():
    text = _spec_text()
    if IRRELEVANT_PROMPT != text:
        n = min(len(IRRELEVANT_PROMPT), len(text))
        i = next((k for k in range(n) if IRRELEVANT_PROMPT[k] != text[k]), n)
        pytest.fail(
            "IRRELEVANT_PROMPT is not byte-identical to the approved text "
            f"in {SPEC_PATH}: lengths {len(IRRELEVANT_PROMPT)} vs "
            f"{len(text)}, first divergence at char {i}: "
            f"{IRRELEVANT_PROMPT[i : i + 40]!r} vs {text[i : i + 40]!r}"
        )


def test_word_count_length_matched_and_under_budget():
    assert len(IRRELEVANT_PROMPT.split()) == EXPECTED_WORDS
    assert IRRELEVANT_SI_WORDS == EXPECTED_WORDS
    assert EXPECTED_WORDS <= WORD_LIMIT
    # length-matched to the constitution within 1% (the gap must separate
    # CONTENT from LENGTH, so the lengths must not differ)
    assert abs(EXPECTED_WORDS - CONSTITUTION_SI_WORDS) <= (
        CONSTITUTION_SI_WORDS * 0.01
    )


def test_registered_exactly_once_and_resolvable():
    from sisteer.data.instructions import REGISTRY, resolve_si
    from sisteer.eval.jailbreak import resolve_instruction

    matches = [i for i in REGISTRY if i.id == "si_irrelevant"]
    assert len(matches) == 1, "si_irrelevant must be registered exactly once"
    assert matches[0].text == IRRELEVANT_PROMPT
    assert matches[0].category == "control"
    assert resolve_si("si_irrelevant") is matches[0]
    # the two-SI resolver (what the fit driver and generate paths use)
    assert resolve_instruction("si_irrelevant") is matches[0]


def test_provenance_recorded():
    assert "IRRELEVANT_PROMPT" in PROVENANCE
    assert "two-si-stage3" in PROVENANCE["IRRELEVANT_PROMPT"]
    assert "4,930" in PROVENANCE["IRRELEVANT_PROMPT"]
    assert "6,380" in PROVENANCE["IRRELEVANT_PROMPT"]


def test_token_count_measured_with_pinned_tokenizer():
    tok = _pinned_tokenizer()
    n_tokens = len(
        tok(IRRELEVANT_PROMPT, add_special_tokens=False)["input_ids"]
    )
    assert n_tokens == EXPECTED_TOKENS == IRRELEVANT_SI_TOKENS

    from sisteer.models import build_chat_prompt

    rendered = build_chat_prompt(tok, "", system=IRRELEVANT_PROMPT)
    n_rendered = len(tok(rendered, add_special_tokens=True)["input_ids"])
    assert n_rendered == EXPECTED_TEACHER_RENDER_TOKENS


def test_two_si_config_carries_the_third_si_and_its_max_len():
    yaml = pytest.importorskip("yaml", reason="pyyaml not installed")
    cfg = yaml.safe_load(TWO_SI_CONFIG.read_text(encoding="utf-8"))
    assert "si_irrelevant" in cfg["si_ids"]
    # its per-SI max_len holds the rendered teacher prompt + the shared
    # completion cap + EOS (the same arithmetic the constitution test runs)
    assert cfg["max_len_by_si"]["si_irrelevant"] >= (
        EXPECTED_TEACHER_RENDER_TOKENS + cfg["completion_tokens"] + 1
    )
    tok = _pinned_tokenizer()
    from sisteer.models import build_chat_prompt

    rendered = build_chat_prompt(tok, "", system=IRRELEVANT_PROMPT)
    n_rendered = len(tok(rendered, add_special_tokens=True)["input_ids"])
    assert cfg["max_len"] >= n_rendered + cfg["completion_tokens"] + 1, (
        f"max_len={cfg['max_len']} cannot hold the rendered irrelevant-SI "
        f"teacher prompt ({n_rendered}) + completion cap "
        f"({cfg['completion_tokens']}) + EOS"
    )
