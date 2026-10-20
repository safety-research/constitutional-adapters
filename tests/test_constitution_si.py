"""Fidelity tests for the expanded 8-section CONSTITUTION_SI.

The normative source is the fenced ```text block in the vendored copy
tests/fixtures/specs/constitution_si.md (point CONSTITUTION_SI_SPEC at
another copy to check against that instead). The assembled
CONSTITUTION_PROMPT must be byte-identical to that block; these tests
PARSE the source file rather than carrying a second copy of the text in
Python, because two copies drift.

CPU-only, no network: the token-count checks use the locally cached
pinned tokenizer under HF_HUB_OFFLINE=1 and skip — with a reason —
when the cache is absent. The byte-identity check never skips.
"""

from __future__ import annotations

import os
import pathlib
import re

# Must precede any transformers/huggingface_hub import (they read it at
# import time): these tests never touch the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

from sisteer.data.safety_texts import (
    CONSTITUTION_PROMPT,
    CONSTITUTION_SECTIONS,
    CONSTITUTION_SI_TOKENS,
    CONSTITUTION_SI_WORDS,
    PROVENANCE,
)

# The vendored normative source ships with the repository, so the check runs
# in every checkout (a plain CPU container included); the override is for
# checking against a copy held elsewhere.
SPEC_PATH = pathlib.Path(
    os.environ.get(
        "CONSTITUTION_SI_SPEC",
        pathlib.Path(__file__).resolve().parent
        / "fixtures/specs/constitution_si.md",
    )
)

# The eight published headings, in constitution document order. Note the
# CURLY apostrophe (U+2019) in the first — the published form, and the
# one documented one-character difference from the old repo dict key.
EXPECTED_HEADINGS = (
    "Claude’s core values",
    "Balancing helpfulness with other values",
    "Being broadly ethical",
    "Being honest",
    "Avoiding harm",
    "The costs and benefits of actions",
    "Being broadly safe",
    "Safe behaviors",
)

WORD_LIMIT = 5000  # hard limit (spec)
EXPECTED_WORDS = 4896  # measured: len(CONSTITUTION_PROMPT.split())
EXPECTED_TOKENS = 5872  # measured: pinned tokenizer, add_special_tokens=False
# Rendered as a teacher chat prompt (system=CONSTITUTION_PROMPT, empty
# user turn, add_special_tokens=True): SI tokens + 16 of template
# overhead on the NousResearch pin (measured 2026-08-20). The
# grandfathered meta-llama pin the completed runs used rendered 5,908
# (+36 overhead — its template silently prepends a 25-token knowledge-
# cutoff system block); the spec's 5,913 figure is THAT render plus a
# ~5-token synthetic attack prompt. The raw SI token count (5,872) is
# identical on both mirrors — same vocabulary, different template.
EXPECTED_TEACHER_RENDER_TOKENS = 5888

# The lane's target pin since 2026-08-20 (configs/two_si.yaml — asserted
# below): the NousResearch mirror. meta-llama @ 0e9e39f2 is grandfathered
# for the runs completed before the switch.
PINNED_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
PINNED_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"

TWO_SI_CONFIG = pathlib.Path(__file__).resolve().parents[1] / "configs/two_si.yaml"


def _spec_block() -> str:
    """The fenced ```text block from the normative source file.

    A missing source FAILS (never skips): byte-identity is the only
    guard against the served prompt drifting from the published text,
    and silently skipping it would let exactly that drift through.
    """
    if not SPEC_PATH.exists():
        pytest.fail(
            f"normative source {SPEC_PATH} not found — the byte-identity "
            "check cannot run without it. Set CONSTITUTION_SI_SPEC to a "
            "copy of constitution_si.md; do not skip this check."
        )
    md = SPEC_PATH.read_text(encoding="utf-8")
    m = re.search(r"^```text\n(.*?)\n```\s*$", md, flags=re.S | re.M)
    assert m, f"no fenced ```text block found in {SPEC_PATH}"
    return m.group(1)


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


def test_prompt_byte_identical_to_spec_block():
    block = _spec_block()
    if CONSTITUTION_PROMPT != block:
        n = min(len(CONSTITUTION_PROMPT), len(block))
        i = next(
            (k for k in range(n) if CONSTITUTION_PROMPT[k] != block[k]), n
        )
        pytest.fail(
            "CONSTITUTION_PROMPT is not byte-identical to the fenced "
            f"block in {SPEC_PATH}: lengths {len(CONSTITUTION_PROMPT)} vs "
            f"{len(block)}, first divergence at char {i}: "
            f"{CONSTITUTION_PROMPT[i : i + 40]!r} vs {block[i : i + 40]!r}"
        )


def test_under_word_limit_with_exact_recorded_count():
    n_words = len(CONSTITUTION_PROMPT.split())
    assert n_words < WORD_LIMIT, (
        f"CONSTITUTION_PROMPT is {n_words} words — over the hard "
        f"{WORD_LIMIT}-word limit"
    )
    # The exact measured count, recorded here and as the module constant.
    assert n_words == EXPECTED_WORDS == CONSTITUTION_SI_WORDS


def test_eight_headings_present_in_document_order():
    assert tuple(CONSTITUTION_SECTIONS) == EXPECTED_HEADINGS
    # And each heading renders into the assembled prompt, in order.
    pos = -1
    for heading in EXPECTED_HEADINGS:
        nxt = CONSTITUTION_PROMPT.find(heading, pos + 1)
        assert nxt > pos, f"heading {heading!r} missing or out of order"
        pos = nxt
    # The apostrophe is the published curly form, consistently: the
    # straight-quote variant of the first heading must NOT appear.
    assert "Claude's core values" not in CONSTITUTION_PROMPT


def test_si_constitution_resolves_from_registry_to_this_exact_text():
    from sisteer.data.instructions import REGISTRY, resolve_si

    matches = [i for i in REGISTRY if i.id == "si_constitution"]
    assert len(matches) == 1, "si_constitution must be registered exactly once"
    assert matches[0].text == CONSTITUTION_PROMPT
    assert resolve_si("si_constitution") is matches[0]


def test_provenance_recorded():
    assert "2026-08-18" in PROVENANCE["CONSTITUTION_SECTIONS"]
    assert "CC0" in PROVENANCE["CONSTITUTION_SECTIONS"]
    assert "anthropic.com/constitution" in PROVENANCE["CONSTITUTION_SECTIONS"]
    # Complete-vs-excerpted accounting is part of the record.
    assert "Complete" in PROVENANCE["CONSTITUTION_SECTIONS"]
    assert "Excerpted" in PROVENANCE["CONSTITUTION_SECTIONS"]
    assert "5,872" in PROVENANCE["CONSTITUTION_PROMPT"]
    assert "4,896" in PROVENANCE["CONSTITUTION_PROMPT"]


def test_token_count_measured_with_pinned_tokenizer():
    tok = _pinned_tokenizer()
    n_tokens = len(
        tok(CONSTITUTION_PROMPT, add_special_tokens=False)["input_ids"]
    )
    assert n_tokens == EXPECTED_TOKENS == CONSTITUTION_SI_TOKENS

    # The number training actually pays: the SI rendered as a teacher
    # chat prompt (before the attack text) — what max_len must cover.
    from sisteer.models import build_chat_prompt

    rendered = build_chat_prompt(tok, "", system=CONSTITUTION_PROMPT)
    n_rendered = len(tok(rendered, add_special_tokens=True)["input_ids"])
    assert n_rendered == EXPECTED_TEACHER_RENDER_TOKENS


# -- configs/two_si.yaml: the constitution-capable max_len ---------------


def _two_si_cfg() -> dict:
    yaml = pytest.importorskip("yaml", reason="pyyaml not installed")
    return yaml.safe_load(TWO_SI_CONFIG.read_text(encoding="utf-8"))


def test_two_si_config_declares_both_arms_and_six_controls():
    cfg = _two_si_cfg()
    # stage 3 added the length-matched irrelevant control as a third SI
    # (tests/test_irrelevant_si.py byte-locks its text)
    assert cfg["si_ids"] == ["safety_refusal", "si_constitution",
                             "si_irrelevant"]
    assert cfg["model"] == PINNED_MODEL
    assert cfg["model_revision"] == PINNED_REVISION
    assert cfg["dtype"] == "bfloat16"
    assert cfg["layer"] == 15
    assert cfg["scale"] == 1.0
    # Shared completion cap: applied to BOTH arms so supervision volume
    # is not a third difference between them.
    assert cfg["completion_tokens"] == 512
    # the declared control set, never silently trimmed;
    # FOUR since the rule of 2026-09-11 retired alpaca_eval and wmdp.
    assert cfg["controls"] == [
        "xstest",
        "or_bench_hard_1k",
        "mmlu",
        "gsm8k",
    ]


def test_two_si_config_max_len_holds_the_constitution():
    cfg = _two_si_cfg()
    # Never the inherited stage-2 value (the crash it caused).
    assert cfg["max_len"] != 768
    tok = _pinned_tokenizer()
    from sisteer.models import build_chat_prompt

    rendered = build_chat_prompt(tok, "", system=CONSTITUTION_PROMPT)
    n_rendered = len(tok(rendered, add_special_tokens=True)["input_ids"])
    # Recompute the config's own arithmetic: rendered SI prompt + the
    # shared completion cap + EOS must fit, with the attack-prompt
    # allowance on top (the config comment derives 8192 from these).
    assert cfg["max_len"] >= n_rendered + cfg["completion_tokens"] + 1, (
        f"max_len={cfg['max_len']} cannot hold the rendered constitution "
        f"teacher prompt ({n_rendered}) + completion cap "
        f"({cfg['completion_tokens']}) + EOS"
    )
