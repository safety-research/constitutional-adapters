"""OR-Bench 80k / hard-1k leakage guard — content-level subtraction tests.

OR-Bench-hard-1k is a near-verbatim subset of or-bench-80k (1,318 of its
1,319 rows appear exactly in the 80k file, measured 2026-08-19 at the pinned
revision), so every 80k fit sample must subtract the hard-1k eval rows by
NORMALISED exact match (strip, lowercase, collapse whitespace) BEFORE the
deterministic subsample. These tests exercise that guard on SYNTHETIC
strings only — invented prompts, no real OR-Bench rows (repo privacy rule).
"""

import pytest

from sisteer.data.content_sets import (
    load_orbench,
    subtract_eval_controls,
    subtract_hard_1k,
)

# Invented stand-ins for hard-1k eval rows (all benign, all synthetic).
HARD_1K = [
    "Explain how a fictional detective catalogues rare buttons.",
    "Describe the migration pattern of imaginary glass butterflies.",
    "Outline a training plan for a competitive cloud-watching team.",
]

# Invented stand-ins for 80k rows genuinely NOT in hard-1k.
CLEAN = [
    "Write a haiku about a lighthouse that runs on clockwork.",
    "Summarize the rules of an invented board game called Quibble.",
    "Draft a newsletter announcing the town's annual kite festival.",
]

# A prompt that shares most words with HARD_1K[1] but genuinely differs
# (one substituted word) — must survive the subtraction.
NEAR_MISS = "Describe the migration pattern of imaginary glass moths."

# Normalisation-equivalent variants of hard-1k rows: whitespace-only and
# case-only perturbations that a raw-bytes match would miss.
WHITESPACE_VARIANT = (
    "  Explain how a fictional\tdetective   catalogues rare buttons.  ")
CASE_VARIANT = HARD_1K[2].upper()


def test_subtract_hard_1k_removes_exactly_the_leaked_rows():
    eighty_k = [
        CLEAN[0],
        HARD_1K[0],            # verbatim leak
        CLEAN[1],
        WHITESPACE_VARIANT,    # whitespace-only variant of HARD_1K[0]
        CASE_VARIANT,          # case-only variant of HARD_1K[2]
        HARD_1K[1],            # verbatim leak
        NEAR_MISS,             # genuinely different — must be kept
        CLEAN[2],
    ]
    got = subtract_hard_1k(eighty_k, HARD_1K)
    # Exactly the leaked rows (verbatim + normalised variants) are gone;
    # everything else survives in input order.
    assert got == [CLEAN[0], CLEAN[1], NEAR_MISS, CLEAN[2]]


def test_subtract_hard_1k_matches_normalised_not_raw():
    # A whitespace/case-only variant IS removed (normalised match)...
    assert subtract_hard_1k([WHITESPACE_VARIANT], HARD_1K) == []
    assert subtract_hard_1k([CASE_VARIANT], HARD_1K) == []
    # ...while a genuinely different prompt is kept.
    assert subtract_hard_1k([NEAR_MISS], HARD_1K) == [NEAR_MISS]


def test_subtract_eval_controls_same_normalised_semantics():
    prompts = [CLEAN[0], WHITESPACE_VARIANT, NEAR_MISS]
    got = subtract_eval_controls(prompts, HARD_1K)
    assert got == [CLEAN[0], NEAR_MISS]


def _write_csv(path, prompts):
    import pandas as pd

    pd.DataFrame({"prompt": prompts,
                  "category": ["synthetic"] * len(prompts)}
                 ).to_csv(path, index=False)
    return str(path)


def test_load_orbench_subtracts_hard_1k_before_subsample(tmp_path):
    eighty = _write_csv(tmp_path / "eighty.csv",
                        CLEAN + HARD_1K + [CASE_VARIANT])
    hard = _write_csv(tmp_path / "hard.csv", HARD_1K)
    # n exceeds what survives the subtraction: the realised sample must be
    # honest (only the 3 clean rows), proving the subtraction ran BEFORE
    # the deterministic subsample rather than after it.
    got = load_orbench("or-bench-80k", n=6, path=eighty, hard_1k_path=hard)
    assert sorted(got) == sorted(CLEAN)


def test_load_orbench_path_override_without_hard_1k_skips_subtraction(tmp_path):
    # Offline `path` override with no hard_1k_path: caller owns the file's
    # contents and tests must never hit the hub — no subtraction.
    eighty = _write_csv(tmp_path / "eighty.csv", CLEAN + HARD_1K)
    got = load_orbench("or-bench-80k", path=eighty)
    assert sorted(got) == sorted(CLEAN + HARD_1K)


def test_load_orbench_rejects_hard_1k_path_on_non_80k_config(tmp_path):
    hard = _write_csv(tmp_path / "hard.csv", HARD_1K)
    with pytest.raises(ValueError, match="hard_1k_path"):
        load_orbench("or-bench-hard-1k", path=hard, hard_1k_path=hard)
