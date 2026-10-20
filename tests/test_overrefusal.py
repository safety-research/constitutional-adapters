"""Pure-logic tests for the over-refusal measurement scripts."""

import json
import sys
from pathlib import Path

import pyarrow as pa

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from alphakl_overrefusal import orbench_rows, xstest_rows  # noqa: E402
from judge_overrefusal import item_key, parse_label  # noqa: E402


def test_xstest_split_contrast_types_are_the_unsafe_half():
    table = pa.table({
        "type": ["homonyms", "contrast_homonyms", "safe_targets",
                 "contrast_privacy"],
        "prompt": ["a", "b", "c", "d"],
    })
    rows = xstest_rows(table)
    assert [r["subset"] for r in rows] == ["safe", "unsafe", "safe", "unsafe"]
    assert all(r["dataset"] == "xstest" for r in rows)


def test_orbench_rows_are_all_safe():
    table = pa.table({"prompt": ["p1", "p2"], "category": ["deception", "harm"]})
    rows = orbench_rows(table)
    assert all(r["subset"] == "safe" for r in rows)
    assert rows[0]["kind"] == "deception"


def test_parse_label_tolerates_wrapping_and_rejects_junk():
    assert parse_label("C") == "C"
    assert parse_label("  I.") == "I"
    assert parse_label("label: p") == "P"
    assert parse_label("compliance") is None  # bare word, no isolated letter
    assert parse_label("") is None


def test_item_key_is_stable_and_collision_averse():
    a = {"prompt": "p", "response": "r"}
    assert item_key(a) == item_key(dict(a))
    assert item_key(a) != item_key({"prompt": "pr", "response": ""})


def test_cpi_aggregation_matches_xstest_shape():
    from collections import Counter

    from sisteer.eval.jailbreak_artifacts import xstest_cpi

    cpi = xstest_cpi(Counter({"C": 6, "P": 1, "I": 3}))
    assert cpi["n"] == 10
    assert cpi["full_refusal_frac"] == 0.3
    assert cpi["partial_refusal_frac"] + cpi["full_refusal_frac"] == 0.4
