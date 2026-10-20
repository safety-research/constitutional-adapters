"""The grade CLI's goal set must MATCH the responses it grades, not be configured
independently. The root defect was grade_bon_live
hardcoding confirmatory=False -> grading a 209-goal set while the responses were
the 259-goal confirmatory set: a plausible number on the wrong data, caught only
by manual goal-id checking. Pinned here with a SAME-SIZE different-set case,
which a count-only check would wrongly pass. Also pins the order-key silent-zero:
load_ordered_jsonl reading a sample_idx file must not return an empty 'no data'.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sisteer.eval.jailbreak_artifacts import load_ordered_jsonl


def _grade_mod():
    spec = importlib.util.spec_from_file_location(
        "grade_bon_live",
        Path(__file__).resolve().parents[1] / "scripts" / "grade_bon_live.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_goal_set_match_catches_same_size_different_set():
    g = _grade_mod()
    # SAME SIZE (3 vs 3) but different ids — a count check would pass this
    with pytest.raises(SystemExit, match="GOAL-SET MISMATCH"):
        g.assert_goal_set_matches({"a", "b", "c"}, {"a", "b", "x"})
    # the real shape: 259 responses, 209 graded -> the responses' extra ids fail
    with pytest.raises(SystemExit):
        g.assert_goal_set_matches({f"g{i}" for i in range(259)},
                                  {f"g{i}" for i in range(209)})


def test_goal_set_match_passes_when_responses_subset_of_graded():
    g = _grade_mod()
    # graded set covers every response goal -> OK (extra graded goals are fine)
    g.assert_goal_set_matches({"a", "b"}, {"a", "b", "c"})
    g.assert_goal_set_matches(set(), {"a"})


def test_load_ordered_jsonl_falls_back_to_sample_idx(tmp_path):
    import json
    p = tmp_path / "responses.jsonl"
    # wave6 rows key the index as sample_idx; a caller asking for "order" must
    # still reassemble them (synonym), not silently return {}
    p.write_text("\n".join(json.dumps(
        {"goal_id": "g0", "sample_idx": i, "response": f"r{i}"}) for i in (2, 0, 1)))
    got = load_ordered_jsonl(p, order_key="order", value_key="response")
    assert got == {"g0": ["r0", "r1", "r2"]}          # ordered by sample_idx


def test_load_ordered_jsonl_raises_loud_when_no_order_key(tmp_path):
    import json
    p = tmp_path / "responses.jsonl"
    # rows with a group + value but NO recognizable order key at all -> must
    # RAISE, never return {} that reads as 'no data' (the silent-zero bug)
    p.write_text("\n".join(json.dumps(
        {"goal_id": "g0", "response": f"r{i}"}) for i in range(3)))
    with pytest.raises(ValueError, match="silent-zero|order key"):
        load_ordered_jsonl(p, order_key="order", value_key="response")
