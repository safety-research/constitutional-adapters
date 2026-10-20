"""The exact-three-seed completeness gate: rows are counted per seed BEFORE
any reduction, duplicates refuse in either insertion order (identical rows included), a
foreign seed refuses, an incomplete or non-eligible triplet is `unavailable` naming the
seed and reason, and no spread is ever computed over survivors."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
if str(SEEDREP) not in sys.path:
    sys.path.insert(0, str(SEEDREP))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SEEDREP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load("triplet_gate")
Record = tg.Record


def test_complete_triplet_returns_three_values_in_seed_order():
    t = tg.require_triplet([Record(3, 0.3, True), Record(1, 0.1, True), Record(2, 0.2, True)])
    assert t.values == {1: 0.1, 2: 0.2, 3: 0.3}
    assert t.as_list() == [0.1, 0.2, 0.3]
    assert t[2] == 0.2
    assert [r.seed for r in t.records] == [1, 2, 3]


def test_plain_tuples_and_dicts_are_accepted_as_records():
    t = tg.require_triplet([(1, "a", True), (2, "b", True, None), {"seed": 3, "value": "c", "eligible": True}])
    assert t.as_list() == ["a", "b", "c"]


def test_missing_seed_is_unavailable_naming_the_seed():
    with pytest.raises(tg.TripletUnavailable) as ei:
        tg.require_triplet([Record(1, 1.0, True), Record(2, 2.0, True)])
    msg = str(ei.value)
    assert msg.startswith("unavailable: incomplete replicate triplet")
    assert "s3 — no store" in msg
    assert ei.value.missing == [(3, "no store")]
    trip, why = tg.gate([Record(1, 1.0, True), Record(2, 2.0, True)])
    assert trip is None and why == msg


def test_present_but_ineligible_seed_is_unavailable_with_its_reason():
    recs = [Record(1, 1.0, True), Record(2, None, False, "not admitted: recovery verdict not recovered"),
            Record(3, 3.0, True)]
    with pytest.raises(tg.TripletUnavailable) as ei:
        tg.require_triplet(recs)
    assert "s2 — not admitted: recovery verdict not recovered" in str(ei.value)
    # an ineligible record with no reason still names the seed
    with pytest.raises(tg.TripletUnavailable) as ei2:
        tg.require_triplet([Record(1, 1.0, True), Record(2, 2.0, True), Record(3, None, False)])
    assert "s3 — not eligible" in str(ei2.value)


@pytest.mark.parametrize("order", [
    [Record(1, 1.0, True), Record(2, 2.0, True), Record(3, 3.0, True), Record(3, 3.5, True)],   # s3a then s3b
    [Record(3, 3.5, True), Record(1, 1.0, True), Record(3, 3.0, True), Record(2, 2.0, True)],   # s3b, ..., s3a
])
def test_two_distinct_records_for_one_seed_refuse_in_either_order(order):
    with pytest.raises(tg.DuplicateReplicate) as ei:
        tg.require_triplet(order)
    assert "duplicate replicate for s3 (2 records)" in str(ei.value)
    assert isinstance(ei.value, tg.TripletRefused)
    # a refusal is a fault, not a gap: gate() does not swallow it into an `unavailable`
    with pytest.raises(tg.DuplicateReplicate):
        tg.gate(order)


def test_two_identical_records_for_one_seed_refuse_never_deduplicated():
    dup = Record(3, 3.0, True)
    with pytest.raises(tg.DuplicateReplicate):
        tg.require_triplet([Record(1, 1.0, True), Record(2, 2.0, True), dup, dup])
    # even when the duplicate is ineligible on one copy — the count is what refuses
    with pytest.raises(tg.DuplicateReplicate):
        tg.require_triplet([Record(1, 1.0, True), Record(2, 2.0, True), Record(3, 3.0, True), Record(3, None, False, "x")])


def test_mapping_form_with_two_element_sequence_refuses_identically():
    with pytest.raises(tg.DuplicateReplicate) as ei:
        tg.require_triplet({1: [Record(1, 1.0, True)], 2: [Record(2, 2.0, True)],
                            3: [Record(3, 3.0, True), Record(3, 3.0, True)]})
    assert "s3 (2 records)" in str(ei.value)
    # the happy mapping form works and a bare record under a key is one row, not two
    t = tg.require_triplet({1: [Record(1, 1.0, True)], 2: Record(2, 2.0, True), 3: [(3, 3.0, True)]})
    assert t.as_list() == [1.0, 2.0, 3.0]
    # a record filed under the wrong key is a fault
    with pytest.raises(tg.TripletRefused):
        tg.require_triplet({1: [Record(2, 1.0, True)], 2: [Record(2, 2.0, True)], 3: [Record(3, 3.0, True)]})


def test_foreign_seed_refuses_the_original_is_not_a_replicate():
    with pytest.raises(tg.UnexpectedSeed) as ei:
        tg.require_triplet([Record(0, 0.0, True), Record(1, 1.0, True), Record(2, 2.0, True), Record(3, 3.0, True)])
    assert "unexpected seed s0" in str(ei.value)
    assert "never a replicate" in str(ei.value)


def test_duplicate_is_checked_before_completeness():
    # s2 missing AND s1 duplicated: the duplicate (a fault) wins over the gap
    with pytest.raises(tg.DuplicateReplicate):
        tg.require_triplet([Record(1, 1.0, True), Record(1, 1.0, True), Record(3, 3.0, True)])


def test_spread_takes_exactly_three_values_and_refuses_survivors():
    s = tg.spread([1.0, 2.0, 4.0])
    assert s["n"] == 3 and s["range"] == 3.0 and s["min"] == 1.0 and s["max"] == 4.0
    assert s["sd_ddof"] == 1 and abs(s["sd"] - 1.5275252316519468) < 1e-12
    assert abs(s["mean"] - 7 / 3) < 1e-12
    with pytest.raises(tg.TripletRefused):
        tg.spread([1.0, 2.0])
    with pytest.raises(tg.TripletRefused):
        tg.spread([1.0, 2.0, float("nan")])


def test_unavailable_message_format_lists_every_missing_seed_sorted():
    msg = tg.unavailable_message([(3, "no store"), (1, "not admitted")])
    assert msg == "unavailable: incomplete replicate triplet (missing: s1 — not admitted, s3 — no store)"
