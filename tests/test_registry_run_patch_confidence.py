"""A ``run_patch``'s ``field_confidence_json`` is read PER COLUMN by the fold (``reducer._refresh_provenance``): each entry the patch
carries replaces the standing entry of its column, an entry of null withdraws the standing one, the entries of the columns the patch
names nothing about stand, and a column the patch states outright without an entry keeps its standing entry with the ``unstated`` flag
shed. A FULL ``run`` record's JSON is the row's whole, as before.

Why: a hook that cannot read the row -- the standup's endpoint record stamping the served positions mode -- must state an ``unverified``
value's reason in the patch itself (the validator reads it there) without erasing the row's other entries, which the earlier wholesale
read did. Whole-log audit at the rule's landing (read-only): no ``run_patch`` record in the log carried the column, so nothing already
folded moves under the rule; the synthetic events below are the only exercise of it.
"""
from __future__ import annotations

from sisteer.registry import reducer as RD
from sisteer.registry import synth

T = synth.t


def _ev(i, payload, kind="full", extra=None):
    body = dict(payload)
    body["_kind"] = kind
    if extra:
        body["_extra"] = dict(extra)
    name = "run_patch" if kind == "patch" else "run"
    return RD.HistoryEvent(T(i), f"{i:064x}", name, f"records/x/{i:06d}-{name}.jsonl", body)


def _full(**over):
    return _ev(1, synth.run_payload("k", "none", "none", 0, field_confidence_json={"a": {"confidence": "unverified", "reason": "no a"},
                                                                                    "b": {"confidence": "derived", "source": "rule b"}},
                                    **over))


def _state(*events):
    out = RD.fold("run", list(events))
    return out[-1][0]


def _conf(state):
    return RD._conf_of(state.get("field_confidence_json"))


def test_a_patch_confidence_json_merges_per_column_and_a_full_record_restates_the_whole():
    patch = _ev(2, {"notes": "served", "field_confidence_json": {"b": {"confidence": "recorded", "source": "endpoint"},
                                                                  "c": {"confidence": "unverified", "reason": "no c"}}}, kind="patch")
    st = _state(_full(), patch)
    assert _conf(st) == {"a": {"confidence": "unverified", "reason": "no a"},                      # names nothing about `a`: it stands
                         "b": {"confidence": "recorded", "source": "endpoint"},                     # its entry for `b` replaces the standing one
                         "c": {"confidence": "unverified", "reason": "no c"}}                       # a new column's entry lands
    assert st["notes"] == "served"
    # the column's stamp is the patch's (it carried the column), and the replay is a pure function of the events
    assert RD.fold("run", [_full(), patch])[-1][1]["field_confidence_json"] == [T(2), f"{2:064x}"]
    assert _state(_full(), patch) == st
    # a later FULL record's JSON is the row's whole: `a` and `c` are gone, as every full statement restates the row
    full2 = _ev(3, synth.run_payload("k", "none", "none", 0, field_confidence_json={"d": {"confidence": "derived", "source": "rule d"}}))
    assert _conf(_state(_full(), patch, full2)) == {"d": {"confidence": "derived", "source": "rule d"}}
    # the string form the log carries folds the same as the dict form
    import json
    patch_s = _ev(2, {"notes": "served", "field_confidence_json": json.dumps({"b": {"confidence": "recorded", "source": "endpoint"},
                                                                               "c": {"confidence": "unverified", "reason": "no c"}})}, kind="patch")
    assert _conf(_state(_full(), patch_s)) == _conf(st)


def test_a_null_entry_withdraws_the_standing_one_and_a_patch_without_the_column_moves_no_entry():
    assert _conf(_state(_full(), _ev(2, {"field_confidence_json": {"a": None}}, kind="patch"))) == {"b": {"confidence": "derived", "source": "rule b"}}
    # a patch that states a column without carrying the JSON keeps every standing entry here: none is flagged, so there is nothing to shed
    # (a flagged column it states sheds the flag -- the test below)
    assert _conf(_state(_full(), _ev(2, {"notes": "n"}, kind="patch"))) == _conf(_state(_full()))
    # an empty JSON on a patch names nothing: the standing entries stand (wholesale, it used to erase them)
    assert _conf(_state(_full(), _ev(2, {"field_confidence_json": {}}, kind="patch"))) == _conf(_state(_full()))


def test_a_patch_stating_a_flagged_column_sheds_the_flag_whether_or_not_it_carries_an_entry_for_it():
    owner = _ev(1, synth.run_payload("k", "none", "none", 0, field_confidence_json={}, n_items_generated=None))
    reader = _ev(2, {"run_id": owner.payload["run_id"], "n_items_generated": 4,
                     "field_confidence_json": {"n_items_generated": {"confidence": "recorded", "source": "export"}}},
                 kind="patch", extra={"unstated_columns": ["n_items_generated"]})
    flagged = _state(owner, reader)
    assert _conf(flagged)["n_items_generated"] == {"confidence": "recorded", "source": "export", "unstated": True}
    # an unmarked patch stating the column with its own entry: the entry replaces, no flag
    own = _ev(3, {"n_items_generated": 9, "field_confidence_json": {"n_items_generated": {"confidence": "recorded", "source": "store"}}}, kind="patch")
    assert _conf(_state(owner, reader, own))["n_items_generated"] == {"confidence": "recorded", "source": "store"}
    # an unmarked patch stating the column with no entry for it (JSON carried for another column): the standing entry stands, flag shed
    other = _ev(3, {"n_items_generated": 9, "field_confidence_json": {"notes": {"confidence": "recorded", "source": "hand"}}}, kind="patch")
    assert _conf(_state(owner, reader, other)) == {"n_items_generated": {"confidence": "recorded", "source": "export"},
                                                  "notes": {"confidence": "recorded", "source": "hand"}}


def test_a_patch_with_no_confidence_json_keeps_the_untouched_entries_and_sheds_only_the_flag_of_the_columns_it_states():
    """The records page's statement of a no-JSON patch, pinned on a flagged row: it adds, replaces and withdraws no entry, the entries of
    the columns it leaves alone stand flag and all, a column it states keeps its entry with the flag shed -- and that column is then an
    unmarked statement a later marked reader no longer refreshes, while the still-flagged column is."""
    owner = _ev(1, synth.run_payload("k", "none", "none", 0, field_confidence_json={}, n_items_generated=None, n_items_judged=None))
    reader = _ev(2, {"run_id": owner.payload["run_id"], "n_items_generated": 4, "n_items_judged": 4,
                     "field_confidence_json": {"n_items_generated": {"confidence": "recorded", "source": "export"},
                                               "n_items_judged": {"confidence": "recorded", "source": "judge"}}},
                 kind="patch", extra={"unstated_columns": ["n_items_generated", "n_items_judged"]})
    flagged = {"n_items_generated": {"confidence": "recorded", "source": "export", "unstated": True},
               "n_items_judged": {"confidence": "recorded", "source": "judge", "unstated": True}}
    assert _conf(_state(owner, reader)) == flagged
    # states neither flagged column: every entry stands, flag included
    assert _conf(_state(owner, reader, _ev(3, {"notes": "n"}, kind="patch"))) == flagged
    # states ONE flagged column, with no JSON at all: that column's entry is retained with the flag shed, the other's stands, flag included
    bare = _ev(3, {"n_items_generated": 4}, kind="patch")
    st = _state(owner, reader, bare)
    assert st["n_items_generated"] == 4
    assert _conf(st) == {"n_items_generated": {"confidence": "recorded", "source": "export"},
                         "n_items_judged": {"confidence": "recorded", "source": "judge", "unstated": True}}
    # the consequence: the restated count is an unmarked statement now, so a later marked reader is skipped over it and refreshes only
    # the column that is still flagged
    reader2 = _ev(4, {"run_id": owner.payload["run_id"], "n_items_generated": 7, "n_items_judged": 7,
                      "field_confidence_json": {"n_items_generated": {"confidence": "recorded", "source": "export-2"},
                                                "n_items_judged": {"confidence": "recorded", "source": "judge-2"}}},
                  kind="patch", extra={"unstated_columns": ["n_items_generated", "n_items_judged"]})
    st2 = _state(owner, reader, bare, reader2)
    assert st2["n_items_generated"] == 4 and st2["n_items_judged"] == 7
    assert _conf(st2) == {"n_items_generated": {"confidence": "recorded", "source": "export"},
                          "n_items_judged": {"confidence": "recorded", "source": "judge-2", "unstated": True}}
    # without the bare patch the same reader refreshes both: the shed flag alone is what stranded the count
    st3 = _state(owner, reader, reader2)
    assert st3["n_items_generated"] == 7 and st3["n_items_judged"] == 7
    assert _conf(st3) == {"n_items_generated": {"confidence": "recorded", "source": "export-2", "unstated": True},
                          "n_items_judged": {"confidence": "recorded", "source": "judge-2", "unstated": True}}
