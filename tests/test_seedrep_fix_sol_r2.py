"""The Qwen seed-replicate battery — both findings of the code review, each RE-VALIDATED
in prose below and pinned with known-bad inputs.

P1 — `petri_target_turn_draws` failed OPEN: it `continue`d on a missing / non-list `target_turns`, a non-mapping entry
or invalid fields (a missing list and `"target_turns": "bad"` vanished; `turn: True` became sample index 1), so a
consumer got zero or fewer draws for a scenario still filed in `results.json` and `bind_rows` could say AGREES after
evidence was silently discarded (re-validated at dd43a18f: the three probe rows returned `[]`, `[]` and one draw at
sample_idx 1). Now FAIL-CLOSED: refuses missing / non-list lists, every malformed entry on its ORIGINAL type, a
completion id that disagrees with the status, duplicate request ids, divergent sibling rows, a scenario with no
successful attempt, an empty input; returns `PetriTurnDraws(draws, n_successful, n_failed, n_scenarios)` with
`len(draws) == n_successful` by construction.

P2 — `set_label()` and the public adapters coerced with `int(sample_idx)` BEFORE the strict validator (re-validated at
dd43a18f: `set_label(..., True)` → header `g/1`, `"2"` → 2, `2.5` → 2), aliasing one draw to another valid sample. Now
`check_sample_idx` validates the ORIGINAL value (non-bool int >= 0) and `make_label` is the ONE Label constructor; no
adapter coerces."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from seedrep_fixes_common import S, VMODEL, hook_off, lane_on, ws3g_modules  # noqa: F401
from sisteer.capeval.tinycap import contracts as C
from sisteer.eval import request_evidence_client as rev
from sisteer.eval.attacks import served_target as st

TASK = "petri/self_preservation"
BAD_INDEXES = [True, False, "2", "0", 2.0, 2.5, -1, None, [2], 3j]


def _entry(turn, rid, cid, status, task=TASK):
    return {"turn": turn, "request_id": rid, "completion_id": cid, "status": status, "task": task}


def _rows(turns, *, item="scn_a", task=TASK, metrics=("self_preservation", "eval_awareness", "scenario_realism")):
    """One scenario's dimension rows as `convert_eval_log` writes them: the SAME list on every row."""
    return [{"task": task, "item_id": item, "metric": m, "extra_json": json.dumps({"dimension": m, "scenario_id": item,
                                                                                   rev.PETRI_TARGET_TURNS_ROW_KEY: turns})}
            for m in metrics]


GOOD = [_entry(1, "r1", "cmpl-1", 200), _entry(2, "r2", None, 500), _entry(2, "r3", "cmpl-2", 200), _entry(3, "r4", "cmpl-3", 200)]


# --------------------------------------------------------------------------- #
# P1 — fail closed                                                             #
# --------------------------------------------------------------------------- #

def test_draws_over_a_well_formed_scenario_count_every_successful_attempt_and_nothing_else():
    r = rev.petri_target_turn_draws(_rows(GOOD))
    assert isinstance(r, rev.PetriTurnDraws)
    assert [(d["sample_idx"], d["request_id"], d["completion_id"]) for d in r.draws] == [(1, "r1", "cmpl-1"), (2, "r3", "cmpl-2"), (3, "r4", "cmpl-3")]
    assert (r.n_successful, r.n_failed, r.n_scenarios) == (3, 1, 1) and len(r.draws) == r.n_successful
    # two scenarios, one aggregate row (the mean row: no attempts by construction) — counted per scenario, the aggregate skipped
    two = _rows(GOOD) + _rows([_entry(1, "r9", "cmpl-9", 200)], item="scn_b") + [
        {"task": TASK, "item_id": C.AGGREGATE_ITEM_ID if hasattr(C, "AGGREGATE_ITEM_ID") else "__aggregate__", "metric": "self_preservation__mean", "extra_json": None}]
    r2 = rev.petri_target_turn_draws(two)
    assert (r2.n_successful, r2.n_failed, r2.n_scenarios) == (4, 1, 2) and [d["item_id"] for d in r2.draws] == ["scn_a"] * 3 + ["scn_b"]


@pytest.mark.parametrize("label, rows, match", [
    ("missing list", [{**r, "extra_json": json.dumps({"dimension": "d"})} for r in _rows(GOOD)], "carries no target_turns"),
    ("list is a str", _rows("bad"), "str, not a list"),
    ("list is a mapping", _rows({"turn": 1}), "dict, not a list"),
    ("entry not a mapping", _rows(["r1"]), "not a mapping"),
    ("turn True", _rows([_entry(True, "r1", "cmpl-1", 200)]), r"turn True \(bool\)"),
    ("turn '2'", _rows([_entry("2", "r1", "cmpl-1", 200)]), r"turn '2' \(str\)"),
    ("turn 0", _rows([_entry(0, "r1", "cmpl-1", 200)]), "turn 0"),
    ("turn 2.0", _rows([_entry(2.0, "r1", "cmpl-1", 200)]), r"turn 2.0 \(float\)"),
    ("request_id empty", _rows([_entry(1, "", "cmpl-1", 200)]), "request_id ''"),
    ("request_id int", _rows([_entry(1, 7, "cmpl-1", 200)]), "request_id 7"),
    ("status True", _rows([_entry(1, "r1", "cmpl-1", True)]), r"status True \(bool\)"),
    ("status '200'", _rows([_entry(1, "r1", "cmpl-1", "200")]), r"status '200' \(str\)"),
    ("2xx without completion", _rows([_entry(1, "r1", None, 200)]), "200 attempt without a completion_id"),
    ("failed with completion", _rows([_entry(1, "r1", "cmpl-1", 500)]), "failed attempt .* carrying completion_id"),
    ("status None with completion", _rows([_entry(1, "r1", "cmpl-1", None)]), "failed attempt .* carrying completion_id"),
    ("foreign task", _rows([_entry(1, "r1", "cmpl-1", 200, task="petri/delusion_sycophancy")]), "another behaviour's binding"),
    ("unknown key", _rows([{**_entry(1, "r1", "cmpl-1", 200), "note": "x"}]), r"missing \[\], unknown \['note'\]"),
    ("duplicate request id", _rows([_entry(1, "r1", "cmpl-1", 200), _entry(2, "r1", "cmpl-2", 200)]), "recorded twice"),
    ("no successful attempt", _rows([_entry(1, "r1", None, 500), _entry(1, "r2", None, None)]), "none successful"),
    ("empty list", _rows([]), "none successful"),
    ("row without extra", [{"task": TASK, "item_id": "scn_a", "metric": "m", "extra_json": None}], "no readable extra_json"),
    ("extra_json not JSON", [{"task": TASK, "item_id": "scn_a", "metric": "m", "extra_json": "{not json"}], "not JSON"),
    ("row without task", [{"task": "", "item_id": "scn_a", "metric": "m", "extra_json": "{}"}], "not non-empty strings"),
])
def test_every_malformed_or_missing_list_refuses_by_name_never_fewer_draws(label, rows, match):
    with pytest.raises(rev.RequestEvidenceError, match=match):
        rev.petri_target_turn_draws(rows)


def test_a_divergent_sibling_row_a_partially_invalid_list_and_an_empty_input_refuse():
    rows = _rows(GOOD)
    # one dimension row of the scenario carries a DIFFERENT list (one attempt dropped): torn store, refused — not "fewer draws"
    rows[1]["extra_json"] = json.dumps({"dimension": "eval_awareness", rev.PETRI_TARGET_TURNS_ROW_KEY: GOOD[:-1]})
    with pytest.raises(rev.RequestEvidenceError, match="differs from row 0's for the same scenario"):
        rev.petri_target_turn_draws(rows)
    # a reordered copy is divergent too (the list is ORDERED — wire order)
    rows[1]["extra_json"] = json.dumps({"dimension": "eval_awareness", rev.PETRI_TARGET_TURNS_ROW_KEY: list(reversed(GOOD))})
    with pytest.raises(rev.RequestEvidenceError, match="differs from row 0's"):
        rev.petri_target_turn_draws(rows)
    # a list whose FIRST entries are fine and whose last is malformed: refused whole, no partial draws
    with pytest.raises(rev.RequestEvidenceError, match=r"\[3\]: turn True"):
        rev.petri_target_turn_draws(_rows(GOOD[:3] + [_entry(True, "r5", "cmpl-5", 200)]))
    with pytest.raises(rev.RequestEvidenceError, match="no rows"):
        rev.petri_target_turn_draws([])
    with pytest.raises(rev.RequestEvidenceError, match="only aggregate rows"):
        rev.petri_target_turn_draws([{"task": TASK, "item_id": "__aggregate__", "metric": "m", "extra_json": None}])


def _without(e, *keys):
    return {k: v for k, v in e.items() if k not in keys}


@pytest.mark.parametrize("label, entries, match", [
    ("a valid success + an entry missing status AND completion_id (the third review's repro: a stripped success must not read as a failure)",
     [_entry(1, "r1", "cmpl-1", 200), _without(_entry(2, "r2", "cmpl-2", 200), "status", "completion_id")],
     r"missing \['completion_id', 'status'\]"),
    ("a failed entry missing completion_id", [_entry(1, "r1", "cmpl-1", 200), _without(_entry(2, "r2", None, 500), "completion_id")],
     r"missing \['completion_id'\]"),
    ("a status-less entry missing status", [_entry(1, "r1", "cmpl-1", 200), _without(_entry(2, "r2", None, None), "status")],
     r"missing \['status'\]"),
    ("an entry missing task", [_without(_entry(1, "r1", "cmpl-1", 200), "task")], r"missing \['task'\]"),
    ("an entry missing turn", [_without(_entry(1, "r1", "cmpl-1", 200), "turn")], r"missing \['turn'\]"),
    ("an entry missing request_id", [_without(_entry(1, "r1", "cmpl-1", 200), "request_id")], r"missing \['request_id'\]"),
    ("missing AND unknown at once", [{**_without(_entry(1, "r1", "cmpl-1", 200), "status"), "code": 200}],
     r"missing \['status'\], unknown \['code'\]"),
    ("an empty mapping", [{}], r"missing \['completion_id', 'request_id', 'status', 'task', 'turn'\], unknown \[\]"),
])
def test_a_missing_key_is_never_a_recorded_none_through_both_readers(label, entries, match):
    """The entry validator held unknown keys out but read the required ones with `.get()`, so an entry MISSING
    `status` and `completion_id` was indistinguishable from an explicitly recorded None — a stripped SUCCESS became a
    failure, its draw vanished, and every count still balanced (the scenario kept another success, so the zero-success
    refusal could not catch it). Exact key-set equality now: refused by name through BOTH readers."""
    with pytest.raises(rev.RequestEvidenceError, match=match):
        rev.petri_target_turn_draws(_rows(entries))
    with pytest.raises(rev.RequestEvidenceError, match=match):
        rev.petri_target_turns_of_sample({"store": {rev.PETRI_TARGET_TURNS_STORE_KEY: entries}}, task=TASK, scenario_id="scn_a")
    # the same entries WITH the keys present (None recorded, or the success intact) are exactly what is accepted
    five = {"turn", "request_id", "completion_id", "status", "task"}
    # the counterpart WITH every key present and coherent (a completion means 2xx): exactly what is accepted
    whole = [e if set(e) == five else
             {**_entry(e.get("turn", 9), e.get("request_id", "r9"), e.get("completion_id"), 200 if e.get("completion_id") else None,
                       task=TASK), **{k: v for k, v in e.items() if k in five and k != "status"}}
             for e in entries if e]
    if whole and all(e.get("task") == TASK for e in whole) and len({e["request_id"] for e in whole}) == len(whole):
        r = rev.petri_target_turn_draws(_rows(whole))
        assert len(r.draws) == r.n_successful == sum(1 for e in whole if e["status"] is not None and 200 <= e["status"] < 300)


def test_the_converters_reader_shares_the_validator_completion_iff_2xx_and_unique_ids():
    good = {"store": {rev.PETRI_TARGET_TURNS_STORE_KEY: GOOD}}
    assert rev.petri_target_turns_of_sample(good, task=TASK, scenario_id="scn_a") == GOOD
    assert rev.petri_target_turns_of_sample({"store": {}}, task=TASK, scenario_id="scn_a") is None
    for bad, match in [([_entry(1, "r1", None, 200)], "without a completion_id"),
                       ([_entry(1, "r1", "c", 500)], "carrying completion_id"),
                       ([_entry(1, "r1", "c", 200), _entry(2, "r1", "d", 200)], "recorded twice"),
                       ([_entry(True, "r1", "c", 200)], r"turn True \(bool\)"),
                       ("bad", "str, not a list")]:
        with pytest.raises(rev.RequestEvidenceError, match=match):
            rev.petri_target_turns_of_sample({"store": {rev.PETRI_TARGET_TURNS_STORE_KEY: bad}}, task=TASK, scenario_id="scn_a")


# --------------------------------------------------------------------------- #
# P2 — the ORIGINAL index value, validated once, before any Label exists        #
# --------------------------------------------------------------------------- #

def test_check_sample_idx_and_make_label_refuse_every_alias_and_accept_plain_ints():
    for bad in BAD_INDEXES:
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            rev.check_sample_idx(bad)
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            rev.make_label("static/x", "g", bad)
    assert rev.check_sample_idx(0) == 0 and rev.check_sample_idx(7) == 7
    assert rev.make_label("static/x", "g", 2) == rev.Label("static/x", "g", 2)
    assert rev.label_headers("static/x", "g", 2)[C.HEADER_SAMPLE] == "g/2"


def test_set_label_and_labelled_refuse_the_aliases_and_leave_no_label_behind():
    for bad in BAD_INDEXES:
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            rev.set_label("static/x", "g", bad)
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            with rev.labelled("static/x", "g", bad):
                pytest.fail("the body ran under an aliased label")
        assert rev.current_label() is None
    with rev.labelled("static/x", "g", 2) as lab:
        assert lab == rev.Label("static/x", "g", 2)


def test_the_multiturn_label_fn_and_petri_turn_headers_refuse_the_aliases(tmp_path, monkeypatch):
    lane_on(monkeypatch, tmp_path / "attempt", extra={"s": S})
    fn = rev.multiturn_label_fn("pair", {"gid": "hb_0042"})
    assert fn is not None and fn(3) == ("multi_turn/pair", "hb_0042", 3)
    for bad in BAD_INDEXES:
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            fn(bad)
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            rev.petri_turn_headers(SimpleNamespace(task=TASK, item_id="scn_a", sample_idx=bad))
    assert rev.petri_turn_headers(SimpleNamespace(task=TASK, item_id="scn_a", sample_idx=4))[C.HEADER_SAMPLE] == "scn_a/4"


def test_ws1a_bon_and_ws1c_misalign_seams_refuse_the_aliases(tmp_path, monkeypatch):
    client = lane_on(monkeypatch, tmp_path / "attempt", extra={"s": S})
    bon = rev.maybe_request_headers_fn(rev.LEG_BON)
    mis = rev.MisalignRequestEvidence(client)
    assert bon is not None
    good = bon("hb_0042", 5, 0)
    assert good[C.HEADER_SAMPLE] == "hb_0042/5" and good[C.HEADER_TASK] == "strongreject_bon" and C.HEADER_REQUEST_ID in good
    assert mis.headers_for(family="agentic", behavior="sabotage", item_id="it_1", sample_idx=1) == {
        C.HEADER_TASK: "agentic/sabotage", C.HEADER_SAMPLE: "it_1/1"}
    n_before = client.n_requests
    for bad in BAD_INDEXES:
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            bon("hb_0042", bad, 0)
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            mis.headers_for(family="agentic", behavior="sabotage", item_id="it_1", sample_idx=bad)
    assert client.n_requests == n_before   # nothing minted, nothing recorded for a refused label


def test_served_chat_target_set_request_context_refuses_the_aliases_and_keeps_no_context(tmp_path, monkeypatch):
    lane_on(monkeypatch, tmp_path / "attempt", extra={"s": S})
    target = st.ServedChatTarget(lambda payload, **kw: {}, virtual_model=VMODEL, cell_id="none", serving_engine="lora_native")
    for bad in BAD_INDEXES:
        if bad is None:
            continue   # None is set_request_context's own "needs the index" refusal (ServedTargetError), pinned below
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            target.set_request_context(kind="multi_turn", attack="pair", goal_id="hb_0042", turn=bad, trial=0)
        with pytest.raises(rev.RequestEvidenceError, match="sample index"):
            target.set_request_context(kind="static", family="grok420", goal_id="g9", sample_idx=bad)
        assert target._take_request_context() is None
    with pytest.raises(st.ServedTargetError):
        target.set_request_context(kind="static", family="grok420", goal_id="g9", sample_idx=None)
    target.set_request_context(kind="multi_turn", attack="pair", goal_id="hb_0042", turn=3, trial=0)
    assert target._take_request_context() == rev.Label("multi_turn/pair", "hb_0042", 3)


# --------------------------------------------------------------------------- #
# The REAL adopted reader (audit_common.petri_target_turn_draws, tip >= 8ecdd9a03) agrees with this contract         #
# --------------------------------------------------------------------------- #

def test_ws3g_adopted_reader_yields_the_same_draws_and_refuses_the_same_core_known_bads():
    """Cross-check against the REAL adopted reader (`audit_common.petri_target_turn_draws`, tip >= 8ecdd9a03).
    Agreement on the good case and the shared known-bads is asserted outright. The ONE recorded divergence — an
    all-failed scenario — refuses on both sides since the correction (495c86b6) sits in the same tree; the former
    flip rule (`[]` | `AuditRefusal`, pinned by an env key) was retired at the battery-lane cut (cross-ownership 7 closed)."""
    from pathlib import Path as _P

    ac, _rea = ws3g_modules()
    theirs = getattr(ac, "petri_target_turn_draws", None)
    if theirs is None:
        pytest.skip("WS3-G's audit_common has not adopted petri_target_turn_draws on this tree")
    mine = rev.petri_target_turn_draws(_rows(GOOD))
    got = theirs(_rows(GOOD), where=_P("/nonexistent/cell"))
    assert [(d.task, d.item_id, d.sample_idx, d.request_id, d.completion_id) for d in got] == \
        [(d["task"], d["item_id"], d["sample_idx"], d["request_id"], d["completion_id"]) for d in mine.draws]
    assert len(got) == mine.n_successful == 3
    # the core known-bads refuse on BOTH sides (theirs raises their AuditRefusal): no list, a non-list, a boolean turn,
    # a 2xx without its completion, a duplicate request id, a foreign task
    for rows in ([{**r, "extra_json": json.dumps({"dimension": "d"})} for r in _rows(GOOD)], _rows("bad"),
                 _rows([_entry(True, "r1", "cmpl-1", 200)]), _rows([_entry(1, "r1", None, 200)]),
                 _rows([_entry(1, "r1", "cmpl-1", 200), _entry(2, "r1", "cmpl-2", 200)]),
                 _rows([_entry(1, "r1", "cmpl-1", 200, task="petri/delusion_sycophancy")])):
        with pytest.raises(rev.RequestEvidenceError):
            rev.petri_target_turn_draws(rows)
        with pytest.raises(ac.AuditRefusal):
            theirs(rows, where=_P("/nonexistent/cell"))
    # the ONE divergence, on record (cross-ownership 7 of impl_ws1dp.md): a filed scenario whose every attempt FAILED —
    # this helper refuses (a scenario the audit would otherwise pass with zero draws); their reader at 8ecdd9a03 yields
    # zero draws for it (fail-open on exactly that case); their correction (495c86b6) refuses it, and this tree pins that.
    all_failed = _rows([_entry(1, "r1", None, 500)])
    with pytest.raises(rev.RequestEvidenceError, match="none successful"):
        rev.petri_target_turn_draws(all_failed)
    # audit_common >= 495c86b6 (now in this tree): the all-failed scenario REFUSES on its side too —
    # the former `zero_draws` compatibility branch and its SISTEER_WS3G_ALL_FAILED_EXPECT pin are gone
    with pytest.raises(ac.AuditRefusal):
        theirs(all_failed, where=_P("/nonexistent/cell"))
    divergent = _rows(GOOD)
    divergent[2]["extra_json"] = json.dumps({"dimension": "scenario_realism", rev.PETRI_TARGET_TURNS_ROW_KEY: GOOD[:-1]})
    with pytest.raises(rev.RequestEvidenceError, match="differs from row 0"):
        rev.petri_target_turn_draws(divergent)
    with pytest.raises(ac.AuditRefusal):
        theirs(divergent, where=_P("/nonexistent/cell"))
