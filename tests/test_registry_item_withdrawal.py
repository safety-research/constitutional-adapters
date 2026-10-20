"""A producer retracts item_result rows it filed under the wrong run: the `item_withdrawal` record (migration 0020).

The validator refuses a malformed payload BY NAME (a missing run_id / items / reason, an empty or non-list items, an item key that is
not an object, carries an unknown field, lacks item_id / metric_id, carries a non-integer index, or repeats an earlier key; a non-string
reason) and fills the item_result defaults for an absent index. The fold, on a fixture shaped like the motivating case (one run, one item
with three generation-time metric rows that belong to another run), marks exactly the named natively emitted rows with the record's sha
and reason, every current-item predicate excludes them while the raw table and the other items' rows stand, `v_paper_inputs_head` moves
(and moves again for a row re-filed above the withdrawal, the paper rebuild trigger's regression of 2026-10-01),
and `registry rebuild --verify` equality holds over a shuffled copy of the log. A withdrawal whose run or whose named row is not in the
table yet is dependency-blocked (never refused) and folds when it lands. A row re-filed under the same key with a greater order key than
the withdrawal's stands as a new measurement (the marks read NULL again); one with a smaller order key than the live row is ignored; a row
written BEFORE the withdrawal but delivered after it is marked all the same (the marks are a pure function of the records). The same
record twice is a duplicate; a later record naming the key with another reason replaces the mark; a revision-tagged row of the key is
never targeted and becomes the key's current row once the native row is withdrawn.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.identity import canonical_json

from test_registry_core_fixes import _rebuild_equal

#: the motivating shape: one item of a per-behavior run with three generation-time metric rows (judgment_id ''), filed under the wrong
#: run; the names are synthetic (the first metric is the synthetic eval's headline, so v_item_headline sees the rows too)
WRONG_ITEM = "synthetic_scenario_a"
OTHER_ITEM = "synthetic_scenario_b"
METRICS = (("asr", 9.0), ("awareness_score", 1.0), ("realism_score", 9.0))
REASON = "filed under the wrong run: the rows are the derived control run's, not this run's"


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def _ingest(root, **kw):
    kw.setdefault("workers", 0)
    kw.setdefault("min_free", 0)
    return H.ingest_once(root, **kw)


def _q(root, sql, *params):
    with DB.reading(DB.db_path(root)) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _v(rec, path="fixture"):
    return VAL.validate_record(rec, path, R.record_sha(rec))


def _item_row(run_id: str, item_id: str, metric_id: str, value: float, written: str, source_revision_id: str = "") -> Dict[str, object]:
    """A generation-time item_result record (the whole row; judgment_id '' and turn_index -1, as the motivating rows carry)."""
    payload = {"run_id": run_id, "item_id": item_id, "metric_id": metric_id, "trial_index": 0, "sample_index": 0, "turn_index": -1,
               "judgment_id": "", "value": value, "label": None, "api_pool": None, "created_utc": written, "judged_utc": None,
               "source_revision_id": source_revision_id, "withdrawn_by_revision_id": None}
    return synth.rec("item_result", payload, written)


def _keys(item_id: str = WRONG_ITEM) -> List[Dict[str, object]]:
    return [{"item_id": item_id, "metric_id": m, "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": ""} for m, _ in METRICS]


def _withdrawal(run_id: str, written: str, items=None, reason: str = REASON, **more) -> Dict[str, object]:
    payload: Dict[str, object] = {"run_id": run_id, "items": items if items is not None else _keys(), "reason": reason}
    payload.update(more)
    return synth.rec("item_withdrawal", payload, written)


def _marks(root, run_id: str, item_id: str):
    return {r[0]: (r[1], r[2]) for r in _q(root, "SELECT metric_id, withdrawn_by_record_sha256, withdrawn_reason FROM item_result "
                                               "WHERE run_id = ? AND item_id = ? ORDER BY metric_id", run_id, item_id)}


def _fixture(tmp_path, through: str = "generated"):
    """Dimensions, the WRONG run A and the run B the rows belong to, and A's six item rows (three of the wrong item, three of another)
    at written_utc t(110): returns (root, rid_a, rid_b)."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid_a, recs_a = synth.run_records("wrong", 100, through=through)
    rid_b, recs_b = synth.run_records("right", 200, through=through)
    _write(root, "a", "000001-run.jsonl", recs_a)
    _write(root, "b", "000001-run.jsonl", recs_b)
    rows = [_item_row(rid_a, item, m, v, synth.t(110)) for item in (WRONG_ITEM, OTHER_ITEM) for m, v in METRICS]
    _write(root, "a", "000002-item_result.jsonl.gz", rows, gz=True)
    return root, rid_a, rid_b


# ---- the validator: malformed payloads are refused by name ------------------------------------------------------------------

def test_item_withdrawal_is_a_record_kind_with_the_documented_contract():
    assert "item_withdrawal" in R.RECORD_TYPES and "item_withdrawal" in R.EFFECTS and "item_withdrawal" not in R.RECORD_TYPE_TABLE
    assert R.REQUIRED_FIELDS["item_withdrawal"] == ("run_id", "items", "reason")
    assert R.EFFECTS["item_withdrawal"].optional == ("evidence_uri", "belongs_to_run_id")
    assert R.ITEM_WITHDRAWAL_KEY_FIELDS == ("item_id", "metric_id", "trial_index", "sample_index", "turn_index", "judgment_id")
    assert R.NATURAL_KEYS["item_withdrawal"] == ("run_id",) + R.ITEM_WITHDRAWAL_KEY_FIELDS
    assert "item_withdrawal" in S.RUN_SCALE_TABLES and "item_withdrawal" in S.VIEW_REFERENCES["v_paper_inputs_head"]
    # the two marks are derived (no producer carries them; the whole-row rule ignores them) and trail the item-scale tail
    cols = S.TABLE_BY_NAME["item_result"].column_names
    assert cols[-3:] == ["ingest_seq", "withdrawn_by_record_sha256", "withdrawn_reason"]
    for c in ("withdrawn_by_record_sha256", "withdrawn_reason"):
        assert S.TABLE_BY_NAME["item_result"].column(c).derived and c not in R.ITEM_WHOLE_ROW_COLUMNS["item_result"]


def test_valid_withdrawal_normalizes_absent_indices_to_the_item_result_defaults():
    rec = _withdrawal("r", synth.t(0), items=[{"item_id": "q1", "metric_id": "asr"}], belongs_to_run_id="r2", evidence_uri="synthetic:///why")
    v = _v(rec)
    assert isinstance(v, VAL.Validated), v
    assert v.table is None and v.extra == {}
    assert v.payload == {"run_id": "r", "reason": REASON, "belongs_to_run_id": "r2", "evidence_uri": "synthetic:///why",
                         "items": [{"item_id": "q1", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": ""}]}
    # an unknown top-level key is parked, never dropped
    rec2 = _withdrawal("r", synth.t(0), items=[{"item_id": "q1", "metric_id": "asr"}], ticket="synthetic-42")
    v2 = _v(rec2)
    assert isinstance(v2, VAL.Validated) and v2.extra == {"ticket": "synthetic-42"} and "ticket" not in v2.payload
    # the record's own parents: the run (the rows are the harvester's composite check)
    assert VAL.parent_references(v) == [("run", "run_id", "r")]


@pytest.mark.parametrize("mutate,field,needle", [
    (lambda p: p.pop("run_id"), "run_id", "required field 'run_id' missing"),
    (lambda p: p.pop("items"), "items", "required field 'items' missing"),
    (lambda p: p.pop("reason"), "reason", "required field 'reason' missing"),
    (lambda p: p.__setitem__("reason", ""), "reason", "required field 'reason' missing"),
    (lambda p: p.__setitem__("reason", 7), "reason", "reason must be a non-empty string"),
    (lambda p: p.__setitem__("items", []), "items", "non-empty list"),
    (lambda p: p.__setitem__("items", {"item_id": "q1", "metric_id": "asr"}), "items", "non-empty list"),
    (lambda p: p.__setitem__("items", ["q1"]), "items[0]", "must be an object"),
    (lambda p: p.__setitem__("items", [{"item_id": "q1", "metric_id": "asr", "label": "x"}]), "items[0].label", "unknown item-key field 'label'"),
    (lambda p: p.__setitem__("items", [{"item_id": "q1", "metric_id": "asr", "run_id": "r"}]), "items[0].run_id", "unknown item-key field 'run_id'"),
    (lambda p: p.__setitem__("items", [{"metric_id": "asr"}]), "items[0].item_id", "missing or not a non-empty string"),
    (lambda p: p.__setitem__("items", [{"item_id": "q1", "metric_id": ""}]), "items[0].metric_id", "missing or not a non-empty string"),
    (lambda p: p.__setitem__("items", [{"item_id": "q1", "metric_id": "asr", "trial_index": "0"}]), "items[0].trial_index", "must be an integer"),
    (lambda p: p.__setitem__("items", [{"item_id": "q1", "metric_id": "asr", "turn_index": True}]), "items[0].turn_index", "must be an integer"),
    (lambda p: p.__setitem__("items", [{"item_id": "q1", "metric_id": "asr", "judgment_id": 3}]), "items[0].judgment_id", "must be a string"),
    (lambda p: p.__setitem__("items", [{"item_id": "q1", "metric_id": "asr"}, {"item_id": "q2", "metric_id": "asr"},
                                       {"item_id": "q1", "metric_id": "asr", "turn_index": -1}]), "items[2]", "repeats items[0]"),
    (lambda p: p.__setitem__("belongs_to_run_id", 5), "belongs_to_run_id", "must be a string"),
    (lambda p: p.__setitem__("evidence_uri", ["a"]), "evidence_uri", "must be a string"),
])
def test_malformed_withdrawal_is_refused_naming_the_field(mutate, field, needle):
    rec = _withdrawal("r", synth.t(0), items=[{"item_id": "q1", "metric_id": "asr"}])
    mutate(rec["payload"])
    out = _v(rec, "records/c/r/000003-item_withdrawal.jsonl#1")
    assert isinstance(out, VAL.Malformed), out
    assert out.field == field, (out.field, out.reason)
    assert needle in out.reason, out.reason
    assert out.path == "records/c/r/000003-item_withdrawal.jsonl#1" and out.record_type == "item_withdrawal"


# ---- the fold on the motivating shape ----------------------------------------------------------------------------------------

def test_withdrawal_marks_the_named_rows_and_every_current_predicate_excludes_them(tmp_path):
    root, rid_a, rid_b = _fixture(tmp_path)
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    head_before = _q(root, "SELECT ingest_seq FROM v_paper_inputs_head")[0][0] or 0      # no in-paper member yet: NULL
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ?", rid_a)[0][0] == 6
    wrec = _withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b, evidence_uri="synthetic:///audit/misfiled.md")
    _write(root, "a", "000003-item_withdrawal.jsonl", [wrec])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    sha = R.record_sha(wrec)
    # the three named native rows carry the record's sha and reason; the other item's rows carry nothing
    assert _marks(root, rid_a, WRONG_ITEM) == {m: (sha, REASON) for m, _ in METRICS}
    assert _marks(root, rid_a, OTHER_ITEM) == {m: (None, None) for m, _ in METRICS}
    # one item_withdrawal row per key, the record's facts on each
    rows = _q(root, "SELECT item_id, metric_id, trial_index, sample_index, turn_index, judgment_id, reason, evidence_uri, belongs_to_run_id "
                    "FROM item_withdrawal WHERE run_id = ? ORDER BY metric_id", rid_a)
    assert rows == [(WRONG_ITEM, m, 0, 0, -1, "", REASON, "synthetic:///audit/misfiled.md", rid_b) for m, _ in sorted(METRICS)]
    fv = json.loads(_q(root, "SELECT field_versions_json FROM item_withdrawal WHERE run_id = ? AND metric_id = 'asr'", rid_a)[0][0])
    assert fv["reason"] == [synth.t(300), sha]
    # the raw table keeps the rows; every current-item predicate excludes them and keeps the other item's
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE run_id = ?", rid_a)[0][0] == 6
    for view in ("v_item_result_current", "v_item_result_provenances"):
        items = _q(root, f"SELECT item_id, metric_id FROM {view} WHERE run_id = ? ORDER BY item_id, metric_id", rid_a)
        assert items == [(OTHER_ITEM, m) for m, _ in sorted(METRICS)], (view, items)
    assert _q(root, "SELECT item_id FROM v_item_headline WHERE run_id = ?", rid_a) == [(OTHER_ITEM,)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_conflict WHERE run_id = ?", rid_a)[0][0] == 0
    # the head moved to the withdrawal file's rank (a member in its own right)
    head_after = _q(root, "SELECT ingest_seq FROM v_paper_inputs_head")[0][0]
    wseq = _q(root, "SELECT ingest_seq FROM record_ingest WHERE file_uri = 'records/a/000003-item_withdrawal.jsonl'")[0][0]
    assert head_after == wseq > head_before
    assert _q(root, "SELECT COUNT(*) FROM item_withdrawal WHERE ingest_seq = ?", wseq)[0][0] == 3
    # the record is indexed under its run, so `registry history` and the status page find it
    assert ("records/a/000003-item_withdrawal.jsonl",) in _q(root, "SELECT record_uri FROM record_run_index WHERE run_id = ?", rid_a)
    _rebuild_equal(root, tmp_path, seed=3)


def test_withdrawal_before_its_run_is_blocked_on_the_run_then_folds(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid_a, recs_a = synth.run_records("wrong", 100, through="generated")
    wrec = _withdrawal(rid_a, synth.t(300))
    _write(root, "a", "000003-item_withdrawal.jsonl", [wrec])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 1, rep.refusals
    assert _q(root, "SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records") == [("item_withdrawal", "run", rid_a)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal")[0][0] == 0
    _write(root, "a", "000001-run.jsonl", recs_a)
    _write(root, "a", "000002-item_result.jsonl.gz", [_item_row(rid_a, WRONG_ITEM, m, v, synth.t(110)) for m, v in METRICS], gz=True)
    rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    assert _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0
    assert _marks(root, rid_a, WRONG_ITEM) == {m: (R.record_sha(wrec), REASON) for m, _ in METRICS}
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ?", rid_a)[0][0] == 0
    _rebuild_equal(root, tmp_path, seed=5)


def test_withdrawal_naming_a_row_not_yet_folded_is_blocked_on_that_row_then_folds(tmp_path):
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)
    missing = {"item_id": "synthetic_scenario_c", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": ""}
    wrec = _withdrawal(rid_a, synth.t(300), items=_keys()[:1] + [missing])
    _write(root, "a", "000003-item_withdrawal.jsonl", [wrec])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 1, rep.refusals
    blocked = _q(root, "SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records")
    assert blocked == [("item_withdrawal", "item_result", canonical_json({"run_id": rid_a, **missing}))]
    assert _marks(root, rid_a, WRONG_ITEM)["asr"] == (None, None)            # nothing folds until every named row exists
    _write(root, "a", "000004-item_result.jsonl.gz", [_item_row(rid_a, "synthetic_scenario_c", "asr", 1.0, synth.t(120))], gz=True)
    rep = _ingest(root)
    assert rep.refused == 0 and _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0
    sha = R.record_sha(wrec)
    assert _marks(root, rid_a, WRONG_ITEM) == {"asr": (sha, REASON), "awareness_score": (None, None), "realism_score": (None, None)}
    assert _marks(root, rid_a, "synthetic_scenario_c") == {"asr": (sha, REASON)}
    _rebuild_equal(root, tmp_path, seed=11)


def test_withdrawal_delivered_in_the_same_file_set_as_its_rows_folds_in_that_cycle(tmp_path):
    """The rows and the withdrawal arrive together: the item batch is not in the table when the withdrawal's parents are checked, so
    it blocks and the cycle's re-evaluation folds it -- one `ingest_once` ends with the marks in place."""
    root, rid_a, rid_b = _fixture(tmp_path)
    wrec = _withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b)
    _write(root, "a", "000003-item_withdrawal.jsonl", [wrec])
    rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    assert _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0
    assert _marks(root, rid_a, WRONG_ITEM) == {m: (R.record_sha(wrec), REASON) for m, _ in METRICS}
    _rebuild_equal(root, tmp_path, seed=13)


# ---- the re-filed row and the pure function ---------------------------------------------------------------------------------

def test_a_row_refiled_with_a_greater_order_key_than_the_withdrawal_stands_and_clears_the_marks(tmp_path):
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)
    wrec = _withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b)
    _write(root, "a", "000003-item_withdrawal.jsonl", [wrec])
    _ingest(root)
    sha = R.record_sha(wrec)
    assert _marks(root, rid_a, WRONG_ITEM)["asr"] == (sha, REASON)
    # re-filed AFTER the withdrawal (written t(400) > t(300)): the whole-row upsert replaces the row, the marks read NULL again
    _write(root, "a", "000004-item_result.jsonl.gz", [_item_row(rid_a, WRONG_ITEM, "asr", 0.5, synth.t(400))], gz=True)
    rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    assert _marks(root, rid_a, WRONG_ITEM) == {"asr": (None, None), "awareness_score": (sha, REASON), "realism_score": (sha, REASON)}
    assert _q(root, "SELECT value FROM v_item_result_current WHERE run_id = ? AND item_id = ? AND metric_id = 'asr'", rid_a, WRONG_ITEM) == [(0.5,)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ?", rid_a)[0][0] == 4
    # the item_withdrawal row stands (the record is history; the row's order key beat it)
    assert _q(root, "SELECT COUNT(*) FROM item_withdrawal WHERE run_id = ?", rid_a)[0][0] == 3
    # a re-filed row with a SMALLER order key than the live one is ignored, as always: value and marks unchanged
    _write(root, "a", "000005-item_result.jsonl.gz", [_item_row(rid_a, WRONG_ITEM, "awareness_score", 0.1, synth.t(105))], gz=True)
    _ingest(root)
    assert _q(root, "SELECT value FROM item_result WHERE run_id = ? AND item_id = ? AND metric_id = 'awareness_score'", rid_a, WRONG_ITEM) == [(1.0,)]
    assert _marks(root, rid_a, WRONG_ITEM)["awareness_score"] == (sha, REASON)
    _rebuild_equal(root, tmp_path, seed=17)


def test_the_head_follows_the_withdrawal_and_then_the_row_refiled_above_it(tmp_path):
    """The paper rebuild trigger (`v_paper_inputs_head`; `paper_rebuild.sh` builds only when it moved) moves on BOTH halves of a
    retraction: the withdrawal's fold (the item_withdrawal member) and a row later re-filed ABOVE it, which clears the marks and is
    current again -- at a rank neither the withdrawal table nor the 0015 item member can see, so the head reads the native row of every
    withdrawal key by primary key. Found in review (2026-10-01): the withdrawal moved the head, the re-filed row did not, and the per-item
    readings the paper draws from `v_item_headline` stayed without the restored row until an unrelated record moved the head."""
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)

    def head():
        return _q(root, "SELECT ingest_seq FROM v_paper_inputs_head")[0][0]

    def rank(name):
        return _q(root, "SELECT ingest_seq FROM record_ingest WHERE file_uri LIKE ?", f"%{name}%")[0][0]

    head0 = head() or 0
    _write(root, "a", "000003-item_withdrawal.jsonl", [_withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b)])
    _ingest(root)
    assert head() == rank("000003-item_withdrawal") > head0
    assert _q(root, "SELECT item_id FROM v_item_headline WHERE run_id = ?", rid_a) == [(OTHER_ITEM,)]
    # the re-filed row (t(400) > t(300)): current again, and the head is its file's rank
    _write(root, "a", "000004-item_result.jsonl.gz", [_item_row(rid_a, WRONG_ITEM, "asr", 0.5, synth.t(400))], gz=True)
    rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    assert _marks(root, rid_a, WRONG_ITEM)["asr"] == (None, None)
    assert head() == rank("000004-item_result") > rank("000003-item_withdrawal")
    assert _q(root, "SELECT item_id FROM v_item_headline WHERE run_id = ? ORDER BY item_id", rid_a) == [(WRONG_ITEM,), (OTHER_ITEM,)]
    # a row re-filed BELOW the withdrawal (t(250) < t(300)) replaces the live row (t(110)) and stays marked; its file's rank moves the
    # head all the same -- the member reads the row's rank, marked or not: a rebuild that reads an unchanged frame, the 0015 superset price
    _write(root, "a", "000005-item_result.jsonl.gz", [_item_row(rid_a, WRONG_ITEM, "awareness_score", 0.7, synth.t(250))], gz=True)
    _ingest(root)
    assert _marks(root, rid_a, WRONG_ITEM)["awareness_score"] != (None, None)
    assert head() == rank("000005-item_result") > rank("000004-item_result")
    # the plan drives from the withdrawal table and seeks the item rows by key: never a walk of item_result
    with DB.reading(DB.db_path(root)) as conn:
        plan = [r[3] for r in conn.execute("EXPLAIN QUERY PLAN SELECT ingest_seq FROM v_paper_inputs_head")]
    iw = next(i for i, p in enumerate(plan) if p.startswith(("SCAN iw", "SEARCH iw")))
    ir = next(i for i, p in enumerate(plan) if p.startswith("SEARCH ir"))
    assert iw < ir and not any("idx_item_result_revision" in p for p in plan), plan
    _rebuild_equal(root, tmp_path, seed=23)


def test_a_row_written_before_the_withdrawal_but_delivered_after_it_is_marked(tmp_path):
    """The pure function: a row's standing is decided by the two order keys, never by arrival order -- a row written at t(250),
    below the withdrawal's t(300), takes the mark when it lands after the withdrawal folded (the post-flush re-derivation), and
    the replay of a shuffled log reaches the same marks."""
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)
    wrec = _withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b)
    _write(root, "a", "000003-item_withdrawal.jsonl", [wrec])
    _ingest(root)
    sha = R.record_sha(wrec)
    _write(root, "a", "000004-item_result.jsonl.gz", [_item_row(rid_a, WRONG_ITEM, "asr", 0.7, synth.t(250))], gz=True)
    rep = _ingest(root)
    assert rep.refused == 0, rep.refusals
    assert _q(root, "SELECT value FROM item_result WHERE run_id = ? AND item_id = ? AND metric_id = 'asr'", rid_a, WRONG_ITEM) == [(0.7,)]
    assert _marks(root, rid_a, WRONG_ITEM) == {m: (sha, REASON) for m, _ in METRICS}
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ? AND item_id = ?", rid_a, WRONG_ITEM)[0][0] == 0
    _rebuild_equal(root, tmp_path, seed=19)
    _rebuild_equal(root, tmp_path, seed=23)


def test_duplicate_withdrawal_is_a_no_op_and_a_later_record_replaces_the_mark(tmp_path):
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)
    w1 = _withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b)
    _write(root, "a", "000003-item_withdrawal.jsonl", [w1])
    _ingest(root)
    _write(root, "a", "000004-item_withdrawal-repush.jsonl", [w1])            # the same record under another file name
    rep = _ingest(root)
    assert rep.duplicates == 1 and rep.refused == 0 and rep.blocked == 0
    assert _marks(root, rid_a, WRONG_ITEM) == {m: (R.record_sha(w1), REASON) for m, _ in METRICS}
    assert _q(root, "SELECT COUNT(*) FROM row_history WHERE table_name = 'item_withdrawal'")[0][0] == 3
    # a later record naming two of the keys with another reason: those two take its sha and reason, the third keeps the first's
    w2 = _withdrawal(rid_a, synth.t(350), items=_keys()[:2], reason="second look: the two rows are the control run's", belongs_to_run_id=rid_b)
    _write(root, "a", "000005-item_withdrawal.jsonl", [w2])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _marks(root, rid_a, WRONG_ITEM) == {"asr": (R.record_sha(w2), "second look: the two rows are the control run's"),
                                               "awareness_score": (R.record_sha(w2), "second look: the two rows are the control run's"),
                                               "realism_score": (R.record_sha(w1), REASON)}
    assert _q(root, "SELECT COUNT(*) FROM item_withdrawal WHERE run_id = ?", rid_a)[0][0] == 3
    assert _q(root, "SELECT COUNT(*) FROM row_history WHERE table_name = 'item_withdrawal'")[0][0] == 5
    _rebuild_equal(root, tmp_path, seed=29)


def test_an_earlier_record_arriving_late_does_not_move_a_mark_the_later_record_set(tmp_path):
    """Order independence on the item_withdrawal side: the key's fold is the LATEST record by order key, so a record written at
    t(280) that lands after the t(300) one leaves the t(300) mark in place (and the replay agrees)."""
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)
    late = _withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b)
    early = _withdrawal(rid_a, synth.t(280), reason="an earlier, superseded note", belongs_to_run_id=rid_b)
    _write(root, "a", "000003-item_withdrawal.jsonl", [late])
    _ingest(root)
    _write(root, "a", "000004-item_withdrawal.jsonl", [early])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _marks(root, rid_a, WRONG_ITEM) == {m: (R.record_sha(late), REASON) for m, _ in METRICS}
    assert _q(root, "SELECT reason FROM item_withdrawal WHERE run_id = ? AND metric_id = 'asr'", rid_a) == [(REASON,)]
    _rebuild_equal(root, tmp_path, seed=31)


def test_a_revision_tagged_row_of_the_key_is_never_targeted_and_becomes_current(tmp_path):
    """A producer retracts ITS OWN rows (source_revision_id = ''); an adapter revision's row of the same key is the revision's to
    carry. Once the native row is withdrawn, the revision's row is the key's current provenance."""
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)
    rev = "synthetic_source:rev1"
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": "synthetic_source", "adapter": "synthetic_adapter",
                                           "observed_utc": synth.t(150), "inputs_json": {"synthetic:///in": "a" * 64}, "accepted": 1,
                                           "priority": 0}, synth.t(150))
    tagged = _item_row(rid_a, WRONG_ITEM, "asr", 0.25, synth.t(151), source_revision_id=rev)
    _write(root, "s", "000001-source_revision.jsonl", [marker])
    _write(root, "s", "000002-item_result.jsonl.gz", [tagged], gz=True)
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    # two provenances of the key: the native row ranks first
    assert _q(root, "SELECT source_revision_id, value FROM v_item_result_current WHERE run_id = ? AND item_id = ? AND metric_id = 'asr'",
              rid_a, WRONG_ITEM) == [("", 9.0)]
    wrec = _withdrawal(rid_a, synth.t(300), items=_keys()[:1], belongs_to_run_id=rid_b)
    _write(root, "a", "000003-item_withdrawal.jsonl", [wrec])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    rows = _q(root, "SELECT source_revision_id, withdrawn_by_record_sha256 FROM item_result WHERE run_id = ? AND item_id = ? AND metric_id = 'asr' "
                    "ORDER BY source_revision_id", rid_a, WRONG_ITEM)
    assert rows == [("", R.record_sha(wrec)), (rev, None)]
    assert _q(root, "SELECT source_revision_id, value FROM v_item_result_current WHERE run_id = ? AND item_id = ? AND metric_id = 'asr'",
              rid_a, WRONG_ITEM) == [(rev, 0.25)]
    assert _q(root, "SELECT n_provenances FROM v_item_conflict WHERE run_id = ? AND item_id = ? AND metric_id = 'asr'", rid_a, WRONG_ITEM) == []
    _rebuild_equal(root, tmp_path, seed=37)


def test_snapshot_item_export_reads_the_current_rows_without_the_withdrawn_ones(tmp_path):
    """The snapshot's item export reads `v_item_result_current`: the withdrawn rows leave it too, and the two mark
    columns ride along as NULL text."""
    pytest.importorskip("pyarrow")
    from sisteer.registry import snapshot as SN
    root, rid_a, rid_b = _fixture(tmp_path)
    _ingest(root)
    _write(root, "a", "000003-item_withdrawal.jsonl", [_withdrawal(rid_a, synth.t(300), belongs_to_run_id=rid_b)])
    _ingest(root)
    out = str(tmp_path / "item_result.parquet")
    with DB.reading(DB.db_path(root)) as conn:
        n = SN._export_parquet(conn, "item_result", out)
    assert n == 3
    import pyarrow.parquet as pq
    t = pq.read_table(out)
    assert set(t.column("item_id").to_pylist()) == {OTHER_ITEM}
    assert t.column("withdrawn_by_record_sha256").to_pylist() == [None] * 3
