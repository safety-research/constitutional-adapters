"""The three core follow-ups the adapter workstreams deferred to the joint landing, each reproduced on a
known-bad input before it was fixed.

(1) A record whose JSON is invalid is a REFUSAL naming the record, never an exception that ends the cycle. The case that
    bit: a non-finite float (``Infinity`` / ``-Infinity`` / ``NaN``) — Python's ``json.loads`` accepts the literals, RFC 8259
    and SQLite's ``json_valid`` do not — passed the validator and crashed the revision staging transaction with an
    IntegrityError (``json_valid(record_json)``); in a plain file it folded a REAL column silently or refused with a bare
    IntegrityError. Now the validator refuses it naming the field's path in every path (validator, plain file, revision
    constituent). A line that is not JSON at all refuses the FILE naming the line (pinned beside it: that path already held).
(2) A held (pending) revision whose source has since ACCEPTED a later revision is superseded: its blocked records are marked
    stale naming the successor, so the blocked queue and the status page stop reporting them as pending work. The revision
    still activates — harmlessly, already superseded — if its dependency ever lands, so live equals replay.
(3) ``v_run_degeneration_current`` ranks a row carrying a number above every placeholder of the same measurement key:
    ``measurement_state = 'measured'`` comes right after the native-first rule and BEFORE the declared source priority
    (the placeholder that hid a number in the field came from a higher-priority store reader), then the standing order
    (priority, source_id, revision_id) decides among rows of equal informativeness.
"""
from __future__ import annotations

import gzip
import json
import os

import pytest

from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry import validate as VAL

from test_registry_core_fixes import (JUDGE_ID_3, JudgeNamingAdapter, StoreAdapter, _adapter_setup, _ingest, _judge3, _live_items,
                                      _pass, _publish_b, _q, _rebuild_equal, _rev_dir, _root, _store, _write)

NONFINITE = [pytest.param(float("inf"), id="inf"), pytest.param(float("-inf"), id="-inf"), pytest.param(float("nan"), id="nan")]
RID = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)


def _dumps_lenient(rec) -> str:
    """What a producer that never checked its numbers writes: canonical shape, non-finite floats spelled as the JSON-invalid
    literals Python emits by default (the bytes the r2 export carried on 2,256 rows)."""
    return json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_lines(path: str, lines) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = "".join(line + "\n" for line in lines).encode("utf-8")
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as fh:
            fh.write(data)
    else:
        with open(path, "wb") as fh:
            fh.write(data)
    return path


def _file_of(rev_dir: str, record_type: str) -> str:
    for f in sorted(os.listdir(rev_dir)):
        m = R.RECORD_FILE_RE.match(f)
        if m and m.group("type") == record_type:
            return os.path.join(rev_dir, f)
    raise AssertionError((record_type, sorted(os.listdir(rev_dir))))


def _rewrite_one(path: str, record_type: str, mutate) -> int:
    """Rewrite a published file with ``mutate(rec)`` applied to its first record of ``record_type``; returns that line."""
    recs = [rec for _, rec, _ in R.read_record_file(path)]
    line = next(i for i, rec in enumerate(recs, start=1) if rec["record_type"] == record_type)
    mutate(recs[line - 1])
    os.remove(path)
    _write_lines(path, [_dumps_lenient(r) for r in recs])
    return line


def _metric(value, metric_id="ratio", extra=None, i=9001):
    p = {"run_id": "r", "judgment_id": "j", "metric_id": metric_id, "threshold": 0.5, "value": value, "k": 1, "n": 4, "computed_by": "producer"}
    if extra is not None:
        p["extra"] = extra
    return synth.rec("run_metric", p, synth.t(i))


def _accepted(root, rev):
    return _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev)[0][0]


def _head(root, source_id):
    return _q(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 AND superseded_by_revision_id IS NULL",
              source_id)


def _blocked_of(root, rev):
    return _q(root, "SELECT stale, json_extract(extra_json, '$.superseded_by') FROM blocked_records "
                    "WHERE json_extract(extra_json, '$.revision_id') = ? ORDER BY record_uri", rev)


# ---- (1) a non-finite float is a named refusal, in every path ------------------------------------------------------


@pytest.mark.parametrize("value", NONFINITE)
def test_validator_refuses_a_nonfinite_float_naming_its_path(value):
    assert isinstance(VAL.validate_record(_metric(0.25), "x#0", "0" * 64), VAL.Validated)
    top = VAL.validate_record(_metric(value), "x#1", "1" * 64)
    assert isinstance(top, VAL.Malformed), top
    assert (top.field, top.record_type, top.sha, top.path) == ("value", "run_metric", "1" * 64, "x#1")
    assert "finite" in top.reason and "JSON" in top.reason
    nested = VAL.validate_record(_metric(0.25, extra={"stats": {"ratios": [0.1, value]}}), "x#2", "2" * 64)
    assert isinstance(nested, VAL.Malformed) and nested.field == "extra.stats.ratios[1]", nested
    envelope = _metric(0.25)
    envelope["producer"]["load"] = value
    env = VAL.validate_record(envelope, "x#3", "3" * 64)
    assert isinstance(env, VAL.Malformed) and env.field == "producer.load", env


@pytest.mark.parametrize("value", NONFINITE)
def test_nonfinite_float_in_a_plain_file_is_refused_naming_the_field_and_never_folded(tmp_path, value):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=2)
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    _ingest(root)

    def metric(metric_id, v, extra=None, i=9001):
        p = {"run_id": rid, "judgment_id": jid, "metric_id": metric_id, "threshold": 0.5, "value": v, "k": 1, "n": 4, "computed_by": "producer"}
        if extra is not None:
            p["extra"] = extra
        return synth.rec("run_metric", p, synth.t(i))
    recs = [metric("fine", 0.5, i=9000), metric("ratio", value, i=9001), metric("parked", 0.5, extra={"steer_live_ratio": value}, i=9002)]
    path = _write_lines(os.path.join(root, "records", "x", "000001-run_metric.jsonl"), [_dumps_lenient(r) for r in recs])
    uri = H.logical_uri(root, path)
    rep = _ingest(root)
    assert rep.refused == 2, rep.refusals
    rows = _q(root, "SELECT record_uri, field, reason FROM record_refusal ORDER BY record_uri")
    assert [(r[0], r[1]) for r in rows] == [(f"{uri}#2", "value"), (f"{uri}#3", "extra.steer_live_ratio")], rows
    assert all("finite" in r[2] for r in rows), rows
    # the valid sibling folded; the non-finite value reached no REAL column and no *_json column
    assert _q(root, "SELECT metric_id FROM run_metric WHERE run_id = ? ORDER BY metric_id", rid) == [("asr",), ("fine",)]
    assert _q(root, "SELECT status, n_ok, n_refused FROM record_delivery WHERE file_uri = ?", uri) == [("partial", 1, 2)]
    page = status.status_json(root)
    assert {row["field"] for row in page["refused"]} == {"value", "extra.steer_live_ratio"}
    _rebuild_equal(root, tmp_path)


@pytest.mark.parametrize("value", NONFINITE)
def test_nonfinite_float_in_a_revision_constituent_holds_the_revision_instead_of_ending_the_cycle(tmp_path, value):
    """`stage_records` raised IntegrityError (json_valid(record_json)) on a run-scale constituent carrying `inf`, and the
    exception left `ingest_once` — no refusal, no held revision, the cycle dead. The constituent is refused by name and the
    revision held like any other malformed constituent."""
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    old_metric = _q(root, "SELECT value FROM run_metric WHERE run_id = ?", RID)[0][0]
    old_items = _live_items(root, RID)
    rev2 = _publish_b(root, store_root, ad, [0.9, 0.9, 0.9], synth.t(3000))
    metric_path = _file_of(_rev_dir(root, ad, rev2), "run_metric")
    line = _rewrite_one(metric_path, "run_metric", lambda rec: rec["payload"].__setitem__("value", value))
    metric_uri = H.logical_uri(root, metric_path)
    assert any(tok in open(metric_path, encoding="utf-8").read() for tok in ("Infinity", "NaN"))     # the bytes are not JSON
    ing = _ingest(root)                                     # before the fix: sqlite3.IntegrityError out of the staging transaction
    assert ing.revisions_held >= 1 and ing.refused >= 1, ing.refusals
    assert _accepted(root, rev2) == 0
    refusal = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev2)[0][0]
    assert refusal.startswith("activation pending:") and f"{metric_uri}#{line}" in refusal and "finite" in refusal, refusal
    rows = _q(root, "SELECT reason FROM record_refusal WHERE record_uri = ?", f"{metric_uri}#{line}")
    assert len(rows) == 1 and rows[0][0].startswith("value:") and "finite" in rows[0][0], rows
    # the previous revision stays entirely current; nothing of the held one is visible
    assert _live_items(root, RID) == old_items
    assert _q(root, "SELECT value FROM run_metric WHERE run_id = ?", RID)[0][0] == pytest.approx(old_metric)
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ? AND value = 0.9", RID) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_record WHERE revision_id = ? AND file_uri = ?", rev2, metric_uri) == [(0,)]
    ing2 = _ingest(root)                                    # a later cycle refuses again, quietly; still no crash
    assert ing2.revisions_held >= 1 and _accepted(root, rev2) == 0 and _live_items(root, RID) == old_items
    _rebuild_equal(root, tmp_path)


def test_a_line_that_is_not_json_refuses_the_plain_file_naming_the_line(tmp_path):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    lines = [I.canonical_json(r) for r in dims]
    lines.insert(2, '{"record_type": "model", "schema_version": 1, "payload": {"model_id": "m", "family": broken')
    path = _write_lines(os.path.join(root, "records", "dims", "000001-dims.jsonl"), lines)
    uri = H.logical_uri(root, path)
    rep = _ingest(root)
    assert rep.refused == 1 and rep.ok == 0, (rep.refused, rep.ok, rep.refusals)
    assert any(f"{uri}#3: not JSON" in r for r in rep.refusals), rep.refusals      # the LOGICAL uri: replay-stable
    delivery = _q(root, "SELECT status, refusal_reason FROM record_delivery WHERE file_uri = ?", uri)
    assert len(delivery) == 1 and delivery[0][0] == "refused" and delivery[0][1].startswith(f"{uri}#3: not JSON"), delivery
    n_models = sum(1 for r in dims if r["record_type"] == "model")
    assert _q(root, "SELECT COUNT(*) FROM model") == [(0,)]                         # the file is one unit: nothing of it folded
    rep2 = _ingest(root)
    assert rep2.refused == 1 and _q(root, "SELECT COUNT(*) FROM model") == [(0,)]   # and it keeps refusing, never crashing
    _write_lines(path, [ln for ln in lines if "broken" not in ln])                  # the producer rewrites the file: it folds
    rep3 = _ingest(root)
    assert rep3.refused == 0 and rep3.ok - rep3.arrival_records == len(dims) and _q(root, "SELECT COUNT(*) FROM model") == [(n_models,)]


def test_a_line_that_is_not_json_in_a_revision_file_holds_the_revision_naming_the_file(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    old_items = _live_items(root, RID)
    rev2 = _publish_b(root, store_root, ad, [0.9, 0.9, 0.9], synth.t(3000))
    item_path = _file_of(_rev_dir(root, ad, rev2), "item_result")
    lines = [I.canonical_json(rec) for _, rec, _ in R.read_record_file(item_path)]
    lines.insert(1, "{this is not json")
    os.remove(item_path)
    _write_lines(item_path, lines)
    item_uri = H.logical_uri(root, item_path)
    ing = _ingest(root)
    assert ing.revisions_held >= 1
    assert _accepted(root, rev2) == 0
    refusal = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev2)[0][0]
    assert refusal.startswith(f"activation pending: record_file {item_uri}") and f"{item_uri}#2: not JSON" in refusal, refusal
    rows = _q(root, "SELECT reason FROM record_refusal WHERE record_uri = ?", item_uri)
    assert len(rows) == 1 and rows[0][0].startswith(f"{item_uri}#2: not JSON"), rows
    assert _live_items(root, RID) == old_items
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", rev2) == [(0,)]
    _rebuild_equal(root, tmp_path)


# ---- (2) a superseded pending revision's blocked records are stale, not pending work -----------------------------


def test_blocked_records_of_a_pending_revision_the_source_moved_past_are_marked_stale_naming_the_successor(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    rep1, _ = _pass(root, ad, synth.t(2000))
    rev1 = rep1.results[0].revision_id
    source_id = _q(root, "SELECT source_id FROM source_revision WHERE revision_id = ?", rev1)[0][0]
    # revision 2 names an absent judge: held, its constituent and marker blocked — pending work, rightly
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9, 0.9])
    rep2, ing2 = _pass(root, JudgeNamingAdapter(store_root, JUDGE_ID_3), synth.t(3000))
    rev2 = rep2.results[0].revision_id
    assert ing2.revisions_held >= 1 and _accepted(root, rev2) == 0
    assert _blocked_of(root, rev2) and all(r == (0, None) for r in _blocked_of(root, rev2)), _blocked_of(root, rev2)
    # revision 3 of the SAME source activates: the source moved past revision 2, which can never become its head
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.5, 0.5, 0.5])
    rep3, ing3 = _pass(root, ad, synth.t(4000))
    rev3 = rep3.results[0].revision_id
    assert ing3.refused == 0 and _accepted(root, rev3) == 1 and _head(root, source_id) == [(rev3,)]
    assert _live_items(root, RID) == {"q0": 0.5, "q1": 0.5, "q2": 0.5}
    assert _accepted(root, rev2) == 0
    rows = _blocked_of(root, rev2)
    assert rows and all(r == (1, rev3) for r in rows), rows           # the defect: these stayed stale = 0 (pending work)
    page = status.status_json(root)["blocked"]
    assert page and all(row["stale"] == 1 for row in page if rev2 in row["record_uri"]), page
    assert not any(row["stale"] == 0 for row in page), page             # nothing else is waiting
    # a re-check must not clear the mark, and a NEWER pending revision of the source is still pending work
    _ingest(root)
    assert all(r == (1, rev3) for r in _blocked_of(root, rev2)), _blocked_of(root, rev2)
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.7, 0.7, 0.7])
    rep4, _ = _pass(root, JudgeNamingAdapter(store_root, JUDGE_ID_3), synth.t(5000))
    rev4 = rep4.results[0].revision_id
    assert _accepted(root, rev4) == 0
    assert _blocked_of(root, rev4) and all(r == (0, None) for r in _blocked_of(root, rev4)), _blocked_of(root, rev4)
    assert all(r == (1, rev3) for r in _blocked_of(root, rev2))
    _rebuild_equal(root, tmp_path)
    # the dependency lands: both pending revisions activate; revision 2 is superseded on arrival, revision 4 is the head
    _write(root, "dims", "000002-judge3.jsonl", [_judge3()])
    ing5 = _ingest(root)
    assert ing5.refused == 0, ing5.refusals
    assert _accepted(root, rev2) == 1 and _accepted(root, rev4) == 1
    assert _q(root, "SELECT superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev2) == [(rev3,)]
    assert _head(root, source_id) == [(rev4,)] and _live_items(root, RID) == {"q0": 0.7, "q1": 0.7, "q2": 0.7}
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    _rebuild_equal(root, tmp_path, seed=11)


# ---- (3) a measured degeneration row outranks a placeholder of the same key -------------------------------------


def _rev(rid, i, src, priority=0):
    return synth.rec("source_revision", {"revision_id": rid, "source_id": src, "adapter": "synthetic", "observed_utc": synth.t(i),
                                         "inputs_json": {"a": ("%064x" % i)}, "accepted": 1, "priority": priority}, synth.t(i))


def _degen(run_id, state, i, rev, measure="union5", **vals):
    p = {"run_id": run_id, "judgment_id": "", "measure": measure, "scope": "all", "measurement_state": state, "numerator": None,
         "denominator": None, "denominator_unit": None, "pooling_json": {}, "aggregation": "single_detector", "rate": None,
         "definition_version": "glossary@" + "a" * 8, "threshold_params_json": {}, "computed_by": "registry", "computed_utc": None,
         "source_uri": None, "notes": None, "source_revision_id": rev}
    p.update(vals)
    return synth.rec("run_degeneration", p, synth.t(i))


def test_current_degeneration_row_is_the_measured_one_whatever_the_placeholders_source_or_priority(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=2)
    run = ids["run_ids"][0]
    # src:a — lexically first AND higher priority (a store reader) — carries only the placeholder; src:z the number
    _write(root, "a", "000001-revs.jsonl", [_rev("rev_a", 9000, "src:a", priority=10), _rev("rev_z", 9000, "src:z", priority=0)])
    _write(root, "d", "000002-degen.jsonl", [_degen(run, "not_measured", 9001, "rev_a"),
                                             _degen(run, "measured", 9002, "rev_z", numerator=3, denominator=200, denominator_unit="items", rate=0.015)])
    rep = _ingest(root)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refusals, rep.blocked)
    assert _q(root, "SELECT COUNT(*) FROM run_degeneration WHERE run_id = ? AND measure = 'union5'", run) == [(2,)]   # both provenances kept
    cur = _q(root, "SELECT measurement_state, rate, source_revision_id FROM v_run_degeneration_current WHERE run_id = ? AND measure = 'union5'", run)
    assert cur == [("measured", 0.015, "rev_z")], cur                    # the defect: the placeholder won on priority / source id
    assert _q(root, "SELECT union5__all_state, union5__all_rate FROM v_run_degeneration_wide WHERE run_id = ?", run) == [("measured", 0.015)]
    assert status.status_json(root)["degeneration_not_measured"] == []
    # among rows of equal informativeness the standing order decides: two placeholders → priority, then source id
    _write(root, "d", "000003-degen.jsonl", [_degen(run, "not_measured", 9003, "rev_a", measure="zlib_collapse"),
                                             _degen(run, "not_applicable", 9004, "rev_z", measure="zlib_collapse")])
    _write(root, "d", "000004-degen.jsonl", [_degen(run, "not_measured", 9005, "rev_a", measure="contentless", notes="a"),
                                             _degen(run, "not_measured", 9006, "rev_z", measure="contentless", notes="z")])
    rep2 = _ingest(root)
    assert rep2.refused == 0, rep2.refusals
    assert _q(root, "SELECT source_revision_id FROM v_run_degeneration_current WHERE run_id = ? AND measure = 'zlib_collapse'", run) == [("rev_a",)]
    assert _q(root, "SELECT notes FROM v_run_degeneration_current WHERE run_id = ? AND measure = 'contentless'", run) == [("a",)]
    # and two numbers still fall to the standing order — priority first
    _write(root, "d", "000005-degen.jsonl", [_degen(run, "measured", 9007, "rev_a", measure="classifier", numerator=1, denominator=10, denominator_unit="items", rate=0.1),
                                             _degen(run, "measured", 9008, "rev_z", measure="classifier", numerator=2, denominator=10, denominator_unit="items", rate=0.2)])
    _ingest(root)
    assert _q(root, "SELECT rate FROM v_run_degeneration_current WHERE run_id = ? AND measure = 'classifier'", run) == [(0.1,)]
    assert _q(root, "SELECT run_id, measure, COUNT(*) FROM v_run_degeneration_current GROUP BY run_id, judgment_id, measure, scope HAVING COUNT(*) > 1") == []
    _rebuild_equal(root, tmp_path)
