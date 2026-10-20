"""Re-activation by id of a revision that carries the synthetic anchor-decision input (registry-hardening 3a, defect 2).

A read that decides anchor comparisons publishes its decision set under TWO fingerprints BY DESIGN (``runner._read_and_publish``):
the marker's ``inputs_json`` folds the PRE-read fingerprint -- what ``plan_groups`` attached, the revision id is minted from it -- while the
decision input's WATERMARK carries the POST-read one, what the read actually decided (an ``anchor_decision_drift`` fact when they differ and
no file moved). ``runner.inputs_from_committed`` demanded, for every URI of the marker, a committed watermark fact of the SAME sha, so a
revision whose read decided differently from the pre-read replay could never be re-activated from committed bytes: ``petri_stats``
``e84fc1dffc1be901`` (13 inputs incl. ``registry://anchor-decision/petri_stats:petri``; inputs_json sha 37c25dd383c6…, committed watermark
sha 530647c50a6c…) was refused by every window's pull and the adapter not read whole since.

The fixture drives the real path: a store adapter whose read of one group logs an anchor decision (``anchor_current.decision_entry``), and
``runner.anchor_decision_inputs`` patched to attach a pre-read fingerprint that differs from what the read decides -- the marker, the
watermark and the drift fact come out of the runner itself, nothing is written by hand. Then the revision is superseded and re-activated by id:
(a) with the drift, accepted -- the bundle marker's ``inputs_json`` equals the committed one byte for byte (the id stands), the decision
watermark carries the COMMITTED watermark's fingerprint (the post-read one; the next pass's replay on an unmoved frame compares equal), the
file watermark its sha; (b) with NO watermark fact for the decision input at all, accepted, the decision watermark from the ``inputs_json``
value; (c) a FILE input whose watermark fact is missing or carries another sha still refuses by name (unchanged); (d) ``inputs_from_committed``
builds an ``AnchorDecisionInput`` for the decision URI.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import anchor_current as AC
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _setup, _store

ADAPTER = SyntheticStoreAdapter.name
SID = f"{ADAPTER}:s2"
DECISION_URI = AC.decision_uri(SID)
#: the fingerprint the pre-read replay gives (what `plan_groups` attaches as the group's decision input): differs from what the read decides
PRE_FP = "1" * 64
PRODUCER = {"script": "tests.reactivate_decision_input", "code_sha": synth.code_sha("a"), "session": None, "host": "TESTHOST", "substrate": "grading_host"}


class DecidingStoreAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter whose read of s2 logs ONE anchor comparison with its decision (`anchor_current.decision_entry`): the runner
    fingerprints the log into the group's synthetic decision input, records it on the marker and writes the decision WATERMARK from
    it -- the post-read fingerprint."""

    def read(self, ctx, group, inputs):
        if group == "s2":
            comparison = {"run_id": "run-steered-s2", "judgment_id": "j-s2", "metric_id": "asr", "threshold": 0.5, "computation_id": "",
                          "judge_id": synth.JUDGE_ID, "needs_items": False}
            decision = {"anchor_run_id": "run-anchor-1", "state": "own_current", "own_reason": "current", "anchor_kind": "unsteered"}
            ctx.decision_log.append(AC.decision_entry(comparison, decision))
        yield from super().read(ctx, group, inputs)


def _rows(root, sql, params=()):
    with DB.reading(DB.db_path(root)) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def _pass(root, ad, now):
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=now, producer=PRODUCER)
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked, [r.error for r in rep.results])
    return rep


def _head(root, source_id=SID):
    return _rows(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", (source_id,))


def _marker_file(root, rev) -> str:
    rev_dir = os.path.join(H.records_dir(root), "adapters", ADAPTER, rev)
    return next(os.path.join(rev_dir, f) for f in sorted(os.listdir(rev_dir)) if "source_revision" in f)


def _marker_records(path) -> List[Dict]:
    return [rec for _, rec, _ in R.read_record_file(path)]


def _committed_marker(root, rev) -> Dict:
    return next(rec["payload"] for rec in _marker_records(_marker_file(root, rev)) if rec["record_type"] == "source_revision")


def _watermarks_of(records) -> Dict[str, str]:
    return {rec["payload"]["input_uri"]: rec["payload"]["sha256"] for rec in records if rec["record_type"] == "source_watermark"}


def _rewrite_marker(path, keep):
    """Rewrite a committed marker file with the records `keep` admits (a record, or a changed copy of it; None drops it) -- the committed
    bytes a re-activation stands on, edited the way the C6 tests edit them."""
    out = [k for k in (keep(rec) for rec in _marker_records(path)) if k is not None]
    os.remove(path)
    R.write_records(path, out)


def _decided_and_superseded(tmp_path, monkeypatch):
    """A grading host where s2's head once carried a decision input with ANCHOR DRIFT and is now superseded: `(root, stores, adapter, rev_a, F)` --
    rev_a's marker folds PRE_FP for the decision URI, its committed watermark carries F (what the read decided), and rev_b (s2 changed) is
    the head."""
    root, _unused = _setup(tmp_path)
    stores = str(tmp_path / "stores")
    s2 = _store(stores, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = DecidingStoreAdapter(stores)
    rep0 = _pass(root, ad, synth.t(2000))                 # the first read: no decision input in the inputs yet; the decision watermark is written from the read
    assert rep0.groups_refreshed == 2, [r.error for r in rep0.results]
    rev0 = _head(root)[0][0]
    (watermark_f,) = _rows(root, "SELECT sha256 FROM source_watermark WHERE source_id = ? AND input_uri = ?", (SID, DECISION_URI))[0]
    assert len(watermark_f) == 64 and watermark_f != PRE_FP and DECISION_URI not in _committed_marker(root, rev0)["inputs_json"]
    # the pre-read replay of the head's recorded set gives PRE_FP (a frame that decides differently from what the read will decide)
    monkeypatch.setattr(RUN, "anchor_decision_inputs",
                        lambda adapter, groups, conn, ctx: {"s2": AC.AnchorDecisionInput.of(SID, PRE_FP)} if "s2" in list(groups) else {})
    rep1 = _pass(root, ad, synth.t(2100))
    (res_a,) = [r for r in rep1.results if r.group == "s2"]
    assert res_a.accepted and rep1.anchor_decisions_changed == 1 and rep1.groups_refreshed == 1, [r.error for r in rep1.results]
    rev_a = res_a.revision_id
    committed = _committed_marker(root, rev_a)
    assert committed["inputs_json"] == {s2: R.file_sha256(s2), DECISION_URI: PRE_FP}
    assert committed["extra"]["anchor_decisions"]["pre_read_fingerprint"] == PRE_FP and committed["extra"]["anchor_decisions"]["fingerprint"] == watermark_f
    assert _watermarks_of(_marker_records(_marker_file(root, rev_a))) == {s2: R.file_sha256(s2), DECISION_URI: watermark_f}
    assert _rows(root, "SELECT json_extract(extra_json, '$.anchor_decision_drift.post_read_fingerprint') FROM source_status WHERE source_revision_id = ?",
                 (rev_a,)) == [(watermark_f,)]
    assert rev_a == RUN.revision_id_for(SID, [AB.check_input(s2, role="report"), AC.AnchorDecisionInput.of(SID, PRE_FP)], RUN.emission_version_of(ad))
    # s2 changes: rev_a is superseded by rev_b, the shape a reverted store (or a cluster shard's refused-known claim) re-activates from
    _store(stores, "s2", "legacy_store_2", "none", [0.1, 0.9, 0.5])
    rep2 = _pass(root, ad, synth.t(2200))
    rev_b = next(r.revision_id for r in rep2.results if r.group == "s2")
    assert rev_b != rev_a and _head(root) == [(rev_b,)]
    return root, stores, ad, rev_a, watermark_f


def _reactivate(root, ad, rev, now):
    with DB.reading(DB.db_path(root)) as conn:
        return RUN.reactivate_by_id(root, ad, rev, conn, now=now)


def test_a_revision_with_anchor_drift_reactivates_by_id_with_the_committed_inputs_json_and_the_post_read_decision_watermark(tmp_path, monkeypatch):
    root, stores, ad, rev_a, watermark_f = _decided_and_superseded(tmp_path, monkeypatch)
    s2 = os.path.join(stores, "s2", "store.json")
    committed = _committed_marker(root, rev_a)
    res = _reactivate(root, ad, rev_a, synth.t(2300))
    assert res.accepted and res.reactivated and res.revision_id == rev_a and res.group == "s2" and res.records_written > 0, res.error
    bundles = RUN.complete_bundles(root, ADAPTER, rev_a)
    assert len(bundles) == 1
    records = _marker_records(bundles[0][1])
    marker = next(rec["payload"] for rec in records if rec["record_type"] == "source_revision")
    # the committed inputs_json byte for byte (the id stands); the committed decision set restated
    assert marker["inputs_json"] == committed["inputs_json"] and marker["revision_id"] == rev_a
    assert marker["extra"]["anchor_decisions"] == committed["extra"]["anchor_decisions"]
    assert marker["extra"]["reactivated_from_committed_bytes"]["inputs"] == 2
    # the file watermark carries its committed sha; the decision watermark the COMMITTED watermark's fingerprint -- the post-read one, never PRE_FP
    assert _watermarks_of(records) == {s2: committed["inputs_json"][s2], DECISION_URI: watermark_f}
    decision_marks = [rec["payload"] for rec in records if rec["record_type"] == "source_watermark" and rec["payload"]["input_uri"] == DECISION_URI]
    assert decision_marks[0]["size"] == 0 and decision_marks[0]["mtime_utc"] == AC.DECISION_MTIME and decision_marks[0]["last_error"] is None
    # the harvester folds the bundle: rev_a is the head again and the database's decision watermark is the post-read fingerprint
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0, (ing.refusals, ing.blocked)
    assert _head(root) == [(rev_a,)]
    assert _rows(root, "SELECT sha256 FROM source_watermark WHERE source_id = ? AND input_uri = ?", (SID, DECISION_URI)) == [(watermark_f,)]
    # what the watermark is FOR: the next pass's replay on an unmoved frame (F) reads the decision input as unchanged; a moved one (PRE_FP) as changed
    with DB.reading(DB.db_path(root)) as conn:
        marks = RUN._watermarks(conn, ADAPTER)
        quiet = RUN.plan_groups(ad, ad.discover(), marks, RUN.emission_version_of(ad), conn, decision_inputs={"s2": AC.AnchorDecisionInput.of(SID, watermark_f)})
        moved = RUN.plan_groups(ad, ad.discover(), marks, RUN.emission_version_of(ad), conn, decision_inputs={"s2": AC.AnchorDecisionInput.of(SID, PRE_FP)})
    assert quiet.decision_changed == set() and moved.decision_changed == {"s2"}
    # a second call: the id is current now -> nothing written, no second bundle
    res2 = _reactivate(root, ad, rev_a, synth.t(2400))
    assert res2.accepted and not res2.reactivated and res2.files == [] and len(RUN.complete_bundles(root, ADAPTER, rev_a)) == 1


def test_a_decision_input_with_no_committed_watermark_fact_reactivates_with_the_inputs_json_fingerprint(tmp_path, monkeypatch):
    root, stores, ad, rev_a, watermark_f = _decided_and_superseded(tmp_path, monkeypatch)
    s2 = os.path.join(stores, "s2", "store.json")
    committed = _committed_marker(root, rev_a)
    _rewrite_marker(_marker_file(root, rev_a), lambda rec: None if rec["record_type"] == "source_watermark" and rec["payload"]["input_uri"] == DECISION_URI else rec)
    assert DECISION_URI not in RUN.committed_marker_facts(os.path.join(H.records_dir(root), "adapters", ADAPTER, rev_a))["watermarks"]
    res = _reactivate(root, ad, rev_a, synth.t(2300))
    assert res.accepted and res.reactivated, res.error
    records = _marker_records(RUN.complete_bundles(root, ADAPTER, rev_a)[0][1])
    marker = next(rec["payload"] for rec in records if rec["record_type"] == "source_revision")
    assert marker["inputs_json"] == committed["inputs_json"]
    assert _watermarks_of(records) == {s2: committed["inputs_json"][s2], DECISION_URI: PRE_FP}       # no committed fact: the marker's own value
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and _head(root) == [(rev_a,)]


def test_a_file_input_with_a_missing_or_mismatched_watermark_fact_still_refuses_by_name(tmp_path, monkeypatch):
    root, stores, ad, rev_a, watermark_f = _decided_and_superseded(tmp_path, monkeypatch)
    s2 = os.path.join(stores, "s2", "store.json")
    path = _marker_file(root, rev_a)
    original = _marker_records(path)
    # (i) the file's watermark fact missing
    _rewrite_marker(path, lambda rec: None if rec["record_type"] == "source_watermark" and rec["payload"]["input_uri"] == s2 else rec)
    res = _reactivate(root, ad, rev_a, synth.t(2300))
    assert not res.accepted and res.error.startswith("re-activation by id refused:"), res
    assert f"the marker names input {s2} (sha " in res.error and "no matching source_watermark fact" in res.error and DECISION_URI not in res.error
    assert RUN.complete_bundles(root, ADAPTER, rev_a) == []
    # (ii) the file's watermark fact carries another sha
    os.remove(path)
    R.write_records(path, original)
    _rewrite_marker(path, lambda rec: (dict(rec, payload=dict(rec["payload"], sha256="0" * 64))
                                       if rec["record_type"] == "source_watermark" and rec["payload"]["input_uri"] == s2 else rec))
    res2 = _reactivate(root, ad, rev_a, synth.t(2350))
    assert not res2.accepted and f"the marker names input {s2} (sha " in res2.error and "no matching source_watermark fact" in res2.error
    assert RUN.complete_bundles(root, ADAPTER, rev_a) == [] and _head(root) != [(rev_a,)]


def test_inputs_from_committed_builds_an_anchor_decision_input_from_the_marker_alone():
    file_uri = "/synthetic/stores/s2/store.json"
    fp = "ab" * 32
    facts = {"marker": "<marker>", "revision": {"inputs_json": {file_uri: "cd" * 32, DECISION_URI: fp}},
             "watermarks": {file_uri: {"sha256": "cd" * 32, "size": 42, "mtime_utc": "2026-10-01T00:00:00.000Z"},
                            DECISION_URI: {"sha256": "ef" * 32, "size": 0, "mtime_utc": AC.DECISION_MTIME}},     # the post-read fingerprint: not demanded to match
             "states": []}
    out = RUN.inputs_from_committed(facts)
    assert [i.identity_uri for i in out] == [file_uri, DECISION_URI] and RUN.inputs_json(out) == facts["revision"]["inputs_json"]
    file_in, decision_in = out
    assert not AC.is_decision_input(file_in) and file_in.size == 42 and file_in.mtime_utc == "2026-10-01T00:00:00.000Z" and file_in.sha256 == "cd" * 32
    assert isinstance(decision_in, AC.AnchorDecisionInput) and AC.is_decision_input(decision_in) and decision_in.sha256 == fp
    assert decision_in.size == 0 and decision_in.mtime_utc == AC.DECISION_MTIME and decision_in.role == AC.DECISION_ROLE and decision_in.path == ""
    assert AB.watermark_of(decision_in) == (fp, 0, AC.DECISION_MTIME)
    # the watermark inputs of a re-activation: the file as built, the decision input at the COMMITTED fact's fingerprint
    stamped = RUN._committed_watermark_inputs(facts, out)
    assert [i.sha256 for i in stamped] == ["cd" * 32, "ef" * 32] and isinstance(stamped[1], AC.AnchorDecisionInput)
    assert RUN._committed_watermark_inputs(dict(facts, watermarks={file_uri: facts["watermarks"][file_uri]}), out) == out      # no fact: the marker's value
    # a decision input with no watermark fact is admitted; a file input with none, or with another sha, refuses by name
    assert RUN.inputs_json(RUN.inputs_from_committed(dict(facts, watermarks={file_uri: facts["watermarks"][file_uri]}))) == facts["revision"]["inputs_json"]
    with pytest.raises(AB.AdapterRefusal, match="names input /synthetic/stores/s2/store.json .* no matching source_watermark fact"):
        RUN.inputs_from_committed(dict(facts, watermarks={DECISION_URI: facts["watermarks"][DECISION_URI]}))
    with pytest.raises(AB.AdapterRefusal, match="no matching source_watermark fact"):
        RUN.inputs_from_committed(dict(facts, watermarks={file_uri: dict(facts["watermarks"][file_uri], sha256="00" * 32)}))
    # an EMPTY inputs_json is a retirement's: not this function's to re-activate (reactivate_by_id's own path re-activates it from the database)
    with pytest.raises(AB.AdapterRefusal, match="inputs_json is empty: an empty revision is a RETIREMENT"):
        RUN.inputs_from_committed(dict(facts, revision={"inputs_json": {}}))
    # a malformed decision fingerprint on the marker refuses by name too
    with pytest.raises(AB.AdapterRefusal, match="malformed fingerprint"):
        RUN.inputs_from_committed(dict(facts, revision={"inputs_json": {file_uri: "cd" * 32, DECISION_URI: "short"}}))
