"""`registry reconcile run-patch-extra` (reconcile.py): the top-level `extra` a legacy `run_patch` carried (the writer's central stamps
before the routing change) was dropped at the fold; the pass re-files, per run, ONLY the keys the live `run.extra_json` still lacks, as an
ordinary record into the run's own directory -- equal keys skipped, a conflict (the live value differs) reported and never overwritten.
Folded and rebuilt through the real harvester; the CLI's `--apply` is gated (exit 2, conflicts listed) in this release and the library
path is what the tests write through.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import reconcile as RC
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

STAMP_A = {"id": "legacy_stamp_v1", "sha256": "c" * 64, "keys": {"OVERRIDE_ID": "legacy_stamp_v1"}}
STAMP_B = {"id": "legacy_stamp_v2", "sha256": "d" * 64, "keys": {"OVERRIDE_ID": "legacy_stamp_v2"}}
OVERRIDES_TEXT = "OVERRIDE_ID=operator_shell_v9\nexport OVERRIDE_INIT_SCHEME=random_lecun_1e-3\n"


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def _seed_run(root, key="k", i0=100):
    """Dimensions plus one run through `running`, in the run's OWN directory `records/<campaign>/<run_dir_name(run_id)>/`."""
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records(key, i0, through="running")
    sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(rid))
    _write(root, sub, "000001-run.jsonl", recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return rid, sub


def _legacy(rid, fields, extra, t):
    """A `run_patch` as the pre-C1 writer shaped it: the stamps under a top-level `extra` the fold drops."""
    return synth.rec("run_patch", {"run_id": rid, "fields": dict(fields), "extra": extra}, synth.t(t))


def _live_extra(root, rid):
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    conn.close()
    assert row is not None, rid
    return json.loads(row[0]) if row[0] else {}


def _read_one(path):
    (_i, rec, _s), = list(R.read_record_file(path, compute_sha=False))
    return rec


def _rebuild_equal(root, tmp_path, rid):
    live = DB.connect(DB.db_path(root), readonly=True)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = {k: v for k, v in H.verify_equal(live, rebuilt).items() if v}
    assert not diff, diff
    sql = "SELECT extra_json, field_versions_json FROM run WHERE run_id = ?"
    assert tuple(live.execute(sql, (rid,)).fetchone()) == tuple(rebuilt.execute(sql, (rid,)).fetchone())
    live.close()
    rebuilt.close()


def _run_files(root, sub):
    return sorted(fn for fn in os.listdir(os.path.join(root, "records", sub)))


# --------------------------------------------------------------------------------------------
# the plan's first test: legacy dropped patch -> reconcile -> the run gains the missing keys + reconciled_q11 -> rebuild equal -> a
# second pass emits nothing
# --------------------------------------------------------------------------------------------


def test_a_legacy_dropped_patch_is_reconciled_missing_keys_only_rebuild_equal_and_a_second_pass_writes_nothing(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    legacy = _legacy(rid, {"job_name": "legacy"}, {"run_overrides": STAMP_A, "cluster_label": "TESTSITE"}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [legacy])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    before = _live_extra(root, rid)
    assert "run_overrides" not in before and "cluster_label" not in before          # the documented drop (C1's pinned contract)
    assert before["verified_fields"] == {"serving_path": "lora_native"}            # the lifecycle's own keys are there
    # the dry run: both keys missing, nothing written, the log untouched
    dry = RC.reconcile_run_patch_extra(root, apply=False)
    assert dry.files_read == 1 and dry.records_read == 1 and not dry.applied and dry.written == []
    (run,) = dry.runs
    assert run.run_id == rid and run.bucket == "missing" and run.directory == os.path.join(root, "records", sub)
    assert run.lost == {"run_overrides": STAMP_A, "cluster_label": "TESTSITE"} and run.missing == run.lost
    assert run.equal == [] and run.conflict == {} and run.pending == {} and not run.no_live_row and run.malformed == []
    assert run.from_record_sha256s == [R.record_sha(legacy)]
    assert dry.by_bucket()["missing"] == 1 and dry.keys_by_bucket() == {"missing": 2, "equal": 0, "conflict": 0, "pending": 0}
    assert _run_files(root, sub) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    # apply: ONE run_patch, the plan's shape exactly, in the run's own directory, validating clean, with no top-level extra
    app = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = app.runs
    assert app.applied and app.written == [run.written] and run.refused is None
    assert run.written == os.path.join(root, "records", sub, "000003-run_patch.jsonl")
    assert _run_files(root, sub) == ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_patch.jsonl"]
    rec = _read_one(run.written)
    assert rec["record_type"] == "run_patch" and rec["producer"]["script"] == RC.PRODUCER_SCRIPT
    assert rec["payload"] == {"run_id": rid, "fields": {"extra_json": {
        "run_overrides": STAMP_A, "cluster_label": "TESTSITE",
        "reconciled_q11": {"from_record_sha256s": [R.record_sha(legacy)], "decision": "Q11"}}}}
    assert "extra" not in rec["payload"]
    v = VAL.validate_record(rec, "fixture", R.record_sha(rec))
    assert isinstance(v, VAL.Validated) and v.extra["run_overrides"] == STAMP_A and v.extra["reconciled_q11"]["decision"] == "Q11"
    # the ordinary fold: the run gains the missing keys and the provenance; the lifecycle's keys stay; the patch's typed columns untouched
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    after = _live_extra(root, rid)
    assert after["run_overrides"] == STAMP_A and after["cluster_label"] == "TESTSITE"
    assert after["reconciled_q11"] == {"from_record_sha256s": [R.record_sha(legacy)], "decision": "Q11"}
    assert after["verified_fields"] == before["verified_fields"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT job_name, field_versions_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    conn.close()
    assert row[0] == "legacy" and json.loads(row[1])["job_name"] == [synth.t(150), R.record_sha(legacy)]
    _rebuild_equal(root, tmp_path, rid)
    # a second pass: the keys are equal now; the reconciliation record itself is not a lost record; nothing to write
    again = RC.reconcile_run_patch_extra(root, apply=True)
    (run2,) = again.runs
    assert run2.bucket == "equal" and sorted(run2.equal) == ["cluster_label", "run_overrides"] and run2.missing == {}
    assert again.written == [] and run2.written is None and run2.refused is None
    assert _run_files(root, sub) == ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_patch.jsonl"]


# --------------------------------------------------------------------------------------------
# the plan's second test: a historical lost value followed by a newer correctly folded value is a CONFLICT -- reported, nothing written
# for that key, the live value untouched; a missing key of the same run is still re-filed
# --------------------------------------------------------------------------------------------


def test_a_lost_value_followed_by_a_newer_folded_value_is_a_conflict_reported_and_never_overwritten(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    legacy = _legacy(rid, {"job_name": "legacy"}, {"run_overrides": STAMP_A, "cluster_label": "TESTSITE"}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [legacy])
    # a later patch states run_overrides the routed way, with ANOTHER value: the live row carries newer provenance
    routed = synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": {"run_overrides": STAMP_B}}}, synth.t(160))
    _write(root, sub, "000003-run_patch.jsonl", [routed])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _live_extra(root, rid)["run_overrides"] == STAMP_B and "cluster_label" not in _live_extra(root, rid)
    dry = RC.reconcile_run_patch_extra(root, apply=False)
    (run,) = dry.runs
    assert run.bucket == "conflict"
    assert run.conflict == {"run_overrides": {"lost": STAMP_A, "live": STAMP_B}}
    assert run.missing == {"cluster_label": "TESTSITE"} and run.equal == []
    assert dry.by_bucket()["conflict"] == 1 and dry.keys_by_bucket() == {"missing": 1, "equal": 0, "conflict": 1, "pending": 0}
    text = RC.format_report(dry)
    assert "CONFLICT run_overrides" in text and "legacy_stamp_v1" in text and "legacy_stamp_v2" in text and "missing: cluster_label" in text
    # apply re-files ONLY the missing key; the conflicting key is not in the record; the live value stands after the fold
    app = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = app.runs
    assert run.written == os.path.join(root, "records", sub, "000004-run_patch.jsonl")
    rec = _read_one(run.written)
    assert rec["payload"]["fields"]["extra_json"] == {"cluster_label": "TESTSITE",
                                                      "reconciled_q11": {"from_record_sha256s": [R.record_sha(legacy)], "decision": "Q11"}}
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    after = _live_extra(root, rid)
    assert after["run_overrides"] == STAMP_B and after["cluster_label"] == "TESTSITE"
    _rebuild_equal(root, tmp_path, rid)
    # and the conflict is STILL reported afterwards (the lost value stays visible in the log for the owner), nothing more written
    again = RC.reconcile_run_patch_extra(root, apply=True)
    (run2,) = again.runs
    assert run2.bucket == "conflict" and run2.conflict == {"run_overrides": {"lost": STAMP_A, "live": STAMP_B}}
    assert run2.equal == ["cluster_label"] and run2.missing == {} and again.written == []


# --------------------------------------------------------------------------------------------
# the merge: several lost records fold in (written_utc, sha) order, latest wins per key, whatever the file order; .gz files are read
# --------------------------------------------------------------------------------------------


def test_several_lost_records_merge_in_written_order_and_gz_files_are_read(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    older = _legacy(rid, {"job_name": "older"}, {"run_overrides": STAMP_A, "cluster_label": "TESTSITE"}, 150)
    newer = _legacy(rid, {"job_name": "newer"}, {"run_overrides": STAMP_B}, 170)
    _write(root, sub, "000002-run_patch.jsonl", [newer], gz=True)       # the NEWER record in the EARLIER file, gzipped
    _write(root, sub, "000003-run_patch.jsonl", [older])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    dry = RC.reconcile_run_patch_extra(root, apply=False)
    (run,) = dry.runs
    assert dry.files_read == 2 and len(run.records) == 2
    assert [r.written_utc for r in run.records] == [synth.t(150), synth.t(170)]
    assert run.lost == {"run_overrides": STAMP_B, "cluster_label": "TESTSITE"}       # latest wins per key; the older key survives
    assert run.missing == run.lost
    assert run.from_record_sha256s == [R.record_sha(older), R.record_sha(newer)]     # both carried a re-filed key, merge order
    # the pure rules, alone
    assert RC.merge_lost(list(reversed(run.records))) == run.lost
    missing, equal, conflict, pending = RC.classify_keys({"a": {"x": 1, "y": 2}, "b": 1, "c": 2}, {"a": {"y": 2, "x": 1}, "b": 2}, {"c": "sha"})
    assert missing == {} and equal == ["a"] and conflict == {"b": {"lost": 1, "live": 2}} and pending == {"c": "sha"}
    missing, equal, conflict, pending = RC.classify_keys({"a": 1}, None)
    assert missing == {"a": 1} and equal == [] and conflict == {} and pending == {}


# --------------------------------------------------------------------------------------------
# the loud buckets: pending (a reconciliation record not yet folded), no_live_row, malformed, a directory that is not the run's own
# --------------------------------------------------------------------------------------------


def test_a_reconciliation_record_not_yet_folded_is_pending_and_never_written_twice(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    legacy = _legacy(rid, {"job_name": "legacy"}, {"run_overrides": STAMP_A}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [legacy])
    H.ingest_once(root, workers=0, min_free=0)
    first = RC.reconcile_run_patch_extra(root, apply=True)
    assert len(first.written) == 1
    written_sha = R.record_sha(_read_one(first.written[0]))
    # no ingest in between: the key is still missing live, but its reconciliation is in the log -> pending, nothing written
    second = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = second.runs
    assert run.bucket == "pending" and run.pending == {"run_overrides": written_sha} and run.missing == {}
    assert second.written == [] and second.by_bucket()["pending"] == 1 and second.keys_by_bucket()["pending"] == 1
    assert "pending run_overrides" in RC.format_report(second)
    assert _run_files(root, sub) == ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_patch.jsonl"]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    (run,) = RC.reconcile_run_patch_extra(root, apply=True).runs
    assert run.bucket == "equal" and run.pending == {}


def test_a_run_with_no_live_row_or_a_malformed_lost_extra_or_a_foreign_directory_is_reported_not_written(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    # (a) a lost patch of a run that never folded (its run record never arrived): blocked at ingest, no row to compare against
    ghost = "synthcampaign:ghost:0123456789ab:0123456789abcdef"
    ghost_sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(ghost))
    _write(root, ghost_sub, "000001-run_patch.jsonl", [_legacy(ghost, {"job_name": "ghost"}, {"run_overrides": STAMP_A}, 150)])
    # (b) a lost `extra` that is not an object: no keys the fold could have merged
    bad = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "bad"}, "extra": "not-an-object"}, synth.t(150))
    _write(root, sub, "000002-run_patch.jsonl", [bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.blocked == 1 and rep.refused == 0, (rep.blocked, rep.refusals)
    dry = RC.reconcile_run_patch_extra(root, apply=True)
    by_id = {r.run_id: r for r in dry.runs}
    assert by_id[ghost].bucket == "no_live_row" and by_id[ghost].no_live_row and by_id[ghost].written is None
    assert by_id[rid].bucket == "malformed" and by_id[rid].malformed == [f"{R.record_sha(bad)}: top-level extra is a str, not an object"]
    assert by_id[rid].lost == {} and by_id[rid].written is None
    assert dry.written == [] and dry.by_bucket()["no_live_row"] == 1 and dry.by_bucket()["malformed"] == 1
    text = RC.format_report(dry)
    assert "no_live_row" in text and "malformed" in text and "/ no_live_row 1" in text and "/ malformed 1" in text
    # (c) a run whose lost record sits in a directory that is not spool/campaign/run_dir_name(run_id): the writer would write
    # elsewhere, so apply refuses that run by name -- the key is still reported missing
    rid2, _sub2 = _seed_run_second(root)
    foreign = os.path.join(synth.CAMPAIGN, "some_other_directory")
    legacy2 = _legacy(rid2, {"job_name": "legacy2"}, {"cluster_label": "TESTSITE"}, 150)
    _write(root, foreign, "000001-run_patch.jsonl", [legacy2])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    app = RC.reconcile_run_patch_extra(root, apply=True)
    run2 = {r.run_id: r for r in app.runs}[rid2]
    assert run2.bucket == "missing" and run2.missing == {"cluster_label": "TESTSITE"} and run2.written is None
    assert run2.refused and "not spool/campaign" in run2.refused and app.written == []
    assert sorted(os.listdir(os.path.join(root, "records", foreign))) == ["000001-run_patch.jsonl"]
    # (d) a run with a malformed lost record AND a missing key: refused until adjudicated
    good = _legacy(rid, {"job_name": "good"}, {"cluster_label": "TESTSITE"}, 160)
    _write(root, sub, "000003-run_patch.jsonl", [good])
    H.ingest_once(root, workers=0, min_free=0)
    run = {r.run_id: r for r in RC.reconcile_run_patch_extra(root, apply=True).runs}[rid]
    assert run.bucket == "missing" and run.missing == {"cluster_label": "TESTSITE"} and run.malformed and run.written is None
    assert run.refused == "a lost record of this run is not an object; adjudicate before re-filing"


def _seed_run_second(root):
    rid, recs = synth.run_records("k2", 300, through="running")
    sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(rid))
    _write(root, sub, "000001-run.jsonl", recs)
    return rid, sub


# --------------------------------------------------------------------------------------------
# the writer stamps nothing of its own on a reconciliation record: an operator's $SISTEER_RUN_OVERRIDES must not become a new
# statement over a live run_overrides
# --------------------------------------------------------------------------------------------


def test_the_reconciliation_writer_adds_no_central_stamp_even_with_an_overrides_file_in_the_environment(tmp_path, monkeypatch):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    legacy = _legacy(rid, {"job_name": "legacy"}, {"run_overrides": STAMP_A, "cluster_label": "TESTSITE"}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [legacy])
    routed = synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": {"run_overrides": STAMP_A}}}, synth.t(160))   # equal
    _write(root, sub, "000003-run_patch.jsonl", [routed])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    f = tmp_path / "operator_shell.env"
    f.write_text(OVERRIDES_TEXT)
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    assert emit.run_overrides_stamp()["id"] == "operator_shell_v9"         # an ordinary producer writer WOULD stamp this
    app = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = app.runs
    assert run.equal == ["run_overrides"] and run.missing == {"cluster_label": "TESTSITE"}
    rec = _read_one(run.written)
    assert rec["payload"]["fields"]["extra_json"] == {"cluster_label": "TESTSITE",
                                                      "reconciled_q11": {"from_record_sha256s": [R.record_sha(legacy)], "decision": "Q11"}}
    assert "extra" not in rec["payload"]
    assert len(rec["producer"]["code_sha"]) == 40 or rec["producer"]["code_sha"] == "unverified"
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    after = _live_extra(root, rid)
    assert after["run_overrides"] == STAMP_A and after["cluster_label"] == "TESTSITE"       # the live stamp untouched
    _rebuild_equal(root, tmp_path, rid)


#: the overrides file a legacy stamp was read from, as the runs saw it: one commented line the old reader kept the comment of
LEGACY_FILE_TEXT = "OVERRIDE_ID=legacy_stamp_v3\nOVERRIDE_EFFECTIVE_BATCH=64   # unchanged\n"
LEGACY_FILE_SHA = hashlib.sha256(LEGACY_FILE_TEXT.encode()).hexdigest()
POLLUTED_STAMP = {"id": "legacy_stamp_v3", "path": "legacy_stamp.env", "sha256": LEGACY_FILE_SHA, "keys": {"OVERRIDE_ID": "legacy_stamp_v3", "OVERRIDE_EFFECTIVE_BATCH": "64   # unchanged"}}
CLEAN_STAMP = {**POLLUTED_STAMP, "keys": {"OVERRIDE_ID": "legacy_stamp_v3", "OVERRIDE_EFFECTIVE_BATCH": "64"}}
#: the file's digest and key set, but a value the file never carried: a later statement, neither reading -- a CONFLICT, never equal
DIVERGENT_STAMP = {**POLLUTED_STAMP, "keys": {"OVERRIDE_ID": "legacy_stamp_v3", "OVERRIDE_EFFECTIVE_BATCH": "32"}}


def _legacy_file(tmp_path):
    f = tmp_path / "legacy_stamp.env"
    f.write_text(LEGACY_FILE_TEXT)
    return str(f)


def test_a_lost_run_overrides_stamp_compares_as_a_reading_of_its_file_by_the_given_sources_and_is_re_filed_as_its_record_stated_it(tmp_path, capsys):
    """A legacy stamp carries the inline comment text the old overrides-file reader kept; the live value may already be restated from the
    file (`registry reconcile env-comments`). With the file given (`--source`, env-comments' own), the two are statements of ONE file --
    each value of both a reading of it, the old reader's or the fixed one's -- `equal`, never `conflict`. A live value that is NEITHER
    reading (a later statement, the file's digest kept as provenance) stays a CONFLICT for the operator, whichever side carries it;
    without the file only identical stamps are equal; a stamp that differs in its digest, path or key SET stays a conflict; a value under
    any other key compares by canonical JSON as before. This pass re-reads nothing: a MISSING stamp is re-filed as the legacy record
    stated it (env-comments corrects it from the file once folded)."""
    src = _legacy_file(tmp_path)
    srcs = RC.overrides_sources([src], include_defaults=False)
    assert set(srcs) == {LEGACY_FILE_SHA} and srcs[LEGACY_FILE_SHA].keys == CLEAN_STAMP["keys"] and srcs[LEGACY_FILE_SHA].historical_keys == POLLUTED_STAMP["keys"]
    equal_ = ({}, ["run_overrides"], {}, {})
    assert RC.classify_keys({"run_overrides": POLLUTED_STAMP}, {"run_overrides": CLEAN_STAMP}, None, srcs) == equal_
    assert RC.classify_keys({"run_overrides": CLEAN_STAMP}, {"run_overrides": POLLUTED_STAMP}, None, srcs) == equal_
    assert RC.classify_keys({"run_overrides": POLLUTED_STAMP}, {"run_overrides": POLLUTED_STAMP}) == equal_                     # identical: equal with no file
    assert RC.classify_keys({"run_overrides": POLLUTED_STAMP}, {}, None, srcs) == ({"run_overrides": POLLUTED_STAMP}, [], {}, {})    # re-filed as stated
    # without the file, the polluted and the clean stamp are a conflict (nothing says they are one file's two readings)
    assert RC.classify_keys({"run_overrides": POLLUTED_STAMP}, {"run_overrides": CLEAN_STAMP}) == ({}, [], {"run_overrides": {"lost": POLLUTED_STAMP, "live": CLEAN_STAMP}}, {})
    # a value neither reading is a conflict WITH the file, on either side
    for lost, live in ((POLLUTED_STAMP, DIVERGENT_STAMP), (DIVERGENT_STAMP, CLEAN_STAMP), (CLEAN_STAMP, DIVERGENT_STAMP)):
        assert RC.classify_keys({"run_overrides": lost}, {"run_overrides": live}, None, srcs) == ({}, [], {"run_overrides": {"lost": lost, "live": live}}, {})
    for other in ({**CLEAN_STAMP, "keys": {**CLEAN_STAMP["keys"], "OVERRIDE_MODEL": "m"}}, {**CLEAN_STAMP, "sha256": "f" * 64}, {**CLEAN_STAMP, "path": "renamed.env"}):
        missing, equal, conflict, _p = RC.classify_keys({"run_overrides": POLLUTED_STAMP}, {"run_overrides": other}, None, srcs)
        assert not missing and not equal and conflict == {"run_overrides": {"lost": POLLUTED_STAMP, "live": other}}
    assert RC.classify_keys({"cluster_label": "x # y"}, {"cluster_label": "x"}, None, srcs) == ({}, [], {"cluster_label": {"lost": "x # y", "live": "x"}}, {})
    # through the log: a legacy dropped patch with the polluted stamp and a routed record with the clean one -> equal; a run with only the
    # legacy patch -> missing, re-filed verbatim, and equal to itself on the next dry run; a run whose routed record restated the stamp
    # with a value the file never carried -> conflict, named, nothing written
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    _write(root, sub, "000002-run_patch.jsonl", [_legacy(rid, {"job_name": "legacy"}, {"run_overrides": POLLUTED_STAMP}, 150)])
    _write(root, sub, "000003-run_patch.jsonl", [synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": {"run_overrides": CLEAN_STAMP}}}, synth.t(160))])
    rid2, sub2 = _seed_run_second(root)
    _write(root, sub2, "000002-run_patch.jsonl", [_legacy(rid2, {"job_name": "legacy"}, {"run_overrides": POLLUTED_STAMP}, 150)])
    rid3, recs3 = synth.run_records("k3", 400, through="running")
    sub3 = os.path.join(synth.CAMPAIGN, I.run_dir_name(rid3))
    _write(root, sub3, "000001-run.jsonl", recs3)
    _write(root, sub3, "000002-run_patch.jsonl", [_legacy(rid3, {"job_name": "legacy"}, {"run_overrides": POLLUTED_STAMP}, 150)])
    _write(root, sub3, "000003-run_patch.jsonl", [synth.rec("run_patch", {"run_id": rid3, "fields": {"extra_json": {"run_overrides": DIVERGENT_STAMP}}}, synth.t(160))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    app = RC.reconcile_run_patch_extra(root, apply=True, sources=[src])
    assert [s_.path for s_ in app.sources if s_.sha256 == LEGACY_FILE_SHA] == [src]
    by = {r.run_id: r for r in app.runs}
    assert by[rid].equal == ["run_overrides"] and not by[rid].conflict and by[rid].written is None
    assert by[rid2].missing == {"run_overrides": POLLUTED_STAMP} and by[rid2].written
    assert by[rid3].conflict == {"run_overrides": {"lost": POLLUTED_STAMP, "live": DIVERGENT_STAMP}} and by[rid3].equal == [] and by[rid3].written is None
    assert _read_one(by[rid2].written)["payload"]["fields"]["extra_json"]["run_overrides"] == POLLUTED_STAMP
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _live_extra(root, rid2)["run_overrides"] == POLLUTED_STAMP and _live_extra(root, rid3)["run_overrides"] == DIVERGENT_STAMP
    assert RC.reconcile_run_patch_extra(root, sources=[src]).by_bucket() == {**{b: 0 for b in RC.BUCKETS}, "equal": 2, "conflict": 1}
    # WITHOUT the file the restated run reads as a conflict too: the sources are what make the two readings one file
    assert RC.reconcile_run_patch_extra(root).by_bucket() == {**{b: 0 for b in RC.BUCKETS}, "equal": 1, "conflict": 2}
    # the CLI takes --source on run-patch-extra (the landing window passes env-comments' sources to it): the report names the files, the
    # totals read with them; --json carries the sources; a named path that is not there is refused
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--source", src]) == 0
    out = capsys.readouterr().out
    assert f"overrides sources (a run_overrides stamp compares as a reading of its file by these, matched by sha256): " in out
    assert f"  {LEGACY_FILE_SHA[:8]} {'legacy_stamp_v3':<36} {src}" in out and "runs 3: missing 0 / equal 2 / conflict 1" in out
    assert f"CONFLICT run_overrides: lost=" in out and f"conflict    {rid3}" in out
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--json", "--source", src]) == 0
    d = json.loads(capsys.readouterr().out)
    assert {"path": src, "sha256": LEGACY_FILE_SHA, "id": "legacy_stamp_v3"} in d["sources"] and d["runs_by_bucket"]["equal"] == 2
    assert cli.main(["--root", root, "reconcile", "run-patch-extra"]) == 0
    assert "runs 3: missing 0 / equal 1 / conflict 2" in capsys.readouterr().out
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--source", str(tmp_path / "nowhere.env")]) == 1
    assert "refused: no overrides file or directory at" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# the CLI: the dry run prints the buckets; --json is the whole report; --apply is GATED (exit 2, conflicts listed, nothing written)
# --------------------------------------------------------------------------------------------


def test_the_cli_dry_run_prints_the_buckets_and_apply_is_gated_with_exit_2_listing_the_conflicts(tmp_path, capsys):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    legacy = _legacy(rid, {"job_name": "legacy"}, {"run_overrides": STAMP_A, "cluster_label": "TESTSITE"}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [legacy])
    routed = synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": {"run_overrides": STAMP_B}}}, synth.t(160))
    _write(root, sub, "000003-run_patch.jsonl", [routed])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    files_before = _run_files(root, sub)
    assert RC.APPLY_ENABLED is False                                      # this release: dry-run only
    assert cli.main(["--root", root, "reconcile", "run-patch-extra"]) == 0
    out = capsys.readouterr()
    assert f"conflict    {rid}" in out.out and "CONFLICT run_overrides" in out.out and "missing: cluster_label" in out.out
    assert "runs 1: missing 0 / equal 0 / conflict 1" in out.out and "keys: missing 1 / equal 0 / conflict 1" in out.out
    assert "wrote" not in out.out and out.err == ""
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--dry-run"]) == 0
    assert "runs 1: missing 0 / equal 0 / conflict 1" in capsys.readouterr().out
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["runs"] == 1 and d["runs_by_bucket"]["conflict"] == 1 and d["keys_by_bucket"] == {"missing": 1, "equal": 0, "conflict": 1, "pending": 0}
    assert d["per_run"][0]["conflict"] == {"run_overrides": {"lost": STAMP_A, "live": STAMP_B}} and d["written"] == [] and not d["applied"]
    # --apply: gated -- exit 2, the report on stdout, the refusal and the conflicts on stderr, NOTHING written
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--apply"]) == 2
    out = capsys.readouterr()
    assert "CONFLICT run_overrides" in out.out
    assert "refused: --apply is gated" in out.err and "APPLY_ENABLED" in out.err and "1 run(s) carry a conflicting key" in out.err
    assert f"  {rid} run_overrides: lost=" in out.err and "legacy_stamp_v1" in out.err and "legacy_stamp_v2" in out.err
    assert _run_files(root, sub) == files_before
    assert _live_extra(root, rid)["run_overrides"] == STAMP_B and "cluster_label" not in _live_extra(root, rid)
    # --all lists the equal runs too; without it an all-equal run prints only in the totals
    routed2 = synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": {"cluster_label": "TESTSITE", "run_overrides": STAMP_A}}}, synth.t(170))
    _write(root, sub, "000004-run_patch.jsonl", [routed2])
    H.ingest_once(root, workers=0, min_free=0)
    assert cli.main(["--root", root, "reconcile", "run-patch-extra"]) == 0
    out = capsys.readouterr().out
    assert rid not in out and "runs 1: missing 0 / equal 1 / conflict 0" in out
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--all"]) == 0
    out = capsys.readouterr().out
    assert f"equal       {rid}" in out and "equal: cluster_label, run_overrides" in out
    # --dry-run and --apply exclude each other; a root with no record log is refused by name
    with pytest.raises(SystemExit):
        cli.main(["--root", root, "reconcile", "run-patch-extra", "--dry-run", "--apply"])
    assert cli.main(["--root", str(tmp_path / "nowhere"), "reconcile", "run-patch-extra"]) == 1
    assert "refused: " in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# Only a record the fold ACCEPTED is evidence. A legacy patch with a valid run_id and a well-formed stamp that the harvester
# REFUSED (its verify.serving_path disagreed with the row) rolled back before its row_history insert; the run row's existence
# says nothing about it. Its stamp is excluded, reported by status, never re-filed.
# --------------------------------------------------------------------------------------------


def test_a_refused_legacy_patch_is_not_evidence_its_stamp_is_excluded_and_reported_and_never_re_filed(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    # the run was registered via lora_native (synth's lifecycle patch verifies it); this patch claims hf -> VerifyMismatch -> refused
    refused = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "refused"}, "verify": {"serving_path": "hf"},
                                      "extra": {"run_overrides": STAMP_A}}, synth.t(150))
    _write(root, sub, "000002-run_patch.jsonl", [refused])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1 and rep.blocked == 0 and "VerifyMismatch" in rep.refusals[0], rep.refusals
    live = _live_extra(root, rid)
    assert "run_overrides" not in live
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT job_name FROM run WHERE run_id = ?", (rid,)).fetchone()[0] != "refused"   # nothing of it folded
    assert R.record_sha(refused) not in RC.accepted_run_patch_shas(conn, rid)
    assert RC.unaccepted_status(conn, R.record_sha(refused)).startswith("refused: VerifyMismatch: run_patch.verify['serving_path'] = 'hf'")
    assert len(RC.accepted_run_patch_shas(conn, rid)) == 1                                                  # the lifecycle's own patch
    conn.close()
    # the dry run: the run's ONLY lost record is unaccepted -> no evidence; bucket `unaccepted`; nothing missing; nothing to write
    dry = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = dry.runs
    assert run.bucket == "unaccepted" and run.accepted == [] and len(run.records) == 1
    assert list(run.unaccepted) == [R.record_sha(refused)] and run.status_of(R.record_sha(refused)) == "refused"
    assert run.lost == {} and run.missing == {} and run.equal == [] and run.conflict == {} and run.pending == {}
    assert run.written is None and run.refused is None and dry.written == []
    assert dry.by_bucket()["unaccepted"] == 1 and dry.records_by_status() == {"accepted": 0, "refused": 1, "blocked": 0, "not_ingested": 0}
    text = RC.format_report(dry)
    assert f"unaccepted  {rid}" in text and f"unaccepted {R.record_sha(refused)[:12]}: refused: VerifyMismatch" in text
    assert "/ unaccepted 1" in text and "lost records: accepted 0 / refused 1 / blocked 0 / not_ingested 0" in text
    assert dry.as_dict()["per_run"][0]["unaccepted"] == run.unaccepted and dry.as_dict()["records_by_status"]["refused"] == 1
    assert _run_files(root, sub) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    # an ACCEPTED legacy patch of the same run (cluster_label): re-filed alone; the refused record is not a source, its key not re-filed
    good = _legacy(rid, {"job_name": "good"}, {"cluster_label": "TESTSITE"}, 160)
    _write(root, sub, "000003-run_patch.jsonl", [good])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    app = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = app.runs
    assert run.bucket == "missing" and run.lost == {"cluster_label": "TESTSITE"} and run.missing == run.lost
    assert [r.sha for r in run.accepted] == [R.record_sha(good)] and run.from_record_sha256s == [R.record_sha(good)]
    assert list(run.unaccepted) == [R.record_sha(refused)] and run.unaccepted[R.record_sha(refused)].startswith("refused: VerifyMismatch")
    assert run.written == os.path.join(root, "records", sub, "000004-run_patch.jsonl") and run.refused is None
    rec = _read_one(run.written)
    assert rec["payload"]["fields"]["extra_json"] == {"cluster_label": "TESTSITE",
                                                      "reconciled_q11": {"from_record_sha256s": [R.record_sha(good)], "decision": "Q11"}}
    assert app.records_by_status() == {"accepted": 1, "refused": 1, "blocked": 0, "not_ingested": 0}
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    after = _live_extra(root, rid)
    assert after["cluster_label"] == "TESTSITE" and "run_overrides" not in after        # the refused evidence never reached the row
    _rebuild_equal(root, tmp_path, rid)
    # a refused record that is the NEWEST and names a key the accepted evidence also names does not win the merge either
    refused2 = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "refused2"}, "verify": {"serving_path": "hf"},
                                       "extra": {"cluster_label": "OTHERSITE", "run_overrides": STAMP_B}}, synth.t(170))
    _write(root, sub, "000005-run_patch.jsonl", [refused2])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1, rep.refusals
    again = RC.reconcile_run_patch_extra(root, apply=True)
    (run2,) = again.runs
    assert run2.lost == {"cluster_label": "TESTSITE"} and run2.bucket == "equal" and run2.equal == ["cluster_label"]
    assert set(run2.unaccepted) == {R.record_sha(refused), R.record_sha(refused2)} and again.written == []
    assert _live_extra(root, rid)["cluster_label"] == "TESTSITE"


def test_a_lost_record_the_fold_has_not_weighed_yet_holds_the_run_until_an_ingest(tmp_path):
    """A legacy patch in the log that no cycle has folded yet (`not_ingested`) may still fold and change which record is latest, so
    apply holds the run and names the record; after the ingest the pass re-files from the whole accepted evidence."""
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    first = _legacy(rid, {"job_name": "first"}, {"run_overrides": STAMP_A}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [first])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    later = _legacy(rid, {"job_name": "later"}, {"run_overrides": STAMP_B, "cluster_label": "TESTSITE"}, 160)
    _write(root, sub, "000003-run_patch.jsonl", [later])                              # in the log, NOT ingested
    held = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = held.runs
    assert run.unaccepted == {R.record_sha(later): "not_ingested"} and run.unfolded == run.unaccepted
    assert run.lost == {"run_overrides": STAMP_A} and run.missing == run.lost and run.bucket == "missing"   # the folded evidence only
    assert run.written is None and held.written == []
    assert run.refused == f"a lost record of this run is not folded yet ({R.record_sha(later)[:12]} not_ingested); ingest first, then re-run"
    assert "refused: a lost record of this run is not folded yet" in RC.format_report(held)
    assert held.records_by_status() == {"accepted": 1, "refused": 0, "blocked": 0, "not_ingested": 1}
    assert _run_files(root, sub) == ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_patch.jsonl"]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    app = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = app.runs
    assert run.unaccepted == {} and run.lost == {"run_overrides": STAMP_B, "cluster_label": "TESTSITE"} and run.missing == run.lost
    assert run.from_record_sha256s == [R.record_sha(first), R.record_sha(later)] and run.written and run.refused is None
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    after = _live_extra(root, rid)
    assert after["run_overrides"] == STAMP_B and after["cluster_label"] == "TESTSITE"
    _rebuild_equal(root, tmp_path, rid)


def test_a_blocked_lost_record_of_a_run_with_no_live_row_is_reported_by_status(tmp_path):
    root = _root(tmp_path)
    _seed_run(root)
    ghost = "synthcampaign:ghost:0123456789ab:0123456789abcdef"
    ghost_sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(ghost))
    lost = _legacy(ghost, {"job_name": "ghost"}, {"run_overrides": STAMP_A}, 150)
    _write(root, ghost_sub, "000001-run_patch.jsonl", [lost])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.blocked == 1 and rep.refused == 0, (rep.blocked, rep.refusals)
    dry = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = dry.runs
    assert run.bucket == "no_live_row" and run.unaccepted == {R.record_sha(lost): f"blocked: run {ghost}"}
    assert run.lost == {"run_overrides": STAMP_A} and run.missing == {} and run.written is None and dry.written == []
    assert dry.records_by_status() == {"accepted": 0, "refused": 0, "blocked": 1, "not_ingested": 0}
    assert f"unaccepted {R.record_sha(lost)[:12]}: blocked: run {ghost}" in RC.format_report(dry)


def test_apply_holds_a_run_with_a_pending_key_or_an_unfolded_record_and_calls_no_writer(tmp_path):
    """The pure hold (reconcile.apply_reconciliations): a run with a missing key AND a pending reconciliation record, or AND a
    blocked / not-ingested lost record, gets `refused` naming the record; the writer is never constructed."""
    calls = []

    def factory(run, now):
        calls.append(run.run_id)
        raise AssertionError("the writer must not be built for a held run")

    rec = RC.LostExtra("r", synth.t(150), "a" * 64, "/x/000001-run_patch.jsonl", {"cluster_label": "TESTSITE"})
    pend = RC.RunReconciliation("r", "/x", [rec], lost=dict(rec.extra), missing={"cluster_label": "TESTSITE"},
                                pending={"run_overrides": "b" * 64})
    blocked = RC.RunReconciliation("r", "/x", [rec], lost=dict(rec.extra), missing={"cluster_label": "TESTSITE"},
                                   unaccepted={"c" * 64: "blocked: run r"})
    refused_only = RC.RunReconciliation("r", "/x", [rec], lost=dict(rec.extra), missing={"cluster_label": "TESTSITE"},
                                        unaccepted={"d" * 64: "refused: VerifyMismatch: x"})
    RC.apply_reconciliations([pend, blocked], writer_factory=factory)
    assert calls == []
    assert pend.refused == f"a reconciliation record of this run is not folded yet ({'b' * 12} for run_overrides); ingest first, then re-run"
    assert blocked.refused == f"a lost record of this run is not folded yet ({'c' * 12} blocked: run r); ingest first, then re-run"
    assert pend.written is None and blocked.written is None
    # a refused record is final: it is excluded, and the run is NOT held on its account
    written = []

    class W:
        def write(self, kind, payload):
            written.append((kind, payload))
            return "/x/000002-run_patch.jsonl"

    RC.apply_reconciliations([refused_only], writer_factory=lambda run, now: W())
    assert refused_only.refused is None and refused_only.written == "/x/000002-run_patch.jsonl"
    assert written == [("run_patch", {"run_id": "r", "fields": {"extra_json": {"cluster_label": "TESTSITE",
                                                                                 "reconciled_q11": {"from_record_sha256s": ["a" * 64], "decision": "Q11"}}}})]


# --------------------------------------------------------------------------------------------
# The marker is append-only. One key reconciled and folded, then a legacy patch for the same run arrives from another machine with a
# different lost key: the second record's marker carries the live marker's sources AND the new ones -- the reducer merges `extra`
# shallowly (the latest marker replaces the whole object), so without the union the first key would keep its value but lose its
# reconciliation provenance.
# --------------------------------------------------------------------------------------------


def test_a_second_reconciliation_after_a_newly_arrived_lost_patch_keeps_the_first_markers_provenance(tmp_path):
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    first = _legacy(rid, {"job_name": "first"}, {"run_overrides": STAMP_A}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [first])
    H.ingest_once(root, workers=0, min_free=0)
    one = RC.reconcile_run_patch_extra(root, apply=True)
    assert len(one.written) == 1
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    live = _live_extra(root, rid)
    assert live["run_overrides"] == STAMP_A and live["reconciled_q11"] == {"from_record_sha256s": [R.record_sha(first)], "decision": "Q11"}
    # an older producer's legacy patch arrives later (another machine's spool), losing a DIFFERENT key
    arrived = _legacy(rid, {"job_name": "arrived"}, {"cluster_label": "TESTSITE"}, 160)
    _write(root, sub, "000004-run_patch.jsonl", [arrived])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert "cluster_label" not in _live_extra(root, rid)
    two = RC.reconcile_run_patch_extra(root, apply=True)
    (run,) = two.runs
    assert run.bucket == "missing" and run.missing == {"cluster_label": "TESTSITE"} and run.equal == ["run_overrides"]
    assert run.live_reconciled == {"from_record_sha256s": [R.record_sha(first)], "decision": "Q11"}
    assert run.from_record_sha256s == [R.record_sha(arrived)]                        # this record's OWN sources
    assert run.written == os.path.join(root, "records", sub, "000005-run_patch.jsonl")
    rec = _read_one(run.written)
    assert rec["payload"]["fields"]["extra_json"] == {
        "cluster_label": "TESTSITE",
        "reconciled_q11": {"from_record_sha256s": [R.record_sha(first), R.record_sha(arrived)], "decision": "Q11"}}   # the union, live first
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    after = _live_extra(root, rid)
    assert after["run_overrides"] == STAMP_A and after["cluster_label"] == "TESTSITE"
    assert after["reconciled_q11"] == {"from_record_sha256s": [R.record_sha(first), R.record_sha(arrived)], "decision": "Q11"}
    _rebuild_equal(root, tmp_path, rid)
    three = RC.reconcile_run_patch_extra(root, apply=True)
    (run3,) = three.runs
    assert run3.bucket == "equal" and sorted(run3.equal) == ["cluster_label", "run_overrides"] and three.written == []
    # the pure rule: the live list kept in order, new sources appended once, a malformed live list ignored
    assert RC.marker_sha256s({"from_record_sha256s": ["s1", "s2"]}, ["s2", "s3"]) == ["s1", "s2", "s3"]
    assert RC.marker_sha256s({}, ["s1"]) == ["s1"] and RC.marker_sha256s({"from_record_sha256s": "junk"}, ["s1"]) == ["s1"]
    assert RC.marker_sha256s({"from_record_sha256s": ["s1", 7]}, []) == ["s1"]


# --------------------------------------------------------------------------------------------
# the command writes NOTHING of its own under the root -- not even the state/last_run stamp every other command leaves -- on a dry
# run or on a gated --apply; SQLite's own side files of a read-only open are the one allowance, pinned empty
# --------------------------------------------------------------------------------------------


def _tree(root):
    """Every file under the root with its size and mtime: the whole-root no-write check."""
    out = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


#: SQLite's own side files of the WAL-mode database: a read-only (`mode=ro`) connection CREATES them when no other connection holds
#: them open -- any reader does, `sqlite3 file:...?mode=ro` included -- and cannot remove them on close (no write lock); the WAL stays
#: empty, the next writer's close reclaims both. Measured on sqlite 3.45 (2026-10-03). `immutable=1` would avoid them and is NOT used:
#: it disables locking and WAL reads, so against the live timers' writer it reads a silently stale snapshot.
_SQLITE_SIDE_FILES = ("registry.sqlite-wal", "registry.sqlite-shm")


def _tree_without_sqlite_side_files(root):
    """`_tree` with SQLite's two side files set aside: (every other file, the side files that were present)."""
    tree = _tree(root)
    side = {f: tree.pop(f) for f in _SQLITE_SIDE_FILES if f in tree}
    return tree, side


def test_the_cli_leaves_no_last_run_stamp_and_nothing_but_sqlites_own_side_files_on_a_dry_run_or_a_gated_apply(tmp_path, capsys):
    """Finding 3: a dry run and a gated `--apply` write nothing of their own under the root -- no `state/last_run` stamp, no record,
    not a byte of the database. The ONE thing either can leave is SQLite's (`_SQLITE_SIDE_FILES`): shown here by a bare read-only
    open of the same database leaving exactly those two files before the command ever runs, and by the command adding nothing to
    the root beyond them, with the WAL empty."""
    root = _root(tmp_path)
    rid, sub = _seed_run(root)
    legacy = _legacy(rid, {"job_name": "legacy"}, {"run_overrides": STAMP_A, "cluster_label": "TESTSITE"}, 150)
    _write(root, sub, "000002-run_patch.jsonl", [legacy])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert not os.path.exists(os.path.join(root, "state", "last_run"))
    assert not any(os.path.exists(os.path.join(root, f)) for f in _SQLITE_SIDE_FILES)   # the ingest's writer closed; SQLite reclaimed them
    _live_extra(root, rid)                                                              # a bare read-only open of the WAL database, closed...
    assert all(os.path.exists(os.path.join(root, f)) for f in _SQLITE_SIDE_FILES)       # ...leaves SQLite's two side files: the reader's, not the command's
    before, side0 = _tree_without_sqlite_side_files(root)
    assert side0["registry.sqlite-wal"][0] == 0
    assert cli.main(["--root", root, "reconcile", "run-patch-extra"]) == 0
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--dry-run", "--json"]) == 0
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--apply"]) == 2
    out = capsys.readouterr()
    assert "missing: cluster_label" in out.out and "refused: --apply is gated" in out.err
    after, side = _tree_without_sqlite_side_files(root)
    assert after == before                             # every other file as it was (size + mtime), the database file included; nothing new
    assert set(side) == set(_SQLITE_SIDE_FILES) and side["registry.sqlite-wal"][0] == 0   # the WAL holds no frame: nothing reached the database
    assert not os.path.exists(os.path.join(root, "state", "last_run"))               # the generic stamp main() leaves for every other command


# --------------------------------------------------------------------------------------------
# `registry reconcile multi-turn-scope`: the job-start default `single_turn` on a multi-turn eval, re-derived from the row's mode and the
# eval's category, as ONE run_patch per run into the run's own directory; the dry run counts by (model, category); never the database
# --------------------------------------------------------------------------------------------

MT_EVAL_ID = "synthmt"
MT_ITEM_SET_ID = "synthmt_259"
MT_INSTRUMENT = synth.instrument_fields(eval_id=MT_EVAL_ID, item_set_id=MT_ITEM_SET_ID, harness="pyrit", max_turns=10,
                                        attacker_model_id="synth_attacker", context_window_tokens=49152)
MT_INSTRUMENT_ID = str(MT_INSTRUMENT["instrument_id"])


def _mt_dimensions(i0=30):
    """A multi-turn eval beside the synthetic static one: its item set and an every-token instrument (the synthetic attacker is a model row)."""
    return [
        synth.rec("eval", {"eval_id": MT_EVAL_ID, "eval_category": "multi_turn", "headline_metric_id": "asr", "metric_direction": "lower_is_better",
                           "attacker_required": 1, "paper_status": "in_paper"}, synth.t(i0)),
        synth.rec("item_set", {"item_set_id": MT_ITEM_SET_ID, "eval_id": MT_EVAL_ID, "dataset_version": "synthetic/synthmt@v1", "n_items": 259,
                               "items_sha256": synth.SHA256_B, "source_uri": "synthetic://items/synthmt_259.jsonl", "item_id_scheme": "g<index>",
                               "subsample_rule": "full"}, synth.t(i0 + 1)),
        synth.rec("model", {"model_id": "synth_attacker", "family": "qwen36", "size_label": "27b", "checkpoint_kind": "instruct",
                            "hf_repo": "synthetic/attacker-27b", "hf_revision": "e" * 40, "chat_template_sha256": synth.SHA256_A,
                            "generation_config_present": 1, "generation_config_json": {"temperature": 1.0}, "thinking_channel": "available",
                            "paper_status": "reference"}, synth.t(i0 + 2)),
        synth.rec("instrument", MT_INSTRUMENT, synth.t(i0 + 3)),
    ]


def _mt_run(root, key, i0, object_id, scope, eval_id=MT_EVAL_ID, item_set_id=MT_ITEM_SET_ID, instrument_id=MT_INSTRUMENT_ID, object_key=None, **over):
    """A run through `running` on the given eval with the given scope stamped (the job-start default, or the derived value)."""
    object_key = object_key or ("none" if object_id == "none" else "constitution_lora@a1b2c3d4")
    rid = I.run_id(synth.CAMPAIGN, key, object_key, eval_id, instrument_id)
    rid2, recs = synth.run_records(key, i0, object_key, object_id, through="running", run_id=rid, eval_id=eval_id, item_set_id=item_set_id,
                                   instrument_id=instrument_id, multi_turn_steering_scope=scope, **over)
    assert rid2 == rid
    sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(rid))
    _write(root, sub, "000001-run.jsonl", recs)
    return rid, sub


#: a confidence entry run A's registration carries on ANOTHER column: the correction record never carries it, and it stands after the fold
A_LIVE_CONFIDENCE = {"code_sha": {"confidence": "unverified", "reason": "git rev-parse HEAD unavailable in the registering checkout"}}


#: the synthetic system instruction's prompted object (`si:<sha>`) and the run-key spelling a prompted entry registers under
SI_OBJECT = synth.object_si()
SI_OBJECT_ID = str(SI_OBJECT["object_id"])
SI_OBJECT_KEY = "constitution_si@prompted"


def _scope_corpus(tmp_path, prompted=False):
    """Four runs: A steered on the multi-turn eval with the default (SELECTED; its registration carries a confidence entry on another
    column); B unsteered on it (not_applicable, not selected); C steered on the static eval with single_turn (its category is single-turn,
    not selected); D steered on the multi-turn eval already carrying the derived scope (not selected). With ``prompted``, a fifth: E, a
    PROMPTED run (object `si:<sha>`, kind system_instruction) on the multi-turn eval with the default and the instrument's `all` copied
    onto its mode column, as a job-start registration stamps it (SELECTED; its kind decides its scope, never its mode)."""
    root = _root(tmp_path)
    dims = synth.seed_dimensions() + _mt_dimensions()
    if prompted:
        dims.append(synth.rec("object", SI_OBJECT, synth.t(34)))
    _write(root, "dims", "000001-dims.jsonl", dims)
    a, sub_a = _mt_run(root, "mtA", 100, synth.OBJECT_SHA, "single_turn", field_confidence_json=A_LIVE_CONFIDENCE)
    b, _ = _mt_run(root, "mtB", 110, "none", "not_applicable")
    c, _ = _mt_run(root, "stC", 120, synth.OBJECT_SHA, "single_turn", eval_id=synth.EVAL_ID, item_set_id=synth.ITEM_SET_ID, instrument_id=synth.INSTRUMENT_ID)
    d, _ = _mt_run(root, "mtD", 130, synth.OBJECT_SHA, "every_turn_prefill_and_decode")
    runs = {"a": (a, sub_a), "b": b, "c": c, "d": d}
    if prompted:
        e, sub_e = _mt_run(root, "mtE", 140, SI_OBJECT_ID, "single_turn", object_key=SI_OBJECT_KEY, applied_positions_mode="all",
                           transfer_kind="prompted_only", strength_coefficient=None, strength_nominal=None, strength_nominal_source=None,
                           is_operating_point=None)
        runs["e"] = (e, sub_e)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return root, runs


def _live_scope(root, rid):
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT multi_turn_steering_scope, field_confidence_json, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    conn.close()
    return row[0], json.loads(row[1] or "{}"), json.loads(row[2] or "{}")


def test_multi_turn_scope_dry_run_selects_the_defaulted_steered_multi_turn_runs_and_counts_them_by_model_and_category(tmp_path, capsys):
    root, runs = _scope_corpus(tmp_path)
    a, _sub = runs["a"]
    assert _live_scope(root, a)[0] == "single_turn"                        # the job-start default, folded as the producer stated it
    rep = RC.reconcile_multi_turn_scope(root, apply=False)
    assert [r.run_id for r in rep.runs] == [a] and not rep.applied and rep.written == [] and rep.held == []
    (run,) = rep.runs
    assert (run.model_id, run.eval_category, run.applied_positions_mode, run.live_scope) == (synth.MODEL_ID, "multi_turn", "all", "single_turn")
    assert (run.scope, run.confidence) == ("every_turn_prefill_and_decode", "derived") and "applied_positions_mode=all on a multi-turn eval" in run.source
    assert run.pending is None
    assert rep.by_group() == {(synth.MODEL_ID, "multi_turn"): 1} and rep.by_scope() == {"every_turn_prefill_and_decode": 1}
    d = rep.as_dict()
    assert d["by_model_and_category"] == [{"model_id": synth.MODEL_ID, "eval_category": "multi_turn", "runs": 1}] and d["runs"] == 1 and d["written"] == 0
    # the CLI's dry run: the count by (model, category), exit 0, nothing written under the root (the generic last_run stamp included)
    before, _side = _tree_without_sqlite_side_files(root)
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope"]) == 0
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--dry-run", "--json"]) == 0
    out = capsys.readouterr().out
    assert f"  {synth.MODEL_ID:<24} {'multi_turn':<12} {1:>6}" in out and "runs 1 in 1 (model, category) group(s); nothing written (dry run)" in out
    assert '"by_derived_scope": {\n  "every_turn_prefill_and_decode": 1\n }' in out
    after, _side = _tree_without_sqlite_side_files(root)
    assert after == before and not os.path.exists(os.path.join(root, "state", "last_run"))


def test_multi_turn_scope_apply_writes_one_patch_per_run_that_folds_rebuild_equal_and_a_second_pass_selects_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path)
    a, sub_a = runs["a"]
    # the record carries the scope's own confidence entry and no other; the row's live entry on another column stands after the fold
    assert _live_scope(root, a)[1] == A_LIVE_CONFIDENCE
    assert _run_files(root, sub_a) == ["000001-run.jsonl"]
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    (run,) = app.runs
    assert app.applied and app.written == [run.written] and run.refused is None and app.held == []
    assert run.written == os.path.join(root, "records", sub_a, "000002-run_patch.jsonl")       # the run's OWN directory, next in sequence
    rec = _read_one(run.written)
    assert rec["record_type"] == "run_patch" and rec["producer"]["script"] == RC.SCOPE_PRODUCER_SCRIPT and rec["payload"].get("extra") in (None, {})
    assert rec["payload"] == {"run_id": a, "fields": {
        "multi_turn_steering_scope": "every_turn_prefill_and_decode",
        "field_confidence_json": {"multi_turn_steering_scope": {"confidence": "derived", "source": "applied_positions_mode=all on a multi-turn eval"}},
        "extra_json": {RC.SCOPE_RECONCILED_KEY: {"replaced": "single_turn", "derived_from": {"applied_positions_mode": "all", "eval_category": "multi_turn",
                                                                                              "object_kind": "lora"},
                                                 "pass": RC.SCOPE_PRODUCER_SCRIPT}}}}
    assert isinstance(VAL.validate_record(rec, run.written, R.record_sha(rec)), VAL.Validated)
    # before the ingest a second apply holds the run (its record is in the log, not folded) and the CLI says so with exit 2
    again = RC.reconcile_multi_turn_scope(root, apply=True)
    assert [r.run_id for r in again.runs] == [a] and again.written == [] and again.runs[0].pending == R.record_sha(rec) and "not folded yet" in again.runs[0].refused
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--apply"]) == 2
    err = capsys.readouterr().err
    assert "1 run(s) held" in err
    # the fold: the scope moves, the confidence entry lands beside the row's others, the marker is in extra_json; the other three rows stand
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    scope, conf, extra = _live_scope(root, a)
    assert scope == "every_turn_prefill_and_decode" and conf["multi_turn_steering_scope"]["confidence"] == "derived"
    assert conf["code_sha"] == A_LIVE_CONFIDENCE["code_sha"] and set(conf) == {"code_sha", "multi_turn_steering_scope"}   # the row's other entry stands
    assert extra[RC.SCOPE_RECONCILED_KEY]["replaced"] == "single_turn" and extra["verified_fields"] == {"serving_path": "lora_native"}
    assert _live_scope(root, runs["b"])[0] == "not_applicable" and _live_scope(root, runs["c"])[0] == "single_turn" and _live_scope(root, runs["d"])[0] == "every_turn_prefill_and_decode"
    _rebuild_equal(root, tmp_path, a)
    # idempotent: the predicate selects nothing now, the apply writes nothing, the CLI exits 0 on both
    third = RC.reconcile_multi_turn_scope(root, apply=True)
    assert third.runs == [] and third.written == []
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--apply"]) == 0
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"]


def test_multi_turn_scope_payload_carries_the_scopes_own_confidence_entry_and_no_other():
    """The record states the provenance of the one column it patches; the row's other entries stand under the fold's per-column read
    of a patch's confidence JSON, so the record needs no sight of them (and the pass reads none)."""
    run = RC.ScopeCorrection("r", "c", "m", "e", "petri", "lora", "non_template", "single_turn", "every_turn_prefill_and_decode", "derived",
                             "applied_positions_mode=non_template on a multi-turn eval")
    payload = RC.scope_correction_payload(run)
    assert payload["fields"]["field_confidence_json"] == {
        "multi_turn_steering_scope": {"confidence": "derived", "source": "applied_positions_mode=non_template on a multi-turn eval"}}
    assert payload["fields"]["multi_turn_steering_scope"] == "every_turn_prefill_and_decode"
    assert payload["fields"]["extra_json"][RC.SCOPE_RECONCILED_KEY]["derived_from"] == {"applied_positions_mode": "non_template", "eval_category": "petri", "object_kind": "lora"}
    assert not hasattr(run, "live_confidence")
    unresolved = RC.ScopeCorrection("r", "c", "m", "e", "petri", "lora", "unverified", "single_turn", "unverified", "unverified",
                                    "applied_positions_mode='unverified' unresolved")
    entry = RC.scope_correction_payload(unresolved)["fields"]["field_confidence_json"]["multi_turn_steering_scope"]
    assert entry == {"confidence": "unverified", "source": "applied_positions_mode='unverified' unresolved", "reason": "applied_positions_mode='unverified' unresolved"}
    # a prompted run: the marker names the kind the derivation read beside the mode it did not act on
    prompted = RC.ScopeCorrection("r", "c", "m", "e", "petri", "system_instruction", "all", "single_turn", "not_applicable", "derived",
                                  "object_kind=system_instruction: a prompted object steers no positions")
    fields = RC.scope_correction_payload(prompted)["fields"]
    assert fields["multi_turn_steering_scope"] == "not_applicable"
    assert fields["extra_json"][RC.SCOPE_RECONCILED_KEY]["derived_from"] == {"applied_positions_mode": "all", "eval_category": "petri", "object_kind": "system_instruction"}


def test_multi_turn_scope_reads_the_objects_kind_so_a_prompted_run_derives_not_applicable_beside_a_steered_runs_mode_scope(tmp_path, monkeypatch, capsys):
    """The predicate selects a prompted run (object `si:<sha>`, kind system_instruction) that registered with the default and the
    instrument's `all` on its mode column exactly as it selects a steered one -- but the derivation reads the object's kind, so the
    prompted run's record states `not_applicable` with the kind's reason, never the mode table's every_turn_prefill_and_decode; the steered
    run beside it still derives from its mode. Dry run, apply, fold, rebuild-equal and idempotence, both runs together."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path, prompted=True)
    a, _sub_a = runs["a"]
    e, sub_e = runs["e"]
    assert _live_scope(root, e)[0] == "single_turn"
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT o.object_kind, r.applied_positions_mode FROM run r JOIN object o ON o.object_id = r.object_id WHERE r.run_id = ?",
                              (e,)).fetchone()) == ("system_instruction", "all")
    conn.close()
    rep = RC.reconcile_multi_turn_scope(root, apply=False)
    by_id = {r.run_id: r for r in rep.runs}
    assert set(by_id) == {a, e} and rep.written == [] and rep.held == []
    assert (by_id[a].object_kind, by_id[a].scope, by_id[a].confidence) == ("lora", "every_turn_prefill_and_decode", "derived")
    assert (by_id[e].object_kind, by_id[e].applied_positions_mode, by_id[e].live_scope) == ("system_instruction", "all", "single_turn")
    assert (by_id[e].scope, by_id[e].confidence, by_id[e].source) == ("not_applicable", "derived", "object_kind=system_instruction: a prompted object steers no positions")
    assert rep.by_group() == {(synth.MODEL_ID, "multi_turn"): 2}
    assert rep.by_scope() == {"every_turn_prefill_and_decode": 1, "not_applicable": 1}
    assert rep.by_kind() == {("lora", "every_turn_prefill_and_decode"): 1, ("system_instruction", "not_applicable"): 1}
    d = rep.as_dict()
    assert d["by_object_kind"] == [{"object_kind": "lora", "scope": "every_turn_prefill_and_decode", "runs": 1},
                                   {"object_kind": "system_instruction", "scope": "not_applicable", "runs": 1}]
    assert {c["run_id"]: c["object_kind"] for c in d["corrections"]} == {a: "lora", e: "system_instruction"}
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope"]) == 0
    out = capsys.readouterr().out
    assert "derived scope: every_turn_prefill_and_decode 1, not_applicable 1" in out
    assert "by object kind: lora -> every_turn_prefill_and_decode 1, system_instruction -> not_applicable 1" in out
    # apply: the prompted run's record states the kind's scope and names the kind it read beside the mode it did not act on
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    assert app.held == [] and len(app.written) == 2
    rec_e = _read_one(os.path.join(root, "records", sub_e, "000002-run_patch.jsonl"))
    assert rec_e["payload"] == {"run_id": e, "fields": {
        "multi_turn_steering_scope": "not_applicable",
        "field_confidence_json": {"multi_turn_steering_scope": {"confidence": "derived", "source": "object_kind=system_instruction: a prompted object steers no positions"}},
        "extra_json": {RC.SCOPE_RECONCILED_KEY: {"replaced": "single_turn", "derived_from": {"applied_positions_mode": "all", "eval_category": "multi_turn",
                                                                                              "object_kind": "system_instruction"},
                                                 "pass": RC.SCOPE_PRODUCER_SCRIPT}}}}
    assert isinstance(VAL.validate_record(rec_e, "<mem>", R.record_sha(rec_e)), VAL.Validated)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    scope_e, conf_e, extra_e = _live_scope(root, e)
    assert scope_e == "not_applicable" and conf_e["multi_turn_steering_scope"]["confidence"] == "derived"
    assert extra_e[RC.SCOPE_RECONCILED_KEY]["derived_from"]["object_kind"] == "system_instruction"
    assert _live_scope(root, a)[0] == "every_turn_prefill_and_decode"
    assert _live_scope(root, runs["b"])[0] == "not_applicable" and _live_scope(root, runs["c"])[0] == "single_turn" and _live_scope(root, runs["d"])[0] == "every_turn_prefill_and_decode"
    _rebuild_equal(root, tmp_path, e)
    again = RC.reconcile_multi_turn_scope(root, apply=True)
    assert again.runs == [] and again.written == []


def test_multi_turn_scope_refuses_the_pass_by_name_when_a_selected_run_names_an_object_with_no_row(tmp_path):
    """The object's kind decides the scope; a run whose object row is missing (the schema's reference says it exists) cannot be read, and
    the pass refuses whole rather than deriving from the mode what may be a prompted run's scope. Checked on a bare database that holds
    the three tables the selection reads."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE run (run_id TEXT, campaign TEXT, model_id TEXT, eval_id TEXT, object_id TEXT, applied_positions_mode TEXT, multi_turn_steering_scope TEXT);"
        "CREATE TABLE eval (eval_id TEXT, eval_category TEXT);"
        "CREATE TABLE object (object_id TEXT, object_kind TEXT);"
        "INSERT INTO eval VALUES ('mt', 'multi_turn');"
        "INSERT INTO object VALUES ('aaaa', 'lora');"
        "INSERT INTO run VALUES ('c:mt:i:1', 'c', 'm', 'mt', 'aaaa', 'all', 'single_turn');"
        "INSERT INTO run VALUES ('c:mt:i:2', 'c', 'm', 'mt', 'orphan', 'all', 'single_turn');")
    with pytest.raises(ValueError, match=r"1 run\(s\) of the class name an object with no object row \(c:mt:i:2 -> object orphan\)"):
        RC.reconcile_multi_turn_scope(str(tmp_path), apply=False, conn=conn)
    conn.execute("INSERT INTO object VALUES ('orphan', 'system_instruction')")
    conn.executescript("CREATE TABLE row_history (table_name TEXT, natural_key_json TEXT, record_type TEXT, record_sha256 TEXT);")
    # the hold reads the whole log for the selected runs' statements: a root with no record log is refused by name, never read as empty
    with pytest.raises(FileNotFoundError, match="no record log at"):
        RC.reconcile_multi_turn_scope(str(tmp_path), apply=False, conn=conn)
    os.makedirs(os.path.join(str(tmp_path), "records"))
    rep = RC.reconcile_multi_turn_scope(str(tmp_path), apply=False, conn=conn)
    assert {r.run_id: (r.object_kind, r.scope) for r in rep.runs} == {"c:mt:i:1": ("lora", "every_turn_prefill_and_decode"), "c:mt:i:2": ("system_instruction", "not_applicable")}
    conn.close()


def _statement(rid, fields, t, verify=None):
    """A `run_patch` another producer filed for the run (a standup's endpoint patch, an older hook's mode-only patch)."""
    payload = {"run_id": rid, "fields": dict(fields)}
    if verify:
        payload["verify"] = dict(verify)
    return synth.rec("run_patch", payload, synth.t(t))


def test_multi_turn_scope_holds_a_run_whose_mode_or_scope_an_unfolded_record_states_wherever_in_the_log_it_lies(tmp_path, monkeypatch, capsys):
    """Three selected runs with the job-start default. A: a standup's endpoint patch (the real hook's payload: served `response`, the scope
    re-derived beside it) pushed into a directory that is NOT the run's own and not yet ingested. F: an older hook's mode-only patch in
    its own directory, not yet ingested. G: a statement of the mode the fold REFUSED (its verify disagrees with the row). The pass holds A
    and F (the dry run names the record and the columns it states; the apply writes nothing for them and the CLI exits 2) and corrects G
    (a refusal is final). After the ingest A reads (`response`, `final_header_and_current_decode_only`) -- never the stale correction's
    `every_turn_prefill_and_decode` beside a served `response` -- F reads (`response`, `single_turn`) and the next pass derives F's scope from
    its NEW mode; everything rebuilds equal."""
    from sisteer.registry import hooks
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path)
    a, sub_a = runs["a"]
    f, sub_f = _mt_run(root, "mtF", 150, synth.OBJECT_SHA, "single_turn")
    g, sub_g = _mt_run(root, "mtG", 160, synth.OBJECT_SHA, "single_turn")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    # G's statement, refused at the fold (the row was served lora_native): final, so it never holds the run
    refused = _statement(g, {"applied_positions_mode": "response"}, 300, verify={"serving_path": "lens"})
    _write(root, sub_g, "000002-run_patch.jsonl", [refused])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1 and RC.unaccepted_status(DB.connect(DB.db_path(root), readonly=True), R.record_sha(refused)).startswith("refused:")
    # A's endpoint patch, the hook's own payload, in a foreign directory (a push under another name); F's mode-only patch in its own
    endpoint = {"attested_max_num_seqs": 48, "engine": "lora_native", "adapters": {"k": {"petri_intervention_hash": "a1b2c3d4e5f60718", "layers": [1]}},
                "served_positions": {"k": ["response"]},
                "engine_identity": {"versions": {"vllm": "0.26.0"}, "adapters": {"k": {"petri_source_kind": "lora_r4", "petri_intervention_hash": "a1b2c3d4e5f60718"}}}}
    payload = hooks.serving_patch_payload(endpoint, "k", {"run_id": a, "campaign": synth.CAMPAIGN, "eval_category": "multi_turn", "expected": {}})
    assert payload["fields"]["applied_positions_mode"] == "response" and payload["fields"]["multi_turn_steering_scope"] == "final_header_and_current_decode_only"
    a_patch = synth.rec("run_patch", payload, synth.t(310))
    _write(root, os.path.join(synth.CAMPAIGN, "pushed-from-a-standup"), "000001-run_patch.jsonl", [a_patch])
    f_patch = _statement(f, {"applied_positions_mode": "response", "field_confidence_json": {"applied_positions_mode": {"confidence": "recorded", "source": "an older hook"}}}, 320)
    _write(root, sub_f, "000002-run_patch.jsonl", [f_patch])
    # the dry run: all three selected, A and F pending with the record and the columns it states, G free
    dry = RC.reconcile_multi_turn_scope(root, apply=False)
    by_id = {r.run_id: r for r in dry.runs}
    assert set(by_id) == {a, f, g}
    assert (by_id[a].pending, by_id[a].pending_columns, by_id[a].pending_own) == (R.record_sha(a_patch), ("applied_positions_mode", "multi_turn_steering_scope"), False)
    assert (by_id[f].pending, by_id[f].pending_columns, by_id[f].pending_own) == (R.record_sha(f_patch), ("applied_positions_mode",), False)
    assert by_id[g].pending is None
    text = RC.format_scope_report(dry)
    assert "pending (a record stating the run's mode or scope is in the log, not folded yet): 2 run(s)" in text
    assert f"  {a} record {R.record_sha(a_patch)[:12]} states applied_positions_mode, multi_turn_steering_scope" in text
    assert f"  {f} record {R.record_sha(f_patch)[:12]} states applied_positions_mode" in text
    d = dry.as_dict()
    assert {c["run_id"]: (c["pending_columns"], c["pending_own"]) for c in d["corrections"]} == {
        a: (["applied_positions_mode", "multi_turn_steering_scope"], False), f: (["applied_positions_mode"], False), g: ([], False)}
    # the apply (the CLI): A and F held by name, G written, exit 2 saying what holds them
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--apply"]) == 2
    err = capsys.readouterr().err
    assert ("2 run(s) held: a record stating each one's mode or scope (this pass's own correction, a standup's endpoint patch, or a full run record "
            "carrying either column) is already in the log and not folded yet") in err
    assert _run_files(root, sub_a) == ["000001-run.jsonl"] and _run_files(root, sub_f) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    assert _run_files(root, sub_g) == ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_patch.jsonl"]
    # a second apply before the ingest: A and F as before, and G now behind its own correction record
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    held = {r.run_id: r for r in app.held}
    assert set(held) == {a, f, g} and app.written == []
    assert held[a].refused == f"a record of this run stating applied_positions_mode, multi_turn_steering_scope is in the log and not folded yet ({R.record_sha(a_patch)[:12]}); ingest first, then re-run"
    assert held[f].refused.startswith("a record of this run stating applied_positions_mode is in the log and not folded yet")
    assert held[g].pending_own and held[g].pending_columns == ("multi_turn_steering_scope",)
    assert held[g].refused.startswith("this pass's own correction record of this run stating multi_turn_steering_scope is in the log and not folded yet")
    assert _run_files(root, sub_g) == ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_patch.jsonl"]
    # the ingest folds the three statements: A consistent, F's mode moved, G corrected
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = {rid: tuple(conn.execute("SELECT applied_positions_mode, multi_turn_steering_scope, served_max_num_seqs FROM run WHERE run_id = ?", (rid,)).fetchone())
            for rid in (a, f, g)}
    conn.close()
    assert rows[a] == ("response", "final_header_and_current_decode_only", 48)
    assert rows[f] == ("response", "single_turn", None)
    assert rows[g] == ("all", "every_turn_prefill_and_decode", None)
    # the next pass: A and G are no longer selected; F is, and its scope now derives from the served mode
    again = RC.reconcile_multi_turn_scope(root, apply=True)
    assert [r.run_id for r in again.runs] == [f] and again.held == [] and len(again.written) == 1
    (run_f,) = again.runs
    assert (run_f.applied_positions_mode, run_f.scope, run_f.confidence) == ("response", "final_header_and_current_decode_only", "derived")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _live_scope(root, f)[0] == "final_header_and_current_decode_only" and _live_scope(root, a)[0] == "final_header_and_current_decode_only"
    assert RC.reconcile_multi_turn_scope(root, apply=True).runs == []
    _rebuild_equal(root, tmp_path, a)                                      # every table equal, the foreign-directory record included


def test_multi_turn_scope_reads_the_runs_own_directory_again_at_the_write(tmp_path, monkeypatch):
    """The second guard: a statement that lands in the run's own directory AFTER the whole-log walk and before the write (here: the walk
    stubbed to have seen nothing) holds the run at the write -- nothing is written, the hold names the record -- and a refused statement
    there does not."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path)
    a, sub_a = runs["a"]
    late = _statement(a, {"applied_positions_mode": "response", "field_confidence_json": {"applied_positions_mode": {"confidence": "recorded", "source": "x"}}}, 300)
    _write(root, sub_a, "000002-run_patch.jsonl", [late])
    monkeypatch.setattr(RC, "scope_statements", lambda records_root, run_ids=None: {})
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    (run,) = app.runs
    assert app.written == [] and run.pending == R.record_sha(late) and run.pending_columns == ("applied_positions_mode",) and not run.pending_own
    assert run.refused == f"a record of this run stating applied_positions_mode is in the log and not folded yet ({R.record_sha(late)[:12]}); ingest first, then re-run"
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    # the statement folded: the row's mode moved, the pass derives from it and writes
    monkeypatch.undo()
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    (run,) = app.runs
    assert run.pending is None and run.written == os.path.join(root, "records", sub_a, "000003-run_patch.jsonl")
    assert (run.applied_positions_mode, run.scope) == ("response", "final_header_and_current_decode_only")
    assert RC.scope_correction_payload(run)["fields"]["extra_json"][RC.SCOPE_RECONCILED_KEY]["derived_from"]["applied_positions_mode"] == "response"


def _full_run(rid, key, i, **over):
    """A later full `run` record of an existing run (a re-registration of the identity, a producer's row, a store's opening): the
    synthetic registration payload with the run id bound and the multi-turn eval's identity, `over` on top."""
    payload = synth.run_payload(key, "constitution_lora@a1b2c3d4", synth.OBJECT_SHA, i, run_id=rid, eval_id=MT_EVAL_ID, item_set_id=MT_ITEM_SET_ID,
                                instrument_id=MT_INSTRUMENT_ID, **over)
    return synth.rec("run", payload, synth.t(i))


def test_run_statement_files_name_the_two_record_types_files_and_nothing_else(tmp_path):
    """The hold's walk reads `*-run.jsonl` and `*-run_patch.jsonl` (`.gz` too) under every directory -- the files the two record types are
    written to by name -- and never the state, ingest, item or dimension files; an absent log is refused by name."""
    root = tmp_path / "records"
    for sub, names in (("a/r1", ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_state.jsonl", "000004-item_result.jsonl.gz", "000005-run.jsonl.gz"]),
                       ("harvester/d", ["000001-run_ingest.jsonl", "000002-run_patch.jsonl.gz"]), ("dims", ["000001-dims.jsonl"])):
        (root / sub).mkdir(parents=True)
        for n in names:
            (root / sub / n).write_bytes(b"")
    got = [os.path.relpath(p, root) for p in RC.run_statement_files(str(root))]
    assert got == ["a/r1/000001-run.jsonl", "a/r1/000002-run_patch.jsonl", "a/r1/000005-run.jsonl.gz", "harvester/d/000002-run_patch.jsonl.gz"]
    assert [os.path.relpath(p, root) for p in RC.run_patch_files(str(root))] == ["a/r1/000002-run_patch.jsonl", "harvester/d/000002-run_patch.jsonl.gz"]
    assert RC.SCOPE_STATEMENT_RECORD_TYPES == ("run", "run_patch")
    with pytest.raises(FileNotFoundError):
        RC.run_statement_files(str(tmp_path / "absent"))


def test_multi_turn_scope_holds_a_run_behind_an_unfolded_full_run_record_as_behind_a_patch(tmp_path, monkeypatch, capsys):
    """Three selected runs with the job-start default, each with a full `run` record in the log the fold has not weighed: A, a
    re-registration of the identity in its own directory moving the mode to `response` and the scope with it; F, a producer's record under a
    FOREIGN directory restating `response` beside the default scope; G, a store's opening with its copies of the plan's `all` / `single_turn`
    marked unstated (`hooks.STORE_PLANNED_COLUMNS`), whose effect is the fold's call. The pass holds all three (the dry run names the record,
    its type and the columns; the apply writes nothing and the CLI exits 2) -- before this the walk read `run_patch` files alone, so A's
    pending `response` was invisible and the pass wrote a scope derived from the database's `all`, leaving (`response`,
    `every_turn_prefill_and_decode`) after the ingest. After the ingest A reads (`response`, `final_header_and_current_decode_only`) and is no
    longer selected; F reads (`response`, `single_turn`) and the next pass derives from `response`; G's marked copies were skipped over the
    registration's values, so it reads (`all`, `single_turn`) and the next pass derives from `all`. Everything rebuilds equal."""
    from sisteer.registry import hooks
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path)
    a, sub_a = runs["a"]
    f, sub_f = _mt_run(root, "mtF", 150, synth.OBJECT_SHA, "single_turn")
    g, sub_g = _mt_run(root, "mtG", 160, synth.OBJECT_SHA, "single_turn")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    accepted_a = RC.accepted_run_statement_shas(conn, a)
    assert RC.accepted_run_patch_shas(conn, a) < accepted_a and len(accepted_a) == len(RC.accepted_run_patch_shas(conn, a)) + 1   # the registration's `run` event too
    conn.close()
    full_a = _full_run(a, "mtA", 300, applied_positions_mode="response", multi_turn_steering_scope="final_header_and_current_decode_only",
                       field_confidence_json=dict(A_LIVE_CONFIDENCE, applied_positions_mode={"confidence": "recorded", "source": "a re-registration"}))
    _write(root, sub_a, "000002-run.jsonl", [full_a])
    full_f = _full_run(f, "mtF", 310, applied_positions_mode="response")
    _write(root, os.path.join(synth.CAMPAIGN, "pushed-from-a-producer"), "000001-run.jsonl", [full_f])
    full_g = _full_run(g, "mtG", 320, extra={"store_key": "k", hooks.UNSTATED_COLUMNS_KEY: list(hooks.STORE_PLANNED_COLUMNS)})
    _write(root, sub_g, "000002-run.jsonl", [full_g])
    for rec in (full_a, full_f, full_g):
        assert isinstance(VAL.validate_record(rec), VAL.Validated), getattr(VAL.validate_record(rec), "reason", None)
    both = ("applied_positions_mode", "multi_turn_steering_scope")
    # the dry run: all three pending, each behind its full record, the record type named
    dry = RC.reconcile_multi_turn_scope(root, apply=False)
    by_id = {r.run_id: r for r in dry.runs}
    assert set(by_id) == {a, f, g}
    for rid, rec in ((a, full_a), (f, full_f), (g, full_g)):
        assert (by_id[rid].pending, by_id[rid].pending_columns, by_id[rid].pending_own, by_id[rid].pending_type) == (R.record_sha(rec), both, False, "run"), rid
        assert by_id[rid].hold_reason == (f"a full run record of this run stating applied_positions_mode, multi_turn_steering_scope is in the log and not folded yet "
                                          f"({R.record_sha(rec)[:12]}); ingest first, then re-run")
    text = RC.format_scope_report(dry)
    assert "pending (a record stating the run's mode or scope is in the log, not folded yet): 3 run(s)" in text
    assert f"  {a} record {R.record_sha(full_a)[:12]} states applied_positions_mode, multi_turn_steering_scope (a full run record)" in text
    assert {c["run_id"]: c["pending_type"] for c in dry.as_dict()["corrections"]} == {a: "run", f: "run", g: "run"}
    # the apply (the CLI): every run held, nothing written, exit 2 naming a full run record among what holds a run
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--apply"]) == 2
    err = capsys.readouterr().err
    assert "3 run(s) held: a record stating each one's mode or scope (this pass's own correction, a standup's endpoint patch, or a full run record carrying either column)" in err
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run.jsonl"] and _run_files(root, sub_f) == ["000001-run.jsonl"]
    assert _run_files(root, sub_g) == ["000001-run.jsonl", "000002-run.jsonl"]
    # the ingest folds the three records: A moved whole, F's mode moved, G's marked copies skipped over the registration's statements
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = {rid: tuple(conn.execute("SELECT applied_positions_mode, multi_turn_steering_scope, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone())
            for rid in (a, f, g)}
    assert R.record_sha(full_a) in RC.accepted_run_statement_shas(conn, a) and R.record_sha(full_a) not in RC.accepted_run_patch_shas(conn, a)
    conn.close()
    assert rows[a][:2] == ("response", "final_header_and_current_decode_only")
    assert rows[f][:2] == ("response", "single_turn")
    assert rows[g][:2] == ("all", "single_turn") and json.loads(rows[g][2])["store_key"] == "k" and hooks.UNSTATED_COLUMNS_KEY not in json.loads(rows[g][2])
    # the next pass: A is no longer selected; F and G are, each derived from its folded mode, nothing pending
    again = RC.reconcile_multi_turn_scope(root, apply=True)
    derived = {r.run_id: (r.applied_positions_mode, r.scope, r.confidence, r.pending) for r in again.runs}
    assert derived == {f: ("response", "final_header_and_current_decode_only", "derived", None), g: ("all", "every_turn_prefill_and_decode", "derived", None)}
    assert again.held == [] and len(again.written) == 2
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _live_scope(root, f)[0] == "final_header_and_current_decode_only" and _live_scope(root, g)[0] == "every_turn_prefill_and_decode"
    assert _live_scope(root, a)[0] == "final_header_and_current_decode_only"
    assert RC.reconcile_multi_turn_scope(root, apply=True).runs == []
    _rebuild_equal(root, tmp_path, a)                                      # every table equal, the foreign-directory record included


def test_multi_turn_scope_reads_a_full_run_record_in_the_runs_own_directory_again_at_the_write(tmp_path, monkeypatch):
    """The second guard reads full `run` records too: with the whole-log walk stubbed blind, a re-registration landing in the run's own
    directory before the write holds the run at the write, by type and name; after the ingest the pass derives from the moved mode."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path)
    a, sub_a = runs["a"]
    late = _full_run(a, "mtA", 300, applied_positions_mode="response", field_confidence_json=A_LIVE_CONFIDENCE)
    _write(root, sub_a, "000002-run.jsonl", [late])
    monkeypatch.setattr(RC, "scope_statements", lambda records_root, run_ids=None: {})
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    (run,) = app.runs
    assert app.written == [] and (run.pending, run.pending_columns, run.pending_own, run.pending_type) == (R.record_sha(late), ("applied_positions_mode", "multi_turn_steering_scope"), False, "run")
    assert run.refused == (f"a full run record of this run stating applied_positions_mode, multi_turn_steering_scope is in the log and not folded yet "
                           f"({R.record_sha(late)[:12]}); ingest first, then re-run")
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run.jsonl"]
    monkeypatch.undo()
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    (run,) = app.runs
    assert run.pending is None and run.written == os.path.join(root, "records", sub_a, "000003-run_patch.jsonl")
    assert (run.applied_positions_mode, run.scope) == ("response", "final_header_and_current_decode_only")


# --------------------------------------------------------------------------------------------
# the verify and the fixed point (the closing review's HIGH): a statement published into ANOTHER directory after the pass's last read
# --------------------------------------------------------------------------------------------


def _endpoint_patch(run_id, t):
    """The real hook's endpoint payload for the run -- served `response`, the scope that follows it -- as the standup's record."""
    from sisteer.registry import hooks
    endpoint = {"attested_max_num_seqs": 48, "engine": "lora_native", "adapters": {"k": {"petri_intervention_hash": "a1b2c3d4e5f60718", "layers": [1]}},
                "served_positions": {"k": ["response"]},
                "engine_identity": {"versions": {"vllm": "0.26.0"}, "adapters": {"k": {"petri_source_kind": "lora_r4", "petri_intervention_hash": "a1b2c3d4e5f60718"}}}}
    payload = hooks.serving_patch_payload(endpoint, "k", {"run_id": run_id, "campaign": synth.CAMPAIGN, "eval_category": "multi_turn", "expected": {}})
    assert payload["fields"]["applied_positions_mode"] == "response" and payload["fields"]["multi_turn_steering_scope"] == "final_header_and_current_decode_only"
    return synth.rec("run_patch", payload, synth.t(t))


def _mode_scope(conn_or_root, rid):
    """(applied_positions_mode, multi_turn_steering_scope) of the run, from a root's live database or an open connection."""
    conn = DB.connect(DB.db_path(conn_or_root), readonly=True) if isinstance(conn_or_root, str) else conn_or_root
    row = conn.execute("SELECT applied_positions_mode, multi_turn_steering_scope FROM run WHERE run_id = ?", (rid,)).fetchone()
    if isinstance(conn_or_root, str):
        conn.close()
    return tuple(row)


def test_multi_turn_scope_race_a_foreign_directory_endpoint_patch_after_the_last_read_is_folded_over_and_the_second_pass_converges(tmp_path, monkeypatch, capsys):
    """THE RACE (the closing review's HIGH). The pass reads A's row (`all`, `single_turn`), walks the log, re-reads A's own directory at the
    write -- and THEN a standup publishes A's endpoint patch (served `response`, the scope that follows it) into another record directory:
    the `writer_factory` seam is exactly the moment after that last read and before the write, so the patch lands there and nothing of the
    pass sees it. The pass writes its `all`-derived correction; the one ingest folds both, the later-written correction winning: (`response`,
    `every_turn_prefill_and_decode`), the rebuild agreeing. The verify lists A by name (the CLI exits 3); the second pass selects A -- its
    scope differs from the derived value, the predicate being consistency -- derives from the FOLDED `response` and writes; the ingest leaves
    (`response`, `final_header_and_current_decode_only`); the verify reads zero (exit 0) live and rebuilt alike; a third pass writes nothing."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path)
    a, sub_a = runs["a"]
    foreign = os.path.join(synth.CAMPAIGN, "pushed-from-a-standup")
    a_patch = _endpoint_patch(a, 310)
    published = []

    def racing_writer(root_, run, now):
        # the standup's publish, between the pass's last read of the run's own directory and its write
        _write(root_, foreign, "000001-run_patch.jsonl", [a_patch])
        published.append(run.run_id)
        return RC._scope_writer_for(root_, run, now)

    app = RC.reconcile_multi_turn_scope(root, apply=True, writer_factory=racing_writer)
    (run,) = app.runs
    assert published == [a] and run.written and run.refused is None and (run.applied_positions_mode, run.scope) == ("all", "every_turn_prefill_and_decode")
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    # the fold: both records accepted, the later-written correction wins -- the inconsistent row the HIGH names; the rebuild agrees
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert {R.record_sha(a_patch), R.record_sha(_read_one(run.written))} <= RC.accepted_run_statement_shas(conn, a)
    conn.close()
    assert _mode_scope(root, a) == ("response", "every_turn_prefill_and_decode")
    (tmp_path / "raced").mkdir()
    _rebuild_equal(root, tmp_path / "raced", a)
    # the verify: A listed by name with what the row reads and what the derivation reads; exit 3; no hold, nothing written, no last_run stamp
    before, _side = _tree_without_sqlite_side_files(root)
    ver = RC.verify_multi_turn_scope(root)
    assert ver.verify and not ver.applied and ver.written == [] and ver.held == []
    assert [(r.run_id, r.applied_positions_mode, r.live_scope, r.scope, r.pending) for r in ver.runs] == [(a, "response", "every_turn_prefill_and_decode", "final_header_and_current_decode_only", None)]
    d = ver.as_dict()
    assert d["verify"] is True and d["applied"] is False and d["runs"] == 1 and d["corrections"][0]["run_id"] == a
    text = RC.format_scope_report(ver)
    assert text.splitlines()[1] == f"  {a} (response, every_turn_prefill_and_decode) -> final_header_and_current_decode_only [lora]" and text.endswith("inconsistent 1 run(s)")
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--verify"]) == 3
    out = capsys.readouterr().out
    assert f"  {a} (response, every_turn_prefill_and_decode) -> final_header_and_current_decode_only [lora]" in out and "inconsistent 1 run(s)" in out
    after, _side = _tree_without_sqlite_side_files(root)
    assert after == before and not os.path.exists(os.path.join(root, "state", "last_run"))
    assert _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl"]
    # the second pass: A selected (inconsistent), nothing pending (both records are folded), derived from the folded mode, written
    second = RC.reconcile_multi_turn_scope(root, apply=True)
    (run2,) = second.runs
    assert run2.pending is None and run2.refused is None and run2.written == os.path.join(root, "records", sub_a, "000003-run_patch.jsonl")
    assert (run2.applied_positions_mode, run2.live_scope, run2.scope, run2.confidence) == ("response", "every_turn_prefill_and_decode", "final_header_and_current_decode_only", "derived")
    rec2 = _read_one(run2.written)
    assert rec2["payload"]["fields"]["multi_turn_steering_scope"] == "final_header_and_current_decode_only"
    assert rec2["payload"]["fields"]["extra_json"][RC.SCOPE_RECONCILED_KEY] == {
        "replaced": "every_turn_prefill_and_decode", "derived_from": {"applied_positions_mode": "response", "eval_category": "multi_turn", "object_kind": "lora"},
        "pass": RC.SCOPE_PRODUCER_SCRIPT}
    assert isinstance(VAL.validate_record(rec2, run2.written, R.record_sha(rec2)), VAL.Validated)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _mode_scope(root, a) == ("response", "final_header_and_current_decode_only")
    assert _live_scope(root, a)[1]["multi_turn_steering_scope"]["source"] == "applied_positions_mode=response on a multi-turn eval"
    # the fixed point: the verify reads zero live (exit 0) and over the rebuilt database, whose row agrees; a third pass writes nothing
    assert RC.verify_multi_turn_scope(root).runs == []
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--verify"]) == 0
    assert capsys.readouterr().out.rstrip().endswith("inconsistent 0 run(s); the class is consistent")
    _rebuild_equal(root, tmp_path, a)
    rebuilt = DB.connect(str(tmp_path / "rebuilt.sqlite"), readonly=True)
    assert _mode_scope(rebuilt, a) == ("response", "final_header_and_current_decode_only") and RC.verify_multi_turn_scope(root, conn=rebuilt).runs == []
    rebuilt.close()
    third = RC.reconcile_multi_turn_scope(root, apply=True)
    assert third.runs == [] and third.written == [] and _run_files(root, sub_a) == ["000001-run.jsonl", "000002-run_patch.jsonl", "000003-run_patch.jsonl"]
    assert _live_scope(root, runs["b"])[0] == "not_applicable" and _live_scope(root, runs["c"])[0] == "single_turn" and _live_scope(root, runs["d"])[0] == "every_turn_prefill_and_decode"


def test_multi_turn_scope_verify_reads_zero_on_a_consistent_registry_and_the_predicate_is_consistency_not_the_default(tmp_path, monkeypatch, capsys):
    """The predicate is consistency: beside A (the job-start default) the pass selects H, whose scope an adapter stamped `unverified` while
    the mode was unresolved and whose mode a later record settled to `all` (the live class beside the defaulted rows) -- its scope differs
    from the derived value -- and never D, which already carries it. The verify lists exactly those (exit 3, the JSON shape); after the
    apply and the fold it reads zero (exit 0, both renderings), writes nothing under the root, and the pass selects nothing; the rebuild
    agrees. `--verify` is multi-turn-scope's: run-patch-extra refuses it by name."""
    monkeypatch.setattr(emit.socket, "gethostname", lambda: "TESTHOST")
    root, runs = _scope_corpus(tmp_path)
    a, _sub_a = runs["a"]
    h, sub_h = _mt_run(root, "mtH", 170, synth.OBJECT_SHA, "unverified", applied_positions_mode="all",
                       field_confidence_json={"multi_turn_steering_scope": {"confidence": "unverified", "reason": "applied_positions_mode='unverified' unresolved"}})
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _mode_scope(root, h) == ("all", "unverified")
    ver = RC.verify_multi_turn_scope(root)
    assert {r.run_id: (r.live_scope, r.scope, r.object_kind) for r in ver.runs} == {a: ("single_turn", "every_turn_prefill_and_decode", "lora"),
                                                                                     h: ("unverified", "every_turn_prefill_and_decode", "lora")}
    dry = RC.reconcile_multi_turn_scope(root, apply=False)
    assert {r.run_id for r in dry.runs} == {a, h} and dry.by_group() == {(synth.MODEL_ID, "multi_turn"): 2}
    assert "multi_turn_steering_scope differs from the derived value" in RC.format_scope_report(dry).splitlines()[0]
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--verify", "--json"]) == 3
    d = json.loads(capsys.readouterr().out)
    assert d["verify"] is True and d["applied"] is False and d["runs"] == 2 and d["written"] == 0 and d["held"] == 0
    assert {c["run_id"]: (c["live_scope"], c["scope"]) for c in d["corrections"]} == {a: ("single_turn", "every_turn_prefill_and_decode"), h: ("unverified", "every_turn_prefill_and_decode")}
    app = RC.reconcile_multi_turn_scope(root, apply=True)
    assert len(app.written) == 2 and app.held == []
    rec_h = _read_one(os.path.join(root, "records", sub_h, "000002-run_patch.jsonl"))
    assert rec_h["payload"]["fields"]["extra_json"][RC.SCOPE_RECONCILED_KEY]["replaced"] == "unverified"
    assert isinstance(VAL.validate_record(rec_h, "<mem>", R.record_sha(rec_h)), VAL.Validated)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    assert _mode_scope(root, h) == ("all", "every_turn_prefill_and_decode") and _mode_scope(root, a) == ("all", "every_turn_prefill_and_decode")
    assert _live_scope(root, h)[1]["multi_turn_steering_scope"] == {"confidence": "derived", "source": "applied_positions_mode=all on a multi-turn eval"}
    # consistent: the verify reads zero in both renderings, exits 0, leaves nothing under the root; the pass selects nothing; rebuild equal
    before, _side = _tree_without_sqlite_side_files(root)
    assert RC.verify_multi_turn_scope(root).runs == []
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--verify"]) == 0
    assert cli.main(["--root", root, "reconcile", "multi-turn-scope", "--verify", "--json"]) == 0
    out = capsys.readouterr().out
    assert "inconsistent 0 run(s); the class is consistent" in out and '"verify": true' in out and '"runs": 0' in out
    after, _side = _tree_without_sqlite_side_files(root)
    assert after == before and not os.path.exists(os.path.join(root, "state", "last_run"))
    assert RC.reconcile_multi_turn_scope(root, apply=True).runs == [] and RC.reconcile_multi_turn_scope(root, apply=False).runs == []
    _rebuild_equal(root, tmp_path, h)
    # --verify is multi-turn-scope's
    assert cli.main(["--root", root, "reconcile", "run-patch-extra", "--verify"]) == 1
    assert "refused: --verify belongs to the correction actions (multi-turn-scope, env-comments, fit-effective-batch" in capsys.readouterr().err
