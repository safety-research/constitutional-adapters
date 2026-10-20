"""Ruling expiry vs evidence validity: every case asserts the PAPER SELECTION of the
covered rows, not only the stamp. Missing / path-absent evidence keeps an exclusion active; an empty
root set never expires; a mixed {old, new} root set expires (as the reader); a re-filed served id
retires the ruling; incoherent / stale evidence makes the covered keys indeterminate and the build
REFUSES; an expired ruling row is never deleted; expiry replays on rebuild byte-for-byte.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import synth

PARQUET = "synthetic:///export/results.parquet"
LANE_FILE = "synthetic:///export/cap4096_declarations.json"


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _roster(i=1):
    return synth.rec("roster_entry", {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": "fig", "model_id": synth.MODEL_ID,
                                      "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID,
                                      "object_selector_json": {"paper_label": "constitution_lora"}, "strength_rule": "operating_point",
                                      "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}, synth.t(50))


def _exclude_ruling(expires_when, rid="R1", run_id=None, action="exclude", scope=None, **extra):
    p = {"ruling_id": rid, "round_id": "r_synth", "action": action, "scope_kind": "run" if run_id else "key",
         "scope_json": {"run_id": run_id} if run_id else (scope or {}), "reason": "superseded reading", "ruled_utc": synth.t(60),
         "ruled_by": "user", "source_ref": "test", "expires_when_json": expires_when}
    p.update(extra)
    return synth.rec("ruling", p, synth.t(60))


def _state(input_uri, json_path, found, resolves, members, i=70, source_id="src", sha="1" * 64):
    return synth.rec("source_state", {"source_id": source_id, "input_uri": input_uri, "json_path": json_path, "found": found,
                                      "path_resolves": resolves, "current_members_json": members, "observed_utc": synth.t(i),
                                      "input_sha256": sha if found else None}, synth.t(i))


def _setup(tmp_path, rulings, states):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    _write(root, "roster", "000001-roster.jsonl", [_roster()] + list(rulings) + list(states))
    H.ingest_once(root, workers=0, min_free=0)
    return root, ids


def _sel(conn, run_id):
    r = conn.execute("SELECT selection_state, in_current_paper FROM v_paper_current WHERE run_id = ?", (run_id,)).fetchone()
    return r[0], r[1]


def _ruling_row(conn, rid="R1"):
    return dict(conn.execute("SELECT evidence_state, expired_utc FROM ruling WHERE ruling_id = ?", (rid,)).fetchone())


def test_missing_lane_file_keeps_exclusion_active(tmp_path):
    ew = {"kind": "grades_root_absent", "input_uri": LANE_FILE, "json_path": "models.synth.objects.constitution", "root": "grades_old"}
    root, ids = _setup(tmp_path, [_exclude_ruling(ew, run_id="LORA")], [_state(LANE_FILE, "models.synth.objects.constitution", 0, 0, [])])
    lora = ids["run_ids"][1]
    # re-scope the ruling to the real run id
    conn = DB.connect(DB.db_path(root), readonly=True)
    conn.close()
    root2 = str(tmp_path / "reg2")
    H.ensure_layout(root2)
    ids = synth.write_corpus(root2, n_runs=2, n_items=3)
    lora = ids["run_ids"][1]
    _write(root2, "roster", "000001-roster.jsonl", [_roster(), _exclude_ruling(ew, run_id=lora),
                                                    _state(LANE_FILE, "models.synth.objects.constitution", 0, 0, [])])
    H.ingest_once(root2, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root2), readonly=True)
    assert _ruling_row(conn) == {"evidence_state": "missing", "expired_utc": None}
    assert _sel(conn, lora) == ("excluded", 0)
    conn.close()


def _with_ruling(tmp_path, ew, state_records, name="reg", action="exclude", extra_rulings=()):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    lora = ids["run_ids"][1]
    _write(root, "roster", "000001-roster.jsonl", [_roster(), _exclude_ruling(ew, run_id=lora, action=action)] + list(extra_rulings) + list(state_records))
    H.ingest_once(root, workers=0, min_free=0)
    return root, lora


def test_missing_json_path_keeps_exclusion_active(tmp_path):
    ew = {"kind": "grades_root_absent", "input_uri": LANE_FILE, "json_path": "models.synth.objects.constitution", "root": "grades_old"}
    root, lora = _with_ruling(tmp_path, ew, [_state(LANE_FILE, "models.synth.objects.constitution", 1, 0, [])])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn) == {"evidence_state": "path_absent", "expired_utc": None}
    assert _sel(conn, lora) == ("excluded", 0)
    conn.close()


def test_empty_root_set_keeps_exclusion_active(tmp_path):
    ew = {"kind": "grades_root_absent", "input_uri": LANE_FILE, "json_path": "p", "root": "grades_old"}
    root, lora = _with_ruling(tmp_path, ew, [_state(LANE_FILE, "p", 1, 1, [])])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn) == {"evidence_state": "ok", "expired_utc": None}
    assert _sel(conn, lora) == ("excluded", 0)
    conn.close()


def test_mixed_roots_old_and_new_expires(tmp_path):
    ew = {"kind": "grades_root_absent", "input_uri": LANE_FILE, "json_path": "p", "root": "grades_old"}
    root, lora = _with_ruling(tmp_path, ew, [_state(LANE_FILE, "p", 1, 1, ["grades_old", "grades_new"])])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = _ruling_row(conn)
    assert row["evidence_state"] == "ok" and row["expired_utc"] is not None
    assert _sel(conn, lora) == ("selected", 1)
    conn.close()


def test_historical_old_run_plus_current_new_expires(tmp_path):
    """The registry keeps runs under the old root; membership = {new} alone → expired; old run stays."""
    ew = {"kind": "grades_root_absent", "input_uri": LANE_FILE, "json_path": "p", "root": "grades_old"}
    root, lora = _with_ruling(tmp_path, ew, [_state(LANE_FILE, "p", 1, 1, ["grades_new"])])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn)["expired_utc"] is not None
    assert _sel(conn, lora) == ("selected", 1)
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2   # historical rows untouched
    conn.close()


def test_served_id_absent_retires_on_refiling(tmp_path):
    ew = {"kind": "served_id_absent", "source": "src", "input_uri": PARQUET, "member_key": "op_ids", "served_id": "old_op"}
    root, lora = _with_ruling(tmp_path, ew, [_state(PARQUET, "op_ids", 1, 1, ["old_op", "other"])])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn) == {"evidence_state": "ok", "expired_utc": None} and _sel(conn, lora) == ("excluded", 0)
    conn.close()
    # the producer re-files at the ruled point: the current export no longer carries the old id
    _write(root, "roster", "000002-state.jsonl", [_state(PARQUET, "op_ids", 1, 1, ["new_op", "other"], i=80)])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.harvester_records >= 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = _ruling_row(conn)
    assert row["expired_utc"] is not None and _sel(conn, lora) == ("selected", 1)
    conn.close()


def test_served_record_disagreement_is_indeterminate_and_refuses(tmp_path):
    ew = {"kind": "served_record_at_ruled_point", "artifact_uri": PARQUET, "json_path": "served.value", "equals": 0.05}
    root, lora = _with_ruling(tmp_path, ew, [_state(PARQUET, "served.value", 1, 1, [0.1])])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn)["evidence_state"] == "incoherent" and _ruling_row(conn)["expired_utc"] is None
    assert _sel(conn, lora) == ("indeterminate", 0)
    with pytest.raises(paper.RulingIndeterminate) as ei:
        paper.assert_roster_dispositions(conn)
    assert "R1" in str(ei.value) and lora in str(ei.value)
    conn.close()


def test_input_sha_pinned_guard(tmp_path):
    ew = {"kind": "input_sha_pinned", "artifact_uri": PARQUET, "json_path": "$", "sha256_prefix": "abcd"}
    root, lora = _with_ruling(tmp_path, ew, [_state(PARQUET, "$", 1, 1, ["x"], sha="abcd" + "0" * 60)])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn)["evidence_state"] == "ok" and _sel(conn, lora) == ("excluded", 0)
    conn.close()
    _write(root, "roster", "000002-state.jsonl", [_state(PARQUET, "$", 1, 1, ["x"], i=80, sha="ffff" + "0" * 60)])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn)["evidence_state"] == "incoherent" and _sel(conn, lora) == ("indeterminate", 0)
    conn.close()


def test_stale_ruling_is_indeterminate_and_refuses(tmp_path):
    """A supersede ruling whose served object is the operating_point ruling's target is STALE."""
    op = synth.rec("ruling", {"ruling_id": "R_op", "round_id": "r_synth", "action": "operating_point", "scope_kind": "key",
                              "scope_json": {"model_id": synth.MODEL_ID}, "target_json": {"object_id": synth.OBJECT_SHA},
                              "reason": "ruled point", "ruled_utc": synth.t(61), "ruled_by": "user", "source_ref": "test"}, synth.t(61))
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    lora = ids["run_ids"][1]
    sup = _exclude_ruling(None, rid="R_sup", action="supersede", scope={"object_id": synth.OBJECT_SHA, "served_object_sha256": synth.OBJECT_SHA})
    del sup["payload"]["expires_when_json"]
    _write(root, "roster", "000001-roster.jsonl", [_roster(), op, sup])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _ruling_row(conn, "R_sup")["evidence_state"] == "stale"
    assert _sel(conn, lora) == ("indeterminate", 0)
    with pytest.raises(paper.RulingIndeterminate):
        paper.assert_roster_dispositions(conn)
    conn.close()


def test_expired_ruling_row_never_deleted_and_expiry_replays_on_rebuild(tmp_path):
    ew = {"kind": "served_id_absent", "source": "src", "input_uri": PARQUET, "member_key": "op_ids", "served_id": "old_op"}
    root, lora = _with_ruling(tmp_path, ew, [_state(PARQUET, "op_ids", 1, 1, ["new_op"])])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = _ruling_row(conn)
    assert row["expired_utc"] is not None
    assert conn.execute("SELECT COUNT(*) FROM ruling").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM ruling_state").fetchone()[0] == 1
    # a second cycle writes no second evaluation (the state did not flip)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.harvester_records == 0
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    b = DB.connect(out, readonly=True)
    assert dict(b.execute("SELECT evidence_state, expired_utc FROM ruling").fetchone()) == row
    assert b.execute("SELECT evidence_json FROM ruling_state").fetchone()[0] == conn.execute("SELECT evidence_json FROM ruling_state").fetchone()[0]
    diff = H.verify_equal(conn, b)
    assert not {k: v for k, v in diff.items() if v}
    conn.close()
    b.close()


def test_evaluate_expiry_unit_cases():
    conn = DB.connect_memory()
    from sisteer.registry.migrations import apply_migrations
    apply_migrations(conn)
    assert paper.evaluate_expiry(conn, {"ruling_id": "x", "expires_when_json": None}) is None
    ev = paper.evaluate_expiry(conn, {"ruling_id": "x", "expires_when_json": json.dumps({"kind": "bogus"})})
    assert ev.evidence_state == "incoherent"
    ev2 = paper.evaluate_expiry(conn, {"ruling_id": "x", "expires_when_json": {"kind": "served_id_absent", "input_uri": "u", "member_key": "k",
                                                                                "served_id": "s"}})
    assert ev2.evidence_state == "missing" and not ev2.expired
