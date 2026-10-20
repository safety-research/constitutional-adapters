"""The replay-equivalent fold: every case asserts live == replay on every table
including `row_history` and `field_versions_json` (the arrival ledger excluded) and the EXACT
historical prefix state of every history row.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import reducer as RD
from sisteer.registry import synth
from sisteer.registry import vocab as V


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def _ingest(root, **kw):
    return H.ingest_once(root, workers=0, min_free=0, derive_facts=False, **kw)


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    """Rebuild from a differently enumerated copy of the log; assert every table equal."""
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        rel = os.path.relpath(f, os.path.join(root, "records"))
        dest = os.path.join(copy, "records", rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    finally:
        a.close()
        b.close()


def _history(conn, table, key):
    rows = conn.execute("SELECT record_type, folded_state_json FROM row_history WHERE table_name = ? AND natural_key_json = ? "
                        f"ORDER BY {RD.ORDER_SQL}", (table, I.canonical_json(key))).fetchall()
    return [(r[0], json.loads(r[1])) for r in rows]


def _seed_and_run(root, through="running"):
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through=through)
    _write(root, "c", "000001-run.jsonl", recs)
    return rid, recs


def test_correction_supersedes_and_keeps_history(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _ingest(root)
    m = synth.seed_dimensions()[2]  # the instruct model record
    corr = dict(m)
    payload = dict(m["payload"], architecture_note="corrected note")
    del payload["generation_config_json"]          # a field the correction does NOT carry
    corr["payload"] = payload
    corr["written_utc"] = synth.t(500)
    _write(root, "dims", "000002-correction.jsonl", [corr])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT architecture_note, generation_config_json, field_versions_json FROM model WHERE model_id = ?",
                       (synth.MODEL_ID,)).fetchone()
    assert row[0] == "corrected note"
    assert json.loads(row[1]) == {"temperature": 0.7, "top_p": 0.8, "top_k": 20}   # kept from the original record
    fv = json.loads(row[2])
    assert fv["architecture_note"][0] == synth.t(500)
    assert fv["family"][0] == synth.t(500)                    # a full record stamps every column it carries (§1.5)
    assert fv["generation_config_json"][0] == synth.t(2)      # a column it does not carry keeps its ORIGINAL stamp
    hist = _history(conn, "model", {"model_id": synth.MODEL_ID})
    assert len(hist) == 2 and hist[0][1].get("architecture_note") is None and hist[1][1]["architecture_note"] == "corrected note"
    conn.close()
    _rebuild_equal(root, tmp_path)


def test_reversed_disjoint_patches_fold_like_replay(tmp_path):
    """x=0,y=0; patch B@t2 sets y=2 then patch A@t1 sets x=1 → current x=1,y=2; A's history row reads
    x=1,y=0 (prefix excludes B); B's row is REWRITTEN to x=1,y=2."""
    root = _root(tmp_path)
    rid, _ = _seed_and_run(root)
    _ingest(root)
    B = synth.rec("run_patch", {"run_id": rid, "fields": {"gpu_count": 2}}, synth.t(202))
    A = synth.rec("run_patch", {"run_id": rid, "fields": {"node_hostname": "TESTHOST-A"}}, synth.t(201))
    _write(root, "c", "000002-B.jsonl", [B])
    _ingest(root)
    _write(root, "c", "000003-A.jsonl", [A])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT node_hostname, gpu_count, field_versions_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert (row[0], row[1]) == ("TESTHOST-A", 2)
    fv = json.loads(row[2])
    assert fv["node_hostname"] == [synth.t(201), R.record_sha(A)] and fv["gpu_count"] == [synth.t(202), R.record_sha(B)]
    hist = _history(conn, "run", {"run_id": rid})
    states = [s for _, s in hist]
    a_state = states[-2]
    b_state = states[-1]
    assert a_state["node_hostname"] == "TESTHOST-A" and a_state["gpu_count"] == 1      # prefix excludes B (gpu_count from the lifecycle patch)
    assert b_state["node_hostname"] == "TESTHOST-A" and b_state["gpu_count"] == 2      # rewritten
    conn.close()
    _rebuild_equal(root, tmp_path)


def test_late_older_full_record_sets_only_unstamped_fields(tmp_path):
    root = _root(tmp_path)
    rid, recs = _seed_and_run(root)
    _ingest(root)
    # a late FULL run record older than the running state: its lifecycle_state=queued must NOT revert the state
    late = synth.rec("run", dict(synth.run_payload("k", i=100), notes="late full record"), synth.t(100))
    late["written_utc"] = "2026-01-01T00:01:40.500Z"  # between the run record (t100) and running (t101)
    _write(root, "c", "000002-late.jsonl", [late])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT lifecycle_state, notes, node_hostname FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert row[0] == "running" and row[1] == "late full record" and row[2] == "TESTHOST"
    conn.close()
    _rebuild_equal(root, tmp_path)


def test_equal_timestamp_patches_same_field_resolve_by_sha(tmp_path):
    root = _root(tmp_path)
    rid, _ = _seed_and_run(root, through="generated")
    _ingest(root)
    p1 = synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 111}}, synth.t(300))
    p2 = synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 222}}, synth.t(300))
    winner = 111 if R.record_sha(p1) > R.record_sha(p2) else 222
    for order in ((p1, p2), (p2, p1)):
        r = _root(tmp_path, f"reg_{order[0]['payload']['fields']['n_items_generated']}")
        _seed_and_run(r, through="generated")
        _ingest(r)
        _write(r, "c", "000002-first.jsonl", [order[0]])
        _ingest(r)
        _write(r, "c", "000003-second.jsonl", [order[1]])
        _ingest(r)
        conn = DB.connect(DB.db_path(r), readonly=True)
        assert conn.execute("SELECT n_items_generated FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == winner
        hist = _history(conn, "run", {"run_id": rid})
        last_two = [s["n_items_generated"] for _, s in hist[-2:]]
        loser = 333 - winner
        assert last_two == [loser, winner]  # the smaller sha's row carries the prefix state, the larger sha's the winner
        conn.close()
        _rebuild_equal(r, tmp_path, shuffle_seed=order[0]["payload"]["fields"]["n_items_generated"])


def test_history_suffix_recomputed_for_three_out_of_order_events(tmp_path):
    root = _root(tmp_path)
    rid, _ = _seed_and_run(root)
    _ingest(root)
    e3 = synth.rec("run_patch", {"run_id": rid, "fields": {"gpu_type": "H200"}}, synth.t(303))
    e1 = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "j1"}}, synth.t(301))
    e2 = synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "j2", "gpu_count": 8}}, synth.t(302))
    for i, e in enumerate((e3, e1, e2), start=2):
        _write(root, "c", f"00000{i}-e.jsonl", [e])
        _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    hist = _history(conn, "run", {"run_id": rid})
    s1, s2, s3 = [s for _, s in hist[-3:]]
    assert s1["job_name"] == "j1" and "gpu_type" not in s1 and s1["gpu_count"] == 1
    assert s2["job_name"] == "j2" and s2["gpu_count"] == 8 and "gpu_type" not in s2
    assert s3["job_name"] == "j2" and s3["gpu_count"] == 8 and s3["gpu_type"] == "H200"
    row = conn.execute("SELECT job_name, gpu_count, gpu_type FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert tuple(row) == ("j2", 8, "H200")
    conn.close()
    _rebuild_equal(root, tmp_path)


def test_rebuild_from_differently_enumerated_mirror_equal(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=2, n_items=20)
    H.ingest_once(root, workers=0, min_free=0)
    for seed in (1, 2, 3):
        _rebuild_equal(root, tmp_path, shuffle_seed=seed)


def test_late_record_is_head_plus_one_and_moves_nothing(tmp_path):
    """ingest_seq is the dense rank of record FILES in ARRIVAL order: a file older than every ranked one by its
    producer stamp still takes the NEXT rank when it arrives, every existing seq keeps its value (rows follow their
    file), the cycle's dense pass has no work, and a rebuild from the log reproduces the order from the arrival facts."""
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    for i, d in enumerate(dims):
        _write(root, "dims", f"{i + 1:06d}-dim.jsonl", [d])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    before = {r[0]: r[1] for r in conn.execute("SELECT model_id, ingest_seq FROM model")}
    n = conn.execute("SELECT COUNT(*) FROM record_ingest").fetchone()[0]
    assert n == len(dims) + 1                                   # the dimension files + the cycle's one file_arrival facts file
    conn.close()
    late = synth.rec("si_text", {"si_id": "older_si", "si_text_sha256": "f" * 64, "si_kind": "irrelevant_control",
                                 "source_uri": "synthetic://x"}, "2025-12-31T00:00:00.000Z")
    _write(root, "late", "000001-late.jsonl", [late])
    rep = _ingest(root)
    assert rep.reranks == 0 and rep.rerank_rows == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    after = {r[0]: r[1] for r in conn.execute("SELECT model_id, ingest_seq FROM model")}
    assert after == before
    assert conn.execute("SELECT ingest_seq FROM si_text WHERE si_id = 'older_si'").fetchone()[0] == n + 1
    seqs = sorted(r[0] for r in conn.execute("SELECT ingest_seq FROM record_ingest"))
    assert seqs == list(range(1, n + 3))                        # + the late file + its cycle's facts file
    conn.close()
    _rebuild_equal(root, tmp_path)


def test_item_rows_fold_whole_row_greatest_order_key_wins_without_history(tmp_path):
    root = _root(tmp_path)
    rid, recs = _seed_and_run(root, through="judged")
    _ingest(root)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    base = synth.item_records(rid, jid, 400, 1)[0]
    v1 = dict(base); v1["payload"] = dict(base["payload"], value=0.1); v1["written_utc"] = synth.t(401)
    v2 = dict(base); v2["payload"] = dict(base["payload"], value=0.2); v2["written_utc"] = synth.t(402)
    v3 = dict(base); v3["payload"] = dict(base["payload"], value=0.3); v3["written_utc"] = synth.t(403)
    for i, v in enumerate((v2, v3, v1), start=2):   # out of order
        _write(root, "c", f"00000{i}-item.jsonl.gz", [v], gz=True)
        _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT value, written_utc FROM item_result WHERE run_id = ?", (rid,)).fetchone()
    assert row[0] == 0.3 and row[1] == R.utc_to_epoch_ms(synth.t(403))
    assert conn.execute("SELECT COUNT(*) FROM row_history WHERE table_name = 'item_result'").fetchone()[0] == 0
    cols = [r[1] for r in conn.execute("PRAGMA table_info(item_result)")]
    assert "field_versions_json" not in cols and "written_utc" in cols and "record_sha256" in cols
    conn.close()
    _rebuild_equal(root, tmp_path)


def test_registry_history_folds_item_key_from_log_equals_live_row(tmp_path):
    root = _root(tmp_path)
    rid, recs = _seed_and_run(root, through="judged")
    _ingest(root)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    base = synth.item_records(rid, jid, 400, 1)[0]
    vals = []
    for i, val in enumerate((0.5, 0.7, 0.9), start=2):
        v = dict(base); v["payload"] = dict(base["payload"], value=val); v["written_utc"] = synth.t(400 + i)
        vals.append(v)
        _write(root, "c", f"00000{i}-item.jsonl.gz", [v], gz=True)
    # an unrelated run's item file must NOT be read
    rid2, recs2 = synth.run_records("other", 900, through="judged")
    _write(root, "o", "000001-run.jsonl", recs2)
    _write(root, "o", "000002-item.jsonl.gz", synth.item_records(rid2, I.judgment_id(rid2, synth.JUDGE_ID), 950, 1), gz=True)
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    key = {k: base["payload"][k] for k in R.NATURAL_KEYS["item_result"]}
    states = H.history(root, conn, "item_result", key)
    assert [s["value"] for s in states] == [0.5, 0.7, 0.9]
    assert all("/c/" in s["_record_uri"] for s in states)
    live = conn.execute("SELECT value FROM item_result WHERE run_id = ? AND item_id = ?", (rid, key["item_id"])).fetchone()[0]
    assert states[-1]["value"] == live
    files = [r[0] for r in conn.execute("SELECT record_uri FROM record_run_index WHERE run_id = ?", (rid,))]
    assert files and all("/o/" not in f for f in files)
    conn.close()


def test_run_scale_history_for_run_key_via_cli_helper(tmp_path):
    root = _root(tmp_path)
    rid, _ = _seed_and_run(root, through="judged")
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    states = H.history(root, conn, "run", {"run_id": rid})
    seq = [s["state"]["lifecycle_state"] for s in states]
    assert seq[0] == "queued" and seq[-1] == "judged" and "running" in seq and "generated" in seq
    conn.close()


def test_judgment_claim_moves_generated_run_to_judging_and_abandon_reverts(tmp_path):
    root = _root(tmp_path)
    rid, _ = _seed_and_run(root, through="generated")
    _ingest(root)
    jid = I.judgment_id(rid, synth.JUDGE_ID_2, "sess", synth.t(500))
    claim = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID_2, "status": "in_progress",
                                   "claimed_by_session": "sess", "claimed_utc": synth.t(500)}, synth.t(500))
    _write(root, "c", "000002-claim.jsonl", [claim])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "judging"
    conn.close()
    ab = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID_2, "status": "abandoned",
                                "claimed_by_session": "sess", "claimed_utc": synth.t(500)}, synth.t(600))
    _write(root, "c", "000003-abandon.jsonl", [ab])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "generated"
    assert conn.execute("SELECT status FROM judgment").fetchone()[0] == "abandoned"
    conn.close()
    _rebuild_equal(root, tmp_path)


def test_ruling_state_folds_latest_evidence_and_earliest_expiry(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    ruling = synth.rec("ruling", {"ruling_id": "R1", "round_id": "r_synth", "action": "exclude", "scope_kind": "run",
                                  "scope_json": {"run_id": "x"}, "reason": "synthetic", "ruled_utc": synth.t(1), "ruled_by": "user",
                                  "source_ref": "test"}, synth.t(1))
    _write(root, "rul", "000001-ruling.jsonl", [ruling])
    evs = [
        synth.rec("ruling_state", {"ruling_id": "R1", "evaluated_utc": synth.t(30), "evidence_state": "ok", "expired": 1}, synth.t(30)),
        synth.rec("ruling_state", {"ruling_id": "R1", "evaluated_utc": synth.t(10), "evidence_state": "missing", "expired": 0}, synth.t(10)),
        synth.rec("ruling_state", {"ruling_id": "R1", "evaluated_utc": synth.t(20), "evidence_state": "ok", "expired": 1}, synth.t(20)),
    ]
    for i, e in enumerate(evs, start=2):
        _write(root, "rul", f"00000{i}-state.jsonl", [e])
        _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT evidence_state, expired_utc, evidence_evaluated_utc FROM ruling WHERE ruling_id = 'R1'").fetchone()
    assert tuple(row) == ("ok", synth.t(20), synth.t(30))
    assert conn.execute("SELECT COUNT(*) FROM ruling_state").fetchone()[0] == 3
    conn.close()
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# run_ingest may carry judged_utc (a rule-scored run's scoring time); the fold sets it only for an ingested stamp
# --------------------------------------------------------------------------------------------


def test_run_ingest_record_shape_with_and_without_judged_utc():
    from sisteer.registry import validate as VAL
    from sisteer.registry.reducer import HistoryEvent, effects_of, fold
    prod = synth.producer()
    plain = R.make_record("run_ingest", {"run_id": "r1", "ingested_utc": synth.t(20)}, prod, "id-plain", synth.t(20))
    v = VAL.validate_record(plain)
    assert isinstance(v, VAL.Validated) and set(v.payload) == {"run_id", "ingested_utc"}
    scored = R.make_record("run_ingest", {"run_id": "r1", "ingested_utc": synth.t(20), "judged_utc": synth.t(5)}, prod, "id-scored", synth.t(20))
    v2 = VAL.validate_record(scored)
    assert isinstance(v2, VAL.Validated) and v2.payload["judged_utc"] == synth.t(5)
    bad = R.make_record("run_ingest", {"run_id": "r1", "ingested_utc": synth.t(20), "judged_utc": "yesterday"}, prod, "id-bad", synth.t(20))
    assert isinstance(VAL.validate_record(bad), VAL.Malformed)
    # any other key is dropped (strict keys), never parked
    extra = R.make_record("run_ingest", {"run_id": "r1", "ingested_utc": synth.t(20), "note": "x"}, prod, "id-extra", synth.t(20))
    v3 = VAL.validate_record(extra)
    assert isinstance(v3, VAL.Validated) and set(v3.payload) == {"run_id", "ingested_utc"} and v3.extra == {}
    eff = effects_of(v2)
    assert len(eff) == 1 and eff[0].payload == {"state": "ingested", "event_utc": synth.t(20), "judged_utc": synth.t(5)}
    assert effects_of(v)[0].payload == {"state": "ingested", "event_utc": synth.t(20)}
    # the fold: a rule-scored run goes generated → ingested and takes judged_utc from the stamp; a run that already has
    # a judged_utc keeps its own
    base = [HistoryEvent(synth.t(0), "a" * 64, "run", "c/run", {"_kind": "full", "run_id": "r1", "lifecycle_state": "queued", "queued_utc": synth.t(0)}),
            HistoryEvent(synth.t(1), "b" * 64, "run_state", "c/run", {"_kind": "state", "state": "running", "event_utc": synth.t(1)}),
            HistoryEvent(synth.t(3), "c" * 64, "run_state", "c/run", {"_kind": "state", "state": "generated", "event_utc": synth.t(3)})]
    stamp = HistoryEvent(synth.t(20), "d" * 64, "run_ingest", "harvester/x", {"_kind": "state", "state": "ingested", "event_utc": synth.t(20), "judged_utc": synth.t(5)})
    rows = fold("run", base + [stamp])
    last = rows[-1][0]
    assert last["lifecycle_state"] == "ingested" and last["judged_utc"] == synth.t(5) and last["ingested_utc"] == synth.t(20)
    assert rows[-1][1]["judged_utc"] == [synth.t(20), "d" * 64]
    judged = HistoryEvent(synth.t(7), "e" * 64, "run_state", "c/run", {"_kind": "state", "state": "judged", "event_utc": synth.t(7)})
    rows = fold("run", base + [judged, stamp])
    assert rows[-1][0]["judged_utc"] == synth.t(7) and rows[-1][0]["lifecycle_state"] == "ingested"


# --------------------------------------------------------------------------------------------
# the lifecycle is monotonic in the fold: a state event never moves a run backwards along the progression
# --------------------------------------------------------------------------------------------


def _lifecycle_base():
    from sisteer.registry.reducer import HistoryEvent
    return [HistoryEvent(synth.t(0), "a" * 64, "run", "c/run", {"_kind": "full", "run_id": "r1", "lifecycle_state": "queued", "queued_utc": synth.t(0)}),
            HistoryEvent(synth.t(1), "b" * 64, "run_state", "c/run", {"_kind": "state", "state": "running", "event_utc": synth.t(1)}),
            HistoryEvent(synth.t(3), "c" * 64, "run_state", "c/run", {"_kind": "state", "state": "generated", "event_utc": synth.t(3)})]


def _state(i: int, sha: str, state: str, **extra):
    from sisteer.registry.reducer import HistoryEvent
    return HistoryEvent(synth.t(i), sha * 64, "run_ingest" if state == "ingested" else "run_state",
                        "harvester/x" if state == "ingested" else "c/run", {"_kind": "state", "state": state, "event_utc": synth.t(i), **extra})


def test_fold_holds_a_state_event_that_would_move_the_lifecycle_backwards():
    """A reader re-emitting its rows at a new version re-sends the run's `judged`; ordered by written_utc it folds AFTER
    the harvester's `ingested` stamp. The fold records it (its history row stands) and moves nothing: lifecycle_state,
    its stamp and judged_utc are the earlier records'."""
    from sisteer.registry.reducer import fold, lifecycle_regresses, lifecycle_regressions_held
    base = _lifecycle_base()
    judged, stamp, again = _state(7, "e", "judged"), _state(20, "d", "ingested"), _state(30, "f", "judged")
    events = base + [judged, stamp, again]
    rows = fold("run", events)
    last, fv = rows[-1]
    assert last["lifecycle_state"] == "ingested" and last["ingested_utc"] == synth.t(20)
    assert last["judged_utc"] == synth.t(7), "the held event's event_utc must not overwrite judged_utc"
    assert fv["lifecycle_state"] == [synth.t(20), "d" * 64] and fv["judged_utc"] == [synth.t(7), "e" * 64]
    assert rows[-1][0] == rows[-2][0], "the held event's snapshot is the previous one: it moved nothing"
    assert lifecycle_regressions_held(events, rows) == 1
    assert lifecycle_regressions_held(events, rows, start=len(events) - 1) == 1
    assert lifecycle_regressions_held(events, rows, start=0) == lifecycle_regressions_held(events[:-1], rows[:-1]) + 1
    # a rule-scored run: `generated` re-sent after the stamp that took it generated → ingested
    scored = _state(20, "d", "ingested", judged_utc=synth.t(5))
    events = base + [scored, _state(30, "f", "generated")]
    rows = fold("run", events)
    last = rows[-1][0]
    assert last["lifecycle_state"] == "ingested" and last["generated_utc"] == synth.t(3) and last["judged_utc"] == synth.t(5)
    assert lifecycle_regressions_held(events, rows) == 1
    # every step of the progression is guarded, not only `ingested`: `running` re-sent after `generated` is held too
    events = base + [_state(30, "f", "running")]
    rows = fold("run", events)
    assert rows[-1][0]["lifecycle_state"] == "generated" and rows[-1][0]["started_utc"] == synth.t(1)
    assert lifecycle_regressions_held(events, rows) == 1
    assert lifecycle_regresses("ingested", "judged") and lifecycle_regresses("judged", "generated") and lifecycle_regresses("generated", "queued")
    assert not lifecycle_regresses("generated", "judged") and not lifecycle_regresses("judged", "judged") and not lifecycle_regresses(None, "queued")


def test_fold_still_advances_forward_and_to_terminal_states():
    from sisteer.registry.reducer import fold, lifecycle_regresses, lifecycle_regressions_held
    base = _lifecycle_base()
    rows = fold("run", base + [_state(7, "e", "judged")])
    assert rows[-1][0]["lifecycle_state"] == "judged" and rows[-1][0]["judged_utc"] == synth.t(7)
    # a terminal / lateral state after `ingested` applies: superseded and withdrawn carry no rank
    for terminal in ("superseded", "withdrawn"):
        events = base + [_state(7, "e", "judged"), _state(20, "d", "ingested"), _state(40, "f", terminal)]
        rows = fold("run", events)
        assert rows[-1][0]["lifecycle_state"] == terminal and rows[-1][0]["ingested_utc"] == synth.t(20)
        assert lifecycle_regressions_held(events, rows) == 0
        assert not lifecycle_regresses("ingested", terminal)
    events = base + [_state(9, "e", "failed")]
    rows = fold("run", events)
    assert rows[-1][0]["lifecycle_state"] == "failed" and rows[-1][0]["finished_utc"] == synth.t(9)
    assert lifecycle_regressions_held(events, rows) == 0
    # from a terminal state any state event applies (no rank to compare against)
    assert not lifecycle_regresses("failed", "generated") and not lifecycle_regresses("superseded", "judged")


def test_fold_treats_judging_as_a_transient_beside_generated():
    from sisteer.registry.reducer import fold, lifecycle_regressions_held
    base = _lifecycle_base()
    events = base + [_state(5, "e", "judging"), _state(6, "f", "generated"), _state(7, "g", "judged"), _state(8, "h", "judging")]
    rows = fold("run", events)
    assert [r[0]["lifecycle_state"] for r in rows[len(base):]] == ["judging", "generated", "judged", "judged"]
    assert lifecycle_regressions_held(events, rows) == 1, "only `judging` after `judged` is backwards"


def _claim_event(i: int, sha: str, jid: str, status: str):
    """A judgment record's second effect on the run key (`effects_of`: kind judgment_state, the judgment id and its status)."""
    from sisteer.registry.reducer import HistoryEvent
    return HistoryEvent(synth.t(i), sha * 64, "judgment", "c/judgment", {"_kind": "judgment_state", "judgment_id": jid, "status": status})


@pytest.mark.parametrize("ending", ["abandoned", "superseded"])
def test_fold_moves_a_judging_run_back_only_when_its_last_open_claim_ends(ending):
    """Two judges claim a generated run; the first judgment ends -- abandoned, or complete and then superseded -- while the second
    judge's claim is still in progress: the run stays `judging` (its stamp still the claim that moved it there) and reads `generated`
    again only when the second claim ends. The open claims are a function of the run key's ordered events alone (the unconditional
    move back read a run another judge was still judging as `generated`)."""
    from sisteer.registry.reducer import fold
    base = _lifecycle_base()
    a, b = "judgment-a", "judgment-b"
    claims = [_claim_event(5, "e", a, "in_progress"), _claim_event(6, "f", b, "in_progress")]
    if ending == "superseded":
        claims.append(_claim_event(7, "g", a, "complete"))
    events = base + claims + [_claim_event(8, "h", a, ending)]
    rows = fold("run", events)
    assert [r[0]["lifecycle_state"] for r in rows[len(base):]] == ["judging"] * (len(events) - len(base))
    assert rows[-1][1]["lifecycle_state"] == [synth.t(5), "e" * 64], "the stamp is the claim that moved the run to judging"
    assert rows[-1][0] == rows[-2][0], "ending one judgment while another claim is open moves nothing"
    # the second claim ends: the run reads generated again (nothing has judged it), stamped by that event
    done = fold("run", events + [_claim_event(9, "i", b, "abandoned")])
    assert done[-1][0]["lifecycle_state"] == "generated" and done[-1][1]["lifecycle_state"] == [synth.t(9), "i" * 64]
    # the single-claim case is unchanged: the one claim ending moves the run back at once
    single = fold("run", base + [_claim_event(5, "e", a, "in_progress"), _claim_event(8, "h", a, ending)])
    assert [r[0]["lifecycle_state"] for r in single[len(base):]] == ["judging", "generated"]
    # a claim re-filed in_progress stays one open claim; a judged_utc already on the row brings the run back as judged
    from sisteer.registry.reducer import HistoryEvent
    patched = HistoryEvent(synth.t(6), "f" * 64, "run_patch", "c/run", {"_kind": "patch", "judged_utc": synth.t(6)})
    again = fold("run", base + [_claim_event(5, "e", a, "in_progress"), patched, _claim_event(7, "g", a, "in_progress"),
                                _claim_event(8, "h", a, ending)])
    assert [r[0]["lifecycle_state"] for r in again[len(base):]] == ["judging", "judging", "judging", "judged"]
# --------------------------------------------------------------------------------------------
# the same-stamp tie-break: at one written_utc a run_state to a lateral/terminal state
# folds AFTER a progression-state event -- order key (written_utc, tie_rank, record_sha256), the same in SQL and in Python
# --------------------------------------------------------------------------------------------


def _adversarial_pair(rid: str, stamp: str):
    """A `judged` and a `superseded` run_state at ONE written_utc whose shas would have put `judged` LAST under the pre-tie-break
    `(written_utc, record_sha256)` order -- the fast-windows pair (declared_common emits both in one batch with one stamp), chosen
    so the test discriminates the orders instead of riding the sha draw."""
    judged = synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": stamp, "evidence_kind": "completion_artifact"}, stamp)
    for n in range(256):
        sup = synth.rec("run_state", {"run_id": rid, "state": "superseded", "event_utc": stamp, "evidence_kind": "ruling",
                                      "note": f"superseded by a later run of the same key ({n})"}, stamp)
        if R.record_sha(sup) < R.record_sha(judged):
            return judged, sup
    raise AssertionError("no adversarial note in 256 tries")


def _judgment(rid: str, stamp: str, n: int = 200):
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    return jid, synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": stamp, "finished_utc": stamp,
                                       "n_items_judged": n, "status": "complete", "api_pool": "lowprio"}, stamp)


def test_same_stamp_judged_and_superseded_fold_to_superseded_whatever_the_arrival_order(tmp_path):
    """Fresh emission: the pair in one file, judged's file first, superseded's file first -- the run ends `superseded` with
    `judged_utc` set every time, the history reads judged THEN superseded (the fold's order, not the shas'), and a shuffled
    rebuild of the log equals the live rows."""
    stamp = synth.t(300)
    for seed, arrangement in enumerate(("one_file", "judged_first", "superseded_first"), start=21):
        root = _root(tmp_path, arrangement)
        rid, _ = _seed_and_run(root, through="generated")
        _ingest(root)
        judged, sup = _adversarial_pair(rid, stamp)
        _, judgment = _judgment(rid, stamp)
        if arrangement == "one_file":
            _write(root, "c", "000002-pair.jsonl", [judgment, sup, judged])          # the lateral event FIRST in the file
            _ingest(root)
        elif arrangement == "judged_first":
            _write(root, "c", "000002-judged.jsonl", [judgment, judged])
            _ingest(root)
            _write(root, "c", "000003-superseded.jsonl", [sup])
            _ingest(root)
        else:
            _write(root, "c", "000002-superseded.jsonl", [sup])
            _ingest(root)
            assert _q1(root, "SELECT lifecycle_state, judged_utc FROM run WHERE run_id = ?", rid) == ("superseded", None)
            _write(root, "c", "000003-judged.jsonl", [judgment, judged])
            _ingest(root)
        conn = DB.connect(DB.db_path(root), readonly=True)
        try:
            row = conn.execute("SELECT lifecycle_state, judged_utc, field_versions_json FROM run WHERE run_id = ?", (rid,)).fetchone()
            assert (row[0], row[1]) == ("superseded", stamp), (arrangement, row[0], row[1])
            fv = json.loads(row[2])
            assert fv["lifecycle_state"] == [stamp, R.record_sha(sup)] and fv["judged_utc"] == [stamp, R.record_sha(judged)]
            hist = H.history(root, conn, "run", {"run_id": rid})
            at_stamp = [h["payload"].get("state") for h in hist if h["written_utc"] == stamp and h["record_type"] == "run_state"]
            assert at_stamp == ["judged", "superseded"], (arrangement, at_stamp)      # the judgment's own run event rides the stamp too
            assert [h["state"]["lifecycle_state"] for h in hist][-2:] == ["judged", "superseded"]
            assert [h["state"].get("judged_utc") for h in hist][-2:] == [stamp, stamp], "the lateral event folds over the judged one, keeping its utc"
            n_events = conn.execute("SELECT COUNT(*) FROM row_history WHERE table_name = 'run_state_event' AND natural_key_json LIKE ?",
                                    (f"%{rid}%",)).fetchone()[0]
            assert n_events >= 5                                                       # queued, running, generated, judged, superseded
        finally:
            conn.close()
        _rebuild_equal(root, tmp_path, shuffle_seed=seed)


def _q1(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        row = conn.execute(sql, params).fetchone()
        return tuple(row) if row is not None else None          # the connection's row factory is sqlite3.Row
    finally:
        conn.close()


def test_order_key_in_python_is_the_sql_order_and_the_rank_reads_the_record_bytes(tmp_path):
    """One order, two spellings: `sorted(events, key=order_key)` equals what `_load_events` reads under ORDER_SQL on a history dense
    with same-stamp events of every kind, ORDER_DESC_SQL is its exact reverse, and the two differ from the pre-tie-break sha order
    somewhere (the rank matters). The rank itself: 1 only for a run_state to a lateral/terminal state."""
    assert RD.LATERAL_STATES == ("failed", "cancelled", "superseded", "withdrawn")
    assert set(RD.LATERAL_STATES) | set(V.LIFECYCLE_PROGRESSION) | {"judging"} == set(V.LIFECYCLE_STATES)
    for st in V.LIFECYCLE_PROGRESSION + ("judging",):
        assert RD.tie_rank("run_state", {"state": st}) == 0, st
    for st in RD.LATERAL_STATES:
        assert RD.tie_rank("run_state", {"state": st}) == 1, st
    assert RD.tie_rank("run_ingest", {"state": "ingested", "event_utc": synth.t(1)}) == 0
    assert RD.tie_rank("run_patch", {"state": "superseded"}) == 0 and RD.tie_rank("run", {"lifecycle_state": "superseded"}) == 0
    assert RD.tie_rank("judgment", {"status": "superseded"}) == 0 and RD.tie_rank("run_state", {}) == 0
    # the pure fold over a Python-ordered pair: judged then superseded, however the shas fall
    judged = RD.HistoryEvent(synth.t(7), "f" * 64, "run_state", "c/run", {"_kind": "state", "state": "judged", "event_utc": synth.t(7)})
    sup = RD.HistoryEvent(synth.t(7), "a" * 64, "run_state", "c/run", {"_kind": "state", "state": "superseded", "event_utc": synth.t(7)})
    assert RD.order_key(judged) == (synth.t(7), 0, "f" * 64) and RD.order_key(sup) == (synth.t(7), 1, "a" * 64)
    events = sorted(_lifecycle_base() + [sup, judged], key=RD.order_key)
    assert [e.sha[0] for e in events[-2:]] == ["f", "a"]
    last, fv = RD.fold("run", events)[-1]
    assert last["lifecycle_state"] == "superseded" and last["judged_utc"] == synth.t(7) and fv["lifecycle_state"] == [synth.t(7), "a" * 64]
    # the SQL order on a dense synthetic history
    root = _root(tmp_path)
    _seed_and_run(root, through="running")
    _ingest(root)
    key = I.canonical_json({"run_id": "r-order"})
    rng = random.Random(5)
    conn = DB.connect(DB.db_path(root))
    try:
        for i in range(400):
            stamp = synth.t(rng.randrange(40))
            rt = rng.choice(("run_state", "run_state", "run_state", "run_ingest", "run_patch", "run"))
            if rt == "run_state":
                payload = {"_kind": "state", "state": rng.choice(V.LIFECYCLE_STATES), "event_utc": stamp}
            elif rt == "run_ingest":
                payload = {"_kind": "state", "state": "ingested", "event_utc": stamp}
            elif rt == "run_patch":
                payload = {"_kind": "patch", "n_items_judged": i}
            else:
                payload = {"_kind": "full", "run_id": "r-order", "lifecycle_state": rng.choice(V.LIFECYCLE_STATES)}
            conn.execute("INSERT INTO row_history(table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, "
                         "payload_json, folded_state_json) VALUES (?,?,?,?,?,?,?,?)",
                         ("run", key, stamp, R.record_sha({"i": i}), rt, "synthetic/u", I.canonical_json(payload), "{}"))
        conn.commit()
        events = RD.Reducer(conn)._load_events("run", key)
        assert len(events) == 400
        slots = [(e.written_utc, e.sha) for e in events]
        assert slots == [(e.written_utc, e.sha) for e in sorted(events, key=RD.order_key)]
        assert slots != [(e.written_utc, e.sha) for e in sorted(events, key=lambda e: (e.written_utc, e.sha))], "the rank changed the order somewhere"
        desc = conn.execute("SELECT written_utc, record_sha256 FROM row_history WHERE table_name = 'run' AND natural_key_json = ? "
                            f"ORDER BY {RD.ORDER_DESC_SQL}", (key,)).fetchall()
        assert [(r[0], r[1]) for r in desc] == slots[::-1]
        top2 = conn.execute("SELECT written_utc, record_sha256 FROM row_history WHERE table_name = 'run' AND natural_key_json = ? "
                            f"ORDER BY {RD.ORDER_DESC_SQL} LIMIT 2", (key,)).fetchall()
        assert [(r[0], r[1]) for r in top2] == slots[-1:-3:-1]
        # at a stamp holding both kinds, every rank-0 event precedes every rank-1 event
        by_stamp = {}
        for e in events:
            by_stamp.setdefault(e.written_utc, []).append(e.tie_rank)
        mixed = [ranks for ranks in by_stamp.values() if 0 in ranks and 1 in ranks]
        assert mixed and all(ranks == sorted(ranks) for ranks in mixed)
    finally:
        conn.close()


def test_other_events_in_the_same_stamp_batch_fold_the_same_beside_the_pair(tmp_path):
    """A run whose one-stamp batch carries a judgment, a run_patch and a run_metric beside the judged/superseded pair: every column
    the other events set folds exactly as it does without the superseded event; only `lifecycle_state` (and its stamp) differ, and
    the metric row is byte-identical."""
    stamp = synth.t(300)
    folded = {}
    for seed, (name, with_sup) in enumerate((("pair", True), ("judged_only", False)), start=31):
        root = _root(tmp_path, name)
        rid, _ = _seed_and_run(root, through="generated")
        _ingest(root)
        judged, sup = _adversarial_pair(rid, stamp)
        jid, judgment = _judgment(rid, stamp, n=150)
        patch = synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": 150, "judge_id": synth.JUDGE_ID, "notes": "same-stamp batch"}}, stamp)
        metric = synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.4, "k": 60,
                                          "n": 150, "computed_by": "producer"}, stamp)
        batch = [judgment, patch, metric, judged] + ([sup] if with_sup else [])
        _write(root, "c", "000002-batch.jsonl", batch)
        _ingest(root)
        conn = DB.connect(DB.db_path(root), readonly=True)
        try:
            conn.row_factory = sqlite3.Row
            run = dict(conn.execute("SELECT * FROM run WHERE run_id = ?", (rid,)).fetchone())
            metrics = [tuple(r) for r in conn.execute("SELECT run_id, judgment_id, metric_id, threshold, value, k, n, field_versions_json "
                                                      "FROM run_metric WHERE run_id = ? ORDER BY metric_id", (rid,)).fetchall()]
        finally:
            conn.close()
        folded[name] = (run, metrics)
        _rebuild_equal(root, tmp_path, shuffle_seed=seed)
    a, b = folded["pair"][0], folded["judged_only"][0]
    assert a["lifecycle_state"] == "superseded" and b["lifecycle_state"] == "judged"
    assert a["n_items_judged"] == b["n_items_judged"] == 150 and a["judge_id"] == b["judge_id"] == synth.JUDGE_ID
    assert a["notes"] == b["notes"] == "same-stamp batch" and a["judged_utc"] == b["judged_utc"] == stamp
    differ = {k for k in a if a[k] != b[k]}
    assert differ - {"record_uri"} == {"lifecycle_state", "field_versions_json"}, differ      # record_uri names the key's LAST event
    fva, fvb = json.loads(a["field_versions_json"]), json.loads(b["field_versions_json"])
    assert {k: v for k, v in fva.items() if k != "lifecycle_state"} == {k: v for k, v in fvb.items() if k != "lifecycle_state"}
    assert folded["pair"][1] == folded["judged_only"][1] and len(folded["pair"][1]) == 1
