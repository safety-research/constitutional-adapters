"""The raw readers of the item tables read VISIBLE rows only, and the status page shows reclaim. On synthetic corpora:

1. ``registry degeneracy-rebuild`` (``degeneracy_artifact.registry_rows``): a run whose union5 reading is carried by a re-read source
   (the superseded revision and its head), a pending revision of that source and a second source's head counts ONE row, the current
   one -- the raw read summed every provenance the table holds, a re-read or a second source counting the run twice or more.
2. ``emit_contrasts._filed_item_metrics``: a steered run whose only item rows ride a pending revision, or a superseded one, is
   ``no_items`` (the raw read named the metric filed, and the pair went ``due`` for items that pair to nothing); a pass reads each
   steered run's filed metrics once, from a memo of its own that hands every caller a copy.
3-5. ``refusals.resolve``: the resolution reads ``v_<table>_current`` at the measurement key; a refusal naming a revision carries the
   revision's block; an unaccepted revision's own row at the full key is the separately labelled ``staged_row``; ``revision_reclaimed``
   when no current row stands at the key and reclaim deleted the revision's rows in the table -- on direct rows, and end to end
   through the fold's own reclaim;
6-7. the status page: the reclaim line (revisions due, the ledger per table, the last cycle's deletions, rehydrations and stop, read
   from the chain, the ledger and the metrics log), a re-activation held on rehydration listed by name and counted in a tile, a
   mode file the harvester refuses shown red, and a line whose three sections did not all run live this render never green: unknown
   with no cached rows (its counts not computed, never 0), stale with the last good rows' stamp and age, red over a hold in them.
"""
from __future__ import annotations

import json
import os

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import degeneracy_artifact as DA
from sisteer.registry import harvest as H
from sisteer.registry import hooks as RH
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import refusals as RF
from sisteer.registry import status
from sisteer.registry import synth

from test_native_judging_cardinality import EC, _synth_root  # noqa: E402  -- the emitter module and its synthetic world
from test_registry_status_ops import _render  # noqa: E402
from test_registry_core_fixes import _root  # noqa: E402

LEDGER = "item_reclaim_ledger"


def _ingest(root, refused=0):
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == refused and rep.blocked == 0, rep.as_dict()
    return rep


def _revision(rid, source, i, accepted=1):
    """A ``source_revision`` record in a plain file: its rows below ride it, tagged with ``rid``."""
    return synth.rec("source_revision", {"revision_id": rid, "source_id": source, "adapter": "seedrep", "observed_utc": synth.t(i),
                                         "inputs_json": {"a": "%064x" % i}, "accepted": accepted, "priority": 0}, synth.t(i))


def _union5(run_id, judgment_id, rev, num, den, i, by_reason):
    row = RH.measured_row(run_id, "union5", "all", num, den, "replies", judgment_id=judgment_id, computed_utc=synth.t(i),
                          source_uri=f"stores/{rev}/union.json ›cells[x]")
    row.update(source_revision_id=rev, extra={"by_reason": by_reason})
    return synth.rec("run_degeneration", row, synth.t(i))


def _ro(root):
    return DB.connect(DB.db_path(root), readonly=True)


# --------------------------------------------------------------------------------------------
# 1. degeneracy-rebuild reads the current row per measurement key
# --------------------------------------------------------------------------------------------


def test_degeneracy_rebuild_counts_the_current_union5_row_once_across_a_reread_a_pending_revision_and_a_second_source(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=3, with_items=False)
    _ingest(root)
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    src = os.path.join(root, "records", "synth_sources")
    R.write_records(os.path.join(src, "000001-revisions.jsonl"),
                    [_revision("revA1", "seedrep:a", 9000), _revision("revA2", "seedrep:a", 9100), _revision("revAP", "seedrep:a", 9200, accepted=0),
                     _revision("revB1", "seedrep:b", 9050)])
    _ingest(root)                                                     # the revisions fold before any row cites them
    d = os.path.join(root, "records", synth.CAMPAIGN, I.run_dir_name(rid))
    # four provenances of ONE measurement key (run, judgment, union5, all), told apart by their numbers: the re-read's old and new
    # revision, the source's pending read, and the second source's head
    rows = {"revA1": (40, 400, {"classify": 30, "at_cap": 10, "zlib+at_cap": 10}), "revA2": (12, 200, {"classify": 9, "at_cap": 5, "rep_loop": 3}),
            "revAP": (77, 700, {"classify": 77}), "revB1": (8, 100, {"contentless": 8, "at_cap": 2})}
    for n, (rev, (num, den, by_reason)) in enumerate(sorted(rows.items())):
        R.write_records(os.path.join(d, R.record_file_name(60 + n, "run_degeneration")), [_union5(rid, jid, rev, num, den, 9300 + n, by_reason)])
    _ingest(root)
    conn = _ro(root)
    sup = conn.execute("SELECT superseded_by_revision_id FROM source_revision WHERE revision_id = 'revA1'").fetchone()[0]
    assert sup == "revA2"
    raw = conn.execute("SELECT source_revision_id, numerator, denominator, extra_json FROM run_degeneration WHERE run_id = ? AND measure = 'union5' "
                       "AND scope = 'all' ORDER BY source_revision_id", (rid,)).fetchall()
    # the pending read and both heads stay whatever reclaim did to the superseded revision's copy: the raw read still multiplies
    assert {r[0] for r in raw} >= {"revA2", "revAP", "revB1"}, raw
    cur = conn.execute("SELECT source_revision_id, numerator, denominator, extra_json FROM v_run_degeneration_current WHERE run_id = ? "
                       "AND measure = 'union5' AND scope = 'all'", (rid,)).fetchall()
    assert len(cur) == 1 and cur[0][0] in ("revA2", "revB1"), cur
    out = DA.registry_rows(conn)
    conn.close()
    assert len(out) == 1, out
    want_num = DA.union_excluding_cap(json.loads(cur[0][3])["by_reason"])
    assert (out[0]["numerator"], out[0]["denominator"]) == (want_num, cur[0][2])
    assert out[0]["degen_rate_pct"] == round(100.0 * want_num / cur[0][2], 3)
    # what the raw read summed (every provenance it found), never what the reader reports now
    raw_den = sum(int(r[2]) for r in raw)
    assert raw_den > cur[0][2] and out[0]["denominator"] != raw_den


# --------------------------------------------------------------------------------------------
# 2. emit_contrasts: a metric filed only under a pending or superseded revision is not filed
# --------------------------------------------------------------------------------------------


def test_contrasts_due_reads_no_items_for_a_run_whose_rows_ride_only_a_pending_or_superseded_revision(tmp_path, monkeypatch):
    root, ids = _synth_root(tmp_path, monkeypatch, [("s_none", 100, "none", "none", "judged", 1, None),
                                                    ("s_nat", 200, "constitution_lora@deadbeef", synth.OBJECT_SHA, "judged", 3, None),
                                                    ("s_pend", 300, "constitution_lora@deadbeef", synth.OBJECT_SHA, "judged", None, None),
                                                    ("s_sup", 400, "constitution_lora@deadbeef", synth.OBJECT_SHA, "judged", None, None)])
    (r_none, _j), (r_nat, _jn), (r_pend, j_pend), (r_sup, j_sup) = ids["s_none"], ids["s_nat"], ids["s_pend"], ids["s_sup"]
    src = os.path.join(root, "records", "synth_sources")
    R.write_records(os.path.join(src, "000001-revisions.jsonl"),
                    [_revision("revP", "seedrep:pend", 9000, accepted=0), _revision("rev1", "seedrep:sup", 9000)])
    _ingest(root)
    for rid, jid, rev, n in ((r_pend, j_pend, "revP", 60), (r_sup, j_sup, "rev1", 61)):
        d = os.path.join(root, "records", synth.CAMPAIGN, I.run_dir_name(rid))
        R.write_records(os.path.join(d, R.record_file_name(n, "item_result", gz=True)),
                        synth.item_records(rid, jid, 305, 200, seed=2, source_revision_id=rev), gz=True)
    _ingest(root)
    # a later read of the second source carries no rows for the run: rev1 is superseded, its rows read by nothing
    R.write_records(os.path.join(src, "000002-rev2.jsonl"), [_revision("rev2", "seedrep:sup", 9100)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT superseded_by_revision_id FROM source_revision WHERE revision_id = 'rev1'").fetchone()[0] == "rev2"
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND source_revision_id = 'revP'", (r_pend,)).fetchone()[0] == 200
    for rid in (r_pend, r_sup):
        assert conn.execute("SELECT COUNT(*) FROM v_item_result_current WHERE run_id = ?", (rid,)).fetchone()[0] == 0
        assert EC._filed_item_metrics(conn, rid) == []
    assert EC._filed_item_metrics(conn, r_nat) == ["asr"]
    due = {d["run_id"]: d for d in EC.contrasts_due(conn, campaign=synth.CAMPAIGN)}
    conn.close()
    assert (due[r_nat]["status"], due[r_nat]["anchor_run_id"]) == ("due", r_none)
    for rid in (r_pend, r_sup):
        assert due[rid]["status"] == "no_items" and due[rid]["filed_item_metric_ids"] == [], due[rid]


FILED_METRICS_SQL = "SELECT DISTINCT metric_id FROM v_item_result_current WHERE run_id = "


def test_contrasts_due_reads_each_steered_runs_filed_item_metrics_once_per_pass_and_hands_out_copies(tmp_path, monkeypatch):
    """A pass read a steered run's filed item metrics twice (the same-view keep check, then the pair check). It reads them once per run,
    from a memo that belongs to the pass (a new pass reads again) and hands every caller its own list, so one caller changing what it
    got never changes what the next one reads. The pass's output is what reading afresh at every call gives."""
    root, ids = _synth_root(tmp_path, monkeypatch, [("s_none", 100, "none", "none", "judged", 1, None),
                                                    ("s_nat", 200, "constitution_lora@deadbeef", synth.OBJECT_SHA, "judged", 3, None),
                                                    ("s_bare", 300, "constitution_lora@deadbeef", synth.OBJECT_SHA, "judged", None, None)])
    (r_none, _j), (r_nat, _jn), (r_bare, _jb) = ids["s_none"], ids["s_nat"], ids["s_bare"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        seen = []

        def reads():
            out = {}
            for sql in seen:
                if FILED_METRICS_SQL in sql:
                    rid = sql.split(FILED_METRICS_SQL, 1)[1].split("'")[1]
                    out[rid] = out.get(rid, 0) + 1
            return out

        conn.set_trace_callback(seen.append)
        due = EC.contrasts_due(conn, campaign=synth.CAMPAIGN)
        conn.set_trace_callback(None)
        by = {d["run_id"]: d for d in due}
        assert (by[r_nat]["status"], by[r_nat]["anchor_run_id"]) == ("due", r_none), by[r_nat]
        assert by[r_bare]["status"] == "no_items" and by[r_bare]["filed_item_metric_ids"] == [], by[r_bare]
        assert reads() == {r_nat: 1, r_bare: 1}, reads()
        # the pass's output, unchanged against a pass that reads afresh at every call
        with monkeypatch.context() as m:
            m.setattr(EC, "_filed_item_metrics_memo", lambda c: (lambda run_id: EC._filed_item_metrics(c, run_id)))
            assert EC.contrasts_due(conn, campaign=synth.CAMPAIGN) == due
        # the memo: one read per run, a fresh list per call, and nothing shared with another pass's memo
        seen.clear()
        conn.set_trace_callback(seen.append)
        filed = EC._filed_item_metrics_memo(conn)
        first = filed(r_nat)
        first.append("changed by a caller")
        assert filed(r_nat) == ["asr"] and filed(r_nat) is not filed(r_nat)
        assert reads() == {r_nat: 1}, reads()
        assert EC._filed_item_metrics_memo(conn)(r_nat) == ["asr"] and reads() == {r_nat: 2}, reads()
        conn.set_trace_callback(None)
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# 3-5. refusals.resolve: current rows at the measurement key; the revision block; the labelled staged row; revision_reclaimed
# --------------------------------------------------------------------------------------------


def _bad(rec, i, rid="11111111-2222-3333-4444-555555555555"):
    """``rec`` with a value the column rule refuses by name, re-stamped at ``i``."""
    bad = json.loads(json.dumps(rec))
    bad["payload"]["value"] = "not a number"
    bad["written_utc"] = synth.t(i)
    bad["record_id"] = rid
    return bad


def test_a_refusal_naming_a_pending_revision_resolves_against_the_current_row_and_labels_the_revisions_staged_row(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=3, with_items=False)
    _ingest(root)
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    R.write_records(os.path.join(root, "records", "c", "000003-native.jsonl"), synth.item_records(rid, jid, 300, n_items=3, seed=1))  # t(302)
    R.write_records(os.path.join(root, "records", "synth_sources", "000001-revP.jsonl"), [_revision("revP", "seedrep:pending", 9000, accepted=0)])
    _ingest(root)
    staged = synth.item_records(rid, jid, 330, n_items=3, seed=2, source_revision_id="revP")          # the pending read's rows, t(332)
    R.write_records(os.path.join(root, "records", "c", "000004-staged.jsonl"), staged[1:2])           # q00001 only
    _ingest(root)
    bad_q0 = _bad(staged[0], 340)                                    # no staged row at its key
    bad_q1 = _bad(staged[1], 320, "11111111-2222-3333-4444-666666666666")   # EARLIER than the staged row at its key
    R.write_records(os.path.join(root, "records", "c", "000005-bad.jsonl"), [bad_q0, bad_q1])
    _ingest(root, refused=2)
    conn = _ro(root)
    try:
        native = {r[0]: r[1] for r in conn.execute("SELECT item_id, source_revision_id FROM v_item_result_current WHERE run_id = ?", (rid,))}
        assert native == {"q00000": "", "q00001": "", "q00002": ""}
        e0 = RF.explain(root, conn, R.record_sha(bad_q0))["records"][0]["resolution"]
        e1 = RF.explain(root, conn, R.record_sha(bad_q1))["records"][0]["resolution"]
        lines = RF.explain_lines(RF.explain(root, conn, R.record_sha(bad_q1)))
    finally:
        conn.close()
    for e in (e0, e1):
        # the CURRENT row at the measurement key is the native one, written before either refusal: an earlier row stands
        assert e["scale"] == "item" and e["live_row"] is True and e["live_source_revision_id"] == "" and e["state"] == "earlier_row_stands", e
        assert e["live_order_key"][0] == synth.t(302)
        rv = e["revision"]
        assert (rv["revision_id"], rv["present"], rv["accepted"], rv["superseded_by_revision_id"], rv["reclaimed"]) == ("revP", True, False, None, False), rv
        assert rv["in_reclaim_ledger"] is False and rv["ledger_carriers"] == 0 and rv["ledger_rows_deleted"] == 0
        assert rv["refusal"] is None
    # the pending revision's own row at the FULL key, labelled apart: none at q00000; at q00001 one LATER than the refused record
    # (the raw by-revision read made that row the resolution: later_row_folded, a row no reader sees)
    assert e0["staged_row"] == {"present": False}
    assert e1["staged_row"] == {"present": True, "order_key": [synth.t(332), R.record_sha(staged[1])], "later": True}
    assert any("staged row (unaccepted revision, read by no reader)" in ln and "later than the refused record" in ln for ln in lines), lines
    assert any("revision revP: accepted 0" in ln for ln in lines), lines


def _insert_revision(conn, rid, source, utc, accepted, superseded_by=None, reclaimed=0):
    conn.execute("INSERT INTO source_revision (revision_id, source_id, adapter, observed_utc, inputs_json, accepted, refusal, superseded_by_revision_id, "
                 "priority, reclaimed, field_versions_json, extra_json, record_uri, ingest_seq) VALUES (?,?,?,?,?,?,NULL,?,0,?,'{}','{}',?,1)",
                 (rid, source, "seedrep", utc, "{}", accepted, superseded_by, reclaimed, f"records/adapters/seedrep/{rid}/000000-source_revision.jsonl"))


def _insert_ledger(conn, table, rev, record_file_id, rows):
    conn.execute(f"INSERT INTO {LEDGER} (table_name, revision_id, record_file_id, file_uri, expected_rows, expected_digest_fold, rows_deleted, "
                 "deleted_digest_fold, deleted_digest_bind, parsed_with) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (table, rev, record_file_id, f"records/adapters/seedrep/{rev}/{record_file_id:06d}-{table}.jsonl", rows, "a" * 64, rows,
                  "a" * 64, "b" * 64, "stamp-1"))


def test_resolve_reads_revision_reclaimed_only_where_no_current_row_stands_and_the_ledger_lists_the_revisions_rows(tmp_path):
    root = _root(tmp_path)
    assert cli.main(["--root", root, "init"]) == 0
    conn = DB.connect(DB.db_path(root))
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        _insert_revision(conn, "rev2", "seedrep:s", synth.t(9100), 1)
        _insert_revision(conn, "rev1", "seedrep:s", synth.t(9000), 1, superseded_by="rev2", reclaimed=1)
        _insert_revision(conn, "revP", "seedrep:p", synth.t(9200), 0)
        _insert_ledger(conn, "item_result", "rev1", 7, 3)
        _insert_ledger(conn, "item_flags", "rev2", 8, 2)                    # another table's ledger row names nothing for item_result
        conn.commit()
        key = {"run_id": "r1", "item_id": "q00000", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1, "judgment_id": ""}

        def refused(rev):
            return {"record_type": "item_result", "payload": dict(key, source_revision_id=rev), "written_utc": synth.t(400)}
        res = RF.resolve(conn, refused("rev1"), "f" * 64)
        assert res["state"] == "revision_reclaimed" and res["live_row"] is False and "staged_row" not in res, res
        assert res["revision"] == {"revision_id": "rev1", "present": True, "accepted": True, "superseded_by_revision_id": "rev2", "reclaimed": True,
                                   "refusal": None, "in_reclaim_ledger": True, "ledger_carriers": 1, "ledger_rows_deleted": 3}
        assert "revision_reclaimed" in RF.RESOLUTIONS
        # the head: accepted, not superseded, nothing in the ledger -> no_row (and no staged-row probe: it is accepted)
        res = RF.resolve(conn, refused("rev2"), "f" * 64)
        assert res["state"] == "no_row" and res["revision"]["in_reclaim_ledger"] is False and "staged_row" not in res, res
        # the table matters: rev1's item_flags rows were never deleted -> an item_flags refusal of rev1 reads no_row
        flags = {"record_type": "item_flags", "payload": {k: v for k, v in key.items() if k != "metric_id"} | {"source_revision_id": "rev1"},
                 "written_utc": synth.t(400)}
        res = RF.resolve(conn, flags, "f" * 64)
        assert res["table"] == "item_flags" and res["state"] == "no_row" and res["revision"]["in_reclaim_ledger"] is False, res
        # an unaccepted revision: the labelled by-revision probe; a revision with no row at all: present False, the probe too
        res = RF.resolve(conn, refused("revP"), "f" * 64)
        assert res["state"] == "no_row" and res["revision"]["accepted"] is False and res["staged_row"] == {"present": False}, res
        res = RF.resolve(conn, refused("revX"), "f" * 64)
        assert res["revision"] == {"revision_id": "revX", "present": False, "accepted": False} and res["staged_row"] == {"present": False}, res
        # a native record names no revision: the resolution's shape is unchanged
        res = RF.resolve(conn, refused(""), "f" * 64)
        assert res == {"state": "no_row", "scale": "item", "live_row": False, "table": "item_result", "natural_key": res["natural_key"],
                       "refused_order_key": [synth.t(400), "f" * 64]}
    finally:
        conn.close()


def test_a_refusal_of_a_superseded_revision_reclaim_emptied_resolves_revision_reclaimed_end_to_end(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=3, with_items=False)
    _ingest(root)
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    src = os.path.join(root, "records", "synth_sources")
    R.write_records(os.path.join(src, "000001-rev1.jsonl"), [_revision("rev1", "seedrep:s", 9000)])
    _ingest(root)
    rows = synth.item_records(rid, jid, 300, n_items=4, seed=1, source_revision_id="rev1")
    R.write_records(os.path.join(root, "records", "c", "000003-rev1-items.jsonl"), rows[:3])       # the carrier reclaim deletes
    bad = _bad(rows[3], 310)                                                                         # a key rev1 never filed a row at
    R.write_records(os.path.join(root, "records", "c", "000004-rev1-bad.jsonl"), [bad])
    _ingest(root, refused=1)
    conn = _ro(root)
    pre = RF.explain(root, conn, R.record_sha(bad))["records"][0]["resolution"]
    conn.close()
    assert pre["state"] == "no_row" and pre["revision"]["accepted"] is True and pre["revision"]["superseded_by_revision_id"] is None, pre
    # a later read of the source carries no item rows: rev1 is superseded, and the cycle's reclaim deletes its carrier
    R.write_records(os.path.join(src, "000002-rev2.jsonl"), [_revision("rev2", "seedrep:s", 9100)])
    _ingest(root)
    conn = _ro(root)
    try:
        assert tuple(conn.execute("SELECT superseded_by_revision_id, reclaimed FROM source_revision WHERE revision_id = 'rev1'").fetchone()) == ("rev2", 1)
        assert conn.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = 'rev1'").fetchone()[0] == 0
        assert tuple(conn.execute(f"SELECT COUNT(*), SUM(rows_deleted) FROM {LEDGER} WHERE table_name = 'item_result' AND revision_id = 'rev1'").fetchone()) == (1, 3)
        rep = RF.explain(root, conn, R.record_sha(bad))
    finally:
        conn.close()
    res = rep["records"][0]["resolution"]
    assert res["state"] == "revision_reclaimed" and res["live_row"] is False, res
    assert res["revision"]["reclaimed"] is True and res["revision"]["ledger_rows_deleted"] == 3 and "staged_row" not in res
    assert any("resolution: revision_reclaimed" in ln for ln in RF.explain_lines(rep))
    assert any("1 carrier(s), 3 row(s) deleted by reclaim in item_result" in ln for ln in RF.explain_lines(rep))


# --------------------------------------------------------------------------------------------
# 6-7. the status page: the reclaim line and the held rehydrations
# --------------------------------------------------------------------------------------------


def _status_root(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=1, n_items=5)
    _ingest(root)
    return root


def _append_metrics(root, lines):
    with open(os.path.join(root, "state", "metrics.jsonl"), "a", encoding="utf-8") as fh:
        for ln in lines:
            fh.write(json.dumps(ln) + "\n")


HOLD_REASON = ("rehydration of records/adapters/seedrep/revA/000007-item_result.jsonl (item_result) does not reproduce the 5 deleted rows: "
               "digest mismatch")


def test_status_reclaim_line_reads_the_chain_the_ledger_and_the_cycle_and_lists_a_held_rehydration(tmp_path):
    root = _status_root(tmp_path)
    conn = DB.connect(DB.db_path(root))
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_revision(conn, "revB", "seedrep:a", synth.t(9100), 1)
    _insert_revision(conn, "revA", "seedrep:a", synth.t(9000), 1, superseded_by="revB", reclaimed=0)      # due
    _insert_revision(conn, "revD", "seedrep:c", synth.t(9300), 1)
    _insert_revision(conn, "revC", "seedrep:c", synth.t(9200), 1, superseded_by="revD", reclaimed=1)      # reclaimed
    _insert_revision(conn, "revP", "seedrep:p", synth.t(9400), 0)                                         # pending: not counted
    _insert_ledger(conn, "item_result", "revC", 11, 5)
    _insert_ledger(conn, "run_degeneration", "revA", 13, 1)
    conn.execute(f"INSERT INTO {LEDGER} (table_name, revision_id, record_file_id, file_uri, expected_rows, expected_digest_fold, rows_deleted, "
                 "deleted_digest_fold, deleted_digest_bind, parsed_with) VALUES ('item_flags','revC',12,'records/x/000012-item_flags.jsonl',4,?,2,?,?,'stamp-1')",
                 ("a" * 64, "a" * 64, "b" * 64))                                                            # a stop left it part-deleted
    marker = "records/adapters/seedrep/revA/_activations/20260101T000000Z/000000-source_revision.jsonl"
    conn.execute("INSERT INTO blocked_records (record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, first_seen_utc, "
                 "last_checked_utc, attempts, stale, extra_json) VALUES (?,?,?,?,?,?,?,1,0,?)",
                 ("c" * 64, marker, "source_revision", LEDGER, "revA", synth.t(9500), synth.t(9500),
                  json.dumps({"revision_id": "revA", "blocked_on": HOLD_REASON, "kind": "rehydration"})))
    conn.commit()
    conn.close()
    utc = synth.t(9600)
    _append_metrics(root, [{"kind": "reclaim", "table": "item_result", "revision": "revC", "record_file_id": 11, "rows": 5, "files": 0, "seconds": 0.1,
                            "utc": utc},
                           {"kind": "reclaim", "rows": 10000, "files": 0, "seconds": 0.1, "utc": utc},          # pass (i)'s un-mark: no table, not a deletion
                           {"kind": "rehydrate", "revision": "revE", "rows": 3, "inline": False, "seconds": 0.2, "utc": utc},
                           {"kind": "reclaim_stop", "reason": "wal", "wal_bytes": 2 ** 31, "busy": 1, "utc": utc},
                           {"kind": "cycle", "seconds": 3.0, "stopped": None, "files": 0, "utc": utc}])
    # the cycle's own record, as a cycle of the code before the reclaim fields wrote it: the last cycle's facts come from its metrics lines
    cycle_path = os.path.join(root, "state", "harvest_cycle.json")
    with open(cycle_path, encoding="utf-8") as fh:
        record = {k: v for k, v in json.load(fh).items() if not k.startswith("reclaim_")}
    record.update({"last_completed_cycle_utc": utc, "cycle_utc": utc})
    with open(cycle_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    last = status.status_json(root)["reclaim"]["last_cycle"]
    assert (last["reclaim_rows_deleted"], last["reclaim_rows_rehydrated"], last["reclaim_stopped"], last["utc"]) == (5, 3, "wal", utc), last
    # and as this code writes it, carrying the same facts itself (harvest.write_cycle_state): the record is what the line reads
    record.update({"reclaim_rows_deleted": 5, "reclaim_rows_rehydrated": 3, "reclaim_stopped": "wal", "reclaim_mode": "normal"})
    with open(cycle_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    ctx = status.status_json(root)
    rc = ctx["reclaim"]
    assert (rc["n_superseded"], rc["n_due"]) == (2, 1), rc
    assert rc["ledger"] == {"item_flags": {"carriers": 1, "revisions": 1, "rows_deleted": 2, "part_deleted": 1},
                            "item_result": {"carriers": 1, "revisions": 1, "rows_deleted": 5, "part_deleted": 0},
                            "run_degeneration": {"carriers": 1, "revisions": 1, "rows_deleted": 1, "part_deleted": 0}}, rc["ledger"]
    assert (rc["rows_deleted"], rc["carriers"], rc["part_deleted"], rc["holds"]) == (8, 3, 1, 1)
    last = rc["last_cycle"]
    assert (last["reclaim_rows_deleted"], last["reclaim_rows_rehydrated"], last["reclaim_stopped"], last["utc"]) == (5, 3, "wal", utc), last
    assert rc["tail"]["rows_deleted"] >= 5 and rc["tail"]["rows_rehydrated"] >= 3 and rc["tail"]["batches"] >= 1
    assert rc["mode"] == "normal" and rc["mode_error"] is None and rc["mode_file"] is False, rc
    assert rc["level"] == "red" and not rc["errors"]           # a held rehydration needs an operator
    holds = ctx["rehydration_holds"]
    assert [(h["revision_id"], h["blocked_on"], h["missing_parent_table"]) for h in holds] == [("revA", HOLD_REASON, LEDGER)]
    for name in ("reclaim_revisions", "reclaim_ledger", "rehydration_holds"):
        assert name in status.QUERIES and ctx["section_state"][name]["state"] == status.STATE_LIVE, (name, ctx["section_state"][name])
    html = _render(root, tmp_path)
    assert 'id="reclaim"' in html and "1 of 2 accepted, superseded revision(s) due" in html
    assert "8 row(s) over 3 carrier file(s)" in html and "1 part-deleted" in html and "stopped on <code>wal</code>" in html
    assert "Held on rehydration" in html and "does not reproduce the 5 deleted rows" in html and "revA" in html
    assert "re-activations held on rehydration" in html


def test_status_reclaim_line_on_a_quiet_root_and_a_mode_file_the_harvester_refuses(tmp_path):
    root = _status_root(tmp_path)
    rc = status.status_json(root)["reclaim"]
    assert rc["mode"] == "normal" and rc["mode_error"] is None and rc["holds"] == 0 and rc["rows_deleted"] == 0 and rc["n_due"] == 0, rc
    assert rc["level"] == "green" and rc["last_cycle"]["reclaim_rows_deleted"] == 0 and rc["last_cycle"]["reclaim_stopped"] is None, rc
    with open(os.path.join(root, "state", "reclaim_mode.json"), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    rc = status.status_json(root)["reclaim"]
    assert rc["mode_error"] and rc["mode_file"] is True and rc["level"] == "red", rc
    html = _render(root, tmp_path)
    assert "the mode file is refused" in html and "the harvester deletes nothing while it stands" in html


RECLAIM_SECTIONS = ("reclaim_revisions", "reclaim_ledger", "rehydration_holds")


def _insert_hold(conn, rid="revA"):
    """The marker row of a re-activation held on rehydration (``blocked_records``, blocked on the ledger)."""
    marker = f"records/adapters/seedrep/{rid}/_activations/20260101T000000Z/000000-source_revision.jsonl"
    conn.execute("INSERT INTO blocked_records (record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, first_seen_utc, "
                 "last_checked_utc, attempts, stale, extra_json) VALUES (?,?,?,?,?,?,?,1,0,?)",
                 ("c" * 64, marker, "source_revision", LEDGER, rid, synth.t(9500), synth.t(9500),
                  json.dumps({"revision_id": rid, "blocked_on": HOLD_REASON, "kind": "rehydration"})))


def _render_budget_spent(m):
    """The render budget is spent before the first section runs: every section is served from its cache, or from nothing."""
    real = status.status_config
    m.setattr(status, "status_config", lambda root: {**real(root), "render_budget_s": 0.0})


def _reclaim_paragraph(html):
    start = html.index('id="reclaim"')
    return html[html.rindex("<p ", 0, start):html.index("</p>", start)]


def _holds_tile(html):
    end = html.index("re-activations held on rehydration")
    return html[html.rindex('<div class="tile">', 0, end):end]


def test_status_reclaim_line_is_not_green_when_its_sections_were_not_computed(tmp_path, monkeypatch):
    """A render whose budget ran out before the three reclaim sections read no rows for them, and the line showed 0 held re-activations,
    0 deleted rows and green over a root that HAS a held rehydration. The line is green only when all three ran live this render: with no
    cached rows it is ``unknown`` and says the counts were not computed (the tile shows ``?``, never a green 0); served from the last good
    rows it is ``stale``, with their stamp and age; and a hold in the rows it does have keeps it red, which outranks both."""
    root = _status_root(tmp_path)
    conn = DB.connect(DB.db_path(root))
    conn.execute("PRAGMA foreign_keys=OFF")
    _insert_revision(conn, "revB", "seedrep:a", synth.t(9100), 1)
    _insert_revision(conn, "revA", "seedrep:a", synth.t(9000), 1, superseded_by="revB", reclaimed=0)      # due
    _insert_hold(conn)
    conn.commit()
    conn.close()
    assert not os.path.isdir(status.sections_dir(root)), "a fresh root has no cached section"

    # (a) not computed this render, nothing cached: the counts are unknown, never zeros and never green
    with monkeypatch.context() as m:
        _render_budget_spent(m)
        ctx = status.status_json(root)
        html = _render(root, tmp_path, "unknown.html")
    assert {n: ctx["section_state"][n]["state"] for n in RECLAIM_SECTIONS} == {n: status.STATE_NO_CACHE for n in RECLAIM_SECTIONS}
    rc = ctx["reclaim"]
    assert rc["level"] == "unknown" and rc["data_state"] == "unknown", rc
    assert rc["sections_not_live"] == {n: status.STATE_NO_CACHE for n in RECLAIM_SECTIONS}, rc
    assert (rc["holds"], rc["holds_state"], rc["n_due"], rc["n_superseded"]) == (None, "unknown", None, None), rc
    assert (rc["rows_deleted"], rc["carriers"], rc["part_deleted"], rc["as_of"], rc["age_s"]) == (None, None, None, None, None), rc
    para = _reclaim_paragraph(html)
    assert para.startswith('<p class="stale" id="reclaim">'), para
    assert "Counts not computed this render" in para and "unknown, not zero" in para, para
    assert "revisions due not computed" in para and "held re-activations: <strong>?</strong>" in para, para
    assert "0 row(s) over" not in para and "<strong>0</strong>" not in para, para
    tile = _holds_tile(html)
    assert '<b class="level-amber">?</b>' in tile and "level-green" not in tile, tile

    # a live render: the hold is red, and the render caches the three sections' rows
    _render(root, tmp_path, "live.html")
    rc = status.status_json(root)["reclaim"]
    assert (rc["level"], rc["data_state"], rc["holds"], rc["holds_state"], rc["sections_not_live"]) == ("red", "live", 1, "live", {}), rc
    assert (rc["as_of"], rc["age_s"]) == (None, None), rc

    # (b) served from those cached rows: stale, with the rows' stamp and age; the hold in them is a fact and keeps the line red
    with monkeypatch.context() as m:
        _render_budget_spent(m)
        ctx = status.status_json(root)
        html = _render(root, tmp_path, "stale_held.html")
    states = {n: ctx["section_state"][n] for n in RECLAIM_SECTIONS}
    assert {n: s["state"] for n, s in states.items()} == {n: status.STATE_CACHED for n in RECLAIM_SECTIONS}
    rc = ctx["reclaim"]
    as_of = min(str(s["as_of"]) for s in states.values())
    assert rc["data_state"] == "stale" and rc["as_of"] == as_of and isinstance(rc["age_s"], int) and 0 <= rc["age_s"] < 3600, rc
    assert rc["sections_not_live"] == {n: status.STATE_CACHED for n in RECLAIM_SECTIONS}, rc
    assert rc["level"] == "red" and (rc["holds"], rc["holds_state"]) == (1, "stale") and (rc["n_superseded"], rc["n_due"]) == (1, 1), rc
    para = _reclaim_paragraph(html)
    assert para.startswith('<p class="late" id="reclaim">') and f"Counts as of <code>{as_of}</code>" in para, para
    assert '<b class="level-red">1</b>' in _holds_tile(html)

    # the hold restored away: a live render caches rows without it (green), and a render served from them is stale, never green
    conn = DB.connect(DB.db_path(root))
    conn.execute("DELETE FROM blocked_records WHERE missing_parent_table = ?", (LEDGER,))
    conn.commit()
    conn.close()
    _render(root, tmp_path, "live2.html")
    assert status.status_json(root)["reclaim"]["level"] == "green"
    with monkeypatch.context() as m:
        _render_budget_spent(m)
        ctx = status.status_json(root)
        html = _render(root, tmp_path, "stale.html")
    rc = ctx["reclaim"]
    assert (rc["level"], rc["data_state"], rc["holds"], rc["holds_state"]) == ("stale", "stale", 0, "stale"), rc
    assert rc["as_of"] == min(str(ctx["section_state"][n]["as_of"]) for n in RECLAIM_SECTIONS) and rc["age_s"] is not None, rc
    para = _reclaim_paragraph(html)
    assert para.startswith('<p class="stale" id="reclaim">') and f"Counts as of <code>{rc['as_of']}</code>" in para, para
    tile = _holds_tile(html)
    assert '<b class="level-amber">0</b>' in tile and "level-green" not in tile, tile


def test_reclaim_summary_level_is_green_only_when_all_three_sections_ran_live():
    """The precedence of the line's level over a hand-built context: red (a refused mode file, a failed query, a hold in the rows the line
    has) > unknown (a section with no rows this render and none cached, a failed one, or one the context does not name) > stale (served
    from the last good rows) > amber (the last cycle's reclaim stopped early) > green. Only a line whose three sections all ran live is
    green or amber."""
    now_s = R.utc_to_epoch_ms(synth.t(9000)) / 1000.0 + 120

    def ctx(states, stopped=None, mode_error=None, holds=(), error=None):
        c = {"reclaim_state": {"mode": "normal", "mode_error": mode_error, "last_cycle": {"reclaim_stopped": stopped}},
             "reclaim_revisions": [{"n_superseded": 2, "n_due": 1}], "reclaim_ledger": [], "rehydration_holds": list(holds),
             "section_state": {n: {"state": s, "as_of": synth.t(9000 + i)} for i, (n, s) in enumerate(states.items())}}
        if error:
            c[error] = [{"error": "no such table"}]
        return c

    live = {n: status.STATE_LIVE for n in RECLAIM_SECTIONS}
    one = lambda state: {**live, "reclaim_ledger": state}   # noqa: E731
    hold = {"revision_id": "revA", "blocked_on": HOLD_REASON, "missing_parent_table": LEDGER}

    def level(c):
        rc = status.reclaim_summary(c, now_s=now_s)
        return rc["level"], rc["data_state"]

    assert level(ctx(live)) == ("green", "live")
    assert level(ctx(live, stopped="wal")) == ("amber", "live")
    assert level(ctx(one(status.STATE_CACHED))) == ("stale", "stale")
    assert level(ctx(one(status.STATE_CACHED), stopped="wal")) == ("stale", "stale")
    assert level(ctx(one(status.STATE_NO_CACHE))) == ("unknown", "unknown")
    assert level(ctx({**one(status.STATE_NO_CACHE), "rehydration_holds": status.STATE_CACHED})) == ("unknown", "unknown")
    assert level(ctx({n: s for n, s in live.items() if n != "reclaim_ledger"})) == ("unknown", "unknown")   # not named: not known to have run
    assert level(ctx(one(status.STATE_NO_CACHE), mode_error="ModeStateInvalid: x")) == ("red", "unknown")
    assert level(ctx(one(status.STATE_ERROR), error="reclaim_ledger")) == ("red", "unknown")
    assert level(ctx({**one(status.STATE_NO_CACHE), "rehydration_holds": status.STATE_CACHED}, holds=[hold])) == ("red", "unknown")
    rc = status.reclaim_summary(ctx(one(status.STATE_CACHED)), now_s=now_s)
    assert (rc["as_of"], rc["age_s"], rc["sections_not_live"]) == (synth.t(9001), 119, {"reclaim_ledger": status.STATE_CACHED}), rc
    assert (rc["rows_deleted"], rc["holds"], rc["holds_state"]) == (0, 0, "live"), rc   # cached rows are counts; stale, not unknown
    rc = status.reclaim_summary(ctx(one(status.STATE_NO_CACHE)), now_s=now_s)
    assert (rc["rows_deleted"], rc["carriers"], rc["part_deleted"], rc["holds"], rc["n_due"]) == (None, None, None, 0, 1), rc


def test_reclaim_sums_count_each_deleted_and_rehydrated_row_once(tmp_path):
    """The status page's and ``prune-superseded status``'s sums over the metrics log equal what the cycles themselves reported. A
    rehydration outside a transaction writes its rows in bounded transactions, each logged ``rehydrate`` with its share of the rows, beside
    the revision's own ``rehydrate`` line carrying the total; pass (i)'s un-mark logs ``reclaim`` with the batch size, not rows deleted.
    Neither is counted a second time."""
    from sisteer.registry import prune as P
    from sisteer.registry import prune_archive as PA
    import test_registry_prune_fold as F
    # rev_a deleted, re-activated by a bundle (rehydrated first), rev_b deleted, re-activated (rehydrated), rev_a deleted again
    root, rev_a, rev_b = F._reactivation_world(tmp_path / "w", None, tmp_path / "stores")
    tail = P.metrics_tail(root)
    assert any(m.get("kind") == "rehydrate" and not m.get("revision") and m.get("rows") for m in tail)   # the per-transaction lines
    # the last cycle's facts as a record that predates the reclaim fields leaves them: derived from that cycle's own lines
    cycle_path = os.path.join(root, "state", "harvest_cycle.json")
    with open(cycle_path, encoding="utf-8") as fh:
        record = json.load(fh)
    with open(cycle_path, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in record.items() if not k.startswith("reclaim_")}, fh)
    last_line = [m for m in tail if m.get("kind") == "cycle"][-1]
    assert last_line["reclaim_rows_rehydrated"] > 0 and last_line["reclaim_rows_deleted"] > 0
    last = status.reclaim_state(root)["last_cycle"]
    assert (last["reclaim_rows_deleted"], last["reclaim_rows_rehydrated"]) == (last_line["reclaim_rows_deleted"],
                                                                                  last_line["reclaim_rows_rehydrated"]), (last, last_line)
    # a head row carrying a mark from before 0033: pass (i) clears it in a `reclaim` transaction logged with the batch size
    conn = DB.connect(DB.db_path(root))
    conn.execute("UPDATE item_result SET withdrawn_by_revision_id = ? WHERE source_revision_id = ?", (rev_a, rev_b))
    conn.close()
    F._ingest(root, 5100)
    tail = P.metrics_tail(root)
    assert any(m.get("kind") == "reclaim" and not m.get("table") and m.get("rows") == H.RECLAIM_BATCH_ROWS for m in tail)
    cycles = [m for m in tail if m.get("kind") == "cycle"]
    deleted, rehydrated = (sum(int(c["reclaim_rows_deleted"]) for c in cycles), sum(int(c["reclaim_rows_rehydrated"]) for c in cycles))
    assert deleted > 0 and rehydrated > 0
    st = status.reclaim_state(root)["tail"]
    assert (st["rows_deleted"], st["rows_rehydrated"]) == (deleted, rehydrated), (st, deleted, rehydrated)
    assert PA.status(root)["recent_reclaim_rows"] == deleted
