"""The harvester: idempotent ingest; burst coalesced into one transaction; the free-space
guard pauses and reports; gzip-after-ingest stays readable and waits for acks; refused records are
listed, never skipped; the lock excludes a second writer; harvester facts are records first;
completion-gated `ingested` (S6); the blocked queue is re-evaluated each cycle; inbox merge + conflicts.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def test_ingest_twice_changes_nothing(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=2, n_items=10)
    r1 = H.ingest_once(root, workers=0, min_free=0)
    assert r1.ok > 0 and r1.refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    snap = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in R.ALL_TABLES}
    conn.close()
    r2 = H.ingest_once(root, workers=0, min_free=0)
    assert r2.ok == 0 and r2.refused == 0 and not r2.changed and r2.harvester_records == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in R.ALL_TABLES} == snap
    conn.close()


def test_burst_coalesced_one_transaction(tmp_path):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    for i, r in enumerate(dims):
        _write(root, "burst", f"{i:06d}-d.jsonl", [r])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.files == len(dims) + 1 and rep.transactions == 3             # one chunk + the arrival facts' fold + the dense re-rank
    # 120 files → 3 transactions of ≤ 50
    root2 = str(tmp_path / "reg2")
    H.ensure_layout(root2)
    for i in range(120):
        _write(root2, "b", f"{i:06d}-d.jsonl", [synth.rec("si_text", {"si_id": f"si{i}", "si_text_sha256": ("%064x" % (i + 1)),
                                                                    "si_kind": "irrelevant_control", "source_uri": "synthetic://x"}, synth.t(i))])
    rep2 = H.ingest_once(root2, workers=0, min_free=0)
    assert rep2.files == 121 and rep2.transactions == 5 and rep2.ok == 240 and rep2.arrival_records == 120   # three chunks + the arrival facts' fold + the re-rank


def test_free_space_guard_pauses_and_reports(tmp_path, monkeypatch):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=1, n_items=5)
    monkeypatch.setattr(H, "free_bytes", lambda path: 1024 ** 3)      # a controlled reading: 1 GB free, whatever the host has
    rep = H.ingest_once(root, workers=0, min_free=2 * 1024 ** 3)       # a 2 GB floor → paused
    assert rep.paused and rep.ok == 0 and rep.free_bytes == 1024 ** 3
    flag = json.load(open(os.path.join(root, "state", "paused.json")))
    assert ("5 GB" in flag["reason"] or "floor" in flag["reason"]) and flag["free_bytes"] == 1024 ** 3
    assert H.paused(root) is not None
    # records stayed in the spool; the next cycle above the floor ingests them and clears the flag
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert not rep2.paused and rep2.ok > 0
    assert H.paused(root) is None


def test_free_bytes_is_a_positive_int_never_above_the_filesystem_total(tmp_path, monkeypatch):
    real = H.free_bytes(str(tmp_path))
    assert isinstance(real, int) and real > 0 and real <= shutil.disk_usage(str(tmp_path)).total
    # a container filesystem reporting an absurd free figure (the Modal CPU suite: ~9.2e18) is clamped to its total,
    # so no free-space guard can take space for infinite
    fake = type("du", (), {"total": 500 * 1024 ** 3, "used": 100 * 1024 ** 3, "free": 9223372036188258304})()
    monkeypatch.setattr(H.DB.shutil, "disk_usage", lambda path: fake)
    assert H.free_bytes(str(tmp_path)) == 500 * 1024 ** 3
    ok, free = H.free_space_guard(str(tmp_path), min_free=600 * 1024 ** 3)
    assert not ok and free == 500 * 1024 ** 3                          # the guard sees the clamped figure


def test_gzip_after_ingest_readable_and_only_after_acks(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    H.ingest_once(root, workers=0, min_free=0, gzip_min_age_s=0)
    plain = os.path.join(root, "records", "dims", "000001-dims.jsonl")
    assert not os.path.exists(plain) and os.path.exists(plain + ".gz")   # no mirrors configured → compressed at once
    # readable and identical
    recs = list(R.read_record_file(plain + ".gz"))
    assert len(recs) == len(synth.seed_dimensions())
    rep = H.ingest_once(root, workers=0, min_free=0, gzip_min_age_s=0)
    assert rep.ok == 0 and rep.duplicates == 0 and rep.files == 0  # logical sha unchanged → nothing to do
    # with mirrors configured, gzip waits for every destination's ack
    root2 = str(tmp_path / "reg2")
    H.ensure_layout(root2)
    with open(os.path.join(root2, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m1")}]}, fh)
    _write(root2, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    H.ingest_once(root2, workers=0, min_free=0, gzip_min_age_s=0)
    plain2 = os.path.join(root2, "records", "dims", "000001-dims.jsonl")
    assert os.path.exists(plain2)  # not acked → not compressed
    conn = DB.connect(DB.db_path(root2), readonly=True)
    seq = conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0]
    conn.close()
    ack = synth.rec("mirror_ack", {"destination": "cluster_a", "acked_through_seq": seq, "manifest_name": "CHECKSUMS-x", "file_count": 1,
                                   "bytes": 1, "checksums_sha256": "0" * 64, "acked_utc": synth.t(999)}, synth.t(999))
    _write(root2, "harvester", "ack.jsonl", [ack])
    H.ingest_once(root2, workers=0, min_free=0, gzip_min_age_s=0)
    assert not os.path.exists(plain2) and os.path.exists(plain2 + ".gz")


def test_refused_record_listed_not_skipped(tmp_path):
    root = _root(tmp_path)
    good = synth.seed_dimensions()
    bad = synth.rec("model", {"model_id": "m_bad", "family": "gemma4", "size_label": "x", "checkpoint_kind": "chatty",
                              "hf_repo": "r", "hf_revision": "a" * 40, "chat_template_sha256": "b" * 64, "generation_config_present": 0,
                              "thinking_channel": "not_available", "paper_status": "in_paper"}, synth.t(50))
    _write(root, "dims", "000001-dims.jsonl", good + [bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.ok - rep.arrival_records == len(good) and rep.refused == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    r = conn.execute("SELECT record_type, field, reason FROM record_refusal").fetchone()
    assert r[0] == "model" and r[1] == "checkpoint_kind" and "chatty" in r[2]
    assert conn.execute("SELECT status FROM record_delivery").fetchone()[0] == "partial"
    conn.close()
    from sisteer.registry import status
    ctx = status.status_json(root)
    assert any("checkpoint_kind" in (row.get("field") or "") for row in ctx["refused"])


def test_lock_excludes_second_writer(tmp_path):
    root = _root(tmp_path)
    with DB.write_lock(root):
        with pytest.raises(DB.LockHeld):
            with DB.write_lock(root, timeout_s=0.2):
                pass
    with DB.write_lock(root):
        pass


def test_harvester_facts_are_records_first(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=1, n_items=5)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.harvester_records == 1
    day_dir = os.path.join(root, "records", "harvester")
    files = [f for _, _, fs in os.walk(day_dir) for f in fs]
    assert any(f.startswith("run_ingest-") for f in files)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT lifecycle_state, ingested_utc, record_uri FROM run").fetchone()
    assert row[0] == "ingested" and row[1] is not None
    # the ROW points at the harvester record that produced it
    assert "records/harvester/" in row[2]
    assert conn.execute("SELECT COUNT(*) FROM record_ingest WHERE file_uri LIKE 'records/harvester/%run_ingest%'").fetchone()[0] == 1
    conn.close()


def test_nonterminal_states_survive_repeated_cycles(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    expected = {}
    for k, through in enumerate(("queued", "running", "generated")):
        rid, recs = synth.run_records(f"k{k}", 100 + 20 * k, through=through)
        _write(root, "c", f"00000{k}-run.jsonl", recs)
        expected[rid] = through
    # failed and cancelled runs
    for k, st in enumerate(("failed", "cancelled")):
        rid, recs = synth.run_records(f"f{k}", 300 + 20 * k, through="running")
        recs.append(synth.rec("run_state", {"run_id": rid, "state": st, "event_utc": synth.t(300 + 20 * k + 5), "evidence_kind": "scheduler_state",
                                            "scheduler_state": st.upper()}, synth.t(300 + 20 * k + 5)))
        _write(root, "c", f"00001{k}-run.jsonl", recs)
        expected[rid] = st
    for _ in range(5):
        rep = H.ingest_once(root, workers=0, min_free=0)
        assert rep.harvester_records == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    states = {r[0]: r[1] for r in conn.execute("SELECT run_id, lifecycle_state FROM run")}
    assert states == expected
    assert conn.execute("SELECT COUNT(*) FROM run WHERE ingested_utc IS NOT NULL").fetchone()[0] == 0
    inflight = {r[0] for r in conn.execute("SELECT run_id FROM v_in_flight")}
    assert inflight == {rid for rid, st in expected.items() if st in ("queued", "running", "generated")}
    pending = {r[0] for r in conn.execute("SELECT run_id FROM v_pending_judgment WHERE is_instrument_judge = 1")}
    assert pending == {rid for rid, st in expected.items() if st == "generated"}
    assert conn.execute("SELECT finished_utc FROM run WHERE lifecycle_state = 'failed'").fetchone()[0] is not None
    conn.close()


def test_ingested_only_when_completion_requirements_hold(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    # judged, but the judgment is PARTIAL (coverage 10 < 200) → stays judged
    rid, recs = synth.run_records("k", 100, through="judged")
    recs = [r for r in recs if r["record_type"] != "judgment"]
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    recs.append(synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": 10,
                                       "status": "partial"}, synth.t(106)))
    _write(root, "c", "000001-run.jsonl", recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "judged"
    ok, why = H.completion_requirements_hold(conn, rid)
    assert not ok and "no complete judgment" in why
    conn.close()
    # a complete judgment with coverage → ingested on the next cycle
    _write(root, "c", "000002-j.jsonl", [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID,
                                                                "n_items_judged": 200, "status": "complete"}, synth.t(110))])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.harvester_records == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "ingested"
    conn.close()
    # a judged run whose store lacks its completion artifact never becomes ingested
    root2 = _root(tmp_path / "b")
    _write(root2, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid2, recs2 = synth.run_records("k2", 100, through="judged")
    recs2 = [r for r in recs2 if not (r["record_type"] == "run_patch" and "store_completion_sha256" in r["payload"].get("fields", {}))]
    _write(root2, "c", "000001-run.jsonl", recs2)
    H.ingest_once(root2, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root2), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "judged"
    conn.close()


# --------------------------------------------------------------------------------------------
# coverage counts every outcome the judgment recorded for a planned item: judged, declined, ERRORED (2026-09-21)
# --------------------------------------------------------------------------------------------


def _judged_root_with_judgment(tmp_path, sub, judgment_fields, metric_rows=()):
    """A judged run (200 planned = generated, item set 200, completion artifact present) whose one complete judgment by the
    instrument's judge carries ``judgment_fields`` in place of the corpus writer's full-coverage judgment, and whose metric rows
    are ``metric_rows`` (the writer's `asr` row is dropped)."""
    root = _root(tmp_path / sub)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="judged")
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    recs = [r for r in recs if r["record_type"] not in ("judgment", "run_metric")]
    recs.append(synth.rec("judgment", dict({"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "status": "complete"},
                                           **judgment_fields), synth.t(106)))
    for i, m in enumerate(metric_rows):
        recs.append(synth.rec("run_metric", dict({"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.2,
                                                  "k": 38, "n": 190, "computed_by": "producer"}, **m), synth.t(109 + i)))
    _write(root, "c", "000001-run.jsonl", recs)
    return root, rid, jid


def _state(root):
    with DB.reading(DB.db_path(root)) as conn:
        return conn.execute("SELECT lifecycle_state FROM run").fetchone()[0]


def test_judge_errors_count_toward_completion_coverage_and_the_fold_stays_monotonic(tmp_path):
    """194 judged + 4 declined + 2 errored = 200 planned -> `ingested`. Under the rule before 2026-09-21 (judged + declined only)
    the same judgment read 198 < 200 and the run stayed `judged` forever -- the or_bench_hard (1,314 + 4 + 1 = 1,319) and xstest
    (249 + 1 = 250) shape the orxs_seedrep reader's dry run found. The regression guard below evaluates that old arithmetic on
    the same rows and asserts it IS short, so the stamp is attributable to the errors and to nothing else."""
    root, rid, jid = _judged_root_with_judgment(tmp_path, "err", {"n_items_judged": 194, "n_judge_refusals": 4, "n_errors": 2})
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(150))
    assert rep.refused == 0 and rep.harvester_records == 1
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT lifecycle_state, ingested_utc FROM run").fetchone()) == ("ingested", synth.t(150))
        assert H.completion_requirements_hold(conn, rid) == (False, "already ingested")
        required = conn.execute(f"SELECT {S.required_coverage_sql('r')} FROM run r").fetchone()[0]
        old_rule = conn.execute("SELECT COALESCE(n_items_judged, 0) + COALESCE(n_judge_refusals, 0) FROM judgment").fetchone()[0]
        assert (required, old_rule) == (200, 198) and old_rule < required, "the pre-2026-09-21 arithmetic would have held this run"
        assert conn.execute(f"SELECT {S.judgment_covered_items_sql('jm')} FROM judgment jm").fetchone()[0] == 200
        # the one predicate: the grading queue stops offering the run the harvester ingested
        assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE run_id = ? AND is_instrument_judge = 1", (rid,)).fetchone()[0] == 0
    # a reader re-emitting the run's `judged` after the stamp folds (provenance), is held, and moves nothing
    _write(root, "c", "000002-reemit.jsonl", [_reemit(rid, "judged", 200)])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(250))
    assert rep.refused == 0 and rep.lifecycle_regressions_held == 1 and rep.harvester_records == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT lifecycle_state, ingested_utc FROM run").fetchone()) == ("ingested", synth.t(150))


def test_errors_on_the_judgments_own_metric_rows_count_when_the_judgment_did_not_count_them(tmp_path):
    """The judgment row counted no errors (`n_errors` NULL, the multi-turn readers' shape); its own metric rows carry
    `n_error_outcomes` = 10 on every row: 190 judged + 10 errored = 200 -> `ingested`."""
    rows = [{"threshold": 0.5, "n_error_outcomes": 10}, {"threshold": 0.7, "value": 0.1, "k": 19, "n_error_outcomes": 10}]
    root, rid, jid = _judged_root_with_judgment(tmp_path, "rmerr", {"n_items_judged": 190}, rows)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.harvester_records == 1
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "ingested"
        assert conn.execute("SELECT n_errors FROM judgment").fetchone()[0] is None
        assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE judgment_id = ? AND n_error_outcomes = 10", (jid,)).fetchone()[0] == 2


def test_the_judgments_own_error_count_wins_over_its_metric_rows(tmp_path):
    """A judgment that COUNTED zero errors is short even when a metric row carries `n_error_outcomes`: the judgment's own count is
    the authoritative one and the metric rows are read only when the judgment did not count."""
    root, rid, jid = _judged_root_with_judgment(tmp_path, "own", {"n_items_judged": 190, "n_errors": 0}, [{"n_error_outcomes": 10}])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.harvester_records == 0 and _state(root) == "judged"
    with DB.reading(DB.db_path(root)) as conn:
        ok, why = H.completion_requirements_hold(conn, rid)
        assert not ok and "190 < 200" in why


def test_incomplete_items_never_count_toward_coverage(tmp_path):
    """190 judged + 5 declined + 0 errored, and 5 items the producer filed as `n_incomplete`: 195 < 200, the run stays `judged`
    (an incomplete item has no outcome), the grading queue keeps offering it with the same 5 missing."""
    root, rid, jid = _judged_root_with_judgment(tmp_path, "inc", {"n_items_judged": 190, "n_judge_refusals": 5, "n_errors": 0},
                                                [{"n_incomplete": 5, "n_error_outcomes": 0}])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.harvester_records == 0 and _state(root) == "judged"
    with DB.reading(DB.db_path(root)) as conn:
        ok, why = H.completion_requirements_hold(conn, rid)
        assert not ok and "195 < 200" in why
        row = conn.execute("SELECT pending_reason, items_missing FROM v_pending_judgment WHERE run_id = ? AND is_instrument_judge = 1",
                           (rid,)).fetchone()
        assert tuple(row) == ("coverage_short", 5)
    # a further cycle derives nothing either: the rule is stable, not a one-cycle miss
    assert H.ingest_once(root, workers=0, min_free=0).harvester_records == 0 and _state(root) == "judged"


def test_blocked_queue_reevaluated_each_cycle(tmp_path):
    root = _root(tmp_path)
    rid, recs = synth.run_records("k", 100, through="running")
    _write(root, "c", "000001-run.jsonl", recs)          # no dimensions yet → the run is blocked on model
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.blocked >= 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    b = conn.execute("SELECT missing_parent_table, attempts FROM blocked_records WHERE record_type = 'run'").fetchone()
    assert b[0] == "model" and b[1] == 1
    conn.close()
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT attempts FROM blocked_records WHERE record_type = 'run'").fetchone()[0] == 2
    conn.close()
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rep = H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert tuple(conn.execute("SELECT lifecycle_state, node_hostname FROM run").fetchone()) == ("running", "TESTHOST")
    conn.close()


def test_inbox_merge_and_conflict(tmp_path):
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    inbox = os.path.join(root, "inbox", "cluster_a", "dims", "seed")
    R.write_records(os.path.join(inbox, "000001-dims.jsonl"), dims)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.ok - rep.arrival_records == len(dims)
    assert os.path.exists(os.path.join(root, "records", "dims", "seed", "000001-dims.jsonl"))
    assert not os.path.exists(os.path.join(inbox, "000001-dims.jsonl"))
    # a same-named file with DIFFERENT content is a producer bug: kept as .conflict-<sha8>, listed
    R.write_records(os.path.join(inbox, "000001-dims.jsonl"), dims[:2])
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.conflicts == 1
    conflicts = [f for f in os.listdir(os.path.join(root, "records", "dims", "seed")) if ".conflict-" in f]
    assert len(conflicts) == 1
    # a same-named file with the SAME content is simply dropped
    R.write_records(os.path.join(inbox, "000001-dims.jsonl"), dims)
    rep3 = H.ingest_once(root, workers=0, min_free=0)
    assert rep3.conflicts == 0 and not os.path.exists(os.path.join(inbox, "000001-dims.jsonl"))
    # THE LOGICAL FILENAME: after the harvester compressed the accepted record, `000001-dims.jsonl` and its `.gz` twin are ONE record --
    # a same-named arrival beside the twin is compared by LOGICAL sha, never installed as a second active form
    rep_gz = H.ingest_once(root, workers=0, min_free=0, gzip_min_age_s=0)
    plain = os.path.join(root, "records", "dims", "seed", "000001-dims.jsonl")
    assert rep_gz.gzipped >= 1 and not os.path.exists(plain) and os.path.exists(plain + ".gz")
    twin_sha = R.file_logical_sha(plain + ".gz")
    R.write_records(os.path.join(inbox, "000001-dims.jsonl"), dims)                 # the same logical content as the twin: a duplicate, dropped
    rep4 = H.ingest_once(root, workers=0, min_free=0)
    assert rep4.conflicts == 0 and not os.path.exists(os.path.join(inbox, "000001-dims.jsonl")) and not os.path.exists(plain)
    R.write_records(os.path.join(inbox, "000001-dims.jsonl"), dims[:3])             # other bytes: kept aside, the twin stays the one active form
    rep5 = H.ingest_once(root, workers=0, min_free=0)
    conflicts = sorted(f for f in os.listdir(os.path.join(root, "records", "dims", "seed")) if ".conflict-" in f)
    assert rep5.conflicts == 1 and len(conflicts) == 2 and not os.path.exists(plain), conflicts
    assert any(f.startswith("000001-dims.jsonl.conflict-") for f in conflicts) and R.file_logical_sha(plain + ".gz") == twin_sha
    # the symmetric case: an incoming `.gz`-named file beside an existing PLAIN form -- the same logical content a duplicate, other bytes aside
    inbox2 = os.path.join(root, "inbox", "cluster_a", "dims", "seed2")
    R.write_records(os.path.join(inbox2, "000002-dims.jsonl"), dims)
    H.ingest_once(root, workers=0, min_free=0)
    plain2 = os.path.join(root, "records", "dims", "seed2", "000002-dims.jsonl")
    assert os.path.exists(plain2)
    R.write_records(os.path.join(inbox2, "000002-dims.jsonl.gz"), dims, gz=True)
    rep6 = H.ingest_once(root, workers=0, min_free=0)
    assert rep6.conflicts == 0 and not os.path.exists(plain2 + ".gz") and not os.path.exists(os.path.join(inbox2, "000002-dims.jsonl.gz"))
    R.write_records(os.path.join(inbox2, "000002-dims.jsonl.gz"), dims[:2], gz=True)
    rep7 = H.ingest_once(root, workers=0, min_free=0)
    assert rep7.conflicts == 1 and not os.path.exists(plain2 + ".gz") and os.path.exists(plain2)
    assert any(f.startswith("000002-dims.jsonl.gz.conflict-") for f in os.listdir(os.path.join(root, "records", "dims", "seed2")))


def test_worker_pool_parse_matches_inline(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=2, n_items=10)
    paths = list(R.iter_record_files(os.path.join(root, "records")))
    a = H.parse_files(root, paths, workers=0)
    b = H.parse_files(root, paths, workers=2)
    assert [(p.uri, p.file_sha, p.n_records, p.first_key) for p in a] == [(p.uri, p.file_sha, p.n_records, p.first_key) for p in b]
    # item rows are built in the worker (F3): the same tuples inline and pooled, and the order of files is preserved
    assert [[pi.row for pi in p.items] for p in a] == [[pi.row for pi in p.items] for p in b]
    assert [p.uri for p in b] == [H.logical_uri(root, x) for x in paths]


def test_stat_sidecar_skips_hashing_unchanged_files_and_sees_changes(tmp_path, monkeypatch):
    """F12: the second cycle stats every known file and hashes none; an appended file is hashed and re-ingested."""
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=2, n_items=10)
    H.ingest_once(root, workers=0, min_free=0)
    side = json.load(open(os.path.join(root, "state", H.FILE_STAT_NAME)))
    assert side and all({"size", "mtime_ns", "ino", "sha"} <= set(v) for v in side.values())
    calls = []
    real = R.file_logical_sha

    def counting(path):
        calls.append(path)
        return real(path)
    monkeypatch.setattr(H.R, "file_logical_sha", counting)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert H.new_or_changed_files(root, conn) == []
    assert calls == []                                  # nothing decompressed on an idle cycle
    path = _write(root, "dims", "000002-more.jsonl", [synth.rec("si_text", {"si_id": "s2", "si_text_sha256": "2" * 64,
                                                                       "si_kind": "irrelevant_control", "source_uri": "synthetic://x"}, synth.t(1))])
    work = H.new_or_changed_files(root, conn)
    assert [w[0] for w in work] == [path] and calls == [path]
    conn.close()
    # a same-size, same-inode rewrite with a NEW mtime is re-hashed; unchanged bytes → not re-ingested
    os.utime(path, ns=(os.stat(path).st_atime_ns, os.stat(path).st_mtime_ns + 10 ** 9))
    H.ingest_once(root, workers=0, min_free=0)
    calls.clear()
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert H.new_or_changed_files(root, conn) == [] and calls == []
    conn.close()


def test_transactions_capped_by_rows_and_files_and_items_sorted_by_key(tmp_path):
    """F11 / F2: a 3 × 400-row item burst under a 500-row cap lands in 3 transactions; the batch is written in key order."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="judged")
    _write(root, "c", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    for i in range(3):
        items = synth.item_records(rid, jid, 200 + i, 400, seed=i)
        items.reverse()   # delivered in reverse key order
        _write(root, "c", f"00001{i}-items.jsonl.gz", items, gz=True)
    old_cap = H.ITEM_ROWS_PER_TRANSACTION
    H.ITEM_ROWS_PER_TRANSACTION = 500
    try:
        rep = H.ingest_once(root, workers=0, min_free=0)
    finally:
        H.ITEM_ROWS_PER_TRANSACTION = old_cap
    assert rep.ok - rep.arrival_records == 1200 and rep.duplicates == 0   # each later file's rows carry a greater order key → 400 updates, no duplicates
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 400
    # the live row is the greatest order key (the third file's stamp), whatever the delivery order
    assert conn.execute("SELECT MIN(written_utc) FROM item_result").fetchone()[0] == R.utc_to_epoch_ms(synth.t(204))
    conn.close()
    assert rep.transactions == 5 and rep.records == 1200 + rep.arrival_records   # three windows + the arrival facts' fold + the re-rank


def test_chunk_parsed_caps():
    def pf(n, k):
        p = H.ParsedFile(f"p{k}", f"records/x/{k:06d}-item_result.jsonl", "s")
        p.items = [H.ParsedItem(i, "%064x" % i, "2026-01-01T00:00:00.000Z", "item_result", (), (), (), "") for i in range(n)]
        return p
    chunks = list(H.chunk_parsed([pf(10, 1), pf(10, 2), pf(85, 3), pf(1, 4), pf(200, 5), pf(1, 6)], max_files=3, max_rows=100))
    # ≤ 3 files AND ≤ 100 rows per chunk; the 200-row file is written as two 100-row parts sharing its ledger row
    assert [[len(x.items) for x in c] for c in chunks] == [[10, 10], [85, 1], [100], [100], [1]]
    parts = [x for c in chunks for x in c if not x.is_single]
    assert [(x.part_index, x.last) for x in parts] == [(0, False), (1, True)] and all(x.total_records == 200 and x.uri == parts[0].uri for x in parts)
    assert parts[0].first_key == parts[1].first_key == ("2026-01-01T00:00:00.000Z", "%064x" % 0)
    # revision directories are routed to apply_revision, never through the chunker
    root = "/r"
    plain, dirs, dims = H.split_work(root, [(f"/r/records/adapters/ad/{'a' * 16}/000001-run.jsonl", None),
                                            (f"/r/records/adapters/ad/{'a' * 16}/000002-item_result.jsonl.gz", None),
                                            (f"/r/records/adapters/ad/_activations/{'b' * 16}/20260101/000001-run.jsonl", None),
                                            (f"/r/records/adapters/ad/_activations/{'b' * 16}/20260101-source_watermark.jsonl", None),
                                            ("/r/records/c/000001-run.jsonl", None)])
    assert dirs == [f"/r/records/adapters/ad/{'a' * 16}", f"/r/records/adapters/ad/_activations/{'b' * 16}/20260101"]
    assert [p_ for p_, _ in plain] == [f"/r/records/adapters/ad/_activations/{'b' * 16}/20260101-source_watermark.jsonl", "/r/records/c/000001-run.jsonl"]
    assert dims == []


def test_an_ingested_file_appended_after_its_fold_is_refused_as_a_replacement_and_nothing_of_it_folds(tmp_path):
    """An ingested record file is immutable. Until 2026-10-06 this test pinned the opposite -- the appended file was re-parsed and its
    new lines ingested as a CHANGED file (the same ledger row, its sha replaced) -- which is the silent replacement the L2c-replay probe
    measured. The appended bytes are now refused by name; the ledger row keeps its sha and count, the rows folded from the first bytes
    stand, and a correction is a NEW record file (tests/test_registry_immutable_ingested_file.py)."""
    root = _root(tmp_path)
    dims = synth.seed_dimensions()
    path = _write(root, "dims", "000001-dims.jsonl", dims[:5])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    sha0, n0 = tuple(conn.execute("SELECT file_sha256, n_records FROM record_ingest").fetchone())
    models0 = conn.execute("SELECT COUNT(*) FROM model").fetchone()[0]
    conn.close()
    R.write_records(path, dims[5:])  # appended lines: the file's sha changed
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.ok == 0 and rep.duplicates == 0 and rep.refused == 1 and rep.files_replaced_refused == 1
    assert rep.files == 2 and rep.refusal_facts == 1                    # the appended file and the cycle's refusal-fact file
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT file_sha256, n_records FROM record_ingest").fetchone()) == (sha0, n0) and n0 == 5
    assert conn.execute("SELECT COUNT(*) FROM model").fetchone()[0] == models0
    assert tuple(conn.execute("SELECT record_uri FROM record_refusal WHERE record_sha256 = ?", (R.file_logical_sha(path),)).fetchone()) == (H.logical_uri(root, path),)
    conn.close()
    # the same appended bytes again: already judged under this code, not listed, nothing refused twice
    rep2 = H.ingest_once(root, workers=0, min_free=0)
    assert rep2.files == 0 and rep2.refused == 0


def test_paper_head_state_file_written(tmp_path):
    root = _root(tmp_path)
    synth.write_corpus(root, n_runs=1, n_items=5)
    H.ingest_once(root, workers=0, min_free=0)
    assert os.path.exists(os.path.join(root, "state", "paper_head"))


# --------------------------------------------------------------------------------------------
# rule-scored runs (an instrument without a judge): the generation-time headline metric completes the run
# --------------------------------------------------------------------------------------------


def _rule_scored_root(tmp_path, **kw):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid, recs = synth.rule_scored_run_records("cap", 100, **kw)
    _write(root, "c", "000001-run.jsonl", recs)
    return root, rid


def _run_ingest_payloads(root):
    out = []
    for dp, _, fs in os.walk(os.path.join(root, "records", "harvester")):
        for f in fs:
            if f.startswith("run_ingest-"):
                with open(os.path.join(dp, f), "r", encoding="utf-8") as fh:
                    out += [json.loads(line)["payload"] for line in fh if line.strip()]
    return out


def test_rule_scored_run_is_ingested_from_its_headline_metric(tmp_path):
    """A run of an instrument without a judge never gets a judgment row; its generation-time headline metric with full
    coverage completes it: the harvester's own run_ingest stamps it ingested with judged_utc = the metric's computed_utc
    (rule scoring happens at generation), the record carries that scoring time, the run leaves v_in_flight and is pending
    for no judge; an LLM-scored generated run beside it is untouched (no judgment yet) and pending for its instrument's
    judge, and the artifact sha the readers mostly did not record is not required of the rule-scored run."""
    root, rid = _rule_scored_root(tmp_path)
    rid_llm, recs = synth.run_records("k", 300, through="generated")
    _write(root, "c", "000002-llm.jsonl", recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.harvester_records == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT lifecycle_state, generated_utc, judged_utc, ingested_utc, store_completion_sha256 FROM run WHERE run_id = ?",
                       (rid,)).fetchone()
    assert row["lifecycle_state"] == "ingested" and row["ingested_utc"] is not None
    assert row["judged_utc"] == synth.t(105) and row["store_completion_sha256"] is None
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_in_flight WHERE run_id = ?", (rid,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0
    assert H.completion_requirements_hold(conn, rid) == (False, "already ingested")
    llm = conn.execute("SELECT lifecycle_state, judged_utc, ingested_utc FROM run WHERE run_id = ?", (rid_llm,)).fetchone()
    assert tuple(llm) == ("generated", None, None)
    assert [r[0] for r in conn.execute("SELECT required_judge_id FROM v_pending_judgment WHERE run_id = ? AND is_instrument_judge = 1", (rid_llm,))] == [synth.JUDGE_ID]
    assert H.completion_requirements_hold(conn, rid_llm) == (False, "lifecycle_state=generated")
    conn.close()
    payloads = _run_ingest_payloads(root)
    assert len(payloads) == 1 and payloads[0]["run_id"] == rid and payloads[0]["judged_utc"] == synth.t(105)
    assert set(payloads[0]) == {"run_id", "ingested_utc", "judged_utc"}
    # a further cycle derives nothing: the run is ingested
    assert H.ingest_once(root, workers=0, min_free=0).harvester_records == 0


def test_rule_scored_run_short_of_coverage_stays_generated_until_the_metric_covers(tmp_path):
    root, rid = _rule_scored_root(tmp_path, n_scored=150)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.harvester_records == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT lifecycle_state, judged_utc, ingested_utc FROM run").fetchone()) == ("generated", None, None)
    ok, why = H.completion_requirements_hold(conn, rid)
    assert not ok and "150 < 200" in why and "rule-scored" in why
    assert conn.execute("SELECT COUNT(*) FROM v_in_flight WHERE run_id = ?", (rid,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE run_id = ?", (rid,)).fetchone()[0] == 0
    conn.close()
    # the producer re-files the headline metric with full coverage → ingested on the next cycle, judged at that scoring time
    _write(root, "c", "000002-metric.jsonl", [synth.rec("run_metric", {"run_id": rid, "metric_id": "acc", "value": 0.5, "k": 100, "n": 200,
                                                                      "computed_by": "producer", "computed_utc": synth.t(120)}, synth.t(120))])
    assert H.ingest_once(root, workers=0, min_free=0).harvester_records == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT lifecycle_state, judged_utc FROM run").fetchone()
    assert row[0] == "ingested" and row[1] == synth.t(120)
    conn.close()


def test_rule_scored_run_without_a_headline_metric_stays_generated(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid, recs = synth.rule_scored_run_records("cap", 100, with_artifact=True)
    _write(root, "c", "000001-run.jsonl", [r for r in recs if r["record_type"] != "run_metric"])
    assert H.ingest_once(root, workers=0, min_free=0).harvester_records == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run").fetchone()[0] == "generated"
    ok, why = H.completion_requirements_hold(conn, rid)
    assert not ok and why.startswith("no headline metric row")
    conn.close()


def test_rule_scored_judged_utc_falls_back_to_generated_utc(tmp_path):
    root, rid = _rule_scored_root(tmp_path, computed_utc=False)
    assert H.ingest_once(root, workers=0, min_free=0).harvester_records == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT lifecycle_state, generated_utc, judged_utc FROM run").fetchone()
    assert row[0] == "ingested" and row[2] == row[1] == synth.t(103)
    conn.close()


def test_rule_scored_ingest_replays_deterministically(tmp_path):
    """The run_ingest record carries the scoring time, so a rebuild from the log lands the same row (state, judged_utc,
    ingested_utc) as the live fold — the record, never a re-derivation, is what replay reads."""
    root, rid = _rule_scored_root(tmp_path)
    assert H.ingest_once(root, workers=0, min_free=0).harvester_records == 1
    q = "SELECT lifecycle_state, generated_utc, judged_utc, ingested_utc FROM run WHERE run_id = ?"
    with DB.reading(DB.db_path(root)) as conn:
        live = tuple(conn.execute(q, (rid,)).fetchone())
    into = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, into, workers=0)
    with DB.reading(into) as conn:
        replay = tuple(conn.execute(q, (rid,)).fetchone())
    assert live == replay and live[0] == "ingested" and live[2] == synth.t(105)


# --------------------------------------------------------------------------------------------
# the lifecycle is monotonic in the live fold, and the ingest stamp re-fires for a row whose lifecycle fell behind it
# --------------------------------------------------------------------------------------------


def _reemit(rid, state, i):
    """What a legacy-store reader sends again at a new EMISSION_VERSION: the run's final run_state, stamped now."""
    return synth.rec("run_state", {"run_id": rid, "state": state, "event_utc": synth.t(i), "evidence_kind": "completion_artifact"}, synth.t(i))


def test_reemitted_run_state_after_the_ingest_stamp_is_held_and_counted(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, recs = synth.run_records("k", 100, through="judged")
    _write(root, "c", "000001-run.jsonl", recs)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(150))
    assert rep.harvester_records == 1 and rep.lifecycle_regressions_held == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT lifecycle_state, ingested_utc FROM run").fetchone()) == ("ingested", synth.t(150))
    # the reader re-emits the run's `judged` later than the stamp: folded, held, counted; the row does not move
    _write(root, "c", "000002-reemit.jsonl", [_reemit(rid, "judged", 200)])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(250))
    assert rep.refused == 0 and rep.ok - rep.arrival_records == 1 and rep.lifecycle_regressions_held == 1
    assert rep.harvester_records == 0, "nothing to re-stamp: the row never left `ingested`"
    with DB.reading(DB.db_path(root)) as conn:
        row = conn.execute("SELECT lifecycle_state, ingested_utc, judged_utc FROM run").fetchone()
        assert tuple(row) == ("ingested", synth.t(150), synth.t(107))
        # the held event is in the history (provenance) with the unchanged state folded after it
        hist = conn.execute("SELECT json_extract(payload_json, '$.state'), json_extract(folded_state_json, '$.lifecycle_state') "
                            "FROM row_history WHERE table_name = 'run' AND written_utc = ?", (synth.t(200),)).fetchall()
        assert [tuple(h) for h in hist] == [("judged", "ingested")]
        assert conn.execute("SELECT COUNT(*) FROM run_state_event WHERE run_id = ? AND state = 'judged'", (rid,)).fetchone()[0] == 2
    assert H.ingest_once(root, workers=0, min_free=0, now=synth.t(300)).lifecycle_regressions_held == 0
    assert len(_run_ingest_payloads(root)) == 1


def test_ingest_stamp_refires_for_a_row_whose_lifecycle_fell_behind_its_ingested_utc(tmp_path, monkeypatch):
    """The live defect of 2026-09-21: rows folded BEFORE the fold held backwards events sit at `judged` / `generated` with
    an `ingested_utc`. The stamp's candidates are the rows not currently `ingested` whose completion holds — never
    `ingested_utc IS NULL` — so the next cycle stamps every such row again, once; a row at `ingested` and a `superseded`
    row are not candidates."""
    from sisteer.registry import reducer as RED
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid_llm, recs_llm = synth.run_records("k", 100, through="judged")          # LLM-scored: judged → ingested, then regressed
    rid_cap, recs_cap = synth.rule_scored_run_records("cap", 100)              # rule-scored: generated → ingested, then regressed
    rid_ok, recs_ok = synth.run_records("k2", 100, through="judged")           # stays ingested: never stamped twice
    rid_sup, recs_sup = synth.run_records("k3", 100, through="judged")         # ingested, then superseded: never re-stamped
    _write(root, "c", "000001-run.jsonl", recs_llm + recs_cap + recs_ok + recs_sup)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(150))
    assert rep.refused == 0 and rep.harvester_records == 4
    # the pre-fix fold: the guard disabled, the re-emitted states move the rows backwards (derive_facts=False keeps this
    # cycle from stamping them straight back, so the regressed state — the live registry's — is observable)
    monkeypatch.setattr(RED, "lifecycle_regresses", lambda current, target: False, raising=False)
    _write(root, "c", "000002-reemit.jsonl", [_reemit(rid_llm, "judged", 200), _reemit(rid_cap, "generated", 200), _reemit(rid_sup, "superseded", 200)])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(250), derive_facts=False)
    assert rep.refused == 0 and rep.ok - rep.arrival_records == 3
    monkeypatch.undo()
    with DB.reading(DB.db_path(root)) as conn:
        q = lambda rid: tuple(conn.execute("SELECT lifecycle_state, ingested_utc FROM run WHERE run_id = ?", (rid,)).fetchone())
        assert q(rid_llm) == ("judged", synth.t(150)) and q(rid_cap) == ("generated", synth.t(150))
        assert q(rid_ok) == ("ingested", synth.t(150)) and q(rid_sup) == ("superseded", synth.t(150))
        assert H.completion_requirements_hold(conn, rid_llm) == (True, "ok"), "complete, and not currently ingested"
        assert H.completion_requirements_hold(conn, rid_cap) == (True, "ok")
        assert H.completion_requirements_hold(conn, rid_ok) == (False, "already ingested")
        assert H.completion_requirements_hold(conn, rid_sup)[0] is False
    # the next cycle stamps the two regressed rows again — and only those
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(300))
    assert rep.harvester_records == 2 and rep.refused == 0 and rep.lifecycle_regressions_held == 0
    with DB.reading(DB.db_path(root)) as conn:
        rows = {r[0]: tuple(r[1:]) for r in conn.execute("SELECT run_id, lifecycle_state, ingested_utc, judged_utc, generated_utc FROM run")}
    assert rows[rid_llm] == ("ingested", synth.t(300), synth.t(107), synth.t(103))
    assert rows[rid_cap] == ("ingested", synth.t(300), synth.t(105), synth.t(103)), "the scoring time the first stamp carried stands"
    assert rows[rid_ok] == ("ingested", synth.t(150), synth.t(107), synth.t(103)), "a row at `ingested` is not stamped again"
    assert rows[rid_sup][:2] == ("superseded", synth.t(150)), "a superseded row is never stamped back to `ingested`"
    payloads = _run_ingest_payloads(root)
    assert len(payloads) == 6 and sorted(p["run_id"] for p in payloads if p["ingested_utc"] == synth.t(300)) == sorted([rid_cap, rid_llm])
    assert all(set(p) == {"run_id", "ingested_utc"} for p in payloads if p["ingested_utc"] == synth.t(300)), "judged_utc already stands on both rows"
    # nothing left to stamp, and a rebuild folds the same rows
    assert H.ingest_once(root, workers=0, min_free=0, now=synth.t(350)).harvester_records == 0
    rebuilt = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, rebuilt, workers=0)
    with DB.reading(DB.db_path(root)) as a, DB.reading(rebuilt) as b:
        diff = H.verify_equal(a, b)
    assert diff["run"] == 0 and diff["run_state_event"] == 0, {k: v for k, v in diff.items() if v}


# --------------------------------------------------------------------------------------------
# the fold-order evidence rule (reducer.LifecycleEvidenceMissing)
# --------------------------------------------------------------------------------------------

def test_an_ingest_stamp_ahead_of_its_judged_evidence_in_fold_order_is_blocked_not_refused_and_folds_when_the_evidence_lands(tmp_path):
    """The R3 bootstrap refused 128 `run_ingest` records with `trg_run_state_utc`: each was retried over a history holding only events
    stamped AFTER it (its first carrier activated later), so the refold put `ingested` first, held every later `judged` as a regression,
    and the upsert's INSERT half carried no `judged_utc` -- while `_missing_lifecycle_evidence` had passed on the ROW, which showed the
    column. One key, no replay: a run folded from records stamped at t(400..); a stamp written at t(200) -- earlier than every record of
    the key -- is BLOCKED on `run_state_event <run>:running` (the first utc the fold lacks), never refused, and the row stands; the
    evidence stamped before the stamp arrives (the run's records re-emitted at t(100..)) and the next cycle's blocked-queue pass folds
    the stamp: `ingested`, every utc set, the later `judged` events held as regressions, no refusal anywhere."""
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    rid, late = synth.run_records("k", 400, through="judged")
    _write(root, "late", "000001-run.jsonl", late)
    assert H.ingest_once(root, workers=0, min_free=0, now=synth.t(450), derive_facts=False).refused == 0
    stamp = H.write_harvester_record(root, "run_ingest", {"run_id": rid, "ingested_utc": synth.t(200)}, synth.t(200))
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(460), derive_facts=False)
    assert rep.refused == 0 and rep.blocked == 1, (rep.refused, rep.blocked, rep.refusals)
    with DB.reading(DB.db_path(root)) as conn:
        assert tuple(conn.execute("SELECT lifecycle_state, ingested_utc FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("judged", None)
        assert conn.execute("SELECT COUNT(*) FROM record_refusal").fetchone()[0] == 0
        blocked = conn.execute("SELECT record_uri, missing_parent_table, missing_parent_id FROM blocked_records").fetchall()
        assert [tuple(b) for b in blocked] == [(H.logical_uri(root, stamp) + "#1", "run_state_event", f"{rid}:running")], blocked
        assert tuple(conn.execute("SELECT status, n_blocked FROM record_delivery WHERE file_uri = ?", (H.logical_uri(root, stamp),)).fetchone()) == ("blocked", 1)
    # the evidence that precedes the stamp in fold order lands: the key's early records
    _, early = synth.run_records("k", 100, through="judged")
    _write(root, "early", "000001-run.jsonl", early)
    rep2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(470), derive_facts=False)
    assert rep2.refused == 0 and rep2.unblocked == 1, (rep2.refused, rep2.unblocked, rep2.refusals)
    with DB.reading(DB.db_path(root)) as conn:
        row = conn.execute("SELECT lifecycle_state, queued_utc, started_utc, generated_utc, judged_utc, ingested_utc FROM run WHERE run_id = ?", (rid,)).fetchone()
        assert tuple(row) == ("ingested", synth.t(100), synth.t(101), synth.t(103), synth.t(107), synth.t(200)), tuple(row)
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM record_refusal").fetchone()[0] == 0
    assert rep2.lifecycle_regressions_held >= 1          # the t(400..) `judged` and the states before it fold after the stamp and are held
    # and the replay of the same log lands the same row
    rebuilt = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, rebuilt, workers=0)
    with DB.reading(DB.db_path(root)) as a, DB.reading(rebuilt) as b:
        diff = {k: v for k, v in H.verify_equal(a, b).items() if v}
        assert not diff, diff
        assert b.execute("SELECT COUNT(*) FROM record_refusal").fetchone()[0] == 0


def test_lifecycle_evidence_missing_names_the_first_absent_state_in_progression_order():
    from sisteer.registry import reducer as RED
    assert RED.lifecycle_evidence_missing({"lifecycle_state": "ingested", "queued_utc": "q", "started_utc": "s", "generated_utc": "g",
                                           "judged_utc": "j", "ingested_utc": "i"}) == []
    assert RED.lifecycle_evidence_missing({"lifecycle_state": "ingested", "queued_utc": "q", "started_utc": "s", "generated_utc": "g",
                                           "ingested_utc": "i"}) == ["judged_utc"]
    assert RED.lifecycle_evidence_missing({"lifecycle_state": "ingested", "queued_utc": "q", "ingested_utc": "i"}) == ["started_utc", "generated_utc", "judged_utc"]
    assert RED.lifecycle_evidence_missing({"lifecycle_state": "failed", "queued_utc": "q"}) == ["finished_utc"]
    assert RED.lifecycle_evidence_missing({}) == []
    exc = RED.LifecycleEvidenceMissing("r1", "ingested", ["started_utc", "judged_utc"])
    assert exc.parent() == ("run_state_event", "r1:running") and not isinstance(exc, ValueError)
    assert "ingested" in str(exc) and "started_utc" in str(exc) and "blocked" in str(exc)
