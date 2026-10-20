"""The replay's revision path made cheap without changing what it writes.

The live shape of 2026-09-17 — a log of ~2,500 adapter revision directories whose dependencies (the dimension seeds)
had not been loaded, replayed with `registry rebuild` — ran for hours: every staging window re-ranked every seq-bearing
table (revision files sort before the plain files a replay applies first), every file paid three full scans of the
item tables for a staging check whose answer the ledger already knew, every run-scale file was streamed twice, and every
held revision was re-attempted whole each pass though its blocker had not moved. Each fix here is pinned by a test that
asserts the OBSERVABLE it removes (a re-rank per window, a second streamed pass, an activation attempt, a table scan)
AND that the replay still equals the live database row for row.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import snapshot as SN
from sisteer.registry import synth
from sisteer.registry.adapters import runner as RUN

from test_registry_core_fixes import (JUDGE_ID_3, JudgeNamingAdapter, StoreAdapter, _adapter_setup, _ingest, _judge3, _live_items,
                                      _pass, _q, _rebuild_equal, _store, _write)


def _si(i):
    return synth.rec("si_text", {"si_id": f"late_si_{i}", "si_text_sha256": ("%064x" % (0xe0 + i)), "si_kind": "irrelevant_control",
                                 "source_uri": "synthetic://x"}, synth.t(i))


def _verify_equal(root, out):
    with DB.reading(DB.db_path(root)) as a, DB.reading(out) as b:
        diff = H.verify_equal(a, b)
        bad = {k: v for k, v in diff.items() if v}
        ledger = "SELECT file_uri, file_sha256, first_written_utc, n_records, ingest_seq FROM record_ingest ORDER BY ingest_seq"
        same_ledger = a.execute(ledger).fetchall() == b.execute(ledger).fetchall()
    return bad, same_ledger


# ---- (1) ranks are made dense ONCE, at the end of a replay ---------------------------------------------------------------

def test_replay_ranks_once_at_the_end_and_equals_live_when_revision_files_sort_before_plain_ones(tmp_path, monkeypatch):
    """Revision directories and activation bundles stamped EARLIER than plain files the plain phase has already ranked:
    every staging window used to re-rank every seq-bearing table. Now no non-final finalize does any work, exactly one
    final one does, and the ledger, `ingest_seq` on every row and every compared table equal the live database."""
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4]),
                                                 ("s2", "store_2", "none", [0.1, 0.9])])
    ad = StoreAdapter(store_root)
    rep0, ing0 = _pass(root, ad, synth.t(2000))                                  # two revision directories stamped t2000
    assert rep0.groups_refreshed == 2 and ing0.refused == 0
    _write(root, "late", "000001-late.jsonl", [_si(2500)])                       # a plain file stamped later
    _ingest(root)
    with DB.reading(DB.db_path(root)) as conn:
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), reread=True)   # activation bundles stamped t3000
    assert rep1.groups_reread == 2
    _ingest(root)
    _write(root, "late", "000002-late.jsonl", [_si(3500)])
    _ingest(root)
    calls = []
    real = H.Ranker.finalize

    def spy(self, final=False):
        n = real(self, final=final)
        calls.append((self.defer, final, n))
        return n
    monkeypatch.setattr(H.Ranker, "finalize", spy)
    out = str(tmp_path / "rebuilt.sqlite")
    rep = H.rebuild(root, out, workers=0)
    replay = [c for c in calls if c[0]]                                            # the replay's ranker only
    assert replay and all(n == 0 for _, final, n in replay if not final), replay   # no per-window re-rank
    finals = [c for c in replay if c[1]]
    assert len(finals) == 1 and finals[0][2] > 0                                   # exactly one dense pass, and it had work
    assert rep.reranks > 0                                                         # the inversions were seen and counted
    bad, same_ledger = _verify_equal(root, out)
    assert not bad and same_ledger, bad
    with DB.reading(out) as b:
        assert b.execute("SELECT COUNT(*) FROM source_status").fetchone()[0] == 2
        # ingest_seq on every seq-bearing table points into the dense ledger
        for t in H.SEQ_TABLES:
            orphan = b.execute(f"SELECT COUNT(*) FROM {t} WHERE ingest_seq NOT IN (SELECT ingest_seq FROM record_ingest) AND ingest_seq <> 0").fetchone()[0]
            assert orphan == 0, t


def test_deferred_ranker_replay_of_a_plain_corpus_equals_live(tmp_path):
    """A plain-files-only corpus (no revision directory): the deferred ranker's one final pass gives the same dense
    ledger the live ingest built incrementally (the peeked stamp orders a chunk, the exact first key orders the ledger,
    so even here a file can sort before the head and the final pass has work)."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=2, n_items=3)
    _ingest(root)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    bad, same_ledger = _verify_equal(root, out)
    assert not bad and same_ledger


# ---- (2) the replay lands whole or not at all --------------------------------------------------------------------------

def test_rebuild_writes_a_building_file_and_renames_it_into_place_only_when_complete(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=4)
    _ingest(root)
    out = str(tmp_path / "rebuilt.sqlite")
    seen = {}
    real = H.Ingestor.reevaluate_blocked

    def spy(self, now=None):
        seen["building"] = os.path.exists(out + DB.REBUILD_TMP_SUFFIX)
        seen["into"] = os.path.exists(out)
        seen["journal_mode"] = self.conn.execute("PRAGMA journal_mode").fetchone()[0]
        seen["synchronous"] = self.conn.execute("PRAGMA synchronous").fetchone()[0]
        return real(self, now)
    monkeypatch.setattr(H.Ingestor, "reevaluate_blocked", spy)
    H.rebuild(root, out, workers=0)
    assert seen == {"building": True, "into": False, "journal_mode": "memory", "synchronous": 0}   # mid-build: the temp file only
    assert os.path.exists(out) and not os.path.exists(out + DB.REBUILD_TMP_SUFFIX)
    with DB.reading(out) as b:
        assert b.execute("PRAGMA journal_mode").fetchone()[0] == "wal"                            # the mode every connection expects
        assert b.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == 4
    bad, same_ledger = _verify_equal(root, out)
    assert not bad and same_ledger


def test_failed_rebuild_leaves_neither_the_database_nor_its_building_file(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=2)
    _ingest(root)
    out = str(tmp_path / "rebuilt.sqlite")

    def boom(self, now=None):
        raise RuntimeError("synthetic failure inside the replay")
    monkeypatch.setattr(H.Ingestor, "reevaluate_blocked", boom)
    with pytest.raises(RuntimeError):
        H.rebuild(root, out, workers=0)
    assert not os.path.exists(out)
    assert not any(os.path.exists(out + DB.REBUILD_TMP_SUFFIX + s) for s in ("", "-wal", "-shm", "-journal"))
    # a stale .building file from an interrupted build never blocks the next one
    open(out + DB.REBUILD_TMP_SUFFIX, "wb").write(b"garbage")
    monkeypatch.undo()
    H.rebuild(root, out, workers=0)
    assert os.path.exists(out) and not os.path.exists(out + DB.REBUILD_TMP_SUFFIX)


def test_footprint_watcher_counts_a_database_still_being_built(tmp_path):
    """The design-scale gate watches `into`; a build in flight lives at `into.building` until the rename."""
    into = str(tmp_path / "x.sqlite")
    open(into + DB.REBUILD_TMP_SUFFIX, "wb").write(b"\0" * 4096)
    w = SN.FootprintWatcher([into], 1.0)
    assert w.sample() == 4096 and w.peak_breakdown[into] == 4096
    os.remove(into + DB.REBUILD_TMP_SUFFIX)
    open(into, "wb").write(b"\0" * 8192)
    assert w.sample() == 8192


def test_replay_keeps_only_the_withdrawn_mark_index_and_rebuilds_the_rest(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=2)
    _ingest(root)
    out = str(tmp_path / "rebuilt.sqlite")
    kept_during = {}
    real = H.Ingestor.reclaim

    def spy(self, now):
        kept_during["idx"] = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL")}
        return real(self, now)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(H.Ingestor, "reclaim", spy)
        H.rebuild(root, out, workers=0)
    assert kept_during["idx"] == set(H.REBUILD_KEEP_INDEXES)                        # only the kept ones exist during the load
    with DB.reading(DB.db_path(root)) as a, DB.reading(out) as b:
        q = "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL ORDER BY name"
        assert a.execute(q).fetchall() == b.execute(q).fetchall()                   # every index is back afterwards


# ---- (3) a run-scale-only file of a revision is streamed once; a first staging never scans the item tables ------------

def test_revision_files_without_items_are_read_once_and_a_first_staging_scans_no_item_table(tmp_path, monkeypatch):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    passes = []
    real = H.stream_parts

    def spy(root_, path, sha, max_rows, want="all", skip=0, keep_raw=False):
        passes.append((os.path.basename(path), want))
        return real(root_, path, sha, max_rows, want=want, skip=skip, keep_raw=keep_raw)
    monkeypatch.setattr(H, "stream_parts", spy)
    scans = []
    conn = DB.open_or_create(DB.db_path(root))
    conn.set_trace_callback(lambda sql: scans.append(sql) if sql.startswith("SELECT 1 FROM item_") and "source_revision_id = " in sql else None)
    try:
        ing = H.ingest_once(root, conn=conn, workers=0, min_free=0)
    finally:
        conn.set_trace_callback(None)
        conn.close()
    assert ing.refused == 0 and ing.revisions_held == 0
    def kind(name):
        return R.RECORD_FILE_RE.match(name).group("type")
    items_passes = sorted(name for name, want in passes if want == "items")
    # the items pass reads exactly the files that carry item-scale records (item_result; source_status is item-scale too) …
    assert items_passes and all(kind(n) in R.ITEM_SCALE_RECORD_TYPES for n in items_passes), passes
    run_scale_files = sorted(name for name, want in passes if want == "run_scale")
    # … and never the run-scale-only ones, which every batch file was read for once
    assert {kind(n) for n in run_scale_files} >= {"artifact", "judgment", "run", "run_state", "run_metric", "item_result"}
    assert not {n for n in run_scale_files if kind(n) not in R.ITEM_SCALE_RECORD_TYPES} & set(items_passes)
    assert scans == [], scans[:3]                                                            # a never-ranked file: nothing to clear
    # bookkeeping of the files read once is complete: every batch file has its staging-derived delivery row and ledger row
    n_files = len([f for f in os.listdir(os.path.join(root, "records", "adapters", ad.name, rev))])
    assert _q(root, "SELECT COUNT(*) FROM record_ingest WHERE file_uri LIKE ?", f"records/adapters/{ad.name}/{rev}/%")[0][0] == n_files
    assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE ? AND status = 'ok'", f"records/adapters/{ad.name}/{rev}/%")[0][0] == n_files
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}
    _rebuild_equal(root, tmp_path)


def test_selective_read_yields_no_sha_for_skipped_records_and_the_same_sha_for_kept_ones(tmp_path):
    path = R.write_records(str(tmp_path / "000001-mixed.jsonl"), [_si(1), _si(2)])
    full = list(R.read_record_file(path))
    lazy = list(R.read_record_file(path, compute_sha=False))
    assert [s for _, _, s in lazy] == [None, None] and [r for _, r, _ in lazy] == [r for _, r, _ in full]
    parts = list(H.stream_parts(str(tmp_path), path, None, 10, want="run_scale"))
    assert [pr.sha for pr in parts[-1].records] == [s for _, _, s in full]                  # kept records carry the real sha


# ---- (3b) a replay with workers parses the revisions' item files ahead, in the pool; the rows are the same -------------

def test_replay_with_workers_prefetches_item_files_in_the_pool_and_equals_the_streamed_replay(tmp_path, monkeypatch):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6]),
                                                 ("s2", "store_2", "none", [0.1, 0.9, 0.3]),
                                                 ("s3", "store_3", "constitution_lora@aaaa", [0.5])])
    ad = StoreAdapter(store_root)
    rep0, ing0 = _pass(root, ad, synth.t(2000))
    assert rep0.groups_refreshed == 3 and ing0.refused == 0
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9])            # a second revision of one source
    rep1, ing1 = _pass(root, ad, synth.t(3000))
    assert rep1.groups_refreshed == 1 and ing1.refused == 0
    streamed_items = []
    real = H.stream_parts

    def spy(root_, path, sha, max_rows, want="all", skip=0, keep_raw=False):
        if want == "items":
            streamed_items.append(os.path.basename(path))
        return real(root_, path, sha, max_rows, want=want, skip=skip, keep_raw=keep_raw)
    monkeypatch.setattr(H, "stream_parts", spy)
    out0 = str(tmp_path / "rebuilt0.sqlite")
    H.rebuild(root, out0, workers=0)                                                        # streamed items pass
    n_streamed_inline = len(streamed_items)
    assert n_streamed_inline == 8                                                           # item_result + source_status (item-scale) per directory
    streamed_items.clear()
    out3 = str(tmp_path / "rebuilt3.sqlite")
    rep = H.rebuild(root, out3, workers=3)                                                  # prefetched in the pool
    assert streamed_items == []                                                             # no in-process items pass at all
    assert rep.refused == 0 and rep.revisions_held == 0
    for out in (out0, out3):
        bad, same_ledger = _verify_equal(root, out)
        assert not bad and same_ledger, (out, bad)
    with DB.reading(out0) as a, DB.reading(out3) as b:
        # the two replays agree on every row, the arrival ledger included (its clock columns excepted: each replay read its own)
        for t, cols in (("item_result", "*"), ("record_ingest", "*"), ("source_revision", "*"), ("revision_staging_file", "*"),
                        ("record_delivery", "file_uri, file_sha256, status, refusal_reason, producer_script, producer_session, schema_version, "
                                            "record_type_counts_json, n_records, n_ok, n_refused, n_blocked, n_duplicate")):
            q = f"SELECT {cols} FROM {t} ORDER BY 1, 2, 3"
            assert [tuple(r) for r in a.execute(q).fetchall()] == [tuple(r) for r in b.execute(q).fetchall()], t


def test_prefetcher_hands_out_each_file_once_in_order_and_none_for_an_unknown_one(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    paths = [R.write_records(str(tmp_path / f"{i:06d}-item_result.jsonl"), [_si(i)]) for i in range(1, 7)]   # any records will do
    with ProcessPoolExecutor(max_workers=2) as ex:
        pf = H._Prefetcher(str(tmp_path), ex, paths, window=2)
        assert len(pf.queue) == 2 and len(pf.pending) == 4
        assert pf.take(paths[0]).uri.endswith("000001-item_result.jsonl")
        assert pf.take(paths[2]).uri.endswith("000003-item_result.jsonl")                 # the passed-over file is dropped
        assert pf.take(paths[1]) is None                                                    # and cannot be taken later
        assert pf.take(str(tmp_path / "never-submitted.jsonl")) is None
        assert pf.take(paths[5]).uri.endswith("000006-item_result.jsonl") and not pf.queue and not pf.pending
    # the file selection: item-typed names only, below the streaming size
    d = tmp_path / "rev"
    d.mkdir()
    for name in ("000001-run.jsonl", "000002-item_result.jsonl.gz", "000003-item_flags.jsonl", "000004-source_status.jsonl", "999998-source_revision.jsonl"):
        (d / name).write_bytes(b"")
    assert [os.path.basename(p) for p in H._Prefetcher.item_files([str(d)])] == ["000002-item_result.jsonl.gz", "000003-item_flags.jsonl", "000004-source_status.jsonl"]


# ---- (4) a held revision whose blocker has not arrived is not re-attempted ---------------------------------------------

def test_held_revision_is_skipped_while_its_blocker_is_absent_and_activates_when_it_lands(tmp_path, monkeypatch):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rid = I.run_id(synth.CAMPAIGN, "store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.9, 0.9, 0.9])
    ad2 = JudgeNamingAdapter(store_root, JUDGE_ID_3)                       # the new revision's judgment names an absent judge
    rep, ing = _pass(root, ad2, synth.t(3000))
    rev2 = rep.results[0].revision_id
    # held on the first attempt; the cycle's own blocked-queue pass finds its attempt memo standing (the same files, marker, code and
    # absent parent, recorded moments ago) and skips it -- the held count is once per cycle, the skip counted beside it (before the fix
    # the queue pass re-attempted and counted the same hold a second time: 2)
    assert ing.revisions_held == 1 and ing.revisions_held_skipped == 1 and ing.revision_transactions == 1
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    marker = _q(root, "SELECT attempts, last_checked_utc, missing_parent_table, missing_parent_id FROM blocked_records "
                      "WHERE record_type = 'source_revision' AND json_extract(extra_json, '$.revision_id') = ?", rev2)
    assert len(marker) == 1 and marker[0][2:] == ("judge", JUDGE_ID_3)
    attempts0 = marker[0][0]
    activations = []
    real = H.Ingestor._activate_revision

    def spy(self, rev, *a, **k):
        activations.append(rev)
        return real(self, rev, *a, **k)
    monkeypatch.setattr(H.Ingestor, "_activate_revision", spy)
    # nothing new: the re-attempt is skipped (no activation attempt), the revision stays held, its bookkeeping advances
    ing2 = _ingest(root, now=synth.t(3100))
    assert activations == [] and ing2.revisions_held == 1 and ing2.revision_transactions == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(0,)]
    marker2 = _q(root, "SELECT attempts, last_checked_utc FROM blocked_records WHERE record_type = 'source_revision' "
                       "AND json_extract(extra_json, '$.revision_id') = ?", rev2)
    assert marker2[0][0] == attempts0 + 1 and marker2[0][1] == synth.t(3100)
    assert _live_items(root, rid) == {"q0": 0.2, "q1": 0.8, "q2": 0.4}
    # the blocker lands: the same cycle re-attempts and activates
    _write(root, "dims", "000002-judge3.jsonl", [_judge3()])
    ing3 = _ingest(root, now=synth.t(3200))
    assert activations == [rev2] and ing3.refused == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    assert _live_items(root, rid) == {"q0": 0.9, "q1": 0.9, "q2": 0.9}
    assert _q(root, "SELECT COUNT(*) FROM blocked_records")[0][0] == 0
    _rebuild_equal(root, tmp_path)


def test_parent_presence_is_checked_under_every_referencing_column(tmp_path):
    """`si_text` is referenced by id AND by sha; a blocker recorded as (`si_text`, <sha>) is present when a row carries
    that sha under either column — the skip declares a parent absent only when no column carries it."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _ingest(root)
    conn = DB.open_or_create(DB.db_path(root))
    try:
        ing = H.Ingestor(conn, root)
        sha = _q(root, "SELECT si_text_sha256 FROM si_text LIMIT 1")[0][0]
        sid = _q(root, "SELECT si_id FROM si_text LIMIT 1")[0][0]
        assert not ing._parent_still_absent("si_text", sha) and not ing._parent_still_absent("si_text", sid)
        assert ing._parent_still_absent("si_text", "f" * 64) and ing._parent_still_absent("model", "no-such-model")
        assert ing._parent_still_absent("source_revision", "0" * 16)
        assert not ing._parent_still_absent("run_state_event", "r:generated")   # a pseudo-parent: never declared absent
        assert not ing._parent_still_absent("record", "records/x#1")
    finally:
        conn.close()


# ---- (5) the design-scale app's shape: its test left with the Modal stub it imported (the app under
# scripts/modal/ is frozen Modal code outside the registry's ownership and is not exercised from here)
