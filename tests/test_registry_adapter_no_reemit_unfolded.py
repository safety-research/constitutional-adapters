"""The runner treats a fold that has not happened as PENDING, never as absent (2026-09-29).

The live shape: ``regrade_twin`` held 97 published revisions (``accepted = 0``, each blocked on a run its judgment named) whose
inputs carried a REFUSED watermark stamp from 2026-09-17 with no attempt id. Every 10-minute pass planned the 97 groups on that
stamp and, the revision directory being published, re-read each of them fresh as an activation bundle
``_activations/<rev>/<stamp>/`` (~18k records apiece); the harvester could not fold 97 new bundles inside its 45-minute unit
bound, was killed, and the next pass -- reading a database the fold had never reached -- re-emitted all 97 again: 17 passes, four
harvester cycles killed in a row, the paper frame frozen for four hours. Three rules end it, each pinned here:

1. a revision whose marker (its directory's, or a complete bundle's) is on the record log with no ``record_delivery`` row is
   PENDING FOLD (``runner.pending_fold``): nothing is written for it whatever selected the group, the pass reports
   ``activations_pending``, and the harvester's next cycle folds it -- on the grading host pass, on a ``--shard`` pass, and on a cluster
   shard reading the reader-state sidecar;
2. a HELD revision is re-read only by an explicit ``--reread``, never on a refused stamp or ``--retry-refused``;
3. one adapter writes at most ``MAX_REACTIVATIONS_PER_PASS`` activation bundles per incremental pass; the rest are deferred by name.

Fixtures are invented (the synthetic store adapter of ``test_registry_adapter_pass``); nothing here reads a real store.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Sequence

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import reader_state as RS
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _cli, _pass_and_ingest, _setup, _store  # noqa: E402
from test_registry_cluster_read import PRODUCER, _cluster_pass, _grading_host, _export, _rows  # noqa: E402


class _Killed(Exception):
    """Stands for the unit's SIGTERM at TimeoutStartSec: the cycle dies where it stands."""


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _pass(root, adapter, now, **kw):
    """The pass alone, no fold (the adapters timer's ``--no-ingest``)."""
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return RUN.run_pass(root, [adapter], conn=conn, now=now, **kw)
    finally:
        conn.close()


def _fold(root, **kw):
    return H.ingest_once(root, workers=0, min_free=0, **kw)


def _bundles(root, adapter_name, rev):
    return [s for s, _ in RUN.complete_bundles(root, adapter_name, rev)]


def _all_bundles(root, adapter_name):
    base = os.path.join(root, "records", "adapters", adapter_name, "_activations")
    if not os.path.isdir(base):
        return {}
    return {rev: _bundles(root, adapter_name, rev) for rev in sorted(os.listdir(base))}


def _three_stores(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2])
    return root, store_root


# ---- (1) the fold that has not happened is pending, not absent -------------------------------------------------------

def test_an_emission_bump_folded_by_a_killed_cycle_is_pending_on_the_next_pass_and_nothing_is_reemitted(tmp_path, monkeypatch):
    """The spec's reproduction: three groups, an EMISSION_VERSION bump publishes three new revision directories, the harvester's
    fold is killed before any marker lands -> the next pass writes NOTHING and reports ``activations_pending = 3``; the fold then
    completes -> status rows present -> nothing pending, nothing planned. A ``--shard`` pass over the same state honors the rule."""
    root, store_root = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep0, ing0 = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep0.groups_refreshed == 3 and ing0.refused == 0 and ing0.blocked == 0
    v1 = {r.group: r.revision_id for r in rep0.results}
    assert _q(root, "SELECT COUNT(*) FROM source_status") == [(3,)]
    # the legitimate one-time emission change: every group's head was minted under version 1, so every group is planned
    ad.EMISSION_VERSION = 2
    rep1 = _pass(root, ad, synth.t(3000))
    assert rep1.groups_refreshed == 3 and rep1.activations_pending == 0 and all(r.accepted and not r.reread for r in rep1.results)
    v2 = {r.group: r.revision_id for r in rep1.results}
    assert set(v2.values()).isdisjoint(v1.values())
    # the harvester dies at its unit bound before the first marker folds: the staging windows it committed stay committed (a
    # `prepare` placeholder row, accepted = 0, no refusal), no marker has a delivery row

    def killed(self, *a, **k):
        raise _Killed("TimeoutStartSec")
    monkeypatch.setattr(H.Ingestor, "_activate_revision", killed)
    with pytest.raises(_Killed):
        _fold(root)
    monkeypatch.undo()
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 0 AND refusal IS NULL") == [(1,)]     # the first directory's placeholder
    assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE '%source_revision%' AND file_uri LIKE ?", f"%{v2['s1']}%") == [(0,)]
    assert all(RUN.pending_fold(root, ad.name, rev, DB.connect(DB.db_path(root), readonly=True)) is not None for rev in v2.values())
    # the next pass (the same inputs: their watermarks are in the unfolded markers, so every group reads as changed): NOT re-emitted
    rep2 = _pass(root, ad, synth.t(4000))
    assert rep2.groups_seen == 3 and rep2.activations_pending == 3 and rep2.groups_held == 3, rep2.as_dict()   # planned on the head version
    assert rep2.groups_refreshed == 0 and rep2.groups_refused == 0 and rep2.records_written == 0 and rep2.reactivations_deferred == 0
    assert all(r.activation_pending and r.held and not r.accepted and r.files == [] and r.revision_id == v2[r.group] for r in rep2.results)
    assert all("not folded yet" in r.error and "pending" in r.error for r in rep2.results)
    assert _all_bundles(root, ad.name) == {}                                              # no _activations bundle, no bytes
    assert sorted(os.listdir(os.path.join(root, "records", "adapters", ad.name))) == sorted(set(v1.values()) | set(v2.values()))
    assert rep2.as_dict()["activations_pending"] == 3 and "reactivations_deferred" in rep2.as_dict()
    # a grading host --shard pass over the same state: each shard reports the pending revisions it owns, nothing is written
    reps = [_pass(root, ad, synth.t(4100 + i), shard=(i, 2)) for i in range(2)]
    assert sum(r.activations_pending for r in reps) == 3 and all(r.records_written == 0 for r in reps) and _all_bundles(root, ad.name) == {}
    # an explicit --reread waits for the fold too: the bundle it would write is already what the log carries
    rep2b = _pass(root, ad, synth.t(4200), reread=True)
    assert rep2b.activations_pending == 3 and rep2b.groups_reread == 0 and _all_bundles(root, ad.name) == {}
    # the uninterrupted fold: the three directories activate, their status rows land, and nothing is pending any more
    ing3 = _fold(root)
    assert ing3.refused == 0 and ing3.blocked == 0 and ing3.revisions_held == 0 and ing3.revisions_applied == 3, ing3.as_dict()
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 1") == [(6,)]
    assert sorted(r[0] for r in _q(root, "SELECT source_revision_id FROM source_status")) == sorted(list(v1.values()) + list(v2.values()))
    assert all(RUN.pending_fold(root, ad.name, rev, DB.connect(DB.db_path(root), readonly=True)) is None for rev in v2.values())
    rep4 = _pass(root, ad, synth.t(5000))
    assert rep4.results == [] and rep4.activations_pending == 0 and rep4.inputs_changed == 0
    assert _all_bundles(root, ad.name) == {}


def test_a_complete_bundle_on_the_log_is_pending_until_its_marker_is_delivered_and_a_torn_bundle_is_not_a_bundle(tmp_path, monkeypatch):
    """``pending_fold`` reads the harvester's marker convention: a ``<stamp>/`` directory under ``_activations/<rev>/`` is a complete
    bundle when its ``*-source_revision.jsonl`` marker is present; it is folded when ``record_delivery`` holds that marker's uri
    (``ok`` or ``blocked``). A stamp directory without a marker (a publisher mid-rename) and the plain ``<stamp>-source_watermark``
    re-stamp files are not bundles."""
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    monkeypatch.setattr(RUN, "_source_status_records", lambda *a, **k: [])       # a head committed before status rows were written: no status row
    rep0, ing0 = _pass_and_ingest(root, ad, synth.t(2000))
    monkeypatch.undo()
    rev = rep0.results[0].revision_id
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert RUN.pending_fold(root, ad.name, rev, conn) is None and RUN.complete_bundles(root, ad.name, rev) == []
    # a touched input (same bytes): the head has no status row -> a re-read bundle -- written once
    p = os.path.join(store_root, "s1", "store.json")
    st = os.stat(p)
    os.utime(p, (st.st_atime + 5, st.st_mtime + 5))
    rep1 = _pass(root, ad, synth.t(3000))
    (res1,) = rep1.results
    assert res1.accepted and res1.reread and res1.reactivated and rep1.activations_pending == 0
    stamps = _bundles(root, ad.name, rev)
    assert len(stamps) == 1
    marker_uri = RUN.pending_fold(root, ad.name, rev, conn)
    assert marker_uri == f"records/adapters/{ad.name}/_activations/{rev}/{stamps[0]}/999998-source_revision.jsonl"
    # a torn stamp directory (no marker) beside it is not a bundle and never pending; a watermark re-stamp file is not either
    torn = os.path.join(RUN.activations_dir(root, ad.name, rev), "20990101000000000")
    os.makedirs(torn)
    R.write_records(os.path.join(torn, "000001-artifact.jsonl"), [])
    R.write_records(os.path.join(RUN.activations_dir(root, ad.name, rev), "20990101000000001-source_watermark.jsonl"), [])
    assert [s for s, _ in RUN.complete_bundles(root, ad.name, rev)] == stamps
    # every later pass, whatever selects the group: pending, no second bundle
    os.utime(p, (st.st_atime + 10, st.st_mtime + 10))
    for i, kw in enumerate(({}, {"retry_refused": True}, {"reread": True}, {"full": True})):
        rep = _pass(root, ad, synth.t(3100 + i), **kw)
        assert rep.activations_pending == 1 and rep.groups_refreshed == 0 and rep.records_written == 0, (kw, rep.as_dict())
        assert _bundles(root, ad.name, rev) == stamps
    # the fold delivers the marker: folded, not pending; the status row is on the database; the next pass writes nothing
    ing = _fold(root)
    # two directories taken: the bundle (activated) and the torn stamp directory, whose one file the harvester blocks until a marker
    # arrives (its `marker_path is None` path) -- which is exactly why a torn directory is never a complete bundle here
    assert ing.refused == 0 and ing.revisions_applied == 2 and ing.blocked == 1 and ing.revision_transactions == 1
    assert RUN.pending_fold(root, ad.name, rev, conn) is None
    assert _q(root, "SELECT COUNT(*) FROM source_status WHERE source_revision_id = ?", rev) == [(1,)]
    rep2 = _pass(root, ad, synth.t(4000))
    assert rep2.activations_pending == 0 and rep2.records_written == 0 and _bundles(root, ad.name, rev) == stamps
    # no database: the pass cannot tell, and says so by answering None (the caller re-activates as before)
    assert RUN.pending_fold(root, ad.name, rev, None) is None
    conn.close()


# ---- (2) the live shape: a HELD revision with a stale refused stamp is not re-read on the stamp ------------------------

class StableStampsAdapter(SyntheticStoreAdapter):
    """The synthetic store adapter with the live readers' property: a re-read of unchanged inputs emits the SAME records (the
    ``item_result.created_utc`` is the store's time, not the read's), so two fresh reads differ only in ``written_utc`` -- the shape of
    the 2026-09-29 backlog's bundles (verified on the live log) and what the harvester's no-op rule compares."""

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for batch in super().read(ctx, group, inputs):
            for rec in batch.records:
                if rec["record_type"] == "item_result":
                    rec["payload"]["created_utc"] = synth.t(1500)
                    rec["record_id"] = I.adapter_record_id(rec["payload"], "item_result")
            yield batch


class HoldingAdapter(StableStampsAdapter):
    """The synthetic store adapter, plus one ``judgment`` naming a run nothing publishes: the harvester holds every revision it reads on
    that missing parent (``activation pending: run <id> (...judgment.jsonl#N)``) -- the live ``regrade_twin`` hold, in miniature."""
    name = "holding_store"

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for batch in super().read(ctx, group, inputs):
            orphan_run = I.run_id(synth.CAMPAIGN, "never_published", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)
            batch.add("judgment", {"judgment_id": I.judgment_id(orphan_run, synth.JUDGE_ID), "run_id": orphan_run, "judge_id": synth.JUDGE_ID,
                                   "n_items_judged": 1, "status": "complete"}, ctx.producer, ctx.now)
            yield batch


def test_a_held_revision_with_a_stale_refused_stamp_is_not_reread_per_pass_and_only_an_explicit_reread_writes_a_bundle_once(tmp_path):
    """Pass 1 refuses (a broken store) and stamps the input refused; the store is rewritten and pass 2 publishes a revision the
    harvester HOLDS on a missing run, so the clean watermarks sit in the held marker and the refused stamp stands. Before the fix
    every later pass re-read the group on that stamp and wrote a fresh bundle, held again, forever. Now: the incremental pass
    and ``--retry-refused`` leave it alone (held, the stamp named); an explicit ``--reread`` writes ONE bundle; while that bundle is
    unfolded every pass -- the explicit re-read included -- reports it pending; once the harvester holds the bundle too, the pass
    is back to held with no new copy."""
    root, store_root = _setup(tmp_path)
    ad = HoldingAdapter(store_root)
    p = os.path.join(store_root, "s1", "store.json")
    ad.broken.add(p)
    rep1, ing1 = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep1.groups_refused == 1 and ing1.refused == 0
    marks = _q(root, "SELECT last_error, json_extract(extra_json, '$.attempt_revision_id') FROM source_watermark WHERE adapter = ?", ad.name)
    assert len(marks) == 1 and "synthetic read failure" in marks[0][0] and marks[0][1]
    # the store is rewritten (new bytes: the refused attempt's id no longer stands for the inventory) and read successfully -- but held
    ad.broken.clear()
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9])
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    (res2,) = rep2.results
    assert res2.accepted and not res2.reread and ing2.revisions_held >= 1 and ing2.blocked >= 1     # held once, then skipped by the closing re-evaluation
    rev = res2.revision_id
    assert _q(root, "SELECT accepted, substr(refusal, 1, 24) FROM source_revision WHERE revision_id = ?", rev) == [(0, "activation pending: run ")]
    # the stale stamp: the successful read's clean watermarks are in the held marker, so the database still says REFUSED
    assert _q(root, "SELECT COUNT(*) FROM source_watermark WHERE adapter = ? AND last_error IS NOT NULL", ad.name) == [(1,)]
    assert _all_bundles(root, ad.name) == {}
    # (a) the incremental pass, then --retry-refused: the group is planned on its stamp and LEFT ALONE -- no bundle, no bytes
    for i, kw in enumerate(({}, {"retry_refused": True}, {})):
        rep = _pass(root, ad, synth.t(4000 + i), **kw)
        (res,) = rep.results
        assert rep.groups_held == 1 and rep.groups_refreshed == 0 and rep.groups_refused == 0 and rep.records_written == 0, (kw, rep.as_dict())
        assert res.held and not res.accepted and not res.reread and res.files == [] and res.revision_id == rev
        assert "held by the harvester" in res.error and "refused stamp" in res.error and "`adapters run --reread`" in res.error
        assert "activation pending: run" in res.error
        assert _all_bundles(root, ad.name) == {}
    # (b) the operator's explicit --reread: the one cause that re-reads a held revision -- ONE bundle, the fresh read
    rep5 = _pass(root, ad, synth.t(5000), reread=True)
    (res5,) = rep5.results
    assert res5.accepted and res5.reread and res5.reactivated and rep5.groups_reread == 1 and rep5.activations_pending == 0
    assert _all_bundles(root, ad.name) == {rev: [RUN._compact(synth.t(5000))]}
    # (c) unfolded, the bundle is PENDING for every pass -- the incremental one, --retry-refused, and another explicit --reread
    for i, kw in enumerate(({}, {"retry_refused": True}, {"reread": True})):
        rep = _pass(root, ad, synth.t(5100 + i), **kw)
        assert rep.activations_pending == 1 and rep.groups_reread == 0 and rep.records_written == 0, (kw, rep.as_dict())
        assert _all_bundles(root, ad.name) == {rev: [RUN._compact(synth.t(5000))]}
    # (d) the harvester folds the bundle -- and holds it on the same missing run: delivered (blocked), so no longer pending; the
    # incremental pass is back to "held", still writing nothing
    ing6 = _fold(root)
    assert ing6.revisions_held >= 1 and ing6.revisions_applied == 1
    ((stamp, marker_path),) = RUN.complete_bundles(root, ad.name, rev)          # a fresh read's bundle: the marker takes the read's own seq
    assert stamp == RUN._compact(synth.t(5000)) and RUN._is_marker_file(os.path.basename(marker_path))
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, marker_path)) == [("blocked",)]
    rep7 = _pass(root, ad, synth.t(6000))
    assert rep7.groups_held == 1 and rep7.activations_pending == 0 and rep7.records_written == 0 and rep7.results[0].held
    assert _all_bundles(root, ad.name) == {rev: [RUN._compact(synth.t(5000))]}
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]


# ---- (3) the per-pass cap ----------------------------------------------------------------------------------------------

def test_an_adapter_writes_at_most_the_cap_of_bundles_per_incremental_pass_defers_the_rest_by_name_and_reread_is_uncapped(tmp_path, monkeypatch):
    root, store_root = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    monkeypatch.setattr(RUN, "_source_status_records", lambda *a, **k: [])       # three heads committed before status rows were written: none
    rep0, ing0 = _pass_and_ingest(root, ad, synth.t(2000))
    monkeypatch.undo()
    assert rep0.groups_refreshed == 3 and ing0.refused == 0 and _q(root, "SELECT COUNT(*) FROM source_status") == [(0,)]
    revs = {r.group: r.revision_id for r in rep0.results}

    def touch(offset):
        for g in ("s1", "s2", "s3"):
            p = os.path.join(store_root, g, "store.json")
            st = os.stat(p)
            os.utime(p, (st.st_atime + offset, st.st_mtime + offset))
    # every input touched with identical bytes: each head is re-read for its status row (a bundle each) -- capped at 2 per pass
    touch(5)
    monkeypatch.setattr(RUN, "MAX_REACTIVATIONS_PER_PASS", 2)
    rep1 = _pass(root, ad, synth.t(3000))
    assert rep1.groups_reread == 2 and rep1.reactivations_deferred == 1 and rep1.groups_held == 0 and rep1.groups_refused == 0, rep1.as_dict()
    deferred = [r for r in rep1.results if r.deferred]
    assert len(deferred) == 1 and not deferred[0].accepted and deferred[0].files == [] and deferred[0].group == "s3"     # groups read in sorted order
    assert "MAX_REACTIVATIONS_PER_PASS = 2" in deferred[0].error and "2 activation bundle(s)" in deferred[0].error
    assert {g for g, stamps in _all_bundles(root, ad.name).items() if stamps} == {revs["s1"], revs["s2"]}
    assert rep1.as_dict()["reactivations_deferred"] == 1
    # the fold, then the next pass takes the deferred group (its watermark still names the old mtime): one bundle, nothing deferred
    ing1 = _fold(root)
    assert ing1.refused == 0 and ing1.blocked == 0 and ing1.revisions_applied == 2
    rep2 = _pass(root, ad, synth.t(4000))
    assert rep2.groups_reread == 1 and rep2.reactivations_deferred == 0 and rep2.results[0].group == "s3", rep2.as_dict()
    assert set(_all_bundles(root, ad.name)) == set(revs.values())
    _fold(root)
    assert _q(root, "SELECT COUNT(*) FROM source_status") == [(3,)]
    # the operator's explicit --reread runs uncapped: three bundles in one pass under the same cap of 2
    rep3 = _pass(root, ad, synth.t(5000), reread=True)
    assert rep3.groups_reread == 3 and rep3.reactivations_deferred == 0 and rep3.activations_pending == 0, rep3.as_dict()
    assert all(len(stamps) == 2 for stamps in _all_bundles(root, ad.name).values())


def test_a_watermark_restamp_of_a_head_with_a_status_row_is_free_of_the_cap(tmp_path, monkeypatch):
    """A touched input of a head that HAS its status row re-stamps its watermarks as a plain file (no bundle): never counted."""
    root, store_root = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep0, ing0 = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep0.groups_refreshed == 3 and ing0.refused == 0
    for g in ("s1", "s2", "s3"):
        p = os.path.join(store_root, g, "store.json")
        st = os.stat(p)
        os.utime(p, (st.st_atime + 5, st.st_mtime + 5))
    monkeypatch.setattr(RUN, "MAX_REACTIVATIONS_PER_PASS", 1)
    rep1 = _pass(root, ad, synth.t(3000))
    assert rep1.groups_refreshed == 3 and rep1.reactivations_deferred == 0 and rep1.groups_reread == 0
    assert all(not r.reactivated and r.files and r.files[0].endswith("-source_watermark.jsonl") for r in rep1.results)
    assert all(not stamps for stamps in _all_bundles(root, ad.name).values())


# ---- the cluster-read path ---------------------------------------------------------------------------------------------

def test_a_cluster_shard_reads_a_published_not_folded_id_as_pending_on_the_grading_host_and_publishes_nothing(tmp_path):
    """The reader-state sidecar carries no delivery ledger, so a shard reads pending off ``published_dirs``: an id listed as a revision
    directory or an activation bundle with no ``source_revision`` row is on the grading host's log unfolded -> refused by name AND counted
    under ``activations_pending``. An id with a row (accepted or held) is refused as before and not called pending: the sidecar
    cannot see whether a bundle of an accepted head has folded, and says nothing it does not know."""
    root, home, stores, ad = _grading_host(tmp_path)
    # s3 rewritten and read on the grading host, published, NOT folded; s1's head re-read on the grading host (a bundle), NOT folded
    _store(str(stores), "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2, 0.1])
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000), producer=PRODUCER)
    assert rep.groups_refreshed == 1 and rep.results[0].group == "s3"
    s3_rev = rep.results[0].revision_id
    with DB.reading(DB.db_path(root)) as conn:
        rep_b = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3100), producer=PRODUCER, reread=True)
    s1 = next(r for r in rep_b.results if r.group == "s1")
    assert s1.reread and s1.reactivated and any("/_activations/" in f for f in s1.files)
    assert next(r for r in rep_b.results if r.group == "s3").activation_pending           # the unfolded s3 directory: pending on the grading host too
    sidecar = str(tmp_path / "rs.sqlite")
    _export(root, sidecar, home)
    conn = RS.open_reader_state(sidecar, [ad.name])
    try:
        kinds = RS.known_dir_kinds(conn)
        assert kinds[s3_rev] == {"revision"} and kinds[s1.revision_id] == {"revision", "activation"}
        cluster = RUN.ClusterRead.from_sidecar(conn)
        assert s3_rev in cluster.pending_dirs and s1.revision_id in cluster.pending_dirs and cluster.known_dirs >= cluster.pending_dirs
        assert cluster.pending_on_grading_host(conn, s3_rev) and not cluster.pending_on_grading_host(conn, s1.revision_id)   # s1 has an accepted row
    finally:
        conn.close()
    cluster_root = str(tmp_path / "cluster_root")
    H.ensure_layout(cluster_root)
    rep_c = _cluster_pass(cluster_root, ad, sidecar, synth.t(4000), full=True)
    by = {r.group: r for r in rep_c.results}
    assert by["s3"].refused_known and by["s3"].activation_pending and "not folded yet" in by["s3"].error and "pending" in by["s3"].error
    assert by["s1"].refused_known and not by["s1"].activation_pending and "source_revision row, accepted" in by["s1"].error
    assert rep_c.groups_refused_known == 3 and rep_c.activations_pending == 1 and rep_c.groups_refreshed == 0
    assert not os.path.isdir(os.path.join(H.records_dir(cluster_root), "adapters"))
    assert rep_c.as_dict()["activations_pending"] == 1
    # a legacy construction (known ids only) still refuses by name; it just cannot call anything pending
    legacy = RUN.ClusterRead(known_dirs={s3_rev})
    assert legacy.known(None, s3_rev) == "a published directory on the grading host" and not legacy.pending_on_grading_host(None, s3_rev)


# ---- the CLI surface ---------------------------------------------------------------------------------------------------

def test_the_adapters_run_summary_carries_the_pending_and_deferred_counts_and_names_a_pending_group(tmp_path, monkeypatch, capsys):
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep0, ing0 = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep0.groups_refreshed == 1 and ing0.refused == 0
    ad.EMISSION_VERSION = 2
    rep1 = _pass(root, ad, synth.t(3000))            # the new revision directory, published and not folded
    assert rep1.groups_refreshed == 1
    rc, summary, tail, err, lock = _cli(root, ad, monkeypatch, capsys, "--no-ingest")
    assert rc == 0 and summary["activations_pending"] == 1 and summary["reactivations_deferred"] == 0 and summary["groups_held"] == 1
    assert "pending fold: synthetic_store s1:" in err and "not folded yet" in err
    assert not lock.called and tail["ingest"].startswith("skipped")


def test_a_decision_changed_group_whose_new_revision_is_published_and_not_folded_is_pending_and_nothing_is_reemitted(tmp_path):
    """The anchor decision input (adapter re-anchoring) under the pending-fold rule: a ruling moves the anchor, the pass publishes the
    re-anchored revision, and until the harvester folds it the ACCEPTED head is still the old one -- so the next pass's pre-read fingerprint
    still differs and the group plans as changed, lands on the published directory, and writes NOTHING (pending). After the fold the head
    carries the new set and the pass is quiet."""
    import _reanchor_fixtures as X
    root, stores = X.contrast_world(tmp_path)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))              # the runs fold first: the contrast read's frame holds them
    ad = X.ContrastStoreAdapter(stores)
    rep0 = _pass(root, ad, synth.t(2000))
    assert rep0.groups_refreshed == 3
    ing0 = _fold(root)
    assert ing0.refused == 0 and ing0.blocked == 0
    sid = RUN.source_id_for(ad.name, "s1")
    rev1, _ = X.head_of(root, sid)
    X.file_ruling(root, "r-exclude-own", "exclude", X.run_id_of(X.OWN), 2100)
    rep_a = _pass(root, X.ContrastStoreAdapter(stores), synth.t(2200))            # the re-anchored revision: published, NOT folded
    assert rep_a.groups_refreshed == 1 and rep_a.anchor_decisions_changed == 1
    new = rep_a.results[0].revision_id
    assert new != rev1 and os.path.isdir(os.path.join(H.records_dir(root), "adapters", ad.name, new))
    rep_b = _pass(root, X.ContrastStoreAdapter(stores), synth.t(2300))
    assert rep_b.anchor_decisions_changed == 1 and rep_b.groups_held == 1 and rep_b.activations_pending == 1
    assert rep_b.groups_refreshed == 0 and rep_b.records_written == 0
    (res_b,) = rep_b.results
    assert res_b.held and res_b.activation_pending and res_b.revision_id == new and res_b.files == [] and "not folded yet" in res_b.error
    assert not os.path.isdir(RUN.activations_dir(root, ad.name, new)) and _all_bundles(root, ad.name) == {}
    ing = _fold(root)
    assert ing.refused == 0 and ing.blocked == 0 and X.head_of(root, sid)[0] == new
    rep_c = _pass(root, X.ContrastStoreAdapter(stores), synth.t(2400))
    assert rep_c.results == [] and rep_c.anchor_decisions_changed == 0 and rep_c.groups_held == 0
