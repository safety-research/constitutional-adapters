"""The harvester makes DURABLE progress inside a bounded cycle (2026-09-29).

The recovery cycles of 2026-09-29 folded a backlog of activation bundles for 80 minutes and logged two ``transaction`` lines in
that time, which read as "the fold commits nothing until the end". It does not: every staging window and every activation of a
revision directory or bundle is its own committed transaction (``kind = "staging"`` / ``"activation"`` on the metrics log; the
``transaction`` kind is the plain-file chunks'), and the held path's one transaction was committed too -- outside the timed path,
so it logged nothing. Pinned here: a cycle killed between bundles keeps every bundle it finished; the held path is a timed
``kind = "held"`` transaction; every bundle logs one ``kind = "bundle"`` line; ``ingest_once(budget_s=...)`` stops taking new
bundles and plain chunks when the budget is spent, finishes the one in hand, reports ``stopped = "budget"`` with what it left, and
the next cycle continues; a completed cycle records ``state/harvest_cycle.json`` (a killed one does not); the CLI wires ``--budget-s``.
"""
from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import synth
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _setup, _store  # noqa: E402
from test_registry_adapter_no_reemit_unfolded import HoldingAdapter, StableStampsAdapter  # noqa: E402


class _Killed(Exception):
    """Stands for the unit's SIGTERM at TimeoutStartSec."""


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _metrics(root, now):
    with open(os.path.join(root, "state", "metrics.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("utc") == now]


def _three_published(tmp_path):
    """Three revision directories published by one pass and NOT folded."""
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2])
    ad = SyntheticStoreAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    assert rep.groups_refreshed == 3
    return root, ad, {r.group: r.revision_id for r in rep.results}


def _cycle_state(root):
    return H.read_cycle_state(root)


def test_a_cycle_killed_between_bundles_keeps_every_bundle_it_finished_and_the_next_cycle_folds_the_rest(tmp_path, monkeypatch):
    root, ad, revs = _three_published(tmp_path)
    inner = H.Ingestor.apply_revision
    calls = []

    def die_on_third(self, rev_dir, now, **kw):
        calls.append(rev_dir)
        if len(calls) == 3:
            raise _Killed("TimeoutStartSec")            # the unit kills the process before the third bundle is touched
        return inner(self, rev_dir, now, **kw)
    monkeypatch.setattr(H.Ingestor, "apply_revision", die_on_third)
    before = _cycle_state(root)                     # the fixture's own completed fold wrote it; the killed cycle must not move it
    now1 = synth.t(2500)
    with pytest.raises(_Killed):
        H.ingest_once(root, workers=0, min_free=0, now=now1)
    monkeypatch.undo()
    # the two finished bundles are COMMITTED: accepted rows, status rows, delivered markers -- read through a fresh connection
    done = [os.path.basename(d) for d in calls[:2]]
    assert sorted(r[0] for r in _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1")) == sorted(done)
    assert sorted(r[0] for r in _q(root, "SELECT source_revision_id FROM source_status")) == sorted(done)
    assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE '%source_revision%' AND status = 'ok'") == [(2,)]
    third = next(r for r in revs.values() if r not in done)
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE revision_id = ?", third) == [(0,)]
    # the metrics log shows the two bundles (one `bundle` line each, ok), their staging windows and activations, and no `cycle` line
    m = _metrics(root, now1)
    assert [r["result"] for r in m if r["kind"] == "bundle"] == ["ok", "ok"]
    assert sum(1 for r in m if r["kind"] == "activation") == 2 and sum(1 for r in m if r["kind"] == "staging") > 2
    assert not any(r["kind"] == "cycle" for r in m) and _cycle_state(root) == before and before["cycle_utc"] != now1
    # the next cycle takes only the third: it is the only new work
    now2 = synth.t(2600)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert ing.revisions_applied == 1 and ing.revision_transactions == 1 and ing.refused == 0 and ing.blocked == 0 and ing.stopped is None
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 1") == [(3,)]
    assert [r["result"] for r in _metrics(root, now2) if r["kind"] == "bundle"] == ["ok"]
    state = _cycle_state(root)
    assert state and state["last_completed_cycle_utc"] == ing.completed_utc and state["cycle_utc"] == now2 and state["stopped"] is None
    assert state["revisions_applied"] == 1 and state["revisions_deferred_budget"] == 0 and ing.elapsed_s >= 0


def test_a_budget_stops_new_bundles_between_them_finishes_the_one_in_hand_and_the_next_cycle_continues(tmp_path, monkeypatch):
    root, ad, revs = _three_published(tmp_path)
    clock = {"t": 0.0}
    inner = H.Ingestor.apply_revision

    def ten_seconds_each(self, rev_dir, now, **kw):
        out = inner(self, rev_dir, now, **kw)
        clock["t"] += 10.0
        return out
    monkeypatch.setattr(H.Ingestor, "apply_revision", ten_seconds_each)
    now1 = synth.t(2500)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now1, budget_s=15.0, clock=lambda: clock["t"])
    # bundle 1 at t = 0 (taken), bundle 2 at t = 10 (within budget: taken, finishes at t = 20), bundle 3 asked at t = 20 > 15: NOT taken
    assert ing.stopped == "budget" and ing.revisions_applied == 2 and ing.revisions_deferred_budget == 1 and ing.plain_files_deferred_budget == 0
    assert ing.budget_s == 15.0 and ing.elapsed_s == 20.0 and ing.refused == 0 and ing.blocked == 0
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 1") == [(2,)]
    m = _metrics(root, now1)
    assert [r["result"] for r in m if r["kind"] == "bundle"] == ["ok", "ok"]
    (cycle,) = [r for r in m if r["kind"] == "cycle"]
    assert cycle["stopped"] == "budget" and cycle["bundles"] == 2 and cycle["bundles_deferred"] == 1 and cycle["budget_s"] == 15.0
    # the closing steps still ran: the arrival facts, the dense pass, the paper head, the completed-cycle state
    assert any(r["kind"] == "rerank" for r in m) and ing.arrival_facts >= 1
    state = _cycle_state(root)
    assert state["stopped"] == "budget" and state["revisions_deferred_budget"] == 1 and state["last_completed_cycle_utc"] == ing.completed_utc
    # the next cycle, unbudgeted: the third bundle only
    monkeypatch.undo()
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2600))
    assert ing2.stopped is None and ing2.revisions_applied == 1 and ing2.revisions_deferred_budget == 0
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 1") == [(3,)]
    assert _cycle_state(root)["stopped"] is None


def test_a_spent_budget_takes_no_plain_chunk_and_no_bundle_and_reports_what_it_left(tmp_path):
    root, ad, revs = _three_published(tmp_path)
    # plain files beside the three directories: the roster file the fixture wrote is folded already; add a fresh plain record file
    from sisteer.registry import records as R
    R.write_records(os.path.join(root, "records", "notes", "000001-dims.jsonl"), synth.seed_dimensions()[:2])
    ing = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500), budget_s=0.0)
    assert ing.stopped == "budget" and ing.revisions_applied == 0 and ing.revisions_deferred_budget == 3 and ing.plain_files_deferred_budget == 1
    assert ing.files == 0 or ing.arrival_facts == 0        # nothing of the new work folded
    assert _q(root, "SELECT COUNT(*) FROM source_revision") == [(0,)]
    assert _cycle_state(root)["stopped"] == "budget" and _cycle_state(root)["plain_files_deferred_budget"] == 1
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2600))
    assert ing2.stopped is None and ing2.revisions_applied == 3 and ing2.plain_files_deferred_budget == 0
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 1") == [(3,)]


def test_the_held_path_is_one_timed_transaction_and_a_held_bundle_logs_its_line(tmp_path):
    root, store_root = _setup(tmp_path)
    ad = HoldingAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    assert rep.groups_refreshed == 1
    now = synth.t(2500)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert ing.revisions_held >= 1 and ing.revisions_applied == 1 and ing.revision_transactions == 1   # held once; the closing re-evaluation skips the exact hold
    m = _metrics(root, now)
    kinds = [r["kind"] for r in m]
    assert kinds.count("held") == 1 and kinds.count("activation") == 1 and kinds.count("bundle") == 1
    (bundle,) = [r for r in m if r["kind"] == "bundle"]
    assert bundle["result"] == "blocked" and bundle["dir"] == f"records/adapters/{ad.name}/{rep.results[0].revision_id}" and bundle["seconds"] >= 0
    (held,) = [r for r in m if r["kind"] == "held"]
    assert held["files"] == len(os.listdir(os.path.join(root, bundle["dir"])))
    # counted with the cycle's transactions: every timed transaction logs (kind, files, rows, seconds) -- the held commit among them
    timed = [r for r in m if {"files", "rows", "seconds"} <= set(r)]
    assert ing.transactions == len(timed) and ing.transactions >= ing.staging_transactions + ing.revision_transactions + 1 + 1, ing.as_dict()
    assert [r["kind"] for r in timed if r["kind"] not in ("staging",)][:2] == ["activation", "held"]       # the hold is recorded right after the rollback
    assert _q(root, "SELECT accepted FROM source_revision") == [(0,)]
    (cycle,) = [r for r in m if r["kind"] == "cycle"]
    assert cycle["bundles"] == 1 and cycle["bundles_held"] >= 1 and cycle["stopped"] is None


def test_the_ingest_cli_wires_the_budget_and_prints_the_completion_stamp(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    rep = H.IngestReport(stopped="budget", budget_s=5.0, elapsed_s=6.5, completed_utc="2026-01-01T00:00:07.000Z", revisions_deferred_budget=4,
                         plain_files_deferred_budget=2)
    lock = mock.MagicMock(name="ingest_under_lock", return_value=rep)
    monkeypatch.setattr(H, "ingest_under_lock", lock)
    assert cli.main(["--root", root, "ingest", "--workers", "0", "--budget-s", "5"]) == 0
    assert lock.call_args.kwargs["budget_s"] == 5.0
    out = capsys.readouterr()
    summary = json.loads(out.out)
    assert summary["last_completed_cycle_utc"] == "2026-01-01T00:00:07.000Z" and summary["stopped"] == "budget" and summary["budget_s"] == 5.0
    assert "stopped: budget" in out.err and "4 revision director(ies)" in out.err and "2 plain file(s)" in out.err
    # without the flag: no budget
    lock.reset_mock()
    assert cli.main(["--root", root, "ingest", "--workers", "0"]) == 0
    assert lock.call_args.kwargs["budget_s"] is None


# ---- no-op bundles: the legacy backlog's shape ----------------------------

def _stable_banned_metric_adapter(store_root):
    """``_BannedMetricAdapter`` (a run_metric under a banned spelling: the revision is held on a REFUSED constituent) with the live
    readers' stable stamps, so its re-reads repeat their content."""
    from test_registry_adapter_reread import _BannedMetricAdapter

    class _Stable(_BannedMetricAdapter):
        def read(self, ctx, group, inputs):
            for batch in super().read(ctx, group, inputs):
                for rec in batch.records:
                    if rec["record_type"] == "item_result":
                        from sisteer.registry import identity as I
                        rec["payload"]["created_utc"] = synth.t(1500)
                        rec["record_id"] = I.adapter_record_id(rec["payload"], "item_result")
                yield batch
    return _Stable(store_root)


def _direct_reread_bundles(root, ad, stamps):
    """The legacy backlog's shape: several fresh-read bundles of one held revision on the log at once -- what the runner wrote once per
    pass before the pending-fold rule (which now refuses the second and later ones), so they are written here through the runner's
    bundle writer directly, bypassing refresh_group."""
    from sisteer.registry.adapters.base import AdapterContext
    out = []
    for now in stamps:
        conn = DB.connect(DB.db_path(root), readonly=True)
        try:
            ctx = AdapterContext(root=root, now=now, producer=dict(H.harvester_producer(root), script="sisteer.registry.adapters.runner"),
                                 legacy_spellings={}, conn=conn)
            inputs = ad.discover()
            res = RUN.reread_committed(root, ad, ad.comparison_group(inputs[0]), inputs, ctx, conn)
        finally:
            conn.close()
        assert res.accepted and res.reactivated, res
        out.append(res)
    return out


def _orphan_run_file(root, now):
    """The run the HoldingAdapter's judgment names and nothing publishes -- published now as a plain record file."""
    from sisteer.registry import identity as I
    from sisteer.registry import records as R
    rid = I.run_id(synth.CAMPAIGN, "never_published", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)
    p = synth.run_payload("never_published", "none", "none", i=900)
    p.update(run_id=rid, lifecycle_state="generated", started_utc=synth.t(901), generated_utc=synth.t(902), n_items_generated=1)
    return R.write_records(os.path.join(root, "records", "parents", "000001-run.jsonl"), [synth.rec("run", p, now)])


def test_identical_bundles_of_a_held_revision_fold_as_no_ops_are_never_attempted_and_leave_the_queue_when_the_parent_lands(tmp_path, monkeypatch):
    root, store_root = _setup(tmp_path)
    ad = HoldingAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    stamps = [synth.t(3000), synth.t(3100), synth.t(3200)]
    bundles = _direct_reread_bundles(root, ad, stamps)          # three fresh reads: the same records but for written_utc
    assert [s for s, _ in RUN.complete_bundles(root, ad.name, rev)] == [RUN._compact(s) for s in stamps]
    # ONE cycle over the directory and its three bundles: the directory is attempted (staged, held on the missing run); every bundle
    # repeats its content and is a no-op -- ranked and delivered blocked under the recorded hold, nothing staged, nothing validated,
    # and NOT attempted at the close (its marker is as exact a hold as the directory's)
    now = synth.t(4000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert ing.revisions_applied == 4 and ing.revisions_noop == 3 and ing.revision_transactions == 1, ing.as_dict()
    m = _metrics(root, now)
    lines = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in m if r["kind"] == "bundle"]
    assert lines == [(rev, "blocked", None, None)] + [(RUN._compact(s), "blocked", "held", None) for s in stamps], lines
    assert sum(1 for r in m if r["kind"] == "held") == 1 and sum(1 for r in m if r["kind"] == "held_noop") == 3
    assert sum(1 for r in m if r["kind"] == "activation") == 1
    for b in bundles:
        for f in b.files:
            uri = H.logical_uri(root, f)
            assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", uri) == [("blocked",)], f
            assert _q(root, "SELECT COUNT(*) FROM record_ingest WHERE file_uri = ?", uri) == [(1,)], f      # ranked: the ledger and the arrival facts carry it
    from sisteer.registry import records as R
    dir_items = next(f for f in sorted(os.listdir(os.path.join(root, "records", "adapters", ad.name, rev))) if "item_result" in f)
    dir_fid = R.record_file_id_for_uri(H.logical_uri(root, os.path.join(root, "records", "adapters", ad.name, rev, dir_items)))
    assert _q(root, "SELECT DISTINCT record_file_id FROM item_result WHERE source_revision_id = ?", rev) == [(dir_fid,)]   # the directory's rows alone
    assert _q(root, "SELECT accepted, substr(refusal, 1, 24) FROM source_revision WHERE revision_id = ?", rev) == [(0, "activation pending: run ")]
    rows = _q(root, "SELECT record_uri, json_extract(extra_json, '$.no_op') FROM blocked_records WHERE record_type = 'source_revision' "
                    "AND json_extract(extra_json, '$.revision_id') = ? AND record_uri LIKE '%/_activations/%' ORDER BY record_uri", rev)
    assert [(RUN._compact(st) in u, flag) for st, (u, flag) in zip(stamps, rows)] == [(True, "held")] * 3, rows   # every no-op stands, tagged
    (cycle,) = [r for r in m if r["kind"] == "cycle"]
    assert cycle["bundles"] == 4 and cycle["bundles_noop"] == 3
    # the next cycle: nothing new, nothing re-attempted (both holds exact)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(4100))
    assert ing2.revisions_applied == 0 and ing2.revisions_noop == 0 and ing2.revision_transactions == 0
    # the missing parent arrives. A spent budget takes no plain chunk: the run file waits, and so does the hold
    _orphan_run_file(root, synth.t(4200))
    ing3 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(4300), budget_s=0.0)
    assert ing3.stopped == "budget" and ing3.plain_files_deferred_budget == 1 and ing3.revisions_deferred_budget == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    # a budget that lasts through the main pass but not into the closing re-evaluation: the run folds, the directory's re-attempt is
    # deferred by name (the no-op bundle is skipped, not deferred: the carrier re-attempts for it)
    clock = {"t": 0.0}
    inner_chunk = H.Ingestor.ingest_chunk

    def chunk_then_ten_seconds(self, chunk, now):
        inner_chunk(self, chunk, now)
        clock["t"] += 10.0
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", chunk_then_ten_seconds)
    ing3b = H.ingest_once(root, workers=0, min_free=0, now=synth.t(4350), budget_s=5.0, clock=lambda: clock["t"])
    monkeypatch.undo()
    assert ing3b.stopped == "budget" and ing3b.plain_files_deferred_budget == 0 and ing3b.revisions_deferred_budget == 1, ing3b.as_dict()
    assert ing3b.revision_transactions == 0 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    # unbudgeted: the directory activates (already staged: activation only); every row of the revision leaves the queue, the no-op
    # bundles with it, none of them ever applied
    ing4 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(4400))
    assert ing4.unblocked >= 1 and ing4.refused == 0 and ing4.stopped is None, ing4.as_dict()
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM source_status WHERE source_revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT DISTINCT record_file_id FROM item_result WHERE source_revision_id = ?", rev) == [(dir_fid,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ?", rev) == [(0,)]
    # the three no-op bundles, bundles of the head now, fold their FACTS in stamp order -- the marker (the head re-observed, its
    # watermarks) and the status row -- with their identical measurement files delivered duplicate; the head's rows stay the directory's
    lines4 = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, synth.t(4400)) if r["kind"] == "bundle"]
    assert lines4 == [(rev, "ok", None, True)] + [(RUN._compact(st), "ok", "head", True) for st in stamps], lines4
    for _, marker in RUN.complete_bundles(root, ad.name, rev):
        assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, marker)) == [("ok",)]
        for fn in os.listdir(os.path.dirname(marker)):
            if "item_result" in fn or "judgment" in fn:
                assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, os.path.join(os.path.dirname(marker), fn))) == [("duplicate",)]
    assert _q(root, "SELECT observed_utc FROM source_revision WHERE revision_id = ?", rev) == [(stamps[-1],)]
    assert _q(root, "SELECT record_uri FROM source_revision WHERE revision_id = ?", rev) == [(H.logical_uri(root, RUN.complete_bundles(root, ad.name, rev)[-1][1]),)]
    assert _q(root, "SELECT DISTINCT observed_utc FROM source_watermark WHERE adapter = ?", ad.name) == [(stamps[-1],)]        # the watermarks advanced
    assert _q(root, "SELECT observed_utc FROM source_status WHERE source_revision_id = ?", rev) == [(stamps[-1],)]
    from test_registry_core_fixes import _rebuild_equal
    _rebuild_equal(root, tmp_path)


def test_a_corrected_read_of_a_held_revision_is_attempted_in_full_an_identical_one_is_a_no_op_and_a_replay_agrees_exactly(tmp_path):
    """The 2026-09-19 strength_mtdiff shape under the no-op rule: a directory held on a constituent the validator REFUSES, then an
    identical re-read (a no-op: never attempted, no second refusal row), then a CORRECTED reader's bundle -- its content differs, so it
    is attempted in full the moment it arrives and activates. A replay of the whole log makes the same decisions from the same bytes
    and reaches the same rows exactly -- the refusal history included."""
    from test_registry_core_fixes import _rebuild_equal
    root, store_root = _setup(tmp_path)
    ad = _stable_banned_metric_adapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    ing0 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500))
    assert ing0.revisions_held >= 1 and ing0.refused >= 1 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]
    # another identical read (the same banned metric): a no-op, never attempted -- and the directory is an EXACT hold since 2026-09-30
    # (its refusing file and the judging code are unchanged), so it is not re-attempted either: no bundle line, the refusal history does
    # not grow and refused_utc does not move (test_registry_refused_hold_exact.py)
    _direct_reread_bundles(root, ad, [synth.t(3000)])
    now1 = synth.t(3500)
    ing1 = H.ingest_once(root, workers=0, min_free=0, now=now1)
    assert ing1.revisions_noop == 1 and ing1.revision_transactions == 0
    lines1 = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now1) if r["kind"] == "bundle"]
    assert lines1 == [(RUN._compact(synth.t(3000)), "blocked", "held", None)], lines1   # the no-op's fold only: neither it nor the exact hold is attempted
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    # the corrected reader: the same inputs, the metric under its proper name -- attempted in full in the main pass, accepted
    fixed = StableStampsAdapter(store_root)
    _direct_reread_bundles(root, fixed, [synth.t(4000)])
    now2 = synth.t(4500)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    lines = [(r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now2) if r["kind"] == "bundle"]
    # the correction activates in the main pass; at the close the earlier no-op -- a copy of the superseded directory's content, older
    # than the correction -- is a STALE copy: delivered duplicate, its rows gone, nothing folded and nothing re-refused
    assert lines == [("ok", None, None), ("ok", "stale", True)], lines
    assert ing2.revisions_noop == 1 and ing2.refused == 0
    for _, marker in RUN.complete_bundles(root, ad.name, rev)[:1]:
        assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, marker)) == [("duplicate",)]
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev) == [(1, None)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)] and _q(root, "SELECT metric_id FROM run_metric") == [("asr",)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]
    # the replay, every bundle in one cycle, reaches the same rows EXACTLY: heads, items, status, the queue, the refusal history (a
    # rebuild must not rewrite which constituents were refused: those rows are displayed and consulted by sha)
    _rebuild_equal(root, tmp_path)


def test_several_identical_bundles_across_cycles_never_grow_the_refusal_history_live_or_on_replay(tmp_path):
    """Two identical bundles in one cycle, one more in the next, of a revision held on a refused constituent: no-ops all, none attempted,
    one refusal row (the directory's) throughout, on both paths."""
    from test_registry_core_fixes import _rebuild_equal
    root, store_root = _setup(tmp_path)
    ad = _stable_banned_metric_adapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500))
    _direct_reread_bundles(root, ad, [synth.t(3000), synth.t(3100)])
    now1 = synth.t(3500)
    ing1 = H.ingest_once(root, workers=0, min_free=0, now=now1)
    s1, s2, s3 = (RUN._compact(synth.t(t)) for t in (3000, 3100, 4000))
    lines = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now1) if r["kind"] == "bundle"]
    assert lines == [(s1, "blocked", "held", None), (s2, "blocked", "held", None)], lines          # no attempt: the directory's hold is exact
    assert ing1.revisions_noop == 2 and ing1.revision_transactions == 0 and _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]
    _direct_reread_bundles(root, ad, [synth.t(4000)])
    now2 = synth.t(4500)
    H.ingest_once(root, workers=0, min_free=0, now=now2)
    lines2 = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now2) if r["kind"] == "bundle"]
    assert lines2 == [(s3, "blocked", "held", None)], lines2
    assert _q(root, "SELECT COUNT(*) FROM record_refusal") == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM record_refusal WHERE record_uri LIKE '%/_activations/%'") == [(0,)]
    # the queue: the directory's marker row and every no-op bundle's (exact holds; their facts fold when the carrier activates)
    rows = _q(root, "SELECT record_uri FROM blocked_records WHERE record_type = 'source_revision' ORDER BY record_uri")
    assert len(rows) == 4 and all(any(st in u for (u,) in rows) for st in (s1, s2, s3)), rows
    _rebuild_equal(root, tmp_path)


def _replay_diff(root, tmp_path):
    import random
    import shutil
    from sisteer.registry import records as R
    copy = str(tmp_path / "replay_copy")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(7).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / "replayed.sqlite")
    H.rebuild(copy, out, workers=0)
    with DB.reading(DB.db_path(root)) as a, DB.reading(out) as b:
        return {k: v for k, v in H.verify_equal(a, b).items() if v}


def test_a_bundle_of_the_accepted_head_folds_marker_and_status_only_and_re_inserts_no_item_row(tmp_path):
    """A -> B -> A: the re-activation bundle of A lands and A is the head again; a second, identical bundle (the shape the runner wrote
    once per pass) folds its marker (A re-observed) and status row and delivers its measurement files as duplicates -- no staging,
    no item row re-inserted (their record_file_id stays the first bundle's), nothing for reclaim to flip."""
    from sisteer.registry.adapters.base import AdapterContext
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep_a, ing_a = _pass_and_ingest_local(root, ad, synth.t(2000))
    rev_a = rep_a.results[0].revision_id
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    rep_b, ing_b = _pass_and_ingest_local(root, ad, synth.t(3000))
    rev_b = rep_b.results[0].revision_id
    assert rev_b != rev_a and ing_b.refused == 0
    # the revert: the pass re-activates A (a full re-emission bundle); a second identical bundle is written directly (the runner now
    # refuses it as pending)
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    with DB.reading(DB.db_path(root)) as conn:
        rep_c = RUN.run_pass(root, [ad], conn=conn, now=synth.t(4000))
        (res_c,) = rep_c.results
        assert res_c.accepted and res_c.reactivated and res_c.revision_id == rev_a and any("/_activations/" in f for f in res_c.files)
        ctx = AdapterContext(root=root, now=synth.t(4100), producer=dict(H.harvester_producer(root), script="sisteer.registry.adapters.runner"),
                             legacy_spellings={}, conn=conn)
        inputs = ad.discover()
        res_d = RUN.reactivate_committed(root, ad, "s1", inputs, ctx, conn)
    assert res_d.accepted and res_d.reactivated and res_d.records_written > 0
    stamps = [s for s, _ in RUN.complete_bundles(root, ad.name, rev_a)]
    assert stamps == [RUN._compact(synth.t(4000)), RUN._compact(synth.t(4100))]
    now = synth.t(5000)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert ing.refused == 0 and ing.blocked == 0 and ing.revisions_applied == 2 and ing.revisions_noop == 1
    lines = [r for r in _metrics(root, now) if r["kind"] == "bundle"]
    assert [(r["result"], r.get("no_op")) for r in lines] == [("ok", None), ("ok", "head")]
    assert not any("artifact" in f for f in res_d.files)                     # a re-emission carries no artifact file, and is a no-op still
    # the head is A, re-observed by the second bundle's marker; its rows are the FIRST bundle's (one record_file_id), not re-inserted
    assert _q(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1",
              f"{ad.name}:s1") == [(rev_a,)]
    assert ing.revision_transactions == 2                                                                  # both markers folded; one staging of rows
    assert _q(root, "SELECT observed_utc FROM source_revision WHERE revision_id = ?", rev_a) == [(synth.t(4100),)]     # the head re-observed by the second bundle's marker
    first_marker, second_marker = (m for _, m in RUN.complete_bundles(root, ad.name, rev_a))
    first_fid = _q(root, "SELECT DISTINCT record_file_id FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NULL", rev_a)
    assert len(first_fid) == 1
    from sisteer.registry import records as R
    first_items = next(f for f in sorted(os.listdir(os.path.dirname(first_marker))) if "item_result" in f)   # the committed file's own name, re-emitted
    assert first_fid[0][0] == R.record_file_id_for_uri(H.logical_uri(root, os.path.join(os.path.dirname(first_marker), first_items)))
    second_dir = os.path.dirname(second_marker)
    for fn in sorted(os.listdir(second_dir)):
        uri = H.logical_uri(root, os.path.join(second_dir, fn))
        (status,) = _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", uri)[0]
        expect = "ok" if ("source_revision" in fn or "source_status" in fn) else "duplicate"          # the facts fold, the measurements are duplicates
        assert status == expect, (fn, status)
    assert _q(root, "SELECT COUNT(*) FROM source_status WHERE source_revision_id = ?", rev_a) == [(1,)]     # the re-emitted status row REPLACES (same key)
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE source_revision_id = ?", rev_a) == [(3,)]
    _rebuild_equal_local(root, tmp_path)


class CorrectedHoldingAdapter(HoldingAdapter):
    """The HoldingAdapter after a fix to its metric computation: the same orphan judgment (so its revision holds on the same missing
    run), a different run_metric value -- the correction the earlier reproduction lost."""

    def read(self, ctx, group, inputs):
        for batch in super().read(ctx, group, inputs):
            for rec in batch.records:
                if rec["record_type"] == "run_metric":
                    rec["payload"]["value"] = 0.99
            yield batch


def test_a_corrected_bundle_of_a_held_revision_is_applied_in_full_when_the_parent_lands_never_marked_duplicate(tmp_path):
    """The revision's directory (run_metric 0.4667) is held on a missing run; a CORRECTED re-read bundle (0.99) arrives --
    its content differs, so it is attempted in full and held on the same run, the latest attempted carrier. The run lands: the closing
    re-evaluation attempts the directory FIRST (activating it, the head with 0.4667) and then the bundle -- which the head shortcut
    must not mark duplicate: its content differs from what the head applied, so it is staged in full and the corrected value is what
    the database carries. Then an IDENTICAL re-emission of the head is the duplicate case."""
    root, store_root = _setup(tmp_path)
    held = HoldingAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [held], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    ing0 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500))
    assert ing0.revisions_held >= 1 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    (bundle,) = _direct_reread_bundles(root, CorrectedHoldingAdapter(store_root), [synth.t(3000)])
    metric_file = next(f for f in bundle.files if "run_metric" in f)
    now1 = synth.t(3500)
    ing1 = H.ingest_once(root, workers=0, min_free=0, now=now1)
    lines1 = [(r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now1) if r["kind"] == "bundle"]
    assert lines1 == [("blocked", None, None)] and ing1.revisions_noop == 0 and ing1.revisions_held >= 1, lines1   # attempted in full, held
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    # the run lands: directory first (the head at 0.4667), then the corrected bundle as a head bundle -- staged in full, not duplicate
    _orphan_run_file(root, synth.t(3600))
    now2 = synth.t(4000)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert ing2.refused == 0 and ing2.unblocked >= 1, ing2.as_dict()
    lines2 = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now2) if r["kind"] == "bundle"]
    assert lines2 == [(rev, "ok", None, True), (RUN._compact(synth.t(3000)), "ok", None, True)], lines2
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    assert _q(root, "SELECT value FROM run_metric WHERE metric_id = 'asr'") == [(0.99,)]
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, metric_file)) == [("ok",)]
    assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE status = 'duplicate'") == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)] and _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)]
    from test_registry_core_fixes import _rebuild_equal
    _rebuild_equal(root, tmp_path)
    # an IDENTICAL re-emission of the head's applied bundle (the corrected one): marker and status fold, measurement files duplicate
    with DB.reading(DB.db_path(root)) as conn:
        head_dir = H.Ingestor(conn, root)._head_applied_dir(rev)
    assert head_dir and head_dir.endswith(RUN._compact(synth.t(3000)))                     # the corrected bundle's marker folded last
    staged = os.path.join(root, "staging", "adapters", held.name, f"{rev}-copy")
    _, n = RUN._reemit_committed(head_dir, staged, synth.t(4200), rev)
    assert n > 0
    from sisteer.registry import records as R
    marker = [rec for _, rec, _ in R.read_record_file(next(os.path.join(head_dir, f) for f in os.listdir(head_dir) if "source_revision" in f))]
    R.write_records(os.path.join(staged, "999998-source_revision.jsonl"), [dict(m, written_utc=synth.t(4200)) for m in marker])
    RUN.publish_staged(staged, os.path.join(RUN.activations_dir(root, held.name, rev), RUN._compact(synth.t(4200))))
    now3 = synth.t(4500)
    ing3 = H.ingest_once(root, workers=0, min_free=0, now=now3)
    assert ing3.revisions_noop == 1 and ing3.refused == 0
    assert [(r["result"], r.get("no_op")) for r in _metrics(root, now3) if r["kind"] == "bundle"] == [("ok", "head")]
    assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE status = 'duplicate'")[0][0] >= 1
    assert _q(root, "SELECT value FROM run_metric WHERE metric_id = 'asr'") == [(0.99,)]
    _rebuild_equal(root, tmp_path, seed=8)


def _rebuild_equal_local(root, tmp_path):
    from test_registry_core_fixes import _rebuild_equal
    _rebuild_equal(root, tmp_path)


class SplitOrphanAdapter(HoldingAdapter):
    """HoldingAdapter with its orphan judgment in a batch of its OWN: the revision carries a second, separate judgment file -- the
    erroneous file a corrected reader removes."""

    def read(self, ctx, group, inputs):
        for batch in StableStampsAdapter.read(self, ctx, group, inputs):
            yield batch
            from sisteer.registry.adapters import base as AB
            orphan_run = I.run_id(synth.CAMPAIGN, "never_published", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)
            extra = AB.RecordBatch(RUN.source_id_for(self.name, group), group)
            extra.add("judgment", {"judgment_id": I.judgment_id(orphan_run, synth.JUDGE_ID), "run_id": orphan_run, "judge_id": synth.JUDGE_ID,
                                   "n_items_judged": 1, "status": "complete"}, ctx.producer, ctx.now)
            yield extra


class CorrectedSplitAdapter(StableStampsAdapter):
    """The SplitOrphanAdapter corrected: the same records but for the orphan judgment file, which it no longer emits."""
    name = SplitOrphanAdapter.name


def test_a_corrected_bundle_that_removes_a_file_is_attempted_in_full_on_the_held_path_and_on_the_head_path(tmp_path):
    """The content check must compare the COMPLETE measurement multisets in both directions. A held directory carries an
    erroneous judgment in a separate file; the corrected re-read emits every other file identically and omits that one. A one-directional
    check ("every new file matches a carrier file") called it a no-op and the revision stayed held on the judgment the correction
    removed. Now it is attempted in full and activates. The head path has the same rule: a bundle of the head that omits a file the
    head carries is staged in full, never marked duplicate."""
    root, store_root = _setup(tmp_path)
    ad = SplitOrphanAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    rev_dir = os.path.join(root, "records", "adapters", ad.name, rev)
    assert sum(1 for f in os.listdir(rev_dir) if "judgment" in f) == 2                              # the orphan judgment in a file of its own
    ing0 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500))
    assert ing0.revisions_held >= 1 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    # (held path) the corrected read: one judgment file fewer, everything else identical -> NOT a no-op (a one-directional check --
    # every bundle file matches a carrier file -- would have said it was): attempted in full, staged, and the hold is REPORTED
    (bundle,) = _direct_reread_bundles(root, CorrectedSplitAdapter(store_root), [synth.t(3000)])
    measurement = [f for f in bundle.files if not any(k in f for k in ("source_revision", "source_status", "artifact"))]
    assert sum(1 for f in bundle.files if "judgment" in f) == 1
    with DB.reading(DB.db_path(root)) as conn:
        ing_ = H.Ingestor(conn, root)
        got, have = ing_._measurement_multiset("", measurement), ing_._measurement_multiset(rev_dir)
        assert all(k in have for k in got) and got != have, (got, have)            # the one-directional test passes, the multisets differ
        assert ing_._same_measurements(rev_dir, measurement) is False
    now1 = synth.t(3500)
    ing1 = H.ingest_once(root, workers=0, min_free=0, now=now1)
    lines1 = [(r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now1) if r["kind"] == "bundle"]
    assert lines1 == [("blocked", None, None)] and ing1.revisions_noop == 0 and ing1.revisions_held >= 1, lines1
    # the bundle was staged (its own files are the revision's staged rows now) -- and the revision stays held, because a revision's
    # fold is the UNION of what was staged for it and the directory's orphan judgment is still staged (the held attempt's sound
    # files fold with the bundle; no record is ever withdrawn inside one revision). A correction that REMOVES a record is a new
    # revision (a bumped EMISSION_VERSION); what this rule guarantees is that it is never mistaken for a no-op
    assert _q(root, "SELECT accepted, substr(refusal, 1, 24) FROM source_revision WHERE revision_id = ?", rev) == [(0, "activation pending: run ")]
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ? AND file_uri LIKE '%/_activations/%'", rev)[0][0] >= 1
    for f in bundle.files:
        assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, f)) == [("blocked",)]
    from test_registry_core_fixes import _rebuild_equal
    _rebuild_equal(root, tmp_path)
    # (head path) a second, fuller revision: the orphan's run published first, so a SplitOrphan read activates as the head with both
    # judgment files; a bundle omitting the orphan's file is a bundle of the head whose measurements differ -> staged in full
    _orphan_run_file(root, synth.t(3600))
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.6])
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(4000))
    rev2 = rep2.results[0].revision_id
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(4500))
    assert ing2.refused == 0 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM judgment") == [(2,)] and _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    # a re-emission of the head's committed bytes (the bundle shape) with the orphan's judgment file left out -- a re-read of an
    # accepted head carries no measurement files at all, so the omitting bundle is built from the head's directory here
    head_dir = os.path.join(root, "records", "adapters", ad.name, rev2)
    staged = os.path.join(root, "staging", "adapters", ad.name, f"{rev2}-omit")
    _, n = RUN._reemit_committed(head_dir, staged, synth.t(5000), rev2)
    assert n > 0
    from sisteer.registry import records as R
    judgment_files = sorted(f for f in os.listdir(staged) if "judgment" in f)
    assert len(judgment_files) == 2
    os.remove(os.path.join(staged, judgment_files[1]))                                             # the erroneous, separate judgment file
    marker = [rec for _, rec, _ in R.read_record_file(next(os.path.join(head_dir, f) for f in os.listdir(head_dir) if "source_revision" in f))]
    R.write_records(os.path.join(staged, "999998-source_revision.jsonl"), [dict(m, written_utc=synth.t(5000)) for m in marker])
    files2 = RUN.publish_staged(staged, os.path.join(RUN.activations_dir(root, ad.name, rev2), RUN._compact(synth.t(5000))))
    assert sum(1 for f in files2 if "judgment" in f) == 1
    with DB.reading(DB.db_path(root)) as conn:
        ing_ = H.Ingestor(conn, root)
        measurement2 = [f for f in files2 if not any(k in f for k in ("source_revision", "source_status", "artifact"))]
        got, have = ing_._measurement_multiset("", measurement2), ing_._measurement_multiset(head_dir)
        assert all(k in have for k in got) and got != have and ing_._same_measurements(head_dir, measurement2) is False
    now3 = synth.t(5500)
    ing3 = H.ingest_once(root, workers=0, min_free=0, now=now3)
    lines3 = [(r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now3) if r["kind"] == "bundle"]
    assert lines3 == [("ok", None, None)] and ing3.revisions_noop == 0 and ing3.refused == 0, lines3      # staged in full, never a duplicate
    assert _q(root, "SELECT COUNT(*) FROM record_delivery WHERE status = 'duplicate'") == [(0,)]
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, next(f for f in files2 if "judgment" in f))) == [("ok",)]
    marker2 = next(f for f in files2 if "source_revision" in f)                                   # the fold's carrier moved to the bundle's marker
    assert _q(root, "SELECT record_uri FROM source_revision WHERE revision_id = ?", rev2) == [(H.logical_uri(root, marker2),)]
    _rebuild_equal(root, tmp_path, seed=8)


class RefusingStatusHoldingAdapter(HoldingAdapter):
    """HoldingAdapter whose read reports a SOURCE-LEVEL refusal in its status facts (identical measurements)."""

    def status_extra(self, group):
        return {"refuse": ["judge rows missing for this source (synthetic)"]}


class RefusingStatusAdapter(StableStampsAdapter):
    name = StableStampsAdapter.name

    def status_extra(self, group):
        return {"refuse": ["judge rows missing for this source (synthetic)"]}


def test_identical_measurements_with_newer_status_facts_fold_the_facts_on_the_head_path_and_after_a_hold(tmp_path):
    """(1) Two re-reads of a NON-head accepted revision before any fold, identical measurements, the second reporting a
    source-level refusal: the first re-activates the revision, the second is a bundle of the head -- its measurement files duplicates,
    its marker and status folded -- so `source_status.ok` reads 0 with the refusal named and the watermarks carry the second read's
    stamp. (2) The same facts through the held path: identical no-op bundles fold their marker and status, in stamp order, when the
    carrier activates."""
    from test_registry_core_fixes import _rebuild_equal
    # ---- (1) the head path: A -> B -> A, two re-reads of A while B is the head ----
    root, store_root = _setup(tmp_path)
    ad = StableStampsAdapter(store_root)
    rep_a, ing_a = _pass_and_ingest_local(root, ad, synth.t(2000))
    rev_a = rep_a.results[0].revision_id
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8])
    rep_b, ing_b = _pass_and_ingest_local(root, ad, synth.t(3000))
    assert rep_b.results[0].revision_id != rev_a and ing_b.refused == 0
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])
    (b1,) = _direct_reread_bundles(root, ad, [synth.t(4000)])                                  # A re-read while B is the head: ok
    (b2,) = _direct_reread_bundles(root, RefusingStatusAdapter(store_root), [synth.t(4100)])   # identical measurements, status refuse
    assert b1.revision_id == rev_a == b2.revision_id
    now = synth.t(4500)
    ing = H.ingest_once(root, workers=0, min_free=0, now=now)
    lines = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op")) for r in _metrics(root, now) if r["kind"] == "bundle"]
    assert lines == [(RUN._compact(synth.t(4000)), "ok", None), (RUN._compact(synth.t(4100)), "ok", "head")], lines
    assert ing.refused == 0 and ing.blocked == 0 and ing.revisions_noop == 1
    assert _q(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 ORDER BY observed_utc DESC LIMIT 1", f"{ad.name}:s1") == [(rev_a,)]
    ok, extra, seen = _q(root, "SELECT ok, extra_json, observed_utc FROM source_status WHERE source_revision_id = ?", rev_a)[0]
    assert ok == 0 and "judge rows missing" in json.loads(extra)["refuse"][0] and seen == synth.t(4100)
    assert _q(root, "SELECT ok FROM v_source_status WHERE adapter = ?", ad.name) == [(0,)]
    assert _q(root, "SELECT DISTINCT observed_utc FROM source_watermark WHERE adapter = ?", ad.name) == [(synth.t(4100),)]
    assert _q(root, "SELECT observed_utc FROM source_revision WHERE revision_id = ?", rev_a) == [(synth.t(4100),)]
    for f in b2.files:
        (status,) = _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, f))[0]
        assert status == ("ok" if ("source_revision" in f or "source_status" in f) else "duplicate"), (f, status)
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)]
    _rebuild_equal(root, tmp_path)
    # ---- (2) the held path: the facts of identical no-op bundles fold when the carrier activates ----
    root2, store2 = _setup(tmp_path / "second")
    held = HoldingAdapter(store2)
    with DB.reading(DB.db_path(root2)) as conn:
        rep = RUN.run_pass(root2, [held], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    H.ingest_once(root2, workers=0, min_free=0, now=synth.t(2500))
    _direct_reread_bundles(root2, held, [synth.t(3000)])
    _direct_reread_bundles(root2, RefusingStatusHoldingAdapter(store2), [synth.t(3100)])
    ing2 = H.ingest_once(root2, workers=0, min_free=0, now=synth.t(3500))
    assert ing2.revisions_noop == 2 and _q(root2, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    _orphan_run_file(root2, synth.t(3600))
    now3 = synth.t(4000)
    ing3 = H.ingest_once(root2, workers=0, min_free=0, now=now3)
    lines3 = [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root2, now3) if r["kind"] == "bundle"]
    assert lines3 == [(rev, "ok", None, True), (RUN._compact(synth.t(3000)), "ok", "head", True), (RUN._compact(synth.t(3100)), "ok", "head", True)], lines3
    ok, extra, seen = _q(root2, "SELECT ok, extra_json, observed_utc FROM source_status WHERE source_revision_id = ?", rev)[0]
    assert ok == 0 and "judge rows missing" in json.loads(extra)["refuse"][0] and seen == synth.t(3100)
    assert _q(root2, "SELECT DISTINCT observed_utc FROM source_watermark WHERE adapter = ?", held.name) == [(synth.t(3100),)]
    assert _q(root2, "SELECT observed_utc FROM source_revision WHERE revision_id = ?", rev) == [(synth.t(3100),)]
    assert _q(root2, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    _rebuild_equal(root2, tmp_path / "second")


def test_no_op_bundles_facts_survive_a_budget_stop_and_a_kill_right_after_the_carrier_activates(tmp_path, monkeypatch):
    """The carrier's activation deleted every queue row of its revision, the no-op bundles' markers included,
    so a budget stop (or a kill) between that activation and the retries that fold those bundles' facts stranded the facts for good --
    their files already delivered `blocked`, no row left to revisit them. Now (1) a carrier and its no-op bundles are ONE unit of work
    the budget never parts: a clock that expires right after the activation still sees the bundles' facts folded in the same cycle,
    and the cycle's end is a state the replay reproduces exactly; (2) a KILL between the two leaves the bundles' rows standing, the
    next cycle folds their facts, and the replay is exact then."""
    from test_registry_core_fixes import _rebuild_equal

    def held_with_two_noops(root_dir):
        root, store_root = _setup(root_dir)
        held = HoldingAdapter(store_root)
        with DB.reading(DB.db_path(root)) as conn:
            rep = RUN.run_pass(root, [held], conn=conn, now=synth.t(2000))
        rev = rep.results[0].revision_id
        H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500))
        _direct_reread_bundles(root, held, [synth.t(3000)])
        _direct_reread_bundles(root, RefusingStatusHoldingAdapter(store_root), [synth.t(3100)])
        ing = H.ingest_once(root, workers=0, min_free=0, now=synth.t(3500))
        assert ing.revisions_noop == 2 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
        _orphan_run_file(root, synth.t(3600))
        return root, held, rev

    def bundle_lines(root, now):
        return [(r["dir"].rsplit("/", 1)[-1], r["result"], r.get("no_op"), r.get("retry")) for r in _metrics(root, now) if r["kind"] == "bundle"]

    # ---- (1) the budget expires right after the carrier's activation: the no-op bundles' facts fold in the same cycle ----
    root, held, rev = held_with_two_noops(tmp_path / "budget")
    clock = {"t": 0.0}
    inner = H.Ingestor.apply_revision

    def ten_seconds_each(self, rev_dir, now, **kw):
        out = inner(self, rev_dir, now, **kw)
        clock["t"] += 10.0                               # the carrier's activation alone spends the 5-s budget
        return out
    monkeypatch.setattr(H.Ingestor, "apply_revision", ten_seconds_each)
    now1 = synth.t(4000)
    ing1 = H.ingest_once(root, workers=0, min_free=0, now=now1, budget_s=5.0, clock=lambda: clock["t"])
    monkeypatch.undo()
    assert bundle_lines(root, now1) == [(rev, "ok", None, True), (RUN._compact(synth.t(3000)), "ok", "head", True), (RUN._compact(synth.t(3100)), "ok", "head", True)]
    assert ing1.stopped is None and ing1.revisions_deferred_budget == 0 and ing1.refused == 0, ing1.as_dict()
    assert _q(root, "SELECT ok FROM source_status WHERE source_revision_id = ?", rev) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    _rebuild_equal(root, tmp_path / "budget")
    # ---- (2) a kill between the carrier's activation and the retries: the rows stand, the next cycle folds the facts ----
    root2, held2, rev2 = held_with_two_noops(tmp_path / "kill")
    calls = []

    def die_at_the_first_bundle_retry(self, rev_dir, now, **kw):
        if kw.get("retry") and "/_activations/" in rev_dir:
            raise _Killed("TimeoutStartSec")
        calls.append(rev_dir)
        return inner(self, rev_dir, now, **kw)
    monkeypatch.setattr(H.Ingestor, "apply_revision", die_at_the_first_bundle_retry)
    with pytest.raises(_Killed):
        H.ingest_once(root2, workers=0, min_free=0, now=synth.t(4000))
    monkeypatch.undo()
    assert calls and calls[-1].endswith(rev2)                                                    # the carrier activated ...
    assert _q(root2, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev2) == [(1,)]
    assert _q(root2, "SELECT ok FROM source_status WHERE source_revision_id = ?", rev2) == [(1,)]   # ... and the newer facts have not folded
    rows = _q(root2, "SELECT record_uri, json_extract(extra_json, '$.no_op') FROM blocked_records WHERE record_type = 'source_revision' ORDER BY record_uri")
    assert len(rows) == 2 and all("/_activations/" in u and flag == "held" for u, flag in rows), rows   # the no-op markers stand; the carrier's rows are gone
    for _, marker in RUN.complete_bundles(root2, held2.name, rev2):
        assert _q(root2, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root2, marker)) == [("blocked",)]
    now2 = synth.t(4500)
    ing2 = H.ingest_once(root2, workers=0, min_free=0, now=now2)
    assert bundle_lines(root2, now2) == [(RUN._compact(synth.t(3000)), "ok", "head", True), (RUN._compact(synth.t(3100)), "ok", "head", True)]
    assert ing2.refused == 0 and _q(root2, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    ok, extra, seen = _q(root2, "SELECT ok, extra_json, observed_utc FROM source_status WHERE source_revision_id = ?", rev2)[0]
    assert ok == 0 and "judge rows missing" in json.loads(extra)["refuse"][0] and seen == synth.t(3100)
    assert _q(root2, "SELECT DISTINCT observed_utc FROM source_watermark WHERE adapter = ?", held2.name) == [(synth.t(3100),)]
    for _, marker in RUN.complete_bundles(root2, held2.name, rev2):
        assert _q(root2, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root2, marker)) == [("ok",)]
    _rebuild_equal(root2, tmp_path / "kill")


def test_a_re_read_of_a_touched_input_folds_its_artifact_facts_with_identical_measurements(tmp_path):
    """The artifact files sit outside the measurement comparison (the inputs are fixed by the revision id) but they are
    FACTS -- the input's `verified_utc` is its mtime, which a re-read of a touched, byte-identical input advances -- so they fold like
    the marker and the status, on the held path (at the carrier's activation) as on the head path; a re-emission without artifact
    files is a no-op still (the re-emission test above)."""
    from test_registry_core_fixes import _rebuild_equal
    root, store_root = _setup(tmp_path)
    held = HoldingAdapter(store_root)
    p = os.path.join(store_root, "s1", "store.json")
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [held], conn=conn, now=synth.t(2000))
    rev = rep.results[0].revision_id
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(2500))
    (old_verified,) = _q(root, "SELECT verified_utc FROM artifact WHERE uri_primary = ?", p)[0] if _q(root, "SELECT COUNT(*) FROM artifact WHERE uri_primary = ?", p) == [(1,)] else (None,)
    assert old_verified is None                                                                   # held: nothing of the revision is visible yet
    # the input touched: same bytes, a later mtime -> the re-read's artifact carries the new verified_utc; its measurements are identical
    st = os.stat(p)
    os.utime(p, (st.st_atime + 600, st.st_mtime + 600))
    new_mtime = os.stat(p).st_mtime_ns
    (bundle,) = _direct_reread_bundles(root, held, [synth.t(3000)])
    with DB.reading(DB.db_path(root)) as conn:
        ing_ = H.Ingestor(conn, root)
        rev_dir = os.path.join(root, "records", "adapters", held.name, rev)
        assert ing_._same_measurements(rev_dir, [f for f in bundle.files if not any(k in f for k in ("source_revision", "source_status", "artifact"))])
        assert ing_._content_sha(next(f for f in bundle.files if "artifact" in f)) != ing_._content_sha(next(os.path.join(rev_dir, f) for f in os.listdir(rev_dir) if "artifact" in f))
    ing1 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(3500))
    assert ing1.revisions_noop == 1                                                              # the held no-op: identical measurements
    _orphan_run_file(root, synth.t(3600))
    now2 = synth.t(4000)
    ing2 = H.ingest_once(root, workers=0, min_free=0, now=now2)
    assert ing2.refused == 0 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(1,)]
    (verified,) = _q(root, "SELECT verified_utc FROM artifact WHERE uri_primary = ?", p)[0]
    from sisteer.registry import records as R
    assert R.utc_to_epoch_ms(verified) == new_mtime // 1_000_000                                # the touched input's mtime, the re-read's fact
    artifact_file = next(f for f in bundle.files if "artifact" in f)
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, artifact_file)) == [("ok",)]      # a fact, never a duplicate
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", H.logical_uri(root, next(f for f in bundle.files if "item_result" in f))) == [("duplicate",)]
    _rebuild_equal(root, tmp_path)


def _pass_and_ingest_local(root, adapter, now):
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [adapter], conn=conn, now=now)
    return rep, H.ingest_once(root, workers=0, min_free=0)


def test_revision_directories_are_grouped_by_source_in_ledger_order_within_the_source(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = SyntheticStoreAdapter(store_root)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    revs = {r.group: r.revision_id for r in rep.results}
    base = os.path.join(root, "records", "adapters", ad.name)
    d1, d2 = os.path.join(base, revs["s1"]), os.path.join(base, revs["s2"])
    b1a, b1b = (os.path.join(base, "_activations", revs["s1"], s) for s in ("20260101000100000", "20260101000200000"))
    b2a = os.path.join(base, "_activations", revs["s2"], "20260101000150000")
    # the ledger order interleaves the two sources' bundles by stamp; the grouping keeps each source's own order and pulls its
    # bundles behind its directory. Sources not in the database yet are named by their marker files.
    ledger = [d1, d2, b1a, b2a, b1b]
    for b, src in ((b1a, d1), (b1b, d1), (b2a, d2)):
        os.makedirs(b)
        import shutil
        shutil.copy(next(os.path.join(src, f) for f in os.listdir(src) if "source_revision" in f), os.path.join(b, "000009-source_revision.jsonl"))
    with DB.reading(DB.db_path(root)) as conn:
        assert H.group_revision_dirs(root, ledger, conn) == [d1, b1a, b1b, d2, b2a]
        assert H.source_of_revision_dir(root, b2a, conn) == f"{ad.name}:s2" and H.source_of_revision_dir(root, d1, None) == f"{ad.name}:s1"
    # a torn directory (no marker) keeps its revision id as its key and its place
    torn = os.path.join(base, "_activations", revs["s2"], "20260101000300000")
    os.makedirs(torn)
    with DB.reading(DB.db_path(root)) as conn:
        assert H.source_of_revision_dir(root, torn, conn) == revs["s2"]      # unknown to the database, no marker: the revision id stands in
        assert H.group_revision_dirs(root, [d2, torn, d1], conn) == [d2, torn, d1]
