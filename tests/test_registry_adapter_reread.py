"""``registry adapters run --reread``: every committed source is read again and a bundle under
``_activations/<rev>/<stamp>/`` carries the read's ``source_status`` facts and a fresh marker. A current head whose keys
the read reproduces has NO measurement re-emitted (the log grows by the status rows, not by the revision); a read that
carries different keys publishes its measurements and withdrawals like a fresh revision; a refused re-read leaves the
head current and files only the watermarks' ``last_error``. The live case: 24 adapters' heads committed by a runner that
filed no status rows (main between #183 and #185, 2026-09-17 14:17–14:45Z).
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Sequence

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from test_registry_adapter_pass import SyntheticStoreAdapter, _pass_and_ingest, _setup, _store  # noqa: E402


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _reread(root, adapter, now, force=False):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep = RUN.run_pass(root, [adapter], conn=conn, now=now, reread=True, force=force)
    finally:
        conn.close()
    ing = H.ingest_once(root, workers=0, min_free=0)
    return rep, ing


def _committed_without_status(root, store_root, monkeypatch, now, n_groups=1):
    """Heads committed by a runner that filed no status rows (the pre-#185 code)."""
    ad = SyntheticStoreAdapter(store_root)
    monkeypatch.setattr(RUN, "_source_status_records", lambda *a, **k: [])
    rep, ing = _pass_and_ingest(root, ad, now)
    monkeypatch.undo()
    assert rep.groups_refreshed == n_groups and ing.refused == 0 and ing.blocked == 0
    assert _q(root, "SELECT COUNT(*) FROM source_status") == [(0,)]
    return ad, {r.group: r.revision_id for r in rep.results}


def _snapshot(root):
    return {
        "items_current": _q(root, "SELECT COUNT(*) FROM v_item_result_current")[0][0],
        "item_rows": _q(root, "SELECT COUNT(*) FROM item_result")[0][0],
        "runs": _q(root, "SELECT COUNT(*) FROM run")[0][0],
        "obs_current": _q(root, "SELECT COUNT(*) FROM observation WHERE current = 1")[0][0],
        "heads": _q(root, "SELECT source_id, revision_id FROM source_revision WHERE accepted = 1 AND superseded_by_revision_id IS NULL ORDER BY 1"),
        "paper": _q(root, "SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1")[0][0],
    }


def test_reread_files_status_for_committed_heads_without_re_emitting_their_measurements(tmp_path, monkeypatch):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad, revs = _committed_without_status(root, store_root, monkeypatch, synth.t(2000), n_groups=2)
    before = _snapshot(root)
    # an ordinary pass finds nothing to do: unchanged inputs never re-read (the gap the re-read closes)
    rep1, ing1 = _pass_and_ingest(root, ad, synth.t(2500))
    assert rep1.groups_refreshed == 0 and rep1.records_written == 0 and _q(root, "SELECT COUNT(*) FROM source_status") == [(0,)]
    # the re-read
    rep, ing = _reread(root, ad, synth.t(3000))
    assert rep.groups_reread == 2 and rep.groups_refreshed == 2 and rep.groups_refused == 0
    assert rep.inputs_changed == 0 and rep.withdrawals == 0
    assert rep.records_written == 2                                    # one status record per group, nothing else
    for r in rep.results:
        assert r.reread and r.accepted and r.revision_id == revs[r.group]
        assert f"{os.sep}_activations{os.sep}{r.revision_id}{os.sep}" in r.files[0]
        names = sorted(os.path.basename(f).split("-", 1)[1] for f in r.files)
        assert names == ["source_revision.jsonl", "source_status.jsonl"], r.files
    assert ing.refused == 0 and ing.blocked == 0 and ing.ok > 0
    # the status rows exist now, tagged with the heads' own revisions; the paper's contract reads them
    rows = _q(root, "SELECT source_id, ok, cells_measured, source_revision_id, observed_utc FROM source_status ORDER BY source_id")
    assert rows == [("synthetic_store:s1", 1, 1, revs["s1"], synth.t(3000)), ("synthetic_store:s2", 1, 1, revs["s2"], synth.t(3000))]
    (st,) = _q(root, "SELECT ok, cells_measured, extra_json FROM v_source_status")
    assert st[0] == 1 and st[1] == 2
    by = json.loads(st[2])["by_source"]
    assert by["synthetic_store:s1"]["record_counts"] == {"item_result": 3, "judgment": 1, "run": 1, "run_metric": 1, "run_state": 1}
    # the heads are re-observed, not superseded; no measurement row was written or moved
    after = _snapshot(root)
    assert after == before
    marks = _q(root, "SELECT revision_id, observed_utc, superseded_by_revision_id FROM source_revision ORDER BY source_id")
    assert marks == [(revs["s1"], synth.t(3000), None), (revs["s2"], synth.t(3000), None)]
    assert _q(root, "SELECT COUNT(*) FROM row_history WHERE table_name = 'source_revision'") == [(4,)]   # history keeps the first markers
    # idempotent: a second re-read re-files the head's status row (the row is keyed by revision: the later read of
    # the same revision is the row) and nothing else
    rep2, ing2 = _reread(root, ad, synth.t(3500))
    assert rep2.groups_reread == 2 and rep2.records_written == 2 and ing2.refused == 0
    assert _q(root, "SELECT COUNT(*) FROM source_status") == [(2,)] and _q(root, "SELECT COUNT(*) FROM v_source_status_current") == [(2,)]
    assert _q(root, "SELECT DISTINCT observed_utc FROM source_status") == [(synth.t(3500),)]
    assert _snapshot(root) == before


def test_reread_that_refuses_files_only_the_watermarks_error_and_keeps_the_head_current(tmp_path, monkeypatch):
    root, store_root = _setup(tmp_path)
    ad, revs = _committed_without_status(root, store_root, monkeypatch, synth.t(2000))
    before = _snapshot(root)
    p = os.path.join(store_root, "s1", "store.json")
    ad.broken.add(p)
    rep, ing = _reread(root, ad, synth.t(3000))
    assert rep.groups_refused == 1 and rep.groups_reread == 0 and rep.records_written == 0
    (res,) = rep.results
    assert not res.accepted and "synthetic read failure" in (res.error or "")
    # the refusal is visible where the runner files read failures; the head keeps its accepted marker and its rows
    assert _q(root, "SELECT last_error FROM source_watermark WHERE input_uri = ?", p)[0][0].endswith("synthetic read failure")
    assert _q(root, "SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", revs["s1"]) == [(1, None)]
    assert not os.path.isdir(os.path.join(RUN.activations_dir(root, ad.name, revs["s1"])))
    refused = RUN.refused_dir(root, ad.name, revs["s1"])
    assert [n.split("-", 1)[1] for n in sorted(os.listdir(refused))] == ["source_watermark.jsonl"]
    assert _snapshot(root) == before and _q(root, "SELECT COUNT(*) FROM source_status") == [(0,)]


class _ShrinkingAdapter(SyntheticStoreAdapter):
    """The same store read by a changed adapter: the last item of every store is no longer emitted."""

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for batch in super().read(ctx, group, inputs):
            dropped = [r for r in batch.records if r["record_type"] == "item_result"][-1]
            batch.records.remove(dropped)
            batch.item_keys = [k for k in batch.item_keys if k["item_id"] != dropped["payload"]["item_id"]]
            yield batch


def test_reread_whose_keys_differ_changes_no_measurement_and_names_the_drift(tmp_path, monkeypatch):
    """A revision's rows are keyed by its id and no withdrawal applies inside one revision, so a re-read can never remove a
    key: the read's records are dropped and the drift is a FACT in the status row for the reader-side gates."""
    root, store_root = _setup(tmp_path)
    ad, revs = _committed_without_status(root, store_root, monkeypatch, synth.t(2000))
    before = _snapshot(root)
    assert before["items_current"] == 3
    changed = _ShrinkingAdapter(store_root)
    rep, ing = _reread(root, changed, synth.t(3000))
    assert rep.groups_reread == 1 and rep.records_written == 1 and ing.refused == 0 and ing.blocked == 0
    (res,) = rep.results
    assert res.revision_id == revs["s1"]
    assert sorted(os.path.basename(f).split("-", 1)[1] for f in res.files) == ["source_revision.jsonl", "source_status.jsonl"]
    assert _snapshot(root) == before                                              # nothing measured moved
    (row,) = _q(root, "SELECT ok, cells_measured, detail, extra_json FROM source_status")
    extra = json.loads(row[3])
    assert row[0] == 1 and row[1] == 1
    assert extra["reread"] == {"of_revision": revs["s1"], "head_revision": revs["s1"], "reactivated": False,
                               "key_drift": {"keys_absent_from_read": 0, "keys_new_in_read": 0, "items_absent_from_read": 1, "items_new_in_read": 0}}
    assert "KEY DRIFT" in row[2] and "items_absent_from_read 1" in row[2] and "measurements unchanged" in row[2]
    assert extra["record_counts"]["item_result"] == 2                             # what the read carried, beside the head's 3 rows
    # the faithful adapter's re-read reports agreement
    rep2, _ = _reread(root, ad, synth.t(3500))
    (row2,) = _q(root, "SELECT detail, extra_json FROM source_status")
    assert "its keys reproduced, measurements unchanged" in row2[0]
    assert json.loads(row2[1])["reread"]["key_drift"] == {"keys_absent_from_read": 0, "keys_new_in_read": 0, "items_absent_from_read": 0, "items_new_in_read": 0}


def test_reread_of_a_committed_non_head_revision_reactivates_its_committed_bytes_with_a_status_row(tmp_path, monkeypatch):
    """A reverted input: the current inputs match a revision that is committed but superseded. The re-read re-activates
    it the way the incremental pass does (committed bytes, withdrawals of the outgoing head's keys) and files its status."""
    root, store_root = _setup(tmp_path)
    ad, revs = _committed_without_status(root, store_root, monkeypatch, synth.t(2000))
    rev_a = revs["s1"]
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9, 0.1])   # v2: 5 items
    rep_b, _ = _pass_and_ingest(root, ad, synth.t(3000))
    rev_b = rep_b.results[0].revision_id
    assert rev_b != rev_a and _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(5,)]
    _store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4])              # reverted to v1: rev_a's inputs
    rep, ing = _reread(root, ad, synth.t(4000))
    assert rep.groups_reread == 1 and ing.refused == 0 and ing.blocked == 0
    (res,) = rep.results
    assert res.revision_id == rev_a and f"{os.sep}_activations{os.sep}{rev_a}{os.sep}" in res.files[0]
    names = sorted(os.path.basename(f).split("-", 1)[1] for f in res.files)
    assert "item_result.jsonl.gz" in names and "run.jsonl" in names and "source_status.jsonl" in names and "source_revision.jsonl" in names
    assert res.records_written > 1 and res.withdrawals >= 1
    assert _q(root, "SELECT revision_id FROM source_revision WHERE source_id = 'synthetic_store:s1' AND accepted = 1 "
                    "AND superseded_by_revision_id IS NULL") == [(rev_a,)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)]
    assert _q(root, "SELECT n FROM run_metric") == [(3,)]
    rows = _q(root, "SELECT source_revision_id, ok, extra_json FROM v_source_status_current")
    assert [(r[0], r[1]) for r in rows] == [(rev_a, 1)]
    assert json.loads(rows[0][2])["reread"]["reactivated"] is True and json.loads(rows[0][2])["reread"]["head_revision"] == rev_b


def test_reread_cli_flag_reaches_the_runner(root_less_cli=None):
    from sisteer.registry import cli
    ns = cli.build_parser().parse_args(["adapters", "run", "--reread", "--adapter", "synthetic_store"])
    assert ns.reread is True and ns.adapter == ["synthetic_store"]
    assert cli.build_parser().parse_args(["adapters", "run", "--incremental"]).reread is False


class _BannedMetricAdapter(SyntheticStoreAdapter):
    """The pre-#207 r2_export shape: a run_metric under a banned degeneration spelling, refused by the validator, so the
    revision is HELD (accepted = 0) while its sound files stay staged."""

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for batch in super().read(ctx, group, inputs):
            metric = next(r for r in batch.records if r["record_type"] == "run_metric")
            bad = dict(metric["payload"], metric_id="at_cap_share")
            batch.add("run_metric", bad, ctx.producer, ctx.now)
            yield batch


def test_reread_of_a_held_revision_publishes_the_fresh_read_and_the_revision_is_accepted(tmp_path):
    """The live 2026-09-17 case: the pre-#207 r2_export revision is held on a banned metric spelling, the corrected adapter
    reads the SAME inputs (same revision id, its directory exists), and neither the incremental pass nor a re-read that
    re-emits the committed bytes can ever replace it. A re-read of a never-accepted directory publishes the fresh read."""
    root, store_root = _setup(tmp_path)
    held = _BannedMetricAdapter(store_root)
    rep0, ing0 = _pass_and_ingest(root, held, synth.t(2000))
    rev = rep0.results[0].revision_id
    assert rep0.groups_refreshed == 1 and ing0.refused >= 1
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(0,)] and _q(root, "SELECT COUNT(*) FROM run_metric") == [(0,)]
    # the ordinary pass leaves a held revision alone: same inputs, same revision id, the directory exists — before this rule
    # every pass re-emitted the whole revision as another activation bundle that was held again (176 MB per pass, live)
    bundles_before = sorted(os.listdir(RUN.activations_dir(root, held.name, rev))) if os.path.isdir(RUN.activations_dir(root, held.name, rev)) else []
    rep1, _ = _pass_and_ingest(root, SyntheticStoreAdapter(store_root), synth.t(2500))
    assert rep1.groups_held == 1 and rep1.groups_refreshed == 0 and rep1.groups_refused == 0 and rep1.records_written == 0
    # the result names the hold's reason — the harvester's own words for the refused constituent, so the operator reads WHY
    # rather than hunting for a runner defect (the 2026-09-20 relu_paired night)
    (held_reason,) = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev)[0]
    assert held_reason.startswith("activation pending: ") and f"held on: {held_reason}" in rep1.results[0].error
    assert (sorted(os.listdir(RUN.activations_dir(root, held.name, rev))) if os.path.isdir(RUN.activations_dir(root, held.name, rev)) else []) == bundles_before
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    # the corrected adapter's re-read
    rep, ing = _reread(root, SyntheticStoreAdapter(store_root), synth.t(3000))
    assert rep.groups_reread == 1 and ing.blocked == 0
    (res,) = rep.results
    assert res.revision_id == rev and f"{os.sep}_activations{os.sep}{rev}{os.sep}" in res.files[0]
    names = sorted(os.path.basename(f).split("-", 1)[1] for f in res.files)
    assert "run_metric.jsonl" in names and "item_result.jsonl.gz" in names and "source_status.jsonl" in names and "source_revision.jsonl" in names
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev) == [(1, None)]
    assert _q(root, "SELECT metric_id FROM run_metric") == [("asr",)]                 # the fresh read's metrics, none under a banned spelling
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)]
    assert _q(root, "SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1") == [(1,)]
    (st,) = _q(root, "SELECT ok, extra_json FROM v_source_status")
    assert st[0] == 1 and json.loads(st[1])["reread"]["fresh_read"] is True and json.loads(st[1])["reread"]["reactivated"] is False
    # the refused constituent of the held attempt stays refused by sha (visible), nothing else is refused
    assert _q(root, "SELECT COUNT(*) FROM record_refusal WHERE reason LIKE '%banned metric spelling%'")[0][0] >= 1


class _StateOfUnregisteredRunAdapter(SyntheticStoreAdapter):
    """A fresh read carrying a lifecycle state for a run nobody registered (strength_mtdiff's over-refusal runs reached
    ``judged`` with no judgment row, 2026-09-19): the dependency check blocks it on a parent that is not among the revision's
    constituents, so the bundle is held on a hold of its own."""

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for batch in super().read(ctx, group, inputs):
            state = next(r for r in batch.records if r["record_type"] == "run_state")
            batch.add("run_state", dict(state["payload"], run_id=state["payload"]["run_id"] + "ghost"), ctx.producer, ctx.now, observe=False)
            yield batch


def _rebuild_agrees(root, tmp_path, tables=("source_revision", "blocked_records", "run_metric", "item_result")):
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    a, b = DB.connect(DB.db_path(root), readonly=True), DB.connect(out, readonly=True)
    try:
        diff = H.verify_equal(a, b)
    finally:
        a.close()
        b.close()
    return {t: diff[t] for t in tables}


def test_refused_original_stands_down_for_the_fresh_read_bundle_and_the_marker_names_the_bundles_hold(tmp_path):
    """The live 2026-09-19 strength_mtdiff shape. A revision directory is held on a constituent a CHECK refused; the corrected
    adapter's re-read publishes the fresh read as an activation bundle of the same revision, and that bundle is held on a parent
    of its own (a lifecycle state for a run nobody registered). Before the fix the original directory — whose bytes never change —
    re-attempted every cycle, refused on the same line and rewrote the shared marker row with that stale reason, so every
    surface read `activation pending: record .../run.jsonl` for a revision that was really waiting on a run. The original
    stands down (its rows leave the queue) and the marker row names the bundle's hold; a corrected re-read then activates, the
    queue and the staging ledgers are empty, the paper rows appear, and a rebuild of the log reaches the same state."""
    root, store_root = _setup(tmp_path)
    held = _BannedMetricAdapter(store_root)
    rep0, ing0 = _pass_and_ingest(root, held, synth.t(2000))
    rev = rep0.results[0].revision_id
    orig_dir = f"records/adapters/{held.name}/{rev}/"
    assert ing0.refused >= 1 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE record_uri LIKE ?", orig_dir + "%")[0][0] >= 1
    (refusal0,) = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev)[0]
    assert refusal0.startswith("activation pending: record " + orig_dir)
    # the re-read whose bundle is held on a run of its own: the original's refusal is no longer the revision's reason
    rep1, ing1 = _reread(root, _StateOfUnregisteredRunAdapter(store_root), synth.t(3000))
    assert rep1.groups_reread == 1 and ing1.blocked >= 1
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev) == [(0,)]
    (refusal1,) = _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev)[0]
    assert refusal1.startswith("activation pending: run ") and refusal1.endswith("ghost (records/adapters/synthetic_store/_activations/"
                                                                                   f"{rev}/20260101005000000/000004-run_state.jsonl#2)"), refusal1
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE record_uri LIKE ?", orig_dir + "%") == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE record_uri LIKE ? AND record_type = 'source_revision'",
              f"records/adapters/{held.name}/_activations/{rev}/%") == [(1,)]
    # another cycle: the original is not re-attempted (its stale reason does not come back), the bundle's hold stands
    ing2 = H.ingest_once(root, workers=0, min_free=0)
    assert _q(root, "SELECT refusal FROM source_revision WHERE revision_id = ?", rev) == [(refusal1,)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE record_uri LIKE ?", orig_dir + "%") == [(0,)]
    assert ing2.revisions_held == 1
    # the corrected read: the revision activates from the bundle's records, nothing of it stays queued or staged
    rep3, ing3 = _reread(root, SyntheticStoreAdapter(store_root), synth.t(4000))
    assert rep3.groups_reread == 1
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev) == [(1, None)]
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ?", rev) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM revision_staging_record WHERE revision_id = ?", rev) == [(0,)]
    assert _q(root, "SELECT metric_id FROM run_metric") == [("asr",)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(3,)]
    assert _q(root, "SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1") == [(1,)]
    assert _q(root, "SELECT ok FROM v_source_status") == [(1,)]
    H.ingest_once(root, workers=0, min_free=0)            # a further cycle re-attempts nothing
    assert _q(root, "SELECT COUNT(*) FROM blocked_records") == [(0,)]
    assert all(n == 0 for n in _rebuild_agrees(root, tmp_path).values()), _rebuild_agrees(root, tmp_path)


def test_refused_first_read_is_superseded_by_the_next_accepted_read_and_the_status_reads_ok(tmp_path):
    """The regrade_twin shape (2026-09-17 → 09-18): the first read of a source refuses (no marker directory, a `_refused/<rev>/`
    marker with accepted = 0), its input then changes (the judge row arrives), and `--reread` reads it fresh under a NEW revision
    id, which is accepted. The marker fold links accepted chains only, so the refused marker stayed the source's open refusal
    for good. The fold rule stamps it `superseded_by_revision_id` = the accepted read, and `v_source_status` reads ok = 1 from
    the accepted head rather than the superseded refusal; a rebuild stamps the same rows."""
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    path = os.path.join(store_root, "s1", "store.json")
    ad.broken.add(path)
    rep0, ing0 = _pass_and_ingest(root, ad, synth.t(2000))
    assert rep0.groups_refused == 1 and ing0.refused == 0
    (rev0, acc0, ref0) = _q(root, "SELECT revision_id, accepted, refusal FROM source_revision")[0]
    assert acc0 == 0 and "synthetic read failure" in ref0
    assert _q(root, "SELECT ok FROM v_source_status") == [(0,)]
    # the input changes (a judge row lands) and the source is read again: a fresh read under a new revision id
    with open(path) as fh:
        store = json.load(fh)
    store["items"].append({"id": "q3", "score": 0.6})
    with open(path, "w") as fh:
        json.dump(store, fh)
    rep1, ing1 = _reread(root, SyntheticStoreAdapter(store_root), synth.t(3000))
    assert rep1.groups_refreshed == 1 and ing1.blocked == 0 and ing1.refused == 0
    (res,) = rep1.results
    assert res.accepted and res.revision_id != rev0
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", res.revision_id) == [(1, None)]
    assert _q(root, "SELECT accepted, superseded_by_revision_id FROM source_revision WHERE revision_id = ?", rev0) == [(0, res.revision_id)]
    rows = _q(root, "SELECT ok, revision_id FROM v_source_status")
    assert rows == [(1, res.revision_id)], rows
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current") == [(4,)]
    assert all(n == 0 for n in _rebuild_agrees(root, tmp_path, tables=("source_revision",)).values())


# --------------------------------------------------------------------------------------------
# The adapter's EMISSION_VERSION is part of the revision identity; `--reread --force` is the nonce lever
# --------------------------------------------------------------------------------------------

NEW_JUDGE_SPELLING = "synthetic judge (framework 1.0, exact match); an unscored draw leaves the denominator"


class _ParkingAdapter(SyntheticStoreAdapter):
    """The same store read by an adapter whose CODE now parks a field on the run (`run.extra_json.judge_legacy`, the #253 math500
    shape) — with its EMISSION_VERSION left where it was."""

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for batch in super().read(ctx, group, inputs):
            run = next(r for r in batch.records if r["record_type"] == "run")
            run["payload"]["extra"] = dict(run["payload"].get("extra") or {}, judge_legacy=NEW_JUDGE_SPELLING)
            yield batch


class _ParkingAdapterV2(_ParkingAdapter):
    """The same parking adapter, its version bumped for the park (the rule)."""
    EMISSION_VERSION = 2


def _run_extra(root, key):
    return [json.loads(r[0] or "{}").get(key) for r in _q(root, "SELECT extra_json FROM run ORDER BY run_id")]


def test_revision_id_folds_the_emission_version_and_the_default_reproduces_the_pre_rule_id():
    inputs = [AB.Input("u1", "/p1", "a" * 64, 1, synth.t(1)), AB.Input("u2", "/p2", "b" * 64, 2, synth.t(2))]
    sid = RUN.source_id_for("synthetic_store", "s1")
    pre_rule = I.sha16(I.canonical_json({"source_id": sid, "inputs": {"u1": "a" * 64, "u2": "b" * 64}}))   # the formula before the emission-version fold
    assert RUN.revision_id_for(sid, inputs) == pre_rule                       # undeclared: every live head's id is reproduced
    assert RUN.revision_id_for(sid, inputs, 1) == RUN.revision_id_for(sid, inputs, "1") == pre_rule
    assert RUN.revision_id_for(sid, inputs, 2) != pre_rule and RUN.revision_id_for(sid, inputs, "2") == RUN.revision_id_for(sid, inputs, 2)
    assert RUN.revision_id_for(sid, inputs, "1+emitter2") not in (pre_rule, RUN.revision_id_for(sid, inputs, 2))
    assert RUN.revision_id_for(sid, inputs, None, synth.t(9)) not in (pre_rule, RUN.revision_id_for(sid, inputs, None, synth.t(10)))
    # the declaration is read off the adapter; a malformed one refuses by name rather than minting ids under it
    assert AB.emission_version_of(SyntheticStoreAdapter) == "1" and AB.emission_version_of(_ParkingAdapterV2) == "2"
    for bad in (True, 1.5, "", "a b", ["2"], "x" * 65):
        class _Bad(SyntheticStoreAdapter):
            EMISSION_VERSION = bad
        with pytest.raises((TypeError, ValueError), match="EMISSION_VERSION"):
            AB.emission_version_of(_Bad)


def test_emission_version_bump_publishes_a_new_revision_where_an_unbumped_park_change_is_dropped(tmp_path):
    """Known-bad first (today's behaviour, intended for a code-unchanged read): the same inputs read by an adapter that parks a
    new field but did not bump reproduce the head's id and the read's records are dropped — the live math500 / over-refusal
    case. Then the rule: the bump is a new revision — published by the INCREMENTAL pass, no `--reread` — the old head is
    superseded, the paper rows carry the new park, and a rebuild of the log equals the live index."""
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep0, ing0 = _pass_and_ingest(root, ad, synth.t(2000))
    rev0 = rep0.results[0].revision_id
    assert ing0.refused == 0 and _run_extra(root, "judge_legacy") == [None]
    assert _q(root, "SELECT json_extract(extra_json, '$.emission_version') FROM source_revision") == [("1",)]
    before = _snapshot(root)
    # known-bad: the park changed, the version did not -> same id, records dropped, the live row keeps the old (absent) value
    rep1, ing1 = _reread(root, _ParkingAdapter(store_root), synth.t(3000))
    assert rep1.groups_reread == 1 and rep1.groups_refreshed == 1 and rep1.groups_forced == 0 and ing1.refused == 0
    assert rep1.results[0].revision_id == rev0 and rep1.results[0].reread
    assert _run_extra(root, "judge_legacy") == [None] and _snapshot(root) == before
    assert "its keys reproduced, measurements unchanged" in _q(root, "SELECT detail FROM v_source_status")[0][0]
    rep1b, _ = _pass_and_ingest(root, _ParkingAdapter(store_root), synth.t(3100))          # nor does the incremental pass move
    assert rep1b.groups_refreshed == 0 and rep1b.records_written == 0
    # the rule: the bump is a new revision, and the incremental pass finds it without being told
    bumped = _ParkingAdapterV2(store_root)
    rep2, ing2 = _pass_and_ingest(root, bumped, synth.t(4000))
    assert rep2.inputs_changed == 0 and rep2.groups_refreshed == 1 and rep2.groups_reread == 0 and rep2.groups_forced == 0
    assert ing2.refused == 0 and ing2.blocked == 0
    (res,) = rep2.results
    rev2 = res.revision_id
    assert rev2 != rev0 and not res.reread and res.accepted and res.records_written > 1
    assert rev2 == RUN.revision_id_for(RUN.source_id_for(bumped.name, "s1"), bumped.discover(), 2)
    assert f"{os.sep}_activations{os.sep}" not in res.files[0] and f"{os.sep}{rev2}{os.sep}" in res.files[0]   # a revision directory, not a bundle
    heads = _q(root, "SELECT revision_id, accepted, superseded_by_revision_id, json_extract(extra_json, '$.emission_version') "
                     "FROM source_revision ORDER BY observed_utc")
    assert heads == [(rev0, 1, rev2, "1"), (rev2, 1, None, "2")]
    assert _run_extra(root, "judge_legacy") == [NEW_JUDGE_SPELLING]
    assert _q(root, "SELECT json_extract(run_extra_json, '$.judge_legacy') FROM v_paper_long WHERE in_current_paper = 1") == [(NEW_JUDGE_SPELLING,)]
    after = _snapshot(root)
    assert after["items_current"] == 3 and after["runs"] == 1 and after["paper"] == 1 and after["heads"] == [("synthetic_store:s1", rev2)]
    assert after["item_rows"] == 6                                            # both revisions' rows are records (history keeps rev0's)
    assert _q(root, "SELECT DISTINCT source_revision_id FROM v_item_result_current") == [(rev2,)]      # the head's rows are the current ones
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", rev0) == [(3,)]
    assert _q(root, "SELECT COUNT(*) FROM withdrawal") == [(0,)]              # the same keys under a new revision: nothing was REMOVED
    assert _q(root, "SELECT ok, revision_id FROM v_source_status") == [(1, rev2)]
    # idempotent under the new version: a re-read reproduces rev2 and drops its records; the incremental pass writes nothing
    rep3, _ = _reread(root, _ParkingAdapterV2(store_root), synth.t(5000))
    assert rep3.groups_reread == 1 and rep3.results[0].revision_id == rev2 and rep3.records_written == 1
    rep4, _ = _pass_and_ingest(root, _ParkingAdapterV2(store_root), synth.t(5500))
    assert rep4.groups_refreshed == 0 and rep4.records_written == 0
    assert _q(root, "SELECT COUNT(*) FROM source_revision") == [(2,)]
    # a rebuild replays the log: both revisions are records, the same head, the same rows
    diff = _rebuild_agrees(root, tmp_path, tables=("source_revision", "run", "run_metric", "item_result", "source_status", "observation"))
    assert all(n == 0 for n in diff.values()), diff


def test_reread_force_publishes_a_new_revision_regardless_of_inputs_and_records_the_nonce(tmp_path):
    """The operational lever: an adapter whose code changed with no version bump. `--reread --force` publishes the read as
    a NEW revision whatever its inputs — the pass stamp is the id's nonce and the marker records it — the old head is superseded,
    and a second forced pass is another revision again (a nonce is never reproduced)."""
    root, store_root = _setup(tmp_path)
    rep0, _ = _pass_and_ingest(root, SyntheticStoreAdapter(store_root), synth.t(2000))
    rev0 = rep0.results[0].revision_id
    changed = _ParkingAdapter(store_root)                                     # a park change nobody versioned
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        with pytest.raises(ValueError, match="re-read only"):
            RUN.run_pass(root, [changed], conn=conn, now=synth.t(2500), force=True)      # force without reread is refused
        rep1 = RUN.run_pass(root, [changed], conn=conn, now=synth.t(3000), reread=True, force=True)
    finally:
        conn.close()
    ing1 = H.ingest_once(root, workers=0, min_free=0)
    assert rep1.groups_refreshed == 1 and rep1.groups_forced == 1 and rep1.groups_reread == 0 and rep1.inputs_changed == 0
    assert ing1.refused == 0 and ing1.blocked == 0
    (res,) = rep1.results
    rev1 = res.revision_id
    assert res.forced and res.accepted and not res.reread and rev1 != rev0
    assert rev1 == RUN.revision_id_for(RUN.source_id_for(changed.name, "s1"), changed.discover(), 1, synth.t(3000))
    marks = _q(root, "SELECT revision_id, accepted, superseded_by_revision_id, json_extract(extra_json, '$.forced'), "
                     "json_extract(extra_json, '$.emission_version') FROM source_revision ORDER BY observed_utc")
    assert marks == [(rev0, 1, rev1, None, "1"), (rev1, 1, None, synth.t(3000), "1")]
    assert _run_extra(root, "judge_legacy") == [NEW_JUDGE_SPELLING]
    assert _q(root, "SELECT json_extract(run_extra_json, '$.judge_legacy') FROM v_paper_long WHERE in_current_paper = 1") == [(NEW_JUDGE_SPELLING,)]
    snap = _snapshot(root)
    assert snap["items_current"] == 3 and snap["runs"] == 1 and snap["paper"] == 1 and snap["heads"] == [("synthetic_store:s1", rev1)]
    assert _q(root, "SELECT ok, revision_id FROM v_source_status") == [(1, rev1)]
    # the incremental pass after a forced read is QUIET: the head's marker records the version this code carries and no input
    # moved. (Keyed on the recorded version, not on the head's id: a nonce id never equals the reconstructed one, and comparing
    # ids would have re-activated rev0 here -- the park-less bytes -- and silently undone the forced read.)
    rep2, ing2 = _pass_and_ingest(root, changed, synth.t(4000))
    assert rep2.groups_refreshed == 0 and rep2.records_written == 0 and ing2.refused == 0
    assert _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1 AND superseded_by_revision_id IS NULL") == [(rev1,)]
    assert _run_extra(root, "judge_legacy") == [NEW_JUDGE_SPELLING]
    # a plain re-read of the forced head: its id is not this code's unforced id, so the pass reads the UNFORCED revision fresh --
    # rev0's id, whose directory exists -> the committed rev0 bytes are re-activated (a reverted input's path). That is the cost
    # of a nonce head under an unversioned code change, and why --force is the one-off and the bump is the rule.
    rep3, _ = _reread(root, changed, synth.t(4500))
    assert rep3.groups_refreshed == 1 and rep3.results[0].revision_id == rev0
    assert _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1 AND superseded_by_revision_id IS NULL") == [(rev0,)]
    assert _q(root, "SELECT DISTINCT source_revision_id FROM v_item_result_current") == [(rev0,)]
    # (how `run.extra_json` folds across revisions -- the harvester merges parked keys rather than replacing the object -- is the
    # harvester's rule, not pinned here; the head and the current item rows are rev0's)
    # a second forced pass is another revision again (a nonce is never reproduced)
    rep4, _ = _reread(root, changed, synth.t(5000), force=True)
    assert rep4.groups_forced == 1 and rep4.results[0].revision_id not in (rev0, rev1)
    assert _q(root, "SELECT COUNT(*) FROM source_revision") == [(3,)]
    assert _run_extra(root, "judge_legacy") == [NEW_JUDGE_SPELLING]
    diff = _rebuild_agrees(root, tmp_path, tables=("source_revision", "run", "run_metric", "item_result", "source_status"))
    assert all(n == 0 for n in diff.values()), diff
    # the CLI: the flag reaches the runner, and --force without --reread is refused before anything is loaded
    from sisteer.registry import cli
    ns = cli.build_parser().parse_args(["adapters", "run", "--reread", "--force", "--adapter", "synthetic_store"])
    assert ns.force is True and ns.reread is True
    assert cli.build_parser().parse_args(["adapters", "run", "--reread"]).force is False
    assert cli.cmd_adapters(cli.build_parser().parse_args(["adapters", "run", "--force"])) == 2


def test_every_registered_adapter_declares_a_well_formed_emission_version_and_the_changed_readers_bumped():
    """The declarations of the live adapters parse under the rule, and the readers whose parks changed are not at the default:
    the eleven `RowEmitter` readers carry the shared emitter's version (`paper_dose`); math500 and arditi_allpos each carry their
    own bump beside it (arditi_allpos: `paper_dose` = the raw coefficient c, the paper's RAW-frame dose for that source);
    overrefusal_items its own. `cap4096_declarations` borrows a path constant from `declared_common` and emits no row, so it
    stays."""
    import importlib
    from sisteer.registry import adapters as ADP
    from sisteer.registry.adapters import declared_common as DC
    versions = {}
    for mod_name in ADP.adapter_modules():
        mod = importlib.import_module(f"sisteer.registry.adapters.{mod_name}")
        for cls in ADP.adapter_classes(mod):
            versions[cls.name] = AB.emission_version_of(cls)
    assert len(versions) > 20
    # emitter 3 (2026-09-21, substrate provenance): every declared run's field_confidence_json.substrate names its evidence, a reader's bare
    # constant files `assumed_default` (never `recorded`), a grading-host-only store path files the substrate unknown -- a shared-emitter park
    # change, so one bump republishes every declared source; the readers whose OWN substrate logic changed bump their own half too
    # emitter 4 (2026-10-03, adapter re-anchoring): the anchor currency is consulted for every declared contrast
    # (`ctx.anchor_currency().choose`), `extra.anchor_selection` lands on every contrast, a comparison with no current anchor at the exact
    # key parks on the metric row's `extra.delta_block` and its old key is withdrawn by the new head, and a re-picked anchor files the
    # registry's own statistic under the family's `REANCHOR_STATISTIC` -- a shared-emitter change, so one bump republishes every declared
    # source (the window plans it); relu_paired 15 -> 16 and em_misalign 10 -> 11 bump their own halves too (their own anchor sites -- the
    # Petri ledger scan, the injected seed-view / 16,384-step anchors -- now stamp the selection)
    assert DC.EMITTER_VERSION == 4
    # 2026-09-23: American spelling of the prose the readers write on their records (defence -> defense: effect_kind / effect_text /
    # method_text, the parked contrast notes, the pending dispositions' reasons, the substrate reasons) -- arditi_strength 1 -> 2,
    # alphasteer_rebake 2 -> 3, strength_ladder 2 -> 3, relu_paired 10 -> 11, em_misalign 9 -> 10, r2_export 7 -> 8, arditi_vendor_mt 2 -> 3,
    # seedrep_cluster 8 -> 9, seedrep_statics 3 -> 4, arditi_allpos 3 -> 4; inputs unchanged, so only the bump republishes. Run ids
    # never carry record prose (identity.run_id hashes campaign / eval / instrument / producer keys), so no id moves.
    # 2026-09-24 (the over-refusal headline becomes the plain refused rate): every over-refusal source files `xst_refused` /
    # `orb_refused` BESIDE `over_refusal_graded` on the same run and judgment -- k = the judged benign items labeled full_refusal, n = the graded
    # row's n, Wilson 95 %, the same abstentions; a steered reading's contrast against its same-standup unsteered reading on the new metric --
    # so the switch (chunk 2) strips no source from the paper's over-refusal figures: arditi_strength 2 -> 3, alphasteer_rebake 3 -> 4,
    # strength_ladder 3 -> 4, arditi_allpos 4 -> 5, sampled_controls 2 -> 3, r2_export 8 -> 9 (relu_paired 13 -> 14 on its own branch). The
    # graded rows' bytes do not move (dry run against the live mirrors, the landing record); every over-refusal run gains a sibling record
    # 2026-09-24 (a THIRD over-refusal row): every over-refusal source files the label mean `over_refusal_label_mean`
    # (full 1 / partial 0.5 / compliance 0 by label, an at-cap reply by its label -- the graded metric without its 0.5 at-cap override -- over the
    # graded row's judged items, a normal 95 % interval from the exact per-item variance of the {0, 0.5, 1} distribution, the same abstentions; a
    # steered reading's contrast against its same-standup unsteered reading from a fresh generator) beside the graded and refused-rate rows:
    # arditi_strength 3 -> 4, alphasteer_rebake 4 -> 5, strength_ladder 4 -> 5, arditi_allpos 5 -> 6, sampled_controls 3 -> 4, r2_export 9 -> 10,
    # relu_paired 14 -> 15, cellstore 5 -> 6, orxs_seedrep 3 -> 4, strength_mtdiff 8 -> 9, seedrep_controls 1 -> 2. The graded and refused-rate
    # rows' bytes do not move (dry run against the live mirrors, the landing record); every over-refusal run with a label census gains a record
    assert versions["arditi_strength"] == DC.emission_version(4) == f"4+emitter{DC.EMITTER_VERSION}"
    # alphasteer_rebake 2: every row names its substrate witness (rung / MMLU routing tables and defense sampler-resolution paths derived,
    # the OR-Bench record's `cluster` field recorded, the refit default assumed with its basis, the refit multi-turn unknown)
    assert versions["alphasteer_rebake"] == DC.emission_version(5)     # 5: the label-mean siblings (the refit reference carries no partial count and files none); 4: the refused-rate siblings on the XSTest ladder, the OR-Bench ladder and the refit reference
    # bcb_bfcl 2: the source's substrate is Modal (every store manifest's usage_ledger on /vol/tinycap), assumed by default and derived
    # from the manifest where one is read; version 1's `cluster_a` was contradicted by every manifest
    assert versions["bcb_bfcl"] == DC.emission_version(2)
    # strength_ladder 2: the node / job / config rule files `derived`, never a bare value
    assert versions["strength_ladder"] == DC.emission_version(5)       # 5: the label-mean sibling; 4: the refused-rate sibling on every graded XSTest row (read, not recomputed)
    # math500 3: the substrate per served store from its ledger binding / launch paths (one llama entry on cluster B); 2: the distilled label
    assert versions["math500"] == DC.emission_version(3)
    # relu_paired's own half 4: half 3 (the sweep grammar, best-of-N, MMLU, Petri, the difference_relu arm, expected-instrument dispositions,
    # the objects' parked stem / fit-variant prefix) plus the token-cap ladder (base + `_cap<N>` rung stores under one ladder instrument, the
    # highest rung superseding the rungs below, a > 1 % base rung filed non-final, the paper's pooled `pooled_t0_t1` run per (point, family)),
    # the controls' 1 % rule on their own at-cap share, the multi-turn seed rule the target sent (`per_goal_turn`), and the same-standup `none`
    # runs + contrasts of the statics / multi-turn attacks / controls. Own half 5 (2026-09-20, the coverage audit's four reader gaps): a static
    # rung is final by the ladder's DECLARED exit / the cap-coherent at-cap share (loops never climb) and files its raw / coherent / loop shares;
    # the ungraded 16,384 ceiling rung files as a `generated` run (`exit_ceiling`) and the judged rung below it reads `exit_ceiling_ungraded_above`;
    # judgments carry the store's grading code sha per root; statics / controls carry the served tree (run.code_sha + extra.served_tree_sha).
    # Own half 6 (2026-09-20, grading provenance by attack): the per-attack sidecar `grading_provenance.by_attack.json` names each judgment's
    # grading unit where it exists (`recorded`, the rewritten store-level file filed as extra.superseded_provenance; a sidecar lacking the attack
    # refuses the judgment by name); a store with neither file falls back to the live SHA_GRADING pin stamped `unverified` (v5 left it NULL).
    # Own half 7 (2026-09-21, the relabel-do-not-remeasure rule): a steered over-refusal reading whose point grades no unsteered
    # `none` carries the base model's own unsteered controls as an EXTERNAL anchor on its metric's delta_block — bound by identity to the
    # sampled_controls run, `external_base_lane`, the dose readout's anchor_source, no delta (another source's value; a closed anchor_kind
    # vocabulary) — so every paired over-refusal metric row's extra_json changes. Own half 8 (2026-09-21): the producing cluster of every
    # non-Petri run is derived per run from the store's own evidence (shard routing table / controls manifest / spool, prefix rule, `derived`;
    # pooled members must agree; nothing named -> `unverified` with the reason) where v7 stamped `modal, recorded` from a constant, so every
    # such run's bytes change; v7 was read live before this landed, so the re-read needs the bump to republish
    # 9 (2026-09-21): the substrate claims spelled under the emitter's one contract (routing-table / spool paths `_substrate_derived`, the Petri
    # drive.log `cluster=` the producer field, recorded); 8 derived them per run from store evidence against the old contract
    # 10 (2026-09-21): the over-refusal controls' escalation verdict is the lane walk's declared exit (`ladder.json`
    # left_ladder_at), the judged store at the walk's completion cap the reading; the raw at-cap share is the gate's spool count (gen_stats) with
    # the walk's decomposition as degeneration; one-trial readings `single_trial_final_rung`; an unrequested higher store provenance-only -- ten
    # pairs the store-block rule left pending file, so every controls row's bytes change
    # 11 (2026-09-23): American spelling of the reader's record prose (defence -> defense), the shared bump above; inputs unchanged
    # 12 (2026-09-23): the controls' at-cap decomposition at the completion cap is the walk record's RECORDED `decisions[<cap>]`,
    # compared like for like with the record's own `n_at_cap[<cap>]`, and the store's gen_stats count is compared with that recorded count;
    # `left_ladder_at`'s derived copy of the four classes rides extra (`left_ladder_at_classes_by_trial`), informational, never a hold condition --
    # 10 / 11 compared the copy with the store count and held paired_d1p25_s1 t1 OR-Bench on a copy that summed to the 512 count (110; decisions["2048"]
    # and n_at_cap["2048"] both 107 = gen_stats). Every controls row read through a walk changes bytes (the degeneration measures' source text, the
    # new extra keys); no value moves
    # 13 (2026-09-24): the controls INPUTS are the flip_pass / ladders CENSUS by name (every store of a point at every cap and trial,
    # every walk record), never the owner's index's list of pairs -- a set-scoped index rewrite shrank four groups' input lists 14-25 % with every file
    # on disk and the input-drop guard refused their reads (s1, gemma4e4b_s1, llama8b_s1, nemotron3nano4b_s1); the index decides finality where it
    # names a pair, the pair's own walk record where it does not (the judged store at the declared completion cap), neither -> PENDING by name as
    # before. Every controls row gains `extra.controls_index_names_pair` (1 / 0) and `extra.finality_source` (`index` / `walk_record`); no value moves
    # 14 (2026-09-24): every over-refusal controls reading (pooled_final_rung / single_trial_final_rung; xstest and or_bench_hard; steered and the
    # same-standup `none`) files the plain refused rate `xst_refused` / `orb_refused` BESIDE `over_refusal_graded` on the same run and judgment (k = full
    # refusals, n = the judged items of the scored benign subset, Wilson 95 %, the same n_abstentions; a pooled row pools k / n per item as the graded row
    # does; a graded same-standup `none` gives the new metric its own paired contrast, the external anchor parks its binding on it too) -- the ids
    # cellstore / strength_mtdiff / arditi_vendor_mt already file, so the difference_relu family carries the headline candidate being switched to.
    # The graded metric row's bytes do not move; every controls run gains a sibling run_metric (and `extra.over_refusal_headline`), so the bump republishes
    # 16 (2026-10-03): the anchor currency decides every contrast (the Petri rows' own ledger scan names the own anchor, `anchor_current` its
    # currency; a T2 re-pick refuses by name); `extra.anchor_selection` on every contrast
    assert versions["relu_paired"] == DC.emission_version(16) == f"16+emitter{DC.EMITTER_VERSION}"      # 15: the label-mean row on every controls reading
    # the 2026-09-19 relabel moved the difference_relu label off the distilled student (now difference_relu_distilled, key
    # diff_psr_distilled): every reader that parks that cell's arm / label bumped its own half (the registry's census of its sources) to 2;
    # 3 (2026-09-21) is each one's substrate provenance: gsm8k_p200 per served store from its launch / ledger paths (9 of 142 entries are cluster B),
    # llama8b_wave from the view sidecar's dir keys (else the source's assumed default), relu_native off the grades root's cluster tag (the
    # anchors unknown, the GSM8K cards an assumed default), em_misalign from the store's `substrate` field / manifest path (else assumed)
    for name in ("relu_native", "gsm8k_p200", "llama8b_wave"):
        assert versions[name] == DC.emission_version(3), (name, versions[name])
    # em_misalign 4 (2026-09-21): the three-seed family's agentic-misalignment stores (source misalign_seedrep, six views under the run-state
    # mirror, the shared anchor parked on every metric by identity) -- new groups and new ids, so the census moves
    # 5 (2026-09-21): the plain 3,000-cap none a seed-view lane pairs against is registered as a run (rung 0 of its ladder, its own instrument) when
    # its store judged it, the re-judge as a second judgment on it, and every seed row's parked anchor block as a run_contrast against it
    # 6 (2026-09-21): the 16,384 step of the agentic ladder (source misalign_16k, five roots under the run-state mirror by absolute path, the canon
    # root's none the step anchor) AND every agentic row's decode provenance read from its meta.json (the wire body recorded, the engine fill
    # derived on decode_resolved_json, runner_forced_T1 the derived policy match; no decode VALUE moved) -- new groups and new markers, so the census moves
    # 7 (2026-09-22): the completion-cap MERGES of the agentic ladder (source misalign_ladder, six roots under the run-state mirror by absolute
    # path; the merged whole-battery value on the [3000, 16384] ladder read at its final level, every contrast parked -- no merged baseline) --
    # new groups, so the census moves
    # em_misalign 8 (2026-09-22): the paired-write difference-ReLU agentic roots of the other models (source misalign_paired_allmodels, read by path
    # under the seed roots' mirror) and judge-pool ESCALATION CHAINS (`lowprio>highprio`: the pool of record is the last element, the chains counted on
    # the judgment) accepted on every root -- version 7 held the bl merged leaking key and the s2 murder key on them, so the bump re-files both
    # em_misalign 11 (2026-10-03): the anchor currency decides every contrast -- the injected seed-view / 16,384-step anchors are the OWN anchor and
    # its declared judgment (the second judgment), the held-out agentic_murder `none` parks its contrasts (strict) and the new head withdraws
    # their old keys; a re-picked anchor files the registry's unpaired rate difference (REANCHOR_STATISTIC); `extra.anchor_selection` on every contrast
    assert versions["em_misalign"] == DC.emission_version(12), versions["em_misalign"]      # 12: a row whose object the registry does not carry is parked by name (registry follow-ups-4 item 8)
    # strength_mtdiff 3 (2026-09-21): the XSTest rows are read from the `controls_cap<N>/` trees BY NAME -- one run per (standup, object,
    # cap) on the sampled vendor-preset ladder instrument through the shared served-controls reader (headline, graded mean, items, at-cap
    # measure, same-standup contrast), the dose stamped from the stem and family, the judge pool and the grading tree's dirty paths per
    # store -- and the pooled `xstest_rates.json` mean (filed by 1-2 as the cap-512 value under the superseded greedy instrument; its
    # array positions carry no cap) is no longer a value on any row. New ids and values over the same inputs, so it bumps.
    # 4 (2026-09-21): the response-level degeneration handoff (`degeneracy/degeneracy_dose.json`, the peer instrument imported
    # unmodified) is filed as `run_degeneration` rows on the multi-turn runs the reader mints -- matched by (standup, eval, object key), the
    # vocabulary's measures, `not_measured` placeholders for the two the instrument lacks, the file an input of every standup it names -- beside every v3
    # record re-emitted byte-identical; new rows over the same standups, so it bumps.
    # 5 (2026-09-21): the silent zero -- the dose-1.25 standup read twice as an accepted EMPTY revision because every run row refused on the
    # standup's own spelling missing from legacy_spellings.yaml model_keys, with the refusals in source_state only. The run's model now
    # resolves through the SOURCE'S spelling (the eight standups' bytes do not move), a standup whose aggregates file no run refuses the
    # group by name, every row refusal rides the status facts, a handoff row whose decisive block reads absent is held by name, and the
    # newest `degeneracy_dose.<stamp>.json` is the handoff input. New runs (the 1.25 standup) and new facts, so it bumps.
    # 6 (2026-09-22): one degeneracy rule across the paper. The handoff's `any` is no longer a MEASURED `union5` row (the
    # frame read it as `frac_degenerate_union5` on all 36 multi-turn rows of the nine standups): `union5` is the `not_measured` placeholder,
    # the `any` reading is parked on the run (`extra_json.degeneration_any_rule`), and the summary's by_run_id route lands the canonical union5.
    # Same inputs; the run records and the union5 rows change, so it bumps
    # 7 (2026-09-22): ONE FAMILY, ANY SEED -- the family segment reads as `<family>s<N>` by one rule, `FAMILIES` keyed by the seed-free
    # family with the staged infix parametrised by the seed, so the seed-2 / seed-3 difference-LoRA standups file instead of refusing
    # as an unknown family; every steered run carries `extra.fit_replicate_label` / `extra.fit_seed`. The seed-1 run ids do not move (no
    # rename); the stamps change the run records, so it bumps
    # 8 (2026-09-24): BOTH controls file -- the OR-Bench-hard stores are staged beside the XSTest ones, each side at the cap it left the
    # ladder at -- each store under the ONE control its manifest declares (two controls, or one this reader does not file, refuse by name
    # BEFORE the parse, never with a dimension lookup's text), on its own ladder instrument; the contrast's anchor is chosen HERE, per control,
    # among the stores this read files (the store's own cap when it holds one, else the unsteered side's exit cap with both caps on the
    # contrast) and the contrast is guarded like the store read (the shared reader's first-in-listing anchor pick, read outside the per-store
    # guard, had refused six whole seeded standups with `Unresolved` on the OR-Bench unsteered store). No run id moves and the XSTest records
    # re-emit byte-identical; the OR-Bench rows are new and the facts (`by_cap.<cap>.<control>`, `runs_by_control`, `contrast_rows`,
    # `contrasts_refused`, `contrasts_cross_cap`) change, so it bumps -- once, for this landing
    # 9 (2026-09-24): the controls stores gain the label-mean row and contrast through the shared store reader
    # 10 (2026-10-03): every controls contrast is decided by the pass's anchor currency and
    # carries `extra.anchor_selection`; the version-8 per-control pick is the OWN anchor, kept whatever the two caps while it is a current run
    # with a usable reading; a retired, unfolded, ambiguous, refused or absent own anchor searches the EXACT key -- a same-source store under
    # another standup directory is re-read through its own cap's reader and paired per item (`sibling_none`), another source's roster-matched
    # run re-anchors the refused-rate headline as the registry's unpaired rate difference while the label-mean and graded contrasts park by
    # name, a cap-only match parks (the "else any cap" tier names an own anchor and widens nothing) -- and the own pairing's numbers park on the
    # metric row's `extra.delta_block`. Every contrast's and every steered headline row's bytes move, so it bumps; no run id moves
    assert versions["strength_mtdiff"] == "10"
    # 2026-09-20: every reader of a FITTED object's runs that had filed `is_operating_point` NULL now stamps it (1.0 = the operating
    # point) — multiturn_grades 3 and r2_export 4 beside their filed nominal (lineage_common.run_payload_base); the sampled
    # statics / controls and the shared store reader (cellstore 2, overrefusal_items 4, sampled_controls 2) from the served key's dose
    # (`_items_common.fitted_strength_payload`); r2_export 4 also reads a run's judged count from its counted row whatever row the export
    # lists first (the Petri unsteered judgments of the views carrying degeneration rows had `n_items_judged` NULL), and resolves the
    # all-positions object key as its own object kind (`vector_all` under the label `arditi_allpos`) instead of the prefill arm's Petri view
    # object, so its three multi-turn rows of that key change; `multiturn_grades` reads no root carrying that key today, so that grammar
    # change leaves its bytes alone (its 3 is the fitted operating point alone). cellstore also = the Arditi vendor re-run's controls
    # roots (arditi_vendor_mt 1 is that re-run's own reader), and an unjudged store skipped rather than refused.
    # r2_export 5 and cellstore 3, not 4 and 2: two branches each bumped those two readers by one on their own (r2_export 3 → 4,
    # cellstore 1 → 2) for DIFFERENT changes, so the merge of the two bumps once more — a rebuild off either branch's version republishes
    # under the union. overrefusal_items 4 and sampled_controls 2 belong to one of those branches alone: the other's one default change to
    # the shared StoreReader (serving_path read off the serving identity when the manifest spells a path the column does not know) fires on
    # none of the 74 / 175 manifests those two readers discover (every one spells `lens` or `passthrough`, measured 2026-09-20), so their
    # bytes did not move.
    # multiturn_grades 4 (2026-09-24): the reader also discovers the roots the registry's own
    # declaration (`configs/registry/multiturn_declared_roots.yaml`, a sha'd `_lane_files` input) names when no lane file does -- the two
    # si-constitution run-2 stores (gemma pair, nemotron crescendo) -- and files a root whose harness attacker-identity check is REFUSED
    # only under the ruling the declaration names, stamping `extra.attacker_attestation` and the attacker fields' confidence entries
    # `attested_independently` on every row it files from such a root. The lane-file roots' rows do not move (the gate and the stamp
    # fire on declared roots alone); the two new groups are new rows, so it bumps
    assert versions["multiturn_grades"] == "4"
    # r2_export 6 (2026-09-22): the judged item count of every over-refusal run the export carries a count for. 118 over-refusal judgments
    # (59 per eval) read `n_items_judged` NULL at 5: 80 controls-store runs whose only rows are the at-cap decomposition (`at_cap_share_all`,
    # "of the N replies judged (every label)") filed under the store's own producer key, 16 sampled re-run runs whose `over_refusal_rate_full*`
    # rows count judged LABELS over two trials (the item count is the pooled block's `n_items` in the sha-pinned `final_rung.json`), 22 whose
    # only judged-metric row is pending (still NULL, the reason on the judgment). `v_roster_match`'s coverage test (judged + declined + errored
    # >= items) failed every one, so the four Llama ReLU-steer over-refusal runs read `not_in_roster` under roster v4's `figR_overrefusal_*`
    # entries. The export's inputs are unchanged (309 pins, same shas), so only the bump republishes the group (the 2026-09-22 scratch dry run
    # against the live database read-only: the same 2,040 runs and 1,852 judgments, 96 counts filled, 22 left NULL with the reason, 64
    # over-refusal runs move out of `not_in_roster` -- 2 selected, 25 superseded, 5 excluded per eval -- and nothing loses selection)
    # r2_export 7 (2026-09-22): the over-refusal HEADLINE (`over_refusal_graded`) derived on the sampled re-run's runs from their two pooled
    # label rates -- exact for the two-term label mean (full 1, partial 0.5 over the shared judged labels), computed_by registry under its own
    # computation row, naming the two source rows and the convention terms it lacks (the at-cap 0.5 term, per-item pooling). The paper's
    # reader draws the headline only, so the four Llama ReLU-steer over-refusal runs selected at 6 could not draw. New rows over the same
    # inputs, the rate rows unchanged, so it bumps
    # r2_export 9 (2026-09-24): the export's own full-refusal rates -- `over_refusal_rate_full` (sampled re-run, pooled judged LABELS over two
    # trials, n about 2 x the items) and `over_refusal_rate` (greedy 512 store, one draw per item) -- TWINNED under the registry's refused-rate id
    # for the run's eval on the same run, judgment and computation, same value / k / n, the source spelling and unit on the twin's extra; and the
    # degeneracy mirror pass stamps an over-refusal run's graded row and twins beside the eval's headline row, so the headline switch of chunk 2
    # moves none of its bytes. New rows over the same inputs, so it bumps
    # r2_export 10 (2026-09-24): the label mean `over_refusal_label_mean` DERIVED from the two pooled label rates on every run that files
    # both (numerically the derived graded row here: the export carries no at-cap cross-tab); a run with the greedy `over_refusal_rate` alone has no
    # partial count and gets none. New rows over the same inputs, so it bumps
    # r2_export 11 (2026-10-03): every contrast is decided by the pass's anchor currency -- the own unsteered run
    # when current (the producer's record plus `extra.anchor_selection`), else a current unsteered run of the EXACT key (sibling_none / borrowed) with
    # the REGISTRY's statistic (an unpaired rate difference from the row's k / n and the anchor's reading; a per-behavior paired difference over
    # registry item rows for petri_score) under its own computation row, the producer's delta parked on the metric row; no current anchor parks by
    # name and the old key is withdrawn; the `full_dose` kind checks its 1x anchor's currency and parks a retired one on `delta_vs_full_dose_block`.
    # Every contrast's bytes move (the selection block), two computation ids move (they quote the version), so it bumps
    # r2_export 12 (2026-10-03, commit 4): the refused-rate TWIN's contrast (`xst_refused` / `orb_refused`)
    # is a comparison of its own -- decided at the twin's id against the anchor's reading at that id, its own entry in the decision set, parked on
    # the twin row by name, re-anchored from the twin's own count -- where 11 relabeled the source spelling's decision and so filed no refused-rate
    # contrast whenever the source parked on an eligible run carrying only the refused rate (live: every other over-refusal source's unsteered
    # runs). Its own number because the runner replays only the comparisons a head RECORDED: an 11 head that parked the source recorded
    # no twin comparison, its replay matched its fingerprint and the group stayed quiet with the contrast absent; the bump re-reads it
    assert versions["r2_export"] == "12"
    # bon_stores 3 (2026-09-21): the same stamp reaches the campaign best-of-N runs -- every fitted object's 512 and climb run is built by
    # `lineage_common.run_payload_base` and files `is_operating_point` 1 with `strength_unit` nominal_only at the campaign nominal 1.0 -- but
    # that change bumped r2_export and multiturn_grades only, so bon_stores' 64 fitted run records (32 objects x the 512 store and its climb) stayed
    # at the 2026-09-20 14:56Z read (NULL / the unit default) with their 76 inputs unchanged, and `v_roster_match`'s operating_point rule
    # passed none of them (four readings had no in-frame best-of-N row). The stamp is the only change; 2 read the registered
    # paired-write points.
    # bon_stores 4 (2026-09-22): a registered point's 512 reading is filed on NO run -- neither relu_paired's run id (version 2's attach)
    # nor a second 512 run of this reader (version 2's contract, kept) -- its counts ride the climb run's `extra.store_512_counts` and the
    # rows not filed are counted in the facts. Version 3's first read carrying the gemma / nemotron paired-write points filed 48
    # degeneration / metric records on 8 run ids it did not emit, two of them runs relu_paired never files (no best-of-N aggregate in its
    # grade roots for those points), and the harvester held the whole revision on the first missing parent -- the two points' paper-facing
    # climb rows included. Rows withdrawn and the climb records' extra changed, so it bumps; a self-check now refuses a batch whose
    # degeneration / metric / judgment / state / contrast records name a run or judgment the batch does not emit
    # bon_stores 5 (2026-09-23, for the appendix best-of-N curve figure): the best-of-N CURVE rides every climb run
    # as seven `asr_at_n<N>` run_metric rows (N in 1, 2, 4, 8, 16, 32, 64 at 0.5 -- seedrep_cluster's and bon_native4096's spelling, the
    # paper's `asr` at `bon_n` N through paper.METRIC_LEGACY, `extra.bon_n` = N), read off the statistics file's `asr_at_n_4096_cf_t0.5`
    # grid: the producer's p, k = p x n where an integer, n the headline block's goals, no interval. Version 4 parked the grid on the
    # headline row's extra only, so the figure could draw no climb run of this reader (the Gemma / Nemotron standups, the paired-write
    # points). New rows over the same inputs, every other record byte-identical, so it bumps
    assert versions["bon_stores"] == "5"
    # cellstore 4 and arditi_vendor_mt 2 (2026-09-20, the vendor re-run's second pass): the opted-in controls roots count `n_items_judged`
    # over the SCORED benign subset (XSTest's 250 safe prompts, as every declared over-refusal reader files it; the live rows said 449 / 450)
    # and file the eval's own headline `over_refusal_graded` beside `xst_refused` — emitted rows moved, so cellstore bumps (the default
    # construction, the other roots, is unchanged and needs none). arditi_vendor_mt files the steered run on the sha its shards attest
    # instead of probing the live `object` table (version 1's probe latched the c = 0.15 refusal: its revision id hashed unchanged inputs,
    # so the object row landing later never re-read it), admits the decode replicates as trials of their own groups, and parks the bound
    # objects / twin copies in status_extra — new and changed rows over the same inputs, so it bumps
    # cellstore 5 (2026-09-21, the cap-2048 XSTest controls, at-cap 0 in all six stores): the vendor roots are every
    # `controls_cap<N>` rung under the store's two parents (`controls_cap_candidates`; 512 / 2,048 declared and named when absent) and opt into
    # `cap_provenance` — per store, the cap's 0-based rung on the instrument's declared ladder (`token_cap_rung_index` /
    # `token_cap_ladder_json`), the cap it was escalated from, the pool the judge driver FINISHED on and whether its ladder escalated
    # (`extra.judge_pool` / `judge_pool_escalated` / `judge_pool_concurrency`; the judgment's `api_pool` from the live calls when one pool),
    # and the grading harness's sha / branch / `harness_dirty_paths` as recorded (the judgment's `grading_code_sha`). The cap-512 rows
    # re-emit with these parks and keep every id and value; the default construction (the other roots) is unchanged and needs none.
    # arditi_vendor_mt stays at 2: the controls rungs are read here, not by that module
    # cellstore 6 (2026-09-24): the opted-in roots file the label mean beside the refused rate and the graded row, with its paired contrast
    # cellstore 7 / overrefusal_items 5 / sampled_controls 5 (2026-09-30): the shared StoreReader derives a fitted object's `transfer_kind` from the
    # manifest's `fit_provenance.model_id` OR the ingested object row's `fit_model_id`, and refuses a store with neither (or an unresolvable /
    # disagreeing fit model) by name instead of filing the `unverified` sentinel -- a default change of the reader, so its three constructors bump
    # together; every store carrying `fit_provenance.model_id` emits the same bytes, the two alphasteer_f22 stores file for the first time
    # cellstore 8 / overrefusal_items 6 / sampled_controls 6 (2026-10-01, commit 27): the shared StoreReader records the decode parameters a run
    # SENT from the store's recorder evidence (resolved_sampler.json, verdict AGREES) when the manifest's gen carries no sampler — the
    # thinking_card cells — so the derived decode-policy stamp can read `exact`; every constructor bumps
    # cellstore 9 / overrefusal_items 7 / sampled_controls 7 (2026-10-01, 27b): SENT from the client's request evidence only, RESOLVED from the
    # recorder, and the recorder a per-store revision input — every constructor bumps
    # cellstore 8 (2026-10-03, commit 6): every contrast of the shared store reader is decided by the pass's
    # anchor currency and carries `extra.anchor_selection`; the same directory's unsteered store (or the one the caller names) is the OWN anchor,
    # kept while current; a retired, unfolded or absent own anchor (until now: no contrast at all, in silence) searches the exact key -- a
    # same-source candidate whose store is on this mirror is re-read and paired per item, any other re-pick files the registry's unpaired rate
    # difference for the refused-rate headline and parks the label-mean and graded contrasts by name -- the own pairing's numbers parked on the
    # metric row's `extra.delta_block`; the rule-scored accuracy contrast re-picks only a same-source store on this mirror. Every contrast's and
    # every steered headline row's bytes move, so the reader and its three named constructors bump together; overrefusal_items and
    # sampled_controls construct the StoreReader alone and file their contrasts elsewhere, so they stand
    assert versions["cellstore"] == "10" and versions["overrefusal_items"] == "7" and versions["sampled_controls"] == "7"   # cellstore 10 = above both parents (9 = 27b, 8 = the anchor currency); 7 / 7 = the merged files (27b)
    # sampled_static 3 (2026-10-03, commit 7): every pooled contrast is decided by the pass's anchor currency and
    # carries `extra.anchor_selection`; the same tag's pooled `none` is the OWN anchor, kept while current; a re-picked anchor REFUSES by name (the pooled
    # contrast is the pin's paired bootstrap over the same tag's goals, k NULL on both sides); a tag without a `none` aggregate, or no current anchor at
    # the exact key, parks the comparison on the pooled steered row's `extra.delta_block` (until now: a silent skip) and names it in `row_refusals`.
    # Every pooled steered row's bytes move, so it bumps (one republish of its 513 groups, `reader_traversal`); ctl_neg1_static stays at 1 (no pooled
    # comparison on that reader: its standups serve the steered object alone and declare their anchor on the run)
    assert versions["sampled_static"] == "4" and versions["ctl_neg1_static"] == "1"      # 4: a cell no standup record names is parked by name, the group publishes (follow-ups-4 item 8)
    # sampled_controls 3: every pooled over-refusal run files `xst_refused` / `orb_refused` beside `over_refusal_graded` on the same
    # run, judgment and computation -- the full-refusal indicator pooled per item over the trials the way the graded score pools, the graded row's
    # n and abstentions, Wilson 95 %; a steered run with a same-tag `none` gets a second paired contrast on the new metric from its OWN generator,
    # so the graded contrasts' draws (the paper's numbers at the pin) do not move
    # sampled_controls 4: the label mean pooled per item like the graded row, a third paired contrast from a third generator;
    # 5 (2026-09-30): the shared reader's transfer_kind rule (above), a default change of the reader every constructor bumps for;
    # 6 (commit 27): the shared reader's recorder decode evidence (above), again every constructor bumps
    # 7 (27b): the sent/resolved split + the recorder as a revision input, every constructor bumps
    assert versions["sampled_controls"] == "7"
    # sampled_static STAYS at 2 after `derive.served_key_dose` learned the seeded dose-family spelling (`qwen35_4b_d1p25_loradiff64s1` -> 1.25,
    # 2026-09-24): the eleven seeded groups it changes had been REFUSED whole on the sidecar cross-check (the trailing grammar read them as
    # 1.0), so no accepted row's bytes move -- the previously refused groups file for the first time; every other tag spells what it did.
    # strength_mtdiff (8) and cellstore (5) also stay: the controls reader stamps the stem's dose over the shared grammar's and drops its park,
    # so its rows re-read byte-identical (dry run against the mirror, the landing record), and no cellstore / sampled_controls / static_padding
    # root carries the seeded spelling
    # arditi_vendor_mt 4 (2026-10-03, commit 5): every contrast is decided by the pass's anchor currency and
    # carries `extra.anchor_selection`; the aggregate's own `none` entry is the OWN anchor, kept when current and usable (the paired per-goal
    # record as before, the anchor judgment from the frame); a retired or unfolded own anchor, or an aggregate WITHOUT a `none` entry (until
    # now: no contrast at all, the missing entry's refusal line alone), searches the exact key and a re-picked unsteered run of the source
    # (another dose step's standup) files the registry's unpaired rate difference from the steered recount and the candidate's reading under
    # its own computation row (paired per goal when the candidate's aggregate is in the read), the own pairing's numbers parked on the steered
    # `asr` row's `extra.delta_block`; no current anchor parks by name and the old key is withdrawn. Every contrast's and every steered `asr`
    # row's bytes move, so it bumps
    assert versions["arditi_vendor_mt"] == "4"
    # petri_stats 2 (2026-10-03, commit 5; asserted for the first time -- the reader declared no version
    # before and read as the default "1", so its first explicit value must differ from that default to force a re-read): every contrast is
    # decided by the pass's anchor currency and carries `extra.anchor_selection`; the audited `none` this reader composes is the OWN anchor,
    # kept when current and usable (the sidecar's paired block as before, `anchor_judgment_id` from the frame instead of ''); a retired or
    # never-folded own anchor (the sidecar's anchor block naming a refused `none` row used to leave a composed id on the contrast that
    # dependency-blocked the whole revision) searches the exact key and a re-picked unsteered run files the registry's per-behavior paired
    # difference over both runs' item rows under its own computation row, the sidecar's block parked on the metric row's `extra.delta_block`;
    # no current anchor parks by name and the old key is withdrawn. Every contrast's and contrast-bearing metric row's bytes move, so it bumps
    # 3: the qwen35_4b_it x arditi_allpos cell resolves through the declared extra manifest
    # petri_stats 4 (2026-10-07): the reader's `run` records state the run's identity columns and mark every other carried
    # column `extra.unstated_columns` (a null carried as a withdrawal) -- the export owns the paper rows, and two full records restating one
    # row flipped its `lane_source_legacy` / clocks / `store_uri` / `serving_path` on every re-read of either adapter (749 shared run ids);
    # a cell with no producer clock is refused by name instead of stamped with the read time. Every run record's bytes move; items / metrics
    # / judgments / contrasts are unchanged.
    assert versions["petri_stats"] == "4"
    # seedrep_cluster 1: the seed-replicate mirror reader (2026-09-20), its first version
    # seedrep_cluster 3: a rule-scored MMLU run's judged count is the items scored and it files no state event (v2 left the live MMLU runs
    # `generated`); 2: the unfitted keys' runs file as decode-seed trials (trial_index = the seed index) (2026-09-20)
    # 4: the seed-replicate multi-turn stores (s2 / s3 judged, s1 held) and the substrate read per run from the store's routing table (2026-09-21)
    # 5: the seed-1 multi-turn runs file like s2 / s3 (2026-09-21); the v4 held dispositions go
    # 6: the substrate off a routing-table path files `derived`, not `recorded` -- a prefix is a derivation (2026-09-21)
    # 7: the graded best-of-N stores of s1 / s2 / s3 (`grades_bon/<tag>/`) on the seedrep_battery premint's derived instrument (pinned; refused by
    # name on any disagreeing fact) under the reader's own ids, the goal any-hit unit, the per-goal items; the seed-1 rows carry the premint's
    # queued id in extra.premint_run_id for the supersede ruling (2026-09-21)
    # 8: the single-object tags (`grades_bon/<stem>_seedrep_s<N>/`: the Arditi all-positions refusal steer's best-of-N at the three seed bases) file as decode-seed
    # trials of ONE object on the seed view (DECODE_TRIAL_RULE), with the cross-store contrast against the seed view's `none`, the battery's plan id in
    # extra.premint_run_id and a reader-minted instrument when the store disagrees with the premint's; the seed-selection tags re-emit unchanged bytes (2026-09-21)
    # 9: American spelling of the prose the reader writes on its records (2026-09-23)
    # 10: a single-object tag's graded config sha passes through a certified scrub-map row when the release rewrote the wave config by declared rules
    # only; the run names sha_check verified_via_map and the graded sha (2026-10-19)
    assert versions["seedrep_cluster"] == "10"
    # gsm8k_seedrep 4 (2026-09-21): the Arditi all-positions object -- its three seed rows (`<tag>/arditi_allpos/`) file as DECODE-SEED TRIALS of ONE
    # object on the seed views (seedrep_cluster's DECODE_TRIAL_RULE: trial_index = seed_base = the seed index, fit_seed unset, the operating point's
    # strength columns, resolved through the all-positions roster / pins and bound to the producer's eligibility record), and the matrix's own
    # default-seed row under `arditi_allpos_paper_objects/` files as a single-object NON-seed target on a reader-minted instrument; the seed views'
    # rows re-emit byte-identical to 3
    # 3: the paired views s<N>p (their own `none` anchor, the composed difference_relu object, tag-prefixed producer keys), a
    # run_contrast per steered run against its own view's `none`, the per-pair final-state record as a cross-check record; 2: the third state
    # widened to the at-cap items the ladder never escalated (the bound's interval and the unresolved count move on four pairs; the cross-check
    # binds to the producer's final-state keys); 1: the three-seed family's GSM8K stores (2026-09-21)
    assert versions["gsm8k_seedrep"] == "4"
    # bigcodebench_seedrep 1 and bfcl_seedrep 1 (2026-09-23): the seed-replicate BigCodeBench-200 / BFCL-200 stores from the Modal volume (tree
    # 8365992) -- two readers over one shared reading (`_bcbw_common`): one store holds both tasks and each reader owns its own copy; one group per
    # SEED (the paired tag's composed row pairs across standups with s<N>/none); a row files only under the gate record's PASS attesting lora_native;
    # a launch-set row with no store files nothing; a seed tag whose `none` has not run files its rows without a contrast and names the reason on
    # extra.anchor; decode_seed_rule per_item / seed_base NULL / trial_index 0 on every row; their first versions.
    # 2 (2026-09-23): every row files decode_seed_rule `none` -- the greedy protocol's rule, the instrument's and the
    # registered policy row's (per_item was Inspect's incidental derivation, kept on extra.decode_seed_rule_text / seed_axis), so the
    # harvester's derived decode_policy_match reads `exact` instead of `no_match`; the paired tag's own `none` (class anchor) is that
    # tag's anchor when it has filed, the cross-standup s<N>/none the fallback (one shared `anchor_plan`); values unchanged. Folded into the same
    # unpublished 2 (the mirror of 2026-09-23 14:19Z): the launch set's ROW TABLE is launch + carried (`_bcbw_common.ROW_TABLE_RULE`; the census moved
    # the 25 pass-1 rows to `carried`, which a reader of `launch` alone refused), the census's anchor declaration carried on extra.launch_set
    # (2026-09-26, no bump: the composed row's fallback contrast -- ACROSS tags against s<N>/none, extra.cross_standup true -- files anchor_kind sibling_none,
    # another standup of the same model (vocab.ANCHOR_KINDS; a `none` across seeds / standups is a fair anchor under the same decode
    # settings), instead of same_standup_none, through ONE rule for the six capability readers (`_bcbw_common.anchor_kind_for`; until then every contrast filed
    # same_standup_none and told the two apart on extra.cross_standup alone). The live mirrors re-read BYTE FOR BYTE: every composed row's own `none` has filed,
    # so no current contrast crosses standups; the 3 cross-standup rows per reader in the registry are the fallback rows of the revision BEFORE the paired
    # `none` landed, sitting beside the own-none contrast that replaced them (run_contrast's key carries anchor_run_id) -- a stale-row question for the fold,
    # not a reader emission; the fixture pin in tests/test_registry_adapter_bcbw_emission_pin.py is unchanged)
    # 3 (2026-10-03, commit 7): every contrast of the six capability readers is decided by the pass's anchor
    # currency (`_bcbw_common.CURRENCY_RULE`) and carries `extra.anchor_selection` -- the row's own pairing stands while current, a retired / unfolded /
    # absent own anchor searches the exact key (a same-source `none` of another standup re-anchors the row: paired per item when it is a row of the batch,
    # else the registry's unpaired rate difference under its own computation row), no current anchor parks on the VALUE row's `extra.delta_block` and in
    # `row_refusals`; the model qwen35_35b_a3b admitted. Every reader of the family bumps by one together (the shared `_emit_contrasts` moved).
    # (Commit 7's own `u0` standup grammar and its unrecognized-row-dir rule were superseded at the merge of main 6eaa2c210, 2026-10-05, by main's
    # unsteered row `<model key>_u0` read in the s1 group, the window-matched baseline and the undeclared-row-dir refusal -- UNSTEERED_ROW_RULE /
    # WINDOW_ANCHOR_RULE / UNDECLARED_ROW_DIR_RULE; the currency decision stands on main's plans.)
    # (Later the same day, no further bump -- the 3 above is still unlanded: the currency's same-standup tier (the row's own tag's `none`
    # current from an earlier read, absent from or refused in this batch) files same_standup_none WITHIN the standup, `reanchor_tier` on every re-anchored
    # contrast; the `$.row_refusals` state row is written on every read, the empty list included, so a cleared park is seen to clear; the files of an
    # unrecognized row dir were inputs of every group (that rule is main's undeclared-row-dir refusal since the merge above). The fixture pin in
    # tests/test_registry_adapter_bcbw_emission_pin.py moves on the state rows alone: the empty refusal row)
    # (Later again the same day, still no bump: the `$.row_refusals` state row is keyed on the GROUP -- `registry://row-refusals/<source_id>`,
    # its sha the inputs' fingerprint (`lineage_common.ROW_REFUSALS_LEDGER_RULE`) -- never on the group's first input, and a re-activation replays the
    # committed revision's own state rows instead of asking the adapter, whose ledger is the last read's. The fixture pin moves on the ledger row's key
    # and sha alone; the records stand. sampled_static (3, unlanded) takes the same key)
    assert versions["bigcodebench_seedrep"] == "3" and versions["bfcl_seedrep"] == "3"
    # bigcodebench_allmodels 1 and bfcl_allmodels 1 (2026-09-23): the appendix families' BigCodeBench-200 / BFCL-200 stores (gemma4e4b / llama8b /
    # nemotron3nano4b: one seed tag each and its paired tag, native serving, the protocol decode) through the SAME shared reading bound to the appendix model set
    # (`_bcbw_common.MODEL_KEYS`, `TaskSpec.models`): a fitted key's staged id is the launch set's pin key pinned in the harness pins (the roster stages no s1 / s1p
    # selection for the families), the `main` canary row's stores file under the s1 tag with their objects the roster's main selection's (extra.tag_legacy_original),
    # llama8b runs no BFCL row; their first
    # versions. The seedrep readers stay at 2: the model-key table re-reads the live qwen mirrors byte for byte (the dry run of 2026-09-23 -- every run / metric /
    # item / contrast record equal; the new extra blocks are stamped on appendix rows alone)
    # (2026-09-24 03:39Z, no bump: a COMPOSED s1p key the harness pins do not carry resolves through the store's tree pins by the launched object_sha256
    # (`_bcbw_common.PINS_KEY_RULE`; extra.launched_pins_key on THOSE rows alone) -- the 3 / 2 rows that refused on the live mirrors file; every row that filed
    # before re-emits byte for byte (the fixture pin in tests/test_registry_adapter_bcbw_emission_pin.py))
    # 2 (2026-10-03, commit 7): the anchor currency on every contrast and the pipeline test's qwen35_35b_a3b admitted with its u0 standup (the bcbw seedrep note)
    assert versions["bigcodebench_allmodels"] == "2" and versions["bfcl_allmodels"] == "2"
    # bigcodebench_full 1 and bfcl_full 1 (2026-09-24): the FULL-suite BigCodeBench (all 1,140 tasks) / BFCL (all 3,981 items) stores from the Modal volume
    # through the same shared reading (`_bcbw_common`) with the full item-set attestation (`ITEM_SET_RULE`; the producer's contract 2026-09-24 00:22Z / 00:32Z):
    # the item count from the mirror's records/item_sets.json + the pin file it names (by file sha256), the manifest's tinycap.<key>_full_item_set = {seed null,
    # n == of == that count, ids_sha256 = contracts.ids_sha256 of the pin's ids} checked against it and against item_sets.yaml's declaration, the label `full`
    # on every aggregate, the launch-set row's item_set {id, sha256, n}, the outcome record's count -- on NEW instruments (bigcodebench_full_protocol_greedy_4096 /
    # bfcl_full_protocol_T0001_1024, item sets bigcodebench_full_1140 / bfcl_full_3981) with run.item_set_id the set's; the addendum of 02:31Z (a second producer
    # tree, pins by hash within the set, the launch set's trees and a row's sha, relaunched rows with a parked first store, relaunches.json); their first versions.
    # The four 200-item readers stay where they are: the count and the item-set knobs are TaskSpec fields at their 200-item defaults, their emission over the
    # synthetic fixtures is pinned equal to 429ed730c (tests/test_registry_adapter_bcbw_emission_pin.py), and the live seedrep mirrors re-read byte for byte. The
    # roster-set parse reading the producer's LIST-shaped rosters.json (beside the mapping the fixtures write) makes the live appendix mirrors FILE (27 / 18 runs)
    # where 429ed730c refused every store with `rosters.json names no roster` -- a read of what was always there, the same facts checked, no record of theirs re-spelled
    # bigcodebench_full 2 and bfcl_full 2 (2026-09-25): the INSTRUCT-DIRECT tag admitted (`_bcbw_common.INSTRUCT_DIRECT_RULE` / `INSTRUCT_ANCHOR_RULE`; the producer's
    # declaration 19:13Z): `<model key>_it` rows (gemma4e4b / llama8b / qwen35_4b; row dir `<model key>-f22it`, launch-set selection `it`, block and transfer_kind
    # instruct_direct, seed_base null, class arm) file under view_legacy `<model key>_it` on the family's s1 group, their object through the harness roster's `it`
    # selection, transfer_kind DERIVED from fit_provenance.model_id vs the served model, their contrast against `<model key>_s1/none` by alias (the launch set's
    # `it/none` anchor row has no store; no `_it` none run files). Every seed-tag record re-emits byte for byte (the dry run over the live mirrors: 63 + 53 rows);
    # the four 200-item readers stay where they are (`TaskSpec.instruct_direct` false; the fixture pin in tests/test_registry_adapter_bcbw_emission_pin.py unchanged)
    # bigcodebench_full 3 and bfcl_full 3 (2026-09-26): the instruct-direct rows' alias contrast against `<model key>_s1/none` (extra.cross_standup true) files
    # anchor_kind sibling_none -- the same model's `none` served by another standup (vocab.ANCHOR_KINDS) -- instead of same_standup_none, through the shared
    # `_bcbw_common.anchor_kind_for`; the 9 / 6 live cross-standup rows move on that field alone (the stamped INSTRUCT_ANCHOR_RULE text keeps its 2026-09-25
    # spelling byte for byte, by design), every seed-tag and own-none record re-emits byte for byte (the dry run over the live mirrors); the four 200-item readers
    # stay where they are (allmodels: no contrast of theirs crosses standups, 16 / 24 own-none contrasts live; seedrep: byte for byte, the note above)
    # bigcodebench_full 4 and bfcl_full 4 (2026-10-03, commit 7): the anchor currency on every contrast (the bcbw seedrep note)
    assert versions["bigcodebench_full"] == "4" and versions["bfcl_full"] == "4"
    # bigcodebench_full_shared 1 and bfcl_full_shared 1 (2026-09-28): the SHARED-STANDUP comparison stores (tinycap-shared-standup) through the solo
    # full readers PARAMETRIZED (`_bcbw_common.SharedSpec` on `TaskSpec`; the solo readers' `shared` is None and their live emission is byte for byte the base
    # tree's -- the dry run over the live mirrors, 72 + 59 runs): the `_shared` spelling and run ids (the store key is byte-identical to the solo's), the repeat row
    # keyed by its segment, the row engine's standup identity (one standup id per row launch; max_loras / residual_slot_armed RECORDED), the comparison fields from
    # the launch row and the gate's comparison block, the borrowed run_contrast against the solo run read from the solo mirror beside the in-run contrast against the
    # row's shared none (sibling_none for a relaunch on its own standup), the instruct row's own none as a comparison row; their first versions
    # 2 (2026-09-29): the store-attestation decision admits landing 1's real stores, every one of which version 1 refused; the bump because the live
    # database holds ACCEPTED zero-run version-1 revisions of both sources with v_source_status.ok = 1, and an unchanged group is re-read only on an input or
    # emission change -- under version 1 the corrected reader would leave them current
    # 3 (2026-10-03, commit 7): the anchor currency on every in-run contrast and the solo run's currency checked before the borrowed comparison files (the bcbw seedrep note)
    assert versions["bigcodebench_full_shared"] == "4" and versions["bfcl_full_shared"] == "4"      # 4: the stand-in contrast files anchor_kind solo_stand_in
    # bigcodebench_full_shared_l2 1 and bfcl_full_shared_l2 1 (2026-09-30): LANDING 2 of the shared-standup comparison stores -- the same tuples driven
    # again by another harness (K = 2 drivers per row + a CPU verification pass), published as SEPARATE sources so no run id collides with landing 1's 66 ingested rows --
    # through landing 1's readers parametrized once more (`_bcbw_common.SharedSpec.harness_variant` / `base_source`): the producer key `<tag>_shared_l2/<store key>`
    # (SHARED_L2_RUN_ID_RULE; the spelling stays `<tag>_shared`), the verified store's standup id read from the manifest and cross-checked against the row runner's
    # earlier RESULT line when the last line is the verifier's re-stamp without standup= (SHARED_VERIFIED_STANDUP_RULE; landing 1's stores keep the last-line rule and
    # re-emit byte for byte), the bound launch set's base-source own_source recorded as provenance; their first versions. Landing 1's readers stay at 2.
    # 2 (2026-10-03, commit 7): the anchor currency on every contrast (the bcbw seedrep note)
    assert versions["bigcodebench_full_shared_l2"] == "3" and versions["bfcl_full_shared_l2"] == "3"      # 3: solo_stand_in (follow-ups-4 item 3)
    # bigcodebench_full_shared_l3 1 and bfcl_full_shared_l3 1 (2026-10-01): LANDING 3 of the shared-standup comparison stores (the fan-out test on the shared-standup
    # harness) -- the gemma row's four tuples driven once more, published as SEPARATE sources so no run id collides with landing 1's or landing 2's -- through landing 1's
    # readers parametrized as landing 2's are (`harness_variant` l3; `landing_rule` = SHARED_L3_RULE stamped on the rows): the fourteen-key verified block recorded and never
    # pinned, the token-present re-stamp read off the last line, the non-null legacy route scalar recorded and never read as identity; their first versions. Landing 1's
    # readers stay at 2 and landing 2's at 1 (the one reader change, the gated `landing_rule` stamp, leaves their records byte for byte).
    # 2 (2026-10-03, commit 7): the anchor currency on every contrast (the bcbw seedrep note)
    assert versions["bigcodebench_full_shared_l3"] == "3" and versions["bfcl_full_shared_l3"] == "3"      # 3: solo_stand_in (follow-ups-4 item 3)
    # ifeval_seedrep 1 (2026-09-23): the seed-replicate IFEval stores from the Modal volume (tree 8365992) read at the COMPLETION CAP -- one run per
    # (model, tag, object key) at the cap its 2,048 -> 4,096 -> 8,192 ladder reached, the value the reader's draw-wise merge (each cap step's re-drawn draws
    # replace their truncated shallower draws, a loop never re-drawn, recounted over the 1,082 draws), the per-store readings checked against the
    # producer's 65-row record and the merged reading against the ladder plan's decisions, a row filed only under a done dispatcher record attesting
    # lora_native (a rate-limited attestation read judged by the store); decode_seed_rule per_item / seed_base NULL / trial_index 0; its first version
    # 2 (2026-09-23, the mirror of 15:24Z): the launch set's ROW TABLE is launch + carried (`_bcbw_common.ROW_TABLE_RULE`; the census moved the 30 pass-1
    # rows to `carried`); a paired tag's `none` reads its dispatcher record from records/dispatch_paired_none/ (`DISPATCH_ROOTS_RULE`, the pass-1 root
    # winning when both name it) and, dispatched outside the ladder matrix, files under `PLAN_ABSENT_RULE` (base-only, its at-cap share at or below the
    # plan records' declared trigger, its at-cap draws `unclassified`, rep_ratio_loop / truncation_only not_measured, exit `not_planned`); every row's
    # extra.dispatch[<cap>] names the root it was read from; the paired tag's own `none` is that tag's anchor (the same day's rule, `anchor_plan`);
    # the parked *.incomplete-* roots join the *.failed-* roots as history never read. The seed rows' values are unchanged (checked against the live rows)
    # (2026-09-26, no bump: the composed row's fallback contrast files anchor_kind sibling_none instead of same_standup_none (`_bcbw_common.anchor_kind_for`, the
    # rule shared with the BigCodeBench / BFCL / MATH-500 readers); the live mirror re-reads byte for byte -- every composed row's own `none` has filed; its 3
    # cross-standup rows in the registry are the earlier revision's fallback rows, see the bcbw seedrep note)
    # ifeval_seedrep 3 (2026-10-04, commit 8): every contrast is decided by the pass's anchor currency and carries
    # extra.anchor_selection; a comparison with no current anchor parks on the VALUE row's extra.delta_block (the bcbw readers' emission 3 rule)
    assert versions["ifeval_seedrep"] == "3"
    # gsm8k_paired_allmodels 1 (2026-09-22): the paired_allmodels GSM8K document -- the GSM8K reader's SECOND source (a subclass of gsm8k_seedrep over the
    # run-state mirror gsm8k_paired_allmodels/): the paired-write difference ReLU on gemma4e4b_it / llama8b_it / nemotron3nano4b and nemotron's constitution
    # ReLU under the default Inspect seed rule, per-model thinking / sampling / decode policy, the object attested by the store and bound to the registry's
    # object row, no `none` (the in-frame unsteered anchors parked on every metric as pooled_in_frame_anchors); its first version. The seed-replicate
    # reader stays at 4: its helpers gained keyword parameters whose defaults are its own values, so its groups re-emit byte-identical
    # 2 (2026-09-22): the producer's records at the mirror root (store pins, the off-task three-number record, the final-state record) bound when present,
    # and a leg the driver refused (nemotron3nano4b x the constitution ReLU: THINKING_UNSERVABLE predicate A) filed as a REFUSAL STATE -- a `not_served`
    # disposition spelling `thinking_unservable_predicate_a` -- instead of a bare pair refusal; the live mirror carries the records, so every row's extra moves
    assert versions["gsm8k_paired_allmodels"] == "2"
    # math500_seedrep 1 (2026-09-23): the seed-replicate MATH-500 stores (cluster B, tree 4c8268da) -- the MATH-500 sibling of gsm8k_seedrep as a
    # STANDALONE reader (the GSM8K reader's helpers imported, its module untouched): one pass at 16,384 (no cap step, final == base), one group per SEED
    # (the paired tag's composed row reads in its seed tag's batch and pairs across standups with s<N>/none), the launch set's pin key / expected
    # sha as the identity declaration, the Arditi rows as decode-seed trials of one object declared unpaired (arditi_unpaired_value); its first version.
    # Still 1 after 2026-09-23's paired-none anchor rule and the launch + carried ROW TABLE: both change what the reader parks only for inputs that do
    # not exist yet in its mirror (a paired `none`, a carried row); the current mirror re-reads byte for byte (the dry run's content digests)
    # (2026-09-26, no bump: the composed row's fallback contrast files anchor_kind sibling_none instead of same_standup_none (`_bcbw_common.anchor_kind_for`, the
    # rule shared with the BigCodeBench / BFCL / IFEval readers); the live mirror re-reads byte for byte -- every composed row's own `none` has filed; its 3
    # cross-standup rows in the registry are the earlier revision's fallback rows, see the bcbw seedrep note)
    # math500_seedrep 2 (2026-10-04, commit 8): every contrast is decided by the pass's anchor currency and carries
    # extra.anchor_selection; a park lands on the VALUE row's extra.delta_block; every steered run carries extra.anchor; the run and value records are
    # added once the contrasts are decided
    assert versions["math500_seedrep"] == "2"
    # orxs_seedrep 1 (2026-09-21): the seed-replicate over-refusal controls through the shared served-controls reader under the source's own
    # registered name (cellstore.OrxsSeedrepAdapter) — its first version. cellstore itself stays at 5: every addition the new root uses is an
    # opt-in StoreRoot field (file_unjudged, object_resolution, ladder_parent, strip_trial_suffix, substrate_from_routing_table) and the
    # generic adapter's roots do not name any of them, so their bytes do not move.
    # orxs_seedrep 2 (2026-09-21, the completion-cap rule, cellstore.COMPLETION_CAP_RULE): the final-cap row of every (seed, store
    # id, control) is the scored merge (extra.completion_cap_merged / final_rung / n_at_cap_by_rung / the residual as the at-cap share,
    # cross-checked against the producer's results_final_merged.json + results_by_rung.json, now declared inputs of every group), every lower-cap row
    # of a pair that escalated is provenance only (no judgment / metric / item / state row however judged its root), and contrasts pair the
    # steered pair's final-cap row against `none`'s final-cap row whatever the caps (extra.contrast_caps). Rows withdrawn, rows moved: it bumps
    # orxs_seedrep 3 (2026-09-21, the NON-seed target under the same mirror: the `none` + `arditi_allpos` OR-Bench controls at the Arditi
    # all-positions point c = 0.3, `qwen35_4b_ar_c0p3_t0`): a target directory is read by its GRAMMAR (StoreRoot.object_resolution target_grammar,
    # cellstore.orxs_target_object) -- a seed tag as before, an Arditi point naming the Arditi ladder's object of its rung with the rung's strength
    # columns; a pair whose exit the ladder has not declared files `generated` rows with extra.final_cap_undeclared instead of refusing; a
    # non-seed target with no row in the producer's result files merges on the ladder record alone (`derived`). The seed views' rows are byte-identical
    # to 2 (the seed-1 tree, judged at final caps only, was read by 2 unchanged); the version moves because a target 2 refused now files
    # orxs_seedrep 4 (2026-09-24): its opted-in stores gain the label-mean row and contrast
    # 5 (2026-10-03, commit 6; the shared reader's 8): every merged contrast is decided by the pass's anchor
    # currency -- the same target directory's `none` final-cap row is the OWN anchor, kept whatever the two caps while current; a retired or
    # unfolded one re-picks a current unsteered run of the exact key (another seed tag's store on this mirror re-read and paired per item; another
    # source's run as the unpaired refused-rate headline); the merge's own-anchor refusals stand (this reader borrows nothing on its own authority)
    assert versions["orxs_seedrep"] == "5"
    # mmlu_full 1 (2026-09-21): the Qwen MMLU read at the FULL 1,531-item width through the shared served-controls reader under the source's
    # own registered name (cellstore.MmluFullAdapter) -- its first version: one run per store on the NEW instrument mmlu_letter_prefill_full_1531
    # (item set mmlu_1531), chosen per store by the attestation (resolved_config.mmlu.item_set present == the full spec; the manifest's ids hash to
    # the item set's items_sha256), 1,531 acc items, the accuracy contrast against the same tag's `none`. cellstore itself STAYS at 5: every
    # addition the new roots use is an opt-in StoreRoot field (source_controls, require_mmlu_full, accuracy_contrast, producer_tree_sha, the
    # mmlu_full_grammar resolver) or keyed on an attestation no existing store carries (`resolved_config.mmlu.item_set` is present IFF the
    # full pool ran), so the generic adapter's roots re-read byte-identical (proved by a dry run of the Arditi vendor / Arditi OR-Bench roots
    # from origin/main and from this tree into two scratch roots: identical record files)
    # mmlu_full 2 (2026-10-03, commit 6; the shared reader's 8): the accuracy contrast is decided by the pass's
    # anchor currency (judge None: the rule-scored readings); the same tag's `none` is the OWN anchor, a re-pick pairs per item only when the
    # candidate's store is on this mirror (the rule-scored row carries a value and n, never a count, and a derived count is not a count), no current
    # anchor parks by name on the steered `acc` row
    assert versions["mmlu_full"] == "2"
    # ctl_neg1_controls 2 (2026-10-03, commit 6; the shared reader's 8): the battery holds no unsteered store, so
    # every steered over-refusal comparison is decided on the exact key alone -- the paper's roster-matched seed-standup `none` of another source
    # (the declared pairing) files the registry's unpaired refused-rate headline (`borrowed`), the item-paired contrasts park by name, and no
    # current anchor parks every comparison by name on the steered rows (until now: no contrast and no record of why)
    assert versions["ctl_neg1_controls"] == "2"
    # seedrep_statics 1 (2026-09-21): the legacy-store reader of the seed-1 static stores, folded onto the premint's queued runs; its first version
    # 2 (2026-09-21): the misalignment input -- the judged seed-1 agentic store (<RS>/misalign/staged/) minted as new runs (the battery queued none), one
    # judgment per run, per-response items, the at-cap share, contrasts against the same-standup `none`; the statics / best-of-N groups re-emit unchanged bytes
    # 3 (2026-09-21): the ladder re-draw -- the consumer re-drew the renellm s1 base store's 12 at-cap items at 8,192: one NEW run per (object key,
    # cap) with replacements under the reader's own stage-axis producer key (never the premint's id, never a patch of the base row), the final state
    # composed item-wise (replace, never add), one item_flags row per item of the final state, one pending disposition per level run naming the unjudged
    # items and the declared `judged.jsonl` contract; every base row gains extra.final_state / final_cap / final_run_id and re-emits otherwise unchanged
    assert versions["seedrep_statics"] == "4"
    # seedrep_controls 1 (2026-09-21): the legacy-store reader of the seed-1 served-controls store (XSTest / OR-Bench-Hard generated, MMLU
    # rule-scored), folded onto the premint's queued runs under the certifying tree's ids; its first version
    assert versions["seedrep_controls"] == "2"       # 2: the label-mean row beside the two it already files, with its contrast
    # seedrep_petri 3 (2026-09-21): the pooled `petri_pooled` run per (seed, object), derived from the six behaviour runs' own files (its instrument
    # minted per seed -- from the premint's six rows for s1, the seeded pooled row for s2 / s3 -- every item filed again under it, the paired
    # contrast against the seed's `none` pooled run, the judge-chain markers rolled up onto its judgment, a seventh pending disposition), and
    # the per-row judge provenance carried onto every item; 2: the seed-2 / seed-3 stores (the minted binding); 1: the seed-1
    # store on its premint
    assert versions["seedrep_petri"] == "3"
    # objects_bank 4 (2026-09-25): a dose ladder's 1.0 ANCHOR of a LoRA subtraction (lora_subtract.py's derived_meta.json + dose_scaled_from.anchor,
    # parent sha == own sha) reads as the subtraction -- derivation lora_subtraction, parents = the two fits, fit_effective_batch through both
    # parents' fit records, the stamp kept as evidence -- where it read as a dose_scale copy whose only parent was itself (four objects);
    # 3: the registered instance reads the checkout's seeds (its factory passed none, so every fitted object read `other`), spells the label
    # from (object_kind, campaign stem) and parks the stem / fit-variant prefix
    # objects_bank 5 (2026-10-03): `status` is filed only where the bank STATES one (a directory under _withdrawn/ or _superseded_by_it64/,
    # a superseded_by.json naming this object's fit fingerprint); a bundle with no such evidence restates the registry's folded status for the id
    # (the default `approved` only for an object with no row, refused when there is no connection to check), `extra_json.status_origin` names
    # which, and the status entry of field_confidence_json says so -- where 4 filed `approved` over the producers' `reference` and demoted three
    # paper reference objects. 300 objects re-emit on the next pass; every restated status is the one already folded, so no live status moves.
    # objects_bank 6 (2026-10-04, the rule over 5's restatement): the same bundle OMITS `status` (no column, no
    # `unstated_columns` marker, no `status_origin`; `extra_json.status_unstated` names why the bank states none); the fold keeps the producer's
    # folded status on omission and the harvester blocks a brand-new object on its own row until its producer states one. The 301 bank-carried
    # objects re-emit once more; an omitted column is no statement, so no live status moves.
    # objects_bank 7 (2026-10-05): (a) every producer-owned column follows 6's omission rule -- `fit_code_sha` is omitted whenever its
    # value would be the bank default (`unverified`) and `derivation_params_json` unless it is a sidecar's own `params` block verbatim (the
    # assembled dicts were abbreviations of the producer's record: eight of a dose stamp's twenty-one keys against a producer's thirteen of its
    # own), `extra_json.producer_owned_unstated` naming each; (b) a cluster location's `bytes` is the OBJECT's size there, not the staging
    # record's directory total (now `staged_dir_bytes`; 9 of 9 live records carried the total beside object.pt's sha). Every bank row re-emits;
    # no folded producer value moves and `artifact_bytes` was right all along.
    # objects_bank 8 (2026-10-05, 7's rule extended to the two remaining producer-owned stamps): a dose COPY's `fit_writer` and
    # `fit_finished_utc` are the copy's own dose_scaled_from stamp's `producer` / `written_at` (derived_meta.json's stamp first, then
    # train_budget_meta.json's, then the all-positions derivation's built_by / built_at) and are OMITTED when the stamp records neither --
    # where 7 read the parent's builder / built_at / writer / COMPLETE time copied beside the copy (a subtraction's dose children filed
    # `scripts/sdfbase/lora_subtract.py` and the parent's build time). Every bank row re-emits; the fold keeps a producer's native statement
    # on omission, so no live producer value moves.
    # objects_bank 9 (2026-10-07): a bundle of a kind declared OUT OF SCOPE by name (`legacy_spellings.yaml:
    # out_of_scope_object_kinds` -- the 40 synthetic Gaussian control directions under `_controls/`, the lofit / alphakl_soft / token_appended /
    # bidpo fits) is PARKED instead of refused: an accepted revision with no object record and a `source_status` fact (`extra.refuse` +
    # `extra.parked`) citing the table, so the 48 standing refusals end and the adapter stops reading REFUSED in every window and tick
    # (a per-group refusal reads the whole adapter REFUSED and aborts the window). Every bank row re-emits; no object record changes.
    # objects_bank 10 (2026-10-13, the rule for the 92 dose copies `reconcile fit-effective-batch` listed as ambiguous): a dose
    # COPY's `fit_effective_batch` is its PARENT's effective batch, stated `derived` from ONE source, the parent's fit record copied beside
    # the copy -- the copy's own `train_budget_meta.json` (`fit_sidecars._dose_batch_from_parent`: a file of
    # the copy's own group, so no other group's directory is read) -- NULL by name when that record names no integer or the copy carries
    # none, where 9 left the column unstated ("nothing was fit") beside the parent's hyperparameter block. The bank's own re-emission is
    # the correction (no owner record is written for a dose copy); every bank row re-emits and only a dose copy's `fit_effective_batch`
    # moves, from NULL to its parent's 8 or 64. Agreement with the parent's own registry row is reconcile's read-only cross-check.
    assert versions["objects_bank"] == "10"
    # the one reader whose raw-frame dose is not the nominal: its own half moved when its `paper_dose` park became the raw c;
    # 3: its substrate claims spelled by kind (the realpath rule derived, the manifest's served_cluster recorded), never a bare value
    # 5: the refused-rate sibling on every graded XSTest row
    # 6: the label-mean sibling on every graded XSTest row
    assert versions["arditi_allpos"] == DC.emission_version(6) == f"6+emitter{DC.EMITTER_VERSION}"
    # overrefusal_items 5 (2026-09-30): the shared store reader's transfer_kind rule (the manifest's fit model OR the object row's, refused by
    # name otherwise); 4: the fitted objects' dose / operating point through the shared store reader; 3: runs keyed `<target dir>/<store key>`
    # — the unsteered store's bare key recurred across a model's target dirs (b2)
    # 6 (commit 27): the shared reader's recorder decode evidence, every constructor bumps
    # 7 (27b): the sent/resolved split + the recorder as a revision input, every constructor bumps
    assert versions["overrefusal_items"] == "7"
    from sisteer.registry.adapters import cap4096_declarations as C4
    assert versions[C4.NAME] == "1"
    # static_padding 1 (2026-09-23): the PADDED static conditions on qwen35_4b, filed on reader-minted padded instruments (migration 0012: the pad
    # and the filler-contract hash are instrument identity), the store recounted from the byte-for-byte mirror, the producer degeneration sidecar read
    # under the judgment-less key; its first version. 2 (2026-09-24): every run's extra gains the SERVING-JOB SET its shards name (a store sharded
    # across several Slurm jobs is one store served by several attested-identical instances and files where it had refused), the standup grammar
    # gains the seed-replicate suffix (`_obj_s<N>`; `training_seed` on the steered rows' extra) and a steered standup with no reference on disk refuses
    # instead of filing on a self-minted instrument; inputs unchanged, so the bump alone republishes the 66 ingested runs (ids byte-identical)
    assert versions["static_padding"] == "2"


class _LabelledAdapterV1(SyntheticStoreAdapter):
    """The declared readers' two-part label (`declared_common.emission_version`: the reader's own version beside the shared
    emitter's) at the value every live `1+emitter2` head was minted under."""
    EMISSION_VERSION = "1+emitter2"


class _LabelledAdapterV2(SyntheticStoreAdapter):
    """The same reader, its own half bumped (relu_paired after #259)."""
    EMISSION_VERSION = "2+emitter2"


class _OtherReader(SyntheticStoreAdapter):
    """A second adapter over the SAME store files: its watermarks name the same input URIs under its own name."""
    name = "other_reader"


def _revision_rows(root, adapter_name):
    return _q(root, "SELECT source_id, revision_id, accepted, superseded_by_revision_id, json_extract(extra_json, '$.emission_version') "
                    "FROM source_revision WHERE adapter = ? ORDER BY observed_utc, source_id", adapter_name)


def _group_inputs(adapter, group):
    return [i for i in adapter.discover() if adapter.comparison_group(i) == group]


def test_bumped_version_and_new_groups_publish_on_the_incremental_pass_whatever_another_adapter_watermarked(tmp_path):
    """The live relu_paired shape (2026-09-19): one head minted under `1+emitter2`, the reader bumped to `2+emitter2`, and two
    comparison groups discovered for the first time (no accepted head) — while a second adapter had watermarked the same store
    files. The incremental pass publishes THREE revision directories (no bundle): the bumped group by the version rule, the new
    groups because nothing of theirs is processed, and another adapter's watermarks are not this adapter's (`_watermarks` is keyed
    by adapter). A `--reread` of the same code then reproduces the three ids and publishes no fourth; the plain pass writes nothing."""
    root, store_root = _setup(tmp_path)
    rep0, ing0 = _pass_and_ingest(root, _LabelledAdapterV1(store_root), synth.t(2000))
    rev0 = rep0.results[0].revision_id
    assert rep0.groups_refreshed == 1 and ing0.refused == 0 and ing0.blocked == 0
    assert _revision_rows(root, "synthetic_store") == [("synthetic_store:s1", rev0, 1, None, "1+emitter2")]
    _store(store_root, "s2", "legacy_store_2", "constitution_lora@aaaa", [0.6, 0.7])
    _store(store_root, "s3", "legacy_store_3", "constitution_lora@aaaa", [0.1, 0.9, 0.5])
    other = _OtherReader(store_root)
    rep_o, ing_o = _pass_and_ingest(root, other, synth.t(2500))
    assert rep_o.groups_refreshed == 3 and ing_o.refused == 0 and ing_o.blocked == 0
    uris = sorted(i.uri for i in other.discover())
    assert _q(root, "SELECT DISTINCT adapter FROM source_watermark WHERE input_uri IN (?, ?, ?) ORDER BY 1", *uris) == [("other_reader",), ("synthetic_store",)]
    assert _q(root, "SELECT COUNT(*) FROM source_watermark WHERE adapter = 'synthetic_store'") == [(1,)]     # s1 only: s2, s3 are unprocessed to it
    bumped = _LabelledAdapterV2(store_root)
    rep, ing = _pass_and_ingest(root, bumped, synth.t(3000))
    assert (rep.inputs_seen, rep.inputs_changed) == (3, 2)              # s2, s3 are new inputs; s1 is changed by the version rule alone
    assert rep.groups_refreshed == 3 and rep.groups_reread == 0 and rep.groups_held == 0 and rep.groups_refused == 0 and rep.groups_forced == 0
    assert ing.refused == 0 and ing.blocked == 0
    by_group = {r.group: r for r in rep.results}
    assert sorted(by_group) == ["s1", "s2", "s3"]
    for g, res in by_group.items():
        assert res.accepted and not res.reread and not res.held and res.records_written > 1
        assert res.revision_id == RUN.revision_id_for(RUN.source_id_for(bumped.name, g), _group_inputs(bumped, g), "2+emitter2")
        assert f"{os.sep}_activations{os.sep}" not in res.files[0] and f"{os.sep}{res.revision_id}{os.sep}" in res.files[0]   # a revision directory
    rev1, rev2, rev3 = (by_group[g].revision_id for g in ("s1", "s2", "s3"))
    assert rev1 != rev0
    assert _revision_rows(root, "synthetic_store") == [("synthetic_store:s1", rev0, 1, rev1, "1+emitter2"),
                                                       ("synthetic_store:s1", rev1, 1, None, "2+emitter2"),
                                                       ("synthetic_store:s2", rev2, 1, None, "2+emitter2"),
                                                       ("synthetic_store:s3", rev3, 1, None, "2+emitter2")]
    # idempotent under the new version: a re-read reproduces the three ids and publishes no fourth; the plain pass writes nothing
    rep_rr, _ = _reread(root, _LabelledAdapterV2(store_root), synth.t(4000))
    assert rep_rr.groups_reread == 3 and rep_rr.groups_refreshed == 3 and rep_rr.groups_forced == 0
    assert {r.revision_id for r in rep_rr.results} == {rev1, rev2, rev3}
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE adapter = 'synthetic_store'") == [(4,)]
    rep_p, _ = _pass_and_ingest(root, _LabelledAdapterV2(store_root), synth.t(4500))
    assert rep_p.groups_refreshed == 0 and rep_p.groups_held == 0 and rep_p.records_written == 0


def test_reread_of_a_bumped_adapter_publishes_the_new_revision_instead_of_rereading_the_old_head(tmp_path):
    """`--reread` run first, before any incremental pass, on a reader whose version moved: the head's directory is not the bumped
    id's, so the read is a fresh revision directory (as the CLI help says), not a bundle re-observing the old head."""
    root, store_root = _setup(tmp_path)
    rep0, _ = _pass_and_ingest(root, _LabelledAdapterV1(store_root), synth.t(2000))
    rev0 = rep0.results[0].revision_id
    rep, ing = _reread(root, _LabelledAdapterV2(store_root), synth.t(3000))
    assert rep.groups_refreshed == 1 and rep.groups_reread == 0 and rep.groups_forced == 0 and ing.refused == 0 and ing.blocked == 0
    (res,) = rep.results
    assert res.revision_id != rev0 and not res.reread and f"{os.sep}{res.revision_id}{os.sep}" in res.files[0]
    assert f"{os.sep}_activations{os.sep}" not in res.files[0]
    assert _revision_rows(root, "synthetic_store") == [("synthetic_store:s1", rev0, 1, res.revision_id, "1+emitter2"),
                                                       ("synthetic_store:s1", res.revision_id, 1, None, "2+emitter2")]


def test_a_published_directory_the_index_has_not_folded_is_left_alone_by_the_incremental_pass(tmp_path):
    """The window between publication and ingest (relu_paired, 2026-09-20: three revisions published at 22:54Z and folded at 00:48Z
    were re-emitted whole as activation bundles by the passes at 00:32Z and 00:43Z). The directory exists, the database has no row
    for it, its watermarks have not folded, so the group reads as changed — and the pass writes NOTHING: the harvester folds the
    directory itself, and a bundle of its bytes is a full copy that holds again on whatever the directory holds on."""
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rep0 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))          # published, NOT ingested
        rev = rep0.results[0].revision_id
        assert rep0.groups_refreshed == 1 and os.path.isdir(os.path.join(H.records_dir(root), "adapters", ad.name, rev))
        assert _q(root, "SELECT COUNT(*) FROM source_revision") == [(0,)]
        rep1 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2500))
    finally:
        conn.close()
    assert rep1.inputs_changed == 1 and rep1.groups_held == 1 and rep1.groups_refreshed == 0 and rep1.groups_refused == 0
    assert rep1.records_written == 0
    (res1,) = rep1.results
    assert res1.held and not res1.accepted and not res1.reread and res1.revision_id == rev and res1.files == []
    assert "not ingested yet" in res1.error and "not re-emitted" in res1.error
    assert not os.path.isdir(RUN.activations_dir(root, ad.name, rev))
    # the harvester folds the directory: the revision is the accepted head, and the next pass writes nothing
    ing = H.ingest_once(root, workers=0, min_free=0)
    assert ing.refused == 0 and ing.blocked == 0
    assert _q(root, "SELECT revision_id, accepted FROM source_revision") == [(rev, 1)]
    rep2, _ = _pass_and_ingest(root, ad, synth.t(3000))
    assert rep2.groups_refreshed == 0 and rep2.groups_held == 0 and rep2.records_written == 0
    assert not os.path.isdir(RUN.activations_dir(root, ad.name, rev))


# ---- reactivation re-stamps every committed record to ONE written_utc; the judged/superseded pair still folds superseded ----------
# (the reducer's same-stamp tie-break; the declared readers' emission shape)


class _SupersedingStoreAdapter(SyntheticStoreAdapter):
    """The declared readers' shape (declared_common): a store whose row names a successor run emits the run's `judged` AND its
    `superseded` run_state in ONE batch, both at the pass clock -- the pair whose order the pre-tie-break fold drew by sha."""
    name = "synthetic_superseding_store"

    def read(self, ctx: AB.AdapterContext, group: str, inputs: Sequence[AB.Input]) -> Iterator[AB.RecordBatch]:
        for inp in inputs:
            for batch in super().read(ctx, group, [inp]):
                with open(inp.path) as fh:
                    store = json.load(fh)
                if store.get("superseded_by"):
                    rid = I.run_id(synth.CAMPAIGN, store["key"], store["object"], synth.EVAL_ID, synth.INSTRUMENT_ID)
                    batch.add("run_state", {"run_id": rid, "state": "superseded", "event_utc": ctx.now, "evidence_kind": "ruling",
                                            "note": f"superseded by {store['superseded_by']}"}, ctx.producer, ctx.now, observe=False)
                yield batch


def _store_with_successor(store_root, name, key, obj, scores, successor):
    path = _store(store_root, name, key, obj, scores)
    with open(path) as fh:
        store = json.load(fh)
    store["superseded_by"] = successor
    with open(path, "w") as fh:
        json.dump(store, fh)
    return path


def _run_state_records(directory):
    out = []
    for dp, _, fs in os.walk(directory):
        for f in sorted(fs):
            for _, rec, _ in R.read_record_file(os.path.join(dp, f)):
                if rec.get("record_type") == "run_state":
                    out.append(rec)
    return out


def test_reactivation_restamps_the_committed_pair_to_one_written_utc_and_the_run_still_folds_superseded(tmp_path):
    """Committed-byte reactivation: `_reemit_committed` copies every measurement record of the reverted-to revision with ONE fresh
    written_utc (the pass clock), so the pair's order is again a same-stamp question -- and the reactivation clock is CHOSEN here
    so that the re-emitted shas would have drawn `judged` under the old order. The run folds `superseded`, the history reads
    judged then superseded at that stamp, and a rebuild agrees."""
    root, store_root = _setup(tmp_path)
    _store_with_successor(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4], "run-b")
    ad = _SupersedingStoreAdapter(store_root)
    rid = I.run_id(synth.CAMPAIGN, "legacy_store_1", "constitution_lora@aaaa", synth.EVAL_ID, synth.INSTRUMENT_ID)
    # A: the fresh emission -- the pair at one stamp in one revision
    rep_a, ing_a = _pass_and_ingest(root, ad, synth.t(2000))
    assert ing_a.refused == 0 and ing_a.blocked == 0, ing_a.refusals
    rev_a = rep_a.results[0].revision_id
    committed = _run_state_records(os.path.join(root, "records", "adapters", ad.name, rev_a))
    assert sorted(r["payload"]["state"] for r in committed) == ["judged", "superseded"] and len({r["written_utc"] for r in committed}) == 1
    assert _q(root, "SELECT lifecycle_state, judged_utc FROM run WHERE run_id = ?", rid) == [("superseded", synth.t(1003))]
    # B: changed values supersede revision A
    _store_with_successor(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.9, 0.9], "run-b")
    rep_b, ing_b = _pass_and_ingest(root, ad, synth.t(3000))
    rev_b = rep_b.results[0].revision_id
    assert rev_b != rev_a and ing_b.refused == 0 and _q(root, "SELECT n FROM run_metric WHERE run_id = ?", rid) == [(2,)]
    # back to A's bytes: the incremental pass re-activates A's committed records with ONE fresh written_utc. Pick the clock at which
    # the re-emitted pair's shas are adversarial to the old `(written_utc, record_sha256)` order (superseded's sha below judged's).
    _store_with_successor(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4], "run-b")
    judged = next(r for r in committed if r["payload"]["state"] == "judged")
    sup = next(r for r in committed if r["payload"]["state"] == "superseded")

    def restamped_sha(rec, now):
        return R.record_sha(dict(rec, written_utc=now, payload=RUN.sanitise_free_form("run_state", rec["payload"])))

    now = next(synth.t(4000 + k) for k in range(256) if restamped_sha(sup, synth.t(4000 + k)) < restamped_sha(judged, synth.t(4000 + k)))
    rep_c, ing_c = _pass_and_ingest(root, ad, now)
    assert ing_c.refused == 0 and ing_c.blocked == 0, ing_c.refusals
    assert [r.revision_id for r in rep_c.results] == [rev_a]
    assert _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1 AND superseded_by_revision_id IS NULL AND source_id = ?",
              RUN.source_id_for(ad.name, "s1")) == [(rev_a,)]
    reemitted = _run_state_records(RUN.activations_dir(root, ad.name, rev_a))
    assert sorted(r["payload"]["state"] for r in reemitted) == ["judged", "superseded"] and {r["written_utc"] for r in reemitted} == {now}
    assert R.record_sha(next(r for r in reemitted if r["payload"]["state"] == "superseded")) < R.record_sha(next(r for r in reemitted if r["payload"]["state"] == "judged"))
    # the fold: superseded, deterministically -- the old order would have drawn judged at this stamp
    assert _q(root, "SELECT lifecycle_state, judged_utc FROM run WHERE run_id = ?", rid) == [("superseded", synth.t(1003))]
    assert _q(root, "SELECT n FROM run_metric WHERE run_id = ?", rid) == [(3,)]
    with DB.reading(DB.db_path(root)) as conn:
        hist = H.history(root, conn, "run", {"run_id": rid})
    at_now = [h["payload"]["state"] for h in hist if h["written_utc"] == now and h["record_type"] == "run_state"]
    assert at_now == ["judged", "superseded"]
    assert [h["state"]["lifecycle_state"] for h in hist if h["written_utc"] == now][-1] == "superseded"
    assert _rebuild_agrees(root, tmp_path, tables=("run", "run_state_event", "run_metric", "source_revision", "row_history")) == \
        {"run": 0, "run_state_event": 0, "run_metric": 0, "source_revision": 0, "row_history": 0}
# ============================================================================================
# the anchor decision as an input of the read
# ============================================================================================

import _reanchor_fixtures as X  # noqa: E402
from sisteer.registry.adapters import anchor_current as AC  # noqa: E402
from test_registry_cluster_read import _sampled_static_world  # noqa: E402

_SID = RUN.source_id_for("synthetic_store", "s1")


def _rev_inputs(root, rev):
    return json.loads(_q(root, "SELECT inputs_json FROM source_revision WHERE revision_id = ?", rev)[0][0])


def test_the_decision_input_joins_its_group_by_name_never_through_comparison_group(tmp_path):
    """On the path-grouped `sampled_static` fixture the URI handed to `comparison_group` would mint a phantom `registry:...` group; through
    `decision_inputs` it lands in the grades-dir group it belongs to, and nowhere else."""
    base = str(tmp_path / "A")
    os.makedirs(base)
    ad = _sampled_static_world(base)(base)
    inputs = ad.discover()
    groups = sorted({ad.comparison_group(i) for i in inputs})
    g = groups[0]
    s = AC.AnchorDecisionInput.of(RUN.source_id_for(ad.name, g), "a" * 64)
    assert ad.comparison_group(s) != g and ad.comparison_group(s).startswith("registry:")          # the phantom the routing would mint
    plan = RUN.plan_groups(ad, inputs, {}, "1", None, decision_inputs={g: s})
    assert sorted(plan.groups) == groups and not any(k.startswith("registry:") for k in plan.groups)
    assert plan.groups[g][-1] is s and not any(AC.is_decision_input(i) for k, v in plan.groups.items() if k != g for i in v)
    assert g in plan.changed and plan.decision_changed == {g}
    # a decision input of a group with no file input this pass is dropped here (the vanished path retires its watermark)
    plan2 = RUN.plan_groups(ad, inputs, {}, "1", None, decision_inputs={"/nowhere/grades": s})
    assert not any(AC.is_decision_input(i) for v in plan2.groups.values() for i in v) and plan2.decision_changed == set()


def test_a_moved_fingerprint_marks_exactly_its_group_changed_and_an_equal_one_leaves_it_quiet(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = SyntheticStoreAdapter(store_root)
    inputs = ad.discover()
    sid1, sid2 = RUN.source_id_for(ad.name, "s1"), RUN.source_id_for(ad.name, "s2")
    s1 = AC.AnchorDecisionInput.of(sid1, "1" * 64)
    marks_files = {i.identity_uri: RUN.Watermark(RUN.source_id_for(ad.name, ad.comparison_group(i)), AB.watermark_of(i), None) for i in inputs}
    marks = dict(marks_files, **{s1.identity_uri: RUN.Watermark(sid1, AB.watermark_of(s1), None)})
    quiet = RUN.plan_groups(ad, inputs, marks, "1", None, decision_inputs={"s1": s1})
    assert quiet.changed == set() and quiet.decision_changed == set() and quiet.inputs_changed == 0 and quiet.groups["s1"][-1] is s1
    moved = RUN.plan_groups(ad, inputs, marks, "1", None, decision_inputs={"s1": AC.AnchorDecisionInput.of(sid1, "2" * 64)})
    assert moved.changed == {"s1"} and moved.decision_changed == {"s1"} and moved.inputs_changed == 1 and "s2" not in moved.changed
    # a decision input with no watermark yet is a new input (read), and is not named as a decision change under --full alone
    fresh = RUN.plan_groups(ad, inputs, marks_files, "1", None, decision_inputs={"s2": AC.AnchorDecisionInput.of(sid2, "3" * 64)})
    assert fresh.changed == {"s2"} and fresh.decision_changed == {"s2"}
    # s1's decision watermark with no decision input this pass (its head stopped recording a set) is a GONE input of s1: one re-read
    gone = RUN.plan_groups(ad, inputs, marks, "1", None)
    assert gone.changed == {"s1"} and gone.decision_changed == set() and gone.gone_in_group == {"s1": {s1.identity_uri}} and gone.inputs_changed == 0
    full = RUN.plan_groups(ad, inputs, marks, "1", None, full=True, decision_inputs={"s1": s1})
    assert full.changed == {"s1", "s2"} and full.decision_changed == set()
    # the standing-refusal rule reads the decision input too: the refused attempt's id was minted with it
    refused_rev = RUN.revision_id_for(sid1, quiet.groups["s1"], "1")
    stamped = dict(marks)
    for i in quiet.groups["s1"]:
        stamped[i.identity_uri] = RUN.Watermark(sid1, AB.watermark_of(i), "synthetic refusal", refused_rev)
    standing = RUN.plan_groups(ad, inputs, stamped, "1", None, decision_inputs={"s1": s1})
    assert standing.refused_standing == ["s1"] and "s1" not in standing.changed
    retried = RUN.plan_groups(ad, inputs, stamped, "1", None, decision_inputs={"s1": AC.AnchorDecisionInput.of(sid1, "2" * 64)})
    assert "s1" in retried.changed and retried.decision_changed == {"s1"} and retried.refused_standing == []


def test_a_preserved_group_is_left_as_it_stands_whatever_selected_it_and_its_decision_watermark_is_no_gone_input(tmp_path):
    """When the pass's anchor currency cannot be loaded, a group whose head recorded a decision set has no synthetic
    input in discovery, so its decision watermark would read as a GONE input and select the group for a read the frame cannot serve.
    `plan_groups(preserved=...)` leaves such a group exactly as it stands: never in `changed` (a gone watermark, a file change, `--full`,
    `--reread`), its watermark not in `gone_in_group`, not standing-refused but named in `preserved`; the other groups plan as usual; a
    group not present is ignored; a `reader_traversal` adapter preserves every group when it preserves one."""
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    ad = SyntheticStoreAdapter(store_root)
    inputs = ad.discover()
    sid1 = RUN.source_id_for(ad.name, "s1")
    s1 = AC.AnchorDecisionInput.of(sid1, "1" * 64)
    marks_files = {i.identity_uri: RUN.Watermark(RUN.source_id_for(ad.name, ad.comparison_group(i)), AB.watermark_of(i), None) for i in inputs}
    marks = dict(marks_files, **{s1.identity_uri: RUN.Watermark(sid1, AB.watermark_of(s1), None)})
    gone = RUN.plan_groups(ad, inputs, marks, "1", None)
    assert gone.changed == {"s1"} and gone.gone_in_group == {"s1": {s1.identity_uri}} and gone.preserved == {}
    kept = RUN.plan_groups(ad, inputs, marks, "1", None, preserved={"s1": "no frame"})
    assert kept.changed == set() and kept.gone_in_group == {} and kept.preserved == {"s1": "no frame"} and kept.refused_standing == []
    assert kept.inputs_changed == 0 and kept.decision_changed == set() and sorted(kept.groups) == ["s1", "s2"]
    # a file change, --full and --reread read the other groups and never the preserved one
    changed_file = dict(marks, **{inputs[0].identity_uri: RUN.Watermark(sid1, ("0" * 64, 1, synth.t(1)), None)})
    assert ad.comparison_group(inputs[0]) == "s1"
    assert RUN.plan_groups(ad, inputs, changed_file, "1", None).changed == {"s1"}
    assert RUN.plan_groups(ad, inputs, changed_file, "1", None, preserved={"s1": "no frame"}).changed == set()
    full = RUN.plan_groups(ad, inputs, marks, "1", None, full=True, preserved={"s1": "no frame"})
    assert full.changed == {"s2"} and full.preserved == {"s1": "no frame"}
    assert RUN.plan_groups(ad, inputs, marks, "1", None, reread=True, preserved={"s1": "no frame"}).changed == {"s2"}
    # a preserved group with a standing refused stamp is named preserved, not standing
    refused_rev = RUN.revision_id_for(sid1, [i for i in inputs if ad.comparison_group(i) == "s1"] + [s1], "1")
    stamped = dict(marks)
    for i in [i for i in inputs if ad.comparison_group(i) == "s1"] + [s1]:
        stamped[i.identity_uri] = RUN.Watermark(sid1, AB.watermark_of(i), "synthetic refusal", refused_rev)
    assert RUN.plan_groups(ad, inputs, stamped, "1", None, decision_inputs={"s1": s1}).refused_standing == ["s1"]
    standing = RUN.plan_groups(ad, inputs, stamped, "1", None, preserved={"s1": "no frame"})
    assert standing.refused_standing == [] and "s1" not in standing.changed and standing.preserved == {"s1": "no frame"}
    # a group that is not present is ignored; a traversal adapter preserves every group when it preserves one
    absent = RUN.plan_groups(ad, inputs, marks, "1", None, preserved={"s9": "no frame"})
    assert absent.preserved == {} and absent.changed == {"s1"}
    trav = SyntheticStoreAdapter(store_root)
    trav.rng_scope = "reader_traversal"
    assert RUN.plan_groups(trav, inputs, marks, "1", None).changed == {"s1", "s2"}
    whole = RUN.plan_groups(trav, inputs, marks, "1", None, preserved={"s1": "no frame"})
    assert whole.changed == set() and whole.preserved == {"s1": "no frame", "s2": "no frame"}


def test_revision_id_folds_the_decision_fingerprint_and_inputs_json_names_the_decision_uri():
    files = [AB.Input("u1", "/p1", "a" * 64, 1, synth.t(1))]
    s_a, s_b = AC.AnchorDecisionInput.of(_SID, "1" * 64), AC.AnchorDecisionInput.of(_SID, "2" * 64)
    base = RUN.revision_id_for(_SID, files, 2)
    with_a, with_b = RUN.revision_id_for(_SID, files + [s_a], 2), RUN.revision_id_for(_SID, files + [s_b], 2)
    assert len({base, with_a, with_b}) == 3 and with_a == RUN.revision_id_for(_SID, files + [AC.AnchorDecisionInput.of(_SID, "1" * 64)], 2)
    assert RUN.inputs_json(files + [s_a]) == {"u1": "a" * 64, AC.decision_uri(_SID): "1" * 64}
    # the inputs-dropped comparison sees files only, on both sides
    prev = {"u1": "a" * 64, AC.decision_uri(_SID): "9" * 64}
    cmp_ = RUN.inputs_dropped_comparison(SyntheticStoreAdapter("/nonexistent"), "s1", files, "rev0", prev)
    assert cmp_ is not None and (cmp_["inputs_previous"], cmp_["inputs_now"], cmp_["inputs_dropped"], cmp_["tripped"]) == (1, 1, 0, False)
    assert RUN.inputs_dropped_comparison(SyntheticStoreAdapter("/nonexistent"), "s1", files, "rev0", {AC.decision_uri(_SID): "9" * 64}) is None


def test_a_ruling_that_moves_an_anchor_is_a_changed_input_a_new_revision_a_withdrawal_and_then_quiet(tmp_path):
    """End to end on a tmp registry: a head with no recorded set reads once under the bump and records it (one publish); nothing
    moved -> quiet; an `exclude` on the own unsteered run -> exactly that group changed with no file change and no bump, a NEW revision id
    whose `inputs_json` names the decision URI, the contrast re-anchored on the same-source sibling, the old (run, anchor) key withdrawn,
    the paper frame showing the re-anchored contrast once -> quiet; a `hold_out` on the chosen anchor -> the comparison PARKS (no contrast,
    the key withdrawn, "none" in the recorded set, the park named in `row_refusals`) -> quiet; the lift (the same ruling id re-filed as
    `reference`) -> the decision is the exclude-time one again, so its revision is RE-ACTIVATED (a reverted input), the bundle marker
    restating the committed decision set -> quiet. The post-read watermark is the recorded fingerprint at every step."""
    root, stores = X.contrast_world(tmp_path)
    steered, own, sib = X.run_id_of(X.STEERED, "constitution_lora@aaaa"), X.run_id_of(X.OWN), X.run_id_of(X.SIBLING)
    # a head minted before this code: no recorded set, no decision watermark, no synthetic input
    rep0, _ = X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(2000))
    rev0, extra0 = X.head_of(root, _SID)
    assert rep0.groups_refreshed == 3 and rep0.anchor_decisions_changed == 0 and "anchor_decisions" not in extra0 and X.decision_watermark(root, _SID) is None
    # the bump: every group read once; s1 records its one comparison, the own unsteered run current -> own_current, same_standup_none
    rep1, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2100))
    assert (rep1.groups_refreshed, rep1.inputs_changed, rep1.anchor_decisions_changed) == (3, 0, 0)
    rev1, extra1 = X.head_of(root, _SID)
    block1 = extra1["anchor_decisions"]
    assert rev1 != rev0 and block1["rule"] == AC.DECISION_RULE and (block1["n_comparisons"], block1["n_parked"]) == (1, 0) and block1["pre_read_fingerprint"] is None
    (entry1,) = block1["comparisons"]
    assert entry1["comparison"]["run_id"] == steered and entry1["comparison"]["key"]["own_anchor_run_id"] == own
    assert entry1["decision"] == {"anchor_run_id": own, "state": "own_current", "own_reason": "current", "anchor_kind": "same_standup_none",
                                  "anchor_judgment_id": I.judgment_id(own, synth.JUDGE_ID), "computation_id": "",
                                  "reading": {"value": 0.5, "k": 1.0, "n": 2, "ci_lo": None, "ci_hi": None}, "items_digest": None,
                                  "steered_items_digest": None, "steered_item_conflicts": None, "flags_digest": None, "steered_flags_digest": None, "why": ""}
    assert X.decision_watermark(root, _SID) == (block1["fingerprint"], None) and block1["input_uri"] == AC.decision_uri(_SID)
    assert list(_rev_inputs(root, rev1)) == [os.path.join(stores, "s1", "store.json")]                 # the first read's id came from the files alone
    assert X.paper_contrast(root, steered) == [(own, "same_standup_none", 1)] and len(X.current_contrast_keys(root, _SID)) == 1
    # the unsteered groups made no comparison: no set recorded, no decision watermark
    for g in ("s2", "s4"):
        assert "anchor_decisions" not in X.head_of(root, RUN.source_id_for("synthetic_store", g))[1]
    assert _q(root, "SELECT COUNT(*) FROM source_watermark WHERE input_uri LIKE 'registry://%'") == [(1,)]
    # nothing moved: quiet (no result, no directory, the head stands)
    rep2, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2200))
    assert rep2.results == [] and rep2.groups_refreshed == 0 and rep2.anchor_decisions_changed == 0 and X.head_of(root, _SID)[0] == rev1
    # the own unsteered run excluded: exactly s1, no file change, no bump -> a new id, re-anchored on the sibling, the old key withdrawn
    X.file_ruling(root, "r-exclude-own", "exclude", own, 2300)
    assert _q(root, "SELECT is_current FROM v_run_current WHERE run_id = ?", own) == [(0,)]
    rep3, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2400))
    assert [r.group for r in rep3.results] == ["s1"] and (rep3.groups_refreshed, rep3.inputs_changed, rep3.anchor_decisions_changed) == (1, 1, 1)
    (res3,) = rep3.results
    rev3, extra3 = X.head_of(root, _SID)
    assert res3.accepted and res3.withdrawals == 1 and res3.revision_id == rev3 and rev3 != rev1 and f"{os.sep}_activations{os.sep}" not in res3.files[0]
    block3 = extra3["anchor_decisions"]
    assert block3["comparisons"][0]["decision"] == {"anchor_run_id": sib, "state": "sibling_same_source", "own_reason": "excluded", "anchor_kind": "sibling_none",
                                                    "anchor_judgment_id": I.judgment_id(sib, synth.JUDGE_ID), "computation_id": "",
                                                    "reading": {"value": 0.5, "k": 1.0, "n": 2, "ci_lo": None, "ci_hi": None}, "items_digest": None,
                                                    "steered_items_digest": None, "steered_item_conflicts": None, "flags_digest": None, "steered_flags_digest": None,
                                                    "why": "selection_state excluded (ruling r-exclude-own)"}
    assert block3["pre_read_fingerprint"] == block3["fingerprint"] and X.decision_watermark(root, _SID) == (block3["fingerprint"], None)
    inputs3 = _rev_inputs(root, rev3)
    assert set(inputs3) == {os.path.join(stores, "s1", "store.json"), AC.decision_uri(_SID)} and inputs3[AC.decision_uri(_SID)] == block3["fingerprint"]
    assert rev3 == RUN.revision_id_for(_SID, [i for i in X.ContrastStoreAdapter(stores).discover() if os.path.basename(os.path.dirname(i.uri)) == "s1"]
                                       + [AC.AnchorDecisionInput.of(_SID, block3["fingerprint"])], 2)
    assert (extra3["inputs_previous"], extra3["inputs_now"], extra3["inputs_dropped"]) == (1, 1, 0)             # the decision input is no file
    assert _q(root, "SELECT reason, from_revision_id, to_revision_id, table_name FROM withdrawal WHERE source_id = ?", _SID) == [("removed_measurement", rev1, rev3, "run_contrast")]
    assert X.paper_contrast(root, steered) == [(sib, "sibling_none", 1)]
    assert [json.loads(k)["anchor_run_id"] for k in X.current_contrast_keys(root, _SID)] == [sib]
    assert _q(root, "SELECT COUNT(*) FROM run_contrast WHERE run_id = ?", steered) == [(2,)]                   # the old row stays a record
    assert _q(root, "SELECT json_extract(extra_json, '$.anchor_decision_drift') FROM source_status WHERE source_revision_id = ?", rev3) == [(None,)]
    rep4, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2500))
    assert rep4.results == [] and X.head_of(root, _SID)[0] == rev3
    # the chosen anchor held out: the comparison parks, the key is withdrawn, the set carries it as "none"
    X.file_ruling(root, "r-hold-sib", "hold_out", sib, 2600)
    assert _q(root, "SELECT is_current, held_out FROM v_run_current WHERE run_id = ?", sib) == [(0, 1)]
    ad5 = X.ContrastStoreAdapter(stores)
    rep5, _ = X.pass_and_fold(root, ad5, synth.t(2700))
    rev5, extra5 = X.head_of(root, _SID)
    assert [r.group for r in rep5.results] == ["s1"] and rep5.results[0].withdrawals == 1 and rev5 not in (rev1, rev3)
    block5 = extra5["anchor_decisions"]
    d5 = block5["comparisons"][0]["decision"]
    assert (block5["n_comparisons"], block5["n_parked"]) == (1, 1) and d5["anchor_run_id"] == AC.PARKED and d5["state"] == "no_current_anchor"
    assert d5["own_reason"] == "excluded" and "held_out (ruling r-hold-sib)" in d5["why"] and d5["anchor_kind"] is None
    assert X.paper_contrast(root, steered) == [(None, None, None)] and X.current_contrast_keys(root, _SID) == []
    assert len(ad5.last_report.refusals) == 1 and X.PARK_PHRASE in ad5.last_report.refusals[0]
    assert X.PARK_PHRASE in _q(root, "SELECT json_extract(extra_json, '$.row_refusals[0]') FROM source_status WHERE source_revision_id = ?", rev5)[0][0]
    assert X.decision_watermark(root, _SID) == (block5["fingerprint"], None)
    rep6, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2800))
    assert rep6.results == []
    # the lift: the same ruling id re-filed as `reference`; the decision is the exclude-time one, so rev3 is RE-ACTIVATED (a reverted input)
    X.file_ruling(root, "r-hold-sib", "reference", sib, 2900)
    assert _q(root, "SELECT is_current, held_out FROM v_run_current WHERE run_id = ?", sib) == [(1, 0)]
    rep7, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(3000))
    (res7,) = rep7.results
    assert res7.accepted and res7.reactivated and res7.revision_id == rev3 and f"{os.sep}_activations{os.sep}" in res7.files[0] and rep7.anchor_decisions_changed == 1
    rev7, extra7 = X.head_of(root, _SID)
    assert rev7 == rev3 and extra7["anchor_decisions"]["comparisons"][0]["decision"]["anchor_run_id"] == sib     # the committed set restated
    assert X.paper_contrast(root, steered) == [(sib, "sibling_none", 1)] and X.decision_watermark(root, _SID) == (block3["fingerprint"], None)
    rep8, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(3100))
    assert rep8.results == [] and X.head_of(root, _SID)[0] == rev3
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE source_id = ?", _SID) == [(4,)]                 # v1, v2, exclude, hold-out
    # a rebuild of the log equals the live index
    diff = _rebuild_agrees(root, tmp_path, tables=("source_revision", "run_contrast", "observation", "withdrawal", "source_watermark"))
    assert all(n == 0 for n in diff.values()), diff


def test_a_reader_that_stops_filing_contrasts_loses_the_decision_input_in_one_re_read_and_is_then_quiet(tmp_path):
    """The head recorded a set; the bumped reader consults no anchor. The bump's read carries the decision input in its id and records
    no set; the next pass sees a gone input, re-reads once under an id without it and retires the decision watermark; then quiet. The
    inputs-dropped comparison never counts the synthetic input."""
    root, stores = X.contrast_world(tmp_path)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))                   # the runs fold first: the v2 read's frame holds them
    X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2000))
    rev1, extra1 = X.head_of(root, _SID)
    assert "anchor_decisions" in extra1 and extra1["anchor_decisions"]["n_parked"] == 0
    rep2, _ = X.pass_and_fold(root, X.StoppedFilingAdapter(stores), synth.t(2100))
    assert rep2.groups_refreshed == 3 and rep2.anchor_decisions_changed == 0
    rev2, extra2 = X.head_of(root, _SID)
    assert rev2 != rev1 and "anchor_decisions" not in extra2 and AC.decision_uri(_SID) in _rev_inputs(root, rev2)
    assert X.decision_watermark(root, _SID) == (extra1["anchor_decisions"]["fingerprint"], None)             # the stale watermark stands
    rep3, _ = X.pass_and_fold(root, X.StoppedFilingAdapter(stores), synth.t(2200))
    assert [r.group for r in rep3.results] == ["s1"] and rep3.groups_refreshed == 1 and rep3.inputs_changed == 0 and rep3.anchor_decisions_changed == 0
    rev3, extra3 = X.head_of(root, _SID)
    assert rev3 != rev2 and list(_rev_inputs(root, rev3)) == [os.path.join(stores, "s1", "store.json")]
    assert (extra3["inputs_previous"], extra3["inputs_dropped"]) == (1, 0) and rep3.results[0].accepted and not rep3.results[0].inputs_dropped
    sha, err = X.decision_watermark(root, _SID)
    assert err == RUN.VANISHED_PREFIX + AC.decision_uri(_SID)
    rep4, _ = X.pass_and_fold(root, X.StoppedFilingAdapter(stores), synth.t(2300))
    assert rep4.results == [] and X.head_of(root, _SID)[0] == rev3


def test_a_reread_of_a_group_whose_decision_set_stands_is_a_bundle_of_the_same_id_carrying_the_set(tmp_path):
    """`--reread` is unaffected in the steady state: the decision input unmoved, the committed id is reproduced and read
    again for its facts, the bundle's marker restates the read's decision set, and the next incremental pass is quiet. One corollary of the
    first-landing rule (a head with no recorded set gets no decision input, so the FIRST read under this code mints its id from the files
    alone) is pinned first: the first `--reread` after it carries the decision input in the id, so it is a fresh revision of the SAME keys
    (no withdrawal, no bundle), and every `--reread` after that is the bundle."""
    root, stores = X.contrast_world(tmp_path)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))
    X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2000))
    rev1, extra1 = X.head_of(root, _SID)
    b1 = extra1["anchor_decisions"]
    assert b1["n_parked"] == 0 and b1["pre_read_fingerprint"] is None and AC.decision_uri(_SID) not in _rev_inputs(root, rev1)
    # the first --reread after the first landing: the decision input joins the id -> a fresh revision directory of the same keys
    rep, ing = _reread(root, X.ContrastStoreAdapter(stores), synth.t(2100))
    assert rep.groups_refreshed == 3 and rep.groups_reread == 2 and ing.refused == 0 and ing.blocked == 0
    s1 = next(r for r in rep.results if r.group == "s1")
    assert not s1.reread and s1.accepted and s1.withdrawals == 0 and s1.revision_id != rev1 and f"{os.sep}_activations{os.sep}" not in s1.files[0]
    rev2, extra2 = X.head_of(root, _SID)
    b2 = extra2["anchor_decisions"]
    assert rev2 == s1.revision_id and _rev_inputs(root, rev2)[AC.decision_uri(_SID)] == b1["fingerprint"] == b2["fingerprint"]
    assert b2["comparisons"] == b1["comparisons"] and b2["pre_read_fingerprint"] == b1["fingerprint"]
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ?", _SID) == [(0,)]
    assert len(X.current_contrast_keys(root, _SID)) == 1
    # the steady state: every --reread is a bundle of the same id, the set restated, and the incremental pass stays quiet
    rep_b, ing_b = _reread(root, X.ContrastStoreAdapter(stores), synth.t(2200))
    assert rep_b.groups_reread == 3 and rep_b.groups_refreshed == 3 and ing_b.refused == 0 and ing_b.blocked == 0
    s1_b = next(r for r in rep_b.results if r.group == "s1")
    assert s1_b.reread and s1_b.revision_id == rev2 and f"{os.sep}_activations{os.sep}" in s1_b.files[0]
    rev3, extra3 = X.head_of(root, _SID)
    assert rev3 == rev2 and extra3["anchor_decisions"]["comparisons"] == b1["comparisons"] and extra3["anchor_decisions"]["fingerprint"] == b1["fingerprint"]
    assert X.decision_watermark(root, _SID) == (b1["fingerprint"], None)
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE source_id = ?", _SID) == [(3,)]      # v1, the first read, the first re-read
    rep3, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2300))
    assert rep3.results == []


def test_a_vanished_group_retires_its_decision_watermark_with_its_files_and_states_no_registry_uri(tmp_path):
    root, stores = X.contrast_world(tmp_path)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))
    X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2000))
    assert len(X.current_contrast_keys(root, _SID)) == 1
    import shutil
    shutil.rmtree(os.path.join(stores, "s1"))
    rep, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2100))
    (res,) = rep.results
    assert res.group == "s1" and res.accepted and res.withdrawals >= 1 and res.records_written == 0
    assert _q(root, "SELECT last_error FROM source_watermark WHERE source_id = ? AND input_uri = ?", _SID, AC.decision_uri(_SID)) == [(RUN.VANISHED_PREFIX + AC.decision_uri(_SID),)]
    states = _q(root, "SELECT DISTINCT input_uri FROM source_state WHERE found = 0")
    assert states == [(os.path.join(stores, "s1", "store.json"),)]                        # the file only; the decision input is no file
    assert X.current_contrast_keys(root, _SID) == []
    rep2, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2200))
    assert rep2.results == []


def test_a_comparison_whose_anchor_is_not_yet_folded_parks_and_the_next_pass_files_it_on_the_fresh_frame(tmp_path):
    """No provisional anchors: the pass that EMITS the unsteered run cannot anchor on it -- the frame it
    loaded predates the fold -- so the comparison parks `not_in_frame`, recorded as "none" in the set; after the ingest the pre-read finds the
    own run current, the fingerprint moves, the group re-reads under a new id and files the contrast; the third pass is quiet. Recovery is
    autonomous: no operator step, no bump."""
    root, stores = X.contrast_world(tmp_path)
    steered, own = X.run_id_of(X.STEERED, "constitution_lora@aaaa"), X.run_id_of(X.OWN)
    ad1 = X.ContrastStoreAdapter(stores)
    rep1, _ = X.pass_and_fold(root, ad1, synth.t(2000))
    assert rep1.groups_refreshed == 3 and rep1.anchor_decisions_changed == 0
    rev1, extra1 = X.head_of(root, _SID)
    block1 = extra1["anchor_decisions"]
    d1 = block1["comparisons"][0]["decision"]
    assert (block1["n_comparisons"], block1["n_parked"]) == (1, 1) and d1["anchor_run_id"] == AC.PARKED and d1["own_reason"] == "not_in_frame"
    assert ad1.choices["s1"].state == "no_current_anchor" and ad1.choices["s1"].own_reason == "not_in_frame"
    assert X.PARK_PHRASE in _q(root, "SELECT json_extract(extra_json, '$.row_refusals[0]') FROM source_status WHERE source_revision_id = ?", rev1)[0][0]
    assert X.paper_contrast(root, steered) == [(None, None, None)] and X.current_contrast_keys(root, _SID) == []
    assert _q(root, "SELECT COUNT(*) FROM blocked_records WHERE record_type = 'run_contrast'") == [(0,)]     # nothing held on a missing parent
    # the fold made the unsteered runs current; the next pass re-reads exactly s1 and files the contrast
    rep2, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2100))
    assert [r.group for r in rep2.results] == ["s1"] and rep2.anchor_decisions_changed == 1 and rep2.inputs_changed == 1
    rev2, extra2 = X.head_of(root, _SID)
    d2 = extra2["anchor_decisions"]["comparisons"][0]["decision"]
    assert rev2 != rev1 and d2["anchor_run_id"] == own and d2["state"] == "own_current" and extra2["anchor_decisions"]["n_parked"] == 0
    assert X.paper_contrast(root, steered) == [(own, "same_standup_none", 1)] and rep2.results[0].withdrawals == 0
    rep3, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2200))
    assert rep3.results == [] and X.head_of(root, _SID)[0] == rev2


def test_a_corrected_anchor_reading_under_an_unchanged_identity_re_reads_the_contrast_on_the_next_pass(tmp_path):
    """End to end: the own unsteered store is corrected -- the same key and object, so the same run, judgment and
    computation ids; one score moved, so `value` 0.5 -> 0.6 with k and n unchanged -- and nothing of the steered group moves. The pass that
    re-reads the corrected store cannot see the correction (its frame predates the fold) and leaves the steered contrast's stale delta for one
    pass; the pass after the fold replays the recorded comparison against the corrected reading, the fingerprint moves, the steered group
    re-reads under a new id with no file change and files the delta against the corrected value under the SAME (run, anchor) key -- replaced,
    not withdrawn; the paper frame's one contrast carries the corrected delta -- and the next pass is quiet."""
    root, stores = X.contrast_world(tmp_path)
    steered, own = X.run_id_of(X.STEERED, "constitution_lora@aaaa"), X.run_id_of(X.OWN)
    steered_value = (0.2 + 0.8 + 0.4) / 3
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))
    X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2000))
    rev1, extra1 = X.head_of(root, _SID)
    d1 = extra1["anchor_decisions"]["comparisons"][0]["decision"]
    assert d1["anchor_run_id"] == own and d1["reading"] == {"value": 0.5, "k": 1.0, "n": 2, "ci_lo": None, "ci_hi": None}
    delta_sql = "SELECT delta FROM run_contrast WHERE run_id = ? AND anchor_run_id = ?"
    assert _q(root, delta_sql, steered, own) == [(pytest.approx(steered_value - 0.5),)]
    assert X.pass_once(root, X.ContrastStoreAdapter(stores), synth.t(2100)).results == []
    # the correction
    _store(stores, "s2", X.OWN, "none", [0.3, 0.9])
    rep2, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2200))
    assert [r.group for r in rep2.results] == ["s2"] and rep2.results[0].accepted and rep2.anchor_decisions_changed == 0      # s1: the frame predates the fold
    assert _q(root, "SELECT value, k, n FROM run_metric WHERE run_id = ?", own) == [(pytest.approx(0.6), 1, 2)]
    assert X.head_of(root, _SID)[0] == rev1 and _q(root, delta_sql, steered, own) == [(pytest.approx(steered_value - 0.5),)]        # stale, for one pass
    # the pass after the fold: the replayed decision carries the corrected reading -> exactly s1, no file change, a new id, the delta corrected
    rep3, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2300))
    assert [r.group for r in rep3.results] == ["s1"] and (rep3.anchor_decisions_changed, rep3.inputs_changed, rep3.groups_refreshed) == (1, 1, 1)
    (res3,) = rep3.results
    rev3, extra3 = X.head_of(root, _SID)
    assert res3.accepted and rev3 == res3.revision_id and rev3 != rev1 and res3.withdrawals == 0 and f"{os.sep}_activations{os.sep}" not in res3.files[0]
    block3 = extra3["anchor_decisions"]
    d3 = block3["comparisons"][0]["decision"]
    assert (d3["anchor_run_id"], d3["anchor_judgment_id"], d3["computation_id"], d3["state"]) == (own, d1["anchor_judgment_id"], d1["computation_id"], "own_current")
    assert d3["reading"]["value"] == pytest.approx(0.6) and {k: v for k, v in d3["reading"].items() if k != "value"} == {"k": 1.0, "n": 2, "ci_lo": None, "ci_hi": None}
    assert block3["pre_read_fingerprint"] == block3["fingerprint"] != extra1["anchor_decisions"]["fingerprint"]
    assert _rev_inputs(root, rev3)[AC.decision_uri(_SID)] == block3["fingerprint"] and X.decision_watermark(root, _SID) == (block3["fingerprint"], None)
    assert _q(root, delta_sql, steered, own) == [(pytest.approx(steered_value - 0.6),)]
    assert X.paper_contrast(root, steered) == [(own, "same_standup_none", 1)] and len(X.current_contrast_keys(root, _SID)) == 1
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ?", _SID) == [(0,)]
    assert _q(root, "SELECT json_extract(extra_json, '$.anchor_decision_drift') FROM source_status WHERE source_revision_id = ?", rev3) == [(None,)]
    assert X.pass_once(root, X.ContrastStoreAdapter(stores), synth.t(2400)).results == []


def test_an_unavailable_frame_preserves_the_recorded_groups_refuses_a_fresh_read_by_name_and_the_pass_completes(tmp_path):
    """A head recorded a decision set and the pass's anchor currency cannot be loaded (here a registry whose `v_run_current` is gone -- the
    pre-0021 shape; on a shard, a sidecar exported without its currency table): the group is PRESERVED as it stands -- not read, nothing
    written, its decision watermark neither retired nor re-stamped -- and named with the cause, where planning without the synthetic input
    read its watermark as vanished and selected it. A group read for another cause (a new steered store) whose read asks the frame refuses
    BY NAME for the pass report with NOTHING written -- no refused marker, no watermark stamp, no state row: an unavailable frame is a fault
    of the pass's environment, not a judgment about the inputs, so it never becomes a standing refusal waiting for `--retry-refused` -- and
    the pass completes instead of ending on the loader's exception. With the frame back the preserved group is quiet and the refused group
    is read normally by the next pass."""
    import sqlite3
    root, stores = X.contrast_world(tmp_path)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(1900))
    X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2000))
    rev1, extra1 = X.head_of(root, _SID)
    fp1 = extra1["anchor_decisions"]["fingerprint"]
    adapter_dir = os.path.join(H.records_dir(root), "adapters", "synthetic_store")
    dirs_before = sorted(os.listdir(adapter_dir))
    broken = str(tmp_path / "no_currency.sqlite")
    live, copy = sqlite3.connect(DB.db_path(root)), sqlite3.connect(broken)
    with copy:
        live.backup(copy)
    live.close()
    copy.execute("DROP VIEW v_run_current")
    copy.commit()
    copy.close()

    def broken_pass(now, **kw):
        with DB.reading(broken) as conn:
            return RUN.run_pass(root, [X.ContrastStoreAdapter(stores)], conn=conn, now=now, producer=synth.producer(), **kw)

    rep = broken_pass(synth.t(2100))
    (res,) = rep.results
    assert res.group == "s1" and res.anchor_unavailable and not res.accepted and res.files == [] and res.revision_id == ""
    assert res.error.startswith(f"{_SID}: preserved as it stands") and "v_run_current is not defined" in res.error
    assert (rep.groups_refused, rep.groups_anchor_unavailable, rep.groups_refreshed, rep.inputs_changed, rep.anchor_decisions_changed) == (1, 1, 0, 0, 0)
    assert rep.as_dict()["groups_anchor_unavailable"] == 1 and rep.refused_standing == {}
    assert sorted(os.listdir(adapter_dir)) == dirs_before                                                 # nothing written
    X.ingest(root)
    assert X.head_of(root, _SID)[0] == rev1 and X.decision_watermark(root, _SID) == (fp1, None)
    # a new steered store: read for its own cause, its read asks the frame -> refused by name and stamped; the pass completes
    X.store_with_anchor(stores, "s5", "legacy_store_5", "constitution_lora@bbbb", [0.5, 0.5], anchor_key=X.OWN)
    sid5 = RUN.source_id_for("synthetic_store", "s5")
    rep2 = broken_pass(synth.t(2200))
    assert [r.group for r in rep2.results] == ["s1", "s5"] and (rep2.groups_refused, rep2.groups_anchor_unavailable, rep2.groups_refreshed) == (2, 2, 0)
    res5 = rep2.results[1]
    assert res5.anchor_unavailable and not res5.accepted and not res5.inputs_dropped and res5.revision_id
    assert res5.files == [] and rep2.refused_standing == {}                                                # nothing written, nothing standing
    assert res5.error.startswith(f"{sid5}: the anchor currency of the pass is unavailable, the group refuses whole: v_run_current is not defined")
    assert sorted(os.listdir(adapter_dir)) == dirs_before and not os.path.isdir(os.path.join(adapter_dir, "_refused"))
    X.ingest(root)
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE source_id = ?", sid5) == [(0,)]              # no refused marker
    assert _q(root, "SELECT COUNT(*) FROM source_watermark WHERE source_id = ? OR input_uri LIKE ?", sid5, f"%{os.sep}s5{os.sep}%") == [(0,)]  # no stamp
    assert _q(root, "SELECT COUNT(*) FROM source_state WHERE input_uri LIKE ?", f"%{os.sep}s5{os.sep}%") == [(0,)]                # no state row
    assert X.head_of(root, _SID)[0] == rev1 and X.decision_watermark(root, _SID) == (fp1, None)
    # the frame still unavailable: the same refusal again, still nothing written (no `--retry-refused` involved: the inputs are still new)
    rep2b = broken_pass(synth.t(2250))
    assert [r.group for r in rep2b.results] == ["s1", "s5"] and rep2b.results[1].files == [] and rep2b.results[1].anchor_unavailable and rep2b.refused_standing == {}
    assert sorted(os.listdir(adapter_dir)) == dirs_before
    # the frame back: s1 quiet; s5 is read normally by the next pass, as a new group, with no flag
    rep3, _ = X.pass_and_fold(root, X.ContrastStoreAdapter(stores), synth.t(2300))
    (res3,) = rep3.results
    assert res3.group == "s5" and res3.accepted and not res3.anchor_unavailable and rep3.groups_anchor_unavailable == 0 and rep3.refused_standing == {}
    assert res3.revision_id == res5.revision_id                                                             # the id the refused attempt would have minted
    assert _q(root, "SELECT accepted FROM source_revision WHERE source_id = ?", sid5) == [(1,)]
    assert _q(root, "SELECT COUNT(*) FROM source_watermark WHERE source_id = ? AND last_error IS NOT NULL", sid5) == [(0,)]
    assert X.paper_contrast(root, X.run_id_of("legacy_store_5", "constitution_lora@bbbb")) == [(X.run_id_of(X.OWN), "same_standup_none", 1)]
    assert X.pass_once(root, X.ContrastStoreAdapter(stores), synth.t(2500)).results == []


def test_a_conflicting_accepted_item_provenance_moves_the_paired_fingerprint_and_the_pair_is_refused_by_name(tmp_path):
    """End to end over the real item objects: an item-paired comparison anchors on the own unsteered run and records the digest of its
    selected item rows; a SECOND accepted provenance of one of its items that disagrees with the selected row -- ranked below the adapter's
    by the provenance tie-break, so `v_item_result_current` keeps selecting the original row and the digest is equal -- is a conflict
    `v_item_conflict` reports and the paired-reading API refuses. The next pass must not leave the filed contrast current: the pre-read
    replay refuses the own anchor by name (`item_conflict`, the item named), the fingerprint moves, exactly that group is re-read with no
    file change, the contrast re-anchors on the same-source sibling (its items agree) and the old key is withdrawn; the sibling conflicted
    too, the comparison PARKS with both conflicts named and the key is withdrawn -> quiet."""
    root, stores = X.contrast_world(tmp_path)
    steered, own, sib = X.run_id_of(X.STEERED, "constitution_lora@aaaa"), X.run_id_of(X.OWN), X.run_id_of(X.SIBLING)
    own_j, sib_j = I.judgment_id(own, synth.JUDGE_ID), I.judgment_id(sib, synth.JUDGE_ID)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(2000))
    rep1, _ = X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2100))
    assert (rep1.groups_refreshed, rep1.anchor_decisions_changed) == (3, 0)
    rev1, extra1 = X.head_of(root, _SID)
    (entry1,) = extra1["anchor_decisions"]["comparisons"]
    d1 = entry1["decision"]
    assert entry1["comparison"]["needs_items"] is True and d1["anchor_run_id"] == own and d1["state"] == "own_current" and d1["own_reason"] == "current"
    assert isinstance(d1["items_digest"], str) and len(d1["items_digest"]) == 64
    assert isinstance(d1["steered_items_digest"], str) and len(d1["steered_items_digest"]) == 64 and d1["steered_item_conflicts"] == 0   # the steered side, folded by the plain pass
    assert X.paper_contrast(root, steered) == [(own, "same_standup_none", 1)]
    with DB.reading(DB.db_path(root)) as conn:
        digest1 = AC.AnchorCurrency.load(conn).items_digest("asr", own, own_j)
    assert digest1 == d1["items_digest"]
    assert _q(root, "SELECT value FROM v_item_result_current WHERE run_id = ? AND item_id = 'q0' AND metric_id = 'asr'", own) == [(0.1,)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_conflict WHERE run_id IN (?, ?)", own, sib) == [(0,)]
    assert X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2150))[0].results == []
    # a second accepted provenance of the own anchor's q0 disagrees (0.5 against 0.1): the selected row and the digest of the selected rows
    # are unchanged, the conflict relation reports the key and the paired-reading API refuses the pair
    X.conflicting_provenance(root, own, own_j, "q0", 0.5, 2200)
    own_rev = X.head_of(root, RUN.source_id_for("synthetic_store", "s2"))[0]
    assert _q(root, "SELECT value, source_revision_id FROM v_item_result_current WHERE run_id = ? AND item_id = 'q0' AND metric_id = 'asr'", own) == [(0.1, own_rev)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_provenances WHERE run_id = ? AND item_id = 'q0' AND metric_id = 'asr'", own) == [(2,)]
    assert _q(root, "SELECT item_id, n_provenances, n_distinct_values, value_min, value_max FROM v_item_conflict WHERE run_id = ?", own) == [("q0", 2, 2, 0.1, 0.5)]
    with DB.reading(DB.db_path(root)) as conn:
        frame = AC.AnchorCurrency.load(conn)
        assert frame.items_digest("asr", own, own_j) == digest1 and frame.items_conflicts("asr", own, own_j) == ["q0"] and frame.items_conflicts("asr", sib, sib_j) == []
    from sisteer import registry as REG
    with pytest.raises(ValueError, match="conflicting visible provenances"):
        REG.paired(steered, own, "asr", root=root)
    # the next pass: the fingerprint moved, exactly s1 re-read with no file change, re-anchored on the sibling, the own anchor refused by name
    rep3, _ = X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2300))
    assert [r.group for r in rep3.results] == ["s1"] and (rep3.groups_refreshed, rep3.inputs_changed, rep3.anchor_decisions_changed) == (1, 1, 1)
    (res3,) = rep3.results
    rev3, extra3 = X.head_of(root, _SID)
    assert res3.accepted and res3.withdrawals == 1 and res3.revision_id == rev3 and rev3 != rev1
    block3 = extra3["anchor_decisions"]
    d3 = block3["comparisons"][0]["decision"]
    conflict_why = (f"1 item(s) at (asr, 0.5) under judgment {own_j} carry conflicting visible provenances (q0; v_item_conflict): "
                    "a statistic paired over items refuses the pair while the sources disagree")
    assert (d3["anchor_run_id"], d3["state"], d3["own_reason"], d3["anchor_kind"]) == (sib, "sibling_same_source", "item_conflict", "sibling_none")
    assert d3["why"] == conflict_why and isinstance(d3["items_digest"], str) and d3["items_digest"] != d1["items_digest"]
    assert d3["steered_items_digest"] == d1["steered_items_digest"] and d3["steered_item_conflicts"] == 0                # the steered side unchanged
    assert block3["pre_read_fingerprint"] == block3["fingerprint"] and X.decision_watermark(root, _SID) == (block3["fingerprint"], None)
    sel3 = _q(root, "SELECT json_extract(extra_json, '$.anchor_selection') FROM run_contrast WHERE run_id = ? AND anchor_run_id = ?", steered, sib)
    assert json.loads(sel3[0][0])["own_anchor_why"] == conflict_why and json.loads(sel3[0][0])["own_reason"] == "item_conflict"
    assert json.loads(sel3[0][0])["rejected"] == [[own, conflict_why]]
    assert X.paper_contrast(root, steered) == [(sib, "sibling_none", 1)]
    assert _q(root, "SELECT reason, from_revision_id, to_revision_id FROM withdrawal WHERE source_id = ?", _SID) == [("removed_measurement", rev1, rev3)]
    assert X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2400))[0].results == []
    # the sibling conflicted too: the comparison PARKS, both conflicts named, the key withdrawn, then quiet
    X.conflicting_provenance(root, sib, sib_j, "q1", 0.05, 2500)
    assert _q(root, "SELECT item_id FROM v_item_conflict WHERE run_id = ?", sib) == [("q1",)]
    ad5 = X.PairedContrastStoreAdapter(stores)
    rep5, _ = X.pass_and_fold(root, ad5, synth.t(2600))
    assert [r.group for r in rep5.results] == ["s1"] and rep5.results[0].withdrawals == 1 and rep5.anchor_decisions_changed == 1
    rev5, extra5 = X.head_of(root, _SID)
    block5 = extra5["anchor_decisions"]
    d5 = block5["comparisons"][0]["decision"]
    assert rev5 not in (rev1, rev3) and (block5["n_comparisons"], block5["n_parked"]) == (1, 1)
    assert (d5["anchor_run_id"], d5["state"], d5["own_reason"], d5["items_digest"], d5["reading"]) == (AC.PARKED, "no_current_anchor", "item_conflict", None, None)
    assert d5["steered_items_digest"] == d1["steered_items_digest"] and d5["steered_item_conflicts"] == 0
    assert d5["why"].startswith(f"own anchor {own}: {conflict_why}; the unsteered run(s) at the key are not eligible: ")
    assert f"{own} ({conflict_why})" in d5["why"] and f"{sib} (1 item(s) at (asr, 0.5) under judgment {sib_j} carry conflicting visible provenances (q1; v_item_conflict)" in d5["why"]
    assert X.paper_contrast(root, steered) == [(None, None, None)] and X.current_contrast_keys(root, _SID) == []
    assert len(ad5.last_report.refusals) == 1 and X.PARK_PHRASE in ad5.last_report.refusals[0] and "conflicting visible provenances" in ad5.last_report.refusals[0]
    assert X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2700))[0].results == []
    assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE source_id = ?", _SID) == [(4,)]                 # v1, v2, the re-anchor, the park


def test_a_conflicting_provenance_of_a_steered_item_parks_the_pair_and_a_steered_item_change_re_derives_the_contrast(tmp_path):
    """End to end over the real item objects: the paired decision carries the STEERED side.
    A second accepted provenance disagreeing with a steered item -- the selected row unchanged, no store file changed, the anchor clean --
    is a conflict `v_item_conflict` reports on the steered run and `registry.paired` refuses; the next pass's replay moves the fingerprint,
    exactly that group is re-read, the comparison parks under `item_conflict` with the steered run named (the pool still on the record), the
    old key is withdrawn, then quiet. And the steered rows are part of the decision: a new steered group read before its own rows fold
    records `steered_items_digest` None, the fold moves it, the next pass re-reads the group alone and re-derives the contrast under a new
    revision with the same key (no withdrawal), then quiet. The rebuilt log equals the live index at the end."""
    root, stores = X.contrast_world(tmp_path)
    steered, own, sib = X.run_id_of(X.STEERED, "constitution_lora@aaaa"), X.run_id_of(X.OWN), X.run_id_of(X.SIBLING)
    steered_j = I.judgment_id(steered, synth.JUDGE_ID)
    X.pass_and_fold(root, SyntheticStoreAdapter(stores), synth.t(2000))
    X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2100))
    rev1, extra1 = X.head_of(root, _SID)
    d1 = extra1["anchor_decisions"]["comparisons"][0]["decision"]
    assert d1["anchor_run_id"] == own and len(d1["steered_items_digest"]) == 64 and d1["steered_item_conflicts"] == 0
    with DB.reading(DB.db_path(root)) as conn:
        frame = AC.AnchorCurrency.load(conn)
        assert frame.pair_items_digest("asr", steered, steered_j) == d1["steered_items_digest"] and frame.pair_items_conflicts("asr", steered, steered_j) == []
    assert X.paper_contrast(root, steered) == [(own, "same_standup_none", 1)]
    assert X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2150))[0].results == []
    # a second accepted provenance of the STEERED run's q0 disagrees (0.9 against 0.2): selected row unchanged, no file changed, the anchor clean
    X.conflicting_provenance(root, steered, steered_j, "q0", 0.9, 2200)
    assert _q(root, "SELECT value FROM v_item_result_current WHERE run_id = ? AND item_id = 'q0' AND metric_id = 'asr'", steered) == [(0.2,)]
    assert _q(root, "SELECT item_id, n_distinct_values FROM v_item_conflict WHERE run_id = ?", steered) == [("q0", 2)]
    assert _q(root, "SELECT COUNT(*) FROM v_item_conflict WHERE run_id IN (?, ?)", own, sib) == [(0,)]
    with DB.reading(DB.db_path(root)) as conn:
        frame = AC.AnchorCurrency.load(conn)
        assert frame.pair_items_digest("asr", steered, steered_j) == d1["steered_items_digest"] and frame.pair_items_conflicts("asr", steered, steered_j) == ["q0"]
    from sisteer import registry as REG
    with pytest.raises(ValueError, match="conflicting visible provenances"):
        REG.paired(steered, own, "asr", root=root)
    # the next pass: the fingerprint moved, exactly s1 re-read with no file change, the comparison PARKED under item_conflict naming the steered run
    ad3 = X.PairedContrastStoreAdapter(stores)
    rep3, _ = X.pass_and_fold(root, ad3, synth.t(2300))
    assert [r.group for r in rep3.results] == ["s1"] and (rep3.groups_refreshed, rep3.inputs_changed, rep3.anchor_decisions_changed) == (1, 1, 1)
    (res3,) = rep3.results
    rev3, extra3 = X.head_of(root, _SID)
    assert res3.accepted and res3.withdrawals == 1 and res3.revision_id == rev3 and rev3 != rev1
    block3 = extra3["anchor_decisions"]
    d3 = block3["comparisons"][0]["decision"]
    why3 = (f"the steered run {steered}: 1 item(s) at (asr, 0.5) under judgment {steered_j} carry conflicting visible provenances (q0; v_item_conflict): "
            "a statistic paired over items refuses the pair while the sources disagree")
    assert (block3["n_comparisons"], block3["n_parked"]) == (1, 1) and block3["pre_read_fingerprint"] == block3["fingerprint"]
    assert (d3["anchor_run_id"], d3["state"], d3["own_reason"], d3["anchor_kind"], d3["items_digest"], d3["reading"]) == (AC.PARKED, "no_current_anchor", "item_conflict", None, None, None)
    assert d3["why"] == why3 and d3["steered_item_conflicts"] == 1 and d3["steered_items_digest"] == d1["steered_items_digest"]
    assert ad3.choices["s1"].n_candidates == 2 and {c.run_id for c in ad3.choices["s1"].candidates} == {own, sib}      # the pool on the record
    assert X.paper_contrast(root, steered) == [(None, None, None)] and X.current_contrast_keys(root, _SID) == []
    assert _q(root, "SELECT reason, from_revision_id, to_revision_id FROM withdrawal WHERE source_id = ?", _SID) == [("removed_measurement", rev1, rev3)]
    assert len(ad3.last_report.refusals) == 1 and X.PARK_PHRASE in ad3.last_report.refusals[0] and f"the steered run {steered}" in ad3.last_report.refusals[0]
    assert X.decision_watermark(root, _SID) == (block3["fingerprint"], None)
    assert X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2400))[0].results == []
    # a NEW steered group (own anchor s2, already folded) read before its own rows fold: the decision records the steered digest None; the
    # fold moves it, the next pass re-reads exactly that group and re-derives the contrast under a new revision with the same key, then quiet
    X.store_with_anchor(stores, "s5", "legacy_store_5", "constitution_lora@bbbb", [0.5, 0.5], anchor_key=X.OWN)
    sid5, s5 = RUN.source_id_for("synthetic_store", "s5"), X.run_id_of("legacy_store_5", "constitution_lora@bbbb")
    rep5, _ = X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2500))
    assert [r.group for r in rep5.results] == ["s5"] and rep5.results[0].accepted
    rev5, extra5 = X.head_of(root, sid5)
    d5 = extra5["anchor_decisions"]["comparisons"][0]["decision"]
    assert d5["anchor_run_id"] == own and d5["steered_items_digest"] is None and d5["steered_item_conflicts"] == 0
    assert X.paper_contrast(root, s5) == [(own, "same_standup_none", 1)]
    rep6, _ = X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2600))
    assert [r.group for r in rep6.results] == ["s5"] and (rep6.inputs_changed, rep6.anchor_decisions_changed) == (1, 1)
    (res6,) = rep6.results
    rev6, extra6 = X.head_of(root, sid5)
    d6 = extra6["anchor_decisions"]["comparisons"][0]["decision"]
    assert res6.accepted and res6.withdrawals == 0 and rev6 == res6.revision_id and rev6 != rev5 and not res6.reactivated
    assert d6["anchor_run_id"] == own and len(d6["steered_items_digest"]) == 64 and d6["steered_item_conflicts"] == 0 and d6["items_digest"] == d5["items_digest"]
    assert X.paper_contrast(root, s5) == [(own, "same_standup_none", 1)] and len(X.current_contrast_keys(root, sid5)) == 1
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ?", sid5) == [(0,)]
    assert X.pass_and_fold(root, X.PairedContrastStoreAdapter(stores), synth.t(2700))[0].results == []
    diff = _rebuild_agrees(root, tmp_path, tables=("source_revision", "run_contrast", "observation", "withdrawal", "source_watermark", "item_result"))
    assert all(v == 0 for v in diff.values()), diff

