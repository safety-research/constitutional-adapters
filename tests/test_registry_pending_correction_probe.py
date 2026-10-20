"""A claim refuses while an identity correction of its run is pending OUTSIDE the run's own record files.

`pending.claim` folds the run's OWN record files first (every kind under `records/<campaign>/<run_dir>/`: a same-id `run` correction, a
`run_patch` of `judge_id`) and binds the identity the fold leaves. A correction that waits where the scoped fold must not reach -- an
adapter revision directory (atomic per revision, the harvester's), a held revision's staged constituents, a blocked record, a cluster push
still in the inbox -- would make a claim bound now `identity_stale` the moment the harvester's cycle folds it, the grading wasted. So before
anything is written `pending.pending_identity_corrections` probes those places, bounded, against the harvester's durable state, and the
claim refuses by name (`ClaimPendingCorrection`, a `ClaimNotFolded`: exit 1 = retry after the next cycle) when a pending record would MOVE
the binding; one that restates the identity (a held re-read of the same store) refuses nothing.
"""
from __future__ import annotations

import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from test_registry_judgment_currency import (INSTRUMENT_X2, INSTRUMENT_X2_ID, INSTRUMENT_X3, INSTRUMENT_X3_ID, _complete, _frame, _generated,
                                             _ingest, _judgment_rows, _q, _rebuild_equal, _root, _worklist_row, _write)

REV = "0123456789abcdef"


def _probe(root, rid):
    with DB.reading(DB.db_path(root)) as conn:
        return P.pending_identity_corrections(root, conn, rid)


def _run_identity(root, rid):
    return _q(root, "SELECT r.instrument_id AS i, r.judge_id AS j FROM run r WHERE r.run_id = ?", rid)[0]


def _claim_files(root, rid):
    d = os.path.join(root, "records", "claims", I.run_dir_name(rid))
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _correction(rid, instrument_id, i, **over):
    """A same-id `run` record of ``rid`` onto ``instrument_id`` (an adapter's or a producer's re-emission of the run)."""
    corrected = synth.run_payload("k", i=100, instrument_id=instrument_id, **over)
    corrected["run_id"] = rid
    return synth.rec("run", corrected, synth.t(i))


def _revision_dir(root, adapter, rev, run_recs, when):
    """An adapter revision directory as the runner publishes it (one directory rename: the run batch file and the `source_revision`
    marker, `records/adapters/<adapter>/<rev>/`), not yet seen by any harvester cycle."""
    d = os.path.join(root, "records", "adapters", adapter, rev)
    R.write_records(os.path.join(d, "000001-run.jsonl"), run_recs)
    marker = {"revision_id": rev, "source_id": f"{adapter}:synth", "adapter": adapter, "observed_utc": synth.t(when),
              "inputs_json": {"synthetic:///store": "0" * 64}, "accepted": 1}
    # the revision's membership, as the runner's marker file carries it (the run-set-drop guard reads the run ids from it)
    keys = sorted({R.canonical_json({"run_id": r["payload"]["run_id"]}) for r in run_recs})
    observation = {"source_id": f"{adapter}:synth", "revision_id": rev, "table_name": "run", "observation_keys": keys}
    R.write_records(os.path.join(d, "000002-source_revision.jsonl"), [synth.rec("source_revision", marker, synth.t(when)),
                                                                      synth.rec("observation", observation, synth.t(when))])
    return d


def test_an_unactivated_adapter_revision_correcting_the_instrument_refuses_the_claim_until_the_cycle_activates_it(tmp_path):
    """The plain case first: nothing pending, the probe is empty, a claim binds and releases. Then an adapter revision carrying a `run`
    record that moves the run to X2 is published (the directory, not yet activated): the claim refuses `ClaimPendingCorrection` naming the
    revision, with NOTHING written (no new claim file, no judgment row, the run row and the revision untouched -- the scoped fold never
    opens a revision directory). The harvester's cycle activates it; the next claim binds X2."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    assert _probe(root, rid) == []
    c0 = P.claim(root, rid, synth.JUDGE_ID, "s-0", now=synth.t(1000))
    assert P.release(root, rid, synth.JUDGE_ID, "s-0", now=synth.t(1001))
    files_before, rows_before = _claim_files(root, rid), _judgment_rows(root, rid)
    _revision_dir(root, "synthad", REV, [_correction(rid, INSTRUMENT_X2_ID, 2000)], 2000)
    why = _probe(root, rid)
    assert len(why) == 1 and why[0].startswith(f"unprocessed revision {REV} (records/adapters/synthad/{REV}/000001-run.jsonl#1)") and INSTRUMENT_X2_ID in why[0]
    with pytest.raises(P.ClaimPendingCorrection, match="pending outside its own record files") as ei:
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    assert isinstance(ei.value, P.ClaimNotFolded) and ei.value.run_id == rid and "unprocessed revision" in str(ei.value)
    assert _claim_files(root, rid) == files_before and _judgment_rows(root, rid) == rows_before                    # nothing written
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID                                                  # the revision was not folded
    assert _q(root, "SELECT COUNT(*) AS n FROM source_revision")[0]["n"] == 0
    # the cycle activates the revision; the correction is the folded identity and the probe is empty again
    rep = _ingest(root, synth.t(3500))
    assert rep.refused == 0, rep.refusals
    if _q(root, "SELECT COUNT(*) AS n FROM source_revision WHERE revision_id = ? AND accepted = 1", REV)[0]["n"] == 0:
        _ingest(root, synth.t(3600))
    assert _q(root, "SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 1, "refusal": None}]
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID and _probe(root, rid) == []
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    rows = _judgment_rows(root, rid)
    assert c1 != c0 and rows[c1]["status"] == "in_progress" and rows[c1]["claimed_instrument_id"] == INSTRUMENT_X2_ID
    assert rows[c0]["status"] == "abandoned"
    _rebuild_equal(root, tmp_path)


DIM_REV = "a1b2c3d4e5f60718"


def _dimension_bundle(root, rev, recs):
    """A dimension seed revision directory as `adapters.dimensions.write_seed_records` publishes one (`records/dimensions/<rev>/`, one
    directory rename): applied FIRST in the harvester's cycle, before the plain files and the blocked queue."""
    by_type = {}
    for r in recs:
        by_type.setdefault(r["record_type"], []).append(r)
    paths = []
    for i, (rt, rs) in enumerate(sorted(by_type.items()), 1):
        paths.append(R.write_records(os.path.join(root, "records", "dimensions", rev, R.record_file_name(i, rt)), rs))
    return paths


def test_a_correction_blocked_on_an_absent_parent_is_pending_once_the_parent_is_present_or_spooled(tmp_path):
    """The literal predicate on the blocked queue, with the cycle's ORDER: a revision's `run` record names an instrument no row exists for
    (X3 unseeded), so the cycle HOLDS the revision (marker `accepted = 0`, kind blocked). While X3 is absent and nowhere on the spool the
    next cycle folds nothing of it (`harvest.parent_absent` True, nothing pending) -- the probe is empty and the claim binds the folded
    identity. Then X3 is PUBLISHED as a dimension bundle and not yet harvested: the next cycle folds the bundle BEFORE it re-evaluates the
    queue, so the correction folds in that very cycle -- the probe names the held revision and the claim refuses (no scoped fold in
    between). ONE full cycle later the run is on X3, the earlier claim is identity-stale and the same session's re-claim binds X3."""
    root, rid = _generated(tmp_path)
    _revision_dir(root, "synthad", REV, [_correction(rid, INSTRUMENT_X3_ID, 2000)], 2000)
    rep = _ingest(root, synth.t(2500))
    assert rep.refused == 0, rep.refusals
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 0}]        # held: X3 has no row
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID
    assert _probe(root, rid) == []                                                                   # nothing folds next cycle
    c0 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    assert _judgment_rows(root, rid)[c0]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    # the parent is published (a dimension bundle on the spool, not harvested): the next cycle folds it first, then the hold
    _dimension_bundle(root, DIM_REV, [synth.rec("instrument", INSTRUMENT_X3, synth.t(3100))])
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID and _q(root, "SELECT COUNT(*) AS n FROM instrument WHERE instrument_id = ?", INSTRUMENT_X3_ID)[0]["n"] == 0
    why = _probe(root, rid)
    assert why and all(w.startswith("held revision ") and INSTRUMENT_X3_ID in w for w in why), why
    with pytest.raises(P.ClaimPendingCorrection, match="held revision"):
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(3200))
    assert _ingest(root, synth.t(3500)).refused == 0                                                  # ONE full cycle: bundle, then the queue
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 1}]
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X3_ID and _probe(root, rid) == []
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))                                  # the stale-bound c0 is held by nobody
    rows = _judgment_rows(root, rid)
    assert c1 != c0 and rows[c0]["status"] == "abandoned" and rows[c1]["claimed_instrument_id"] == INSTRUMENT_X3_ID
    _rebuild_equal(root, tmp_path)


def test_an_aged_hold_is_as_pending_as_a_fresh_one(tmp_path):
    """`blocked_records.stale` is an age mark (`BLOCKED_STALE_S`, seven days), not a retry predicate -- the
    queue re-evaluates EVERY row. A held correction waits eight days (its marker and constituent read `stale = 1`); its instrument is then
    published; the probe must still name it, and the next cycle activates it."""
    root, rid = _generated(tmp_path)
    _revision_dir(root, "synthad", REV, [_correction(rid, INSTRUMENT_X3_ID, 2000)], 2000)
    assert _ingest(root, synth.t(2500)).refused == 0
    assert _ingest(root, synth.t(2500 + 9 * 86400)).refused == 0                                      # eight+ days later: the rows age
    aged = _q(root, "SELECT record_type, stale FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ? ORDER BY record_type", REV)
    assert aged and all(r["stale"] == 1 for r in aged), aged
    assert _probe(root, rid) == []                                                                   # still nothing to fold: X3 absent
    _dimension_bundle(root, DIM_REV, [synth.rec("instrument", INSTRUMENT_X3, synth.t(2500 + 9 * 86400 + 10))])
    why = _probe(root, rid)
    assert why and all(w.startswith("held revision ") and INSTRUMENT_X3_ID in w for w in why), why      # aged, and pending all the same
    with pytest.raises(P.ClaimPendingCorrection):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(2500 + 9 * 86400 + 20))
    assert _ingest(root, synth.t(2500 + 9 * 86400 + 100)).refused == 0
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X3_ID and _probe(root, rid) == []
    _rebuild_equal(root, tmp_path)


def test_a_probe_miss_costs_one_regrade_never_a_wrong_row(tmp_path, monkeypatch):
    """The contract, end to end: the probe is best-effort; the correctness belt is the identity-stale rule. A miss is FORCED (the probe
    returns nothing) while an unprocessed revision corrects the run to X2: the claim binds X; the judgment completes under that binding;
    one real cycle folds the correction. Then: the judgment is identity-stale (out of the frame, covering nothing), `v_pending_judgment`
    lists the run again for the judge (`identity_stale`), and a second claim binds X2. One re-graded run, never a wrong row -- which is
    what lets a future gap in the probe's table be a cost rather than a blocker."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    _revision_dir(root, "synthad", REV, [_correction(rid, INSTRUMENT_X2_ID, 2000)], 2000)
    assert _probe(root, rid)                                                                         # the probe WOULD see it
    monkeypatch.setattr(P, "pending_identity_corrections", lambda *a, **k: [])                       # the forced miss
    c0 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    monkeypatch.undo()
    assert _judgment_rows(root, rid)[c0]["claimed_instrument_id"] == synth.INSTRUMENT_ID              # bound to the identity about to be replaced
    _complete(root, rid, c0, synth.JUDGE_ID, "s-A", 3000, 3100)                                       # graded under that binding
    assert _ingest(root, synth.t(3500)).refused == 0                                                  # the correction and the completion fold
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID
    rows = _judgment_rows(root, rid)
    assert rows[c0]["status"] == "complete" and rows[c0]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    assert _frame(root, rid) == []                                                                   # identity-stale: out of the frame
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM judgment j JOIN run r ON r.run_id = j.run_id LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                            f"WHERE j.judgment_id = ? AND {S.judgment_identity_current_sql('j', 'r', 'i')}", (c0,)).fetchone()[0] == 0
        ok, why = H.completion_requirements_hold(conn, rid)
        assert not ok and "identity_stale" in why
    assert _worklist_row(root, rid, synth.JUDGE_ID)["pending_reason"] == "identity_stale"            # pending again for the judge
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    assert c1 != c0 and _judgment_rows(root, rid)[c1]["claimed_instrument_id"] == INSTRUMENT_X2_ID    # the re-grade binds the current identity
    _rebuild_equal(root, tmp_path)


def _rule_scored_run(tmp_path):
    """A run on the rule-scored instrument (it names no judge) with `judge_id` set by a run_patch, under its OWN directory."""
    root = _root(tmp_path)
    # on a root another seed already folded, the rule-scored dimensions go into a NEW file: an ingested record file is immutable,
    # so appending them to 000001-dims.jsonl would be refused by name (the first seed's dims are already in; only the extra ones are new)
    first = os.path.join(root, "records", "dims", "000001-dims.jsonl")
    if os.path.exists(first) or os.path.exists(first + ".gz"):
        _write(root, "dims", "000002-dims.jsonl", synth.seed_rule_scored_dimensions())
    else:
        _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid, recs = synth.rule_scored_run_records("rk", 100, through="generated", n_items=4)
    recs.append(synth.rec("run_patch", {"run_id": rid, "fields": {"judge_id": synth.JUDGE_ID}}, synth.t(150)))
    _write(root, os.path.join("c", I.run_dir_name(rid)), "000001-run.jsonl", recs)
    assert _ingest(root, synth.t(900)).refused == 0
    assert _run_identity(root, rid) == {"i": synth.RULE_INSTRUMENT_ID, "j": synth.JUDGE_ID}
    return root, rid


def _judge_patch(rid, judge, i):
    return synth.rec("run_patch", {"run_id": rid, "fields": {"judge_id": judge}}, synth.t(i))


def test_a_cluster_pushed_run_patch_moving_the_fallback_judge_refuses_the_claim_until_the_inbox_is_merged(tmp_path):
    """`run_patch.fields.judge_id` moves the binding when the instrument names no judge. Pushed by a cluster, the patch sits in
    `inbox/<host>/<campaign>/<run_dir>/` until the cycle merges it: the claim refuses naming the inbox, nothing written; after the cycle
    the run's fallback judge is JUDGE_ID_2 and a claim binds it."""
    root, rid = _rule_scored_run(tmp_path)
    run_dir = I.run_dir_name(rid)
    R.write_records(os.path.join(root, "inbox", "cluster_a", "c", run_dir, "000002-run_patch.jsonl"), [_judge_patch(rid, synth.JUDGE_ID_2, 2000)])
    why = _probe(root, rid)
    assert len(why) == 1 and why[0].startswith(f"inbox cluster_a/c/{run_dir} (inbox/cluster_a/c/{run_dir}/000002-run_patch.jsonl#1)") and synth.JUDGE_ID_2 in why[0]
    with pytest.raises(P.ClaimPendingCorrection, match="inbox"):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    assert _claim_files(root, rid) == [] and _judgment_rows(root, rid) == {}
    assert _run_identity(root, rid)["j"] == synth.JUDGE_ID                                                       # the inbox is the cycle's to merge
    assert _ingest(root, synth.t(3500)).refused == 0
    assert _run_identity(root, rid)["j"] == synth.JUDGE_ID_2 and _probe(root, rid) == []
    assert not os.path.exists(os.path.join(root, "inbox", "cluster_a", "c", run_dir, "000002-run_patch.jsonl"))
    c1 = P.claim(root, rid, synth.JUDGE_ID_2, "s-B", now=synth.t(4000))
    assert _judgment_rows(root, rid)[c1]["claimed_required_judge_id"] == synth.JUDGE_ID_2
    _rebuild_equal(root, tmp_path)


def test_a_spooled_run_patch_in_the_runs_own_directory_is_folded_first_and_bound_without_refusal(tmp_path):
    """The widened scoped set: a `run_patch` of `judge_id` spooled under the run's own directory is the producer's own statement; the
    claim folds it first (no cycle) and binds the moved judge -- no refusal, the probe empty."""
    root, rid = _rule_scored_run(tmp_path)
    _write(root, os.path.join("c", I.run_dir_name(rid)), "000002-run_patch.jsonl", [_judge_patch(rid, synth.JUDGE_ID_2, 2000)])
    assert _run_identity(root, rid)["j"] == synth.JUDGE_ID and _probe(root, rid) == []                            # own file: the fold's, not the probe's
    c1 = P.claim(root, rid, synth.JUDGE_ID_2, "s-A", now=synth.t(3000))
    assert _run_identity(root, rid)["j"] == synth.JUDGE_ID_2                                                       # folded by the claim
    assert _judgment_rows(root, rid)[c1]["claimed_required_judge_id"] == synth.JUDGE_ID_2
    _rebuild_equal(root, tmp_path)


def test_a_pending_record_that_restates_the_identity_refuses_nothing(tmp_path):
    """A held or unseen adapter re-read restates the run's identity (another store uri, the same instrument and judge): no correction, the
    probe is empty and the claim binds the folded identity -- refusing on it would refuse grading for as long as the revision waits."""
    root, rid = _generated(tmp_path)
    _revision_dir(root, "synthad", REV, [_correction(rid, synth.INSTRUMENT_ID, 2000, store_uri="synthetic:///store/moved")], 2000)
    assert _probe(root, rid) == []
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    assert _judgment_rows(root, rid)[c1]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    assert _ingest(root, synth.t(3500)).refused == 0
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID
    assert P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000)) == c1                                      # still the current-bound claim
    _rebuild_equal(root, tmp_path)


def test_binding_move_of_is_the_pure_rule(tmp_path):
    """The comparison the probe applies: a `run` record moving the instrument, or the fallback judge where the instrument names none, or
    naming an instrument with no row; a `run_patch` moving `judge_id` only where the instrument names no judge; everything else restates."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    with DB.reading(DB.db_path(root)) as conn:
        folded = P.run_identity_binding(conn, rid)
        assert folded == (synth.INSTRUMENT_ID, synth.JUDGE_ID)
        same = _correction(rid, synth.INSTRUMENT_ID, 5000)
        other_run = _correction("not-this-run", INSTRUMENT_X2_ID, 5000)
        moved = _correction(rid, INSTRUMENT_X2_ID, 5000)
        unknown = _correction(rid, INSTRUMENT_X3_ID, 5000)
        assert P.binding_move_of(conn, rid, same, folded, None) is None and P.binding_move_of(conn, rid, other_run, folded, None) is None
        assert "moves the binding" in P.binding_move_of(conn, rid, moved, folded, None)
        assert "no instrument row yet" in P.binding_move_of(conn, rid, unknown, folded, None)
        # a record stamped BEFORE the standing statement of the columns it would move folds into the history and moves the row nothing
        # (per-column latest statement): the run was stated at t(100), so a correction stamped t(1) is no move
        assert P.binding_move_of(conn, rid, _correction(rid, INSTRUMENT_X2_ID, 1), folded, None) is None
        assert P._standing_stamps(conn, rid)["instrument_id"] == synth.t(100)
        # the seeded instrument names a judge: a run_patch of judge_id moves the fallback, not the required judge -> no move
        assert P.binding_move_of(conn, rid, _judge_patch(rid, synth.JUDGE_ID_2, 5000), folded, None) is None
        assert P.binding_move_of(conn, rid, synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": 3}}, synth.t(5000)), folded, None) is None
        assert P.binding_move_of(conn, rid, synth.rec("judgment", {"run_id": rid}, synth.t(5000)), folded, None) is None
        # an instrument naming no judge: the fallback IS the required judge, so the patch moves it
        rule_folded = (synth.RULE_INSTRUMENT_ID, synth.JUDGE_ID)
    root2, rid2 = _rule_scored_run(tmp_path)
    with DB.reading(DB.db_path(root2)) as conn:
        assert P.run_identity_binding(conn, rid2) == rule_folded
        assert "moves judge_id" in P.binding_move_of(conn, rid2, _judge_patch(rid2, synth.JUDGE_ID_2, 5000), rule_folded, synth.JUDGE_ID)
        assert P.binding_move_of(conn, rid2, _judge_patch(rid2, synth.JUDGE_ID, 5000), rule_folded, synth.JUDGE_ID) is None
        assert P.binding_move_of(conn, rid2, _judge_patch(rid2, synth.JUDGE_ID_2, 1), rule_folded, synth.JUDGE_ID) is None       # older than the t(150) patch


def test_the_cli_exits_1_on_a_pending_correction_like_any_not_folded_refusal(tmp_path, monkeypatch, capsys):
    root, rid = _generated(tmp_path)
    monkeypatch.setattr(P, "claim", lambda *a, **k: (_ for _ in ()).throw(P.ClaimPendingCorrection(rid, "test: an unprocessed revision")))
    rc = cli.main(["--root", root, "pending-judgment", "--claim", "--run-id", rid, "--judge", synth.JUDGE_ID, "--session", "s"])
    err = capsys.readouterr().err
    assert rc == 1 and err.startswith("claim refused: ClaimPendingCorrection: ") and "pending outside its own record files" in err and "retry after the next harvester cycle" in err


# --------------------------------------------------------------------------------------------
# activation bundles are probed; a superseded hold is not a pending correction; every hit is a record the next cycle folds
# --------------------------------------------------------------------------------------------


def _bundle_dir(root, adapter, rev, stamp, run_recs, when):
    """An activation bundle as the runner publishes one (`records/adapters/<adapter>/_activations/<rev>/<stamp>/`, the stamp digits only as
    `harvest.REVISION_DIR_RE` requires: the revision's committed run records re-emitted with a later stamp and a fresh marker with the SAME
    revision id), not yet folded by any cycle."""
    d = os.path.join(root, "records", "adapters", adapter, "_activations", rev, stamp)
    R.write_records(os.path.join(d, "000001-run.jsonl"), run_recs)
    marker = {"revision_id": rev, "source_id": f"{adapter}:synth", "adapter": adapter, "observed_utc": synth.t(when),
              "inputs_json": {"synthetic:///store": "0" * 64}, "accepted": 1}
    R.write_records(os.path.join(d, "999998-source_revision.jsonl"), [synth.rec("source_revision", marker, synth.t(when))])
    return d


REV_B = "fedcba9876543210"


def test_an_unfolded_activation_bundle_of_an_older_revision_refuses_the_claim_until_the_cycle_activates_it(tmp_path):
    """Revision A carried the run on X (the seeded instrument) and was accepted; the newer head B corrected it to X2; a
    re-activation of A publishes X again as a bundle under `_activations/A/<stamp>/`. A grader claiming before the harvester folds the
    bundle would bind X2 and read `identity_stale` the moment the bundle activates: the claim refuses naming the bundle, nothing written;
    after the cycle the run reads X again and the claim binds X. The probe finds the bundle through the revisions that CARRIED the run
    (its row_history uris), never by walking the activation directories."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    _revision_dir(root, "synthad", REV, [_correction(rid, synth.INSTRUMENT_ID, 2000)], 2000)         # A: the run on X
    assert _ingest(root, synth.t(2100)).refused == 0
    _revision_dir(root, "synthad", REV_B, [_correction(rid, INSTRUMENT_X2_ID, 2200)], 2200)           # B, the head: corrected to X2
    assert _ingest(root, synth.t(2300)).refused == 0
    assert _q(root, "SELECT revision_id, accepted, superseded_by_revision_id AS s FROM source_revision ORDER BY observed_utc") == \
        [{"revision_id": REV, "accepted": 1, "s": REV_B}, {"revision_id": REV_B, "accepted": 1, "s": None}]
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID and _probe(root, rid) == []
    # a re-activation of A: X re-emitted under _activations/A/<stamp>/, unfolded
    _bundle_dir(root, "synthad", REV, "20260104000000000", [_correction(rid, synth.INSTRUMENT_ID, 3000)], 3000)
    why = _probe(root, rid)
    assert len(why) == 1 and why[0].startswith(f"unfolded activation bundle synthad/_activations/{REV}/20260104000000000 (") and synth.INSTRUMENT_ID in why[0], why
    with pytest.raises(P.ClaimPendingCorrection, match="unfolded activation bundle"):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3100))
    assert _claim_files(root, rid) == [] and _judgment_rows(root, rid) == {} and _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID
    # the cycle activates the bundle: the run is back on X, the probe is empty, the claim binds X
    assert _ingest(root, synth.t(3500)).refused == 0
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID and _probe(root, rid) == []
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(4000))
    assert _judgment_rows(root, rid)[c1]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    # a bundle the cycle has already processed is never a hit again (the delivery row of its marker is the bound)
    assert _probe(root, rid) == []
    _rebuild_equal(root, tmp_path)


def test_a_superseded_held_correction_is_not_pending_and_the_accepted_head_rules(tmp_path):
    """Finding 2: revision A is HELD with a staged correction onto an instrument no row exists for; a newer revision B of the same source
    is accepted and restates the run's identity. The harvester stamps A `superseded_by_revision_id = B`; while the instrument is absent
    nothing re-attempts A, and A's correction is stamped before B's statement, so it is never a binding move: the probe ignores it and
    the claim binds the folded identity -- else every later claim of the run would be refused with no route out. (The harvester re-attempts
    a superseded `blocked` hold once its parent lands -- only a `refused` hold is exact -- and the row keeps B's later statement.)"""
    root, rid = _generated(tmp_path)
    _revision_dir(root, "synthad", REV, [_correction(rid, INSTRUMENT_X3_ID, 2000)], 2000)             # A: held (X3 unseeded)
    assert _ingest(root, synth.t(2100)).refused == 0
    assert _q(root, "SELECT accepted, superseded_by_revision_id AS s FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 0, "s": None}]
    assert _probe(root, rid) == []                                                                   # parent absent: nothing folds next cycle
    _revision_dir(root, "synthad", REV_B, [_correction(rid, synth.INSTRUMENT_ID, 2200, store_uri="synthetic:///store/again")], 2200)   # B: accepted, same identity
    for i in range(2):
        assert _ingest(root, synth.t(2300 + i)).refused == 0
    assert _q(root, "SELECT accepted, superseded_by_revision_id AS s FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 0, "s": REV_B}]
    assert _q(root, "SELECT COUNT(*) AS n FROM revision_staging_record WHERE revision_id = ?", REV)[0]["n"] >= 1       # the staging row stays
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID
    assert _probe(root, rid) == []                                                                   # superseded: parked, not pending
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    assert _judgment_rows(root, rid)[c1]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    # the parent lands: the harvester DOES re-attempt a superseded `blocked` hold (only a refused one is exact) and A activates -- yet the
    # run keeps B's later statement (per-column latest), so A's older-stamped correction was never a binding move; the probe stayed empty
    # and the claim stands
    _write(root, "dims", "000002-x3.jsonl", [synth.rec("instrument", INSTRUMENT_X3, synth.t(2400))])
    assert _probe(root, rid) == []
    for i in range(2):
        assert _ingest(root, synth.t(2500 + i)).refused == 0
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 1}]
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID and _probe(root, rid) == []
    assert P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3100)) == c1
    assert _judgment_rows(root, rid)[c1]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    _rebuild_equal(root, tmp_path)


def test_a_torn_revision_directory_and_a_malformed_correction_are_not_pending(tmp_path):
    """Every hit names a record the next cycle folds: a revision directory without its marker (a crash between staging and publication)
    applies nothing, and a correction the validator refuses is a refusal fact, not a pending identity -- neither refuses the claim."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    torn = os.path.join(root, "records", "adapters", "synthad", REV)
    R.write_records(os.path.join(torn, "000001-run.jsonl"), [_correction(rid, INSTRUMENT_X2_ID, 2000)])     # no marker file
    assert _probe(root, rid) == []
    bad = _correction(rid, INSTRUMENT_X2_ID, 2100)
    bad["payload"]["lifecycle_state"] = "not-a-state"                                                      # the validator refuses it
    R.write_records(os.path.join(root, "inbox", "cluster_a", "c", I.run_dir_name(rid), "000002-run.jsonl"), [bad])
    assert _probe(root, rid) == []
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    assert _judgment_rows(root, rid)[c1]["claimed_instrument_id"] == synth.INSTRUMENT_ID


# --------------------------------------------------------------------------------------------
# a marker-only release bundle folds the staged correction; an exact refused hold is never re-attempted
# --------------------------------------------------------------------------------------------


#: Seconds from ``synth.t``'s base (2026-01-01) to a day after the run-set drop guard landed on main (``H.RUN_SET_DROP_UNDECLARED_SINCE``,
#: 2026-09-23T18:30:43Z): the one test that expects the guard's hold stamps its revisions, bundle and cycles past the gate.
_SINCE_SHIFT = (R.utc_to_epoch_ms("2026-09-24T00:00:00.000Z") - R.utc_to_epoch_ms("2026-01-01T00:00:00.000Z")) // 1000


def _s(i):
    return i + _SINCE_SHIFT


def _t(i):
    return synth.t(_s(i))


def _held_marker(root, rev):
    return _q(root, "SELECT json_extract(extra_json, '$.kind') AS kind, stale FROM blocked_records WHERE record_type = 'source_revision' "
                    "AND json_extract(extra_json, '$.revision_id') = ? ORDER BY record_uri", rev)


def test_an_operators_release_of_a_run_drop_hold_makes_the_staged_correction_pending_before_the_release_cycle(tmp_path, capsys):
    """Accepted revision A carried runs R and S on X; the newer head B carries R and S on X2 plus T. A re-activation of A
    HOLDS on the run-set-drop guard (T dropped undeclared): A's marker stays `accepted = 1`, the bundle's `run` file is staged under A. A
    `run_drop` hold folds nothing by itself (probe empty, a claim binds X2). `registry revision accept` publishes a MARKER-ONLY release
    bundle: the next cycle folds A's staged rows -- the probe names the release bundle and the staged correction, the claim refuses with
    nothing written; after the cycle R is back on X and the claim binds X. The revisions, the bundle and the cycles are stamped after the
    guard landed on main (``_s`` / ``_t``: the run-set drop guard is versioned by ``H.RUN_SET_DROP_UNDECLARED_SINCE`` and judges only a
    marker observed from that stamp on)."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    sid, s_recs = synth.run_records("s", 300, through="generated", n_items=4)
    tid, t_recs = synth.run_records("t", 400, through="generated", n_items=4)
    s_run, t_run = s_recs[0]["payload"], t_recs[0]["payload"]
    def on(payload, instrument_id, i):
        q = dict(payload, instrument_id=instrument_id)
        return synth.rec("run", q, synth.t(i))
    _revision_dir(root, "synthad", REV, [_correction(rid, synth.INSTRUMENT_ID, _s(2000)), on(s_run, synth.INSTRUMENT_ID, _s(2001))], _s(2000))     # A
    assert _ingest(root, _t(2100)).refused == 0
    _revision_dir(root, "synthad", REV_B, [_correction(rid, INSTRUMENT_X2_ID, _s(2200)), on(s_run, INSTRUMENT_X2_ID, _s(2201)), on(t_run, INSTRUMENT_X2_ID, _s(2202))], _s(2200))   # B
    assert _ingest(root, _t(2300)).refused == 0
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID
    assert _q(root, "SELECT COUNT(*) AS n FROM observation WHERE revision_id = ?", REV_B)[0]["n"] == 3                 # the head names R, S, T
    # a re-activation of A (R and S on X, T dropped, no declaration): held on the run-set-drop guard, A still accepted, its rows staged
    _bundle_dir(root, "synthad", REV, "20260924010000000", [_correction(rid, synth.INSTRUMENT_ID, _s(3000)), on(s_run, synth.INSTRUMENT_ID, _s(3001))], _s(3000))
    assert _ingest(root, _t(3100)).refused == 0
    assert _held_marker(root, REV) == [{"kind": H.RUN_SET_DROP_KIND, "stale": 0}], _held_marker(root, REV)
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 1}]
    assert _q(root, "SELECT COUNT(*) AS n FROM revision_staging_record WHERE revision_id = ? AND file_uri LIKE '%_activations%'", REV)[0]["n"] >= 1
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID and _probe(root, rid) == []           # a run_drop hold folds nothing next cycle
    c0 = P.claim(root, rid, synth.JUDGE_ID, "s-0", now=_t(3200))
    assert P.release(root, rid, synth.JUDGE_ID, "s-0", now=_t(3201))
    files_before, rows_before = _claim_files(root, rid), _judgment_rows(root, rid)
    # the operator releases the hold: a marker-only bundle the next cycle folds with A's staged rows
    rc = cli.main(["--root", root, "revision", "accept", "--revision-id", REV, "--reason", "T retired on purpose"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    release = out.out.strip()
    assert "/_activations/" in release and release.endswith("-source_revision.jsonl") and os.path.isfile(release)
    why = _probe(root, rid)
    assert len(why) == 1 and why[0].startswith(f"release bundle synthad/_activations/{REV}/") and "folds staged" in why[0] and synth.INSTRUMENT_ID in why[0], why
    with pytest.raises(P.ClaimPendingCorrection, match="release bundle"):
        P.claim(root, rid, synth.JUDGE_ID, "s-A", now=_t(3300))
    assert _claim_files(root, rid) == files_before and _judgment_rows(root, rid) == rows_before and _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID
    # the release cycle: A activates again as the head, R is back on X, the probe is empty, the claim binds X
    for i in range(3):
        assert _ingest(root, _t(3500 + i)).refused == 0
        if _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID:
            break
    assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID, _held_marker(root, REV)
    assert _probe(root, rid) == []
    c1 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=_t(4000))
    assert _judgment_rows(root, rid)[c1]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    _rebuild_equal(root, tmp_path)


def test_an_exact_refused_hold_is_never_pending_until_a_repair_makes_it_retryable(tmp_path):
    """Finding 2 of the re-check: a revision stages a valid `run` correction to X2 and a second file refuses (a malformed metric). Its
    marker is `accepted = 0`, not superseded, on disk -- yet the harvester's `_refused_hold_exact` rule skips it every cycle while the bytes
    and the validator stamp stand (`harvest.refused_hold_exact`), so the staged correction is PARKED: three cycles without repair, the probe
    empty each time, the claim binds the folded identity. The route out is a new revision of the source: an unprocessed directory the
    next cycle folds -- pending, the claim refuses; the cycle activates it and R reads X2."""
    root, rid = _generated(tmp_path, extra_dims=[synth.rec("instrument", INSTRUMENT_X2, synth.t(31))])
    d = _revision_dir(root, "synthad", REV, [_correction(rid, INSTRUMENT_X2_ID, 2000)], 2000)
    bad = synth.rec("run_metric", {"run_id": rid, "judgment_id": "nope", "metric_id": "asr", "threshold": 0.5, "value": "not-a-number", "computed_by": "producer"}, synth.t(2001))
    R.write_records(os.path.join(d, "000003-run_metric.jsonl"), [bad])
    rep = _ingest(root, synth.t(2100))
    assert rep.refused >= 1
    assert _held_marker(root, REV) == [{"kind": "refused", "stale": 0}], _held_marker(root, REV)
    assert _q(root, "SELECT accepted, superseded_by_revision_id AS s FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 0, "s": None}]
    assert _q(root, "SELECT COUNT(*) AS n FROM revision_staging_record WHERE revision_id = ?", REV)[0]["n"] >= 1       # the valid correction is staged
    for i in range(3):                                                                               # cycles without repair: nothing changes
        assert _probe(root, rid) == []
        assert _ingest(root, synth.t(2500 + i)).refused == 0
        assert _run_identity(root, rid)["i"] == synth.INSTRUMENT_ID and _held_marker(root, REV) == [{"kind": "refused", "stale": 0}]
    c0 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    assert _judgment_rows(root, rid)[c0]["claimed_instrument_id"] == synth.INSTRUMENT_ID
    # the route out is a NEW revision of the source carrying the correction (the record log is append-only: nobody edits the refused
    # file in place): an unprocessed revision directory the next cycle folds -> pending, the claim refuses; the cycle activates it, A is
    # superseded by it and R reads X2
    _revision_dir(root, "synthad", REV_B, [_correction(rid, INSTRUMENT_X2_ID, 3200)], 3200)
    why = _probe(root, rid)
    assert why and all(w.startswith(f"unprocessed revision {REV_B}") and INSTRUMENT_X2_ID in w for w in why), why
    with pytest.raises(P.ClaimPendingCorrection, match="unprocessed revision"):
        P.claim(root, rid, synth.JUDGE_ID, "s-B", now=synth.t(3300))
    assert _ingest(root, synth.t(3500)).refused == 0
    assert _run_identity(root, rid)["i"] == INSTRUMENT_X2_ID and _probe(root, rid) == []
    assert _q(root, "SELECT accepted, superseded_by_revision_id AS s FROM source_revision WHERE revision_id = ?", REV) == [{"accepted": 0, "s": REV_B}]
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# The one-re-grade guarantee is scoped to instrument-judged runs; the judge-less J1 -> J2 case is pinned, not hidden
# --------------------------------------------------------------------------------------------


def test_the_one_regrade_guarantee_is_scoped_to_instrument_judged_runs(tmp_path, monkeypatch):
    """A judge-less instrument's run is judged by its fallback `run.judge_id` = J1. A cluster-pushed `run_patch` moves it to J2; the probe
    is FORCED to miss; J1's claim binds (RULE_INSTRUMENT, J1) and the judgment completes under it; one real cycle merges and folds the patch.
    Then: J1's judgment is identity-stale (out of the frame, covering nothing); the run stays VISIBLE in `v_pending_judgment` under J1 with
    `pending_reason = identity_stale` and `is_instrument_judge = 0` -- the grade daemon's default selection (`is_instrument_judge = 1`)
    never takes it; and the view does NOT list the run for J2 (`P.pending(conn, J2)` is empty: a judge joins a judge-less instrument's run
    only through a judgment it already has) -- the pending view's known limitation. No automatic re-grade restores the measurement; the
    PRODUCER re-judges with J2, and that claim binds (RULE_INSTRUMENT, J2). This is what limits the one-re-grade guarantee to
    instrument-judged runs."""
    root, rid = _rule_scored_run(tmp_path)
    run_dir = I.run_dir_name(rid)
    R.write_records(os.path.join(root, "inbox", "cluster_a", "c", run_dir, "000002-run_patch.jsonl"), [_judge_patch(rid, synth.JUDGE_ID_2, 2000)])
    assert _probe(root, rid)                                                                         # the probe WOULD see the inbox patch
    monkeypatch.setattr(P, "pending_identity_corrections", lambda *a, **k: [])                       # the forced miss
    c0 = P.claim(root, rid, synth.JUDGE_ID, "s-A", now=synth.t(3000))
    monkeypatch.undo()
    assert (_judgment_rows(root, rid)[c0]["claimed_instrument_id"], _judgment_rows(root, rid)[c0]["claimed_required_judge_id"]) == (synth.RULE_INSTRUMENT_ID, synth.JUDGE_ID)
    _complete(root, rid, c0, synth.JUDGE_ID, "s-A", 3000, 3100)                                       # J1 grades under the old fallback
    assert _ingest(root, synth.t(3500)).refused == 0                                                  # the patch and the completion fold
    assert _run_identity(root, rid) == {"i": synth.RULE_INSTRUMENT_ID, "j": synth.JUDGE_ID_2}
    assert _judgment_rows(root, rid)[c0]["status"] == "complete"
    assert all(j != c0 for j, _m in _frame(root, rid))                                               # identity-stale: J1's rows out of the frame (the rule metric stays)
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM judgment j JOIN run r ON r.run_id = j.run_id LEFT JOIN instrument i ON i.instrument_id = r.instrument_id "
                            f"WHERE j.judgment_id = ? AND {S.judgment_identity_current_sql('j', 'r', 'i')}", (c0,)).fetchone()[0] == 0
        # (a rule-scored run is `ingested` on its rule metric whatever its judgments, so the completion requirement reads `already
        # ingested` here, not `identity_stale`: the identity test above is the pin)
        # the run is listed under J1 (identity_stale, not an instrument judge) and NOT under J2: the view's judge join for a judge-less
        # instrument is a judge with a judgment on the run, and J2 has none
        j1 = _worklist_row(root, rid, synth.JUDGE_ID)
        assert j1 is not None and j1["pending_reason"] == "identity_stale" and j1["is_instrument_judge"] == 0, j1
        assert _worklist_row(root, rid, synth.JUDGE_ID_2) is None
        assert [r for r in P.pending(conn, synth.JUDGE_ID_2) if r["run_id"] == rid] == []
        assert [r["pending_reason"] for r in P.pending(conn, synth.JUDGE_ID) if r["run_id"] == rid] == ["identity_stale"]
    # no automatic re-grade: the producer re-judges with J2, and that claim binds the new fallback
    c1 = P.claim(root, rid, synth.JUDGE_ID_2, "s-producer", now=synth.t(4000))
    rows = _judgment_rows(root, rid)
    assert c1 != c0 and (rows[c1]["claimed_instrument_id"], rows[c1]["claimed_required_judge_id"]) == (synth.RULE_INSTRUMENT_ID, synth.JUDGE_ID_2)
    _rebuild_equal(root, tmp_path)
