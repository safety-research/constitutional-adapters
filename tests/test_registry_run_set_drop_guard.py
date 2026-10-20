"""The run-set drop guard of the harvester (`harvest.Ingestor._undeclared_run_drop`): a `source_revision` marker that would
move a source's head while no longer naming run ids the accepted head carries, none of them declared, is PUBLISHED BUT NOT
ACCEPTED -- the existing held state (`source_revision.accepted = 0`, `refusal` = `activation pending: source_revision <head>
(run_set_drop_undeclared: ...)` naming the source, the counts and up to eight dropped run ids; the marker in `blocked_records` with
`kind = run_drop`) -- and the head stays current, so the paper frame does not move. A revision declares a drop through the marker's
`extra.declared_run_drops` (`{"run_ids": [ids or globs] | "*", "reason", "declared_by"}`), which the runner writes from the adapter's
`runs_retired`, for the ids that left with a declared retired input (`inputs_retired`), and for a vanished group (`input_vanished`);
the operator releases a held one with `registry revision accept --revision-id X --reason ...`, an append-only bundle record.

The live case: a producer reshaped `records/launch_sets.json` under the live incremental readers; the readers
validated rows against one branch of the new shape, refused 25 of the 30 rows per source, and the pass published ACCEPTED revisions
with 5 runs each (the s1 groups: 10 -> 0; the s2 / s3 groups: 7 -> 3 and 8 -> 2 with every previous id replaced by a new one), 50 paper
rows withdrawn with `accepted: 1` and no error; a later pass published three EMPTY revisions. Nothing compared the run set against the
head. The first test is that shape, at 6 -> 2 and 6 -> 0. The rule is a NET SHRINK (`runs_now < runs_previous`, decided 2026-09-23): the
s2 / s3 shape -- every id replaced AND fewer of them -- holds (the second test); a re-keying that replaces N ids with N others does not
shrink and publishes, the framework's standing contract (the third test).

What must NOT fire: a first read, a superset, the same run ids re-emitted under a new revision id, a declared drop (`runs_retired`),
the drops that follow a declared `inputs_retired` input, a vanished group's empty revision. Fixtures are the synthetic store adapter of
test_registry_core_fixes with a per-row refusal switch; nothing here reads a real store.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB
from sisteer.registry.adapters import runner as RUN

from test_registry_core_fixes import StoreAdapter, _adapter_setup, _ingest, _q, _rebuild_equal, _store

#: Every read of these worlds is stamped AFTER the guard landed on main (``H.RUN_SET_DROP_UNDECLARED_SINCE``, 2026-09-23T18:30:43Z): the
#: rule is VERSIONED and judges only a marker observed from that stamp on, so a world
#: stamped at ``synth.t``'s January base would fold every shrink the way the pre-rule harvester did (the last tests below say so).
_SINCE_BASE = "2026-09-24T00:00:00.000Z"


def _t(i):
    return synth.t(i, base=_SINCE_BASE)


GROUP = "one_group"
SOURCE = RUN.source_id_for("synthetic_store", GROUP)


class RefusingStoreAdapter(StoreAdapter):
    """The synthetic store adapter with the 2026-09-23 defect on a switch: every store whose key is in `refuse_keys` is refused
    row by row (read, then dropped -- the input is still an input, so the inputs-dropped guard has nothing to say), and
    `EMISSION_VERSION` mints a new revision id over the same inputs, the way the live readers' version bumps do."""

    def __init__(self, store_root: str):
        super().__init__(store_root, group_of=lambda inp: GROUP)
        self.refuse_keys: set = set()
        self.EMISSION_VERSION = 1

    def read(self, ctx, group, inputs):
        kept = []
        for inp in inputs:
            with open(inp.path) as fh:
                if json.load(fh)["key"] in self.refuse_keys:
                    continue
            kept.append(inp)
        yield from super().read(ctx, group, kept)


def _stores(n, start=0):
    return [(f"s{i:03d}", f"store_{i:03d}", "none", [0.25 + (i % 3) * 0.25]) for i in range(start, start + n)]


def _rid(i):
    return I.run_id(synth.CAMPAIGN, f"store_{i:03d}", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)


def _setup(tmp_path, n=6):
    root, store_root = _adapter_setup(tmp_path, _stores(n))
    return root, store_root, RefusingStoreAdapter(store_root)


def _pass(root, ad, now):
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=now)
    return rep, _ingest(root)


def _head(root):
    rows = _q(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 AND superseded_by_revision_id IS NULL", SOURCE)
    assert len(rows) == 1, rows
    return rows[0][0]


def _paper_rows(root):
    return _q(root, "SELECT COUNT(*) FROM v_paper_long WHERE lane_source_legacy = 'synth_source'")[0][0]


def _marker(root, rev):
    return _q(root, "SELECT accepted, refusal, superseded_by_revision_id, extra_json FROM source_revision WHERE revision_id = ?", rev)[0]


def _extra(root, rev):
    return json.loads(_marker(root, rev)[3])


def _blocked_marker(root, rev):
    return _q(root, "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind'), stale FROM blocked_records "
                    "WHERE record_type = 'source_revision' AND json_extract(extra_json, '$.revision_id') = ? ORDER BY record_uri", rev)


# ============================================================================================
# the pure comparison
# ============================================================================================

def test_the_comparison_trips_on_an_undeclared_net_shrink_and_on_nothing_else():
    prev = {"r1", "r2", "r3", "r4", "r5", "r6"}
    strict = H.run_set_drop_comparison(prev, {"r1", "r2"}, None)
    assert (strict["runs_previous"], strict["runs_now"], strict["runs_dropped"], strict["runs_added"]) == (6, 2, 4, 0)
    assert strict["runs_dropped_undeclared"] == 4 and strict["runs_dropped_undeclared_sample"] == ["r3", "r4", "r5", "r6"] and strict["tripped"]
    empty = H.run_set_drop_comparison(prev, set(), None)
    assert empty["runs_dropped"] == 6 and empty["tripped"]
    # a NET shrink with additions trips: the 2026-09-23 s2 / s3 shape (every previous id replaced, fewer of them)
    mixed = H.run_set_drop_comparison({"a", "b", "c"}, {"x", "y"}, None)
    assert (mixed["runs_dropped"], mixed["runs_added"], mixed["runs_dropped_undeclared"], mixed["tripped"]) == (3, 2, 3, True)
    # a re-keying (N -> N) or a growth with drops (N -> N+k) does not shrink and does not trip; the counts are on the record
    same = H.run_set_drop_comparison({"a", "b", "c"}, {"x", "y", "z"}, None)
    assert (same["runs_dropped"], same["runs_added"], same["tripped"]) == (3, 3, False)
    assert not H.run_set_drop_comparison(prev, prev - {"r1"} | {"r9"}, None)["tripped"]
    assert not H.run_set_drop_comparison(prev, prev - {"r1"} | {"r9", "r10"}, None)["tripped"]
    assert H.run_set_drop_comparison(prev, prev - {"r1", "r2"} | {"r9"}, None)["tripped"]
    assert not H.run_set_drop_comparison(prev, prev, None)["tripped"]
    assert not H.run_set_drop_comparison(prev, prev | {"r7"}, None)["tripped"]
    assert not H.run_set_drop_comparison(set(), {"r1"}, None)["tripped"]
    # declarations: exact ids, globs, "*"; a partial declaration leaves the rest undeclared
    assert not H.run_set_drop_comparison(prev, {"r1", "r2"}, {"run_ids": ["r3", "r4", "r5", "r6"], "reason": "x"})["tripped"]
    assert not H.run_set_drop_comparison(prev, {"r1", "r2"}, {"run_ids": ["r[3-6]"], "reason": "x"})["tripped"]
    assert not H.run_set_drop_comparison(prev, set(), {"run_ids": "*", "reason": "x"})["tripped"]
    partial = H.run_set_drop_comparison(prev, {"r1", "r2"}, {"run_ids": ["r3", "r4"], "reason": "x"})
    assert (partial["runs_dropped_declared"], partial["runs_dropped_undeclared"], partial["runs_dropped_undeclared_sample"]) == (2, 2, ["r5", "r6"])
    # the sample is bounded; the count is exact
    big = H.run_set_drop_comparison({f"r{i:02d}" for i in range(30)}, set(), None)
    assert big["runs_dropped_undeclared"] == 30 and len(big["runs_dropped_undeclared_sample"]) == H.RUN_SET_DROP_NAMED
    # the declaration's shape is checked; a malformed one declares nothing and says why
    assert H.declared_run_drops_of(None) == (None, None) and H.declared_run_drops_of({}) == (None, None)
    ok, why = H.declared_run_drops_of({"declared_run_drops": {"run_ids": "*", "reason": "because"}})
    assert ok == {"run_ids": "*", "reason": "because"} and why is None
    for bad in ("x", {"run_ids": "r1", "reason": "x"}, {"run_ids": [], "reason": "x"}, {"run_ids": ["r1", ""], "reason": "x"},
                {"run_ids": ["r1"]}, {"run_ids": ["r1"], "reason": "  "}):
        decl, why = H.declared_run_drops_of({"declared_run_drops": bad})
        assert decl is None and why and why.startswith("declared_run_drops")
    # the ids a set of observation keys speaks of: every run-scale key carrying a run_id, once
    keys = [I.canonical_json({"run_id": "a"}), I.canonical_json({"run_id": "a", "judgment_id": "j", "metric_id": "m", "threshold": 0.5, "computation_id": ""}),
            I.canonical_json({"instrument_id": "i"}), {"run_id": "b"}, "not json", I.canonical_json({"run_id": ""})]
    assert H.run_ids_of_observation_keys(keys) == {"a", "b"}


# ============================================================================================
# the live shape: 6 accepted; 2 undeclared -> held; 0 -> held; the same 6 -> accepted; declared -> withdrawn; the operator
# ============================================================================================

def test_the_2026_09_23_shape_is_held_the_frame_stands_and_a_declaration_or_the_operator_releases_it(tmp_path, capsys):
    root, store_root, ad = _setup(tmp_path, 6)
    rep1, ing1 = _pass(root, ad, _t(2000))
    assert rep1.groups_refreshed == 1 and ing1.refused == 0 and ing1.blocked == 0 and ing1.revisions_held == 0
    rev1 = _head(root)
    assert _paper_rows(root) == 6
    assert "run_set" not in _extra(root, rev1) and H.DECLARED_RUN_DROPS_KEY not in _extra(root, rev1)     # a first read has no head to compare with
    # the defect: the reader refuses four of the six rows over the same inputs, under a new version -- the runner publishes an
    # accepted revision (it cannot know), the pass reports it refreshed, and the HARVESTER holds it
    ad.refuse_keys = {f"store_{i:03d}" for i in range(2, 6)}
    ad.EMISSION_VERSION = 2
    rep2, ing2 = _pass(root, ad, _t(3000))
    assert rep2.groups_refreshed == 1 and rep2.groups_refused == 0 and rep2.withdrawals > 0
    rev2 = rep2.results[0].revision_id
    assert rev2 != rev1
    # one activation attempt, rolled back; the cycle's closing re-evaluation then skips the hold:
    # the held count is once per cycle under the attempt memo (registry follow-ups-4 item 10): the cycle's own blocked-queue pass finds
    # the hold's attempt identity standing and skips the re-attempt (before item 10 it re-attempted and counted the same hold again: 2)
    assert ing2.refused == 0 and ing2.revisions_held_run_drop == 1 and ing2.revision_transactions == 1 and ing2.revisions_held == 1 and ing2.revisions_held_skipped == 1
    accepted, refusal, superseded, _ = _marker(root, rev2)
    assert accepted == 0 and superseded is None
    dropped = sorted(_rid(i) for i in range(2, 6))
    assert refusal.startswith(f"activation pending: source_revision {rev1} ({H.RUN_SET_DROP_KEY}: {SOURCE} no longer names 4 of the 6 run ids of the accepted head {rev1} and declares none of them; the run set shrank 6 -> 2 (0 added); the head stays current. Dropped (4 of 4 named): ")
    assert ", ".join(dropped) in refusal and f"registry revision accept --revision-id {rev2}" in refusal and "runs_retired" in refusal
    assert _blocked_marker(root, rev2) == [("source_revision", rev1, H.RUN_SET_DROP_KIND, 0)]
    assert _q(root, "SELECT json_extract(extra_json, '$.rule') FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ?", rev2) == [(H.RUN_SET_DROP_RULE,)]
    # the frame did not move: the head, the paper rows, no withdrawal, every observation of the head current; the runner's record of
    # what it saw is on the marker (run_set), with no declaration
    assert _head(root) == rev1 and _paper_rows(root) == 6
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ?", SOURCE) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND revision_id = ? AND current = 1 AND table_name = 'run'", SOURCE, rev1) == [(6,)]
    assert _q(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND revision_id = ?", SOURCE, rev2) == [(0,)]
    ex2 = _extra(root, rev2)
    assert ex2 == {} or H.DECLARED_RUN_DROPS_KEY not in ex2       # the placeholder row carries no extra: the marker never folded
    # the status the paper reads: not ok, the hold named, the accepted read current
    ok, detail = _q(root, "SELECT ok, detail FROM v_source_status WHERE adapter = 'synthetic_store'")[0]
    assert ok == 0 and detail.startswith(f"latest read REFUSED (revision {rev2} at ") and H.RUN_SET_DROP_KEY in detail and f"the accepted read {rev1} stays current" in detail
    # the next cycles: the hold is exact while the head stands, so nothing is re-attempted (no activation transaction) and the
    # incremental pass leaves the published directory alone, naming the hold
    ing2b = _ingest(root)
    assert ing2b.revisions_held == 1 and ing2b.revision_transactions == 0 and _head(root) == rev1 and _paper_rows(root) == 6
    with DB.reading(DB.db_path(root)) as conn:
        rep2b = RUN.run_pass(root, [ad], conn=conn, now=_t(3100))
    assert rep2b.groups_held == 1 and rep2b.groups_refreshed == 0 and rep2b.results[0].held
    assert "held by the harvester" in rep2b.results[0].error and H.RUN_SET_DROP_KEY in rep2b.results[0].error
    assert not os.path.isdir(os.path.join(root, "records", "adapters", "synthetic_store", "_activations", rev2))   # nothing re-emitted
    # the 16:06Z shape: the reader refuses EVERY row -- an empty revision, held the same way; the frame still whole
    ad.refuse_keys = {f"store_{i:03d}" for i in range(6)}
    ad.EMISSION_VERSION = 3
    rep3, ing3 = _pass(root, ad, _t(4000))
    rev3 = rep3.results[0].revision_id
    assert rep3.groups_refreshed == 1 and ing3.revisions_held_run_drop == 1 and ing3.revision_transactions == 1     # rev2 skipped (still held), rev3 attempted once
    accepted3, refusal3, _, _ = _marker(root, rev3)
    assert accepted3 == 0 and f"{H.RUN_SET_DROP_KEY}: {SOURCE} no longer names 6 of the 6 run ids of the accepted head {rev1}" in refusal3
    assert "the run set shrank 6 -> 0 (0 added)" in refusal3 and "(+0 more)" not in refusal3
    assert _head(root) == rev1 and _paper_rows(root) == 6 and _blocked_marker(root, rev3)[0][2] == H.RUN_SET_DROP_KIND
    assert [r[0] for r in _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 0 ORDER BY observed_utc")] == [rev2, rev3]
    # the corrected reader: the same six run ids under a new revision id -- nothing dropped, accepted at once; the held markers are
    # superseded (a later accepted read) and their queue rows go stale
    ad.refuse_keys = set()
    ad.EMISSION_VERSION = 4
    rep4, ing4 = _pass(root, ad, _t(5000))
    rev4 = rep4.results[0].revision_id
    # the two held revisions are superseded by rev4 and never re-attempted (no activation, no re-comparison): the hold's provenance
    # -- the refusal naming rev1, the queue rows -- stands, and nothing of theirs folds
    assert rep4.groups_refreshed == 1 and rep4.withdrawals == 0 and ing4.refused == 0 and ing4.revisions_held_run_drop == 0 and ing4.revision_transactions == 1
    assert _head(root) == rev4 and _paper_rows(root) == 6
    assert f"of the accepted head {rev1}" in _marker(root, rev2)[1] and _marker(root, rev2)[0] == 0
    ex4 = _extra(root, rev4)
    assert ex4["run_set"] == {"previous_revision_id": rev1, "runs_previous": 6, "runs_now": 6, "runs_dropped": 0, "runs_added": 0}
    assert H.DECLARED_RUN_DROPS_KEY not in ex4
    assert _q(root, "SELECT revision_id, superseded_by_revision_id FROM source_revision WHERE accepted = 0 ORDER BY observed_utc") == [(rev2, rev4), (rev3, rev4)]
    assert all(r[3] == 1 for r in _blocked_marker(root, rev2) + _blocked_marker(root, rev3))
    # a DECLARED drop: the reader retires four runs on purpose and says so (runs_retired) -- accepted, the four withdrawn at the switch
    ad.refuse_keys = {f"store_{i:03d}" for i in range(2, 6)}
    ad.EMISSION_VERSION = 5
    ad.runs_retired = {"run_ids": dropped, "reason": "the four stores measure a retired condition (synthetic)"}
    rep5, ing5 = _pass(root, ad, _t(6000))
    rev5 = rep5.results[0].revision_id
    assert rep5.groups_refreshed == 1 and rep5.withdrawals > 0 and ing5.refused == 0 and ing5.revisions_held_run_drop == 0 and ing5.revision_transactions == 1
    assert _head(root) == rev5 and _paper_rows(root) == 2
    ex5 = _extra(root, rev5)
    assert ex5["run_set"] == {"previous_revision_id": rev4, "runs_previous": 6, "runs_now": 2, "runs_dropped": 4, "runs_added": 0}
    assert ex5[H.DECLARED_RUN_DROPS_KEY] == {"run_ids": dropped, "reason": "the four stores measure a retired condition (synthetic)",
                                             "declared_by": ["adapter:runs_retired"], "runs_dropped_at_read": 4}
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ? AND from_revision_id = ? AND to_revision_id = ? AND table_name = 'run'",
              SOURCE, rev4, rev5) == [(4,)]
    assert _q(root, "SELECT COUNT(*) FROM v_paper_current WHERE selection_state = 'withdrawn'") == [(4,)]
    # a declaration that covers only part of the drop holds on the rest, by name
    ad.refuse_keys = {f"store_{i:03d}" for i in range(0, 6)}
    ad.EMISSION_VERSION = 6
    ad.runs_retired = {"run_ids": [_rid(0)], "reason": "one more retired (synthetic)"}
    rep6, ing6 = _pass(root, ad, _t(7000))
    rev6 = rep6.results[0].revision_id
    assert ing6.revisions_held_run_drop == 1 and _head(root) == rev5 and _paper_rows(root) == 2
    accepted6, refusal6, _, _ = _marker(root, rev6)
    assert accepted6 == 0 and f"no longer names 1 of the 2 run ids of the accepted head {rev5}" in refusal6 and "1 dropped and declared" in refusal6
    assert f"Dropped (1 of 1 named): {_rid(1)}" in refusal6
    # the OPERATOR releases it: a declaration bundle, an append-only record; the next cycle folds it, the head moves, provenance on the row
    assert cli.main(["--root", root, "revision", "held"]) == 0
    held_out = capsys.readouterr().out
    assert rev6 in held_out and rev2 not in held_out                       # superseded holds are not listed without --all
    assert cli.main(["--root", root, "revision", "held", "--all"]) == 0 and rev2 in capsys.readouterr().out
    assert cli.main(["--root", root, "revision", "accept", "--revision-id", rev6, "--reason", "the last store is retired too (synthetic)",
                     "--accepted-by", "the test"]) == 0
    bundle_path = capsys.readouterr().out.strip()
    assert bundle_path.startswith(os.path.join(root, "records", "adapters", "synthetic_store", "_activations", rev6) + os.sep)
    assert bundle_path.endswith("000001-source_revision.jsonl") and os.path.isfile(bundle_path)
    assert _head(root) == rev5 and _paper_rows(root) == 2                 # nothing moved before the fold
    ing7 = _ingest(root)
    assert ing7.refused == 0 and ing7.revisions_held_run_drop == 0 and ing7.revision_transactions >= 1
    assert _head(root) == rev6 and _paper_rows(root) == 0
    accepted7, refusal7, _, _ = _marker(root, rev6)
    assert accepted7 == 1 and refusal7 is None
    ex6 = _extra(root, rev6)
    assert ex6[H.DECLARED_RUN_DROPS_KEY] == {"run_ids": "*", "reason": "the last store is retired too (synthetic)",
                                             "declared_by": ["operator:registry revision accept"], "accepted_by": "the test"}
    held_reason = ex6["operator_accept"]["held_reason"]                # the queue row's reason: the detail inside the marker's `activation pending: ... (...)`
    assert held_reason.startswith(f"{H.RUN_SET_DROP_KEY}: {SOURCE} no longer names 1 of the 2 run ids") and refusal6.endswith(f"({held_reason})")
    assert ex6["operator_accept"]["of_marker"].endswith("-source_revision.jsonl")
    assert ex6["run_set"]["runs_dropped"] == 2                            # the runner's record of the read rides along
    assert _q(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND revision_id = ? AND current = 1", SOURCE, rev6)[0][0] == 0   # an empty revision
    assert _q(root, "SELECT COUNT(*) FROM observation WHERE source_id = ? AND revision_id = ? AND current = 1", SOURCE, rev5) == [(0,)]
    assert _q(root, "SELECT COUNT(*) FROM v_paper_current WHERE selection_state = 'withdrawn'") == [(6,)]
    assert _blocked_marker(root, rev6) == []                               # the original directory's queue rows left with the activation
    # the operator command refuses what a declaration cannot release
    assert cli.main(["--root", root, "revision", "accept", "--revision-id", rev6, "--reason", "again"]) == 1
    assert "holds no marker" in capsys.readouterr().err   # accepted: nothing held
    assert cli.main(["--root", root, "revision", "accept", "--revision-id", rev2, "--reason", "old"]) == 1
    assert "superseded by the later accepted revision" in capsys.readouterr().err
    assert cli.main(["--root", root, "revision", "accept", "--revision-id", "0" * 16, "--reason", "x"]) == 1
    assert "no source_revision row" in capsys.readouterr().err
    assert cli.main(["--root", root, "revision", "accept", "--revision-id", rev6]) == 2
    # live equals replay: the held directories hold again on replay, the bundle activates
    _rebuild_equal(root, tmp_path)


def test_a_revision_that_replaces_every_run_id_with_fewer_is_a_net_shrink_and_holds(tmp_path):
    """The 2026-09-23 s2 / s3 shape: the surviving rows were NEW run ids (the conditions that had grown the record), every previous id
    gone -- 3 -> 2 here (7 -> 3 live) with 3 dropped and 2 added. Fewer runs than the head with undeclared drops: held; the counts on
    the marker's `run_set` and in the reason."""
    root, store_root, ad = _setup(tmp_path, 3)
    _pass(root, ad, _t(2000))
    rev1 = _head(root)
    assert _paper_rows(root) == 3
    for name, key, obj, scores in _stores(2, start=3):                     # two new stores arrive; the reader refuses the three old ones
        _store(store_root, name, key, obj, scores)
    ad.refuse_keys = {f"store_{i:03d}" for i in range(3)}
    rep, ing = _pass(root, ad, _t(3000))
    rev2 = rep.results[0].revision_id
    assert rep.groups_refreshed == 1 and rep.groups_inputs_dropped == 0 and rep.withdrawals >= 3 and ing.revisions_held_run_drop == 1
    assert _head(root) == rev1 and _paper_rows(root) == 3
    accepted, refusal, _, _ = _marker(root, rev2)
    assert accepted == 0 and f"no longer names 3 of the 3 run ids of the accepted head {rev1}" in refusal
    assert "the run set shrank 3 -> 2 (2 added)" in refusal
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE to_revision_id = ?", rev2) == [(0,)]        # nothing of it folded
    # the reader corrected: five runs, a superset of the head -- accepted, the held marker superseded
    ad.refuse_keys = set()
    ad.EMISSION_VERSION = 2
    rep3, ing3 = _pass(root, ad, _t(4000))
    assert ing3.revisions_held_run_drop == 0 and _head(root) == rep3.results[0].revision_id and _paper_rows(root) == 5
    assert _extra(root, _head(root))["run_set"] == {"previous_revision_id": rev1, "runs_previous": 3, "runs_now": 5, "runs_dropped": 0, "runs_added": 2}
    assert _marker(root, rev2)[2] == _head(root)


def test_a_re_keying_that_replaces_every_run_id_with_as_many_does_not_shrink_and_publishes(tmp_path):
    """N -> N with every id replaced (the 2026-09-20 `overrefusal_items` pass that re-keyed every run; a store replaced under the same
    group, `test_removed_measurement_is_withdrawn`): the source names as many runs as it did, so nothing shrank -- the revision
    publishes, the old keys withdraw as `removed_measurement`, and the marker's `run_set` records the N dropped and N added."""
    root, store_root, ad = _setup(tmp_path, 3)
    _pass(root, ad, _t(2000))
    rev1 = _head(root)
    for name, key, obj, scores in _stores(3, start=3):                     # three new stores arrive; the reader refuses the three old ones
        _store(store_root, name, key, obj, scores)
    ad.refuse_keys = {f"store_{i:03d}" for i in range(3)}
    rep, ing = _pass(root, ad, _t(3000))
    rev2 = rep.results[0].revision_id
    assert rep.groups_refreshed == 1 and rep.withdrawals >= 3 and ing.revisions_held_run_drop == 0 and ing.refused == 0
    assert _head(root) == rev2 and _paper_rows(root) == 3
    assert _extra(root, rev2)["run_set"] == {"previous_revision_id": rev1, "runs_previous": 3, "runs_now": 3, "runs_dropped": 3, "runs_added": 3}
    assert H.DECLARED_RUN_DROPS_KEY not in _extra(root, rev2)
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE to_revision_id = ? AND table_name = 'run' AND reason = 'removed_measurement'", rev2) == [(3,)]
    assert _q(root, "SELECT COUNT(*) FROM v_paper_current WHERE selection_state = 'withdrawn'") == [(3,)]


# ============================================================================================
# the legitimate shrinks still publish, declared by the runner
# ============================================================================================

def test_a_declared_input_retirement_and_a_vanished_group_declare_their_run_drops_and_publish(tmp_path):
    root, store_root, ad = _setup(tmp_path, 4)
    _pass(root, ad, _t(2000))
    rev1 = _head(root)
    # two of four stores leave discovery, declared retired: the read publishes and the runner declares the two run ids that left
    gone = [os.path.join(store_root, f"s{i:03d}", "store.json") for i in (2, 3)]
    for p in gone:
        os.remove(p)
    ad.inputs_retired = gone
    rep2, ing2 = _pass(root, ad, _t(3000))
    rev2 = rep2.results[0].revision_id
    assert rep2.groups_refreshed == 1 and rep2.groups_inputs_dropped == 0 and ing2.revisions_held == 0 and ing2.refused == 0
    assert _head(root) == rev2 and _paper_rows(root) == 2
    ex2 = _extra(root, rev2)
    assert ex2["inputs_retired"] == 2 and ex2["run_set"]["runs_dropped"] == 2
    assert ex2[H.DECLARED_RUN_DROPS_KEY]["run_ids"] == sorted(_rid(i) for i in (2, 3))
    assert ex2[H.DECLARED_RUN_DROPS_KEY]["declared_by"] == ["runner:inputs_retired"] and ex2[H.DECLARED_RUN_DROPS_KEY]["runs_dropped_at_read"] == 2
    assert "inputs_retired: 2 input(s)" in ex2[H.DECLARED_RUN_DROPS_KEY]["reason"]
    # the whole group leaves discovery: the empty revision declares "*" (input_vanished) and publishes, as before
    for i in (0, 1):
        shutil.rmtree(os.path.join(store_root, f"s{i:03d}"))
    rep3, ing3 = _pass(root, ad, _t(4000))
    assert rep3.withdrawals >= 1 and ing3.revisions_held == 0 and ing3.refused == 0
    rev3 = _head(root)
    assert rev3 != rev2 and _paper_rows(root) == 0
    ex3 = _extra(root, rev3)
    assert ex3[H.DECLARED_RUN_DROPS_KEY]["run_ids"] == "*" and ex3[H.DECLARED_RUN_DROPS_KEY]["declared_by"] == ["runner:input_vanished"]
    assert ex3["run_set"] == {"previous_revision_id": rev2, "runs_previous": 2, "runs_now": 0, "runs_dropped": 2, "runs_added": 0}
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE reason = 'input_vanished' AND table_name = 'run'")[0][0] == 2
    _rebuild_equal(root, tmp_path)


def test_a_re_activation_of_a_smaller_published_revision_is_held_undeclared_and_the_head_stands(tmp_path):
    """A reverted input set below the inputs-dropped threshold re-activates a published revision as a bundle (no observation
    records re-emitted: its membership is in the table) -- the guard reads that membership and holds the bundle when it drops a
    head run id undeclared."""
    root, store_root, ad = _setup(tmp_path, 12)
    _pass(root, ad, _t(2000))
    rev_small = _head(root)
    _store(store_root, "s012", "store_012", "none", [0.5])                  # a thirteenth store: the head grows to 13 runs
    rep2, _ = _pass(root, ad, _t(3000))
    rev_big = rep2.results[0].revision_id
    assert _head(root) == rev_big and _paper_rows(root) == 13
    os.remove(os.path.join(store_root, "s012", "store.json"))              # it leaves again: 1 of 13 inputs (7.7 %, under the 10 % input guard)
    rep3, ing3 = _pass(root, ad, _t(4000))
    assert rep3.groups_refreshed == 1 and rep3.groups_inputs_dropped == 0 and rep3.results[0].revision_id == rev_small
    assert ing3.revisions_held_run_drop == 1 and _head(root) == rev_big and _paper_rows(root) == 13
    # the revision's row is accepted (superseded) and says nothing of the hold: the QUEUE names it -- the bundle's marker, kind run_drop
    accepted, refusal, superseded, _ = _marker(root, rev_small)
    assert accepted == 1 and refusal is None and superseded == rev_big
    (blocked,) = _blocked_marker(root, rev_small)
    assert blocked == ("source_revision", rev_big, H.RUN_SET_DROP_KIND, 0)
    reason = _q(root, "SELECT json_extract(extra_json, '$.blocked_on'), record_uri FROM blocked_records WHERE json_extract(extra_json, '$.revision_id') = ?", rev_small)[0]
    assert reason[0].startswith(f"{H.RUN_SET_DROP_KEY}: {SOURCE} no longer names 1 of the 13 run ids of the accepted head {rev_big}") and "/_activations/" in reason[1]
    bundles = os.listdir(os.path.join(root, "records", "adapters", "synthetic_store", "_activations", rev_small))
    assert len(bundles) == 1
    # the store still gone: the pass sees the held bundle and re-emits NOTHING (no second copy of the revision per pass)
    with DB.reading(DB.db_path(root)) as conn:
        rep3b = RUN.run_pass(root, [ad], conn=conn, now=_t(4500))
    assert rep3b.groups_held == 1 and rep3b.groups_refreshed == 0 and "re-activation bundle is held" in rep3b.results[0].error
    assert H.RUN_SET_DROP_KEY in rep3b.results[0].error
    assert os.listdir(os.path.join(root, "records", "adapters", "synthetic_store", "_activations", rev_small)) == bundles
    ing3b = _ingest(root)
    assert ing3b.revision_transactions == 0 and _head(root) == rev_big
    # the store is back: the big revision is the head already, the pass re-stamps its watermarks and nothing is held
    _store(store_root, "s012", "store_012", "none", [0.5])
    rep4, ing4 = _pass(root, ad, _t(5000))
    assert rep4.results[0].revision_id == rev_big and ing4.revisions_held_run_drop == 0 and _paper_rows(root) == 13
    _rebuild_equal(root, tmp_path)


def test_malformed_run_drop_declarations_refuse_by_name(tmp_path):
    root, store_root, ad = _setup(tmp_path, 3)
    _pass(root, ad, _t(2000))
    ad.EMISSION_VERSION = 2
    for bad, match in ((["r1"], "runs_retired must be a dict"), ({"run_ids": "r1", "reason": "x"}, "runs_retired.run_ids must be"),
                       ({"run_ids": [], "reason": "x"}, "runs_retired.run_ids must be"), ({"run_ids": ["r1", 2], "reason": "x"}, "runs_retired.run_ids must be"),
                       ({"run_ids": ["r1"]}, "runs_retired.reason must be"), ({"run_ids": "*", "reason": " "}, "runs_retired.reason must be")):
        ad.runs_retired = bad
        with pytest.raises(ValueError, match="synthetic_store: " + match):
            with DB.reading(DB.db_path(root)) as conn:
                RUN.run_pass(root, [ad], conn=conn, now=_t(3000))
    assert AB.runs_retired_of(object(), GROUP) is None
    assert AB.runs_retired_of(type("A", (), {"runs_retired": lambda self, g: None})(), GROUP) is None
    assert AB.runs_retired_of(type("A", (), {"runs_retired": {"run_ids": ["b", "a", "a"], "reason": " why "}})(), GROUP) == {"run_ids": ["a", "b"], "reason": "why"}
    assert AB.runs_retired_of(type("A", (), {"runs_retired": {"run_ids": "*", "reason": "all"}})(), GROUP) == {"run_ids": "*", "reason": "all"}


# ============================================================================================
# rule versioning: the rule applies from the stamp it landed on main
# ============================================================================================

def test_the_rule_is_versioned_by_its_landing_stamp_at_the_boundary():
    """``RUN_SET_DROP_UNDECLARED_SINCE`` is the live harvester's ACTIVATION of the rule -- the base checkout's fast-forward to the merge
    of f4ad71fd7 (6c07514e4, 2026-09-23 14:16:01 -0700) in the registry's stamp format -- not the commit's own date
    (``RUN_SET_DROP_UNDECLARED_LANDED``, 2026-09-23T18:30:43Z), which one live acceptance postdates: relu_paired:s2 d366f7644c815fcf,
    observed 20:22:17Z, an undeclared 88 -> 86 shrink the live fold accepted before the code reached it. A marker observed one
    millisecond before the activation is judged as the pre-rule fold judged it, one at or after it by the rule; an unreadable stamp counts
    as after (the safe side)."""
    since = H.RUN_SET_DROP_UNDECLARED_SINCE
    assert since == "2026-09-23T21:16:01.000Z" and R.UTC_MS_RE.match(since)
    assert H.RUN_SET_DROP_UNDECLARED_LANDED == "2026-09-23T18:30:43.000Z" and H.RUN_SET_DROP_UNDECLARED_LANDED < since
    ms = R.utc_to_epoch_ms(since)
    assert H.run_set_drop_rule_applies(since) is True
    assert H.run_set_drop_rule_applies(R.epoch_ms_to_utc(ms + 1)) is True
    assert H.run_set_drop_rule_applies(R.epoch_ms_to_utc(ms - 1)) is False
    assert H.run_set_drop_rule_applies("2026-09-23T16:04:52.933Z") is False      # the latest of the R3 bootstrap's 45 pre-rule acceptances
    assert H.run_set_drop_rule_applies("2026-09-23T20:22:17.033Z") is False      # d366f7644c815fcf: accepted live after the commit, before the activation
    assert H.run_set_drop_rule_applies("2026-09-23T21:58:10.449Z") is True       # the first read of that source folded under the rule
    assert H.run_set_drop_rule_applies("2026-09-23T23:21:54.868Z") is True       # the earliest live hold under the rule
    assert H.run_set_drop_rule_applies(synth.t(3000)) is False and H.run_set_drop_rule_applies(_t(3000)) is True
    assert H.run_set_drop_rule_applies("") is True and H.run_set_drop_rule_applies("not a stamp") is True


def test_a_pre_rule_shrink_folds_as_the_live_fold_of_its_day_did_and_a_post_rule_shrink_is_held(tmp_path):
    """The R3 bootstrap's 51: revisions the live harvester ACCEPTED before the guard existed were held by a replay applying today's
    rule -- 45 observed before it landed, 6 chained behind the stale heads the holds left. Versioned, the replay admits them. Here the
    2026-09-23 shape (6 -> 2, undeclared) stamped at ``synth.t``'s January base FOLDS: the head moves, the four runs withdraw, no hold,
    no `run_drop` marker, and the replay equals the live fold; the SAME shape stamped after the landing (``_t``) is HELD, as every test
    above shows. The check script reproduces the two replays from the database alone: the January acceptance is the one hold of the
    unversioned replay (before the stamp), the versioned replay holds nothing, and no live hold predates the stamp."""
    root, store_root, ad = _setup(tmp_path, 6)
    rep1, ing1 = _pass(root, ad, synth.t(2000))                    # January stamps: before the rule landed
    rev1 = _head(root)
    assert ing1.revisions_held_run_drop == 0 and _paper_rows(root) == 6
    ad.refuse_keys = {f"store_{i:03d}" for i in range(2, 6)}
    ad.EMISSION_VERSION = 2
    rep2, ing2 = _pass(root, ad, synth.t(3000))
    rev2 = rep2.results[0].revision_id
    assert rep2.groups_refreshed == 1 and rep2.withdrawals > 0
    assert ing2.revisions_held_run_drop == 0 and ing2.revisions_held == 0 and ing2.refused == 0
    assert _head(root) == rev2 and _paper_rows(root) == 2
    accepted, refusal, superseded, _ = _marker(root, rev2)
    assert accepted == 1 and refusal is None and superseded is None
    assert _blocked_marker(root, rev2) == []
    assert _extra(root, rev2)["run_set"] == {"previous_revision_id": rev1, "runs_previous": 6, "runs_now": 2, "runs_dropped": 4, "runs_added": 0}
    assert _q(root, "SELECT COUNT(*) FROM withdrawal WHERE source_id = ? AND from_revision_id = ? AND to_revision_id = ? AND table_name = 'run'",
              SOURCE, rev1, rev2) == [(4,)]
    assert _q(root, "SELECT COUNT(*) FROM v_paper_current WHERE selection_state = 'withdrawn'") == [(4,)]
    _rebuild_equal(root, tmp_path)
    # the operator's check, against this database: the unversioned replay would hold rev2 (observed before the stamp), the versioned one
    # holds nothing, and no live hold predates the stamp
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_set_drop_versioning_check", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                                                                  "scripts", "registry", "run_set_drop_versioning_check.py"))
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    revs, ids = chk.load(DB.db_path(root))
    unversioned = chk.simulate(revs, ids, None)
    assert [(h["revision_id"], h["compared_with"], h["runs_previous"], h["runs_now"], h["before_stamp"]) for h in unversioned] == [(rev2, rev1, 6, 2, True)]
    assert chk.simulate(revs, ids, H.RUN_SET_DROP_UNDECLARED_SINCE) == []
    expect = tmp_path / "expect.json"
    expect.write_text(json.dumps([rev2]))
    out = tmp_path / "check.json"
    assert chk.main(["--db", DB.db_path(root), "--expect-ids", str(expect), "--json", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["n_unversioned_holds"] == 1 and doc["n_unversioned_before_stamp"] == 1 and doc["n_versioned_holds"] == 0
    assert doc["n_live_holds"] == 0 and doc["n_live_holds_before_stamp"] == 0 and doc["expect"] == {"n": 1, "missing": [], "unexpected": []}
    assert chk.main(["--db", DB.db_path(root), "--expect-ids", str(expect), "--since", "2026-01-01T00:00:00.000Z"]) == 2    # gated at the base: rev2 held
    # the same shape after the landing: held, the head stands (the first test of this module in one breath)
    root2, store_root2, ad2 = _setup(tmp_path / "after", 6)
    _pass(root2, ad2, _t(2000))
    head2 = _head(root2)
    ad2.refuse_keys = {f"store_{i:03d}" for i in range(2, 6)}
    ad2.EMISSION_VERSION = 2
    rep2b, ing2b = _pass(root2, ad2, _t(3000))
    assert ing2b.revisions_held_run_drop == 1 and _head(root2) == head2 and _paper_rows(root2) == 6
    assert _blocked_marker(root2, rep2b.results[0].revision_id) == [("source_revision", head2, H.RUN_SET_DROP_KIND, 0)]
    revs2, ids2 = chk.load(DB.db_path(root2))
    assert chk.simulate(revs2, ids2, None) == [] and chk.simulate(revs2, ids2, H.RUN_SET_DROP_UNDECLARED_SINCE) == []     # held live: not an accepted shrink
    assert chk.main(["--db", DB.db_path(root2)]) == 0


def test_a_pre_rule_re_activation_bundle_is_judged_by_its_own_observed_stamp(tmp_path):
    """A bundle re-observes a committed revision with a LATER ``observed_utc`` (the head re-observed): the gate reads the marker in hand --
    the bundle's own stamp -- so a re-activation observed after the landing is judged by the rule even when the revision it re-activates
    was first observed before it, as the live fold of the bundle's day judged it."""
    root, store_root, ad = _setup(tmp_path, 12)
    _pass(root, ad, synth.t(2000))                                          # January: 12 runs, before the rule
    rev_small = _head(root)
    _store(store_root, "s012", "store_012", "none", [0.5])
    rep2, _ = _pass(root, ad, synth.t(3000))                                # January: the head grows to 13
    rev_big = rep2.results[0].revision_id
    assert _head(root) == rev_big and _paper_rows(root) == 13
    os.remove(os.path.join(store_root, "s012", "store.json"))
    rep3, ing3 = _pass(root, ad, _t(4000))                                  # after the landing: the small revision re-activated as a bundle
    assert rep3.results[0].revision_id == rev_small and ing3.revisions_held_run_drop == 1
    assert _head(root) == rev_big and _paper_rows(root) == 13
    (blocked,) = _blocked_marker(root, rev_small)
    assert blocked == ("source_revision", rev_big, H.RUN_SET_DROP_KIND, 0)
    _rebuild_equal(root, tmp_path)
