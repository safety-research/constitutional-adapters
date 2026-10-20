"""A source's revision switch is atomic for readers, and has been since the staging / activation split;
pinned here after the 2026-09-21 investigation of a paper arm that read as empty for ~40 minutes (`relu_paired`'s
`difference_relu` x over-refusal rows).

What the harvester does with an adapter revision directory (``records/adapters/<adapter>/<rev>/``): the item rows are
STAGED into the live item tables tagged with the pending revision (read by nothing: `item_visible` admits the head's rows
only) and the run-scale records into `revision_staging_record`, one bounded transaction per window; then ONE activation
transaction folds the run / judgment / run_metric / instrument / object constituents, the marker's `source_revision`
(``accepted = 1``: the head moves), its `observation` and `withdrawal` records, and recomputes `observation.current` once
per source — old rows 0, new rows 1 — before it commits. Readers (WAL snapshot isolation) see the old frame or the new
one, never neither and never a mix. The hypothesis this file refutes: that the marker folds in an earlier chunk
transaction than the new rows, demoting the old head's observations minutes before the new ones land.

What actually emptied the arm on 2026-09-21: the adapter's 22:19:36Z read of every `relu_paired` source saw 125 fewer
inputs than its 20:43Z read (a whole store tree absent from discovery for a while), so the revision it published carried
NONE of the arm's runs and WITHDREW them (`withdrawal` rows, reason `removed_measurement`: 9 run, 9 judgment, 9 run_metric
keys of `relu_paired:s1`). The 22:50Z cycle activated that revision atomically; from then until the 23:43Z re-read
(activated 23:44Z, the inputs back) the frame showed exactly what the head carried — no over-refusal rows. That is the
registry's contract (a key the head no longer carries is withdrawn), not a fold artefact. Since the inputs-dropped guard such a
read is REFUSED unless the adapter declares the drop (`inputs_retired`); the last test here shows both: the undeclared drop refused
with the head current and the frame never dipping, the declared drop withdrawn at the one-transaction switch.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import synth
from sisteer.registry.adapters import runner as RUN

from test_registry_core_fixes import StoreAdapter, _adapter_setup, _ingest, _q, _store

GROUP = "one_group"
SOURCE = RUN.source_id_for("synthetic_store", GROUP)
#: a per-store run below the item-set size, above the window cap used here, so a revision stages in several windows
ITEMS_PER_STORE = 120
WINDOW_ROWS = 100
_BEGIN = re.compile(r"^\s*BEGIN", re.IGNORECASE)
_SRCREV_UPSERT = re.compile(r"^\s*INSERT INTO source_revision\b", re.IGNORECASE)
_CURRENT_UPDATE = re.compile(r"^\s*UPDATE observation SET current\b", re.IGNORECASE)
#: the revision's own run-scale rows (`run` is left out: the harvester's `run_ingest` fact re-upserts it in the facts transaction)
_NEW_ROWS = re.compile(r"^\s*INSERT INTO (run_metric|judgment)\b", re.IGNORECASE)


def _scores(seed: int, n: int = ITEMS_PER_STORE):
    return [((i * 37 + seed * 11) % 100) / 100.0 for i in range(n)]


def _rid(key: str, obj: str) -> str:
    return I.run_id(synth.CAMPAIGN, key, obj, synth.EVAL_ID, synth.INSTRUMENT_ID)


def _frame_count(c, run_ids) -> int:
    q = ",".join("?" * len(run_ids))
    return int(c.execute(f"SELECT COUNT(*) FROM v_paper_long WHERE run_id IN ({q})", list(run_ids)).fetchone()[0])


def _current_state(c, run_ids):
    q = ",".join("?" * len(run_ids))
    return [tuple(r) for r in c.execute(
        f"SELECT pc.run_id, pc.selection_state FROM v_paper_current pc WHERE pc.run_id IN ({q}) ORDER BY 1", list(run_ids)).fetchall()]


def _head(root, source_id=SOURCE):
    rows = _q(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 AND superseded_by_revision_id IS NULL", source_id)
    assert len(rows) <= 1, rows
    return rows[0][0] if rows else None


class _Poller:
    """A second, read-only connection sampling the paper frame every ``every`` seconds on its own thread."""

    def __init__(self, root, run_ids, every=0.005):
        self.root, self.run_ids, self.every = root, list(run_ids), every
        self.samples = []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        c = DB.connect(DB.db_path(self.root), readonly=True)
        try:
            while not self._stop.is_set():
                self.samples.append((time.monotonic(), _frame_count(c, self.run_ids)))
                time.sleep(self.every)
        finally:
            c.close()

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()


@pytest.fixture
def slow_commits(monkeypatch):
    """Every write transaction of the cycle is followed by a pause, so the poller samples between every two commits."""
    real = H.Ingestor._timed_transaction

    def paused(self, body, files, rows, now, kind="transaction"):
        real(self, body, files, rows, now, kind)
        time.sleep(0.03)
    monkeypatch.setattr(H.Ingestor, "_timed_transaction", paused)
    monkeypatch.setattr(H, "ITEM_ROWS_PER_TRANSACTION", WINDOW_ROWS)


def _setup(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("a", "store_a", "constitution_lora@aaaa", _scores(1)),
                                                 ("b", "store_b", "none", _scores(2))])
    ad = StoreAdapter(store_root, group_of=lambda inp: GROUP)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    assert rep.groups_refreshed == 1 and rep.results[0].accepted
    ing = _ingest(root)
    assert ing.refused == 0 and ing.blocked == 0 and ing.revision_transactions == 1
    return root, store_root, ad


# --------------------------------------------------------------------------------------------
# the switch: one transaction, and a reader never sees fewer rows than min(old, new)
# --------------------------------------------------------------------------------------------


def test_multi_window_revision_switch_is_one_transaction_and_a_reader_never_sees_fewer_than_min_old_new(tmp_path, slow_commits, monkeypatch):
    root, store_root, ad = _setup(tmp_path)
    rid_a, rid_b, rid_c = _rid("store_a", "constitution_lora@aaaa"), _rid("store_b", "none"), _rid("store_c", "none")
    run_ids = [rid_a, rid_b, rid_c]
    rev1 = _head(root)
    with DB.reading(DB.db_path(root)) as c:
        n_old = _frame_count(c, run_ids)
    assert n_old == 2
    # revision 2: store_a re-graded, store_c new -> three runs, 360 item rows, staged in >= 4 windows of 100
    _store(store_root, "a", "store_a", "constitution_lora@aaaa", _scores(5))
    _store(store_root, "c", "store_c", "none", _scores(3))
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000))
    assert rep.groups_refreshed == 1 and rep.results[0].accepted
    rev2 = rep.results[0].revision_id
    assert rev2 != rev1
    # inside the activation, at the moment the head is recomputed: a second connection still reads the OLD frame whole
    inside = []
    real_recompute = H.Reducer._recompute_current

    def spy(self, source_id, head="?"):
        if self.deferred_current is None and source_id == SOURCE:      # the once-per-source call at the end of the activation body
            c = DB.connect(DB.db_path(root), readonly=True)
            inside.append((_frame_count(c, run_ids), _head(root)))
            c.close()
        return real_recompute(self, source_id, head)
    monkeypatch.setattr(H.Reducer, "_recompute_current", spy)
    conn = DB.open_or_create(DB.db_path(root))
    stmts = []
    conn.set_trace_callback(stmts.append)
    try:
        with _Poller(root, run_ids) as poller:
            ing = H.ingest_once(root, conn=conn, workers=0, min_free=0)
    finally:
        conn.set_trace_callback(None)
        conn.close()
    assert ing.refused == 0 and ing.blocked == 0 and ing.revisions_held == 0 and ing.revision_transactions == 1
    assert ing.staging_transactions >= 4, ing.staging_transactions                    # several bounded windows before the switch
    assert inside == [(n_old, rev1)], inside                                          # the old frame, whole, until the very last statement
    counts = [n for _, n in poller.samples]
    assert len(counts) >= 20 and min(counts) >= min(n_old, 3) and set(counts) <= {2, 3}, (len(counts), sorted(set(counts)))
    assert counts[0] == 2 and counts[-1] == 3 and counts == sorted(counts)            # one step, old -> new, never back
    # the statements: every write that moves the head, demotes the old rows or lands the new run-scale rows sits in ONE
    # BEGIN ... COMMIT, and no earlier transaction of the cycle touches source_revision or observation.current
    seg, seg_of = -1, []
    for s in stmts:
        if _BEGIN.match(s):
            seg += 1
        seg_of.append(seg)
    where = {"marker": {seg_of[i] for i, s in enumerate(stmts) if _SRCREV_UPSERT.match(s)},
             "current": {seg_of[i] for i, s in enumerate(stmts) if _CURRENT_UPDATE.match(s)},
             "new_rows": {seg_of[i] for i, s in enumerate(stmts) if _NEW_ROWS.match(s)}}
    assert len(where["current"]) == 1 and where["current"] == where["new_rows"], where
    assert where["marker"] <= where["current"] | {min(where["marker"])}, where          # the pending placeholder (accepted = 0) in prepare(), then the fold
    act = next(iter(where["current"]))
    order = [("marker" if _SRCREV_UPSERT.match(s) else "current" if _CURRENT_UPDATE.match(s) else "rows")
             for i, s in enumerate(stmts) if seg_of[i] == act and (_SRCREV_UPSERT.match(s) or _CURRENT_UPDATE.match(s) or _NEW_ROWS.match(s))]
    assert order.index("current") > order.index("marker") and order[-1] == "current"    # fold, marker, then the one recompute
    with DB.reading(DB.db_path(root)) as c:
        assert _frame_count(c, run_ids) == 3 and _head(root) == rev2
        assert [tuple(r) for r in c.execute("SELECT revision_id, current, COUNT(*) FROM observation WHERE source_id = ? AND table_name = 'run_metric' GROUP BY 1, 2 ORDER BY 1", (SOURCE,))] == \
            sorted([(rev1, 0, 2), (rev2, 1, 3)])
    kinds = [json.loads(line)["kind"] for line in open(os.path.join(root, "state", "metrics.jsonl"))]
    assert kinds.count("activation") == 2                                              # one per revision, both cycles


# --------------------------------------------------------------------------------------------
# an interrupted fold leaves the old head current and resumes
# --------------------------------------------------------------------------------------------


def test_a_fold_interrupted_between_staging_windows_leaves_the_old_head_current_and_resumes(tmp_path, slow_commits, monkeypatch):
    root, store_root, ad = _setup(tmp_path)
    rid_a, rid_b, rid_c = _rid("store_a", "constitution_lora@aaaa"), _rid("store_b", "none"), _rid("store_c", "none")
    run_ids = [rid_a, rid_b, rid_c]
    rev1 = _head(root)
    _store(store_root, "a", "store_a", "constitution_lora@aaaa", _scores(5))
    _store(store_root, "c", "store_c", "none", _scores(3))
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000))
    rev2 = rep.results[0].revision_id
    # the crash: the process dies after the second staging window has committed
    windows = []
    real = H.Ingestor._stage_window

    def dying(self, rev, furi, part, *a, **kw):
        if part.items:                                   # the run-scale-only files' bookkeeping windows carry no item rows
            if len(windows) == 1:
                raise RuntimeError("simulated crash between staging windows")
            windows.append(1)
        return real(self, rev, furi, part, *a, **kw)
    monkeypatch.setattr(H.Ingestor, "_stage_window", dying)
    with pytest.raises(RuntimeError):
        H.ingest_once(root, workers=0, min_free=0)
    with DB.reading(DB.db_path(root)) as c:
        assert _head(root) == rev1 and _frame_count(c, run_ids) == 2                       # the old head, the old frame
        assert _current_state(c, run_ids) == [(rid_a, "selected"), (rid_b, "selected")] or len(_current_state(c, run_ids)) == 2
        assert c.execute("SELECT accepted FROM source_revision WHERE revision_id = ?", (rev2,)).fetchone()[0] == 0   # a pending placeholder only
        staged = c.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev2,)).fetchone()[0]
        assert 0 < staged < 3 * ITEMS_PER_STORE                                            # some windows landed, tagged pending ...
        assert c.execute(f"SELECT COUNT(*) FROM v_item_result_current WHERE source_revision_id = ?", (rev2,)).fetchone()[0] == 0   # ... and read by nothing
        assert c.execute("SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE ?", (f"records/adapters/synthetic_store/{rev2}/%",)).fetchone()[0] == 0
    # the next cycle resumes: files whose windows all committed are not re-staged, the interrupted one starts over, then the switch
    monkeypatch.setattr(H.Ingestor, "_stage_window", real)
    with _Poller(root, run_ids) as poller:
        ing = _ingest(root)
    assert ing.refused == 0 and ing.blocked == 0 and ing.revisions_held == 0 and ing.revision_transactions == 1
    counts = [n for _, n in poller.samples]
    assert counts and min(counts) >= 2 and counts == sorted(counts) and counts[-1] == 3, sorted(set(counts))
    with DB.reading(DB.db_path(root)) as c:
        assert _head(root) == rev2 and _frame_count(c, run_ids) == 3
        assert c.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev2,)).fetchone()[0] == 3 * ITEMS_PER_STORE
        assert c.execute("SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ?", (rev2,)).fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE ? AND status = 'ok'", (f"records/adapters/synthetic_store/{rev2}/%",)).fetchone()[0] > 0


# --------------------------------------------------------------------------------------------
# what a read that drops a run does: refused undeclared (the head stays); declared, a withdrawal at the switch, restored by the next
# read that carries it
# --------------------------------------------------------------------------------------------


def test_a_read_that_drops_a_run_is_refused_undeclared_and_declared_withdraws_it_at_the_switch_then_the_next_read_restores_it(tmp_path, slow_commits):
    root, store_root, ad = _setup(tmp_path)
    rid_a, rid_b = _rid("store_a", "constitution_lora@aaaa"), _rid("store_b", "none")
    run_ids = [rid_a, rid_b]
    rev1 = _head(root)
    # store_b leaves discovery (the 2026-09-21 shape: a store tree absent for one read): half the source's inputs, undeclared — the
    # inputs-dropped guard REFUSES the read before any store is opened (tests/test_registry_input_drop_guard.py); the head stays
    # current and the frame never dips while the refused marker folds
    path_b = os.path.join(store_root, "b", "store.json")
    os.remove(path_b)
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3000))
    (res,) = rep.results
    assert rep.groups_refused == 1 and rep.groups_inputs_dropped == 1 and rep.groups_refreshed == 0 and rep.withdrawals == 0
    assert not res.accepted and res.inputs_dropped and res.error.startswith(f"{SOURCE}: {RUN.INPUTS_DROPPED_KEY}: 1 of the 2 inputs of the previous accepted revision {rev1}")
    rev2 = res.revision_id
    assert rev2 != rev1
    with _Poller(root, run_ids) as poller:
        ing = _ingest(root)
    assert ing.refused == 0 and ing.revision_transactions == 0
    counts = [n for _, n in poller.samples]
    assert counts and set(counts) == {2}, sorted(set(counts))
    with DB.reading(DB.db_path(root)) as c:
        assert _head(root) == rev1 and _frame_count(c, run_ids) == 2
        assert tuple(c.execute("SELECT accepted, substr(refusal, 1, 70) FROM source_revision WHERE revision_id = ?", (rev2,)).fetchone()) == \
            (0, f"{SOURCE}: {RUN.INPUTS_DROPPED_KEY}: 1 of the 2 inputs of the previous"[:70])
        assert c.execute("SELECT COUNT(*) FROM withdrawal WHERE source_id = ?", (SOURCE,)).fetchone()[0] == 0
        assert dict(_current_state(c, run_ids))[rid_b] != "withdrawn"
    # the same drop DECLARED (the adapter retires store_b on purpose — a reader change, read by the window's --retry-refused): the
    # read publishes, and the withdrawal lands at the ONE activation transaction — the frame steps 2 -> 1 once, never below, never to a blank
    ad.inputs_retired = [path_b]
    with DB.reading(DB.db_path(root)) as conn:
        rep2 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3500), retry_refused=True)
    (res2,) = rep2.results
    assert rep2.groups_refreshed == 1 and rep2.groups_inputs_dropped == 0 and res2.accepted and not res2.inputs_dropped and res2.withdrawals > 0
    assert res2.revision_id == rev2                                                       # the same inputs: the id the refused read named
    with _Poller(root, run_ids) as poller:
        ing2 = _ingest(root)
    assert ing2.refused == 0 and ing2.revision_transactions == 1
    counts = [n for _, n in poller.samples]
    assert counts[0] == 2 and counts[-1] == 1 and set(counts) == {2, 1} and counts == sorted(counts, reverse=True), sorted(set(counts))
    with DB.reading(DB.db_path(root)) as c:
        assert _head(root) == rev2 and _frame_count(c, run_ids) == 1
        cmp = json.loads(c.execute("SELECT extra_json FROM source_revision WHERE revision_id = ?", (rev2,)).fetchone()[0])
        assert (cmp["inputs_previous"], cmp["inputs_now"], cmp["inputs_dropped"], cmp["inputs_retired"]) == (2, 1, 0, 1)
        assert c.execute("SELECT accepted, refusal FROM source_revision WHERE revision_id = ?", (rev2,)).fetchone()[0] == 1
        # v_paper_current still lists the dropped run, as withdrawn; v_paper_long has no metric row for it (its key is current nowhere)
        assert dict(_current_state(c, run_ids))[rid_b] == "withdrawn"
        assert [tuple(r) for r in c.execute("SELECT reason, COUNT(*) FROM withdrawal WHERE source_id = ? AND from_revision_id = ? AND to_revision_id = ? GROUP BY 1 ORDER BY 1",
                                            (SOURCE, rev1, rev2))] == [("removed_items", 1), ("removed_measurement", 3)]   # its items; run, judgment, run_metric keys
    # the store is back: the same inputs give the same revision id, re-activated as a bundle (a superset of the head's inputs, so
    # nothing to declare); the frame is whole again, in one step
    _store(store_root, "b", "store_b", "none", _scores(2))
    with DB.reading(DB.db_path(root)) as conn:
        rep3 = RUN.run_pass(root, [ad], conn=conn, now=synth.t(4000))
    assert rep3.groups_refreshed == 1 and rep3.results[0].revision_id == rev1
    with _Poller(root, run_ids) as poller:
        ing3 = _ingest(root)
    assert ing3.refused == 0 and ing3.revision_transactions == 1
    counts = [n for _, n in poller.samples]
    assert counts[0] == 1 and counts[-1] == 2 and counts == sorted(counts), sorted(set(counts))
    with DB.reading(DB.db_path(root)) as c:
        assert _head(root) == rev1 and _frame_count(c, run_ids) == 2
        assert dict(_current_state(c, run_ids))[rid_b] != "withdrawn"
