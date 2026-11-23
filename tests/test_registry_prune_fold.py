"""The reclaim rule for superseded revisions' item rows (``prune-rule/3``, src/sisteer/registry/prune.py; migration 0033).

An accepted, SUPERSEDED revision's item rows (``item_result``, ``item_flags``, ``run_degeneration``) are deleted by the harvester's reclaim,
carrier by carrier, where the record log reproduces them exactly; every deletion is counted in ``item_reclaim_ledger``, and a deleted
carrier is re-folded from the log before anything is written under its revision again (rehydrate before write), so every later fold sees
what the code that kept every copy held. These tests pin, each against a rebuild of the shuffled log (``_rebuild_equal``) and, where the
claim is about readers, against the same world run with reclaim HELD (nothing ever deleted):

* stale copies go after a re-read or a forced re-read (no regrowth per re-read), in all three tables;
* pending, held and refused revisions' rows, staged rows, ``source_status`` and ``observation`` stay;
* re-activation (an accept, a fresh-read bundle's own rows, the late arrivals) restores exactly the hold run's visible rows;
* the flag, the marks, the visibility guard, the modes, the WAL / disk admission and a stop mid-carrier;
* a carrier the current code does not reproduce stays (``unauthentic``), and a rehydration that cannot authenticate HOLDS, writes nothing,
  and is released by an operator remedy (``prune_archive.rehydrate``).
"""
from __future__ import annotations

import gzip
import inspect
import json
import os
import random
import re
import shutil
import sqlite3
import threading

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import prune as PR
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import runner as RUN

from test_registry_core_fixes import JUDGE_ID_3, JudgeNamingAdapter, StoreAdapter, _judge3, _q, _rebuild_equal, _roster_entry, _store, _write

LORA = "constitution_lora@aaaa"
ADAPTER = "synthetic_store"
ITEM_OBJECTS = ("v_item_result_current", "v_item_flags_current", "v_run_degeneration_current", "v_item_pointer", "v_item_conflict",
                "v_item_result_provenances", "v_item_headline", "v_paper_long")


# --------------------------------------------------------------------------------------------
# a world: one store, one comparison group, an adapter that files rows in all three reclaimed tables
# --------------------------------------------------------------------------------------------

class MTAdapter(StoreAdapter):
    """The synthetic store adapter, plus one ``item_flags`` row per item and one measured ``run_degeneration`` row per run: every table
    reclaim deletes from carries rows (each in its own record file, the runner writes one file per record type)."""

    def read(self, ctx, group, inputs):
        from sisteer.registry import hooks as RH
        for batch in super().read(ctx, group, inputs):
            items = [r for r in batch.records if r["record_type"] == "item_result"]
            if items:
                rid, jid = items[0]["payload"]["run_id"], items[0]["payload"]["judgment_id"]
                for r in items:
                    p = {c: None for c in R.ITEM_WHOLE_ROW_COLUMNS["item_flags"]}
                    p.update(run_id=rid, item_id=r["payload"]["item_id"], trial_index=0, sample_index=0, turn_index=-1, judgment_id=jid,
                             response_tokens=int(round(float(r["payload"]["value"]) * 100)))
                    batch.add("item_flags", p, ctx.producer, ctx.now)
                k = sum(1 for r in items if float(r["payload"]["value"]) > 0.5)
                row = RH.measured_row(rid, "union5", "all", k, len(items), "replies", judgment_id=jid, computed_utc=ctx.now)
                row.pop("source_revision_id", None)
                batch.add("run_degeneration", row, ctx.producer, ctx.now)
            yield batch


def _ingest(root, i=None, **kw):
    kw.setdefault("workers", 0)
    kw.setdefault("min_free", 0)
    if i is not None:
        kw.setdefault("now", synth.t(i))
    return H.ingest_once(root, **kw)


def _world(base, scores=(0.2, 0.8, 0.4), mode=None, adapter=MTAdapter, store_root=None):
    """``base``/reg with the synthetic dimensions and a roster, ``base``/stores with one store (or ``store_root``: two worlds that must mint
    the same revision ids read the SAME store path, which the input uris and so the ids carry); ``mode`` = "hold" writes the mode file."""
    base = str(base)
    root = os.path.join(base, "reg")
    H.ensure_layout(root)
    if mode == "hold":
        PR.ReclaimMode.hold().write_state(root, synth.t(1))
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    _ingest(root, 10)
    store_root = str(store_root or os.path.join(base, "stores"))
    os.makedirs(store_root, exist_ok=True)
    _store(store_root, "s1", "store_1", LORA, list(scores))
    return root, store_root, adapter(store_root)


def _read(root, ad, i, ingest=True, **kw):
    """One adapter pass stamped ``t(i)`` (and the harvester cycle after it); returns the group's revision id."""
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(i), **kw)
    assert rep.results, rep
    if ingest:
        _ingest(root, i + 1)
    return rep.results[0].revision_id


def _rows(root, rev):
    return {t: _q(root, f"SELECT COUNT(*) FROM {t} WHERE source_revision_id = ?", rev)[0][0] for t in PR.PRUNE_TABLES}


def _ledger(root, rev):
    return {t: n for t, n in _q(root, "SELECT table_name, SUM(rows_deleted) FROM item_reclaim_ledger WHERE revision_id = ? GROUP BY table_name", rev)}


def _reclaimed(root, rev):
    return _q(root, "SELECT reclaimed FROM source_revision WHERE revision_id = ?", rev)[0][0]


def _head(root):
    rows = _q(root, "SELECT revision_id FROM source_revision WHERE source_id = ? AND accepted = 1 AND superseded_by_revision_id IS NULL",
              RUN.source_id_for(ADAPTER, "s1"))
    assert len(rows) == 1, rows
    return rows[0][0]


def _live_items(root):
    return dict(_q(root, "SELECT item_id, value FROM v_item_result_current"))


def _visible(root):
    """Every reader-facing object over the item tables, row for row (sorted, values normalised as rebuild equality normalises them), the
    rank column aside: two roots compared here read their store files at different wall-clock times, so the harvester's own fact files
    (which carry the stores' mtimes through the watermarks) can tie-break into another order between them -- within ONE root the ranks are
    pinned by its rebuild (``_rebuild_equal``)."""
    out = {}
    with DB.reading(DB.db_path(root)) as conn:
        for o in ITEM_OBJECTS:
            cur = conn.execute(f"SELECT * FROM {o}")
            cols = [d[0] for d in cur.description]
            keep = [i for i, c in enumerate(cols) if c != "ingest_seq"]
            out[o] = ([cols[i] for i in keep], sorted(repr(tuple(H._norm(r[i]) for i in keep)) for r in cur.fetchall()))
    return out


def _rev_dir(root, rev):
    return os.path.join(root, "records", "adapters", ADAPTER, rev)


def _bundle(root, rev, stamp_i, edit=None, observed_i=None, files=None, src=None):
    """(``stamp_i`` / ``observed_i`` may be a UTC string instead of a ``synth.t`` index.)"""
    return _bundle_at(root, rev, stamp_i if isinstance(stamp_i, str) else synth.t(stamp_i), edit,
                      None if observed_i is None else (observed_i if isinstance(observed_i, str) else synth.t(observed_i)), files, src)


def _bundle_at(root, rev, stamp, edit=None, observed=None, files=None, src=None):
    """Publish an activation bundle of ``rev`` by hand, atomically (one directory rename, the runner's way): the records of ``src``
    (default: the revision's own directory) restamped ``t(stamp_i)``, ``edit(rec)`` applied to each (None drops it), the marker's
    ``observed_utc`` set to ``t(observed_i)`` when given (newer than the head's: a re-activation; unchanged: the revision stays superseded),
    ``files`` (a predicate on the file name) choosing the files. Returns the bundle directory."""
    src = src or _rev_dir(root, rev)
    out = os.path.join(root, "records", "adapters", ADAPTER, "_activations", rev, "".join(ch for ch in stamp if ch.isdigit()))
    tmp = os.path.join(root, "staging", "bundles", os.path.basename(out))
    for fn in sorted(os.listdir(src)):
        if files is not None and not files(fn):
            continue
        recs = []
        for _, rec, _ in R.read_record_file(os.path.join(src, fn)):
            rec = json.loads(json.dumps(rec))
            rec["written_utc"] = stamp
            if rec["record_type"] == "source_revision" and observed is not None:
                rec["payload"]["observed_utc"] = observed
            if edit is not None:
                rec = edit(rec)
                if rec is None:
                    continue
            recs.append(rec)
        R.write_records(os.path.join(tmp, fn[:-3] if fn.endswith(".gz") else fn), recs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.rename(tmp, out)
    return out


def _set_values(values):
    """An ``edit`` for :func:`_bundle`: item values by item id (item_flags follow), the run's degeneration count recomputed."""
    def edit(rec):
        p = rec["payload"]
        if rec["record_type"] == "item_result" and p["item_id"] in values:
            p["value"] = values[p["item_id"]]
        elif rec["record_type"] == "item_flags" and p["item_id"] in values:
            p["response_tokens"] = int(round(values[p["item_id"]] * 100))
        return rec
    return edit


def _item_record(root, rev, item_id):
    """The revision directory's own item_result record of ``item_id`` (read from the log: the row may be deleted)."""
    d = _rev_dir(root, rev)
    for fn in sorted(os.listdir(d)):
        for _, rec, _ in R.read_record_file(os.path.join(d, fn)):
            if rec["record_type"] == "item_result" and rec["payload"]["item_id"] == item_id:
                return rec
    raise AssertionError((rev, item_id))


def _plain_item(root, rev, item_id, value, stamp_i, name):
    """A revision-tagged item_result record in a PLAIN file outside records/adapters/ (harvest.py admits one: write point 3)."""
    rec = _item_record(root, rev, item_id)
    p = dict(rec["payload"], value=value)
    assert p["source_revision_id"] == rev
    return _write(root, "late", name, [synth.rec("item_result", p, synth.t(stamp_i))])


def _plans(root, rev, mode=None):
    with DB.reading(DB.db_path(root)) as conn:
        return PR.revision_plan(conn, root, rev, mode or PR.ReclaimMode.normal())


# --------------------------------------------------------------------------------------------
# 1-8: what is deleted and what is kept
# --------------------------------------------------------------------------------------------

def test_reread_deletes_the_superseded_revisions_rows_in_all_three_tables(tmp_path):
    root, store_root, ad = _world(tmp_path)
    rev_a = _read(root, ad, 2000)
    assert _rows(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    _store(store_root, "s1", "store_1", LORA, [0.9, 0.1, 0.3])
    rev_b = _read(root, ad, 3000)
    assert rev_b != rev_a and _head(root) == rev_b
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}                       # the stale copy is gone, in every table
    assert _rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    assert _ledger(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}   # ... and counted, carrier by carrier
    assert _reclaimed(root, rev_a) == 1 and _reclaimed(root, rev_b) == 0
    assert _live_items(root) == {"q0": 0.9, "q1": 0.1, "q2": 0.3}
    cycle = H.read_cycle_state(root)                                       # the cycle's record carries its reclaim (the status page reads it)
    assert (cycle["reclaim_rows_deleted"], cycle["reclaim_rows_rehydrated"], cycle["reclaim_stopped"], cycle["reclaim_mode"]) == (7, 0, None, "normal")
    led = _q(root, "SELECT table_name, file_uri, expected_rows, rows_deleted, expected_digest_fold = deleted_digest_fold FROM item_reclaim_ledger")
    assert all(r[2] == r[3] and r[4] == 1 and f"/{rev_a}/" in r[1] for r in led), led   # whole carriers: deleted == expected
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def test_forced_reread_leaves_one_copy_after_reclaim(tmp_path):
    """No regrowth: a forced re-read publishes a NEW revision of the same content; once the cycle's reclaim ran, the table holds one copy."""
    root, store_root, ad = _world(tmp_path)
    first = _read(root, ad, 2000)
    copies = [first]
    for i in (3000, 4000, 5000):
        rev = _read(root, ad, i, reread=True, force=True)
        assert rev not in copies
        copies.append(rev)
        assert _head(root) == rev
        assert {t: _q(root, f"SELECT COUNT(*) FROM {t}")[0][0] for t in PR.PRUNE_TABLES} == \
            {"item_result": 3, "item_flags": 3, "run_degeneration": 1}                     # one copy, whatever the number of re-reads
    assert all(_rows(root, r) == {t: 0 for t in PR.PRUNE_TABLES} for r in copies[:-1])
    assert _q(root, "SELECT COUNT(DISTINCT revision_id) FROM item_reclaim_ledger")[0][0] == 3
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def test_pending_held_and_refused_revisions_keep_their_rows(tmp_path):
    root, store_root, ad = _world(tmp_path)
    rev_a = _read(root, ad, 2000)
    # a pending revision: its judgment names a judge the registry lacks, so the whole revision is held (its rows staged, invisible)
    _store(store_root, "s1", "store_1", LORA, [0.9, 0.9, 0.9])
    held = JudgeNamingAdapter(store_root, JUDGE_ID_3)
    rev_p = _read(root, held, 3000)
    assert _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev_p) == [(0,)]
    assert _rows(root, rev_p)["item_result"] == 3 and _rows(root, rev_a)["item_result"] == 3     # the head and the pending one stay
    # a refused read (new bytes, then the read fails) publishes no item row at all; the pending one still stays after another cycle's reclaim
    _store(store_root, "s1", "store_1", LORA, [0.7, 0.7, 0.7])
    ad.broken.add(os.path.join(store_root, "s1", "store.json"))
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(3500))
    assert rep.groups_refused == 1
    ad.broken.clear()
    _ingest(root, 3501)
    assert _rows(root, rev_p)["item_result"] == 3 and _rows(root, rev_a)["item_result"] == 3
    # a later accepted read supersedes the head AND the pending revision: the old head's rows go, the pending one's never do
    _store(store_root, "s1", "store_1", LORA, [0.5, 0.5, 0.5])
    rev_c = _read(root, ad, 4000)
    assert _head(root) == rev_c
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    assert _rows(root, rev_p)["item_result"] == 3 and _q(root, "SELECT accepted FROM source_revision WHERE revision_id = ?", rev_p) == [(0,)]
    assert not _plans(root, rev_p)                                                          # never a candidate: not accepted
    _rebuild_equal(root, tmp_path)


def test_rows_staged_by_a_held_reactivation_bundle_are_kept_and_accept_activates_them(tmp_path):
    """The held re-activation bundle (the run-set drop guard's: a reverted input set re-activates a smaller revision, which drops a head run
    undeclared): its files are STAGED under the accepted, superseded revision and kept by every reclaim; the operator's accept activates
    them, and the outgoing head's rows go."""
    import test_registry_run_set_drop_guard as RSD
    root, store_root, ad = RSD._setup(tmp_path, 12)
    RSD._pass(root, ad, RSD._t(2000))
    rev_small = RSD._head(root)
    _store(store_root, "s012", "store_012", "none", [0.5])
    rep2, _ = RSD._pass(root, ad, RSD._t(3000))
    rev_big = rep2.results[0].revision_id
    assert _rows(root, rev_small)["item_result"] == 0 and _ledger(root, rev_small) == {"item_result": 12}
    os.remove(os.path.join(store_root, "s012", "store.json"))
    rep3, ing3 = RSD._pass(root, ad, RSD._t(4000))
    assert rep3.results[0].revision_id == rev_small and ing3.revisions_held_run_drop == 1 and RSD._head(root) == rev_big
    # write point 1 re-folded the deleted directory carrier before the bundle staged; the bundle's rows are staged and protected
    assert ing3.reclaim_rows_rehydrated == 12
    staged = {u for (u,) in _q(root, "SELECT file_uri FROM revision_staging_file WHERE revision_id = ?", rev_small)}
    assert staged and all("/_activations/" in u for u in staged)
    plans = _plans(root, rev_small)
    by_class = {}
    for p in plans:
        by_class.setdefault(p.klass, []).append(p)
    assert {p.file_uri for p in by_class.get(PR.STAGED, [])} & staged
    assert not any(p.file_uri in staged for p in plans if p.klass == PR.DELETABLE)
    n_staged = _rows(root, rev_small)["item_result"]
    assert n_staged == 12
    _ingest(root)
    assert _rows(root, rev_small)["item_result"] == n_staged                               # another cycle's reclaim keeps them
    assert cli.main(["--root", root, "revision", "accept", "--revision-id", rev_small, "--reason", "the thirteenth store is retired (synthetic)"]) == 0
    _ingest(root)
    assert RSD._head(root) == rev_small
    assert _rows(root, rev_small)["item_result"] == 12 and _rows(root, rev_big)["item_result"] == 0
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE source_revision_id = ?", rev_small)[0][0] == 12
    _rebuild_equal(root, tmp_path)


def test_source_status_and_observation_rows_are_never_deleted(tmp_path):
    root, store_root, ad = _world(tmp_path)
    rev_a = _read(root, ad, 2000)
    before_status = _q(root, "SELECT COUNT(*) FROM source_status WHERE source_revision_id = ?", rev_a)[0][0]
    before_obs = _q(root, "SELECT COUNT(*) FROM observation WHERE revision_id = ?", rev_a)[0][0]
    assert before_status >= 1 and before_obs >= 1
    _store(store_root, "s1", "store_1", LORA, [0.1, 0.1, 0.1])
    _read(root, ad, 3000)
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    assert _q(root, "SELECT COUNT(*) FROM source_status WHERE source_revision_id = ?", rev_a)[0][0] == before_status
    assert _q(root, "SELECT COUNT(*) FROM observation WHERE revision_id = ?", rev_a)[0][0] == before_obs
    assert "source_status" not in PR.PRUNE_TABLES and "observation" not in PR.PRUNE_TABLES
    _rebuild_equal(root, tmp_path)


def _reactivation_world(base, mode, stores):
    root, store_root, ad = _world(base, mode=mode, store_root=stores)
    rev_a = _read(root, ad, 2000)
    _store(store_root, "s1", "store_1", LORA, [0.9, 0.9])                 # B: changed values and a shrink
    rev_b = _read(root, ad, 3000)
    _store(store_root, "s1", "store_1", LORA, [0.2, 0.8, 0.4])            # A's bytes again: re-activated by a bundle
    rev_c = _read(root, ad, 4000)
    assert rev_c == rev_a and _head(root) == rev_a
    _store(store_root, "s1", "store_1", LORA, [0.9, 0.9])                 # and B again
    assert _read(root, ad, 5000) == rev_b
    return root, rev_a, rev_b


def test_reactivation_restores_the_same_visible_rows_as_with_the_hold(tmp_path):
    """Re-activation is unchanged: the same history in a root whose reclaim is held (nothing ever deleted) and in a normal one: every reader-facing item
    object is equal row for row at every step's end, and the normal root equals its rebuild."""
    held, a1, b1 = _reactivation_world(tmp_path / "held", "hold", tmp_path / "stores")
    root, a2, b2 = _reactivation_world(tmp_path / "normal", None, tmp_path / "stores")
    assert (a1, b1) == (a2, b2)
    assert _visible(held) == _visible(root)
    assert _rows(held, a1)["item_result"] > 0 and _rows(root, a2)["item_result"] == 0 and _q(held, "SELECT COUNT(*) FROM item_reclaim_ledger")[0][0] == 0
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def _fresh_read_world(base, mode, stores):
    """Rows a revision got from a FRESH-READ bundle's own files (not in its directory, so a re-emission of the directory cannot
    bring them back) -- the bundle files an extra item; the revision is superseded, then re-activated by the runner's re-emission."""
    root, store_root, ad = _world(base, mode=mode, store_root=stores)
    rev_a = _read(root, ad, 2000)
    bundle = _bundle(root, rev_a, 2500, files=lambda fn: "item_result" in fn or "source_revision" in fn)
    item_file = next(os.path.join(bundle, fn) for fn in os.listdir(bundle) if "item_result" in fn)
    recs = [rec for _, rec, _ in R.read_record_file(item_file)]
    q3 = json.loads(json.dumps(next(r for r in recs if r["payload"]["item_id"] == "q2")))
    q3["payload"]["item_id"], q3["payload"]["value"] = "q3", 0.7
    os.remove(item_file)
    R.write_records(item_file, recs + [q3])
    _ingest(root, 2501)
    assert sorted(_live_items(root)) == ["q0", "q1", "q2", "q3"]
    _store(store_root, "s1", "store_1", LORA, [0.9, 0.9, 0.9])
    rev_b = _read(root, ad, 3000)
    _store(store_root, "s1", "store_1", LORA, [0.2, 0.8, 0.4])
    assert _read(root, ad, 4000) == rev_a and _head(root) == rev_a
    return root, rev_a, rev_b


def test_fresh_read_bundle_rows_return_on_reactivation(tmp_path):
    held, a1, _ = _fresh_read_world(tmp_path / "held", "hold", tmp_path / "stores")
    root, a2, b2 = _fresh_read_world(tmp_path / "normal", None, tmp_path / "stores")
    assert a1 == a2
    assert _live_items(root)["q3"] == 0.7                                   # the bundle's own row is back: rehydrated, not re-emitted
    assert _visible(held) == _visible(root)
    assert _rows(root, b2) == {t: 0 for t in PR.PRUNE_TABLES}
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def test_partially_displaced_carrier_is_kept(tmp_path):
    """A newer plain-file record of one key displaces the directory's row: the directory carrier no longer holds every key it parses to
    (unauthentic, count), so it is kept -- a re-fold of it could otherwise resurrect the displaced key."""
    root, store_root, ad = _world(tmp_path)
    rev_a = _read(root, ad, 2000)
    _plain_item(root, rev_a, "q0", 0.55, 2500, "000001-q0.jsonl")
    _ingest(root, 2501)
    assert _live_items(root)["q0"] == 0.55
    _store(store_root, "s1", "store_1", LORA, [0.1, 0.1, 0.1])
    _read(root, ad, 3000)
    plans = {(p.table, p.file_uri.rsplit("/", 1)[-1]): p for p in _plans(root, rev_a) if p.table == "item_result"}
    dir_carrier = next(p for (t, f), p in plans.items() if "item_result" in f)
    assert dir_carrier.klass == PR.UNAUTHENTIC_COUNT and dir_carrier.rows == 2
    assert _rows(root, rev_a) == {"item_result": 2, "item_flags": 0, "run_degeneration": 0}    # the plain carrier was whole: deleted
    assert _ledger(root, rev_a) == {"item_result": 1, "item_flags": 3, "run_degeneration": 1}
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# 9-14: late arrivals, each with a reclaim between the arrivals
# --------------------------------------------------------------------------------------------

def _superseded(tmp_path, mode=None, name="w"):
    root, store_root, ad = _world(tmp_path / name, mode=mode, store_root=tmp_path / "stores")
    rev_a = _read(root, ad, 2000)
    _store(store_root, "s1", "store_1", LORA, [0.9, 0.9, 0.9])
    rev_b = _read(root, ad, 3000)
    return root, store_root, ad, rev_a, rev_b


def test_an_older_plain_record_after_its_winner_was_deleted_loses_as_on_a_rebuild(tmp_path):
    """The sequence: the directory's row (stamped t(2000)) is deleted; a plain-file record of the same revision and key stamped
    EARLIER arrives. Rehydration restores the winner first, so the late record loses the guard exactly as it does on a rebuild."""
    root, _, _, rev_a, _ = _superseded(tmp_path)
    assert _rows(root, rev_a)["item_result"] == 0
    _plain_item(root, rev_a, "q0", 0.01, 1500, "000001-old.jsonl")
    rep = _ingest(root, 3501)
    assert rep.reclaim_rows_rehydrated == 7 and rep.blocked == 0          # every deleted carrier of the revision, re-folded first
    assert _rows(root, rev_a)["item_result"] == 0                          # deleted again at the cycle's end: no row on either path
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE value = 0.01")[0][0] == 0
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def test_a_newer_plain_record_after_deletion_displaces_the_key_and_the_directory_carrier_is_kept(tmp_path):
    root, _, _, rev_a, _ = _superseded(tmp_path)
    _plain_item(root, rev_a, "q0", 0.66, 3400, "000001-new.jsonl")
    _ingest(root, 3401)
    p = next(p for p in _plans(root, rev_a) if p.table == "item_result" and f"/{rev_a}/" in p.file_uri)
    assert p.klass == PR.UNAUTHENTIC_COUNT and p.rows == 2                # rehydrated, then displaced: kept whole
    assert _rows(root, rev_a)["item_result"] == 2                          # the plain carrier (one row, authentic) went
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def _rsd_world(base, stores, n, mode):
    """The run-set drop guard's world (its adapter refuses rows on a switch; its stamps are after the guard landed) over a shared store path."""
    import test_registry_run_set_drop_guard as RSD
    root = os.path.join(str(base), "reg")
    H.ensure_layout(root)
    if mode == "hold":
        PR.ReclaimMode.hold().write_state(root, RSD._t(1))
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    _ingest(root)
    stores = str(stores)
    for name, key, obj, scores in RSD._stores(n):
        if not os.path.exists(os.path.join(stores, name, "store.json")):
            _store(stores, name, key, obj, scores)
    return root, stores, RSD.RefusingStoreAdapter(stores)


def test_a_held_reactivation_staged_after_deletion_then_accepted_matches_the_hold_run_and_a_rebuild(tmp_path):
    import test_registry_run_set_drop_guard as RSD

    def world(base, mode):
        root, store_root, ad = _rsd_world(base, tmp_path / "stores", 12, mode)
        RSD._pass(root, ad, RSD._t(2000))
        small = RSD._head(root)
        _store(store_root, "s012", "store_012", "none", [0.5])
        RSD._pass(root, ad, RSD._t(3000))
        os.remove(os.path.join(store_root, "s012", "store.json"))
        RSD._pass(root, ad, RSD._t(4000))                                  # held: the bundle's rows staged under `small`
        assert cli.main(["--root", root, "revision", "accept", "--revision-id", small, "--reason", "retired (synthetic)"]) == 0
        _ingest(root)
        assert RSD._head(root) == small
        return root
    held = world(tmp_path / "held", "hold")
    root = world(tmp_path / "normal", None)
    assert _visible(held) == _visible(root)
    _rebuild_equal(root, tmp_path)


def test_a_fresh_read_bundle_into_a_superseded_revision_after_deletion_folds_as_without_deletion(tmp_path):
    """The M3 class: a fresh read folds into an accepted revision that is superseded and whose rows were deleted; the bundle's marker keeps
    the revision's observed_utc, so it stays superseded."""
    held, *_ = _superseded(tmp_path, "hold", name="held")
    root, _, _, rev_a, rev_b = _superseded(tmp_path, None, name="normal")
    for r in (held, root):
        _bundle(r, rev_a, 3500, edit=_set_values({"q1": 0.33}), files=lambda fn: "item_" in fn or "source_revision" in fn)
        rep = _ingest(r, 3501)
        assert rep.revisions_held == 0 and _head(r) == rev_b
    assert _visible(held) == _visible(root)
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def test_two_bundles_folding_out_of_stamp_order_across_a_supersession_and_a_deletion(tmp_path):
    def world(name, mode):
        root, store_root, ad = _world(tmp_path / name, mode=mode, store_root=tmp_path / "stores")
        rev_a = _read(root, ad, 2000)
        early = _bundle(root, rev_a, 2100, edit=_set_values({"q0": 0.11}), files=lambda fn: "item_" in fn or "source_revision" in fn)
        late = _bundle(root, rev_a, 2200, edit=_set_values({"q0": 0.22}), files=lambda fn: "item_" in fn or "source_revision" in fn)
        hidden = os.path.join(root, "staging", "parked-" + os.path.basename(early))
        os.rename(early, hidden)                                           # the later bundle arrives first (the earlier parked off the log)
        _ingest(root, 2201)
        _store(store_root, "s1", "store_1", LORA, [0.9, 0.9, 0.9])
        _read(root, ad, 3000)                                              # supersession; the later bundle's rows deleted
        os.rename(hidden, early)                                           # the earlier bundle arrives last
        _ingest(root, 3501)
        return root, rev_a, late
    held, *_ = world("held", "hold")
    root, rev_a, _ = world("normal", None)
    assert _visible(held) == _visible(root)
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def _held_small(base, stores, mode):
    """The run-set drop guard's held re-activation: twelve stores (``small``), a thirteenth (``big``), the thirteenth gone again -- the
    re-activation of ``small`` drops a head run undeclared and is HELD, its bundle STAGED under the accepted, superseded ``small``."""
    import test_registry_run_set_drop_guard as RSD
    root, stores, ad = _rsd_world(base, stores, 12, mode)
    RSD._pass(root, ad, RSD._t(2000))
    small = RSD._head(root)
    _store(stores, "s012", "store_012", "none", [0.5])
    RSD._pass(root, ad, RSD._t(3000))
    big = RSD._head(root)
    os.remove(os.path.join(stores, "s012", "store.json"))
    _, ing = RSD._pass(root, ad, RSD._t(4000))
    assert ing.revisions_held_run_drop == 1 and RSD._head(root) == big
    acts = os.path.join(root, "records", "adapters", ADAPTER, "_activations", small)
    (stamp,) = os.listdir(acts)
    return root, stores, ad, small, big, os.path.join(acts, stamp)


def test_a_staged_bundle_replaced_by_a_newer_bundle_after_deletion(tmp_path):
    """The stale-bundle clearing (harvest._stage_revision): the held re-activation's files are STAGED under ``small``, whose own rows reclaim
    deleted when ``big`` superseded it; a NEWER bundle of ``small`` -- one measurement corrected, the drop declared -- clears them whole
    (rehydrating first) and activates. The result is the hold run's, and a rebuild's."""
    import test_registry_run_set_drop_guard as RSD

    def world(name, mode):
        root, stores, ad, small, big, b1 = _held_small(tmp_path / name, tmp_path / "stores", mode)
        assert _q(root, "SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ?", small)[0][0] > 0

        def declare(rec):
            p = rec["payload"]
            if rec["record_type"] == "source_revision":
                p["extra"] = dict(p.get("extra") or {}, **{H.DECLARED_RUN_DROPS_KEY: {"run_ids": "*", "reason": "retired (synthetic)",
                                                                                      "declared_by": ["test"]}})
            elif rec["record_type"] == "item_result" and p["run_id"] == RSD._rid(0):
                p["value"] = 0.123
            return rec
        _bundle(root, small, RSD._t(4500), edit=declare, observed_i=RSD._t(4500), src=b1)
        _ingest(root)
        assert RSD._head(root) == small
        assert not _q(root, "SELECT 1 FROM revision_staging_file WHERE revision_id = ?", small)
        assert _q(root, "SELECT value FROM v_item_result_current WHERE run_id = ?", RSD._rid(0)) == [(0.123,)]
        assert _rows(root, big)["item_result"] == (13 if mode == "hold" else 0)
        return root
    held = world("held", "hold")
    root = world("normal", None)
    assert _visible(held) == _visible(root)
    _rebuild_equal(root, tmp_path)


def test_a_bundle_directory_gone_resets_reclaimed_and_its_rows_stay(tmp_path):
    """The held bundle's directory disappears (a quarantine): its staging ledger is discarded and the revision's ``reclaimed`` reset -- the
    rows it staged lose their protection; reclaim then finds their file missing and keeps them, by name, and the revision stays due (a
    missing carrier is planned again every pass, so the file's return would be seen)."""
    root, stores, ad, small, big, b1 = _held_small(tmp_path / "w", tmp_path / "stores", None)
    assert _reclaimed(root, small) == 1 and _rows(root, small)["item_result"] == 12
    shutil.rmtree(b1)
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        ing.reevaluate_blocked(R.utc_now())
        assert conn.execute("SELECT reclaimed FROM source_revision WHERE revision_id = ?", (small,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM revision_staging_file WHERE revision_id = ?", (small,)).fetchone()[0] == 0
    _ingest(root)
    plans = _plans(root, small)
    assert plans and all(p.klass == PR.MISSING for p in plans), [(p.klass, p.detail) for p in plans]
    assert _rows(root, small)["item_result"] == 12 and _reclaimed(root, small) == 0


def _chain_move_world(base, stores, mode):
    """A, then B (A superseded), then A again (B superseded and, in a normal root, deleted); then a bundle of A that RE-STATES A's own
    older observed_utc: the chain moves under it and B -- whose rows reclaim deleted -- is the head again, though nothing was written
    under B (the property test's seed 18)."""
    root, store_root, ad = _world(base, mode=mode, store_root=stores)
    rev_a = _read(root, ad, 2000)
    _store(store_root, "s1", "store_1", LORA, [0.5, 0.5])
    rev_b = _read(root, ad, 3000)
    _store(store_root, "s1", "store_1", LORA, [0.2, 0.8, 0.4])
    assert _read(root, ad, 4000) == rev_a and _head(root) == rev_a
    return root, rev_a, rev_b


def test_a_chain_move_by_another_revisions_marker_rehydrates_the_new_head_before_it_is_visible(tmp_path):
    held, a1, b1 = _chain_move_world(tmp_path / "held", tmp_path / "stores", "hold")
    root, rev_a, rev_b = _chain_move_world(tmp_path / "normal", tmp_path / "stores", None)
    assert (a1, b1) == (rev_a, rev_b) and _rows(root, rev_b) == {t: 0 for t in PR.PRUNE_TABLES} and _ledger(root, rev_b)
    for r in (held, root):
        _bundle(r, rev_a, 5000, files=lambda fn: "source_revision" in fn)     # the marker as the directory wrote it: A's first observed_utc
        rep = _ingest(r, 5001)
        assert rep.revisions_held == 0 and _head(r) == rev_b                    # the chain moved: B is the head again
    assert _rows(root, rev_b) == {"item_result": 2, "item_flags": 2, "run_degeneration": 1} and not _ledger(root, rev_b)
    assert _visible(held) == _visible(root) and _live_items(root) == {"q0": 0.5, "q1": 0.5}
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def test_a_chain_move_onto_a_head_whose_rows_cannot_be_restored_is_held_and_the_old_head_stays(tmp_path, monkeypatch):
    root, rev_a, rev_b = _chain_move_world(tmp_path / "w", tmp_path / "stores", None)
    monkeypatch.setattr(VAL, "validate_record", _refuse_q1_of(rev_b))
    _bundle(root, rev_a, 5000, files=lambda fn: "source_revision" in fn)
    rep = _ingest(root, 5001)
    assert rep.revisions_held == 1 and _head(root) == rev_a                     # the activation that would move the chain is held
    assert _live_items(root) == {"q0": 0.2, "q1": 0.8, "q2": 0.4} and _rows(root, rev_b)["item_result"] == 0
    hold = _q(root, "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind') FROM blocked_records "
                    "WHERE record_type = 'source_revision' AND json_extract(extra_json, '$.revision_id') = ?", rev_a)
    assert hold == [("item_reclaim_ledger", rev_b, H.REHYDRATION_KIND)]
    _ingest(root, 5101)                                                       # the hold stands another cycle, head or not
    assert _head(root) == rev_a and _q(root, "SELECT COUNT(*) FROM blocked_records WHERE json_extract(extra_json, '$.kind') = ?", H.REHYDRATION_KIND) == [(1,)]
    # the operator's remedy (the current code's rows: the refused record stays out, by name) releases it; the held bundle folds and B leads
    from sisteer.registry import prune_archive as PA
    res = PA.rehydrate(root, rev_b, accept_reparse=True, reason="the refused record stays out (synthetic)")
    assert not _ledger(root, rev_b) and res["rows_restored"] == 4                 # item_result q0, item_flags q0 + q1, run_degeneration
    _ingest(root, 5201, retry_held=True)
    assert _head(root) == rev_b and not _q(root, "SELECT 1 FROM blocked_records WHERE json_extract(extra_json, '$.kind') = ?", H.REHYDRATION_KIND)


def _plain_marker(root, rev, stamp_i, observed_i, name):
    """The revision directory's own ``source_revision`` record of ``rev``, restamped ``t(stamp_i)`` and observed ``t(observed_i)``, alone in
    a PLAIN file outside records/adapters/: a re-statement that moves the source's chain with no write under any revision, folded in a
    chunk of plain files (never an activation)."""
    d = _rev_dir(root, rev)
    for fn in sorted(os.listdir(d)):
        for _, rec, _ in R.read_record_file(os.path.join(d, fn)):
            if rec["record_type"] == "source_revision":
                rec = json.loads(json.dumps(rec))
                rec["written_utc"] = synth.t(stamp_i)
                rec["payload"]["observed_utc"] = synth.t(observed_i)
                return _write(root, "late", name, [rec])
    raise AssertionError(rev)


@pytest.mark.parametrize("cause", ["validator", "missing_file"])
def test_a_plain_restatement_onto_rows_that_cannot_be_restored_holds_that_record_and_the_cycle_goes_on(tmp_path, monkeypatch, cause):
    """A ``source_revision`` record in a plain file re-states rev_a (superseded, its rows deleted) as observed now, so the chain would make it
    the head again. Its deleted carriers cannot be restored -- the current validator refuses one of their records, or a carrier's log file
    is gone -- so that record alone is held on rev_a's ledger (kind ``rehydration``): the old head stays, nothing of rev_a is written, an
    independent file in the same cycle and the next cycle's files still fold, and status lists the hold. The operator's remedy (the
    current code's rows, or the archived bytes) releases it: the record folds on the next cycle and rev_a leads with its rows."""
    from sisteer.registry import prune_archive as PA
    root, rev_a, rev_b, out = _archive_then_delete(tmp_path, f"w-{cause}")
    assert _head(root) == rev_b and _ledger(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    gone = None
    if cause == "validator":
        monkeypatch.setattr(VAL, "validate_record", _refuse_q1_of(rev_a))
    else:
        (uri,), = _q(root, "SELECT file_uri FROM item_reclaim_ledger WHERE revision_id = ? AND table_name = 'item_result'", rev_a)
        gone = H.resolve_uri(root, uri)
        shutil.move(gone, str(tmp_path / "aside"))                         # out of the record tree: the harvester reads every file in it
    _plain_marker(root, rev_a, 4000, 4000, "000001-restate.jsonl")
    _write(root, "roster", "000002-roster.jsonl", [_roster_entry(2, {"paper_label": "independent_a"})])
    _ingest(root, 4001)                                                   # completes: the failed rehydration no longer escapes the cycle
    assert _head(root) == rev_b and _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    assert _ledger(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    assert _q(root, "SELECT COUNT(*) FROM roster_entry WHERE roster_entry_id = 're2'") == [(1,)]
    hold = _q(root, "SELECT record_type, missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind'), "
                    "json_extract(extra_json, '$.blocked_on') FROM blocked_records WHERE missing_parent_table = ?", PR.LEDGER_TABLE)
    assert [h[:4] for h in hold] == [("source_revision", PR.LEDGER_TABLE, rev_a, H.REHYDRATION_KIND)], hold
    assert "rehydration of" in hold[0][4] and ("the current validator refuses" if cause == "validator" else "not on disk") in hold[0][4], hold
    _write(root, "roster", "000003-roster.jsonl", [_roster_entry(3, {"paper_label": "independent_b"})])
    _ingest(root, 4101)                                                   # the next cycle: the hold stands and later files fold
    assert _head(root) == rev_b and _q(root, "SELECT COUNT(*) FROM roster_entry WHERE roster_entry_id = 're3'") == [(1,)]
    held = PA.status(root)["held_rehydrations"]
    assert [(h["revision"], h["record_type"], h["kind"]) for h in held] == [(rev_a, "source_revision", H.REHYDRATION_KIND)], held
    if cause == "validator":
        res = PA.rehydrate(root, rev_a, accept_reparse=True, reason="the refused record stays out (synthetic)")
        assert res["rows_restored"] == 6                                  # all but the refused item_result q1
    else:
        res = PA.rehydrate(root, rev_a, archive_dirs=[out])
        assert res["rows_restored"] == 7                                  # the archived bytes: the missing file is not read
    assert not _ledger(root, rev_a)
    _ingest(root, 4201)
    assert _head(root) == rev_a and not _q(root, "SELECT 1 FROM blocked_records WHERE missing_parent_table = ?", PR.LEDGER_TABLE)
    assert _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE source_revision_id = ?", rev_a) == [(2 if cause == "validator" else 3,)]
    if gone is not None:
        shutil.move(str(tmp_path / "aside"), gone)
        _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# 15: the model-based property test
# --------------------------------------------------------------------------------------------

VALUE_SETS = ([0.2, 0.8, 0.4], [0.9, 0.1, 0.3], [0.5, 0.5])
N_SEQUENCES = int(os.environ.get("PRUNE_PROPERTY_SEQUENCES", "200"))
N_CHUNKS = 8


def _ops(seed):
    rnd = random.Random(seed)
    ops = [("read", 0)]
    for _ in range(rnd.randint(3, 7)):
        kind = rnd.choice(("read", "read", "force", "fresh", "plain_old", "plain_new"))
        ops.append((kind, rnd.randrange(3) if kind == "read" else rnd.randrange(1 << 30)))
    return ops, rnd.choice((1, 2, 10_000)), [rnd.random() < 0.7 for _ in ops], [rnd.random() < 0.3 for _ in ops]


def mode_name(mode):
    return mode or "normal"


def _apply_ops(base, seed, mode):
    """Run seed's random history in a fresh world: reads of three value sets (new revisions and re-activations), forced re-reads, fresh-read
    bundles into a random accepted revision, plain-file records of a random accepted revision older or newer than its own; a cycle (and so
    a reclaim, sometimes stopped mid-way) after most operations, never only at the end."""
    ops, batch_rows, cycle_after, stop_early = _ops(seed)
    root, store_root, ad = _world(base / mode_name(mode), mode=mode, store_root=base / "stores")
    stats = {"deleted": 0, "rehydrated": 0, "stopped": 0}

    def cycle(i):
        rep = _ingest(root, i)
        stats["deleted"] += rep.reclaim_rows_deleted
        stats["rehydrated"] += rep.reclaim_rows_rehydrated
        stats["stopped"] += int(rep.reclaim_stopped is not None)
    old_rows = H.RECLAIM_BATCH_ROWS
    H.RECLAIM_BATCH_ROWS = batch_rows
    try:
        for n, ((kind, arg), cyc, early) in enumerate(zip(ops, cycle_after, stop_early)):
            i = 2000 + 100 * n
            rnd = random.Random(seed * 1000 + n)
            # in fold order (rowid), never by id: an id hashes the store's path, which differs between runs, and the seed must name one history
            accepted = [r for (r,) in _q(root, "SELECT revision_id FROM source_revision WHERE accepted = 1 AND adapter = ? ORDER BY rowid", ADAPTER)]
            if kind == "read":
                _store(store_root, "s1", "store_1", LORA, VALUE_SETS[arg])
                _read(root, ad, i, ingest=False)
            elif kind == "force" and accepted:
                _read(root, ad, i, ingest=False, reread=True, force=True)
            elif kind == "fresh" and accepted:
                rev = rnd.choice(accepted)
                _bundle(root, rev, i, edit=_set_values({f"q{rnd.randrange(2)}": round(rnd.random(), 3)}),
                        files=lambda fn: "item_" in fn or "source_revision" in fn)
            elif kind in ("plain_old", "plain_new") and accepted:
                rev = rnd.choice(accepted)
                rec = _item_record(root, rev, "q0")
                stamp = R.utc_to_epoch_ms(rec["written_utc"]) // 1000 - R.utc_to_epoch_ms(synth.t(0)) // 1000
                _plain_item(root, rev, "q0", round(rnd.random(), 3), (stamp - 1) if kind == "plain_old" else i, f"{n:06d}-plain.jsonl")
            if cyc:
                if early and mode != "hold":
                    budget = [rnd.randint(0, 3)]

                    def hook(ing, plan, budget=budget):
                        budget[0] -= 1
                        return "budget" if budget[0] < 0 else None
                    H.RECLAIM_BEFORE_BATCH_HOOK = hook
                try:
                    cycle(i + 50)
                finally:
                    H.RECLAIM_BEFORE_BATCH_HOOK = None
        cycle(9000)                                                        # quiescence: a cycle with nothing new, no stop
        cycle(9100)
    finally:
        H.RECLAIM_BATCH_ROWS = old_rows
        H.RECLAIM_BEFORE_BATCH_HOOK = None
    return root, stats


@pytest.mark.parametrize("chunk", range(N_CHUNKS))
def test_random_histories_live_equals_rebuild_and_the_hold_run(tmp_path, chunk):
    """~200 seeded histories (``PRUNE_PROPERTY_SEQUENCES``), split over the chunks: for each, the normal root in arrival order equals its
    shuffled rebuild at quiescence, and its reader-facing item objects equal the same history's with reclaim held."""
    total = {"deleted": 0, "rehydrated": 0, "stopped": 0}
    for seed in range(chunk, N_SEQUENCES, N_CHUNKS):
        base = tmp_path / f"s{seed}"
        root, stats = _apply_ops(base, seed, None)
        held, held_stats = _apply_ops(base, seed, "hold")
        assert held_stats["deleted"] == held_stats["rehydrated"] == 0
        for k in total:
            total[k] += stats[k]
        assert _visible(held) == _visible(root), (seed, _ops(seed)[0])
        n_sup = _q(root, "SELECT COUNT(*) FROM source_revision WHERE accepted = 1 AND superseded_by_revision_id IS NOT NULL")[0][0]
        assert _q(root, "SELECT COUNT(*) FROM source_revision WHERE reclaimed = 1")[0][0] == n_sup, seed
        _rebuild_equal(root, base, seed=seed)
        shutil.rmtree(base, ignore_errors=True)
    if N_SEQUENCES >= 4 * N_CHUNKS:
        # the histories exercise what the test is for: deletions, re-folds of deleted carriers, and stops mid-way
        assert total["deleted"] > 0 and total["rehydrated"] > 0 and total["stopped"] > 0, total


# --------------------------------------------------------------------------------------------
# 16-27: the flag, the marks, the guard, the modes, the admission, stops, authentication
# --------------------------------------------------------------------------------------------

def test_reclaimed_resets_on_every_item_write_and_on_directory_gone(tmp_path):
    root, _, _, rev_a, rev_b = _superseded(tmp_path)
    assert _reclaimed(root, rev_a) == 1
    # a plain-file item write under the revision resets it (inside the chunk's transaction) -- and the cycle's reclaim records it again
    _plain_item(root, rev_a, "q1", 0.01, 1000, "000001-late.jsonl")
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        ing.ingest_parsed(H.iter_parsed(root, [(os.path.join(root, "records", "late", "000001-late.jsonl"), None)], 0, H.ITEM_ROWS_PER_TRANSACTION),
                          synth.t(3500))
        assert conn.execute("SELECT reclaimed FROM source_revision WHERE revision_id = ?", (rev_a,)).fetchone()[0] == 0
        assert ing.report.reclaim_rows_rehydrated == 7
        ing.reclaim(synth.t(3500))
        assert conn.execute("SELECT reclaimed FROM source_revision WHERE revision_id = ?", (rev_a,)).fetchone()[0] == 1


def test_reclaimed_equals_accepted_and_superseded_at_quiescence(tmp_path):
    root, store_root, ad = _world(tmp_path)
    for i, scores in enumerate(([0.2, 0.8, 0.4], [0.9, 0.1, 0.3], [0.5, 0.5], [0.2, 0.8, 0.4])):
        _store(store_root, "s1", "store_1", LORA, scores)
        _read(root, ad, 2000 + 1000 * i)
    rows = _q(root, "SELECT revision_id, accepted, superseded_by_revision_id, reclaimed FROM source_revision")
    assert rows and all(r[3] == int(r[1] == 1 and r[2] is not None) for r in rows), rows
    _rebuild_equal(root, tmp_path)


def test_no_head_row_carries_a_mark_after_pass_i_and_retained_rows_are_untouched(tmp_path, monkeypatch):
    """Pass (i) still clears a head's leftover marks (written before 0033); reclaim writes no new mark and modifies no row it keeps."""
    root, store_root, ad = _world(tmp_path)
    rev_a = _read(root, ad, 2000)
    _plain_item(root, rev_a, "q0", 0.55, 2500, "000001-q0.jsonl")         # a displacing row: the directory carrier will be KEPT
    _ingest(root, 2501)
    _store(store_root, "s1", "store_1", LORA, [0.9, 0.9, 0.9])
    rev_b = _read(root, ad, 3000)
    with DB.open_or_create(DB.db_path(root)) as conn, DB.transaction(conn):
        conn.execute("UPDATE item_result SET withdrawn_by_revision_id = 'pre0033' WHERE source_revision_id IN (?, ?)", (rev_a, rev_b))
    kept_before = _q(root, "SELECT * FROM item_result WHERE source_revision_id = ? ORDER BY item_id", rev_a)
    assert len(kept_before) == 2
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    rep = _ingest(root, 3500)
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL", rev_b)[0][0] == 0
    assert rep.reclaim_transactions == 3                                   # pass (i): one per mark
    assert _q(root, "SELECT * FROM item_result WHERE source_revision_id = ? ORDER BY item_id", rev_a) == kept_before   # untouched, marks too
    assert _q(root, "SELECT COUNT(*) FROM item_result WHERE withdrawn_by_revision_id IS NOT NULL AND source_revision_id <> ?", rev_a)[0][0] == 0


def test_a_visible_row_is_never_deleted(tmp_path, monkeypatch):
    """The revision becomes the head between its plan and the DELETE: the batch's EXISTS clause deletes nothing, the count check rolls the
    batch back, and the revision is planned again later."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))
    calls = []

    def hook(ing, plan):
        if not calls:
            ing.conn.execute("UPDATE source_revision SET superseded_by_revision_id = NULL WHERE revision_id = ?", (rev_a,))
        calls.append(plan.table)
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", hook)
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        ing.reclaim(synth.t(4000))
        assert calls and ing.report.reclaim_rows_deleted == 0
        assert conn.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev_a,)).fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM item_reclaim_ledger").fetchone()[0] == 0
        assert conn.execute("SELECT reclaimed FROM source_revision WHERE revision_id = ?", (rev_a,)).fetchone()[0] == 0


def test_hold_mode_deletes_nothing_and_rebuild_ignores_the_mode(tmp_path):
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    assert _rows(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1} and _reclaimed(root, rev_a) == 0
    rep = _ingest(root, 3500)
    assert rep.reclaim_mode == "hold" and rep.reclaim_rows_deleted == 0
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)                                        # the mode file is the live root's state: a rebuild runs normal
    with DB.reading(out) as b:
        assert b.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev_a,)).fetchone()[0] == 0
    os.remove(PR.ReclaimMode.state_path(root))                             # mode normal: the next cycle prunes, and live equals the rebuild
    rep2 = _ingest(root, 3600)
    assert rep2.reclaim_mode == "normal" and rep2.reclaim_rows_deleted == 7
    _rebuild_equal(root, tmp_path)


def test_a_rebuilds_statistics_describe_its_final_item_tables(tmp_path):
    """A rebuild deletes the superseded revisions' rows at its very end, after recreating its indexes (which gathers the statistics):
    the item tables' sqlite_stat1 rows must count the rows the rebuilt file holds, not the ones the deletion removed."""
    root, _, _, rev_a, _ = _superseded(tmp_path)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    with DB.reading(out) as conn:
        assert conn.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", (rev_a,)).fetchone()[0] == 0
        for t in PR.PRUNE_TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            stats = {idx: int(str(stat).split()[0]) for idx, stat in conn.execute("SELECT idx, stat FROM sqlite_stat1 WHERE tbl = ?", (t,))}
            assert stats and set(stats.values()) == {n}, (t, n, stats)


def test_an_invalid_mode_file_deletes_nothing_and_says_so(tmp_path):
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    with open(PR.ReclaimMode.state_path(root), "w") as fh:
        fh.write('{"format": "reclaim-mode/1", "mode": "cohort", "manifests": [{"dir": "/nonexistent", "manifest_sha256": "x"}]}\n')
    rep = _ingest(root, 3500)
    assert rep.reclaim_stopped == "mode" and rep.reclaim_rows_deleted == 0 and _rows(root, rev_a)["item_result"] == 3


def test_an_error_inside_reclaim_is_named_and_the_live_cycle_completes(tmp_path, monkeypatch):
    """Reclaim runs after the cycle's fold committed: an unexpected exception in it is recorded by name (a `reclaim_error` metrics line,
    `reclaim_stopped = "error"` in the cycle's record) and the cycle's closing steps still run; the next cycle deletes as usual."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))

    def broken(*a, **kw):
        raise RuntimeError("synthetic reclaim defect")
    monkeypatch.setattr(PR, "revision_plan", broken)
    rep = _ingest(root, 3500)
    assert rep.reclaim_stopped == "error" and rep.completed_utc is not None and _rows(root, rev_a)["item_result"] == 3
    assert H.read_cycle_state(root)["reclaim_stopped"] == "error"
    with open(os.path.join(root, "state", "metrics.jsonl"), encoding="utf-8") as fh:
        errors = [json.loads(ln) for ln in fh if '"reclaim_error"' in ln]
    assert errors and "synthetic reclaim defect" in errors[-1]["error"]
    monkeypatch.undo()
    rep2 = _ingest(root, 3600)
    assert rep2.reclaim_stopped is None and _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}


def _wal(path):
    try:
        return os.path.getsize(path + "-wal")
    except OSError:
        return 0


def test_wal_admission_with_a_pinned_reader(tmp_path, monkeypatch):
    """A reader holds a snapshot across five reclaim invocations with a tiny WAL bound: the WAL never exceeds the bound plus one batch,
    every invocation after the first writes nothing and reports ``wal``; once the reader lets go, reclaim proceeds and the TRUNCATE
    checkpoint empties the WAL."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))
    path = DB.db_path(root)
    with DB.open_or_create(path) as conn:
        DB.wal_checkpoint_truncate(conn)
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    monkeypatch.setattr(H, "RECLAIM_WAL_MAX_BYTES", 1)
    reader = DB.connect(path, readonly=True)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM item_result").fetchone()          # a snapshot pinned: no checkpoint can pass it
    batch_wal = []
    try:
        for k in range(5):
            with DB.open_or_create(path) as conn:
                before = _wal(path)
                ing = H.Ingestor(conn, root)
                ing.reclaim(synth.t(4000 + k), wal_guard=True, min_free=0)
                batch_wal.append(_wal(path) - before)
                if k == 0:
                    assert ing.report.reclaim_rows_deleted <= 1               # the first batch may land (the WAL was empty) ...
                else:
                    assert ing.report.reclaim_rows_deleted == 0 and ing.report.reclaim_stopped == "wal"   # ... then nothing while pinned
                assert ing.report.reclaim_stopped == "wal"
        assert all(d <= max(batch_wal[0], 0) for d in batch_wal[1:])
    finally:
        reader.execute("ROLLBACK")
        reader.close()
    monkeypatch.setattr(H, "RECLAIM_WAL_MAX_BYTES", 1024 ** 3)
    with DB.open_or_create(path) as conn:
        ing = H.Ingestor(conn, root)
        ing.reclaim(synth.t(4100), wal_guard=True, min_free=0)
        assert ing.report.reclaim_stopped is None and _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
        busy, _, _ = DB.wal_checkpoint_truncate(conn)
        assert busy == 0
    assert _wal(path) == 0


def test_disk_admission_stops_reclaim_below_the_floor(tmp_path, monkeypatch):
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))
    monkeypatch.setattr(H.DB, "free_bytes", lambda p: 10)
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        ing.reclaim(synth.t(4000), wal_guard=True, min_free=1000)
        assert ing.report.reclaim_stopped == "disk" and ing.report.reclaim_rows_deleted == 0
    monkeypatch.setattr(H.DB, "free_bytes", lambda p: 10 ** 15)
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        ing.reclaim(synth.t(4001), wal_guard=True, min_free=1000)
        assert ing.report.reclaim_stopped is None and ing.report.reclaim_rows_deleted == 7


def test_budget_stop_and_resume(tmp_path, monkeypatch):
    """A stop lands MID-CARRIER: the half-deleted carrier is in the ledger with its running totals; a fold into the revision before the next
    pass rehydrates it in full; the next pass deletes it again; the end equals a rebuild."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    seen = []

    def hook(ing, plan):
        seen.append(plan.table)
        return "budget" if len(seen) > 2 else None                        # two batches, then the stop
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", hook)
    rep = _ingest(root, 3500)
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", None)
    assert rep.reclaim_stopped == "budget" and rep.reclaim_rows_deleted == 2
    half = _q(root, "SELECT table_name, expected_rows, rows_deleted FROM item_reclaim_ledger WHERE revision_id = ?", rev_a)
    assert half and half[0][1] == 3 and half[0][2] == 2                   # 2 of 3 rows of the first carrier, accounted for
    assert _reclaimed(root, rev_a) == 0
    _plain_item(root, rev_a, "q1", 0.02, 1000, "000001-late.jsonl")       # a fold into the revision before the next pass
    rep2 = _ingest(root, 3600)
    assert rep2.reclaim_rows_rehydrated == 2                               # the half-deleted carrier, whole again
    assert rep2.reclaim_stopped is None and _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}   # and pruned again, whole
    _rebuild_equal(root, tmp_path, seed=7)
    _rebuild_equal(root, tmp_path, seed=13)


def test_the_budget_stops_reclaim_between_batches(tmp_path, monkeypatch):
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    clock = [0.0]

    def fake_monotonic():                                                  # every reading one second later than the last
        clock[0] += 1.0
        return clock[0]
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        with monkeypatch.context() as m:
            m.setattr(H.time, "monotonic", fake_monotonic)
            ing.reclaim(synth.t(3650), budget_s=4.0)
        assert ing.report.reclaim_stopped == "budget" and 0 < ing.report.reclaim_rows_deleted < 7
    rep = _ingest(root, 3700)                                              # the next pass resumes (no budget in a test cycle's way)
    assert rep.reclaim_stopped is None and _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    _rebuild_equal(root, tmp_path)


def test_the_reclaim_budget_default_is_the_codes_fallback(tmp_path):
    """``[harvest] reclaim_budget_s`` is an explicit default in the operator's config, equal to the fallback the cycle uses when the
    config is torn, and the rendered default config carries it."""
    from sisteer.registry import config as CFG
    assert CFG.DEFAULTS["harvest"]["reclaim_budget_s"] == 600 == H.RECLAIM_BUDGET_S
    assert "reclaim_budget_s = 600" in CFG.render_default(str(tmp_path))
    assert CFG.load(str(tmp_path)).get("harvest", {}).get("reclaim_budget_s") == 600


def test_static_every_v_object_reads_item_tables_only_through_visibility():
    """Every v_* object that reads an item table RAW carries the visibility predicate once per raw read (or reads only native
    rows, ``source_revision_id = ''``); every other object reads them through those. So deleting an invisible row changes no reader."""
    raw = re.compile(r"\b(?:FROM|JOIN)\s+(item_result|item_flags|run_degeneration)\b(?![\w.])")
    visible = "sr_v.accepted = 1 AND sr_v.superseded_by_revision_id IS NULL"
    readers = {}
    for name in S.VIEW_NAMES:
        sql = S.VIEW_SQL[name]
        hits = raw.findall(sql)
        if not hits:
            continue
        readers[name] = len(hits)
        if sql.count(visible) >= len(hits):
            continue
        assert name == "v_paper_inputs_head" and "ir.source_revision_id = ''" in sql and len(hits) == 1, (name, hits)
    assert set(readers) == {"v_run_degeneration_current", "v_item_result_current", "v_item_flags_current", "v_item_result_provenances",
                            "v_item_pointer", "v_paper_inputs_head"}, readers
    assert "withdrawn_by_revision_id" not in S.VIEW_SQL["v_paper_inputs_head"]


def test_force_accept_of_a_pending_superseded_revision_is_unchanged(tmp_path):
    """A run-drop hold of a PENDING revision a later accepted read superseded: ``revision accept --force`` makes it the head; its rows were
    never deletable (not accepted), and the outgoing head's rows go once it is superseded."""
    import test_registry_run_set_drop_guard as RSD
    root, store_root, ad = RSD._setup(tmp_path, 6)
    RSD._pass(root, ad, RSD._t(2000))
    rev1 = RSD._head(root)
    ad.refuse_keys = {f"store_{i:03d}" for i in range(2, 6)}
    ad.EMISSION_VERSION = 2
    rep2, ing2 = RSD._pass(root, ad, RSD._t(3000))
    pend = rep2.results[0].revision_id
    assert ing2.revisions_held_run_drop == 1 and RSD._head(root) == rev1
    n_pend = _rows(root, pend)["item_result"]
    assert n_pend == 2
    ad.refuse_keys = set()
    ad.EMISSION_VERSION = 3
    rep3, _ = RSD._pass(root, ad, RSD._t(4000))
    rev3 = RSD._head(root)
    assert rev3 not in (rev1, pend) and _rows(root, rev1)["item_result"] == 0              # the first head, superseded, pruned
    assert _rows(root, pend)["item_result"] == n_pend                                       # pending and superseded: kept
    assert cli.main(["--root", root, "revision", "accept", "--revision-id", pend, "--reason", "deliberate (synthetic)", "--force"]) == 0
    _ingest(root)
    assert RSD._head(root) == pend and _rows(root, pend)["item_result"] == n_pend
    assert _rows(root, rev3)["item_result"] == 0
    _rebuild_equal(root, tmp_path)


def test_unauthentic_carriers_are_kept(tmp_path, monkeypatch):
    """A carrier the current code builds differently (a changed row builder) is kept as unauthentic (digest); a partly displaced one as
    unauthentic (count); both with their rows untouched."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))
    real = H.item_row_prefix

    def changed(plan, v, rfid):
        row = real(plan, v, rfid)
        if plan.table == "item_flags":
            i = plan.columns.index("response_tokens")
            row = row[:i] + ((row[i] or 0) + 1,) + row[i + 1:]
        return row
    monkeypatch.setattr(H, "item_row_prefix", changed)
    plans = {p.table: p for p in _plans(root, rev_a)}
    assert plans["item_flags"].klass == PR.UNAUTHENTIC_DIGEST and plans["item_result"].klass == PR.DELETABLE
    _ingest(root, 3500)
    assert _rows(root, rev_a) == {"item_result": 0, "item_flags": 3, "run_degeneration": 0}
    assert _reclaimed(root, rev_a) == 1                                    # nothing deletable is left: the kept carrier is a class, not work


def test_a_carrier_whose_file_holds_a_record_of_its_table_the_validator_refuses_is_kept(tmp_path, monkeypatch):
    """A plain file carries two records of one key of an accepted revision: the winner and an older duplicate the validator refuses. The
    accepted record reproduces the stored row exactly, but rehydration re-parses the whole file and refuses it, so a deletion could never
    be undone: the carrier is kept (``unauthentic_refused``, naming the refused record), its row survives reclaim, and the kept carrier is
    a class, not work."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    loser = 0.123
    real = VAL.validate_record

    def validate(rec, where, sha=None):
        p = rec.get("payload") or {}
        if rec.get("record_type") == "item_result" and p.get("value") == loser:
            return VAL.Malformed("value", "refused by a later rule (synthetic)", where, "item_result", sha or "")
        return real(rec, where, sha) if sha is not None else real(rec, where)
    monkeypatch.setattr(VAL, "validate_record", validate)
    base = _item_record(root, rev_a, "q0")["payload"]
    _write(root, "late", "000001-dup.jsonl", [synth.rec("item_result", dict(base, value=loser), synth.t(3100)),
                                              synth.rec("item_result", dict(base, value=0.77), synth.t(3200))])
    _ingest(root, 3300)
    (rfid,), = _q(root, "SELECT record_file_id FROM item_result WHERE source_revision_id = ? AND value = 0.77", rev_a)
    os.remove(PR.ReclaimMode.state_path(root))
    _ingest(root, 3500)
    assert _q(root, "SELECT value FROM item_result WHERE source_revision_id = ? AND record_file_id = ?", rev_a, rfid) == [(0.77,)]
    assert _reclaimed(root, rev_a) == 1 and not _ledger(root, rev_a).get("item_result")
    (p,) = [p for p in _plans(root, rev_a) if (p.table, p.record_file_id) == ("item_result", rfid)]
    assert p.klass == PR.UNAUTHENTIC_REFUSED and "the current validator refuses" in p.detail, (p.klass, p.detail)


def _archive_then_delete(tmp_path, name):
    """A superseded revision archived (prune_archive, the one-time cohort's tool) while reclaim was held, then deleted in normal mode."""
    from sisteer.registry import prune_archive as PA
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold", name=name)
    out = str(tmp_path / f"{name}-archive")
    PA.archive(DB.db_path(root), root, out)
    os.remove(PR.ReclaimMode.state_path(root))
    _ingest(root, 3500)
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    return root, rev_a, rev_b, out


def _refuse_q1_of(rev):
    real = VAL.validate_record

    def validate(rec, where, sha=None):
        p = rec.get("payload") or {}
        if rec.get("record_type") == "item_result" and p.get("source_revision_id") == rev and p.get("item_id") == "q1":
            return VAL.Malformed("value", "refused by a later rule (synthetic)", where, "item_result", sha or "")
        return real(rec, where, sha) if sha is not None else real(rec, where)
    return validate


def test_validator_change_then_marker_only_reactivation_holds_and_never_drops_rows(tmp_path, monkeypatch):
    from sisteer.registry import prune_archive as PA
    root, rev_a, rev_b, out = _archive_then_delete(tmp_path, "w1")
    monkeypatch.setattr(VAL, "validate_record", _refuse_q1_of(rev_a))
    # a marker-only bundle that re-activates rev_a (the shape `revision accept` writes: the marker file alone, a newer observed_utc)
    _bundle(root, rev_a, 4000, observed_i=4000, files=lambda fn: "source_revision" in fn)
    rep = _ingest(root, 4001)
    assert rep.revisions_held == 1 and _head(root) == rev_b                # held: the old head stays visible
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}           # nothing written
    assert _ledger(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    hold = _q(root, "SELECT missing_parent_table, missing_parent_id, json_extract(extra_json, '$.kind'), json_extract(extra_json, '$.blocked_on') "
                    "FROM blocked_records WHERE record_type = 'source_revision' AND json_extract(extra_json, '$.revision_id') = ?", rev_a)
    assert hold and hold[0][:3] == ("item_reclaim_ledger", rev_a, H.REHYDRATION_KIND), hold
    assert "rehydration of" in hold[0][3] and "the current validator refuses" in hold[0][3], hold[0][3]
    # the plain-file variant (write point 3): a revision-tagged row of rev_a is blocked on the ledger, nothing written
    _plain_item(root, rev_a, "q2", 0.5, 1000, "000001-late.jsonl")
    _ingest(root, 4101)
    assert _rows(root, rev_a)["item_result"] == 0
    assert _q(root, "SELECT missing_parent_table, missing_parent_id FROM blocked_records WHERE record_type = 'item_result'") == [("item_reclaim_ledger", rev_a)]
    # the exact remedy: the archived bytes, digest-checked against the ledger; the held activation then proceeds
    before = {t: _q(root, f"SELECT COUNT(*) FROM {t}")[0][0] for t in PR.PRUNE_TABLES}
    res = PA.rehydrate(root, rev_a, archive_dirs=[out])
    assert res["rows_restored"] == 7 and not _ledger(root, rev_a)
    with DB.reading(DB.db_path(root)) as conn:
        for t in PR.PRUNE_TABLES:
            rows = [tuple(r) for r in conn.execute(f"SELECT {', '.join(PR.columns_of(t))} FROM {t} WHERE source_revision_id = ?", (rev_a,))]
            arch = [r for r in PA.iter_archive_rows(out, t) if r[PR.columns_of(t).index("source_revision_id")] == rev_a]
            assert PR.Digest.of_rows(t, rows, PR.BIND_EXCLUDE) == PR.Digest.of_rows(t, arch, PR.BIND_EXCLUDE), t   # exact bytes back
    assert {t: _q(root, f"SELECT COUNT(*) FROM {t}")[0][0] for t in PR.PRUNE_TABLES} == {t: before[t] + n for t, n in
                                                                                          {"item_result": 3, "item_flags": 3, "run_degeneration": 1}.items()}
    _ingest(root, 4201, retry_held=True)
    assert _head(root) == rev_a and _q(root, "SELECT COUNT(*) FROM v_item_result_current WHERE source_revision_id = ?", rev_a)[0][0] == 3
    assert not _q(root, "SELECT 1 FROM blocked_records WHERE missing_parent_table = 'item_reclaim_ledger'")


def test_a_changed_parser_then_reactivation_holds_and_accept_reparse_restores_the_current_codes_rows(tmp_path, monkeypatch):
    from sisteer.registry import prune_archive as PA
    root, rev_a, rev_b, _ = _archive_then_delete(tmp_path, "w2")
    real = H.item_row_prefix

    def changed(plan, v, rfid):                                            # a later parser builds item_result's value differently
        row = real(plan, v, rfid)
        if plan.table == "item_result":
            i = plan.columns.index("value")
            row = row[:i] + (round((row[i] or 0) + 0.01, 6),) + row[i + 1:]
        return row
    monkeypatch.setattr(H, "item_row_prefix", changed)
    _bundle(root, rev_a, 4000, observed_i=4000, files=lambda fn: "source_revision" in fn)
    rep = _ingest(root, 4001)
    assert rep.revisions_held == 1 and _head(root) == rev_b and _rows(root, rev_a)["item_result"] == 0
    assert "digest mismatch" in _q(root, "SELECT json_extract(extra_json, '$.blocked_on') FROM blocked_records WHERE record_type = 'source_revision' "
                                         "AND json_extract(extra_json, '$.revision_id') = ?", rev_a)[0][0]
    res = PA.rehydrate(root, rev_a, accept_reparse=True, reason="the corrected parser's values stand (synthetic)")
    assert res["rows_restored"] == 7 and not _ledger(root, rev_a)
    vals = sorted(v for (v,) in _q(root, "SELECT value FROM item_result WHERE source_revision_id = ?", rev_a))
    assert vals == pytest.approx(sorted(round(x + 0.01, 6) for x in (0.2, 0.8, 0.4)))        # the current code's rows
    _ingest(root, 4101, retry_held=True)
    assert _head(root) == rev_a


def test_rehydration_authenticates_partial_deletions(tmp_path, monkeypatch):
    """A fold arrives into a carrier a stop left half-deleted: present + deleted reconstruct the original, rehydration restores it exactly
    (row for row equal to the hold run's), and the end equals a rebuild."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    original = {t: _q(root, f"SELECT * FROM {t} WHERE source_revision_id = ? ORDER BY 1, 2, 3, 4", rev_a) for t in PR.PRUNE_TABLES}
    os.remove(PR.ReclaimMode.state_path(root))
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    n = []
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", lambda ing, plan: (n.append(1), "budget" if len(n) > 1 else None)[1])
    _ingest(root, 3500)
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", None)
    led = _q(root, "SELECT table_name, expected_rows, rows_deleted FROM item_reclaim_ledger WHERE revision_id = ?", rev_a)
    assert len(led) == 1 and led[0][2] == 1 and led[0][1] == 3
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        assert ing._rehydrate_before_write(rev_a, synth.t(3600)) == 1
        restored = {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} WHERE source_revision_id = ? ORDER BY 1, 2, 3, 4", (rev_a,))]
                    for t in PR.PRUNE_TABLES}
    assert restored == {t: [tuple(r) for r in rows] for t, rows in original.items()}
    assert not _ledger(root, rev_a) and _reclaimed(root, rev_a) == 0
    _ingest(root, 3700)
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    _rebuild_equal(root, tmp_path)


def _insert_statements(trace, table):
    """The INSERT statements SQLite ran into ``table``, read from the connection's trace. Python reports a statement when it starts AND
    again, as the same expanded text, at every trigger firing inside it, so one statement is one run of identical lines (the rows of
    two statements differ, so two statements never make one run)."""
    prefix = f"INSERT INTO {table} ("
    runs, last = 0, None
    for line in trace:
        if line.startswith(prefix) and line != last:
            runs += 1
        last = line
    return runs


def test_a_chain_move_rehydrates_inside_the_record_savepoint_in_multi_row_statements(tmp_path, monkeypatch):
    """A plain-file ``source_revision`` record re-heads a revision whose rows reclaim deleted; the harvester rehydrates it INSIDE the
    record's savepoint, on its temp_store=MEMORY connection, where every INSERT statement's release walks the whole in-memory sub-journal
    the savepoint has built -- one statement per row was quadratic in the rows there. restore_chunk writes multi-row statements: the
    INSERT statements per carrier, counted from SQLite's trace (not timed), are at most ceil(rows / p) + log2(p), p the largest power of
    two whose rows fit the variable limit (lowered here, so a carrier of 100 rows spans many statements). The rows come back exactly as
    the hold run kept them, and the end equals a rebuild."""
    n, limit = 100, 200
    root, store_root, ad = _world(tmp_path / "w", scores=[i / 100 for i in range(n)], mode="hold", store_root=tmp_path / "stores")
    rev_a = _read(root, ad, 2000)
    _store(store_root, "s1", "store_1", LORA, [1 - i / 100 for i in range(n)])
    rev_b = _read(root, ad, 3000)
    assert rev_b != rev_a and _head(root) == rev_b
    original = {t: _q(root, f"SELECT * FROM {t} WHERE source_revision_id = ? ORDER BY 1, 2, 3, 4", rev_a) for t in PR.PRUNE_TABLES}
    os.remove(PR.ReclaimMode.state_path(root))
    _ingest(root, 3500)
    assert _rows(root, rev_a) == {t: 0 for t in PR.PRUNE_TABLES}
    assert _ledger(root, rev_a) == {"item_result": n, "item_flags": n, "run_degeneration": 1}
    real, calls = PR.rehydrate, []

    def traced(ingestor, revision_id, now):
        conn = ingestor.conn
        callers = {f.function for f in inspect.stack(0)}
        temp_store = conn.execute("PRAGMA temp_store").fetchone()[0]
        old = conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)
        trace = []
        conn.set_trace_callback(trace.append)
        try:
            return real(ingestor, revision_id, now)
        finally:
            conn.set_trace_callback(None)
            conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old)
            calls.append((revision_id, conn.in_transaction, temp_store, "_apply_validated" in callers, trace))
    monkeypatch.setattr(PR, "rehydrate", traced)
    _plain_marker(root, rev_a, 4000, 4000, "000001-restate.jsonl")
    rep = _ingest(root, 4001)
    assert rep.blocked == 0 and rep.reclaim_rows_rehydrated == 2 * n + 1 and _head(root) == rev_a
    # one rehydration, in the record's own fold (its savepoint, inside the chunk's transaction), on a temp_store=MEMORY (2) connection
    assert [c[:4] for c in calls] == [(rev_a, True, 2, True)], [c[:4] for c in calls]
    trace = calls[0][4]
    for t, rows in (("item_result", n), ("item_flags", n), ("run_degeneration", 1)):
        p = 1 << ((limit // len(PR.columns_of(t))).bit_length() - 1)
        bound = -(-rows // p) + p.bit_length() - 1
        k = _insert_statements(trace, t)
        # one statement per row (k == rows) exceeds the bound for both 100-row carriers
        assert 0 < k <= bound and (rows == 1 or bound < rows), (t, k, p, bound)
    assert {t: _q(root, f"SELECT * FROM {t} WHERE source_revision_id = ? ORDER BY 1, 2, 3, 4", rev_a) for t in PR.PRUNE_TABLES} == original
    assert not _ledger(root, rev_a)
    _rebuild_equal(root, tmp_path)


def _scratch_upsert():
    """A scratch table with the item tables' shape where it matters here: a BEFORE INSERT trigger that RAISE(ABORT)s (a parent check), and
    :func:`db.upsert_sql`'s whole-row upsert guarded like the item fold (the greater (w, sha) wins; a tie on both adopts the smaller f)."""
    c = sqlite3.connect(":memory:", isolation_level=None)
    c.execute("CREATE TABLE parent (id TEXT PRIMARY KEY)")
    c.executemany("INSERT INTO parent VALUES (?)", [("p0",), ("p1",)])
    c.execute("CREATE TABLE t (k TEXT NOT NULL, p TEXT NOT NULL, w INTEGER, sha TEXT, f INTEGER, v REAL, PRIMARY KEY (k, p))")
    c.execute("CREATE TRIGGER t_fk BEFORE INSERT ON t WHEN NOT EXISTS (SELECT 1 FROM parent WHERE id = NEW.p) "
              "BEGIN SELECT RAISE(ABORT, 't.p: no parent'); END")
    sql = DB.upsert_sql("t", ("k", "p", "w", "sha", "f", "v"), ("k", "p"),
                        guard="(excluded.w, excluded.sha) > (t.w, t.sha) OR (excluded.w = t.w AND excluded.sha = t.sha AND excluded.f < t.f)")
    return c, sql


@pytest.mark.parametrize("per", [1, 3, 8, 256])
def test_multi_row_restore_statements_end_where_one_statement_per_row_ends(monkeypatch, per):
    """restore_chunk's multi-row statements (``prune._insert_rows``) against one statement per row (``executemany``, as restore_chunk
    wrote before), on random rows whose keys repeat -- within one statement and across statement boundaries -- over rows already present:
    the same table at every seed. A row the trigger refuses raises IntegrityError either way (the caller's savepoint rolls back)."""
    monkeypatch.setattr(PR, "RESTORE_ROWS_PER_STATEMENT", per)

    def row(rng):
        return (f"k{rng.randrange(10)}", rng.choice(("p0", "p1")), rng.randrange(3), rng.choice("ab"), rng.randrange(3), rng.random())
    for seed in range(40):
        rng = random.Random(seed)
        present = [row(rng) for _ in range(rng.randrange(8))]
        rows = [row(rng) for _ in range(rng.randrange(1, 70))]
        ends = []
        for write in (lambda c, sql: c.executemany(sql, rows), lambda c, sql: PR._insert_rows(c, sql, rows)):
            c, sql = _scratch_upsert()
            c.executemany(sql, present)
            c.execute("BEGIN")
            write(c, sql)
            ends.append(c.execute("SELECT * FROM t ORDER BY k, p").fetchall())
            c.execute("COMMIT")
        assert ends[0] == ends[1], seed
    refused = [("k1", "p0", 1, "a", 0, 0.5)] * 5 + [("k2", "no-such-parent", 1, "a", 0, 0.5)] + [("k3", "p1", 1, "a", 0, 0.5)] * 5
    for write in (lambda c, sql: c.executemany(sql, refused), lambda c, sql: PR._insert_rows(c, sql, refused)):
        c, sql = _scratch_upsert()
        with pytest.raises(sqlite3.IntegrityError, match="no parent"):
            write(c, sql)
    c, sql = _scratch_upsert()
    with pytest.raises(ValueError, match="a row of 5 values"):
        PR._insert_rows(c, sql, [("k1", "p0", 1, "a", 0, 0.5), ("k1", "p0", 1, "a", 0)])
    with pytest.raises(ValueError, match="not a one-row VALUES upsert"):
        PR._insert_rows(c, "INSERT INTO t SELECT * FROM t WHERE true", [("k1",)])


def test_restore_statement_sizes_are_powers_of_two_that_cover_the_rows():
    """Every statement size is a power of two at most ``per`` (so a table meets at most log2(p) + 1 shapes, whatever its row counts), the
    sizes sum to the rows, and there are at most ceil(n / p) + log2(p) of them."""
    for per in (1, 2, 3, 9, 256, 1000):
        p = 1 << (per.bit_length() - 1)
        shapes = set()
        for n in range(0, 3000):
            sizes = PR._statement_sizes(n, per)
            assert sum(sizes) == n and all(s & (s - 1) == 0 and s <= p for s in sizes), (per, n)
            assert len(sizes) <= -(-n // p) + p.bit_length() - 1, (per, n, sizes)
            shapes.update(sizes)
        assert len(shapes) == p.bit_length(), (per, shapes)


@pytest.mark.parametrize("change", ["truncate", "replace"])
def test_a_resumed_carrier_whose_file_changed_is_missing_and_keeps_its_remaining_rows(tmp_path, monkeypatch, change):
    """A stop leaves a carrier half-deleted; its ledger row carries the expected rows, so the next pass of the same code does not re-parse
    it. Its file is then cut short or replaced (one byte more, which parses to the same rows): the next pass still checks the file's
    logical sha256 against the ingested one, classifies the carrier ``missing``, deletes none of its remaining rows and leaves the
    ledger's count where the stop left it."""
    root, store_root, ad, rev_a, rev_b = _superseded(tmp_path, "hold")
    os.remove(PR.ReclaimMode.state_path(root))
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    n = []
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", lambda ing, plan: (n.append(1), "budget" if len(n) > 1 else None)[1])
    _ingest(root, 3500)
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", None)
    (table, rfid, uri, expected, deleted), = _q(root, "SELECT table_name, record_file_id, file_uri, expected_rows, rows_deleted "
                                                     "FROM item_reclaim_ledger WHERE revision_id = ?", rev_a)
    assert (expected, deleted) == (3, 1)
    remaining = _q(root, f"SELECT COUNT(*) FROM {table} WHERE source_revision_id = ? AND record_file_id = ?", rev_a, rfid)[0][0]
    assert remaining == 2
    path = H.resolve_uri(root, uri)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        data = fh.read()
    with opener(path, "wb") as fh:
        fh.write(data[:data.index(b"\n") + 1] if change == "truncate" else data + b"\n")
    resumed = [p for p in _plans(root, rev_a) if (p.table, p.record_file_id) == (table, rfid)]
    assert len(resumed) == 1 and resumed[0].klass == PR.MISSING and "is not the ingested" in resumed[0].detail, resumed
    with DB.open_or_create(DB.db_path(root)) as conn:
        ing = H.Ingestor(conn, root)
        ing.reclaim(synth.t(3600))
    assert _q(root, f"SELECT COUNT(*) FROM {table} WHERE source_revision_id = ? AND record_file_id = ?", rev_a, rfid)[0][0] == 2
    assert _q(root, "SELECT rows_deleted FROM item_reclaim_ledger WHERE revision_id = ? AND table_name = ? AND record_file_id = ?",
              rev_a, table, rfid) == [(1,)]
    # the missing carrier keeps its revision due: its file returning is no item write, so nothing else would plan the revision again
    assert _reclaimed(root, rev_a) == 0
    with opener(path, "wb") as fh:                                        # the original bytes return (an operator restores the file)
        fh.write(data)
    with DB.open_or_create(DB.db_path(root)) as conn:
        H.Ingestor(conn, root).reclaim(synth.t(3700))
    assert _q(root, f"SELECT COUNT(*) FROM {table} WHERE source_revision_id = ? AND record_file_id = ?", rev_a, rfid)[0][0] == 0
    assert _q(root, "SELECT rows_deleted FROM item_reclaim_ledger WHERE revision_id = ? AND table_name = ? AND record_file_id = ?",
              rev_a, table, rfid) == [(3,)]                               # resumed from the ledger and finished
    assert _reclaimed(root, rev_a) == 1
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the digest and the encoding the ledger and the archive share
# --------------------------------------------------------------------------------------------

def test_the_row_encoding_round_trips_and_the_digest_is_additive():
    vals = [None, 0, -1, 2 ** 63 - 1, -(2 ** 63), 0.0, -0.0, 5e-324, 1.5, float("inf"), "", "ünï code", b"\x00\xff", 3]
    enc = PR.encode_row(vals)
    back = PR.decode_row(enc)
    assert [repr(v) for v in back] == [repr(v) for v in vals]
    assert enc.decode("ascii")                                             # ASCII bytes, whatever the text
    t = "run_degeneration"
    cols = PR.columns_of(t)
    rows = [tuple(i if c != "rate" else i / 3 for c in cols) for i in range(5)]
    a = PR.Digest.of_rows(t, rows[:2])
    b = PR.Digest.of_rows(t, rows[2:])
    assert (a + b) == PR.Digest.of_rows(t, rows) and (a + b).copy().sub(b) == a
    assert PR.Digest.from_hex(5, (a + b).hex()) == a + b
    i = cols.index("ingest_seq")
    moved = [r[:i] + (99,) + r[i + 1:] for r in rows]
    assert PR.Digest.of_rows(t, moved, PR.BIND_EXCLUDE) == PR.Digest.of_rows(t, rows, PR.BIND_EXCLUDE)
    assert PR.Digest.of_rows(t, moved) != PR.Digest.of_rows(t, rows)


def test_the_mode_file_round_trips_and_a_cohort_without_a_verified_archive_is_refused(tmp_path):
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, "state"))
    assert PR.ReclaimMode.from_state(root).kind == "normal"
    PR.ReclaimMode.hold().write_state(root, synth.t(1))
    assert PR.ReclaimMode.from_state(root).kind == "hold"
    PR.ReclaimMode.normal().write_state(root, synth.t(2))
    assert not os.path.exists(PR.ReclaimMode.state_path(root)) and PR.ReclaimMode.from_state(root).kind == "normal"
    with pytest.raises(PR.ModeStateInvalid):
        PR.ReclaimMode.cohort([str(tmp_path / "no-archive")])
    with pytest.raises(PR.ModeStateInvalid):
        PR.ReclaimMode.cohort([])
