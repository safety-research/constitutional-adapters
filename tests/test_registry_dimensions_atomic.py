"""The dimension seeds land atomically for readers (2026-09-22). ``registry dimensions load`` publishes one seed revision as
``records/dimensions/<rev>/`` by ONE directory rename, and the harvester applies the directory in ONE transaction, before the
cycle's plain files (``harvest.Ingestor.apply_dimension_bundle``). Before this change the nine files were plain files, folded
in ≤ 50-file chunk transactions interleaved by stamp with every other pending file of the cycle — hundreds on the live
database, 60–140 s per chunk — so a reader between two commits saw the new revision's ``model`` / ``eval`` / ``item_set`` rows
beside the previous revision's ``instrument`` / ``decode_policy`` / ``object`` rows (reproduced on a scratch root with 45
older-stamped plain files: the files landed at ranks 56–60 and 61–64 in two transactions). Pinned here: the routing, the single
transaction (traced on the writer's own connection) placed first in the cycle, a concurrent reader that never sees a mixed
dimension set nor fewer paper rows than before, the one-rename publication and ``--force``'s append, the CLI report's timings,
the rebuild's equality, and the lock refusal that publishes nothing.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import threading
import time

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import dimensions as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = os.path.join(ROOT, "configs", "registry")
PROD = synth.producer("tests.registry_dimensions_atomic")
REV1, REV2 = "a" * 16, "b" * 16
#: the dimension tables the bundle writes whose statements the trace test locates (si_text is left out: the plain files beside
#: the bundle are si_text records too)
_DIM_INSERT = re.compile(r"^\s*INSERT INTO (model|eval|item_set|judge|instrument|object)\b", re.IGNORECASE)


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    return root


def _dimension_revision(root: str, rev: str, i0: int, marker: str):
    """``synth.seed_dimensions`` written the way the loader publishes a revision: one ``<seq>-<type>.jsonl`` per record type
    under ``records/dimensions/<rev>/``; ``marker`` lands in the ``notes`` of the model / eval / judge / instrument rows and in
    the item set's ``dataset_version``, so a reader can tell which revision each table shows."""
    by = {}
    for r in synth.seed_dimensions(i0):
        rt = str(r["record_type"])
        p = dict(r["payload"])
        if rt in ("model", "eval", "judge", "instrument"):
            p["notes"] = marker
        if rt == "item_set":
            p["dataset_version"] = f"synthetic/syntheval@{marker}"
        by.setdefault(rt, []).append(synth.rec(rt, p, str(r["written_utc"]), PROD))
    d = os.path.join(root, "records", "dimensions", rev)
    return [R.write_records(os.path.join(d, R.record_file_name(seq, rt)), recs) for seq, (rt, recs) in enumerate(by.items(), start=1)]


def _plain_files(root: str, n: int, i0: int = 0):
    """``n`` one-record plain files stamped from ``synth.t(i0)`` — the other pending files of a cycle, sorted BEFORE a later-stamped
    bundle by the ledger order, which is what split the nine files across two chunk transactions before this change."""
    for i in range(n):
        R.write_records(os.path.join(root, "records", "other", f"{i:06d}-si_text.jsonl"),
                        [synth.rec("si_text", {"si_id": f"other_si_{i}", "si_text_sha256": "%064x" % (i + 1), "si_kind": "irrelevant_control",
                                               "source_uri": "synthetic://other"}, synth.t(i0 + i), PROD)])


def _dim_state(c) -> str:
    """Which revision the dimension tables show: the marker when every table agrees, else ``mixed:<markers>``."""
    vals = {c.execute("SELECT notes FROM model WHERE model_id = ?", (synth.MODEL_ID,)).fetchone()[0],
            c.execute("SELECT notes FROM eval WHERE eval_id = ?", (synth.EVAL_ID,)).fetchone()[0],
            c.execute("SELECT notes FROM judge WHERE judge_id = ?", (synth.JUDGE_ID,)).fetchone()[0],
            c.execute("SELECT notes FROM instrument WHERE instrument_id = ?", (synth.INSTRUMENT_ID,)).fetchone()[0],
            c.execute("SELECT dataset_version FROM item_set WHERE item_set_id = ?", (synth.ITEM_SET_ID,)).fetchone()[0].rsplit("@", 1)[1]}
    return vals.pop() if len(vals) == 1 else "mixed:" + ",".join(sorted(str(v) for v in vals))


def _paper_rows(c) -> int:
    return int(c.execute("SELECT COUNT(*) FROM v_paper_long").fetchone()[0])


# --------------------------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------------------------


def test_split_work_routes_a_seed_revision_directory_and_leaves_other_dimension_files_plain():
    root = "/r"
    work = [(f"/r/records/dimensions/{REV1}/000001-model.jsonl", "s1"),
            ("/r/records/c/000001-run.jsonl", None),
            (f"/r/records/dimensions/{REV1}/000006-instrument.jsonl.gz", "s6"),
            ("/r/records/dimensions/seed/000001-dimensions.jsonl", None),          # synth.write_corpus's plain seed file: not a revision
            (f"/r/records/adapters/ad/{'c' * 16}/000001-run.jsonl", None),
            (f"/r/records/dimensions/{REV2}/000001-model.jsonl", None)]
    plain, dirs, dims = H.split_work(root, work)
    assert [p for p, _ in plain] == ["/r/records/c/000001-run.jsonl", "/r/records/dimensions/seed/000001-dimensions.jsonl"]
    assert dirs == [f"/r/records/adapters/ad/{'c' * 16}"]
    assert dims == [(f"/r/records/dimensions/{REV1}", [(f"/r/records/dimensions/{REV1}/000001-model.jsonl", "s1"),
                                                       (f"/r/records/dimensions/{REV1}/000006-instrument.jsonl.gz", "s6")]),
                    (f"/r/records/dimensions/{REV2}", [(f"/r/records/dimensions/{REV2}/000001-model.jsonl", None)])]
    assert H.dimension_revision_of_uri(f"records/dimensions/{REV1}/000001-model.jsonl") == REV1
    assert H.dimension_revision_of_uri("records/dimensions/seed/000001-dimensions.jsonl") is None
    assert H.dimension_revision_of_uri(f"records/adapters/dimensions/{REV1}/000001-model.jsonl") is None


# --------------------------------------------------------------------------------------------
# one transaction, first in the cycle
# --------------------------------------------------------------------------------------------


def test_seed_revision_folds_in_one_transaction_before_the_plain_files(tmp_path):
    root = _root(tmp_path)
    _plain_files(root, 45)                       # stamped before the bundle: the pre-change chunker put 45 + 5 of the files in one transaction
    files = _dimension_revision(root, REV1, 100, "rev1")
    conn = DB.open_or_create(DB.db_path(root))
    stmts = []
    conn.set_trace_callback(stmts.append)
    try:
        rep = H.ingest_once(root, conn=conn, workers=0, min_free=0)
    finally:
        conn.set_trace_callback(None)
        conn.close()
    assert rep.refused == 0 and rep.blocked == 0 and rep.dimension_transactions == 1 and rep.dimension_write_s > 0
    # every write to a dimension table the bundle carries sits in ONE BEGIN…COMMIT, and it is the cycle's FIRST transaction
    seg, seg_of = -1, []
    for s in stmts:
        if s.strip().upper().startswith("BEGIN"):
            seg += 1
        seg_of.append(seg)
    dim_segments = {seg_of[i] for i, s in enumerate(stmts) if _DIM_INSERT.match(s)}
    assert dim_segments == {0}, dim_segments
    assert seg >= 2                              # then the plain chunk, the arrival facts, the dense pass
    # the bundle's files took the cycle's first ledger ranks, in file order
    conn = DB.connect(DB.db_path(root), readonly=True)
    first = [r[0] for r in conn.execute("SELECT file_uri FROM record_ingest ORDER BY ingest_seq LIMIT ?", (len(files),))]
    assert first == [H.logical_uri(root, f) for f in files]
    assert conn.execute("SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE ? AND status = 'ok'", (f"records/dimensions/{REV1}/%",)).fetchone()[0] == len(files)
    conn.close()
    # the metrics log names the transaction
    kinds = [json.loads(line)["kind"] for line in open(os.path.join(root, "state", "metrics.jsonl"))]
    assert kinds[0] == "dimensions" and kinds.count("dimensions") == 1


def test_a_second_cycle_refolds_only_the_changed_file_still_in_one_transaction(tmp_path):
    root = _root(tmp_path)
    files = _dimension_revision(root, REV1, 100, "rev1")
    assert H.ingest_once(root, workers=0, min_free=0).dimension_transactions == 1
    # one file appended to (a --force re-publication appends): the bundle re-applies just that file, in one transaction
    model_file = next(f for f in files if f.endswith("-model.jsonl"))
    R.write_records(model_file, [synth.rec("model", dict(next(r for r in synth.seed_dimensions(100) if r["record_type"] == "model")["payload"],
                                                          notes="rev1b"), synth.t(500), PROD)])
    rep = H.ingest_once(root, workers=0, min_free=0)
    # the changed file alone (its arrival fact was written by the first cycle); its two unchanged records read as duplicates
    assert rep.dimension_transactions == 1 and rep.files == 1 and rep.ok == 1 and rep.duplicates == 2 and rep.refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT notes FROM model WHERE model_id = ?", (synth.FIT_MODEL_ID,)).fetchone()[0] == "rev1b"
    conn.close()


# --------------------------------------------------------------------------------------------
# the reader's side
# --------------------------------------------------------------------------------------------


def test_reader_sees_the_old_or_the_new_dimension_set_and_never_fewer_paper_rows(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _dimension_revision(root, REV1, 100, "rev1")
    rid, recs = synth.run_records("k0", 200, n_items=20)
    synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in recs])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    ro = DB.connect(DB.db_path(root), readonly=True)
    n_before = _paper_rows(ro)
    assert n_before >= 1 and _dim_state(ro) == "rev1"
    # revision 2 beside 45 older-stamped plain files: the shape that split the nine files across two chunk transactions before
    _plain_files(root, 45)
    _dimension_revision(root, REV2, 1000, "rev2")
    inside = []

    def hook(dim_dir):
        # INSIDE the bundle's transaction, after the fold and before the commit: a second connection still reads the old set, whole
        assert dim_dir.endswith(os.path.join("records", "dimensions", REV2))
        c = DB.connect(DB.db_path(root), readonly=True)
        for _ in range(5):
            inside.append((_dim_state(c), _paper_rows(c)))
            time.sleep(0.05)
        c.close()
    monkeypatch.setattr(H, "DIMENSION_APPLY_HOOK", hook)
    samples = []
    stop = threading.Event()

    def poll():
        c = DB.connect(DB.db_path(root), readonly=True)
        while not stop.is_set():
            samples.append((_dim_state(c), _paper_rows(c)))
            time.sleep(0.01)
        c.close()
    th = threading.Thread(target=poll)
    th.start()
    try:
        rep2 = H.ingest_once(root, workers=0, min_free=0)
    finally:
        stop.set()
        th.join()
    assert rep2.dimension_transactions == 1 and rep2.refused == 0 and rep2.blocked == 0
    assert len(inside) == 5 and all(state == "rev1" and n == n_before for state, n in inside), inside
    states = {state for state, _ in samples}
    assert states <= {"rev1", "rev2"} and "rev1" in states, states          # never a mix; the poller ran during the transaction
    assert all(n >= n_before for _, n in samples), [n for _, n in samples if n < n_before]
    assert _dim_state(ro) == "rev2" and _paper_rows(ro) >= n_before
    ro.close()


# --------------------------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------------------------


def test_publish_is_one_directory_rename_and_force_appends(tmp_path):
    root = _root(tmp_path)
    bundle = D.build_seed_bundle(root, SEEDS, producer=PROD, now="2026-09-16T12:00:00.000Z", verify_si=False)
    assert not os.path.exists(D.seed_revision_dir(root, bundle.revision_id))
    paths = D.publish_seed_records(root, bundle)
    out = D.seed_revision_dir(root, bundle.revision_id)
    assert len(paths) == 9 and sorted(os.listdir(out)) == sorted(os.path.basename(p) for p in paths)
    assert not os.path.exists(os.path.join(root, "staging", "dimensions"))          # nothing staged is left behind
    assert D.publish_seed_records(root, bundle) == []                                # already on the log
    n1 = sum(1 for _ in R.read_record_file(paths[0]))
    assert n1 == bundle.counts()["model"]
    forced = D.publish_seed_records(root, bundle, force=True)
    assert forced == paths and sum(1 for _ in R.read_record_file(paths[0])) == 2 * n1     # appended, never rewritten
    # a file gzipped after ingest is appended as a further gzip member under its logical name
    with open(paths[0], "rb") as src, gzip.open(paths[0] + ".gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove(paths[0])
    forced2 = D.publish_seed_records(root, bundle, force=True)
    assert paths[0] + ".gz" in forced2 and sum(1 for _ in R.read_record_file(paths[0] + ".gz")) == 3 * n1
    # write_seed_records is the composition
    bundle2, paths2 = D.write_seed_records(root, SEEDS, producer=PROD, now="2026-09-16T12:00:00.000Z", verify_si=False)
    assert bundle2.revision_id == bundle.revision_id and paths2 == []


def test_publish_gives_way_to_a_concurrent_publication_of_the_same_revision(tmp_path, monkeypatch):
    root = _root(tmp_path)
    bundle = D.build_seed_bundle(root, SEEDS, producer=PROD, now="2026-09-16T12:00:00.000Z", verify_si=False)
    out = D.seed_revision_dir(root, bundle.revision_id)
    real_rename = os.rename

    def racing_rename(src, dst):
        if dst == out:
            os.makedirs(out)
            R.write_records(os.path.join(out, "000001-model.jsonl"), bundle.records["model"])   # another load published first
        return real_rename(src, dst)
    monkeypatch.setattr(os, "rename", racing_rename)
    assert D.publish_seed_records(root, bundle) == []
    assert sorted(os.listdir(out)) == ["000001-model.jsonl"] and not os.path.exists(os.path.join(root, "staging", "dimensions"))


# --------------------------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------------------------


def test_cli_load_reports_one_dimension_transaction_and_its_lock_time(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    assert cli.main(["dimensions", "load", "--seeds", SEEDS, "--no-verify-si"]) == 0
    rep = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert rep["files_written"] == 9 and rep["ingest"]["refused"] == 0 and rep["ingest"]["blocked"] == 0
    assert rep["ingest"]["dimension_transactions"] == 1
    assert 0 < rep["ingest"]["dimension_write_s"] <= rep["lock_hold_s"] and rep["build_s"] >= 0
    # the same revision again: nothing published, no dimension transaction, the lock still taken and released
    assert cli.main(["dimensions", "load", "--seeds", SEEDS, "--no-verify-si"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["files_written"] == 0 and "already on the record log" in rep["note"] and rep["ingest"]["dimension_transactions"] == 0


def test_cli_load_refuses_when_the_harvester_lock_is_held_and_publishes_nothing(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "reg")
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["init"]) == 0
    with DB.write_lock(root):
        assert cli.main(["dimensions", "load", "--seeds", SEEDS, "--no-verify-si", "--lock-timeout", "0.2"]) == 1
    err = capsys.readouterr().err
    assert "harvester lock" in err
    assert not os.path.isdir(os.path.join(root, "records", "dimensions"))
    assert not os.path.exists(os.path.join(root, "staging", "dimensions"))


# --------------------------------------------------------------------------------------------
# rebuild
# --------------------------------------------------------------------------------------------


def test_rebuild_replays_a_seed_revision_directory_equal_to_live(tmp_path):
    root = _root(tmp_path)
    _plain_files(root, 5)
    _dimension_revision(root, REV1, 100, "rev1")
    rid, recs = synth.run_records("k0", 200, n_items=20)
    synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in recs])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and rep.dimension_transactions == 1
    into = str(tmp_path / "rebuilt.sqlite")
    rrep = H.rebuild(root, into, workers=0)
    assert rrep.dimension_transactions == 1 and rrep.refused == 0
    live = DB.connect(DB.db_path(root), readonly=True)
    rebuilt = DB.connect(into, readonly=True)
    diff = H.verify_equal(live, rebuilt)
    assert not {k: v for k, v in diff.items() if v}, diff
    live.close()
    rebuilt.close()
