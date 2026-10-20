"""Rebuild equality: build live in three batches (a correction record, a late older
update, a ruling expiry evaluation and a harvester run_ingest included), rebuild into a tmp DB FROM A
SHUFFLED COPY of the log, and diff every table except the arrival ledger — `ingest_seq`,
`run.ingested_utc`, `ruling.expired_utc` and `row_history` included. Also `registry rebuild --verify`.
"""
from __future__ import annotations

import os
import random
import shutil
import sqlite3

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth


def _write(root, sub, name, recs, gz=False):
    return R.write_records(os.path.join(root, "records", sub, name), recs, gz=gz)


def _shuffled_copy(root, dest, seed):
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(seed).shuffle(files)
    for f in files:
        rel = os.path.relpath(f, os.path.join(root, "records"))
        d = os.path.join(dest, "records", rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(f, d)


def test_rebuild_from_synthetic_log_equals_live(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    # batch 1: dimensions + two judged runs with items
    ids = synth.write_corpus(root, n_runs=2, n_items=20)
    H.ingest_once(root, workers=0, min_free=0)
    # batch 2: a correction record and a late older update
    m = synth.seed_dimensions()[2]
    corr = dict(m, written_utc=synth.t(700))
    corr["payload"] = dict(m["payload"], notes="corrected")
    late = synth.rec("run_patch", {"run_id": ids["run_ids"][0], "fields": {"job_name": "late-older"}}, synth.t(101))
    _write(root, "b2", "000001-b2.jsonl", [corr, late])
    H.ingest_once(root, workers=0, min_free=0)
    # batch 3: a self-retiring ruling whose evidence expires it (source_state present) → ruling_state fact
    ruling = synth.rec("ruling", {"ruling_id": "R_exp", "round_id": "r_synth", "action": "exclude", "scope_kind": "run",
                                  "scope_json": {"run_id": ids["run_ids"][1]}, "reason": "synthetic exclusion",
                                  "expires_when_json": {"kind": "served_id_absent", "source": "src", "input_uri": "synthetic:///export.parquet",
                                                        "member_key": "op_ids", "served_id": "old_op"},
                                  "ruled_utc": synth.t(800), "ruled_by": "user", "source_ref": "test"}, synth.t(800))
    state = synth.rec("source_state", {"source_id": "src", "input_uri": "synthetic:///export.parquet", "json_path": "op_ids", "found": 1,
                                       "path_resolves": 1, "current_members_json": ["new_op"], "observed_utc": synth.t(801),
                                       "input_sha256": "1" * 64}, synth.t(801))
    _write(root, "b3", "000001-b3.jsonl", [ruling, state])
    rep = H.ingest_once(root, workers=0, min_free=0)
    live = DB.connect(DB.db_path(root), readonly=True)
    assert live.execute("SELECT expired_utc FROM ruling WHERE ruling_id = 'R_exp'").fetchone()[0] is not None
    assert live.execute("SELECT COUNT(*) FROM run WHERE lifecycle_state = 'ingested'").fetchone()[0] == 2
    assert live.execute("SELECT job_name FROM run WHERE run_id = ?", (ids["run_ids"][0],)).fetchone()[0] == "late-older"
    # rebuild from a shuffled copy
    copy = str(tmp_path / "copy")
    _shuffled_copy(root, copy, seed=11)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(copy, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = H.verify_equal(live, rebuilt)
    assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    # the derived facts replayed byte-for-byte
    for sql in ("SELECT ingested_utc FROM run ORDER BY run_id", "SELECT expired_utc, evidence_state FROM ruling",
                "SELECT ingest_seq FROM record_ingest ORDER BY ingest_seq", "SELECT COUNT(*) FROM row_history"):
        assert live.execute(sql).fetchall() == rebuilt.execute(sql).fetchall(), sql
    live.close()
    rebuilt.close()


def test_rebuild_refuses_to_overwrite(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=2)
    out = str(tmp_path / "x.sqlite")
    H.rebuild(root, out, workers=0)
    import pytest
    with pytest.raises(FileExistsError):
        H.rebuild(root, out, workers=0)


def test_cli_rebuild_verify_reports_zero_diffs(tmp_path, capsys):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=5)
    H.ingest_once(root, workers=0, min_free=0)
    out = str(tmp_path / "rebuilt.sqlite")
    rc = cli.main(["--root", root, "rebuild", "--from", root, "--into", out, "--verify", "--workers", "0"])
    assert rc == 0
    assert "0 differing rows" in capsys.readouterr().out


def test_verify_detects_a_difference(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=5)
    H.ingest_once(root, workers=0, min_free=0)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    c = DB.connect(out)                                   # this code's connection: the fold-order fence refuses a run write by a bare sqlite3.connect
    c.execute("UPDATE run SET notes = 'tampered'")
    c.commit()
    c.close()
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    diff = H.verify_equal(a, b)
    assert diff["run"] == 2 and all(v == 0 for k, v in diff.items() if k != "run")
    a.close()
    b.close()


# --------------------------------------------------------------------------------------------
# the harvester's facts fold after the revision phase
# --------------------------------------------------------------------------------------------

def _late_carrier_world(tmp_path, derive_facts=False):
    """The R3 shape, synthetic: one run carried by two revisions of source A -- the FIRST held until a `computation` its metric cites
    lands with source B's revision, the SECOND free of it -- and the harvester's `run_ingest` stamped between them. Ledger order:
    A1 (held), the stamp, B1, A2. The live cycles fold it in that order (A1 activates with B1, the stamp folds right after it, A2's
    re-emitted `judged` is held as a regression) and the row reads `ingested`. A replay that folds the plain files before the revision
    directories blocks the stamp on its missing run, applies A2 in the revision phase (A1 still held: the source-major order puts A's
    directories before B's) and retries the stamp over A2's events alone -- `ingested` ahead of every `judged`, so the refold has no
    `judged_utc` and the upsert's INSERT half fires `trg_run_state_utc` (the 128 refusals of the R3 bootstrap). Returns
    ``(root, run_id, uris)``."""
    from sisteer.registry import identity as I
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    comp = I.computation_id("synthetic.reader_stats", "1", "none")
    p = synth.run_payload("k", i=1000)
    p.update(lifecycle_state="generated", started_utc=synth.t(1001), generated_utc=synth.t(1002), n_items_generated=3, n_items_judged=3,
             judge_id=synth.JUDGE_ID, store_completion_sha256=synth.SHA256_B, store_completion_artifact_uri="synthetic:///store/k/results.json")
    rid = str(p["run_id"])
    jid = I.judgment_id(rid, synth.JUDGE_ID)

    def revision(adapter, rev, when, recs, keys):
        d = os.path.join(root, "records", "adapters", adapter, rev)
        R.write_records(os.path.join(d, "000001-batch.jsonl"), recs)
        src = f"{adapter}:synth"
        marker = {"revision_id": rev, "source_id": src, "adapter": adapter, "observed_utc": synth.t(when),
                  "inputs_json": {"synthetic:///store": "0" * 64}, "accepted": 1}
        recs2 = [synth.rec("source_revision", marker, synth.t(when))]
        if keys:
            obs = {"source_id": src, "revision_id": rev, "table_name": "run", "observation_keys": [R.canonical_json(k) for k in keys]}
            recs2.append(synth.rec("observation", obs, synth.t(when)))
        R.write_records(os.path.join(d, "000002-source_revision.jsonl"), recs2)

    def carrier(when, computation=None):
        metric = {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": 1, "n": 3, "computed_by": "producer"}
        if computation:
            metric["computation_id"] = computation
        return [synth.rec("run", p, synth.t(when)),
                synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "n_items_judged": 3, "status": "complete"}, synth.t(when)),
                synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(1003), "evidence_kind": "completion_artifact"}, synth.t(when)),
                synth.rec("run_metric", metric, synth.t(when))]
    revision("srca", "a1a1a1a1a1a1a1a1", 2000, carrier(2000, computation=comp), [{"run_id": rid}])        # A1: waits for B1's computation
    stamp = H.write_harvester_record(root, "run_ingest", {"run_id": rid, "ingested_utc": synth.t(2043)}, synth.t(2043))
    revision("srcb", "b1b1b1b1b1b1b1b1", 3000, [synth.rec("computation", {"computation_id": comp, "algorithm": "synthetic.reader_stats", "version": "1",
                                                                          "seed_provenance": "none", "code_sha": "unverified"}, synth.t(3000))], [])
    revision("srca", "a2a2a2a2a2a2a2a2", 4000, carrier(4000), [{"run_id": rid}])                            # A2: the run re-emitted, nothing to wait for
    for when in (2500, 2600, 3500, 4500):
        rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(when), derive_facts=derive_facts)
        assert rep.refused == 0, rep.refusals
    return root, rid, H.logical_uri(root, stamp)


def _delivered(root):
    with DB.reading(DB.db_path(root)) as conn:
        return [r[0] for r in conn.execute("SELECT file_uri FROM record_delivery")]


def test_a_harvester_stamp_whose_first_carrier_activates_late_folds_on_replay_as_live_did(tmp_path):
    """The one-key reproduction of the R3 bootstrap's 128 refusals: live reads `ingested`; the replay must too, with no
    `run_ingest` refusal -- the facts fold after the revision phase and its blocked-queue pass (so the stamp meets a complete history and
    applies on its first attempt), and the reducer's fold-order evidence rule blocks rather than refuses whatever the order."""
    root, rid, stamp_uri = _late_carrier_world(tmp_path)
    live = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(live.execute("SELECT lifecycle_state, ingested_utc FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("ingested", synth.t(2043))
    assert live.execute("SELECT COUNT(*) FROM record_refusal").fetchone()[0] == 0
    assert live.execute("SELECT accepted FROM source_revision WHERE revision_id = 'a1a1a1a1a1a1a1a1'").fetchone()[0] == 1
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0, files=_delivered(root))
    rebuilt = DB.connect(out, readonly=True)
    assert tuple(rebuilt.execute("SELECT lifecycle_state, ingested_utc FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("ingested", synth.t(2043))
    assert rebuilt.execute("SELECT COUNT(*) FROM record_refusal WHERE record_type = 'run_ingest'").fetchone()[0] == 0
    assert rebuilt.execute("SELECT COUNT(*) FROM record_refusal").fetchone()[0] == 0
    # the stamp met its run and its judged evidence on the first attempt: folded `ok`, never blocked (live blocked it once, at the cycle
    # before A1 activated -- the first attempt's verdict is bookkeeping outside rebuild equality, as record_delivery is)
    assert tuple(rebuilt.execute("SELECT status, n_ok, n_blocked FROM record_delivery WHERE file_uri = ?", (stamp_uri,)).fetchone()) == ("ok", 1, 0)
    assert rebuilt.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    diff = {k: v for k, v in H.verify_equal(live, rebuilt).items() if v}
    assert not diff, diff
    live.close()
    rebuilt.close()


def test_partition_harvester_files_keeps_each_lists_order_and_splits_on_the_harvester_prefix(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    paths = [os.path.join(root, u) for u in ("records/x/000001-run.jsonl", "records/harvester/2026-01-01/run_ingest-aaaaaaaaaaaa.jsonl",
                                             "records/adapters/ad/0123456789abcdef/000001-run.jsonl", "records/harvester/2026-01-02/ruling_state-bbbbbbbbbbbb.jsonl",
                                             "rulings/r2_v1/000001-ruling.jsonl")]
    producer, facts = H.partition_harvester_files(root, [(p, None) for p in paths])
    assert [H.logical_uri(root, p) for p, _ in producer] == ["records/x/000001-run.jsonl", "records/adapters/ad/0123456789abcdef/000001-run.jsonl", "rulings/r2_v1/000001-ruling.jsonl"]
    assert [H.logical_uri(root, p) for p, _ in facts] == ["records/harvester/2026-01-01/run_ingest-aaaaaaaaaaaa.jsonl", "records/harvester/2026-01-02/ruling_state-bbbbbbbbbbbb.jsonl"]
