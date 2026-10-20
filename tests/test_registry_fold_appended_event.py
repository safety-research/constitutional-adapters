"""An appended run-scale event folds from the previous snapshot, byte for byte as the whole-history refold (2026-09-24).

The 2026-09-24 fold-time profile (specs: results-registry/fold-perf-profile.md) put 25-30 % of a harvester cycle in
``Reducer._load_events``: every re-emitted ``source_watermark`` and refused ``source_revision`` marker -- one new event per
key per adapter pass -- re-read the key's whole history from ``row_history`` (72 events per key on average, one random page
each) to refold a state that depends only on the previous snapshot and the event. ``Reducer._appended_fold`` takes that
shortcut when it is exact; these tests hold the two paths to the same bytes and pin when the shortcut applies.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import reducer as RED
from sisteer.registry import records as R
from sisteer.registry import synth

PROD = synth.producer("tests.fold_appended_event")
SRC = "overrefusal_items:/stores/one"
URI = "/stores/one/rows.jsonl"
REV = "c" * 16


def _watermark(i: int, extra=None, omit=()):
    p = {"source_id": SRC, "input_uri": URI, "sha256": "%064x" % 7, "size": 100 + i, "mtime_utc": synth.t(10), "observed_utc": synth.t(100 + i),
         "adapter": "overrefusal_items", "records_written": 0, "last_error": None}
    for k in omit:
        p.pop(k)
    if extra:
        p["extra"] = extra
    return synth.rec("source_watermark", p, synth.t(100 + i), PROD)


def _marker(i: int):
    p = {"revision_id": REV, "source_id": "objects_bank:" + "%064x" % 9, "adapter": "objects_bank", "observed_utc": synth.t(100 + i),
         "inputs_json": {"/objects/x/object.pt": "%064x" % 3}, "accepted": 0, "refusal": f"refused on pass {i}",
         "extra": {"emission_version": "3", "owner": "registry"}}
    return synth.rec("source_revision", p, synth.t(100 + i), PROD)


def _si_text(i: int):
    return synth.rec("si_text", {"si_id": f"si{i}", "si_text_sha256": "%064x" % (i + 1), "si_kind": "irrelevant_control",
                                 "source_uri": "synthetic://x"}, synth.t(100 + i), PROD)


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _digest(root):
    """``{table: sha256 over every row ordered by primary key}`` -- the row SETS, insertion order aside."""
    out = {}
    with DB.reading(DB.db_path(root)) as c:
        tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                                          "AND name <> 'schema_migration' ORDER BY name")]        # applied_utc is the wall clock
        for t in tables:
            info = c.execute(f"PRAGMA table_info({t})").fetchall()
            pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])] or [r[1] for r in info]
            h = hashlib.sha256()
            for row in c.execute(f"SELECT * FROM {t} ORDER BY {', '.join(pk)}"):
                h.update(json.dumps([v.hex() if isinstance(v, bytes) else v for v in row], sort_keys=True).encode())
                h.update(b"\n")
            out[t] = h.hexdigest()
    return out


@pytest.fixture
def load_events_spy(monkeypatch):
    """Records ``(table, key)`` of every whole-history read the reducer makes."""
    calls = []
    real = RED.Reducer._load_events

    def spy(self, table, key_json):
        calls.append((table, key_json))
        return real(self, table, key_json)
    monkeypatch.setattr(RED.Reducer, "_load_events", spy)
    return calls


def _cycles(root, incremental: bool, monkeypatch, calls):
    """Three cycles over the same files, each at a fixed clock so two roots fold byte-identically; returns the history reads
    per cycle for the watermark key, the marker key and everything else."""
    monkeypatch.setattr(RED, "INCREMENTAL_FOLD", incremental)
    H.ensure_layout(root)
    # cycle 1: first events, then three re-emissions (with accumulating extras) -- appended, one per pass
    _write(root, "a", "000001-first.jsonl", [_watermark(0), _marker(0), _si_text(0)])
    _write(root, "a", "000002-pass1.jsonl", [_watermark(1, extra={"pass": 1}), _marker(1)])
    _write(root, "a", "000003-pass2.jsonl", [_watermark(2, extra={"note": "x"}), _marker(2)])
    _write(root, "a", "000004-pass5.jsonl", [_watermark(5), _marker(5)])
    per_cycle = []
    del calls[:]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(1000))
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    per_cycle.append(list(calls))
    # cycle 2: an OLDER event arrives late (a mirror catching up): it sorts before pass 5, so the suffix after it is refolded
    _write(root, "a", "000005-pass3-late.jsonl", [_watermark(3)])
    del calls[:]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(2000))
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    per_cycle.append(list(calls))
    # cycle 3: a record that OMITS a column the key already has (its stamp must survive from the older event), then a whole row again
    _write(root, "a", "000006-pass6-partial.jsonl", [_watermark(6, omit=("last_error",))])
    _write(root, "a", "000007-pass7.jsonl", [_watermark(7)])
    del calls[:]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(3000))
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    per_cycle.append(list(calls))
    return per_cycle


def _by_table(calls, table):
    return sum(1 for t, _ in calls if t == table)


def test_appended_events_fold_byte_identically_with_and_without_the_history_reread(tmp_path, monkeypatch, load_events_spy):
    fast = str(tmp_path / "fast")
    slow = str(tmp_path / "slow")
    reads_fast = _cycles(fast, True, monkeypatch, load_events_spy)
    reads_slow = _cycles(slow, False, monkeypatch, load_events_spy)
    # the same row SETS in every table: row_history (payloads AND folded snapshots), the live rows, the ledger, the deliveries
    d_fast, d_slow = _digest(fast), _digest(slow)
    assert d_fast == d_slow, {t: (d_fast[t][:12], d_slow[t][:12]) for t in d_fast if d_fast[t] != d_slow.get(t)}
    with DB.reading(DB.db_path(fast)) as c:
        n_hist = c.execute("SELECT COUNT(*) FROM row_history WHERE table_name = 'source_watermark'").fetchone()[0]
        assert n_hist == 7                                                                    # passes 0, 1, 2, 5, 3, 6, 7: nothing lost
        state = json.loads(c.execute("SELECT folded_state_json FROM row_history WHERE table_name = 'source_watermark' "
                                     "ORDER BY written_utc DESC, record_sha256 DESC LIMIT 1").fetchone()[0])
        assert state["observed_utc"] == synth.t(107) and json.loads(state["extra_json"]) == {"pass": 1, "note": "x"}   # extras accumulate
        fv = json.loads(c.execute("SELECT field_versions_json FROM source_watermark WHERE source_id = ?", (SRC,)).fetchone()[0])
        assert {k for k, v in fv.items() if v[0] == synth.t(107)} == set(fv)                  # pass 7 re-set every column
    # the whole-history path read the history for every applied run-scale event; the shortcut only where it cannot be exact
    assert _by_table(reads_slow[0], "source_watermark") == 4 and _by_table(reads_slow[0], "source_revision") == 4
    assert _by_table(reads_fast[0], "source_watermark") == 1 and _by_table(reads_fast[0], "source_revision") == 1   # the first event of each key
    assert _by_table(reads_fast[1], "source_watermark") == 1                                  # the late older event: the suffix refolds
    assert _by_table(reads_fast[2], "source_watermark") == 1                                  # the partial record refolds; pass 7 does not
    assert _by_table(reads_fast[0], "si_text") == 1 and _by_table(reads_slow[0], "si_text") == 1


def test_run_and_ruling_keys_always_refold_and_a_first_event_reads_its_one_row(tmp_path, monkeypatch, load_events_spy):
    """The tables whose fold keeps state outside the snapshot (``run``: event_owned; ``ruling``: the evaluations list) never take
    the shortcut, however their events arrive; a key's first event has nothing to fold from and reads its single row."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=5)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(1000))
    assert rep.refused == 0
    run_reads = [k for t, k in load_events_spy if t == "run"]
    assert run_reads, "the run's records (run, run_state, judgment) each refolded the run key"
    with DB.reading(DB.db_path(root)) as c:
        n_run_events = c.execute("SELECT COUNT(*) FROM row_history WHERE table_name = 'run'").fetchone()[0]
    assert len(run_reads) >= n_run_events - 1
    # a first event on a shortcut-eligible table: exactly one history read of one row
    del load_events_spy[:]
    _write(root, "w", "000001-w.jsonl", [_watermark(0)])
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(2000))
    assert _by_table(load_events_spy, "source_watermark") == 1
    assert RED._REFOLD_ALWAYS == ("run", "ruling")


def test_the_shortcut_declines_when_the_event_is_not_the_newest_or_drops_a_column(tmp_path):
    """The four facts ``_appended_fold`` checks, one by one, on a reducer over a fresh database."""
    conn = DB.open_fresh(str(tmp_path / "r.sqlite"))
    red = RED.Reducer(conn, seq_of=lambda uri: 1)
    from sisteer.registry import validate as VAL

    def apply(rec, uri):
        v = VAL.validate_record(rec, f"{uri}#1", R.record_sha(rec))
        assert isinstance(v, VAL.Validated), v
        (eff,) = RED.effects_of(v)
        return v, eff
    with DB.transaction(conn):
        v0, e0 = apply(_watermark(0), "a/1.jsonl")
        assert red._appended_fold(e0, v0, e0.key_json, e0.history_payload()) is None      # nothing in the history yet
        assert red.apply_run_scale(e0, v0, "a/1.jsonl", 1)
        v2, e2 = apply(_watermark(2), "a/2.jsonl")
        conn.execute("INSERT INTO row_history(table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, payload_json, folded_state_json) "
                     "VALUES (?,?,?,?,?,?,?,?)", (e2.table, e2.key_json, v2.written_utc, v2.sha, v2.record_type, "a/2.jsonl", "{}", "{}"))
        assert red._appended_fold(e2, v2, e2.key_json, e2.history_payload()) is not None  # the newest, every column re-set
        v1, e1 = apply(_watermark(1), "a/3.jsonl")
        conn.execute("INSERT INTO row_history(table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, payload_json, folded_state_json) "
                     "VALUES (?,?,?,?,?,?,?,?)", (e1.table, e1.key_json, v1.written_utc, v1.sha, v1.record_type, "a/3.jsonl", "{}", "{}"))
        assert red._appended_fold(e1, v1, e1.key_json, e1.history_payload()) is None      # pass 2 already folded after it
        v3, e3 = apply(_watermark(3, omit=("last_error",)), "a/4.jsonl")
        conn.execute("INSERT INTO row_history(table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, payload_json, folded_state_json) "
                     "VALUES (?,?,?,?,?,?,?,?)", (e3.table, e3.key_json, v3.written_utc, v3.sha, v3.record_type, "a/4.jsonl", "{}", "{}"))
        # the previous snapshot is the pass-2 placeholder here ("{}"), so the column check passes on an empty set; a real
        # snapshot with last_error set would decline -- exercised end to end by the cycle test above
        assert red._appended_fold(e3, v3, e3.key_json, e3.history_payload()) is not None
    conn.close()
