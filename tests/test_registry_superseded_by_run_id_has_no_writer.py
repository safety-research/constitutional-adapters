"""Pinned here (the remaining backlog stays a backlog): NO fold writes ``run.superseded_by_run_id`` or ``run.status_ruling_id``.

A producer retires a run with a ``run_state superseded`` record whose ``extra`` names the successor and the ruling (the future reducer
rule's INPUT); the owner files a ``supersede`` ruling scoped to the run whose ``target_json`` names the successor. Neither
moves the two columns: the run folds ``lifecycle_state = superseded`` -- which every state-keyed reader keys on (v_run_current's
lifecycle rule, the pending readers, v_superseded: ``lifecycle_state = 'superseded' OR superseded_by_run_id IS NOT NULL``) -- with
``superseded_by_run_id`` NULL and ``status_ruling_id`` NULL; the extra rides the run_state_event row; the ruling folds as its own row;
and a ``run_patch`` naming either column is refused by name (neither is in RUN_PATCHABLE_FIELDS).

The other direction of the backlog -- the reducer filling ``superseded_by_run_id`` from a supersede ruling's ``target_json`` -- is NOT
done here on purpose: it changes folded rows at rebuild (710 ``supersede`` / ``run`` rulings carried ``target_json.superseded_by_run_id``
in the live registry on 2026-10-05, their runs' column NULL), which is a STOP change: its own window and migration. This file pins
today's rule so that change is a decision someone makes, never a drift. Synthetic corpus only (sisteer.registry.synth).
"""
from __future__ import annotations

import json
import os

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

RULING_ID = "placeholder:test-supersede-run"


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def test_neither_a_superseded_run_state_nor_a_supersede_ruling_sets_the_successor_columns(tmp_path):
    root = _root(tmp_path)
    old_id, old = synth.run_records("retired", 100, through="generated")
    new_id, new = synth.run_records("successor", 200, through="generated")
    _write(root, "runs", "000001-runs.jsonl", old + new)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    # the producer's retirement: the successor and the ruling under extra (the readers' input, never a column of the run)
    retire = synth.rec("run_state", {"run_id": old_id, "state": "superseded", "event_utc": synth.t(300), "evidence_kind": "ruling",
                                     "evidence_uri": f"ruling://{RULING_ID}", "extra": {"superseded_by_run_id": new_id, "ruling_id": RULING_ID}},
                       synth.t(300))
    # the ruling: scope the retired run, target the successor (the shape of the live `placeholder:` supersede rulings)
    ruling = synth.rec("ruling", {"ruling_id": RULING_ID, "round_id": "r_synth", "action": "supersede", "scope_kind": "run",
                                  "scope_json": {"run_id": old_id}, "target_json": {"superseded_by_run_id": new_id},
                                  "reason": "the successor re-ran the same point on the fixed tree", "ruled_utc": synth.t(299),
                                  "ruled_by": "test", "source_ref": "test"}, synth.t(301))
    _write(root, "rulings", "000001-rulings.jsonl", [ruling])
    _write(root, "runs", "000002-retire.jsonl", [retire])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        row = conn.execute("SELECT lifecycle_state, superseded_by_run_id, status_ruling_id FROM run WHERE run_id = ?", (old_id,)).fetchone()
        assert tuple(row) == ("superseded", None, None), tuple(row)
        ev = json.loads(conn.execute("SELECT extra_json FROM run_state_event WHERE run_id = ? AND state = 'superseded'", (old_id,)).fetchone()[0])
        assert ev["superseded_by_run_id"] == new_id and ev["ruling_id"] == RULING_ID, ev
        run_extra = json.loads(conn.execute("SELECT COALESCE(extra_json, '{}') FROM run WHERE run_id = ?", (old_id,)).fetchone()[0])
        assert "superseded_by_run_id" not in run_extra and "ruling_id" not in run_extra, run_extra   # a state event's extra stays on the event
        rul = conn.execute("SELECT action, scope_kind, scope_json, target_json FROM ruling WHERE ruling_id = ?", (RULING_ID,)).fetchone()
        assert tuple(rul)[:2] == ("supersede", "run") and json.loads(rul[2]) == {"run_id": old_id} and json.loads(rul[3]) == {"superseded_by_run_id": new_id}
        # the state-keyed readers retire the row on lifecycle_state alone, successor column NULL
        sup = conn.execute("SELECT kind, superseded_by, ruling_id, state FROM v_superseded WHERE kind = 'run' AND id = ?", (old_id,)).fetchall()
        assert [tuple(r) for r in sup] == [("run", None, None, "superseded")], [tuple(r) for r in sup]
        cur = conn.execute("SELECT is_current FROM v_run_current WHERE run_id = ?", (old_id,)).fetchone()
        assert cur is not None and cur[0] == 0, cur
        succ = conn.execute("SELECT lifecycle_state, superseded_by_run_id, status_ruling_id FROM run WHERE run_id = ?", (new_id,)).fetchone()
        assert tuple(succ) == ("generated", None, None), tuple(succ)
    finally:
        conn.close()
    # a run_patch naming either column is refused by name, before anything folds
    for col, val in (("superseded_by_run_id", new_id), ("status_ruling_id", RULING_ID)):
        bad = synth.rec("run_patch", {"run_id": old_id, "fields": {col: val}}, synth.t(310))
        v = VAL.validate_record(bad, "<memory>")
        assert isinstance(v, VAL.Malformed) and v.field == col and "RUN_PATCHABLE_FIELDS" in v.reason, (col, v)
