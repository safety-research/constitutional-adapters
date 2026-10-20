"""Why `strength_ladder` and `relu_native` had four accepted revisions each and no row in `v_source_status` (2026-09-18), and the fix.

Neither adapter opts out of the status fact: a FRESH read of either over its fixture writes a `source_status` file into the
revision directory and the row lands in the table (the first two tests pin that for both). The live gap was the runner's
RE-ACTIVATION path: both adapters' heads were committed before the status fact existed (no status file in those revision
directories), and every later pass found their inputs touched but their content identity unchanged, so `reactivate_committed`
wrote a fresh marker and nothing else -- the head stayed current and invisible. The third test reproduces exactly that with
the synthetic store adapter (a revision published without a status row, then a touched input) and pins the fix: the pass
re-reads a published, accepted revision that has no status row and files one.

Fixtures are invented (tests/registry_declared_fixtures.py, the synthetic store adapter); nothing here reads a real store.
"""
from __future__ import annotations

import json
import os
import time

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import relu_native as RN
from sisteer.registry.adapters import runner as RUN
from sisteer.registry.adapters import strength_ladder as SL

import registry_declared_fixtures as F  # noqa: E402
from test_registry_adapter_pass import SyntheticStoreAdapter, _pass_and_ingest, _setup, _store  # noqa: E402


def _registry(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    H.ingest_once(root, workers=0, min_free=0)
    return root


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _status_files(root, adapter_name, rev):
    d = os.path.join(root, "records", "adapters", adapter_name, rev)
    return sorted(f for f in os.listdir(d) if "source_status" in f)


def test_strength_ladder_read_writes_a_source_status_record(tmp_path):
    root = _registry(tmp_path)
    fx = F.build_strength(str(tmp_path / "strength"))
    decl = SL.StrengthRecordDecl(fx["path"], fx["source"], ("const_lora", "diff_lora"))
    ad = SL.StrengthLadderAdapter(records=(decl,))
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    (res,) = rep.results
    assert res.accepted, res.error
    assert _status_files(root, "strength_ladder", res.revision_id), "the revision directory carries no source_status file"
    rows = _q(root, "SELECT lane_source_legacy, adapter, source_id, owner, source_revision_id FROM source_status WHERE adapter = 'strength_ladder'")
    assert rows == [("strength_lane", "strength_ladder", "strength_ladder:strength_lane", "Strength", res.revision_id)]
    (extra,) = _q(root, "SELECT extra_json FROM source_status WHERE adapter = 'strength_ladder'")
    facts = json.loads(extra[0])
    assert "roster" in facts and "row_refusals" in facts and facts["refuse"] == []


def test_relu_native_read_writes_a_source_status_record(tmp_path):
    root = _registry(tmp_path)
    allowance, _ = RN.passthrough_anchor_allowance()
    fx = F.build_relu_native(str(tmp_path / "relu"), allowance)
    ad = RN.ReluNativeAdapter(rs=fx["rs"], paired_root=fx["paired"])
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    (res,) = rep.results
    assert res.accepted, res.error
    assert _status_files(root, "relu_native", res.revision_id)
    rows = _q(root, "SELECT lane_source_legacy, adapter, source_id, owner, source_revision_id FROM source_status WHERE adapter = 'relu_native'")
    assert rows == [("relu_native", "relu_native", "relu_native:llama8b", "registry", res.revision_id)]
    (extra,) = _q(root, "SELECT extra_json FROM source_status WHERE adapter = 'relu_native'")
    facts = json.loads(extra[0])
    assert facts["paired_root"]["status"] == "present" and "admitted" in facts


def test_a_published_revision_without_a_status_row_gets_one_when_its_inputs_are_touched(tmp_path, monkeypatch):
    """The live gap: a head committed before the status fact existed (no status row) whose inputs are later
    touched. Before the fix the pass re-stamped the watermarks and filed nothing; now it re-reads the committed
    revision for its facts."""
    root, store_root = _setup(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    # a publish predating the status fact: the runner of that day filed no status row
    monkeypatch.setattr(RUN, "_source_status_records", lambda *a, **k: [])
    rep, ing = _pass_and_ingest(root, ad, synth.t(2000))
    monkeypatch.undo()
    (res,) = rep.results
    assert res.accepted and ing.refused == 0
    assert _q(root, "SELECT revision_id, accepted FROM source_revision") == [(res.revision_id, 1)]
    assert _q(root, "SELECT COUNT(*) FROM source_status") == [(0,)] and _q(root, "SELECT COUNT(*) FROM v_source_status") == [(0,)]
    # the input is touched (same bytes, a later mtime): the content identity is unchanged, so the pass re-activates the head
    p = os.path.join(store_root, "s1", "store.json")
    st = os.stat(p)
    os.utime(p, (st.st_atime + 5, st.st_mtime + 5))
    rep2, ing2 = _pass_and_ingest(root, ad, synth.t(3000))
    (res2,) = rep2.results
    assert res2.accepted and res2.revision_id == res.revision_id and res2.reread, res2
    assert ing2.refused == 0
    rows = _q(root, "SELECT lane_source_legacy, adapter, source_id, ok, source_revision_id FROM source_status")
    assert rows == [("synthetic", "synthetic_store", "synthetic_store:s1", 1, res.revision_id)]
    assert _q(root, "SELECT lane_source_legacy, adapter, ok FROM v_source_status") == [("synthetic", "synthetic_store", 1)]
    (extra,) = _q(root, "SELECT extra_json FROM source_status")
    facts = json.loads(extra[0])
    assert facts["reread"]["of_revision"] == res.revision_id and facts["reread"]["reactivated"] is False
    assert facts["reread"]["key_drift"] == {"keys_absent_from_read": 0, "keys_new_in_read": 0, "items_absent_from_read": 0, "items_new_in_read": 0}
    # a third touch: the head now HAS a status row, so the plain re-stamp path is taken again (no second status row)
    os.utime(p, (st.st_atime + 10, st.st_mtime + 10))
    rep3, _ = _pass_and_ingest(root, ad, synth.t(4000))
    assert rep3.results[0].accepted and not rep3.results[0].reread
    assert _q(root, "SELECT COUNT(*) FROM source_status") == [(1,)]
