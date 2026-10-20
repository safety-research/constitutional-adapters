"""Object lineage, first-class, and an object record that omits its status -- plan commit C6, the owner addenda of
2026-10-02 / 2026-10-03.

`object.parent_object_id` (FK, dependency-blocked at fold) and `object.named_role` (vocab.OBJECT_NAMED_ROLES) fold from the record's
own field when stated, else are LIFTED from `extra.parent_object_id` / `extra.named_role` (the keys producers wrote before the columns
existed); an explicit null resets; the parent equal to the object itself and a role outside the vocabulary refuse by name; the migration's
Python post-step lifts the existing rows' extra values ONCE, idempotently, to the bytes a fresh fold writes.

An `object` record may omit `status`: it folds normally when a row exists (the folded status and its stamp kept), is dependency-
blocked on its own row when none does (a mirror's statement waits for the producer's), a stated status still moves the column, an explicit
null is refused by name, and replay is equal (an omitted column is no statement on replay either).
"""
from __future__ import annotations

import json
import os
import random
import shutil

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import records as R
from sisteer.registry import reducer as RED
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V

PARENT = synth.OBJECT_SHA                 # the seeded fitted object: the parent of every child below
CHILD_A = "1" * 64
CHILD_B = "2" * 64
CHILD_C = "3" * 64
ORPHAN = "4" * 64
LATE_PARENT = "5" * 64


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    assert H.ingest_once(root, workers=0, min_free=0, now=synth.t(500)).refused == 0
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root, now):
    return H.ingest_once(root, workers=0, min_free=0, now=now)


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _row(root, oid):
    rows = _q(root, "SELECT * FROM object WHERE object_id = ?", oid)
    return rows[0] if rows else None


def _blocked(root):
    return _q(root, "SELECT record_type, missing_parent_table, missing_parent_id FROM blocked_records ORDER BY missing_parent_id")


#: a payload field to REMOVE (an omission), as opposed to None (an explicit null, a statement)
OMIT = object()


def _child(sha, i, **over):
    """An object record of a child checkpoint; ``over`` sets payload fields (None is an explicit null; OMIT removes the field)."""
    p = synth.object_lora(sha=sha, paper_label="constitution_lora", role="constitution")
    for k, val in over.items():
        if val is OMIT:
            p.pop(k, None)
        else:
            p[k] = val
    p.setdefault("extra", {})
    p["extra"] = dict(p["extra"], named_roles=[])
    return synth.rec("object", p, synth.t(i))


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = {k: v for k, v in H.verify_equal(a, b).items() if v}
        assert not diff, diff
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------------------------------------
# The fold
# --------------------------------------------------------------------------------------------


def test_lineage_folds_from_the_field_and_from_extra_and_an_explicit_null_resets(tmp_path):
    root = _root(tmp_path)
    a = _child(CHILD_A, 1000, parent_object_id=PARENT, named_role="epoch_1")                       # the columns, stated
    b = _child(CHILD_B, 1001, extra={"parent_object_id": PARENT, "named_role": "final"})           # the keys, parked under extra (the old shape)
    _write(root, "objs", "000001-objects.jsonl", [a, b])
    rep = _ingest(root, synth.t(2000))
    assert rep.refused == 0 and rep.blocked == 0
    ra, rb = _row(root, CHILD_A), _row(root, CHILD_B)
    assert (ra["parent_object_id"], ra["named_role"]) == (PARENT, "epoch_1")
    assert (rb["parent_object_id"], rb["named_role"]) == (PARENT, "final")
    fva, fvb = json.loads(ra["field_versions_json"]), json.loads(rb["field_versions_json"])
    assert fva["parent_object_id"] == fva["named_role"] == [synth.t(1000), R.record_sha(a)]
    assert fvb["parent_object_id"] == fvb["named_role"] == [synth.t(1001), R.record_sha(b)]      # the lifted column carries the lifting event's stamp
    assert json.loads(rb["extra_json"])["named_role"] == "final" and json.loads(rb["extra_json"])["named_roles"] == []   # the keys stay in extra
    assert "named_role" not in json.loads(ra["extra_json"])
    # the history payload keeps the record as validated: the lifted column is NOT in payload_json, it is in the folded state
    hist = _q(root, "SELECT payload_json, folded_state_json FROM row_history WHERE table_name = 'object' AND natural_key_json = ?", RED.canonical_json({"object_id": CHILD_B}))
    assert "named_role" not in json.loads(hist[0]["payload_json"]) and json.loads(hist[0]["payload_json"])["_extra"]["named_role"] == "final"
    assert json.loads(hist[0]["folded_state_json"])["named_role"] == "final"
    # an explicit null resets; a later record silent on the column leaves the reset standing (the lift reads the EVENT's extra, not the accumulation)
    _write(root, "objs", "000002-reset.jsonl", [_child(CHILD_B, 1100, named_role=None, extra={})])
    assert _ingest(root, synth.t(2100)).refused == 0
    rb = _row(root, CHILD_B)
    assert rb["named_role"] is None and rb["parent_object_id"] == PARENT and json.loads(rb["extra_json"])["named_role"] == "final"
    _write(root, "objs", "000003-silent.jsonl", [_child(CHILD_B, 1200, extra={})])
    assert _ingest(root, synth.t(2200)).refused == 0
    rb = _row(root, CHILD_B)
    assert rb["named_role"] is None and rb["parent_object_id"] == PARENT
    # a later record stating the role moves it again
    _write(root, "objs", "000004-role.jsonl", [_child(CHILD_B, 1300, extra={"named_role": "epoch_2"})])
    assert _ingest(root, synth.t(2300)).refused == 0
    assert _row(root, CHILD_B)["named_role"] == "epoch_2"
    _rebuild_equal(root, tmp_path)


def test_a_child_whose_parent_has_not_folded_is_dependency_blocked_from_the_field_and_from_extra(tmp_path):
    root = _root(tmp_path)
    _write(root, "objs", "000001-children.jsonl", [_child(CHILD_A, 1000, parent_object_id=LATE_PARENT, named_role="epoch_1"),
                                                   _child(CHILD_B, 1001, extra={"parent_object_id": LATE_PARENT, "named_role": "epoch_2"})])
    rep = _ingest(root, synth.t(2000))
    assert rep.refused == 0 and rep.blocked == 2
    assert _blocked(root) == [{"record_type": "object", "missing_parent_table": "object", "missing_parent_id": LATE_PARENT}] * 2
    assert _row(root, CHILD_A) is None and _row(root, CHILD_B) is None
    _write(root, "objs", "000002-parent.jsonl", [synth.rec("object", synth.object_lora(sha=LATE_PARENT, paper_label="control_lora", role="control"), synth.t(1002))])
    rep = _ingest(root, synth.t(3000))
    assert rep.refused == 0 and rep.unblocked == 2 and _blocked(root) == []
    assert _row(root, CHILD_A)["parent_object_id"] == LATE_PARENT and _row(root, CHILD_B)["parent_object_id"] == LATE_PARENT
    assert _row(root, CHILD_B)["named_role"] == "epoch_2"
    _rebuild_equal(root, tmp_path)


def test_a_role_outside_the_vocabulary_and_a_self_parent_are_refused_by_name_in_both_spellings():
    def verdict(rec):
        return VAL.validate_record(rec, "<t>", R.record_sha(rec))
    assert V.OBJECT_NAMED_ROLES == ("epoch_1", "epoch_2", "intermediate", "final")
    for rec in (_child(CHILD_A, 1, named_role="epoch_9"), _child(CHILD_A, 1, extra={"named_role": "epoch_9"})):
        out = verdict(rec)
        assert isinstance(out, VAL.Malformed) and out.field == "named_role" and "OBJECT_NAMED_ROLES" in out.reason
    for rec in (_child(CHILD_A, 1, parent_object_id=CHILD_A), _child(CHILD_A, 1, extra={"parent_object_id": CHILD_A})):
        out = verdict(rec)
        assert isinstance(out, VAL.Malformed) and out.field == "parent_object_id" and "own id" in out.reason
    out = verdict(_child(CHILD_A, 1, extra={"parent_object_id": 7}))
    assert isinstance(out, VAL.Malformed) and out.field == "parent_object_id" and "must be a string" in out.reason
    ok = verdict(_child(CHILD_A, 1, extra={"parent_object_id": PARENT, "named_role": "intermediate"}))
    assert isinstance(ok, VAL.Validated) and ("object", "object_id", PARENT) in VAL.parent_references(ok)      # the extra-carried parent is a dependency
    ok2 = verdict(_child(CHILD_A, 1, parent_object_id=PARENT))
    assert ("object", "object_id", PARENT) in VAL.parent_references(ok2)
    assert VAL.parent_references(verdict(_child(CHILD_A, 1, extra={"named_role": "final"}))) == [a for a in VAL.parent_references(verdict(_child(CHILD_A, 1))) ]
    # a stated top-level null outranks a stale extra parent, exactly as the fold does (lift_object_fields: the column stated, nothing lifted):
    # the record is no dependency on a parent the fold would not set
    cleared = verdict(_child(CHILD_A, 1, parent_object_id=None, extra={"parent_object_id": LATE_PARENT, "named_role": "final"}))
    assert isinstance(cleared, VAL.Validated) and cleared.payload["parent_object_id"] is None
    assert ("object", "object_id", LATE_PARENT) not in VAL.parent_references(cleared) and ("object", "object_id", PARENT) not in VAL.parent_references(cleared)
    assert VAL.parent_references(cleared) == VAL.parent_references(verdict(_child(CHILD_A, 1)))


def test_a_stated_null_parent_beside_a_stale_extra_parent_naming_a_missing_object_folds_and_resets_instead_of_blocking(tmp_path):
    """The dependency pre-check read `p.get(col) is None` and so took a STATED null for an
    omission, consulting the extra parent -- a record clearing the lineage (`parent_object_id: null`) while still carrying an old
    `extra.parent_object_id` naming a parent that never folds was dependency-blocked indefinitely on a value the fold (which gives the
    top-level null precedence) would never have set. The extra parent is a dependency only when the record states no column at all."""
    root = _root(tmp_path)
    _write(root, "o", "000001-child.jsonl", [_child(CHILD_A, 1, extra={"parent_object_id": PARENT, "named_role": "epoch_1"})])
    rep = _ingest(root, synth.t(1000))
    assert rep.refused == 0 and rep.blocked == 0 and _row(root, CHILD_A)["parent_object_id"] == PARENT
    # the clearing record: the column stated null, the stale extra key naming an object that never folds
    _write(root, "o", "000002-clear.jsonl", [_child(CHILD_A, 2, parent_object_id=None, extra={"parent_object_id": LATE_PARENT, "named_role": "epoch_1"})])
    rep = _ingest(root, synth.t(2000))
    assert rep.refused == 0 and rep.blocked == 0 and _blocked(root) == []                     # not blocked on LATE_PARENT
    row = _row(root, CHILD_A)
    assert row["parent_object_id"] is None and row["named_role"] == "epoch_1"                 # the stated null reset the column
    assert json.loads(row["extra_json"])["parent_object_id"] == LATE_PARENT                   # the extra accumulation is untouched
    assert json.loads(row["field_versions_json"])["parent_object_id"][0] == synth.t(2)         # stamped by the clearing record
    # a record stating NO column beside the same stale extra key is the dependency it always was: blocked until the parent folds
    _write(root, "o", "000003-omitted.jsonl", [_child(CHILD_A, 3, parent_object_id=OMIT, extra={"parent_object_id": LATE_PARENT, "named_role": "epoch_1"})])
    rep = _ingest(root, synth.t(3000))
    assert rep.refused == 0 and rep.blocked == 1
    assert _blocked(root) == [{"record_type": "object", "missing_parent_table": "object", "missing_parent_id": LATE_PARENT}]
    assert _row(root, CHILD_A)["parent_object_id"] is None
    _rebuild_equal(root, tmp_path)


def test_the_migration_lift_rewrites_the_live_rows_and_history_to_the_fresh_folds_bytes_and_is_idempotent(tmp_path):
    """The rows as the code before the migration left them (extra carrying the keys, the columns NULL, the history folded without them)
    are brought to EXACTLY what the new fold writes -- the live row, its field stamps and every folded state -- by `lift_object_lineage`;
    a second run changes nothing; a row that carried no keys is untouched; the count line says what moved."""
    root = _root(tmp_path)
    _write(root, "objs", "000001-objects.jsonl", [_child(CHILD_A, 1000, extra={"parent_object_id": PARENT, "named_role": "epoch_1"}),
                                                   _child(CHILD_B, 1001, extra={"named_role": "intermediate"}),
                                                   _child(CHILD_C, 1002)])
    _write(root, "objs", "000002-again.jsonl", [_child(CHILD_A, 1100, extra={"parent_object_id": PARENT, "named_role": "epoch_1"}, artifact_bytes=99)])
    assert _ingest(root, synth.t(2000)).refused == 0
    want_rows = {r["object_id"]: r for r in _q(root, "SELECT * FROM object")}
    want_hist = _q(root, "SELECT natural_key_json, written_utc, record_sha256, folded_state_json FROM row_history WHERE table_name = 'object' ORDER BY 1, 2, 3")
    # downgrade to the pre-lift shape: columns NULL, stamps without them, history folded without them
    conn = DB.connect(DB.db_path(root))
    for oid, r in want_rows.items():
        fv = json.loads(r["field_versions_json"])
        fv.pop("parent_object_id", None)
        fv.pop("named_role", None)
        conn.execute("UPDATE object SET parent_object_id = NULL, named_role = NULL, field_versions_json = ? WHERE object_id = ?", (RED.canonical_json(fv), oid))
    for h in want_hist:
        st = json.loads(h["folded_state_json"])
        st.pop("parent_object_id", None)
        st.pop("named_role", None)
        conn.execute("UPDATE row_history SET folded_state_json = ? WHERE table_name = 'object' AND natural_key_json = ? AND written_utc = ? AND record_sha256 = ?",
                     (RED.canonical_json(st), h["natural_key_json"], h["written_utc"], h["record_sha256"]))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM object WHERE parent_object_id IS NOT NULL OR named_role IS NOT NULL").fetchone()[0] == 0
    note = M.lift_object_lineage(conn)
    conn.commit()
    assert note.startswith("lifted object lineage from extra_json: parent_object_id 1, named_role 2 (2 object row(s) rewritten, ")
    assert "object key(s) checked" in note
    got_rows = {r["object_id"]: dict(r) for r in conn.execute("SELECT * FROM object").fetchall()}
    assert got_rows == want_rows                                            # byte for byte: columns, stamps, extra, uri, rank
    got_hist = [dict(r) for r in conn.execute("SELECT natural_key_json, written_utc, record_sha256, folded_state_json FROM row_history WHERE table_name = 'object' ORDER BY 1, 2, 3").fetchall()]
    assert got_hist == want_hist
    note2 = M.lift_object_lineage(conn)
    assert note2.startswith("lifted object lineage from extra_json: parent_object_id 0, named_role 0 (0 object row(s) rewritten, 0 history event(s) refolded")
    conn.close()
    assert M.POST_APPLY["record_refusal_fact"] is M.lift_object_lineage
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# Status may be omitted when a row exists
# --------------------------------------------------------------------------------------------


def test_an_object_record_omitting_status_keeps_the_folded_status_and_its_stamp_and_waits_when_no_row_exists(tmp_path):
    root = _root(tmp_path)
    producer = _child(CHILD_A, 1000, status="reference")
    _write(root, "producer", "000001-objects.jsonl", [producer])
    assert _ingest(root, synth.t(2000)).refused == 0
    before = _row(root, CHILD_A)
    assert before["status"] == "reference" and json.loads(before["field_versions_json"])["status"] == [synth.t(1000), R.record_sha(producer)]
    # a mirror's record, later, omitting status and restating nothing else of it: folds, status and its stamp untouched, its own column moved
    mirror = _child(CHILD_A, 1100, status=OMIT, artifact_bytes=4321)
    assert "status" not in mirror["payload"]
    assert isinstance(VAL.validate_record(mirror, "<t>", R.record_sha(mirror)), VAL.Validated)
    _write(root, "mirror", "000001-objects.jsonl", [mirror, _child(ORPHAN, 1101, status=OMIT)])
    rep = _ingest(root, synth.t(2100))
    assert rep.refused == 0 and rep.blocked == 1
    after = _row(root, CHILD_A)
    assert after["status"] == "reference" and after["artifact_bytes"] == 4321
    fv = json.loads(after["field_versions_json"])
    assert fv["status"] == [synth.t(1000), R.record_sha(producer)] and fv["artifact_bytes"] == [synth.t(1100), R.record_sha(mirror)]
    assert _blocked(root) == [{"record_type": "object", "missing_parent_table": "object", "missing_parent_id": ORPHAN}] and _row(root, ORPHAN) is None
    # the producer's statement for the orphan arrives (stamped BEFORE the mirror's): the mirror's record folds behind it, status the producer's
    _write(root, "producer", "000002-orphan.jsonl", [_child(ORPHAN, 1050, status="pending_fit")])
    rep = _ingest(root, synth.t(2200))
    assert rep.refused == 0 and rep.unblocked == 1 and _blocked(root) == []
    assert _row(root, ORPHAN)["status"] == "pending_fit"
    # a stated status still moves the column; an explicit null is refused by name
    _write(root, "producer", "000003-status.jsonl", [_child(CHILD_A, 1200, status="withdrawn")])
    assert _ingest(root, synth.t(2300)).refused == 0 and _row(root, CHILD_A)["status"] == "withdrawn"
    null_status = _child(CHILD_A, 1300)
    null_status["payload"]["status"] = None
    out = VAL.validate_record(null_status, "<t>", R.record_sha(null_status))
    assert isinstance(out, VAL.Malformed) and out.field == "status" and "omit the field" in out.reason
    assert R.OBJECT_OMITTABLE_WITH_ROW == ("status",) and "status" in R.REQUIRED_FIELDS["object"]       # required of the ROW, documented as such
    _rebuild_equal(root, tmp_path)


def test_a_status_less_record_ordered_before_the_producers_in_one_chunk_folds_once_the_row_exists(tmp_path):
    """The mirror's record stamped EARLIER than the producer's, both delivered in one cycle: the mirror's folds first in order-key order,
    finds no row and is blocked; the producer's folds; the re-evaluation folds the mirror's as an EARLIER event -- the live row is the
    producer's status (the latest statement), and replay from any order agrees."""
    root = _root(tmp_path)
    mirror = _child(CHILD_A, 900, status=OMIT, artifact_bytes=7)
    producer = _child(CHILD_A, 1000, status="approved", artifact_bytes=8)
    _write(root, "mix", "000001-objects.jsonl", [mirror, producer])
    rep = _ingest(root, synth.t(2000))
    assert rep.refused == 0
    row = _row(root, CHILD_A)
    assert row["status"] == "approved" and row["artifact_bytes"] == 8
    assert _blocked(root) == []
    assert len(_q(root, "SELECT 1 AS one FROM row_history WHERE table_name = 'object' AND natural_key_json = ?", RED.canonical_json({"object_id": CHILD_A}))) == 2
    _rebuild_equal(root, tmp_path)
