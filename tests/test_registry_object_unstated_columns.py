"""`extra.unstated_columns`: a carried value that is not a statement.

A MIRROR that cannot omit a NOT NULL column it does not state marks it: the marker makes the carried value a non-statement in the fold
-- applied only when no row exists yet for the id, skipped over an existing row whatever the stamps' order, its provenance (the
`field_confidence_json` entry, the `extra.<column>_origin` key) skipped with it; a pure function of the record and the prior state, so
a rebuild agrees. The validator admits it on `object` records only, every name a column of `object` other than the key and carried in
the payload. It was written for the objects bank: a producer's `reference` record stamped BEFORE the bank pass but folded AFTER it
lost to the bank's newer restated `approved`, and an unchanged bank never re-emits. The bank now OMITS `status` instead (an `object`
record may omit `status`; the fold keeps the folded value, the harvester blocks a brand-new object on its own row until its producer
states one), so the bank writes no marker -- a synthetic mirror record stands in for the end-to-end cases below, and that case is
pinned in its current form.
"""
from __future__ import annotations

import json
import os
import shutil

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import reducer as RD
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import runner as RUN
from test_registry_adapter_objects_bank import _adapter, _bank, _bank_records, _producer_records, _registry

T = synth.t


# ---- the validator ------------------------------------------------------------------------------------------------------------


def _object_record(unstated, **over):
    p = dict(synth.object_lora(), **over)
    p["extra"] = {"unstated_columns": unstated}
    return synth.rec("object", p, T(100))


def test_validator_admits_the_marker_on_object_records_with_carried_columns_only():
    ok = VAL.validate_record(_object_record(["status"]))
    assert isinstance(ok, VAL.Validated) and ok.extra["unstated_columns"] == ["status"] and ok.payload["status"] == "approved"
    ok2 = VAL.validate_record(_object_record(["status", "paper_label"]))
    assert isinstance(ok2, VAL.Validated)
    for bad, why in ((["object_id"], "natural key"), (["no_such_column"], "not a column"), (["notes"], "not carried"),
                     ([], "non-empty list"), ("status", "non-empty list"), ([1], "non-empty list"), (["status", "status"], "repeats")):
        m = VAL.validate_record(_object_record(bad))
        assert isinstance(m, VAL.Malformed) and m.field == "extra.unstated_columns" and why in m.reason, (bad, getattr(m, "reason", None))
    # admitted on the tables whose fold honors it -- object, and run since the follow-up (tests/test_registry_run_unstated_columns.py);
    # nowhere else
    rid, recs = synth.run_records("k", 0, through="judged")
    jrec = next(r for r in recs if r["record_type"] == "judgment")
    j = dict(jrec["payload"], extra={"unstated_columns": ["status"]})
    m = VAL.validate_record(synth.rec("judgment", j, T(100)))
    assert isinstance(m, VAL.Malformed) and m.field == "extra.unstated_columns" and "object, run records only" in m.reason
    assert RD.UNSTATED_COLUMNS_KEY == VAL.UNSTATED_COLUMNS_KEY == "unstated_columns" and RD.UNSTATED_TABLES == VAL.UNSTATED_TABLES == ("object", "run")
    assert RD.ORIGIN_TABLES == ("object",)        # the `_origin` convention stays the object mirrors'
    assert RD.UNSTATED_REFRESH_TABLES == ("run",) # the refresh-by-flag rule is run's; object keeps the rule for byte-equal rebuilds
    assert VAL.UNSTATED_REFRESH_TABLES == RD.UNSTATED_REFRESH_TABLES    # and so is the marked-null withdrawal the validator admits (round 2)


# ---- the fold --------------------------------------------------------------------------------------------------------------------


def _ev(i, payload, extra=None):
    return RD.HistoryEvent(T(i), f"{i:064x}", "object", f"records/x/{i:06d}-object.jsonl", dict(payload, _kind="full", **({"_extra": extra} if extra else {})))


def test_fold_applies_an_unstated_column_to_a_new_row_only_and_parks_no_marker():
    base = synth.object_lora()
    # a brand-new row: the carried value is the initial value (origin bank_default), the marker is not parked
    out = RD.fold("object", [_ev(1, dict(base, status="approved"), {"unstated_columns": ["status"], "status_origin": "bank_default"})])
    state, fv = out[-1]
    assert state["status"] == "approved" and fv["status"] == [T(1), f"{1:064x}"]
    assert json.loads(state["extra_json"]) == {"status_origin": "bank_default"}
    # an existing row: the carried value is skipped, the folded value and its stamp stand, every other column applies -- and the
    # skipped column's PROVENANCE stays the standing statement's: its field_confidence_json entry and its <column>_origin key
    prod_conf = {"status": {"confidence": "recorded", "source": "the producer's statement"}, "notes": {"confidence": "recorded", "source": "p"}}
    bank_conf = {"status": {"confidence": "assumed_default", "source": "no object row at emit time …"}, "notes": {"confidence": "recorded", "source": "b"}}
    out = RD.fold("object", [_ev(1, dict(base, status="reference", notes="producer", field_confidence_json=json.dumps(prod_conf))),
                             _ev(2, dict(base, status="approved", notes="bank", field_confidence_json=json.dumps(bank_conf)),
                                 {"unstated_columns": ["status"], "status_origin": "bank_default", "other_origin": "bank"})])
    state, fv = out[-1]
    assert state["status"] == "reference" and fv["status"] == [T(1), f"{1:064x}"]
    assert state["notes"] == "bank" and fv["notes"] == [T(2), f"{2:064x}"]
    conf = json.loads(state["field_confidence_json"])
    assert conf["status"] == prod_conf["status"] and conf["notes"] == bank_conf["notes"]       # status: the producer's entry; notes: the bank's
    assert fv["field_confidence_json"] == [T(2), f"{2:064x}"]                                     # the column itself is the event's (merged)
    assert json.loads(state["extra_json"]) == {"other_origin": "bank"}                            # status_origin not applied; an unrelated origin is
    # the converse: an event that EXPLICITLY states the column sheds the earlier record's origin -- replaced by its own when it carries
    # one, removed otherwise -- so the origin always describes the standing statement (a bank-created row's `bank_default` otherwise
    # outlives the producer's later `reference`)
    bank_first = _ev(1, dict(base, status="approved", field_confidence_json=json.dumps(bank_conf)), {"unstated_columns": ["status"], "status_origin": "bank_default"})
    producer_later = _ev(2, dict(base, status="reference", field_confidence_json=json.dumps(prod_conf)))
    reread = _ev(3, dict(base, status="reference", field_confidence_json=json.dumps(bank_conf)), {"unstated_columns": ["status"], "status_origin": "registry_row"})
    ruling = _ev(4, dict(base, status="reference", field_confidence_json=json.dumps(prod_conf)), {"status_origin": "ruling"})
    seq = RD.fold("object", [bank_first, producer_later, reread, ruling])
    def three(i):
        st, _ = seq[i]
        return st["status"], json.loads(st["field_confidence_json"])["status"], json.loads(st.get("extra_json") or "{}").get("status_origin")
    assert three(0) == ("approved", bank_conf["status"], "bank_default")          # the bank creates the row: its default, its provenance
    assert three(1) == ("reference", prod_conf["status"], None)                   # the producer states: value, entry, and the origin is GONE
    assert three(2) == ("reference", prod_conf["status"], None)                   # an unstated re-read keeps all three
    assert three(3) == ("reference", prod_conf["status"], "ruling")               # a statement carrying its own origin sets it
    # the standing row has no entry for the skipped column: the event's entry is dropped, not applied
    out = RD.fold("object", [_ev(1, dict(base, status="reference", field_confidence_json=json.dumps({}))),
                             _ev(2, dict(base, status="approved", field_confidence_json=json.dumps(bank_conf)), {"unstated_columns": ["status"]})])
    assert "status" not in json.loads(out[-1][0]["field_confidence_json"]) and "notes" in json.loads(out[-1][0]["field_confidence_json"])
    # the OUTCOME is order-independent: the producer's statement, stamped earlier, wins whichever folds first
    later_first = RD.fold("object", [_ev(1, dict(base, status="approved"), {"unstated_columns": ["status"]}),
                                     _ev(2, dict(base, status="reference"))])
    assert later_first[-1][0]["status"] == "reference"
    # a stated status (no marker) still overwrites, as every statement does
    stated = RD.fold("object", [_ev(1, dict(base, status="reference")), _ev(2, dict(base, status="withdrawn"))])
    assert stated[-1][0]["status"] == "withdrawn"


def _copy_log(root, dest):
    for f in R.iter_record_files(os.path.join(root, "records")):
        rel = os.path.relpath(f, os.path.join(root, "records"))
        dest_f = os.path.join(dest, "records", rel)
        os.makedirs(os.path.dirname(dest_f), exist_ok=True)
        shutil.copy2(f, dest_f)


def _status(reg, sha):
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        row = conn.execute("SELECT status, record_uri, extra_json FROM object WHERE object_id = ?", (sha,)).fetchone()
        return (row[0], row[1], json.loads(row[2] or "{}")) if row else None
    finally:
        conn.close()


def _provenance(reg, sha):
    """(status, field_confidence_json.status, extra_json.status_origin) -- the three fields that must agree on a row."""
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        row = conn.execute("SELECT status, field_confidence_json, extra_json FROM object WHERE object_id = ?", (sha,)).fetchone()
        return row[0], json.loads(row[1] or "{}").get("status"), json.loads(row[2] or "{}").get("status_origin")
    finally:
        conn.close()


PRODUCER_CONF = {"confidence": "recorded", "source": "the producer's statement"}


def _pass(reg, ad, now, **kw):
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        return RUN.run_pass(reg, [ad], conn=conn, now=now, **kw)
    finally:
        conn.close()


def _ingest(reg):
    rep = H.ingest_once(reg, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    if rep.blocked:
        rep = H.ingest_once(reg, workers=0, min_free=0)
    return rep


def _mirror_record(payload, status, origin, when, conf=None):
    """A MIRROR's object record carrying ``status`` as a NON-statement (the marker): the shape a reader of a NOT NULL column it cannot
    omit files, built over a bank payload (every other column the object's own facts)."""
    p = dict(payload, status=status)
    p["field_confidence_json"] = dict(p.get("field_confidence_json") or {}, status=conf or {"confidence": "assumed_default", "source": "the mirror's default"})
    p["extra"] = dict({k: v for k, v in (p.get("extra") or {}).items() if k != "status_unstated"}, unstated_columns=["status"], status_origin=origin)
    return synth.rec("object", p, when)


def _blocked(reg):
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        return sorted((r[0], r[1]) for r in conn.execute("SELECT missing_parent_table, missing_parent_id FROM blocked_records WHERE record_type = 'object'"))
    finally:
        conn.close()


def test_sols_case_a_producer_reference_stamped_before_the_pass_and_folded_after_it_wins(tmp_path):
    """A producer `reference` stamped before the bank pass and folded after it wins. The bank reads a brand-new object (no row) and files
    NO status: the record waits on its own row (dependency-blocked, never refused, never a default row). The producer's `reference`,
    stamped a day earlier, was still in the spool and folds after the bank's pass: the row reads `reference`, the blocked bank record
    then folds behind it as the latest carrier and moves nothing of the column -- and keeps reading `reference` when the bank re-emits."""
    root, pins, shas = _bank(tmp_path)
    reg = _registry(tmp_path)
    ad = _adapter(root, pins)
    assert _pass(reg, ad, T(9000)).groups_refused == 0
    _ingest(reg)
    assert _status(reg, shas["base"]) is None and ("object", shas["base"]) in _blocked(reg)     # no row, no default: the bank's record waits
    published = _bank_records(reg)
    assert "status" not in published[shas["base"]] and "unstated_columns" not in published[shas["base"]]["extra"]
    # the producer's record: stamped BEFORE the pass, spooled, folded now; the blocked bank record follows it
    producer = _producer_records(root, pins, [(shas["base"], "reference")])
    R.write_records(os.path.join(reg, "records", "producer", "000001-object.jsonl"), [synth.rec("object", p, T(50)) for p in producer])
    _ingest(reg)
    status, uri, extra = _status(reg, shas["base"])
    assert status == "reference", (status, uri)
    assert uri.startswith("records/adapters/objects_bank/") and ("object", shas["base"]) not in _blocked(reg)   # the bank's record is the latest carrier; it stated no status
    # the value is the producer's and no origin reaches the row; the confidence entry for status is NOT on the live row: the
    # omitting record carries a field_confidence_json without the column and that column is a latest-statement column like any other
    # (the producer's entry stays in the history) -- the row claims no evidence for a status the bank did not carry
    assert _provenance(reg, shas["base"]) == ("reference", None, None)
    # a forced re-read of the unchanged bank (a new revision, a newer record) changes nothing
    assert _pass(reg, ad, T(9900), reread=True, force=True).groups_refused == 0
    _ingest(reg)
    assert _provenance(reg, shas["base"]) == ("reference", None, None)


def test_the_marker_variant_and_the_bank_evidence_variant(tmp_path):
    """A producer `approved` row exists; a MIRROR record carries `approved` unstated (the marker) -> the row keeps the producer's value,
    entry and (absent) origin; a producer `reference` stamped before both folds after -> `reference`. Then the bank: it omits `status`
    for the plain objects (the value stands, the entry leaves with the omitting record), and the superseded twin's `superseded` and
    the withdrawn copy's `withdrawn` are its OWN evidence: stated, applied over `reference` rows, no marker on those records. A rebuild
    agrees row for row."""
    root, pins, shas = _bank(tmp_path)
    prior = _producer_records(root, pins, [(shas["base"], "approved"), (shas["old"], "reference"), (shas["withdrawn"], "reference"),
                                           (shas["dose"], "approved"), (shas["it64"], "approved")])
    reg = _registry(tmp_path, prior_objects=prior)
    assert _provenance(reg, shas["base"]) == ("approved", PRODUCER_CONF, None)
    # the mirror's marked record over the existing row: its carried `approved` matches the row, its provenance is NOT applied over the
    # producer's (the standing statement's confidence stays; no origin key reaches the row); the mirror's record is the latest carrier
    R.write_records(os.path.join(reg, "records", "mirror", "000001-object.jsonl"), [_mirror_record(prior[0], "approved", "mirror_default", T(9100))])
    _ingest(reg)
    status, uri, extra = _status(reg, shas["base"])
    assert (status, uri.startswith("records/mirror/")) == ("approved", True) and "unstated_columns" not in extra
    assert _provenance(reg, shas["base"]) == ("approved", PRODUCER_CONF, None)
    late = _producer_records(root, pins, [(shas["base"], "reference")])
    R.write_records(os.path.join(reg, "records", "producer", "000001-object.jsonl"), [synth.rec("object", p, T(50)) for p in late])
    _ingest(reg)
    assert _provenance(reg, shas["base"]) == ("reference", PRODUCER_CONF, None)
    # the bank: omission for the plain objects, the bank's own evidence stated over the `reference` rows of the twin and the withdrawn copy
    ad = _adapter(root, pins)
    assert _pass(reg, ad, T(9200)).groups_refused == 0
    _ingest(reg)
    published = _bank_records(reg)
    assert "status" not in published[shas["base"]] and "unstated_columns" not in published[shas["base"]]["extra"]
    assert published[shas["old"]]["status"] == "superseded" and "unstated_columns" not in published[shas["old"]]["extra"]
    assert published[shas["withdrawn"]]["status"] == "withdrawn" and "unstated_columns" not in published[shas["withdrawn"]]["extra"]
    assert _status(reg, shas["old"])[0] == "superseded" and _status(reg, shas["withdrawn"])[0] == "withdrawn"
    assert _status(reg, shas["base"])[0] == "reference" and _status(reg, shas["base"])[1].startswith("records/adapters/objects_bank/")
    assert _provenance(reg, shas["base"]) == ("reference", None, None)                    # The omitting record carries no entry for the column
    assert _blocked(reg) == []
    # rebuild equality: the fold is a pure function of the log in order, so a replay agrees row for row
    copy = str(tmp_path / "copy")
    _copy_log(reg, copy)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(reg), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, diff
        assert b.execute("SELECT status FROM object WHERE object_id = ?", (shas["base"],)).fetchone()[0] == "reference"
        assert b.execute("SELECT status FROM object WHERE object_id = ?", (shas["old"],)).fetchone()[0] == "superseded"
        rb = b.execute("SELECT field_confidence_json, extra_json FROM object WHERE object_id = ?", (shas["base"],)).fetchone()
        assert "status" not in json.loads(rb[0]) and "status_origin" not in json.loads(rb[1] or "{}")
    finally:
        a.close()
        b.close()


def test_a_later_producer_statement_sheds_the_mirror_default_origin_end_to_end(tmp_path):
    """The closing case in a scratch registry, with a MIRROR record as the carrier: the mirror creates the row (approved, unstated,
    origin mirror_default -- the only statement there is); a LATER-stamped producer record states `reference` with no origin ->
    (reference, producer's entry, None), through the live appended-fold shortcut; an unstated mirror re-read keeps all three; a producer
    record carrying its own `status_origin` sets it; a rebuild agrees."""
    root, pins, shas = _bank(tmp_path)
    reg = _registry(tmp_path)
    base = _producer_records(root, pins, [(shas["base"], "approved")])[0]
    mirror_conf = {"confidence": "assumed_default", "source": "the mirror's default"}
    R.write_records(os.path.join(reg, "records", "mirror", "000001-object.jsonl"), [_mirror_record(base, "approved", "mirror_default", T(9000), mirror_conf)])
    _ingest(reg)
    assert _provenance(reg, shas["base"]) == ("approved", mirror_conf, "mirror_default")
    assert "unstated_columns" not in _status(reg, shas["base"])[2]                        # a fold directive, never parked
    producer = _producer_records(root, pins, [(shas["base"], "reference")])
    R.write_records(os.path.join(reg, "records", "producer", "000001-object.jsonl"), [synth.rec("object", p, T(9500)) for p in producer])
    _ingest(reg)                                                   # the key's LAST event: the live path's appended-fold shortcut
    assert _provenance(reg, shas["base"]) == ("reference", PRODUCER_CONF, None)
    R.write_records(os.path.join(reg, "records", "mirror", "000002-object.jsonl"), [_mirror_record(base, "reference", "mirror_row", T(9900), mirror_conf)])
    _ingest(reg)                                                   # an unstated re-read of the mirror
    assert _provenance(reg, shas["base"]) == ("reference", PRODUCER_CONF, None)
    ruled = [dict(p, extra=dict(p["extra"], status_origin="ruling")) for p in producer]
    R.write_records(os.path.join(reg, "records", "producer", "000002-object.jsonl"), [synth.rec("object", p, T(9950)) for p in ruled])
    _ingest(reg)
    assert _provenance(reg, shas["base"]) == ("reference", PRODUCER_CONF, "ruling")
    copy = str(tmp_path / "copy")
    _copy_log(reg, copy)
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(reg), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, diff
        rb = b.execute("SELECT status, field_confidence_json, extra_json FROM object WHERE object_id = ?", (shas["base"],)).fetchone()
        assert (rb[0], json.loads(rb[1])["status"], json.loads(rb[2]).get("status_origin")) == ("reference", PRODUCER_CONF, "ruling")
    finally:
        a.close()
        b.close()
