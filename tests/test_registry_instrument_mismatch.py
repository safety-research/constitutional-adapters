"""`v_instrument_mismatch`: a run whose SENT decode / cap / serving fields disagree with the instrument it
claims is listed by the SQL object, naming the differing fields; a run whose sent fields equal the declaration is not.
Pinned here for the first time — the DDL shipped the object and nothing referenced it.

The known-bad row is the nemotron case: `top_p` sent as 1.0 against a declared 0.95. The identity tie-out is explicit: the
instrument id RECOMPUTED from the bad run's stored fields (`identity.instrument_id` over the declaration with the run's sent
values in place) differs from the id the run claims, while the good run's recomputation reproduces it. Synthetic corpus only.
"""
from __future__ import annotations

import os

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import status
from sisteer.registry import synth

#: a vendor-preset declaration (the nemotron-style row): every value the run must send back; positions `none` because the
#: runs below are unsteered (synth.run_payload files `applied_positions_mode = none` for object `none`)
VENDOR = dict(decode_source="vendor_recommendation", decode_temperature=0.7, decode_top_p=0.95, decode_top_k=20,
              applied_positions_mode="none")
SENT_OF = {"decode_temperature": "decode_temperature_sent", "decode_top_p": "decode_top_p_sent", "decode_top_k": "decode_top_k_sent"}
#: instrument identity fields the run row also carries under the same name
SHARED = ("token_cap_tokens", "serving_path", "applied_positions_mode", "thinking_mode")


def _run(inst, key, **over):
    sent = {SENT_OF[k]: inst[k] for k in SENT_OF}
    p = synth.run_payload(key, instrument_id=inst["instrument_id"], decode_source=inst["decode_source"], **sent)
    p.update(over)
    p["run_id"] = I.run_id(synth.CAMPAIGN, key, "none", synth.EVAL_ID, str(inst["instrument_id"]))
    return p


def _recomputed_instrument_id(inst, run_row):
    """The instrument id the run's STORED identity fields mint: the declaration with the run's sent decode values and its
    cap / serving / positions / thinking fields in place of the declared ones."""
    fields = {k: v for k, v in inst.items() if k != "instrument_id"}
    for declared, sent in SENT_OF.items():
        fields[declared] = run_row[sent]
    for k in SHARED:
        fields[k] = run_row[k]
    return I.instrument_id(fields)


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=2)
    inst = synth.instrument_fields(**VENDOR)
    good = _run(inst, "good_store")
    bad = _run(inst, "bad_store", decode_top_p_sent=1.0)             # the nemotron case: top_p 1.0 sent against 0.95 declared
    flagged = _run(inst, "flagged_store", instrument_match="mismatch")   # equal fields, but the producer's own match flag says no
    recs = [synth.rec("instrument", inst, synth.t(20))]
    for k, p in enumerate((good, bad, flagged)):
        recs.append(synth.rec("run", p, synth.t(7000 + k)))
        recs.append(synth.rec("run_state", {"run_id": p["run_id"], "state": "queued", "event_utc": synth.t(7000 + k),
                                            "evidence_kind": "submit_wrapper"}, synth.t(7000 + k)))
    R.write_records(os.path.join(root, "records", "mismatch", "000001-mismatch.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return root, inst, good, bad, flagged


def test_known_bad_row_is_listed_naming_the_field_and_the_good_row_is_not(tmp_path):
    root, inst, good, bad, flagged = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    ours = (good["run_id"], bad["run_id"], flagged["run_id"])
    listed = {r[0]: r[1:] for r in conn.execute(
        "SELECT run_id, differing_fields, decode_top_p_sent, declared_top_p, instrument_match FROM v_instrument_mismatch "
        "WHERE run_id IN (?, ?, ?)", ours)}
    assert set(listed) == {bad["run_id"], flagged["run_id"]}, listed
    assert listed[bad["run_id"]] == ("top_p", 1.0, 0.95, "match")
    assert listed[flagged["run_id"]] == ("", 0.95, 0.95, "mismatch")   # the flag alone lists a row; no field differs
    assert good["run_id"] not in listed
    # the good run is in the database and joins the same instrument: its absence is a verdict, not a missing row
    assert conn.execute("SELECT instrument_id FROM run WHERE run_id = ?", (good["run_id"],)).fetchone()[0] == inst["instrument_id"]
    conn.close()


def test_listed_row_is_the_one_whose_stored_fields_mint_a_different_instrument_id(tmp_path):
    """The mismatch the SQL object reports is exactly an identity disagreement: recomputing `identity.instrument_id` from the
    run's stored fields reproduces the claimed id for the good run and a different id for the bad one."""
    root, inst, good, bad, _ = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    conn.row_factory = None
    cols = ["run_id", "instrument_id"] + list(SENT_OF.values()) + list(SHARED)
    rows = {r[0]: dict(zip(cols, r)) for r in conn.execute(f"SELECT {', '.join(cols)} FROM run WHERE run_id IN (?, ?)",
                                                            (good["run_id"], bad["run_id"]))}
    conn.close()
    assert _recomputed_instrument_id(inst, rows[good["run_id"]]) == rows[good["run_id"]]["instrument_id"] == inst["instrument_id"]
    assert _recomputed_instrument_id(inst, rows[bad["run_id"]]) != rows[bad["run_id"]]["instrument_id"]
    # and the sent value that moved the id is the one the SQL object names
    assert rows[bad["run_id"]]["decode_top_p_sent"] == 1.0 and inst["decode_top_p"] == 0.95


def test_status_page_section_lists_the_mismatch(tmp_path):
    root, _inst, good, bad, flagged = _setup(tmp_path)
    ctx = status.status_json(root)
    ids = {r["run_id"]: r["differing_fields"] for r in ctx["instrument_mismatch"]}
    assert good["run_id"] not in ids
    assert {k: v for k, v in ids.items() if k in (bad["run_id"], flagged["run_id"])} == {bad["run_id"]: "top_p", flagged["run_id"]: ""}
