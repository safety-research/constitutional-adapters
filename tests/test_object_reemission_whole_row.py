"""An `object` re-emission folds by object_id (campaign is a column, not part of the key), latest statement wins PER STATED
COLUMN: a corrective record overwrites exactly the columns it states — a wrong campaign flips the row's campaign, an explicit null
blanks a column, an omitted column is left standing — and never a record written earlier than the current fold (the registry
owner's hazard note, stage E 2026-10-02, pinned here against the reducer's actual semantics: the campaign flip of a front-door model's
parents and V to the paper frame's sdfbase_full22 was executed as such re-emissions from the complete sidecars). Synthetic throughout."""
from __future__ import annotations

import hashlib
import json

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.records import record_file_id_for_uri, record_sha

from test_registry_schema import _seeded

TAIL = {"record_uri", "ingest_seq", "field_versions_json"}       # the ingest bookkeeping a re-fold legitimately changes


def _apply(conn, recs, uri: str, utc: str) -> list:
    """Ingest the records as the harvester would; returns each record's outcome (`ok`, or `duplicate` for a record id already folded)."""
    ing = H.Ingestor(conn, "/nonexistent", live=False)
    out = []
    with DB.transaction(conn):
        ranker = H.Ranker(conn)
        ex = H._Existence(conn)
        seq = ranker.assign(uri, hashlib.sha256(uri.encode()).hexdigest(), utc, len(recs))
        for i, r in enumerate(recs, start=1):
            v = VAL.validate_record(r, uri, record_sha(r))
            assert isinstance(v, VAL.Validated), v
            v.uri = f"{uri}#{i}"
            out.append(ing.apply_validated(v, seq, uri, record_file_id_for_uri(uri), {}, set(), ex, utc))
    assert all(o in ("ok", "duplicate") for o in out), out
    return out


def _row(conn, object_id: str) -> dict:
    cur = conn.execute("SELECT * FROM object WHERE object_id = ?", (object_id,))
    cols = [d[0] for d in cur.description]
    r = cur.fetchone()
    assert r is not None
    return dict(zip(cols, r))


def _diff(a: dict, b: dict) -> set:
    return {k for k in a if a.get(k) != b.get(k)}


def test_a_complete_reemission_changes_only_the_corrected_column():
    conn = _seeded()                                          # seeds synth.object_lora() under synth.CAMPAIGN at t(11)
    base = synth.object_lora()
    oid = base["object_id"]
    before = _row(conn, oid)
    assert before["campaign"] == synth.CAMPAIGN and before["fit_lr"] is not None
    # the corrective record: the COMPLETE payload, campaign changed, written later
    fixed = dict(base, campaign="paper_frame")
    _apply(conn, [synth.rec("object", fixed, synth.t(40))], "reemit", synth.t(40))
    after = _row(conn, oid)
    assert after["campaign"] == "paper_frame"
    assert _diff(before, after) - TAIL == {"campaign"}, _diff(before, after)
    assert json.loads(after["field_versions_json"])["campaign"][0] == synth.t(40)


def test_a_reemission_overwrites_only_what_it_states_an_explicit_null_resets_and_an_older_one_does_not_win():
    """The fold's TRUE semantics, pinned for the operator: PER COLUMN, the latest-written record that STATES it wins (reducer.fold
    replays the key's history in written_utc order and sets state[k] for the keys each record carries; _write_live_row upserts the
    columns the fold set). So a re-emission that OMITS a column leaves the prior value standing — the hazard is an explicit null (or
    a wrong value), which does overwrite — and a record written EARLIER than the current fold cannot win back a column a later record
    stated (it does supply a column NO later record ever stated). The native hook states only non-null values, so a corrective
    re-emission from the same sidecars can add or change what it states and never blanks a column by omission."""
    conn = _seeded()
    base = synth.object_lora()
    oid = base["object_id"]
    before = _row(conn, oid)
    assert before["fit_lr"] is not None and before["fit_alpha"] is not None
    # omitted columns are KEPT (columns the pairing rule does not require, so the partial record itself validates)
    partial = {k: v for k, v in base.items() if k not in ("fit_lr", "fit_alpha")}
    partial["campaign"] = "paper_frame"
    _apply(conn, [synth.rec("object", partial, synth.t(41))], "partial", synth.t(41))
    after = _row(conn, oid)
    assert after["campaign"] == "paper_frame" and after["fit_lr"] == before["fit_lr"] and after["fit_alpha"] == before["fit_alpha"]
    assert _diff(before, after) - TAIL == {"campaign"}, _diff(before, after)
    # an EXPLICIT null resets
    nulled = dict(base, campaign="paper_frame", fit_lr=None)
    _apply(conn, [synth.rec("object", nulled, synth.t(42))], "nulled", synth.t(42))
    assert _row(conn, oid)["fit_lr"] is None
    # an OLDER complete record (written before the fold's latest; distinct bytes so it is folded, not deduplicated) cannot win back
    # a column a later record stated (campaign, fit_lr) — but a column it ALONE states (notes) is taken: latest statement per column
    assert _apply(conn, [synth.rec("object", dict(base, notes="an older statement"), synth.t(5))], "older", synth.t(5)) == ["ok"]
    stale = _row(conn, oid)
    assert stale["campaign"] == "paper_frame" and stale["fit_lr"] is None and stale["notes"] == "an older statement"
    # a LATER complete record restores it, campaign with it (notes stays: no later record states it)
    _apply(conn, [synth.rec("object", dict(base, campaign="paper_frame"), synth.t(43))], "restore", synth.t(43))
    restored = _row(conn, oid)
    assert _diff(before, restored) - TAIL == {"campaign", "notes"}


def test_the_validator_refuses_a_derived_object_under_the_old_objective_before_the_spool():
    """The 2026-10-02 class: the record the derived emitters wrote (fit_target_kind derived, fit_objective ce) is refused by the
    writer-side validator naming the column and the CHECK, so it never reaches the spool as an "emitted" record."""
    base = synth.object_lora()
    derived = dict(base, object_id="f" * 64, artifact_sha256="f" * 64, object_kind="composed", paper_label="difference_relu",
                   object_role="difference", fit_target_kind="derived", derivation="composition", fit_objective="ce",
                   fit_positions="inherited_from_parents", parent_object_ids_json=[base["object_id"], "a" * 64])
    for col in ("fit_corpus_id", "fit_corpus_sha256", "fit_teacher_si_id", "fit_teacher_si_text_sha256", "fit_prompt_pool_id",
                "fit_prompt_pool_sha256", "fit_contrast_sets_json", "si_text_sha256", "si_token_count"):
        derived.pop(col, None)
    m = VAL.validate_record(synth.rec("object", derived, synth.t(50)), "memory")
    assert isinstance(m, VAL.Malformed) and m.field == "fit_objective" and "object_fit_target_kind_pairing" in m.reason
    ok = VAL.validate_record(synth.rec("object", dict(derived, fit_objective="not_applicable"), synth.t(50)), "memory")
    assert isinstance(ok, VAL.Validated), getattr(ok, "reason", ok)
    assert VAL.fit_target_pairing_problem({"fit_target_kind": "unknown_kind"}) is None       # the vocabulary check names an unknown kind
