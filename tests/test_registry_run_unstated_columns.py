"""`extra.unstated_columns` on `run` records (2026-10-07): a stats reader's record onto the export's run row.

The flap this ends: two adapters (`petri_stats`, `r2_export`) mint ONE run id per Petri audit and each files a full `run` record; the
fold is per-column latest statement, so the live row's `lane_source_legacy`, `generated_utc` / `queued_utc` / `started_utc`, `store_uri`,
`serving_path`, `notes`, the counts and `field_confidence_json` flipped to whichever adapter re-read last -- 749 shared run ids live, the
same `written_utc` resolved by record sha -- without any new information. The reader's record now states the run's IDENTITY columns and
marks every other carried column (`lineage_common.stats_reader_run_record`); the fold applies a marked column to a new row, or where the
standing row has no value, and skips it over a standing value whatever the stamps' order. Synthetic events throughout.

Round 2 of the review (2026-10-07): a null the reader carries is no longer omitted but carried and marked -- a WITHDRAWAL the fold applies
only over the reader's own (flagged) earlier value, so a corrected sidecar that no longer attests `serving_identity` / `intervention_hash`
clears them; an owner's value and a standing null are untouched.
"""
from __future__ import annotations

import json

from sisteer.registry import reducer as RD
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import lineage_common as LC

T = synth.t
READER, OWNER = "stats_reader_synth", "export_synth"


def _run(**over):
    return synth.run_payload("k", "none", "none", 0, **over)


def _reader_record(**over):
    """What the stats reader files: identity stated, the rest marked, a null carried as a withdrawal."""
    return LC.stats_reader_run_record(_run(lane_source_legacy=READER, **over))


def _ev(i, payload, kind="full"):
    extra = dict(payload.get("extra") or {})
    body = {k: v for k, v in payload.items() if k != "extra"}
    body["_kind"] = kind
    if extra:
        body["_extra"] = extra
    return RD.HistoryEvent(T(i), f"{i:064x}", "run", f"records/x/{i:06d}-run.jsonl", body)


def _fold(*events):
    out = RD.fold("run", list(events))
    return out[-1][0], out[-1][1]


def _conf(st):
    """The row's confidence entries, whether the synthetic events carried the column as a dict or as the log's canonical string."""
    return RD._conf_of(st.get("field_confidence_json"))


# ---- the record the reader files ---------------------------------------------------------------------------------------------------


def test_stats_reader_run_record_states_identity_marks_the_rest_and_carries_a_null_as_a_withdrawal():
    p = _run(lane_source_legacy=READER, notes=None, pass_label=None, serving_identity="proxy+synthetic", n_items_planned=60)
    rec = LC.stats_reader_run_record(p)
    marker = rec["extra"]["unstated_columns"]
    assert set(LC.RUN_IDENTITY_COLUMNS) <= set(rec) | {"view_legacy", "campaign_cell_id_legacy", "judge_id"}   # identity stated where carried
    assert not (set(marker) & set(LC.RUN_IDENTITY_COLUMNS)) and "extra" not in marker and "field_confidence_json" not in marker
    # a null is carried AND marked (round 2 of the review, 2026-10-07): a withdrawal the fold applies only over the reader's own earlier
    # value -- the record used to omit it, and a withdrawn serving_identity stood as current
    assert rec["notes"] is None and rec["pass_label"] is None and "notes" in marker and "pass_label" in marker
    assert set(marker) == {k for k in rec if k not in LC.RUN_IDENTITY_COLUMNS and k not in ("extra", "field_confidence_json")} and marker == sorted(marker)
    assert rec["lane_source_legacy"] == READER and rec["serving_identity"] == "proxy+synthetic"
    # the provenance carrier keeps the entries of the marked columns only (the reader's evidence for what it may apply); an identity
    # column's entry and an entry for a column the record does not carry are dropped, so the reader's re-read moves no provenance of a
    # column it does not state
    assert set(rec["field_confidence_json"]) <= set(marker) and not (set(rec["field_confidence_json"]) & set(LC.RUN_IDENTITY_COLUMNS))
    full = {"serving_identity": {"confidence": "recorded", "source": "the sidecar"}, "model_id": {"confidence": "recorded", "source": "the model key"},
            "node_hostname": {"confidence": "unverified", "reason": "not on the report row"}}
    rec2 = LC.stats_reader_run_record(_run(lane_source_legacy=READER, serving_identity="proxy+synthetic", field_confidence_json=full))
    assert rec2["field_confidence_json"] == {"serving_identity": full["serving_identity"]} and "field_confidence_json" not in rec2["extra"]["unstated_columns"]
    assert p["notes"] is None and "unstated_columns" not in (p.get("extra") or {})       # the input is not changed
    # the record validates: the marker names carried, non-key run columns -- the nulls included
    ok = VAL.validate_record(synth.rec("run", rec, T(100)))
    assert isinstance(ok, VAL.Validated), getattr(ok, "reason", None)
    assert ok.extra["unstated_columns"] == marker


def test_validator_admits_the_marker_on_run_records_with_the_same_rules_as_object():
    base = _run(lane_source_legacy=READER)
    assert "notes" not in base                                                   # an OMITTED column is the "not carried" case
    for bad, why in (((["run_id"]), "natural key"), (["no_such_column"], "not a column"), (["notes"], "not carried"), ([], "non-empty list"),
                     (["lane_source_legacy", "lane_source_legacy"], "repeats")):
        m = VAL.validate_record(synth.rec("run", dict(base, extra={"unstated_columns": bad}), T(100)))
        assert isinstance(m, VAL.Malformed) and m.field == "extra.unstated_columns" and why in m.reason, (bad, getattr(m, "reason", None))
    ok = VAL.validate_record(synth.rec("run", dict(base, extra={"unstated_columns": ["lane_source_legacy", "serving_path"]}), T(100)))
    assert isinstance(ok, VAL.Validated), getattr(ok, "reason", None)
    # ONE rule differs from object: a marked column CARRIED AS NULL is admitted on run (a withdrawal; round 2 of the review, 2026-10-07)
    ok2 = VAL.validate_record(synth.rec("run", dict(base, notes=None, extra={"unstated_columns": ["notes"]}), T(100)))
    assert isinstance(ok2, VAL.Validated) and ok2.payload["notes"] is None and ok2.extra["unstated_columns"] == ["notes"], getattr(ok2, "reason", None)
    assert VAL.UNSTATED_REFRESH_TABLES == RD.UNSTATED_REFRESH_TABLES == ("run",)


# ---- the fold ---------------------------------------------------------------------------------------------------------------------------


def _owner_record(**over):
    """The owner's full record (the export): states everything it carries, omits the sidecar-only columns."""
    p = _run(lane_source_legacy=OWNER, generated_utc="2026-09-16T20:32:11.000Z", queued_utc="2026-09-16T20:32:11.000Z",
             started_utc="2026-09-16T20:32:11.000Z", store_uri="~/synthetic/consolidated/agg.json", notes="the owner's note",
             serving_path="lora_native", **over)
    p.pop("serving_identity", None)
    p.pop("n_items_planned", None)
    return p


def _reader(**over):
    """The stats reader's record with its usual sidecar values; ``over`` wins over every default."""
    fields = dict(generated_utc="2026-09-07T22:58:40.343Z", queued_utc="2026-09-07T22:58:40.343Z", started_utc="2026-09-07T22:58:40.343Z",
                  store_uri="~/.local/state/synthetic/audit/store", serving_path="lens", serving_identity="proxy+synthetic+lens",
                  n_items_planned=60, notes=None)
    fields.update(over)
    return _reader_record(**fields)


PROVENANCE = ("lane_source_legacy", "generated_utc", "queued_utc", "started_utc", "store_uri", "serving_path", "notes")


def test_without_the_marker_two_readers_flip_the_row_on_every_re_read_the_reproduction():
    """The defect as it stood: a full record from each adapter, the fold per-column latest, the row's provenance follows the write order."""
    # the old shape: a full statement of every column the reader carries, nulls included, no marker
    stripped = {k: v for k, v in _run(lane_source_legacy=READER, generated_utc="2026-09-07T22:58:40.343Z", queued_utc="2026-09-07T22:58:40.343Z",
                                      started_utc="2026-09-07T22:58:40.343Z", store_uri="~/.local/state/synthetic/audit/store", serving_path="lens",
                                      serving_identity="proxy+synthetic+lens", n_items_planned=60, notes=None).items() if k != "extra"}
    row_a, _ = _fold(_ev(1, stripped), _ev(2, _owner_record()))
    row_b, _ = _fold(_ev(1, stripped), _ev(2, _owner_record()), _ev(3, stripped))
    row_c, _ = _fold(_ev(1, stripped), _ev(2, _owner_record()), _ev(3, stripped), _ev(4, _owner_record()))
    assert row_a["lane_source_legacy"] == OWNER and row_b["lane_source_legacy"] == READER and row_c["lane_source_legacy"] == OWNER
    assert row_a["generated_utc"] != row_b["generated_utc"] and row_b["serving_path"] == "lens" and row_c["serving_path"] == "lora_native"
    assert row_b["notes"] is None and row_c["notes"] == "the owner's note"       # the reader's explicit null erased the owner's note


def test_with_the_marker_the_owner_row_stands_whatever_the_order_and_a_hole_is_filled_once():
    # owner first: the reader's re-reads move nothing the owner stated; the sidecar-only columns the owner omits are FILLED, once
    seq = RD.fold("run", [_ev(1, _owner_record()), _ev(2, _reader()), _ev(3, _owner_record()), _ev(4, _reader())])
    for i, (st, fv) in enumerate(seq):
        assert st["lane_source_legacy"] == OWNER and st["generated_utc"] == "2026-09-16T20:32:11.000Z" and st["serving_path"] == "lora_native", i
        assert st["store_uri"] == "~/synthetic/consolidated/agg.json" and st["notes"] == "the owner's note", i
    st, fv = seq[-1]
    assert st["serving_identity"] == "proxy+synthetic+lens" and st["n_items_planned"] == 60           # the owner left them null: filled
    # filled at 2 and REFRESHED by the reader's re-read at 4 (its own value, flagged `unstated` in the row's confidence entry); the owner's
    # provenance keeps the owner's stamp (3) -- the reader's re-read moved none of it
    assert fv["serving_identity"] == [T(4), f"{4:064x}"] and fv["lane_source_legacy"] == [T(3), f"{3:064x}"]
    conf = _conf(st)
    assert conf["serving_identity"].get("unstated") is True and conf["n_items_planned"].get("unstated") is True
    assert not any(ent.get("unstated") for c, ent in conf.items() if c in PROVENANCE)                   # the owner's columns carry no flag
    assert json.loads(st.get("extra_json") or "{}").get("unstated_columns") is None                      # a directive, never parked
    # reader first (it mints the row): its values are the initial ones; the owner's statement then wins and the reader never moves it back
    seq = RD.fold("run", [_ev(1, _reader()), _ev(2, _owner_record()), _ev(3, _reader())])
    first, _ = seq[0]
    assert first["lane_source_legacy"] == READER and first["serving_path"] == "lens" and first["serving_identity"] == "proxy+synthetic+lens"
    assert "notes" not in first or first.get("notes") is None
    for st, _ in seq[1:]:
        assert st["lane_source_legacy"] == OWNER and st["generated_utc"] == "2026-09-16T20:32:11.000Z" and st["serving_path"] == "lora_native"
        assert st["notes"] == "the owner's note" and st["serving_identity"] == "proxy+synthetic+lens"
    # order-independence of the OUTCOME: both orders end on the owner's provenance plus the reader's fill
    a = _fold(_ev(1, _owner_record()), _ev(2, _reader()))[0]
    b = _fold(_ev(1, _reader()), _ev(2, _owner_record()))[0]
    assert all(a[c] == b[c] for c in PROVENANCE) and a["serving_identity"] == b["serving_identity"] == "proxy+synthetic+lens"


def test_the_reader_keeps_stating_the_identity_and_the_skipped_columns_keep_the_owners_provenance():
    owner_conf = {"generated_utc": {"confidence": "derived", "source": "the provenance file's mtime"}, "model_id": {"confidence": "recorded", "source": "o"}}
    reader_conf = {"generated_utc": {"confidence": "derived", "source": "the audit's completed_at"}, "model_id": {"confidence": "recorded", "source": "r"},
                   "serving_identity": {"confidence": "recorded", "source": "the sidecar's serving_identity"}}
    st, fv = _fold(_ev(1, _owner_record(field_confidence_json=json.dumps(owner_conf))), _ev(2, _reader(field_confidence_json=json.dumps(reader_conf))))
    conf = _conf(st)
    assert conf["generated_utc"]["source"] == "the provenance file's mtime"           # the skipped column's provenance is the owner's
    assert conf["model_id"]["source"] == "o"                                           # an identity entry the reader's record does not restate
    assert conf["serving_identity"] == {"confidence": "recorded", "source": "the sidecar's serving_identity", "unstated": True}   # the filled column: the reader's entry, flagged
    assert fv["field_confidence_json"] == [T(2), f"{2:064x}"]                         # the carrier moved (the fill's entry and flag), the owner's entries intact
    assert fv["run_id"] == [T(2), f"{2:064x}"] and fv["model_id"] == [T(2), f"{2:064x}"]   # identity: stated by the reader, equal values, its stamp
    # an explicit null the reader DOES state (an identity column it carries as None is still a statement) would apply -- the helper omits
    # nulls only outside the identity, which is why `judge_id` None stays stated: both minters spell it alike
    rec = LC.stats_reader_run_record(_run(lane_source_legacy=READER, judge_id=None))
    assert "judge_id" in rec and rec["judge_id"] is None and "judge_id" not in rec["extra"]["unstated_columns"]


def test_a_lifecycle_event_still_owns_the_state_column_whatever_either_full_record_says():
    """The marker is about `full` / `patch` records; a `run_state` event owns `lifecycle_state` and its clock as before (the reader's
    `judged` event moves the row forward; neither full record moves it back)."""
    judged = RD.HistoryEvent(T(3), f"{3:064x}", "run", "records/x/000003-run_state.jsonl",
                             {"_kind": "state", "state": "judged", "event_utc": "2026-09-08T00:00:00.000Z"})
    st, _ = _fold(_ev(1, _owner_record()), _ev(2, _reader()), judged, _ev(4, _owner_record()), _ev(5, _reader()))
    assert st["lifecycle_state"] == "judged" and st["judged_utc"] == "2026-09-08T00:00:00.000Z" and st["lane_source_legacy"] == OWNER


# ---- a reader may refresh ITS OWN earlier value, never an owner's ----------------------------------------------------


def _counts(**over):
    return _reader(n_items_generated=over.pop("gen", 4), n_items_judged=over.pop("judged", 4), **over)


def _flag(st, c):
    return (_conf(st).get(c) or {}).get("unstated")


def test_successive_reader_revisions_refresh_the_readers_own_counts_when_no_owner_states_them():
    """Two valid reader records, 4 then 12, no owner: the row reads 12 (the adversarial review's case: "skip over any existing value" froze
    the reader's own counts at 4 on the runs no owner ever states, and nothing reported it). The flag in the row's confidence entry is what
    tells the fold the standing 4 was the reader's, not an owner's."""
    st, fv = _fold(_ev(1, _counts(gen=4, judged=4)), _ev(2, _counts(gen=12, judged=12)))
    assert st["n_items_generated"] == 12 and st["n_items_judged"] == 12 and fv["n_items_generated"] == [T(2), f"{2:064x}"]
    assert _flag(st, "n_items_generated") is True and _flag(st, "lane_source_legacy") is True       # every value is the reader's: flagged
    assert st["lane_source_legacy"] == READER
    # a changed sidecar value (serving_identity) refreshes the same way
    st2, _ = _fold(_ev(1, _counts()), _ev(2, _counts(serving_identity="proxy+synthetic+native")))
    assert st2["serving_identity"] == "proxy+synthetic+native" and _flag(st2, "serving_identity") is True


def test_a_reader_refreshes_its_counts_through_an_owner_that_omits_them_while_the_owners_own_statements_stand():
    owner = _owner_record()                                                            # states lane_source / clocks / store_uri / notes; omits the counts
    owner.pop("n_items_generated", None)
    owner.pop("n_items_judged", None)
    st, fv = _fold(_ev(1, _counts(gen=4, judged=4)), _ev(2, owner), _ev(3, _counts(gen=12, judged=12)))
    assert st["n_items_generated"] == 12 and st["n_items_judged"] == 12                           # the reader's own values: refreshed
    assert st["lane_source_legacy"] == OWNER and st["generated_utc"] == "2026-09-16T20:32:11.000Z" and st["notes"] == "the owner's note"
    assert fv["lane_source_legacy"] == [T(2), f"{2:064x}"] and fv["n_items_generated"] == [T(3), f"{3:064x}"]
    conf = _conf(st)
    assert conf["n_items_generated"].get("unstated") is True and conf["serving_identity"].get("unstated") is True   # carried through the owner's statement
    assert not (conf.get("lane_source_legacy") or {}).get("unstated") and not (conf.get("generated_utc") or {}).get("unstated")


def test_an_owners_statement_is_never_refreshed_by_a_reader_and_a_later_owner_statement_overwrites_and_sheds_the_flag():
    # owner first, stating the count: the reader's 12 is skipped; no flag
    st, _ = _fold(_ev(1, _owner_record(n_items_generated=10, n_items_judged=10)), _ev(2, _counts(gen=12, judged=12)))
    assert st["n_items_generated"] == 10 and not _flag(st, "n_items_generated")
    # reader first (flagged), then the owner states the count (unmarked): the owner's value stands, the flag is shed, a later reader is skipped
    seq = RD.fold("run", [_ev(1, _counts(gen=4, judged=4)), _ev(2, _owner_record(n_items_generated=10, n_items_judged=10)), _ev(3, _counts(gen=12, judged=12))])
    assert [st["n_items_generated"] for st, _ in seq] == [4, 10, 10]
    assert _flag(seq[0][0], "n_items_generated") is True and not _flag(seq[1][0], "n_items_generated") and not _flag(seq[2][0], "n_items_generated")
    # an owner PATCH without a confidence JSON states the column too: the value moves and the flag is shed
    patch = RD.HistoryEvent(T(2), f"{2:064x}", "run", "records/x/000002-run_patch.jsonl", {"_kind": "patch", "n_items_generated": 9})
    st, _ = _fold(_ev(1, _counts(gen=4)), patch, _ev(3, _counts(gen=12)))
    assert st["n_items_generated"] == 9 and not _flag(st, "n_items_generated")


def test_both_same_stamp_orders_end_on_the_owners_provenance_and_the_readers_fill():
    """Live, the two adapters stamped the same written_utc and the sha decided the order: either order ends on the same row."""
    a = _fold(_ev(5, _owner_record()), RD.HistoryEvent(T(5), f"{6:064x}", "run", "records/x/000006-run.jsonl", dict({k: v for k, v in _counts(gen=12, judged=12).items() if k != "extra"}, _kind="full", _extra=_counts()["extra"])))[0]
    b = _fold(RD.HistoryEvent(T(5), f"{4:064x}", "run", "records/x/000004-run.jsonl", dict({k: v for k, v in _counts(gen=12, judged=12).items() if k != "extra"}, _kind="full", _extra=_counts()["extra"])), _ev(5, _owner_record()))[0]
    for c in PROVENANCE + ("serving_identity", "n_items_planned", "n_items_generated", "n_items_judged"):
        assert a.get(c) == b.get(c), c
    assert a["lane_source_legacy"] == OWNER and a["serving_identity"] == "proxy+synthetic+lens" and a["n_items_planned"] == 60
    assert a["n_items_generated"] == 12                                          # the owner record omits the counts: the reader's stand either way
    assert _flag(a, "serving_identity") is True and _flag(b, "serving_identity") is True and not _flag(a, "lane_source_legacy")


# ---- a run_patch carries the marker under the same rules, or is refused ----------------------------------------------

import pytest

_MALFORMED = [(["run_id"], "natural key"), (["no_such_column"], "not a column"), (["notes", "bogus"], "not a column"), (["serving_identity"], "not carried"),
              ([], "non-empty list"), ("notes", "non-empty list"), ([1], "non-empty list"), (["notes", "notes"], "repeats"),
              # round 3 of the review: the flag's own carrier, refused whether or not the record carries the column (the shape below)
              (["field_confidence_json"], "carrier"), (["notes", "field_confidence_json"], "carrier")]


@pytest.mark.parametrize("marker, why", _MALFORMED)
def test_a_malformed_marker_is_refused_by_name_on_a_full_record_and_on_a_patch(marker, why):
    base = _run(lane_source_legacy=READER, notes="n")
    base.pop("serving_identity", None)
    m = VAL.validate_record(synth.rec("run", dict(base, extra={"unstated_columns": marker}), T(100)))
    assert isinstance(m, VAL.Malformed) and m.field == "extra.unstated_columns" and why in m.reason, (marker, getattr(m, "reason", None))
    rid = base["run_id"]
    m = VAL.validate_record(synth.rec("run_patch", {"run_id": rid, "fields": {"notes": "patched", "extra_json": {"unstated_columns": marker}}, "verify": {}}, T(101)))
    assert isinstance(m, VAL.Malformed) and m.field == "extra.unstated_columns" and m.record_type == "run_patch" and why in m.reason, (marker, getattr(m, "reason", None))


def test_a_well_formed_patch_marker_validates_and_folds_as_a_non_statement():
    rid = _run()["run_id"]
    ok = VAL.validate_record(synth.rec("run_patch", {"run_id": rid, "fields": {"notes": "the reader's note", "extra_json": {"unstated_columns": ["notes"]}}, "verify": {}}, T(101)))
    assert isinstance(ok, VAL.Validated) and ok.payload["fields"]["notes"] == "the reader's note" and ok.extra["unstated_columns"] == ["notes"]
    patch = RD.HistoryEvent(T(2), f"{2:064x}", "run", "records/x/000002-run_patch.jsonl", {"_kind": "patch", "notes": "the reader's note", "_extra": {"unstated_columns": ["notes"]}})
    st, _ = _fold(_ev(1, _owner_record()), patch)
    assert st["notes"] == "the owner's note"                                      # skipped over the owner's statement
    st, _ = _fold(_ev(1, _reader()), patch)
    assert st["notes"] == "the reader's note" and _flag(st, "notes") is True         # fills a hole, flagged


# ---- round 2 of the review: a reader WITHDRAWS its own earlier value with an explicit null, never an owner's ------------------------------


def _attested(**over):
    """The reader's record with the sidecar's attestations (`serving_identity`, `intervention_hash`); ``over`` wins."""
    fields = dict(serving_identity="proxy+synthetic+lens", intervention_hash="1111000011110000")
    fields.update(over)
    return _reader(**fields)


def test_a_readers_explicit_null_withdraws_its_own_earlier_serving_identity_and_intervention_hash():
    """Before round 2 the record OMITTED a null, so a corrected sidecar that no longer attested `serving_identity` / `intervention_hash` left
    the reader's earlier values standing as current, with no refusal. Now the null is carried and marked: over the reader's own (flagged)
    value it applies -- the column becomes null, its stamp and provenance the withdrawing event's -- and a later read that attests the
    column again fills it back."""
    st, fv = _fold(_ev(1, _attested()), _ev(2, _attested(serving_identity=None, intervention_hash=None)))
    assert st["serving_identity"] is None and st["intervention_hash"] is None
    assert fv["serving_identity"] == [T(2), f"{2:064x}"] and fv["intervention_hash"] == [T(2), f"{2:064x}"]
    assert _flag(st, "serving_identity") is True and _flag(st, "intervention_hash") is True      # the withdrawal is a marked statement too
    assert st["lane_source_legacy"] == READER and st["n_items_planned"] == 60                       # the reader's other values stand
    st3, fv3 = _fold(_ev(1, _attested()), _ev(2, _attested(serving_identity=None, intervention_hash=None)), _ev(3, _attested(serving_identity="proxy+synthetic+native")))
    assert st3["serving_identity"] == "proxy+synthetic+native" and st3["intervention_hash"] == "1111000011110000"
    assert fv3["serving_identity"] == [T(3), f"{3:064x}"] and fv3["intervention_hash"] == [T(3), f"{3:064x}"]
    # a confidence entry the withdrawing record carries for the column is the row's (flagged); with none, the flag alone stands
    conf = {"serving_identity": {"confidence": "unverified", "reason": "the corrected sidecar records no serving identity"}}
    st4, _ = _fold(_ev(1, _attested(field_confidence_json=json.dumps({"serving_identity": {"confidence": "recorded", "source": "the sidecar"}}))),
                   _ev(2, _attested(serving_identity=None, field_confidence_json=json.dumps(conf))))
    assert st4["serving_identity"] is None and _conf(st4)["serving_identity"] == dict(conf["serving_identity"], unstated=True)
    assert _conf(st)["intervention_hash"] == {"unstated": True}


def test_a_readers_null_never_touches_an_owners_unmarked_value_and_moves_nothing_over_a_standing_null():
    # the owner states notes and (here) a serving_identity of its own: the reader's nulls are skipped -- value, stamp and provenance the owner's
    owner_conf = {"serving_identity": {"confidence": "recorded", "source": "the export"}}
    owner = _owner_record(field_confidence_json=json.dumps(owner_conf))
    owner["serving_identity"] = "the owner's identity"                           # (the helper omits the sidecar-only columns; this owner states one)
    st, fv = _fold(_ev(1, owner), _ev(2, _attested(serving_identity=None, notes=None)))
    assert st["serving_identity"] == "the owner's identity" and st["notes"] == "the owner's note"
    assert fv["serving_identity"] == [T(1), f"{1:064x}"] and fv["notes"] == [T(1), f"{1:064x}"]
    assert _conf(st)["serving_identity"] == owner_conf["serving_identity"] and not _flag(st, "serving_identity") and not _flag(st, "notes")
    # a column the owner never states: the reader fills it (flagged), then withdraws it beside the owner's row; the owner's columns stand
    st2, fv2 = _fold(_ev(1, _owner_record()), _ev(2, _attested()), _ev(3, _attested(intervention_hash=None)))
    assert st2["intervention_hash"] is None and fv2["intervention_hash"] == [T(3), f"{3:064x}"] and st2["serving_identity"] == "proxy+synthetic+lens"
    assert st2["lane_source_legacy"] == OWNER and st2["notes"] == "the owner's note" and fv2["notes"] == [T(1), f"{1:064x}"] and fv2["lane_source_legacy"] == [T(1), f"{1:064x}"]
    # a null over a standing null -- a reader-minted row that never attested the column, a re-read after the withdrawal -- is no event for the
    # column: no value, no stamp, no flag entry (nothing to withdraw)
    st3, fv3 = _fold(_ev(1, _attested(intervention_hash=None)))
    assert st3.get("intervention_hash") is None and "intervention_hash" not in fv3 and "intervention_hash" not in _conf(st3)
    st4, fv4 = _fold(_ev(1, _attested()), _ev(2, _attested(intervention_hash=None)), _ev(3, _attested(intervention_hash=None)))
    assert st4["intervention_hash"] is None and fv4["intervention_hash"] == [T(2), f"{2:064x}"]      # the withdrawal's stamp, not the re-read's
    # the owner's later full record that OMITS the withdrawn column carries the withdrawal's flagged entry forward, as it does any flagged entry
    st5, _ = _fold(_ev(1, _attested()), _ev(2, _attested(intervention_hash=None)), _ev(3, _owner_record()))
    assert st5["intervention_hash"] is None and _flag(st5, "intervention_hash") is True and st5["lane_source_legacy"] == OWNER
    # both same-stamp orders of a withdrawal and an owner record end on the same row
    a = _fold(_ev(1, _attested()), _ev(5, _owner_record()),
              RD.HistoryEvent(T(5), f"{6:064x}", "run", "records/x/000006-run.jsonl", dict({k: v for k, v in _attested(intervention_hash=None).items() if k != "extra"}, _kind="full", _extra=_attested()["extra"])))[0]
    b = _fold(_ev(1, _attested()),
              RD.HistoryEvent(T(5), f"{4:064x}", "run", "records/x/000004-run.jsonl", dict({k: v for k, v in _attested(intervention_hash=None).items() if k != "extra"}, _kind="full", _extra=_attested()["extra"])),
              _ev(5, _owner_record()))[0]
    assert a["intervention_hash"] is None and b["intervention_hash"] is None and a["lane_source_legacy"] == b["lane_source_legacy"] == OWNER


def test_a_patch_marking_a_null_field_validates_and_folds_as_a_withdrawal_and_object_still_refuses_a_marked_null():
    rid = _run()["run_id"]
    ok = VAL.validate_record(synth.rec("run_patch", {"run_id": rid, "fields": {"notes": None, "extra_json": {"unstated_columns": ["notes"]}}, "verify": {}}, T(101)))
    assert isinstance(ok, VAL.Validated) and ok.payload["fields"]["notes"] is None and ok.extra["unstated_columns"] == ["notes"], getattr(ok, "reason", None)
    patch = RD.HistoryEvent(T(2), f"{2:064x}", "run", "records/x/000002-run_patch.jsonl", {"_kind": "patch", "notes": None, "_extra": {"unstated_columns": ["notes"]}})
    st, _ = _fold(_ev(1, _owner_record()), patch)
    assert st["notes"] == "the owner's note"                                      # skipped over the owner's statement
    st, fv = _fold(_ev(1, _reader(notes="the reader's note")), patch)
    assert st["notes"] is None and fv["notes"] == [T(2), f"{2:064x}"] and _flag(st, "notes") is True      # the reader's own: withdrawn
    # object keeps the strict rule: a marked column carried as null is refused there by name (no withdrawal rule; the rebuild is byte for byte)
    obj = dict(synth.object_lora(), notes=None, extra={"unstated_columns": ["notes"]})
    m = VAL.validate_record(synth.rec("object", obj, T(100)))
    assert isinstance(m, VAL.Malformed) and m.field == "extra.unstated_columns" and "carried as null" in m.reason and "not carried" in m.reason, getattr(m, "reason", None)


# ---- round 3 of the review: the flag's CARRIER is never a marked column -----------------------------------------------------------------


def _unverified_owner():
    """The owner's row with a column stated ``unverified`` and the reason the validator requires beside it."""
    conf = {"code_sha": {"confidence": "unverified", "reason": "the producer recorded no code sha"}}
    return _owner_record(code_sha="unverified", field_confidence_json=json.dumps(conf)), conf


def _patch(i, **fields):
    """A ``run_patch`` as (the record the validator reads, the event the fold reads)."""
    rid = _run()["run_id"]
    rec = synth.rec("run_patch", {"run_id": rid, "fields": dict(fields), "verify": {}}, T(100 + i))
    body = {k: v for k, v in fields.items() if k != "extra_json"}
    body["_kind"] = "patch"
    extra = dict(fields.get("extra_json") or {})
    if extra:
        body["_extra"] = extra
    return rec, RD.HistoryEvent(T(i), f"{i:064x}", "run", f"records/x/{i:06d}-run_patch.jsonl", body)


def test_a_marker_naming_the_confidence_carrier_is_refused_on_a_full_record_and_on_a_patch():
    """The carrier of the per-column ``unstated`` flag is never a marked column. Marked, it passed validation and the REDUCER dropped it
    from the marker, so the record's ``field_confidence_json`` folded as an ordinary statement: a patch carrying nothing but ``{}`` with the
    carrier marked erased the standing entries -- the reason an ``unverified`` value requires included -- while the column that reason
    explains kept its value, a row no record states and no refusal reports. The validator refuses the marker by name now; the reducer's drop
    is the belt behind it."""
    assert VAL.UNSTATED_CARRIER_COLUMN == RD.CONFIDENCE_COLUMN == LC.CONFIDENCE_COLUMN == "field_confidence_json"
    base = _run(lane_source_legacy=READER, field_confidence_json={"lane_source_legacy": {"confidence": "recorded", "source": "the store"}})
    m = VAL.validate_record(synth.rec("run", dict(base, extra={"unstated_columns": ["field_confidence_json"]}), T(100)))
    assert isinstance(m, VAL.Malformed) and m.field == "extra.unstated_columns" and "carrier" in m.reason, getattr(m, "reason", None)
    mp, _ = _patch(2, field_confidence_json={}, extra_json={"unstated_columns": ["field_confidence_json"]})
    mp = VAL.validate_record(mp)
    assert isinstance(mp, VAL.Malformed) and mp.field == "extra.unstated_columns" and mp.record_type == "run_patch" and "carrier" in mp.reason, getattr(mp, "reason", None)
    # the object face of the same rule (the marker's other table): refused there too, carried or not
    mo = VAL.validate_record(synth.rec("object", dict(synth.object_lora(), extra={"unstated_columns": ["field_confidence_json"]}), T(100)))
    assert isinstance(mo, VAL.Malformed) and mo.field == "extra.unstated_columns" and "carrier" in mo.reason, getattr(mo, "reason", None)


def test_no_accepted_marked_patch_erases_the_reason_an_unverified_value_requires():
    """The finding's consequence, on the row it lands on: the owner states ``code_sha = 'unverified'`` with its reason. The malformed patch
    is refused before the fold sees it -- and since a patch's confidence JSON is read per column, a ``{}`` it carried would name nothing and
    erase nothing even if it folded (the erasure the refusal was written against is a FULL record's: its JSON is the row's whole); a
    WELL-FORMED marked patch (the reader's own column, its own entry) leaves the owner's reason standing -- a marked statement merges into
    the standing entries, touching only the columns it applies."""
    owner, conf = _unverified_owner()
    assert isinstance(VAL.validate_record(synth.rec("run", owner, T(99))), VAL.Validated)      # the reason is what admits 'unverified'
    st, _ = _fold(_ev(1, owner))
    assert st["code_sha"] == "unverified" and _conf(st)["code_sha"] == conf["code_sha"]
    bad_rec, bad_ev = _patch(2, field_confidence_json={}, extra_json={"unstated_columns": ["field_confidence_json"]})
    assert isinstance(VAL.validate_record(bad_rec), VAL.Malformed)                              # never folded: the fold below is what it WOULD do
    kept, _ = _fold(_ev(1, owner), bad_ev)
    assert kept["code_sha"] == "unverified" and _conf(kept)["code_sha"] == conf["code_sha"]      # per column: `{}` names nothing, the reason stands
    full_erase = _ev(2, dict(owner, field_confidence_json={}, extra={"unstated_columns": ["field_confidence_json"]}))
    erased, _ = _fold(_ev(1, owner), full_erase)
    assert erased["code_sha"] == "unverified" and "code_sha" not in _conf(erased)                # a FULL record's JSON is the whole: the hazard the refusal guards
    ok_rec, ok_ev = _patch(3, serving_identity="proxy+synthetic", field_confidence_json={"serving_identity": {"confidence": "recorded", "source": "the sidecar"}},
                           extra_json={"unstated_columns": ["serving_identity"]})
    assert isinstance(VAL.validate_record(ok_rec), VAL.Validated), getattr(VAL.validate_record(ok_rec), "reason", None)
    st2, _ = _fold(_ev(1, owner), ok_ev)
    assert st2["code_sha"] == "unverified" and _conf(st2)["code_sha"] == conf["code_sha"]       # the owner's reason stands
    assert st2["serving_identity"] == "proxy+synthetic" and _flag(st2, "serving_identity") is True
    # and the same for the reader's own full record, which never marks the carrier (lineage_common keeps its marked columns' entries only)
    st3, _ = _fold(_ev(1, owner), _ev(4, _reader()))
    assert st3["code_sha"] == "unverified" and _conf(st3)["code_sha"] == conf["code_sha"]
