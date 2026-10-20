"""Paper views on a synthetic roster: the derived `in_current_paper` truth table
(roster match, lifecycle, later run supersedes unless a coexist ruling, exclude / supersede / include
rulings, indeterminate evidence), `v_roster_gaps` enumerate-and-subtract with dispositions,
`v_paper_long` skeleton + the legacy-name frame, `v_paper_inputs_head` advancing on withdrawals, and
`v_pending_judgment` eligibility per judge.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import synth


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _roster_entry(i, object_selector=None, object_id=None, strength_rule="operating_point", comparison="fig_synth", judge=synth.JUDGE_ID):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": judge, "strength_rule": strength_rule,
         "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
    if object_id:
        p["object_id"] = object_id
    else:
        p["object_selector_json"] = object_selector or {"paper_label": "constitution_lora"}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _ruling(rid, action, scope_kind, scope, i=60, **extra):
    p = {"ruling_id": rid, "round_id": "r_synth", "action": action, "scope_kind": scope_kind, "scope_json": scope,
         "reason": "synthetic", "ruled_utc": synth.t(i), "ruled_by": "user", "source_ref": "test"}
    p.update(extra)
    return synth.rec("ruling", p, synth.t(i))


def _setup(tmp_path, entries, rulings=(), n_runs=2):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=n_runs, n_items=4)
    _write(root, "roster", "000001-roster.jsonl", list(entries) + list(rulings))
    H.ingest_once(root, workers=0, min_free=0)
    return root, ids


def _pc(conn):
    return {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM v_paper_current")}


def _selector_entry(i, comparison, object_id=None, object_selector=None, instrument_selector=None):
    """A roster entry that SELECTS its instrument (instrument_selector_json set, instrument_id null -- the shape of every pre-v6 entry)."""
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID, "eval_id": synth.EVAL_ID,
         "instrument_selector_json": instrument_selector or {"harness": "bespoke_sisteer", "thinking_mode": "off", "token_cap_tokens": [512]},
         "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point", "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
    if object_id:
        p["object_id"] = object_id
    else:
        p["object_selector_json"] = object_selector or {"paper_label": "constitution_lora"}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def test_a_selector_entry_never_matches_a_padded_instrument_and_a_named_entry_does(tmp_path):
    """Migration 0013: the instrument SELECTOR branch of v_roster_match requires `padding_tokens IS NULL`. A padded instrument
    (0012) equal to the seeded one on every selector-visible field satisfies the selector entries' fields and matches NONE of them, so
    a padded run is `not_in_roster` with an empty comparison list while the unpadded runs are selected as before; once entries NAME the
    padded instrument (roster v6's shape: instrument_id set, selector null) the padded runs are selected under that comparison ALONE,
    the unpadded runs do not match the named entries, v_paper_long carries the pad beside the one comparison id, and v_roster_gaps
    empties -- the frame never pools a padded row with the unpadded rows of its eval."""
    contract = I.padding_identity_sha256({"filler_sha256": "7" * 64, "mode": "fixed", "placement": "before", "separator": "\n\n", "tolerance": 0.02})
    padded = synth.instrument_fields(padding_tokens=16384, padding_identity_sha256=contract)
    padded_id = str(padded["instrument_id"])
    # the pad is the only identity difference from the seeded instrument (so the selector's fields are all satisfied)
    assert {k for k, _ in I.INSTRUMENT_ID_FIELDS if padded.get(k) != synth.INSTRUMENT.get(k)} == {"padding_tokens", "padding_identity_sha256"}
    root, ids = _setup(tmp_path, [_selector_entry(1, "fig_synth", object_id="none"), _selector_entry(2, "fig_synth")])
    none_run, lora_run = ids["run_ids"]
    rid_pn, recs_pn = synth.run_records("padded_none", 400, "none", "none", n_items=4, instrument_id=padded_id)
    rid_pl, recs_pl = synth.run_records("padded_lora", 420, "constitution_lora@deadbeef", synth.OBJECT_SHA, n_items=4, instrument_id=padded_id)
    _write(root, "padded", "000001-padded.jsonl", [synth.rec("instrument", padded, synth.t(40))] + recs_pn + recs_pl)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:5]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT padding_tokens FROM instrument WHERE instrument_id = ?", (padded_id,)).fetchone()[0] == 16384
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_pn,)).fetchone()[0] in ("judged", "ingested")   # complete (the harvester stamps `ingested`)
    # stage 1: selector entries only. The unpadded runs match; the padded runs -- same fields, a pad -- match nothing
    matched = {r[0]: r[1] for r in conn.execute("SELECT run_id, roster_entry_id FROM v_roster_match")}
    assert matched == {none_run: "re1", lora_run: "re2"}, matched
    pc = _pc(conn)
    for rid in (rid_pn, rid_pl):
        assert pc[rid]["selection_state"] == "not_in_roster" and pc[rid]["in_current_paper"] == 0 and json.loads(pc[rid]["comparison_ids_json"]) == []
    assert pc[none_run]["selection_state"] == pc[lora_run]["selection_state"] == "selected"
    assert {r[0] for r in conn.execute("SELECT roster_entry_id FROM v_roster_gaps")} == set()
    conn.close()
    # stage 2: two entries NAME the padded instrument (instrument_id set, selector null), one per object, under a comparison of their own
    def _named(i, comparison, object_id=None):
        p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID, "eval_id": synth.EVAL_ID,
             "instrument_id": padded_id, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point", "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
        if object_id:
            p["object_id"] = object_id
        else:
            p["object_selector_json"] = {"paper_label": "constitution_lora"}
        return synth.rec("roster_entry", p, synth.t(50 + i))
    _write(root, "roster", "000002-named.jsonl", [_named(3, "fig_padded", object_id="none"), _named(4, "fig_padded")])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:5]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM roster_entry WHERE instrument_id = ? AND instrument_selector_json IS NULL", (padded_id,)).fetchone()[0] == 2
    matched = {}
    for r in conn.execute("SELECT run_id, roster_entry_id FROM v_roster_match"):
        matched.setdefault(r[0], set()).add(r[1])
    assert matched == {none_run: {"re1"}, lora_run: {"re2"}, rid_pn: {"re3"}, rid_pl: {"re4"}}, matched
    pc = _pc(conn)
    for rid in (rid_pn, rid_pl):
        assert pc[rid]["selection_state"] == "selected" and pc[rid]["in_current_paper"] == 1
        assert json.loads(pc[rid]["comparison_ids_json"]) == ["fig_padded"]
    for rid in (none_run, lora_run):
        assert json.loads(pc[rid]["comparison_ids_json"]) == ["fig_synth"]
    long = {r[0]: (r[1], r[2], json.loads(r[3])) for r in conn.execute("SELECT run_id, padding_tokens, padding_identity_sha256, comparison_ids_json FROM v_paper_long")}
    assert long[rid_pn] == (16384, contract, ["fig_padded"]) and long[rid_pl] == (16384, contract, ["fig_padded"])
    assert long[none_run] == (None, None, ["fig_synth"]) and long[lora_run] == (None, None, ["fig_synth"])
    assert {r[0] for r in conn.execute("SELECT roster_entry_id FROM v_roster_gaps")} == set()
    assert paper.in_current_paper(conn, rid_pn) == (1, "selected", ["fig_padded"])
    conn.close()


def test_a_zero_pad_instrument_is_never_selector_matched_and_is_reached_by_name(tmp_path):
    """Migration 0014: the recipe's ZERO POINT files as `padding_tokens = 0` WITH its contract hash, never NULL, precisely so that
    0013's selector fence (`i.padding_tokens IS NULL`) keeps it out of the ordinary comparisons: a zero-pad instrument equal to the seeded one
    on every selector-visible field matches NO selector entry (a NULL-padded row would have), reads `not_in_roster` with an empty comparison
    list, and once an entry NAMES it (roster v7's shape) is selected under that comparison alone, with `v_paper_long` exposing 0 -- not NULL --
    beside the one comparison id, while the unpadded runs keep theirs."""
    contract = I.padding_identity_sha256({"filler_sha256": "7" * 64, "mode": "fixed", "placement": "before", "separator": "\n\n", "tolerance": 0.02})
    zero = synth.instrument_fields(padding_tokens=0, padding_identity_sha256=contract)
    zero_id = str(zero["instrument_id"])
    assert {k for k, _ in I.INSTRUMENT_ID_FIELDS if zero.get(k) != synth.INSTRUMENT.get(k)} == {"padding_tokens", "padding_identity_sha256"}
    assert zero_id != str(synth.instrument_fields(padding_tokens=16384, padding_identity_sha256=contract)["instrument_id"]) != synth.INSTRUMENT_ID
    root, ids = _setup(tmp_path, [_selector_entry(1, "fig_synth", object_id="none"), _selector_entry(2, "fig_synth")])
    none_run, lora_run = ids["run_ids"]
    rid_zn, recs_zn = synth.run_records("zero_none", 400, "none", "none", n_items=4, instrument_id=zero_id)
    rid_zl, recs_zl = synth.run_records("zero_lora", 420, "constitution_lora@deadbeef", synth.OBJECT_SHA, n_items=4, instrument_id=zero_id)
    _write(root, "zero", "000001-zero.jsonl", [synth.rec("instrument", zero, synth.t(40))] + recs_zn + recs_zl)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:5]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    assert conn.execute("SELECT padding_tokens FROM instrument WHERE instrument_id = ?", (zero_id,)).fetchone()[0] == 0          # 0, the CHECK accepts it (0014)
    matched = {r[0]: r[1] for r in conn.execute("SELECT run_id, roster_entry_id FROM v_roster_match")}
    assert matched == {none_run: "re1", lora_run: "re2"}, matched                                                              # no selector reaches the zero point
    pc = _pc(conn)
    for rid in (rid_zn, rid_zl):
        assert pc[rid]["selection_state"] == "not_in_roster" and pc[rid]["in_current_paper"] == 0 and json.loads(pc[rid]["comparison_ids_json"]) == []
    conn.close()

    def _named(i, comparison, object_id=None):
        p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID, "eval_id": synth.EVAL_ID,
             "instrument_id": zero_id, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point", "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
        if object_id:
            p["object_id"] = object_id
        else:
            p["object_selector_json"] = {"paper_label": "constitution_lora"}
        return synth.rec("roster_entry", p, synth.t(50 + i))
    _write(root, "roster", "000002-named.jsonl", [_named(3, "fig_padded", object_id="none"), _named(4, "fig_padded")])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:5]
    conn = DB.connect(DB.db_path(root), readonly=True)
    matched = {}
    for r in conn.execute("SELECT run_id, roster_entry_id FROM v_roster_match"):
        matched.setdefault(r[0], set()).add(r[1])
    assert matched == {none_run: {"re1"}, lora_run: {"re2"}, rid_zn: {"re3"}, rid_zl: {"re4"}}, matched
    pc = _pc(conn)
    for rid in (rid_zn, rid_zl):
        assert pc[rid]["selection_state"] == "selected" and json.loads(pc[rid]["comparison_ids_json"]) == ["fig_padded"]
    long = {r[0]: (r[1], r[2], json.loads(r[3])) for r in conn.execute("SELECT run_id, padding_tokens, padding_identity_sha256, comparison_ids_json FROM v_paper_long")}
    assert long[rid_zn] == (0, contract, ["fig_padded"]) and long[rid_zl] == (0, contract, ["fig_padded"])                        # 0 on the view, never NULL
    assert long[none_run] == (None, None, ["fig_synth"]) and long[lora_run] == (None, None, ["fig_synth"])
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE padding_tokens = 0").fetchone()[0] >= 2
    assert conn.execute("SELECT COUNT(*) FROM v_run_wide WHERE instrument_padding_tokens = 0").fetchone()[0] == 2
    conn.close()


def test_in_current_paper_truth_table(tmp_path):
    # run 0 = none object (not in roster), run 1 = constitution_lora (matches the selector)
    root, ids = _setup(tmp_path, [_roster_entry(1)])
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    none_run, lora_run = ids["run_ids"]
    assert pc[none_run]["selection_state"] == "not_in_roster" and pc[none_run]["in_current_paper"] == 0
    assert pc[lora_run]["selection_state"] == "selected" and pc[lora_run]["in_current_paper"] == 1
    assert json.loads(pc[lora_run]["comparison_ids_json"]) == ["fig_synth"]
    assert paper.in_current_paper(conn, lora_run) == (1, "selected", ["fig_synth"])
    # a none-object entry matches the unsteered run under operating_point (no strength)
    conn.close()
    _write(root, "roster", "000002-roster.jsonl", [_roster_entry(2, object_id="none", comparison="baseline")])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    assert pc[none_run]["in_current_paper"] == 1 and json.loads(pc[none_run]["comparison_ids_json"]) == ["baseline"]
    conn.close()


def test_exclude_supersede_include_rulings(tmp_path):
    root, ids = _setup(tmp_path, [_roster_entry(1), _roster_entry(2, object_id="none", comparison="baseline")])
    none_run, lora_run = ids["run_ids"]
    _write(root, "roster", "000002-rulings.jsonl", [_ruling("R_ex", "exclude", "run", {"run_id": lora_run})])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    assert pc[lora_run]["selection_state"] == "excluded" and pc[lora_run]["in_current_paper"] == 0
    assert "R_ex" in pc[lora_run]["ruling_ids_json"]
    conn.close()
    # an include ruling names the excluded run explicitly → selected again (include wins)
    _write(root, "roster", "000003-rulings.jsonl", [_ruling("R_in", "include", "run", {"run_id": lora_run}, i=70)])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _pc(conn)[lora_run]["in_current_paper"] == 1
    conn.close()
    # a key-scoped supersede covering the none run
    _write(root, "roster", "000004-rulings.jsonl", [_ruling("R_sup", "supersede", "key", {"model_id": synth.MODEL_ID, "object_id": "none"}, i=80,
                                                             target_json={"superseded_by": "later-run"})])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert _pc(conn)[none_run]["selection_state"] == "superseded"
    conn.close()


def test_later_run_supersedes_unless_coexist(tmp_path):
    root, ids = _setup(tmp_path, [_roster_entry(1)])
    lora_run = ids["run_ids"][1]
    # a LATER judged run at the same key (same object, eval, instrument, judge)
    rid2, recs = synth.run_records("synthstore_later", 5000, "constitution_lora@deadbeef", synth.OBJECT_SHA, 4)
    _write(root, "later", "000001-run.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    assert pc[lora_run]["selection_state"] == "superseded" and pc[rid2]["selection_state"] == "selected"
    conn.close()
    # a coexist ruling on the later run keeps both (pass_label semantics)
    _write(root, "later", "000002-coexist.jsonl", [_ruling("R_co", "coexist", "run", {"run_id": rid2}, i=90)])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    assert pc[lora_run]["in_current_paper"] == 1 and pc[rid2]["in_current_paper"] == 1
    conn.close()


def test_roster_gaps_enumerate_and_subtract_with_dispositions(tmp_path):
    entries = [_roster_entry(1), _roster_entry(3, object_selector={"paper_label": "difference_lora"}, comparison="fig_gap")]
    root, ids = _setup(tmp_path, entries)
    conn = DB.connect(DB.db_path(root), readonly=True)
    gaps = [dict(r) for r in conn.execute("SELECT * FROM v_roster_gaps")]
    assert [g["roster_entry_id"] for g in gaps] == ["re3"]
    with pytest.raises(paper.RosterGap) as ei:
        paper.assert_roster_dispositions(conn)
    assert "re3" in str(ei.value)
    conn.close()
    disp = synth.rec("disposition", {"disposition_id": "d1", "round_id": "r_synth", "roster_entry_id": "re3", "comparison_id": "fig_gap",
                                     "model_id": synth.MODEL_ID, "eval_id": synth.EVAL_ID, "state": "not_served",
                                     "reason": "difference object never served for this eval", "source_ref": "test",
                                     "ruled_utc": synth.t(99)}, synth.t(99))
    _write(root, "roster", "000002-disp.jsonl", [disp])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM v_roster_gaps").fetchone()[0] == 0
    paper.assert_roster_dispositions(conn)   # no longer raises
    conn.close()


def test_paper_long_frame_and_legacy_names(tmp_path):
    root, ids = _setup(tmp_path, [_roster_entry(1)])
    conn = DB.connect(DB.db_path(root), readonly=True)
    df = paper.paper_long_frame(conn, round_id="r_synth")
    assert len(df) == 1 and df.iloc[0]["metric_id"] == "asr" and df.iloc[0]["value"] == 0.25
    assert df.iloc[0]["paper_label"] == "constitution_lora" and df.iloc[0]["transfer_kind"] == "base_to_instruct"
    legacy = paper.paper_long_frame(conn, round_id="r_synth", legacy_names=True)
    assert "arm" in legacy.columns and "lane_kind" in legacy.columns and legacy.iloc[0]["lane_kind"] == "base→instruct"
    assert "view" in legacy.columns and "cell_id" in legacy.columns
    everything = paper.paper_long_frame(conn, current_only=False)
    assert len(everything) == 2
    conn.close()


def test_paper_inputs_head_advances_on_withdrawal_and_accepted_revision(tmp_path):
    root, ids = _setup(tmp_path, [_roster_entry(1)])
    conn = DB.connect(DB.db_path(root), readonly=True)
    head0 = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
    assert head0 is not None
    conn.close()
    rev = synth.rec("source_revision", {"revision_id": "rev1", "source_id": "src:g", "adapter": "synthetic", "observed_utc": synth.t(9000),
                                        "inputs_json": {"a": "b" * 64}, "accepted": 1}, synth.t(9000))
    _write(root, "adapters", "000001-rev.jsonl", [rev])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    head1 = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
    assert head1 > head0
    conn.close()
    wd = synth.rec("withdrawal", {"source_id": "src:g", "observation_key": json.dumps({"run_id": "gone"}), "from_revision_id": "rev0",
                                  "to_revision_id": "rev1", "table_name": "run", "reason": "removed_measurement"}, synth.t(9001))
    _write(root, "adapters", "000002-wd.jsonl", [wd])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0] > head1
    conn.close()


def test_pending_judgment_eligibility_per_judge(tmp_path):
    root, ids = _setup(tmp_path, [])
    conn = DB.connect(DB.db_path(root), readonly=True)
    # both runs are ingested with a complete judgment by JUDGE_ID → not pending for it, pending for JUDGE_ID_2
    for_1 = {r[0] for r in conn.execute("SELECT run_id FROM v_pending_judgment WHERE required_judge_id = ?", (synth.JUDGE_ID,))}
    for_2 = {r[0] for r in conn.execute("SELECT run_id FROM v_pending_judgment WHERE required_judge_id = ?", (synth.JUDGE_ID_2,))}
    assert for_1 == set() and for_2 == set(ids["run_ids"])
    row = conn.execute("SELECT is_instrument_judge, existing_judgments_json FROM v_pending_judgment WHERE required_judge_id = ? LIMIT 1",
                       (synth.JUDGE_ID_2,)).fetchone()
    assert row[0] == 0 and json.loads(row[1])[0]["status"] == "complete"
    conn.close()


def test_eval_variants_view_separates_instruments(tmp_path):
    root, ids = _setup(tmp_path, [])
    # a second instrument (different cap) for the same eval → a second variant row
    inst2 = synth.instrument_fields(token_cap_tokens=4096, token_cap_ladder_json="[4096]")
    p = synth.run_payload("variant_store", instrument_id=inst2["instrument_id"], token_cap_tokens=4096)
    p["run_id"] = I.run_id(synth.CAMPAIGN, "variant_store", "none", synth.EVAL_ID, str(inst2["instrument_id"]))
    _write(root, "v", "000001-v.jsonl", [synth.rec("instrument", inst2, synth.t(20)), synth.rec("run", p, synth.t(7000)),
                                          synth.rec("run_state", {"run_id": p["run_id"], "state": "queued", "event_utc": synth.t(7000),
                                                                  "evidence_kind": "submit_wrapper"}, synth.t(7000))])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    variants = conn.execute("SELECT instrument_id, token_cap_tokens, n_runs FROM v_eval_variants WHERE eval_id = ? ORDER BY token_cap_tokens",
                            (synth.EVAL_ID,)).fetchall()
    assert len(variants) == 2 and variants[0][1] == 512 and variants[1][1] == 4096 and variants[0][0] != variants[1][0]
    conn.close()


def test_rule_scored_run_enters_the_paper_frame_when_a_roster_entry_matches(tmp_path):
    """The paper frame admits ingested runs. A rule-scored run (no judge) reaches ingested from its headline metric, so a
    roster entry naming its eval and instrument with no judge — the entry reads the instrument's judge, none, and so the
    generation-time metric row (v_roster_match) — selects it; a run whose metric is short stays generated and out."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + synth.seed_rule_scored_dimensions())
    rid, recs = synth.rule_scored_run_records("cap", 100, object_key="lora", object_id=synth.OBJECT_SHA)
    rid_short, recs_short = synth.rule_scored_run_records("cap_short", 200, object_key="lora2", object_id=synth.OBJECT_SHA, n_scored=150)
    _write(root, "c", "000001-run.jsonl", recs + recs_short)
    entry = {"roster_entry_id": "re_cap", "round_id": "r_synth", "comparison_id": "fig_capability", "model_id": synth.MODEL_ID,
             "eval_id": synth.CAP_EVAL_ID, "instrument_id": synth.RULE_INSTRUMENT_ID, "strength_rule": "operating_point",
             "object_selector_json": {"paper_label": "constitution_lora"}, "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
    _write(root, "roster", "000001-roster.jsonl", [synth.rec("roster_entry", entry, synth.t(50))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    assert pc[rid]["lifecycle_state"] == "ingested" and pc[rid]["in_current_paper"] == 1 and pc[rid]["selection_state"] == "selected"
    assert json.loads(pc[rid]["roster_entry_ids_json"]) == ["re_cap"]
    assert pc[rid_short]["lifecycle_state"] == "generated" and pc[rid_short]["in_current_paper"] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE run_id = ? AND in_current_paper = 1", (rid,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM v_roster_gaps WHERE roster_entry_id = 're_cap'").fetchone()[0] == 0
    conn.close()


#: `v_paper_inputs_head` as it read before migration 0015: its first member evaluated the whole paper frame for one number (452 s of every
#: live harvester cycle). Kept here as the reference the head is compared against: the migration chose a SUPERSET (the base tables an
#: in-paper row draws its ingest_seq from, read whole), so the head must read >= this text at every instant -- equal while the newest input
#: is in the frame or a shared member, greater when a run outside the frame lands (a rebuild that reads an unchanged frame is the price).
HEAD_BEFORE_0015 = """SELECT MAX(seq) FROM (
    SELECT MAX(ingest_seq) AS seq FROM v_paper_long WHERE in_current_paper = 1
    UNION ALL SELECT MAX(ingest_seq) FROM roster_entry
    UNION ALL SELECT MAX(ingest_seq) FROM ruling
    UNION ALL SELECT MAX(ingest_seq) FROM ruling_state
    UNION ALL SELECT MAX(ingest_seq) FROM disposition
    UNION ALL SELECT MAX(ingest_seq) FROM source_revision WHERE accepted = 1
    UNION ALL SELECT MAX(ingest_seq) FROM withdrawal
    UNION ALL SELECT MAX(ingest_seq) FROM item_result WHERE withdrawn_by_revision_id IS NOT NULL
)"""
#: a stamp after the grading host's clock: a file so stamped ranks at the ledger head when it arrives (files are ordered by their own
#: stamp; the harvester's facts then take head + 1 ms), so each step below appends ranks instead of re-ranking earlier facts upward
_LATE = 60_000_000


def _heads(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        new = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
        old = conn.execute(HEAD_BEFORE_0015).fetchone()[0]
        frame = conn.execute("SELECT MAX(ingest_seq) FROM v_paper_long WHERE in_current_paper = 1").fetchone()[0]
        return new, old, frame
    finally:
        conn.close()


def test_paper_inputs_head_is_a_superset_of_its_pre_0015_text(tmp_path):
    """At every step the head reads >= the pre-0015 text on the same database. With in-paper rows the newest input (the harvester's own
    `run_ingest` facts on the two selected runs), the two are equal and equal to the frame's own maximum; a ruling, an accepted source
    revision and a withdrawal (members in their own right) move both, equally; a judged run OUTSIDE the frame (a fitted run
    with `is_operating_point` 0: `not_in_roster`, in_current_paper 0) moves the head and leaves the old text standing -- the superset,
    exactly where the migration says it is. Since migration 0021 the run member reads EVERY row (a run that retires leaves the two
    states the 0015 member read, and the frame reads the retirement on every contrast anchored on it), so the same run stopped at
    `generated` moves the head too and leaves the old text standing: the superset again, never a subset."""
    root, ids = _setup(tmp_path, [_roster_entry(1), _roster_entry(2, object_id="none", comparison="baseline")])
    none_run, lora_run = ids["run_ids"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    assert pc[none_run]["in_current_paper"] == 1 and pc[lora_run]["in_current_paper"] == 1
    conn.close()
    new0, old0, frame0 = _heads(root)
    assert new0 is not None and new0 == old0 == frame0
    # a run outside the frame that is NOT complete: the head moves (the run member reads every row since 0021), the old text stands
    rid_gen, recs = synth.run_records("outside_generated", _LATE, "constitution_lora@deadbeef", synth.OBJECT_SHA, 4, through="generated",
                                      is_operating_point=0, strength_nominal=0.5, strength_coefficient=0.5)
    _write(root, "outside", "000001-generated.jsonl", recs)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_gen,)).fetchone()[0] == "generated"
    assert _pc(conn)[rid_gen]["in_current_paper"] == 0
    conn.close()
    new1, old1, _ = _heads(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    top_gen = conn.execute("SELECT ingest_seq FROM run WHERE run_id = ?", (rid_gen,)).fetchone()[0]
    conn.close()
    assert old1 == old0 and new1 > new0 and new1 >= top_gen
    # a ruling: a shared member, both move and stay equal
    _write(root, "roster", "000002-ruling.jsonl", [_ruling("R_in", "include", "run", {"run_id": lora_run}, i=_LATE + 100)])
    H.ingest_once(root, workers=0, min_free=0)
    new2, old2, _ = _heads(root)
    assert new2 == old2 > new1                                         # a shared member: the two texts meet again
    # an accepted source revision, then a withdrawal: both move, equally, whether or not the withdrawn observation was in the paper
    rev = synth.rec("source_revision", {"revision_id": "rev1", "source_id": "src:g", "adapter": "synthetic", "observed_utc": synth.t(_LATE + 200),
                                        "inputs_json": {"a": "b" * 64}, "accepted": 1}, synth.t(_LATE + 200))
    _write(root, "adapters", "000001-rev.jsonl", [rev])
    H.ingest_once(root, workers=0, min_free=0)
    new3, old3, _ = _heads(root)
    assert new3 == old3 > new2
    wd = synth.rec("withdrawal", {"source_id": "src:g", "observation_key": json.dumps({"run_id": "gone"}), "from_revision_id": "rev0",
                                  "to_revision_id": "rev1", "table_name": "run", "reason": "removed_measurement"}, synth.t(_LATE + 300))
    _write(root, "adapters", "000002-wd.jsonl", [wd])
    H.ingest_once(root, workers=0, min_free=0)
    new4, old4, _ = _heads(root)
    assert new4 == old4 > new3
    # a JUDGED run outside the frame: the head moves (its run row and metric row are the newest inputs), the old text stands still
    rid_out, recs = synth.run_records("outside_judged", _LATE + 400, "constitution_lora@deadbeef", synth.OBJECT_SHA, 4,
                                      is_operating_point=0, strength_nominal=0.5, strength_coefficient=0.5)
    _write(root, "outside", "000002-judged.jsonl", recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:5]
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    assert pc[rid_out]["selection_state"] == "not_in_roster" and pc[rid_out]["in_current_paper"] == 0
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid_out,)).fetchone()[0] in ("judged", "ingested")
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (rid_out,)).fetchone()[0] >= 1
    top = conn.execute("SELECT MAX(ingest_seq) FROM run WHERE run_id = ?", (rid_out,)).fetchone()[0]
    conn.close()
    new5, old5, frame5 = _heads(root)
    assert old5 == old4 and frame5 <= old5                             # the frame did not gain a newer row and the old text did not move
    assert new5 > old5 and new5 >= top and new5 > new4                 # the superset: the head follows the run outside the frame
