"""Additive schema items: new vocabulary values, the
`parity_gate` table, `ruling.ruled_utc_precision`, `roster_entry.drawn` / `judge_selector_json` /
`ladder:<record_id>`, comparison-scoped `hold_out` rulings (a `held_out` mark, never an exclusion), the structured `expires_when_json`
(any_of / all_of + guards), and `run_degeneration.measurement_state` with its paired-nullness CHECK,
`_state` columns on the wide view and the status count.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _setup(tmp_path, extra=()):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=3)
    if extra:
        _write(root, "extra", "000001-extra.jsonl", list(extra))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return root, ids


def test_new_vocabulary_values_in_ddl():
    assert "comparison" in V.SCOPE_KINDS and "hold_out" in V.RULING_ACTIONS and "not_applicable" in V.TRANSFER_KINDS
    assert "baseline_published" in V.OBJECT_ROLES and "not_evaluated" in V.EVIDENCE_STATES
    assert V.RULED_UTC_PRECISIONS == ("datetime", "date") and V.MEASUREMENT_STATES == ("measured", "not_measured", "not_applicable", "refused")
    ddl = S.render_schema_sql()
    for frag in ("'comparison'", "'hold_out'", "'baseline_published'", "ruled_utc_precision", "CREATE TABLE parity_gate", "measurement_state",
                 "run_degeneration_measured_pairing", "GLOB 'ladder:*'", "judge_selector_json", "drawn INTEGER NOT NULL DEFAULT 1"):
        assert frag in ddl, frag
    assert "parity_gate" in R.RECORD_TYPES and R.RECORD_TYPE_TABLE["parity_gate"] == "parity_gate"
    assert set(R.REQUIRED_FIELDS["parity_gate"]) == {"gate_id", "paper_sha", "input_shas_json", "registry_ingest_seq", "level", "result", "run_utc"}


def test_parity_gate_record_ingests_and_refuses_bad_level(tmp_path):
    gate = synth.rec("parity_gate", {"gate_id": "g1", "paper_sha": "e" * 40, "paper_checkout_uri": "synthetic:///paper",
                                     "input_shas_json": {"synthetic:///r2.parquet": "f" * 64}, "registry_ingest_seq": 10, "level": "L0",
                                     "result": "pass", "run_utc": synth.t(5000)}, synth.t(5000))
    bad = synth.rec("parity_gate", {"gate_id": "g2", "paper_sha": "e" * 40, "input_shas_json": {}, "registry_ingest_seq": 10, "level": "L9",
                                    "result": "pass", "run_utc": synth.t(5001)}, synth.t(5001))
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "gate", "000001-gate.jsonl", [gate, bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.ok - rep.arrival_records == 1 and rep.refused == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert tuple(conn.execute("SELECT level, result FROM parity_gate").fetchone()) == ("L0", "pass")
    assert "level" in conn.execute("SELECT field FROM record_refusal").fetchone()[0]
    conn.close()


def test_ruling_precision_default_and_roster_drawn_default(tmp_path):
    ruling = synth.rec("ruling", {"ruling_id": "R_date", "round_id": "r_synth", "action": "reference", "scope_kind": "run",
                                  "scope_json": {"run_id": "x"}, "reason": "date-only ruling", "ruled_utc": "2026-09-10T00:00:00.000Z",
                                  "ruled_utc_precision": "date", "ruled_by": "user", "source_ref": "CLAUDE.md"}, synth.t(60))
    ruling2 = synth.rec("ruling", {"ruling_id": "R_dt", "round_id": "r_synth", "action": "reference", "scope_kind": "run",
                                   "scope_json": {"run_id": "y"}, "reason": "r", "ruled_utc": synth.t(61), "ruled_by": "user",
                                   "source_ref": "s"}, synth.t(61))
    entry = synth.rec("roster_entry", {"roster_entry_id": "re_gap", "round_id": "r_synth", "comparison_id": "fig", "model_id": synth.MODEL_ID,
                                       "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "object_id": "none",
                                       "strength_rule": "operating_point", "source_citation": "gap map", "inferred": 0, "drawn": 0}, synth.t(62))
    root, ids = _setup(tmp_path, [ruling, ruling2, entry])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert {r[0]: r[1] for r in conn.execute("SELECT ruling_id, ruled_utc_precision FROM ruling")} == {"R_date": "date", "R_dt": "datetime"}
    assert conn.execute("SELECT drawn FROM roster_entry WHERE roster_entry_id = 're_gap'").fetchone()[0] == 0
    conn.close()


def test_comparison_scoped_hold_out_ruling(tmp_path):
    entry = synth.rec("roster_entry", {"roster_entry_id": "re1", "round_id": "r_synth", "comparison_id": "fig_hold", "model_id": synth.MODEL_ID,
                                       "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID,
                                       "object_selector_json": {"paper_label": "constitution_lora"}, "strength_rule": "operating_point",
                                       "source_citation": "s", "inferred": 0}, synth.t(50))
    hold = synth.rec("ruling", {"ruling_id": "R_hold", "round_id": "r_synth", "action": "hold_out", "scope_kind": "comparison",
                                "scope_json": {"comparison_id": "fig_hold"}, "reason": "figure composition", "ruled_utc": synth.t(60),
                                "ruled_by": "user", "source_ref": "s"}, synth.t(60))
    root, ids = _setup(tmp_path, [entry, hold])
    conn = DB.connect(DB.db_path(root), readonly=True)
    lora = ids["run_ids"][1]
    row = conn.execute("SELECT selection_state, in_current_paper, ruling_ids_json, held_out FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()
    # hold_out: the row stays graded, drawn and IN the paper, and is marked so a pooled mean leaves it out —
    # it is not an exclusion (`excluded` would have dropped the row from v_paper_long)
    assert (row[0], row[1], row[3]) == ("selected", 1, 1) and "R_hold" in row[2]
    assert conn.execute("SELECT held_out FROM v_paper_long WHERE run_id = ?", (lora,)).fetchone()[0] == 1
    conn.close()


def test_ladder_strength_rule_and_judge_selector(tmp_path):
    entry = synth.rec("roster_entry", {"roster_entry_id": "re_ladder", "round_id": "r_synth", "comparison_id": "fig5", "model_id": synth.MODEL_ID,
                                       "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID,
                                       "judge_selector_json": {"judge_model": "judge-model-x"},
                                       "object_selector_json": {"paper_label": "constitution_lora"}, "strength_rule": "ladder:battery_record_7",
                                       "source_citation": "s", "inferred": 0}, synth.t(50))
    root, ids = _setup(tmp_path, [entry])
    none_run, lora = ids["run_ids"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    # the fitted run files a nominal strength → the ladder rule matches it (every strength level the record carries);
    # the unsteered run has none and the selector names constitution_lora → not matched
    assert conn.execute("SELECT in_current_paper FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == 1
    assert conn.execute("SELECT selection_state FROM v_paper_current WHERE run_id = ?", (none_run,)).fetchone()[0] == "not_in_roster"
    conn.close()
    # a second run of the same object at another strength level is ALSO selected (a ladder draws every level, no later-run supersession
    # applies because its judge selector matched both) — drop the nominal instead and the rule stops matching
    patch = synth.rec("run_patch", {"run_id": lora, "fields": {"strength_nominal": None, "strength_nominal_source": None}}, synth.t(7000))
    _write(root, "extra", "000002-patch.jsonl", [patch])
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT in_current_paper FROM v_paper_current WHERE run_id = ?", (lora,)).fetchone()[0] == 0
    conn.close()
    # both judge_id and judge_selector_json set → CHECK refuses
    bad = synth.rec("roster_entry", {"roster_entry_id": "re_bad", "round_id": "r_synth", "comparison_id": "x", "model_id": synth.MODEL_ID,
                                     "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID,
                                     "judge_selector_json": {"judge_model": "judge-model-x"}, "object_id": "none",
                                     "strength_rule": "operating_point", "source_citation": "s", "inferred": 0}, synth.t(51))
    _write(root, "extra", "000003-bad.jsonl", [bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1


def _state(uri, path, members, i, found=1, resolves=1, sha="1" * 64):
    return synth.rec("source_state", {"source_id": "src", "input_uri": uri, "json_path": path, "found": found, "path_resolves": resolves,
                                      "current_members_json": members, "observed_utc": synth.t(i), "input_sha256": sha if found else None}, synth.t(i))


def _self_retiring(rid, run_id, ew):
    return synth.rec("ruling", {"ruling_id": rid, "round_id": "r_synth", "action": "exclude", "scope_kind": "run", "scope_json": {"run_id": run_id},
                                "reason": "s", "ruled_utc": synth.t(60), "ruled_by": "user", "source_ref": "s", "expires_when_json": ew}, synth.t(60))


def test_structured_expiry_any_of_all_of_and_guards(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=1, n_items=2)
    run = ids["run_ids"][0]
    c1 = {"kind": "served_id_absent", "source": "src", "input_uri": "u1", "member_key": "ids", "served_id": "old"}
    c2 = {"kind": "grades_root_absent", "input_uri": "u2", "json_path": "p", "root": "grades_old"}
    guard_ok = {"kind": "input_sha_pinned", "path": "u3", "sha256": "abcd"}
    any_of = _self_retiring("R_any", run, {"kind": "any_of", "conditions": [c1, c2], "guards": [guard_ok]})
    all_of = _self_retiring("R_all", run, {"kind": "all_of", "conditions": [c1, c2]})
    states = [_state("u1", "ids", ["new"], 70), _state("u2", "p", ["grades_old"], 71), _state("u3", "$", ["x"], 72, sha="abcd" + "0" * 60)]
    _write(root, "r", "000001-r.jsonl", [any_of, all_of] + states)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT ruling_id, evidence_state, expired_utc FROM ruling")}
    assert rows["R_any"][0] == "ok" and rows["R_any"][1] is not None       # c1 held → any_of expires
    assert rows["R_all"][0] == "ok" and rows["R_all"][1] is None           # c2 not held → all_of stands
    conn.close()
    # a failing guard can never retire: it makes the evaluation incoherent even though a condition holds
    root2 = str(tmp_path / "reg2")
    H.ensure_layout(root2)
    ids2 = synth.write_corpus(root2, n_runs=1, n_items=2)
    run2 = ids2["run_ids"][0]
    guard_bad = {"kind": "input_sha_pinned", "path": "u3", "sha256": "ffff"}
    _write(root2, "r", "000001-r.jsonl", [_self_retiring("R_g", run2, {"kind": "any_of", "conditions": [c1], "guards": [guard_bad]})] + states)
    H.ingest_once(root2, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root2), readonly=True)
    r = conn.execute("SELECT evidence_state, expired_utc FROM ruling WHERE ruling_id = 'R_g'").fetchone()
    assert r[0] == "incoherent" and r[1] is None
    assert conn.execute("SELECT selection_state FROM v_paper_current WHERE run_id = ?", (run2,)).fetchone()[0] == "indeterminate"
    conn.close()
    # missing evidence under all_of blocks expiry with state missing; under any_of the other condition decides
    root3 = str(tmp_path / "reg3")
    H.ensure_layout(root3)
    ids3 = synth.write_corpus(root3, n_runs=1, n_items=2)
    run3 = ids3["run_ids"][0]
    _write(root3, "r", "000001-r.jsonl", [_self_retiring("R_m_all", run3, {"kind": "all_of", "conditions": [c1, c2]}),
                                          _self_retiring("R_m_any", run3, {"kind": "any_of", "conditions": [c1, c2]}),
                                          _state("u1", "ids", ["new"], 70), _state("u2", "p", [], 71, found=0, resolves=0)])
    H.ingest_once(root3, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root3), readonly=True)
    rows = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT ruling_id, evidence_state, expired_utc FROM ruling")}
    assert rows["R_m_all"] == ("missing", None)
    assert rows["R_m_any"][0] == "ok" and rows["R_m_any"][1] is not None
    conn.close()
    assert paper.evaluate_expiry(conn, {"ruling_id": "x", "expires_when_json": {"kind": "any_of", "conditions": []}}).evidence_state == "incoherent"


def _degen(run_id, measure, state, i, **vals):
    p = {"run_id": run_id, "judgment_id": "", "measure": measure, "scope": "all", "measurement_state": state, "numerator": None,
         "denominator": None, "denominator_unit": None, "pooling_json": {}, "aggregation": "single_detector", "rate": None,
         "definition_version": "glossary@" + "a" * 8, "threshold_params_json": {}, "computed_by": "registry", "computed_utc": None,
         "source_uri": None, "notes": None}
    p.update(vals)
    return synth.rec("run_degeneration", p, synth.t(i))


def test_run_degeneration_measurement_state_pairing_and_wide_view(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=1, n_items=2)
    run = ids["run_ids"][0]
    good_measured = _degen(run, "union5", "measured", 9000, numerator=3, denominator=200, denominator_unit="items", rate=0.015)
    good_not = _degen(run, "zlib_collapse", "not_measured", 9001)
    bad_not_with_number = _degen(run, "classifier", "not_measured", 9002, numerator=1)
    bad_measured_missing_rate = _degen(run, "contentless", "measured", 9003, numerator=1, denominator=10, denominator_unit="items")
    R.write_records(os.path.join(root, "records", "d", "000001-degen.jsonl.gz"), [good_measured, good_not, bad_not_with_number, bad_measured_missing_rate], gz=True)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 2, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = {r[0]: r[1] for r in conn.execute("SELECT measure, measurement_state FROM run_degeneration")}
    assert rows == {"union5": "measured", "zlib_collapse": "not_measured"}
    wide = dict(conn.execute("SELECT union5__all_rate, union5__all_state, zlib_collapse__all_rate, zlib_collapse__all_state "
                             "FROM v_run_degeneration_wide WHERE run_id = ?", (run,)).fetchone())
    assert wide == {"union5__all_rate": 0.015, "union5__all_state": "measured", "zlib_collapse__all_rate": None,
                    "zlib_collapse__all_state": "not_measured"}
    for reason in conn.execute("SELECT reason FROM record_refusal"):
        assert "run_degeneration_measured_pairing" in reason[0]
    conn.close()
    from sisteer.registry import status
    ctx = status.status_json(root)
    assert ctx["degeneration_not_measured"] == [{"eval_category": "static_attack", "measure": "zlib_collapse", "n_run_measure_pairs": 1}]
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    assert "Degeneration not measured" in open(out).read()


def test_direct_insert_violating_measured_pairing_fails_naming_check(tmp_path):
    conn = DB.connect_memory()
    from sisteer.registry.migrations import apply_migrations
    apply_migrations(conn)
    with pytest.raises(sqlite3.IntegrityError) as ei:
        conn.execute("INSERT INTO record_ingest(file_uri, record_file_id, file_sha256, first_written_utc, arrival_utc, n_records, ingest_seq) VALUES ('f', 1, ?, '2026-01-01T00:00:00.000Z', '2026-01-01T00:00:00.000Z', 1, 1)", ("0" * 64,))
        conn.execute("INSERT INTO run_degeneration(run_id, judgment_id, measure, scope, measurement_state, numerator, denominator, "
                     "denominator_unit, pooling_json, aggregation, rate, definition_version, threshold_params_json, computed_by, "
                     "written_utc, record_sha256, extra_json, record_file_id, ingest_seq) VALUES "
                     "('r', '', 'union5', 'all', 'not_measured', 5, NULL, NULL, '{}', 'union', NULL, 'v', '{}', 'registry', 0, ?, '{}', 1, 1)",
                     (b"\x00" * 32,))
    assert "run_degeneration_measured_pairing" in str(ei.value) or "FOREIGN KEY" in str(ei.value)
