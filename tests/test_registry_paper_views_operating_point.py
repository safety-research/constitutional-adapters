"""The operating point of a run WITHOUT a stamp (migration 0011; registry ruling
operating-point-default-fitted-strength-1p0-arditi-c0p3-2026-09-22): a roster
entry's `operating_point` strength rule admits a filed `is_operating_point = 1`, else -- when the stamp is NULL -- a
run of a fallback kind (`schema_def.OPERATING_POINT_FALLBACK_OBJECT_KINDS`) at `strength_nominal` 1.0, or the Arditi
all-positions vector (`vector_all`) at `strength_coefficient` 0.3; a stamp of 0 still excludes; `none` and
`system_instruction` are admitted on their kind as before. v_paper_current and v_paper_long expose the basis of the
admission as `operating_point_basis`.
"""
from __future__ import annotations

import json
import os

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V

# synthetic object hashes (64 hex each): four LoRA objects under ONE paper label so a single selector matches every one of them and
# no two runs share an object (v_paper_current's later-run supersession keys on the object), and one all-positions vector
LORA_A, LORA_B, LORA_C, LORA_D = ("a1" * 32, "b2" * 32, "c3" * 32, "d4" * 32)
ARDITI = "e5" * 32


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _entry(i, object_selector=None, object_id=None, strength_rule="operating_point", comparison="fig_synth"):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": comparison, "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": strength_rule,
         "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0}
    if object_id:
        p["object_id"] = object_id
    else:
        p["object_selector_json"] = object_selector
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _vector_all_object():
    """The Arditi all-positions steering vector: the LoRA fixture's fit record under `object_kind = 'vector_all'` (the fit_target_kind
    pairing CHECK reads the fit fields, not the kind) with the published baseline's label and role."""
    p = synth.object_lora(sha=ARDITI, paper_label="arditi_allpos", role="baseline_published")
    p["object_kind"] = "vector_all"
    return p


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=1, n_items=4)              # the dimensions + the unsteered run
    none_run = ids["run_ids"][0]
    objects = [synth.rec("object", synth.object_lora(sha=sha), synth.t(20 + k)) for k, sha in enumerate((LORA_A, LORA_B, LORA_C, LORA_D))]
    objects.append(synth.rec("object", _vector_all_object(), synth.t(25)))
    _write(root, "dims", "000002-objects.jsonl", objects)
    runs = {}
    recs = []
    # a fitted LoRA at the campaign nominal 1.0 with NO stamp: the fallback admits it
    runs["lora_unstamped_1p0"], r = synth.run_records("lora_unstamped", 200, "lora_a", LORA_A, n_items=4, is_operating_point=None)
    recs += r
    # a filed stamp of 0 at nominal 1.0: never at the point, whatever the nominal says
    runs["lora_stamp0_1p0"], r = synth.run_records("lora_stamp0", 220, "lora_b", LORA_B, n_items=4, is_operating_point=0)
    recs += r
    # a filed stamp of 1 at nominal 0.5: at the point (the stamp wins over the nominal)
    runs["lora_stamp1_0p5"], r = synth.run_records("lora_stamp1_half", 240, "lora_c", LORA_C, n_items=4, is_operating_point=1,
                                                   strength_nominal=0.5, strength_coefficient=0.5)
    recs += r
    # no stamp at nominal 0.5: not at the point (a fixed:0.5 entry still takes it)
    runs["lora_unstamped_0p5"], r = synth.run_records("lora_unstamped_half", 260, "lora_d", LORA_D, n_items=4, is_operating_point=None,
                                                      strength_nominal=0.5, strength_coefficient=0.5)
    recs += r
    # the all-positions vector with NO stamp at c = 0.3 (a nominal other than 1.0, so only the coefficient clause admits it) ...
    runs["arditi_unstamped_c0p3"], r = synth.run_records("arditi_c0p3", 280, "arditi_c0p3", ARDITI, n_items=4, is_operating_point=None,
                                                         strength_nominal=0.5, strength_coefficient=0.3, strength_unit="raw_coefficient")
    recs += r
    # ... and at c = 0.15: not at the point
    runs["arditi_unstamped_c0p15"], r = synth.run_records("arditi_c0p15", 300, "arditi_c0p15", ARDITI, n_items=4, is_operating_point=None,
                                                          strength_nominal=0.5, strength_coefficient=0.15, strength_unit="raw_coefficient")
    recs += r
    _write(root, "c", "000001-runs.jsonl", recs)
    entries = [_entry(1, object_selector={"paper_label": "constitution_lora"}),
               _entry(2, object_id="none", comparison="baseline"),
               _entry(3, object_selector={"paper_label": "arditi_allpos"}, comparison="fig_arditi"),
               _entry(4, object_selector={"paper_label": "constitution_lora"}, strength_rule="fixed:0.5", comparison="fig_half")]
    _write(root, "roster", "000001-roster.jsonl", entries)
    res = H.ingest_once(root, workers=0, min_free=0)
    assert res.refused == 0, res.refusals[:5]
    runs["none"] = none_run
    return root, runs


def _pc(conn):
    return {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM v_paper_current")}


def test_operating_point_fallback_admits_unstamped_fitted_at_nominal_1p0_and_arditi_at_c0p3(tmp_path):
    root, runs = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    pc = _pc(conn)
    entries = lambda k: sorted(json.loads(pc[runs[k]]["roster_entry_ids_json"]))  # noqa: E731
    state = lambda k: (pc[runs[k]]["selection_state"], pc[runs[k]]["operating_point_basis"])  # noqa: E731
    # the fallback: an unstamped fitted LoRA at 1.0 IS at the operating point
    assert pc[runs["lora_unstamped_1p0"]]["is_operating_point"] is None
    assert state("lora_unstamped_1p0") == ("selected", "fallback_nominal_1p0") and entries("lora_unstamped_1p0") == ["re1"]
    # a stamp of 0 at 1.0 is NOT admitted by the operating_point rule (and matches no fixed:0.5 entry either)
    assert state("lora_stamp0_1p0") == ("not_in_roster", "none") and entries("lora_stamp0_1p0") == []
    # a stamp of 1 at 0.5 IS admitted (the stamp wins), and the fixed:0.5 entry takes it too
    assert state("lora_stamp1_0p5") == ("selected", "stamped") and entries("lora_stamp1_0p5") == ["re1", "re4"]
    # no stamp at 0.5: only the fixed:0.5 entry takes it; the operating_point entry does not
    assert state("lora_unstamped_0p5") == ("selected", "none") and entries("lora_unstamped_0p5") == ["re4"]
    # the Arditi all-positions vector: c = 0.3 without a stamp is the operating point, c = 0.15 is not
    assert state("arditi_unstamped_c0p3") == ("selected", "fallback_arditi_c0p3") and entries("arditi_unstamped_c0p3") == ["re3"]
    assert state("arditi_unstamped_c0p15") == ("not_in_roster", "none") and entries("arditi_unstamped_c0p15") == []
    # the unsteered run is unaffected: admitted on its kind, no strength axis
    assert pc[runs["none"]]["strength_nominal"] is None and pc[runs["none"]]["is_operating_point"] is None
    assert state("none") == ("selected", "no_strength_axis") and entries("none") == ["re2"]
    assert {r["operating_point_basis"] for r in pc.values()} <= set(S.OPERATING_POINT_BASES)
    # v_paper_long carries the same basis on every metric row of the run, and the long frame exposes it as a registry-side column
    long = {r[0]: r[1] for r in conn.execute("SELECT run_id, operating_point_basis FROM v_paper_long")}
    assert long[runs["lora_unstamped_1p0"]] == "fallback_nominal_1p0" and long[runs["arditi_unstamped_c0p3"]] == "fallback_arditi_c0p3"
    assert long[runs["lora_stamp1_0p5"]] == "stamped" and long[runs["none"]] == "no_strength_axis"
    assert "operating_point_basis" in paper.LONG_REGISTRY_COLUMNS
    frame = paper.paper_long_frame(conn, "r_synth")
    assert "operating_point_basis" in frame.columns
    # the in-paper rows: the four admitted bases plus `none` on the unstamped 0.5 run the fixed:0.5 entry selects
    assert set(frame["operating_point_basis"]) == {"fallback_nominal_1p0", "stamped", "fallback_arditi_c0p3", "no_strength_axis", "none"}
    # the gap map: every entry an admitted run satisfies is off it
    gaps = {r[0] for r in conn.execute("SELECT roster_entry_id FROM v_roster_gaps")}
    assert gaps == set()
    conn.close()


def test_operating_point_fallback_is_one_named_constant_and_never_the_unsteered_kinds():
    assert S.OPERATING_POINT_RULING_ID == "operating-point-default-fitted-strength-1p0-arditi-c0p3-2026-09-22"
    assert set(S.OPERATING_POINT_FALLBACK_OBJECT_KINDS) <= set(V.OBJECT_KINDS)
    assert not set(S.OPERATING_POINT_FALLBACK_OBJECT_KINDS) & set(S.NO_STRENGTH_AXIS_OBJECT_KINDS)
    assert S.NO_STRENGTH_AXIS_OBJECT_KINDS == ("none", "system_instruction")
    assert S.ARDITI_ALLPOS_OBJECT_KIND in S.OPERATING_POINT_FALLBACK_OBJECT_KINDS and S.ARDITI_ALLPOS_OPERATING_COEFFICIENT == 0.3
    assert S.OPERATING_POINT_FALLBACK_NOMINAL == 1.0
    # the rendered rule spells every kind of the constant once, and every basis value the column can take
    sql = S.VIEW_SQL["v_roster_match"]
    assert sql.count("IN (" + ", ".join(f"'{k}'" for k in S.OPERATING_POINT_FALLBACK_OBJECT_KINDS) + ")") == 1
    for basis in S.OPERATING_POINT_BASES:
        assert f"'{basis}'" in S.operating_point_basis_sql("r", "o"), basis
    assert S.operating_point_sql("r", "o") == f"(({S.operating_point_basis_sql('r', 'o')}) <> 'none')"
    for v in ("v_paper_current", "v_paper_long"):
        assert S.operating_point_basis_sql("r", "o") + " AS operating_point_basis" in S.VIEW_SQL[v], v
