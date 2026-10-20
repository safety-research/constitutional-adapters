"""The utilisation bars are an operational-efficiency gate; an UNMET verdict is admitted by the certificate only through a
readings-bearing waiver DERIVED from the recorded verdict (`seedrep_launch_plan.build_utilisation_waiver`, `stamp_utilisation_waiver.py`),
re-validated field by field (`utilisation_waiver_problems`); every mismatch refuses by name; a waiver present while bars_met is true refuses.
Synthetic verdicts and trees only. The certificate-level admission is exercised in test_seedrep_review_fixes.py (TestSolR3_2_launch_certificate)."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_launch_plan as lp  # noqa: E402
import stamp_utilisation_waiver as stamp  # noqa: E402

W = "c" * 40
EW = "22222222-3333-4444-8444-555555555555"
UNMET_READINGS = {"running_over_max_num_seqs_min": 0.0, "sm_duty_min": 0.76, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.84}
MET_READINGS = {"running_over_max_num_seqs_min": 0.91, "sm_duty_min": 0.88, "aggregate_tok_s_over_s5_min": 0.97, "h200_hours_over_sized_max": 1.1}


def _verdict(readings, **over):
    return lp.utilisation_verdict(bars=readings, source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=W, execution_wave_uuid=EW, attempt_ids=["ta-s5"],
                                  measured_utc="2026-09-20T12:00:00Z", **over)


def test_the_waiver_is_derived_from_the_verdict_and_validates_only_when_every_field_matches():
    bars = _verdict(UNMET_READINGS)
    assert bars["bars_met"] is False
    w = lp.build_utilisation_waiver(bars, W=W, execution_wave_uuid=EW, stamped_utc="2026-09-20T12:01:00Z")
    assert tuple(w) == lp.UTILISATION_WAIVER_KEYS and w["schema"] == "seedrep_utilisation_waiver/1" and w["by"] == "D198" and w["W"] == W
    assert w["bars_sha256"] == lp._canonical_sha256(bars) and len(w["bars_sha256"]) == 64
    assert w["unmet"] == [{"bar": "running_over_max_num_seqs_min", "value": 0.0, "threshold": 0.8, "met": False},
                          {"bar": "sm_duty_min", "value": 0.76, "threshold": 0.85, "met": False},
                          {"bar": "h200_hours_over_sized_max", "value": 1.84, "threshold": 1.5, "met": False}]        # the bars' order, the readings copied
    assert lp.utilisation_waiver_problems(w, bars=bars, W=W, execution_wave_uuid=EW) == []
    assert lp.utilisation_waiver_problems(json.loads(json.dumps(w)), bars=json.loads(json.dumps(bars)), W=W) == []   # a JSON round trip keeps it
    # every mismatch refuses by name
    def probs(waiver=None, bars_=None, W_=W):
        return lp.utilisation_waiver_problems(w if waiver is None else waiver, bars=bars if bars_ is None else bars_, W=W_, execution_wave_uuid=EW)
    assert any("schema" in p for p in probs({**w, "schema": "seedrep_utilisation_waiver/0"}))
    assert any("not the ruling D198" in p for p in probs({**w, "by": "D197"}))
    assert any("is not the tree" in p for p in probs({**w, "W": "d" * 40})) and any("is not the tree" in p for p in probs(W_="d" * 40))
    assert any("does not name this verdict" in p for p in probs({**w, "bars_sha256": "0" * 64}))
    edited = copy.deepcopy(w); edited["unmet"][1]["value"] = 0.86
    assert any("is not the verdict's unmet bars" in p for p in probs(edited))
    dropped = {**w, "unmet": w["unmet"][:2]}
    assert any("is not the verdict's unmet bars" in p for p in probs(dropped))
    reordered = {**w, "unmet": list(reversed(w["unmet"]))}
    assert any("is not the verdict's unmet bars" in p for p in probs(reordered))
    extra_bar = {**w, "unmet": w["unmet"] + [{"bar": "aggregate_tok_s_over_s5_min", "value": 0.97, "threshold": 0.8, "met": True}]}
    assert any("is not the verdict's unmet bars" in p for p in probs(extra_bar))
    assert any("lacks ['stamped_utc']" in p for p in probs({k: v for k, v in w.items() if k != "stamped_utc"}))
    assert any("outside the waiver schema ['note']" in p for p in probs({**w, "note": "hand-typed"}))
    assert any("not an object" in p for p in probs("D198"))
    # the verdict it names was re-recorded with another reading: the sha and the list both refuse
    other = _verdict({**UNMET_READINGS, "sm_duty_min": 0.78})
    pp = probs(bars_=other)
    assert any("does not name this verdict" in p for p in pp) and any("is not the verdict's unmet bars" in p for p in pp)


def test_a_met_verdict_is_never_waived_and_a_broken_verdict_is_not_admitted():
    met = _verdict(MET_READINGS)
    assert met["bars_met"] is True
    with pytest.raises(ValueError, match="bars_met is true -- nothing to waive"):
        lp.build_utilisation_waiver(met, W=W, execution_wave_uuid=EW)
    forged = lp.build_utilisation_waiver(_verdict(UNMET_READINGS), W=W, execution_wave_uuid=EW)
    assert any("present while bars_met is true" in p for p in lp.utilisation_waiver_problems(forged, bars=met, W=W, execution_wave_uuid=EW))
    broken = {**_verdict(UNMET_READINGS), "source": "mirror"}
    assert any("is not a utilisation verdict" in p for p in lp.utilisation_waiver_problems(forged, bars=broken, W=W, execution_wave_uuid=EW))
    with pytest.raises(ValueError, match="not a utilisation verdict"):
        lp.build_utilisation_waiver(broken, W=W, execution_wave_uuid=EW)
    with pytest.raises(ValueError, match="no other ruling admits"):
        lp.build_utilisation_waiver(_verdict(UNMET_READINGS), W=W, execution_wave_uuid=EW, ruling="D197")
    assert lp.utilisation_unmet(met) == [] and len(lp.utilisation_unmet(_verdict(UNMET_READINGS))) == 3


def test_the_stamp_tool_derives_the_waiver_from_the_recorded_verdict_and_refuses_a_met_one(tmp_path, capsys):
    rs = tmp_path / "rs"; rs.mkdir()
    plan = rs / "launch_plan.json"; plan.write_text(json.dumps({"W": W, "W7": W[:7]}))
    (rs / "smokes.json").write_text(json.dumps({"S5": {"W": W, "execution_wave_uuid": EW, "bars": _verdict(UNMET_READINGS)}}))
    assert stamp.main(["--rs", str(rs), "--plan", str(plan), "--ruling", "D198"]) == 0
    out = capsys.readouterr().out
    w = json.loads((rs / lp.UTILISATION_WAIVER_FILE).read_text())
    assert lp.utilisation_waiver_problems(w, bars=_verdict(UNMET_READINGS), W=W, execution_wave_uuid=EW) == [] and lp.read_utilisation_waiver(rs) == w
    assert "unmet=['running_over_max_num_seqs_min', 'sm_duty_min', 'h200_hours_over_sized_max']" in out and "sm_duty_min: value 0.76 vs threshold 0.85" in out
    assert stamp.main(["--rs", str(rs), "--plan", str(plan)]) == 0 and lp.read_utilisation_waiver(rs) == w        # idempotent: an identical re-stamp is a no-op
    assert stamp.main(["--rs", str(rs), "--plan", str(plan), "--ruling", "D197"]) == 2 and "no other ruling admits" in capsys.readouterr().err
    # a DIFFERENT verdict recorded afterwards: the stamped waiver no longer names it and a re-stamp refuses (write-once)
    (rs / "smokes.json").write_text(json.dumps({"S5": {"W": W, "execution_wave_uuid": EW, "bars": _verdict({**UNMET_READINGS, "sm_duty_min": 0.78})}}))
    assert stamp.main(["--rs", str(rs), "--plan", str(plan)]) == 2 and "DIFFERENT waiver is already stamped" in capsys.readouterr().err
    # a met verdict: nothing to waive; no verdict: nothing to waive; another tree's plan: the verdict does not validate
    rs2 = tmp_path / "rs2"; rs2.mkdir()
    (rs2 / "smokes.json").write_text(json.dumps({"S5": {"W": W, "execution_wave_uuid": EW, "bars": _verdict(MET_READINGS)}}))
    assert stamp.main(["--rs", str(rs2), "--plan", str(plan)]) == 2 and "nothing to waive" in capsys.readouterr().err and not (rs2 / lp.UTILISATION_WAIVER_FILE).exists()
    (rs2 / "smokes.json").write_text(json.dumps({"S5": {"W": W, "execution_wave_uuid": EW}}))
    assert stamp.main(["--rs", str(rs2), "--plan", str(plan)]) == 2 and "carries no utilisation verdict" in capsys.readouterr().err
    plan2 = tmp_path / "plan2.json"; plan2.write_text(json.dumps({"W": "d" * 40}))
    (rs2 / "smokes.json").write_text(json.dumps({"S5": {"W": W, "execution_wave_uuid": EW, "bars": _verdict(UNMET_READINGS)}}))
    assert stamp.main(["--rs", str(rs2), "--plan", str(plan2)]) == 2 and "not a utilisation verdict" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# fu3: `unmeasured` is a first-class bar outcome -- the /2 verdict's explicit form, the table-cell mapping, the waiver carrying the why          #
# --------------------------------------------------------------------------- #

#: the REAL three-root table shape (RS/tables/utilisation_7c9060f.json, fu2): Running unmeasured on pasttense (no recorded concurrency) and
#: controls (demand 64 < S 70), measured 21/21 on misalign; SM 76 / 78 / 64 %; tok ratio 1.0; hours 0.457 / 1.103 / 1.84
REAL_ROWS = [
    {"leg": "pasttense", "view": "s1", "attempt_id": "01TESTTASK-ccf6feacddc8", "S": 45, "median_running_saturated": "unmeasured",
     "running_bar": "unmeasured(driver demand not recorded on the root)", "median_sm_decode": 76.0, "sm_bar": "false", "tok_s_ratio_s5": 1.0, "tok_bar": "true",
     "h200_h_ratio": 0.457, "hours_bar": "true", "bars_met": False, "problems": ["driver demand not recorded on the root: 48 child(ren) live during the scrapes have no recorded request concurrency"]},
    {"leg": "misalign", "view": "s1", "attempt_id": "01TESTTASK-b4cbfce42621", "S": 21, "median_running_saturated": 21.0, "running_bar": "true",
     "median_sm_decode": 78.0, "sm_bar": "false", "tok_s_ratio_s5": 1.0, "tok_bar": "true", "h200_h_ratio": 1.103, "hours_bar": "true", "bars_met": False, "problems": []},
    {"leg": "controls", "view": "s1", "attempt_id": "01TESTTASK-dc4bd2f90303", "S": 70, "median_running_saturated": "unmeasured",
     "running_bar": "unmeasured(in-flight never >= S)", "median_sm_decode": 64.0, "sm_bar": "false", "tok_s_ratio_s5": 1.0, "tok_bar": "true",
     "h200_h_ratio": 1.84, "hours_bar": "false", "bars_met": False, "problems": ["in-flight never reached S=70 (max driver demand (served_children_live_concurrency) 64)", "H200-h actual 0.920 is 1.84x the sized 0.5 (bar: <= 1.5x)"]},
]
RUNNING_WHY = ("pasttense/s1 01TESTTASK-ccf6feacddc8: driver demand not recorded on the root; controls/s1 01TESTTASK-dc4bd2f90303: in-flight never >= S")


def _real_verdict(**over):
    readings = lp.utilisation_readings_from_table(REAL_ROWS)
    return lp.utilisation_verdict(bars=readings, source="RS/tables/utilisation_7c9060f.csv", source_sha256="7" * 64, W=W, execution_wave_uuid=EW,
                                  attempt_ids=[r["attempt_id"] for r in REAL_ROWS], measured_utc="2026-09-20T12:00:00Z", **over)


def test_fu3_the_real_three_root_table_builds_a_v2_verdict_with_unmeasured_running_and_a_waiver_carrying_the_reasons(tmp_path, capsys):
    readings = lp.utilisation_readings_from_table(REAL_ROWS)
    assert readings["running_over_max_num_seqs_min"] == {"unmeasured": True, "why": RUNNING_WHY}                     # any unmeasured row -> the bar is unmeasured, every reason named
    assert readings["sm_duty_min"] == 0.64 and readings["aggregate_tok_s_over_s5_min"] == 1.0 and readings["h200_hours_over_sized_max"] == 1.84   # the weakest row per bar
    assert lp.utilisation_reading(REAL_ROWS[1], "running_over_max_num_seqs_min") == 1.0                                # misalign: 21 / 21
    v = _real_verdict()
    assert v["schema"] == lp.UTILISATION_VERDICT_SCHEMA == "seedrep_utilisation_verdict/2" and v["bars_met"] is False
    assert v["bars"]["running_over_max_num_seqs_min"] == {"value": None, "threshold": 0.8, "met": False, "unmeasured": True, "why": RUNNING_WHY}
    assert v["bars"]["sm_duty_min"] == {"value": 0.64, "threshold": 0.85, "met": False} and v["bars"]["h200_hours_over_sized_max"] == {"value": 1.84, "threshold": 1.5, "met": False}
    assert v["bars"]["aggregate_tok_s_over_s5_min"] == {"value": 1.0, "threshold": 0.8, "met": True}
    assert lp.utilisation_verdict_problems(v, W=W, execution_wave_uuid=EW) == [] and lp.utilisation_verdict_problems(json.loads(json.dumps(v)), W=W) == []
    # the waiver copies the unmet list INCLUDING the unmeasured bar with its why; the stamp tool writes it
    w = lp.build_utilisation_waiver(v, W=W, execution_wave_uuid=EW, stamped_utc="2026-09-20T12:01:00Z")
    assert w["unmet"] == [{"bar": "running_over_max_num_seqs_min", "value": None, "threshold": 0.8, "met": False, "unmeasured": True, "why": RUNNING_WHY},
                          {"bar": "sm_duty_min", "value": 0.64, "threshold": 0.85, "met": False},
                          {"bar": "h200_hours_over_sized_max", "value": 1.84, "threshold": 1.5, "met": False}]
    assert lp.utilisation_waiver_problems(w, bars=v, W=W, execution_wave_uuid=EW) == []
    edited = json.loads(json.dumps(w)); edited["unmet"][0]["why"] = "hand-typed"
    assert any("is not the verdict's unmet bars" in p for p in lp.utilisation_waiver_problems(edited, bars=v, W=W, execution_wave_uuid=EW))
    rs = tmp_path / "rs"; rs.mkdir()
    plan = rs / "launch_plan.json"; plan.write_text(json.dumps({"W": W}))
    (rs / "smokes.json").write_text(json.dumps({"S5": {"W": W, "execution_wave_uuid": EW, "bars": v}}))
    assert stamp.main(["--rs", str(rs), "--plan", str(plan)]) == 0
    out = capsys.readouterr().out
    assert "running_over_max_num_seqs_min: value None vs threshold 0.8 (met False)" in out and json.loads((rs / lp.UTILISATION_WAIVER_FILE).read_text())["unmet"][0]["why"] == RUNNING_WHY
    # the table's bar text itself is an admissible reading; a bare `unmeasured` without its reason is not; a foreign string is not
    v2 = lp.utilisation_verdict(bars={**MET_READINGS, "running_over_max_num_seqs_min": "unmeasured(in-flight never >= S)"}, source="t", source_sha256="a" * 64, W=W,
                                execution_wave_uuid=EW, attempt_ids=["x"])
    assert v2["bars"]["running_over_max_num_seqs_min"]["why"] == "in-flight never >= S" and v2["bars_met"] is False
    with pytest.raises(ValueError, match="needs the table's reason"):
        lp.utilisation_verdict(bars={**MET_READINGS, "running_over_max_num_seqs_min": "unmeasured"}, source="t", source_sha256="a" * 64, W=W, execution_wave_uuid=EW, attempt_ids=["x"])
    with pytest.raises(ValueError, match="neither a number nor"):
        lp.utilisation_verdict(bars={**MET_READINGS, "running_over_max_num_seqs_min": "n/a"}, source="t", source_sha256="a" * 64, W=W, execution_wave_uuid=EW, attempt_ids=["x"])
    with pytest.raises(ValueError, match="must be the unmeasured form"):
        lp.utilisation_verdict(bars={**MET_READINGS, "running_over_max_num_seqs_min": {"value": 0.5}}, source="t", source_sha256="a" * 64, W=W, execution_wave_uuid=EW, attempt_ids=["x"])
    with pytest.raises(ValueError, match="neither a number nor `unmeasured`"):
        lp.utilisation_reading({**REAL_ROWS[1], "median_running_saturated": "n/a", "running_bar": "true"}, "running_over_max_num_seqs_min")


def test_fu3_every_malformed_unmeasured_form_refuses_by_name_and_a_v1_record_still_validates():
    v = _real_verdict()
    good = v["bars"]["running_over_max_num_seqs_min"]

    def probs(bar):
        return lp.utilisation_verdict_problems({**v, "bars": {**v["bars"], "running_over_max_num_seqs_min": bar}}, W=W, execution_wave_uuid=EW)
    assert any("is a number while unmeasured is true" in p for p in probs({**good, "value": 0.5}))
    assert any("while unmeasured -- an unmeasured bar is never met" in p for p in probs({**good, "met": True}))
    assert any("value is null without unmeasured true" in p for p in probs({**good, "unmeasured": False}))
    assert any("lacks ['unmeasured', 'why']" in p for p in probs({"value": None, "threshold": 0.8, "met": False}))          # value null without the form
    assert any("why is empty" in p for p in probs({**good, "why": "  "}))
    assert any("extra ['note']" in p for p in probs({**good, "note": "x"}))
    assert any("threshold 0.9 != plan" in p for p in probs({**good, "threshold": 0.9}))
    assert any("bars_met True disagrees" in p for p in lp.utilisation_verdict_problems({**v, "bars_met": True}, W=W, execution_wave_uuid=EW))
    # the unmeasured form in a /1 record refuses; a /1 record with numeric bars validates unchanged
    v1 = {**v, "schema": "seedrep_utilisation_verdict/1"}
    assert any("belongs to a seedrep_utilisation_verdict/2 record" in p for p in lp.utilisation_verdict_problems(v1, W=W, execution_wave_uuid=EW))
    numeric = _verdict(UNMET_READINGS)
    v1_numeric = {**numeric, "schema": "seedrep_utilisation_verdict/1"}
    assert lp.utilisation_verdict_problems(v1_numeric, W=W, execution_wave_uuid=EW) == [] and numeric["schema"] == "seedrep_utilisation_verdict/2"
    assert lp.build_utilisation_waiver(v1_numeric, W=W, execution_wave_uuid=EW)["unmet"] == lp.build_utilisation_waiver(numeric, W=W, execution_wave_uuid=EW)["unmet"]
    assert any("not in ['seedrep_utilisation_verdict/1', 'seedrep_utilisation_verdict/2']" in p for p in lp.utilisation_verdict_problems({**v, "schema": "seedrep_utilisation_verdict/3"}, W=W))


# --------------------------------------------------------------------------- #
# fu4: the row's BAR column speaks first -- a numeric diagnostic beside an `unmeasured(...)` bar never becomes the verdict's reading; #
# the JSON and CSV paths of the table give one verdict                                                                                     #
# --------------------------------------------------------------------------- #

import csv  # noqa: E402
import io  # noqa: E402

for _p in (REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import audit_common as ac  # noqa: E402
import seedrep_audit_fixtures as fx  # noqa: E402
import utilisation_table as ut  # noqa: E402


def _low_coverage_table(tmp_path):
    """The REAL reader over a synthetic root with counter resets: sampling coverage 0.5, the tok/s ratio kept as a
    diagnostic 1.125, the tok bar `unmeasured(sampling coverage 0.50 < 0.8)`, bars_met False -- written as the CLI writes it (CSV + JSON)."""
    root = fx.make_root(tmp_path / "stores", leg="pasttense", view="s1", arms=("none",))
    fx.populate_root(root, s=1, arms=("none",), n_items=1)
    root.write_samplers(reset_scrapes=[3, 5, 7, 9, 11])
    out = tmp_path / "tables" / "utilisation.csv"
    s5 = fx.s5_file(tmp_path / "smokes.json")
    exp = tmp_path / "hours.json"; exp.write_text(json.dumps({"pasttense": 3.0}))
    assert ac.run_main(ut.main, ["--roots", str(root.path), "--out", str(out), "--s5", str(s5), "--expected-hours", str(exp)]) == 0
    rows_json = json.loads(out.with_suffix(".json").read_text())
    with open(out, newline="") as f:
        rows_csv = list(csv.DictReader(f))
    return rows_json, rows_csv


def _verdict_of(rows):
    return lp.utilisation_verdict(bars=lp.utilisation_readings_from_table(rows), source="RS/tables/utilisation.csv", source_sha256="a" * 64, W=W, execution_wave_uuid=EW,
                                  attempt_ids=[str(r["attempt_id"]) for r in rows], measured_utc="2026-09-20T12:00:00Z")


def test_fu4_a_numeric_diagnostic_beside_an_unmeasured_bar_is_not_a_reading_json_and_csv_alike(tmp_path):
    rows_json, rows_csv = _low_coverage_table(tmp_path)
    (r,) = rows_json
    assert r["sampling_coverage"] == 0.5 and r["tok_s_ratio_s5"] == 1.125 and r["tok_bar"] == f"unmeasured(sampling coverage 0.50 < {ut.MIN_SAMPLING_COVERAGE})" and r["bars_met"] is False
    for rows, path in ((rows_json, "json"), (rows_csv, "csv")):
        readings = lp.utilisation_readings_from_table(rows)
        assert readings["aggregate_tok_s_over_s5_min"] == {"unmeasured": True, "why": f"pasttense/s1 {r['attempt_id']}: sampling coverage 0.50 < {ut.MIN_SAMPLING_COVERAGE}"}, path
        assert readings["running_over_max_num_seqs_min"] == 1.0 and readings["sm_duty_min"] == 0.92 and readings["h200_hours_over_sized_max"] == pytest.approx(0.667), path
        v = _verdict_of(rows)
        assert v["bars_met"] is False and v["bars"]["aggregate_tok_s_over_s5_min"]["unmeasured"] is True and v["bars"]["aggregate_tok_s_over_s5_min"]["value"] is None, path
        assert lp.utilisation_verdict_problems(v, W=W, execution_wave_uuid=EW) == [], path
    assert _verdict_of(rows_json) == _verdict_of(rows_csv)                        # the two paths of the same table: ONE verdict
    # every bar: a numeric cell beside an `unmeasured(...)` bar text -> unmeasured, the bar's reason, never the number
    base = dict(REAL_ROWS[1])
    for key, (col, bar_col) in lp.UTILISATION_TABLE_CELLS.items():
        row = {**base, col: 0.99 if key != "running_over_max_num_seqs_min" else 20.9, bar_col: "unmeasured(synthetic reason)"}
        assert lp.utilisation_reading(row, key) == {"unmeasured": True, "why": f"misalign/s1 {base['attempt_id']}: synthetic reason"}, key
        row_csv = {k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v)) for k, v in row.items()}      # the CSV's string cells
        assert lp.utilisation_reading(row_csv, key) == lp.utilisation_reading(row, key), key
    # the three-root fixture aggregation is unchanged
    readings = lp.utilisation_readings_from_table(REAL_ROWS)
    assert readings == {"running_over_max_num_seqs_min": {"unmeasured": True, "why": RUNNING_WHY}, "sm_duty_min": 0.64, "aggregate_tok_s_over_s5_min": 1.0, "h200_hours_over_sized_max": 1.84}
    rows_csv3 = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v)) for k, v in r.items()} for r in REAL_ROWS]
    assert lp.utilisation_readings_from_table(rows_csv3) == readings
