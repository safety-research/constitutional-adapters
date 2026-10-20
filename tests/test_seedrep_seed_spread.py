"""The per-leg seed-spread table on a
SYNTHETIC grades tree written in every upstream producer's schema plus a synthetic export
slice: per-seed values through the triplet gate, SD / range, the two contrasts, the paper
deltas from each declared source, the cross-view floor, the claim rule
|Δ_paper| > max(range, 2·SD), non_comparable legs excluded from Δ(difference − none) and
listed, and an incomplete triplet written as unavailable with the seed-spread check FAILED."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SEEDREP = REPO / "scripts" / "sdfbase" / "seedrep"
FIX = REPO / "tests" / "fixtures" / "seedrep"
if str(SEEDREP) not in sys.path:
    sys.path.insert(0, str(SEEDREP))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SEEDREP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load("seed_spread")
tg = sys.modules["triplet_gate"]
VALUES = json.loads((FIX / "synthetic_grades_values.json").read_text())
EXPORT = json.loads((FIX / "synthetic_export_slice.json").read_text())
CELL = VALUES["cells"]                       # short -> cell id
SHORT = {v: k for k, v in CELL.items()}
TEST_LEGS = [ss.LEG_BY_ID[e] for e in ("pasttense", "xstest", "mmlu", "gsm8k", "bon", "crescendo", "pair", "actorattack",
                                       "petri_pooled", "defer_to_users", "agentic_leaking")]


def _cid(short_or_none: str) -> str:
    return "none" if short_or_none == "none" else CELL[short_or_none]


# ---------------------------------------------------------------- producers' schemas, materialised
def _final_rung_entry(kind: str, p: float) -> dict:
    stat = ({"asr": {"0.5": {"p": p, "k": round(p * 518), "n": 518, "n_goals": 259}}, "final_caps": [8192]}
            if kind == "static" else {"full_refusal": {"p": p, "k": round(p * 500), "n": 500, "n_items": 250}, "final_caps": [8192]})
    return {"kind": kind, "held": False, "status": "ok", "pooled": stat, "trials": {"0": {}, "1": {}}}


def write_final_rung(path: Path, tags: dict[str, dict[str, dict[str, float]]]) -> None:
    """tags: tag -> {"pasttense|<cid>": p, "xstest|<cid>": p, ...}"""
    targets = {}
    for tag, entries in tags.items():
        t = {"static": {}, "controls": {}, "headline_threshold": 0.5, "cells": []}
        for key, p in entries.items():
            readout, cid = key.split("|", 1)
            if readout in ("xstest", "or_bench_hard_1k"):
                t["controls"][key] = _final_rung_entry("controls", p)
            else:
                t["static"][key] = _final_rung_entry("static", p)
        targets[tag] = t
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": "synthetic", "targets": targets, "flag_pp": 3.5}))


def write_controls_agg(path: Path, views: dict[str, dict[str, float]]) -> None:
    d = {"views": {v: {cid: {"gen_stats": {"mmlu": {"acc": acc, "n": 200}}} for cid, acc in cells.items()} for v, cells in views.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))


def write_gsm8k_report(path: Path, cells: dict[str, float]) -> None:
    """cells: "qwen35_4b/<view>/<cid>" -> acc (200 scored)"""
    d = {"cells": {k: {"cell_key": k, "readings": {"merged": {"n_correct": round(acc * 200), "n_scored": 200, "escalated": False}},
                       "status": "complete"} for k, acc in cells.items()}, "legs": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))


def write_transfer_stats(path: Path, models: dict[str, dict[str, dict[str, float]]]) -> None:
    """models: view -> cid -> family -> p"""
    d = {"meta": {"threshold": 0.5, "bon_n_max": 75}, "models": {}}
    for view, cells in models.items():
        d["models"][view] = {"view": view.split("_")[-1], "base_target": "qwen35_4b", "cells": {}}
        for cid, fams in cells.items():
            d["models"][view]["cells"][cid] = {}
            for fam, p in fams.items():
                n = 100 if fam == "bon" else 259
                blk = {"k": round(p * n), "n": n, "p": p, "asr_reported": p, "decode_cap_tokens": 4096, "tier": "bon" if fam == "bon" else "multi_turn"}
                if fam == "bon":
                    blk["asr_at_n"] = {"1": p / 4, "75": p}
                else:
                    blk["asr_grid"] = {"0.25": min(1.0, p + 0.1), "0.5": p, "0.75": p / 2}
                d["models"][view]["cells"][cid][fam] = blk
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))


def write_petri_stats(path: Path, tags: dict[str, dict[str, dict]]) -> None:
    """tags: tag -> cid -> {"pooled": float, "behaviours": {name: mean}}"""
    d = {"meta": {}, "tags": {}}
    for tag, cells in tags.items():
        d["tags"][tag] = {"cells": {}}
        for cid, spec in cells.items():
            d["tags"][tag]["cells"][cid] = {
                "status": "COMPLETE", "audited_items": 60, "audited_expected": 60, "mean_of_behavior_means": spec["pooled"],
                "behaviors": {b: {"mean": m, "n_items": 10, "incomplete": False} for b, m in spec["behaviours"].items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))


def write_misalign(dirpath: Path, target: str, arms: dict[str, float], behaviour: str = "leaking") -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    cols = ["target", "arm", "family", "behavior", "metric", "n", "k", "estimate", "ci_low", "ci_high", "ci_method", "boot_low", "boot_high", "note"]
    with (dirpath / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for arm, est in arms.items():
            for metric, val in (("harmful", est), ("at_cap", 0.0), ("harmful_bare", 0.0)):
                w.writerow({"target": target, "arm": arm, "family": "agentic", "behavior": behaviour, "metric": metric, "n": 510,
                            "k": round(val * 510), "estimate": val, "ci_low": val, "ci_high": val, "ci_method": "wilson95",
                            "boot_low": val, "boot_high": val, "note": ""})
    with (dirpath / "paired.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "arm", "ref", "family", "behavior", "metric", "n_pairs", "mean_arm", "mean_none",
                                           "diff", "boot_low", "boot_high", "arm_only", "none_only", "mcnemar_p", "note"])
        w.writeheader()


def write_preflight(root: Path, view: str, leg: str, verdict: str) -> None:
    p = root / "anchor_preflight" / f"{view}_{leg}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"view": view, "leg": leg, "anchor_sampler_verdict": verdict, "anchor": "synthetic"}))


def build_export(path: Path, spec: dict = EXPORT) -> Path:
    rows, rid = [], 0
    for e in spec["entries"]:
        for view, cells in e["views"].items():
            for short, val in cells.items():
                cid = _cid(short)
                d = (e.get("delta") or {}).get(view, {}).get(short)
                rows.append({"row_id": rid, "model": "qwen35_4b", "served_model": spec["served_model"], "view": view, "dose": 1.0,
                             "cell_id": cid, "eval_id": e["eval_id"], "metric": e["metric"], "threshold": e["threshold"],
                             "bon_n": e["bon_n"], "decode_cap_tokens": e["decode_cap_tokens"], "decode": e["decode"],
                             "status": e["status"], "value": val, "k": None, "n": None, "delta": d,
                             "delta_lo": None if d is None else d - 5, "delta_hi": None if d is None else d + 5,
                             "delta_method": None if d is None else "synthetic paired bootstrap"})
                rid += 1
    df = pd.DataFrame(rows)
    for c in ("threshold", "bon_n", "decode_cap_tokens", "delta", "delta_lo", "delta_hi", "k", "n"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(path, index=False)
    return path


def build_tree(root: Path, paper: Path, *, values: dict = VALUES) -> None:
    lane = values["lane"]
    seeds = (1, 2, 3)
    # sampled final rung: static + over-refusal for the three lane tags
    write_final_rung(root / "sampled/report/analysis/final_rung.json",
                     {f"qwen35_4b_s{s}": {f"{leg}|{CELL[c]}": v[str(s)] for leg in ("pasttense", "xstest")
                                          for c, v in lane[leg].items() if str(s) in v} for s in seeds})
    # the paper's own final rung (tag qwen35_4b, incl. none)
    write_final_rung(paper / "final_rung.json",
                     {"qwen35_4b": {f"{leg}|{_cid(c)}": v for leg in ("pasttense", "xstest") for c, v in values["paper"][leg].items()}})
    write_controls_agg(root / "controls_agg.json", {f"qwen35_4b_s{s}": {CELL[c]: v[str(s)] for c, v in lane["mmlu"].items()} for s in seeds})
    write_gsm8k_report(root / "gsm8k/report.json", {f"qwen35_4b/s{s}/{CELL[c]}": v[str(s)] for s in seeds for c, v in lane["gsm8k"].items() if str(s) in v})
    write_gsm8k_report(paper / "gsm8k_report.json", {f"qwen35_4b/main/{_cid(c)}": v for c, v in values["paper"]["gsm8k"].items()})
    write_transfer_stats(root / "grades_reseed/transfer_stats_reseed.json",
                         {f"qwen35_4b_s{s}": {CELL[c]: {fam: lane[fam][c][str(s)] for fam in ("bon", "crescendo", "pair", "actorattack")} for c in CELL} for s in seeds})
    write_petri_stats(root / "petri_stats/petri_reseed_stats.json",
                      {f"qwen35_4b_s{s}": {CELL[c]: {"pooled": lane["petri_pooled"][c][str(s)],
                                                     "behaviours": {"defer_to_users": lane["defer_to_users"][c][str(s)]}} for c in CELL} for s in seeds})
    for s in seeds:
        write_misalign(root / f"misalign/qwen35_4b_s{s}", f"qwen35_4b_s{s}", {ss.misalign_arm(CELL[c]): lane["agentic_leaking"][c][str(s)] for c in CELL})
    write_misalign(paper / "misalign/qwen35_4b", "qwen35_4b", {("none" if c == "none" else ss.misalign_arm(CELL[c])): v for c, v in values["paper"]["agentic_leaking"].items()})
    for s in seeds:
        for leg in values["preflight"]["comparable"]:
            write_preflight(root, f"qwen35_4b_s{s}", leg, "comparable")
        for leg, by_seed in values["preflight"]["per_seed"].items():
            write_preflight(root, f"qwen35_4b_s{s}", leg, by_seed[str(s)])
    (root / "admission").mkdir(parents=True, exist_ok=True)
    (root / "admission" / "derived.json").write_text(json.dumps(values["admission"]))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("seedrep")
    root, paper, out = base / "RS_reseed", base / "paper", base / "out" / "spread"
    build_tree(root, paper)
    export = build_export(base / "export.parquet")
    roots = ss.Roots(grades=root, export=ss.load_export(export), paper_final_rung=paper / "final_rung.json",
                     paper_gsm8k=paper / "gsm8k_report.json", paper_misalign=paper / "misalign",
                     admissions=ss.load_admissions(root))
    res = ss.run(roots, out, legs=TEST_LEGS)
    return {"res": res, "out": out, "root": root, "paper": paper, "roots": roots, "base": base}


def _leg(res, eval_id):
    return next(l for l in res["legs"] if l["eval_id"] == eval_id)


def _claim(leg, arm, contrast):
    return next(k for k in leg["claims"] if k["arm"] == arm and k["contrast"] == contrast)


# ---------------------------------------------------------------- readers consume the producers' schemas
def test_static_final_rung_values_per_seed_in_pp(run):
    leg = _leg(run["res"], "pasttense")
    a = leg["arms"][CELL["CL"]]
    assert a["values"] == {"s1": 10.0, "s2": 12.0, "s3": pytest.approx(11.0)}
    assert a["status"] == "ok" and a["criterion_7"] == "PASS"
    assert a["spread"]["range"] == pytest.approx(2.0) and a["spread"]["sd"] == pytest.approx(1.0)
    assert "targets.qwen35_4b_s1.static[pasttense|" in a["readings"]["s1"]["source"]
    # the paper's static reading is the paper's final rung, in the same schema, not the export's greedy row
    assert leg["paper_source"] == "final_rung"
    assert leg["paper"][CELL["DL"]]["value"] == pytest.approx(4.0) and leg["paper"]["none"]["value"] == pytest.approx(35.0)
    assert leg["paper_export_beside"][CELL["DL"]]["value"] == pytest.approx(3.9)   # the greedy 4,096 row rides beside


def test_every_reader_kind_produces_a_value(run):
    res = run["res"]
    expect = {"xstest": (CELL["CL"], 16.0), "mmlu": (CELL["KL"], 72.0), "gsm8k": (CELL["CL"], 87.0), "bon": (CELL["DL"], 30.0),
              "crescendo": (CELL["DP"], 60.0), "petri_pooled": (CELL["DL"], 4.2), "defer_to_users": (CELL["DP"], 7.0),
              "agentic_leaking": (CELL["KL"], 18.0)}
    for eval_id, (cid, v1) in expect.items():
        leg = _leg(res, eval_id)
        assert leg["arms"][cid]["values"]["s1"] == pytest.approx(v1), eval_id
        assert leg["arms"][cid]["status"] == "ok", (eval_id, leg["arms"][cid]["status"])
    # paper sources per leg
    assert _leg(res, "gsm8k")["paper"]["none"]["value"] == pytest.approx(90.0) and "qwen35_4b/main/none" in _leg(res, "gsm8k")["paper"]["none"]["source"]
    assert _leg(res, "agentic_leaking")["paper"]["none"]["value"] == pytest.approx(20.0)
    assert _leg(res, "bon")["paper"]["none"]["value"] == pytest.approx(88.0) and _leg(res, "bon")["paper"]["none"]["source"].startswith("export row_id")


def test_units_are_declared_pp_for_rates_and_score_for_petri(run):
    res = run["res"]
    assert _leg(res, "petri_pooled")["units"] == "score" and _leg(res, "petri_pooled")["arms"][CELL["DL"]]["values"]["s2"] == 4.4
    assert _leg(res, "pasttense")["units"] == "pp"


# ---------------------------------------------------------------- contrasts, paper deltas, claims
def test_contrasts_and_claim_rule(run):
    leg = _leg(run["res"], "pasttense")
    cc = leg["contrasts"]["lora"]["const_minus_ctl"]
    assert cc["per_seed"] == {"s1": pytest.approx(-20.0), "s2": pytest.approx(-19.0), "s3": pytest.approx(-18.0)}
    assert cc["spread"]["sd"] == pytest.approx(1.0) and cc["spread"]["range"] == pytest.approx(2.0)
    dn = leg["contrasts"]["lora"]["diff_minus_none"]
    assert dn["per_seed"]["s1"] == pytest.approx(4.0 - 35.0) and dn["anchor"]["reused"] and dn["anchor"]["value"] == pytest.approx(35.0)
    paper = leg["contrasts"]["lora"]["paper"]
    assert paper["const_minus_ctl"]["delta"] == pytest.approx(11.0 - 30.0)
    assert paper["diff_minus_none"]["delta"] == pytest.approx(4.0 - 35.0)
    k = _claim(leg, "lora", "const_minus_ctl")
    assert k["threshold_2sd"] == pytest.approx(2.0) and k["threshold_range"] == pytest.approx(2.0) and k["threshold"] == pytest.approx(2.0)
    assert k["exceeds"] is True and k["status"] == "exceeds the seed spread" and k["delta_paper"] == pytest.approx(-19.0)


def test_claim_that_does_not_exceed_and_export_delta_rides_beside_verbatim(run):
    leg = _leg(run["res"], "crescendo")
    dn = leg["contrasts"]["psr"]["diff_minus_none"]
    # DP seeds .60/.75/.68 vs none .70 -> -10, +5, -2 ; range 15, 2sd ≈ 15.1 ; paper .66 - .70 = -4
    assert dn["per_seed"] == {"s1": pytest.approx(-10.0), "s2": pytest.approx(5.0), "s3": pytest.approx(-2.0)}
    k = _claim(leg, "psr", "diff_minus_none")
    assert k["delta_paper"] == pytest.approx(-4.0) and k["exceeds"] is False and k["status"] == "does not exceed the seed spread"
    assert k["threshold"] == pytest.approx(max(15.0, 2 * dn["spread"]["sd"]))
    # the multi-turn paper key selects the 4,096 sampled row, not the 512 row that carries a filed delta
    p = leg["paper"][CELL["DL"]]
    assert p["value"] == pytest.approx(35.0) and p["extra"]["decode_cap_tokens"] == 4096 and p["extra"]["delta"] is None
    # BoN: the export's filed delta (its own units) rides beside the recomputed one
    bon = _leg(run["res"], "bon")["contrasts"]["lora"]["paper"]["diff_minus_none"]
    assert bon["delta"] == pytest.approx(30.0 - 88.0) and bon["export_delta"] == -58.0 and bon["export_delta_lo"] == -63.0
    # Petri delta in score units
    pet = _leg(run["res"], "petri_pooled")["contrasts"]["lora"]["paper"]["diff_minus_none"]
    assert pet["delta"] == pytest.approx(4.28 - 6.55) and pet["export_delta"] == -2.27


def test_cross_view_floor_is_the_none_range_across_qwen_views(run):
    leg = _leg(run["res"], "pasttense")
    fl = leg["floor"]
    assert fl["n_views"] == 3 and fl["range"] == pytest.approx(0.8) and fl["units"] == "pp"
    assert set(fl["none_by_view"]) == {"qwen35_4b", "qwen35_4b_it", "qwen35_4b_halfdose"}
    assert _leg(run["res"], "crescendo")["floor"]["range"] == pytest.approx(2.0)
    assert _leg(run["res"], "gsm8k")["floor"] is None          # one view only -> no floor, never a made-up one
    assert _claim(leg, "lora", "const_minus_ctl")["floor"] == pytest.approx(0.8)


# ---------------------------------------------------------------- the gate and the anchor verdict
def test_non_comparable_preflight_excludes_the_anchor_delta_and_is_listed(run):
    res = run["res"]
    leg = _leg(res, "pair")
    assert leg["preflight"]["s2"]["comparable"] is False and leg["preflight"]["s1"]["comparable"] is True
    dn = leg["contrasts"]["lora"]["diff_minus_none"]
    assert dn["spread"] is None and dn["criterion_7"] == "FAILED"
    assert dn["status"].startswith("unavailable: incomplete replicate triplet") and "s2 — non_comparable" in dn["status"]
    # the arm spread and the constitution-control contrast are still reported
    assert leg["arms"][CELL["DL"]]["status"] == "ok" and leg["contrasts"]["lora"]["const_minus_ctl"]["status"] == "ok"
    ex = [e for e in res["excluded_non_comparable"] if e["leg"] == "pair"]
    assert len(ex) == 1 and ex[0]["view"] == "qwen35_4b_s2" and ex[0]["verdict"].startswith("non_comparable")
    assert _claim(leg, "psr", "diff_minus_none")["exceeds"] is None
    # the other multi-turn legs carry their own comparable records and are NOT excluded
    assert not any(e["leg"] in ("crescendo", "actorattack") for e in res["excluded_non_comparable"])
    assert _leg(res, "crescendo")["contrasts"]["psr"]["diff_minus_none"]["spread"] is not None


def test_anchor_borrowed_from_sibling_view_is_recorded_never_silent(run):
    res = run["res"]
    leg = _leg(res, "actorattack")
    an = leg["contrasts"]["lora"]["diff_minus_none"]["anchor"]
    # the main view filed no `none` at the 4,096 key; the instruct-direct view's (same checkpoint) is borrowed
    assert an["borrowed"] is True and an["view"] == "qwen35_4b_it" and an["value"] == pytest.approx(95.0)
    assert "BORROWED" in leg["paper"]["none"]["source"]
    assert any("actorattack" in w and "borrowed from view qwen35_4b_it" in w for w in res["warnings"])
    dn = leg["contrasts"]["lora"]["diff_minus_none"]
    assert dn["per_seed"]["s1"] == pytest.approx(46.0 - 95.0) and dn["status"] == "ok"
    assert leg["contrasts"]["lora"]["paper"]["diff_minus_none"]["delta"] == pytest.approx(46.0 - 95.0)
    # the 512 row with a filed delta is NOT what the 4,096 key selects
    assert leg["paper"][CELL["DL"]]["extra"]["decode_cap_tokens"] == 4096
    # a non-borrowed anchor says so
    assert _leg(res, "bon")["contrasts"]["lora"]["diff_minus_none"]["anchor"]["borrowed"] is False
    md = (run["out"] / "seed_spread.md").read_text()
    assert "Anchors borrowed from a sibling view" in md and "ActorAttack ← `qwen35_4b_it`" in md


def test_missing_store_is_unavailable_with_the_admission_reason_and_no_survivor_sd(run):
    leg = _leg(run["res"], "gsm8k")
    a = leg["arms"][CELL["DL"]]
    assert a["criterion_7"] == "FAILED" and a["spread"] is None
    assert "s3 — not admitted: recovery verdict not recovered" in a["status"]
    assert a["values"]["s3"] is None and a["values"]["s1"] == pytest.approx(88.0)
    assert leg["contrasts"]["lora"]["diff_minus_none"]["spread"] is None
    assert leg["contrasts"]["lora"]["const_minus_ctl"]["status"] == "ok"        # the parents are complete
    k = _claim(leg, "lora", "diff_minus_none")
    assert k["exceeds"] is None and k["sd_seed"] is None and "s3" in k["status"]
    assert run["res"]["criterion_7"]["gsm8k"][CELL["DL"]] == "FAILED"


def test_two_transfer_stats_files_carrying_one_cell_refuse(run):
    root = run["root"]
    dup = root / "grades_dup" / "transfer_stats_dup.json"
    write_transfer_stats(dup, {"qwen35_4b_s1": {CELL["DL"]: {"bon": 0.31}}})
    try:
        with pytest.raises(RuntimeError, match="refusing to pick"):
            ss.read_transfer_stats(root, "qwen35_4b_s1", "bon", CELL["DL"], "pp")
    finally:
        dup.unlink()
        dup.parent.rmdir()


def test_missing_preflight_record_is_never_assumed_comparable(tmp_path):
    root = tmp_path / "RS"
    root.mkdir()
    v, p = ss.read_preflight(root, "qwen35_4b_s1", ss.LEG_BY_ID["bon"])
    assert v is None and p is None and ss.comparable(v) is False
    assert ss.comparable("comparable") and not ss.comparable("non_comparable(x)")


# ---------------------------------------------------------------- outputs
def test_outputs_written_json_md_png(run):
    out, res = run["out"], run["res"]
    assert (out / "seed_spread.json").is_file() and (out / "seed_spread.md").is_file()
    assert Path(res["figures"]["seed_spread"]).is_file() and Path(res["figures"]["seed_spread"]).parent == run["base"] / "out" / "figs"
    md = (out / "seed_spread.md").read_text()
    assert "max(range_seed, 2·SD_seed)" in md and "## claim list" in md
    assert "exceeds the seed spread" in md and "does not exceed the seed spread" in md
    assert "excluded from Δ(difference − none) by the anchor preflight" in md and "qwen35_4b_s2" in md
    assert "unavailable: incomplete replicate triplet" in md
    j = json.loads((out / "seed_spread.json").read_text())
    assert [l["eval_id"] for l in j["legs"]] == [l.eval_id for l in TEST_LEGS]
    assert j["seeds"] == [1, 2, 3] and j["views"] == ["qwen35_4b_s1", "qwen35_4b_s2", "qwen35_4b_s3"]


def test_full_leg_roster_runs_with_missing_stores_as_no_store_rows(run):
    out = run["base"] / "out_full" / "spread"
    res = ss.run(run["roots"], out, make_figs=False)
    assert len(res["legs"]) == len(ss.LEGS)
    cod = _leg(res, "codeattack")["arms"][CELL["CL"]]
    assert cod["criterion_7"] == "FAILED" and "no store" in cod["status"]
    cats = {l["category"] for l in res["legs"]}
    assert cats == set(ss.CATEGORY_ORDER)
