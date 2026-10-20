"""The coverage contract (schema_def.judgment_covers_sql) reads
``n_items_judged + n_judge_refusals (+ n_errors) >= planned`` -- so ``n_items_judged`` counts the items WITH A VERDICT and the judge's
refusals are counted apart. One test per grader we own, each asserting judged + refused == the scored rows and judged EXCLUDES the
refused, on synthetic fixtures:

* misalign (scripts/sdfbase/full22/misalign/misalign_judge.py) -- FIXED: it filed `len(mine)` (verdicts AND blocked rows) as
  n_items_judged (509 for 502 + 7 on the s1 control-LoRA leaking run).
* the static sampled grader (scripts/sdfbase/full22/sampled/grade_static_sampled.py) -- verified correct: n_items_judged is
  `n_responses_graded`, the row count of the clean mapping, and every response in it carries a verdict (the live grader is built with
  `allow_unscored=0` and raises on an item still UNSCORED after its retries, so a judge refusal never reaches the count).
* the over-refusal controls (scripts/served_capability_judge.py) -- verified correct: `n_judged` and `n_abstained` are filed apart.
* the multi-turn grader (scripts/sdfbase/seedrep/grade_full22_native.py) -- verified correct: n_items_judged is `n_goals` of the clean
  mapping under the same UNSCORED contract.
* Petri (scripts/petri_run.py) -- verified correct: n_items_judged is the distinct scored items of the frame; the refusals come
  from the analysis footing and are filed as n_judge_refusals.
The two gated graders' filing sites sit inside `main` (driven only by the sampled-world fixtures), so for them the per-item row builders
are exercised and the filing expression itself is pinned from the module source."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity
from sisteer.registry import records as R
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "src", REPO / "scripts" / "sdfbase" / "full22" / "misalign"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def _source(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _goal(gid: str):
    return SimpleNamespace(goal_id=gid, dataset="synthetic", revision="v1", text=f"synthetic goal {gid}")


# --------------------------------------------------------------------------------------------
# misalign
# --------------------------------------------------------------------------------------------


def test_misalign_judged_excludes_the_blocked_rows_and_judged_plus_refused_is_the_scored_rows():
    MJ = _load("misalign_judge_sweep", "scripts/sdfbase/full22/misalign/misalign_judge.py")
    row = {"run_id": "r-leak", "eval_id": "agentic_leaking"}
    responses = [{"item_id": f"L{i}", "behavior": "leaking", "sample_idx": 0} for i in range(10)]
    judged = [{"item_id": f"L{i}", "behavior": "leaking", "sample_idx": 0, "judge_blocked": i >= 7, "classifier_harmful": None if i >= 7 else 0,
               "harmful": None if i >= 7 else 0} for i in range(10)]
    fc = MJ.run_filing_counts(row, judged, responses, [], "agentic", "cellA", len(responses))
    assert fc["n_verdicts"] + fc["n_blocked"] == len(judged) == 10 and fc["n_blocked"] == 3 and fc["n_verdicts"] == 7
    assert fc["status"] == "complete" and fc["n_errors"] == 0
    items = MJ.judged_item_rows("agentic", fc["mine"])
    assert sum(1 for r in items if r["label"] == "judge_blocked") == 3 * 2 and sum(1 for r in items if r["value"] is not None) == 7 * 2
    # the filing site hands the split to the scope (never len(mine))
    src = _source("scripts/sdfbase/full22/misalign/misalign_judge.py")
    assert re.search(r'scope\.complete\(fc\["status"\],\s*n_items_judged=fc\["n_verdicts"\],\s*n_judge_refusals=fc\["n_blocked"\],\s*n_errors=fc\["n_errors"\]\)', src)
    assert 'n_items_judged=len(mine)' not in src


# --------------------------------------------------------------------------------------------
# the static sampled grader
# --------------------------------------------------------------------------------------------


def test_static_sampled_judged_is_the_graded_responses_every_one_with_a_verdict_and_no_refusal_channel():
    gss = _load("grade_static_sampled_sweep", "scripts/sdfbase/full22/sampled/grade_static_sampled.py")
    goals = [_goal("g1"), _goal("g2"), _goal("g3")]
    by_id = {g.goal_id: g for g in goals}
    responses = {"g1": ["r1a", "r1b"], "g2": ["r2a"], "g3": []}            # g3: zero persisted responses -> its asr row only
    scores = {(g.dataset, g.revision, g.text, r): 0.25 * (i + 1) for g in goals for i, r in enumerate(responses[g.goal_id])}
    m = {"n_goals": 3, "n_responses_graded": sum(len(v) for v in responses.values()),
         "per_goal": {"g1": {"queries_to_first_success": 2, "best_score": 0.5, "n_target_forwards": 2},
                      "g2": {"queries_to_first_success": -1, "best_score": 0.25, "n_target_forwards": 1},
                      "g3": {"queries_to_first_success": -1, "best_score": 0.0, "n_target_forwards": 0}}}
    rows = gss.registry_item_rows_static(m, responses, by_id, scores, trial=0, canonical=0.5)
    verdicts = [r for r in rows if r["metric_id"] == "strongreject_score"]
    assert len(verdicts) == m["n_responses_graded"] == 3 and all(r["value"] is not None and r["label"] is None for r in verdicts)
    assert sorted(r["item_id"] for r in rows if r["metric_id"] == "asr") == ["g1", "g2", "g3"]
    # a response without a verdict is a KeyError at the row builder (never a silent refusal count): the grader cannot file it
    with pytest.raises(KeyError):
        gss.registry_item_rows_static(m, {**responses, "g2": ["r2a", "unscored"]}, by_id, scores, trial=0, canonical=0.5)
    src = _source("scripts/sdfbase/full22/sampled/grade_static_sampled.py")
    assert re.search(r'scope\.complete\("complete",\s*n_items_judged=int\(m\.get\("n_responses_graded"\)', src)   # the filing: graded responses
    assert "allow_unscored=0" in src and "n_judge_refusals" not in src                                             # no refusal channel reaches the count


# --------------------------------------------------------------------------------------------
# the over-refusal controls (served_capability_judge)
# --------------------------------------------------------------------------------------------


def _control_items(control: str, n: int = 6) -> list:
    rows = []
    for i in range(n):
        status = "judged" if i % 3 else "abstained"
        rec = {"control": control, "task": f"{control}_task", "item_id": f"{control}-{i}", "request_key": f"k{i}", "judge_model": "judge-model-x",
               "rubric_sha": synth.SHA256_A, "protocol_sha": synth.SHA256_B, "cache_org": "lowprio", "verdict_created_at": 1.0,
               "cached_before_run": False, "status": status}
        if status == "judged":
            rec.update(label="full_refusal" if i % 2 else "compliance", label_source="store")
        else:
            rec.update(abstention_reason="refusal")
        rows.append(rec)
    return rows


def test_controls_judged_and_abstained_are_filed_apart_and_sum_to_the_scored_rows(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    drv = __import__("served_capability_judge")
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = synth.run_records("k_ctl", 100, through="generated")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), recs)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    spool = os.path.join(root, "records")
    monkeypatch.setenv(emit.SPOOL_ENV, spool)
    cell_dir = tmp_path / "controls" / "synthtarget" / "cells" / "constitution_lora__xstest__t0__a0__a1b2c3d4__0f1e2d3c"
    cell_dir.mkdir(parents=True)
    cell = types.SimpleNamespace(target="synthtarget", name=cell_dir.name, cell_dir=cell_dir, judge_model="judge-model-x")
    items = _control_items("xstest")
    n_judged = sum(1 for r in items if r["status"] == "judged")
    n_abst = sum(1 for r in items if r["status"] == "abstained")
    summary = {"target": "synthtarget", "cell": cell_dir.name, "cell_id": "constitution_lora", "intervention_hash": "a1b2c3d4e5f60718", "model_id": synth.MODEL_ID,
               "controls": {"xstest": {"status": "ok", "task": "xstest_task", "n": len(items), "at_cap_share": 0.5,
                                       "over_refusal": {"point_estimate_over_judged": 0.5, "wilson95_over_judged": [0.15, 0.85], "n_judged": n_judged,
                                                        "n": len(items), "n_abstained": n_abst, "abstention_rate": n_abst / len(items),
                                                        "abstention_band_full_n": [0.33, 0.67], "full_refusal_n": 2}}}}
    monkeypatch.setattr(drv, "REGISTRY_RUNS", {f"synthtarget/{cell_dir.name}": {"xstest": {"run_id": rid, "campaign": synth.CAMPAIGN}}})
    drv._registry_record_store(cell, summary, items, types.SimpleNamespace(org="lowprio"))
    recs_out = [rec for p in R.iter_record_files(os.path.join(spool, synth.CAMPAIGN, identity.run_dir_name(rid))) for _, rec, _ in R.read_record_file(p)
                if rec["producer"]["script"] == "scripts/served_capability_judge.py"]
    judgments = [r["payload"] for r in recs_out if r["record_type"] == "judgment"]
    item_rows = [r["payload"] for r in recs_out if r["record_type"] == "item_result"]
    done = judgments[-1]
    assert done["status"] == "complete" and (done["n_items_judged"], done["n_judge_refusals"]) == (n_judged, n_abst) == (4, 2)
    assert done["n_items_judged"] + done["n_judge_refusals"] == len(item_rows) == len(items) == 6
    assert sum(1 for r in item_rows if r["value"] is None and str(r["label"]).startswith("abstained:")) == n_abst      # judged excludes the abstained


# --------------------------------------------------------------------------------------------
# the multi-turn grader
# --------------------------------------------------------------------------------------------


def test_multiturn_judged_is_the_goals_graded_each_reply_with_a_verdict_and_no_refusal_channel():
    gfn = _load("grade_full22_native_sweep", "scripts/sdfbase/seedrep/grade_full22_native.py")
    goals = [_goal("g1"), _goal("g2"), _goal("g3")]
    by_id = {g.goal_id: g for g in goals}
    responses = {"g1": ["t1", "t2", "t3"], "g2": ["t1"]}                       # g3 excluded by the harness (errored / refused render): no per_goal entry
    scores = {(g.dataset, g.revision, g.text, r): 0.1 * (i + 1) for g in goals for i, r in enumerate(responses.get(g.goal_id, []))}
    m = {"n_goals": 2, "per_goal": {"g1": {"queries_to_first_success": 3, "best_score": 0.3}, "g2": {"queries_to_first_success": -1, "best_score": 0.1}}}
    rows = gfn.multiturn_item_rows(m, responses, by_id, ["g1", "g2", "g3"], scores, canonical=0.25)
    asr = [r for r in rows if r["metric_id"] == "asr"]
    verdicts = [r for r in rows if r["metric_id"] == gfn.STRONGREJECT_ITEM_METRIC]
    assert len(asr) == m["n_goals"] == 2 and len(verdicts) == 4 and all(r["value"] is not None for r in verdicts)
    assert [r["item_id"] for r in asr] == ["g1", "g2"]                                                                  # the excluded goal has no row
    src = _source("scripts/sdfbase/seedrep/grade_full22_native.py")
    assert len(re.findall(r'scope\.complete\("complete",\s*n_items_judged=int\(m\["n_goals"\]\)', src)) == 2            # both filing sites: goals graded
    assert "allow_unscored" in src and "n_judge_refusals" not in src and "UNSCORED" in src                              # an unscorable item raises; no refusal count


# --------------------------------------------------------------------------------------------
# Petri
# --------------------------------------------------------------------------------------------


def test_petri_judged_is_the_scored_items_of_the_frame_and_the_footings_refusals_are_filed_apart():
    prun = _load("petri_run_sweep", "scripts/petri_run.py")
    items_scored = ["s1", "s2", "s3"]
    frame = pd.DataFrame([{"arm_name": "none", "task": "petri/self_preservation", "behavior": "self_preservation", "metric": metric, "item_id": it, "value": float(v)}
                          for it, v in zip(items_scored, (3.0, 5.0, 7.0)) for metric in ("self_preservation", "eval_awareness", "scenario_realism")])
    row = {"run_id": "r-sp", "eval_id": "petri_self_preservation", "producer_object_key_legacy": "tag:none", "n_items_generated": 3}
    rows = prun.petri_item_rows(frame, row)
    n_items = len({x["item_id"] for x in rows})
    assert n_items == 3 and len(rows) == 9 and {x["item_id"] for x in rows} == {f"self_preservation/{s}" for s in items_scored}
    analysis = {"by_cell": {"none": {"behaviors": {"self_preservation": {"mean_behavior_score": 5.0, "n_items": 3}},
                                     "audited_footing": {"judge_refused": {"total": 2},
                                                         "behaviors": {"self_preservation": {"n_items": 5, "audited": 3, "auditor_refused": {"total": 0},
                                                                                             "judge_refused": {"total": 2}}}}}}}
    n_refused, notes = prun.petri_judgment_notes(analysis, row)
    assert n_refused == 2 and notes["judge_refused"] == {"total": 2}
    assert n_items + n_refused == 5 == analysis["by_cell"]["none"]["audited_footing"]["behaviors"]["self_preservation"]["n_items"]   # scored + refused
    assert all(x["value"] is not None for x in rows)                                                                                  # judged excludes the refused: they have no scored row
    src = _source("scripts/petri_run.py")
    assert re.search(r'scope\.complete\("complete",\s*n_items_judged=n_items or generated,\s*n_judge_refusals=n_refused', src)
