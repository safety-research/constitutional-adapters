"""The flat per-row export ``paper_rows.json`` + ``paper_rows.csv`` of the two capability
analyses (``scripts/tinycap_gsm8k_paired200.py`` and ``scripts/tinycap_math500_thinking.py``), generated
from an EXISTING report with no estimator change: the pinned field set (exactly, in order); the round-trip
identities ``point_conditional x n_scored == n_correct`` and ``point_score_as_is x n_paired == n_correct``
on every row; a censored cell reads conditional > score-as-is and an uncensored cell reads them equal (same
CIs); ``n_unscored == sum(unscored_reason)``; the bootstrap CI where the report has one and the labelled
exact binomial where it does not (a rung leg); the masked MATH-500 leg read over every item posed; the
rung-state vocabulary from the ladder record; the sensitivity sampling rows listed under ``not_exported``;
the ``--paper-rows-only`` mode on both scripts writing exactly two files beside the report and refusing a
report that is not a production result. Every report here is SYNTHETIC (invented counts, placeholder shas)."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tinycap import paper_rows as PRW

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


G8K = _load("tinycap_gsm8k_paired200", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
M5T = _load("tinycap_math500_thinking", REPO / "scripts" / "tinycap_math500_thinking.py")

PIN, ACS = "1111111", "2222222222222222222222222222222222222222"
BASE, RUNG = 8192, 16384

# --------------------------------------------------------------------------- #
# synthetic GSM8K report: an uncensored cell, a censored + escalated cell, a refused cell
# --------------------------------------------------------------------------- #


def _acc(n_correct, n_scored, *, width=0.03):
    p = n_correct / n_scored
    return {"point": p, "ci_low": max(0.0, p - width), "ci_high": min(1.0, p + width), "acc_minus": p, "acc_plus": p,
            "n_correct": n_correct, "n_scored": n_scored, "n_censored": 0}


def _items(n, *, correct, unscored_at_cap=(), unscored_eos=(), at_cap_scored_wrong=(), source="r8192"):
    """Per-item records the way the GSM8K reading files them (only the flags the export reads)."""
    out = {}
    for i in range(n):
        iid = f"gsm8k_{i:08x}"
        if i in unscored_at_cap:
            out[iid] = {"scored": False, "value": None, "at_cap": True, "think_block_unterminated": True, "source": source}
        elif i in unscored_eos:
            out[iid] = {"scored": False, "value": None, "at_cap": False, "think_block_unterminated": True, "source": source}
        elif i in at_cap_scored_wrong:
            out[iid] = {"scored": True, "value": 0.0, "at_cap": True, "think_block_unterminated": False, "source": source}
        else:
            out[iid] = {"scored": True, "value": 1.0 if i in correct else 0.0, "at_cap": False, "think_block_unterminated": False, "source": source}
    return out


def _reading(items):
    n_scored = sum(1 for it in items.values() if it["scored"])
    return {"items": items, "n": len(items), "n_scored": n_scored, "n_unscored": len(items) - n_scored,
            "n_correct": sum(1 for it in items.values() if it["scored"] and it["value"] == 1.0),
            "acc_stock": None, "n_at_cap_residual": sum(1 for it in items.values() if it["at_cap"]),
            "n_think_unscored": sum(1 for it in items.values() if it["think_block_unterminated"] and not it["scored"]), "n_from_rung": 0}


def _summary(rd, *, cap, n_unterminated_not_at_cap=0):
    n = rd["n"]
    return {"n": n, "n_paired": n, "n_scored": rd["n_scored"], "n_unscored": rd["n_unscored"], "n_at_cap": rd["n_at_cap_residual"],
            "at_cap_rate": rd["n_at_cap_residual"] / n, "n_think_block_unterminated": rd["n_think_unscored"],
            "n_unterminated_not_at_cap": n_unterminated_not_at_cap, "acc": rd["n_correct"] / rd["n_scored"] if rd["n_scored"] else None,
            "max_tokens": cap}


def _ladder(exit_, *, n_at_cap=0, n_eligible=0, rung_store_present=False):
    return {"record_present": True, "record_problem": None, "exit": exit_, "n_at_cap": n_at_cap, "triggered": exit_ == CL.ESCALATE,
            "n_eligible": n_eligible, "rung_store_present": rung_store_present, "escalated": exit_ == CL.ESCALATE}


def _stock(none_reading, cell_reading, n_items):
    """The estimands' `stock` block for one reading: primary over each leg's own scored items, score_as_is over n_items."""
    prim = {"n_items": n_items, "acc_by_leg": {"none": _acc(none_reading["n_correct"], none_reading["n_scored"]),
                                                "cell": _acc(cell_reading["n_correct"], cell_reading["n_scored"])}}
    sai = {"n_items": n_items, "acc_by_leg": {"none": _acc(none_reading["n_correct"], n_items), "cell": _acc(cell_reading["n_correct"], n_items)}}
    return {"primary": prim, "score_as_is": sai}


def gsm8k_report(tmp_path) -> dict:
    """model `qwen35_4b`, view `main`: `none` (uncensored, clean ladder), `constv2-mix-full__lora_r4_ce_minus_ctl`
    (13 at-cap unterminated draws + 1 EOS-in-thoughts, escalated: 4 redrawn at 16,384 — 1 still at cap, 2 correct),
    `constv2-mix-full__psr_ce_all` (a certified refusal: no reading)."""
    n = 200
    none_items = _items(n, correct=set(range(190)))
    none_rd = _reading(none_items)
    cell_id = "constv2-mix-full__lora_r4_ce_minus_ctl"
    cap_ids = set(range(180, 193))            # 13 at-cap, unterminated -> unscored
    eos = {193}                               # 1 unterminated but NOT at cap (EOS inside the thoughts) -> unscored
    cell_items = _items(n, correct=set(range(179)), unscored_at_cap=cap_ids, unscored_eos=eos)
    cell_rd = _reading(cell_items)
    # the rung: 4 eligible re-drawn at 16,384 — 180, 181 correct; 182 wrong; 183 still at cap
    redraws = {}
    merged_items = json.loads(json.dumps(cell_items))
    for i, outcome in ((180, "correct"), (181, "correct"), (182, "wrong"), (183, "at_cap")):
        iid = f"gsm8k_{i:08x}"
        if outcome == "at_cap":
            redraws[iid] = {"new_scored": False, "new_value": None, "still_at_cap": True, "outcome": "still_at_cap"}
            merged_items[iid] = {"scored": False, "value": None, "at_cap": True, "think_block_unterminated": True, "source": "r16384"}
        else:
            v = 1.0 if outcome == "correct" else 0.0
            redraws[iid] = {"new_scored": True, "new_value": v, "still_at_cap": False, "outcome": "extended"}
            merged_items[iid] = {"scored": True, "value": v, "at_cap": False, "think_block_unterminated": False, "source": "r16384"}
    merged_rd = _reading(merged_items)
    rung_summary = {"n": 4, "n_paired": 4, "n_scored": 3, "n_unscored": 1, "n_at_cap": 1, "at_cap_rate": 0.25, "n_think_block_unterminated": 1,
                    "n_unterminated_not_at_cap": 0, "acc": 2 / 3, "max_tokens": RUNG}
    none_key, cell_key, psr_key = "qwen35_4b/main/none", f"qwen35_4b/main/{cell_id}", "qwen35_4b/main/constv2-mix-full__psr_ce_all"
    legs = {
        f"{none_key}@{BASE}": {"leg": f"{none_key}@{BASE}", "cell_key": none_key, "cap": BASE, "rung": False, "missing": None, "refused": None,
                               "summary": _summary(none_rd, cap=BASE), "identity": {}},
        f"{cell_key}@{BASE}": {"leg": f"{cell_key}@{BASE}", "cell_key": cell_key, "cap": BASE, "rung": False, "missing": None, "refused": None,
                               "summary": _summary(cell_rd, cap=BASE, n_unterminated_not_at_cap=1), "identity": {}},
        f"{cell_key}@{RUNG}": {"leg": f"{cell_key}@{RUNG}", "cell_key": cell_key, "cap": RUNG, "rung": True, "missing": None, "refused": None,
                               "summary": rung_summary, "identity": {"cap_rung": {"source_cap": BASE}}},
        f"{psr_key}@{BASE}": {"leg": f"{psr_key}@{BASE}", "cell_key": psr_key, "cap": BASE, "rung": False, "missing": None,
                              "refused": "THINKING_UNSERVABLE: the ON probe completion does not carry the family's think shape",
                              "certified_unservable": True, "identity": {}},
    }
    cells = {
        none_key: {"cell_key": none_key, "model_key": "qwen35_4b", "view": "main", "cell": "none", "fitted": False, "declared": False,
                   "base_leg": f"{none_key}@{BASE}", "rung_leg": None, "ladder": _ladder(CL.EXIT_CLEAN),
                   "readings": {"r8192": none_rd, "merged": {**json.loads(json.dumps(none_rd)), "escalated": False, "same_as_r8192": True}}, "redraws": None},
        cell_key: {"cell_key": cell_key, "model_key": "qwen35_4b", "view": "main", "cell": cell_id, "fitted": True, "declared": False,
                   "base_leg": f"{cell_key}@{BASE}", "rung_leg": f"{cell_key}@{RUNG}",
                   "ladder": _ladder(CL.ESCALATE, n_at_cap=13, n_eligible=4, rung_store_present=True),
                   "readings": {"r8192": cell_rd, "merged": {**merged_rd, "n_from_rung": 3, "escalated": True, "same_as_r8192": False}}, "redraws": redraws},
        psr_key: {"cell_key": psr_key, "model_key": "qwen35_4b", "view": "main", "cell": "constv2-mix-full__psr_ce_all", "fitted": True,
                  "declared": False, "base_leg": f"{psr_key}@{BASE}", "rung_leg": None, "ladder": None, "readings": None, "redraws": None},
    }
    name = f"qwen35_4b/main: {cell_id} - none"
    contrasts = [{"name": name, "model_key": "qwen35_4b", "view": "main", "cell": cell_id, "cell_key": cell_key, "none_key": none_key, "computable": True},
                 {"name": "qwen35_4b/main: constv2-mix-full__psr_ce_all - none", "model_key": "qwen35_4b", "view": "main",
                  "cell": "constv2-mix-full__psr_ce_all", "cell_key": psr_key, "none_key": none_key, "computable": False, "certified_refusal": True}]
    estimands = {name: {"r8192": {"stock": _stock(none_rd, cell_rd, n)}, "merged": {"stock": _stock(none_rd, merged_rd, n)}}}
    return {"experiment": "gsm8k-paired200-synthetic", "task": "inspect/gsm8k_paired200", "sha": PIN, "analysis_code_sha": ACS,
            "gates": {"failed": [], "pending": []}, "unusable_legs": {}, "legs": legs, "cells": cells, "contrasts": contrasts,
            "estimands": estimands, "verdict": {"word": "filed", "verdict": "synthetic"}}


# --------------------------------------------------------------------------- #
# synthetic MATH-500 report: bare uncensored, object censored, a masked lora_ctl2 leg, a sensitivity mode
# --------------------------------------------------------------------------- #


def _m5t_leg(model, cell, mode, kind, *, n=500, n_scored=500, n_at_cap=0, n_unt_not_at_cap=0, fitted=False, primary=True, view=None):
    unscored = n - n_scored
    lk = M5T.leg_key(model, cell, mode)
    leg = {"leg": lk, "model_key": model, "cell": cell, "kind": kind, "mode": mode, "fitted": fitted, "primary": primary, "missing": None,
           "refused": None, "identity": {"cap_rung": None, "max_tokens_by_task": {M5T.TASK: RUNG}},
           "summary": {"n": n, "n_scored": n_scored, "n_unscored": unscored, "n_at_cap": n_at_cap, "at_cap_rate": n_at_cap / n,
                       "n_think_block_unterminated": unscored, "n_unterminated_not_at_cap": n_unt_not_at_cap, "max_tokens": RUNG},
           "unscored": {"n_unscored": unscored, "n_unterminated_at_cap": unscored - n_unt_not_at_cap, "n_unterminated_not_at_cap": n_unt_not_at_cap,
                        "n_unscored_other": 0}}
    if view is not None:
        leg["view"] = view
    return lk, leg


def _m5t_estimands(legs_by_arm, n_items, *, drop=()):
    """`primary` over each leg's own scored items and `score_as_is` over `n_items`; `drop` names the arm whose
    population the mask shrinks by one UNSCORED item (its score_as_is then reads over n_items - 1)."""
    prim, sai = {"n_items": n_items, "acc_by_leg": {}}, {"n_items": n_items, "acc_by_leg": {}}
    for arm, (n_correct, n_scored) in legs_by_arm.items():
        masked = arm in drop
        prim["acc_by_leg"][arm] = {**_acc(n_correct, n_scored), "n_censored": (n_items - 1 if masked else n_items) - n_scored}
        sai["acc_by_leg"][arm] = _acc(n_correct, n_items - 1 if masked else n_items)
    return {"primary": prim, "score_as_is": sai}


def math500_report(tmp_path) -> dict:
    mode = "thinking_card"
    legs = {}
    for cell, kind, kw in (("none", "bare", {}),
                           ("prompted__safety_refusal", "prompted", {}),
                           ("constv2-mix-full__lora_r4_ce_minus_ctl", "object", {"n_scored": 318, "n_at_cap": 186, "fitted": True}),
                           ("constv2ctl2-full__lora_r4_ce", "lora_ctl2", {"n_scored": 489, "n_at_cap": 11, "n_unt_not_at_cap": 1, "fitted": True})):
        lk, leg = _m5t_leg("nemotron3nano4b", cell, mode, kind, **kw)
        legs[lk] = leg
    lk_gen, leg_gen = _m5t_leg("nemotron3nano4b", "none", "battery", "bare", primary=False, n_scored=480, n_at_cap=20)
    legs[lk_gen] = leg_gen
    counts = {"bare": (441, 500), "prompted": (431, 500), "object": (282, 318), "lora_ctl2": (428, 489)}
    comp = {"name": f"nemotron3nano4b/{mode}", "model_key": "nemotron3nano4b", "mode": mode, "primary": True, "usable": True,
            "leg_keys": {arm: M5T.leg_key("nemotron3nano4b", cell, mode) for arm, cell in
                         (("bare", "none"), ("prompted", "prompted__safety_refusal"), ("object", "constv2-mix-full__lora_r4_ce_minus_ctl"),
                          ("lora_ctl2", "constv2ctl2-full__lora_r4_ce"))},
            "masked_items": {"lora_ctl2": ["math_00000001"]},
            "estimands": _m5t_estimands(counts, 500, drop=("lora_ctl2",)),
            "estimands_with_anomalies": _m5t_estimands(counts, 500)}
    sens = {"name": "nemotron3nano4b/battery", "model_key": "nemotron3nano4b", "mode": "battery", "primary": False, "usable": True,
            "leg_keys": {"bare": lk_gen}, "estimands": _m5t_estimands({"bare": (430, 480)}, 500)}
    return {"experiment": "math500-synthetic", "task": M5T.TASK, "sha": PIN, "analysis_code_sha": ACS, "gates": {"failed": []}, "unusable_legs": {},
            "legs": legs, "comparisons": [comp, sens], "estimands": {"nemotron3nano4b": {}}, "mode_delta_records": [],
            "verdict": {"word": "filed", "verdict": "synthetic"}}


# --------------------------------------------------------------------------- #
# the schema and the shared module                                             #
# --------------------------------------------------------------------------- #


def test_serving_commit_is_derived_from_the_standup_record_and_refuses_when_the_two_trees_disagree():
    """`serving_commit` is READ from its authoritative source — `standup_jobs.jsonl[0].sha`, the tree the STANDUP served
    from, carried onto the leg as `standup_record.standup_sha` — and never from `producer.tree_sha`, which is the DRIVER's.
    For these launchers the two are one eval-repo checkout, so PRESENT-AND-EQUAL yields the commit and a DISAGREEMENT
    refuses by name: two stacks under one row is the thing the field exists to catch. Absent, it stays null and the
    export refuses a row of an arm that requires it (pinned by the provenance tests below)."""
    A = G8K
    sha, tree = "a" * 40, "b" * 40
    cell = {"cell_key": "qwen35_4b/it/arditi_allpos", "cell": "arditi_allpos"}

    # (1) PRESENT and EQUAL -> the row carries the standup's tree, and the note names where it came from
    leg = {"standup_record": {"standup_sha": sha, "producer_tree_sha": sha}}
    got, note = A._pr_serving_commit(cell, leg)
    assert got == sha and "standup_jobs.jsonl[0].sha" in note and "producer.tree_sha" in note

    # (2) DISAGREEMENT -> refused BY NAME, naming both trees; never silently preferring either
    with pytest.raises(A.AnalysisError, match="two stacks under one row"):
        A._pr_serving_commit(cell, {"standup_record": {"standup_sha": sha, "producer_tree_sha": tree}})

    # (3) ABSENT (a lens leg has no native standup record) -> null, for the export's own refusal to handle
    assert A._pr_serving_commit(cell, {}) == (None, None)
    assert A._pr_serving_commit(cell, {"standup_record": {"standup_sha": None, "producer_tree_sha": None}}) == (None, None)

    # (4) an EXPLICIT producer stamp still wins over the derivation, and is validated like any other
    got, note = A._pr_serving_commit(cell, {"serving_commit": sha, "standup_record": {"standup_sha": tree, "producer_tree_sha": tree}})
    assert got == sha and "as the store record stamps it" in note


def test_the_field_set_is_pinned_exactly_and_in_order():
    # re-pinned for schema /4 (E3 tier-2 item 8): ``n_refusal_shaped`` and its denominator ``n_refusal_shaped_of`` appended LAST, in
    # that order, after the /3 provenance fields, so every earlier column keeps its position; /2 and /3 documents still read (the absent
    # fields filled null, TOGETHER)
    assert PRW.SCHEMA == "paper_rows/4" and PRW.READABLE_SCHEMAS == ("paper_rows/2", "paper_rows/3", "paper_rows/4")
    assert PRW.FIELDS == (
        "instrument", "model_key", "view", "cell", "arm", "paper_arm", "cap", "leg",
        "n_paired", "n_scored", "n_unscored", "n_at_cap", "at_cap_rate", "n_unterminated_not_at_cap", "n_correct",
        "point_conditional", "ci_conditional_low", "ci_conditional_high",
        "point_score_as_is", "ci_score_as_is_low", "ci_score_as_is_high", "ci_method",
        "rung_state", "object_generation", "selection_outcome", "unscored_reason", "censoring_note", "source",
        "selection", "serving_commit", "fit_provenance",
        "n_refusal_shaped", "n_refusal_shaped_of")
    assert PRW.PROVENANCE_FIELDS == PRW.FIELDS[-5:-2] == ("selection", "serving_commit", "fit_provenance")
    assert PRW.REFUSAL_SHAPED_FIELDS == PRW.FIELDS[-2:] == ("n_refusal_shaped", "n_refusal_shaped_of")
    assert (PRW.REFUSAL_SHAPED_FIELD, PRW.REFUSAL_SHAPED_OF_FIELD) == PRW.REFUSAL_SHAPED_FIELDS
    assert all(f in PRW.COUNT_FIELDS for f in PRW.REFUSAL_SHAPED_FIELDS)
    assert PRW.CENSORING_NOTE == ("at-cap draws that end inside an unterminated think block are UNSCORED and excluded from point_conditional; "
                                  "point_score_as_is counts every such draw as wrong")
    assert PRW.RUNG_STATES == ("complete", "owed", "not_applicable", "refused", "never_run")
    assert PRW.OBJECT_GENERATIONS == ("batch8", "batch64", "none", "n/a")
    # re-pinned 2026-09-15: the oracle spells gemma's SECOND ruling of 2026-09-14 `ruled_2026-09-14_second`; the launcher's
    # vocabulary gained it when the admission record was re-stamped from the candidate label, and the export mirrors it word for word
    assert PRW.SELECTION_OUTCOMES == ("chosen", "a4_fallback", "ruled_2026-09-14", "chosen_confirmed_2026-09-14", "ruled_2026-09-14_second",
                                      "candidate_2026-09-14", "user_override_over_budget_2026-09-16", None)
    # the dated words are the LAUNCHER's vocabulary, mirrored word for word (the export module imports no launcher): the AlphaSteer
    # ladder's words (`SELECTION_OUTCOMES`) plus the all-positions Arditi arm's ONE override word, which the launcher keeps in its own
    # tuple (`ARDITI_ALLPOS_SELECTION_OUTCOMES`) because no other arm may carry it
    G8KC = _load("g8k_cells_for_paper_rows_test", REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py")
    assert tuple(x for x in PRW.SELECTION_OUTCOMES if x is not None) == tuple(G8KC.SELECTION_OUTCOMES) + tuple(G8KC.ARDITI_ALLPOS_SELECTION_OUTCOMES)
    assert PRW.SEL_USER_OVERRIDE_OVER_BUDGET == G8KC.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE == "user_override_over_budget_2026-09-16"
    assert PRW.ARDITI_ALLPOS_SELECTION_OUTCOMES == tuple(G8KC.ARDITI_ALLPOS_SELECTION_OUTCOMES) == (PRW.SEL_USER_OVERRIDE_OVER_BUDGET,)
    assert PRW.DOSE_UNIT_RAW_COEFFICIENT == G8KC.DOSE_UNIT_RAW_COEFFICIENT and PRW.DOSE_UNIT_ALPHA == G8KC.DOSE_UNIT_RESIDUAL_NORM_FRACTION
    assert PRW.UNSCORED_REASONS == ("unterminated_think_block", "hit_cap_mid_solution", "refusal", "other")
    with pytest.raises(PRW.PaperRowsError, match="unknown"):
        PRW.make_row(**{f: None for f in PRW.FIELDS}, extra=1)
    with pytest.raises(PRW.PaperRowsError, match="missing"):
        PRW.make_row(instrument="math500")


def test_paper_arm_is_the_declared_producer_key_never_inferred_from_the_slug(tmp_path):
    """The paper's canonical arm key rides every row from the DECLARED table: the bare ``alphasteer`` slug is the REBAKE
    arm (never ``alphasteer_as_fit``), ``none`` maps to ``none``, and a slug the paper does not draw is ``null`` — a
    statement of absence, not a guess. The verifier holds a row's ``paper_arm`` to the table."""
    assert PRW.PAPER_ARM_KEY_BY_CELL == {
        "none": "none", "alphasteer": "alphasteer_rebake", "arditi": "arditi", "prompted__si_constitution": "si_constitution",
        "constv2-mix-full__lora_r4_ce": "const_lora", "constv2ctl2-full__lora_r4_ce": "ctl_lora",
        "constv2-mix-full__lora_r4_ce_minus_ctl": "diff_lora", "arditi_allpos": "arditi_allpos"}
    # the two Arditi arms are two interventions with two keys: the entry order keeps `arditi` where it was, and the new cell never
    # resolves to the prefill-only arm's key (nor the other way round)
    assert list(PRW.PAPER_ARM_KEY_BY_CELL)[2] == "arditi" and list(PRW.PAPER_ARM_KEY_BY_CELL)[-1] == "arditi_allpos"
    assert PRW.paper_arm_of("arditi_allpos") == "arditi_allpos" != PRW.paper_arm_of("arditi") == "arditi"
    assert PRW.paper_arm_of("alphasteer") == "alphasteer_rebake" and "alphasteer_as_fit" not in PRW.PAPER_ARM_KEYS
    assert PRW.paper_arm_of("none") == "none"
    for slug in ("prompted__safety_refusal", "constv2-mix-full__psr_ce_all", "safety_refusal__lora_r4",
                 "constv2-mix-full__psr_ce_all_minus_ctl_distilled", "alphasteer_as_fit", "made_up_cell", None):
        assert PRW.paper_arm_of(slug) is None, slug
    # the readers never pass paper_arm: make_row fills it from the table, and the verifier holds it there
    rows, _ = G8K.paper_rows_of_report(gsm8k_report(tmp_path), report_path=tmp_path / "report.json")
    by_cell = {r["cell"]: r["paper_arm"] for r in rows}
    assert by_cell == {"none": "none", "constv2-mix-full__lora_r4_ce_minus_ctl": "diff_lora", "constv2-mix-full__psr_ce_all": None}
    assert all(list(r.keys()).index("paper_arm") == list(r.keys()).index("arm") + 1 for r in rows)
    good = next(r for r in rows if r["cell"] == "none" and r["leg"] == "r8192")
    assert PRW.row_problems(good) == []
    assert any("paper_arm" in p for p in PRW.row_problems({**good, "paper_arm": "alphasteer_rebake"}))
    assert any("paper_arm" in p for p in PRW.row_problems({**good, "paper_arm": "alphasteer_as_fit"}))
    assert any("paper_arm" in p for p in PRW.row_problems({**good, "paper_arm": None}))
    explicit = PRW.make_row(**{**{k: good[k] for k in PRW.FIELDS if k not in ("paper_arm", "censoring_note")}, "paper_arm": "none"})
    assert explicit == good


def test_exact_binomial_ci_is_clopper_pearson_and_degenerates_sanely():
    low, high = PRW.exact_binomial_ci(2, 3)
    assert math.isclose(low, 0.0942, abs_tol=1e-3) and math.isclose(high, 0.9916, abs_tol=1e-3)
    low, high = PRW.exact_binomial_ci(170, 200)
    assert low < 170 / 200 < high and math.isclose(low, 0.7928, abs_tol=1e-3) and math.isclose(high, 0.8965, abs_tol=1e-3)
    assert PRW.exact_binomial_ci(0, 5)[0] == 0.0 and PRW.exact_binomial_ci(5, 5)[1] == 1.0
    assert PRW.exact_binomial_ci(0, 0) == (None, None)
    with pytest.raises(PRW.PaperRowsError):
        PRW.exact_binomial_ci(4, 3)


def test_unscored_reason_sums_to_n_unscored_and_names_a_remainder():
    ur, note = PRW.unscored_reason(13, unterminated_think_block=13)
    assert ur == {"unterminated_think_block": 13, "hit_cap_mid_solution": 0, "refusal": 0, "other": 0} and note is None
    ur, note = PRW.unscored_reason(15, unterminated_think_block=13)
    assert ur["other"] == 2 and "2 unscored draw(s)" in note
    with pytest.raises(PRW.PaperRowsError, match="exceed"):
        PRW.unscored_reason(2, unterminated_think_block=3)


def test_the_verifier_catches_a_broken_round_trip_a_bad_vocabulary_and_a_duplicate_key(tmp_path):
    rows, _ = G8K.paper_rows_of_report(gsm8k_report(tmp_path), report_path=tmp_path / "report.json")
    good = next(r for r in rows if r["n_unscored"])
    assert PRW.row_problems(good) == []
    bad = {**good, "point_score_as_is": good["point_score_as_is"] + 0.01}
    assert any("ROUND TRIP point_score_as_is" in p for p in PRW.row_problems(bad))
    bad = {**good, "n_correct": good["n_correct"] + 1}
    assert any("ROUND TRIP" in p for p in PRW.row_problems(bad))
    bad = {**good, "rung_state": "pending"}
    assert any("rung_state" in p for p in PRW.row_problems(bad))
    bad = {**good, "unscored_reason": {**good["unscored_reason"], "other": 5}}
    assert any("sum(unscored_reason)" in p for p in PRW.row_problems(bad))
    bad = {**good, "censoring_note": "truncated solutions are graded wrong"}
    assert any("literal sentence" in p for p in PRW.row_problems(bad))
    assert PRW.row_problems({k: good[k] for k in reversed(PRW.FIELDS)})[0].startswith("fields are")
    with pytest.raises(PRW.PaperRowsError, match="share the key"):
        PRW.verify_rows([good, dict(good)])


# --------------------------------------------------------------------------- #
# GSM8K                                                                        #
# --------------------------------------------------------------------------- #


def test_gsm8k_rows_round_trip_to_their_counts_on_every_row(tmp_path):
    report = gsm8k_report(tmp_path)
    rows, skipped = G8K.paper_rows_of_report(report, report_path=tmp_path / "report.json")
    assert skipped == []
    PRW.verify_rows(rows)
    keys = [(r["cell"], r["leg"]) for r in rows]
    assert keys == [("none", "r8192"), ("none", "merged"),
                    ("constv2-mix-full__lora_r4_ce_minus_ctl", "r8192"), ("constv2-mix-full__lora_r4_ce_minus_ctl", "merged"),
                    ("constv2-mix-full__lora_r4_ce_minus_ctl", "r16384"), ("constv2-mix-full__psr_ce_all", "r8192")]
    for r in rows:
        assert tuple(r.keys()) == PRW.FIELDS and r["instrument"] == "gsm8k_paired200" and r["censoring_note"] == PRW.CENSORING_NOTE
        if r["n_paired"] is None:
            continue
        assert round(r["point_score_as_is"] * r["n_paired"]) == r["n_correct"]
        assert math.isclose(r["point_score_as_is"] * r["n_paired"], r["n_correct"], abs_tol=1e-6)
        if r["n_scored"]:
            assert math.isclose(r["point_conditional"] * r["n_scored"], r["n_correct"], abs_tol=1e-6)
        assert r["n_paired"] == r["n_scored"] + r["n_unscored"] == sum(r["unscored_reason"].values()) + r["n_scored"]
        assert r["source"].startswith(f"report={tmp_path / 'report.json'}; store_pin={PIN}; analysis_code_sha={ACS}; read: ")


def test_gsm8k_a_censored_cell_reads_conditional_above_score_as_is_and_an_uncensored_cell_reads_them_equal(tmp_path):
    rows, _ = G8K.paper_rows_of_report(gsm8k_report(tmp_path), report_path=tmp_path / "report.json")
    by = {(r["cell"], r["leg"]): r for r in rows}
    none = by[("none", "r8192")]
    assert none["n_unscored"] == 0 and none["point_conditional"] == none["point_score_as_is"] == 0.95
    assert (none["ci_conditional_low"], none["ci_conditional_high"]) == (none["ci_score_as_is_low"], none["ci_score_as_is_high"])
    assert none["arm"] == "bare" and none["object_generation"] == "none" and none["rung_state"] == "not_applicable" and none["cap"] == BASE
    cell = by[("constv2-mix-full__lora_r4_ce_minus_ctl", "r8192")]
    # 13 at-cap unterminated + 1 EOS-in-thoughts -> 14 unscored; 179 correct of 186 scored; every unscored draw wrong under score-as-is
    assert (cell["n_paired"], cell["n_scored"], cell["n_unscored"], cell["n_at_cap"], cell["n_correct"]) == (200, 186, 14, 13, 179)
    assert cell["n_unterminated_not_at_cap"] == 1 and cell["at_cap_rate"] == 13 / 200
    assert math.isclose(cell["point_conditional"], 179 / 186) and math.isclose(cell["point_score_as_is"], 179 / 200)
    assert cell["point_conditional"] > cell["point_score_as_is"]
    assert cell["unscored_reason"] == {"unterminated_think_block": 14, "hit_cap_mid_solution": 0, "refusal": 0, "other": 0}
    assert cell["ci_method"] == PRW.CI_BOOTSTRAP and "estimands[" in cell["source"] and "acc_by_leg.cell" in cell["source"]
    assert cell["arm"] == "object" and cell["object_generation"] == "n/a" and "no generation stamp" in cell["source"]
    assert cell["selection_outcome"] is None and cell["rung_state"] == "complete"
    merged = by[("constv2-mix-full__lora_r4_ce_minus_ctl", "merged")]
    # the rung re-drew 4: 2 correct, 1 wrong, 1 still at cap -> 189 scored, 181 correct, 11 unscored (10 at cap + the EOS draw)
    assert (merged["n_paired"], merged["n_scored"], merged["n_unscored"], merged["n_at_cap"], merged["n_correct"]) == (200, 189, 11, 10, 181)
    assert merged["cap"] == RUNG and merged["n_unterminated_not_at_cap"] == 1 and merged["at_cap_rate"] == 10 / 200
    assert math.isclose(merged["point_conditional"], 181 / 189) and math.isclose(merged["point_score_as_is"], 181 / 200)
    assert "the merged reading files no rate" in merged["source"]


def test_gsm8k_the_rung_leg_is_read_over_its_redrawn_subset_with_the_exact_binomial_ci(tmp_path):
    rows, _ = G8K.paper_rows_of_report(gsm8k_report(tmp_path), report_path=tmp_path / "report.json")
    rung = next(r for r in rows if r["leg"] == "r16384")
    assert (rung["cap"], rung["n_paired"], rung["n_scored"], rung["n_unscored"], rung["n_at_cap"], rung["n_correct"]) == (RUNG, 4, 3, 1, 1, 2)
    assert math.isclose(rung["point_conditional"], 2 / 3) and rung["point_score_as_is"] == 0.5 and rung["at_cap_rate"] == 0.25
    assert rung["ci_method"] == PRW.CI_EXACT
    assert (rung["ci_conditional_low"], rung["ci_conditional_high"]) == PRW.exact_binomial_ci(2, 3)
    assert (rung["ci_score_as_is_low"], rung["ci_score_as_is_high"]) == PRW.exact_binomial_ci(2, 4)
    assert rung["unscored_reason"]["unterminated_think_block"] == 1 and rung["rung_state"] == "complete"
    assert "redraws" in rung["source"] and "Clopper-Pearson" in rung["source"] and "not the 200 posed" in rung["source"]


def test_gsm8k_a_refused_cell_is_one_null_count_row_with_rung_state_refused(tmp_path):
    rows, _ = G8K.paper_rows_of_report(gsm8k_report(tmp_path), report_path=tmp_path / "report.json")
    ref = next(r for r in rows if r["cell"] == "constv2-mix-full__psr_ce_all")
    assert all(ref[f] is None for f in PRW.COUNT_FIELDS)
    assert ref["rung_state"] == "refused" and ref["arm"] == "psr_mix" and ref["leg"] == "r8192" and ref["object_generation"] == "n/a"
    assert "readings = null" in ref["source"] and "THINKING_UNSERVABLE" in ref["source"]
    assert PRW.row_problems(ref) == []


def test_gsm8k_rung_states_follow_the_ladder_record(tmp_path):
    report = gsm8k_report(tmp_path)
    cell = report["cells"]["qwen35_4b/main/constv2-mix-full__lora_r4_ce_minus_ctl"]
    legs = report["legs"]
    assert G8K._pr_rung_state(cell, legs)[0] == "complete"
    owed = {**cell, "ladder": {**cell["ladder"], "rung_store_present": False}, "rung_leg": None}
    assert G8K._pr_rung_state(owed, legs)[0] == "owed"
    for exit_, want in ((CL.EXIT_CLEAN, "not_applicable"), (CL.EXIT_BELOW_THRESHOLD, "not_applicable"), (IR.EXIT_TRIGGERED_NO_ELIGIBLE, "not_applicable"),
                        (CL.EXIT_CEILING, "complete"), (CL.EXIT_TRANSPORT_REFUSED, "refused"), (CL.EXIT_NO_RUNG, "never_run")):
        assert G8K._pr_rung_state({**cell, "ladder": _ladder(exit_)}, legs)[0] == want, exit_
    assert G8K._pr_rung_state({**cell, "ladder": {"record_present": False, "record_problem": "absent"}}, legs)[0] == "never_run"
    with pytest.raises(G8K.AnalysisError, match="names no rung state"):
        G8K._pr_rung_state({**cell, "ladder": _ladder("sideways")}, legs)


def test_gsm8k_the_reader_refuses_a_reading_whose_headline_counts_are_not_its_items(tmp_path):
    report = gsm8k_report(tmp_path)
    report["cells"]["qwen35_4b/main/none"]["readings"]["r8192"]["n_correct"] += 1
    with pytest.raises(G8K.AnalysisError, match="items sum to"):
        G8K.paper_rows_of_report(report)
    report = gsm8k_report(tmp_path)
    est = next(iter(report["estimands"].values()))["r8192"]["stock"]["primary"]["acc_by_leg"]["cell"]
    est["n_correct"] -= 1
    with pytest.raises(G8K.AnalysisError, match="counts"):
        G8K.paper_rows_of_report(report)
    report = gsm8k_report(tmp_path)
    report["legs"]["qwen35_4b/main/constv2-mix-full__lora_r4_ce_minus_ctl@8192"]["identity"]["object_generation"] = "batch-64"
    with pytest.raises(G8K.AnalysisError, match="object_generation stamp"):
        G8K.paper_rows_of_report(report)
    report["legs"]["qwen35_4b/main/constv2-mix-full__lora_r4_ce_minus_ctl@8192"]["identity"]["object_generation"] = "batch64"
    report["legs"]["qwen35_4b/main/constv2-mix-full__lora_r4_ce_minus_ctl@8192"]["identity"]["selection_outcome"] = "a4_fallback"
    rows, _ = G8K.paper_rows_of_report(report)
    cell = next(r for r in rows if r["cell"] == "constv2-mix-full__lora_r4_ce_minus_ctl" and r["leg"] == "r8192")
    assert cell["object_generation"] == "batch64" and cell["selection_outcome"] == "a4_fallback" and "no generation stamp" not in cell["source"]


def test_gsm8k_paper_rows_only_writes_exactly_two_files_beside_the_report_and_refuses_a_non_production_report(tmp_path, capsys):
    d = tmp_path / "prod"
    d.mkdir()
    (d / "report.json").write_text(json.dumps(gsm8k_report(tmp_path)), encoding="utf-8")
    (d / "figs").mkdir()
    before = {p.name: p.read_bytes() for p in d.rglob("*") if p.is_file()}
    assert G8K.main(["--paper-rows-only", str(d / "report.json")]) == 0
    after = {p.name for p in d.iterdir()}
    assert after == {"report.json", "figs", PRW.JSON_NAME, PRW.CSV_NAME}
    assert all((d / n).read_bytes() == b for n, b in before.items())          # nothing else touched, the report byte-identical
    doc = json.loads((d / PRW.JSON_NAME).read_text(encoding="utf-8"))
    assert doc["schema"] == PRW.SCHEMA and doc["fields"] == list(PRW.FIELDS) and doc["n_rows"] == 6 == len(doc["rows"])
    assert doc["instrument"] == "gsm8k_paired200" and doc["report"] == str(d / "report.json") and doc["censoring_note"] == PRW.CENSORING_NOTE
    rows = PRW.read_paper_rows(d / PRW.JSON_NAME)
    with (d / PRW.CSV_NAME).open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert [tuple(r.keys()) for r in csv_rows] == [PRW.FIELDS] * 6
    for jr, cr in zip(rows, csv_rows):
        assert cr["cell"] == jr["cell"] and cr["leg"] == jr["leg"]
        assert (json.loads(cr["unscored_reason"]) if cr["unscored_reason"] else None) == jr["unscored_reason"]
        assert (cr["n_correct"] == "" and jr["n_correct"] is None) or int(cr["n_correct"]) == jr["n_correct"]
        assert (cr["point_score_as_is"] == "" and jr["point_score_as_is"] is None) or math.isclose(float(cr["point_score_as_is"]), jr["point_score_as_is"])
    # a report that is not a production result gets no paper rows
    bad = gsm8k_report(tmp_path)
    bad["verdict"] = {"word": None, "verdict": None, "reason": "withheld"}
    e = tmp_path / "notprod"
    e.mkdir()
    (e / "report.json").write_text(json.dumps(bad), encoding="utf-8")
    assert G8K.main(["--paper-rows-only", str(e / "report.json")]) == G8K.EXIT_NOT_PRODUCTION == 3
    assert {p.name for p in e.iterdir()} == {"report.json"}
    assert "NOT a production result" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# MATH-500                                                                     #
# --------------------------------------------------------------------------- #


def test_math500_rows_round_trip_and_the_sensitivity_mode_is_listed_not_exported(tmp_path):
    report = math500_report(tmp_path)
    rows, skipped = M5T.paper_rows_of_report(report, report_path=tmp_path / "report.json")
    PRW.verify_rows(rows)
    assert [r["arm"] for r in rows] == ["bare", "prompted", "object", "lora_ctl2"]
    assert [s["leg"] for s in skipped] == ["nemotron3nano4b/none/battery"] and "sensitivity row" in skipped[0]["reason"]
    for r in rows:
        assert r["instrument"] == "math500" and r["leg"] == "r16384" and r["cap"] == RUNG and r["view"] == "main"
        assert r["rung_state"] == "not_applicable" and "one cap" in r["source"] and "view: not a field" in r["source"]
        assert math.isclose(r["point_score_as_is"] * r["n_paired"], r["n_correct"], abs_tol=1e-6)
        assert math.isclose(r["point_conditional"] * r["n_scored"], r["n_correct"], abs_tol=1e-6)
        assert r["n_paired"] == 500 == r["n_scored"] + r["n_unscored"] and sum(r["unscored_reason"].values()) == r["n_unscored"]
        assert r["ci_method"] == PRW.CI_BOOTSTRAP
    by = {r["arm"]: r for r in rows}
    bare, obj = by["bare"], by["object"]
    assert bare["n_unscored"] == 0 and bare["point_conditional"] == bare["point_score_as_is"] == 441 / 500 and bare["object_generation"] == "none"
    assert (obj["n_scored"], obj["n_unscored"], obj["n_at_cap"], obj["n_correct"]) == (318, 182, 186, 282)
    assert math.isclose(obj["point_conditional"], 282 / 318) and obj["point_score_as_is"] == 282 / 500 and obj["point_conditional"] > obj["point_score_as_is"]
    assert obj["unscored_reason"] == {"unterminated_think_block": 182, "hit_cap_mid_solution": 0, "refusal": 0, "other": 0}
    assert obj["object_generation"] == "n/a" and obj["selection_outcome"] is None and obj["at_cap_rate"] == 186 / 500


def test_math500_a_d48_masked_leg_is_read_over_every_item_posed_from_the_with_anomalies_branch(tmp_path):
    report = math500_report(tmp_path)
    rows, _ = M5T.paper_rows_of_report(report, report_path=tmp_path / "report.json")
    ctl = next(r for r in rows if r["arm"] == "lora_ctl2")
    assert (ctl["n_paired"], ctl["n_scored"], ctl["n_unscored"], ctl["n_correct"], ctl["n_unterminated_not_at_cap"]) == (500, 489, 11, 428, 1)
    assert ctl["point_score_as_is"] == 428 / 500            # not the masked 428 / 499
    assert math.isclose(ctl["point_conditional"], 428 / 489)
    assert "estimands_with_anomalies" in ctl["source"] and "D48" in ctl["source"] and "math_00000001" in ctl["source"] and "499" in ctl["source"]
    # the leg whose view the record carries reads it (no fallback note)
    report["legs"]["nemotron3nano4b/constv2ctl2-full__lora_r4_ce/thinking_card"]["view"] = "it"
    rows, _ = M5T.paper_rows_of_report(report)
    ctl = next(r for r in rows if r["arm"] == "lora_ctl2")
    assert ctl["view"] == "it" and "view: not a field" not in ctl["source"]
    # a leg the report's masked primary shrinks but that files no with-reading refuses
    del report["comparisons"][0]["estimands_with_anomalies"]
    with pytest.raises(M5T.AnalysisError, match="estimands_with_anomalies"):
        M5T.paper_rows_of_report(report)


def test_math500_the_reader_refuses_a_summary_that_disagrees_with_the_estimand(tmp_path):
    report = math500_report(tmp_path)
    report["legs"]["nemotron3nano4b/none/thinking_card"]["summary"]["n_scored"] = 499
    with pytest.raises(M5T.AnalysisError, match="!= the estimand"):
        M5T.paper_rows_of_report(report)
    report = math500_report(tmp_path)
    report["legs"]["nemotron3nano4b/none/thinking_card"]["kind"] = "object"
    with pytest.raises(M5T.AnalysisError, match="kind"):
        M5T.paper_rows_of_report(report)
    report = math500_report(tmp_path)
    report["legs"]["nemotron3nano4b/none/thinking_card"]["unscored"]["n_unterminated_at_cap"] = 3     # 3 named on a leg with 0 unscored
    with pytest.raises(PRW.PaperRowsError, match="exceed"):
        M5T.paper_rows_of_report(report)


def test_math500_paper_rows_only_writes_exactly_two_files_beside_the_report_and_refuses_a_non_production_report(tmp_path, capsys):
    d = tmp_path / "prod"
    d.mkdir()
    report = math500_report(tmp_path)
    (d / "report.json").write_text(json.dumps(report), encoding="utf-8")
    before = (d / "report.json").read_bytes()
    assert M5T.main(["--paper-rows-only", str(d / "report.json")]) == 0
    assert {p.name for p in d.iterdir()} == {"report.json", PRW.JSON_NAME, PRW.CSV_NAME} and (d / "report.json").read_bytes() == before
    doc = json.loads((d / PRW.JSON_NAME).read_text(encoding="utf-8"))
    assert doc["schema"] == PRW.SCHEMA and doc["instrument"] == "math500" and doc["n_rows"] == 4 and len(doc["not_exported"]) == 1
    assert doc["not_exported"][0]["leg"] == "nemotron3nano4b/none/battery"
    with (d / PRW.CSV_NAME).open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 4 and [tuple(r.keys()) for r in csv_rows] == [PRW.FIELDS] * 4
    out = capsys.readouterr().out
    assert "4 rows" in out and "1 sensitivity leg(s)" in out
    bad = math500_report(tmp_path)
    bad["comparisons"][0]["estimands"] = None
    e = tmp_path / "notprod"
    e.mkdir()
    (e / "report.json").write_text(json.dumps(bad), encoding="utf-8")
    assert M5T.main(["--paper-rows-only", str(e / "report.json")]) == M5T.EXIT_NOT_PRODUCTION == 3
    assert {p.name for p in e.iterdir()} == {"report.json"} and "NOT a production result" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# schema /3: the all-positions Arditi arm's three provenance fields             #
# --------------------------------------------------------------------------- #

ALLPOS = "arditi_allpos"
OVERRIDE = "user_override_over_budget_2026-09-16"
HEX64_A, HEX64_B, HEX64_C = "a" * 64, "b" * 64, "c" * 64


def selection_block(**over):
    """A SYNTHETIC selection block in the admission record's shape (invented readings; the real record is never copied here)."""
    block = {"outcome": OVERRIDE, "rule": "highest dose within the over-refusal budget, read at the ladder cap named here",
             "rule_returns": "ar_c0p2", "rule_reading_pp": 5.04, "override_reading_pp": 10.26, "budget_pp": 10, "ladder_cap": 2048,
             "single_trial": True, "dose": {"value": 0.3, "unit": "raw_coefficient"},
             "alpha_equiv": {"value": 0.105, "note": "arithmetic convenience on this model only, never the dose"},
             "replicate_note": "a fresh-seed replicate moves the delta by more than the margin over budget"}
    block.update(over)
    return block


def fit_provenance_block(**over):
    block = {"fitted_on_model": "org/model-4B", "fitted_on_revision": "1" * 40, "rendering": "chat rendering, activations at post-instruction positions",
             "capture_positions": "all", "item_source": "data/fit_prompts.json", "item_source_sha": "2" * 40,
             "realised_counts": {"harmful_train": 123, "harmful_val": 31, "harmless_train": 125, "harmless_val": 31},
             "direction_sha256": HEX64_A, "direction_pt_sha256": HEX64_B, "fit_layer": 13, "fit_position": -5, "direction_norm": 3.29,
             "served_layer": 12, "fit_quality_gate": {"harmful_mean": 5.8, "harmless_mean": -10.9, "harmful_frac_positive": 0.96,
                                                     "harmless_frac_negative": 0.98},
             "fit_record_sha256": HEX64_C}
    block.update(over)
    return block


def allpos_row(**over):
    """A sound schema-/3 row of the all-positions arm WITH a reading and a stamped serving commit (the verifier's happy path)."""
    n, ns, nc = 200, 200, 170
    base = dict(instrument="gsm8k_paired200", model_key="qwen35_4b", view="it", cell=ALLPOS, arm=ALLPOS, cap=BASE, leg="r8192",
                n_paired=n, n_scored=ns, n_unscored=0, n_at_cap=0, at_cap_rate=0.0, n_unterminated_not_at_cap=0, n_correct=nc,
                point_conditional=nc / ns, ci_conditional_low=0.8, ci_conditional_high=0.9,
                point_score_as_is=nc / n, ci_score_as_is_low=0.8, ci_score_as_is_high=0.9, ci_method=PRW.CI_BOOTSTRAP,
                rung_state="not_applicable", object_generation="n/a", selection_outcome=OVERRIDE,
                unscored_reason={"unterminated_think_block": 0, "hit_cap_mid_solution": 0, "refusal": 0, "other": 0},
                source="synthetic", selection=selection_block(), serving_commit="3" * 40, fit_provenance=fit_provenance_block())
    base.update(over)
    return PRW.make_row(**base)


def test_the_all_positions_arm_row_verifies_with_its_three_provenance_fields_and_make_row_defaults_them_null(tmp_path):
    good = allpos_row()
    # /4: the three provenance fields sit just before the last two (`n_refusal_shaped`, `n_refusal_shaped_of`), which make_row defaults
    # null TOGETHER (never counted)
    assert PRW.row_problems(good) == [] and good["paper_arm"] == ALLPOS and tuple(good)[-5:-2] == PRW.PROVENANCE_FIELDS
    assert tuple(good)[-2:] == PRW.REFUSAL_SHAPED_FIELDS and all(good[f] is None for f in PRW.REFUSAL_SHAPED_FIELDS)
    # every EXISTING arm's rows carry the three fields null: the readers pass none, make_row fills them, the verifier requires nothing
    rows, _ = G8K.paper_rows_of_report(gsm8k_report(tmp_path), report_path=tmp_path / "report.json")
    assert rows and all(r["selection"] is None and r["serving_commit"] is None and r["fit_provenance"] is None for r in rows)
    PRW.verify_rows(rows)
    rows, _ = M5T.paper_rows_of_report(math500_report(tmp_path), report_path=tmp_path / "report.json")
    assert rows and all(r["selection"] is None and r["serving_commit"] is None and r["fit_provenance"] is None for r in rows)
    # a row of that arm WITHOUT a reading served nothing: serving_commit null is admitted there, the two declared blocks still required
    null = allpos_row(**{f: None for f in PRW.COUNT_FIELDS}, rung_state="never_run", serving_commit=None)
    assert PRW.row_problems(null) == []
    # a well-formed block on ANOTHER arm's row is verified too (presence is enough to be checked), a malformed one refuses by name
    other = {**rows[0], "selection": selection_block(outcome=rows[0]["selection_outcome"])}
    assert any("rule_returns" in p or "override" in p for p in PRW.row_problems(other)) or PRW.row_problems(other) == []
    assert any("fit_provenance is missing" in p for p in PRW.row_problems({**rows[0], "fit_provenance": {"fit_layer": 13}}))


def test_fit_provenance_missing_or_incomplete_refuses_by_name_and_is_never_filled():
    assert PRW.FIT_PROVENANCE_FIELDS == ("fitted_on_model", "fitted_on_revision", "rendering", "capture_positions", "item_source",
                                         "item_source_sha", "realised_counts", "direction_sha256", "direction_pt_sha256", "fit_layer",
                                         "fit_position", "direction_norm", "served_layer", "fit_quality_gate", "fit_record_sha256")
    assert PRW.CELLS_REQUIRING_FIT_PROVENANCE == (ALLPOS,) == PRW.CELLS_REQUIRING_SERVING_COMMIT == PRW.CELLS_DOSED_IN_RAW_COEFFICIENT
    problems = PRW.row_problems(allpos_row(fit_provenance=None))
    assert len(problems) == 1 and "fit_provenance is null on 'arditi_allpos'" in problems[0]
    for key in PRW.FIT_PROVENANCE_FIELDS:
        fp = fit_provenance_block()
        del fp[key]
        problems = PRW.row_problems(allpos_row(fit_provenance=fp))
        assert len(problems) == 1 and f"fit_provenance is missing ['{key}']" in problems[0], (key, problems)
        fp = fit_provenance_block(**{key: None})
        assert any(f"fit_provenance is missing ['{key}']" in p for p in PRW.row_problems(allpos_row(fit_provenance=fp))), key
    assert any("not a 64-hex sha256" in p for p in PRW.row_problems(allpos_row(fit_provenance=fit_provenance_block(direction_sha256="abc"))))
    assert any("not a non-empty mapping" in p for p in PRW.row_problems(allpos_row(fit_provenance=fit_provenance_block(realised_counts={}))))
    assert any("not an integer" in p for p in PRW.row_problems(allpos_row(fit_provenance=fit_provenance_block(served_layer="12"))))
    # extra keys ride along verbatim (the record carries notes beside the required keys)
    assert PRW.row_problems(allpos_row(fit_provenance=fit_provenance_block(served_layer_note="served = fit - 1 on this arm"))) == []
    with pytest.raises(PRW.PaperRowsError, match="fit_provenance is missing"):
        PRW.verify_rows([allpos_row(fit_provenance=fit_provenance_block(fit_record_sha256=None))])


def test_a_declared_producer_spelling_satisfies_a_key_but_nothing_stands_in_for_an_absent_fact():
    """A naming difference is not a missing fact: the admission record spells the fit items' commit ``item_source_repo_sha``, and a
    block carrying that name satisfies ``item_source_sha``. A key the record names NOWHERE is a different matter — it refuses, and
    no other field stands in for it."""
    assert PRW.FIT_PROVENANCE_ALIASES == {"item_source_sha": ("item_source_repo_sha",)}
    fp = fit_provenance_block()
    fp["item_source_repo_sha"] = fp.pop("item_source_sha")
    assert PRW.fit_provenance_problems(fp, tag="t") == [] and PRW.row_problems(allpos_row(fit_provenance=fp)) == []
    assert PRW.fit_provenance_value(fp, "item_source_sha") == "2" * 40 == PRW.fit_provenance_value(fp, "item_source_repo_sha")
    assert PRW.fit_provenance_value({k: v for k, v in fp.items() if k != "capture_positions"}, "capture_positions") is None
    # the alias is checked like the canonical name, not waved through
    assert any("item_source_sha 'nope' is not a 40-hex git commit" in p for p in PRW.fit_provenance_problems({**fp, "item_source_repo_sha": "nope"}, tag="t"))
    # the two GIT commits are checked in their own shape: a content digest is not a commit
    assert PRW.FIT_PROVENANCE_GIT_SHA_FIELDS == ("fitted_on_revision", "item_source_sha")
    assert any("fitted_on_revision" in p and "40-hex git commit" in p
               for p in PRW.fit_provenance_problems(fit_provenance_block(fitted_on_revision=HEX64_A), tag="t"))
    # an absent fact refuses even though a DIFFERENT position field rides the row (the served mode is not the capture positions)
    fp2 = {k: v for k, v in fit_provenance_block().items() if k != "capture_positions"}
    fp2["served_positions"] = "all"
    assert any("['capture_positions']" in p for p in PRW.fit_provenance_problems(fp2, tag="t"))


def test_the_admission_records_declared_key_set_maps_wholly_into_the_exports_required_set():
    """A MIRROR of the launcher's declared `fit_provenance` key tuple against this export's required set, so the two cannot drift
    silently. EVERY canonical key now has a producer name. It did not always: ``capture_positions`` — the positions the fit
    ACTIVATIONS were captured at — was stated only inside the record's ``rendering`` PROSE, which this export cannot parse and
    refuses to infer, so the arm's rows were unexportable until the record carried it as its own key (2026-09-17). A fact
    stated only in prose is a fact the export does not have.
    The two halves of that seam are handled differently ON PURPOSE, and this test pins both: ``item_source_repo_sha`` is the
    SAME quantity under another name and is an accepted ALIAS (refusing there would reject the authoritative record over
    spelling), while ``capture_positions`` was a MISSING FACT and was added. Alias a synonym; never alias a gap."""
    ruling = _load("steer_admission_ruling_for_paper_rows_test", REPO / "scripts" / "sdfbase" / "steer_admission_ruling.py")
    declared = set(ruling.FIT_PROVENANCE_KEYS)
    satisfied = {k for k in PRW.FIT_PROVENANCE_FIELDS if declared & {k, *PRW.FIT_PROVENANCE_ALIASES.get(k, ())}}
    assert set(PRW.FIT_PROVENANCE_FIELDS) - satisfied == set(), "the record's key set moved: re-read it, do not widen the alias table"
    assert "capture_positions" in declared, "capture_positions must stay a KEY of the record, never prose the export cannot read"
    assert not declared - {k for f in PRW.FIT_PROVENANCE_FIELDS for k in (f, *PRW.FIT_PROVENANCE_ALIASES.get(f, ()))}, \
        "the record declares a key this export does not carry"


def test_the_override_word_requires_a_selection_block_that_is_a_reading_of_something():
    # the word without the block
    problems = PRW.row_problems(allpos_row(selection=None))
    assert len(problems) == 1 and "no selection block" in problems[0] and OVERRIDE in problems[0]
    # a block without its ladder cap is not a reading of anything
    sel = selection_block(); del sel["ladder_cap"]
    problems = PRW.row_problems(allpos_row(selection=sel))
    assert any("selection lacks ['ladder_cap']" in p for p in problems)
    assert any("not a positive integer" in p and "reading's IDENTITY" in p for p in PRW.row_problems(allpos_row(selection=selection_block(ladder_cap=None))))
    # the block's outcome must be the row's word
    assert any("selection.outcome 'chosen' != the row's selection_outcome" in p for p in PRW.row_problems(allpos_row(selection=selection_block(outcome="chosen"))))
    # the override word beside a WITHIN-budget reading is false
    assert any("override_reading_pp 9.5 <= budget_pp 10" in p for p in PRW.row_problems(allpos_row(selection=selection_block(override_reading_pp=9.5))))
    assert any("override_reading_pp 10 <= budget_pp 10" in p for p in PRW.row_problems(allpos_row(selection=selection_block(override_reading_pp=10))))
    # the rule's own reading over budget is not the rule's reading
    assert any("rule_reading_pp 11 > budget_pp 10" in p for p in PRW.row_problems(allpos_row(selection=selection_block(rule_reading_pp=11))))
    # `rule_returns` naming the SELECTED point means the rule chose it — there was no override
    assert any("names the selected dose 0.3" in p for p in PRW.row_problems(allpos_row(selection=selection_block(rule_returns="ar_c0p3"))))
    assert any("names the selected dose 0.3" in p for p in PRW.row_problems(allpos_row(selection=selection_block(rule_returns=0.3))))
    assert any("names no dose the export can read" in p for p in PRW.row_problems(allpos_row(selection=selection_block(rule_returns="lam2"))))
    assert PRW.dose_of_rule_return("ar_c0p2") == 0.2 and PRW.dose_of_rule_return("ar_c1p25") == 1.25 and PRW.dose_of_rule_return(None) is None
    # the single-trial flag is stated, never inferred
    assert any("single_trial False is not true" in p for p in PRW.row_problems(allpos_row(selection=selection_block(single_trial=False))))
    # a null selection_outcome on this arm would read as the rule's own pick
    problems = PRW.row_problems(allpos_row(selection_outcome=None, selection=None))
    assert any("says so in exactly one word" in p for p in problems)
    # a sound block: the rule returns 0.2 within budget, the override reads over budget at 0.3
    assert PRW.row_problems(allpos_row()) == []


def test_an_alpha_shaped_dose_on_the_all_positions_arm_refuses_and_the_alpha_equiv_is_labelled():
    assert any("NEVER an alpha" in p for p in PRW.row_problems(allpos_row(selection=selection_block(dose={"value": 0.105, "unit": "residual_norm_fraction"}))))
    assert any("carries an `alpha` key" in p for p in PRW.row_problems(allpos_row(selection=selection_block(dose={"value": 0.3, "unit": "raw_coefficient", "alpha": 0.105}))))
    assert any("selection.dose.unit 'alpha' not in" in p for p in PRW.row_problems(allpos_row(selection=selection_block(dose={"value": 0.3, "unit": "alpha"}))))
    assert any("selection.dose 0.3 is not a" in p for p in PRW.row_problems(allpos_row(selection=selection_block(dose=0.3))))
    assert any("selection.dose lacks ['unit']" in p for p in PRW.row_problems(allpos_row(selection=selection_block(dose={"value": 0.3}))))
    assert any("alpha_equiv.note is empty" in p for p in PRW.row_problems(allpos_row(selection=selection_block(alpha_equiv={"value": 0.105, "note": ""}))))
    assert any("alpha_equiv 0.105 is not a" in p for p in PRW.row_problems(allpos_row(selection=selection_block(alpha_equiv=0.105))))
    assert PRW.row_problems(allpos_row(selection=selection_block(alpha_equiv=None))) == []


def test_the_new_cell_never_takes_the_prefill_only_arms_paper_key():
    good = allpos_row()
    problems = PRW.row_problems({**good, "paper_arm": "arditi"})
    assert any("paper_arm 'arditi' != the declared key 'arditi_allpos'" in p for p in problems)
    assert any("paper_arm 'arditi_allpos' != the declared key 'arditi'" in p for p in PRW.row_problems({**good, "cell": "arditi", "selection_outcome": None, "selection": None, "fit_provenance": None, "serving_commit": None}))
    assert PRW.make_row(**{k: good[k] for k in PRW.FIELDS if k != "paper_arm"})["paper_arm"] == "arditi_allpos"


def test_serving_commit_is_required_on_a_served_row_of_the_arm_and_never_a_stand_in():
    problems = PRW.row_problems(allpos_row(serving_commit=None))
    assert len(problems) == 1 and "serving_commit is null on a served row of 'arditi_allpos'" in problems[0] and "producer.tree_sha is the DRIVER's tree" in problems[0]
    assert any("not a 40-hex git commit" in p for p in PRW.row_problems(allpos_row(serving_commit="deadbeef")))
    assert any("not a 40-hex git commit" in p for p in PRW.row_problems(allpos_row(serving_commit="A" * 40)))
    assert PRW.row_problems(allpos_row(serving_commit="f" * 40)) == []
    with pytest.raises(PRW.PaperRowsError, match="serving_commit is null"):
        PRW.verify_rows([allpos_row(serving_commit=None)])


def test_the_gsm8k_producer_reads_the_three_fields_off_the_cell_record_and_refuses_a_bad_block(tmp_path):
    """The readers READ the blocks the matrix row declared (the GSM8K reader files them on the cell record beside
    `selection_outcome`), never compose them; a bad block refuses in the producer by name; a served row of the arm with no
    serving commit on record refuses at the writer (the store record carries none today — the documented gap)."""
    report = gsm8k_report(tmp_path)
    ck = "qwen35_4b/main/constv2-mix-full__psr_ce_all"          # the refused cell: no reading, so no serving commit is owed
    cell = report["cells"][ck]
    cell.update(cell=ALLPOS, selection_outcome=OVERRIDE, selection=selection_block(), fit_provenance=fit_provenance_block())
    rows, _ = G8K.paper_rows_of_report(report, report_path=tmp_path / "report.json")
    r = next(x for x in rows if x["cell"] == ALLPOS)
    assert r["paper_arm"] == ALLPOS and r["arm"] == ALLPOS and r["selection_outcome"] == OVERRIDE and r["n_paired"] is None
    assert r["selection"] == selection_block() and r["fit_provenance"] == fit_provenance_block() and r["serving_commit"] is None
    assert "selection: the admission record's block" in r["source"] and "fit_provenance: the admission record's block" in r["source"]
    PRW.verify_rows(rows)
    # the bad twins refuse in the producer, by name
    for bad_sel, needle in ((selection_block(rule_returns="ar_c0p3"), "names the selected dose"),
                            ({k: v for k, v in selection_block().items() if k != "ladder_cap"}, "ladder_cap"),
                            (selection_block(dose={"value": 0.105, "unit": "residual_norm_fraction"}), "NEVER an alpha")):
        report = gsm8k_report(tmp_path)
        report["cells"][ck].update(cell=ALLPOS, selection_outcome=OVERRIDE, selection=bad_sel, fit_provenance=fit_provenance_block())
        with pytest.raises(G8K.AnalysisError, match=needle):
            G8K.paper_rows_of_report(report)
    report = gsm8k_report(tmp_path)
    report["cells"][ck].update(cell=ALLPOS, selection_outcome=OVERRIDE, selection=selection_block(), fit_provenance={"fit_layer": 13})
    with pytest.raises(G8K.AnalysisError, match="fit_provenance is missing"):
        G8K.paper_rows_of_report(report)
    # the word without its block, or the arm without its fit provenance, is refused by the writer (make_row fills null, verify refuses)
    report = gsm8k_report(tmp_path)
    report["cells"][ck].update(cell=ALLPOS, selection_outcome=OVERRIDE)
    rows, _ = G8K.paper_rows_of_report(report)
    with pytest.raises(PRW.PaperRowsError, match="no selection block"):
        PRW.verify_rows(rows)
    # a SERVED row of the arm: the record names no serving commit -> the writer refuses rather than exporting the row
    report = gsm8k_report(tmp_path)
    ck2 = "qwen35_4b/main/constv2-mix-full__lora_r4_ce_minus_ctl"
    report["cells"][ck2].update(cell=ALLPOS, selection_outcome=OVERRIDE, selection=selection_block(), fit_provenance=fit_provenance_block())
    rows, _ = G8K.paper_rows_of_report(report)
    with pytest.raises(PRW.PaperRowsError, match="serving_commit is null on a served row"):
        PRW.verify_rows(rows)
    report["cells"][ck2]["serving_commit"] = "4" * 40          # a producer that stamps it is read verbatim
    rows, _ = G8K.paper_rows_of_report(report)
    PRW.verify_rows(rows)
    assert all(r["serving_commit"] == "4" * 40 for r in rows if r["cell"] == ALLPOS)


def test_the_math500_producer_reads_the_three_fields_off_the_leg_and_a_v2_document_still_reads(tmp_path):
    report = math500_report(tmp_path)
    lk = "nemotron3nano4b/constv2ctl2-full__lora_r4_ce/thinking_card"
    leg = report["legs"][lk]
    leg.update(selection_outcome=OVERRIDE, selection=selection_block(), fit_provenance=fit_provenance_block())
    rows, _ = M5T.paper_rows_of_report(report, report_path=tmp_path / "report.json")
    ctl = next(r for r in rows if r["arm"] == "lora_ctl2")
    assert ctl["selection_outcome"] == OVERRIDE and ctl["selection"] == selection_block() and ctl["fit_provenance"] == fit_provenance_block()
    assert ctl["serving_commit"] is None and "selection: the admission record's block" in ctl["source"]
    report["legs"][lk]["selection"] = selection_block(override_reading_pp=9.0)
    with pytest.raises(M5T.AnalysisError, match="<= budget_pp"):
        M5T.paper_rows_of_report(report)
    report["legs"][lk]["selection"] = selection_block()
    report["legs"][lk]["serving_commit"] = "xyz"
    with pytest.raises(M5T.AnalysisError, match="not a 40-hex git commit"):
        M5T.paper_rows_of_report(report)
    # a /2 document (no provenance fields) reads with them null; a foreign schema refuses
    d = tmp_path / "v2"
    d.mkdir()
    rows, _ = M5T.paper_rows_of_report(math500_report(tmp_path))
    old = {"schema": "paper_rows/2", "fields": [f for f in PRW.FIELDS if f not in PRW.PROVENANCE_FIELDS], "n_rows": len(rows),
           "not_exported": [], "rows": [{k: v for k, v in r.items() if k not in PRW.PROVENANCE_FIELDS} for r in rows]}
    (d / PRW.JSON_NAME).write_text(json.dumps(old), encoding="utf-8")
    back = PRW.read_paper_rows(d / PRW.JSON_NAME)
    assert back == rows and all(tuple(r) == PRW.FIELDS for r in back)
    (d / PRW.JSON_NAME).write_text(json.dumps({**old, "schema": "paper_rows/1"}), encoding="utf-8")
    with pytest.raises(PRW.PaperRowsError, match="not a paper_rows/2 / paper_rows/3 / paper_rows/4 document"):
        PRW.read_paper_rows(d / PRW.JSON_NAME)


def test_large_gap_rows_names_the_cells_where_the_two_primaries_part(tmp_path):
    rows, _ = M5T.paper_rows_of_report(math500_report(tmp_path))
    gaps = PRW.large_gap_rows(rows, pp=5.0)
    assert [g["arm"] for g in gaps] == ["object"] and math.isclose(gaps[0]["gap_pp"], (282 / 318 - 282 / 500) * 100)
    rows, _ = G8K.paper_rows_of_report(gsm8k_report(tmp_path))
    assert [(g["cell"][:12], g["leg"]) for g in PRW.large_gap_rows(rows, pp=5.0)] == [("constv2-mix-", "r8192"), ("constv2-mix-", "merged"), ("constv2-mix-", "r16384")]
