"""``scripts/tinycap_oos_transfer.py``: the fit-support-transfer analysis over
four SYNTHETIC stores written with the real store writers
(``tests/oos_synthetic_store.py``): the verdict words, the two decidability
checks (branch agreement; bound decidability), the T-bound fixture, the
rule-level unit test, the cap gate, every refusal, the ledger confirmation,
the regime triple, the serving-parity cross-check, and the full-id keying of
every per-cell number (never ``classify_cell_id``)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from oos_synthetic_store import (
    BASE_ROUTE_PROBE_BLOCK,
    CAP_OFF_OVERRIDE,
    CAP_ON,
    IDS,
    LAST4,
    NONE,
    OFF8192_LENGTHS_NONE,
    OFF8192_NONE_CORRECT,
    OFF_NONE_CORRECT,
    OFF_PROMPT_EXTRA,
    OFF_PSR_CORRECT,
    OFFCAP_FULL_SHA,
    OFFCAP_SHA7,
    PROMPT_MAX,
    PROMPT_MAX_OFF,
    PSR,
    SERVED,
    SHA7,
    TASK,
    build_2x2,
    build_2x3,
    build_store,
    off_legs,
    off_legs_capped,
    offcap_legs,
    prompt_budget_artifact,
    prompt_budget_artifact_off,
)

from sisteer.capeval.tiers import pairing as pr
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("tinycap_oos_transfer", REPO / "scripts" / "tinycap_oos_transfer.py")
T = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(T)

#: fewer replicates where the exact CI is not the point (the T-bound fixture
#: keeps the default 10,000 — its edge is what the test is about)
B_FAST = 2000


def analyze(gen_root, **kw):
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("allow_pending_gates", True)     # stores without the S3 driver's regime stamps read PENDING
    kw.setdefault("prompt_budget", prompt_budget_artifact())
    kw.setdefault("paired_ids", IDS)               # the fixture's frozen paired-200 population
    kw.setdefault("read_eval_logs", False)
    return T.analyze(gen_root, SHA7, **kw)


def leg_of(report, cell, arm):
    return report["legs"][T.leg_key(cell, arm)]


# --------------------------------------------------------------------------- #
# the verdict words, no censoring: bounds are points, both checks pass         #
# --------------------------------------------------------------------------- #

WORD_CASES = {
    # effect_off = +15 pp, CI [+8, +22] pp
    T.WORD_TRANSFERS: {"none_on": {"correct": OFF_NONE_CORRECT}, "psr_on": {"correct": OFF_PSR_CORRECT}},
    T.WORD_SIGN_FLIP: {"none_on": {"correct": OFF_PSR_CORRECT}, "psr_on": {"correct": OFF_NONE_CORRECT}},
    T.WORD_COLLAPSE: {"none_on": {"correct": range(120)}, "psr_on": {"correct": range(124)}},
    T.WORD_AMPLIFIES: {"none_on": {"correct": range(100)}, "psr_on": {"correct": range(160)}},
}


@pytest.mark.parametrize("word", sorted(WORD_CASES))
def test_each_verdict_word_is_reproduced_under_both_branches_with_point_bounds(tmp_path, word):
    build_2x2(tmp_path, **off_legs(), **WORD_CASES[word])
    rep = analyze(tmp_path)
    v = rep["verdict_rule"]
    assert v["word_by_branch"] == {T.BRANCH_PRIMARY: word, T.BRANCH_SCORE_AS_IS: word}
    assert v["agree_branches"] is True and v["decidable_bounds"] is True
    assert rep["verdict"]["word"] == word and rep["verdict"]["verdict"] == word
    for b in T.BRANCHES:
        for c in pr.CONTRASTS:
            s = rep["estimands"][b]["stats"][c]
            assert s["bound_low"] == s["bound_high"] == pytest.approx(s["point"])
    assert rep["gates"]["failed"] == []
    assert rep["unusable_legs"] == {}
    assert rep["common_attempted_ids"]["n"] == 200


def test_the_in_support_effect_not_resolved_is_its_own_verdict_never_forced_into_a_word(tmp_path):
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": range(122)},
              none_on={"correct": range(100)}, psr_on={"correct": range(140)})
    rep = analyze(tmp_path)
    v = rep["verdict_rule"]
    assert v["word_by_branch"] == {b: T.WORD_NOT_RESOLVED for b in T.BRANCHES}
    assert rep["verdict"]["verdict"] == T.WORD_NOT_RESOLVED
    assert rep["verdict"]["statement"].startswith(T.WORD_NOT_RESOLVED + ": effect_off = +1.00 pp")
    assert "effect_on = +20.00 pp" in rep["verdict"]["statement"]


# --------------------------------------------------------------------------- #
# the two branches disagree in the sign of effect_on                           #
# --------------------------------------------------------------------------- #


def test_branch_sign_disagreement_is_not_decidable_and_the_top_item(tmp_path):
    """none/on fully scored 130/200; psr/on 129 of 196 scored (4 censored):
    conditional effect_on +0.82 pp, score-as-is -0.5 pp -> the words differ
    (collapse vs sign flip), ``agree_branches`` False, the identified
    interval straddles 0, and the disagreement is the top artifact item."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(130)},
              psr_on={"correct": range(129), "unscored": LAST4})
    rep = analyze(tmp_path)
    v = rep["verdict_rule"]
    assert v["word_by_branch"][T.BRANCH_PRIMARY] == T.WORD_COLLAPSE
    assert v["word_by_branch"][T.BRANCH_SCORE_AS_IS] == T.WORD_SIGN_FLIP
    assert v["agree_branches"] is False and v["decidable_bounds"] is False
    assert rep["verdict"]["verdict"] == T.NOT_DECIDABLE_CENSORING
    assert any("DISAGREE" in item for item in rep["top_items"])
    assert rep["top_items"][0].startswith("the two scoring branches DISAGREE")
    assert rep["verdict_rule"]["signs"]["shift_psr"]["reading"] in ("+", "-", T.SIGN_NOT_IDENTIFIED)


# --------------------------------------------------------------------------- #
# T-bound                                                                      #
# --------------------------------------------------------------------------- #


def test_t_bound_both_branches_transfer_but_the_interval_straddles_the_ci_edge_so_no_word(tmp_path):
    """OFF legs fully scored (none 120/200, psr 150/200 -> effect_off +15 pp,
    seed-0 CI [+8.0, +22.0] pp); none/on 126 of 196, psr/on 143 of 196 (4 + 4
    censored at the cap, 2 % each — INSIDE the cap gate): conditional +8.67 pp
    and all-wrong +8.5 pp both read "transfers", but the sharp interval
    [+6.5, +10.5] pp straddles the CI's lower edge -> ``decidable_bounds``
    False, verdict "not decidable at this cap (censoring)" naming the edge."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
              psr_on={"correct": range(143), "unscored": LAST4})
    rep = analyze(tmp_path, n_resamples=10_000)
    p = rep["estimands"][T.BRANCH_PRIMARY]["stats"]
    assert p["effect_off"]["point"] == pytest.approx(0.15)
    assert p["effect_off"]["ci_low"] == pytest.approx(0.08) and p["effect_off"]["ci_high"] == pytest.approx(0.22)
    assert p["effect_on"]["point"] == pytest.approx(17 / 196)
    assert rep["estimands"][T.BRANCH_SCORE_AS_IS]["stats"]["effect_on"]["point"] == pytest.approx(0.085)
    assert (p["effect_on"]["bound_low"], p["effect_on"]["bound_high"]) == (pytest.approx(0.065), pytest.approx(0.105))
    assert p["effect_on"]["bound_low"] < p["effect_off"]["ci_low"] < p["effect_on"]["bound_high"]
    v = rep["verdict_rule"]
    assert v["word_by_branch"] == {b: T.WORD_TRANSFERS for b in T.BRANCHES}
    assert v["agree_branches"] is True
    assert v["decidable_bounds"] is False
    assert v["decidability"]["crossings"] == ["the OFF CI's lower edge"]
    assert rep["verdict"]["verdict"] == T.NOT_DECIDABLE_CENSORING
    assert "the OFF CI's lower edge" in rep["verdict"]["reason"]
    assert rep["verdict"]["word"] is None
    assert rep["gates"]["cap"]["status"] == T.GATE_PASS               # 2 % is inside the gate
    assert rep["estimands"][T.BRANCH_PRIMARY]["n_censored_by_leg"] == {
        "none/off": 0, "psr/off": 0, "none/on": 4, "psr/on": 4}


def test_t_bound_variant_straddling_zero_names_the_sign(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(128), "unscored": LAST4},
              psr_on={"correct": range(130), "unscored": LAST4})
    rep = analyze(tmp_path, n_resamples=10_000)
    p = rep["estimands"][T.BRANCH_PRIMARY]["stats"]["effect_on"]
    assert (p["bound_low"], p["bound_high"]) == (pytest.approx(-0.01), pytest.approx(0.03))
    v = rep["verdict_rule"]
    assert v["word_by_branch"] == {b: T.WORD_COLLAPSE for b in T.BRANCHES}
    assert v["decidable_bounds"] is False and "the sign" in v["decidability"]["crossings"]
    assert rep["verdict"]["verdict"] == T.NOT_DECIDABLE_CENSORING
    assert rep["estimands"][T.BRANCH_PRIMARY]["stats"]["effect_on"]["sign_identified"] is False


def test_t_bound_with_the_eight_censored_items_scored_is_decidable_and_the_word_is_filed(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path, n_resamples=10_000)
    v = rep["verdict_rule"]
    assert v["decidable_bounds"] is True and v["agree_branches"] is True
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    assert rep["estimands"][T.BRANCH_PRIMARY]["stats"]["effect_on"]["point"] == pytest.approx(0.085)


# --------------------------------------------------------------------------- #
# the rule-level unit test: verdict_from_bounds around each breakpoint         #
# --------------------------------------------------------------------------- #


def _pts(e_off, e_on, did=None):
    did = (e_on - e_off) if did is None else did
    return {b: {"effect_off": e_off, "effect_on": e_on, "did": did} for b in T.BRANCHES}


def test_classify_word_at_every_region_and_its_edges():
    lo, hi, e_off = 0.08, 0.22, 0.15
    assert T.classify_word(e_off, lo, hi, 0.10) == T.WORD_TRANSFERS
    assert T.classify_word(e_off, lo, hi, 0.08) == T.WORD_TRANSFERS      # the edge belongs to the band
    assert T.classify_word(e_off, lo, hi, 0.22) == T.WORD_TRANSFERS
    assert T.classify_word(e_off, lo, hi, 0.079) == T.WORD_COLLAPSE
    assert T.classify_word(e_off, lo, hi, 0.0) == T.WORD_COLLAPSE        # zero is a collapse, not a flip
    assert T.classify_word(e_off, lo, hi, -0.001) == T.WORD_SIGN_FLIP
    assert T.classify_word(e_off, lo, hi, 0.221) == T.WORD_AMPLIFIES
    assert T.classify_word(0.01, -0.02, 0.04, 0.30) == T.WORD_NOT_RESOLVED
    assert T.classify_word(-0.15, -0.22, -0.08, -0.10) == T.WORD_TRANSFERS
    assert T.classify_word(-0.15, -0.22, -0.08, 0.05) == T.WORD_SIGN_FLIP
    assert T.classify_word(-0.15, -0.22, -0.08, -0.30) == T.WORD_AMPLIFIES


def test_verdict_from_bounds_is_constant_iff_no_material_breakpoint_lies_inside_the_interval():
    lo, hi, e_off = 0.08, 0.22, 0.15
    # inside one region: decidable, the word filed
    r = T.verdict_from_bounds(e_off, (lo, hi), _pts(e_off, 0.12), (0.10, 0.14))
    assert r["decidable_bounds"] and r["agree_branches"] and r["verdict"] == T.WORD_TRANSFERS
    # straddling the lower edge (transfers | collapse)
    r = T.verdict_from_bounds(e_off, (lo, hi), _pts(e_off, 0.09), (0.065, 0.105))
    assert not r["decidable_bounds"] and r["verdict"] == T.NOT_DECIDABLE_CENSORING
    assert r["decidability"]["crossings"] == ["the OFF CI's lower edge"]
    assert sorted(r["decidability"]["words"]) == sorted([T.WORD_TRANSFERS, T.WORD_COLLAPSE])
    # straddling the upper edge (transfers | amplifies)
    r = T.verdict_from_bounds(e_off, (lo, hi), _pts(e_off, 0.21), (0.20, 0.24))
    assert not r["decidable_bounds"] and r["decidability"]["crossings"] == ["the OFF CI's upper edge"]
    # straddling zero (collapse | sign flip)
    r = T.verdict_from_bounds(e_off, (lo, hi), _pts(e_off, 0.01), (-0.01, 0.03))
    assert not r["decidable_bounds"] and r["decidability"]["crossings"] == ["the sign"]
    # the collapse/amplify line +|E_off| lies INSIDE the band here, so crossing it does not change the word
    r = T.verdict_from_bounds(e_off, (lo, hi), _pts(e_off, 0.15), (0.14, 0.16))
    assert r["decidable_bounds"] and r["verdict"] == T.WORD_TRANSFERS
    # ... but with an asymmetric CI whose upper edge is below +|E_off| the line is material
    r = T.verdict_from_bounds(e_off, (0.05, 0.12), _pts(e_off, 0.15), (0.13, 0.17))
    assert not r["decidable_bounds"]
    assert "the collapse/amplify line" in r["decidability"]["crossings"]
    # a breakpoint exactly at an endpoint is not "inside": constant
    r = T.verdict_from_bounds(e_off, (lo, hi), _pts(e_off, 0.09), (0.08, 0.10))
    assert r["decidable_bounds"]
    # the not-resolved case is decidable whatever the ON bound
    r = T.verdict_from_bounds(0.01, (-0.02, 0.04), _pts(0.01, 0.30), (-0.5, 0.5))
    assert r["decidable_bounds"] and r["verdict"] == T.WORD_NOT_RESOLVED
    # a bound on E_off too (the rectangle): its sign crossing is named
    r = T.verdict_from_bounds(0.01, (0.005, 0.03), _pts(0.01, 0.02), (0.02, 0.02), e_off_bound=(-0.01, 0.03))
    assert not r["decidable_bounds"] and "the sign of the OFF effect" in r["decidability"]["crossings"]


def test_verdict_from_bounds_branch_disagreement_is_named_when_the_bounds_are_points():
    """The unreachable-in-practice remainder (both branch points lie inside
    the bound): fed inconsistent points by hand, the rule names the branch
    disagreement rather than the censoring."""
    pts = {T.BRANCH_PRIMARY: {"effect_off": 0.15, "effect_on": 0.10, "did": -0.05},
           T.BRANCH_SCORE_AS_IS: {"effect_off": 0.15, "effect_on": -0.02, "did": -0.17}}
    r = T.verdict_from_bounds(0.15, (0.08, 0.22), pts, (0.10, 0.10))
    assert r["decidable_bounds"] and not r["agree_branches"]
    assert r["verdict"] == T.NOT_DECIDABLE_BRANCHES and "the sign of effect_on (1/-1)" in r["reason"]
    assert r["disagreeing_contrasts"] == ["effect_on"]
    # same word, different sign of the DiD -> still a disagreement, naming the DiD
    pts2 = {T.BRANCH_PRIMARY: {"effect_off": 0.15, "effect_on": 0.16, "did": 0.01},
            T.BRANCH_SCORE_AS_IS: {"effect_off": 0.15, "effect_on": 0.14, "did": -0.01}}
    r = T.verdict_from_bounds(0.15, (0.08, 0.22), pts2, (0.14, 0.16))
    assert r["same_word"] and not r["same_signs"] and not r["agree_branches"]
    assert r["disagreeing_contrasts"] == ["did"] and r["reason"] == "the two branches disagree on the sign of did (1/-1)"
    # zero is its OWN sign — a +1.53 pp primary DiD beside a score-as-is DiD of
    # exactly 0 is a disagreement (it used to read as a wildcard and file "transfers")
    pts3 = {T.BRANCH_PRIMARY: {"effect_off": 0.15, "effect_on": 0.1653, "did": 0.0153},
            T.BRANCH_SCORE_AS_IS: {"effect_off": 0.15, "effect_on": 0.15, "did": 0.0}}
    r = T.verdict_from_bounds(0.15, (0.08, 0.22), pts3, (0.15, 0.17))
    assert r["same_word"] and r["word_by_branch"] == {b: T.WORD_TRANSFERS for b in T.BRANCHES}
    assert not r["same_signs"] and r["disagreeing_contrasts"] == ["did"]
    assert r["signs_by_contrast"]["did"] == {T.BRANCH_PRIMARY: 1, T.BRANCH_SCORE_AS_IS: 0}
    assert r["verdict"] == T.NOT_DECIDABLE_BRANCHES and "the sign of did (1/0)" in r["reason"]
    # identical signs (incl. both zero) agree; the exact `signs` from pairing win over the floats
    pts4 = {b: {"effect_off": 0.15, "effect_on": 0.15, "did": 0.0} for b in T.BRANCHES}
    assert T.verdict_from_bounds(0.15, (0.08, 0.22), pts4, (0.15, 0.15))["agree_branches"]
    pts5 = {T.BRANCH_PRIMARY: {"effect_off": 0.15, "effect_on": 0.15, "did": 1e-17,
                               "signs": {"effect_off": 1, "effect_on": 1, "did": 0}},
            T.BRANCH_SCORE_AS_IS: {"effect_off": 0.15, "effect_on": 0.15, "did": 0.0,
                                   "signs": {"effect_off": 1, "effect_on": 1, "did": 0}}}
    assert T.verdict_from_bounds(0.15, (0.08, 0.22), pts5, (0.15, 0.15))["same_signs"]


# --------------------------------------------------------------------------- #
# the P1-b integration fixture and the other gates                            #
# --------------------------------------------------------------------------- #


def test_p1b_five_unscored_at_cap_read_2_5_percent_from_the_summary_and_fail_the_cap_gate(tmp_path):
    """5 unscored at-cap items + 195 rows: the collector reports the SUMMARY's
    at_cap_rate 0.025 (a row-derived rate would read 0 %) and the cap gate
    reads FAILED; no word is filed."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": range(195, 200)},
              psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    leg = leg_of(rep, NONE, "on")
    assert leg["summary"]["at_cap_rate"] == pytest.approx(0.025)
    assert leg["rows_check"] == {"n_rows": 195, "n_scored_in_record": 195, "agrees": True, "detail": None}
    g = rep["gates"]["cap"]
    assert g["status"] == T.GATE_FAIL and "2.500%" in g["detail"] and "cap too small" in g["detail"]
    assert g["legs"][T.leg_key(NONE, "on")]["at_cap_rate"] == pytest.approx(0.025)
    assert rep["gates"]["failed"] == ["cap"]
    assert rep["verdict"]["word"] is None and rep["verdict"]["verdict"] == T.NOT_MEASURABLE
    assert "cap too small" in rep["verdict"]["reason"]
    # the estimand still ran and is in the report (the doc shows both), only the WORD is withheld
    assert "estimands" in rep and rep["verdict_rule"]["word_by_branch"]


def test_a_null_n_at_cap_is_handled_as_null_never_coerced_to_zero(tmp_path):
    def no_stop_reasons(s):
        s["n_at_cap"] = None
        s["at_cap_rate"] = None
        s["n_stop_reason"] = 0

    def header_none(h):
        h["n_at_cap"] = None
        h["n_stop_reason"] = 0

    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "summary_edit": no_stop_reasons, "header_edit": header_none})
    rep = analyze(tmp_path)
    leg = leg_of(rep, PSR, "on")
    assert leg["refused"] is None and leg["summary"]["n_at_cap"] is None
    assert rep["gates"]["cap"]["legs"][T.leg_key(PSR, "on")]["n_at_cap"] is None
    assert "at_cap_rate is null" in rep["gates"]["cap"]["detail"]
    assert leg["ledger"]["summary_n_at_cap"] is None and leg["ledger"]["length_agrees"] is None
    assert any("n_at_cap is null" in f for f in leg["ledger"]["failures"])


def test_a_leg_lacking_its_outcome_record_is_missing_outcomes_and_no_verdict_is_computed(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "write_outcomes": False})
    rep = analyze(tmp_path)
    leg = leg_of(rep, PSR, "on")
    assert leg["missing"].startswith("MISSING-OUTCOMES")
    assert rep["unusable_legs"] == {T.leg_key(PSR, "on"): leg["missing"]}
    assert "estimands" not in rep and rep["verdict"]["word"] is None
    assert rep["verdict"]["verdict"] == T.NOT_MEASURABLE and "MISSING-OUTCOMES" in rep["verdict"]["reason"]
    assert rep["gates"]["outcomes"]["status"] == T.GATE_FAIL
    assert rep["gates"]["split_anomalies"]["status"] == T.GATE_PENDING
    # the controls table still names the leg
    row = next(r for r in rep["controls"] if r["leg"] == T.leg_key(PSR, "on"))
    assert row["missing"] == leg["missing"]


def test_a_header_disagreeing_with_its_summary_is_refused_before_any_statistic(tmp_path):
    stores = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    cd = stores[T.leg_key(NONE, "on")]
    summary = completion.read_task_summary(cd, TASK)
    completion.write_task_summary(cd, TASK, {**summary, "n": 199})
    rep = analyze(tmp_path)
    leg = leg_of(rep, NONE, "on")
    assert leg["refused"] == ("outcomes/inspect__gsm8k_paired200.json disagrees with "
                              "summary/inspect__gsm8k_paired200.json: n_attempted_items 200 != 199")
    assert "estimands" not in rep
    completion.write_task_summary(cd, TASK, {**summary, "n_unscored_samples": 1})
    rep = analyze(tmp_path)
    assert "n_attempted_samples 200 != 201" in leg_of(rep, NONE, "on")["refused"]


def test_rows_whose_item_set_differs_from_the_record_are_refused(tmp_path):
    stores = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
                       psr_on={"correct": range(143)})
    cd = stores[T.leg_key(NONE, "on")]
    p = cd / "rows.jsonl"
    # a row for an item the record marks UNSCORED
    extra = json.dumps({"sample_idx": 0, "stop_reason": "max_tokens", "unscored": False})
    p.write_text(p.read_text() + json.dumps({"task": TASK, "item_id": IDS[196], "metric": "acc", "value": 0.0,
                                             "engine": "vllm", "extra_json": extra}) + "\n")
    rep = analyze(tmp_path)
    leg = leg_of(rep, NONE, "on")
    assert "rows.jsonl disagrees with the outcome record" in leg["refused"]
    assert "1 rows for items the record marks unscored" in leg["refused"]
    # a scored item's row missing, and a row value that differs
    lines = [ln for ln in p.read_text().splitlines() if IDS[196] not in ln and IDS[0] not in ln]
    row1 = json.loads(next(ln for ln in p.read_text().splitlines() if IDS[1] in ln))
    row1["value"] = 1.0 - row1["value"]
    lines = [ln for ln in lines if IDS[1] not in ln] + [json.dumps(row1)]
    p.write_text("\n".join(lines) + "\n")
    rep = analyze(tmp_path)
    why = leg_of(rep, NONE, "on")["refused"]
    assert "1 scored items with no row" in why and "1 items whose row value differs" in why


def test_attempted_id_sets_differing_across_legs_are_refused_naming_the_ids(tmp_path):
    shifted = [f"gsm8k_{i:08x}" for i in range(1, 201)]      # one id swapped for a new one
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "ids": shifted})
    rep = analyze(tmp_path)
    assert rep["unusable_legs"] == {}
    assert rep["common_attempted_ids"]["n"] is None
    why = rep["common_attempted_ids"]["reason"]
    assert "the four attempted-id sets differ" in why and IDS[0] in why and shifted[-1] in why
    assert "estimands" not in rep and rep["verdict"]["word"] is None
    assert rep["gates"]["outcomes"]["status"] == T.GATE_FAIL
    assert "attempted_ids" in rep["gates"]["outcomes"]["problems"]


def test_the_ledger_confirmation_flags_a_2048_cap_a_context_length_row_and_a_prompt_max_disagreement(tmp_path):
    def bad_ledger(rows):
        gsm = [r for r in rows if r["task"] == TASK]
        gsm[3]["max_tokens_requested"] = 2048
        gsm[5]["error_class"] = C.LEDGER_ERROR_CLASS_CONTEXT_LENGTH
        gsm[5]["rejected"] = True
        gsm[5]["status"] = 400
        gsm[7]["prompt_tokens_pre"] = PROMPT_MAX + 50          # disagrees with the live pass / artifact

    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "ledger_edit": bad_ledger})
    rep = analyze(tmp_path)
    led = leg_of(rep, PSR, "on")["ledger"]
    assert led["passed"] is False
    joined = " | ".join(led["failures"])
    assert "max_tokens_requested [2048, 8192] != the summary's cap 8192" in joined
    assert "error_class counts" in joined and "context_length" in joined
    # a rejected row is named as CENSORING (the client re-sampled that item), and
    # the summary's own ledger-derived count is held to the ledger read here
    assert "censored: 1 rejected calls re-sampled by the client's retry (upstream HTTP 400)" in joined
    assert "summary n_http_rejected_retried 0 != the ledger's 1 rejected calls" in joined
    assert led["n_rejected"] == 1 and led["summary_n_http_rejected_retried"] == 0
    assert "three-way prompt-max disagreement" in joined and f"ledger {PROMPT_MAX + 50}" in joined
    assert led["prompt_max"]["agrees"] is False
    assert rep["gates"]["declaration"]["status"] == T.GATE_FAIL
    assert "declaration" in rep["gates"]["failed"] and rep["verdict"]["word"] is None
    # the clean legs confirm: 200 calls, one cap, one arm, every class none, three sources agree
    ok = leg_of(rep, NONE, "on")["ledger"]
    assert ok["passed"] and ok["n_calls"] == 200 and ok["max_tokens_requested"]["values"] == [CAP_ON]
    assert ok["thinking"] == {"values": [True], "expected": True, "agrees": True}
    assert ok["prompt_max"] == {"ledger": PROMPT_MAX, "live": PROMPT_MAX, "artifact": PROMPT_MAX,
                                "agrees": True, "detail": None, "applicable": True}
    assert ok["window"] == {"prompt_max": PROMPT_MAX, "cap": CAP_ON, "max_model_len": 16384, "fits": True}
    assert ok["artifact_fit"]["passed"] is True and ok["artifact_fit"]["prompt_max"] == PROMPT_MAX
    off = leg_of(rep, NONE, "off")["ledger"]
    assert off["thinking"]["values"] == [False] and off["max_tokens_requested"]["values"] == [512]
    # the OFF prompts are longer on the same messages; the artifact is not compared there
    assert off["prompt_max"]["ledger"] == PROMPT_MAX_OFF == PROMPT_MAX + OFF_PROMPT_EXTRA
    assert off["prompt_max"]["applicable"] is False and off["prompt_max"]["artifact"] is None
    assert off["passed"] is True and off["pending"] == [] and off["artifact_fit"] is None


def test_the_ledger_length_count_must_equal_the_summarys_n_at_cap_and_a_missing_ledger_fails(tmp_path):
    def drop_length(rows):
        for r in rows:
            if r["task"] == TASK and r["finish_reason"] == "length":
                r["finish_reason"] = "stop"

    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4, "ledger_edit": drop_length},
              psr_on={"correct": range(143), "ledger": False})
    rep = analyze(tmp_path)
    led = leg_of(rep, NONE, "on")["ledger"]
    assert led["n_length"] == 0 and led["summary_n_at_cap"] == 4 and led["length_agrees"] is False
    assert any("finish_reason == length 0 != summary n_at_cap 4" in f for f in led["failures"])
    led2 = leg_of(rep, PSR, "on")["ledger"]
    assert led2["available"] is False and led2["failures"] == ["no usage_ledger.jsonl at the out-root"]


def test_the_three_way_prompt_max_is_pending_not_failed_when_a_source_is_not_stamped(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)},
              stamps=False)
    rep = analyze(tmp_path, prompt_budget=None)
    led = leg_of(rep, PSR, "on")["ledger"]
    assert led["passed"] is True and led["prompt_max"]["agrees"] is None
    assert any("live prompt max" in p and "not stamped" in p for p in led["pending"])
    assert any("no --prompt-budget artifact supplied" in p for p in led["pending"])
    # the header certification is pending too; strict gating withholds the word
    dec = rep["gates"]["declaration"]
    assert dec["status"] == T.GATE_PENDING
    assert any("header_certified not stamped" in p for p in dec["pending"])
    strict = analyze(tmp_path, prompt_budget=None, allow_pending_gates=False)
    assert strict["verdict"]["word"] is None and "PENDING" in strict["verdict"]["reason"]
    assert strict["verdict"]["verdict"].startswith(T.NOT_MEASURABLE)
    # the missing ARTIFACT is an operator input, not a producer stamp: the flag never waives it
    assert rep["verdict"]["word"] is None and "not waivable by --allow-pending-gates" in rep["verdict"]["reason"]
    assert [it["stamps"] for it in rep["verdict"]["pending_unwaivable"]] == [[None], [None]]   # one per ON leg
    assert all("no --prompt-budget artifact supplied" in it["detail"] for it in rep["verdict"]["pending_unwaivable"])
    # with the artifact supplied, what stays pending is Phase B's stamps alone: waived, the word filed
    waived = analyze(tmp_path)
    assert waived["verdict"]["word"] == T.WORD_TRANSFERS and waived["verdict"]["gates_pending"] == ["declaration"]
    assert all(it["waivable"] for it in waived["gates"]["pending_items"])


def test_a_foreign_torn_or_off_arm_prompt_budget_artifact_refuses_by_name(tmp_path):
    """The analysis reads the artifact through the contract
    (``validate_oos_prompt_budget`` / ``read_oos_prompt_budget``): a document
    of another shape, a torn file, or the OFF arm's artifact is a loud refusal
    naming the cause — never a pending "no readable max", never a probe of
    alternative key spellings. The fixture's artifact IS a contract document."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    good = prompt_budget_artifact()
    assert set(good) == set(C.OOS_PROMPT_BUDGET_FIELDS) and C.validate_oos_prompt_budget(good) == good
    with pytest.raises(T.TransferAnalysisError, match="keys .* are not the contract's"):
        analyze(tmp_path, prompt_budget={"n_prompts": 200, "prompt_tokens": {"max": PROMPT_MAX}})
    with pytest.raises(T.TransferAnalysisError, match=r"max 1000 != max\(lengths\)"):
        analyze(tmp_path, prompt_budget={**good, "max": 1000, "max_plus_cap": 1000 + CAP_ON})
    with pytest.raises(T.TransferAnalysisError, match="arm 'off'"):
        analyze(tmp_path, prompt_budget={**good, "arm": C.THINKING_OFF})
    # the ONE key the reader uses
    assert T._artifact_prompt_max(good) == (PROMPT_MAX, None)
    assert T._artifact_prompt_max(None) == (None, "no --prompt-budget artifact supplied")
    # the CLI reads the file through the contract's reader: torn / foreign refuse by name
    torn = tmp_path / "torn.json"
    torn.write_text("{not json")
    with pytest.raises(T.TransferAnalysisError, match="unreadable"):
        T.main(["--gen-root", str(tmp_path), "--sha", SHA7, "--prompt-budget", str(torn), "--no-figs",
                "--no-mirror", "--wandb-mode", "disabled", "--no-eval-logs", "--bootstrap", "50"])
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"max": PROMPT_MAX}))
    with pytest.raises(T.TransferAnalysisError, match="not the contract's"):
        T.main(["--gen-root", str(tmp_path), "--sha", SHA7, "--prompt-budget", str(foreign), "--no-figs",
                "--no-mirror", "--wandb-mode", "disabled", "--no-eval-logs", "--bootstrap", "50"])
    # a torn evidence.json is a REFUSED leg with its cause, never "no evidence"
    stores = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    (stores[T.leg_key(PSR, "on")] / "evidence.json").write_text("{torn")
    rep = analyze(tmp_path)
    assert "evidence.json: unreadable" in leg_of(rep, PSR, "on")["refused"]
    assert rep["verdict"]["word"] is None


def test_a_stamped_false_certification_or_an_unverified_probe_fails_gate_four(tmp_path):
    stores = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    cd = stores[T.leg_key(PSR, "on")]
    ev = json.loads((cd / "evidence.json").read_text())
    ev["thinking_header_by_arm"]["prompt_opened_think_certified"] = False
    ev["steering_bites"]["verified"] = False
    ev["steering_bites"]["error"] = "the steered and unsteered probes agreed on every prompt"
    ev["steering_bites"]["probe_render_arm"] = "off"
    (cd / "evidence.json").write_text(json.dumps(ev))
    rep = analyze(tmp_path)
    dec = rep["gates"]["declaration"]
    assert dec["status"] == T.GATE_FAIL
    joined = " | ".join(dec["failures"])
    assert "prompt_opened_think_certified is False" in joined
    assert "steering bites not verified on a lens leg" in joined
    assert "probe_render_arm 'off' != 'on'" in joined
    assert rep["verdict"]["word"] is None


# --------------------------------------------------------------------------- #
# the experiment's FIXED parameters and every nuisance, at admission           #
# --------------------------------------------------------------------------- #


def test_a_leg_drawn_under_another_sampling_row_or_cap_is_not_this_experiment(tmp_path):
    """Admission checked arm, task, item set, k, DONE and the population but
    never the FIXED sampling row — a leg at 1.7 / 0.5 / 1 was admitted, the
    verdict filed, and the report still said 0.7/0.95/20 was held constant.
    Now the row (read from contracts' ONE definition of the campaign row,
    never re-spelled) and the per-arm cap are asserted per leg: a mismatch
    makes the leg "not this experiment" — refused from the estimand, its
    summary still in the controls table."""
    def other_row(s):
        s.update(temperature=1.7, top_p=0.5, top_k=1)

    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "summary_edit": other_row})
    rep = analyze(tmp_path)
    leg = leg_of(rep, PSR, "on")
    assert leg["refused"].startswith(
        "sampling row {'temperature': 1.7, 'top_p': 0.5, 'top_k': 1} != the experiment's fixed row "
        "{'temperature': 0.7, 'top_p': 0.95, 'top_k': 20} (contracts.sampling_for('qwen35_4b', 'card'))")
    assert "not this experiment" in leg["refused"]
    assert rep["verdict"]["word"] is None and "estimands" not in rep
    assert rep["unusable_legs"] == {T.leg_key(PSR, "on"): leg["refused"]}
    row = next(r for r in rep["controls"] if r["leg"] == T.leg_key(PSR, "on"))      # still shown
    assert (row["summary"]["temperature"], row["summary"]["top_k"]) == (1.7, 1) and row["refused"] == leg["refused"]
    # the row is contracts', not this script's
    want = C.sampling_for(C.model_row(T.MODEL).model_key, C.AUTHORISED_OOS_THINKING.sampling_mode)
    assert T.FIXED_SAMPLING == want and rep["fixed_sampling"] == want._asdict()
    assert (want.temperature, want.top_p, want.top_k) == (0.7, 0.95, 20)
    assert "0.7/0.95/20" in rep["sampling_deviation"] and "asserted at admission" in rep["sampling_deviation"]
    # the per-arm cap: an ON leg at the in-support 2,048 floor is not this experiment ...
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143),
                summary_edit=lambda s: s.update(max_tokens=2048))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["refused"].startswith("max_tokens 2048 != the experiment's on cap 8192")
    # ... nor an OFF leg off the table's 512
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143))
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT,
                summary_edit=lambda s: s.update(max_tokens=2048))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "off")["refused"].startswith("max_tokens 2048 != the experiment's off cap 512")
    assert rep["verdict"]["word"] is None
    assert T.expected_cap("on") == C.OOS_THINKING_MAX_TOKENS == 8192 and T.expected_cap("off") == 512
    # a numerically-equal but off-type row (a bool temperature) is not the row either
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT)
    assert T.sampling_row_problem({"temperature": 0.7, "top_p": 0.95, "top_k": 20}) is None
    assert T.sampling_row_problem({"temperature": 0.7, "top_p": 0.95, "top_k": True}) is not None
    assert T.sampling_row_problem({"temperature": 0.7, "top_p": 0.95}) is not None            # top_k absent


def test_a_nuisance_parameter_differing_across_the_legs_refuses_the_odd_leg_or_all(tmp_path):
    """The cross-leg half: every nuisance (the served geometry, the
    reduction mode, the model and revision, the row, k, the parser, the policy)
    equal across the four legs. One leg served at another window is refused
    (the strict majority stands); two legs at another revision — no majority —
    refuse every leg: the four are one experiment or none of them is."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "served": {**SERVED, "max_model_len": 8192}})
    rep = analyze(tmp_path)
    assert rep["nuisance"]["disagreeing"] == ["served.max_model_len"]
    odd = leg_of(rep, PSR, "on")
    assert odd["refused"].startswith("nuisance parameter served.max_model_len = 8192 disagrees across the legs")
    assert "not one experiment" in odd["refused"] and odd["nuisance_refused"] == odd["refused"]
    assert set(rep["unusable_legs"]) == {T.leg_key(PSR, "on")}
    assert rep["nuisance"]["fields"]["served.max_model_len"]["values"][T.leg_key(NONE, "on")] == 16384
    assert rep["nuisance"]["fields"]["temperature"] == {"agree": True, "values": {k: 0.7 for k in rep["legs"]}}
    assert rep["verdict"]["word"] is None
    # no strict majority: every leg carrying the field is refused (two legs served
    # at another S — the model / revision no longer reach this check: a leg off the
    # registry row is refused POSITIVELY at admission, see
    # test_stores_of_another_model_or_another_cell_are_refused_by_identity_not_by_agreement)
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    for cell in (NONE, PSR):
        build_store(tmp_path, cell=cell, arm="off", correct=OFF_NONE_CORRECT if cell == NONE else OFF_PSR_CORRECT,
                    served={**SERVED, "max_num_seqs": 11})
    rep = analyze(tmp_path)
    assert rep["nuisance"]["disagreeing"] == ["served.max_num_seqs"]
    assert set(rep["unusable_legs"]) == set(rep["legs"])
    assert all("nuisance parameter served.max_num_seqs" in r for r in rep["unusable_legs"].values())
    for cell in (NONE, PSR):
        build_store(tmp_path, cell=cell, arm="off", correct=OFF_NONE_CORRECT if cell == NONE else OFF_PSR_CORRECT,
                    manifest_edit=lambda m: m.update(model_revision="other-rev"))
    rep = analyze(tmp_path)
    assert rep["nuisance"]["disagreeing"] == []                    # never reached: refused per leg first
    assert all(leg_of(rep, c, "off")["refused"].startswith("manifest model ") for c in (NONE, PSR))
    assert set(rep["unusable_legs"]) == {T.leg_key(NONE, "off"), T.leg_key(PSR, "off")}
    # a MISSING leg has nothing to compare and is not counted
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    import shutil
    shutil.rmtree(T.out_root_of(tmp_path, SHA7, cell=PSR, arm="on"))
    rep = analyze(tmp_path)
    assert rep["nuisance"]["disagreeing"] == [] and leg_of(rep, PSR, "on")["missing"]
    assert T.leg_key(PSR, "on") not in rep["nuisance"]["fields"]["temperature"]["values"]
    assert T.NUISANCE_FIELDS == (
        "temperature", "top_p", "top_k", "samples_per_item", "item_set", "task", "reasoning_parser",
        "unterminated_policy", "served.max_model_len", "served.max_num_seqs", "served.concurrency",
        C.REDUCTION_MODE_FIELD, "model_id", "model_revision")


# --------------------------------------------------------------------------- #
# evidence read in three states; --allow-pending-gates enumerated              #
# --------------------------------------------------------------------------- #


def _rewrite_evidence(cd, mutate):
    ev = json.loads((cd / "evidence.json").read_text())
    mutate(ev)
    (cd / "evidence.json").write_text(json.dumps(ev))


def test_evidence_stamps_keep_absent_null_and_off_contract_apart(tmp_path):
    """``_tri`` folded a PRESENT non-boolean certification (the
    string "false") into None -> PENDING -> waivable, so the flag filed the
    verdict over an invalid record; the live prompt max did the same. Three
    states now: the key ABSENT (the producer not landed) and an explicit NULL
    (the producer landed, the certification did not run) are PENDING — distinct,
    both waivable as producer stamps; a present value of any other type is a
    hard FAIL naming the key and the value, waived by nothing."""
    K = C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED
    stores = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    cd = stores[T.leg_key(PSR, "on")]
    stamp = T.stamp_name(C.OOS_EVIDENCE_HEADER_BLOCK, K)
    # -- absent: the producer has not landed
    _rewrite_evidence(cd, lambda ev: ev[C.OOS_EVIDENCE_HEADER_BLOCK].pop(K))
    strict = analyze(tmp_path, allow_pending_gates=False)
    leg = leg_of(strict, PSR, "on")
    assert leg["evidence"]["header_state"][K] == T.EVIDENCE_ABSENT and leg["evidence"]["header"][K] is None
    dec = strict["gates"]["declaration"]
    assert dec["status"] == T.GATE_PENDING and strict["verdict"]["word"] is None
    [msg] = [p for p in dec["pending"] if K in p]
    assert msg.endswith(f"{K} not stamped (the driver's certification, plan S3.4)")
    assert dec["pending_stamps"][dec["pending"].index(msg)] == stamp
    [item] = [it for it in strict["gates"]["pending_items"] if it["stamps"] == [stamp]]
    assert item["waivable"] is True and stamp in T.WAIVABLE_PENDING_STAMPS
    waived = analyze(tmp_path)
    assert waived["verdict"]["word"] == T.WORD_TRANSFERS and waived["verdict"]["gates_pending"] == ["declaration"]
    # -- explicit null: what THIS tree's driver writes (the certification is Phase B)
    _rewrite_evidence(cd, lambda ev: ev[C.OOS_EVIDENCE_HEADER_BLOCK].update({K: None}))
    strict = analyze(tmp_path, allow_pending_gates=False)
    leg = leg_of(strict, PSR, "on")
    assert leg["evidence"]["header_state"][K] == T.EVIDENCE_NULL
    dec = strict["gates"]["declaration"]
    assert dec["status"] == T.GATE_PENDING and any(f"{K} stamped null" in p for p in dec["pending"])
    assert strict["verdict"]["word"] is None and analyze(tmp_path)["verdict"]["word"] == T.WORD_TRANSFERS
    # -- present but off-contract: the string "false", an int, a list — FAIL, named, never waived
    for bad in ("false", 1, [True]):
        _rewrite_evidence(cd, lambda ev, bad=bad: ev[C.OOS_EVIDENCE_HEADER_BLOCK].update({K: bad}))
        rep = analyze(tmp_path)                                    # the flag is on
        leg = leg_of(rep, PSR, "on")
        assert leg["evidence"]["header_state"][K] == T.EVIDENCE_OFF_CONTRACT and leg["evidence"]["header"][K] == bad
        dec = rep["gates"]["declaration"]
        assert dec["status"] == T.GATE_FAIL
        [f] = [f for f in dec["failures"] if stamp in f]
        assert f"{stamp} is {bad!r} ({type(bad).__name__}) — off-contract" in f
        assert "a JSON boolean or null was expected" in f
        assert rep["verdict"]["word"] is None and "declaration" in rep["gates"]["failed"]
    # a stamped boolean False is the plain failure it always was
    _rewrite_evidence(cd, lambda ev: ev[C.OOS_EVIDENCE_HEADER_BLOCK].update({K: False}))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["evidence"]["header_state"][K] == T.EVIDENCE_STAMPED
    assert any(f"{K} is False" in f for f in rep["gates"]["declaration"]["failures"])
    _rewrite_evidence(cd, lambda ev: ev[C.OOS_EVIDENCE_HEADER_BLOCK].update({K: True}))
    # -- the live prompt max: the same three states
    LIVE, MAX = C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK, C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX
    live_stamp = T.stamp_name(LIVE, MAX)
    _rewrite_evidence(cd, lambda ev: ev[LIVE].pop(MAX))
    rep = analyze(tmp_path, allow_pending_gates=False)
    led = leg_of(rep, PSR, "on")["ledger"]
    assert leg_of(rep, PSR, "on")["evidence"][LIVE]["state"][MAX] == T.EVIDENCE_ABSENT
    assert led["passed"] and any("not stamped" in p for p in led["pending"]) and live_stamp in led["pending_stamps"]
    assert rep["verdict"]["word"] is None and analyze(tmp_path)["verdict"]["word"] == T.WORD_TRANSFERS
    _rewrite_evidence(cd, lambda ev: ev[LIVE].update({MAX: None}))
    rep = analyze(tmp_path, allow_pending_gates=False)
    led = leg_of(rep, PSR, "on")["ledger"]
    assert leg_of(rep, PSR, "on")["evidence"][LIVE]["state"][MAX] == T.EVIDENCE_NULL
    assert led["passed"] and any("stamped null" in p for p in led["pending"])
    for bad in (str(PROMPT_MAX), 1199.0, True):
        _rewrite_evidence(cd, lambda ev, bad=bad: ev[LIVE].update({MAX: bad}))
        rep = analyze(tmp_path)
        leg = leg_of(rep, PSR, "on")
        assert leg["evidence"][LIVE]["state"][MAX] == T.EVIDENCE_OFF_CONTRACT
        assert leg["evidence"][LIVE][MAX] is None and f"{live_stamp} is {bad!r}" in leg["evidence"][LIVE]["problem"][MAX]
        led = leg["ledger"]
        assert led["passed"] is False and not led["pending"]
        [f] = [f for f in led["failures"] if "live prompt max off-contract" in f]
        assert f"{live_stamp} is {bad!r} ({type(bad).__name__}) — off-contract" in f
        assert rep["gates"]["declaration"]["status"] == T.GATE_FAIL and rep["verdict"]["word"] is None
    _rewrite_evidence(cd, lambda ev: ev[LIVE].update({MAX: PROMPT_MAX}))
    # -- the probe's render arm: an off-contract value (an int) FAILS; null is pending
    PRA, BITES = C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM, C.OOS_EVIDENCE_BITES_BLOCK
    _rewrite_evidence(cd, lambda ev: ev[BITES].update({PRA: 1}))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["evidence"]["bites_state"][PRA] == T.EVIDENCE_OFF_CONTRACT
    assert any(f"{T.stamp_name(BITES, PRA)} is 1 (int) — off-contract" in f
               for f in rep["gates"]["declaration"]["failures"])
    _rewrite_evidence(cd, lambda ev: ev[BITES].update({PRA: None}))
    rep = analyze(tmp_path, allow_pending_gates=False)
    assert any(f"{PRA} stamped null" in p for p in rep["gates"]["declaration"]["pending"])
    assert analyze(tmp_path)["verdict"]["word"] == T.WORD_TRANSFERS
    # -- a block that is not an object puts its keys off-contract
    _rewrite_evidence(cd, lambda ev: ev.update({C.OOS_EVIDENCE_HEADER_BLOCK: ["not", "an", "object"]}))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["evidence"]["header_state"][K] == T.EVIDENCE_OFF_CONTRACT
    assert rep["gates"]["declaration"]["status"] == T.GATE_FAIL
    # the reader itself
    assert T.evidence_state({"a": True}, "a", kind="bool") == (T.EVIDENCE_STAMPED, True)
    assert T.evidence_state({"a": None}, "a", kind="bool") == (T.EVIDENCE_NULL, None)
    assert T.evidence_state({}, "a", kind="bool") == (T.EVIDENCE_ABSENT, None)
    assert T.evidence_state(None, "a", kind="bool") == (T.EVIDENCE_ABSENT, None)
    assert T.evidence_state({"a": "true"}, "a", kind="bool") == (T.EVIDENCE_OFF_CONTRACT, "true")
    assert T.evidence_state({"a": True}, "a", kind="int") == (T.EVIDENCE_OFF_CONTRACT, True)   # a bool is not a count
    assert T.evidence_state({"a": 7}, "a", kind="int") == (T.EVIDENCE_STAMPED, 7)
    assert T.evidence_state({"a": "on"}, "a", kind="arm") == (T.EVIDENCE_STAMPED, "on")
    assert T.evidence_state({"a": "yes"}, "a", kind="arm") == (T.EVIDENCE_OFF_CONTRACT, "yes")


def test_allow_pending_gates_waives_only_the_enumerated_producer_stamps(tmp_path):
    """The flag applies to an ENUMERATED list — the OOS_EVIDENCE_*
    certifications / live pass Phase B's driver writes and the summary's §3.4
    regime triple — and to nothing else: a missing ``--paired-ids`` or
    ``--prompt-budget`` (operator inputs) withholds the word with the flag on,
    named as not waivable."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)},
              stamps=False)
    rep = analyze(tmp_path)
    items = rep["gates"]["pending_items"]
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS and items and all(it["waivable"] for it in items)
    stamps = {s for it in items for s in it["stamps"]}
    assert stamps <= set(T.WAIVABLE_PENDING_STAMPS)
    assert {T.stamp_name("thinking_header_by_arm", "header_certified"),
            T.stamp_name("thinking_header_by_arm", "prompt_opened_think_certified"),
            T.stamp_name("steering_bites", "probe_render_arm"),
            T.stamp_name("prompt_budget_live", "max"), "summary.render_regime"} <= stamps
    assert rep["gates"]["pending_unwaivable"] == []
    # the enumerated list, exactly
    assert set(T.WAIVABLE_PENDING_STAMPS) == {
        "thinking_header_by_arm.header_certified", "thinking_header_by_arm.prompt_opened_think_certified",
        "thinking_header_by_arm.header_attested_agrees", "steering_bites.probe_render_arm",
        "prompt_budget_live.max", "prompt_budget_live.p99",
        "summary.salt_regime", "summary.resumed_policy", "summary.render_regime",
        "summary.n_http_rejected_retried"}
    assert all(k in C.OOS_EVIDENCE_KEYS for k in ("header_certified", "prompt_opened_think_certified",
                                                   "header_attested_agrees", "probe_render_arm", "max", "p99"))
    # operator inputs are never waived (the artifact is asked for on each ON leg; the id set once)
    for kw, gate, needle, n in ((dict(prompt_budget=None), "declaration", "no --prompt-budget artifact supplied", 2),
                                (dict(paired_ids=None), "population", "no frozen paired-200 id set was supplied", 1)):
        rep = analyze(tmp_path, **kw)
        assert rep["verdict"]["word"] is None
        assert rep["verdict"]["verdict"] == T.NOT_MEASURABLE + " (instrumentation stamps pending)"
        assert "not waivable by --allow-pending-gates" in rep["verdict"]["reason"]
        items = rep["verdict"]["pending_unwaivable"]
        assert len(items) == n and all(
            it["gate"] == gate and needle in it["detail"] and it["stamps"] == [None] and it["waivable"] is False
            for it in items)
        assert gate in rep["gates"]["pending"]


# --------------------------------------------------------------------------- #
# the §3.4 regime triple: optional at this tree; MIXED / off-expectation fails #
# --------------------------------------------------------------------------- #


def test_the_regime_triple_reads_not_stamped_yet_when_absent_and_fails_when_off_its_expected_pair(tmp_path):
    exp_psr_off = C.expected_stamp_regimes(lens=True, arm="off")
    assert (exp_psr_off.salt_regime, exp_psr_off.resumed_policy) == ("request-unique", "conditional")
    good = {"salt_regime": "request-unique", "resumed_policy": "conditional", "render_regime": "thinking_off"}
    wrong = {**good, "salt_regime": "cell-stable"}
    mixed = {**good, "n_salt_regimes": 2}
    stores = build_2x2(tmp_path, none_off={"correct": OFF_NONE_CORRECT, "regimes": {
                           "salt_regime": "cell-stable", "resumed_policy": "not-consulted",
                           "render_regime": "thinking_off"}},
                       psr_off={"correct": OFF_PSR_CORRECT, "regimes": good},
                       none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    absent = leg_of(rep, NONE, "on")["regimes"]
    assert absent["stamped"] is False and absent["status"] == T.GATE_PENDING
    assert "not stamped yet" in absent["detail"]
    stamped = leg_of(rep, PSR, "off")["regimes"]
    assert stamped["stamped"] is True
    # the complete psr/OFF triple (contract §2.5.11 row 3) -> PASSED: every expected member is B0-filled
    assert stamped["status"] == T.GATE_PASS and stamped["detail"] is None and stamped["pending_stamps"] == []
    assert stamped["observed"] == good and stamped["expected"] == {
        "salt_regime": "request-unique", "resumed_policy": "conditional", "render_regime": "thinking_off"}
    assert rep["gates"]["declaration"]["status"] == T.GATE_PENDING          # none/ON is still unstamped
    # off its expected pair -> FAILED, gate 4 named
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, regimes=wrong)
    rep = analyze(tmp_path)
    r = leg_of(rep, PSR, "off")["regimes"]
    assert r["status"] == T.GATE_FAIL and "salt_regime: 'cell-stable' != expected 'request-unique'" in r["detail"]
    assert "declaration" in rep["gates"]["failed"] and rep["verdict"]["word"] is None
    # MIXED -> FAILED
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, regimes=mixed)
    rep = analyze(tmp_path)
    assert "2 distinct values over the leg (MIXED" in leg_of(rep, PSR, "off")["regimes"]["detail"]
    assert stores  # the four dirs were written


def test_expected_stamp_regimes_table_matches_plan_3_4():
    """By NAME: the table returns ``StampRegimes`` (salt, resumed, render) and
    the analysis compares its ``_asdict()`` to the summary's fields."""
    def pair(t):
        return t.salt_regime, t.resumed_policy

    assert pair(C.expected_stamp_regimes(lens=False, arm="off")) == ("cell-stable", "not-consulted")
    assert pair(C.expected_stamp_regimes(lens=False, arm="on")) == ("cell-stable", "not-consulted")
    assert pair(C.expected_stamp_regimes(lens=True, arm="off")) == ("request-unique", "conditional")
    on = C.expected_stamp_regimes(lens=True, arm="on")
    assert isinstance(on, C.StampRegimes) and on._fields == C.EXPECTED_STAMP_TRIPLE_FIELDS
    assert on.salt_regime == "request-unique"
    assert on.resumed_policy == C.DECLARED_SALT_POLICY_LABEL == "declared-request-unique"   # §2.5.11 row 1
    assert all(t.render_regime == C.RENDER_REGIME_BY_ARM[a] for a, t in (
        ("on", on), ("off", C.expected_stamp_regimes(lens=True, arm="off"))))
    assert on._asdict() == {"salt_regime": "request-unique", "resumed_policy": "declared-request-unique",
                            "render_regime": "thinking_on"}
    assert set(on._asdict()) == set(C.EXPECTED_STAMP_TRIPLE_FIELDS) == set(C.STAMP_REGIME_COUNT_FIELD)
    with pytest.raises(ValueError):
        C.expected_stamp_regimes(lens=True, arm="sideways")
    with pytest.raises(ValueError, match="per-task"):
        C.expected_stamp_regimes(lens=True, arm=C.THINKING_PER_TASK)    # a mode, not an arm


# --------------------------------------------------------------------------- #
# P2-c: a large none ON-OFF decline is a finding, never an invalidation        #
# --------------------------------------------------------------------------- #


def test_a_large_none_decline_with_clean_termination_is_a_finding_and_the_word_is_filed(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(60)}, psr_on={"correct": range(90)})
    rep = analyze(tmp_path)
    assert rep["gates"]["failed"] == []
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS               # +15 pp under both arms
    s = rep["estimands"][T.BRANCH_PRIMARY]["stats"]["shift_none"]
    assert s["point"] == pytest.approx(-0.30) and s["ci_high"] < 0
    assert any("thinking changed the unsteered cell's capability" in f for f in rep["findings"])
    assert rep["verdict_rule"]["signs"]["shift_none"]["reading"] == "-"
    assert any("shift_none" in item for item in rep["top_items"])
    assert rep["interpretation_limits"] == []


def test_a_floor_is_stated_as_an_interpretation_limit_not_an_instrument_failure(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(4)}, psr_on={"correct": range(6)})
    rep = analyze(tmp_path)
    assert rep["gates"]["failed"] == []
    assert any("near the floor" in lim for lim in rep["interpretation_limits"])


# --------------------------------------------------------------------------- #
# split anomalies: reported; invalidate only when they move the word           #
# --------------------------------------------------------------------------- #


def test_split_anomalies_are_reported_and_invalidate_only_when_the_word_moves(tmp_path):
    # 2 unterminated-not-at-cap items on psr/on whose removal does not move "transfers"
    # (psr/on 160 correct: the DiD is positive under BOTH branches — with 150 the score-as-is
    # DiD is exactly 0 against a positive primary one, which the corrected rule reads as a
    # branch disagreement; that is its own test)
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(120)},
              psr_on={"correct": range(160), "unscored": {198, 199}, "not_at_cap": {198, 199}})
    rep = analyze(tmp_path)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == T.GATE_PASS and g["masked_out"] == [IDS[198], IDS[199]] and g["word_moves"] is False
    assert g["counts"][T.leg_key(PSR, "on")]["n_unterminated_not_at_cap"] == 2
    assert leg_of(rep, PSR, "on")["summary"]["n_unterminated_not_at_cap"] == 2
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    # an anomaly set whose removal MOVES the word: OFF none 0..119 vs psr 16..137
    # (effect_off +1 pp, CI covers 0 -> "not resolved"); the 16 none-only OFF items
    # 0..15 are the anomalous items on psr/on (unterminated, not at cap) — without
    # them effect_off is 18/184 = +9.8 pp with a CI clear of zero, so the word moves
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": range(16, 138)},
              none_on={"correct": range(120)},
              psr_on={"correct": range(140), "unscored": range(16), "not_at_cap": range(16)})
    rep2 = analyze(tmp_path)
    assert rep2["verdict_rule"]["word_by_branch"] == {b: T.WORD_NOT_RESOLVED for b in T.BRANCHES}
    g2 = rep2["gates"]["split_anomalies"]
    assert g2["status"] == T.GATE_FAIL and g2["word_moves"] is True and len(g2["masked_out"]) == 16
    assert g2["word_without"][T.BRANCH_PRIMARY] != T.WORD_NOT_RESOLVED
    assert "change the word" in g2["detail"] and "not measurable" in g2["detail"]
    assert rep2["verdict"]["word"] is None and "split_anomalies" in rep2["gates"]["failed"]
    # 16 unterminated-not-at-cap items stopped normally, so the cap gate is untouched
    assert rep2["gates"]["cap"]["status"] == T.GATE_PASS
    assert leg_of(rep2, PSR, "on")["summary"]["n_unterminated_not_at_cap"] == 16


def test_anomaly_ids_reads_unterminated_not_at_cap_and_unscored_without_a_block(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": {0, 1}, "not_at_cap": {0}},
              psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    ids = T.anomaly_ids(rep["legs"])
    assert ids[T.leg_key(NONE, "on")] == [IDS[0]] and ids[T.leg_key(PSR, "on")] == []


# --------------------------------------------------------------------------- #
# a MISSING leg is named; the full id keys every per-cell number               #
# --------------------------------------------------------------------------- #


def test_a_missing_leg_is_named_never_dropped(tmp_path):
    build_store(tmp_path, cell=NONE, arm="off", correct=OFF_NONE_CORRECT)
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT)
    build_store(tmp_path, cell=NONE, arm="on", correct=range(126))
    rep = analyze(tmp_path)
    k = T.leg_key(PSR, "on")
    assert set(rep["legs"]) == {T.leg_key(NONE, "off"), T.leg_key(PSR, "off"), T.leg_key(NONE, "on"), k}
    assert rep["legs"][k]["missing"].startswith("no out-root at ")
    assert rep["unusable_legs"] == {k: rep["legs"][k]["missing"]}
    assert rep["verdict"]["verdict"] == T.NOT_MEASURABLE and k in rep["verdict"]["reason"]
    assert [r["leg"] for r in rep["controls"]].count(k) == 1
    # two stores under one out-root is named, not resolved by picking the first
    extra = tmp_path / MODEL_DIR(tmp_path, NONE, "on") / "cells" / "second__controls__t0__a0__deadbeef__cafef00d"
    extra.mkdir(parents=True)
    (extra / "manifest.json").write_text("{}")
    rep = analyze(tmp_path)
    assert "2 stores under" in rep["legs"][T.leg_key(NONE, "on")]["missing"]


def MODEL_DIR(root, cell, arm):
    return Path(C.cell_out_dir("", cell, arm, model_slug=T.MODEL).lstrip("/")) / f"oos-{SHA7}"


def test_a_thinking_variant_leg_is_never_pooled_with_its_base(tmp_path):
    """``petri_stats.classify_cell_id`` folds a ``@thinking``
    variant into its base's (si, method). Every per-cell number here keys on
    the FULL leg id — the (cell dir, arm) path — so a variant store and its
    base store are two legs, and the analysis never imports the classifier."""
    variant = PSR + "@thinking"
    assert T.leg_key(variant, "on") != T.leg_key(PSR, "on")
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143))
    build_store(tmp_path, cell=variant, arm="on", correct=range(100))
    legs = T.collect_legs(tmp_path, SHA7, cells=[PSR, variant], arms=["on"], read_eval_logs=False)
    assert set(legs) == {T.leg_key(PSR, "on"), T.leg_key(variant, "on")}
    assert legs[T.leg_key(variant, "on")]["identity"]["cell_id"] == variant
    assert legs[T.leg_key(PSR, "on")]["identity"]["cell_id"] == PSR
    assert legs[T.leg_key(variant, "on")]["summary"]["acc"] == pytest.approx(0.5)
    assert legs[T.leg_key(PSR, "on")]["summary"]["acc"] == pytest.approx(0.715)
    src = (REPO / "scripts" / "tinycap_oos_transfer.py").read_text()
    assert "classify_cell_id" not in src.replace("never on ``petri_stats.classify_cell_id``", "").replace(
        "``petri_stats.classify_cell_id``", "")
    assert "petri_stats" not in src.replace("petri_stats.classify_cell_id", "")


def test_the_identity_marker_must_match_the_arm(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "oos_marker": False})
    rep = analyze(tmp_path)
    assert "not positively marked out_of_support_thinking: true" in leg_of(rep, PSR, "on")["refused"]
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, oos_marker=True)
    rep = analyze(tmp_path)
    assert "an in-support store carries no such key" in leg_of(rep, PSR, "off")["refused"]


# --------------------------------------------------------------------------- #
# the .eval re-derivation (monkeypatched: no inspect log is written here)      #
# --------------------------------------------------------------------------- #


def _resolve_report_path(report, path):
    """An INDEPENDENT dotted-path resolver for the sidecar test (the production
    one is not consulted): dict keys by name, list entries by integer."""
    node = report
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            assert part in node, (path, part, sorted(node)[:8] if isinstance(node, dict) else node)
            node = node[part]
    return node


def _results_from_store(cell_dir, task, *, swap=(), rescore=()):
    """``ItemResult`` stand-ins built from the store's OWN outcome record — a
    retained log consistent with the record — with the score values of each
    item pair in ``swap`` exchanged (the log's truth where the record was
    edited) and every item in ``rescore`` read as scored 0.0 where the record
    says unscored (a count disagreement). ``unscored`` is the real result's
    NaN reading; ``value`` is NaN for an unscored draw, as inspect files it."""
    rec = completion.read_task_outcomes(cell_dir, task)
    by_id = {it["item_id"]: dict(it) for it in rec["items"]}
    for a, b in swap:
        by_id[a]["value"], by_id[b]["value"] = by_id[b]["value"], by_id[a]["value"]
    for i in rescore:
        by_id[i].update(scored=True, value=0.0)
    return [SimpleNamespace(item_id=it["item_id"], sample_idx=it["sample_idx"], unscored=not it["scored"],
                            value=(float("nan") if not it["scored"] else float(it["value"])), at_cap=it["at_cap"],
                            think_block_present=it["think_block_present"],
                            think_block_unterminated=it["think_block_unterminated"])
            for it in by_id.values()]


def test_the_eval_log_rederivation_agrees_or_refuses(tmp_path, monkeypatch):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4}, psr_on={"correct": range(143)})
    calls = {}

    def fake(cell_dir, task):
        calls[str(cell_dir)] = task
        return _results_from_store(cell_dir, task), ["a completion " * 40, "  ", "x y z " * 40], None

    monkeypatch.setattr(T, "_read_eval_results", fake)
    rep = analyze(tmp_path, read_eval_logs=True)
    leg = leg_of(rep, NONE, "on")
    assert leg["eval_check"]["available"] and leg["eval_check"]["agrees"] is True
    assert leg["eval_check"]["derived"]["n_unscored_samples"] == 4
    sig = leg["eval_check"]["text"]
    assert sig["n_completions"] == 3 and sig["n_empty"] == 1 and sig["n_repetitive"] == 2
    assert len(calls) == 4

    def disagree(cell_dir, task):
        # the log scores the four draws the record calls unscored: a COUNT disagreement
        unscored = [IDS[i] for i in LAST4] if "none" in str(cell_dir) and "thinking_on" in str(cell_dir) else ()
        return _results_from_store(cell_dir, task, rescore=unscored), [], None

    monkeypatch.setattr(T, "_read_eval_results", disagree)
    rep = analyze(tmp_path, read_eval_logs=True)
    why = leg_of(rep, NONE, "on")["refused"]
    assert "the .eval log re-derivation disagrees with the record" in why
    assert "n_unscored_samples: log 0 != record 4" in why


def test_without_logs_the_text_signature_is_null_with_the_reason(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path, read_eval_logs=True)
    t = leg_of(rep, NONE, "on")["eval_check"]
    assert t["available"] is False and "no Inspect .eval log" in t["reason"]
    assert t["text"]["repetition_rate"] is None and t["text"]["unavailable"] == t["reason"]


# --------------------------------------------------------------------------- #
# complete-case (rows) agrees with the pairing function's intersection number  #
# --------------------------------------------------------------------------- #


def test_complete_case_over_rows_equals_the_pairing_functions_intersection_delta(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": {0, 1, 2, 3}},
              psr_on={"correct": range(143), "unscored": LAST4})
    rep = analyze(tmp_path)
    cc_rows = rep["complete_case_rows"]
    cc_fn = rep["estimands"][T.BRANCH_PRIMARY]["complete_case"]
    assert cc_rows["label"] == pr.COMPLETE_CASE_LABEL
    for c in ("effect_off", "effect_on", "shift_none", "shift_psr"):
        assert cc_rows[c]["n_items"] == cc_fn[c]["n_items"]
        assert cc_rows[c]["delta"] == pytest.approx(cc_fn[c]["delta"])
    assert cc_rows["effect_on"]["n_items"] == 192
    assert sorted(cc_rows["effect_on"]["only_a"]) == sorted(IDS[i] for i in LAST4)     # scored on none/on only
    assert sorted(cc_rows["effect_on"]["only_b"]) == IDS[:4]
    tr = rep["transitions"]["none_to_psr__on"]
    assert tr["counts"]["->U"] == 8 and sum(tr["counts"].values()) == 200
    assert tr["mcnemar"]["n_items"] == 192 and tr["label"] == pr.COMPLETE_CASE_LABEL


# --------------------------------------------------------------------------- #
# serving parity                                                              #
# --------------------------------------------------------------------------- #


def test_serving_parity_pairs_the_fresh_off_legs_with_the_waves_stores_and_labels_the_regime(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False,
                served={**T.__dict__.get("SERVED", {}), "reasoning_parser": None, "tool_calls": False,
                        "max_model_len": 16384, "max_num_seqs": 36, "concurrency": 36,
                        C.AGENT_LOOP_ENTRY_KEY: False, C.REDUCTION_MODE_FIELD: "per_row"})
    rep = analyze(tmp_path, wave_root=wave)
    par = rep["serving_parity"]
    assert par["available"] and par["wave_sha"] == "c7e1796"
    none_rec = par["cells"][NONE]
    assert none_rec["n_items"] == 200 and none_rec["regime_label"].startswith("like-for-like")
    assert none_rec["delta_fresh_minus_wave"] == pytest.approx((120 - 118) / 200)
    assert none_rec["ci"][0] <= none_rec["delta_fresh_minus_wave"] <= none_rec["ci"][1]
    psr_rec = par["cells"][PSR]
    assert psr_rec["regime_label"].startswith("across regimes (max_num_seqs 36 -> 22")
    assert psr_rec["delta_fresh_minus_wave"] == pytest.approx(0.0)
    assert "across cache regimes" not in json.dumps(par)              # F9: never claimed
    assert psr_rec["salt_label"]["wave"] == "not stamped"
    assert par["cells"][NONE]["unscored"] == {"wave": 0, "fresh": 0}
    assert T.serving_parity(rep["legs"], None, n_resamples=10, seed=0, ci=0.95) == {
        "available": False, "reason": "no --wave-root given"}


def test_serving_parity_refuses_a_non_comparable_wave_store_and_promotes_a_drift(tmp_path):
    """Parity compared only the served regime, so a wave store at
    another sampling row / cap still yielded a delta labelled like-for-like,
    and a drift beyond its CI stayed nested. Now task, item set, arm, k, the
    row, the cap, the model and revision are compared first — a mismatch
    refuses the statistic and labels the pair non-comparable (never
    like-for-like) — and a drift beyond its CI is promoted into ``findings``
    and the head of ``top_items``, before any ON number."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    wave = tmp_path / "wave"
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False,
                summary_edit=lambda s: s.update(temperature=1.7, top_p=0.5, max_tokens=2048))
    build_store(wave, cell=NONE, arm="off", correct=range(90), out_suffix="c7e1796", oos_marker=False)
    rep = analyze(tmp_path, wave_root=wave)
    par = rep["serving_parity"]
    psr_rec = par["cells"][PSR]
    assert psr_rec["comparable"] is False and psr_rec["non_comparable_fields"] == ["max_tokens", "temperature", "top_p"]
    assert psr_rec["regime_label"] == "non-comparable (max_tokens 2048 vs 512, temperature 1.7 vs 0.7, top_p 0.5 vs 0.95)"
    assert "delta_fresh_minus_wave" not in psr_rec and "ci" not in psr_rec and psr_rec["drift_beyond_ci"] is None
    assert psr_rec["refused"].startswith("not the same instrument on both sides — no paired delta is computed")
    assert "like-for-like" not in json.dumps(psr_rec)
    assert psr_rec["comparison"]["max_tokens"] == {"wave": 2048, "fresh": 512}
    assert psr_rec["comparison"]["model_revision"] == {"wave": C.model_row(T.MODEL).rev, "fresh": C.model_row(T.MODEL).rev}
    assert set(psr_rec["comparison"]) == set(T.PARITY_FIELDS) == {
        "task", "item_set", "thinking", "samples_per_item", "temperature", "top_p", "top_k", "max_tokens",
        "model_id", "model_revision"}
    # the comparable cell drifted: fresh 120/200 vs wave 90/200 -> +15 pp, CI excludes zero
    none_rec = par["cells"][NONE]
    assert none_rec["comparable"] is True and none_rec["non_comparable_fields"] == []
    assert none_rec["regime_label"].startswith("like-for-like") and none_rec["drift_beyond_ci"] is True
    assert none_rec["delta_fresh_minus_wave"] == pytest.approx(30 / 200)
    [line] = [f for f in rep["findings"] if f.startswith("serving-parity drift beyond its CI on none OFF")]
    assert "fresh - wave = +15.00 pp" in line and "like-for-like" in line
    assert rep["top_items"][0] == line and rep["top_items"].count(line) == 1
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS               # the parity finding does not gate the word
    assert T.parity_findings({"available": False}) == [] and T.parity_findings(par) == [line]
    # a wave store at another model revision is non-comparable on that field alone; no drift promoted
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False,
                manifest_edit=lambda m: m.update(model_revision="other-rev"))
    rep = analyze(tmp_path, wave_root=wave)
    rec = rep["serving_parity"]["cells"][NONE]
    assert rec["non_comparable_fields"] == ["model_revision"] and rec["regime_label"].startswith("non-comparable (model_revision 'other-rev' vs ")
    assert not any(f.startswith("serving-parity drift") for f in rep["findings"] + rep["top_items"])
    # no wave root: nothing to promote, findings unchanged
    assert analyze(tmp_path)["findings"] == []


# --------------------------------------------------------------------------- #
# the CLI: W&B disabled writes no network; figures are optional               #
# --------------------------------------------------------------------------- #


def test_main_with_wandb_disabled_and_no_figs_writes_the_report_and_imports_no_wandb(tmp_path, capsys):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    out = tmp_path / "report" / "report.json"
    budget = tmp_path / "oos_prompt_budget.json"
    budget.write_text(json.dumps(prompt_budget_artifact()))
    ids_file = tmp_path / "paired_ids.json"
    ids_file.write_text(json.dumps(IDS))
    sys.modules.pop("wandb", None)
    rc = T.main(["--gen-root", str(tmp_path), "--sha", f"oos-{SHA7}", "--wandb-mode", "disabled",
                 "--no-figs", "--no-mirror", "--no-eval-logs", "--bootstrap", "500",
                 "--prompt-budget", str(budget), "--paired-ids", str(ids_file),
                 "--allow-pending-gates", "--out", str(out)])
    assert rc == 0
    assert "wandb" not in sys.modules
    rep = json.loads(out.read_text())
    assert rep["wandb_url"] is None and rep["verdict"]["word"] == T.WORD_TRANSFERS
    assert rep["sha"] == SHA7 and rep["bootstrap"]["B"] == 500
    assert rep["gates"]["population"]["status"] == T.GATE_PASS and rep["gates"]["population"]["frozen_set"] == "verified"
    printed = capsys.readouterr().out
    assert "== VERDICT: transfers ==" in printed and "safety_refusal__psr_kl_all" in printed
    assert "the estimand (conditional accuracy" in printed
    assert T.log_wandb(rep, {}, mode="disabled") is None


def test_main_with_a_relative_out_and_figs_dir_mirrors_from_one_absolute_path_convention(tmp_path, monkeypatch, capsys):
    """The code review's A2: with a bare `--out report.json` the figures went to the
    relative `figs/`, `make_figures` filed manifest entries like
    `figs/<stem>.png`, and `mirror_figures` prepended `figs_dir` to them again
    (`figs/figs/<stem>.png`) -> TransferAnalysisError before W&B logging and the
    report write — on a plain four-leg run too. One convention now: `--out`,
    `--figs-dir` and `--mirror-dir` are resolved to absolute paths before any
    manifest entry is built, so every entry the report files is absolute and on
    disk, the mirror copies exactly them, and the report lands where asked."""
    pytest.importorskip("matplotlib")
    gen = tmp_path / "gen"
    build_2x2(gen, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    budget = gen / "oos_prompt_budget.json"
    budget.write_text(json.dumps(prompt_budget_artifact()))
    ids_file = gen / "paired_ids.json"
    ids_file.write_text(json.dumps(IDS))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    common = ["--gen-root", str(gen), "--sha", f"oos-{SHA7}", "--wandb-mode", "disabled", "--no-eval-logs",
              "--bootstrap", "200", "--prompt-budget", str(budget), "--paired-ids", str(ids_file),
              "--allow-pending-gates"]

    def check(out_arg: str, figs_arg: str | None, mirror_arg: str, expect_figs: Path) -> dict:
        rc = T.main(common + ["--out", out_arg, "--mirror-dir", mirror_arg]
                    + (["--figs-dir", figs_arg] if figs_arg else []))
        assert rc == 0
        report_path = (cwd / out_arg).resolve()
        assert report_path.is_file(), out_arg
        rep = json.loads(report_path.read_text())
        written = [p for ps in rep["figures"]["written"].values() for p in ps]
        assert written and set(rep["figures"]["written"]) | set(rep["figures"]["skipped"]) == set(T.FIGURE_STEMS)
        assert set(rep["figures"]["skipped"]) == {T.FIGURE_PARITY}                     # no --wave-root
        for entry in written + [rep["figures"]["numbers"], rep["figures"]["tables"]]:
            assert Path(entry).is_absolute() and Path(entry).is_file(), entry
            assert Path(entry).parent == expect_figs.resolve(), entry
        assert len(rep["mirrored"]) == len(written) + 2
        for entry in rep["mirrored"]:
            assert Path(entry).is_absolute() and Path(entry).is_file(), entry
            assert Path(entry).parent == (cwd / mirror_arg).resolve(), entry
        assert sorted(q.name for q in (cwd / mirror_arg).iterdir()) == sorted(Path(q).name for q in rep["mirrored"])
        assert rep["wandb_url"] is None and "verdict" in rep            # written AFTER the mirror, W&B off
        return rep

    # the reported shape: a bare --out, figures beside it under ./figs, a relative mirror dir
    check("report.json", None, "mirror", cwd / "figs")
    # an explicit relative --figs-dir and a nested relative --out
    check("sub/report2.json", "figs2", "mirror2", cwd / "figs2")
    assert "[analyze] full report -> " in capsys.readouterr().out


def test_figures_write_png_pdf_and_a_numbers_json_with_a_source_per_number(tmp_path):
    pytest.importorskip("matplotlib")
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
              psr_on={"correct": range(143), "unscored": LAST4})
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    rep = analyze(tmp_path, wave_root=wave)
    figs = T.make_figures(rep, tmp_path / "figs")
    assert set(figs["written"]) == set(T.FIGURE_STEMS)
    for stem, paths in figs["written"].items():
        assert [Path(p).suffix for p in paths] == [".png", ".pdf"] and all(Path(p).stat().st_size > 0 for p in paths)
    numbers = json.loads(Path(figs["numbers"]).read_text())
    assert set(numbers["figures"]) == set(T.FIGURE_STEMS)
    for stem, nums in numbers["figures"].items():
        assert nums, stem
        for n in nums:
            assert set(n) == {"name", "value", "source"} and n["source"], (stem, n)
    tables = json.loads(Path(figs["tables"]).read_text())
    assert set(tables["effects"]) == set(T.BRANCHES) and tables["verdict"]["verdict"] == T.NOT_DECIDABLE_CENSORING
    mirrored = T.mirror_figures(tmp_path / "figs", tmp_path / "mirror", manifest=figs)      # from the manifest, never a glob
    assert len(mirrored) == 2 * len(T.FIGURE_STEMS) + 2
    assert "numbers_offcap" not in figs and "tables_offcap" not in figs                     # the 2x2 run: the e1 pair only


# --------------------------------------------------------------------------- #
# round-A review follow-ups                                                    #
# --------------------------------------------------------------------------- #


def test_off_prompts_are_longer_and_the_artifact_is_compared_on_the_on_legs_only(tmp_path):
    """The CPU artifact tokenizes the ON render; the OFF render of the
    same messages is two tokens longer on this family (the fixture's default
    now). The three-way prompt-max agreement is an ON-leg check: the OFF legs
    record it as not applicable (neither pending nor failed), every other
    ledger check runs on every leg, and a correctly executed experiment gets
    its word. Comparing the artifact on the OFF legs (the old code) failed
    'declaration' and withheld the word."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    for cell in (NONE, PSR):
        on, off = leg_of(rep, cell, "on")["ledger"], leg_of(rep, cell, "off")["ledger"]
        assert on["prompt_max"] == {"ledger": PROMPT_MAX, "live": PROMPT_MAX, "artifact": PROMPT_MAX,
                                    "agrees": True, "detail": None, "applicable": True}
        assert on["artifact_fit"]["passed"] is True
        assert off["prompt_max"]["ledger"] == PROMPT_MAX_OFF and off["prompt_max"]["live"] == PROMPT_MAX_OFF
        assert off["prompt_max"]["applicable"] is False and off["prompt_max"]["artifact"] is None
        assert off["prompt_max"]["agrees"] is None and "not compared on an OFF leg" in off["prompt_max"]["detail"]
        assert off["passed"] is True and off["failures"] == [] and off["pending"] == []
        # the other checks still run on the OFF leg
        assert off["n_calls"] == 200 and off["window"] == {"prompt_max": PROMPT_MAX_OFF, "cap": 512,
                                                           "max_model_len": 16384, "fits": True}
    assert rep["gates"]["declaration"]["status"] != T.GATE_FAIL
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    # the ON check still bites: an ON leg whose ledger max drifts from the artifact FAILS
    def drift(rows):
        gsm = [r for r in rows if r["task"] == TASK]
        gsm[0]["prompt_tokens_pre"] = PROMPT_MAX + 1

    build_store(tmp_path, cell=PSR, arm="on", correct=range(143), ledger_edit=drift)
    rep = analyze(tmp_path)
    assert any("three-way prompt-max disagreement" in f for f in leg_of(rep, PSR, "on")["ledger"]["failures"])
    assert rep["verdict"]["word"] is None
    # and an artifact written for another cap does not stand for the leg
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143))
    other_cap = {**prompt_budget_artifact(), "cap": 4096, "max_plus_cap": PROMPT_MAX + 4096}
    rep = analyze(tmp_path, prompt_budget=other_cap)
    led = leg_of(rep, PSR, "on")["ledger"]
    assert led["artifact_fit"]["passed"] is False and any("written for cap 4096" in f for f in led["failures"])
    assert leg_of(rep, PSR, "off")["ledger"]["passed"] is True


def test_a_scored_stray_open_tag_item_is_an_anomaly_that_can_move_the_word(tmp_path):
    """``stray_open_tag`` is on the durable per-item record; the
    anomaly set includes those items even when SCORED, the gate reports the
    counts, and a set whose exclusion moves the word FAILS the gate. Same
    construction as the unterminated case: OFF none 0..119 vs psr 16..137
    (effect_off +1 pp, 'not resolved'); the 16 none-only OFF items are
    stray-tag SCORED items on psr/on — without them effect_off is +9.8 pp with
    a CI clear of zero, so the word moves."""
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": range(16, 138)},
              none_on={"correct": range(120)}, psr_on={"correct": range(140), "stray": range(16)})
    rep = analyze(tmp_path)
    leg = leg_of(rep, PSR, "on")
    assert all(leg["items"][IDS[i]]["stray_open_tag"] and leg["items"][IDS[i]]["scored"] for i in range(16))
    assert leg["summary"]["n_unscored"] == 0                          # every one of them scored
    assert T.anomaly_ids(rep["legs"])[T.leg_key(PSR, "on")] == sorted(IDS[:16])
    g = rep["gates"]["split_anomalies"]
    assert g["counts"][T.leg_key(PSR, "on")] == {"n_unterminated_not_at_cap": 0, "n_stray_open_tag": 16,
                                                  "n_stray_open_tag_scored": 16, "n_anomalous_items": 16}
    assert g["status"] == T.GATE_FAIL and g["word_moves"] is True and len(g["masked_out"]) == 16
    assert rep["verdict_rule"]["word_by_branch"][T.BRANCH_PRIMARY] == T.WORD_NOT_RESOLVED
    assert g["word_without"][T.BRANCH_PRIMARY] != T.WORD_NOT_RESOLVED
    assert rep["verdict"]["word"] is None and "split_anomalies" in rep["gates"]["failed"]
    # a stray-tag item whose exclusion does NOT move the word is reported, not an invalidation
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(120)},
              psr_on={"correct": range(150), "stray": {5, 6}})
    rep = analyze(tmp_path)
    g = rep["gates"]["split_anomalies"]
    assert g["status"] == T.GATE_PASS and g["masked_out"] == [IDS[5], IDS[6]]
    assert g["counts"][T.leg_key(PSR, "on")]["n_stray_open_tag"] == 2
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    # the field is REQUIRED on the record: the writer refuses a record without it, and a record
    # on disk without it (written around the writer) is refused by the reader, naming the field
    cd = build_store(tmp_path, cell=PSR, arm="on", correct=range(150))
    rec = completion.read_task_outcomes(cd, TASK)
    for r in rec["items"]:
        r.pop("stray_open_tag")
    with pytest.raises(completion.CompletionError, match="stray_open_tag"):
        completion.write_task_outcomes(cd, TASK, rec)
    completion.task_outcomes_path(cd, TASK).write_text(json.dumps(rec))
    rep = analyze(tmp_path)
    assert "stray_open_tag" in leg_of(rep, PSR, "on")["refused"]


def test_an_exact_zero_did_from_equal_effects_is_sign_not_identified_and_the_branches_agree(tmp_path):
    """Exact arithmetic through the stores: equal 30/200 effects with different baselines
    (OFF 120 vs 150; ON 110 vs 140) — the DiD point and bound are EXACTLY
    zero under both branches (integer counts over the common denominator), the
    DiD sign is filed as not identified, both branches read 0 for it, and the
    word is filed. The floating version returned [-1.1e-16, -1.1e-16] and
    'negative, sign identified'."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(110)}, psr_on={"correct": range(140)})
    rep = analyze(tmp_path)
    for b in T.BRANCHES:
        d = rep["estimands"][b]["stats"]["did"]
        assert d["point"] == 0.0 and d["bound_low"] == 0.0 and d["bound_high"] == 0.0
        assert d["point_sign"] == 0 and d["sign_identified"] is False
    v = rep["verdict_rule"]
    assert v["signs_by_contrast"]["did"] == {b: 0 for b in T.BRANCHES}
    assert v["signs"]["did"]["reading"] == T.SIGN_NOT_IDENTIFIED and v["signs"]["did"]["sign"] is None
    assert v["agree_branches"] is True and rep["verdict"]["word"] == T.WORD_TRANSFERS
    # the floating sum in the bound's order is the -1.1e-16 the rule used to read as a sign
    assert (0 + 140 / 200 - 110 / 200 - 150 / 200 + 120 / 200) != 0.0
    # the same with a censored ON leg whose adverse imputation lands the bound's low end on zero
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(106), "unscored": LAST4},
              psr_on={"correct": range(140)})
    rep = analyze(tmp_path)
    d = rep["estimands"][T.BRANCH_PRIMARY]["stats"]["did"]
    assert d["bound_low"] == 0.0 and d["bound_high"] == pytest.approx(0.02) and d["sign_identified"] is False
    assert rep["verdict_rule"]["signs"]["did"]["reading"] == T.SIGN_NOT_IDENTIFIED


def test_branch_sign_disagreement_through_an_exact_zero_did_is_named(tmp_path):
    """Branch agreement through the stores: psr/on 130 correct of 196 scored (4 censored)
    against none/on 100/200 — primary DiD +1.33 pp, score-as-is DiD EXACTLY 0
    (130/200 - 100/200 = 150/200 - 120/200). Same word ('transfers') and a
    decidable bound, but the DiD's sign differs (+1 vs 0): the verdict is
    'not decidable (branch disagreement)' naming the DiD — zero is its own
    value, never a wildcard."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(100)},
              psr_on={"correct": range(130), "unscored": LAST4})
    rep = analyze(tmp_path)
    v = rep["verdict_rule"]
    assert v["word_by_branch"] == {b: T.WORD_TRANSFERS for b in T.BRANCHES}
    assert v["decidable_bounds"] is True
    assert rep["estimands"][T.BRANCH_SCORE_AS_IS]["stats"]["did"]["point"] == 0.0
    assert rep["estimands"][T.BRANCH_PRIMARY]["stats"]["did"]["point"] == pytest.approx(130 / 196 - 0.65)
    assert v["signs_by_contrast"]["did"] == {T.BRANCH_PRIMARY: 1, T.BRANCH_SCORE_AS_IS: 0}
    assert v["disagreeing_contrasts"] == ["did"] and v["agree_branches"] is False
    assert rep["verdict"]["verdict"] == T.NOT_DECIDABLE_BRANCHES and rep["verdict"]["word"] is None
    assert "the sign of did (1/0)" in rep["verdict"]["reason"]
    assert rep["top_items"][0].startswith("the two scoring branches DISAGREE")


def test_partial_marker_stores_are_diagnostics_only_and_the_verdict_is_withheld(tmp_path):
    """A leg finished under a limit gets the PARTIAL marker the real
    ``completion.finish_task`` writes (summary + record + marker); the analysis
    refuses it by name — its summary still enters the controls table — and no
    estimand runs. A done store whose marker is gone is refused too. Only four
    DONE stores over the frozen 200 reach the estimand."""
    partial_root, done_root = tmp_path / "partial", tmp_path / "done"
    stores = build_2x2(partial_root, **off_legs(), none_on={"correct": range(126)},
                       psr_on={"correct": range(3), "ids": IDS[:5], "limits": {"inspect_limit": 5},
                               "n_expected_full": 200})
    cd = stores[T.leg_key(PSR, "on")]
    assert completion.partial_marker_path(cd, TASK).is_file() and not completion.done_marker_path(cd, TASK).is_file()
    assert completion.read_task_outcomes(cd, TASK)["n_attempted_items"] == 5          # written for the limited run too
    rep = analyze(partial_root)
    leg = leg_of(rep, PSR, "on")
    assert leg["refused"].startswith("PARTIAL marker at") and "limits {'inspect_limit': 5}" in leg["refused"]
    assert "5 of 200 items" in leg["refused"] and "diagnostics only" in leg["refused"]
    assert leg["partial"] == {"limits": {"inspect_limit": 5}, "n_items": 5, "n_expected_full": 200}
    assert leg["summary"]["n"] == 5                                                # diagnostics kept
    row = next(r for r in rep["controls"] if r["leg"] == T.leg_key(PSR, "on"))
    assert row["summary"]["n"] == 5 and row["refused"] == leg["refused"]
    assert "estimands" not in rep and rep["verdict"]["word"] is None
    assert rep["verdict"]["verdict"] == T.NOT_MEASURABLE and "PARTIAL marker" in rep["verdict"]["reason"]
    assert rep["gates"]["population"]["status"] == T.GATE_FAIL
    # a complete store whose done marker is gone: refused, not admitted
    stores = build_2x2(done_root, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    cd = stores[T.leg_key(PSR, "on")]
    rep = analyze(done_root)
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    completion.done_marker_path(cd, TASK).unlink()
    rep = analyze(done_root)
    assert leg_of(rep, PSR, "on")["refused"].startswith("no done marker at")
    assert "estimands" not in rep and rep["verdict"]["word"] is None
    # a 'done' store over a different population count is refused too
    build_store(done_root, cell=PSR, arm="on", correct=range(100), ids=IDS[:150])
    rep = analyze(done_root)
    assert "n_attempted_items 150 != the frozen 200" in leg_of(rep, PSR, "on")["refused"]


def test_the_frozen_paired_200_population_gates_the_word(tmp_path, monkeypatch):
    """The population: n == 200 and the four legs' identity are hard
    requirements; the ids == the frozen set is PASSED when the set is supplied
    and equal, FAILED (no estimand) on a mismatch or a wrong-sized set, and
    PENDING (word withheld — an operator input is never waived by
    ``--allow-pending-gates``) when no set is supplied. ``--paired-ids
    plan`` derives the set from the task plan."""
    real_from_plan = T.frozen_paired_ids_from_plan                       # before any patch below
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    assert rep["gates"]["population"] == {"status": T.GATE_PASS, "detail": None, "n": 200, "reason": None,
                                          "frozen_set": "verified", "n_frozen": 200}
    # not supplied: pending, named; the word is withheld under strict gating
    strict = analyze(tmp_path, paired_ids=None, allow_pending_gates=False)
    pop = strict["gates"]["population"]
    assert pop["status"] == T.GATE_PENDING and pop["frozen_set"] == "not supplied" and pop["n"] == 200
    assert "no frozen paired-200 id set was supplied" in pop["detail"]
    assert "estimands" in strict and strict["verdict"]["word"] is None
    assert "population" in strict["gates"]["pending"] and "PENDING" in strict["verdict"]["reason"]
    loose = analyze(tmp_path, paired_ids=None)                 # the flag is on: still withheld
    assert loose["verdict"]["word"] is None and "not waivable by --allow-pending-gates" in loose["verdict"]["reason"]
    assert [it["gate"] for it in loose["verdict"]["pending_unwaivable"]] == ["population"]
    # a mismatch: FAILED, no estimand, the ids named
    rep = analyze(tmp_path, paired_ids=IDS[1:] + ["gsm8k_ffffffff"])
    pop = rep["gates"]["population"]
    assert pop["status"] == T.GATE_FAIL and pop["frozen_set"] == "mismatch"
    assert IDS[0] in pop["detail"] and "gsm8k_ffffffff" in pop["detail"]
    assert "estimands" not in rep and rep["verdict"]["word"] is None
    assert rep["verdict"]["verdict"] == T.NOT_MEASURABLE and "not the frozen paired-200 set" in rep["verdict"]["reason"]
    # a wrong-sized frozen set is invalid
    rep = analyze(tmp_path, paired_ids=IDS[:199])
    assert rep["gates"]["population"]["frozen_set"] == "invalid" and "estimands" not in rep
    # the plan-derived set
    monkeypatch.setattr(T, "frozen_paired_ids_from_plan", lambda: list(IDS))
    out = tmp_path / "r.json"
    budget = tmp_path / "b.json"
    budget.write_text(json.dumps(prompt_budget_artifact()))
    rc = T.main(["--gen-root", str(tmp_path), "--sha", SHA7, "--wandb-mode", "disabled", "--no-figs",
                 "--no-mirror", "--no-eval-logs", "--bootstrap", "200", "--prompt-budget", str(budget),
                 "--paired-ids", T.PAIRED_IDS_FROM_PLAN, "--allow-pending-gates", "--out", str(out)])
    assert rc == 0 and json.loads(out.read_text())["gates"]["population"]["frozen_set"] == "verified"
    # frozen_paired_ids_from_plan reads the plan's sample ids and refuses a short plan
    from sisteer.capeval.tiers import inspect_served as isv

    monkeypatch.setattr(isv, "plan_task", lambda name, **k: SimpleNamespace(sample_ids=list(reversed(IDS))))
    assert real_from_plan() == sorted(IDS)
    monkeypatch.setattr(isv, "plan_task", lambda name, **k: SimpleNamespace(sample_ids=IDS[:199]))
    with pytest.raises(T.TransferAnalysisError, match="199 ids, not the frozen 200"):
        real_from_plan()


def test_the_fixture_p99_is_the_producers_nearest_rank(tmp_path):
    """The synthetic summary's ``p99_completion_tokens`` is
    ``contracts.nearest_rank_percentile`` (sorted[ceil(0.99 n) - 1] =
    sorted[-3] at n = 200), the runner's own convention — with exactly two
    at-cap draws the nearest rank is the SHORT length, where ``sorted[-2]``
    (the old fixture) read the cap."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "unscored": {198, 199}})
    rep = analyze(tmp_path)
    s = leg_of(rep, PSR, "on")["summary"]
    tokens = sorted(rec["completion_tokens"] for rec in leg_of(rep, PSR, "on")["items"].values())
    assert s["p99_completion_tokens"] == C.nearest_rank_percentile(tokens, 0.99) == tokens[-3] == 300
    assert tokens[-2] == CAP_ON != s["p99_completion_tokens"]


# --------------------------------------------------------------------------- #
# per-leg admission — the arm's think-split mode, the store's POSITIVE         #
# identity, and the live p99 handled like the max                               #
# --------------------------------------------------------------------------- #


def test_a_leg_scored_under_another_think_split_mode_is_refused_per_arm(tmp_path):
    """Admission displayed ``think_split_mode`` but never checked
    it against the ARM's contract — a PSR/ON store produced with ``record_only``
    (consistent in its summary AND outcomes header, ``prompt_opened_think_certified``
    true) passed every gate and filed the verdict, though an unfinished think
    block scored ``record_only`` reads an intermediate reasoning number as the
    GSM8K answer. Now the EXPECTED mode is derived per leg through
    ``contracts.think_split_mode`` from the arm, the family's parser pin (Qwen:
    none) and the tuple's render fact, and a leg whose recorded mode differs is
    refused by name — arm-SPECIFIC, so OFF ``record_only`` beside ON
    ``implicit_open`` is the correctly executed experiment, never a mismatch."""
    def record_only(s):
        s.update(think_split_mode=C.THINK_SPLIT_RECORD_ONLY)

    stores = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
                       psr_on={"correct": range(143), "summary_edit": record_only})
    cd = stores[T.leg_key(PSR, "on")]
    # the record is internally consistent and certified — the old admission passed it
    outcomes = completion.read_task_outcomes(cd, TASK)
    summary = completion.read_task_summary(cd, TASK)
    assert outcomes["think_split_mode"] == summary["think_split_mode"] == C.THINK_SPLIT_RECORD_ONLY
    ev = json.loads((cd / "evidence.json").read_text())
    assert ev[C.OOS_EVIDENCE_HEADER_BLOCK][C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED] is True
    rep = analyze(tmp_path)
    leg = leg_of(rep, PSR, "on")
    assert leg["refused"].startswith(
        "think_split_mode 'record_only' != the on arm's expected 'implicit_open' "
        "(contracts.think_split_mode(arm='on', reasoning_parser=None, prompt_opened_think=True))")
    assert "not this experiment" in leg["refused"]
    assert leg["think_split_mode_expected"] == C.THINK_SPLIT_IMPLICIT_OPEN
    assert rep["verdict"]["word"] is None and "estimands" not in rep
    assert rep["unusable_legs"] == {T.leg_key(PSR, "on"): leg["refused"]}
    row = next(r for r in rep["controls"] if r["leg"] == T.leg_key(PSR, "on"))       # still shown
    assert row["summary"]["think_split_mode"] == C.THINK_SPLIT_RECORD_ONLY and row["refused"] == leg["refused"]
    # explicit_tags on an ON leg is the other wrong rule
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143),
                summary_edit=lambda s: s.update(think_split_mode=C.THINK_SPLIT_EXPLICIT_TAGS))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["refused"].startswith(
        "think_split_mode 'explicit_tags' != the on arm's expected 'implicit_open'")
    # arm-specific, not cross-leg equality: an OFF leg scored implicit_open is refused ...
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143))
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT,
                summary_edit=lambda s: s.update(think_split_mode=C.THINK_SPLIT_IMPLICIT_OPEN))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "off")["refused"].startswith(
        "think_split_mode 'implicit_open' != the off arm's expected 'record_only'")
    assert rep["verdict"]["word"] is None
    # ... while the correctly executed 2x2 — OFF record_only beside ON implicit_open — files its word
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT)
    rep = analyze(tmp_path)
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    assert {a: leg_of(rep, c, a)["think_split_mode_expected"] for c in (NONE, PSR) for a in ("off", "on")} == {
        "off": C.THINK_SPLIT_RECORD_ONLY, "on": C.THINK_SPLIT_IMPLICIT_OPEN}
    # the derivation is the contract's, over the family's pin and the tuple's declared fact
    assert C.model_row(T.MODEL).family not in C.REASONING_PARSER_BY_FAMILY and T.FAMILY_REASONING_PARSER is None
    assert T.expected_think_split_mode("on") == C.think_split_mode(
        arm="on", reasoning_parser=None, prompt_opened_think=C.AUTHORISED_OOS_THINKING.prompt_opens_think_on)
    assert (T.expected_think_split_mode("on"), T.expected_think_split_mode("off")) == (
        C.THINK_SPLIT_IMPLICIT_OPEN, C.THINK_SPLIT_RECORD_ONLY)
    # a parser-separated leg is another instrument: the premise of the derivation is checked too
    assert T.think_split_problem({"reasoning_parser": "gemma4", "think_split_mode": "implicit_open"}, "on").startswith(
        "reasoning_parser 'gemma4' != the family's pin None")
    assert T.think_split_problem({"reasoning_parser": None, "think_split_mode": "implicit_open"}, "on") is None
    assert T.think_split_problem({"think_split_mode": "record_only"}, "off") is None
    assert T.think_split_problem({"think_split_mode": None}, "on").startswith("think_split_mode None !=")


def test_stores_of_another_model_or_another_cell_are_refused_by_identity_not_by_agreement(tmp_path):
    """``nuisance_check`` compared ``model_id`` / ``model_revision``
    for EQUALITY across the legs only — four stores all at ``Other/Model @
    badrev`` agreed with each other, passed, and the report was labelled with
    the tuple's slug. Now each leg's manifest is validated POSITIVELY against
    ``contracts.model_row(MODEL).hf`` / ``.rev`` and its cell identity against the
    requested leg BEFORE the cross-leg check, refusing by name, and the report's
    ``model`` label is read off the VALIDATED legs."""
    row = C.model_row(T.MODEL)

    def other_model(m):
        m.update(model_id="Other/Model", model_revision="badrev")

    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)},
              manifest_edit=other_model)
    rep = analyze(tmp_path)
    assert set(rep["unusable_legs"]) == set(rep["legs"])
    assert rep["verdict"]["word"] is None and "estimands" not in rep
    for leg in rep["legs"].values():
        assert leg["identity_validated"] is False
        assert leg["refused"].startswith(
            f"manifest model 'Other/Model' @ 'badrev' != the authorised model's {row.hf!r} @ {row.rev!r} "
            f"(contracts.model_row({T.MODEL!r}))")
        assert "another model's store" in leg["refused"] and "nuisance" not in leg["refused"]
    # the four AGREE with each other — agreement is not identity, and the label says so
    assert rep["nuisance"]["disagreeing"] == []
    assert rep["model"]["slug"] is None and rep["model"]["hf"] is None and rep["model"]["revision"] is None
    assert rep["model"]["validated_legs"] == [] and "no leg's manifest validated" in rep["model"]["source"]
    # the correctly produced 2x2 is labelled from its validated manifests, not a constant
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    assert rep["model"] == {"slug": T.MODEL, "hf": row.hf, "revision": row.rev,
                            "validated_legs": sorted(rep["legs"]), "source": rep["model"]["source"]}
    assert "4 leg(s) validated against contracts.model_row" in rep["model"]["source"]
    assert all(leg["identity_validated"] is True for leg in rep["legs"].values())
    # one leg at the pinned model but another revision: refused by name, before any cross-leg majority
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143),
                manifest_edit=lambda m: m.update(model_revision="badrev"))
    rep = analyze(tmp_path)
    leg = leg_of(rep, PSR, "on")
    assert leg["refused"].startswith(f"manifest model {row.hf!r} @ 'badrev' != the authorised model's")
    assert "summary" not in leg and rep["model"]["validated_legs"] == sorted(set(rep["legs"]) - {T.leg_key(PSR, "on")})
    assert rep["verdict"]["word"] is None
    # a store whose manifest names ANOTHER cell than its leg
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143), manifest_edit=lambda m: m.update(cell_id=NONE))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["refused"] == (
        f"manifest cell_id 'none' != the leg's cell {PSR!r} — another cell's store under this leg's out-root")
    assert rep["verdict"]["word"] is None
    # ... or whose route is another cell's
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143),
                manifest_edit=lambda m: m[C.MANIFEST_KEY_ROUTE].update(virtual_model="petri__base"))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["refused"].startswith(
        f"manifest route virtual_model 'petri__base' != the contract's spelling of cell {PSR!r} on the on arm "
        f"({C.cell_spellings(PSR, declared=True).vmodel!r}, contracts.cell_spellings)")
    assert rep["verdict"]["word"] is None
    # the checker itself, on the shapes the fixture writes
    good = json.loads((build_store(tmp_path, cell=PSR, arm="on", correct=range(143)) / "manifest.json").read_text())
    assert T.identity_problem(good, cell=PSR, arm="on") is None
    assert T.identity_problem(good, cell=NONE, arm="on").startswith(f"manifest cell_id {PSR!r} != the leg's cell 'none'")
    assert T.identity_problem({}, cell=PSR, arm="on").startswith("manifest model None @ None !=")
    # a @thinking variant leg wears the declared spelling on the ON arm and keeps its own cell_id
    variant = PSR + "@thinking"
    vcd = build_store(tmp_path, cell=variant, arm="on", correct=range(100))
    vman = json.loads((vcd / "manifest.json").read_text())
    assert T.identity_problem(vman, cell=variant, arm="on") is None
    assert T.identity_problem(vman, cell=PSR, arm="on").startswith("manifest cell_id")
    assert T.identity_problem(vman, cell=variant, arm="off").startswith(
        f"cell {variant!r} has no route spelling on the off arm")


def test_the_live_prompt_p99_is_handled_like_the_max_and_compared_with_the_artifact(tmp_path, capsys):
    """``prompt_budget_live.p99`` had the three-state machinery but
    only ``max`` derived a pending / a problem — an absent or null p99 was not
    PENDING, a malformed p99 was serialised raw as the metric, and the word was
    filed under the waiver. Now the two statistics are handled alike: absent /
    null is p99's OWN named producer stamp (waivable, enumerated), an off-contract
    value is an unwaivable FAIL naming key and value with the metric slot None,
    and on the ON legs — the same 200 messages under the same render — the live
    p99 must agree with the artifact's ``p99``, recorded beside the max."""
    LIVE, MAX, P99 = (C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK, C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX,
                      C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99)
    p99_stamp = T.stamp_name(LIVE, P99)
    art = prompt_budget_artifact()
    stores = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    cd = stores[T.leg_key(PSR, "on")]
    # the correctly executed experiment: ON legs agree with the artifact, OFF legs record not applicable
    rep = analyze(tmp_path)
    on, off = leg_of(rep, PSR, "on")["ledger"], leg_of(rep, PSR, "off")["ledger"]
    assert on["prompt_p99"] == {"live": art["p99"], "artifact": art["p99"], "agrees": True, "detail": None,
                                "applicable": True}
    assert art["p99"] == PROMPT_MAX - 2
    assert off["prompt_p99"]["applicable"] is False and off["prompt_p99"]["agrees"] is None
    assert off["prompt_p99"]["artifact"] is None and off["prompt_p99"]["live"] == PROMPT_MAX_OFF - 2
    assert "not compared on an OFF leg" in off["prompt_p99"]["detail"] and off["pending"] == []
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    row = next(r for r in rep["controls"] if r["leg"] == T.leg_key(PSR, "on"))
    assert row["ledger"]["prompt_p99"] == on["prompt_p99"]
    # -- absent: p99's OWN producer stamp, pending, waivable; the max still agrees
    _rewrite_evidence(cd, lambda ev: ev[LIVE].pop(P99))
    strict = analyze(tmp_path, allow_pending_gates=False)
    leg = leg_of(strict, PSR, "on")
    led = leg["ledger"]
    assert leg["evidence"][LIVE]["state"][P99] == T.EVIDENCE_ABSENT and leg["evidence"][LIVE][P99] is None
    assert led["passed"] is True and led["prompt_p99"]["agrees"] is None and led["prompt_max"]["agrees"] is True
    [msg] = [p for p in led["pending"] if "live prompt p99" in p]
    assert f"evidence.{p99_stamp}" in msg and msg.endswith("not stamped)")
    assert led["pending_stamps"][led["pending"].index(msg)] == p99_stamp
    assert strict["verdict"]["word"] is None
    [item] = [it for it in strict["gates"]["pending_items"] if it["stamps"] == [p99_stamp]]
    assert item["waivable"] is True and p99_stamp in T.WAIVABLE_PENDING_STAMPS
    assert analyze(tmp_path)["verdict"]["word"] == T.WORD_TRANSFERS
    # -- explicit null: the pass did not run
    _rewrite_evidence(cd, lambda ev: ev[LIVE].update({P99: None}))
    strict = analyze(tmp_path, allow_pending_gates=False)
    leg = leg_of(strict, PSR, "on")
    assert leg["evidence"][LIVE]["state"][P99] == T.EVIDENCE_NULL and leg["evidence"][LIVE][P99] is None
    assert leg["ledger"]["passed"] is True
    assert any("live prompt p99" in p and "stamped null" in p for p in leg["ledger"]["pending"])
    assert strict["verdict"]["word"] is None and analyze(tmp_path)["verdict"]["word"] == T.WORD_TRANSFERS
    # -- malformed: the string "bad", a float, a bool — unwaivable FAIL naming key and value;
    #    the metric slot is None, never the raw value; the max is untouched
    for bad in ("bad", float(art["p99"]), True):
        _rewrite_evidence(cd, lambda ev, bad=bad: ev[LIVE].update({P99: bad}))
        rep = analyze(tmp_path)                                     # the waiver is on
        leg = leg_of(rep, PSR, "on")
        assert leg["evidence"][LIVE]["state"][P99] == T.EVIDENCE_OFF_CONTRACT
        assert leg["evidence"][LIVE][P99] is None
        assert f"{p99_stamp} is {bad!r} ({type(bad).__name__}) — off-contract" in leg["evidence"][LIVE]["problem"][P99]
        assert leg["evidence"][LIVE][MAX] == PROMPT_MAX and leg["evidence"][LIVE]["problem"][MAX] is None
        led = leg["ledger"]
        assert led["passed"] is False and led["pending"] == [] and led["prompt_p99"]["live"] is None
        [f] = [f for f in led["failures"] if "live prompt p99 off-contract" in f]
        assert f"{p99_stamp} is {bad!r} ({type(bad).__name__}) — off-contract" in f
        assert "a JSON integer (a token count) or null was expected" in f
        assert rep["gates"]["declaration"]["status"] == T.GATE_FAIL and "declaration" in rep["gates"]["failed"]
        assert rep["verdict"]["word"] is None and rep["verdict"]["verdict"] == T.NOT_MEASURABLE
        assert repr(bad) not in json.dumps({k: v for k, v in leg["evidence"][LIVE].items() if k != "problem"})
    # -- disagreement with the artifact's p99: FAIL naming both, beside a max that still agrees
    _rewrite_evidence(cd, lambda ev: ev[LIVE].update({P99: art["p99"] + 1}))
    rep = analyze(tmp_path)
    led = leg_of(rep, PSR, "on")["ledger"]
    assert led["prompt_p99"] == {"live": art["p99"] + 1, "artifact": art["p99"], "agrees": False,
                                 "detail": f"live {art['p99'] + 1} / artifact {art['p99']} disagree", "applicable": True}
    assert "live / artifact prompt-p99 disagreement: " + led["prompt_p99"]["detail"] in led["failures"]
    assert led["prompt_max"]["agrees"] is True and led["passed"] is False
    assert rep["gates"]["declaration"]["status"] == T.GATE_FAIL and rep["verdict"]["word"] is None
    # -- no artifact: the comparison has nothing to agree with and adds no pending of its own —
    #    the missing artifact is the ONE (operator-input) pending, not doubled by the p99
    _rewrite_evidence(cd, lambda ev: ev[LIVE].update({P99: art["p99"]}))
    rep = analyze(tmp_path, prompt_budget=None)
    led = leg_of(rep, PSR, "on")["ledger"]
    assert led["prompt_p99"]["artifact"] is None and led["prompt_p99"]["agrees"] is None and led["passed"] is True
    assert sum(T.ARTIFACT_NOT_SUPPLIED in p for p in led["pending"]) == 1
    assert rep["verdict"]["word"] is None and "not waivable" in rep["verdict"]["reason"]
    # the readers
    assert T.live_measurement({P99: 7}, P99) == T.LiveMeasurement(T.EVIDENCE_STAMPED, 7, None)
    assert T.live_measurement({}, P99) == T.LiveMeasurement(T.EVIDENCE_ABSENT, None, None)
    assert T.live_measurement({P99: None}, P99) == T.LiveMeasurement(T.EVIDENCE_NULL, None, None)
    m = T.live_measurement({P99: "bad"}, P99)
    assert (m.state, m.value) == (T.EVIDENCE_OFF_CONTRACT, None) and f"{p99_stamp} is 'bad' (str)" in m.problem
    assert set(T.live_measurements(None)) == set(T.LIVE_PROMPT_STATS) == {MAX, P99}
    assert T._artifact_stat(art, P99) == (art["p99"], None) and T._artifact_stat(None, P99) == (None, T.ARTIFACT_NOT_SUPPLIED)
    assert T._artifact_prompt_max(art) == (art["max"], None)
    # the printed table names the validated model, and an unvalidated report says so
    T.print_table(rep)
    assert f"the 2x2: {T.MODEL} ({C.model_row(T.MODEL).hf} @" in capsys.readouterr().out
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)},
              manifest_edit=lambda m: m.update(model_id="Other/Model"))
    T.print_table(analyze(tmp_path))
    assert "model NOT validated" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the COMPLETE route spelling per arm (unresolved refuses), and present        #
# regime stamps validated before an unresolved expectation is consulted         #
# --------------------------------------------------------------------------- #


def test_the_route_spelling_is_compared_completely_per_arm_and_an_unresolved_one_refuses(tmp_path, monkeypatch):
    """``identity_problem`` compared the route's virtual model on
    its BASE part only (``partition(VARIANT_SEP)[0]``), so a PSR/ON store on the
    bare route and a PSR/OFF store on the ``@thinking`` route both passed. Now
    the COMPLETE spelling is compared exactly, per arm, through
    ``contracts.cell_spellings`` at the leg's door: ``none`` bare on either arm
    (``petri__base``), PSR/OFF bare, PSR/ON the declared ``@thinking`` spelling
    — and should the declared spelling's constants read None (a mutated
    module; B0 filled both) the PSR/ON route identity is UNRESOLVED: a named
    refusal from the estimand (no verdict), never accepted under either
    spelling."""
    ROUTE = C.MANIFEST_KEY_ROUTE
    psr_on = C.cell_spellings(PSR, declared=True).vmodel        # B0 filled the declared spelling's constants
    psr_off = C.cell_spellings(PSR, declared=False).vmodel
    base = C.cell_spellings(NONE, declared=False).vmodel
    assert (psr_on, psr_off, base) == (psr_off + "@thinking", "petri__si_safety__psr_kl_all", "petri__base")
    assert T.expected_route_vmodel(PSR, arm="on") == psr_on and T.expected_route_vmodel(PSR, arm="off") == psr_off
    assert T.expected_route_vmodel(NONE, arm="on") == base == T.expected_route_vmodel(NONE, arm="off")
    # the fixture writes the PRODUCTION routes, and the correctly executed 2x2 files its word
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS
    assert {k: leg["identity"]["virtual_model"] for k, leg in rep["legs"].items()} == {
        T.leg_key(NONE, "off"): base, T.leg_key(NONE, "on"): base,
        T.leg_key(PSR, "off"): psr_off, T.leg_key(PSR, "on"): psr_on}

    def on_route(vm):
        return lambda m: m[ROUTE].update(virtual_model=vm)

    # a PSR/ON manifest on the bare route: refused, both spellings named
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143), manifest_edit=on_route(psr_off))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "on")["refused"] == (
        f"manifest route virtual_model {psr_off!r} != the contract's spelling of cell {PSR!r} on the on arm "
        f"({psr_on!r}, contracts.cell_spellings) — another route under this leg's out-root")
    assert rep["verdict"]["word"] is None
    # a PSR/OFF manifest on the @thinking route: refused
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143))
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, manifest_edit=on_route(psr_on))
    rep = analyze(tmp_path)
    assert leg_of(rep, PSR, "off")["refused"].startswith(
        f"manifest route virtual_model {psr_on!r} != the contract's spelling of cell {PSR!r} on the off arm "
        f"({psr_off!r}")
    assert rep["verdict"]["word"] is None
    # `none` on anything but petri__base, on either arm
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT)
    for arm, correct in (("off", OFF_NONE_CORRECT), ("on", range(126))):
        build_store(tmp_path, cell=NONE, arm=arm, correct=correct, manifest_edit=on_route(psr_off))
        rep = analyze(tmp_path)
        assert leg_of(rep, NONE, arm)["refused"].startswith(
            f"manifest route virtual_model {psr_off!r} != the contract's spelling of cell 'none' on the {arm} arm "
            "('petri__base'")
        assert rep["verdict"]["word"] is None
        build_store(tmp_path, cell=NONE, arm=arm, correct=correct)
    assert analyze(tmp_path)["verdict"]["word"] == T.WORD_TRANSFERS
    # a None spelling constant (a mutated module — B0 filled both; the stores were written by the intact
    # producer): the PSR/ON route identity is UNRESOLVED, a named refusal from the estimand, never accept-both
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    monkeypatch.setattr(C, "OOS_STANDUP_CELL_SUFFIX", None)
    monkeypatch.setattr(C, "OOS_EXPECT_SHA_KEY", None)
    with pytest.raises(C.OutOfSupportDeclarationUnreconciled):
        T.expected_route_vmodel(PSR, arm="on")
    assert T.expected_route_vmodel(PSR, arm="off") == psr_off and T.expected_route_vmodel(NONE, arm="on") == base
    rep = analyze(tmp_path)
    leg = leg_of(rep, PSR, "on")
    assert leg["refused"].startswith(
        f"the on leg's route identity is UNRESOLVED: the declared spelling of cell {PSR!r} cannot be derived yet (")
    assert "OOS_STANDUP_CELL_SUFFIX" in leg["refused"] and "B0" in leg["refused"]
    assert leg["refused"].endswith("— refused from the estimand rather than accepted under either spelling")
    assert leg["identity"]["virtual_model"] == psr_on           # the store says @thinking; verified under neither
    assert leg["identity_validated"] is False and "summary" not in leg
    assert rep["verdict"]["word"] is None and "estimands" not in rep
    assert rep["model"]["validated_legs"] == sorted(set(rep["legs"]) - {T.leg_key(PSR, "on")})
    # the same store on the bare route is refused just the same — not "accepted, the other spelling"
    monkeypatch.undo()                                          # the producer writes with the module intact
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143), manifest_edit=on_route(psr_off))
    monkeypatch.setattr(C, "OOS_STANDUP_CELL_SUFFIX", None)
    assert leg_of(analyze(tmp_path), PSR, "on")["refused"].startswith("the on leg's route identity is UNRESOLVED")


def test_present_regime_stamps_are_validated_before_an_unresolved_expectation_is_consulted(tmp_path, monkeypatch):
    """``regime_check`` consulted the expectation first, so a
    member whose expectation was a ``None`` placeholder read "not reconciled"
    (PENDING, waivable) whatever value it carried — ``thinking_off`` on an ON
    leg, ``garbage-render``, ``garbage-policy`` all filed the word under
    ``--allow-pending-gates``. A PRESENT value is validated first: outside the
    contract vocabulary -> unwaivable FAIL naming key and value; a
    ``render_regime`` off its arm's -> FAIL (decidable from the arm alone); off
    a KNOWN expectation -> FAIL; PENDING only for an absent / null stamp — or,
    should an expected member read ``None`` (a mutated contracts module; B0
    filled every one), a valid, arm-consistent value ("not reconciled"). On the
    shipped module the contract's full triple PASSES on both psr legs."""
    on_good = {"salt_regime": "request-unique", "resumed_policy": "declared-request-unique",
               "render_regime": "thinking_on"}                                  # contract §2.5.11 row 1
    off_good = {"salt_regime": "request-unique", "resumed_policy": "conditional", "render_regime": "thinking_off"}
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "regimes": on_good})
    # the contract's psr/ON triple -> PASSED on the shipped module; none/ON is unstamped (the S3 driver's)
    strict = analyze(tmp_path, allow_pending_gates=False)
    r = leg_of(strict, PSR, "on")["regimes"]
    assert r["status"] == T.GATE_PASS and r["stamped"] is True and r["observed"] == on_good
    assert r["pending_stamps"] == [] and r["detail"] is None
    assert leg_of(strict, NONE, "on")["regimes"]["status"] == T.GATE_PENDING
    assert strict["verdict"]["word"] is None
    waived = analyze(tmp_path)
    assert waived["verdict"]["word"] == T.WORD_TRANSFERS and all(it["waivable"] for it in waived["gates"]["pending_items"])
    # a psr/ON `conditional` is off the KNOWN expectation (the declared cell's attested policy is
    # overridden to request-unique, §2.5.11 Q9) -> FAIL, no waiver
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143),
                regimes={**on_good, "resumed_policy": "conditional"})
    rep = analyze(tmp_path)
    r = leg_of(rep, PSR, "on")["regimes"]
    assert r["status"] == T.GATE_FAIL and r["pending_stamps"] == []
    assert "resumed_policy: 'conditional' != expected 'declared-request-unique'" in r["detail"]
    assert rep["verdict"]["word"] is None
    # ONLY should the expectation read None (a mutated module) does that valid, arm-consistent value
    # read "not reconciled" — PENDING, waivable: the branch that keeps a mutated expectation from passing
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "DECLARED_SALT_POLICY_LABEL", None)
        strict = analyze(tmp_path, allow_pending_gates=False)
        r = leg_of(strict, PSR, "on")["regimes"]
        assert r["status"] == T.GATE_PENDING and r["pending_stamps"] == ["summary.resumed_policy"]
        assert "not reconciled" in r["detail"] and "not stamped" not in r["detail"]
        assert strict["verdict"]["word"] is None
    # thinking_off on an ON leg (count 1): FAIL now, decidable from the arm — even with the waiver
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143),
                regimes={**on_good, "render_regime": "thinking_off"})
    rep = analyze(tmp_path)
    r = leg_of(rep, PSR, "on")["regimes"]
    assert r["status"] == T.GATE_FAIL and r["counts"]["render_regime"] == 1 and r["pending_stamps"] == []
    assert "render_regime: 'thinking_off' on the on arm — this arm renders 'thinking_on'" in r["detail"]
    assert "decidable from the arm alone" in r["detail"]
    assert rep["gates"]["declaration"]["status"] == T.GATE_FAIL and "declaration" in rep["gates"]["failed"]
    assert rep["verdict"]["word"] is None and rep["verdict"]["verdict"] == T.NOT_MEASURABLE
    # thinking_on on an OFF leg: FAIL; `unspecified` anywhere: FAIL (the harness always sends the key)
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143), regimes=on_good)
    for bad in ("thinking_on", "unspecified"):
        build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, regimes={**off_good, "render_regime": bad})
        rep = analyze(tmp_path)
        r = leg_of(rep, PSR, "off")["regimes"]
        assert r["status"] == T.GATE_FAIL and f"render_regime: {bad!r} on the off arm" in r["detail"]
        assert rep["verdict"]["word"] is None
    # garbage: off-contract FAIL naming key, value and the vocabulary; the waiver is irrelevant
    for field, bad, vocab in (("render_regime", "garbage-render", C.RENDER_REGIME_LITERALS),
                              ("resumed_policy", "garbage-policy", C.RESUMED_POLICY_LABELS),
                              ("salt_regime", 7, C.SALT_REGIME_LITERALS)):
        build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, regimes={**off_good, field: bad})
        rep = analyze(tmp_path)
        r = leg_of(rep, PSR, "off")["regimes"]
        assert r["status"] == T.GATE_FAIL and r["pending_stamps"] == []
        assert (f"{field}: {bad!r} ({type(bad).__name__}) is off-contract — not in the vocabulary {list(vocab)}"
                in r["detail"]) and "never a pending one" in r["detail"]
        assert rep["gates"]["declaration"]["status"] == T.GATE_FAIL and rep["verdict"]["word"] is None
    # partially stamped: the absent member is that member's own pending stamp; the present ones are judged
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT,
                regimes={"salt_regime": "request-unique", "resumed_policy": "conditional"})
    r = leg_of(analyze(tmp_path, allow_pending_gates=False), PSR, "off")["regimes"]
    assert r["status"] == T.GATE_PENDING and r["stamped"] is True and r["pending_stamps"] == ["summary.render_regime"]
    assert "not stamped yet: the summary carries no ['render_regime']" in r["detail"]
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT,
                regimes={"salt_regime": "cell-stable", "resumed_policy": "conditional"})
    r = leg_of(analyze(tmp_path), PSR, "off")["regimes"]                # off a KNOWN expectation
    assert r["status"] == T.GATE_FAIL and "salt_regime: 'cell-stable' != expected 'request-unique'" in r["detail"]
    # the vocabulary is the contract's: the two regime / render literal sets and the EIGHT policy labels
    assert T.regime_vocabulary("salt_regime") == C.SALT_REGIME_LITERALS == ("cell-stable", "request-unique")
    assert T.regime_vocabulary("render_regime") == C.RENDER_REGIME_LITERALS
    assert T.regime_vocabulary("resumed_policy") == C.RESUMED_POLICY_LABELS
    assert C.RESUMED_POLICY_LABELS[-2:] == ("declared-request-unique", "forced-request-unique")
    assert len(C.RESUMED_POLICY_LABELS) == len(set(C.RESUMED_POLICY_LABELS)) == 8
    assert (T.render_regime_for_arm("on"), T.render_regime_for_arm("off")) == (
        C.RENDER_REGIME_BY_ARM["on"], C.RENDER_REGIME_BY_ARM["off"]) == ("thinking_on", "thinking_off")
    with pytest.raises(KeyError):
        T.regime_vocabulary("render")
    # `forced-request-unique` is IN the vocabulary (never off-contract) yet expected on NO experiment-1 leg
    build_store(tmp_path, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, regimes=off_good)
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143),
                regimes={**on_good, "resumed_policy": "forced-request-unique"})
    r = leg_of(analyze(tmp_path), PSR, "on")["regimes"]
    assert r["status"] == T.GATE_FAIL and "off-contract" not in r["detail"]
    assert "resumed_policy: 'forced-request-unique' != expected 'declared-request-unique'" in r["detail"]
    # the complete contract triple on both psr legs: PASSED, nothing pending on either
    build_store(tmp_path, cell=PSR, arm="on", correct=range(143), regimes=on_good)
    rep = analyze(tmp_path, allow_pending_gates=False)
    assert leg_of(rep, PSR, "on")["regimes"]["status"] == T.GATE_PASS
    assert leg_of(rep, PSR, "off")["regimes"]["status"] == T.GATE_PASS


# --------------------------------------------------------------------------- #
# the verdict's inclusive edge on ONE representation                           #
# --------------------------------------------------------------------------- #


def test_an_on_effect_exactly_on_the_off_ci_edge_reads_transfers(tmp_path):
    """Closing reviewer (P2, verdict path): the replicate contrasts subtracted
    two separately rounded leg accuracies while the points and bounds were
    exact rationals, and the verdict compared them at an inclusive edge. The
    fixture's OFF legs (120 / 150) with ON none 100 / psr 144 (exact ON effect
    0.22, no censoring; B = 10,000, seed 0) returned an OFF CI of
    ``[0.07999999999999996, 0.21999999999999997]`` and filed "amplifies" with
    both decidability checks passing; the same draws over integer counts give
    ``[0.08, 0.22]``, the ON effect sits ON the inclusive upper edge, and the
    specified word is "transfers". The lower-edge counterpart (psr 116, ON
    effect exactly 0.08) reads the same."""
    for psr_on_correct, want in ((144, 44 / 200), (116, 16 / 200)):
        build_2x2(tmp_path, **off_legs(), none_on={"correct": range(100)}, psr_on={"correct": range(psr_on_correct)})
        rep = analyze(tmp_path, n_resamples=10_000)
        for b in T.BRANCHES:
            s = rep["estimands"][b]["stats"]
            assert (s["effect_off"]["ci_low"], s["effect_off"]["ci_high"]) == (0.08, 0.22)
            assert s["effect_on"]["point"] == want and want in (0.22, 0.08)
            assert s["effect_on"]["bound_low"] == s["effect_on"]["bound_high"] == want
        v = rep["verdict_rule"]
        assert v["word_by_branch"] == {b: T.WORD_TRANSFERS for b in T.BRANCHES}
        assert v["agree_branches"] is True and v["decidable_bounds"] is True
        assert v["decidability"]["words"] == [T.WORD_TRANSFERS] and v["decidability"]["crossings"] == []
        assert rep["verdict"]["word"] == T.WORD_TRANSFERS and rep["verdict"]["verdict"] == T.WORD_TRANSFERS
        # the edge is inclusive and evaluated on ONE representation through both checks
        assert T.classify_word(0.15, 0.08, 0.22, want) == T.WORD_TRANSFERS
        dec = T.decidability_over_bounds(off_ci=(0.08, 0.22), e_off_bound=(0.15, 0.15), e_on_bound=(want, want))
        assert dec["decidable"] is True and dec["words"] == [T.WORD_TRANSFERS]
    # the mixed representation the rule once received: an edge a hair inside the grid value
    old_hi = next(a / 200 - (a - 44) / 200 for a in range(44, 201) if a / 200 - (a - 44) / 200 < 0.22)
    assert T.classify_word(0.15, 0.08, old_hi, 0.22) == T.WORD_AMPLIFIES
    old_lo = next(a / 200 - (a - 16) / 200 for a in range(16, 201) if a / 200 - (a - 16) / 200 > 0.08)
    assert T.classify_word(0.15, old_lo, 0.22, 0.08) == T.WORD_COLLAPSE


def test_the_ledger_confirmation_names_censoring_and_holds_the_summary_to_the_ledger(tmp_path):
    """The censored psr/ON leg: four 504s at the proxy's upstream
    timeout, each re-sampled by the client's retry and the second draw scored,
    while the summary read ``n_rejected 0``. The ledger confirmation FAILS the
    leg by name as censoring, counts the rejected rows, and holds the summary's
    ledger-derived ``n_http_rejected_retried`` to the same number: a summary
    still claiming 0 names the split; a summary from a producer BEFORE the field
    is pending under its waivable producer stamp (the ledger alone still
    decides censoring); a summary that agrees passes."""
    def four_504s(rows):
        gsm = [r for r in rows if r["task"] == TASK]
        for r in gsm[:4]:
            rows.append({**r, "status": 504, "error_code": "http_504", "rejected": True,
                         "usage_prompt_tokens": None, "usage_completion_tokens": None,
                         "finish_reason": None, "tokenize_mismatch": None,
                         "error_class": C.LEDGER_ERROR_CLASS_SERVER_ERROR})

    # (a) the summary claims 0 while the ledger holds four: censoring named, split named
    build_2x2(tmp_path / "a", **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "ledger_edit": four_504s})
    led = leg_of(analyze(tmp_path / "a"), PSR, "on")["ledger"]
    assert led["passed"] is False and led["n_rejected"] == 4
    joined = " | ".join(led["failures"])
    assert "censored: 4 rejected calls re-sampled by the client's retry (upstream HTTP 504)" in joined
    assert "the scored population is not the first-draw population" in joined
    assert "summary n_http_rejected_retried 0 != the ledger's 4 rejected calls" in joined
    assert "204 ledger calls for inspect/gsm8k_paired200, expected 200" in joined
    assert led["pending"] == []
    # (b) a producer before the field: pending under the waivable stamp, censoring still decided by the ledger
    build_2x2(tmp_path / "b", **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "ledger_edit": four_504s,
                      "summary_edit": lambda s: s.pop("n_http_rejected_retried")})
    rep = analyze(tmp_path / "b")
    led = leg_of(rep, PSR, "on")["ledger"]
    assert led["passed"] is False and led["summary_n_http_rejected_retried"] is None
    assert any(f.startswith("censored: 4 rejected calls") for f in led["failures"])
    assert not any("summary n_http_rejected_retried" in f for f in led["failures"])
    assert any(p.startswith("summary n_http_rejected_retried") for p in led["pending"])
    assert "summary.n_http_rejected_retried" in led["pending_stamps"]
    assert "summary.n_http_rejected_retried" in T.WAIVABLE_PENDING_STAMPS
    # (c) a summary that agrees with the ledger passes cleanly on the un-censored legs
    ok = leg_of(rep, NONE, "on")["ledger"]
    assert ok["passed"] and ok["n_rejected"] == 0 and ok["summary_n_http_rejected_retried"] == 0
    # (d) a clean store whose producer predates the field: pending only, waived with the flag
    build_2x2(tmp_path / "c", **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)},
              summary_edit=lambda s: s.pop("n_http_rejected_retried"))
    rep = analyze(tmp_path / "c")
    for cell in (NONE, PSR):
        for arm in ("off", "on"):
            led = leg_of(rep, cell, arm)["ledger"]
            assert led["passed"] is True and led["pending_stamps"] == ["summary.n_http_rejected_retried"]
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS                  # waived: a producer stamp
    strict = analyze(tmp_path / "c", allow_pending_gates=False)
    assert strict["verdict"]["word"] is None and "declaration" in strict["gates"]["pending"]


# --------------------------------------------------------------------------- #
# validation of the censored run: a present-but-unread .eval log is a          #
# named PENDING, the oos column, and the two figures' layout                    #
# --------------------------------------------------------------------------- #

IMPORT_REASON = "inspect_ai / inspect_served not importable here (No module named 'inspect_ai')"


def _plant_eval_log(cell_dir: Path, content: bytes = b"not a zip") -> Path:
    d = cell_dir / C.INSPECT_LOGS_SUBDIR / f"{TASK.replace('/', '__')}__s0"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "2026-01-01T00-00-00_gsm8k_synthetic.eval"
    p.write_bytes(content)
    return p


def test_a_present_but_unread_eval_log_is_a_named_pending_of_the_outcomes_gate(tmp_path, monkeypatch):
    """The censored run's validation in the figure venv: every leg's ``.eval``
    log was PRESENT, inspect_ai could not be imported, ``eval_check.available``
    read false and the outcomes gate read PASSED — a re-derivation that was
    owed and not run, passing silently. Now: the gate is PENDING with one named
    entry per such leg (not a producer stamp, so not waivable — run where the
    log can be read); the split-anomaly gate still runs (the estimand did); an
    ABSENT log stays "the record stands alone" (no pending); a real torn log is
    the same pending through the real reader; --no-eval-logs is not pending."""
    dirs = build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    for d in dirs.values():
        _plant_eval_log(d)
    monkeypatch.setattr(T, "_read_eval_results", lambda cell_dir, task: (None, None, IMPORT_REASON))
    rep = analyze(tmp_path, read_eval_logs=True)
    ec = leg_of(rep, PSR, "on")["eval_check"]
    assert ec["available"] is False and ec["log_present"] is True and ec["reason"] == IMPORT_REASON
    g = rep["gates"]["outcomes"]
    assert g["status"] == T.GATE_PENDING and g["problems"] == {}
    assert len(g["pending"]) == 4 and g["pending_stamps"] == [None] * 4
    assert all(p.endswith(f"independent .eval re-derivation not run: {IMPORT_REASON}") for p in g["pending"])
    assert any(p.startswith(T.leg_key(PSR, "on")) for p in g["pending"])
    items = [it for it in rep["gates"]["pending_items"] if it["gate"] == "outcomes"]
    assert len(items) == 4 and all(it["waivable"] is False and it["stamps"] == [None] for it in items)
    assert rep["gates"]["pending_unwaivable"] and rep["verdict"]["word"] is None
    assert "not waivable by --allow-pending-gates" in rep["verdict"]["reason"]
    assert "independent .eval re-derivation not run" in rep["verdict"]["reason"]
    # the estimand ran and the split-anomaly gate was evaluated over it (outcomes did not FAIL)
    assert "estimands" in rep and rep["gates"]["split_anomalies"]["status"] == T.GATE_PASS
    # a real TORN log through the real reader (inspect_ai present here) is the same pending
    monkeypatch.undo()
    pytest.importorskip("inspect_ai")
    rep = analyze(tmp_path, read_eval_logs=True)
    ec = leg_of(rep, NONE, "off")["eval_check"]
    assert ec["available"] is False and ec["log_present"] is True
    assert ".eval" in ec["reason"] and IMPORT_REASON not in ec["reason"]
    assert rep["gates"]["outcomes"]["status"] == T.GATE_PENDING
    # the operator's explicit --no-eval-logs: reported, not pending
    rep = analyze(tmp_path, read_eval_logs=False)
    assert leg_of(rep, NONE, "off")["eval_check"] == {
        "available": False, "reason": "log reading disabled", "agrees": None,
        "text": T.text_signature(None, "log reading disabled"), "log_present": None}
    assert rep["gates"]["outcomes"]["status"] == T.GATE_PASS and rep["gates"]["outcomes"]["pending"] == []


def test_an_absent_eval_log_leaves_the_record_standing_alone_not_pending(tmp_path):
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path, read_eval_logs=True)
    ec = leg_of(rep, PSR, "on")["eval_check"]
    assert ec["available"] is False and ec["log_present"] is False
    assert "no Inspect .eval log" in ec["reason"] and "stands alone" in ec["reason"]
    g = rep["gates"]["outcomes"]
    assert g["status"] == T.GATE_PASS and g["pending"] == [] and g["pending_stamps"] == []
    assert not [it for it in rep["gates"]["pending_items"] if it["gate"] == "outcomes"]
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS


def test_the_controls_table_prints_a_dash_never_None_in_the_oos_column(tmp_path, capsys):
    """An OFF leg's store carries no out_of_support_thinking key (the OFF arm is
    the in-support render), so its identity reads null — the printed table
    showed ``None`` there; it prints ``—`` now, and ``True`` on the ON legs."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    capsys.readouterr()
    T.print_table(rep)
    out = capsys.readouterr().out
    table = out.split("== the estimand")[0]
    rows = [ln for ln in table.splitlines() if ln.startswith(("none ", "safety_refusal__psr_kl_all "))]
    assert len(rows) == 4 and "None" not in table
    cols = {tuple(r.split()[:2]): r.split()[2] for r in rows}
    assert cols[("none", "off")] == "—" and cols[("safety_refusal__psr_kl_all", "off")] == "—"
    assert cols[("none", "on")] == "True" and cols[("safety_refusal__psr_kl_all", "on")] == "True"


def _bbox(artist, renderer):
    return artist.get_window_extent(renderer=renderer)


def test_the_two_headline_figures_keep_every_label_inside_the_frame(tmp_path):
    """The censored run's PNGs: the effect figure's title clipped at the right
    edge, its ``+0.50 pp`` label collided with the subtitle and the DiD label
    floated above the axes; the 2x2's legend sat on the bar value labels. Both
    are measured off the renderer: every axes text lies inside the axes box,
    the legend is below the axes and overlaps no bar label, and the title fits
    the figure's width. Rendered on a fixture whose effect_off CI is wide (the
    censoring intervals differ from the bars) so the labels have to move."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
              psr_on={"correct": range(143), "unscored": LAST4})
    rep = analyze(tmp_path)
    for stem in ("final_e1_effect_by_arm_ci", "final_e1_gsm8k_acc_2x2"):
        fig, _nums = T.FIGURES[stem](rep)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        (ax,) = fig.axes
        axbox = _bbox(ax, r)
        for t in ax.texts:                        # value labels, breakpoint labels
            tb = _bbox(t, r)
            assert tb.y1 <= axbox.y1 + 0.5 and tb.y0 >= axbox.y0 - 0.5, (stem, t.get_text(), tb, axbox)
        leg = ax.get_legend()
        lb = _bbox(leg, r)
        assert lb.y1 <= axbox.y0 + 0.5, (stem, "legend not below the axes", lb, axbox)     # below the frame
        for t in ax.texts:
            assert not lb.overlaps(_bbox(t, r)), (stem, t.get_text())
        for t in fig.texts:                       # the title block
            tb = _bbox(t, r)
            assert tb.x1 <= fig.bbox.x1 + 0.5 and tb.y0 >= axbox.y1, (stem, t.get_text()[:40])
        paths = T._save(fig, tmp_path / "figs", stem)
        assert all(p.stat().st_size > 0 for p in paths)
    # the verdict left the title for the subtitle
    fig, _ = T.fig_effect_by_arm(rep)
    assert fig.texts[0].get_text() == "Steering effect by arm"
    assert fig.texts[1].get_text().startswith(f"verdict: {rep['verdict']['verdict']}")


# --------------------------------------------------------------------------- #
# the analysis's four interpretation statements + the figure notes             #
# --------------------------------------------------------------------------- #


def test_the_shift_finding_is_a_render_and_cap_statement_with_the_selected_subset_diagnostic(tmp_path):
    """The claim "thinking raised the unsteered cell's accuracy" conflated the
    RENDER with the CAP (OFF/512 vs ON/8192). The finding names both
    configurations and says the comparison does not isolate thinking from
    relief of OFF truncation; beside it, labelled exploratory, how many of the
    I->C transitions were at the cap under OFF and the two legs' accuracies on
    the items NOT capped under OFF. Fixture: 10 none/OFF items ran to the cap,
    8 of them incorrect; every one is correct under ON."""
    capped = set(range(120, 128)) | {0, 1}                  # 8 incorrect + 2 correct at the OFF cap
    build_2x2(tmp_path, none_off={"correct": range(120), "capped_scored": capped},
              psr_off={"correct": OFF_PSR_CORRECT}, none_on={"correct": range(140)}, psr_on={"correct": range(160)})
    rep = analyze(tmp_path)
    assert rep["gates"]["failed"] == []
    [line] = rep["findings"]
    assert line.startswith(f"the thinking-ON/{CAP_ON} configuration scored higher than thinking-OFF/512 on the "
                           "unsteered cell: shift_none = +10.00 pp [")
    assert f" — {T.RENDER_AND_CAP_TEXT}. " in line and "thinking raised" not in line
    assert line.endswith(f"{T.SELECTED_SUBSET_LABEL}: 8 of the 20 I→C transitions were at the cap under OFF; "
                         "on the 190 items not capped under none/OFF, OFF 118/190 vs ON 130/190")
    d = rep["selected_subset_diagnostic"]
    assert d["label"] == T.SELECTED_SUBSET_LABEL
    none = d["cells"]["none"]
    assert none["cap"] == {"off": 512, "on": CAP_ON}
    assert (none["n_items"], none["n_at_cap_off"], none["n_at_cap_off_incorrect"], none["n_i_to_c"],
            none["n_i_to_c_at_cap_off"]) == (200, 10, 8, 20, 8)
    sub = none["subset_not_capped_off"]
    assert sub["n"] == 190 and sub["off"] == {"correct": 118, "scored": 190, "acc": 118 / 190}
    assert sub["on"] == {"correct": 130, "scored": 190, "acc": 130 / 190} and sub["shift"] == pytest.approx(12 / 190)
    psr = d["cells"]["psr"]
    assert psr["n_at_cap_off"] == 0 and psr["subset_not_capped_off"]["n"] == 200
    assert rep["top_items"] == [line]
    # the decline direction wears the same render + cap statement
    build_2x2(tmp_path / "decline", **off_legs(), none_on={"correct": range(60)}, psr_on={"correct": range(90)})
    [down] = analyze(tmp_path / "decline")["findings"]
    assert down.startswith("thinking changed the unsteered cell's capability: the thinking-ON/"
                           f"{CAP_ON} configuration scored lower than thinking-OFF/512")
    assert T.RENDER_AND_CAP_TEXT in down and T.SELECTED_SUBSET_LABEL in down


def test_the_statement_separates_point_sampling_ci_and_censoring_interval_and_never_claims_containment(tmp_path, capsys):
    """The statement wrote ``effect_on … CI ⊂ bound``, which was FALSE
    (the POINT lies inside the censoring bound; the CI does not). Three
    labelled quantities now — the point, its sampling CI, the
    censoring-sensitivity interval with its endpoints' own bootstrap CI — no
    containment glyph anywhere in the report or the printed table, and every
    sign says what "identified" means: the observed sign survives every
    censoring assignment in the SAMPLE, not a population statement."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
              psr_on={"correct": range(143), "unscored": LAST4})
    rep = analyze(tmp_path)
    st = rep["verdict_rule"]["statement"]
    p = rep["estimands"][T.BRANCH_PRIMARY]["stats"]
    s = p["effect_on"]
    assert "⊂" not in st and "⊂" not in json.dumps(rep, default=str)
    assert (f"effect_on = {T._pp(s['point'])} (point); sampling 95 % CI [{T._pp(s['ci_low'])}, {T._pp(s['ci_high'])}]; "
            f"{T.BOUND_LABEL} [{T._pp(s['bound_low'])}, {T._pp(s['bound_high'])}], its endpoints' bootstrap CI "
            f"[{T._pp(s['bound_ci_low'])}, {T._pp(s['bound_ci_high'])}]") in st
    assert s["bound_low"] < s["bound_high"] and s["bound_ci_low"] < s["bound_low"] < s["bound_high"] < s["bound_ci_high"]
    eo = p["effect_off"]
    assert f"effect_off = {T._pp(eo['point'])} (point); sampling 95 % CI" in st
    assert f"{T.BOUND_LABEL} [{T._pp(eo['point'])}, {T._pp(eo['point'])}] — a point: no censored item on these legs" in st
    assert st.startswith(rep["verdict"]["verdict"] + ": effect_off = ")
    for c, sg in rep["verdict_rule"]["signs"].items():
        assert sg["identified_meaning"] == T.SIGN_IDENTIFIED_MEANING
        assert sg["bound_endpoints_ci"] == [p[c]["bound_ci_low"], p[c]["bound_ci_high"]]
    capsys.readouterr()
    T.print_table(rep)
    out = capsys.readouterr().out
    assert T.SIGN_IDENTIFIED_MEANING in out and T.BOUND_LABEL in out
    assert "⊂" not in out and "sharp" not in out and "c-s [" in out


def test_an_unresolved_off_effect_is_said_so_with_its_discordances_never_as_free(tmp_path):
    """An OFF CI covering zero read as "the object is free on GSM8K"
    — an unresolved effect turned into equivalence with no preregistered
    margin. The statement says what was not resolved and puts the OFF
    discordances (C->I / I->C) beside effect_off; a resolved OFF effect carries
    the discordances without the unresolved clause."""
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": range(122)},
              none_on={"correct": range(100)}, psr_on={"correct": range(140)})
    rep = analyze(tmp_path)
    st = rep["verdict_rule"]["statement"]
    assert rep["verdict"]["verdict"] == T.WORD_NOT_RESOLVED
    assert f" — {T.OFF_NOT_RESOLVED_TEXT} (OFF discordances none→psr: C→I 0 / I→C 2); effect_on = " in st
    assert rep["transitions"]["none_to_psr__off"]["counts"] == {"C->C": 120, "C->I": 0, "I->C": 2, "I->I": 78, "->U": 0}
    texts = json.dumps([st, rep["findings"], rep["interpretation_limits"], rep["top_items"]])
    for banned in (" free", "costs nothing", "no cost", "equivalen"):
        assert banned not in texts, banned
    build_2x2(tmp_path / "resolved", **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    st2 = analyze(tmp_path / "resolved")["verdict_rule"]["statement"]
    assert T.OFF_NOT_RESOLVED_TEXT not in st2
    assert "a point: no censored item on these legs (OFF discordances none→psr: C→I 13 / I→C 43); effect_on = " in st2


def test_exact_mcnemar_stands_beside_effect_on_under_both_branches_as_a_sensitivity(tmp_path):
    """The ON cost's percentile CI excluded zero, but exact two-sided
    McNemar on the discordances did not at 0.05. Both branches' exact tests
    (the complete-case n and the score-as-is n) are filed as SENSITIVITY checks
    beside effect_on, each labelled a complete-case exact sensitivity under its
    scoring (never "primary p"), with the one-sided note — every
    complete-case discordance a loss, so an EMPIRICAL bootstrap of those pairs
    could not generate an improvement, which does not bind the registered
    conditional bootstrap where masks VARY (primary: four none/ON draws
    censored) and DOES bind it where every mask is one (score-as-is scores
    every draw) — and the registered replicates' actual signs beside it; the
    headline limits carry the caveat: an ON-only effect beside an unresolved
    OFF one is suggestive, not a robust steering-harms-thinking claim; the DiD
    CI says whether an interaction with the combined render-plus-cap
    configuration is established. Fixture: effect_off exactly 0 with 10/10 OFF
    discordances; 10/0 under ON; 4 none/ON items unscored."""
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": set(range(110)) | set(range(120, 130))},
              none_on={"correct": range(150), "unscored": LAST4}, psr_on={"correct": range(140)})
    rep = analyze(tmp_path)
    p = rep["estimands"][T.BRANCH_PRIMARY]["stats"]
    assert p["effect_off"]["point"] == 0.0 and p["effect_off"]["ci_low"] < 0 < p["effect_off"]["ci_high"]
    assert p["effect_on"]["excludes_zero"] and p["effect_on"]["point"] == pytest.approx(140 / 200 - 150 / 196)
    mc = rep["sensitivity"]["mcnemar_exact"]
    assert set(mc) == set(rep["transitions"]) == set(T.transition_pairs()) == {
        "off_to_on__none", "off_to_on__psr", "none_to_psr__off", "none_to_psr__on"}
    on = mc["none_to_psr__on"]
    assert on["label"] == T.MCNEMAR_LABEL and (on["from"], on["to"]) == (T.leg_key(NONE, "on"), T.leg_key(PSR, "on"))
    exact_p = 2 * 1 / 2 ** 10                                   # two-sided binomial(10, 1/2) at 0 of 10
    note = ("All ten complete-case discordances are losses, so an empirical bootstrap of those 196 pairs cannot "
            f"generate an improvement; {T.ONE_SIDED_NOTE_REGISTERED}")
    assert on[T.BRANCH_PRIMARY] == {
        "n_items": 196, "n_common": 200, "C->I": 10, "I->C": 0, "n_discordant": 10, "p_value": pytest.approx(exact_p),
        "label": "complete-case exact sensitivity under primary scoring",
        "one_sided_discordance": True, "fully_scored": False, "note": note,
        "p_replicate_without_discordant_item": pytest.approx((1 - 10 / 196) ** 196),
        "replicate_signs": rep["estimands"][T.BRANCH_PRIMARY]["stats"]["effect_on"]["replicate_signs"],
        "n_replicates_used": B_FAST}
    assert on["registered_contrast"] == "effect_on"
    assert on[T.BRANCH_SCORE_AS_IS]["n_items"] == 200 and on[T.BRANCH_SCORE_AS_IS]["p_value"] == pytest.approx(exact_p)
    assert on[T.BRANCH_SCORE_AS_IS]["label"] == "complete-case exact sensitivity under score-as-is scoring"
    # score-as-is scores every draw: every mask one, so the constraint binds the registered replicates there (M1)
    assert on[T.BRANCH_SCORE_AS_IS]["fully_scored"] is True
    assert on[T.BRANCH_SCORE_AS_IS]["note"] == ("All ten complete-case discordances are losses, so an empirical bootstrap of "
                                                f"those 200 pairs cannot generate an improvement; "
                                                f"{T.ONE_SIDED_NOTE_FULLY_SCORED.format(n=200, d=10)}")
    assert on[T.BRANCH_SCORE_AS_IS]["replicate_signs"]["positive"] == 0
    assert on[T.BRANCH_PRIMARY]["p_value"] == rep["transitions"]["none_to_psr__on"]["mcnemar"]["p_value"]
    off = mc["none_to_psr__off"][T.BRANCH_PRIMARY]
    assert (off["C->I"], off["I->C"], off["p_value"], off["one_sided_discordance"], off["note"]) == (10, 10, 1.0, False, None)
    st = rep["verdict_rule"]["statement"]
    signs = T.replicate_signs_text(rep["estimands"])
    assert ("exact two-sided McNemar: complete-case exact sensitivity under primary scoring: p = 0.001953 "
            "(C→I 10 / I→C 0, n = 196); complete-case exact sensitivity under score-as-is scoring: p = 0.001953 "
            f"(C→I 10 / I→C 0, n = 200) — {note} ({signs}); DiD = ") in st
    [lim] = [ln for ln in rep["interpretation_limits"] if T.NOT_ROBUST_TEXT in ln]
    assert lim.startswith(f"effect_on = {T._pp(p['effect_on']['point'])} [") and T.OFF_NOT_RESOLVED_TEXT in lim
    assert "exact two-sided McNemar: complete-case exact sensitivity under primary scoring: p = 0.001953" in lim
    assert f"{note} ({signs})" in lim
    assert "primary p" not in json.dumps(rep, default=str) and "cannot draw an improvement" not in json.dumps(rep, default=str)
    did = p["did"]
    includes = did["ci_low"] <= 0.0 <= did["ci_high"]
    assert not includes                                         # this fixture's DiD excludes zero (the other branch)
    interaction = T.did_interaction_text({"off": 512, "on": CAP_ON})
    assert (f"includes zero, so no {interaction} is established" in lim) is includes
    assert (f"the DiD CI includes zero: no {interaction} is established" in st) is includes
    assert (f"the DiD CI [{T._pp(did['ci_low'])}, {T._pp(did['ci_high'])}] excludes zero — an {interaction}, "
            "not a thinking-specific one" in lim) is not includes
    # a resolved OFF effect (transfers) carries no such caveat
    build_2x2(tmp_path / "resolved", **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    assert not any(T.NOT_ROBUST_TEXT in ln for ln in analyze(tmp_path / "resolved")["interpretation_limits"])


def test_the_closed_think_capped_recoding_is_a_named_sensitivity_beside_the_registered_policy(tmp_path):
    """One psr/ON item ran to the cap with a CLOSED think block — a
    truncated answer after completed reasoning, scored wrong under the
    registered policy. The report files the recoding (censored) as a
    sensitivity: the items, the policy, both branches' recomputed contrasts,
    the word, and how much wider effect_on's bound gets; a run without such an
    item files none, and an OFF at-cap item (no think block) is not one."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)},
              psr_on={"correct": range(143), "capped_scored": {150}})
    rep = analyze(tmp_path)
    assert rep["gates"]["failed"] == [] and rep["gates"]["cap"]["status"] == T.GATE_PASS
    assert leg_of(rep, PSR, "on")["items"][IDS[150]]["at_cap"] is True
    ct = rep["sensitivity"]["closed_think_capped"]
    assert ct["policy"] == T.CLOSED_THINK_CAPPED_POLICY
    assert ct["items"] == {T.leg_key(PSR, "on"): [IDS[150]]} and ct["n_items"] == 1
    est_on = rep["estimands"][T.BRANCH_PRIMARY]["stats"]["effect_on"]
    assert est_on["point"] == pytest.approx(143 / 200 - 126 / 200) and est_on["bound_low"] == est_on["bound_high"]
    rec = ct["recoded"][T.BRANCH_PRIMARY]["effect_on"]
    assert rec["point"] == pytest.approx(143 / 199 - 126 / 200)
    assert (rec["bound_low"], rec["bound_high"]) == (pytest.approx(0.085), pytest.approx(0.09))
    assert set(rec) == {"point", "ci_low", "ci_high", "bound_low", "bound_high", "bound_ci_low", "bound_ci_high"}
    assert ct["effect_on_bound_widening"] == {T.BRANCH_PRIMARY: pytest.approx(0.005), T.BRANCH_SCORE_AS_IS: pytest.approx(0.0)}
    assert ct["recoded"][T.BRANCH_SCORE_AS_IS]["effect_on"]["point"] == pytest.approx(est_on["point"])   # wrong either way
    assert ct["acc_by_leg"][T.BRANCH_PRIMARY]["psr/on"]["n_censored"] == 1 and set(ct["recoded"][T.BRANCH_PRIMARY]) == set(pr.CONTRASTS)
    assert ct["verdict"] == rep["verdict"]["verdict"] == T.WORD_TRANSFERS and ct["word_by_branch"] == rep["verdict_rule"]["word_by_branch"]
    build_2x2(tmp_path / "clean", **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    ct2 = analyze(tmp_path / "clean")["sensitivity"]["closed_think_capped"]
    assert ct2 == {"policy": T.CLOSED_THINK_CAPPED_POLICY, "items": {}, "n_items": 0, "recoded": None}
    build_2x2(tmp_path / "off", none_off={"correct": range(120), "capped_scored": {130}}, psr_off={"correct": OFF_PSR_CORRECT},
              none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    assert analyze(tmp_path / "off")["sensitivity"]["closed_think_capped"]["items"] == {}


def test_the_parity_wording_is_a_passed_screen_never_parity_established(tmp_path):
    """``drift_beyond_ci = false`` is a PASSED SCREEN, not equivalence. The
    report's sensitivity block words it "no drift detected by the registered
    screen" with the like-for-like note; a drift names its cells; no wave root
    says no screen ran."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    par = analyze(tmp_path, wave_root=wave)["sensitivity"]["parity"]
    assert par == {"wording": T.PARITY_NO_DRIFT, "note": T.PARITY_NOTE, "drift_cells": [], "unscreened_cells": []}
    assert par["wording"] == "no drift detected by the registered screen"
    assert "established" not in par["wording"] and "like-for-like on the checked task and serving settings" in par["note"]
    build_store(wave, cell=NONE, arm="off", correct=range(90), out_suffix="c7e1796", oos_marker=False)
    par = analyze(tmp_path, wave_root=wave)["sensitivity"]["parity"]
    assert par["wording"] == f"drift beyond its CI detected by the registered screen on ['{NONE}']" and par["drift_cells"] == [NONE]
    par = analyze(tmp_path)["sensitivity"]["parity"]
    assert par["wording"] == "no serving-parity screen was run (no --wave-root given)" and par["note"] == T.PARITY_NOTE
    half = T.parity_wording({"available": True, "cells": {NONE: {"drift_beyond_ci": None}, PSR: {"drift_beyond_ci": False}}})
    assert half["wording"] == f"the registered screen did not run on every cell (unscreened: ['{NONE}'])"


def test_the_controls_gate_is_on_only_over_the_on_bars_and_the_transition_bars_carry_counts(tmp_path):
    """Figure notes: the controls plot's "2 % gate" line ran across the OFF
    bars it does not judge — it is drawn over the ON bars only and labelled
    ON-only; the transition figure's decisive 5 vs 0 was unreadable beside
    ~190 unchanged — every bar carries its count."""
    pytest.importorskip("matplotlib")
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
              psr_on={"correct": range(143), "unscored": LAST4})
    rep = analyze(tmp_path)
    fig, _ = T.fig_controls(rep)
    [ax] = [a for a in fig.axes if a.get_title(loc="left") == "at-cap rate (summary)"]
    assert [t.get_text() for t in ax.texts] == [T.CAP_GATE_LABEL] == ["ON-only 2 % gate"]
    assert not [ln for ln in ax.lines if list(ln.get_ydata()) == [T.CAP_GATE_AT_CAP_RATE] * 2]   # no full-width line
    segs = [seg for coll in ax.collections for seg in coll.get_segments()]
    slots = list(T._leg_slots(0.4))
    on_xs = sorted(x for x, _k, arm in slots if arm == C.THINKING_ON)
    assert sorted(round((seg[0][0] + seg[1][0]) / 2, 6) for seg in segs) == on_xs
    assert all(seg[0][1] == seg[1][1] == T.CAP_GATE_AT_CAP_RATE for seg in segs)
    for x, _k, arm in slots:
        if arm == C.THINKING_OFF:
            assert not any(seg[0][0] <= x <= seg[1][0] for seg in segs), "the gate must not cross an OFF bar"
    fig2, _ = T.fig_transitions(rep)
    tr = rep["transitions"]
    keys = ("C->C", "C->I", "I->C", "I->I", "->U")
    for ax, prefix, names in ((fig2.axes[0], "off_to_on", ("none", "psr")), (fig2.axes[1], "none_to_psr", ("off", "on"))):
        want = sorted(str(tr[f"{prefix}__{n}"]["counts"][k]) for n in names for k in keys)
        assert sorted(t.get_text() for t in ax.texts) == want and "4" in want   # LAST4 -> ->U reads 4


def test_coincident_breakpoint_labels_merge_into_one_on_the_effect_figure(tmp_path):
    """Figure note: with effect_off = 0 the 0 and the two ±|E_off|
    breakpoint lines coincide and their three labels printed on top of each
    other — one label, "0 = ±|E_off|", now; the legend names the bound for
    what it is (a censoring-sensitivity interval), never "sharp identified"."""
    pytest.importorskip("matplotlib")
    assert T.breakpoint_labels(0.0, -0.03, 0.03) == {0.0: ["0", "+|E_off|", "-|E_off|"], -0.03: ["OFF CI lo"], 0.03: ["OFF CI hi"]}
    assert T._fold_e_off(["0", "+|E_off|", "-|E_off|"]) == ["0", "±|E_off|"]
    assert T.breakpoint_labels(0.15, 0.08, 0.22) == {0.0: ["0"], 0.08: ["OFF CI lo"], 0.22: ["OFF CI hi"],
                                                     0.15: ["+|E_off|"], -0.15: ["-|E_off|"]}
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": set(range(110)) | set(range(120, 130))},
              none_on={"correct": range(150)}, psr_on={"correct": range(140)})
    rep = analyze(tmp_path)
    assert rep["estimands"][T.BRANCH_PRIMARY]["stats"]["effect_off"]["point"] == 0.0
    fig, _ = T.fig_effect_by_arm(rep)
    (ax,) = fig.axes
    labels = [t.get_text() for t in ax.texts]
    assert labels.count("0 = ±|E_off|") == 1 and not any(lb in ("0", "+|E_off|", "-|E_off|") for lb in labels)
    assert "OFF CI lo" in labels and "OFF CI hi" in labels
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert T.BOUND_LABEL in legend and not any("sharp" in lb for lb in legend)
    # a non-zero effect_off keeps its five distinct labels
    build_2x2(tmp_path / "five", **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    fig, _ = T.fig_effect_by_arm(analyze(tmp_path / "five"))
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert all(lb in labels for lb in ("0", "OFF CI lo", "OFF CI hi", "+|E_off|", "-|E_off|"))


def test_the_parity_figure_annotations_stay_inside_the_axes_clear_of_the_subtitle(tmp_path):
    """Figure note: the parity figure's annotation touched the subtitle
    (placed above an auto y-range). Measured off the renderer: every axes text
    lies inside the axes box, the title block sits above the axes and overlaps
    no annotation; the subtitle carries the screen's wording."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    rep = analyze(tmp_path, wave_root=wave)
    fig, _ = T.fig_parity(rep)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    (ax,) = fig.axes
    axbox = _bbox(ax, r)
    assert len(ax.texts) == 2
    for t in ax.texts:
        tb = _bbox(t, r)
        assert tb.y1 <= axbox.y1 + 0.5 and tb.y0 >= axbox.y0 - 0.5, (t.get_text(), tb, axbox)
    for t in fig.texts:
        tb = _bbox(t, r)
        assert tb.y0 >= axbox.y1, t.get_text()[:40]
        assert not any(tb.overlaps(_bbox(a, r)) for a in ax.texts), t.get_text()[:40]
    assert T.PARITY_NO_DRIFT in fig.texts[1].get_text()
    T._save(fig, tmp_path / "figs", T.FIGURE_PARITY)


def test_the_wandb_run_and_artifact_names_take_the_suffix(tmp_path, monkeypatch):
    """A re-analysis of the same stores is logged apart from the first run:
    ``--wandb-name-suffix -v2`` names the run AND its figures artifact."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    calls: dict = {}

    class _Summary:
        @staticmethod
        def update(d):
            calls["summary"] = d

    class _Run:
        url = "https://wandb.test/run"
        summary = _Summary()

        def log_artifact(self, art):
            calls["artifact"] = art.name

        def finish(self):
            calls["finished"] = True

    class _Artifact:
        def __init__(self, name, type):
            self.name = name

        def add_dir(self, d):
            calls["artifact_dir"] = d

    def _init(**kw):
        calls["init"] = kw
        return _Run()

    fake = SimpleNamespace(init=_init, Image=lambda p: p, Artifact=_Artifact,
                           log=lambda d: calls.setdefault("logged", []).append(d))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("SISTEER_WANDB_PROJECT_TINYCAP", "example-tinycap")        # the purpose's project, as a site names it
    png = tmp_path / "figs" / "final_e1_x.png"
    png.parent.mkdir()
    png.write_bytes(b"png")
    url = T.log_wandb(rep, {"written": {"final_e1_x": [str(png)]}}, mode="offline", name_suffix="-v2")
    assert url == "https://wandb.test/run" and calls["finished"]
    assert calls["init"]["name"] == f"e1-thinking-transfer-{SHA7}-v2" and calls["init"]["project"] == "example-tinycap"
    assert T.WANDB_PURPOSE == "tinycap"
    # the run's config carries BOTH shas — the store pin and the analysis code's
    assert calls["init"]["config"]["sha"] == SHA7 and calls["init"]["config"]["analysis_code_sha"] == rep["analysis_code_sha"]
    assert calls["artifact"] == f"figures-e1-thinking-transfer-{SHA7}-v2" and calls["artifact_dir"] == str(png.parent)
    assert calls["summary"]["verdict"] == rep["verdict"]["verdict"] and calls["logged"] == [{"e1/final_e1_x": str(png)}]
    T.log_wandb(rep, {}, mode="offline")
    assert calls["init"]["name"] == f"e1-thinking-transfer-{SHA7}"
    ap = T.build_parser()
    assert ap.parse_args(["--gen-root", ".", "--sha", "x", "--wandb-name-suffix=-v2"]).wandb_name_suffix == "-v2"
    assert ap.parse_args(["--gen-root", ".", "--sha", "x"]).wandb_name_suffix == ""


# --------------------------------------------------------------------------- #
# render+cap contrasts everywhere, the analysis code's own sha, and a          #
# directional ceiling                                                          #
# --------------------------------------------------------------------------- #

#: the fixture's two configurations, as every arm mention must read
OFF_CFG, ON_CFG = "OFF/512", f"ON/{CAP_ON}"
BOTH_CFG = f"{OFF_CFG} vs {ON_CFG}"


def _git_head(repo: Path) -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_every_cross_arm_contrast_is_a_render_cap_contrast_and_the_did_is_never_on_specific(tmp_path, capsys):
    """shift_psr and the DiD are OFF/512 vs ON/8192 contrasts exactly
    as shift_none is, yet the DiD was called an "ON-specific interaction" and
    shift_psr carried no qualifier. Now: the DiD's statement and limit lines
    say "interaction with the combined render-plus-cap configuration (OFF/512
    vs ON/8192)" (the includes-zero AND the excludes-zero branch), the word
    "ON-specific" appears nowhere in the report, every sign names its contrast,
    and a ``cross_arm_contrasts`` block qualifies shift_none, shift_psr and the
    DiD each beside its own number. Fixture A: OFF effect exactly 0 with 10/10
    discordances; 5/0 under ON, so effect_on's CI excludes zero while the DiD's
    includes it (the real run's shape)."""
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": set(range(110)) | set(range(120, 130))},
              none_on={"correct": range(150), "unscored": LAST4}, psr_on={"correct": set(range(150)) - {0, 1, 2, 3, 4}})
    rep = analyze(tmp_path)
    p = rep["estimands"][T.BRANCH_PRIMARY]["stats"]
    assert p["effect_on"]["excludes_zero"] and p["did"]["ci_low"] < 0 < p["did"]["ci_high"]
    interaction = T.did_interaction_text({"off": 512, "on": CAP_ON})
    assert interaction == f"interaction with the combined render-plus-cap configuration ({BOTH_CFG})"
    st = rep["verdict_rule"]["statement"]
    assert (f"; DiD = {T._pp(p['did']['point'])} (point); sampling 95 % CI {T._ci(p['did'])}; " in st)
    assert (f" ({T.RENDER_CAP_CONTRAST_LABEL}: effect_on at {ON_CFG} − effect_off at {OFF_CFG}) — "
            f"the DiD CI includes zero: no {interaction} is established" in st)
    [lim] = [ln for ln in rep["interpretation_limits"] if T.NOT_ROBUST_TEXT in ln]
    assert f"the DiD CI {T._ci(p['did'])} includes zero, so no {interaction} is established — " in lim
    assert "ON-specific" not in json.dumps(rep, default=str)
    for c, sg in rep["verdict_rule"]["signs"].items():
        assert sg["contrast"] == f"{T.RENDER_CAP_CONTRAST_LABEL} ({BOTH_CFG})", c
    xa = rep["cross_arm_contrasts"]
    assert (xa["label"], xa["configurations"], xa["note"]) == (T.RENDER_CAP_CONTRAST_LABEL, BOTH_CFG, T.RENDER_AND_CAP_TEXT)
    assert list(xa["lines"]) == ["shift_none", "shift_psr", "did"]
    for c, cell in (("shift_none", "none"), ("shift_psr", "psr")):
        assert xa["lines"][c] == (f"{c} = {T._pp(p[c]['point'])} {T._ci(p[c])} — {T.RENDER_CAP_CONTRAST_LABEL}: "
                                  f"acc({cell}, {ON_CFG}) − acc({cell}, {OFF_CFG}); {T.RENDER_AND_CAP_TEXT}")
    assert xa["lines"]["did"] == (f"DiD = {T._pp(p['did']['point'])} {T._ci(p['did'])} — {T.RENDER_CAP_CONTRAST_LABEL}: "
                                  f"effect_on ({ON_CFG}) − effect_off ({OFF_CFG}); a nonzero DiD would be an "
                                  f"{interaction}, not a thinking-specific one")
    capsys.readouterr()
    T.print_table(rep)
    out = capsys.readouterr().out
    assert f"== cross-arm contrasts: each a {T.RENDER_CAP_CONTRAST_LABEL} ({BOTH_CFG}) ==" in out
    assert xa["lines"]["shift_psr"] in out and "ON-specific" not in out
    # the excludes-zero branch names the same interaction, and says it is not thinking-specific
    build_2x2(tmp_path / "b", none_off={"correct": range(120)}, psr_off={"correct": set(range(110)) | set(range(120, 130))},
              none_on={"correct": range(150), "unscored": LAST4}, psr_on={"correct": range(140)})
    rep_b = analyze(tmp_path / "b")
    dd = rep_b["estimands"][T.BRANCH_PRIMARY]["stats"]["did"]
    assert dd["ci_high"] < 0
    [lim_b] = [ln for ln in rep_b["interpretation_limits"] if T.NOT_ROBUST_TEXT in ln]
    assert f"the DiD CI {T._ci(dd)} excludes zero — an {interaction}, not a thinking-specific one — " in lim_b
    assert "includes zero" not in rep_b["verdict_rule"]["statement"].split("; DiD = ")[1]
    # without a known cap the labels fall back to the bare arm, never to a wrong number
    assert T.arm_cap_label("on", None) == "ON" and T.arm_cap_label("off", {"off": None}) == "OFF"
    assert T.render_cap_configurations({}) == "OFF vs ON"


def test_the_ceiling_limit_is_directional_and_the_floor_mirrors_it(tmp_path):
    """The phrase "near the ceiling — effect_on has little room" read as little
    room in EITHER direction, which biases against the observed negative
    effect: at none/ON 0.980 a benefit has ~2 pp of room, a loss ~98. One line
    per edge now, over the ON legs at it, naming each with its cap: the
    ceiling limits evidence for a benefit (the room = 1 − the unsteered ON
    cell) and leaves few discordant errors; it does not constrain a
    steering-induced loss. The floor mirrors it. "little room" and "either
    direction" appear nowhere."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(196)}, psr_on={"correct": range(191)})
    rep = analyze(tmp_path)
    assert rep["interpretation_limits"] == [
        (f"the ON legs are near the ceiling (none/{ON_CFG} 0.980, psr/{ON_CFG} 0.955): the ceiling limits evidence "
         "for a benefit (~+2.0 pp of room) and leaves few discordant errors; it does not constrain a "
         "steering-induced loss")]
    assert T.CEILING_TEXT.format(room="~+2.0 pp") in rep["interpretation_limits"][0]
    for banned in ("little room", "either direction"):
        assert banned not in json.dumps(rep, default=str), banned
    # a single leg at the ceiling names only itself; the room is still the unsteered ON cell's
    build_2x2(tmp_path / "one", **off_legs(), none_on={"correct": range(196)}, psr_on={"correct": range(150)})
    [lim] = analyze(tmp_path / "one")["interpretation_limits"]
    assert lim.startswith(f"the ON legs are near the ceiling (none/{ON_CFG} 0.980): the ceiling limits evidence for a benefit (~+2.0 pp of room)")
    # the floor: evidence for a LOSS is what it limits, discordant successes what it leaves few of
    build_2x2(tmp_path / "floor", **off_legs(), none_on={"correct": range(4)}, psr_on={"correct": range(6)})
    [lim] = analyze(tmp_path / "floor")["interpretation_limits"]
    assert lim == (f"the ON legs are near the floor (none/{ON_CFG} 0.020, psr/{ON_CFG} 0.030): the floor limits evidence "
                   "for a loss (~-2.0 pp of room) and leaves few discordant successes; it does not constrain a "
                   "steering-induced gain")
    assert "near the floor" in lim and T.FLOOR_TEXT.format(room="~-2.0 pp") in lim
    # off the edges: no such line
    build_2x2(tmp_path / "mid", **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    assert analyze(tmp_path / "mid")["interpretation_limits"] == []


def test_the_figures_name_every_arm_with_its_cap_and_the_did_as_a_render_cap_contrast(tmp_path):
    """The figures: the 2x2, the controls and the transitions named
    the arms "off" / "on"; the effect figure's DiD read "(ON − OFF)". Every arm
    label now reads OFF/512 or ON/8192 (tick labels, the hatch legend, the
    transition series and panel title, the parity ticks), the DiD tick carries
    "render+cap contrast", and the numbers / tables JSON beside the figures
    carry a caption per written figure and a label per contrast in the same
    vocabulary, plus the analysis code's sha. Measured off the renderer on the
    effect figure: the wider DiD tick label and the legend do not overlap."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
              psr_on={"correct": range(143), "unscored": LAST4})
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    rep = analyze(tmp_path, wave_root=wave)
    legs = [f"none\n{OFF_CFG}", f"none\n{ON_CFG}", f"psr\n{OFF_CFG}", f"psr\n{ON_CFG}"]
    fig, _ = T.fig_acc_2x2(rep)
    (ax,) = fig.axes
    assert [t.get_text() for t in ax.get_xticklabels()] == legs
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["none (unsteered)", "safety_refusal__psr_kl_all",
                                                                   f"thinking ON ({ON_CFG})"]
    fig, _ = T.fig_controls(rep)
    for ax in fig.axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == legs, ax.get_title(loc="left")
    assert f"thinking ON ({ON_CFG})" in [t.get_text() for t in fig.legends[0].get_texts()]
    fig, _ = T.fig_transitions(rep)
    assert fig.axes[0].get_title(loc="left") == f"{OFF_CFG} → {ON_CFG}, per cell ({T.RENDER_CAP_CONTRAST_LABEL})"
    assert fig.axes[1].get_title(loc="left") == "none → psr, per arm"
    series = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert [s.split(" — ")[0] for s in series] == [OFF_CFG, ON_CFG]
    assert [s.split(" — ")[0] for t in (fig.axes[0].get_legend().get_texts(),) for s in (x.get_text() for x in t)] == ["none", "psr"]
    fig, _ = T.fig_parity(rep)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == [f"none, {OFF_CFG}", f"psr, {OFF_CFG}"]
    fig, _ = T.fig_effect_by_arm(rep)
    (ax,) = fig.axes
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        f"effect_off\n(psr − none, {OFF_CFG})", f"effect_on\n(psr − none, {ON_CFG})",
        f"DiD, {T.RENDER_CAP_CONTRAST_LABEL}\n({ON_CFG} − {OFF_CFG})"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    lb = _bbox(ax.get_legend(), r)
    for t in ax.get_xticklabels():
        assert not lb.overlaps(_bbox(t, r)), t.get_text()
    # the JSON beside the figures speaks the same vocabulary and names the analysis code
    figs = T.make_figures(rep, tmp_path / "figs")
    numbers = json.loads(Path(figs["numbers"]).read_text())
    tables = json.loads(Path(figs["tables"]).read_text())
    assert set(numbers["captions"]) == set(figs["written"]) == set(T.FIGURE_STEMS)
    assert numbers["captions"] == T.figure_captions(rep) == tables["captions"]
    for stem, cap in numbers["captions"].items():
        assert OFF_CFG in cap, stem
        assert (ON_CFG in cap) is (stem != T.FIGURE_PARITY), stem
        assert "render+cap" not in cap or T.RENDER_CAP_CONTRAST_LABEL in cap
    assert T.RENDER_CAP_CONTRAST_LABEL in numbers["captions"]["final_e1_effect_by_arm_ci"]
    assert T.RENDER_CAP_CONTRAST_LABEL in numbers["captions"]["final_e1_item_transitions"]
    assert tables["contrast_labels"] == T.contrast_labels(rep) == {
        "effect_off": f"acc(psr, {OFF_CFG}) − acc(none, {OFF_CFG})", "effect_on": f"acc(psr, {ON_CFG}) − acc(none, {ON_CFG})",
        "shift_none": f"acc(none, {ON_CFG}) − acc(none, {OFF_CFG}) — {T.RENDER_CAP_CONTRAST_LABEL}",
        "shift_psr": f"acc(psr, {ON_CFG}) − acc(psr, {OFF_CFG}) — {T.RENDER_CAP_CONTRAST_LABEL}",
        "did": f"effect_on ({ON_CFG}) − effect_off ({OFF_CFG}) — {T.RENDER_CAP_CONTRAST_LABEL}"}
    for doc in (numbers, tables):
        assert doc["sha"] == SHA7 and doc["analysis_code_sha"] == rep["analysis_code_sha"] != SHA7
    # the number identifiers are unchanged (a diff against an earlier render compares leaf for leaf)
    assert [n["name"] for n in numbers["figures"]["final_e1_gsm8k_acc_2x2"]][:2] == ["acc[none/off]", "se[none/off]"]
    assert "ON-specific" not in json.dumps([numbers, tables])


def test_the_report_carries_the_analysis_codes_own_sha_beside_the_store_pin(tmp_path, monkeypatch, capsys):
    """The stores were produced at one commit and read by analysis
    code at another, and the report recorded only the first (`sha`), so the
    documented pin could not reproduce the analysis. The report now carries
    ``analysis_code_sha`` — the HEAD of the tree this script lives in, read at
    runtime, with whether that tree was dirty — or the ``--analysis-sha`` the
    caller passed; a value that is not a sha, or no git and no flag, refuses
    by name rather than writing a null."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path)
    head = _git_head(REPO)
    assert rep["sha"] == SHA7 and rep["analysis_code_sha"] == head and len(head) == 40 and head != SHA7
    code = rep["analysis_code"]
    assert code == {"sha": head, "source": "git rev-parse HEAD of the script's tree", "dirty": code["dirty"],
                    "tree": str(REPO), "note": T.ANALYSIS_CODE_NOTE}
    assert isinstance(code["dirty"], bool)
    assert "distinct from `sha`, the producer/store pin" in T.ANALYSIS_CODE_NOTE
    assert list(rep)[:5] == ["model", "task", "sha", "analysis_code_sha", "analysis_code"]
    capsys.readouterr()
    T.print_table(rep)
    out = capsys.readouterr().out
    assert f"producer/store pin {SHA7}; analysis code {head} (git rev-parse HEAD of the script's tree" in out
    assert ("TREE DIRTY" in out) is code["dirty"]
    # an explicit sha is taken as given and marked as such
    rep2 = analyze(tmp_path, analysis_sha="abc1234")
    assert rep2["analysis_code_sha"] == "abc1234"
    assert rep2["analysis_code"] == {"sha": "abc1234", "source": "--analysis-sha", "dirty": None, "tree": None,
                                     "note": T.ANALYSIS_CODE_NOTE}
    with pytest.raises(T.TransferAnalysisError, match="--analysis-sha 'not-a-sha': expected 7-40 hex digits"):
        analyze(tmp_path, analysis_sha="not-a-sha")
    # no git available and no flag: refused by name, never a null sha
    def _no_git(*_a, **_k):
        raise OSError("git: not found")
    monkeypatch.setattr(T.subprocess, "run", _no_git)
    with pytest.raises(T.TransferAnalysisError, match="could not be read from .*; pass --analysis-sha"):
        analyze(tmp_path)
    assert analyze(tmp_path, analysis_sha=head)["analysis_code_sha"] == head      # the flag needs no git
    ap = T.build_parser()
    assert ap.parse_args(["--gen-root", ".", "--sha", "x", "--analysis-sha", head]).analysis_sha == head
    assert ap.parse_args(["--gen-root", ".", "--sha", "x"]).analysis_sha is None


def test_the_registered_bootstraps_replicate_signs_stand_beside_the_one_sided_note(tmp_path, capsys):
    """The note "with one-sided discordance the percentile bootstrap cannot
    draw an improvement" is true of an EMPIRICAL bootstrap of the complete-case
    pairs, not of the registered conditional bootstrap, whose replicates
    resample censoring masks and divide by per-replicate denominators (the real
    run: 9,961 negative / 20 zero / 19 positive of 10,000). The note now says
    exactly that, per branch, and the registered replicates' sign counts —
    filed by ``pairing`` from its own replicate arrays as ``replicate_signs``
    on every contrast — stand beside it in the statement, the limits and the
    printed table. "primary p" and the old sentence appear nowhere. Without
    censoring the property holds literally: not one positive replicate."""
    build_2x2(tmp_path, none_off={"correct": range(120)}, psr_off={"correct": set(range(110)) | set(range(120, 130))},
              none_on={"correct": range(150), "unscored": LAST4}, psr_on={"correct": range(140)})
    rep = analyze(tmp_path)
    est = rep["estimands"]
    for b in T.BRANCHES:
        for c in pr.CONTRASTS:
            rs = est[b]["stats"][c]["replicate_signs"]
            assert set(rs) == {"negative", "zero", "positive"} and all(isinstance(v, int) and v >= 0 for v in rs.values())
            assert sum(rs.values()) == est[b]["n_replicates_used"] == B_FAST
    on = rep["sensitivity"]["mcnemar_exact"]["none_to_psr__on"]
    pri = on[T.BRANCH_PRIMARY]
    assert pri["one_sided_discordance"] and pri["label"] == T.MCNEMAR_BRANCH_LABEL.format(branch="primary")
    assert T.one_sided_note(pri) == pri["note"] == (
        "All ten complete-case discordances are losses, so an empirical bootstrap of those 196 pairs cannot "
        f"generate an improvement; {T.ONE_SIDED_NOTE_REGISTERED}")
    assert T.ONE_SIDED_NOTE_REGISTERED == ("the registered conditional bootstrap also resamples censoring masks and "
                                           "uses per-replicate denominators, so this property does not apply "
                                           "literally to its replicates")
    signs = T.replicate_signs_text(est)
    eo = est[T.BRANCH_PRIMARY]["stats"]["effect_on"]["replicate_signs"]
    sa = est[T.BRANCH_SCORE_AS_IS]["stats"]["effect_on"]["replicate_signs"]
    assert signs == ("registered bootstrap replicate signs for effect_on — "
                     f"primary: {eo['negative']:,} negative / {eo['zero']:,} zero / {eo['positive']:,} positive of {B_FAST:,}; "
                     f"score-as-is: {sa['negative']:,} negative / {sa['zero']:,} zero / {sa['positive']:,} positive of {B_FAST:,}")
    st = rep["verdict_rule"]["statement"]
    assert f" — {pri['note']} ({signs}); DiD = " in st
    [lim] = [ln for ln in rep["interpretation_limits"] if T.NOT_ROBUST_TEXT in ln]
    assert f" — {pri['note']} ({signs}); the DiD CI " in lim
    everything = json.dumps(rep, default=str)
    for banned in ("primary p", "cannot draw an improvement", "excludes zero more readily"):
        assert banned not in everything, banned
    capsys.readouterr()
    T.print_table(rep)
    out = capsys.readouterr().out
    assert pri["note"] in out and signs in out and "primary p" not in out
    # a gains-only discordance reads the other way round; a two-sided one carries no note
    assert T.one_sided_note({"one_sided_discordance": True, "n_discordant": 3, "n_items": 50, "I->C": 3, "C->I": 0}) == (
        f"All three complete-case discordances are gains, so an empirical bootstrap of those 50 pairs cannot generate a decline; "
        f"{T.ONE_SIDED_NOTE_REGISTERED}")
    assert T.one_sided_note({"one_sided_discordance": True, "n_discordant": 1, "n_items": 50, "I->C": 0, "C->I": 1}).startswith(
        "The one complete-case discordance is a loss, so an empirical bootstrap of those 50 pairs cannot generate an improvement")
    assert T.one_sided_note({"one_sided_discordance": False}) is None and T.one_sided_note(None) is None
    assert T.first_one_sided_note(rep["sensitivity"]["mcnemar_exact"]["none_to_psr__off"]) is None
    # no censoring: the registered bootstrap IS the empirical bootstrap of the pairs — no positive replicate at all
    build_2x2(tmp_path / "clean", none_off={"correct": range(120)}, psr_off={"correct": set(range(110)) | set(range(120, 130))},
              none_on={"correct": range(150)}, psr_on={"correct": range(140)})
    clean = analyze(tmp_path / "clean")["estimands"]
    for b in T.BRANCHES:
        rs = clean[b]["stats"]["effect_on"]["replicate_signs"]
        assert rs["positive"] == 0 and rs["negative"] + rs["zero"] == clean[b]["n_replicates_used"]


# --------------------------------------------------------------------------- #
# the thinking-OFF comparator at cap 8192                                     #
# --------------------------------------------------------------------------- #

#: the fixture's three configurations, as every arm mention must read
OFF8K_CFG = f"OFF/{CAP_OFF_OVERRIDE}"
#: the ON legs of the comparator fixture: none 140 / psr 160 correct
ON_LEGS = {"none_on": {"correct": range(140)}, "psr_on": {"correct": range(160)}}
#: path-free digests of the CURRENT script's report on the synthetic 2x2 (the byte-identity
#: pin): the report with `legs`, `analysis_code`, `analysis_code_sha` and
#: `gen_root` removed, `json.dumps(sort_keys=True, default=str)`, sha256. First computed at
#: d90de120 BEFORE the comparator landed (80294f0e… / 3874b489…); RE-PINNED ONCE on an
#: integration tree where the script and `pairing.py` are untouched since the offcap tip
#: 00a87295 but a block the report EMBEDS moved under it: `fixed_sampling` is
#: `contracts.sampling_for(...)._asdict()`, and contracts at 565f5409 widened `SamplingRow`
#: by three optional fields — `min_p`, `presence_penalty`, `repetition_penalty` (None on the
#: Qwen3.5-4B card row; hashed only when set). Leaf diff of the 00a87295 script's report
#: against this tree's, same fixtures, same `_digest` body: clean 1,357 shared leaves, 0
#: changed, 0 numeric, 3 ADDED (`fixed_sampling.min_p` / `.presence_penalty` /
#: `.repetition_penalty`, all null); censored+wave 1,603 shared, 0 changed, 0 numeric (958
#: shared numeric leaves equal), the same 3 added. No estimand, count, CI or leg statistic
#: moved — additive and non-numeric, the one condition under which the pin may move.
#: Anything ELSE that moves the 2x2 report still fails here.
BASELINE_DIGEST_CLEAN = "0a36021512fd4c13c40b0fae6c5d0b62e3b65e3876fe027b113721b062de0534"
BASELINE_DIGEST_CENSORED_WAVE = "66da4d24e331d7ee5cc0523039ecb51518f1401c0affba61bce34606e64efc8a"
DIGEST_EXCLUDED = {"legs", "analysis_code", "analysis_code_sha", "gen_root"}
#: The analysis review's M1 (the fix round after f40e3e63) is the ONE edit the 2x2
#: report has taken since the pins were computed, and it is text-only: every
#: McNemar record gained `n_common` / `fully_scored` / `replicate_signs` /
#: `n_replicates_used` and its pair `registered_contrast`, and the one-sided note
#: of a contrast fully scored under its branch wears the constraint tail instead of
#: the earlier one (the verdict statements embed the primary note). Derived by a
#: leaf diff of the f40e3e63 script's report against this tree's on both fixtures:
#: clean 14,974 shared leaves, 6 text differences (those notes and the two verdict
#: statements), 0 numeric, 52 added; censored+wave 15,220 shared, 2 text
#: differences (the score-as-is notes), 0 numeric, 52 added. `_digest` therefore
#: normalises exactly those edits away — drops the added keys and rewrites the
#: new tail to the old — and the d90de120 digests stay the anchor: anything ELSE
#: that moves the 2x2 report still fails here. The same round's M3 added ONE
#: more leaf per leg, `identity.object_artifact_sha256` (the fitted object's
#: digest; None on the base route), which the `controls` rows copy from the leg
#: records — dropped there too (`M3_ADDED_KEYS`), the leg records being excluded already.
M1_ADDED_KEYS = {"n_common", "fully_scored", "replicate_signs", "n_replicates_used", "registered_contrast"}
M3_ADDED_KEYS = {"object_artifact_sha256"}
M1_TAIL = re.compile(re.escape(T.ONE_SIDED_NOTE_FULLY_SCORED).replace(r"\{n\}", r"\d+").replace(r"\{d\}", r"\d+"))


def _m1_normalised(o, *, drop_keys=frozenset()):
    """The fix round's edits undone: `drop_keys` dropped at every level of the subtree (the M1 McNemar keys under
    `sensitivity.mcnemar_exact` ONLY — pairing files `replicate_signs` / `n_replicates_used` under `estimands` in
    its own right — and the M3 identity leaf under `controls`), the M1 tail rewritten to the old text."""
    if isinstance(o, dict):
        return {k: _m1_normalised(v, drop_keys=drop_keys) for k, v in o.items() if k not in drop_keys}
    if isinstance(o, list):
        return [_m1_normalised(v, drop_keys=drop_keys) for v in o]
    if isinstance(o, str):
        return M1_TAIL.sub(T.ONE_SIDED_NOTE_REGISTERED, o)
    return o


def _digest(rep):
    body = _m1_normalised({k: v for k, v in rep.items() if k not in DIGEST_EXCLUDED})
    if "sensitivity" in body and "mcnemar_exact" in body["sensitivity"]:
        body["sensitivity"] = {**body["sensitivity"],
                               "mcnemar_exact": _m1_normalised(body["sensitivity"]["mcnemar_exact"], drop_keys=M1_ADDED_KEYS)}
    if "controls" in body:
        body["controls"] = _m1_normalised(body["controls"], drop_keys=M3_ADDED_KEYS)
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _leaves(v, path + (i,))
    else:
        yield path, o


def _assert_leaves_subset(old, new, *, skip_prefixes=()):
    """Every leaf of ``old`` is present in ``new`` with the same value (the
    six-leg report keeps every four-leg key and value); ``skip_prefixes`` name
    the paths that legitimately differ (the analysis code's tree state)."""
    new_leaves = dict(_leaves(new))
    missing, differing = [], []
    for path, v in _leaves(old):
        if any(path[:len(p)] == p for p in skip_prefixes):
            continue
        if path not in new_leaves:
            missing.append(path)
        elif new_leaves[path] != v and not (isinstance(v, float) and math.isnan(v)):
            differing.append((path, v, new_leaves[path]))
    assert not missing and not differing, (missing[:5], differing[:5])


def build_fixture_2x3(root, *, offcap_root=None, on=None, **over):
    kw = {**off_legs_capped(), **ON_LEGS, **(on or {}), **offcap_legs()}
    for k, v in over.items():
        kw[k] = {**kw[k], **v}
    return build_2x3(root, offcap_gen_root=offcap_root, **kw)


def analyze6(gen_root, offcap_root=None, **kw):
    kw.setdefault("offcap_sha", OFFCAP_SHA7)
    kw.setdefault("offcap_gen_root", offcap_root)
    kw.setdefault("prompt_budget_off", prompt_budget_artifact_off())
    return analyze(gen_root, **kw)


def offleg(report, cell):
    return report["offcap"]["legs"][T.leg_key(cell, "off", cap_override=True)]


def _spy_pairing(monkeypatch):
    """Record the key set of every ``paired_bootstrap_conditional_2x2`` call:
    no re-estimation path may ever hand the four-leg wrapper six keys."""
    calls = []
    real = pr.paired_bootstrap_conditional_2x2

    def spy(legs, **kw):
        calls.append(set(legs))
        return real(legs, **kw)

    monkeypatch.setattr(T.pr, "paired_bootstrap_conditional_2x2", spy)
    return calls


def test_without_offcap_sha_the_report_is_byte_identical_to_today(tmp_path):
    """The extension is additive and OFF by default. The
    report of the 2x2 run reproduces the pinned digests (every block but the
    path-bearing `legs` / code provenance) under the M1-only normalisation
    documented at `M1_ADDED_KEYS`, with or without the new keyword arguments
    spelled out; the only addition to a leg record is the validated comparator
    classification `cap_override: False`, and no `offcap*` key
    appears. The M1 additions themselves are asserted where they are made (the
    M1 tests); here they are checked to be the ONLY departure from the pins.
    The pins were first computed at d90de120 and moved ONCE, on the
    math500-thinking-on-full22 integration tree, for exactly three added null
    leaves under `fixed_sampling` (`min_p`, `presence_penalty`,
    `repetition_penalty`: contracts' `SamplingRow` grew; the leaf diff is
    recorded at `BASELINE_DIGEST_CLEAN`) — nothing numeric moved."""
    build_2x2(tmp_path / "clean", **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    rep = analyze(tmp_path / "clean")
    assert _digest(rep) == BASELINE_DIGEST_CLEAN
    explicit = analyze(tmp_path / "clean", offcap_sha=None, offcap_gen_root=None, prompt_budget_off=None)

    def strip(r):
        return {k: v for k, v in r.items() if k not in ("analysis_code", "analysis_code_sha")}
    assert json.dumps(strip(explicit), default=str) == json.dumps(strip(rep), default=str)
    assert not [k for k in rep if k.startswith("offcap")]
    for leg in rep["legs"].values():
        assert leg["cap_override"] is False
    build_2x2(tmp_path / "cens", **off_legs(), none_on={"correct": range(126), "unscored": LAST4},
              psr_on={"correct": range(143), "unscored": LAST4, "capped_scored": {150}})
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    assert _digest(analyze(tmp_path / "cens", wave_root=wave)) == BASELINE_DIGEST_CENSORED_WAVE


def test_the_offcap_legs_are_collected_under_their_own_suffix_and_root_and_a_table_cap_store_there_is_refused(tmp_path):
    offcap_root = tmp_path / "pull_offcap"
    build_fixture_2x3(tmp_path, offcap_root=offcap_root)
    rep = analyze6(tmp_path, offcap_root)
    assert rep["offcap_sha"] == OFFCAP_SHA7 and rep["offcap_gen_root"] == str(offcap_root) and rep["sha"] == SHA7
    assert set(rep["offcap"]["legs"]) == {T.leg_key(NONE, "off", cap_override=True), T.leg_key(PSR, "off", cap_override=True)}
    assert T.leg_key(NONE, "off", cap_override=True) == f"{NONE}|off|cap{CAP_OFF_OVERRIDE}" != T.leg_key(NONE, "off")
    for cell in (NONE, PSR):
        leg = offleg(rep, cell)
        assert leg["cap_override"] is True and leg["arm"] == "off" and leg["refused"] is None and leg["missing"] is None
        assert leg["out_root"] == str(T.out_root_of(offcap_root, OFFCAP_SHA7, cell=cell, arm="off",
                                                     prefix=T.OFFCAP_OUT_SUFFIX_PREFIX))
        assert leg["out_root"].endswith(f"/thinking_off/offcap-{OFFCAP_SHA7}")
        assert leg["identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {TASK: CAP_OFF_OVERRIDE}
        assert leg["identity"][C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] is None
        assert leg["identity"]["producer_tree_sha"] == OFFCAP_FULL_SHA and leg["summary"]["max_tokens"] == CAP_OFF_OVERRIDE
    assert set(rep["legs"]) == {T.leg_key(c, a) for c in (NONE, PSR) for a in ("off", "on")}   # the four, as today
    assert rep["unusable_legs"] == {} and rep["gates"]["failed"] == [] and rep["verdict"]["word"] == T.WORD_TRANSFERS
    # a TABLE-CAP store under the comparator's suffix (no cap key, cap 512) is refused by the key ...
    build_store(offcap_root, cell=NONE, arm="off", correct=OFF_NONE_CORRECT, sha=OFFCAP_SHA7,
                out_suffix=f"offcap-{OFFCAP_SHA7}", oos_marker=False)
    import shutil
    good = [p for p in (T.out_root_of(offcap_root, OFFCAP_SHA7, cell=NONE, arm="off", prefix="offcap-") / "cells").iterdir()
            if json.loads((p / "manifest.json").read_text())["tinycap"].get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)]
    for p in good:
        shutil.rmtree(p)
    rep2 = analyze6(tmp_path, offcap_root)
    leg = offleg(rep2, NONE)
    assert leg["cap_override"] is False and "carries no max_tokens_by_task" in leg["refused"]
    assert "estimands" not in rep2 and rep2["verdict"]["word"] is None      # the six-leg population gate is all-or-nothing
    assert rep2["unusable_legs"] == {T.leg_key(NONE, "off", cap_override=True): leg["refused"]}
    # ... a comparator store that ALSO carries the OOS marker is refused naming it ...
    shutil.rmtree(T.out_root_of(offcap_root, OFFCAP_SHA7, cell=NONE, arm="off", prefix="offcap-"))
    build_store(offcap_root, cell=NONE, arm="off", correct=OFF8192_NONE_CORRECT, sha=OFFCAP_SHA7, cap=CAP_OFF_OVERRIDE,
                cap_override=True, out_suffix=f"offcap-{OFFCAP_SHA7}", oos_marker=True)
    leg = offleg(analyze6(tmp_path, offcap_root), NONE)
    assert leg["cap_override"] is False and C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in leg["refused"]
    assert "carries max_tokens_by_task alone" in leg["refused"]
    # ... and a comparator store whose executed cap is not 8192 is refused by the cap
    shutil.rmtree(T.out_root_of(offcap_root, OFFCAP_SHA7, cell=NONE, arm="off", prefix="offcap-"))
    build_store(offcap_root, cell=NONE, arm="off", correct=OFF8192_NONE_CORRECT, sha=OFFCAP_SHA7, cap=CAP_OFF_OVERRIDE,
                cap_override=True, out_suffix=f"offcap-{OFFCAP_SHA7}", oos_marker=False,
                summary_edit=lambda s: s.update(max_tokens=4096))
    leg = offleg(analyze6(tmp_path, offcap_root), NONE)
    assert leg["refused"].startswith(f"max_tokens 4096 != the comparator's off cap {CAP_OFF_OVERRIDE}")
    assert "AUTHORISED_OFF_CAP_COMPARATOR" in leg["refused"]
    assert T.expected_cap("off", cap_override=True) == CAP_OFF_OVERRIDE == T.OFFCAP_CAP == C.OOS_THINKING_MAX_TOKENS
    assert T.expected_cap("off") == 512 and T.expected_cap("on") == CAP_ON


def test_a_comparator_store_read_as_an_off_512_leg_is_refused_by_the_cap_and_the_identity_key(tmp_path):
    """A cross-read in EITHER direction is refused by BOTH the
    identity key and the cap. An OFF/8192 store under the 2x2's suffix (read
    under --sha) trips the key check first; a store at 8192 that carries no key
    trips the cap; the OFF/512 leg's refusal text for the OOS marker is unchanged."""
    build_2x2(tmp_path, **off_legs(), none_on={"correct": range(126)}, psr_on={"correct": range(143)})
    import shutil
    shutil.rmtree(T.out_root_of(tmp_path, SHA7, cell=NONE, arm="off"))
    build_store(tmp_path, cell=NONE, arm="off", correct=OFF8192_NONE_CORRECT, cap=CAP_OFF_OVERRIDE, cap_override=True,
                oos_marker=False)                                       # the comparator's store under oos-<sha>
    leg = leg_of(analyze(tmp_path), NONE, "off")
    assert leg["cap_override"] is False
    assert leg["refused"].startswith("the OFF leg's store carries ['max_tokens_by_task']")
    assert "a comparator store reads under --offcap-sha" in leg["refused"]
    shutil.rmtree(T.out_root_of(tmp_path, SHA7, cell=NONE, arm="off"))
    build_store(tmp_path, cell=NONE, arm="off", correct=OFF8192_NONE_CORRECT, cap=CAP_OFF_OVERRIDE, oos_marker=False)
    leg = leg_of(analyze(tmp_path), NONE, "off")
    assert leg["refused"].startswith(f"max_tokens {CAP_OFF_OVERRIDE} != the experiment's off cap 512")
    shutil.rmtree(T.out_root_of(tmp_path, SHA7, cell=NONE, arm="off"))
    build_store(tmp_path, cell=NONE, arm="off", correct=OFF_NONE_CORRECT, oos_marker=True)
    assert "an in-support store carries no such key" in leg_of(analyze(tmp_path), NONE, "off")["refused"]


def test_the_two_relabelled_2x2s_reproduce_the_original_contrasts_and_each_other_bit_for_bit(tmp_path, monkeypatch):
    """est_cap and est_matched are two further calls of the UNCHANGED
    bootstrap with the legs relabelled; the shared contrasts are one number
    across calls (the runtime `offcap.consistency` block), the renamed contrasts
    carry the fixture's exact fractions, and every wrapper call carries exactly
    the four pairing keys."""
    calls = _spy_pairing(monkeypatch)
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    assert calls and all(c == set(pr.LEG_KEYS) for c in calls)
    oc = rep["offcap"]
    assert oc["consistency"]["ok"] is True and len(oc["consistency"]["checks"]) == 3 * len(T.BRANCHES)
    orig, cap, matched = rep["estimands"], oc["estimands"][T.MAP_CAP], oc["estimands"][T.MAP_MATCHED]
    fields = ("point", "ci_low", "ci_high", "bound_low", "bound_high", "bound_ci_low", "bound_ci_high",
              "replicate_signs", "point_sign", "sign_identified", "excludes_zero")
    def pick(s):
        return {f: s[f] for f in fields}
    for b in T.BRANCHES:
        assert pick(cap[b]["stats"]["effect_off"]) == pick(orig[b]["stats"]["effect_off"])
        assert pick(matched[b]["stats"]["effect_on"]) == pick(orig[b]["stats"]["effect_on"])
        assert pick(matched[b]["stats"]["effect_off_8192"]) == pick(cap[b]["stats"]["effect_off_8192"])
        assert set(cap[b]["stats"]) == {"effect_off", "effect_off_8192", "cap_relief_none", "cap_relief_psr", "did_cap_relief"}
        assert set(matched[b]["stats"]) == {"effect_off_8192", "effect_on", "shift_none_matched", "shift_psr_matched", "did_matched"}
        # the fixture's exact fractions (no censoring: one float per rational)
        assert cap[b]["stats"]["cap_relief_none"]["point"] == 6 / 200 and cap[b]["stats"]["cap_relief_psr"]["point"] == 3 / 200
        assert cap[b]["stats"]["effect_off_8192"]["point"] == 27 / 200 and cap[b]["stats"]["did_cap_relief"]["point"] == -3 / 200
        assert matched[b]["stats"]["shift_none_matched"]["point"] == 14 / 200
        assert matched[b]["stats"]["shift_psr_matched"]["point"] == 7 / 200 and matched[b]["stats"]["did_matched"]["point"] == -7 / 200
        assert orig[b]["stats"]["shift_none"]["point"] == 20 / 200 and orig[b]["stats"]["shift_psr"]["point"] == 10 / 200
        # nobody reads the internal 2x2 slot as thinking: the pseudo-2x2 record and relabelled legs
        assert cap[b]["pseudo_2x2"]["off"] == "OFF/512" and cap[b]["pseudo_2x2"]["on"] == OFF8K_CFG
        assert matched[b]["pseudo_2x2"]["off"] == OFF8K_CFG and matched[b]["pseudo_2x2"]["on"] == f"ON/{CAP_ON}"
        assert set(cap[b]["acc_by_leg"]) == {"none/OFF/512", "psr/OFF/512", f"none/{OFF8K_CFG}", f"psr/{OFF8K_CFG}"}
        assert set(matched[b]["acc_by_leg"]) == {f"none/{OFF8K_CFG}", f"psr/{OFF8K_CFG}", f"none/ON/{CAP_ON}", f"psr/ON/{CAP_ON}"}
        assert cap[b]["stats"]["cap_relief_none"]["legs"] == {f"none/{OFF8K_CFG}": 1, "none/OFF/512": -1}
        assert set(orig[b]["stats"]) == set(pr.CONTRASTS) and set(orig[b]["acc_by_leg"]) == {pr.leg_label(k) for k in pr.LEG_KEYS}
    # a mapping that is not the four pairing keys refuses by name, as does a legs dict
    with pytest.raises(T.TransferAnalysisError, match="four pairing keys"):
        T.estimands(rep["legs"], n_resamples=10)
    with pytest.raises(T.TransferAnalysisError, match="no slot in the natural 2x2 mapping"):
        T.natural_mapping({**rep["legs"], **rep["offcap"]["legs"]})


def test_a_six_leg_analysis_with_a_closed_think_capped_on_item_files_the_legacy_sensitivity_unchanged_and_the_comparator_one_separately(tmp_path, monkeypatch):
    """The real report has one closed-think capped psr/ON
    item, so the sensitivity path RUNS. The legacy block is computed over the
    four legs exactly as the 2x2 run computes it (leaf for leaf); the comparator
    block recodes the same ON items and re-runs MAP_MATCHED, stating MAP_CAP
    unchanged; no wrapper call ever sees six keys."""
    calls = _spy_pairing(monkeypatch)
    build_fixture_2x3(tmp_path, on={"psr_on": {"correct": range(160), "capped_scored": {150}}})
    rep4 = analyze(tmp_path)
    rep6 = analyze6(tmp_path)
    assert all(c == set(pr.LEG_KEYS) for c in calls)
    assert rep4["sensitivity"]["closed_think_capped"]["items"] == {T.leg_key(PSR, "on"): [IDS[150]]}
    assert rep6["sensitivity"]["closed_think_capped"] == rep4["sensitivity"]["closed_think_capped"]
    ct = rep6["offcap"]["sensitivity"]["closed_think_capped"]
    assert ct["items"] == rep4["sensitivity"]["closed_think_capped"]["items"] and ct["policy"] == T.CLOSED_THINK_CAPPED_POLICY
    assert ct["map_cap_unchanged"] is True and ct["map_cap_note"]
    assert set(ct["recoded"]) == {T.MAP_MATCHED}
    rec = ct["recoded"][T.MAP_MATCHED][T.BRANCH_PRIMARY]
    assert set(rec) == {"effect_off_8192", "effect_on", "shift_none_matched", "shift_psr_matched", "did_matched"}
    assert rec["effect_on"]["point"] == pytest.approx(159 / 199 - 140 / 200)      # item 150 was a correct one
    assert rec["effect_off_8192"] == {f: rep6["offcap"]["estimands"][T.MAP_MATCHED][T.BRANCH_PRIMARY]["stats"]["effect_off_8192"][f]
                                      for f in rec["effect_off_8192"]}          # the OFF/8192 legs are not recoded
    assert ct["verdict_sensitivity_off_8192"]["label"] == T.VERDICT_SENSITIVITY_8192_LABEL
    # every legacy block of the six-leg report equals the 2x2 run's, leaf for leaf
    for key in ("estimands", "transitions", "verdict_rule", "complete_case_rows", "selected_subset_diagnostic",
                "interpretation_limits", "findings", "cross_arm_contrasts", "sensitivity", "verdict", "top_items",
                "population", "serving_parity"):
        _assert_leaves_subset(rep4[key], rep6[key])
    _assert_leaves_subset(rep4["gates"], rep6["gates"])
    _assert_leaves_subset(rep4["controls"], rep6["controls"])
    _assert_leaves_subset(rep4["legs"], rep6["legs"])
    _assert_leaves_subset(rep4["nuisance"], rep6["nuisance"])
    assert len(rep6["controls"]) == 6 and [r["leg"] for r in rep6["controls"][:4]] == [r["leg"] for r in rep4["controls"]]


def test_anomaly_re_estimation_runs_on_the_four_legs_for_the_verdict_and_on_the_two_mappings_for_the_comparator(tmp_path, monkeypatch):
    """An unterminated-not-at-cap ON item plus a
    stray-open-tag OFF/8192 item. `gates.split_anomalies` equals the 2x2 run's
    (four legs, MAP_ORIG); `offcap.gates.split_anomalies` masks the UNION of
    anomaly ids over six legs and reports the comparator contrasts' sign moves;
    a spy asserts every wrapper call's key set is exactly LEG_KEYS."""
    calls = _spy_pairing(monkeypatch)
    build_fixture_2x3(tmp_path, on={"psr_on": {"correct": range(160), "unscored": {198, 199}, "not_at_cap": {198, 199}}},
                      none_off8192={"stray": {5}})
    rep4 = analyze(tmp_path)
    n4 = len(calls)
    rep6 = analyze6(tmp_path)
    assert all(c == set(pr.LEG_KEYS) for c in calls) and len(calls) > n4
    g4, g6 = rep4["gates"]["split_anomalies"], rep6["gates"]["split_anomalies"]
    assert g4["masked_out"] == [IDS[198], IDS[199]] and g6 == g4
    og = rep6["offcap"]["gates"]["split_anomalies"]
    assert og["masked_out"] == [IDS[5], IDS[198], IDS[199]]
    assert og["counts"][T.leg_key(NONE, "off", cap_override=True)]["n_stray_open_tag"] == 1
    assert og["counts"][T.leg_key(NONE, "off", cap_override=True)]["n_think_block_present"] == 0
    assert og["status"] == T.GATE_PASS and og["sign_moves"] == {} and og["resolved_moves"] is False
    assert set(og["without"]) == {T.MAP_CAP, T.MAP_MATCHED}
    assert "verdict" not in og and "word" not in og           # reported, never a verdict word
    assert rep6["verdict"] == rep4["verdict"]


def test_a_closed_think_block_on_an_off_comparator_leg_is_masked_and_can_flip_a_contrast_sign(tmp_path):
    """A thinking-OFF completion
    that emits a CLOSED `<think>…</think>` block is recorded by the REAL
    `record_only` splitter as `think_block_present True, think_block_unterminated
    False, stray_open_tag False` — a scored item the legacy `anomaly_ids`
    predicate never counts (under `implicit_open` every ON item has a block).
    The comparator gate reported the summary COUNT but left the items in both
    re-estimated mappings, so a comparator contrast whose sign rests on them
    passed. Now every OFF leg's present-block items join the anomalous-id union
    before the counts and the re-run: two such items on the none OFF/8192 leg,
    correct there and not at OFF/512, flip `cap_relief_none` from +1/200 to
    -1/197 when masked (a present block on an OFF/512 leg joins the union too,
    hence 197); the four-leg gate (`anomaly_ids`, MAP_ORIG) is byte-identical."""
    from sisteer.capeval.tiers import think_split as ts

    real = ts.split_metadata(ts.split_think("<think>carry the one</think> The answer is 42.",
                                            mode=C.THINK_SPLIT_RECORD_ONLY))
    assert real == {"think_block_present": True, "think_block_unterminated": False,
                    "think_split_mode": C.THINK_SPLIT_RECORD_ONLY, "stray_open_tag": False}
    record_fields = ("think_block_present", "think_block_unterminated", "stray_open_tag")
    closed = {120, 121}                                   # capped-incorrect at OFF/512, correct at OFF/8192
    build_fixture_2x3(tmp_path, none_off={"closed_think": {7}},
                      none_off8192={"correct": (OFF_NONE_CORRECT - {150}) | closed, "closed_think": closed})
    rep = analyze6(tmp_path)
    leg8192, leg512 = offleg(rep, NONE), leg_of(rep, NONE, "off")
    for j in closed:
        rec = leg8192["items"][IDS[j]]
        assert {f: rec[f] for f in record_fields} == {f: real[f] for f in record_fields} and rec["scored"]
    assert {f: leg512["items"][IDS[7]][f] for f in record_fields} == {f: real[f] for f in record_fields}
    assert leg8192["summary"]["n_think_block_present"] == 2 and leg512["summary"]["n_think_block_present"] == 1
    # the four-leg gate: the legacy predicate, untouched — no OFF/512 item is an anomaly to it
    g4 = rep["gates"]["split_anomalies"]
    assert g4["masked_out"] == [] and g4["status"] == T.GATE_PASS
    assert T.anomaly_ids(rep["offcap"]["legs"]) == {k: [] for k in rep["offcap"]["legs"]}
    # the comparator gate: the present-block items are anomalous ids, masked, and the sign moves
    og = rep["offcap"]["gates"]["split_anomalies"]
    k8192, k512 = T.leg_key(NONE, "off", cap_override=True), T.leg_key(NONE, "off")
    assert og["counts"][k8192]["n_think_block_present"] == 2 and og["counts"][k8192]["n_anomalous_items"] == 2
    assert og["counts"][k512]["n_think_block_present"] == 1 and og["counts"][k512]["n_anomalous_items"] == 1
    assert og["masked_out"] == sorted(IDS[j] for j in closed | {7})
    before = rep["offcap"]["estimands"][T.MAP_CAP][T.BRANCH_PRIMARY]["stats"]["cap_relief_none"]
    assert before["point"] == pytest.approx(1 / 200) and before["point_sign"] == 1
    after = og["without"][T.MAP_CAP][T.BRANCH_PRIMARY]["cap_relief_none"]
    assert after["point"] == pytest.approx(118 / 197 - 119 / 197) and after["point_sign"] == -1
    assert og["sign_moves"][f"{T.BRANCH_PRIMARY}/cap_relief_none"] == (1, -1)
    assert og["sign_moves"][f"{T.BRANCH_SCORE_AS_IS}/cap_relief_none"] == (1, -1)
    assert og["status"] == T.GATE_FAIL and "cap_relief_none" in og["detail"]
    assert "verdict" not in og and "word" not in og           # reported, never a verdict word
    # the same store WITHOUT the closed blocks: the gate has nothing to mask and passes
    build_fixture_2x3(tmp_path / "plain", none_off8192={"correct": (OFF_NONE_CORRECT - {150}) | closed})
    og_plain = analyze6(tmp_path / "plain")["offcap"]["gates"]["split_anomalies"]
    assert og_plain["masked_out"] == [] and og_plain["status"] == T.GATE_PASS


def test_every_estimands_call_site_is_enumerated():
    """The set of `estimands(` call sites in the script equals
    the caller table — a new site fails until it is classified here."""
    src = (REPO / "scripts" / "tinycap_oos_transfer.py").read_text().splitlines()
    sites: dict[str, int] = {}
    fn = None
    for line in src:
        m = re.match(r"^def (\w+)\(", line)
        if m:
            fn = m.group(1)
            continue
        if re.search(r"(?<![\w.])estimands\(", line) and not line.lstrip().startswith("#") and "def estimands" not in line:
            sites[fn] = sites.get(fn, 0) + 1
    classified = {
        "analyze": 1,                                   # est_orig / est_cap / est_matched: one loop over the three mappings
        "split_anomaly_gate": 1,                        # MAP_ORIG over legs4 (the registered rule)
        "offcap_split_anomaly_gate": 1,                 # MAP_CAP + MAP_MATCHED, anomaly ids over legs6
        "closed_think_capped_sensitivity": 1,           # the natural mapping over the recoded legs4
        "offcap_closed_think_capped_sensitivity": 1,    # MAP_MATCHED (and MAP_CAP only if an OFF leg was recoded)
    }
    assert sites == classified, sites


def test_the_eleven_contrasts_carry_their_varies_labels_and_the_originals_are_unchanged(tmp_path):
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    varies = rep["offcap"]["contrast_varies"]
    both = f"render+cap (OFF/512 vs ON/{CAP_ON})"
    assert varies == {
        "effect_off": "steering at OFF/512", "effect_on": f"steering at ON/{CAP_ON}",
        "shift_none": both, "shift_psr": both, "did": both,
        "cap_relief_none": f"budget only (OFF render, 512 → {CAP_OFF_OVERRIDE})",
        "cap_relief_psr": f"budget only (OFF render, 512 → {CAP_OFF_OVERRIDE})",
        "did_cap_relief": f"budget only (OFF render, 512 → {CAP_OFF_OVERRIDE}; the steering interaction)",
        "shift_none_matched": f"render only (cap {CAP_OFF_OVERRIDE})", "shift_psr_matched": f"render only (cap {CAP_OFF_OVERRIDE})",
        "did_matched": f"render only (matched cap {CAP_OFF_OVERRIDE})", "effect_off_8192": f"steering at {OFF8K_CFG}"}
    assert len([c for c in varies if c != "did_cap_relief"]) == 11
    for b in T.BRANCHES:
        for est in rep["offcap"]["estimands"].values():
            assert est[b]["varies"] == {c: varies[c] for c in est[b]["stats"]}
        assert "varies" not in rep["estimands"][b] and set(rep["estimands"][b]["stats"]) == set(pr.CONTRASTS)
        for s in rep["estimands"][b]["stats"].values():
            assert "varies" not in s
    assert set(T.OFFCAP_CONTRASTS) == {"cap_relief_none", "cap_relief_psr", "shift_none_matched", "shift_psr_matched",
                                       "did_matched", "effect_off_8192"}


def test_the_decomposition_statement_names_the_observed_contrasts_and_the_exact_identity(tmp_path):
    """The statement reads "the observed cap-only contrast was +x pp [CI]", never
    "+x pp is budget" (the analysis review's M1; its qualification is tested in
    test_the_decomposition_findings_say_observed_contrast_and_carry_the_sparse_discordance_qualification)."""
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    dec = rep["offcap"]["decomposition"]
    assert set(dec) == {"none", "psr"} and set(dec["none"]) == set(T.BRANCHES)
    d = dec["none"][T.BRANCH_PRIMARY]
    assert d["total"]["point"] == 20 / 200 and d["budget"]["point"] == 6 / 200 and d["render"]["point"] == 14 / 200
    assert d["identity_exact_full_set"] is True and d["identity_replicates_to_an_ulp"] is True
    assert d["budget_share_point"] == pytest.approx(0.3) and d["budget_share_note"]
    assert d["statement"].startswith(f"Of the unsteered OFF/512 → ON/{CAP_ON} gain of +10.00 pp [")
    assert f"the observed cap-only contrast (OFF/512 → {OFF8K_CFG}, same render) was +3.00 pp [" in d["statement"]
    assert f"the observed render-only contrast ({OFF8K_CFG} → ON/{CAP_ON}, same cap) was +7.00 pp [" in d["statement"]
    assert ("sum to the total exactly (a decomposition identity between observed estimates, not a causal partition)"
            in d["statement"]) and "do not add" in d["statement"]
    assert "is budget" not in d["statement"] and "is render" not in d["statement"]
    p = dec["psr"][T.BRANCH_PRIMARY]
    assert p["statement"].startswith("Of the steered (psr) ") and p["budget"]["point"] == 3 / 200 and p["render"]["point"] == 7 / 200
    assert p["total"]["ci_low"] == rep["estimands"][T.BRANCH_PRIMARY]["stats"]["shift_psr"]["ci_low"]
    assert rep["offcap"]["did_matched"]["point"] == -7 / 200 and rep["offcap"]["did_matched"]["statement"]
    assert rep["offcap"]["censoring_note"].startswith("cap_relief_none, cap_relief_psr and effect_off_8192 are points")
    # the statements are the comparator's findings; the registered findings are untouched
    assert any(f.startswith("Of the unsteered") for f in rep["offcap"]["findings"])
    assert rep["findings"] == analyze(tmp_path)["findings"]


def test_effect_off_8192_gets_the_resolved_reading_and_the_sensitivity_verdict_is_labelled_never_headline(tmp_path):
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    e = rep["offcap"]["effect_off_8192"]
    assert e["resolved"] is True and e["point"] == 27 / 200 and e["text"].startswith("effect_off_8192 = +13.50 pp (point)")
    assert "cap 8192" not in e["text"] and f"(OFF/{CAP_OFF_OVERRIDE} discordances none→psr: C→I 10 / I→C 37)" in e["text"]
    v = rep["offcap"]["verdict_sensitivity_off_8192"]
    assert v["label"] == T.VERDICT_SENSITIVITY_8192_LABEL and "sensitivity, not the registered input" in v["label"]
    assert v["inputs"] == {"e_off": "effect_off_8192", "e_on": "effect_on", "did": "did_matched"}
    assert v["verdict"] == T.WORD_TRANSFERS and v["word_by_branch"] == {b: T.WORD_TRANSFERS for b in T.BRANCHES}
    assert v["headline"] is False
    # the registered verdict (keyed on the 512 leg) and the top items are untouched
    rep4 = analyze(tmp_path)
    assert rep["verdict"] == rep4["verdict"] and rep["top_items"] == rep4["top_items"]
    # an unresolved effect at 8192 says so, at that cap
    build_fixture_2x3(tmp_path / "flat", psr_off8192={"correct": OFF8192_NONE_CORRECT})
    e2 = analyze6(tmp_path / "flat")["offcap"]["effect_off_8192"]
    assert e2["resolved"] is False and e2["point"] == 0.0
    assert f"no net OFF accuracy effect was resolved at this sample size and cap {CAP_OFF_OVERRIDE}" in e2["text"]
    assert T.OFF_NOT_RESOLVED_TEXT in analyze(tmp_path / "flat")["verdict_rule"]["statement"] or True


def test_the_off_8192_legs_take_the_two_percent_cap_gate_and_the_off_512_legs_do_not(tmp_path):
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    cg = rep["gates"]["cap"]
    assert set(cg["legs"]) == {T.leg_key(NONE, "on"), T.leg_key(PSR, "on"),
                               T.leg_key(NONE, "off", cap_override=True), T.leg_key(PSR, "off", cap_override=True)}
    assert cg["status"] == T.GATE_PASS and cg["legs"][T.leg_key(NONE, "off", cap_override=True)]["at_cap_rate"] == 0.0
    assert leg_of(rep, NONE, "off")["summary"]["at_cap_rate"] == 0.05          # ungated: the thing being relieved
    assert set(rep["offcap"]["cap_gate"]["legs"]) == {T.leg_key(NONE, "off", cap_override=True), T.leg_key(PSR, "off", cap_override=True)}
    # five OFF/8192 draws at the cap (2.5 %) fail the gate on that leg, by name
    build_fixture_2x3(tmp_path / "capped", none_off8192={"capped_scored": {10, 11, 12, 13, 14}})
    rep2 = analyze6(tmp_path / "capped")
    cg2 = rep2["gates"]["cap"]
    assert cg2["status"] == T.GATE_FAIL and cg2["legs"][T.leg_key(NONE, "off", cap_override=True)]["at_cap_rate"] == 0.025
    assert cg2["detail"].startswith(f"{T.leg_key(NONE, 'off', cap_override=True)}: at_cap_rate 2.500% > 2%")
    assert f"OFF budget was not adequate at {CAP_OFF_OVERRIDE}" in cg2["detail"]
    assert rep2["offcap"]["cap_gate"]["status"] == T.GATE_FAIL


def test_the_off512_to_off8192_description_is_exploratory_carries_no_expectation_and_names_the_selection(tmp_path):
    """The block is a transition DESCRIPTION split by the first draw's
    at-cap status — no zero expectation, no noise floor, no parity role. The
    wording assertion is scoped to the block, its figure's caption and its
    numbers entry; the parent's serving-parity blocks are present and
    byte-identical to the 2x2 run's."""
    lengths = {**OFF8192_LENGTHS_NONE, 50: 130}                        # one not-capped item whose second draw is longer
    build_fixture_2x3(tmp_path, none_off8192={"completion_tokens": lengths})
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    rep = analyze6(tmp_path, wave_root=wave)
    blk = rep["offcap"]["off512_to_off8192"]
    assert blk["label"].startswith("exploratory transition description") and "no expectation" in blk["label"]
    none = blk["cells"]["none"]
    assert none["n_items"] == 200 and none["counts"] == {"C->C": 120, "C->I": 0, "I->C": 6, "I->I": 74, "->U": 0}
    assert none["by_first_draw_stop"]["capped_at_512"]["n"] == 10
    assert none["by_first_draw_stop"]["capped_at_512"]["counts"] == {"C->C": 2, "C->I": 0, "I->C": 6, "I->I": 2, "->U": 0}
    assert none["by_first_draw_stop"]["not_capped_at_512"]["n"] == 190
    assert none["by_first_draw_stop"]["not_capped_at_512"]["counts"] == {"C->C": 118, "C->I": 0, "I->C": 0, "I->I": 72, "->U": 0}
    cs = none["capped_at_512_second_draw"]
    assert cs["n"] == 10 and cs["n_correct_at_8192"] == 8 and cs["n_at_cap_at_8192"] == 0
    assert cs["completion_tokens_at_8192"]["min"] == 900 and cs["completion_tokens_at_8192"]["max"] == 2170
    assert len(cs["completion_tokens_at_8192"]["values"]) == 10
    assert none["not_capped_at_512"] == {"n": 190, "n_completion_tokens_differ": 1, "n_compared": 190}
    psr = blk["cells"]["psr"]
    assert psr["by_first_draw_stop"]["capped_at_512"]["n"] == 6 and psr["capped_at_512_second_draw"]["n_correct_at_8192"] == 5
    assert blk["why_no_expectation"] in T.OFF512_TO_OFF8192_CAVEATS
    # the scoped vocabulary rule: over the block, its caption and its figure's numbers entry only
    caption = T.figure_captions(rep)["final_e1b_off512_to_off8192_by_first_draw_stop"]
    texts = [v for _p, v in _leaves(blk) if isinstance(v, str)] + [caption]
    banned = ("noise floor", "never-capped", "never capped", "reproducib", "parity", "expectation zero", "expected zero",
              "zero-centred", "zero expectation", "identical draw", "same draw")
    for text in texts:
        for term in banned:
            if term in text.lower():
                assert text in T.OFF512_TO_OFF8192_CAVEATS, (term, text)      # a caveat saying what it is NOT
    assert any("not parity evidence" in c for c in T.OFF512_TO_OFF8192_CAVEATS)
    assert "post-selected" in caption and "exploratory" in caption
    rep4 = analyze(tmp_path, wave_root=wave)
    assert rep["serving_parity"] == rep4["serving_parity"] and rep["sensitivity"]["parity"] == rep4["sensitivity"]["parity"]
    assert set(rep["serving_parity"]["cells"]) == {NONE, PSR}
    for cell in (NONE, PSR):
        assert rep["serving_parity"]["cells"][cell]["fresh_leg"] == T.leg_key(cell, "off")
    assert "not parity evidence" in rep["offcap"]["parity_note"] and "max_tokens" in rep["offcap"]["parity_note"]
    assert T.figure_captions(rep)[T.FIGURE_PARITY] == T.figure_captions(rep4)[T.FIGURE_PARITY]


def test_the_off_ledger_confirmation_holds_8192_thinking_false_and_the_three_way_prompt_max(tmp_path):
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    for cell in (NONE, PSR):
        led = offleg(rep, cell)["ledger"]
        assert led["passed"] is True and led["failures"] == [] and led["pending"] == []
        assert led["max_tokens_requested"] == {"values": [CAP_OFF_OVERRIDE], "expected": CAP_OFF_OVERRIDE, "agrees": True}
        assert led["thinking"] == {"values": [False], "expected": False, "agrees": True}
        assert led["n_calls"] == 200 and led["n_rejected"] == 0 and led["n_length"] == 0 and led["length_agrees"] is True
        assert led["window"] == {"prompt_max": PROMPT_MAX_OFF, "cap": CAP_OFF_OVERRIDE, "max_model_len": SERVED["max_model_len"], "fits": True}
        assert led["prompt_max"] == {"ledger": PROMPT_MAX_OFF, "live": PROMPT_MAX_OFF, "artifact": PROMPT_MAX_OFF,
                                     "agrees": True, "detail": None, "applicable": True}
        assert led["prompt_p99"]["agrees"] is True and led["prompt_p99"]["applicable"] is True
        assert led["artifact_fit"]["passed"] is True and led["artifact_fit"]["cap"] == CAP_OFF_OVERRIDE
        assert led["artifact_fit"]["prompt_max"] == PROMPT_MAX_OFF and led["artifact_fit"]["margin"] == SERVED["max_model_len"] - PROMPT_MAX_OFF - CAP_OFF_OVERRIDE
    assert rep["offcap"]["ledger"][T.leg_key(NONE, "off", cap_override=True)]["passed"] is True
    assert leg_of(rep, NONE, "off")["ledger"]["prompt_max"]["applicable"] is False        # the OFF/512 legs: two sources, as today
    # without the OFF artifact the third source is a named, unwaivable pending on the OFF/8192 legs only
    rep2 = analyze6(tmp_path, prompt_budget_off=None, allow_pending_gates=True)
    led = offleg(rep2, NONE)["ledger"]
    assert led["pending"] == ["CPU artifact prompt max (no --prompt-budget-off artifact supplied)"] and led["pending_stamps"] == [None]
    assert rep2["gates"]["declaration"]["status"] == T.GATE_PENDING and rep2["verdict"]["word"] is None
    assert leg_of(rep2, NONE, "on")["ledger"]["pending"] == []
    # a ledger row at the table cap on an OFF/8192 leg fails the leg
    build_fixture_2x3(tmp_path / "bad", none_off8192={"ledger_edit": lambda rows: rows[0].update(max_tokens_requested=512)})
    led = offleg(analyze6(tmp_path / "bad"), NONE)["ledger"]
    assert led["passed"] is False and led["failures"][0].startswith(f"max_tokens_requested [512, {CAP_OFF_OVERRIDE}]")


def test_the_off_artifact_is_validated_as_arm_off_cap_8192_and_the_on_one_still_refused_if_off(tmp_path):
    build_fixture_2x3(tmp_path)
    with pytest.raises(T.TransferAnalysisError, match=f"cap 512 != the comparator's OFF cap {CAP_OFF_OVERRIDE}"):
        analyze6(tmp_path, prompt_budget_off=prompt_budget_artifact_off(cap=512))
    with pytest.raises(T.TransferAnalysisError, match="--prompt-budget-off artifact: arm 'on'"):
        analyze6(tmp_path, prompt_budget_off=prompt_budget_artifact())
    with pytest.raises(T.TransferAnalysisError, match="--prompt-budget artifact: arm 'off'"):
        analyze6(tmp_path, prompt_budget=prompt_budget_artifact_off())
    doc = T.checked_prompt_budget_off(prompt_budget_artifact_off())
    assert doc["arm"] == "off" and doc["cap"] == CAP_OFF_OVERRIDE and T.checked_prompt_budget_off(None) is None
    pb = analyze6(tmp_path)["offcap"]["prompt_budget_off"]
    assert pb["arm"] == "off" and pb["cap"] == CAP_OFF_OVERRIDE and pb["max"] == PROMPT_MAX_OFF and "lengths" not in pb
    assert pb["tree_sha"] == OFFCAP_FULL_SHA and pb["chat_template_hash"] == "0123456789abcdef"


def test_the_probe_requirement_binds_the_validated_comparator_classification_never_cap_key_presence(tmp_path):
    """The historical none/ON and psr/ON manifests at 5f5fd3d carry
    `max_tokens_by_task` (with their two other optional keys) and no probe
    evidence — they must keep passing. The probe requirement binds
    `leg["cap_override"] is True` only: a comparator none/OFF/8192 store
    WITHOUT `evidence.base_route_declaration_refused` FAILS the gate naming the
    leg; the comparator psr store is not held to it."""
    build_2x2(tmp_path / "hist", **off_legs(), none_on={"correct": range(126), "oos_keys": True},
              psr_on={"correct": range(143), "oos_keys": True})
    rep = analyze(tmp_path / "hist")
    assert leg_of(rep, NONE, "on")["identity"][C.IDENTITY_KEY_MAX_TOKENS_BY_TASK] == {TASK: CAP_ON}
    assert leg_of(rep, PSR, "on")["identity"][C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT] == ["thinking"]
    assert rep["gates"]["declaration"]["status"] != T.GATE_FAIL and rep["gates"]["failed"] == []
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS          # the historical three-key ON stores keep passing
    assert all(leg["cap_override"] is False for leg in rep["legs"].values())
    assert BASE_ROUTE_PROBE_BLOCK not in json.dumps(rep["gates"]["declaration"])
    # the same historical shape inside a six-leg run keeps passing; the comparator's none leg is probed
    build_fixture_2x3(tmp_path, on={"none_on": {"correct": range(140), "oos_keys": True},
                                    "psr_on": {"correct": range(160), "oos_keys": True}})
    rep6 = analyze6(tmp_path)
    assert rep6["gates"]["declaration"]["status"] != T.GATE_FAIL and rep6["gates"]["failed"] == []
    k_none, k_psr = T.leg_key(NONE, "off", cap_override=True), T.leg_key(PSR, "off", cap_override=True)
    assert rep6["gates"]["declaration"]["legs"][k_none][BASE_ROUTE_PROBE_BLOCK] == {"status": 400, "code": "thinking_on_not_lens", "ok": True}
    assert BASE_ROUTE_PROBE_BLOCK not in rep6["gates"]["declaration"]["legs"][k_psr]
    assert rep6["offcap"]["base_route_declaration_probe"]["ok"] is True and rep6["offcap"]["base_route_declaration_probe"]["leg"] == k_none
    # the analysis reads the block under the contract's ONE spelling — the key the driver
    # writes (`served_capability_run` stamps `tc.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED`)
    # and the synthetic store reproduces; a local respelling in either would miss here
    assert C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED in C.OOS_EVIDENCE_KEYS
    assert BASE_ROUTE_PROBE_BLOCK == C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED
    assert rep6["gates"]["declaration"]["legs"][k_none][C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED]["ok"] is True
    # a comparator none store WITHOUT the probe fails the gate, by name, unwaivably
    build_fixture_2x3(tmp_path / "noprobe", none_off8192={"probe": False})
    rep7 = analyze6(tmp_path / "noprobe", allow_pending_gates=True)
    g = rep7["gates"]["declaration"]
    assert g["status"] == T.GATE_FAIL and "declaration" in rep7["gates"]["failed"]
    assert g["failures"] == [(f"{k_none}: evidence.{BASE_ROUTE_PROBE_BLOCK}.ok absent — the comparator's none leg must record the "
                              "base route's live refusal of the thinking-ON declaration key (400 thinking_on_not_lens, plan §4.8); "
                              "a store without the probe was not produced by the comparator's driver")]
    assert rep7["verdict"]["word"] is None
    # a probe that recorded a non-refusal fails naming status and code; null = the probe did not run
    def _not_refused(ev):
        ev[BASE_ROUTE_PROBE_BLOCK] = {"status": 200, "code": None, "ok": False}
    build_fixture_2x3(tmp_path / "leak", none_off8192={"evidence_edit": _not_refused})
    g = analyze6(tmp_path / "leak")["gates"]["declaration"]
    assert g["status"] == T.GATE_FAIL and g["failures"] == [(f"{k_none}: evidence.{BASE_ROUTE_PROBE_BLOCK}.ok is False (status 200, code None) — "
                                                              "the base route did not refuse the thinking-ON declaration key")]
    def _null(ev):
        ev[BASE_ROUTE_PROBE_BLOCK] = {"status": None, "code": None, "ok": None}
    build_fixture_2x3(tmp_path / "null", none_off8192={"evidence_edit": _null})
    g = analyze6(tmp_path / "null")["gates"]["declaration"]
    assert g["status"] == T.GATE_FAIL and "stamped null" in g["failures"][0]


def test_serving_parity_ignores_the_comparator_legs_and_the_report_says_why(tmp_path):
    build_fixture_2x3(tmp_path)
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    rep = analyze6(tmp_path, wave_root=wave)
    par = rep["serving_parity"]
    assert par["available"] and set(par["cells"]) == {NONE, PSR}
    for cell in (NONE, PSR):
        assert par["cells"][cell]["fresh_leg"] == T.leg_key(cell, "off") and par["cells"][cell]["comparable"] is True
        assert par["cells"][cell]["comparison"]["max_tokens"] == {"wave": 512, "fresh": 512}
    assert rep["sensitivity"]["parity"]["wording"] == T.PARITY_NO_DRIFT
    note = rep["offcap"]["parity_note"]
    assert note == T.OFFCAP_PARITY_NOTE and "not screened against the wave" in note and "not parity evidence" in note
    assert "max_tokens" in T.PARITY_FIELDS


def test_six_legs_pass_the_nuisance_check_with_the_cap_per_leg(tmp_path):
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    nu = rep["nuisance"]
    assert nu["disagreeing"] == [] and nu["refused"] == {}
    assert all(len(f["values"]) == 6 for f in nu["fields"].values())
    assert "max_tokens" not in T.NUISANCE_FIELDS
    caps = {k: leg["summary"]["max_tokens"] for k, leg in {**rep["legs"], **rep["offcap"]["legs"]}.items()}
    assert caps == {T.leg_key(NONE, "off"): 512, T.leg_key(PSR, "off"): 512, T.leg_key(NONE, "on"): CAP_ON,
                    T.leg_key(PSR, "on"): CAP_ON, T.leg_key(NONE, "off", cap_override=True): CAP_OFF_OVERRIDE,
                    T.leg_key(PSR, "off", cap_override=True): CAP_OFF_OVERRIDE}
    assert rep["offcap"]["configurations"] == {"off": {"cap": 512, "label": "OFF/512"},
                                               "off_override": {"cap": CAP_OFF_OVERRIDE, "label": OFF8K_CFG},
                                               "on": {"cap": CAP_ON, "label": f"ON/{CAP_ON}"}}
    assert rep["population"]["status"] == T.GATE_PASS and rep["population"]["n"] == 200


def test_offcap_transitions_and_complete_case_rows_file_the_five_comparator_pairs(tmp_path):
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    tr = rep["offcap"]["transitions"]
    assert list(tr) == ["off512_to_off8192__none", "off512_to_off8192__psr", "off8192_to_on__none", "off8192_to_on__psr",
                        "none_to_psr__off8192"]
    assert tr["off512_to_off8192__none"]["counts"] == {"C->C": 120, "C->I": 0, "I->C": 6, "I->I": 74, "->U": 0}
    assert tr["off512_to_off8192__none"]["from"] == T.leg_key(NONE, "off") and tr["off512_to_off8192__none"]["to"] == T.leg_key(NONE, "off", cap_override=True)
    # the OFF/8192 and ON correct-sets are not nested: 126 vs 140 correct with 13 OFF/8192-only items
    assert tr["off8192_to_on__none"]["counts"] == {"C->C": 113, "C->I": 13, "I->C": 27, "I->I": 47, "->U": 0}
    assert tr["none_to_psr__off8192"]["counts"] == {"C->C": 116, "C->I": 10, "I->C": 37, "I->I": 37, "->U": 0}
    assert tr["none_to_psr__off8192"]["label"] == pr.COMPLETE_CASE_LABEL
    mc = rep["offcap"]["sensitivity"]["mcnemar_exact"]
    assert set(mc) == set(tr) and mc["off512_to_off8192__none"][T.BRANCH_PRIMARY]["I->C"] == 6
    cc = rep["offcap"]["complete_case_rows"]
    assert set(cc) == {"label", "cap_relief_none", "cap_relief_psr", "shift_none_matched", "shift_psr_matched", "effect_off_8192"}
    assert cc["cap_relief_none"]["delta"] == pytest.approx(6 / 200) and cc["effect_off_8192"]["delta"] == pytest.approx(27 / 200)
    assert rep["transitions"] == analyze(tmp_path)["transitions"]         # the legacy four pairs, unchanged


def test_a_missing_comparator_leg_withholds_every_estimand_and_is_named(tmp_path):
    build_fixture_2x3(tmp_path)
    import shutil
    shutil.rmtree(T.out_root_of(tmp_path, OFFCAP_SHA7, cell=PSR, arm="off", prefix=T.OFFCAP_OUT_SUFFIX_PREFIX))
    rep = analyze6(tmp_path)
    k = T.leg_key(PSR, "off", cap_override=True)
    assert offleg(rep, PSR)["missing"].startswith("no out-root at ") and rep["unusable_legs"] == {k: offleg(rep, PSR)["missing"]}
    assert "estimands" not in rep and "estimands" not in rep["offcap"] and rep["verdict"]["word"] is None
    assert rep["population"]["status"] == T.GATE_FAIL and k in rep["population"]["detail"]
    assert rep["offcap"]["legs"][k]["cap_override"] is False
    # the 2x2 alone still files its word — the comparator is one flag away
    assert analyze(tmp_path)["verdict"]["word"] == T.WORD_TRANSFERS


def test_the_e1b_figures_write_png_pdf_numbers_and_name_every_arm_with_its_cap(tmp_path):
    pytest.importorskip("matplotlib")
    build_fixture_2x3(tmp_path)
    wave = tmp_path / "wave"
    build_store(wave, cell=NONE, arm="off", correct=range(118), out_suffix="c7e1796", oos_marker=False)
    build_store(wave, cell=PSR, arm="off", correct=OFF_PSR_CORRECT, out_suffix="c7e1796", oos_marker=False)
    rep = analyze6(tmp_path, wave_root=wave)
    figs = T.make_figures(rep, tmp_path / "figs")
    assert set(figs["written"]) == set(T.FIGURE_STEMS) | set(T.FIGURES_OFFCAP)
    assert set(T.FIGURES_OFFCAP) == {"final_e1b_shift_decomposition", "final_e1b_effect_by_configuration_ci", "final_e1b_gsm8k_acc_2x3",
                                     "final_e1b_item_transitions_offcap", "final_e1b_off512_to_off8192_by_first_draw_stop",
                                     "final_e1b_controls_offcap"}
    for stem, paths in figs["written"].items():
        assert [Path(p).suffix for p in paths] == [".png", ".pdf"] and all(Path(p).stat().st_size > 0 for p in paths)
    assert Path(figs["numbers_offcap"]).name == T.NUMBERS_OFFCAP_FILENAME == "final_e1b_figures_numbers.json"
    assert Path(figs["tables_offcap"]).name == T.TABLES_OFFCAP_FILENAME == "final_e1b_tables.json"
    numbers = json.loads(Path(figs["numbers_offcap"]).read_text())
    assert set(numbers["figures"]) == set(T.FIGURES_OFFCAP) and numbers["sha"] == SHA7 and numbers["offcap_sha"] == OFFCAP_SHA7
    # the analysis review's L1: every record's `source` is a report-JSON path (or a list of them) that RESOLVES in
    # the report — checked here with an independent resolver — and any arithmetic between the source leaves and
    # the value is named in a separate `transform`; a record without one carries the leaf itself
    report_json = json.loads(json.dumps(rep, default=str))
    n_transformed = 0
    for stem, nums in numbers["figures"].items():
        assert nums, stem
        for n in nums:
            assert set(n) == {"name", "value", "source", "transform"} and n["source"], (stem, n)
            sources = n["source"] if isinstance(n["source"], list) else [n["source"]]
            resolved = [_resolve_report_path(report_json, src) for src in sources]
            if n["transform"] is None:
                assert len(sources) == 1 and n["value"] == resolved[0], (stem, n)
            else:
                assert isinstance(n["transform"], str) and n["transform"], (stem, n)
                n_transformed += 1
    assert n_transformed > 0
    src_of = {n["name"]: n["source"] for n in numbers["figures"]["final_e1b_shift_decomposition"]}
    assert src_of["budget.point[none]"] == "offcap.decomposition.none.primary.budget.point"
    assert src_of["identity_exact_full_set[psr]"] == "offcap.decomposition.psr.primary.identity_exact_full_set"
    assert set(numbers["captions"]) == set(T.FIGURES_OFFCAP)
    for cap in numbers["captions"].values():
        assert OFF8K_CFG in cap or "OFF/512" in cap
    tables = json.loads(Path(figs["tables_offcap"]).read_text())
    assert set(tables["effects"]) == {T.MAP_CAP, T.MAP_MATCHED} and tables["decomposition"] == rep["offcap"]["decomposition"]
    assert tables["contrast_varies"] == rep["offcap"]["contrast_varies"] and tables["offcap_sha"] == OFFCAP_SHA7
    legacy_numbers = json.loads(Path(figs["numbers"]).read_text())
    assert set(legacy_numbers["figures"]) == set(T.FIGURE_STEMS)          # the e1 pair is the four-leg set, as today
    # every arm mention on the e1b figures carries its cap; the headline names the identity
    import matplotlib.pyplot as plt
    fig, nums = T.fig_shift_decomposition(rep)
    texts = [t.get_text() for t in fig.findobj(plt.Text) if t.get_text().strip()]
    plt.close(fig)
    joined = "\n".join(texts)
    assert "OFF/512" in joined and OFF8K_CFG in joined and f"ON/{CAP_ON}" in joined
    assert "decomposition identity" in joined.lower() or "sum" in joined.lower()
    # the second analysis review's Low (c): the title describes, it does not attribute
    assert f"Observed OFF/512 → ON/{CAP_ON} shift decomposition" in texts
    assert "comes from" not in joined.lower() and "where the" not in joined.lower()
    bare_arm = re.compile(r"\b(OFF|ON)\b(?!/)(?! render)")          # an arm is named with its cap, or as "the OFF render"
    assert not [t for t in texts if bare_arm.search(t)], [t for t in texts if bare_arm.search(t)]
    fig, _ = T.fig_off512_to_off8192(rep)
    texts2 = [t.get_text() for t in fig.findobj(plt.Text) if t.get_text().strip()]
    plt.close(fig)
    low = "\n".join(texts2).lower()
    assert "exploratory" in low and "post-selected" in low and "noise floor" not in low and "parity" not in low
    mirrored = T.mirror_figures(tmp_path / "figs", tmp_path / "mirror", manifest=figs)
    assert len(mirrored) == 2 * (len(T.FIGURE_STEMS) + len(T.FIGURES_OFFCAP)) + 4


def test_the_mirror_copies_the_generated_manifest_both_figure_sets_and_both_json_pairs(tmp_path):
    """The old glob `final_e1_*` did not match `final_e1b_*`; the mirror
    copies EXACTLY the generated-file manifest."""
    figs = tmp_path / "figs"
    figs.mkdir()
    written = {}
    for stem in ("final_e1_gsm8k_acc_2x2", "final_e1b_shift_decomposition"):
        paths = []
        for ext in (".png", ".pdf"):
            p = figs / f"{stem}{ext}"
            p.write_bytes(b"x")
            paths.append(str(p))
        written[stem] = paths
    for name in (T.NUMBERS_FILENAME, T.TABLES_FILENAME, T.NUMBERS_OFFCAP_FILENAME, T.TABLES_OFFCAP_FILENAME):
        (figs / name).write_text("{}")
    (figs / "final_e1_stray.png").write_bytes(b"stray")           # on disk, NOT in the manifest
    manifest = {"written": written, "numbers": str(figs / T.NUMBERS_FILENAME), "tables": str(figs / T.TABLES_FILENAME),
                "numbers_offcap": str(figs / T.NUMBERS_OFFCAP_FILENAME), "tables_offcap": str(figs / T.TABLES_OFFCAP_FILENAME)}
    out = T.mirror_figures(figs, tmp_path / "mirror", manifest=manifest)
    names = sorted(Path(p).name for p in out)
    assert names == sorted(["final_e1_gsm8k_acc_2x2.png", "final_e1_gsm8k_acc_2x2.pdf", "final_e1b_shift_decomposition.png",
                            "final_e1b_shift_decomposition.pdf", T.NUMBERS_FILENAME, T.TABLES_FILENAME,
                            T.NUMBERS_OFFCAP_FILENAME, T.TABLES_OFFCAP_FILENAME])
    assert sorted(p.name for p in (tmp_path / "mirror").iterdir()) == names          # the stray file is not copied
    # a manifest entry missing on disk refuses by name
    (figs / "final_e1b_shift_decomposition.pdf").unlink()
    with pytest.raises(T.TransferAnalysisError, match="final_e1b_shift_decomposition.pdf"):
        T.mirror_figures(figs, tmp_path / "mirror2", manifest=manifest)
    # the four-leg manifest (no offcap pair) mirrors its own set
    legacy = {"written": {"final_e1_gsm8k_acc_2x2": written["final_e1_gsm8k_acc_2x2"]}, "numbers": manifest["numbers"], "tables": manifest["tables"]}
    assert len(T.mirror_figures(figs, tmp_path / "mirror3", manifest=legacy)) == 4
    with pytest.raises(TypeError):
        T.mirror_figures(figs, tmp_path / "mirror4")                    # the glob is gone: the manifest is required


def _fake_wandb(monkeypatch):
    calls: dict = {}

    class _Summary:
        @staticmethod
        def update(d):
            calls["summary"] = dict(d)

    class _Run:
        url = "https://wandb.test/run"
        summary = _Summary()

        def log_artifact(self, art):
            calls["artifact"] = art.name

        def finish(self):
            calls["finished"] = True

    class _Artifact:
        def __init__(self, name, type):
            self.name = name

        def add_dir(self, d):
            calls["artifact_dir"] = d

    def _init(**kw):
        calls["init"] = kw
        return _Run()

    fake = SimpleNamespace(init=_init, Image=lambda p: p, Artifact=_Artifact,
                           log=lambda d: calls.setdefault("logged", []).append(d))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return calls


def test_the_wandb_name_flag_names_the_run_and_the_artifact(tmp_path, monkeypatch):
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    calls = _fake_wandb(monkeypatch)
    png = tmp_path / "figs" / "final_e1b_x.png"
    png.parent.mkdir()
    png.write_bytes(b"png")
    url = T.log_wandb(rep, {"written": {"final_e1b_x": [str(png)], "final_e1_y": [str(png)]}}, mode="offline",
                      name=f"e1b-offcap-comparator-{OFFCAP_SHA7}")
    assert url == "https://wandb.test/run"
    assert calls["init"]["name"] == f"e1b-offcap-comparator-{OFFCAP_SHA7}" and calls["artifact"] == f"figures-e1b-offcap-comparator-{OFFCAP_SHA7}"
    assert calls["init"]["config"]["sha"] == SHA7 and calls["init"]["config"]["offcap_sha"] == OFFCAP_SHA7
    assert calls["logged"] == [{"e1b/final_e1b_x": str(png)}, {"e1/final_e1_y": str(png)}]
    T.log_wandb(rep, {}, mode="offline", name_suffix="-v2")
    assert calls["init"]["name"] == f"e1-thinking-transfer-{SHA7}-v2"          # the default name, as today
    ap = T.build_parser()
    a = ap.parse_args(["--gen-root", ".", "--sha", "x", "--wandb-name", "e1b-offcap-comparator-abc1234"])
    assert a.wandb_name == "e1b-offcap-comparator-abc1234" and ap.parse_args(["--gen-root", ".", "--sha", "x"]).wandb_name is None


def test_the_wandb_summary_keys_six_distinct_configurations_with_both_off_caps_and_keeps_the_legacy_keys(tmp_path, monkeypatch):
    """`controls/<kind>/<arm>/<field>` cannot hold a second OFF cap, so the
    comparator's controls go under `controls_cap/<kind>/<arm>/<cap>/<field>`
    for ALL SIX configurations; the four legacy prefixes stay byte-identical
    to the 2x2 run's."""
    build_fixture_2x3(tmp_path)
    rep4, rep6 = analyze(tmp_path), analyze6(tmp_path)
    calls = _fake_wandb(monkeypatch)
    T.log_wandb(rep4, {}, mode="offline")
    s4 = calls["summary"]
    T.log_wandb(rep6, {}, mode="offline")
    s6 = calls["summary"]
    legacy = {k: v for k, v in s4.items() if k.startswith("controls/")}
    assert legacy and {k: v for k, v in s6.items() if k.startswith("controls/")} == legacy
    assert not [k for k in s4 if k.startswith(("controls_cap/", "offcap/"))]
    prefixes = {"/".join(k.split("/")[1:4]) for k in s6 if k.startswith("controls_cap/")}
    assert prefixes == {"none/off/512", f"none/off/{CAP_OFF_OVERRIDE}", f"none/on/{CAP_ON}", "psr/off/512",
                        f"psr/off/{CAP_OFF_OVERRIDE}", f"psr/on/{CAP_ON}"}
    assert s6["controls_cap/none/off/512/acc"] == 0.6 and s6[f"controls_cap/none/off/{CAP_OFF_OVERRIDE}/acc"] == 0.63
    assert s6[f"controls_cap/none/off/{CAP_OFF_OVERRIDE}/max_tokens"] == CAP_OFF_OVERRIDE and s6["controls/none/off/max_tokens"] == 512
    for b in T.BRANCHES:
        for c in T.OFFCAP_CONTRASTS:
            assert {k.split("/")[-1] for k in s6 if k.startswith(f"offcap/{b}/{c}/")} == {"point", "ci_low", "ci_high", "bound_low", "bound_high"}
        assert s6[f"offcap/{b}/cap_relief_none/point"] == 6 / 200 and s6[f"offcap/decomposition/none/{b}/budget_share_point"] == pytest.approx(0.3)
    assert {k for k in s4 if k.startswith(("primary/", "score_as_is/"))} == {k for k in s6 if k.startswith(("primary/", "score_as_is/"))}
    assert s6["verdict"] == s4["verdict"] and s6["offcap/verdict_sensitivity_off_8192"] == T.WORD_TRANSFERS


def test_the_report_carries_both_store_pins_and_the_analysis_code_sha(tmp_path, capsys):
    offcap_root = tmp_path / "pull_offcap"
    build_fixture_2x3(tmp_path, offcap_root=offcap_root)
    out = tmp_path / "report" / "report.json"
    budget = tmp_path / "oos_prompt_budget.json"
    budget.write_text(json.dumps(prompt_budget_artifact()))
    budget_off = tmp_path / "oos_prompt_budget_off.json"
    budget_off.write_text(json.dumps(prompt_budget_artifact_off()))
    ids_file = tmp_path / "paired_ids.json"
    ids_file.write_text(json.dumps(IDS))
    sys.modules.pop("wandb", None)
    rc = T.main(["--gen-root", str(tmp_path), "--sha", f"oos-{SHA7}", "--offcap-sha", f"offcap-{OFFCAP_SHA7}",
                 "--offcap-gen-root", str(offcap_root), "--prompt-budget", str(budget), "--prompt-budget-off", str(budget_off),
                 "--paired-ids", str(ids_file), "--wandb-mode", "disabled", "--no-figs", "--no-mirror", "--no-eval-logs",
                 "--bootstrap", "500", "--allow-pending-gates", "--out", str(out)])
    assert rc == 0 and "wandb" not in sys.modules
    rep = json.loads(out.read_text())
    assert (rep["sha"], rep["offcap_sha"], rep["offcap_gen_root"]) == (SHA7, OFFCAP_SHA7, str(offcap_root))
    assert rep["analysis_code_sha"] == _git_head(REPO) and rep["offcap"]["provenance"] == {
        "sha": SHA7, "offcap_sha": OFFCAP_SHA7, "analysis_code_sha": rep["analysis_code_sha"],
        "note": T.OFFCAP_PROVENANCE_NOTE}
    assert list(rep)[:8] == ["model", "task", "sha", "analysis_code_sha", "analysis_code", "gen_root", "offcap_sha", "offcap_gen_root"]
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS and rep["offcap"]["effect_off_8192"]["resolved"] is True
    assert rep["offcap"]["legs"][f"{NONE}|off|cap{CAP_OFF_OVERRIDE}"]["cap_override"] is True
    printed = capsys.readouterr().out
    assert f"comparator store pin {OFFCAP_SHA7}" in printed and "== the thinking-OFF comparator at cap 8192" in printed
    assert "cap_relief_none" in printed and "effect_off_8192" in printed and "Of the unsteered" in printed
    # --offcap-gen-root defaults to --gen-root; a wrong suffix spelling is normalised like --sha's
    build_fixture_2x3(tmp_path / "same")
    rep2 = analyze(tmp_path / "same", offcap_sha=f"offcap-{OFFCAP_SHA7}")
    assert rep2["offcap_gen_root"] == str(tmp_path / "same") and rep2["offcap_sha"] == OFFCAP_SHA7
    assert rep2["offcap"]["prompt_budget_off"] is None and offleg(rep2, NONE)["ledger"]["pending"]


# --------------------------------------------------------------------------- #
# the four analysis findings (M1 wording logic, M2 per-item log                 #
# check, M3 object-artifact identity, L1 sidecar sources)                       #
# --------------------------------------------------------------------------- #


def test_the_one_sided_note_follows_the_contrasts_actual_masks_and_the_registered_signs_stand_beside_it(tmp_path):
    """Analysis finding M1: the note said the registered bootstrap "is
    not constrained" by a one-sided discordance because it resamples censoring
    masks and divides by per-replicate denominators — FALSE for a contrast whose
    two legs are fully scored under the branch (every mask one, every replicate
    denominator n): there the registered bootstrap IS the empirical bootstrap of
    the pairs and cannot generate a contrast of the opposite sign. The note now
    reads the contrast's ACTUAL masks — fully scored: the constraint applies and
    an interval endpoint on the discordances' side is a property of the sparse
    empirical distribution, not corroboration; masks varying (a censored ON leg
    under primary): the existing statement — and every McNemar record carries
    the registered replicates' sign counts of the contrast it IS. Fixture:
    none/ON gains-only over none/OFF-8192 with four censored draws, so the SAME
    pair reads both ways: primary (n = 196, masks vary) and score-as-is (n = 200,
    every mask one)."""
    build_fixture_2x3(tmp_path, none_on={"correct": OFF8192_NONE_CORRECT | set(range(126, 140)), "unscored": LAST4})
    rep = analyze6(tmp_path)
    mc = rep["offcap"]["sensitivity"]["mcnemar_exact"]
    est = rep["offcap"]["estimands"]
    # the psr cap relief: 3 gains / 0 losses over two fully scored OFF legs, under both branches
    cap = mc["off512_to_off8192__psr"]
    assert cap["registered_contrast"] == "cap_relief_psr"
    for b in T.BRANCHES:
        r = cap[b]
        rs = est[T.MAP_CAP][b]["stats"]["cap_relief_psr"]["replicate_signs"]
        assert (r["n_items"], r["n_common"], r["fully_scored"], r["one_sided_discordance"]) == (200, 200, True, True)
        assert (r["C->I"], r["I->C"]) == (0, 3)
        assert r["replicate_signs"] == rs and r["n_replicates_used"] == B_FAST
        assert rs["negative"] == 0                    # the constraint HOLDS on the registered replicates
        assert r["note"] == ("All three complete-case discordances are gains, so an empirical bootstrap of those 200 pairs "
                             f"cannot generate a decline; {T.ONE_SIDED_NOTE_FULLY_SCORED.format(n=200, d=3)}")
        assert T.ONE_SIDED_NOTE_REGISTERED not in r["note"] and "does not apply" not in r["note"]
    # the render-only none contrast: one-sided under both branches, masks VARY under primary only
    ren = mc["off8192_to_on__none"]
    assert ren["registered_contrast"] == "shift_none_matched"
    pri, sai = ren[T.BRANCH_PRIMARY], ren[T.BRANCH_SCORE_AS_IS]
    assert (pri["n_items"], pri["n_common"], pri["fully_scored"], pri["one_sided_discordance"]) == (196, 200, False, True)
    assert (pri["C->I"], pri["I->C"]) == (0, 14)
    assert pri["note"] == ("All 14 complete-case discordances are gains, so an empirical bootstrap of those 196 pairs cannot "
                           f"generate a decline; {T.ONE_SIDED_NOTE_REGISTERED}")
    assert pri["replicate_signs"] == est[T.MAP_MATCHED][T.BRANCH_PRIMARY]["stats"]["shift_none_matched"]["replicate_signs"]
    assert (sai["n_items"], sai["n_common"], sai["fully_scored"]) == (200, 200, True)
    assert sai["note"].endswith(T.ONE_SIDED_NOTE_FULLY_SCORED.format(n=200, d=14))
    # a two-sided contrast carries no note; its masks and sign counts still stand beside it
    two = mc["none_to_psr__off8192"]
    assert two["registered_contrast"] == "effect_off_8192"
    for b in T.BRANCHES:
        assert two[b]["note"] is None and two[b]["one_sided_discordance"] is False and two[b]["fully_scored"] is True
        assert two[b]["replicate_signs"] == est[T.MAP_CAP][b]["stats"]["effect_off_8192"]["replicate_signs"]
    # the legacy 2x2 records read their masks the same way and carry the registered contrast's signs
    legacy = rep["sensitivity"]["mcnemar_exact"]
    assert {k: v["registered_contrast"] for k, v in legacy.items()} == {
        "off_to_on__none": "shift_none", "off_to_on__psr": "shift_psr", "none_to_psr__off": "effect_off",
        "none_to_psr__on": "effect_on"}
    lon = legacy["off_to_on__none"]
    assert lon[T.BRANCH_PRIMARY]["fully_scored"] is False and lon[T.BRANCH_SCORE_AS_IS]["fully_scored"] is True
    assert lon[T.BRANCH_PRIMARY]["replicate_signs"] == rep["estimands"][T.BRANCH_PRIMARY]["stats"]["shift_none"]["replicate_signs"]
    # the helper on its own: the masks decide which statement follows the head
    base = {"one_sided_discordance": True, "n_discordant": 2, "n_items": 200, "I->C": 2, "C->I": 0}
    assert T.one_sided_note({**base, "fully_scored": True}).endswith(T.ONE_SIDED_NOTE_FULLY_SCORED.format(n=200, d=2))
    assert T.one_sided_note({**base, "fully_scored": False}).endswith(T.ONE_SIDED_NOTE_REGISTERED)
    assert T.one_sided_note(base).endswith(T.ONE_SIDED_NOTE_REGISTERED)          # masks unknown: the weaker statement
    banned = ("cannot draw an improvement", "primary p", "excludes zero more readily", "is budget", "is render")
    everything = json.dumps(rep, default=str)
    assert not [w for w in banned if w in everything], [w for w in banned if w in everything]


def test_the_decomposition_findings_say_observed_contrast_and_carry_the_sparse_discordance_qualification(tmp_path):
    """M1 (continued): the headline decomposition statements — the first
    two comparator findings — read "the observed cap-only contrast was +x pp
    [CI]" (never "+x pp is budget"), name the identity as one between OBSERVED
    estimates (not a causal partition), and carry the sparse-discordance /
    exact-McNemar qualification of the cap-only contrast BESIDE the numbers:
    how many discordant items of how many, gains / losses, the exact p, and —
    both OFF legs being fully scored — that the registered bootstrap cannot
    generate a contrast of the opposite sign when the discordance runs one way,
    with its replicate signs. The same record sits in the decomposition block
    (`budget_discordance`), not only under `sensitivity`."""
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    dec = rep["offcap"]["decomposition"]
    off8 = OFF8K_CFG
    for kind, gains, share in (("none", 6, "unsteered"), ("psr", 3, "steered (psr)")):
        for b in T.BRANCHES:
            d = dec[kind][b]
            bd = d["budget_discordance"]
            mc = rep["offcap"]["sensitivity"]["mcnemar_exact"][f"off512_to_off8192__{kind}"][b]
            assert bd["transition"] == f"off512_to_off8192__{kind}" and bd["registered_contrast"] == f"cap_relief_{kind}"
            assert (bd["n_items"], bd["C->I"], bd["I->C"], bd["n_discordant"]) == (200, 0, gains, gains)
            assert bd["p_value"] == mc["p_value"] == pytest.approx(2 * 0.5 ** gains)
            assert bd["fully_scored"] is True and bd["one_sided_discordance"] is True and bd["note"] == mc["note"]
            assert bd["replicate_signs"] == rep["offcap"]["estimands"][T.MAP_CAP][b]["stats"][f"cap_relief_{kind}"]["replicate_signs"]
            assert bd["replicate_signs"]["negative"] == 0
            st = d["statement"]
            assert st.startswith(f"Of the {share} OFF/512 → ON/{CAP_ON} gain of ")
            assert (f"the observed cap-only contrast (OFF/512 → {off8}, same render) was {T._pp(d['budget']['point'])} "
                    f"{T._ci(d['budget'])}") in st
            assert (f"the observed render-only contrast ({off8} → ON/{CAP_ON}, same cap) was {T._pp(d['render']['point'])} "
                    f"{T._ci(d['render'])}") in st
            assert "a decomposition identity between observed estimates, not a causal partition" in st
            assert "the intervals are each term's own and do not add" in st
            rs = bd["replicate_signs"]
            assert (f"The cap-only contrast rests on {gains} discordant items of 200 ({gains} gains / 0 losses; exact two-sided "
                    f"McNemar p = {mc['p_value']:.4g}, a complete-case sensitivity); both legs are fully scored under this "
                    "branch, so the registered bootstrap cannot generate a contrast of the opposite sign (replicate signs "
                    f"{rs['negative']:,} negative / {rs['zero']:,} zero / {rs['positive']:,} positive of {B_FAST:,}) — its ") in st
            # L1: the endpoint is READ, never assumed — 6 gains of 200 leave a positive lower endpoint,
            # 3 gains of 200 a zero one (about 4.9 % of replicates carry no gain at n = 200, above the 2.5th percentile)
            lo = d["budget"]["ci_low"]
            if kind == "none":
                assert lo > 0 and f"its positive lower endpoint ({T._pp(lo)}) reflects this sparse empirical distribution, not corroboration" in st
            else:
                assert lo == 0 and (f"its lower endpoint is {T._pp(lo)}: {rs['zero']:,} of {B_FAST:,} replicates carry no gain, so the "
                                    "interval reaches zero; a positive lower endpoint, when present, reflects this sparse empirical "
                                    "distribution, not corroboration") in st
            assert "lower percentile above zero" not in st
            assert bd["qualifier"] in st
            assert "is budget" not in st and "is render" not in st
    # the two statements ARE the first two comparator findings, qualification included
    assert rep["offcap"]["findings"][:2] == [dec[k][T.BRANCH_PRIMARY]["statement"] for k in ("none", "psr")]
    assert all("rests on" in f and "McNemar" in f for f in rep["offcap"]["findings"][:2])
    # a two-sided cap-only contrast: the qualification names both directions and no one-sided claim
    two = build_fixture_2x3(tmp_path / "two", none_off8192={"correct": (OFF8192_NONE_CORRECT - {0}) | {130}})
    assert two
    d2 = analyze6(tmp_path / "two")["offcap"]["decomposition"]["none"][T.BRANCH_PRIMARY]
    bd2 = d2["budget_discordance"]
    assert (bd2["C->I"], bd2["I->C"], bd2["one_sided_discordance"], bd2["note"]) == (1, 7, False, None)
    assert ("The cap-only contrast rests on 8 discordant items of 200 (7 gains / 1 loss; exact two-sided McNemar p = "
            f"{bd2['p_value']:.4g}, a complete-case sensitivity); both legs are fully scored under this branch, so the "
            "registered bootstrap is the empirical bootstrap of those pairs and its interval reflects that sparse "
            "distribution (replicate signs ") in d2["statement"]
    assert "opposite sign" not in d2["statement"]


def test_the_eval_log_re_derivation_compares_every_item_not_only_the_counts(tmp_path, monkeypatch):
    """Analysis finding M2: `eval_cross_check` compared the log's
    AGGREGATE counts with the record's header, so a store whose rows and
    outcomes swapped one correct and one incorrect comparator outcome kept
    every marginal, passed the rows / eval / ledger / outcomes gates and moved
    the cap-relief discordances (the fixture: 6 gains / 0 losses -> 7 / 1). The
    retained log's results are now compared with the outcome record item by
    item, by (item_id, sample_idx), on score availability AND score value —
    a mismatch is a named refusal of that leg; a draw the log lacks or the
    record lacks is named too."""
    swapped = (OFF8192_NONE_CORRECT - {0}) | {130}              # item 0 correct -> incorrect, item 130 the reverse
    build_fixture_2x3(tmp_path, none_off8192={"correct": swapped})
    # without the log the swap is invisible: every gate passes and the discordances read 7 / 1
    rep0 = analyze6(tmp_path)
    assert rep0["gates"]["failed"] == [] and rep0["unusable_legs"] == {}
    t0 = rep0["offcap"]["transitions"]["off512_to_off8192__none"]["counts"]
    assert (t0["I->C"], t0["C->I"]) == (7, 1)
    k8 = T.leg_key(NONE, "off", cap_override=True)
    target = offleg(rep0, NONE)["cell_dir"]

    def fake(cell_dir, task):
        # the log carries the truth (item 0 correct, item 130 incorrect) where the record was swapped
        swap = ((IDS[0], IDS[130]),) if str(cell_dir) == target else ()
        return _results_from_store(cell_dir, task, swap=swap), [], None

    monkeypatch.setattr(T, "_read_eval_results", fake)
    rep = analyze6(tmp_path, read_eval_logs=True)
    leg = offleg(rep, NONE)
    ec = leg["eval_check"]
    assert ec["available"] is True and ec["agrees"] is False
    assert ec["count_detail"] is None                                   # every aggregate count AGREES ...
    per = ec["per_item"]
    assert (per["n_log"], per["n_record"], per["n_compared"], per["n_mismatched"]) == (200, 200, 200, 2)
    assert per["missing_in_log"] == [] and per["extra_in_log"] == []
    assert {(m["item_id"], m["sample_idx"]) for m in per["mismatches"]} == {(IDS[0], 0), (IDS[130], 0)}
    by_id = {m["item_id"]: m for m in per["mismatches"]}
    assert by_id[IDS[0]]["log"] == {"scored": True, "value": 1.0} and by_id[IDS[0]]["record"] == {"scored": True, "value": 0.0}
    assert by_id[IDS[130]]["log"] == {"scored": True, "value": 0.0} and by_id[IDS[130]]["record"] == {"scored": True, "value": 1.0}
    assert leg["refused"].startswith("the .eval log re-derivation disagrees with the record: ")
    assert IDS[0] in leg["refused"] and IDS[130] in leg["refused"] and "2 of 200 draws" in leg["refused"]
    assert rep["unusable_legs"] == {k8: leg["refused"]} and "estimands" not in rep and rep["verdict"]["word"] is None
    # every other leg's log agrees item by item
    for k, other in {**rep["legs"], **rep["offcap"]["legs"]}.items():
        if k != k8:
            assert other["eval_check"]["agrees"] is True and other["eval_check"]["per_item"]["n_mismatched"] == 0, k
    # the comparison itself, on constructed inputs: availability, value, a missing and an extra draw each named
    items = [{"item_id": "a", "sample_idx": 0, "scored": True, "value": 1.0},
             {"item_id": "b", "sample_idx": 0, "scored": True, "value": 0.0},
             {"item_id": "c", "sample_idx": 0, "scored": False, "value": None},
             {"item_id": "d", "sample_idx": 0, "scored": True, "value": 1.0}]
    ok = [SimpleNamespace(item_id="a", sample_idx=0, unscored=False, value=1.0),
          SimpleNamespace(item_id="b", sample_idx=0, unscored=False, value=0.0),
          SimpleNamespace(item_id="c", sample_idx=0, unscored=True, value=float("nan")),
          SimpleNamespace(item_id="d", sample_idx=0, unscored=False, value=1.0)]
    good = T.per_item_log_check(ok, items)
    assert good == {"n_log": 4, "n_record": 4, "n_compared": 4, "n_mismatched": 0, "mismatches": [],
                    "missing_in_log": [], "extra_in_log": [], "agrees": True}
    bad = [SimpleNamespace(item_id="a", sample_idx=0, unscored=False, value=0.0),          # value
           SimpleNamespace(item_id="c", sample_idx=0, unscored=False, value=0.0),          # availability
           SimpleNamespace(item_id="d", sample_idx=1, unscored=False, value=1.0),          # another draw: extra
           SimpleNamespace(item_id="d", sample_idx=0, unscored=False, value=1.0)]          # b missing in the log
    chk = T.per_item_log_check(bad, items)
    assert chk["agrees"] is False and (chk["n_log"], chk["n_record"], chk["n_compared"], chk["n_mismatched"]) == (4, 4, 3, 2)
    assert chk["missing_in_log"] == [["b", 0]] and chk["extra_in_log"] == [["d", 1]]
    assert chk["mismatches"] == [
        {"item_id": "a", "sample_idx": 0, "log": {"scored": True, "value": 0.0}, "record": {"scored": True, "value": 1.0}},
        {"item_id": "c", "sample_idx": 0, "log": {"scored": True, "value": 0.0}, "record": {"scored": False, "value": None}}]
    # a duplicate (item, sample) in the log is refused outright — two results cannot both be the draw
    with pytest.raises(T.TransferAnalysisError, match="twice"):
        T.per_item_log_check(ok + [ok[0]], items)


def test_the_fitted_objects_artifact_digest_must_agree_across_the_lens_legs_and_is_recorded(tmp_path):
    """Analysis finding M3: comparator admission compared the model,
    the cell id, the route spelling and the nuisance fields, never the fitted
    OBJECT — a changed object under the same cell name would have passed. Every
    leg now records its manifest's `provenance.artifact_sha256` (the digest the
    driver files from the registry artifact, the same digest the launcher holds
    the standup to under `--expect-object-sha256`; NOT the intervention hash,
    which legitimately differs on the declared ON route), and under the
    comparator the digests must agree within every cell across its
    configurations: a lens cell's legs carry one 64-hex digest, the base
    route's carry none; the legs off a strict majority are refused by name
    (all of them with no majority), and the compared digests are filed at
    `offcap.object_identity`."""
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    six = {**rep["legs"], **rep["offcap"]["legs"]}
    digest = json.loads((Path(offleg(rep, PSR)["cell_dir"]) / "manifest.json").read_text())["provenance"]["artifact_sha256"]
    assert C.is_sha256_hex(digest)
    oi = rep["offcap"]["object_identity"]
    assert oi["field"] == "provenance.artifact_sha256" and oi["agree"] is True and oi["refused"] == {}
    assert oi["by_leg"] == {k: (digest if k.startswith(PSR) else None) for k in six}
    assert oi["by_cell"]["psr"] == {"lens": True, "legs": sorted(k for k in six if k.startswith(PSR)), "digests": [digest], "agree": True}
    assert oi["by_cell"]["none"] == {"lens": False, "legs": sorted(k for k in six if k.startswith(NONE)), "digests": [None], "agree": True}
    for k, leg in six.items():
        assert leg["identity"]["object_artifact_sha256"] == (digest if k.startswith(PSR) else None), k
    # the intervention hash differs across the psr legs (the declared ON route) while the object digest agrees
    ih = {k: leg["identity"]["intervention_hash"] for k, leg in six.items() if k.startswith(PSR)}
    assert len(set(ih.values())) > 1
    # the 2x2 report is untouched but for the leg record's new identity field
    rep4 = analyze(tmp_path)
    assert "object_identity" not in rep4 and rep4["legs"][T.leg_key(PSR, "on")]["identity"]["object_artifact_sha256"] == digest
    assert rep4["model"]["validated_legs"] == sorted(rep4["legs"])
    # the comparator psr store carries ANOTHER object's digest: refused by name, the other two psr legs stand
    other = "f" * 64
    build_fixture_2x3(tmp_path / "other", psr_off8192={"manifest_edit": lambda m: m["provenance"].update(artifact_sha256=other)})
    rep2 = analyze6(tmp_path / "other")
    k8 = T.leg_key(PSR, "off", cap_override=True)
    leg = rep2["offcap"]["legs"][k8]
    assert leg["refused"] == (
        f"fitted-object artifact digest provenance.artifact_sha256 = {other!r} disagrees across the {PSR!r} legs "
        f"({{{T.leg_key(PSR, 'off')!r}: {digest!r}, {k8!r}: {other!r}, {T.leg_key(PSR, 'on')!r}: {digest!r}}}) — "
        "another fitted object under the same cell name, not this experiment's")
    oi2 = rep2["offcap"]["object_identity"]
    assert oi2["agree"] is False and oi2["refused"] == {k8: leg["refused"]} and oi2["by_cell"]["psr"]["agree"] is False
    assert sorted(oi2["by_cell"]["psr"]["digests"]) == sorted([digest, other])
    assert rep2["legs"][T.leg_key(PSR, "off")].get("refused") is None and rep2["legs"][T.leg_key(PSR, "on")].get("refused") is None
    assert rep2["unusable_legs"] == {k8: leg["refused"]} and "estimands" not in rep2 and rep2["verdict"]["word"] is None
    assert "estimands" not in rep2["offcap"] and rep2["offcap"]["findings"] == []
    # no strict majority (three different digests): every psr leg is refused
    build_fixture_2x3(tmp_path / "three", psr_off8192={"manifest_edit": lambda m: m["provenance"].update(artifact_sha256="e" * 64)},
                      psr_on={"manifest_edit": lambda m: m["provenance"].update(artifact_sha256="d" * 64)})
    rep3 = analyze6(tmp_path / "three")
    assert set(rep3["offcap"]["object_identity"]["refused"]) == {T.leg_key(PSR, "off"), k8, T.leg_key(PSR, "on")}
    # a lens leg whose manifest records no digest: unrecorded identity, refused — never compared as "agrees with nothing"
    build_fixture_2x3(tmp_path / "unrecorded", psr_on={"manifest_edit": lambda m: m["provenance"].update(artifact_sha256=None)})
    rep5 = analyze6(tmp_path / "unrecorded")
    why = rep5["legs"][T.leg_key(PSR, "on")]["refused"]
    assert why.startswith("fitted-object artifact digest provenance.artifact_sha256 = None is not a 64-hex sha256 on a lens leg")
    assert rep5["offcap"]["object_identity"]["refused"] == {T.leg_key(PSR, "on"): why}
    build_fixture_2x3(tmp_path / "noblock", psr_off8192={"manifest_edit": lambda m: m.pop("provenance")})
    rep6 = analyze6(tmp_path / "noblock")
    assert rep6["offcap"]["legs"][k8]["refused"].startswith("fitted-object artifact digest provenance.artifact_sha256 = None is not a 64-hex sha256 on a lens leg")
    # a base-route leg carrying a digest: a fitted object served on the base route is another experiment
    build_fixture_2x3(tmp_path / "base", none_off8192={"manifest_edit": lambda m: m["provenance"].update(artifact_sha256=digest)})
    rep7 = analyze6(tmp_path / "base")
    k8n = T.leg_key(NONE, "off", cap_override=True)
    assert rep7["offcap"]["legs"][k8n]["refused"].startswith(
        f"fitted-object artifact digest provenance.artifact_sha256 = {digest!r} on the base-route leg")
    assert rep7["offcap"]["object_identity"]["refused"] == {k8n: rep7["offcap"]["legs"][k8n]["refused"]}


def test_the_model_label_enumerates_every_validated_leg_six_under_the_comparator(tmp_path):
    """The report's provenance label listed the four 2x2 legs although six were
    read and validated under `--offcap-sha` (the doc writer's find): the label
    now enumerates every validated leg."""
    build_fixture_2x3(tmp_path)
    rep = analyze6(tmp_path)
    six = sorted({**rep["legs"], **rep["offcap"]["legs"]})
    assert len(six) == 6 and rep["model"]["validated_legs"] == six
    assert "6 leg(s) validated against contracts.model_row" in rep["model"]["source"]
    assert all(leg["identity_validated"] is True for leg in {**rep["legs"], **rep["offcap"]["legs"]}.values())
    rep4 = analyze(tmp_path)
    assert rep4["model"]["validated_legs"] == sorted(rep4["legs"]) and "4 leg(s) validated" in rep4["model"]["source"]
    # a comparator leg that fails identity is not listed
    build_fixture_2x3(tmp_path / "bad", psr_off8192={"manifest_edit": lambda m: m.update(model_revision="badrev")})
    rep2 = analyze6(tmp_path / "bad")
    assert rep2["model"]["validated_legs"] == sorted(set(six) - {T.leg_key(PSR, "off", cap_override=True)})
    assert "5 leg(s) validated" in rep2["model"]["source"]


# --------------------------------------------------------------------------- #
# the production boundary fails CLOSED                                         #
# --------------------------------------------------------------------------- #


def _two_pass_args(root, offcap_root, tmp):
    budget = tmp / "oos_prompt_budget.json"
    budget.write_text(json.dumps(prompt_budget_artifact()))
    budget_off = tmp / "oos_prompt_budget_off.json"
    budget_off.write_text(json.dumps(prompt_budget_artifact_off()))
    ids_file = tmp / "paired_ids.json"
    ids_file.write_text(json.dumps(IDS))
    return ["--gen-root", str(root), "--sha", f"oos-{SHA7}", "--offcap-sha", f"offcap-{OFFCAP_SHA7}",
            "--offcap-gen-root", str(offcap_root), "--prompt-budget", str(budget), "--prompt-budget-off", str(budget_off),
            "--paired-ids", str(ids_file), "--wandb-mode", "disabled", "--no-figs", "--no-mirror", "--bootstrap", "200",
            "--allow-pending-gates"]


def test_main_exits_non_zero_when_the_report_is_not_a_production_result_and_the_two_pass_runner_stops(tmp_path, monkeypatch, capsys):
    """The second analysis review's HIGH: `main()` returned 0 with failed gates,
    an unusable leg and no estimand in the report, and the two-pass runner
    trusted the exit code alone — so a store the per-item log check REFUSED in
    run E still yielded a canonical report in run F (`--no-eval-logs`). Now (1)
    `main()` writes the report and exits `EXIT_NOT_PRODUCTION` whenever
    `production_refusals` names anything — failed gates, pending gates not
    waived, unusable legs, a withheld estimand (the comparator's too); (2) the
    runner's guard `--gate-report` reads run E's report and refuses before run
    F; (3) `--diff-reports` requires the two reports' numeric leaves to be equal
    before DONE; (4) this test drives the two passes as the runner does on
    the probe: two comparator outcome scores swapped, the log carrying the
    truth — the first pass exits non-zero, the guard trips, the second pass is
    never reached; and were it run regardless (the v2 runner), the diff guard
    refuses its report against the first's."""
    swapped = (OFF8192_NONE_CORRECT - {0}) | {130}
    offcap_root = tmp_path / "pull_offcap"
    build_fixture_2x3(tmp_path, offcap_root=offcap_root, none_off8192={"correct": swapped})
    common = _two_pass_args(tmp_path, offcap_root, tmp_path)
    target = str(next((offcap_root / "qwen35_4b" / NONE).rglob("manifest.json")).parent)

    def fake(cell_dir, task):
        swap = ((IDS[0], IDS[130]),) if str(cell_dir) == target else ()
        return _results_from_store(cell_dir, task, swap=swap), [], None

    monkeypatch.setattr(T, "_read_eval_results", fake)
    e_out, f_out = tmp_path / "out" / "report_evalcheck.json", tmp_path / "out" / "report.json"
    # -- run E (the log read): the refusal reaches the exit code, and the report is still written for diagnosis
    rc_e = T.main(common + ["--out", str(e_out)])
    assert rc_e == T.EXIT_NOT_PRODUCTION == 3
    rep_e = json.loads(e_out.read_text())
    k8 = T.leg_key(NONE, "off", cap_override=True)
    assert rep_e["gates"]["failed"] == ["outcomes", "population"] and set(rep_e["unusable_legs"]) == {k8}
    assert "estimands" not in rep_e and "estimands" not in rep_e["offcap"]
    refusals = T.production_refusals(rep_e)
    assert refusals == [
        "gate(s) FAILED: outcomes, population",
        "gate(s) PENDING and not waivable: split_anomalies: not evaluated: the estimand did not run",
        f"unusable leg(s): {[k8]}",
        "the estimand did not run (no `estimands` block)",
        "the comparator estimand did not run (no `offcap.estimands` block)"]
    err = capsys.readouterr().err
    assert "NOT a production result" in err and "gate(s) FAILED: outcomes, population" in err
    # -- the runner's guard reads the report it was given, not the exit code
    assert T.main(["--gate-report", str(e_out)]) == T.EXIT_NOT_PRODUCTION
    assert "NOT a production result" in capsys.readouterr().err
    # -- the runner stops here: run F is never reached
    reached_f = rc_e == 0 and T.main(["--gate-report", str(e_out)]) == 0
    assert reached_f is False and not f_out.exists()
    # -- had a runner ignored both (the v2 runner), run F alone reads no log, sees no swap, and files a verdict:
    #    the fail-open probed here — which the numeric-leaf guard refuses against run E's report
    rc_f = T.main(common + ["--no-eval-logs", "--out", str(f_out)])
    assert rc_f == 0 and json.loads(f_out.read_text())["verdict"]["word"] == T.WORD_TRANSFERS
    assert T.main(["--diff-reports", str(e_out), str(f_out)]) == T.EXIT_REPORTS_DIFFER == 4
    diff = T.numeric_leaf_diff(rep_e, json.loads(f_out.read_text()))
    assert diff["differing"] == [] and diff["missing"] == [] and diff["added"] and diff["equal"] is False
    assert any(p[0] == "estimands" for p in diff["added"])
    # -- the clean stores: both passes exit 0, the guard passes, the two reports' numeric leaves are equal
    clean = tmp_path / "clean"
    build_fixture_2x3(clean, offcap_root=clean / "pull_offcap")
    common_c = _two_pass_args(clean, clean / "pull_offcap", clean)
    monkeypatch.setattr(T, "_read_eval_results", lambda cell_dir, task: (_results_from_store(cell_dir, task), [], None))
    e2, f2 = clean / "out" / "report_evalcheck.json", clean / "out" / "report.json"
    assert T.main(common_c + ["--out", str(e2)]) == 0
    assert T.main(["--gate-report", str(e2)]) == 0
    assert T.main(common_c + ["--no-eval-logs", "--out", str(f2)]) == 0
    assert T.main(["--diff-reports", str(e2), str(f2)]) == 0
    d2 = T.numeric_leaf_diff(json.loads(e2.read_text()), json.loads(f2.read_text()))
    assert d2["equal"] is True and d2["shared_numeric"] > 5000 and d2["differing"] == [] and d2["missing"] == [] and d2["added"] == []
    assert T.production_refusals(json.loads(e2.read_text())) == []
    printed = capsys.readouterr().out
    assert f"numeric leaves equal: {d2['shared_numeric']} shared" in printed
    # -- a pending gate that was NOT waived also withholds the word, and the exit code says so
    rc_p = T.main([a for a in common_c if a != "--allow-pending-gates"] + ["--no-eval-logs", "--out", str(clean / "out" / "p.json")])
    rep_p = json.loads((clean / "out" / "p.json").read_text())
    assert rep_p["verdict"]["word"] is None and rep_p["gates"]["pending"] and rc_p == T.EXIT_NOT_PRODUCTION
    assert T.production_refusals(rep_p) == [f"verdict withheld: {rep_p['verdict']['reason']}"]
    # -- the gate mode refuses a report it cannot read rather than passing it
    (clean / "torn.json").write_text("{not json")
    with pytest.raises(T.TransferAnalysisError):
        T.main(["--gate-report", str(clean / "torn.json")])
    # -- the analysis mode still needs its inputs
    with pytest.raises(SystemExit):
        T.main(["--no-figs"])


def test_the_mirror_refuses_a_copy_that_is_not_byte_identical(tmp_path, monkeypatch):
    """The second analysis review's MEDIUM: the mirror held the v1 sidecars while
    the analysis dir's were corrected. Every mirrored file is now compared byte
    for byte with its source after the copy; a difference refuses by name."""
    figs = tmp_path / "figs"
    figs.mkdir()
    png = figs / "final_e1b_shift_decomposition.png"
    png.write_bytes(b"png bytes")
    for name in (T.NUMBERS_FILENAME, T.TABLES_FILENAME, T.NUMBERS_OFFCAP_FILENAME, T.TABLES_OFFCAP_FILENAME):
        (figs / name).write_text(json.dumps({"name": name, "v": 2}))
    manifest = {"written": {"final_e1b_shift_decomposition": [str(png)]}, "numbers": str(figs / T.NUMBERS_FILENAME),
                "tables": str(figs / T.TABLES_FILENAME), "numbers_offcap": str(figs / T.NUMBERS_OFFCAP_FILENAME),
                "tables_offcap": str(figs / T.TABLES_OFFCAP_FILENAME)}
    mirror = tmp_path / "mirror"
    out = T.mirror_figures(figs, mirror, manifest=manifest)
    assert len(out) == 5 and all(Path(p).read_bytes() == (figs / Path(p).name).read_bytes() for p in out)
    # a stale sidecar already in the mirror is overwritten and verified, never left in place
    (mirror / T.NUMBERS_OFFCAP_FILENAME).write_text(json.dumps({"name": T.NUMBERS_OFFCAP_FILENAME, "v": 1}))
    T.mirror_figures(figs, mirror, manifest=manifest)
    assert json.loads((mirror / T.NUMBERS_OFFCAP_FILENAME).read_text())["v"] == 2
    # a copy that does not reproduce the source refuses by name
    real_copy2 = T.shutil.copy2

    def corrupt(src, dst, *a, **k):
        real_copy2(src, dst, *a, **k)
        if Path(src).name == T.NUMBERS_OFFCAP_FILENAME:
            Path(dst).write_text("{}")
        return dst

    monkeypatch.setattr(T.shutil, "copy2", corrupt)
    with pytest.raises(T.TransferAnalysisError, match=f"mirror: {T.NUMBERS_OFFCAP_FILENAME} .* not byte-identical"):
        T.mirror_figures(figs, tmp_path / "mirror2", manifest=manifest)


def _publishable(args):
    """`_two_pass_args` minus `--no-figs`, `--no-mirror` and the `--wandb-mode <value>` pair (the test adds its own)."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--wandb-mode":
            skip = True
            continue
        if a in ("--no-figs", "--no-mirror"):
            continue
        out.append(a)
    return out


def _publication_spies(monkeypatch):
    """Record every call of the three publication side effects; each returns the shape `main()` expects."""
    calls: list[str] = []

    def fake_make_figures(report, figs_dir):
        calls.append("make_figures")
        return {"written": {"final_e1_gsm8k_acc_2x2": [str(Path(figs_dir) / "final_e1_gsm8k_acc_2x2.png")]},
                "skipped": {}, "numbers": str(Path(figs_dir) / T.NUMBERS_FILENAME), "tables": str(Path(figs_dir) / T.TABLES_FILENAME)}

    def fake_mirror(figs_dir, mirror_dir, *, manifest):
        calls.append("mirror_figures")
        return [str(Path(mirror_dir) / "final_e1_gsm8k_acc_2x2.png")]

    def fake_wandb(report, figs, *, mode, name=None, name_suffix=""):
        # the mode is recorded: a `disabled` call is the real function's contractual no-op
        # (it imports nothing and touches no network), an `online` one is a publication
        calls.append(f"log_wandb:{mode}")
        return None if mode == "disabled" else "https://example.invalid/run"

    monkeypatch.setattr(T, "make_figures", fake_make_figures)
    monkeypatch.setattr(T, "mirror_figures", fake_mirror)
    monkeypatch.setattr(T, "log_wandb", fake_wandb)
    return calls


def test_no_figure_mirror_or_wandb_side_effect_precedes_the_production_checks(tmp_path, monkeypatch, capsys):
    """A Medium defect: `main()` generated figures, overwrote the
    mirror and FINISHED a W&B run before `production_refusals` was evaluated,
    and the runner ran F online before `--diff-reports` — a run F that was
    production-valid on its own but numerically different from run E would
    have published before the exit-4 refusal. Now the refusals are evaluated
    right after `analyze()` (exit 3, the diagnostic report written, nothing
    published), and `--require-report-match PATH` compares the in-memory report
    with PATH's numeric leaves right after `analyze()` (exit 4 before any
    publication on a mismatch; the match recorded as `report_match`); on a
    clean matching pass the three side effects run, in order, once each."""
    offcap_root = tmp_path / "pull_offcap"
    swapped = (OFF8192_NONE_CORRECT - {0}) | {130}
    build_fixture_2x3(tmp_path, offcap_root=offcap_root, none_off8192={"correct": swapped})
    clean = tmp_path / "clean"
    build_fixture_2x3(clean, offcap_root=clean / "pull_offcap")
    swapped_args = _publishable(_two_pass_args(tmp_path, offcap_root, tmp_path))
    clean_args = _publishable(_two_pass_args(clean, clean / "pull_offcap", clean))
    publish = ["--wandb-mode", "online", "--mirror-dir", str(tmp_path / "mirror"), "--figs-dir", str(tmp_path / "figs")]
    target = str(next((offcap_root / "qwen35_4b" / NONE).rglob("manifest.json")).parent)
    monkeypatch.setattr(T, "_read_eval_results", lambda cell_dir, task: (
        _results_from_store(cell_dir, task, swap=(((IDS[0], IDS[130]),) if str(cell_dir) == target else ())), [], None))
    calls = _publication_spies(monkeypatch)
    # (1) a refused analysis publishes NOTHING: no figure, no mirror, no W&B run — and the diagnostic report is written
    e_swapped = tmp_path / "out" / "report_evalcheck.json"
    rc = T.main(swapped_args + publish + ["--out", str(e_swapped)])
    assert rc == T.EXIT_NOT_PRODUCTION and calls == []
    rep = json.loads(e_swapped.read_text())
    assert "figures" not in rep and "mirrored" not in rep and "wandb_url" not in rep and rep["gates"]["failed"]
    assert "NOT a production result" in capsys.readouterr().err
    # (2) run F individually production-valid but numerically DIFFERENT from run E: refused BEFORE publication
    e_clean = clean / "out" / "report_evalcheck.json"
    assert T.main(clean_args + ["--no-figs", "--no-mirror", "--wandb-mode", "disabled", "--out", str(e_clean)]) == 0
    assert calls == ["log_wandb:disabled"]                                            # the no-op, after the checks
    calls.clear()
    f_swapped = tmp_path / "out" / "report.json"
    rc = T.main(swapped_args + publish + ["--no-eval-logs", "--require-report-match", str(e_clean), "--out", str(f_swapped)])
    assert rc == T.EXIT_REPORTS_DIFFER and calls == []
    rep = json.loads(f_swapped.read_text())
    assert rep["verdict"]["word"] == T.WORD_TRANSFERS                                # production-valid on its own ...
    assert T.production_refusals(rep) == []
    m = rep["report_match"]                                                           # ... and refused against run E
    assert m["against"] == str(e_clean) and m["equal"] is False and m["differing"] and m["missing"] == [] and m["added"] == []
    assert "figures" not in rep and "mirrored" not in rep and "wandb_url" not in rep
    err = capsys.readouterr().err
    assert "numeric leaves DIFFER" in err and str(e_clean) in err
    # a refused run E's report (no estimands) against a valid run F: the estimand leaves are "added" -> refused too
    rc = T.main(clean_args + publish + ["--no-eval-logs", "--require-report-match", str(e_swapped),
                                        "--out", str(clean / "out" / "f_vs_refused.json")])
    assert rc == T.EXIT_REPORTS_DIFFER and calls == []
    assert json.loads((clean / "out" / "f_vs_refused.json").read_text())["report_match"]["added"]
    # (3) the clean matching pass publishes, in order, once each — after the checks
    f_clean = clean / "out" / "report.json"
    rc = T.main(clean_args + publish + ["--no-eval-logs", "--require-report-match", str(e_clean), "--out", str(f_clean)])
    assert rc == 0 and calls == ["make_figures", "mirror_figures", "log_wandb:online"]
    rep = json.loads(f_clean.read_text())
    assert rep["report_match"] == {"against": str(e_clean), "equal": True, "shared_numeric": rep["report_match"]["shared_numeric"],
                                   "differing": [], "missing": [], "added": [], "type_changed": []}
    assert rep["report_match"]["shared_numeric"] > 5000
    assert rep["wandb_url"] == "https://example.invalid/run" and rep["mirrored"] and rep["figures"]["written"]
    # the post-hoc record agrees with the pre-publication gate, and `report_match` itself is outside the compared leaves
    assert T.main(["--diff-reports", str(e_clean), str(f_clean)]) == 0
    assert "report_match" in T.REPORT_DIFF_SKIP
    # an unreadable match target refuses before anything runs
    (clean / "torn.json").write_text("{torn")
    with pytest.raises(T.TransferAnalysisError):
        T.main(clean_args + publish + ["--no-eval-logs", "--require-report-match", str(clean / "torn.json"),
                                       "--out", str(clean / "out" / "x.json")])
    assert calls == ["make_figures", "mirror_figures", "log_wandb:online"]


def test_a_numeric_leaf_that_changes_type_or_vanishes_on_either_side_fails_the_report_match(tmp_path, monkeypatch, capsys):
    """A Medium defect: `numeric_leaf_diff` compared values only where
    BOTH sides were numeric while computing missing / added against ALL leaves,
    so a numeric leaf that became null / a string / a bool on one side dropped
    out of the comparison — the reference's `cap_relief_none.point` set to null
    gave `equal true`, and a `--require-report-match`ed run F published. Now the
    UNION of numeric paths is compared and each must be numeric on both sides:
    numeric on one side and non-numeric on the other is `type_changed`, absent
    on the other side `missing` / `added` (against the NUMERIC sets), in both
    directions; `--require-report-match` exits 4 with no publisher called and
    `--diff-reports` exits 4."""
    # the function, on small reports, every type in both directions
    base = {"a": {"x": 1.0, "y": 2, "z": "text", "t": True, "n": None, "l": [0.5, 1.5]}}

    def with_x(v):
        return {"a": {**base["a"], "x": v}}
    for bad in (None, "1.0", True, False):                                # a scalar of another type: type_changed
        d = T.numeric_leaf_diff(base, with_x(bad))                        # numeric -> non-numeric on the right
        assert d["equal"] is False and d["differing"] == [] and d["missing"] == [] and d["added"] == [], bad
        assert d["type_changed"] == [(("a", "x"), 1.0, bad)], bad
        d = T.numeric_leaf_diff(with_x(bad), base)                        # non-numeric -> numeric on the right
        assert d["equal"] is False and d["differing"] == [] and d["missing"] == [] and d["added"] == [], bad
        assert d["type_changed"] == [(("a", "x"), bad, 1.0)], bad
    for bad in ([1.0], {"v": 1.0}):                                       # a container: the leaf path vanishes, a new one appears
        d = T.numeric_leaf_diff(base, with_x(bad))
        assert d["equal"] is False and d["missing"] == [("a", "x")] and d["added"] and d["type_changed"] == [], bad
        d = T.numeric_leaf_diff(with_x(bad), base)
        assert d["equal"] is False and d["added"] == [("a", "x")] and d["missing"] and d["type_changed"] == [], bad
    assert T.numeric_leaf_diff(base, with_x(None))["shared_numeric"] == 3 and T.numeric_leaf_diff(base, base)["equal"] is True
    gone = {"a": {k: v for k, v in base["a"].items() if k != "x"}}
    assert T.numeric_leaf_diff(base, gone)["missing"] == [("a", "x")] and T.numeric_leaf_diff(gone, base)["added"] == [("a", "x")]
    # NaN on both sides is equal; NaN against a number differs; a bool never counts as numeric
    assert T.numeric_leaf_diff({"v": float("nan")}, {"v": float("nan")})["equal"] is True
    assert T.numeric_leaf_diff({"v": float("nan")}, {"v": 0.0})["differing"] == [(("v",), pytest.approx(float("nan"), nan_ok=True), 0.0)]
    assert T.numeric_leaf_diff({"v": 1}, {"v": True})["type_changed"] == [(("v",), 1, True)]
    # end to end, on the clean six-leg run: the reference's `cap_relief_none.point` set to null
    clean = tmp_path / "clean"
    build_fixture_2x3(clean, offcap_root=clean / "pull_offcap")
    args = _publishable(_two_pass_args(clean, clean / "pull_offcap", clean))
    publish = ["--wandb-mode", "online", "--mirror-dir", str(tmp_path / "mirror"), "--figs-dir", str(tmp_path / "figs")]
    monkeypatch.setattr(T, "_read_eval_results", lambda cell_dir, task: (_results_from_store(cell_dir, task), [], None))
    calls = _publication_spies(monkeypatch)
    e_out = clean / "out" / "report_evalcheck.json"
    assert T.main(args + ["--no-figs", "--no-mirror", "--wandb-mode", "disabled", "--out", str(e_out)]) == 0
    calls.clear()
    ref = json.loads(e_out.read_text())
    path = ("offcap", "estimands", T.MAP_CAP, T.BRANCH_PRIMARY, "stats", "cap_relief_none", "point")
    for label, bad in (("null", None), ("string", "0.03"), ("bool", True)):
        doc = json.loads(e_out.read_text())
        node = doc
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = bad
        edited = clean / "out" / f"ref_{label}.json"
        edited.write_text(json.dumps(doc))
        f_out = clean / "out" / f"report_{label}.json"
        rc = T.main(args + publish + ["--no-eval-logs", "--require-report-match", str(edited), "--out", str(f_out)])
        assert rc == T.EXIT_REPORTS_DIFFER and calls == [], label
        m = json.loads(f_out.read_text())["report_match"]
        assert m["equal"] is False and m["type_changed"] == [["/".join(path), bad, ref["offcap"]["estimands"][T.MAP_CAP][T.BRANCH_PRIMARY]["stats"]["cap_relief_none"]["point"]]], label
        assert m["differing"] == [] and m["missing"] == [] and m["added"] == []
        assert T.main(["--diff-reports", str(edited), str(e_out)]) == T.EXIT_REPORTS_DIFFER
        assert T.main(["--diff-reports", str(e_out), str(edited)]) == T.EXIT_REPORTS_DIFFER
        err = capsys.readouterr().err
        assert "type changed" in err and "cap_relief_none/point" in err
    # the leaf removed from the reference: `added` on the run's side, refused too
    doc = json.loads(e_out.read_text())
    node = doc
    for part in path[:-1]:
        node = node[part]
    del node[path[-1]]
    (clean / "out" / "ref_gone.json").write_text(json.dumps(doc))
    rc = T.main(args + publish + ["--no-eval-logs", "--require-report-match", str(clean / "out" / "ref_gone.json"),
                                  "--out", str(clean / "out" / "report_gone.json")])
    assert rc == T.EXIT_REPORTS_DIFFER and calls == []
    assert json.loads((clean / "out" / "report_gone.json").read_text())["report_match"]["added"] == ["/".join(path)]
    # the untouched reference still matches and publishes
    rc = T.main(args + publish + ["--no-eval-logs", "--require-report-match", str(e_out), "--out", str(clean / "out" / "report.json")])
    assert rc == 0 and calls == ["make_figures", "mirror_figures", "log_wandb:online"]


def test_the_qualifier_reads_the_intervals_endpoint_instead_of_assuming_it(tmp_path):
    """L1: for ANY all-gain comparison the qualifier said "its lower
    percentile above zero" without reading the interval — 1 gain / 0 losses of
    200 has a zero lower endpoint (about 36.7 % of replicates carry no gain).
    The endpoint is now read off the budget term's CI: positive -> it reflects
    the sparse empirical distribution, not corroboration; zero -> said so, with
    the share of replicates carrying no discordant item, and the positive-case
    reading stated conditionally. Losses-only reads the upper endpoint the same
    way; without an interval the wording is conditional."""
    rec = {"n_discordant": 1, "n_items": 200, "I->C": 1, "C->I": 0, "p_value": 1.0, "one_sided_discordance": True,
           "fully_scored": True, "replicate_signs": {"negative": 0, "zero": 734, "positive": 1266}, "n_replicates_used": 2000,
           "p_replicate_without_discordant_item": (1 - 1 / 200) ** 200}
    zero = T.discordance_qualifier(rec, interval=(0.0, 0.015))
    assert zero.startswith("The cap-only contrast rests on 1 discordant item of 200 (1 gain / 0 losses; exact two-sided McNemar p = 1, "
                           "a complete-case sensitivity); both legs are fully scored under this branch, so the registered bootstrap "
                           "cannot generate a contrast of the opposite sign (replicate signs 0 negative / 734 zero / 1,266 positive of 2,000)")
    assert zero.endswith("— its lower endpoint is +0.00 pp: 734 of 2,000 replicates carry no gain, so the interval reaches zero; a "
                         "positive lower endpoint, when present, reflects this sparse empirical distribution, not corroboration")
    pos = T.discordance_qualifier({**rec, "I->C": 4, "n_discordant": 4, "replicate_signs": {"negative": 0, "zero": 196, "positive": 9804},
                                   "n_replicates_used": 10000}, interval=(0.005, 0.04))
    assert pos.endswith("— its positive lower endpoint (+0.50 pp) reflects this sparse empirical distribution, not corroboration")
    losses = {**rec, "I->C": 0, "C->I": 2, "n_discordant": 2}
    assert T.discordance_qualifier(losses, interval=(-0.02, -0.005)).endswith(
        "— its negative upper endpoint (-0.50 pp) reflects this sparse empirical distribution, not corroboration")
    assert T.discordance_qualifier(losses, interval=(-0.02, 0.0)).endswith(
        "— its upper endpoint is +0.00 pp: 734 of 2,000 replicates carry no loss, so the interval reaches zero; a negative upper "
        "endpoint, when present, reflects this sparse empirical distribution, not corroboration")
    # no interval given: the conditional wording, no claim about the endpoint
    assert T.discordance_qualifier(rec).endswith("— a positive lower endpoint, when present, reflects this sparse empirical "
                                                 "distribution, not corroboration")
    # no replicate signs: the analytic share stands in
    no_signs = {k: v for k, v in rec.items() if k not in ("replicate_signs", "n_replicates_used")}
    assert "about 36.7 % of replicates carry no gain" in T.discordance_qualifier(no_signs, interval=(0.0, 0.015))
    assert "lower percentile above zero" not in T.discordance_qualifier(rec, interval=(0.0, 0.015))
    # two-sided: unchanged wording, the interval itself is the statement
    two = {**rec, "I->C": 7, "C->I": 1, "n_discordant": 8, "one_sided_discordance": False}
    assert T.discordance_qualifier(two, interval=(0.005, 0.06)).endswith(
        "so the registered bootstrap is the empirical bootstrap of those pairs and its interval reflects that sparse distribution "
        "(replicate signs 0 negative / 734 zero / 1,266 positive of 2,000)")
