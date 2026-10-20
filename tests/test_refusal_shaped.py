"""E3 tier-2 item 8: the REFUSAL-SHAPED class — ``sisteer.capeval.tinycap.refusal_shaped`` (a fixed, judge-free pattern list over
the scored draws' CONTENT channel) counted BESIDE n_correct / n_unscored in the GSM8K paired-200 recorder and the paper-rows export
(schema paper_rows/4: ``n_refusal_shaped`` WITH its denominator ``n_refusal_shaped_of`` — the draws the classifier examined — as the last
two fields, null together when never counted), never changing a denominator or a grader value. Unit tests of the module against KNOWN-GOOD
and KNOWN-BAD inputs (a template refusal, an ordinary answer, an empty ANSWER line, a missing channel; a readout over rows the outcome
record does not know; off-contract readout blocks), then end to end over the SYNTHETIC GSM8K stores of ``tests/gsm8k_synthetic_store.py``
with refusal-shaped completions INJECTED into scored draws (three wrong, one the grader scored correct — filed as a finding)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sisteer.capeval.tinycap import paper_rows as PRW
from sisteer.capeval.tinycap import refusal_shaped as RS

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))
import gsm8k_synthetic_store as G  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_gsm8k_paired200", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
V = A.V
GEMMA, NONE, CONST_LORA = "gemma4e4b", "none", "constv2-mix-full__lora_r4_ce"
B_FAST = 50
#: a TEMPLATE refusal (the generic phrasing the pattern list names; invented, not recorded data) and an ordinary answer
REFUSAL = "I cannot fulfill this request. I am programmed to follow strict safety guidelines, which prohibit this content."
ANSWER = "Step by step: 3 + 4 = 7.\nANSWER: 7"


# --------------------------------------------------------------------------- the module


def test_classify_names_the_patterns_that_matched_and_nothing_else():
    c = RS.classify(REFUSAL)
    assert c["refusal_shaped"] is True and c["text_present"] is True
    assert c["patterns"] == ["cannot_fulfill", "strict_safety_guidelines", "programmed_to_follow"]
    assert RS.classify(ANSWER) == {"refusal_shaped": False, "patterns": [], "text_present": True}
    assert RS.classify("Working...\nANSWER:\n")["patterns"] == ["empty_answer_line"]
    assert RS.classify("ANSWER: 12")["refusal_shaped"] is False                       # a filled answer line is not the empty one
    assert RS.classify("i'm unable to help with that.")["patterns"] == ["unable_to"]   # case-insensitive
    assert RS.classify("I must decline.")["patterns"] == ["must_decline"]
    assert RS.classify("As an AI language model I cannot do arithmetic on request.")["patterns"] == ["cannot_provide_assist_help", "as_an_ai_cannot"] or \
        "as_an_ai_cannot" in RS.classify("As an AI language model I cannot do arithmetic on request.")["patterns"]
    # a missing channel is recorded, never counted
    assert RS.classify(None) == {"refusal_shaped": False, "patterns": [], "text_present": False}
    assert RS.PATTERN_IDS == tuple(pid for pid, _ in RS.PATTERNS) and "empty_answer_line" in RS.PATTERN_IDS
    assert RS.SCHEMA == "refusal_shaped/1" and RS.TEXT_CHANNEL == "content"


def _row(iid, content, *, value):
    return {"item_id": iid, "value": value, "_extra": {"content": content, "reasoning": "some deliberation about whether to answer"}}


def test_readout_counts_over_the_scored_draws_splits_by_grader_outcome_and_refuses_rows_the_record_does_not_know():
    rows = [_row("a", REFUSAL, value=0.0), _row("b", ANSWER, value=1.0), _row("c", "ANSWER:\n", value=0.0), _row("d", REFUSAL, value=1.0)]
    items = {i: {"scored": True, "value": r["value"]} for i, r in zip("abcd", rows)}
    items["e"] = {"scored": False, "value": None}          # an unscored draw has no row and is not read
    ro = RS.readout(rows, items)
    assert ro["schema"] == RS.SCHEMA and ro["n_rows"] == 4 and ro["n_text_present"] == 4
    # the DENOMINATOR the count is over: the draws the classifier actually examined (content channel on record), enumerated by id
    assert ro["n_examined"] == 4 and ro["examined_ids"] == ["a", "b", "c", "d"]
    assert ro["n_refusal_shaped"] == 3 and ro["n_refusal_shaped_correct"] == 1 and ro["n_refusal_shaped_wrong"] == 2
    assert ro["ids"] == ["a", "c", "d"] and ro["ids_correct"] == ["d"]
    # a row with no content channel is neither examined nor counted — and the gap is on record (n_rows - n_examined)
    ro2 = RS.readout(rows + [{"item_id": "f", "value": 0.0, "_extra": {"content": None, "reasoning": "only a reasoning channel"}}],
                     {**items, "f": {"scored": True, "value": 0.0}})
    assert ro2["n_rows"] == 5 and ro2["n_examined"] == 4 and "f" not in ro2["examined_ids"] and ro2["n_refusal_shaped"] == 3
    assert RS.readout_problems(ro2, n_scored=5) == []
    assert ro["by_pattern"]["cannot_fulfill"] == 2 and ro["by_pattern"]["empty_answer_line"] == 1 and ro["by_pattern"]["must_decline"] == 0
    assert ro["patterns_by_id"]["c"] == ["empty_answer_line"] and set(ro["patterns_by_id"]) == {"a", "c", "d"}
    # the reasoning channel is never read: a refusal there alone is not counted
    assert RS.readout([{"item_id": "z", "value": 0.0, "_extra": {"content": ANSWER, "reasoning": REFUSAL}}], {"z": {"scored": True, "value": 0.0}})["n_refusal_shaped"] == 0
    with pytest.raises(RS.RefusalShapedError, match="not in the outcome record"):
        RS.readout([_row("q", REFUSAL, value=0.0)], items)
    with pytest.raises(RS.RefusalShapedError, match="unscored in the outcome record"):
        RS.readout([_row("e", REFUSAL, value=0.0)], items)
    with pytest.raises(RS.RefusalShapedError, match="appears twice"):
        RS.readout([_row("a", REFUSAL, value=0.0), _row("a", REFUSAL, value=0.0)], items)
    assert RS.readout_problems(ro, n_scored=4) == []
    assert RS.readout_problems(ro, n_scored=2) == ["refusal_shaped: n_examined 4 > n_scored 2"]
    assert RS.readout_problems(None, n_scored=4) == ["refusal_shaped: no readout block"]
    # KNOWN-BAD twins of the count/denominator pair: a count without its denominator; a count larger than what was examined; an examined
    # set that does not contain a counted draw
    assert any("count without its denominator" in p for p in RS.readout_problems({k: v for k, v in ro.items() if k != "n_examined"}, n_scored=4))
    assert any("n_refusal_shaped 3 > n_examined 2" in p for p in RS.readout_problems({**ro, "n_examined": 2, "examined_ids": ["a", "b"]}, n_scored=4))
    assert any("examined_ids" in p for p in RS.readout_problems({**ro, "examined_ids": ["a", "b", "c", "e"]}, n_scored=4))
    assert any("schema" in p for p in RS.readout_problems({**ro, "schema": "x"}, n_scored=4))
    assert any("!= correct" in p for p in RS.readout_problems({**ro, "n_refusal_shaped_wrong": 5}, n_scored=4))
    assert any("enumerate" in p for p in RS.readout_problems({**ro, "ids": ["a"]}, n_scored=4))
    assert any("by_pattern" in p for p in RS.readout_problems({**ro, "by_pattern": {"cannot_fulfill": 2}}, n_scored=4))


# --------------------------------------------------------------------------- the export contract


def test_make_row_defaults_the_count_null_and_the_verifier_bounds_it_by_n_scored():
    base = dict(instrument=PRW.INSTRUMENT_GSM8K, model_key=GEMMA, view="main", cell=CONST_LORA, arm="lora_mix", cap=8192, leg="r8192",
                n_paired=200, n_scored=190, n_unscored=10, n_at_cap=10, at_cap_rate=0.05, n_unterminated_not_at_cap=0, n_correct=171,
                point_conditional=0.9, ci_conditional_low=0.85, ci_conditional_high=0.94, point_score_as_is=0.855, ci_score_as_is_low=0.8,
                ci_score_as_is_high=0.9, ci_method=PRW.CI_EXACT, rung_state=PRW.RUNG_NOT_APPLICABLE, object_generation=PRW.GEN_NA,
                selection_outcome=None, unscored_reason=PRW.unscored_reason(10, unterminated_think_block=10)[0], source="a test")
    row = PRW.make_row(**base)
    assert tuple(row) == PRW.FIELDS and tuple(row)[-2:] == ("n_refusal_shaped", "n_refusal_shaped_of")
    assert row["n_refusal_shaped"] is None and row["n_refusal_shaped_of"] is None and PRW.row_problems(row) == []     # never counted: null TOGETHER
    # the count carries its denominator: a count with its n is auditable in a way a share is not
    assert PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=0, n_refusal_shaped_of=190)) == []
    assert PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=190, n_refusal_shaped_of=190)) == []
    assert PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=3, n_refusal_shaped_of=150)) == []                # examined fewer than scored: allowed
    # KNOWN-BAD twins: a count without its denominator, a denominator without its count — each refused by the schema check
    assert any("must be null together or present together" in p for p in PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=3)))
    assert any("must be null together or present together" in p for p in PRW.row_problems(PRW.make_row(**base, n_refusal_shaped_of=190)))
    assert any("n_refusal_shaped 5 > n_refusal_shaped_of 4" in p for p in PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=5, n_refusal_shaped_of=4)))
    assert any("n_refusal_shaped_of 191 > n_scored 190" in p for p in PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=3, n_refusal_shaped_of=191)))
    assert any("not non-negative integers" in p for p in PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=-1, n_refusal_shaped_of=190)))
    assert any("not non-negative integers" in p for p in PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=True, n_refusal_shaped_of=190)))
    assert any("not non-negative integers" in p for p in PRW.row_problems(PRW.make_row(**base, n_refusal_shaped=3, n_refusal_shaped_of=0.5)))
    # a no-reading row: the pair null together with every other count field; a value on either is a stray
    null_row = PRW.make_row(**{**base, **{f: None for f in PRW.COUNT_FIELDS}, "rung_state": PRW.RUNG_REFUSED})
    assert null_row["n_refusal_shaped"] is None and null_row["n_refusal_shaped_of"] is None and PRW.row_problems(null_row) == []
    assert any("carry values" in p for p in PRW.row_problems({**null_row, "n_refusal_shaped": 0, "n_refusal_shaped_of": 0}))
    assert any("carry values" in p for p in PRW.row_problems({**null_row, "n_refusal_shaped_of": 0}))


# --------------------------------------------------------------------------- end to end: the recorder + the export over a synthetic store


def analyze(gen_root, sha=G.SHA7, *, matrix, **kw):
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("sampling_rows", G.SAMPLING_ROWS)
    kw.setdefault("render_facts", G.RENDER_FACTS)
    kw.setdefault("pins", G.pins_doc(G.matrix_rows()))
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    kw.setdefault("overlap", None)
    return A.analyze(gen_root, sha, matrix=matrix, **kw)


def _small(rows, *triples):
    keep = set(triples)
    return [r for r in rows if (r["model_key"], r["view"], r["cell"]) in keep]


@pytest.fixture(scope="module")
def refusal_store(tmp_path_factory):
    """gemma's `main` none anchor and its constitution-LoRA cell, the latter with refusal-shaped content on three WRONG draws and one
    CORRECT draw (the answer line after the refusal text — the grader scored it), plus an empty ANSWER line on a fourth wrong draw."""
    root = tmp_path_factory.mktemp("g8k_refusal")
    rows = _small(G.eight_arm_matrix(with_steer=False), (GEMMA, "main", NONE), (GEMMA, "main", CONST_LORA))
    none_row, cell_row = G.row_for(rows, GEMMA, "main", NONE), G.row_for(rows, GEMMA, "main", CONST_LORA)
    correct = set(G.CORRECT_BY_CELL[CONST_LORA])
    wrong = [j for j in range(len(G.IDS)) if j not in correct]
    refusal_wrong, empty_wrong, refusal_correct = wrong[:3], wrong[3], sorted(correct)[0]
    texts = {j: (REFUSAL, "should I answer this arithmetic problem?") for j in refusal_wrong}
    texts[empty_wrong] = ("Working through it...\nANSWER:\n", None)
    texts[refusal_correct] = (f"{REFUSAL}\nBut the arithmetic is {G.ANSWERS[refusal_correct]}.\nANSWER: {G.ANSWERS[refusal_correct]}", None)
    G.build_store(root, none_row, correct=G.CORRECT_NONE, producer_fields=True)
    G.build_store(root, cell_row, correct=correct, texts=texts, producer_fields=True)
    return {"root": root, "rows": rows, "refusal_wrong": refusal_wrong, "empty_wrong": empty_wrong, "refusal_correct": refusal_correct,
            "n_correct": len(correct)}


def test_the_recorder_counts_the_class_beside_the_accuracy_counts_and_the_export_carries_it_without_moving_a_denominator(refusal_store):
    s = refusal_store
    rep = analyze(str(s["root"]), matrix=s["rows"], waves=["published"])
    lk, lk_none = V.leg_key(GEMMA, "main", CONST_LORA, G.CAP_BASE), V.leg_key(GEMMA, "main", NONE, G.CAP_BASE)
    leg, none_leg = rep["legs"][lk], rep["legs"][lk_none]
    assert A.usable(leg) and A.usable(none_leg), (leg.get("refused"), none_leg.get("refused"))
    ro = leg["refusal_shaped"]
    expect_ids = sorted(G.IDS[j] for j in (*s["refusal_wrong"], s["empty_wrong"], s["refusal_correct"]))
    assert ro["schema"] == RS.SCHEMA and ro["ids"] == expect_ids and ro["n_refusal_shaped"] == 5
    assert ro["n_refusal_shaped_correct"] == 1 and ro["ids_correct"] == [G.IDS[s["refusal_correct"]]] and ro["n_refusal_shaped_wrong"] == 4
    assert ro["by_pattern"]["cannot_fulfill"] == 4 and ro["by_pattern"]["empty_answer_line"] == 1
    assert none_leg["refusal_shaped"]["n_refusal_shaped"] == 0 and none_leg["refusal_shaped"]["ids"] == []
    # the DENOMINATOR: every scored draw of the synthetic store carries a content channel, so the classifier examined exactly the scored draws
    assert ro["n_examined"] == leg["summary"]["n_scored"] == len(ro["examined_ids"]) and set(ro["ids"]) <= set(ro["examined_ids"])
    assert none_leg["refusal_shaped"]["n_examined"] == none_leg["summary"]["n_scored"]
    # the flags ride the reading's items; the reading sums both the count and its denominator; the accuracy counts are the store's, untouched
    ck = V.cell_key(GEMMA, "main", CONST_LORA)
    rd = rep["cells"][ck]["readings"][A.READING_R8192]
    assert rd["n_refusal_shaped"] == 5 == sum(1 for it in rd["items"].values() if it.get(A.ITEM_REFUSAL_SHAPED_KEY))
    assert rd["n_refusal_shaped_of"] == rd["n_scored"] == sum(1 for it in rd["items"].values() if it.get(A.ITEM_REFUSAL_EXAMINED_KEY))
    assert rd["n_correct"] == s["n_correct"] and rd["n_scored"] == leg["summary"]["n_scored"] and rd["n"] == 200
    assert all(rd["items"][G.IDS[j]]["scored"] and rd["items"][G.IDS[j]]["value"] == 0.0 for j in s["refusal_wrong"])
    assert rd["items"][G.IDS[s["refusal_correct"]]]["value"] == 1.0
    # the paper rows: the count WITH its denominator on every reading row (5 of n_scored on the cell, 0 of n_scored on the anchor), the
    # pair as the last two fields in order, the rule in `source`
    prs, _ = A.paper_rows_of_report(rep)
    cell_rows = [r for r in prs if r["cell"] == CONST_LORA]
    none_rows = [r for r in prs if r["cell"] == NONE]
    assert cell_rows and all(r["n_refusal_shaped"] == 5 and r["n_correct"] == s["n_correct"] and r["n_paired"] == 200 for r in cell_rows)
    assert none_rows and all(r["n_refusal_shaped"] == 0 for r in none_rows)
    assert all(r["n_refusal_shaped_of"] == r["n_scored"] and r["n_refusal_shaped_of"] > 0 for r in prs)
    assert all(PRW.row_problems(r) == [] for r in prs) and all("n_refusal_shaped_of" in r["source"] for r in cell_rows)
    assert all(tuple(r) == PRW.FIELDS and tuple(r)[-2:] == ("n_refusal_shaped", "n_refusal_shaped_of") for r in prs)


def test_a_reading_written_before_the_class_existed_exports_null_never_zero_and_an_off_contract_block_refuses(refusal_store):
    s = refusal_store
    rep = analyze(str(s["root"]), matrix=s["rows"], waves=["published"])
    old = json.loads(json.dumps(rep))
    for cell in old["cells"].values():
        for rd in (cell.get("readings") or {}).values():
            rd.pop("n_refusal_shaped", None)
            rd.pop("n_refusal_shaped_of", None)
            for it in rd["items"].values():
                it.pop(A.ITEM_REFUSAL_SHAPED_KEY, None)
                it.pop(A.ITEM_REFUSAL_EXAMINED_KEY, None)
    prs, _ = A.paper_rows_of_report(old)
    assert prs and all(r["n_refusal_shaped"] is None and r["n_refusal_shaped_of"] is None for r in prs)     # null TOGETHER, never 0
    assert all("null together" in r["source"] for r in prs) and all(PRW.row_problems(r) == [] for r in prs)
    # a reading whose headline disagrees with its items refuses by name — the count and the denominator each
    ck = V.cell_key(GEMMA, "main", CONST_LORA)
    torn = json.loads(json.dumps(rep))
    torn["cells"][ck]["readings"][A.READING_R8192]["n_refusal_shaped"] = 1
    with pytest.raises(A.AnalysisError, match="n_refusal_shaped = 1 but its items sum to 5"):
        A.paper_rows_of_report(torn)
    torn_of = json.loads(json.dumps(rep))
    torn_of["cells"][ck]["readings"][A.READING_R8192]["n_refusal_shaped_of"] = 7
    with pytest.raises(A.AnalysisError, match="n_refusal_shaped_of = 7 but its items sum to"):
        A.paper_rows_of_report(torn_of)
    # a reading that filed the count WITHOUT its denominator is a count nobody can audit: refused by name, never exported half-null
    half = json.loads(json.dumps(rep))
    half["cells"][ck]["readings"][A.READING_R8192].pop("n_refusal_shaped_of")
    with pytest.raises(A.AnalysisError, match="without its pair"):
        A.paper_rows_of_report(half)


# --------------------------------------------------------------------------- the MATH-500 reader, the same class the same way


def test_the_math500_reader_files_the_same_readout_on_every_leg_and_its_export_carries_the_count(tmp_path):
    import math500_synthetic_store as S

    M5T = V.M
    rows = [r for r in S.cells_matrix() if r["model_key"] == GEMMA and r["sampling_mode"] == S.MODE_PRIMARY]
    assert {r["cell"] for r in rows} == set(S.CELLS)
    correct = set(S.CORRECT_BY_CELL[S.CELL_OBJECT])
    wrong = [j for j in range(len(S.IDS)) if j not in correct]
    refusal_wrong, refusal_correct = wrong[:2], sorted(correct)[0]
    texts = {j: (REFUSAL, None) for j in refusal_wrong}
    texts[refusal_correct] = (f"{REFUSAL}\nStill, the value is computed below.", None)
    root = tmp_path / "m5t"
    S.build_all(root, rows=rows, overrides={(GEMMA, S.CELL_OBJECT, S.MODE_PRIMARY): {"texts": texts}})
    rep = M5T.analyze(root, S.SHA7, cells=rows, sampling_rows=S.SAMPLING_ROWS, render_facts=S.RENDER_FACTS, math500_ids=S.IDS,
                      n_resamples=B_FAST, read_eval_logs=False, analysis_sha=S.FULL_SHA)
    legs = {lk: leg for lk, leg in rep["legs"].items() if leg.get("model_key") == GEMMA}
    assert legs and all(leg.get("refused") is None for leg in legs.values()), {lk: leg.get("refused") for lk, leg in legs.items()}
    obj = next(leg for leg in legs.values() if leg["cell"] == S.CELL_OBJECT)
    bare = next(leg for leg in legs.values() if leg["cell"] == S.CELL_BARE)
    ro = obj["refusal_shaped"]
    assert ro["schema"] == RS.SCHEMA and ro["n_refusal_shaped"] == 3 and ro["n_refusal_shaped_correct"] == 1
    assert ro["ids"] == sorted(S.IDS[j] for j in (*refusal_wrong, refusal_correct)) and bare["refusal_shaped"]["n_refusal_shaped"] == 0
    assert ro["n_examined"] == obj["summary"]["n_scored"] and bare["refusal_shaped"]["n_examined"] == bare["summary"]["n_scored"]
    prs, _ = M5T.paper_rows_of_report(rep)
    by_cell = {r["cell"]: r for r in prs if r["model_key"] == GEMMA}
    assert by_cell[S.CELL_OBJECT]["n_refusal_shaped"] == 3 and by_cell[S.CELL_BARE]["n_refusal_shaped"] == 0
    # the count carries its denominator on every row: the scored draws the classifier examined
    assert all(r["n_refusal_shaped_of"] == r["n_scored"] and r["n_refusal_shaped_of"] > 0 for r in prs)
    assert by_cell[S.CELL_OBJECT]["n_correct"] == len(correct) and by_cell[S.CELL_OBJECT]["n_paired"] == len(S.IDS)
    assert all(PRW.row_problems(r) == [] for r in prs) and "n_refusal_shaped_of" in by_cell[S.CELL_OBJECT]["source"]
    assert all(tuple(r)[-2:] == ("n_refusal_shaped", "n_refusal_shaped_of") for r in prs)
    # a leg written before the class existed exports the pair null TOGETHER with the note; an off-contract block refuses by name
    old = json.loads(json.dumps(rep))
    for leg in old["legs"].values():
        leg.pop("refusal_shaped", None)
    prs_old, _ = M5T.paper_rows_of_report(old)
    assert prs_old and all(r["n_refusal_shaped"] is None and r["n_refusal_shaped_of"] is None and "null together" in r["source"] for r in prs_old)
    bad = json.loads(json.dumps(rep))
    lk_obj = next(lk for lk, leg in legs.items() if leg["cell"] == S.CELL_OBJECT)
    bad["legs"][lk_obj]["refusal_shaped"]["n_refusal_shaped_wrong"] = 9
    with pytest.raises(M5T.AnalysisError, match="refusal_shaped"):
        M5T.paper_rows_of_report(bad)
