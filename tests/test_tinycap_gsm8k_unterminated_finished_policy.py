"""The GSM8K paired-200 reader's NAMED policy for unterminated-but-FINISHED draws (the rule of 2026-09-13 on the
GSM8K paired-200 artifact: a draw that ended the turn below the cap inside an unterminated think block is a failure of
the object, not a missing measurement).

Two behaviours, one flag (`--unterminated-finished-policy`, `analyze(..., unterminated_finished_policy=...)`):

* `unscored` (the DEFAULT): the store's scoring stands. The report is byte-identical to the one the reader wrote before
  the flag existed — no field is added anywhere, no value moves — so every other lane's report (the Llama wave, the
  eight-arm extension) is untouched.
* `failure`: exactly the draws with `think_block_unterminated and not at_cap` that the store left unscored are re-scored
  WRONG under BOTH think-block branches; the reading files the policy and its rescored count, each rescored item is
  flagged, `design` and the estimand sentence name the policy, and every paper row written under it says so in `source`.
  An AT-CAP unterminated draw (case (b)) is never touched; the redraw records stay the store's outcomes.

The fixtures are the lane's synthetic wave (tests/gsm8k_synthetic_store.py) reduced to Nemotron `main`: the bare cell,
the control-corpus PSR with 16 unterminated-but-finished draws and the control-corpus LoRA with one (the published wave's
shape: 167 / 184 and 186 / 199 scored), the constitution LoRA with two AT-CAP unterminated draws (case (b), 1.0 % — under
the trigger, so no rung) and the prompted cell.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import gsm8k_synthetic_store as G
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_gsm8k_paired200", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
V = A.V

B_FAST = 400
NEMO = "nemotron3nano4b"
NONE, PROMPTED = G.CELL_NONE, G.CELL_PROMPTED
CONST_LORA = "constv2-mix-full__lora_r4_ce"
CTL_PSR, CTL_LORA = "constv2ctl2-full__psr_ce_all", "constv2ctl2-full__lora_r4_ce"
ROWS = G.matrix_rows()
TRIPLES = ((NEMO, "main", NONE), (NEMO, "main", PROMPTED), (NEMO, "main", CTL_PSR), (NEMO, "main", CTL_LORA), (NEMO, "main", CONST_LORA))
#: the published wave's shape: 16 finished-inside-the-block draws on the control PSR, 1 on the control LoRA
FINISHED_PSR = set(range(184, 200))
FINISHED_LORA = {199}
#: case (b) on the constitution LoRA: two draws CUT at the cap inside the block (2 of 200 = 1.0 %, not over the trigger)
AT_CAP_CONST = {198, 199}
POLICY_KEYS = (A.UNTERMINATED_FINISHED_POLICY_KEY, A.N_UNTERMINATED_FINISHED_RESCORED_KEY, A.ITEM_RESCORED_UNTERMINATED_FINISHED_KEY)


def _rows():
    keep = set(TRIPLES)
    return [r for r in ROWS if (r["model_key"], r["view"], r["cell"]) in keep]


def _analyze(gen_root, rows, **kw):
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("sampling_rows", G.SAMPLING_ROWS)
    kw.setdefault("render_facts", G.RENDER_FACTS)
    kw.setdefault("pins", G.pins_doc(ROWS))
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(gen_root, G.SHA7, matrix=rows, **kw)


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    root = tmp_path_factory.mktemp("g8k_unterminated_finished")
    rows = _rows()
    assert len(rows) == len(TRIPLES)
    G.build_all(root, rows=rows, overrides={
        (NEMO, "main", CTL_PSR): {"unscored": FINISHED_PSR, "not_at_cap": FINISHED_PSR},
        (NEMO, "main", CTL_LORA): {"unscored": FINISHED_LORA, "not_at_cap": FINISHED_LORA},
        (NEMO, "main", CONST_LORA): {"unscored": AT_CAP_CONST},
    })
    return root, rows


def _ck(cell):
    return V.cell_key(NEMO, "main", cell)


def _dumps(report):
    return json.dumps(report, sort_keys=True, default=str)


def test_the_synthetic_wave_carries_the_published_shape(wave):
    root, rows = wave
    rep = _analyze(root, rows)
    # no overlap record is built for this reduced wave, so A10 is the ONE failed gate — the same under either policy (below)
    assert A.gates_failed(rep["gates"]) == ["A10_overlap"] and A.production_refusals(rep) == ["gate(s) FAILED: A10_overlap"]
    psr = rep["legs"][V.leg_key(NEMO, "main", CTL_PSR, G.CAP_BASE)]["summary"]
    assert psr["n_unterminated_not_at_cap"] == 16 and psr["n_think_block_unterminated"] == 16 and psr["n_scored"] == 184 and psr["n_at_cap"] == 0
    lora = rep["legs"][V.leg_key(NEMO, "main", CTL_LORA, G.CAP_BASE)]["summary"]
    assert lora["n_unterminated_not_at_cap"] == 1 and lora["n_scored"] == 199 and lora["n_at_cap"] == 0
    const = rep["legs"][V.leg_key(NEMO, "main", CONST_LORA, G.CAP_BASE)]["summary"]
    assert const["n_unterminated_not_at_cap"] == 0 and const["n_think_block_unterminated"] == 2 and const["n_at_cap"] == 2 and const["n_scored"] == 198
    # the store's own policy (gate A4) is the producer's and is untouched by the reader's flag
    assert rep["gates"]["A4_scoring_path"]["status"] == A.GATE_PASS
    assert all(v["unterminated_policy"] == "unscored" for v in rep["gates"]["A4_scoring_path"]["legs"].values())


def test_the_default_policy_writes_nothing_and_is_byte_identical_to_naming_it(wave):
    root, rows = wave
    implicit = _analyze(root, rows)
    explicit = _analyze(root, rows, unterminated_finished_policy=A.UNTERMINATED_FINISHED_UNSCORED)
    text = _dumps(implicit)
    assert text == _dumps(explicit)
    for key in POLICY_KEYS:
        assert key not in text, key
    assert implicit["estimand"] == A.ESTIMAND_TEXT
    assert A.UNTERMINATED_FINISHED_POLICY_KEY not in implicit["design"]
    rd = implicit["cells"][_ck(CTL_PSR)]["readings"]
    for reading in A.READINGS:
        assert rd[reading]["n_scored"] == 184 and rd[reading]["n_unscored"] == 16 and rd[reading]["n_think_unscored"] == 16
        assert all(not rd[reading]["items"][G.IDS[j]]["scored"] for j in FINISHED_PSR)
    # the default is the store's scoring, exactly: `apply` hands the same object back
    items = {"a": {"scored": False, "value": None, "at_cap": False, "think_block_unterminated": True}}
    out, n = A.apply_unterminated_finished_policy(items, A.UNTERMINATED_FINISHED_UNSCORED)
    assert out is items and n == 0


def test_the_failure_policy_rescores_exactly_the_finished_draws_and_names_itself(wave):
    root, rows = wave
    base = _analyze(root, rows)
    rep = _analyze(root, rows, unterminated_finished_policy=A.UNTERMINATED_FINISHED_FAILURE)
    # the policy adds no refusal and changes no gate: the same one failed gate (no overlap record in this wave) as the default
    assert A.production_refusals(rep) == A.production_refusals(base) == ["gate(s) FAILED: A10_overlap"]
    assert {k: v["status"] for k, v in rep["gates"].items() if isinstance(v, dict) and "status" in v} == \
        {k: v["status"] for k, v in base["gates"].items() if isinstance(v, dict) and "status" in v}
    assert rep["design"][A.UNTERMINATED_FINISHED_POLICY_KEY] == A.UNTERMINATED_FINISHED_FAILURE
    assert rep["estimand"] == A.ESTIMAND_TEXT + A.UNTERMINATED_FINISHED_FAILURE_TEXT and "failure" in rep["estimand"]
    # the 16 finished-inside-the-block draws are WRONG and scored; the count and the policy ride the reading
    for cell, finished, n_before in ((CTL_PSR, FINISHED_PSR, 184), (CTL_LORA, FINISHED_LORA, 199)):
        for reading in A.READINGS:
            rd, rd0 = rep["cells"][_ck(cell)]["readings"][reading], base["cells"][_ck(cell)]["readings"][reading]
            assert rd[A.UNTERMINATED_FINISHED_POLICY_KEY] == A.UNTERMINATED_FINISHED_FAILURE
            assert rd[A.N_UNTERMINATED_FINISHED_RESCORED_KEY] == len(finished)
            assert rd["n_scored"] == 200 and rd["n_unscored"] == 0 and rd["n_think_unscored"] == 0
            assert rd["n_correct"] == rd0["n_correct"] and rd0["n_scored"] == n_before
            assert rd["acc_stock"] == pytest.approx(rd["n_correct"] / 200) and rd["acc_stock"] < rd0["acc_stock"]
            for j in finished:
                it = rd["items"][G.IDS[j]]
                assert it["scored"] is True and it["value"] == 0.0 and it[A.ITEM_RESCORED_UNTERMINATED_FINISHED_KEY] is True
                assert it["think_block_unterminated"] is True and it["at_cap"] is False
            assert sum(1 for it in rd["items"].values() if it.get(A.ITEM_RESCORED_UNTERMINATED_FINISHED_KEY)) == len(finished)
    # case (b) — the at-cap unterminated draws — is NOT touched: the constitution LoRA reads exactly as under the default
    for reading in A.READINGS:
        rd, rd0 = rep["cells"][_ck(CONST_LORA)]["readings"][reading], base["cells"][_ck(CONST_LORA)]["readings"][reading]
        assert rd[A.N_UNTERMINATED_FINISHED_RESCORED_KEY] == 0 and rd["n_scored"] == 198 == rd0["n_scored"] and rd["n_think_unscored"] == 2
        assert all(not rd["items"][G.IDS[j]]["scored"] for j in AT_CAP_CONST)
        assert {k: v for k, v in rd.items() if k not in POLICY_KEYS} == rd0
    # so do the cells with no unterminated draw at all
    for cell in (NONE, PROMPTED):
        for reading in A.READINGS:
            rd, rd0 = rep["cells"][_ck(cell)]["readings"][reading], base["cells"][_ck(cell)]["readings"][reading]
            assert rd[A.N_UNTERMINATED_FINISHED_RESCORED_KEY] == 0
            assert {k: v for k, v in rd.items() if k not in POLICY_KEYS} == rd0


def _summary_of(report, cell):
    return next(c for c in report["contrast_summary"] if c["model_key"] == NEMO and c["view"] == "main" and c["cell"] == cell)


def test_under_failure_the_two_branches_coincide_on_the_affected_cells_and_every_other_contrast_is_unchanged(wave):
    root, rows = wave
    base = _analyze(root, rows)
    rep = _analyze(root, rows, unterminated_finished_policy=A.UNTERMINATED_FINISHED_FAILURE)
    for cell in (CTL_PSR, CTL_LORA):
        s1, s0 = _summary_of(rep, cell), _summary_of(base, cell)
        for reading in A.READINGS:
            pt1, pt0 = s1[reading]["point"], s0[reading]["point"]
            for axis in A.AXES:
                # the conditional reading now IS the score-as-is reading on this cell — the policy re-scores the same draws wrong
                assert pt1[axis][A.BRANCH_PRIMARY] == pytest.approx(pt1[axis][A.BRANCH_SCORE_AS_IS])
                assert pt1[axis][A.BRANCH_PRIMARY] == pytest.approx(pt0[axis][A.BRANCH_SCORE_AS_IS])
                assert pt1[axis][A.BRANCH_PRIMARY] < pt0[axis][A.BRANCH_PRIMARY]
            acc1 = s1[reading]["acc_by_leg_stock_primary"]["cell"]
            assert acc1["n_scored"] == 200 and acc1["n_censored"] == 0
    # the unaffected contrasts (the at-cap case-(b) cell, the prompted cell) read leaf for leaf as under the default
    for cell in (CONST_LORA, PROMPTED):
        assert _summary_of(rep, cell) == _summary_of(base, cell)
    # the verdict machinery is the same rule: the words on the unaffected contrasts are the default's
    words1 = {c["name"]: c["merged"]["word"] for c in rep["contrast_summary"] if c["computable"]}
    words0 = {c["name"]: c["merged"]["word"] for c in base["contrast_summary"] if c["computable"]}
    for name in words0:
        if CONST_LORA in name or PROMPTED in name:
            assert words1[name] == words0[name]


def test_the_paper_rows_read_the_policy_and_carry_it_in_source(wave, tmp_path):
    root, rows = wave
    rep = _analyze(root, rows, unterminated_finished_policy=A.UNTERMINATED_FINISHED_FAILURE)
    out = tmp_path / "report.json"
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    written = A.write_paper_rows_beside(rep, out)
    prw_rows = A.PRW.read_paper_rows(written["json"])
    by = {(r["cell"], r["leg"]): r for r in prw_rows}
    for cell, n_finished in ((CTL_PSR, 16), (CTL_LORA, 1)):
        for leg in ("r8192", "merged"):
            r = by[(cell, leg)]
            assert r["n_paired"] == 200 and r["n_scored"] == 200 and r["n_unscored"] == 0
            assert r["n_unterminated_not_at_cap"] == n_finished                       # the per-item FLAG count is the store's fact
            assert r["unscored_reason"]["unterminated_think_block"] == 0            # nothing is left unscored to explain
            assert r["point_conditional"] == pytest.approx(r["point_score_as_is"])
            assert f"{A.UNTERMINATED_FINISHED_POLICY_KEY}=failure: {n_finished} draw(s)" in r["source"]
    for leg in ("r8192", "merged"):
        r = by[(CONST_LORA, leg)]
        assert r["n_scored"] == 198 and r["n_unscored"] == 2 and r["unscored_reason"]["unterminated_think_block"] == 2
        assert r["point_conditional"] > r["point_score_as_is"]
        assert f"{A.UNTERMINATED_FINISHED_POLICY_KEY}=failure: no unterminated-but-finished draw" in r["source"]
    # the default export names no policy at all
    base = _analyze(root, rows)
    out0 = tmp_path / "default" / "report.json"
    out0.parent.mkdir()
    out0.write_text(json.dumps(base, indent=1, default=str), encoding="utf-8")
    text0 = Path(A.write_paper_rows_beside(base, out0)["json"]).read_text(encoding="utf-8")
    assert A.UNTERMINATED_FINISHED_POLICY_KEY not in text0


def test_the_cli_names_the_policy_and_refuses_an_unknown_one():
    ap = A.build_parser()
    assert ap.parse_args([]).unterminated_finished_policy == A.UNTERMINATED_FINISHED_UNSCORED
    assert ap.parse_args(["--unterminated-finished-policy", "failure"]).unterminated_finished_policy == A.UNTERMINATED_FINISHED_FAILURE
    with pytest.raises(SystemExit):
        ap.parse_args(["--unterminated-finished-policy", "wrong"])
    with pytest.raises(A.AnalysisError):
        A.apply_unterminated_finished_policy({}, "wrong")
    with pytest.raises(A.AnalysisError):
        A.Context(sampling_rows=G.SAMPLING_ROWS, render_facts=G.RENDER_FACTS, model_pins={}, store_pin=G.SHA7, expected_refusals={},
                  unterminated_finished_policy="wrong")
