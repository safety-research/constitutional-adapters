"""The GSM8K paired-200 noise-floor post-processor (``scripts/sdfbase/gsm8k/g8k_noise_floor.py``).

Synthetic stores only (``tests/gsm8k_synthetic_store.py``: placeholder shas, invented items, ``TESTHOST``), written through
the REAL producers, read by the REAL analysis (``scripts/tinycap_gsm8k_paired200.py``) into a production report, and then
post-processed here — so the pairs' estimand is the report's own, the ``none`` pair equals the report's ``noise_floor``
block leaf for leaf, and the attestation reads the same manifests / request records / ledger rows the reader reads.
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
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_gsm8k_paired200", REPO / "scripts" / "tinycap_gsm8k_paired200.py")
O = _load("gsm8k_overlap_check", REPO / "scripts" / "gsm8k_overlap_check.py")
NF = _load("g8k_noise_floor", REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_noise_floor.py")
V = A.V

B_FAST = 400
LLAMA, NONE, CONST = G.LLAMA_KEY, G.CELL_NONE, G.CELL_PROMPTED_CONST
BANK_VIEW = {"main": "main", "it": None}
ROWS = [r for r in G.matrix_rows_llama() if r["cell"] in (NONE, CONST)]          # the two no-object cells, both views
DOC = G.matrix_doc_llama(ROWS)
MAIN_CORRECT, IT_CORRECT = set(range(180)), set(range(177)) | {181, 182}         # none: 3 main-only, 2 it-only, 177 both, 18 neither
CONST_CORRECT = set(range(170))                                                   # constitution: identical outcomes


def _overlap(root):
    return O.run(retired_root=root / "retired", stock_root=root / "overlap", models=[LLAMA], canonical_ids=G.IDS, ids_path=None,
                 pairing_rows=G.pairing_rows(), pairing_source="synthetic rows", repo_root=REPO, store_pin=G.SHA7, n_resamples=B_FAST, seed=0,
                 ci=0.95, eval_logs=False)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("g8k_nf")
    correct = {(LLAMA, "main", NONE): MAIN_CORRECT, (LLAMA, "it", NONE): IT_CORRECT, (LLAMA, "main", CONST): CONST_CORRECT, (LLAMA, "it", CONST): CONST_CORRECT}
    overrides = {k: {"bank_view": BANK_VIEW[k[1]], "correct": v} for k, v in correct.items()}
    dirs = G.build_all(root, rows=ROWS, overrides=overrides)
    assert len(dirs) == 4
    return root


@pytest.fixture(scope="module")
def report(store):
    rep = A.analyze(store, G.SHA7, matrix=DOC, canonical_ids=G.IDS, sampling_rows=G.SAMPLING_ROWS, render_facts=G.RENDER_FACTS_ALL,
                    pins=G.pins_doc(ROWS), n_resamples=B_FAST, read_eval_logs=False, analysis_sha=G.FULL_SHA, overlap=_overlap(store))
    assert not A.production_refusals(rep), A.production_refusals(rep)
    return rep


@pytest.fixture(scope="module")
def doc(report):
    return NF.build(report, model_key=None, cells=NF.DEFAULT_CELLS, n_resamples=None, seed=None, ci=None, gen_root=None)


def _pair(doc, cell):
    return next(p for p in doc["pairs"] if p["cell"] == cell)


def test_the_per_item_reading_classes_every_item_on_the_same_judged_outcome_and_lists_the_discordant_ids(doc):
    none = _pair(doc, NONE)
    assert none["computable"] and none["kind"] == "noise_floor" and none["main_key"] == f"{LLAMA}/main/{NONE}" and none["it_key"] == f"{LLAMA}/it/{NONE}"
    for reading in A.READINGS:                       # no cell escalated in this fixture: the merged reading is the 8,192 one
        pi = none["per_item"][reading]
        assert pi["n_items"] == 200 and pi["classes"] == {NF.CLASS_BOTH_CORRECT: 177, NF.CLASS_BOTH_WRONG: 18, NF.CLASS_MAIN_ONLY: 3,
                                                          NF.CLASS_IT_ONLY: 2, NF.CLASS_UNSCORED: 0}
        assert pi["n_discordant"] == 5
        assert pi["discordant_ids"][NF.CLASS_MAIN_ONLY] == sorted(G.IDS[i] for i in (177, 178, 179))
        assert pi["discordant_ids"][NF.CLASS_IT_ONLY] == sorted(G.IDS[i] for i in (181, 182))
    const = _pair(doc, CONST)
    assert const["computable"] and const["per_item"][A.READING_MERGED]["n_discordant"] == 0
    assert const["per_item"][A.READING_MERGED]["classes"][NF.CLASS_BOTH_CORRECT] == 170


def test_the_aggregate_spread_is_the_analysis_own_estimand_and_the_none_pair_equals_the_reports_noise_floor_block(doc, report):
    none = _pair(doc, NONE)
    c = none["readings"][A.READING_MERGED]["compact"]
    assert c["point"] == pytest.approx(-0.005) and c["acc_main"] == pytest.approx(0.9) and c["acc_it"] == pytest.approx(0.895)
    assert c["ci_low"] <= -0.005 <= c["ci_high"] and c["n_scored_by_leg"] == {"main": 200, "it": 200}
    full = none["readings"][A.READING_MERGED]
    assert set(full) >= {"estimands", "mcnemar", "transitions", "verdict", "compact"}
    assert full["transitions"][A.AXIS_STOCK]["counts"] == {"C->C": 177, "C->I": 3, "I->C": 2, "I->I": 18, "->U": 0}
    # the report's own noise_floor block (none(it) - none(main) by the same code) — equal leaf for leaf, and recorded as checked
    ref = next(n for n in report["noise_floor"] if n["model_key"] == LLAMA and n["computable"])
    for r in A.READINGS:
        assert json.dumps(ref["readings"][r]["estimands"], sort_keys=True, default=str) == json.dumps(full["estimands"] if r == A.READING_MERGED
                                                                                                     else none["readings"][r]["estimands"], sort_keys=True, default=str)
    assert none["consistency_with_report"] == {"checked": True, "equal": True,
                                               "detail": "estimands (every axis, both branches) of the report's noise_floor entry vs this pair's"}
    const = _pair(doc, CONST)
    cc = const["readings"][A.READING_MERGED]["compact"]
    assert cc["point"] == 0.0 and cc["ci_low"] <= 0.0 <= cc["ci_high"] and not cc["excludes_zero"]
    assert const["consistency_with_report"] is None          # the report carries no constitution pair; nothing to compare against
    assert doc["bootstrap"] == report["bootstrap"]           # the report's B / seed / level by default
    # a different bootstrap than the report's is refused on the none pair by name (the leaf-for-leaf agreement is the check)
    with pytest.raises(A.AnalysisError, match="differ from the report's own noise_floor block"):
        NF.build(report, model_key=None, cells=(NONE,), n_resamples=B_FAST // 2, seed=None, ci=None, gen_root=None)


def test_the_attestation_reads_both_stores_and_labels_the_pair_a_floor_only_when_every_check_passes(doc):
    for cell in (NONE, CONST):
        att = _pair(doc, cell)["attestation"]
        assert att["label"] == NF.LABEL_FLOOR and att["failing"] == [] and all(c["passed"] for c in att["checks"] if c["attested"]), att["checks"]
        assert all(c["passed"] is None for c in att["checks"] if not c["attested"])
        assert all(c["attested"] for c in att["checks"] if c["name"] in ("serving_geometry", "batch_composition", "judge_cache_state", "request_bytes"))
        names = [c["name"] for c in att["checks"]]
        assert {"serving_geometry", "batch_composition", "judge_cache_state", "request_bytes", "seeds", "cell_identity", "model_revision",
                "served_template_hash", "sampling_row", "cap", "thinking_mode", "producer_tree_sha", "ledger_thinking_absent"} <= set(names)
        assert all(s["equal"] for s in att["shared"] if s["name"] != "identity_hashes"), [s for s in att["shared"] if not s["equal"]]
        assert next(s for s in att["shared"] if s["name"] == "serving_geometry")["present"]
        # the fixture's placeholder hashes carry the view label, so they differ: recorded, compared, never label-deciding
        hashes = next(s for s in att["shared"] if s["name"] == "identity_hashes")
        assert set(hashes["main"]) == {"intervention_hash", "harness_config_hash"} and "hashes differ" in att["label_text"]
        assert att["designed_difference"]["bank_view"] == {"main": "main", "it": None}
        w = att["wire"]
        assert w["n_items"] == 200 and w["n_request_bodies_identical"] == 200 == w["n_common_requests"] and w["n_seeds_identical"] == 200
        assert w["n_prompt_tokens_identical"] == 200 and 0 <= w["n_completions_byte_identical"] <= 200
        judge = next(c for c in att["checks"] if c["name"] == "judge_cache_state")
        assert judge["detail"].startswith(NF.SCORING_SENTENCE) and "deterministic string matching" in judge["detail"]
        assert set(att["incidental"]["main"]) == set(att["incidental"]["it"]) >= {"slurm_job_id", "ts_first", "ts_last", "latency_mean_s"}
        assert "floor" in att["label_text"] and "by construction" in att["label_text"]
        for name in att["not_attestable"]:
            assert name in att["label_text"]
    # the raw store records (request bodies, ledger rows, rows) never travel in the document — the counts summarise them
    for p in doc["pairs"]:
        assert not ({"requests", "ledger", "rows"} & set(p["attestation"]))


def test_a_planted_difference_flips_the_label_and_names_the_check(report):
    rcells, legs = report["cells"], report["legs"]
    fm = NF.store_facts(legs[rcells[f"{LLAMA}/main/{NONE}"]["base_leg"]])
    fi = NF.store_facts(legs[rcells[f"{LLAMA}/it/{NONE}"]["base_leg"]])
    assert NF.attest_pair(fm, fi)["label"] == NF.LABEL_FLOOR
    geo = json.loads(json.dumps(fi))
    geo["geometry"]["max_model_len"] = 99_999
    att = NF.attest_pair(fm, geo)
    assert att["label"] == NF.LABEL_MEASURES and att["failing"] == ["serving_geometry"] and "measures the difference in: serving_geometry" in att["label_text"]
    assert next(s for s in att["shared"] if s["name"] == "serving_geometry")["equal"] is False
    batch = json.loads(json.dumps(fi))
    batch["concurrency"] = (fi["concurrency"] or 0) + 1
    assert NF.attest_pair(fm, batch)["failing"] == ["batch_composition"]
    wire = json.loads(json.dumps(fi))
    some = sorted(wire["requests"])[0]
    wire["requests"][some]["body_sha256"] = "0" * 64
    att = NF.attest_pair(fm, wire)
    assert att["failing"] == ["request_bytes"] and att["wire"]["n_request_bodies_identical"] == 199
    judged = json.loads(json.dumps(fi))
    judged["judge"]["scorer_keys"] = ["think_aware_scorer", "model_graded_judge"]
    att = NF.attest_pair(fm, judged)
    assert att["failing"] == ["judge_cache_state"] and "NOT attested" in next(c for c in att["checks"] if c["name"] == "judge_cache_state")["detail"]
    seeds = json.loads(json.dumps(fi))
    first = sorted(seeds["rows"])[0]
    seeds["rows"][first]["seed"] = (seeds["rows"][first]["seed"] or 0) + 1
    assert NF.attest_pair(fm, seeds)["failing"] == ["seeds"]


def test_the_figure_and_the_document_land_with_every_plotted_number_resolving_in_the_document(doc, tmp_path):
    pytest.importorskip("matplotlib")
    d = json.loads(json.dumps(doc, default=str))
    figs = NF.write_outputs(d, out=tmp_path / "noise_floor.json", figs_dir=tmp_path / "figs")
    stem = f"{NF.STEM}_{LLAMA}"
    assert set(figs["written"]) == {stem} and all(Path(p).is_file() for p in figs["written"][stem])
    assert sorted(Path(p).suffix for p in figs["written"][stem]) == [".pdf", ".png"]
    assert figs["numbers"], "the figure records what it plotted"
    for rec in figs["numbers"]:
        assert A.T._same_leaf(rec["value"], A.T.report_path(d, rec["source"])), rec
    names = {rec["name"] for rec in figs["numbers"]}
    assert f"{LLAMA}: {NONE}(it) - {NONE}(main).{A.READING_MERGED}.point" in names
    assert f"{LLAMA}: {CONST}(it) - {CONST}(main).discordant: main correct only" in names
    written = json.loads((tmp_path / "noise_floor.json").read_text())
    assert written["schema"] == NF.SCHEMA and written["model_key"] == LLAMA and written["thinking_label"] == A.NA_LABEL
    assert written["figure"]["written"] == figs["written"] and [s["name"] for s in written["summary"]] == [p["name"] for p in written["pairs"]]
    assert written["summary"][0]["label"] == NF.LABEL_FLOOR and written["summary"][0][f"n_discordant_{A.READING_MERGED}"] == 5
    # the figure's title: two lines of <= 90 characters (check_title refuses otherwise), axes >= 80 % of the height
    fig, _n = NF.fig_noise_floor(d)
    bold = [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text().split("\n")
    assert len(bold) == 2 and all(len(line) <= A.TITLE_MAX_CHARS for line in bold) and A.NA_LABEL in bold[1]
    assert A.axes_height_fraction(fig) >= A.AXES_MIN_FRACTION
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_the_figure_claims_identity_only_when_every_pair_attests_a_floor_and_names_a_missing_or_failed_attestation(doc, report):
    """Code review: the figure defaulted a MISSING attestation (--no-stores) to the row label "floor" and
    kept the title's "identical by construction" claim even over an explicit `measures_difference` verdict — so a reproduced or
    non-equivalent comparison could be published as measured nondeterminism. Both the row labels and the title derive from the
    attestation now: affirmative wording only when every computable pair attests a floor; an unattested pair says so and
    claims no equivalence; a failed pair names the differing checks."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    def bold_lines(fig):
        return [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text().split("\n")
    def row_labels(fig):
        return [t.get_text() for t in fig.axes[0].get_yticklabels()]
    # (a) the stores read, every check passed: the affirmative title, every row a floor (the state the doc fixture is in)
    d = json.loads(json.dumps(doc, default=str))
    assert all(NF.pair_label(p) == NF.LABEL_FLOOR for p in d["pairs"] if p["computable"])
    fig, _n = NF.fig_noise_floor(d)
    lines = bold_lines(fig)
    assert "identical by construction" in lines[0] and NF.FIGURE_TITLE_TAIL in lines[1] and all(lab.endswith("floor") for lab in row_labels(fig))
    plt.close(fig)
    # (b) --no-stores: no attestation exists -> UNATTESTED rows, a title that claims no equivalence, the summary label says so
    un = NF.build(report, model_key=None, cells=NF.DEFAULT_CELLS, n_resamples=B_FAST, seed=0, ci=0.95, gen_root=None, read_stores=False)
    assert all(p["attestation"] is None and NF.pair_label(p) == NF.LABEL_UNATTESTED for p in un["pairs"] if p["computable"])
    assert all(s["label"] == NF.LABEL_UNATTESTED for s in un["summary"] if s["computable"])
    fig, _n = NF.fig_noise_floor(json.loads(json.dumps(un, default=str)))
    lines = bold_lines(fig)
    assert "UNATTESTED" in lines[0] and "NOT verified" in lines[1] and "identical by construction" not in "\n".join(lines)
    assert all("unattested" in lab and "floor" not in lab.split("\n")[-1].replace("unattested (stores not read)", "") for lab in row_labels(fig))
    assert len(lines) == 2 and all(len(line) <= A.TITLE_MAX_CHARS for line in lines) and A.NA_LABEL in lines[1]
    plt.close(fig)
    # (c) a planted geometry mismatch on one pair: that row "measures a difference", the title names the check, no identity claim
    rcells, legs = report["cells"], report["legs"]
    fm = NF.store_facts(legs[rcells[f"{LLAMA}/main/{NONE}"]["base_leg"]])
    fi = json.loads(json.dumps(NF.store_facts(legs[rcells[f"{LLAMA}/it/{NONE}"]["base_leg"]])))
    fi["geometry"]["max_model_len"] = 99_999
    planted = json.loads(json.dumps(doc, default=str))
    none_pair = next(p for p in planted["pairs"] if p["cell"] == NONE)
    none_pair["attestation"] = NF.attest_pair(fm, fi)
    assert NF.pair_label(none_pair) == NF.LABEL_MEASURES
    fig, _n = NF.fig_noise_floor(planted)
    lines = bold_lines(fig)
    assert "measure a DIFFERENCE" in lines[0] and "serving_geometry" in lines[1] and "identical by construction" not in "\n".join(lines)
    by_cell = {lab.split("\n")[0]: lab.split("\n")[-1] for lab in row_labels(fig)}          # first line = the cell's short name
    assert by_cell[A._short(NONE)] == "measures a difference" and by_cell[A._short(CONST)] == "floor"
    assert len(lines) == 2 and all(len(line) <= A.TITLE_MAX_CHARS for line in lines)
    plt.close(fig)
    # a label outside the vocabulary refuses by name rather than reading as anything
    bad = json.loads(json.dumps(doc, default=str)); bad["pairs"][0]["attestation"]["label"] = "floorish"
    with pytest.raises(NF.AnalysisError, match="not one of"):
        NF.pair_label(bad["pairs"][0])


def test_the_cli_refuses_by_name_and_files_an_absent_cell_as_not_computable(report, tmp_path, capsys):
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report, default=str))
    rc = NF.main(["--report", str(p), "--out", str(tmp_path / "nf.json"), "--cells", f"{NONE},bogus__cell", "--no-stores"])
    assert rc == 0
    out = json.loads((tmp_path / "nf.json").read_text())
    bogus = next(x for x in out["pairs"] if x["cell"] == "bogus__cell")
    assert not bogus["computable"] and "not in the design" in bogus["reason"] and bogus["attestation"] is None
    none = next(x for x in out["pairs"] if x["cell"] == NONE)
    assert none["computable"] and none["attestation"] is None                       # --no-stores: readings + discordance only
    assert "not computable" in capsys.readouterr().out
    with pytest.raises(A.AnalysisError, match="not run in both views"):
        NF.build(report, model_key="gemma4e4b", cells=(NONE,), n_resamples=None, seed=None, ci=None, gen_root=None)
    broken = {**json.loads(json.dumps(report, default=str))}
    broken["gates"] = {**broken["gates"], "A1_ledger": {**broken["gates"]["A1_ledger"], "status": A.GATE_FAIL}, "failed": ["A1_ledger"]}
    with pytest.raises(A.AnalysisError, match="not a production result"):
        NF.build(broken, model_key=None, cells=(NONE,), n_resamples=None, seed=None, ci=None, gen_root=None)


def test_the_post_processor_is_not_a_second_writer_of_the_ladder_record():
    src = (REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_noise_floor.py").read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "cap_" + "ladder.json" not in code and "CAP_LADDER" not in code
