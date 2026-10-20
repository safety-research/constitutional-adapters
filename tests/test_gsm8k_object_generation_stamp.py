"""E3 tier-2 item 5: the GSM8K paired-200 recorder stamps a fitted cell's ``object_generation`` (``batch8`` | ``batch64``) from the
object's FIT PROVENANCE by sha256 — the it-doc admission record's ``fit_record.effective_batch`` (a derived object: its two parents') and
the declared batch-8 fit record ``configs/sdfbase/gsm8k/object_fit_batches.json`` — never from a staged id's spelling; the paper-rows
export reads the stamp off the cell record. Over the SHIPPED records (every it-doc matrix row binds and agrees with its declared word)
and the SYNTHETIC stores of ``tests/gsm8k_synthetic_store.py`` (an it-doc leg stamped through a binding; unstamped without one; a
declared word its provenance contradicts, or that nothing binds, refuses by name), plus KNOWN-BAD records the loaders refuse."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
PRW = A.PRW
G8KC = _load("g8k_cells_for_object_generation_test", REPO / "scripts" / "sdfbase" / "gsm8k" / "g8k_cells.py")
QWEN, NONE, ITDOC_CTL2 = "qwen35_4b", "none", G.ITDOC_CELL_CTL2
B_FAST = 50
ADMISSION = json.loads((REPO / V.ITDOC_ADMISSION_REL).read_text(encoding="utf-8"))
FIT_BATCHES_PATH = REPO / V.OBJECT_FIT_BATCHES_REL


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


# --------------------------------------------------------------------------- the shipped records


def test_the_shipped_records_bind_every_it_doc_object_and_every_it_doc_row_agrees_with_its_declared_word():
    fit_batches = V.load_object_fit_batches(FIT_BATCHES_PATH)
    bindings = V.object_generation_bindings(ADMISSION, fit_batches)
    # the six batch-64 objects (four fitted parents + two derived differences) from the admission record's fit records, the two
    # batch-8 direct-fit objects from the declared record — eight shas, no word read off any staged id
    by_gen = {g: sorted(s for s, b in bindings.items() if b["generation"] == g) for g in (V.OBJECT_GENERATION_BATCH8, V.OBJECT_GENERATION_BATCH64)}
    assert len(by_gen[V.OBJECT_GENERATION_BATCH8]) == 2 and len(by_gen[V.OBJECT_GENERATION_BATCH64]) == 6 and len(bindings) == 8
    assert set(by_gen[V.OBJECT_GENERATION_BATCH8]) == set(fit_batches)
    for sha, e in fit_batches.items():
        assert e["effective_batch"] == 8 and e["source"]["object_pt_sha256_on_disk"] == sha and e["source"]["file"] == "train_budget_meta.json"
        # each declared object is the admission record's `superseded` direct-fit object of its (model, it, cell)
        sup = ADMISSION["entries"][e["model_key"]][e["view"]][e["cell"]]["superseded"]
        assert sup["object_sha256"] == sha and sup["staged_id"] == e["staged_id"]
    for key, by_view in ADMISSION["entries"].items():
        for view, by_cell in by_view.items():
            for cell, e in by_cell.items():
                b = bindings[e["object_sha256"]]
                assert b["generation"] == V.OBJECT_GENERATION_BATCH64 and b["effective_batch"] == 64
                if e["object_kind"] == "fitted":
                    assert e["fit_record"]["effective_batch"] == 64 and "fit_record.effective_batch" in b["source"]
                else:
                    assert "two parents" in b["source"]
    # the it-doc matrix: every fitted row's declared word is its binding's; the prompted rows carry no word
    matrix = json.loads((REPO / "configs" / "sdfbase" / "gsm8k" / "matrix_itdoc.json").read_text(encoding="utf-8"))
    for row in matrix["rows"]:
        stamp, problem = V.object_generation_of_row(row, bindings)
        assert problem is None, problem
        if row["fitted"]:
            assert stamp == row["object_generation"] == G8KC.object_generation_of(row["staged_id"]), row["index"]
        else:
            assert stamp is None and row.get("object_generation") is None
    # the vocabulary is the export's and the launcher's, word for word
    assert set(V.OBJECT_GENERATION_BY_EFFECTIVE_BATCH.values()) == {PRW.GEN_BATCH8, PRW.GEN_BATCH64} == set(G8KC.OBJECT_GENERATIONS)


def test_the_loaders_refuse_known_bad_records_by_name(tmp_path):
    good = json.loads(FIT_BATCHES_PATH.read_text(encoding="utf-8"))
    sha0 = next(iter(good["entries"]))

    def variant(edit):
        doc = json.loads(json.dumps(good))
        edit(doc)
        return doc

    def bad_batch(doc):
        doc["entries"][sha0]["effective_batch"] = 32          # a batch the vocabulary has no word for
    with pytest.raises(A.AnalysisError, match="names no generation word"):
        V.load_object_fit_batches(variant(bad_batch))

    def bad_table(doc):
        doc["generation_by_effective_batch"] = {"8": "batch8", "64": "batch32"}
    with pytest.raises(A.AnalysisError, match="generation_by_effective_batch"):
        V.load_object_fit_batches(variant(bad_table))

    def no_disk_read(doc):
        doc["entries"][sha0]["source"].pop("object_pt_sha256_on_disk")
    with pytest.raises(A.AnalysisError, match="source lacks"):
        V.load_object_fit_batches(variant(no_disk_read))

    def key_mismatch(doc):
        doc["entries"][sha0]["source"]["object_pt_sha256_on_disk"] = "f" * 64
    with pytest.raises(A.AnalysisError, match="!= the key"):
        V.load_object_fit_batches(variant(key_mismatch))

    def bad_key(doc):
        doc["entries"]["not-a-sha"] = doc["entries"].pop(sha0)
    with pytest.raises(A.AnalysisError, match="not a 64-hex"):
        V.load_object_fit_batches(variant(bad_key))

    # a declared sha the admission record does not name as superseded cannot bind
    stranger = json.loads(json.dumps(good))
    e = stranger["entries"].pop(sha0)
    e["source"]["object_pt_sha256_on_disk"] = "a" * 64
    stranger["entries"]["a" * 64] = e
    with pytest.raises(A.AnalysisError, match="not a `superseded` object"):
        V.object_generation_bindings(ADMISSION, V.load_object_fit_batches(stranger))
    # an admission entry whose effective batch has no word refuses; two parents that disagree refuse
    adm = json.loads(json.dumps(ADMISSION))
    adm["entries"][QWEN]["it"][ITDOC_CTL2]["fit_record"]["effective_batch"] = 16
    with pytest.raises(A.AnalysisError, match="effective_batch 16 names no generation word"):
        V.object_generation_bindings(adm, None)
    adm2 = json.loads(json.dumps(ADMISSION))
    adm2["entries"][QWEN]["it"][ITDOC_CTL2]["fit_record"]["effective_batch"] = 8
    with pytest.raises(A.AnalysisError, match="parents' effective batches disagree"):
        V.object_generation_bindings(adm2, None)
    # the same sha bound to two words refuses
    adm3 = json.loads(json.dumps(ADMISSION))
    adm3["entries"][QWEN]["it"][ITDOC_CTL2]["fit_record"]["effective_batch"] = 8
    adm3["entries"][QWEN]["it"]["constv2-mix-full__lora_r4_ce"]["fit_record"]["effective_batch"] = 8
    adm3["parents"][QWEN]["control"]["object_sha256"] = adm3["entries"][QWEN]["it"]["constv2-mix-full__lora_r4_ce"]["object_sha256"]
    adm3["entries"]["gemma4e4b"]["it"][ITDOC_CTL2]["object_sha256"] = adm3["entries"][QWEN]["it"][ITDOC_CTL2]["object_sha256"]
    with pytest.raises(A.AnalysisError, match="bound to"):
        V.object_generation_bindings(adm3, None)


def test_object_generation_of_row_reads_the_binding_and_refuses_a_declared_word_the_provenance_does_not_carry():
    sha = "b" * 64
    bindings = {sha: {"generation": "batch64", "effective_batch": 64, "source": "a test binding"}}
    fitted = {"index": 1, "model_key": QWEN, "view": "it", "cell": ITDOC_CTL2, "fitted": True, "object_sha256": sha, "staged_id": "itdoc__x"}
    assert V.object_generation_of_row(fitted, bindings) == ("batch64", None)                       # the staged id says batch8; the record says batch64: the record wins
    assert V.object_generation_of_row({**fitted, "object_generation": "batch64"}, bindings) == ("batch64", None)
    stamp, problem = V.object_generation_of_row({**fitted, "object_generation": "batch8"}, bindings)
    assert stamp is None and "declares object_generation 'batch8' but its fit provenance binds batch64" in problem
    stamp, problem = V.object_generation_of_row({**fitted, "object_sha256": "c" * 64}, bindings)
    assert (stamp, problem) == (None, None)                                                          # unbound, undeclared: unstamped
    stamp, problem = V.object_generation_of_row({**fitted, "object_sha256": "c" * 64, "object_generation": "batch64"}, bindings)
    assert stamp is None and "no fit provenance binds object" in problem
    stamp, problem = V.object_generation_of_row({**fitted, "object_generation": "batch32"}, bindings)
    assert stamp is None and "not in" in problem
    stamp, problem = V.object_generation_of_row({**fitted, "fitted": False, "object_sha256": None, "object_generation": "batch8"}, bindings)
    assert stamp is None and "non-fitted" in problem
    assert V.object_generation_of_row({**fitted, "fitted": False, "object_sha256": None}, bindings) == (None, None)
    assert V.object_generation_of_row(fitted, None) == (None, None)


# --------------------------------------------------------------------------- the recorder + the export, end to end


@pytest.fixture(scope="module")
def itdoc_store(tmp_path_factory):
    """One root at the primary tree: qwen's `it` none anchor (published) and the it-doc control-LoRA leg (wave itdoc)."""
    root = tmp_path_factory.mktemp("g8k_gen")
    rows = G.itdoc_matrix(launch=[(QWEN, ITDOC_CTL2)])
    none_row, leg_row = G.row_for(rows, QWEN, "it", NONE), G.row_for(rows, QWEN, "it", ITDOC_CTL2)
    G.build_rows(root, [none_row], producer_fields=True)
    G.build_itdoc_leg(root, leg_row, tree_sha=G.FULL_SHA)
    return {"root": root, "rows": [none_row, leg_row], "leg_row": leg_row}


def _itdoc_rows(rep):
    rows, _ = A.paper_rows_of_report(rep)
    return [r for r in rows if r["cell"] == ITDOC_CTL2 and r["view"] == "it"]


def test_a_bound_it_doc_leg_is_stamped_on_the_cell_record_and_the_paper_row_reads_it(itdoc_store):
    s = itdoc_store
    pin = s["leg_row"]["object_sha256"]
    bindings = {pin: {"generation": "batch64", "effective_batch": 64, "source": "synthetic it-doc admission fit_record.effective_batch"}}
    rep = analyze(str(s["root"]), matrix=s["rows"], waves=["published", G.WAVE_ITDOC], object_generations=bindings)
    ck = V.cell_key(QWEN, "it", ITDOC_CTL2)
    assert rep["gates"]["A0_completeness"]["status"] == A.GATE_PASS and rep["unusable_legs"] == {}
    cell = rep["cells"][ck]
    assert cell["object_generation"] == "batch64" and cell["object_generation_source"] == bindings[pin]["source"]
    assert rep["cells"][V.cell_key(QWEN, "it", NONE)]["object_generation"] is None       # no object, no stamp
    prs = _itdoc_rows(rep)
    assert prs and all(r["object_generation"] == "batch64" for r in prs)
    assert all("fit-provenance binding" in r["source"] for r in prs)
    none_rows = [r for r in A.paper_rows_of_report(rep)[0] if r["cell"] == NONE and r["view"] == "it"]
    assert none_rows and all(r["object_generation"] == PRW.GEN_NONE for r in none_rows)


def test_an_unbound_fitted_leg_stays_unstamped_and_exports_n_a(itdoc_store):
    s = itdoc_store
    rep = analyze(str(s["root"]), matrix=s["rows"], waves=["published", G.WAVE_ITDOC], object_generations={})
    cell = rep["cells"][V.cell_key(QWEN, "it", ITDOC_CTL2)]
    assert cell["object_generation"] is None and cell["object_generation_source"] is None
    prs = _itdoc_rows(rep)
    assert prs and all(r["object_generation"] == PRW.GEN_NA for r in prs)


def test_a_declared_word_its_provenance_contradicts_or_that_nothing_binds_refuses_the_analysis_by_name(itdoc_store):
    s = itdoc_store
    pin = s["leg_row"]["object_sha256"]
    declared = [dict(r) for r in s["rows"]]
    declared[1]["object_generation"] = "batch64"
    bindings8 = {pin: {"generation": "batch8", "effective_batch": 8, "source": "a contradicting binding"}}
    with pytest.raises(A.AnalysisError, match="declares object_generation 'batch64' but its fit provenance binds batch8"):
        analyze(str(s["root"]), matrix=declared, waves=["published", G.WAVE_ITDOC], object_generations=bindings8)
    with pytest.raises(A.AnalysisError, match="no fit provenance binds object"):
        analyze(str(s["root"]), matrix=declared, waves=["published", G.WAVE_ITDOC], object_generations={})
    # a declared word that AGREES passes, and the export reads the same word
    bindings64 = {pin: {"generation": "batch64", "effective_batch": 64, "source": "an agreeing binding"}}
    rep = analyze(str(s["root"]), matrix=declared, waves=["published", G.WAVE_ITDOC], object_generations=bindings64)
    assert rep["cells"][V.cell_key(QWEN, "it", ITDOC_CTL2)]["object_generation"] == "batch64"
    assert all(r["object_generation"] == "batch64" for r in _itdoc_rows(rep))
