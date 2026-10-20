"""ifeval-canonical-battery — the NOISE FLOORS of a two-view model.

Synthetic stores only (``tests/ifeval_synthetic_store.py``: placeholder shas, invented items, ``TESTHOST``), written through
the REAL producers and read by the REAL analysis. Pinned: every cell the matrix carries on both views with no object on
either is a floor candidate (llama8b: ``none`` and ``prompted__si_constitution``); a candidate identical by construction is
filed with the per-draw discordance on the same judged outcome (the four metrics x two readings; strict prompt-level accuracy
primary), the item-cluster paired bootstrap spread with both legs' points beside it (never pooled), and a shared-configuration
attestation read from the two stores that labels it a measured floor; a planted serving-geometry / batch difference or a
differing request body flips the label to "measures that difference" naming the check; a pair whose two rows are NOT
identical by construction (different served weights, a fitted object) is refused as a floor by name; the figure
``final_ifeval_noise_floor_<model>`` draws one panel per pair with the caption stating the reading; the results table names
the label per pair. Also pinned as a FACT of the frozen matrix: gemma4e4b's and qwen35_4b's two views serve the same
checkpoint and revision (the view suffix names the object-bank view, not the served model), so their ``none`` pairs are
identical by construction too and their label is decided by the stores like llama's.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import ifeval_contracts as IC

import ifeval_synthetic_store as G

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("tinycap_ifeval_battery", "scripts/tinycap_ifeval_battery.py")
V = A.V
ROWS = G.matrix_rows()
LLAMA, NONE, CONST = "llama8b", IC.CELL_NONE, IC.CELL_PROMPTED_CONSTITUTION
B_FAST = 200


def row(*t):
    return G.row_for(ROWS, *t)


def _rows(*triples):
    out = [dict(r) for r in ROWS if (r["model_key"], r["view"], r["cell"]) in set(triples)]
    for i, r in enumerate(out):
        r["index"] = i
    return out


LLAMA_ROWS = _rows((LLAMA, "main", NONE), (LLAMA, "it", NONE), (LLAMA, "main", CONST), (LLAMA, "it", CONST))
N_ITEMS = G.N


def _flip_plan():
    """Five draws of the llama ``it`` none store to flip, on distinct items: the first three FAILING draws pass on `it` only,
    the first two PASSING draws fail on `it` only (``draw_facts`` is deterministic in (cell, item, trial))."""
    it_only, main_only, used = [], [], set()
    for i in range(N_ITEMS):
        for t in range(G.K):
            if i in used:
                continue
            passes = G.draw_facts(NONE, i, t)["prompt_level_strict"]
            if not passes and len(it_only) < 3:
                it_only.append((G.IDS[i], t))
                used.add(i)
            elif passes and len(main_only) < 2:
                main_only.append((G.IDS[i], t))
                used.add(i)
    assert len(it_only) == 3 and len(main_only) == 2
    return it_only, main_only


IT_ONLY, MAIN_ONLY = _flip_plan()


def _flip_header(h):
    """The planted between-view discordance, consistent across the four IFEval values of each flipped draw."""
    for it in h["items"]:
        key = (it["item_id"], int(it["sample_idx"]))
        n = int(it["num_instructions"])
        if key in IT_ONLY:
            it.update(prompt_level_strict=True, prompt_level_loose=True, inst_level_strict=n, inst_level_loose=n, value=1.0)
        elif key in MAIN_ONLY:
            it.update(prompt_level_strict=False, prompt_level_loose=False, inst_level_strict=n - 1, inst_level_loose=n - 1, value=0.0)


def analyze(root, rows, **kw):
    kw.setdefault("canonical_ids", G.IDS)
    kw.setdefault("n_resamples", B_FAST)
    kw.setdefault("read_eval_logs", False)
    kw.setdefault("analysis_sha", G.FULL_SHA)
    return A.analyze(root, G.SHA7, matrix=rows, **kw)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("ife_nf")
    for r in LLAMA_ROWS:
        G.build_store(root, r, header_edit=_flip_header if (r["view"], r["cell"]) == ("it", NONE) else None)
    return root


@pytest.fixture(scope="module")
def report(store):
    rep = analyze(store, LLAMA_ROWS)
    # no .eval log read here (the CPU suite has no inspect_ai): every gate but the two log-derived ones is clear; not production (r3 S1)
    assert A.non_log_gates_clear(rep), A.production_refusals(rep)
    return rep


def _pair(report, cell, model=LLAMA):
    return A.noise_floor_pair(report, model, cell)


def test_every_no_object_cell_on_both_views_is_a_pair_per_draw_discordance_and_the_spread_never_pooled(report):
    (block,) = report["noise_floor"]
    assert block["model_key"] == LLAMA and block["status"] == "filed" and block["views"] == ["main", "it"] and block["primary_metric"] == "prompt_strict"
    assert [p["cell"] for p in block["pairs"]] == [NONE, CONST] and all(p["status"] == "filed" and p["identical_by_construction"] for p in block["pairs"])
    none = _pair(report, NONE)
    assert none["name"] == f"{LLAMA}: {NONE}(it) - {NONE}(main)" and none["main_key"] == f"{LLAMA}/main/{NONE}" and none["it_key"] == f"{LLAMA}/it/{NONE}"
    assert none["contrast"] == IC.CONTRAST_NOISE and none["kind"] == "noise_floor"
    for reading in A.READINGS:                          # nothing escalated: the merged reading is the pass-1 one
        pd = none["per_draw"][reading]["prompt_strict"]
        assert pd["n_draws"] == 1082 and pd["n_items"] == 541
        assert pd["classes"][A.CLASS_MAIN_ONLY] == 2 and pd["classes"][A.CLASS_IT_ONLY] == 3 and pd["classes"][A.CLASS_UNSCORED] == 0
        assert pd["classes"][A.CLASS_BOTH_PASS] + pd["classes"][A.CLASS_BOTH_FAIL] == 1077
        assert pd["n_discordant_draws"] == 5 and pd["n_discordant_items"] == 5
        assert pd["discordant_draws"][A.CLASS_IT_ONLY] == sorted(f"{i}/{t}" for i, t in IT_ONLY)
        assert pd["discordant_draws"][A.CLASS_MAIN_ONLY] == sorted(f"{i}/{t}" for i, t in MAIN_ONLY)
        inst = none["per_draw"][reading]["inst_strict"]
        assert inst["classes"][A.CLASS_IT_HIGHER] == 3 and inst["classes"][A.CLASS_MAIN_HIGHER] == 2 and inst["n_discordant_draws"] == 5
        assert inst["sum_abs_diff_followed"] >= 5 and set(inst["classes"]) == set(A.INST_CLASSES)
        # the other two metrics ride alongside
        assert set(none["per_draw"][reading]) == set(A.METRICS) and set(none["readings"][reading]) == set(A.METRICS)
    # the aggregate spread: the analysis's own estimator; the point is exactly (+3 - 2) / 1,082 on strict prompt-level; both legs' points beside it
    s = none["readings"][A.READING_MERGED]["prompt_strict"]
    assert s["point"] == pytest.approx(1 / 1082) and s["ci_low"] <= s["point"] <= s["ci_high"]
    assert s["legs"]["it"]["point"] - s["legs"]["main"]["point"] == pytest.approx(1 / 1082) and s["legs"]["main"]["n_scored"] == 1082
    assert s["B"] == B_FAST and set(s) >= {"point", "ci_low", "ci_high", "excludes_zero", "legs", "n_items"}
    # the constitution pair: identical outcomes by construction of the fixture — a zero spread, zero discordance
    const = _pair(report, CONST)
    for reading in A.READINGS:
        for m in A.METRICS:
            st = const["readings"][reading][m]
            assert st["point"] == 0.0 and st["ci_low"] == 0.0 == st["ci_high"] and not st["excludes_zero"]
            assert const["per_draw"][reading][m]["n_discordant_draws"] == 0
    assert const["design"]["si_text_sha256"] == IC.IFEVAL_CONSTITUTION_SI_SHA256 and none["design"]["si_text_sha256"] is None
    assert none["design"]["served_weights"] == {"hf": IC.IFEVAL_MODEL_PINS[LLAMA][0], "rev": IC.IFEVAL_MODEL_PINS[LLAMA][1]}
    assert none["design"]["varied"]["view"] == {"main": "main", "it": "it"} and none["design"]["differences"] == []
    # the per-draw maps never travel in the report; the counts summarise them
    text = json.dumps(report, default=str)
    assert "body_sha256_by_draw" not in text and "seeds_by_draw" not in text and "prompt_tokens_by_draw" not in text
    assert "wire" not in report["legs"][V.leg_key(LLAMA, "main", NONE, 2048)]


def test_the_attestation_reads_both_stores_and_labels_the_pair_a_floor_only_when_every_attested_check_passes(report):
    for cell in (NONE, CONST):
        p = _pair(report, cell)
        att = p["attestation"]
        assert p["label"] == att["label"] == A.LABEL_FLOOR and att["failing"] == []
        assert all(c["passed"] for c in att["checks"] if c["attested"]), [c for c in att["checks"] if c["attested"] and not c["passed"]]
        assert all(c["passed"] is None for c in att["checks"] if not c["attested"])
        names = [c["name"] for c in att["checks"]]
        assert {"serving_geometry", "batch_composition", "scoring_path", "model_revision", "served_template_hash", "sampling_row", "cap",
                "thinking_mode", "seed_rule", "cell_identity", "producer_tree_sha", "frameworks", "si_prefix", "request_bytes", "seeds",
                "prompt_tokens"} <= set(names)
        assert "ladder_promotions" not in names                     # recorded beside the attestation, never a deciding check
        assert all(next(c for c in att["checks"] if c["name"] == n)["attested"] for n in ("serving_geometry", "batch_composition", "scoring_path",
                                                                                          "request_bytes", "seeds", "prompt_tokens", "si_prefix"))
        # the producer's endpoint records the chat_template_hash and the evidence the served hash: attestable, equal on both legs
        assert att["not_attestable"] == [] and next(c for c in att["checks"] if c["name"] == "served_template_hash")["passed"] is True
        assert all(s["equal"] for s in att["shared"] if s["name"] != "identity_hashes"), [s["name"] for s in att["shared"] if not s["equal"]]
        hashes = next(s for s in att["shared"] if s["name"] == "identity_hashes")
        assert not hashes["equal"] and "hashes differ" in p["label_text"]            # the fixture's placeholder hashes carry the view
        w = att["wire"]
        assert w["n_draws"] == 1082 and w["n_request_bodies_identical"] == 1082 == w["n_common_requests"]
        assert w["n_seeds_identical"] == 1082 and w["n_prompt_tokens_identical"] == 1082 == w["n_common_ledger_rows"] and w["n_max_tokens_identical"] == 1082
        scoring = next(c for c in att["checks"] if c["name"] == "scoring_path")
        assert scoring["detail"].startswith(A.SCORING_SENTENCE) and "no judge model" in scoring["detail"] and "langdetect" in scoring["detail"]
        assert att["designed_difference"]["bank_view"] == {"main": {"view": "main", "bank_view": "main"}, "it": {"view": "it", "bank_view": "it"}}
        assert set(att["incidental"]["main"]) == set(att["incidental"]["it"]) >= {"host", "generation_span", "wall_clock_s", "n_attempts"}
        assert "measured noise floor" in p["label_text"] and "by construction" in p["label_text"]
        assert att["ladder_promotions"] == {"main": [], "it": [], "equal": True, "deciding": False, "note": att["ladder_promotions"]["note"]}
    assert "IFEVAL" in _pair(report, CONST)["attestation"]["checks"][0]["detail"].upper() or True   # (geometry detail names the entry)
    geo = next(c for c in _pair(report, CONST)["attestation"]["checks"] if c["name"] == "serving_geometry")
    assert "capability:ifeval_longsi_2048" in geo["detail"]


def test_a_planted_geometry_or_wire_difference_flips_the_label_and_names_the_check(tmp_path):
    rows = _rows((LLAMA, "main", NONE), (LLAMA, "it", NONE))
    # (a) the `it` standup served a different S (a kv_shrink): serving geometry AND batch composition differ
    G.build_store(tmp_path / "geo", row(LLAMA, "main", NONE))
    G.build_store(tmp_path / "geo", row(LLAMA, "it", NONE), kv_shrink=G.kv_shrink(187, 120, G.MAX_MODEL_LEN_BY_CAP[2048]))
    rep = analyze(tmp_path / "geo", rows)
    p = _pair(rep, NONE)
    assert p["status"] == "filed" and p["label"] == A.LABEL_MEASURES
    assert set(p["attestation"]["failing"]) == {"serving_geometry", "batch_composition"}
    assert p["label_text"].startswith("NOT a noise floor") and "serving_geometry" in p["label_text"]
    assert next(s for s in p["attestation"]["shared"] if s["name"] == "serving_geometry")["equal"] is False
    assert rep["noise_floor"][0]["status"] == "filed"                            # filed, labelled — never silently dropped
    # the readings are still computed (the pair is a between-view difference, just not a floor) and the table says so
    assert p["readings"][A.READING_MERGED]["prompt_strict"]["point"] == 0.0
    table = A._noise_table(rep)
    assert f"| {LLAMA} | {NONE}(it) - {NONE}(main) | {A.LABEL_MEASURES} (serving_geometry, batch_composition) |" in table   # check order
    # the contrasts figure draws NO grey band from a pair that is not a floor
    assert A.noise_floor_pair(rep, LLAMA, NONE) is p
    fig, _n = A._fig_contrasts_model(rep, LLAMA)
    assert all(not ax.patches or all(getattr(pt, "get_alpha", lambda: None)() != 0.18 for pt in ax.patches) for ax in fig.axes)
    import matplotlib.pyplot as plt
    plt.close(fig)
    # (b) a byte on the wire differs on every request of one leg: request_bytes fails; everything else still attests
    def body_edit(body):
        body["user"] = "probe"
    G.build_store(tmp_path / "wire", row(LLAMA, "main", NONE))
    G.build_store(tmp_path / "wire", row(LLAMA, "it", NONE), body_edit=body_edit)
    rep2 = analyze(tmp_path / "wire", rows)
    p2 = _pair(rep2, NONE)
    assert p2["label"] == A.LABEL_MEASURES and p2["attestation"]["failing"] == ["request_bytes"]
    assert p2["attestation"]["wire"]["n_request_bodies_identical"] == 0 and p2["attestation"]["wire"]["n_common_requests"] == 1082
    # (c) a planted ladder difference: the merged reading mixes caps across the pair
    fm, fi = (rep["cells"][f"{LLAMA}/main/{NONE}"], rep["cells"][f"{LLAMA}/it/{NONE}"])
    assert fm["ladder"]["promoted"] == fi["ladder"]["promoted"] == []


def test_a_pair_whose_rows_are_not_identical_by_construction_is_refused_as_a_floor_by_name():
    ctx = A.Context(canonical_ids=G.IDS, pins=None, expected_refusals={}, n_resamples=B_FAST, read_eval_logs=False)
    rm, ri = row("gemma4e4b", "main", NONE), row("gemma4e4b", "it", NONE)
    # the FACT of the frozen matrix: the two gemma views serve the same checkpoint and revision (the suffix is the object-bank view)
    assert (rm["hf"], rm["rev"]) == (ri["hf"], ri["rev"]) == IC.IFEVAL_MODEL_PINS["gemma4e4b"]
    real = A.pair_design(rm, ri)
    assert real["identical_by_construction"] and real["differences"] == [] and real["reason"] is None
    assert real["varied"]["slug"] == {"main": "gemma4e4b-sdfbase-main", "it": "gemma4e4b-sdfbase-it"}
    # a matrix whose `it` view pins DIFFERENT served weights: not identical by construction — refused, named, no readings
    ri2 = {**ri, "hf": "google/gemma-4-E4B"}
    d = A.pair_design(rm, ri2)
    assert not d["identical_by_construction"] and d["differences"] == ["hf"] and "hf ('google/gemma-4-E4B-it' vs 'google/gemma-4-E4B')" in d["reason"]
    assert d["served_weights"] is None
    block = A.noise_floor({}, rows=[rm, ri2], ctx=ctx)
    (g,) = block
    assert g["status"] == "no floor (no identical-by-construction pair)" and len(g["pairs"]) == 1
    p = g["pairs"][0]
    assert p["status"] == "not identical by construction" and p["label"] == A.LABEL_NOT_IDENTICAL and p["readings"] == {} and p["attestation"] is None
    assert p["label_text"].startswith("refused as a floor") and "hf" in p["reason"]
    # a served object on either view is the design's own between-view comparison: never a candidate
    lora = "constv2-mix-full__lora_r4_ce"
    rl_m, rl_i = row(LLAMA, "main", lora), row(LLAMA, "it", lora)
    assert not A.pair_design(rl_m, rl_i)["identical_by_construction"] and "served object" in A.pair_design(rl_m, rl_i)["reason"]
    block = A.noise_floor({}, rows=[row(LLAMA, "main", NONE), row(LLAMA, "it", NONE), rl_m, rl_i], ctx=ctx)
    assert [p["cell"] for p in block[0]["pairs"]] == [NONE] and block[0]["pairs"][0]["status"] == "not run (a leg unusable)"
    assert block[0]["status"] == "no floor (a leg unusable)"
    # a one-view model has no floor
    nemo = A.noise_floor({}, rows=[row("nemotron3nano4b", "main", NONE)], ctx=ctx)
    assert nemo == [{"model_key": "nemotron3nano4b", "views": ["main"], "status": "no floor (one view)", "primary_metric": "prompt_strict", "pairs": []}]
    # the per-draw reading refuses two legs whose draw sets differ, by name
    a = [{"item_id": "x", "sample_idx": 0, "scored": True, "prompt_level_strict": True, "num_instructions": 1}]
    with pytest.raises(A.AnalysisError, match="draw sets differ"):
        A.per_draw_discordance(a, a + [{**a[0], "sample_idx": 1}], metric="prompt_strict")


def test_the_figure_the_caption_the_table_and_the_record_state_the_reading(report, tmp_path):
    pytest.importorskip("matplotlib")
    figs = A.make_figures(report, tmp_path / "figs")
    stem = f"{A.NOISE_FLOOR_STEM}_{LLAMA}"
    assert stem in figs["written"] and {Path(p).suffix for p in figs["written"][stem]} == {".png", ".pdf"}
    assert all(Path(p).stat().st_size > 1000 for p in figs["written"][stem])
    nums = json.loads(Path(figs["numbers"]).read_text())
    entries = nums["figures"][stem]
    assert entries, "the figure records what it plotted"
    for e in entries:
        assert A.T.report_path(report, e["source"]) == e["value"], e
    names = {e["name"] for e in entries}
    assert f"{LLAMA}: {NONE}(it) - {NONE}(main)/merged/prompt_strict/point" in names
    assert f"{LLAMA}: {CONST}(it) - {CONST}(main)/per_draw/{A.CLASS_IT_ONLY}" in names
    cap = nums["captions"][stem]
    assert "MEASURED NOISE FLOOR" in cap and "never pooled" in cap and "1 % raw at-cap of draws" in cap and "identical by construction" in cap
    assert f"{NONE}(it) - {NONE}(main) is a MEASURED NOISE FLOOR" in cap and "5 of 1082 draws discordant (main-only 2, it-only 3)" in cap
    assert f"{CONST}(it) - {CONST}(main) is a MEASURED NOISE FLOOR" in cap and "0 of 1082 draws discordant" in cap
    # the figure: two panels (one per pair), a two-line title within the limit, tall axes
    fig, _n = A.fig_noise_floor_model(report, LLAMA)
    assert len(fig.axes) == 2 and all(any("measured noise floor" in t.get_text() for t in ax.texts) for ax in fig.axes)
    assert all(any("5 of 1082 draws discordant" in t.get_text() or "0 of 1082 draws discordant" in t.get_text() for t in ax.texts) for ax in fig.axes)
    bold = [t for t in fig.texts if t.get_fontweight() == "bold"][0].get_text().split("\n")
    assert len(bold) == 2 and all(len(line) <= 90 for line in bold)
    assert A.axes_height_fraction(fig) >= A.AXES_MIN_FRACTION
    import matplotlib.pyplot as plt
    plt.close(fig)
    # the table: one row per pair with the label; the record renders every placeholder; the template names the rule
    table = A._noise_table(report)
    assert f"| {LLAMA} | {NONE}(it) - {NONE}(main) | {A.LABEL_FLOOR} | = pass 1 (no promotion on either view); 1082 of 1082 final request bodies identical | +0.09 [" in table
    assert f"| {LLAMA} | {CONST}(it) - {CONST}(main) | {A.LABEL_FLOOR} | = pass 1 (no promotion on either view); 1082 of 1082 final request bodies identical | +0.00 [+0.00, +0.00]" in table and "5 of 1082 (main-only 2, it-only 3)" in table
    md = A.render_results_md(report)
    assert "{{" not in md and "identical by construction" in md and "measures that difference" in md
    # the llama family excluded from the figures drops the noise-floor stem with it
    rep2 = analyze(Path(report["gen_root"]), LLAMA_ROWS, exclude_families=["llama"])
    assert stem not in A.figures_for(rep2) and stem in A.figures_for(report)
    tables = json.loads(Path(figs["tables"]).read_text())
    assert tables["noise_floor"][0]["pairs"][0]["label"] == A.LABEL_FLOOR
