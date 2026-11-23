"""appendix_caperrors: the error analysis of the BFCL and BigCodeBench losses (app:caperrors). Fast: the committed sidecar, the committed
snapshot's per-item results for a few runs read by run id, synthetic tampering, and the two composition figures built in memory.

Pinned here:
  * the committed sidecar's contract: its schema, the judge stamp (deterministic, no judge id or model), the provenance block (every source
    file with its sha256, registry head 456957 and its snapshot, the bootstrap, a snapshot check that matched every run), the class order
    the figures stack, and every stored point equal to the one recomputed from the run counts (`caperrors_problems` finds nothing);
  * tampering refuses by name: a class count, the judge stamp, a pool naming an unknown run, a stored share, an interval that no longer
    holds its point, a missing provenance file hash;
  * the binding: `item_check` on the committed snapshot's per-item results (metric acc) for a few runs and their anchors reproduces each
    run's k, n, new failures and gains and its object id (on a frame whose item rows are in its SQLite only, where a render left that
    SQLite under the notebook's cache or under SISTEER_SLOW_TESTS=1); a flipped item, a missing anchor and a changed object id each fail by name and
    `assert_bound` refuses;
  * the numbers the manuscript prints: selected rows of tab:caperrors and tab:caperrors-bcb, character for character; the first rows by the
    notebook's first-row rule, `registry_nb.seed_average` (Nemotron and Llama BigCodeBench 39.6 and 41.5); the pooled points
    written out from the run counts independently of `pool_point`;
  * the figures: one segment per nonzero class, its drawn width the sidecar's share, the segments abutting from 0 to 100, the headline and
    detail text, the 22 Qwen bars, the Llama BFCL panel's note in place of bars, and the three saved formats;
  * the builder's restated rules on synthetic input: the declination keyword rules, the model-balanced weights, the scope's refusals.
"""
import copy
import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
for p in (NOTEBOOKS, NOTEBOOKS / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import registry_nb as R  # noqa: E402
import appendix_caperrors as CE  # noqa: E402
import build_appendix_caperrors as B  # noqa: E402

Q = "qwen35_4b_it"
AC_MIRROR = "APPENDIX_FIGS_MIRROR"


@pytest.fixture(scope="module")
def side():
    return CE.load_caperrors()


def _pool(s, ev, m, name, w="runs"):
    q = CE.pool(s, ev, m, name, w)
    assert q is not None, (ev, m, name, w)
    return q


# ---- the committed sidecar ----------------------------------------------------------------------------------------------------------
def test_committed_sidecar_holds_its_contract(side):
    assert CE.caperrors_problems(side) == []
    assert side["schema"] == CE.SCHEMA
    assert side["judge"]["kind"] == "deterministic" and side["judge"]["judge_id"] is None and side["judge"]["judge_model"] is None
    prov = side["provenance"]
    # the analysis's frame, cut at its registry head; the frame CURRENT names may be a later one, and the sidecar's binding to it is
    # checked by `item_check` below and by the notebook cell, whichever frame that is
    assert prov["registry_head"] == 456957 and prov["snapshot"].startswith(f"{prov['registry_head']:012d}-")
    assert re.fullmatch(r"\d{12}-\d{8}T\d{6}-[0-9a-f]{8}", prov["snapshot_check"]["snapshot"])
    assert set(prov["files"]) == set(B.FILES) and all(len(f["sha256"]) == 64 for f in prov["files"].values())
    assert prov["bootstrap"]["B"] == 4000 and prov["bootstrap"]["seed"] == 20261111
    assert prov["snapshot_check"]["runs_checked"] == prov["snapshot_check"]["runs_matching"] == len(side["runs"]) == 82
    assert sum(r["eval_id"] == "bfcl" for r in side["runs"]) == 37 and len(side["anchors"]) == 22
    assert side["bfcl_item_kinds"]["n_relevant"] == 2857 and side["bfcl_item_kinds"]["n_irrelevance"] == 1124
    # no path, store or session field rides along in a run record
    assert not any(k in r for r in side["runs"] for k in ("store_uri", "rows_path", "cell_dir", "store_dir", "eval_log_path"))
    # pinned by registry ids: every run id has the registry's shape; a fitted object's id is its sha256, a prompted system instruction's
    # the registry's `si:<sha256>`
    assert all(r["run_id"].startswith("sdfbase_full22:") for r in side["runs"])
    for r in side["runs"]:
        prefix = "si:" if r["object_label"] in R.PROMPTED else ""
        assert re.fullmatch(prefix + "[0-9a-f]{64}", r["object_id"]), (r["object_label"], r["object_id"])


def test_pooled_points_follow_the_counts_written_out_here(side):
    """The pooled shares and delta from the run counts with plain sums (no `pool_point`), on the rows the manuscript quotes."""
    q = _pool(side, "bfcl", Q, "control_relu")
    rs = [r for r in side["runs"] if r["run_id"] in q["run_ids"]]
    assert [r["seed"] for r in sorted(rs, key=lambda r: r["seed"])] == ["s1", "s2", "s3"]
    lost = sum(r["lost"] for r in rs)
    assert lost == 4501 and sum(r["gained"] for r in rs) == 512
    assert 100 * (512 - 4501) / (3 * 3981) == pytest.approx(q["delta_pp"][0], abs=1e-9)
    nc = sum(r["lost_extra"]["no_call"] for r in rs)
    assert 100 * nc / lost == pytest.approx(q["shares"]["no_call"][0], abs=1e-9) and round(100 * nc / lost, 1) == 79.2
    assert sorted(r["k"] for r in rs) == [1786, 1786, 1787] and sorted(r["lost"] for r in rs) == [1494, 1503, 1504]
    # the model-balanced all-model pool weights each model equally
    mb = _pool(side, "bigcodebench", "all", "all_fitted", "model_balanced")
    by_id = {r["run_id"]: r for r in side["runs"]}
    per = {}
    for rid in mb["run_ids"]:
        per.setdefault(by_id[rid]["model_id"], []).append(by_id[rid])
    wl = {m: 1.0 / len(v) for m, v in per.items()}
    num = sum(wl[m] * r["lost_by_class"]["wrong_code"] for m, v in per.items() for r in v)
    den = sum(wl[m] * r["lost"] for m, v in per.items() for r in v)
    assert 100 * num / den == pytest.approx(mb["shares"]["wrong_code"][0], abs=1e-9) and round(100 * num / den, 1) == 75.0


@pytest.mark.parametrize("mutate, message", [
    (lambda s: s["runs"][0]["lost_by_class"].__setitem__("wrong_call", s["runs"][0]["lost_by_class"]["wrong_call"] + 1), "classes sum to"),
    (lambda s: s["judge"].__setitem__("judge_model", "claude-haiku-4-5"), "judge stamp"),
    (lambda s: s["pools"][0]["run_ids"].append("sdfbase_full22:bfcl:x:unknown"), "run ids missing, repeated or unknown"),
    (lambda s: s["pools"][0]["shares"]["wrong_call"].__setitem__(0, 50.0), "share wrong_call"),
    (lambda s: s["pools"][0]["delta_pp"].__setitem__(1, 0.0), "does not hold"),
    (lambda s: s["provenance"]["files"]["manifest"].pop("sha256"), "lacks a path and a sha256"),
    (lambda s: s.__setitem__("schema", "appendix-caperrors-v0"), "schema"),
])
def test_a_broken_sidecar_is_refused_by_name(side, tmp_path, mutate, message):
    bad = copy.deepcopy(side)
    mutate(bad)
    assert any(message in p for p in CE.caperrors_problems(bad)), CE.caperrors_problems(bad)[:3]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="nothing drawn"):
        CE.load_caperrors(str(path))


# ---- the binding to the registry's per-item results ---------------------------------------------------------------------------------
def _snapshot_items(side, run_ids):
    """The committed frame's per-item results (metric acc) and run table for `run_ids`: from the item parquet where the frame ships one,
    else (a frame whose item tables are SQLite-only, MANIFEST `sqlite_only_tables`) from its SQLite under the notebook's cache, read with
    the notebook cell's own SQL. Reassembling that SQLite (about 4 GB, once per frame) is the slow tier's cost, as in the other modules
    that read the committed frame's item rows: this fast module reads it only where a render (or an earlier run) already left it under
    `_cache/snapshot/<frame>/`, or under SISTEER_SLOW_TESTS=1, and skips by name otherwise."""
    root = REPO / "paper" / "snapshot"
    sid = (root / "CURRENT").read_text().strip()
    snap = root / "snapshots" / sid
    if (snap / "item_result.parquet").exists():
        ir = pq.read_table(snap / "item_result.parquet", columns=["run_id", "item_id", "metric_id", "trial_index", "value", "withdrawn_by_revision_id"],
                           filters=[("run_id", "in", run_ids)]).to_pandas()
        ir = ir[(ir.metric_id == "acc") & ir.withdrawn_by_revision_id.isna()][["run_id", "item_id", "trial_index", "value"]]
        runs = pq.read_table(snap / "run.parquet", columns=["run_id", "object_id"], filters=[("run_id", "in", run_ids)]).to_pandas()
        return ir, runs
    from sisteer.registry import snapshot as SN
    cache = pathlib.Path(R.CACHE) / "snapshot"
    if not (cache / sid / SN.PAPER_FRAME_SQLITE).exists() and os.environ.get("SISTEER_SLOW_TESTS") != "1":
        pytest.skip(f"frame {sid} ships its item rows in the SQLite only, and none is reassembled under {cache} (about 4 GB): a render "
                    "or SISTEER_SLOW_TESTS=1 builds it")
    ids = set(run_ids)
    sub = dict(side, runs=[r for r in side["runs"] if r["run_id"] in ids], anchors=[a for a in side["anchors"] if a["run_id"] in ids])
    con = SN.open_paper_frame(str(root), cache_dir=str(cache))
    try:
        ir = pd.read_sql_query(CE.item_sql(sub), con)[["run_id", "item_id", "trial_index", "value"]]
        runs = pd.read_sql_query(CE.object_sql(sub), con)[["run_id", "object_id"]]
    finally:
        con.close()
    return ir, runs


@pytest.fixture(scope="module")
def few(side):
    """Three runs (a Qwen BFCL seed, a Nemotron BigCodeBench run, the Llama difference ReLU steer on its paired view) and their anchors."""
    pick = [r for r in side["runs"] if (r["model_id"], r["eval_id"], r["object_label"], r["seed"]) == (Q, "bfcl", "constitution_relu", "s2")]
    pick += [r for r in side["runs"] if (r["model_id"], r["eval_id"], r["object_label"]) in
             (("nemotron3nano4b", "bigcodebench", "constitution_lora"), ("llama8b_it", "bigcodebench", "difference_relu"))]
    assert len(pick) == 3
    sub = dict(side, runs=pick, anchors=[a for a in side["anchors"] if a["run_id"] in {r["anchor_run_id"] for r in pick}])
    ids = sorted({r["run_id"] for r in pick} | {r["anchor_run_id"] for r in pick})
    items, runs = _snapshot_items(side, ids)
    return sub, items, runs


def test_item_check_reproduces_the_counts_from_the_snapshot(few):
    sub, items, runs = few
    chk = CE.item_check(sub, items, runs)
    assert chk.ok.all(), chk.why.tolist()
    CE.assert_bound(chk)
    r = sub["runs"][0]
    row = chk.set_index("run_id").loc[r["run_id"]]
    assert (row.k, row.n, row.lost, row.gained) == (r["k"], r["n"], r["lost"], r["gained"]) == (1940, 3981, 1380, 205)
    assert "item_result" in CE.item_sql(sub) and "metric_id = 'acc'" in CE.item_sql(sub) and r["anchor_run_id"] in CE.item_sql(sub)


def test_item_check_names_a_flipped_item_a_missing_anchor_and_a_changed_object(few):
    sub, items, runs = few
    r0 = sub["runs"][0]
    flipped = items.copy()
    i = flipped[(flipped.run_id == r0["run_id"]) & (flipped.value == 1.0)].index[0]
    flipped.loc[i, "value"] = 0.0
    chk = CE.item_check(sub, flipped, runs).set_index("run_id")
    assert not chk.loc[r0["run_id"], "ok"] and "k 1939" in chk.loc[r0["run_id"], "why"]
    with pytest.raises(ValueError, match="rebuild the sidecar"):
        CE.assert_bound(chk.reset_index())
    r1 = sub["runs"][1]
    chk = CE.item_check(sub, items[items.run_id != r1["anchor_run_id"]], runs).set_index("run_id")
    assert chk.loc[r1["run_id"], "why"] == "no item results for its anchor"
    moved = runs.copy()
    moved.loc[moved.run_id == r0["run_id"], "object_id"] = "0" * 64
    chk = CE.item_check(sub, items, moved).set_index("run_id")
    assert not chk.loc[r0["run_id"], "ok"] and "object 000000000000" in chk.loc[r0["run_id"], "why"]
    with pytest.raises(ValueError, match="two rows for one"):
        CE.item_check(sub, pd.concat([items, items.head(1)]), runs)


# ---- the tables -------------------------------------------------------------------------------------------------------------------
def test_tables_print_the_manuscripts_rows(side):
    bf, bc = CE.table_caperrors(side), CE.table_caperrors_bcb(side)
    assert "\\label{tab:caperrors}" in bf and "\\label{tab:caperrors-bcb}" in bc
    assert len(bf.splitlines()) == 42 and len(bc.splitlines()) == 51
    for row in (r"\quad Control ReLU & $-33.4$ {\scriptsize $[-35.0, -31.8]$} & 4,501 & 79.2 (1.5) & 0.1 & 0.2 & 18.2 & 2.3 \\",
                r"\quad Constitution ReLU & $-28.6$ {\scriptsize $[-30.2, -27.1]$} & 4,023 & 85.3 (35.8) & 0.3 & 1.9 & 10.2 & 2.3 \\",
                r"\quad Difference LoRA & $-14.3$ {\scriptsize $[-15.4, -13.2]$} & 2,353 & 13.7 (0.2) & 14.0 & 3.0 & 43.9 & 25.3 \\",
                r"\quad Constitution LoRA & $+0.1$ {\scriptsize $[-0.3, 0.4]$} & 20 & 40.0 (0.0) & 0.0 & 0.0 & 50.0 & 10.0 \\",
                r"\quad Difference ReLU & $-18.9$ {\scriptsize $[-20.3, -17.5]$} & 843 & 0.1 (0.0) & 0.1 & 0.0 & 40.0 & 59.8 \\"):
        assert row in bf.splitlines(), row
    for row in (r"\quad Constitution LoRA & $-24.4$ {\scriptsize $[-27.0, -21.8]$} & 969 & 2.9 & 43.8 & 53.4 & 0.0 \\",
                r"\quad Constitution LoRA & $+0.0$ {\scriptsize $[-1.7, 1.8]$} & 49 & 10.2 & 61.2 & 0.0 & 28.6 \\",
                r"\quad Difference ReLU & $-33.7$ {\scriptsize $[-36.8, -30.6]$} & 416 & 75.0 & 25.0 & 0.0 & 0.0 \\"):
        assert row in bc.splitlines(), row
    firsts = lambda t: [ln for ln in t.splitlines() if ln.startswith("\\quad Unsteered")]  # noqa: E731
    assert firsts(bf) == [f"\\quad Unsteered & ${v}$ & & & & & & \\\\" for v in ("78.3", "78.0", "75.8")]
    assert firsts(bc) == [f"\\quad Unsteered & ${v}$ & & & & & \\\\" for v in ("45.8", "49.6", "39.6", "41.5")]
    assert "Llama-3.1-8B-Instruct (one fit" not in bf and "Llama-3.1-8B-Instruct (one fit" in bc


# Table 6's caption as the paper sets it (sections/app_capability_errors.tex, its revision markup dropped)
MANUSCRIPT_CAPTION = ("BFCL pass rate and new failures by class. First row of each block: unsteered pass rate (\\%). Other rows: change "
                      "against the seed-matched unsteered run (pp, 95\\% item-bootstrap interval), the number of new failures, and their "
                      "classes as a percentage, with explicit declinations in parentheses. An irrelevance call is a call on an item that "
                      "needs none.")


def test_table_6_prints_the_papers_caption_and_the_notebook_prints_the_fuller_reading_under_it(side):
    """The printed Table 6 carries the paper's caption, so it can be pasted; the notebook prints NOTE_BFCL (the first row's seed-average
    rule with its pointer to the tie rule, the class definitions, the imprecise shares) as plain text under it."""
    import registry_nb as R
    assert CE.CAPTION_BFCL == MANUSCRIPT_CAPTION
    assert "\\caption{" + MANUSCRIPT_CAPTION + "}" in CE.table_caperrors(side)
    assert CE.NOTE_BFCL.count(R.SEED_AVERAGE_TIES) == 1 and "$" not in CE.NOTE_BFCL and "\\" not in CE.NOTE_BFCL
    src = (pathlib.Path(CE.__file__).resolve().parent / "make_registry_notebook.py").read_text(encoding="utf-8")
    assert "print(CE.table_caperrors(_ce))\nprint(CE.NOTE_BFCL)\nprint(CE.table_caperrors_bcb(_ce))" in src


def test_first_rows_follow_the_notebooks_seed_average_rule(side):
    """registry_nb.seed_average, the rule of every first row: per seed the unsteered run the most of its arms were differenced against
    (the paired difference ReLU's own run does not move it), averaged over Qwen's three seeds; no cell is flagged for its own run."""
    import registry_nb as R
    lv = CE.own_levels(side, "bigcodebench", "nemotron3nano4b")
    assert sorted(round(v, 3) for v in lv.values()) == [39.035] + [39.561] * 5
    assert CE.unsteered_level(side, "bigcodebench", "nemotron3nano4b") == pytest.approx(39.561, abs=1e-3)
    assert CE.unsteered_level(side, "bfcl", Q) == pytest.approx(np.mean([78.272, 78.247, 78.297]), abs=1e-3)   # the seed average
    rd = CE.anchor_readings(side, "bfcl", Q)
    assert len(rd) == 18 and sorted({s for s, *_ in rd}) == ["s1", "s2", "s3"] and all(f for *_, f in rd)
    assert R.seed_replicated(rd) and CE.unsteered_level(side, "bfcl", Q) == R.seed_average(rd)
    assert not hasattr(CE, "own_anchor_notes")


# ---- the figures ------------------------------------------------------------------------------------------------------------------
def test_qwen_figure_draws_the_sidecars_shares(side, tmp_path, monkeypatch):
    monkeypatch.delenv(AC_MIRROR, raising=False)
    fig = CE.qwen_figure(side, "figA_test_caperrors", out_dirs=[str(tmp_path)])
    seg = CE.drawn_segments(fig)
    assert set(CE.panel_axes(fig)) == {(Q, "bfcl"), (Q, "bigcodebench")}
    assert seg.groupby(["eval_id"]).pool.nunique().to_dict() == {"bfcl": 11, "bigcodebench": 11}
    for (ev, p), g in seg.groupby(["eval_id", "pool"]):
        q = _pool(side, ev, Q, p)
        order = [c for c, _col, _lab in CE.CLASS_STYLE[ev] if q["shares"][c][0] > 0]
        assert g.cls.tolist() == order
        assert np.allclose(g.width, [q["shares"][c][0] for c in order], atol=1e-9)
        assert np.allclose(g.left, np.concatenate([[0], np.cumsum(g.width)[:-1]]), atol=1e-9)
        assert g.width.sum() == pytest.approx(100.0, abs=1e-6)
    texts = {t.get_gid(): t.get_text() for ax in fig.axes for t in ax.texts if t.get_gid()}
    assert texts["caperrors-headline|bfcl|control_relu"] == "No call 79% · format 0.2%"
    assert texts["caperrors-detail|bfcl|constitution_relu"] == "Δ −28.6 pp · 4,023 new failures"
    assert texts["caperrors-headline|bigcodebench|constitution_lora"] == "Format 3% · refusal 53%"
    labels = [t.get_text() for t in CE.panel_axes(fig)[(Q, "bfcl")].get_yticklabels()]
    assert labels[:4] == [R.LABEL["control_relu"], R.LABEL["constitution_relu"], R.LABEL["difference_relu"], "All ReLU steers"]
    assert labels[-1] == R.LABEL["arditi_allpos"]
    plt.close(fig)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"figA_test_caperrors.{e}" for e in ("pdf", "png", "svg")]



def test_models_figure_has_every_model_and_says_llama_has_no_bfcl_run(side):
    fig = CE.models_figure(side, None)
    axes = CE.panel_axes(fig)
    assert len(axes) == 8
    seg = CE.drawn_segments(fig)
    assert seg[seg.model_id == "llama8b_it"].eval_id.unique().tolist() == ["bigcodebench"]
    notes = [t.get_text() for t in axes[("llama8b_it", "bfcl")].texts if (t.get_gid() or "").startswith("caperrors-none")]
    assert notes == ["No BFCL run for Llama-3.1-8B"]
    assert [t.get_text() for t in axes[("llama8b_it", "bfcl")].get_yticklabels()][0] == R.LABEL["control_relu"]
    assert "arditi_allpos" not in set(seg[seg.model_id != Q].pool)        # the Refusal steer has capability runs on Qwen only
    bars = seg.groupby(["model_id", "eval_id"]).pool.nunique().to_dict()
    assert bars == {(Q, "bfcl"): 11, (Q, "bigcodebench"): 11, ("gemma4e4b_it", "bfcl"): 10, ("gemma4e4b_it", "bigcodebench"): 10,
                    ("nemotron3nano4b", "bfcl"): 10, ("nemotron3nano4b", "bigcodebench"): 10, ("llama8b_it", "bigcodebench"): 10}
    q = _pool(side, "bigcodebench", "nemotron3nano4b", "difference_relu")
    g = seg[(seg.model_id == "nemotron3nano4b") & (seg.eval_id == "bigcodebench") & (seg.pool == "difference_relu")]
    assert g.set_index("cls").width["cap_no_program"] == pytest.approx(q["shares"]["cap_no_program"][0]) and round(q["shares"]["cap_no_program"][0], 1) == 53.3
    plt.close(fig)


# ---- the builder's restated rules ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, broad, narrow", [
    ("I can't do that. Booking a flight requires...", True, True),
    ("I’m not going to call the weather tool for you.", True, True),          # curly quote: normalised for the broad rule
    ("Sure, I'm not going to guess the location.", True, False),             # a verb only the broad rule lists
    ("Here is the answer in prose: 42.", False, False),
    ("I can't access real-time data.", False, False),                        # incapacity, not declination wording
])
def test_declination_rules(text, broad, narrow):
    assert B.declination(text) is broad and B.declination_narrow(text) is narrow


def test_model_balanced_weights_and_counts():
    w = B.pool_weights(["a1", "a2", "a3", "b1"], {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}, "model_balanced")
    assert w["b1"] == pytest.approx(3 * w["a1"]) and sum(w.values()) == pytest.approx(4.0)
    assert CE.pool_weights(["a1", "a2", "a3", "b1"], {"a1": "A", "a2": "A", "a3": "A", "b1": "B"}, "model_balanced") == pytest.approx(w)
    with pytest.raises(B.BuildError, match="not a whole number"):
        B._count(2.5)


def test_scope_refuses_a_run_without_one_unsteered_anchor():
    run = dict(run_id="r1", eval_id="bigcodebench", roles=["steered"], view="qwen35_4b_s1", store_view="x-f22s1", arm="control_lora",
               model_id=Q, seed="s1", anchor_run_id="a1")
    anchor = dict(run_id="a1", eval_id="bigcodebench", roles=["anchor"], view="qwen35_4b_s1", arm="unsteered", model_id=Q)
    drawn, anchors = B.scope({"runs": [run, anchor]})
    assert [r["run_id"] for r in drawn] == ["r1"] and [a["run_id"] for a in anchors] == ["a1"]
    with pytest.raises(B.BuildError, match="not an unsteered run"):
        B.scope({"runs": [dict(run, anchor_run_id="a2"), anchor]})
    with pytest.raises(B.BuildError, match="2 anchors"):
        B.scope({"runs": [dict(run, anchor_run_id=["a1", "a1b"]), anchor]})
    # an instruct-direct LoRA (the transfer figure's) and a set-aside duplicate are not drawn here
    assert B.scope({"runs": [dict(run, store_view="x-f22it"), anchor]})[0] == []
    assert B.scope({"runs": [dict(run, roles=["steered", "set_aside_duplicate"]), anchor]})[0] == []
