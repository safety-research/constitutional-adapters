"""appendix_geometry_evenodd: the sign decomposition of the control objects (figA_control_evenodd) and its committed sidecar.

Reads the committed `paper/notebooks/data/appendix_geometry_evenodd.json`; no live registry, no lane files (the builder checked the
sidecar against both when it wrote it: the lane's per-seed tables to 1e-9 and every run's MiniMax-M3 recount exactly); the committed
frame's item rows only for the payload binding below, where its SQLite is already reassembled or under SISTEER_SLOW_TESTS=1. The intervals are
registry_nb's, built from the file's per-unit payloads through the unit source the module registers (this module installs the default
unit provider, which dispatches to it, and no row here reads a registry's items). Pinned here:
  * the sidecar's contract: it loads; the MiniMax-M3 stamp on the judge and a pinned judgment on every reading and anchor run; the
    provenance names every source file with its sha256; 108 readings (control ReLU 44 with the two unpaired seed-1 benign cells named,
    control LoRA 48, Refusal steer 16); `sidecar_problems` refuses a moved delta, another judge, a lone sign, a repeated mark, two anchors
    for one seed's signs, a -1 object that is not the +1 object's negation, and a payload that is missing, out of range or does not give
    its run's reading or its anchor's value (an item set is named by its size and the sha256 of its ids, never spelled);
  * the payloads as a unit source: each run's units give its reading (rate and n, all replies and coherent), a drifted payload stops the
    pool (UnitsMismatch), and every seed pool is item-clustered (no bound, no opaque pool);
  * the arithmetic: the shared anchor cancels from the sign-dependent part and counts once in the sign-independent part;
  * the printed numbers: the sign rows of the appendix's Table (tab:app_evenodd), all replies and coherent, and the per-evaluation parts
    the appendix text quotes, at the precision they are printed;
  * the figure: one mark per reading at its delta, circle for the + sign and diamond for the - sign in each object's paper colour, the
    Refusal steer one unseeded pair per evaluation, no seed-1 ReLU mark on the benign evaluations, 18 mean marks at the notebook's
    category means, the panel titles, and the three formats saved; the line printed under it names the two unpaired cells only;
  * the registry binding on synthetic object, run and judgment tables built from the sidecar: clean, then each corruption named;
  * the payloads against the committed frame: every used run's payload is its pinned MiniMax-M3 judgment's units by the builder's rules,
    at their positions in the evaluation's item set, and each item set's size and sha256 are the frame's; a reordered payload, which
    still gives its run's rate and n, and a tampered item-set digest are named (on a frame whose item rows are in its SQLite only, where
    a render left that SQLite under the notebook's cache or under SISTEER_SLOW_TESTS=1)."""
import copy
import json
import math
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
for _p in (NOTEBOOKS, NOTEBOOKS / "data"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402

import registry_nb as R  # noqa: E402
import appendix_common as AC  # noqa: E402
import appendix_geometry_evenodd as GE  # noqa: E402
import pooled_ci as P  # noqa: E402
import build_appendix_geometry_evenodd as B  # noqa: E402

#: the conftest's synthetic-frame provider stands down: the intervals here read the file's own payloads (`_payload_units`)
USES_ITEM_ROWS = True


@pytest.fixture(scope="module", autouse=True)
def _payload_units():
    """The default unit provider (`registry_nb.item_units`, which reads a row whose `judge_source` is a registered unit source from that
    source: this file's payloads) for this module, set before its module fixtures draw; the synthetic-frame stub put back after it."""
    R.reset_pooled_ci(None)
    yield
    R.reset_pooled_ci(R.no_item_rows)

# tab:app_evenodd's sign rows (MiniMax-M3), as printed: {(object, category): {(part, mode): (value, half-width)}}
TABLE = {
    ("control_relu", "static"): {("even", "all"): (-18.6, 3.6), ("odd", "all"): (-5.2, 6.5), ("even", "coherent"): (-18.8, 3.5), ("odd", "coherent"): (-4.9, 6.9)},
    ("control_relu", "multi-turn"): {("even", "all"): (12.7, 4.8), ("odd", "all"): (-11.0, 4.2), ("even", "coherent"): (10.6, 3.5), ("odd", "coherent"): (-9.2, 2.7)},
    ("control_relu", "over-refusal"): {("even", "all"): (19.8, 1.8), ("odd", "all"): (4.0, 1.1), ("even", "coherent"): (20.6, 2.5), ("odd", "coherent"): (3.2, 3.0)},
    ("control_lora", "static"): {("even", "all"): (-20.6, 5.7), ("odd", "all"): (-0.5, 2.1), ("even", "coherent"): (-20.6, 5.5), ("odd", "coherent"): (-0.6, 2.0)},
    ("control_lora", "multi-turn"): {("even", "all"): (6.5, 4.3), ("odd", "all"): (-1.9, 4.4), ("even", "coherent"): (6.4, 3.9), ("odd", "coherent"): (-1.1, 3.7)},
    ("control_lora", "over-refusal"): {("even", "all"): (12.8, 7.6), ("odd", "all"): (-5.2, 4.8), ("even", "coherent"): (12.7, 7.8), ("odd", "coherent"): (-5.1, 5.0)},
    ("arditi_allpos", "static"): {("even", "all"): (-7.5, 2.6), ("odd", "all"): (23.5, 1.5), ("even", "coherent"): (-7.5, 2.7), ("odd", "coherent"): (23.6, 1.6)},
    ("arditi_allpos", "multi-turn"): {("even", "all"): (23.3, 3.6), ("odd", "all"): (30.6, 2.3), ("even", "coherent"): (23.3, 3.6), ("odd", "coherent"): (30.6, 2.3)},
    ("arditi_allpos", "over-refusal"): {("even", "all"): (3.3, 2.1), ("odd", "all"): (-24.4, 1.4), ("even", "coherent"): (3.3, 2.1), ("odd", "coherent"): (-24.4, 1.4)},
}
# D(+theta) and D(-theta) the appendix text quotes for the control ReLU steer on multi-turn attacks
RELU_MT = {"Dp": (1.7, 3.4), "Dn": (23.7, 8.2)}


@pytest.fixture(scope="module")
def ge():
    return GE.load_geometry_evenodd()


@pytest.fixture(scope="module")
def fig_and_dir(ge, tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    fig = GE.figure_geometry_evenodd(ge, GE.FIGURE, out_dirs=[str(out)])
    yield fig, out
    plt.close(fig)


def _r1(x) -> float:
    return float(f"{x:.1f}")


# ---- the sidecar ---------------------------------------------------------------------------------------------------------------
def test_sidecar_loads_with_its_judge_and_provenance(ge):
    assert GE.sidecar_problems(ge) == []
    assert ge["judge"]["model"] == "minimax-m3"
    assert ge["judge"]["strongreject"] == {**ge["judge"]["strongreject"], "judge_model": "minimax-m3", "threshold": 0.5}
    assert ge["judge"]["over_refusal"]["judge_model"] == "minimax-m3"
    names = {f["name"] for f in ge["provenance"]["source_files"]}
    assert {"tables/A1_seed_relu_items_minimax-m3.csv", "tables/A3_eval_borrowed_items_minimax-m3.csv", "work/run_recount_live_minimax-m3.parquet",
            "frames_work/runs_scope.parquet"} <= names
    assert all(len(f["sha256"]) == 64 for f in ge["provenance"]["source_files"])
    assert ge["provenance"]["run_selection"]["registry_head"] == 456957
    # the run selection's run table, held to that frame's MANIFEST sha256 by the builder (`--selection-snapshot`)
    assert ge["provenance"]["run_selection"]["run_table_sha256"] == "52da9713fdf4ca8d8da4febdcb69bea576b82e5139eed4f910ddca5342577064"
    c = ge["provenance"]["checks"]
    assert c["max_abs_diff_vs_lane_seed_tables_items"] < 1e-9 and c["max_abs_diff_vs_lane_seed_tables_coherent"] < 1e-9
    assert c["max_abs_diff_registry_recount_vs_lane"] < 1e-9
    assert all(r["judgment_id"] for r in ge["rows"]) and all(all(a["judgment_ids"]) for a in ge["anchors"].values())


def test_sidecar_holds_the_readings_and_names_the_unpaired_cells(ge):
    df = pd.DataFrame(ge["rows"])
    assert df.groupby("object").size().to_dict() == {"control_relu": 44, "control_lora": 48, "arditi_allpos": 16}
    assert set(df[df.object == "arditi_allpos"].seed) == {""} and set(df[df.object != "arditi_allpos"].seed) == {"s1", "s2", "s3"}
    assert sorted(df.loc[(df.object == "control_relu") & (df.category == "over-refusal"), "seed"].unique()) == ["s2", "s3"]
    un = {(u["object"], u["seed"], u["eval"], u["missing_sign"]) for u in ge["unpaired"]}
    assert un == {("control_relu", "s1", "xstest", -1), ("control_relu", "s1", "or_bench_hard", -1)}
    assert all(u["registry_runs_of_missing_sign"] == [] and u["why"] for u in ge["unpaired"])
    objs = {(o["object"], o["seed"], o["sign"]): o for o in ge["objects"]}
    assert len(objs) == 14
    for (o, s, sg), x in objs.items():
        if sg == -1:
            assert x["parent_object_id"] == objs[(o, s, 1)]["object_id"] and x["base_scale"] == -1.0


def _flip(p):
    k = next(i for i, x in enumerate(p["v"]) if x is not None)
    p["v"][k] = 1.0 - p["v"][k]


def _drop_degenerate(g):
    for r in g["rows"]:
        if r["rate_coherent"] is not None and g["payloads"][r["run_id"]]["degenerate"]:
            g["payloads"][r["run_id"]]["degenerate"].pop()
            return
    raise AssertionError("no reading with a degenerate unit")


def _mutated(ge, fn):
    g = copy.deepcopy(ge)
    fn(g)
    return GE.sidecar_problems(g)


@pytest.mark.parametrize("what, fn, needle", [
    ("a moved delta", lambda g: g["rows"][0].update(delta=g["rows"][0]["delta"] + 1.0), "does not follow"),
    ("another judge", lambda g: g["judge"].update(model="claude-haiku-4-5"), "judge stamp"),
    ("a lone sign", lambda g: g["rows"].pop(1), "one sign only"),
    ("a repeated mark", lambda g: g["rows"].append(copy.deepcopy(g["rows"][0])), "repeats a mark"),
    ("two anchors for one seed's signs", lambda g: (g["anchors"].__setitem__("pasttense|99", copy.deepcopy(g["anchors"]["pasttense|01"])),
                                                    g["rows"][1].update(anchor="pasttense|99")), "different anchors"),
    ("a -1 object that is not the negation", lambda g: next(o for o in g["objects"] if o["sign"] == -1).update(base_scale=1.0), "negated"),
    ("a reading without its judgment", lambda g: g["rows"][2].update(judgment_id=""), "judgment"),
    ("an unpaired cell drawn", lambda g: g["unpaired"].append(dict(object="control_lora", seed="s1", eval="xstest", why="x")), "unpaired"),
    # the binding waiver covers only a pinned run filed between the run selection's snapshot and the build, listed once
    ("a run read from the file that the sidecar does not pin", lambda g: g["provenance"]["run_selection"]["runs_not_in_snapshot"].append(
        dict(run_id="s9:unpinned:000000000000:0000000000000000", ingest_seq=500000)), "runs_not_in_snapshot"),
    ("a run read from the file filed before the snapshot", lambda g: g["provenance"]["run_selection"]["runs_not_in_snapshot"][0].update(
        ingest_seq=456957), "runs_not_in_snapshot"),
    ("a run read from the file filed after the build", lambda g: g["provenance"]["run_selection"]["runs_not_in_snapshot"][0].update(
        ingest_seq=g["provenance"]["verdicts"]["registry_head_at_build"] + 1), "runs_not_in_snapshot"),
    ("a run read from the file listed twice", lambda g: g["provenance"]["run_selection"]["runs_not_in_snapshot"].append(
        copy.deepcopy(g["provenance"]["run_selection"]["runs_not_in_snapshot"][0])), "runs_not_in_snapshot"),
    # the per-unit payloads the intervals read
    ("no payloads", lambda g: g.pop("payloads"), "no per-unit payloads"),
    ("a reading's run without its payload", lambda g: g["payloads"].pop(g["rows"][0]["run_id"]), "without one"),
    ("a payload for a run nothing uses", lambda g: g["payloads"].__setitem__("x:pasttense:0:0", copy.deepcopy(g["payloads"][g["rows"][0]["run_id"]])),
     "no reading or anchor uses"),
    ("a flipped verdict", lambda g: _flip(g["payloads"][g["rows"][0]["run_id"]]), "its payload reads"),
    ("a verdict out of range", lambda g: g["payloads"][g["rows"][0]["run_id"]]["v"].__setitem__(0, 0.5), "verdicts in"),
    ("a degenerate unit dropped", lambda g: _drop_degenerate(g), "coherent reading"),
    ("an anchor run's verdict flipped", lambda g: _flip(g["payloads"][g["anchors"]["pasttense|01"]["runs_used"][0]]), "anchor pasttense|01"),
    ("a short payload", lambda g: g["payloads"][g["rows"][0]["run_id"]]["v"].pop(), "one verdict per item"),
    ("an item set without its digest", lambda g: g["item_sets"]["pasttense"].pop("ids_sha256"), "item_sets['pasttense']"),
    ("an item set of another size", lambda g: g["item_sets"]["pasttense"].update(n_items=g["item_sets"]["pasttense"]["n_items"] + 1),
     "one verdict per item"),
])
def test_sidecar_problems_refuse_by_name(ge, what, fn, needle):
    bad = _mutated(ge, fn)
    assert bad and any(needle in b for b in bad), (what, bad)


# ---- the arithmetic and the printed numbers ----------------------------------------------------------------------------------
def test_the_shared_anchor_cancels_from_the_sign_dependent_part(ge):
    """Per seed and evaluation (one reading each sign): the sign-dependent part's se is the two runs' own se alone, sqrt(se_p^2 / 4 +
    se_n^2 / 4), whatever the anchor; the sign-independent part carries the anchor's item-clustered se once; a sign's own se is the run's
    Wilson interval on its goals (the lane's `se_steered`) for an attack success rate."""
    sp = GE.seed_parts(ge).set_index(["object", "seed", "eval"])
    by = {(r["object"], r["seed"], r["eval"], r["sign"]): r for r in ge["rows"]}
    for (o, s, e), x in sp.iterrows():
        p, n = by[(o, s, e, 1)], by[(o, s, e, -1)]
        se_p, se_n = GE._key(p["run_id"], "all")[1], GE._key(n["run_id"], "all")[1]
        se_a = R.combo_se(GE.anchor_weights(ge, p["anchor"]))
        assert x.even == pytest.approx((p["delta"] + n["delta"]) / 2) and x.odd == pytest.approx((p["delta"] - n["delta"]) / 2)
        assert x.odd_se == pytest.approx(math.sqrt(se_p ** 2 / 4 + se_n ** 2 / 4), rel=1e-9)
        assert x.Dp_se == pytest.approx(math.hypot(se_p, se_a), rel=1e-9) and x.Dn_se == pytest.approx(math.hypot(se_n, se_a), rel=1e-9)
        assert x.even_se >= se_a - 1e-12 and x.odd_se < math.hypot(x.Dp_se, x.Dn_se) / 2   # the anchor counted twice would widen it
        for r_, se_ in ((p, se_p), (n, se_n)):              # the lane's Wilson interval on its graded goals; z 1.959964 there
            if r_["category"] != "over-refusal" and not r_["n_ungraded"]:
                assert se_ == pytest.approx(r_["se_steered"], rel=1e-7)
            elif r_["category"] != "over-refusal":          # a goal filed but not graded is in the reading's n, as in its rate
                k = r_["rate"] / 100.0 * (r_["n_units"] + r_["n_ungraded"])
                lo, hi = R._judge_rules().wilson(k, r_["n_units"] + r_["n_ungraded"])
                assert se_ == pytest.approx(100.0 * (hi - lo) / 3.92, rel=1e-9)


def test_each_runs_units_are_its_payload_and_give_its_reading(ge):
    """The unit source: every reading's unit row pools by units that give its stored rate and n (all replies, ungraded goals counted as
    not attacked; coherent, the graded units without a degenerate reply), every unit id in its evaluation's item set; a payload that no
    longer gives its run's reading stops the pool (UnitsMismatch), and the file's own payload put back pools again."""
    GE.register_payloads(ge)
    for r in ge["rows"]:
        for mode, rate, n in (("all", r["rate"], r["n_units"] + r["n_ungraded"]), ("coherent", r["rate_coherent"], r["n_coherent"])):
            if rate is None:
                continue
            row = GE._ROW_OF[(r["run_id"], mode)]
            assert row["judge_source"] == GE.UNIT_SOURCE[mode] and R.reading_rederived(row)
            u = R.row_units(row)
            assert u.value == pytest.approx(rate / 100.0, abs=1e-12) and u.n == n
            ids = {f"{r['eval']}:{k}" for k in range(ge["item_sets"][r["eval"]]["n_items"])}
            assert set(u.table.index) <= ids and u.present == frozenset(ids)
    g = copy.deepcopy(ge)
    rid = g["rows"][0]["run_id"]
    _flip(g["payloads"][rid])
    try:
        GE.register_payloads(g)                            # the reading's row keeps its stored rate; its units are the drifted payload's
        assert GE._ROW_OF[(rid, "all")]["value"] == pytest.approx(ge["rows"][0]["rate"] / 100.0, abs=1e-15)
        with pytest.raises(P.UnitsMismatch):
            R.row_units(GE._ROW_OF[(rid, "all")])
    finally:
        GE.register_payloads(ge)
    assert R.row_units(GE._ROW_OF[(rid, "all")]).value == pytest.approx(ge["rows"][0]["rate"] / 100.0, abs=1e-12)


def test_every_seed_pool_is_item_clustered_and_the_marks_follow_the_notebooks_rule(ge):
    """No pool takes a bound and none is opaque; each drawn mark's interval is its delta +/- 1.96 hypot(the run's own se, the anchor's
    runs item-clustered), and the table's D(+) and D(-) are the figure's mean marks."""
    R.reset_pooled_ci(None)
    cp = GE.category_parts(ge).set_index(["object", "category"])
    rows = GE.rows(ge)
    assert R.POOLED_CI_REPORT["bound"] == {} and R.POOLED_CI_REPORT["opaque_seed_pools"] == 0 and R.POOLED_CI_REPORT["seed_pools"] > 0
    assert R.POOLED_CI_REPORT["seed_overlap_min"] == 1.0
    assert set(cp.loc[[("control_relu", "static"), ("control_lora", "static")], "even_how"]) == {R.SEED_CLUSTERED}
    assert cp.loc[("arditi_allpos", "static"), "Dp_how"] == R.SEED_ONE_FIT
    by = {(r["object"], r["seed"], r["eval"], GE.CONDITION[r["sign"]]): r for r in ge["rows"]}
    for _, x in rows.iterrows():
        r = by[(x.object, x.seed, x["eval"], x.condition)]
        se = math.hypot(x.se_steer, R.combo_se(x.anchor_terms))
        assert (x.ci_lo, x.ci_hi) == (pytest.approx(r["delta"] - 1.96 * se, abs=1e-12), pytest.approx(r["delta"] + 1.96 * se, abs=1e-12))
        assert x.steer_terms == {R.unit_key(GE._ROW_OF[(r["run_id"], "all")]): 1.0} and x.split_anchor_terms == x.anchor_terms


@pytest.mark.parametrize("mode", ["all", "coherent"])
def test_category_parts_are_the_appendix_table(ge, mode):
    cp = GE.category_parts(ge, coherent=(mode == "coherent")).set_index(["object", "category"])
    for (o, c), want in TABLE.items():
        for (part, m), (v, ci) in want.items():
            if m == mode:
                assert (_r1(cp.loc[(o, c), part]), _r1(cp.loc[(o, c), f"{part}_ci"])) == (v, ci), (o, c, part, mode)
    x = cp.loc[("control_relu", "multi-turn")]
    if mode == "all":
        assert {q: (_r1(x[q]), _r1(x[f"{q}_ci"])) for q in RELU_MT} == RELU_MT
    assert cp.loc[("control_relu", "over-refusal"), "seeds"] == "s2,s3" and cp.loc[("arditi_allpos", "static"), "n_seeds"] == 1


def test_eval_parts_are_the_numbers_the_text_quotes(ge):
    ep = GE.eval_parts(ge).set_index(["object", "eval"])
    odd = lambda o, e: _r1(ep.loc[(o, e), "odd"])
    assert [odd("arditi_allpos", e) for e in ("pasttense", "codeattack", "renellm", "grok420")] == [17.0, 24.9, 26.1, 26.1]
    assert [odd("arditi_allpos", e) for e in ("crescendo", "actorattack", "xstest", "or_bench_hard")] == [31.9, 29.3, -13.4, -35.3]
    assert (odd("control_relu", "grok420"), _r1(ep.loc[("control_relu", "grok420"), "odd_ci"])) == (-19.4, 3.7)
    assert (odd("control_lora", "pasttense"), odd("control_lora", "renellm"), odd("control_lora", "grok420")) == (12.0, 10.7, -27.3)
    assert (_r1(ep.loc[("control_relu", "crescendo"), "anchor"]), _r1(ep.loc[("control_relu", "actorattack"), "anchor"])) == (94.3, 72.9)


def test_unsteered_levels_and_the_coherent_moves(ge):
    assert {c: _r1(v) for c, v in GE.unsteered_levels(ge).items()} == {"static": 82.9, "multi-turn": 16.4, "over-refusal": 58.7}
    assert {c: _r1(v) for c, v in GE.unsteered_levels(ge, coherent=True).items()} == {"static": 82.9, "multi-turn": 16.4, "over-refusal": 58.6}
    mv = GE.coherent_moves(ge)
    assert (_r1(mv["marks"]), _r1(mv["means"])) == (12.5, 3.9)
    assert "ActorAttack" in mv["marks_where"] and "multi-turn" in mv["means_where"]


def test_table_prints_the_sign_rows(ge):
    t = GE.table_geometry_evenodd(ge)
    assert r"\emph{Unsteered (seed average)} & $82.9$ & $82.9$ & $16.4$ & $16.4$ & $58.7$ & $58.6$ \\" in t
    assert r"\quad sign-dependent & $-5.2\pm6.5$ & $-4.9\pm6.9$ & $-11.0\pm4.2$ & $-9.2\pm2.7$ & $+4.0\pm1.1$ & $+3.2\pm3.0$ \\" in t
    assert r"\quad $D(-\theta)$ & $-31.0\pm3.5$" in t and t.count(r"\addlinespace") == 2


# ---- the figure ----------------------------------------------------------------------------------------------------------------
def _vertices(m: str) -> np.ndarray:
    ms = MarkerStyle(m)
    return ms.get_path().transformed(ms.get_transform()).vertices


def _marks(fig) -> dict:
    out = {}
    for ax in fig.axes:
        for coll in ax.collections:
            g = coll.get_gid()
            if g and g.startswith("paired|"):
                out[g] = (float(coll.get_offsets()[0][1]), coll.get_paths()[0].vertices.copy(), tuple(np.round(coll.get_facecolor()[0][:3], 3)))
    return out


def test_figure_draws_every_reading_at_its_delta_in_its_shape_and_colour(ge, fig_and_dir):
    fig, _ = fig_and_dir
    m = _marks(fig)
    seeds = {k: v for k, v in m.items() if "|mean|" not in k}
    assert len(seeds) == len(ge["rows"]) == 108
    shape = {c: _vertices(AC.CONDITION_MARKERS[i]) for i, c in enumerate(GE.CONDITIONS)}
    for r in ge["rows"]:
        y, verts, face = seeds[f"paired|{GE.CONDITION[r['sign']]}|{r['object']}|{r['seed']}|{r['category']}|{r['eval']}"]
        assert y == pytest.approx(r["delta"], abs=1e-12)
        assert np.allclose(verts, shape[GE.CONDITION[r["sign"]]])
        assert face == tuple(np.round(matplotlib.colors.to_rgb(R.COLOR[r["object"]]), 3))
    assert not any(k.startswith("paired|") and "|control_relu|s1|over-refusal|" in k for k in m)
    assert sum(1 for k in seeds if "|arditi_allpos||" in k) == 16
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["static", "multi-turn", "benign compliance"]   # R.style titles top-left


def test_figure_means_are_the_category_means(ge, fig_and_dir):
    fig, _ = fig_and_dir
    means = {k: v[0] for k, v in _marks(fig).items() if "|mean|" in k}
    assert len(means) == 18
    cp = GE.category_parts(ge).set_index(["object", "category"])
    pm = AC.paired_means(GE.rows(ge), GE.CATEGORIES)
    assert pm.drawn.all()
    for _, r in pm.iterrows():
        q = "Dp" if r.condition == "plus" else "Dn"
        assert means[f"paired|{r.condition}|{r.object}|mean|{r.category}|{R.MEAN_COLUMN}"] == pytest.approx(r.y, abs=1e-12)
        assert r.y == pytest.approx(cp.loc[(r.object, r.category), q], abs=1e-9)
        assert (r.hi - r.lo) / 2 == pytest.approx(cp.loc[(r.object, r.category), f"{q}_ci"], abs=1e-9)   # the same seed pool


def test_figure_is_saved_and_the_line_under_it_names_the_unpaired_cells(ge, fig_and_dir):
    _, out = fig_and_dir
    assert sorted(p.name for p in out.iterdir()) == [f"{GE.FIGURE}.{x}" for x in ("pdf", "png", "svg")]
    nd = GE.not_drawn(ge)
    assert len(nd) == 2 and all("Control ReLU steer s1" in x and "α = −1" in x for x in nd)


# ---- the registry binding ------------------------------------------------------------------------------------------------------
def _tables(ge):
    objs, plus = [], {(o["object"], o["seed"]): o["object_id"] for o in ge["objects"] if o["sign"] == 1}
    for o in ge["objects"]:
        objs.append(dict(object_id=o["object_id"], paper_label=o["object"], fit_replicate_label=o["seed"] if o["sign"] == 1 and o["seed"] else None,
                         parent_object_ids_json=json.dumps([plus[(o["object"], o["seed"])]]) if o["sign"] == -1 else None))
    runs = [dict(run_id=r["run_id"], object_id=r["object_id"], eval_id=r["eval"], model_id="qwen35_4b_it", lifecycle_state="ingested") for r in ge["rows"]]
    runs += [dict(run_id=x, object_id="none", eval_id=a["eval"], model_id="qwen35_4b_it", lifecycle_state="ingested")
             for a in ge["anchors"].values() for x in a["runs"]]
    jj = [dict(judgment_id=r["judgment_id"], run_id=r["run_id"], status="complete", judge_model="minimax-m3") for r in ge["rows"]]
    jj += [dict(judgment_id=j, run_id=x, status="complete", judge_model="minimax-m3")
           for a in ge["anchors"].values() for j, x in zip(a["judgment_ids"], a["runs_used"])]
    return pd.DataFrame(objs), pd.DataFrame(runs).drop_duplicates("run_id"), pd.DataFrame(jj).drop_duplicates("judgment_id")


def test_binding_is_clean_then_names_each_break(ge):
    o, r, j = _tables(ge)
    assert GE.bind_problems(ge, o, r, j) == [] and GE.bind_problems(ge, o, r, None) == []
    o2 = o.copy(); o2.loc[0, "paper_label"] = "control_lora" if o2.loc[0, "paper_label"] != "control_lora" else "control_relu"
    assert any("labels it" in b for b in GE.bind_problems(ge, o2, r, j))
    r2 = r.copy(); r2.loc[r2.run_id == ge["rows"][0]["run_id"], "eval_id"] = "xstest"
    assert any(ge["rows"][0]["run_id"] in b for b in GE.bind_problems(ge, o, r2, j))
    j2 = j.copy(); j2.loc[0, "status"] = "superseded"
    assert any("status superseded" in b for b in GE.bind_problems(ge, o, r, j2))
    assert "judgment" in GE.judgments_sql(ge) and GE.runs_sql(ge).count("'") == 2 * len(r)


def test_a_run_filed_after_the_selection_snapshot_is_read_from_the_file_on_an_older_source_only(ge):
    """The two regenerated seed-1 Pliny runs (ingested after the run selection's snapshot at head 456957): a source at that head that lacks
    them reads them from the file, named; a newer source that lacks them, or one whose copy disagrees, is refused as for any run."""
    later = ge["provenance"]["run_selection"]["runs_not_in_snapshot"]
    assert [x["run_id"].split(":")[0] for x in later] == ["s1regen_pliny", "s1regen_pliny"] and all(x["ingest_seq"] > 456957 for x in later)
    ids = {x["run_id"] for x in later}
    o, r, j = _tables(ge)
    old = r[~r.run_id.isin(ids)].copy()
    assert GE.runs_read_from_file(ge, old, head=456957) == sorted(ids)
    assert GE.bind_problems(ge, o, old, None, head=456957) == []
    old.attrs["head"] = 456957                                          # R.query stamps the source's head on its frame
    assert GE.bind_problems(ge, o, old, None) == []
    newer = max(x["ingest_seq"] for x in later)
    assert GE.runs_read_from_file(ge, old, head=newer) == [] and len(GE.bind_problems(ge, o, old, j, head=newer)) == 2
    assert GE.runs_read_from_file(ge, r, head=456957) == []                # a source that holds them binds them


# ---- the payloads against the committed frame's item rows ----------------------------------------------------------------------
#: the pinned judgments' item rows with the columns the builder's ITEMS_SQL reads, from the frame's `item_result` table (a paper frame
#: has no `v_item_result_current` view; withdrawn rows are left out, as `registry_nb.item_rows_sql` leaves them out)
FRAME_ITEMS_SQL = ("SELECT ir.run_id, ir.item_id, ir.metric_id, ir.trial_index, ir.judgment_id, ir.value, ir.label, ir.extra_json, "
                   "j.status, j.ingest_seq, j.judge_id, g.judge_model, g.rubric_name "
                   "FROM item_result ir JOIN judgment j ON j.judgment_id = ir.judgment_id JOIN judge g ON g.judge_id = j.judge_id "
                   "WHERE ir.withdrawn_by_revision_id IS NULL AND ir.run_id IN ({ph}) AND ir.judgment_id IN ({ph})")


def _frame_items(ge) -> pd.DataFrame:
    """The committed frame's item rows of every judgment the sidecar pins (the readings' runs and the anchors' used runs), read-only from
    its SQLite. Reassembling that SQLite (about 4 GB, once per frame) is the slow tier's cost, as in `test_appendix_caperrors`: this
    module reads it only where a render (or an earlier run) already left it under `_cache/snapshot/<frame>/`, or under
    SISTEER_SLOW_TESTS=1, and skips by name otherwise."""
    from sisteer.registry import snapshot as SN
    root = REPO / "paper" / "snapshot"
    sid = (root / "CURRENT").read_text().strip()
    cache = pathlib.Path(R.CACHE) / "snapshot"
    if not (cache / sid / SN.PAPER_FRAME_SQLITE).exists() and os.environ.get("SISTEER_SLOW_TESTS") != "1":
        pytest.skip(f"frame {sid} ships its item rows in the SQLite only, and none is reassembled under {cache} (about 4 GB): a render "
                    "or SISTEER_SLOW_TESTS=1 builds it")
    jm = GE._judgments(ge)
    rids = sorted(jm)
    con = SN.open_paper_frame(str(root), cache_dir=str(cache))
    try:
        parts = [pd.read_sql_query(FRAME_ITEMS_SQL.format(ph=",".join("?" * len(ch))), con, params=ch + [jm[r] for r in ch])
                 for ch in (rids[i:i + 150] for i in range(0, len(rids), 150))]
    finally:
        con.close()
    items = pd.concat(parts, ignore_index=True)
    return items[items.judgment_id.eq(items.run_id.map(jm))].reset_index(drop=True)


def frame_payload_problems(ge, items: pd.DataFrame) -> list:
    """Every way the sidecar's payloads are not the frame's items, one phrase each (empty when they are): each used run's pinned
    MiniMax-M3 judgment holds complete item rows; its units by the builder's rules (`build.recount_run`: a goal's best StrongREJECT score
    over 0.5, an item's C/P/I label score, XSTest's safe subset; one trial per unit) are its payload's verdicts at their positions in the
    evaluation's item set, the union of that evaluation's runs' unit ids, sorted (a null exactly where the frame holds no unit of a
    goal); and each item set's size and sha256 are the ones the file names. `payload_problems` cannot see this: a payload whose verdicts
    sit at the wrong positions still gives its run's rate and n."""
    out, units = [], {}
    for rid, p in sorted(ge["payloads"].items()):
        x = items[items.run_id.eq(rid)]
        if not len(x) or set(x.judge_model) != {GE.JUDGE} or set(x.status) != {"complete"}:
            out.append(f"payload {rid}: its pinned judgment holds no complete MiniMax-M3 item rows in the frame")
            continue
        u = B.recount_run(x, SimpleNamespace(run_id=rid, eval_id=p["eval"], pass_label="", n=float("nan")))["units"]
        if u.trial_index.nunique() > 1 or u.item_id.duplicated().any():
            out.append(f"payload {rid}: the frame holds more than one trial per unit")
            continue
        units[rid] = dict(zip(u.item_id, u.v.astype(float)))
    by_eval = {}
    for rid, u in units.items():
        by_eval.setdefault(ge["payloads"][rid]["eval"], set()).update(u)
    for e, ids in sorted(by_eval.items()):
        ids = sorted(ids)
        if B.item_set_entry(ids) != ge["item_sets"].get(e):
            out.append(f"item_sets[{e!r}]: the frame's unit ids give {B.item_set_entry(ids)}, the file names {ge['item_sets'].get(e)}")
            continue
        for rid in sorted(r for r, p in ge["payloads"].items() if p["eval"] == e and r in units):
            mine = {ids[k]: float(y) for k, y in enumerate(ge["payloads"][rid]["v"]) if y is not None}
            off = sum(mine.get(i) != units[rid].get(i) for i in set(mine) | set(units[rid]))
            if off:
                out.append(f"payload {rid}: {off} verdict(s) are not the frame's at their positions")
    return out


@pytest.fixture(scope="module")
def frame_items(ge):
    return _frame_items(ge)


def test_every_payload_is_its_runs_frame_items_at_their_positions(ge, frame_items):
    """Each used run's payload is its pinned MiniMax-M3 judgment's units in the committed frame, by position in its evaluation's item set
    (151 runs, 4 ungraded goals among them), and the 8 item sets the file names by size and sha256 are the frame's; then a reordered
    payload (the same rate and n), a tampered item-set digest and a run the frame does not hold are each named."""
    assert frame_payload_problems(ge, frame_items) == []
    assert len(ge["payloads"]) == frame_items.run_id.nunique() == 151 and len(ge["item_sets"]) == 8
    assert sum(y is None for p in ge["payloads"].values() for y in p["v"]) == 4
    rid, p = next((r, p) for r, p in sorted(ge["payloads"].items()) if p["eval"] == "pasttense")
    i, j = p["v"].index(1.0), p["v"].index(0.0)
    bad = copy.deepcopy(ge)
    bad["payloads"][rid]["v"][i], bad["payloads"][rid]["v"][j] = 0.0, 1.0
    assert GE.payload_problems(bad) == []                               # the rate and n still hold, so the file's contract cannot see it
    assert frame_payload_problems(bad, frame_items) == [f"payload {rid}: 2 verdict(s) are not the frame's at their positions"]
    bad = copy.deepcopy(ge)
    bad["item_sets"]["xstest"]["ids_sha256"] = "0" * 64
    assert [x.split(":")[0] for x in frame_payload_problems(bad, frame_items)] == ["item_sets['xstest']"]
    assert f"payload {rid}: its pinned judgment holds no complete MiniMax-M3 item rows in the frame" in \
        frame_payload_problems(ge, frame_items[frame_items.run_id.ne(rid)])


def test_the_figure_and_table_are_registered_lane_fixed_on_minimax_m3(ge):
    """The render stamp says the sign decomposition is MiniMax-M3's under every policy: the cell registers GE.LANE_FIGURES with
    GE.lane_fixed_text before it draws the figure, which prints the table after it."""
    text = GE.lane_fixed_text(ge)
    assert text.startswith("minimax-m3: StrongREJECT at 0.5 per goal and over-refusal labels per item, judges ")
    assert text.endswith("judges " + ", ".join(i[:8] for i in ge["judge"]["judge_ids"])) and len(ge["judge"]["judge_ids"]) == 3
    src = (pathlib.Path(GE.__file__).resolve().parent / "make_registry_notebook.py").read_text(encoding="utf-8")
    cell = src[src.index('code("5f821770"'):]
    cell = cell[:cell.index('\n""")')]
    reg = "R.register_lane_fixed(GE.LANE_FIGURES, GE.lane_fixed_text(ge))"
    assert cell.count(reg) == 1 and cell.index(reg) < cell.index("GE.figure_geometry_evenodd(") < cell.index("GE.table_geometry_evenodd(")
