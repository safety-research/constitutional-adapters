"""The generator-comparison appendix figure (`figA_qwen_generator_ablation`) on the control-pool appendix's five panels.

Fast, no registry: the committed file, a PROVISIONAL synthetic file (the synthetic registry's derivation before the wave is filed: the
committed static family, every later reading pending) and a FULL one derived from the synthetic registry with the whole wave filed
(`appendix_generator_world`, both derived unpinned), and synthetic main-figure frames (the control-pool tests'
`main_frames`: unsteered runs, the fitted seeds, the decode-only conditions, anchored by `registry_nb.anchored` itself). Pinned here:
  * the structure: Figure 11's categories (`registry_nb.ctlstop_categories`) and grid (`registry_nb.CTLSTOP_GRID`), five panels titled
    as there, a y axis of its own on every panel (multi-query and multi-turn each labelled, not shared), 72 per-seed gids and 10
    mean gids, a reading not yet in the data drawn as the gap glyph and named under the figure (`not_drawn`), the over-refusal mark
    minus the change in the label mean x 100, the capability mean `registry_nb.category_means` over MMLU and GSM8K then
    `registry_nb.pool_seeds`;
  * the anchors: the RULE is Figure 11's in the same frames (each row's anchor runs, value, route and cap flag are the ones
    `registry_nb.ctlstop_anchored` gives the same reading as a control-pool row), anchored once per condition (both at once
    refuses: the cell key has no condition), the interval is `registry_nb.anchored`'s on the same frame, a static reading above every
    comparator's rung is left without a delta and named;
  * the arithmetic: the Qwen - Opus difference is ASR_Opus - ASR_Qwen over matched seeds and does not move with the anchors, the
    readings recompute independently from the payloads at thresholds 0.5 and 0.75, the static pre-fixed verdict is -7.56 pp
    (-11.06 to -4.07) at 0.5, GSM8K's coverage and its all-200 sensitivity, the drift line;
  * the notebook: the md/code pair sits after the control-pool comparison cell (76fdab54) and directly before the zip cell, passes
    ctl_md as the reference frame, registers the lane-fixed judges before the figure is written and names the figure once; a
    snapshot render opens no registry and a live render refuses a registry that differs."""
import ast
import copy
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
TESTS = pathlib.Path(__file__).resolve().parent
for _p in (str(NOTEBOOKS), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import registry_nb as R  # noqa: E402
import appendix_common as AC  # noqa: E402
import appendix_generator as AG  # noqa: E402
import appendix_generator_sidecar as G  # noqa: E402
import appendix_generator_world as W  # noqa: E402
import test_registry_nb_ctlstop_appendix as CT  # noqa: E402  (its synthetic main-figure frames)

FIGURE = "figA_qwen_generator_ablation"
STATIC_CAPS = {e: [4096.0, 8192.0] for e in G.STATIC}
GSM8K_CAPS = {"gsm8k": [8192.0, 16384.0]}
#: the static per-cell counts at strict > 0.5 (the comparison filing's), (eval, arm, seed) -> k of 259
K = G.FILING_K


@pytest.fixture(autouse=True, scope="module")
def _offline():
    """As the control-pool tests run: no registry read for headline metrics, the re-judge policy, the undetermined-goal bound lifted
    for the synthetic main-figure payloads (which leave a few percent of goals unscored)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "headline_metric_ids", lambda: {})
        mp.setattr(R, "JUDGES", "rejudge")
        mp.setattr(R._judge_rules(), "SR_MAX_UNDETERMINED", 1.0)
        yield


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIGS", str(tmp_path / "figs"))
    monkeypatch.delenv(AC.MIRROR_ENV, raising=False)
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def committed():
    return G.load_rows()


@pytest.fixture(scope="module")
def full(committed):
    """The file a derivation gives once the whole wave is filed (synthetic later families, the committed static family)."""
    c = W.registry(committed)
    W.file_wave(c, rng=np.random.default_rng(7))
    with W.unpinned():
        d = G.derive(c)
    c.close()
    assert d["pending"] == []
    return d


@pytest.fixture(scope="module")
def frames(_offline):
    """Figure 11's synthetic frames (ctl_op, ctl_pool, ctl_md): static ladders on 4,096 and 8,192, GSM8K's on 8,192 and 16,384."""
    op, d, _cm, _p = CT.main_frames(list(R.CTLSTOP_EVALS), caps={**STATIC_CAPS, **GSM8K_CAPS})
    return op, None, d


def _gen(data, frames):
    op, pool, md = frames
    frame = AG.lane_frame(data)
    gen_d = AG.anchor(frame, op, pool, md)
    return dict(data=data, frame=frame, gen_d=gen_d, rows=AG.tidy_rows(gen_d, data))


@pytest.fixture(scope="module")
def gen(full, frames):
    return _gen(full, frames)


@pytest.fixture(scope="module")
def provisional(committed, frames):
    """A provisional file, as the committed one was while the wave ran: the synthetic registry's derivation before any later reading is
    filed (the committed static family, every later reading pending)."""
    c = W.registry(committed)
    with W.unpinned():
        d = G.derive(c)
    c.close()
    assert d["pending"]
    return _gen(d, frames)


def _gids(fig) -> dict:
    """{gid: (x, y)} of every scatter mark on the canvas."""
    out = {}
    for ax in fig.axes:
        for coll in ax.collections:
            gid = coll.get_gid()
            if gid:
                off = coll.get_offsets()
                out[gid] = (float(off[0][0]), float(off[0][1]))
    return out


# ---- 1. structure -----------------------------------------------------------------------------------------------------------------
def test_figure_14s_categories_grid_and_options():
    assert AG.CATEGORIES == R.ctlstop_categories(R.MODEL_ID) and AG.LAYOUT == R.CTLSTOP_GRID
    o = AG.figure_options()
    assert (o["layout"], o["sharey"], o["conditions"], o["objects"], o["mean_rule"], o["unit"]) == (
        R.CTLSTOP_GRID, False, ["opus_written", "qwen_written"], ["constitution_lora_1m"], "complete", 2.4)
    assert o["categories"] is AG.CATEGORIES and AG.OBJECT not in R.COLOR      # its own key, never Figure 3's constitution LoRA


def test_the_full_figure_draws_72_seed_marks_and_10_means_on_five_panels(gen):
    rows = gen["rows"]
    assert len(rows) == 72 and rows.delta.notna().all()
    fig = AG.draw(rows)
    gids = _gids(fig)
    seed = [g for g in gids if g.startswith("paired|") and "|mean|" not in g]
    mean = [g for g in gids if g.startswith("paired|") and "|mean|" in g]
    assert len(seed) == 72 and len(mean) == 10 and not [g for g in gids if g.startswith("paired-pending")]
    assert [ax.get_title(loc="left") for ax in fig.axes] == [c for row in R.CTLSTOP_GRID for c in row]   # Figure 11's titles
    by = {ax.get_title(loc="left"): ax for ax in fig.axes}
    for cat in AG.CATEGORIES:
        assert by[cat].get_ylabel(), cat                                  # every panel labels its own y axis
    assert not by["multi-query"].get_shared_y_axes().joined(by["multi-query"], by["multi-turn"])
    assert not by["over-refusal"].get_shared_y_axes().joined(by["over-refusal"], by["capability"])
    assert AG.not_drawn(rows) == []


def test_the_provisional_file_draws_gap_glyphs_named_under_the_figure(provisional):
    rows, data = provisional["rows"], provisional["data"]
    assert len(rows) == 72 and data["pending"]
    # THE COUNTS COME FROM THE FILE, so a mid-wave provisional file (written again as runs are filed) holds this as the day-0 one does:
    # every grid reading not pending is drawn, every pending one is a gap, and a panel's mean is a gap where any of its readings is
    cat_of = {e: c for c, (evals, _s, _y) in AG.CATEGORIES.items() for e in evals}
    pend = {(G.CONDITION[p["arm"]], f"s{p['seed']}", p["eval_id"]) for p in data["pending"] if p["arm"] != G.UNSTEERED}
    drawn = rows[rows.delta.notna()]
    assert {(r.condition, r.seed, r["eval"]) for _, r in drawn.iterrows()} == {
        (c, s, e) for e in AG.EVALS for c in AG.CONDITIONS for s in AG.SEEDS} - pend
    assert len(drawn) == 72 - len(pend) and len(pend) == len(data["pending"])            # nothing static or drift is pending
    gids = _gids(AG.draw(rows))
    seed = [g for g in gids if "|mean|" not in g]
    mean = [g for g in gids if "|mean|" in g]
    assert len(seed) == 72 and len(mean) == 10
    assert sum(g.startswith("paired-pending|") for g in seed) == len(pend)
    assert sum(g.startswith("paired-pending|") for g in mean) == len({(cat_of[e], c) for c, _s, e in pend})
    for g in [g for g in gids if g.startswith("paired-pending|")]:
        assert gids[g][1] == 0.0                                          # the gap glyph sits on the zero line
    named = " ".join(AG.not_drawn(rows))
    for p in data["pending"]:
        assert f"s{p['seed']} {R.EVAL_LABEL[p['eval_id']]}" in named


def test_the_over_refusal_mark_is_minus_the_change_in_the_label_mean(gen):
    gids = _gids(AG.draw(gen["rows"]))
    x = gen["gen_d"][gen["gen_d"].eval_id.isin(["xstest", "or_bench_hard"])]
    assert len(x) == 12
    for r in x.itertuples():
        assert r.metric_id == R.OR_METRIC and r.d_y == pytest.approx(r.y - r.anchor_value, abs=1e-9)
        y = gids[f"paired|{r.condition}|{AG.OBJECT}|{r.seed}|over-refusal|{r.eval_id}"][1]
        assert y == pytest.approx(-r.d_y, abs=1e-9)                       # benign compliance: minus the label mean's change x 100


def test_the_capability_mean_is_category_means_over_mmlu_and_gsm8k_then_pool_seeds(gen):
    mm = AG.mean_marks(gen["rows"])
    gd = gen["gen_d"]
    for c in AG.CONDITIONS:
        f = gd[gd.condition.eq(c) & gd.eval_id.isin(["mmlu", "gsm8k"])].copy()
        cm = R.category_means(f, categories={"capability": R.CTLSTOP_CAPABILITY})
        pooled = R.pool_seeds(cm).iloc[0]
        got = mm.loc[("capability", c)]
        assert (got.y, got.lo, got.hi) == pytest.approx((pooled.y, pooled.lo, pooled.hi), abs=1e-9)
        assert got.y == pytest.approx(f.groupby("seed").d_y.mean().mean(), abs=1e-9)   # by hand: within seed, then the seeds


# ---- 2. anchors ---------------------------------------------------------------------------------------------------------------------
def test_the_anchor_rule_is_figure_14s_in_the_same_frames(gen, frames):
    """Each reading, anchored as a control-pool row of the comparison figure would be (relabelled to a slot it does not hold there),
    takes exactly the anchor the figure's own call gives it: the RULE, held to the same reference frame."""
    op, pool, md = frames
    gd = gen["gen_d"]
    for c in AG.CONDITIONS:
        assert R.ctlstop_anchor_problems(gd[gd.condition.eq(c)], md) == []
        rows = gen["frame"][gen["frame"].role.eq("drawn") & gen["frame"].condition.eq(c)].copy()
        rows["object_label"], rows["slot"], rows["rule"] = "control_lora", "control_lora", "control_pool"
        as_ctl = R.ctlstop_anchored(op, rows.reset_index(drop=True), pool, md).set_index("run_id")
        mine = gd[gd.condition.eq(c)].set_index("run_id")
        for col in ("d_y", "d_lo", "d_hi", "anchor_run", "anchor_value", "route", "cap_mismatch", "anchor_draws"):
            pd.testing.assert_series_equal(mine[col].sort_index(), as_ctl[col].sort_index(), check_names=False, check_dtype=False)


def test_both_conditions_in_one_call_refuse(gen, frames):
    op, pool, md = frames
    drawn = gen["frame"][gen["frame"].role.eq("drawn")].reset_index(drop=True)
    with pytest.raises(AssertionError, match="anchoring kept 36 of 72"):
        R.ctlstop_anchored(op, drawn, pool, md)


def test_the_interval_is_anchoreds_on_the_same_frame(gen, frames):
    """Off over-refusal (whose rows take a bound run first), each row's delta and interval are `registry_nb.anchored`'s on the frame's
    unsteered rows plus that condition's rows, and the interval carries the anchor's error beside the reading's own."""
    op, pool, md = frames
    for c in AG.CONDITIONS:
        fr = gen["frame"]
        rows = fr[fr.role.eq("drawn") & fr.condition.eq(c) & ~fr.eval_id.isin(R.SHARED_FRAME_ANCHOR)]
        ref = R.anchored(pd.concat([op[op.object_label.eq("none")], rows], ignore_index=True), pool=pool)
        ref = ref[ref.object_label.eq(AG.OBJECT)].set_index("run_id")
        mine = gen["gen_d"].set_index("run_id").loc[ref.index]
        for col in ("d_y", "d_lo", "d_hi", "anchor_value"):
            np.testing.assert_allclose(mine[col].astype(float), ref[col].astype(float), rtol=0, atol=1e-12)
        own_se = (mine.hi - mine.lo) / 3.92
        assert (((mine.d_hi - mine.d_lo) / 2 / 1.96) >= own_se - 1e-12).all()


def test_a_static_reading_above_every_comparators_rung_is_left_out_and_named(provisional):
    """With the static panel's runs on 4,096 alone, a reading the ladder read at 8,192 has no same-cap comparator: no delta, its route
    names why, the gap glyph stands in its place and the line under the figure names it."""
    op, d, _cm, _p = CT.main_frames(list(R.CTLSTOP_EVALS), caps=GSM8K_CAPS)
    gen_d = AG.anchor(provisional["frame"], op, None, d)
    rows = AG.tidy_rows(gen_d, provisional["data"])
    gap = rows[(rows.category == "static") & rows.delta.isna()]
    assert len(gap) == sum(1 for r in provisional["data"]["rows"]
                           if r["role"] == "drawn" and r["eval_id"] in G.STATIC and r["token_cap_tokens"] == 8192) > 0
    assert gap.route.str.startswith(R.CTLSTOP_RUNG_GAP_ROUTE).all()
    assert sum("not drawn - no same-cap comparator" in n for n in AG.not_drawn(rows)) == len(gap)


# ---- 3. arithmetic ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("t", [0.5, 0.75])
def test_the_readings_recompute_independently_at_two_thresholds(gen, monkeypatch, t):
    monkeypatch.setattr(R, "SR_THRESHOLD", t)
    data = gen["data"]
    f = AG.lane_frame(data).set_index("run_id")
    for r in data["rows"]:
        if r["role"] == "rung":
            continue
        e = data["items"][r["run_id"]]
        x = f.loc[r["run_id"]]
        fam = R.CTLSTOP_EVALS[r["eval_id"]]["family"]
        if fam == "strongreject":                                          # every unit is scored: every goal is determined
            k = sum(1 for b in e["best"][0] if b is not None and b > t)
            assert (x.k, x.n) == (k, len(e["best"][0])) and x.value == pytest.approx(k / len(e["best"][0]), abs=1e-12)
        elif fam == "over_refusal":
            sub = e["headline_subset"]
            labs = [lab for lab, s in zip(e["label"], e["subset"]) if lab is not None and (sub is None or s == sub)]
            assert x.n == len(labs) and x.value == pytest.approx(np.mean([(2 - lab) / 2 for lab in labs]), abs=1e-12)
        else:
            n_items = len(e["base"])
            final = [(rg if at is not None else b) for b, rg, at in zip(e["base"], e.get("rung") or [None] * n_items,
                                                                       e.get("rung_at_cap") or [None] * n_items)]
            sc = [v for v in final if v is not None]
            assert (x.k, x.n) == (sum(sc), len(sc))
    if t == 0.5:
        st = {(r["eval_id"], r["arm"], r["seed"]): r["run_id"] for r in data["rows"] if r["eval_id"] in G.STATIC and r["role"] != "rung"}
        assert {c: int(f.loc[rid].k) for c, rid in st.items()} == K


def test_the_static_verdict_at_half(gen):
    v = AG.verdict(gen["frame"])
    assert v["pooled"][:3] == pytest.approx((-7.561132561132562, -11.055149461498955, -4.0671156607661665), abs=1e-9)
    assert v["pooled_no_renellm"][:3] == pytest.approx((-8.108108108108109, -12.882379257436263, -3.3338369587799512), abs=1e-9)
    by_hand = [np.mean([100.0 * (K[(e, "opus", s)] - K[(e, "qwen", s)]) / 259 for e in AG.VERDICT]) for s in (1, 2, 3)]
    assert list(v["pooled"][3]) == pytest.approx(by_hand, abs=1e-9)


def test_the_corpus_difference_is_asr_opus_minus_asr_qwen_and_does_not_move_with_the_anchors(gen, frames):
    rd = AG.readings(gen["frame"]).set_index(["condition", "seed", "eval_id"])
    for cat, (evals, sign, _y) in AG.CATEGORIES.items():
        m, lo, hi, seeds = AG.corpus_difference(gen["frame"], evals, sign)
        for s, got in zip(AG.SEEDS, seeds):
            want = np.mean([sign * (rd.loc[("qwen_written", s, e)].y - rd.loc[("opus_written", s, e)].y) for e in evals])
            assert got == pytest.approx(want, abs=1e-12)
        assert (m, lo, hi) == pytest.approx(AG.t2_interval(seeds), abs=1e-12)
    st = AG.corpus_difference(gen["frame"], AG.VERDICT, -1)
    asr = lambda c, s: np.mean([rd.loc[(c, s, e)].y for e in AG.VERDICT])
    assert list(st[3]) == pytest.approx([asr("opus_written", s) - asr("qwen_written", s) for s in AG.SEEDS], abs=1e-12)
    # move every unsteered anchor by a tenth: the drawn deltas move, the difference read from the readings cannot
    op, pool, _md = frames
    op2 = op.copy()
    none = op2.object_label.eq("none")
    op2.loc[none, ["value", "ci_lo", "ci_hi"]] = (op2.loc[none, ["value", "ci_lo", "ci_hi"]] * 0.9).values
    gd2 = AG.anchor(gen["frame"], op2, pool, R.anchored(op2, pool=None))
    assert not np.allclose(gd2.sort_values("run_id").d_y.values, gen["gen_d"].sort_values("run_id").d_y.values)
    assert AG.corpus_difference(AG.lane_frame(gen["data"]), AG.VERDICT, -1) == st


def test_a_seeds_two_marks_differ_by_the_reading_difference_where_they_share_an_anchor(gen):
    gd = gen["gen_d"].set_index(["condition", "seed", "eval_id"])
    shared = 0
    for e in G.EVALS:
        for s in AG.SEEDS:
            o, q = gd.loc[("opus_written", s, e)], gd.loc[("qwen_written", s, e)]
            if o.anchor_run == q.anchor_run:
                shared += 1
                assert q.d_y - o.d_y == pytest.approx(q.y - o.y, abs=1e-9)
    assert shared >= 30                                                   # every pair but the static ones read on different rungs


def test_gsm8k_coverage_and_the_all_200_sensitivity(gen):
    data = copy.deepcopy(gen["data"])
    rid = next(r["run_id"] for r in data["rows"] if r["eval_id"] == "gsm8k" and r["role"] == "drawn" and r["rung_run_id"] is None)
    e = data["items"][rid]
    for i in range(3):
        e["base"][i], e["base_at_cap"][i] = None, True                    # three answers the cap cut, no rung to redraw them
    cov = AG.gsm8k_coverage(data, AG.lane_frame(data))
    assert len(cov) == 6
    row = next(r for r in cov.itertuples() if r.n < 200)
    assert row.n == 197 and row.items == 200
    assert row.acc == pytest.approx(100.0 * row.k / 197) and row.acc_all == pytest.approx(100.0 * row.k / 200)
    assert "scored of 200" in AG.summary_text(data, gen["gen_d"], gen["rows"], None)


def test_the_drift_line_is_the_lanes_unsteered_run_against_the_pooled_anchor(gen, frames):
    out = AG.drift(gen["frame"], gen["gen_d"], frames[2])
    assert [x["eval_id"] for x in out] == list(G.STATIC)
    dr = gen["frame"][gen["frame"].role.eq("drift_anchor")].set_index("eval_id")
    for x in out:
        a = dr.loc[x["eval_id"]]
        assert x["rerun"] == pytest.approx(100.0 * a.value, abs=1e-12) and x["cap"] == a.token_cap_tokens
        peers = gen["gen_d"][gen["gen_d"].eval_id.eq(x["eval_id"]) & gen["gen_d"].token_cap_tokens.eq(x["cap"])]
        assert x["anchor"] == pytest.approx(float(peers.anchor_value.iat[0])) and x["drift"] == pytest.approx(x["rerun"] - x["anchor"])


def test_the_printed_lines(gen, provisional):
    lines = AG.summary_text(gen["data"], gen["gen_d"], gen["rows"], None).splitlines()
    assert lines[0].startswith("VERDICT (pre-fixed") and "-7.56 pp defense success, 95% -11.06 to -4.07" in lines[0]
    # the panel headlines' Qwen - Opus is read from the readings, so the line above them says it need not be the gap between the marks
    assert lines[1] == AG.PANEL_NOTE and "anchor-free from the readings" in lines[1] and "gap between the two marks" in lines[1]
    for cat in AG.CATEGORIES:
        line = next(ln for ln in lines if ln.startswith(f"{cat} ("))
        assert ("the pre-fixed verdict" if cat == "static" else "descriptive") in line and "not drawn" not in line
        assert lines.index(line) > 1
    gsm = next(ln for ln in lines if ln.startswith("GSM8K coverage (thinking on"))
    # every unscored answer the cap cut: on the first rung and never redrawn, or redrawn and cut again on the last
    assert "one cut at 8,192 that the ladder did not redraw, or one it redrew that the 16,384 cap cut again" in gsm
    assert "last rung" not in gsm and any(ln.startswith("drift (") for ln in lines)
    pend = provisional["data"]["pending"]
    if pend:
        ptext = AG.summary_text(provisional["data"], provisional["gen_d"], provisional["rows"], None)
        assert f"{len(pend)} reading(s) not yet in the data" in ptext
        for cat, (evals, _s, _y) in AG.CATEGORIES.items():
            missing = sum(1 for p in pend if p["eval_id"] in evals)
            if missing:                                                    # counted from the file: holds at day 0 and mid-wave
                assert f"not yet computed ({len(evals) * 6 - missing} of {len(evals) * 6} readings in)" in ptext, cat


def test_the_marks_table_flags_an_anchor_at_another_cap(gen, provisional):
    """No glyph on the canvas, the flag in the table, as the main figures do: `other` exactly where the anchored row carries cap_mismatch
    off over-refusal (a static reading on a rung no unsteered run was read at), on over-refusal the route the anchored row took (these
    frames bind every fitted over-refusal row: `bound`), empty on a gap."""
    for g in (gen, provisional):
        rows = g["rows"]
        t = AG.marks_table(rows)
        assert list(t.columns)[-2:] == ["cap", "anchor cap"] and len(t) == len(rows)
        shared = rows["eval"].isin(R.SHARED_FRAME_ANCHOR)
        has = rows.delta.notna()
        assert (t["anchor cap"][~has] == "").all() and (t["anchor cap"][has & shared] == "bound").all()
        assert (rows.anchor_binding[has & shared] == R.CTLSTOP_ANCHOR_BOUND).all() and rows.anchor_binding[~shared].isna().all()
        other = has & ~shared & rows.cap_mismatch.astype(bool)
        assert (t["anchor cap"][other] == "other").all() and (t["anchor cap"][has & ~shared & ~other] == "same").all()
    rows = gen["rows"].copy()                     # these frames hold unsteered runs on both static rungs: no static row carries the flag
    i = rows.index[rows.category.eq("static")][0]
    rows.loc[i, "cap_mismatch"] = True
    assert AG.marks_table(rows)["anchor cap"].iat[i] == "other" and AG.marks_table(gen["rows"])["anchor cap"].iat[i] == "same"


def test_a_pooled_over_refusal_anchor_prints_as_pooled(gen):
    """THE POOLED SHAPE (Figure 11's fitted over-refusal rows carry no registry binding, as in a live render whose registry holds no
    current contrast for them: `test_registry_nb_ctlstop_appendix.main_frames(bound=False)`): every over-refusal mark is differenced
    against the runs pooled for its seed and the table says `pooled`, never `bound`; every other mark's flag is the bound shape's."""
    op, d, _cm, _p = CT.main_frames(list(R.CTLSTOP_EVALS), caps={**STATIC_CAPS, **GSM8K_CAPS}, bound=False)
    gen_d = AG.anchor(gen["frame"], op, None, d)
    rows = AG.tidy_rows(gen_d, gen["data"])
    shared = rows["eval"].isin(R.SHARED_FRAME_ANCHOR) & rows.delta.notna()
    assert int(shared.sum()) == 2 * 2 * 3                                 # XSTest and OR-Bench-Hard, two corpora, three seeds
    assert (rows.anchor_binding[shared] == R.CTLSTOP_ANCHOR_POOLED).all()
    for r in rows[shared].itertuples():
        assert r.anchor_draws == len(CT.pooled_runs(r.eval).split(","))
    t, t_bound = AG.marks_table(rows), AG.marks_table(gen["rows"])
    assert (t["anchor cap"][shared] == "pooled").all() and not (t["anchor cap"] == "bound").any()
    assert (t["anchor cap"][~shared] == t_bound["anchor cap"][~shared]).all()


def test_an_over_refusal_mark_with_no_binding_refuses():
    """The route is the anchored row's, never assumed: an over-refusal mark with a delta and no `anchor_binding` raises."""
    rows = pd.DataFrame([dict(eval="xstest", condition="opus_written", seed="s1", delta=-1.0, cap_mismatch=False, anchor_binding=None),
                         dict(eval="mmlu", condition="opus_written", seed="s1", delta=0.5, cap_mismatch=False, anchor_binding=None)])
    with pytest.raises(ValueError, match="1 over-refusal mark.*no anchor binding"):
        AG.anchor_cap(rows)
    with pytest.raises(ValueError, match="no anchor binding"):
        AG.anchor_cap(rows.drop(columns=["anchor_binding"]))
    rows.loc[0, "anchor_binding"] = "pooled"
    assert list(AG.anchor_cap(rows)) == ["pooled", "same"]


def test_saving_writes_three_formats_with_the_lane_fixed_stamp(gen, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "LANE_FIXED", {})
    R.register_lane_fixed(AG.LANE_FIGURES, AG.lane_fixed_text())
    AG.draw(gen["rows"], FIGURE, out_dirs=[str(tmp_path)])
    for ext in ("png", "pdf", "svg"):
        assert (tmp_path / f"{FIGURE}.{ext}").stat().st_size > 1000
    assert AG.lane_fixed_text() in (tmp_path / f"{FIGURE}.svg").read_text(encoding="utf-8")


# ---- 4. the data and the notebook ---------------------------------------------------------------------------------------------------
class _Conn:
    def close(self):
        pass


def test_a_snapshot_render_opens_no_registry(committed, monkeypatch):
    monkeypatch.setattr(G, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("a snapshot render opened the registry")))
    assert AG.load("snapshot")["provenance"]["payload_sha256"] == committed["provenance"]["payload_sha256"]


def test_a_live_render_refuses_a_registry_that_differs_or_cannot_be_read(committed, monkeypatch):
    monkeypatch.setattr(G, "connect", lambda *a, **k: _Conn())
    monkeypatch.setattr(G, "report_file_check", lambda conn: ("", []))
    monkeypatch.setattr(G, "derive", lambda conn: copy.deepcopy(committed))
    assert AG.load("live")["schema"] == G.SCHEMA
    moved = copy.deepcopy(committed)
    moved["rows"][0]["view_legacy"] = "qwen35_4b_qg1ms2"
    monkeypatch.setattr(G, "derive", lambda conn: moved)
    with pytest.raises(AG.GeneratorSidecarMismatch, match="differs from the live registry"):
        AG.load("live")
    monkeypatch.setattr(G, "derive", lambda conn: (_ for _ in ()).throw(G.Refused(["a reason"])))
    with pytest.raises(AG.GeneratorSidecarMismatch, match="no longer yields"):
        AG.load("live")
    monkeypatch.setattr(G, "connect", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x")))
    with pytest.raises(AG.GeneratorSidecarMismatch, match="cannot open the registry"):
        AG.load("live")


def _cells():
    src = (NOTEBOOKS / "make_registry_notebook.py").read_text(encoding="utf-8")
    return sorted((n.lineno, n.func.id, n.args[0].value, n.args[1].value) for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("md", "code") and len(n.args) >= 2)


def test_the_cells_sit_after_the_comparison_cell_and_before_the_zip_cell():
    """The two cells sit after the control-pool comparison cell (76fdab54), whose frames they read, and before the zip cell; the
    distillation-target appendix's cells sit between the comparison cell and them, so the lane-fixed registration here stamps none of
    their files, and the appendix cells that read data files sit between them and the zip cell."""
    cells = _cells()
    ids = [c[2] for c in cells]
    i = ids.index("be6069de") - 1
    assert ids.index("76fdab54") <= i
    assert ids[i + 1:i + 3] == ["be6069de", "431232a9"] and cells[i + 1][1] == "md" and cells[i + 2][1] == "code"
    assert i + 2 < ids.index("b7d4e9f2")             # the appendix cells that read data files sit between it and the zip cell
    code = cells[i + 2][3]
    assert "AG.anchor(AG.lane_frame(gen_data), ctl_op, ctl_pool, ctl_md)" in code       # Figure 11's frames, ctl_md the reference
    assert code.index("R.register_lane_fixed(AG.LANE_FIGURES, AG.lane_fixed_text())") < code.index("AG.draw(")
    named = [(c[2], ast.unparse(n.func)) for c in cells if c[1] == "code" for n in ast.walk(ast.parse(c[3]))
             if isinstance(n, ast.Call) and any(isinstance(a, ast.Constant) and a.value == FIGURE
                                                for a in list(n.args) + [k.value for k in n.keywords])]
    assert named == [("431232a9", "AG.draw")]


def test_the_caption_names_the_five_panels_and_the_one_pre_fixed_comparison():
    md = next(c[3] for c in _cells() if c[2] == "be6069de")
    for word in ("Best-of-75", "PAIR", "Crescendo", "ActorAttack", "XSTest", "OR-Bench-Hard", "MMLU", "GSM8K", "Figure 11",
                 "pre-fixed", "descriptive", "MiniMax-M3", "1M tokens", "effective batch of 8", "pooled runs at another cap"):
        assert word in md, word
    assert len(md.split()) < 330
    # the over-refusal anchor in words true of a bound render and a pooled one alike: never "the run" the fitted points take
    flat = " ".join(md.split())
    assert "the over-refusal reference Figure 11 uses for that seed" in flat and "fitted points of the same seed take" not in flat


def test_the_marks_tables_header_names_both_over_refusal_routes():
    code = " ".join(next(c[3] for c in _cells() if c[2] == "431232a9").split())
    assert "bound = the one run the registry binds to them" in code and "pooled = the runs pooled for them" in code
    assert "over-refusal's per-seed run" not in code


def test_the_cells_prose_is_clean():
    """The control-pool cells' containment rule, held here too: no demo marker, ruling, decision id, attribution or absolute path in
    either cell, and no process word in the prose."""
    cells = {c[2]: c for c in _cells()}
    for cid in ("be6069de", "431232a9"):
        body = cells[cid][3]
        assert not [m for m in CT.DEMO_MARKERS if m in body], cid
        assert not re.search(r"\b(?:ruling|r2v4|decisions?\.md|D\d+|D-[a-z])\b|/home/|~/|\bthe user\b|\bthe owner\b", body), cid
    assert not re.search(r"\b(?:sidecar|registry|lane|wave|judgment id|run id)\b", cells["be6069de"][3])
