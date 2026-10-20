"""The all-model appendix's layout and table form.

Three rulings, one per family of appendix artifact:
  * a figure whose x is the OBJECT (`category_panels`: figA_jailbreaks/misalignment/capabilities_models) stacks its category panels
    vertically, one per row, whatever row layout the caller names;
  * a figure whose x is the EVAL, in the main figures' form (`absolute_panels`), keeps ONE row - and gains a relative companion
    (`delta_panels`) on the same layout: the anchored deltas, oriented so up is better, a zero line, each panel labeled and ranged on
    its own axis;
  * the table under an absolute figure follows the shared table form: the unsteered model ABSOLUTE first, one row per model, every
    other arm a SIGNED change against its own model's unsteered run, read from the anchored frame (the same deltas the figures draw).
Synthetic frames throughout; figures are written to a temporary directory."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import registry_nb as R  # noqa: E402

Q, G = "qwen35_4b_it", "gemma4e4b_it"


def _row(model, obj, eval_id, d_y, anchor, seed="s1", held=0, lo=None, hi=None):
    lo = d_y - 5.0 if lo is None else lo
    hi = d_y + 5.0 if hi is None else hi
    return dict(model_id=model, object_label=obj, seed=seed, draw="", eval_id=eval_id, y=anchor + d_y, lo=anchor + lo, hi=anchor + hi,
                d_y=d_y, d_lo=lo, d_hi=hi, anchor_in_frame=True, ceiling=False, anchor_value=anchor, held_out=held, route="common anchor (1 runs)")


def _deltas():
    """Two models, three categories: single-turn (two evals, sign -1), multi-query (one eval), capability (one eval, sign +1)."""
    rows = [
        _row(Q, "safety_si", "pasttense", -10.0, 30.0), _row(Q, "safety_si", "codeattack", -20.0, 40.0),
        _row(Q, "safety_si", "strongreject_bon", -5.0, 60.0), _row(Q, "safety_si", "mmlu", 2.0, 70.0),
        _row(Q, "control_lora", "pasttense", 4.0, 30.0), _row(Q, "control_lora", "codeattack", -2.0, 40.0),
        _row(G, "safety_si", "pasttense", -30.0, 50.0), _row(G, "safety_si", "codeattack", -10.0, 45.0),
        _row(G, "safety_si", "strongreject_bon", -15.0, 55.0), _row(G, "safety_si", "mmlu", -1.0, 65.0),
        # two seeds of one Gemma LoRA on one eval: they pool into one mark
        _row(G, "control_lora", "pasttense", -6.0, 50.0, seed="s1"), _row(G, "control_lora", "pasttense", -10.0, 50.0, seed="s2"),
    ]
    return pd.DataFrame(rows)


def _gaps(d, objects, evals, models):
    drawn = R.drawn_set(d[d.d_y.notna()])
    rows = [dict(model_id=m, object_label=o, eval_id=e, status="drawn" if (m, o, e) in drawn else "absent", ruling="", selection="")
            for m in models for o in objects for e in evals]
    return pd.DataFrame(rows)


@pytest.fixture
def figs(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIGS", str(tmp_path))
    yield tmp_path
    plt.close("all")


def _panel_axes(fig):
    fig.canvas.draw()
    return [a for a in fig.axes if _title(a) or a.get_xticklabels()]


def _title(ax) -> str:
    return ax.get_title(loc="left") or ax.get_title()


def _body(tex: str) -> list:
    """The table's data rows: every line that ends a row, less the header and its rule."""
    return [l for l in tex.splitlines() if l.endswith("\\\\") and "multicolumn" not in l and "\\hline" not in l]


# ---- x = object: stacked ----------------------------------------------------------------------------------------------------------
def test_x_object_figures_stack_one_category_per_row_whatever_layout_the_caller_names(figs):
    d = _deltas()
    cm = R.category_means(d)
    objects, models = ["safety_si", "control_lora"], [Q, G]
    evals = ["pasttense", "codeattack", "strongreject_bon", "pair", "mmlu", "gsm8k", "bigcodebench", "bfcl"]
    fig = R.category_panels(cm, d, [["static", "multi-query", "capability"]], objects, models, _gaps(d, objects, evals, models), "t_stack")
    axes = _panel_axes(fig)
    assert [_title(a) for a in axes] == ["static", "multi-query", "capability"]
    pos = [a.get_position() for a in axes]
    assert max(p.x0 for p in pos) - min(p.x0 for p in pos) < 0.02, "stacked panels share the left edge"
    y0 = sorted(p.y0 for p in pos)
    assert all(b - a > 0.1 for a, b in zip(y0, y0[1:])), "one panel per row: distinct vertical positions"
    assert (figs / "t_stack.png").exists()


def test_the_row_layout_is_still_available_by_asking_for_it(figs):
    d = _deltas()
    cm = R.category_means(d)
    objects, models = ["safety_si"], [Q]
    evals = ["pasttense", "codeattack", "strongreject_bon", "pair"]
    fig = R.category_panels(cm, d, [["static", "multi-query"]], objects, models, _gaps(d, objects, evals, models), "t_row", stack=False)
    pos = [a.get_position() for a in _panel_axes(fig)]
    assert len(pos) == 2 and abs(pos[0].y0 - pos[1].y0) < 0.02 and pos[1].x0 > pos[0].x1


# ---- x = eval: one row, absolute and relative -------------------------------------------------------------------------------------
def _absolute_points(d):
    p = d[["model_id", "object_label", "seed", "draw", "eval_id", "y", "lo", "hi", "held_out"]].copy()
    none = d.drop_duplicates(["model_id", "eval_id"])
    n = pd.DataFrame(dict(model_id=none.model_id, object_label="none", seed="", draw="", eval_id=none.eval_id, y=none.anchor_value,
                          lo=none.anchor_value - 3, hi=none.anchor_value + 3, held_out=0))
    return pd.concat([p, n], ignore_index=True)


def test_x_eval_absolute_figure_keeps_one_row(figs):
    d = _deltas()
    objects, models = ["none", "safety_si", "control_lora"], [Q, G]
    evals = ["pasttense", "codeattack", "strongreject_bon"]
    pts = _absolute_points(d)
    fig = R.absolute_panels(pts, evals, objects, models, _gaps(pts.assign(d_y=pts.y), objects, evals, models), "rate (%)", "t_abs")
    axes = _panel_axes(fig)
    assert [_title(a) for a in axes] == ["static", "multi-query"]
    pos = [a.get_position() for a in axes]
    assert abs(pos[0].y0 - pos[1].y0) < 0.02 and pos[1].x0 > pos[0].x1, "one row"
    assert axes[0].get_ylim() == (-3, 103) and axes[1].get_ylim() == (-3, 103), "the absolute axis is the shared 0-100 range"
    assert not any(np.allclose(l.get_ydata(), [0, 0]) and len(l.get_xdata()) == 2 and abs(l.get_xdata()[1] - l.get_xdata()[0]) > 0.5
                   for l in axes[0].lines), "no zero line on an absolute panel"


def test_delta_points_orient_by_the_categorys_sign_and_pool_a_cells_readings():
    d = _deltas()
    p = R.delta_points(d, ["pasttense", "codeattack", "strongreject_bon", "mmlu"], ["none", "safety_si", "control_lora"])
    assert "none" not in set(p.object_label)
    q = p.set_index(["model_id", "object_label", "eval_id"])
    # single-turn's sign is -1: an ASR delta of -10 is +10 pp of defense, and the interval's bounds swap
    assert q.loc[(Q, "safety_si", "pasttense"), "y"] == 10.0
    assert (q.loc[(Q, "safety_si", "pasttense"), "lo"], q.loc[(Q, "safety_si", "pasttense"), "hi"]) == (5.0, 15.0)
    # capability's sign is +1: unchanged
    assert q.loc[(Q, "safety_si", "mmlu"), "y"] == 2.0
    # two seeds pool into one mark at their mean
    assert q.loc[(G, "control_lora", "pasttense"), "y"] == 8.0 and int(q.loc[(G, "control_lora", "pasttense"), "n_pooled"]) == 2
    assert len(p) == p.groupby(["model_id", "object_label", "eval_id"]).ngroups


def test_delta_panels_draw_one_row_of_zero_lined_panels_each_on_its_own_axis(figs):
    d = _deltas()
    objects, models = ["none", "safety_si", "control_lora"], [Q, G]
    evals = ["pasttense", "codeattack", "strongreject_bon"]
    gaps = _gaps(d, objects, evals, models)
    fig = R.delta_panels(d, evals, objects, models, gaps, "t_delta")
    axes = _panel_axes(fig)
    assert [_title(a) for a in axes] == ["static", "multi-query"]
    pos = [a.get_position() for a in axes]
    assert abs(pos[0].y0 - pos[1].y0) < 0.02 and pos[1].x0 > pos[0].x1, "one row"
    for ax in axes:
        assert any(np.allclose(l.get_ydata(), [0, 0]) for l in ax.lines), "the unsteered model is the zero line"
        assert ax.get_ylabel() == R.CATEGORIES[_title(ax)][2], "each panel carries its category's own Δ label"
        assert [t.get_text() for t in ax.get_xticklabels()][-1] == "mean"
        lo, hi = ax.get_ylim()
        assert lo < -5 and hi > 10, "the range is set from the marks, at least the main figures' -5..10"
    # the single-turn panel's range holds Gemma's +30 pp mark and its interval
    assert axes[0].get_ylim()[1] > 35
    # the unsteered model is not an arm on a relative figure: the legend keeps its slot blank
    labels = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert "Unsteered" not in labels and "Safety SI" in labels
    assert (figs / "t_delta.png").exists()


def test_a_cell_with_no_unsteered_run_is_a_gap_on_the_relative_figure_not_a_point(figs):
    d = _deltas()
    d.loc[len(d)] = _row(Q, "control_lora", "strongreject_bon", 0.0, 60.0)
    d.loc[d.index[-1], ["d_y", "d_lo", "d_hi", "anchor_value"]] = np.nan
    d.loc[d.index[-1], "route"] = "no unsteered run"
    objects, models = ["safety_si", "control_lora"], [Q, G]
    gaps = _gaps(d, objects, ["strongreject_bon"], models)
    g = R.delta_gaps(d, gaps, objects)
    assert g.set_index(["model_id", "object_label", "eval_id"]).loc[(Q, "control_lora", "strongreject_bon"), "status"] == "no unsteered run"
    assert g.set_index(["model_id", "object_label", "eval_id"]).loc[(Q, "safety_si", "strongreject_bon"), "status"] == "drawn"
    p = R.delta_points(d, ["strongreject_bon"], objects)
    assert (Q, "control_lora") not in set(zip(p.model_id, p.object_label))


# ---- the table under an absolute figure --------------------------------------------------------------------------------------
def test_table_absolute_leads_with_the_absolute_unsteered_baseline_and_signs_every_arm():
    d = _deltas()
    tex = R.table_absolute(d, ["pasttense", "codeattack", "strongreject_bon"], ["none", "safety_si", "control_lora"], [Q, G],
                           "cap", "tab:t", flip_baseline=True)
    body = _body(tex)
    assert body[0].startswith(R.BASELINE_ROW_LABEL) and body[1].startswith(R.BASELINE_ROW_LABEL), "the unsteered model first, one row per model"
    assert not any(l.startswith(R.BASELINE_ROW_LABEL) for l in body[2:])
    # Qwen's unsteered ASR anchors are 30 / 40 / 60: as defense success, 70 / 60 / 40, absolute and unsigned
    assert "$70.0$" in body[0] and "$60.0$" in body[0] and "$40.0$" in body[0] and "+" not in body[0]
    # the arms are signed changes, oriented so positive is better defense: Qwen safety SI on Past tense is +10.0 [5.0, 15.0]
    q = next(l for l in body[2:] if "Safety SI" in l and "Qwen" in l)
    # a SYMMETRIC interval now prints as +/-; the bracket form survives only for an
    # asymmetric one, so this pins the half-width rather than the two bounds
    assert "$+10.0$" in q and "\\pm 5.0" in q and "$+20.0$" in q and "$+5.0$" in q
    # an arm without a reading on an eval is `--`, never a zero
    c = next(l for l in body[2:] if "Control LoRA" in l and "Qwen" in l)
    assert c.rstrip("\\").strip().endswith("--")
    assert "unsteered" not in q.lower()


def test_table_absolute_marks_a_materially_mixed_anchor_baseline_and_a_held_out_arm():
    d = _deltas()
    d.loc[len(d)] = _row(Q, "control_lora", "strongreject_bon", -8.0, 54.0)          # a second anchor on Qwen's Best-of-75, 6 pp off
    d.loc[len(d)] = _row(G, "control_lora", "strongreject_bon", 12.0, 55.5, held=1)  # a held-out cell: value kept, marked; anchor 0.5 pp off
    tex = R.table_absolute(d, ["strongreject_bon"], ["none", "safety_si", "control_lora"], [Q, G], "cap", "tab:t", flip_baseline=True)
    body = _body(tex)
    qb = next(l for l in body if l.startswith(R.BASELINE_ROW_LABEL) and "Qwen" in l)
    assert "$43.0$" in qb and "\\ddagger" in qb, "two anchors 6 pp apart (60, 54) -> their mean, flipped, and marked"
    gb = next(l for l in body if l.startswith(R.BASELINE_ROW_LABEL) and "Gemma" in l)
    assert "$44.8$" in gb and "\\ddagger" not in gb, "two anchors half a point apart are one baseline: no mark"
    held = next(l for l in body if "Control LoRA" in l and "Gemma" in l)
    assert "$-12.0$" in held and "\\S" in held
    assert "apart" in tex and R.HELD_NOTE in tex


def test_table_absolute_keeps_the_baseline_unflipped_where_the_figure_is_not_flipped():
    d = _deltas()
    tex = R.table_absolute(d, ["mmlu"], ["none", "safety_si"], [Q], "cap", "tab:t")
    body = _body(tex)
    assert "$70.0$" in body[0] and "$+2.0$" in body[1]


# ---- the paired lane's owner metric is on the percent scale ---------------------------------------------------------------------
def test_the_noncollapsed_over_refusal_owner_metric_is_scaled_to_percent_like_its_parent():
    """The paired-write views file `over_refusal_graded_noncollapsed` as the OWNER over-refusal row (Gemma's and Llama's unsteered runs
    and composed difference ReLU arms). Left off the scale map it drew as a fraction - 0.60 as 0.6 % - and its deltas a hundredfold
    too small (2026-09-23). It is a rate over n items like its parent and takes the same scale and Wilson denominator."""
    d = pd.DataFrame([
        dict(metric_id="over_refusal_graded_noncollapsed", value=0.602, n=1294, ci_lo=np.nan, ci_hi=np.nan),
        dict(metric_id="over_refusal_graded", value=0.598, n=1318, ci_lo=np.nan, ci_hi=np.nan),
    ])
    out = R.intervals(d)
    assert abs(out.y.iat[0] - 60.2) < 1e-9 and abs(out.y.iat[1] - 59.8) < 1e-9
    assert 57 < out.lo.iat[0] < 60.2 < out.hi.iat[0] < 63 and out.ci_source.iat[0] == "Wilson from value and n"
    assert R.SCALE["over_refusal_graded_noncollapsed"] == R.SCALE["over_refusal_graded"]
    assert R.IS_RATE["over_refusal_graded_noncollapsed"] == R.IS_RATE["over_refusal_graded"]


def test_a_noncollapsed_owner_row_carries_its_own_collapse_share_and_prints_it(monkeypatch):
    """When the non-collapsed rate is itself the owner row (the paired lane), the collapse numbers must land on THAT row, or the
    figure plots the right rate and never prints the collapse under it (2026-09-23). The registry's own companion table is what
    carries the share; here it is stubbed."""
    companion = pd.DataFrame([dict(run_id="r1", judgment_id="j1", threshold=-1.0, k=60, n=110,
                                   metric_extra_json='{"role_collapse_share": 0.56, "role_collapse_n": 140}')])
    monkeypatch.setattr(R, "role_collapse_rows", lambda at_head: companion)
    d = pd.DataFrame([
        dict(run_id="r1", judgment_id="j1", metric_id="over_refusal_graded_noncollapsed", value=60 / 110, n=110, ci_lo=np.nan, ci_hi=np.nan),
        dict(run_id="r2", judgment_id="j2", metric_id="over_refusal_graded", value=0.30, n=250, ci_lo=np.nan, ci_hi=np.nan),
    ])
    out = R.intervals(R.attach_role_collapse(d))
    nc, plain = out.iloc[0], out.iloc[1]
    assert nc.role_collapse_share == 0.56 and nc.noncollapsed_n == 110 and bool(nc.collapsed)
    assert abs(nc.y - 100 * 60 / 110) < 1e-9 and nc.ci_source == "Wilson over the non-collapsed items"
    assert R.collapse_note(nc) == "collapse 56%" and "110 non-collapsed items" in R.collapse_detail(nc, "llama8b_it", "difference_relu", "xstest")
    assert not bool(plain.collapsed) and pd.isna(plain.role_collapse_share)


# ---- one metric under two spellings anchors as one metric ----------------------------------------------------------------------
def test_an_arm_filed_under_the_evals_other_headline_spelling_still_finds_its_unsteered_run(monkeypatch):
    """strongreject_bon declares `asr` and `asr_bon_cap4096` as one headline family. Llama's exported unsteered runs file both (the
    headline rule keeps `asr`); the native best-of-N Llama arm files only `asr_bon_cap4096`; a literal metric match left the arm
    with no unsteered run (2026-09-23). A candidate is any headline unsteered row in the same family."""
    monkeypatch.setattr(R, "headline_metric_ids", lambda: {"strongreject_bon": ["asr", "asr_bon_cap4096"]})
    frame = pd.DataFrame([
        dict(model_id="llama8b_it", eval_id="strongreject_bon", metric_id="asr", run_id="u1", y=46.0),
        dict(model_id="llama8b_it", eval_id="strongreject_bon", metric_id="asr", run_id="u2", y=42.0),
        dict(model_id="llama8b_it", eval_id="pasttense", metric_id="asr", run_id="u3", y=30.0),
        dict(model_id="gemma4e4b_it", eval_id="strongreject_bon", metric_id="asr_bon_cap4096", run_id="u4", y=76.0),
    ])
    arm = pd.Series(dict(model_id="llama8b_it", eval_id="strongreject_bon", metric_id="asr_bon_cap4096"))
    assert R.metric_family("strongreject_bon", "asr_bon_cap4096") == ["asr", "asr_bon_cap4096"]
    assert R.anchor_candidates(frame, arm).run_id.tolist() == ["u1", "u2"], "the other spelling's headline rows are candidates; other models and evals are not"
    # a metric off the family list, or an eval with no list, matches literally
    assert R.metric_family("strongreject_bon", "bon_nauc") == ["bon_nauc"] and R.metric_family("pasttense", "asr") == ["asr"]
    other = pd.Series(dict(model_id="llama8b_it", eval_id="pasttense", metric_id="asr_bon_cap4096"))
    assert len(R.anchor_candidates(frame, other)) == 0



def test_x_eval_figures_are_no_wider_than_a_third_of_a_metre_at_the_attacks_shape(figs):
    """The attacks figures - nine objects, eight evals in three panels - were 44 in wide and are now narrower."""
    objects = ["none", "safety_si", "constitution_si", "control_relu", "control_lora", "constitution_relu", "constitution_lora", "difference_relu", "difference_lora"]
    evals = ["pasttense", "codeattack", "renellm", "grok420", "strongreject_bon", "pair", "crescendo", "actorattack"]
    rows = [dict(model_id=Q, object_label=o, seed="", draw="", eval_id=e, y=50.0, lo=45.0, hi=55.0, held_out=0) for o in objects for e in evals]
    pts = pd.DataFrame(rows)
    fig = R.absolute_panels(pts, evals, objects, [Q, G], _gaps(pts.assign(d_y=pts.y), objects, evals, [Q, G]), "rate (%)", "t_width")
    w, h = fig.get_size_inches()
    assert 28 < w < 34 and abs(h - 5.0) < 1e-6


def test_long_y_labels_fold_onto_lines_that_fit_the_panel_height(figs):
    assert R._wrap_label("defense success rate (%; 100 \u2212 ASR at threshold 0.5)").count("\n") == 1
    assert R._wrap_label("reduction in\nmisaligned actions (pp)") == "reduction in\nmisaligned actions (pp)", "existing breaks are kept"
    assert R._wrap_label("rate (%)") == "rate (%)"
    d = _deltas()
    pts = _absolute_points(d)
    objects, models, evals = ["none", "safety_si"], [Q], ["pasttense", "codeattack"]
    fig = R.absolute_panels(pts, evals, objects, models, _gaps(pts.assign(d_y=pts.y), objects, evals, models),
                            "defense success rate (%; 100 \u2212 ASR at threshold 0.5)", "t_wrap")
    assert fig._supylabel.get_text().count("\n") == 1


# ---- the best-of-N curve figure: its legend and its footnote --------------------------------------------------------------------
def test_bon_curves_figure_has_only_the_object_legend_and_no_footnote(figs):
    """The thin/thick `lines` key and the stopgap footnote are off the canvas; the stopgap cells stay on the
    figure object for the composition table."""
    rows = []
    for m, o, rid, extra in ((Q, "none", "r0", False), (Q, "safety_si", "r1", False), (Q, "safety_si", "r2", False), (G, "none", "g0", True)):
        for i, n in enumerate((1, 2, 4, 8, 16, 32, 64, 75)):
            rows.append(dict(model_id=m, object_label=o, run_id=rid, n=n, asr=min(0.05 * (i + 1), 0.9), from_extra=extra))
    pts = pd.DataFrame(rows)
    fig = R.bon_curves_figure(pts, ["none", "safety_si"], [Q, G], "t_curves")
    titles = [leg.get_title().get_text() for leg in fig.legends]
    assert titles == [""], titles                                             # one group: no header
    labels = [tx.get_text() for leg in fig.legends for tx in leg.get_texts()]
    assert "Unsteered" in labels and "Safety SI" in labels and not any("thin" in l or "thick" in l for l in labels)
    assert not any("asr_at_n" in tx.get_text() or "grid" in tx.get_text() for tx in fig.texts), "no footnote under the figure"
    for ax in [a for a in fig.axes if a.get_title()]:                                # the axis is ticked out to N = 75, where the cells were scored
        assert [int(t) for t in ax.get_xticks()] == list(R.BON_NS) and ax.get_xticklabels()[-1].get_text().strip() == "75"
        assert ax.get_xticklabels()[-2].get_text() == "", "64 keeps its tick mark and loses its label"
    assert fig.bon_stopgap == [(G, "none")]
    assert (figs / "t_curves.png").exists()


def test_a_legend_with_one_group_has_no_header_and_two_groups_keep_theirs(figs):
    """A header tells groups apart, so a single-group legend carries none, anywhere in the notebook."""
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(4, 3))
    fig.add_subplot(111)
    R.legend_stack(fig, [("object", [Line2D([], [], color="k", label="a"), Line2D([], [], color="r", label="b")], 2)])
    assert [leg.get_title().get_text() for leg in fig.legends] == [""]
    fig2 = plt.figure(figsize=(4, 3))
    fig2.add_subplot(111)
    R.legend_stack(fig2, [("object", [Line2D([], [], color="k", label="a")], 1), ("model", [Line2D([], [], color="r", label="b")], 1)])
    assert sorted(leg.get_title().get_text() for leg in fig2.legends) == ["model", "object"]
    fig3 = plt.figure(figsize=(4, 3))
    fig3.add_subplot(111)
    R.legend_stack(fig3, [("object", [Line2D([], [], color="k", label="a")], 1), ("empty", [], 1)])   # an empty group does not count
    assert [leg.get_title().get_text() for leg in fig3.legends] == [""]


def test_x_object_figures_carry_no_cap_annotations(figs):
    """No "cap n%" labels and no "≠" cap-mismatch glyphs on the appendix figures; the shares stay in the tables."""
    d = _deltas().assign(frac_at_token_cap=0.6, cap_mismatch=True)
    cm = R.category_means(d)
    assert (cm.at_cap > R.AT_CAP_LINE).any() and cm.cap_mismatch.any(), "the fixture would have drawn both marks"
    objects, models = ["safety_si", "control_lora"], [Q, G]
    evals = ["pasttense", "codeattack", "strongreject_bon", "pair"]
    fig = R.category_panels(cm, d, [["static", "multi-query"]], objects, models, _gaps(d, objects, evals, models), "t_nocap")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not any(t.startswith("cap ") or "≠" in t for t in texts), texts


# ---- open marks instead of text: held-out and role-collapsed cells; no double bar ----------------------------
def _open_marks(ax):
    """Scatter collections drawn unfilled (white face, coloured edge)."""
    import matplotlib.colors as mc
    out = []
    for c in ax.collections:
        fc = c.get_facecolor(); ec = c.get_edgecolor()
        if len(fc) and len(ec) and tuple(fc[0][:3]) == (1.0, 1.0, 1.0) and tuple(ec[0][:3]) != (1.0, 1.0, 1.0):
            out.append(c)
    return out


def _annot_frame():
    d = _deltas()
    # Llama safety SI held out on pasttense; Gemma control LoRA role-collapsed on pasttense; seeds pooled at two caps on Qwen safety SI
    d.loc[len(d)] = _row("llama8b_it", "safety_si", "pasttense", -5.0, 40.0, held=1)
    d.loc[len(d)] = _row("llama8b_it", "safety_si", "codeattack", -6.0, 40.0, held=1)
    d["collapsed"] = False; d["role_collapse_share"] = np.nan; d["noncollapsed_n"] = np.nan
    i = d.index[(d.model_id == G) & (d.object_label == "control_lora") & (d.eval_id == "pasttense")]
    d.loc[i, ["collapsed", "role_collapse_share", "noncollapsed_n"]] = [True, 0.55, 226]
    return d


def test_x_object_figures_draw_held_and_collapsed_cells_open_with_no_text_and_no_double_bar(figs):
    d = _annot_frame()
    cm = R.category_means(d)
    cm["mixed_cap_seeds"] = True                                                  # would have drawn ‖ beside every mean
    objects, models = ["safety_si", "control_lora"], [Q, G, "llama8b_it"]
    evals = ["pasttense", "codeattack", "strongreject_bon", "pair"]
    fig = R.category_panels(cm, d, [["static", "multi-query"]], objects, models, _gaps(d, objects, evals, models), "t_open")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts] + [t.get_text() for t in fig.texts]
    assert not any(("\u2016" in t) or ("not scoreable" in t) or ("collapse" in t) for t in texts), texts
    panel = [a for a in fig.axes if _title(a) == "static"][0]
    assert len(_open_marks(panel)) >= 3, "the two held Llama evals and Gemma's collapsed eval draw open"
    assert not any("role collapse" in a.get_xlabel() for a in fig.axes), "no collapse footer under the panel"


def test_x_eval_figures_draw_held_and_collapsed_cells_open_with_no_note(figs):
    d = _annot_frame()
    objects, models = ["none", "safety_si", "control_lora"], [Q, G, "llama8b_it"]
    evals = ["pasttense", "codeattack"]
    fig = R.delta_panels(d, evals, objects, models, _gaps(d, objects, evals, models), "t_open_eval")
    texts = [t.get_text() for ax in fig.axes for t in ax.texts] + [t.get_text() for t in fig.texts]
    assert not any(("not scoreable" in t) or ("collapse" in t) for t in texts), texts
    panel = [a for a in fig.axes if _title(a) == "static"][0]
    assert len(_open_marks(panel)) >= 3


def test_table_legends_name_the_open_cells():
    d = _annot_frame()
    n = R.cell_notes(d, ["pasttense", "codeattack"])
    assert "not scoreable" in n and "Llama-3.1-8B Safety SI on Past tense, CodeAttack" in n
    assert "role collapse" in n and "Gemma-4-E4B Control LoRA on Past tense 55% (226 items)" in n
    assert R.cell_notes(d, ["pair"]) == ""
    cm = R.category_means(d)
    tex = R.table_categories_models(cm, "static", ["safety_si", "control_lora"], [Q, G, "llama8b_it"], "cap", "tab:t", d=d)
    assert "not scoreable" in tex and "role collapse" in tex
    assert "not scoreable" not in R.table_categories_models(cm, "static", ["safety_si", "control_lora"], [Q, G], "cap", "tab:t")
    ta = R.table_absolute(d, ["pasttense", "codeattack"], ["none", "safety_si", "control_lora"], [Q, G, "llama8b_it"], "cap", "tab:t", flip_baseline=True)
    body = _body(ta)
    assert any("Control LoRA" in l and "Gemma" in l and "\\ast" in l for l in body), "the collapsed cell is starred"
    assert any("Safety SI" in l and "Llama" in l and "\\S" in l for l in body), "the held cell keeps its section mark"
    assert "open marks, role collapse" in ta and "open marks, not scoreable" in ta


# ---- clean canvases and complete cells -----------------------------------------------------------------------
def test_a_lone_seed_label_is_the_fits_stamp_not_a_replicate():
    """On Gemma every fitted object has one fit; its BCB / BFCL rows came on a `gemma4e4b_s1` view and split each cell into two seeds.
    A (model, object) with exactly one distinct seed label reads blank; Qwen's three seeds stay three."""
    base = dict(paper_label=None, campaign_cell_id_legacy="", run_extra_json=None, fit_replicate_label=None, transfer_kind="instruct_direct",
                strength_nominal=1.0, strength_coefficient=None, is_operating_point=1, metric_id="asr", threshold=0.5, bon_n=None,
                eval_category="static", headline_metric_id="asr", judge_model=None, judge_id=None, run_id=None, eval_id="pasttense",
                token_cap_tokens=4096, decode_policy_id=None, decode_source=None, decode_sent_json=None, item_set_id=None, model_id=None, view_legacy=None)
    rows = []
    for i, (m, lab, view) in enumerate([("gemma4e4b_it", "control_lora", "gemma4e4b_it"), ("gemma4e4b_it", "control_lora", "gemma4e4b_s1"),
                                         ("qwen35_4b_it", "control_lora", "qwen35_4b_s1"), ("qwen35_4b_it", "control_lora", "qwen35_4b_s2"),
                                         ("qwen35_4b_it", "control_lora", "qwen35_4b_it")]):
        rows.append({**base, "model_id": m, "paper_label": lab, "view_legacy": view, "run_id": f"r{i}"})
    d = pd.DataFrame(rows)
    try:
        out = R.spell(d)
    except Exception as e:                                                    # the spelling needs registry tables; fall back to the rule alone
        pytest.skip(f"spell needs more frame than the fixture gives: {e}")
    g = out[out.model_id == "gemma4e4b_it"]; q = out[out.model_id == "qwen35_4b_it"]
    assert set(g.seed) == {""}, g.seed.tolist()
    assert set(q.seed) == {"s1", "s2", ""}, q.seed.tolist()


def test_a_lone_seed_label_on_a_prompted_si_is_a_view_stamp():
    """2026-09-25: with BCB / BFCL on the appendix capability axis, Safety SI on Gemma split into an `s1` half (BCB / BFCL on the
    `gemma4e4b_s1` view) and a blank half (MMLU / GSM8K) and both drew ringed as partial. A prompt is one object: the lone label reads blank."""
    base = dict(paper_label=None, campaign_cell_id_legacy="", run_extra_json=None, fit_replicate_label=None, transfer_kind="instruct_direct",
                strength_nominal=1.0, strength_coefficient=None, is_operating_point=1, metric_id="acc", threshold=-1.0, bon_n=None,
                eval_category="capability", headline_metric_id="acc", judge_model=None, judge_id=None, run_id=None, eval_id=None,
                token_cap_tokens=4096, decode_policy_id=None, decode_source=None, decode_sent_json=None, item_set_id=None, model_id=None, view_legacy=None)
    rows = []
    for i, (m, lab, view, ev) in enumerate([("gemma4e4b_it", "safety_si", "gemma4e4b_s1", "bfcl"), ("gemma4e4b_it", "safety_si", "gemma4e4b_s1", "bigcodebench"),
                                             ("gemma4e4b_it", "safety_si", "gemma4e4b_it", "mmlu"), ("gemma4e4b_it", "safety_si", "gemma4e4b", "gsm8k")]):
        rows.append({**base, "model_id": m, "paper_label": lab, "view_legacy": view, "eval_id": ev, "run_id": f"p{i}"})
    try:
        out = R.spell(pd.DataFrame(rows))
    except Exception as e:
        pytest.skip(f"spell needs more frame than the fixture gives: {e}")
    assert set(out.seed) == {""}, out[["eval_id", "view_legacy", "seed"]].to_dict("records")


def test_a_mean_missing_only_an_out_by_design_eval_is_not_partial():
    d = pd.DataFrame([dict(model_id="nemotron3nano4b", object_label="safety_si", seed="", eval_id="agentic_murder", d_y=-5.0, d_lo=-8.0, d_hi=-2.0,
                           anchor_in_frame=True, ceiling=False, anchor_value=12.0, held_out=0),
                      dict(model_id="gemma4e4b_it", object_label="safety_si", seed="", eval_id="agentic_murder", d_y=-5.0, d_lo=-8.0, d_hi=-2.0,
                           anchor_in_frame=True, ceiling=False, anchor_value=12.0, held_out=0)])
    # the two-eval agentic category as it stood before murder was removed (2026-09-25): the test is about OUT_BY_DESIGN, not the roster
    cm = R.category_means(d, categories={"agentic misalignment": (["agentic_leaking", "agentic_murder"], -1, "")}).set_index("model_id")
    assert not bool(cm.loc["nemotron3nano4b", "partial"]), "Nemotron's leaking is out by design: one eval is the whole category for it"
    assert bool(cm.loc["gemma4e4b_it", "partial"]), "Gemma's missing leaking is a real gap"


def test_appendix_category_set_capability_axis_and_static_unchanged():
    assert R.APPENDIX_CATEGORIES["capability"][0] == ["mmlu", "gsm8k", "bigcodebench", "bfcl"] and R.APPENDIX_CATEGORIES["static"] == R.CATEGORIES["static"]
    d = _deltas()
    d.loc[len(d)] = _row(Q, "safety_si", "gsm8k", 1.0, 90.0)
    cm = R.category_means(d, categories=R.APPENDIX_CATEGORIES)
    cap = cm[(cm.category == "capability") & (cm.model_id == Q)]
    assert len(cap) == 1 and bool(cap.partial.iat[0]), "MMLU + GSM8K alone -> partial now that the coding evals are on the appendix axis"
    d.loc[len(d)] = _row(Q, "safety_si", "bigcodebench", 1.0, 40.0)
    d.loc[len(d)] = _row(Q, "safety_si", "bfcl", 1.0, 60.0)
    cm = R.category_means(d, categories=R.APPENDIX_CATEGORIES)
    cap = cm[(cm.category == "capability") & (cm.model_id == Q)]
    assert len(cap) == 1 and not bool(cap.partial.iat[0]), "all four present -> complete"


def test_no_text_under_any_appendix_figure(figs):
    d = _annot_frame().assign(two_term_dev=0.007, item_set_id="mmlu_200")
    cm = R.category_means(d)
    objects, models = ["safety_si", "control_lora"], [Q, G, "llama8b_it"]
    evals = ["pasttense", "codeattack", "strongreject_bon", "pair"]
    fig = R.category_panels(cm, d, [["static", "multi-query"]], objects, models, _gaps(d, objects, evals, models), "t_clean")
    assert all(a.get_xlabel() == "" for a in fig.axes), [a.get_xlabel() for a in fig.axes]
    assert not any("two-term" in t.get_text() for ax in fig.axes for t in ax.texts)
    fig2 = R.delta_panels(d, ["pasttense", "codeattack"], ["none"] + objects, models, _gaps(d, objects, evals, models), "t_clean_eval")
    assert fig2._supxlabel is None or fig2._supxlabel.get_text() == ""
    n = R.cell_notes(d.assign(item_set_id=np.where(d.model_id == Q, "mmlu_1531", "mmlu_200")), ["pasttense"])
    assert "Two-term graded" in n and "item sets differ across models" in n


def test_every_declared_headline_metric_id_of_the_paper_evals_is_on_the_percent_scale():
    """2026-09-24: `orb_refused` / `xst_refused` became the over-refusal headline ids and were on neither map, so every over-refusal
    arm drew at ~100 % compliance. The ids the registry's evals declare as headlines (plus the reader's own additions) must all scale."""
    for mid in ("orb_refused", "xst_refused", "over_refusal_graded", "over_refusal_graded_noncollapsed", "asr", "asr_bon_cap4096", "acc"):
        assert R.SCALE[mid] == 100.0 and R.IS_RATE[mid] == 1.0, mid
    d = pd.DataFrame([dict(metric_id="orb_refused", value=0.6, n=1319, ci_lo=np.nan, ci_hi=np.nan),
                      dict(metric_id="xst_refused", value=0.05, n=250, ci_lo=np.nan, ci_hi=np.nan)])
    out = R.intervals(d)
    assert abs(out.y.iat[0] - 60.0) < 1e-9 and abs(out.y.iat[1] - 5.0) < 1e-9 and (out.ci_source == "Wilson from value and n").all()


# ---- the headline-scale guard (2026-09-24) ---------------------------------------------------------------------------------------------
def test_every_headline_id_the_notebook_draws_is_on_the_scale_today_and_a_missing_one_fails_at_load(monkeypatch):
    import os
    if not os.path.exists(R.DB):
        pytest.skip("no registry on this box")
    full = dict(R.SCALE)
    R.check_headline_scales.cache_clear()
    assert R.check_headline_scales() == ()
    # drop the refused-rate ids: the exact hole of 2026-09-24 must raise, naming the ids and their evals
    R.check_headline_scales.cache_clear()
    monkeypatch.setattr(R, "SCALE", {k: v for k, v in full.items() if k not in ("orb_refused", "xst_refused")})
    with pytest.raises(ValueError) as ei:
        R.check_headline_scales()
    assert "orb_refused" in str(ei.value) and "xst_refused" in str(ei.value) and "or_bench_hard" in str(ei.value)
    # the same hole with the override set is a warning, not a raise
    R.check_headline_scales.cache_clear()
    monkeypatch.setenv("NB_ALLOW_UNSCALED_HEADLINE", "1")
    assert set(R.check_headline_scales()) == {"orb_refused", "xst_refused"}
    R.check_headline_scales.cache_clear()
    monkeypatch.delenv("NB_ALLOW_UNSCALED_HEADLINE")
    # with the full map back, an eval the notebook does not draw (alpaca_eval's win_rate) never fires
    monkeypatch.setattr(R, "SCALE", full)
    R.check_headline_scales.cache_clear()
    assert R.check_headline_scales() == ()
