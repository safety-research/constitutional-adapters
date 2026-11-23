"""appendix_common: the paired comparison figure the appendix ports share. Synthetic rows only; no registry, no committed data.

Pinned here:
  * the tidy table's contract: the refusals (a missing column, an empty key, an object mixing seeded and unseeded rows, a repeated mark,
    an interval that does not hold its point or has one bound, a (category, eval) the categories do not declare, a non-boolean `open`)
    and the orientation by the category's sign; an unseeded object's one mark per cell at its slot's centre;
  * the mean marks: the seed means and the seed-pooled mean equal the notebook's rule written out here independently (within seed
    se = sqrt(sum se_e^2) / k; over seeds se = sqrt(se_within^2 + se_between^2)), and the drawn mean marks sit at those values;
  * the canvas: one mark per (object, seed, eval, condition) and per (object, condition) mean, each condition's shape (circle, diamond,
    square), each object's `R.COLOR`, the conditions left to right within a seed, the joining line through a seed's marks;
  * the mean rule: a pending reading draws the gap glyph and keeps its condition's mean off under "complete", ringed under "ring", and
    `not_drawn` names it;
  * the legend (objects two per column, the conditions, then 95% CI), the layouts (side by side with a shared y, stacked on one x
    scale), the width cap, and `save_figure`'s three formats in every target directory."""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402

import registry_nb as R  # noqa: E402
import appendix_common as AC  # noqa: E402

OBJS = ("control_lora", "difference_lora")
SEEDS = ("s1", "s2", "s3")
MT = {"multi-turn": R.CATEGORIES["multi-turn"]}          # sign -1: the rows below hold attack-success deltas
EVALS = list(MT["multi-turn"][0])


def _rows(conditions=("published", "control_pool"), objects=OBJS, category="multi-turn", evals=None) -> pd.DataFrame:
    """Deterministic attack-success deltas (negative = more defense) with asymmetric intervals, one per (object, seed, eval, condition)."""
    out = []
    for oi, o in enumerate(objects):
        for si, sd in enumerate(SEEDS):
            for ei, e in enumerate(evals or EVALS):
                for ci, c in enumerate(conditions):
                    v = -(5.0 + 20.0 * oi + 3.0 * si + 4.0 * ei - 2.0 * ci)
                    out.append(dict(object=o, seed=sd, category=category, eval=e, condition=c, delta=v, ci_lo=v - 2.0 - si - ci,
                                    ci_hi=v + 3.0 + ei, n=100 + si))
    return pd.DataFrame(out)


def _gid_marks(fig) -> dict:
    """{gid: (x, y, marker vertices, facecolor, size)} of every tagged scatter on every axis."""
    out = {}
    for ax in fig.axes:
        for coll in ax.collections:
            gid = coll.get_gid()
            if gid:
                (x, y), = coll.get_offsets()
                face = tuple(coll.get_facecolor()[0]) if len(coll.get_facecolor()) else None
                out[gid] = (float(x), float(y), coll.get_paths()[0].vertices, face, float(coll.get_sizes()[0]))
    return out


def _marker_vertices(m: str) -> np.ndarray:
    ms = MarkerStyle(m)
    return ms.get_path().transformed(ms.get_transform()).vertices


def _legend_columns(fig) -> list:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    cols = {}
    for t in fig.legends[0].get_texts():
        cols.setdefault(round(t.get_window_extent(r).x0), []).append(t.get_text())
    return [cols[x] for x in sorted(cols)]


@pytest.fixture(autouse=True)
def _no_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIGS", str(tmp_path / "figs"))
    monkeypatch.delenv(AC.MIRROR_ENV, raising=False)
    yield
    plt.close("all")


# ---- 1. the tidy table ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns="ci_hi"), "lack column"),
    (lambda d: d.assign(object=d.object.where(d.index != 0, "")), "empty value"),
    (lambda d: d.assign(seed=d.seed.where(d.index != 0, "")), "mix rows with and without a seed"),
    (lambda d: pd.concat([d, d.iloc[:1]], ignore_index=True), "repeat a mark"),
    (lambda d: d.assign(ci_lo=d.ci_lo.where(d.index != 0, d.delta + 1)), "does not hold its delta"),
    (lambda d: d.assign(ci_hi=d.ci_hi.where(d.index != 0, np.nan)), "one interval bound"),
    (lambda d: d.assign(eval=d["eval"].where(d.index != 0, "pair")), "do not declare"),
    (lambda d: d.assign(open="no"), "must hold booleans"),
])
def test_tidy_refuses_by_name(mutate, message):
    with pytest.raises(ValueError, match=message):
        AC.tidy(mutate(_rows()), MT)


def test_tidy_orients_by_the_category_sign_and_swaps_the_bounds():
    rows = _rows()
    t = AC.tidy(rows, MT)
    assert np.allclose(t.y, -rows.delta) and np.allclose(t.lo, -rows.ci_hi) and np.allclose(t.hi, -rows.ci_lo)
    plus = AC.tidy(rows, {"multi-turn": (EVALS, 1, "y")})
    assert np.allclose(plus.y, rows.delta) and np.allclose(plus.lo, rows.ci_lo)
    # the three category spellings: the notebook's 3-tuple, (evals, y label), and a bare eval list (sign +1, the default label)
    assert AC.resolve_categories(rows, {"m": (EVALS, "lab")})["m"] == (EVALS, 1, "lab")
    assert AC.resolve_categories(rows, {"m": EVALS})["m"] == (EVALS, 1, AC.DEFAULT_YLABEL)
    assert AC.resolve_categories(rows, None) == {"multi-turn": (EVALS, 1, AC.DEFAULT_YLABEL)}


# ---- 2. the means --------------------------------------------------------------------------------------------------------------------
def test_seed_and_pooled_means_follow_the_notebook_rule_written_out():
    rows = _rows()
    t = AC.tidy(rows, MT)
    sm = AC.seed_means(rows, MT)
    pm = AC.paired_means(rows, MT)
    assert len(sm) == len(OBJS) * 2 * len(SEEDS) and len(pm) == len(OBJS) * 2
    for (o, c), g in t.groupby(["object", "condition"]):
        ys, ses = [], []
        for sd, gs in g.groupby("seed"):
            y = gs.y.mean()
            se = np.sqrt((((gs.hi - gs.lo) / 3.92) ** 2).sum()) / len(gs)
            r = sm[(sm.object == o) & (sm.condition == c) & (sm.seed == sd)].iloc[0]
            assert r.y == pytest.approx(y) and r.lo == pytest.approx(y - 1.96 * se) and r.hi == pytest.approx(y + 1.96 * se)
            assert r.k == 2 and r.n == gs.n.sum()
            ys.append(y)
            ses.append(se)
        ys, ses = np.array(ys), np.array(ses)
        # rows without their delta's parts cannot be split (2026-11-21): the identical-seeds bound over the seeds' whole se, plus the
        # between-fit variance (the seeds' item clustering needs `steer_terms`; test_registry_nb_seed_ci.py pins that path)
        se = np.hypot(ses.mean(), ys.std(ddof=1) / np.sqrt(len(ys)))
        p = pm[(pm.object == o) & (pm.condition == c)].iloc[0]
        assert p.y == pytest.approx(ys.mean()) and p.lo == pytest.approx(ys.mean() - 1.96 * se) and p.hi == pytest.approx(ys.mean() + 1.96 * se)
        assert p.drawn and not p.ring and p.why == "" and p.n_seeds == 3 and p.seeds == "s1,s2,s3"


def test_paper_specific_lookups_never_reach_a_ports_rows():
    """Rows under a decode-only label (safety_si) and a category named Petri keep their seeds and their own evals: the notebook's
    decode-only pooling and Petri's pooled eval apply to the paper's frame, not to rows handed to the figure."""
    rows = _rows(objects=("safety_si",), category="Petri", evals=["petri_defer_to_users", "petri_self_preservation"])
    cats = {"Petri": (["petri_defer_to_users", "petri_self_preservation"], 1, "y")}
    sm = AC.seed_means(rows, cats)
    assert sorted(sm.seed.unique()) == list(SEEDS) and (sm.k == 2).all()


# ---- 3. the canvas -------------------------------------------------------------------------------------------------------------------
def test_figure_marks_shapes_colours_order_and_mean_positions():
    rows = _rows()
    fig = AC.paired_comparison_figure(rows, categories=MT, conditions=["published", "control_pool"],
                                      condition_labels={"published": "published stopping rule", "control_pool": "control-pool stopping rule"})
    marks = {g: v for g, v in _gid_marks(fig).items() if g.startswith("paired|")}
    seeds_ = [g for g in marks if "|mean|" not in g]
    means_ = [g for g in marks if "|mean|" in g]
    assert len(seeds_) == len(OBJS) * len(SEEDS) * len(EVALS) * 2 and len(means_) == len(OBJS) * 2
    for g, (x, y, verts, face, size) in marks.items():
        _, cond, obj, sd, cat, ev = g.split("|")
        assert np.allclose(verts, _marker_vertices(AC.CONDITION_MARKERS[["published", "control_pool"].index(cond)])), g
        assert np.allclose(face[:3], matplotlib.colors.to_rgb(R.COLOR[obj])), g
        assert size == pytest.approx((AC.MEAN_MARK if sd == "mean" else AC.SEED_MARK) * AC.MARKER_AREA[AC.CONDITION_MARKERS[["published", "control_pool"].index(cond)]])
    assert AC.CONDITION_MARKERS[:2] == ("o", "D")
    for o in OBJS:
        for sd in SEEDS:
            for e in EVALS:
                assert marks[f"paired|published|{o}|{sd}|multi-turn|{e}"][0] < marks[f"paired|control_pool|{o}|{sd}|multi-turn|{e}"][0]
    pm = AC.paired_means(rows, MT)
    for _, p in pm.iterrows():
        assert marks[f"paired|{p.condition}|{p.object}|mean|multi-turn|mean"][1] == pytest.approx(p.y)
    t = AC.tidy(rows, MT).set_index(["object", "seed", "eval", "condition"])
    assert marks["paired|control_pool|difference_lora|s2|multi-turn|actorattack"][1] == pytest.approx(t.loc[("difference_lora", "s2", "actorattack", "control_pool"), "y"])
    ax = fig.axes[0]
    assert [x.get_text() for x in ax.get_xticklabels()] == ["Crescendo", "ActorAttack", "mean"]
    assert ax.get_xticklabels()[-1].get_fontweight() == "bold" and ax.get_title(loc="left") == "multi-turn"
    assert ax.get_ylim()[0] <= -5 and ax.get_ylim()[1] >= 10
    assert _legend_columns(fig) == [["Control LoRA", "Difference LoRA"], ["published stopping rule", "control-pool stopping rule"], ["95% CI", " "]]
    handles = fig.legends[0].legend_handles
    assert handles[2].get_marker() == "o" and handles[3].get_marker() == "D"


def test_three_conditions_take_the_square_and_one_line_joins_a_seeds_three_marks():
    rows = _rows(conditions=("a", "b", "c"))
    fig = AC.paired_comparison_figure(rows, categories=MT)
    marks = _gid_marks(fig)
    assert np.allclose(marks["paired|c|control_lora|s1|multi-turn|crescendo"][2], _marker_vertices("s"))
    xs = [marks[f"paired|{c}|control_lora|s1|multi-turn|crescendo"][0] for c in "abc"]
    assert xs == sorted(xs)
    lines = [ln for ln in fig.axes[0].lines if len(ln.get_xdata()) == 3]
    assert any(np.allclose(ln.get_xdata(), xs) for ln in lines)
    assert [t for t in _legend_columns(fig)[1]] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="default shapes"):
        AC.paired_comparison_figure(_rows(conditions=("a", "b", "c", "d")), categories=MT)


def test_a_pending_reading_draws_the_gap_glyph_and_keeps_its_mean_off_unless_ringed():
    rows = _rows()
    hit = (rows.object == "difference_lora") & (rows.seed == "s2") & (rows["eval"] == "actorattack") & (rows.condition == "control_pool")
    rows.loc[hit, ["delta", "ci_lo", "ci_hi"]] = np.nan
    fig = AC.paired_comparison_figure(rows, categories=MT)
    gids = set(_gid_marks(fig))
    assert "paired-pending|control_pool|difference_lora|s2|multi-turn|actorattack" in gids
    assert "paired|control_pool|difference_lora|mean|multi-turn|mean" not in gids
    assert "paired-pending|control_pool|difference_lora|mean|multi-turn|mean" in gids
    assert "paired|published|difference_lora|mean|multi-turn|mean" in gids               # the other condition's mean stands
    assert R.CTLSTOP_PENDING_TEXT in [t.get_text() for t in fig.legends[0].get_texts()]
    assert AC.not_drawn(rows, MT) == ["Difference LoRA, control_pool: no mean mark, 1 reading(s) not yet in the data (s2 ActorAttack)"]
    ringed = AC.paired_comparison_figure(rows, categories=MT, mean_rule="ring")
    assert "paired|control_pool|difference_lora|mean|multi-turn|mean" in set(_gid_marks(ringed))
    assert AC.RING_TEXT in [t.get_text() for t in ringed.legends[0].get_texts()]
    assert R.RING_HANDLE.get_label() not in [t.get_text() for t in ringed.legends[0].get_texts()]
    rings = [c for c in ringed.axes[0].collections                      # the ring: an untagged, unfilled scatter over the mean mark
             if c.get_gid() is None and (len(c.get_facecolor()) == 0 or c.get_facecolor()[0][3] == 0)]
    assert len(rings) == 1
    pm = AC.paired_means(rows, MT, mean_rule="ring").set_index(["object", "condition"])
    assert pm.loc[("difference_lora", "control_pool"), "ring"] and pm.loc[("difference_lora", "control_pool"), "k"] == 1


def test_an_unseeded_object_draws_one_mark_per_condition_at_its_slot_centre_and_its_mean_is_its_eval_mean():
    """A prompted SI's rows carry no seed: one mark per (eval, condition), the pair centred on the slot, and a mean over the evals with
    no between-seed term (one series), drawn under the "complete" rule because it covers every seed the object has (none)."""
    seeded = _rows(objects=("control_lora",))
    si = _rows(objects=("safety_si",)).query("seed == 's1'").assign(seed="")
    rows = pd.concat([seeded, si], ignore_index=True)
    fig = AC.paired_comparison_figure(rows, categories=MT)
    marks = _gid_marks(fig)
    pub, new = marks["paired|published|safety_si||multi-turn|crescendo"], marks["paired|control_pool|safety_si||multi-turn|crescendo"]
    centre = 0 - 0.41 + (0.82 / 2) * 0.5                     # crescendo's group, the first of two slots (R.OBJECTS puts the SI first)
    assert (pub[0] + new[0]) / 2 == pytest.approx(centre) and pub[0] < new[0]
    pm = AC.paired_means(rows, MT).set_index(["object", "condition"])
    t = AC.tidy(si, MT)
    for c in ("published", "control_pool"):
        g = t[t.condition == c]
        se = np.sqrt((((g.hi - g.lo) / 3.92) ** 2).sum()) / len(g)
        p = pm.loc[("safety_si", c)]
        assert p.drawn and p.n_seeds == 1 and p.y == pytest.approx(g.y.mean()) and p.hi == pytest.approx(g.y.mean() + 1.96 * se)
        assert marks[f"paired|{c}|safety_si|mean|multi-turn|mean"][1] == pytest.approx(p.y)


def test_an_object_outside_the_palette_needs_a_colour_and_open_rows_draw_open():
    rows = _rows(objects=("irrelevant_si",)).assign(open=lambda d: d.seed.eq("s1"))
    with pytest.raises(ValueError, match="no colour for object"):
        AC.paired_comparison_figure(rows, categories=MT)
    fig = AC.paired_comparison_figure(rows, categories=MT, colors={"irrelevant_si": "#4a52b5"}, labels={"irrelevant_si": "Irrelevant SI"})
    marks = _gid_marks(fig)
    assert np.allclose(marks["paired|published|irrelevant_si|s1|multi-turn|crescendo"][3][:3], (1, 1, 1))          # open: white face
    assert np.allclose(marks["paired|published|irrelevant_si|s2|multi-turn|crescendo"][3][:3], matplotlib.colors.to_rgb("#4a52b5"))
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "Irrelevant SI" in labels and R.OPEN_HANDLE.get_label() in labels


# ---- 4. layouts, width and saving ----------------------------------------------------------------------------------------------------
def _two_categories():
    st = _rows(category="static", evals=list(R.CATEGORIES["static"][0]))
    return pd.concat([st, _rows()], ignore_index=True), {"static": R.CATEGORIES["static"], **MT}


def _px_per_x(ax) -> float:
    return float(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])


def test_side_by_side_panels_share_a_row_and_y_axis_and_stacked_panels_one_x_scale():
    rows, cats = _two_categories()
    fig = AC.paired_comparison_figure(rows, categories=cats)
    fig.canvas.draw()
    a, b = fig.axes
    assert a.get_title(loc="left") == "static" and b.get_title(loc="left") == "multi-turn"
    assert a.get_position().y0 == pytest.approx(b.get_position().y0) and a.get_position().x0 < b.get_position().x0
    assert a.get_ylim() == b.get_ylim() and b.get_ylabel() == "" and not any(t.get_visible() and t.get_text() for t in b.get_yticklabels())
    col = AC.paired_comparison_figure(rows, categories=cats, layout="column")
    col.canvas.draw()
    a, b = col.axes
    assert a.get_position().y0 > b.get_position().y0 and b.get_ylabel() != ""
    assert _px_per_x(a) == pytest.approx(_px_per_x(b), rel=0.02)                       # one x scale down the column
    with pytest.raises(ValueError, match="must name each category once"):
        AC.paired_comparison_figure(rows, categories=cats, layout=[["static"]])


def test_the_width_cap_narrows_the_groups_and_shrinks_the_marks():
    rows = _rows()
    wide = AC.paired_comparison_figure(rows, categories=MT, max_width=None)
    narrow = AC.paired_comparison_figure(rows, categories=MT, max_width=8.0)
    assert narrow.get_size_inches()[0] == pytest.approx(8.0) and wide.get_size_inches()[0] > 8.0
    g = "paired|published|control_lora|s1|multi-turn|crescendo"
    assert _gid_marks(narrow)[g][4] < _gid_marks(wide)[g][4]


def test_save_figure_writes_three_formats_into_figs_and_every_mirror(tmp_path, monkeypatch):
    fig = AC.paired_comparison_figure(_rows(), categories=MT)
    mirror, extra = tmp_path / "mirror", tmp_path / "extra"
    monkeypatch.setenv(AC.MIRROR_ENV, str(mirror))
    paths = AC.save_figure(fig, "t_paired", extra_dirs=[str(extra)])
    for d in (pathlib.Path(R.FIGS), mirror, extra):
        for ext in AC.FORMATS:
            assert (d / f"t_paired.{ext}").is_file() and str(d / f"t_paired.{ext}") in paths
    with pytest.raises(ValueError, match="bare file stem"):
        AC.save_figure(fig, "a/b")
    named = AC.paired_comparison_figure(_rows(), "t_named", categories=MT)
    assert (pathlib.Path(R.FIGS) / "t_named.svg").is_file() and (mirror / "t_named.png").is_file()
    plt.close(named)
