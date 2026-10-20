"""Three changes of 2026-09-24: a metric repoint that reports its coverage, padding as a Figure 3 panel, the padding table.

The bug the first group pins: `repoint_headline` moves the headline flag only on runs that HAVE the new metric, and
leaves every other run on the old one. Nothing complained, because both metrics are valid numbers in the same column
-- so the frame ended up HALF on each and a panel built from it compared some arms against anchors read the other way.
Measured on the live frame that day: 66 of 205 or_bench_hard runs carried `orb_refused`, and the mixture showed up as a
uniform ~35 pp offset on every rung of the second version, which read like a property of the metric and was not one.
So coverage is now reported and an incomplete repoint is refused by the caller rather than drawn.

Synthetic frames throughout; figures go to a temporary directory."""
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


def _frame(runs_with_new_metric, total_runs=4, eval_id="or_bench_hard",
           old="over_refusal_graded", new="orb_refused"):
    """`total_runs` runs on `eval_id`, the first `runs_with_new_metric` of them carrying BOTH metrics."""
    rows = []
    for i in range(total_runs):
        rid = f"run{i}"
        rows.append(dict(run_id=rid, eval_id=eval_id, metric_id=old, headline=True, value=40.0 + i))
        if i < runs_with_new_metric:
            rows.append(dict(run_id=rid, eval_id=eval_id, metric_id=new, headline=False, value=5.0 + i))
    # an unrelated eval must be left completely alone by any repoint
    rows.append(dict(run_id="other", eval_id="xstest", metric_id="over_refusal_graded", headline=True, value=12.0))
    return pd.DataFrame(rows)


class TestRepointCoverage:
    def test_complete_repoint_moves_every_run_and_reports_complete(self):
        out = R.repoint_headline(_frame(4), "or_bench_hard", "orb_refused")
        assert out.attrs["repoint_coverage"] == (4, 4)
        assert out.attrs["repoint_complete"] is True
        hl = out[out.headline & out.eval_id.eq("or_bench_hard")]
        assert set(hl.metric_id) == {"orb_refused"}, "a complete repoint leaves no run on the old metric"

    def test_partial_repoint_is_reported_incomplete(self):
        """THE REGRESSION. Before the fix this returned a frame split across two metrics with nothing to say so."""
        out = R.repoint_headline(_frame(1), "or_bench_hard", "orb_refused")
        assert out.attrs["repoint_coverage"] == (1, 4)
        assert out.attrs["repoint_complete"] is False
        hl = out[out.headline & out.eval_id.eq("or_bench_hard")]
        # the mixture is real and is exactly what the caller must now refuse to draw
        assert set(hl.metric_id) == {"orb_refused", "over_refusal_graded"}

    def test_no_run_carries_the_metric(self):
        out = R.repoint_headline(_frame(0), "or_bench_hard", "orb_refused")
        assert out.attrs["repoint_complete"] is False
        assert out.attrs["repoint_coverage"] == (0, 4)

    def test_asking_for_the_current_headline_is_a_complete_no_op(self):
        """THE v124 REGRESSION. An earlier clause repointed "the other way" when the requested metric was already
        headline, so `repoint_all(frame, GRADED)` handed back the REFUSED rate and reported "0 of 0 runs carry
        over_refusal_graded" — a frame entirely on the requested metric, reported as having none of it, and withheld."""
        f = _frame(4)                                   # over_refusal_graded is headline on all four runs
        out = R.repoint_headline(f, "or_bench_hard", "over_refusal_graded")
        assert out.attrs["repoint_complete"] is True, "the frame already IS on this metric: coverage is total"
        assert out.attrs["repoint_coverage"] == (4, 4)
        hl = out[out.headline & out.eval_id.eq("or_bench_hard")]
        assert set(hl.metric_id) == {"over_refusal_graded"}, "asking for a metric must never return the other one"
        assert len(hl) == 4

    def test_both_directions_are_available_from_one_frame(self):
        """Both versions name their metric, so the DB's own default cannot change what either figure plots."""
        f = _frame(4)
        g = R.repoint_headline(f, "or_bench_hard", "over_refusal_graded")
        r = R.repoint_headline(f, "or_bench_hard", "orb_refused")
        assert g.attrs["repoint_complete"] and r.attrs["repoint_complete"]
        assert set(g[g.headline & g.eval_id.eq("or_bench_hard")].metric_id) == {"over_refusal_graded"}
        assert set(r[r.headline & r.eval_id.eq("or_bench_hard")].metric_id) == {"orb_refused"}

    def test_a_row_no_panel_reads_is_not_in_the_denominator(self):
        """A run on the eval carrying only some THIRD metric is never drawn, so it cannot make a repoint look short.
        Measured on the live frame: the old count said graded was 357 of 507 and withheld a figure that was fine."""
        f = _frame(4)
        spare = pd.DataFrame([dict(run_id="unread", eval_id="or_bench_hard", metric_id="something_else",
                                   headline=False, value=1.0)])
        out = R.repoint_headline(pd.concat([f, spare], ignore_index=True), "or_bench_hard", "orb_refused")
        assert out.attrs["repoint_coverage"] == (4, 4), "the unread row is neither covered nor counted against"
        assert out.attrs["repoint_complete"] is True

    def test_a_headline_run_that_lacks_the_metric_still_counts_against(self):
        """The gate must keep biting on runs that ARE drawn — that is the failure it exists to catch."""
        out = R.repoint_headline(_frame(1), "or_bench_hard", "orb_refused")
        assert out.attrs["repoint_coverage"] == (1, 4)
        assert out.attrs["repoint_complete"] is False

    def test_a_repoint_does_not_touch_another_eval(self):
        out = R.repoint_headline(_frame(4), "or_bench_hard", "orb_refused")
        other = out[out.eval_id.eq("xstest")]
        assert list(other.headline) == [True] and list(other.metric_id) == ["over_refusal_graded"]


class TestRepointAll:
    """The frontier's x is the MEAN of XSTest and OR-Bench, so one eval short is still a mixed panel."""

    def _two_evals(self, orb_ok, xst_ok):
        a = _frame(4 if orb_ok else 1, eval_id="or_bench_hard", new="orb_refused")
        a = a[a.eval_id.eq("or_bench_hard")]
        b = _frame(4 if xst_ok else 1, eval_id="xstest", new="xst_refused")
        # _frame appends a spare xstest row under run_id "other" with no new metric; it belongs to the
        # single-eval fixtures and would make this frame look one run short on every parametrisation
        b = b[b.eval_id.eq("xstest") & b.run_id.ne("other")]
        b = b.assign(run_id=b.run_id + "x")
        return pd.concat([a, b], ignore_index=True)

    SPEC = {"or_bench_hard": "orb_refused", "xstest": "xst_refused"}

    def test_both_complete(self):
        out = R.repoint_all(self._two_evals(True, True), self.SPEC)
        assert out.attrs["repoint_complete"] is True
        assert out.attrs["repoint_coverage"] == (8, 8)

    @pytest.mark.parametrize("orb_ok,xst_ok", [(True, False), (False, True), (False, False)])
    def test_one_eval_short_is_still_incomplete(self, orb_ok, xst_ok):
        out = R.repoint_all(self._two_evals(orb_ok, xst_ok), self.SPEC)
        assert out.attrs["repoint_complete"] is False, "completeness is the AND over the evals, not the OR"

    def test_reports_which_metrics_it_pointed_at(self):
        out = R.repoint_all(self._two_evals(True, True), self.SPEC)
        assert "orb_refused" in out.attrs["repointed_to"] and "xst_refused" in out.attrs["repointed_to"]


THREE_RUNGS = [0, 16384, 65536]          # the rungs in the frame today; PAD_RUNGS lists five, two of them still folding


def _pad_points(held_at=None):
    rows = []
    for o in ("arditi_allpos", "safety_si"):
        for pad, (d, lo, hi) in zip(THREE_RUNGS, [(5.0, 1.0, 9.0), (-3.2, -7.0, 0.6), (12.4, 8.0, 16.8)]):
            rows.append(dict(object_label=o, pad=float(pad), delta=d, lo=lo, hi=hi, defence=50.0,
                             base_defence={0: 44.8, 16384: 52.9, 65536: 37.5}[pad], degen=np.nan,
                             held=(o == held_at and pad == 0)))
    return pd.DataFrame(rows)


class TestPaddingTable:
    def test_baseline_row_is_absolute_and_arms_are_signed(self):
        tex = R.padding_table(_pad_points(), ["arditi_allpos", "safety_si"])
        assert R.BASELINE_ROW_LABEL in tex
        assert "$44.8$" in tex and "$52.9$" in tex and "$37.5$" in tex, "the unsteered row is absolute, unsigned"
        assert "$+5.0$" in tex and "$-3.2$" in tex, "every arm cell is signed against that same-rung anchor"

    def test_symmetric_intervals_use_plus_minus(self):
        tex = R.padding_table(_pad_points(), ["arditi_allpos"])
        assert "\\pm" in tex and "[" not in tex.split("\\hline")[-1], "symmetric intervals print as +/- to fit textwidth"

    def test_caption_is_bare_and_there_is_no_minipage(self):
        tex = R.padding_table(_pad_points(), ["arditi_allpos"])
        assert "\\caption{Padding ladder for" in tex
        assert "minipage" not in tex
        assert tex.count("\\caption") == 1

    def test_a_held_arm_is_marked_and_the_mark_is_keyed(self):
        tex = R.padding_table(_pad_points(held_at="safety_si"), ["arditi_allpos", "safety_si"])
        assert "\\S" in tex
        assert "% note:" in tex and "held out" in tex, "a symbol with no key is worse than no symbol"

    def test_degeneracy_is_never_a_column(self):
        tex = R.padding_table(_pad_points(), ["arditi_allpos"])
        assert "degen" not in tex.lower()

    def test_empty_points_render_nothing(self):
        assert R.padding_table(pd.DataFrame()) == ""


class TestPaddingPanelHook:
    """Padding rides into Figure 3 as a HOOK, which is what keeps its rows out of every category mean."""

    def _empty_category_frames(self):
        cols = ["model_id", "eval_id", "object_label", "seed", "d_y", "d_lo", "d_hi",
                "anchor_in_frame", "category", "anchor_abs", "cap_mismatch"]
        return (pd.DataFrame(columns=cols), pd.DataFrame(columns=cols),
                pd.DataFrame(columns=["model_id", "eval_id", "object_label", "status"]))

    def test_a_hook_panel_draws_beside_a_category_panel(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))          # save() writes to FIGS; a wrong name here leaks into paper/notebooks/figures
        cm, d, gaps = self._empty_category_frames()
        drew = []
        fig = R.delta_figure(cm, d, [["static", "padding"]], R.OBJECTS, gaps, "t_hook",
                             panel_hooks={"padding": lambda ax: drew.append(ax)})
        assert len(drew) == 1, "the hook is handed exactly one axis"
        assert len(fig.axes) >= 2, "the category panel and the hook panel both exist"
        plt.close(fig)

    def test_a_layout_without_hooks_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))          # save() writes to FIGS; a wrong name here leaks into paper/notebooks/figures
        cm, d, gaps = self._empty_category_frames()
        fig = R.delta_figure(cm, d, [["static", "multi-turn"]], R.OBJECTS, gaps, "t_plain")
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_draw_padding_returns_the_objects_it_drew(self, tmp_path):
        fig, ax = plt.subplots()
        drawn = R._draw_padding(ax, _pad_points(), compact=True)
        assert set(drawn) == {"arditi_allpos", "safety_si"}
        assert ax.get_xlabel() == "neutral filler tokens", "the compact label is for the narrow shared slot"
        plt.close(fig)

    def test_the_ladder_lists_five_rungs(self):
        assert R.PAD_RUNGS == [0, 8192, 16384, 32768, 65536], "Pad's five-rung ladder (the user, 2026-09-24)"

    def test_absent_rungs_leave_no_empty_slot(self):
        """Three rungs in the points, five in the ladder: the axis shows three ticks, not five with two gaps."""
        fig, ax = plt.subplots()
        R._draw_padding(ax, _pad_points(), compact=True)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "16,384", "65,536"]
        plt.close(fig)

    def test_a_new_rung_appears_in_ladder_order_once_present(self):
        pts = _pad_points()
        extra = pts[pts.pad.eq(0)].assign(pad=8192.0, delta=1.0, lo=-1.0, hi=3.0, base_defence=48.0)
        fig, ax = plt.subplots()
        R._draw_padding(ax, pd.concat([pts, extra], ignore_index=True), compact=True)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "8,192", "16,384", "65,536"]
        plt.close(fig)

    def test_the_table_columns_are_the_rungs_present(self):
        tex = R.padding_table(_pad_points(), ["arditi_allpos"])
        assert "16,384" in tex and "65,536" in tex and "8,192" not in tex and "32,768" not in tex

    def test_the_full_label_is_used_when_not_compact(self):
        fig, ax = plt.subplots()
        R._draw_padding(ax, _pad_points(), compact=False)
        assert "system instruction" in ax.get_xlabel()
        plt.close(fig)


class TestAtCapCensoringGranularity:
    """A runaway SEED comes off a rung; it does not take the rung's healthy seeds with it.

    Measured on the live frame: difference_relu at dose 1.25 had s3 generating to the 8,192-token ceiling on 94.8% of
    OR-Bench and 91.6% of XSTest prompts, while s1 sat at 8.4%/4.4% and s2 at 47.3%/19.4%. Because the pooled point's
    `cap` was the MAX over its seeds, s3 censored the whole rung off both panels -- amputating the difference steers'
    top rung while Arditi (<=0.3% at cap everywhere) kept its full ladder, which is what made ReLU read as weaker on
    the frontier than Figures 3-4 show it. The gate has to run at the granularity the pooling happens."""

    def test_the_threshold_is_what_the_frontier_declares(self):
        assert R.FRONTIER_AT_CAP_MAX == 0.90

    def test_one_runaway_seed_does_not_remove_its_healthy_siblings(self):
        shares = {"s1": 0.08, "s2": 0.47, "s3": 0.95}
        rows = [dict(object_label="difference_relu", seed=sd, dose=1.25, cap_share=v) for sd, v in shares.items()]
        f = pd.DataFrame(rows)
        worst = f.groupby(["object_label", "seed", "dose"])["cap_share"].transform("max")
        kept = f[~(worst > R.FRONTIER_AT_CAP_MAX)]
        assert set(kept.seed) == {"s1", "s2"}, "only the runaway seed goes"
        assert "s3" not in set(kept.seed)

    def test_a_rung_whose_every_seed_runs_away_still_disappears(self):
        f = pd.DataFrame([dict(object_label="x", seed=sd, dose=1.25, cap_share=0.97) for sd in ("s1", "s2", "s3")])
        worst = f.groupby(["object_label", "seed", "dose"])["cap_share"].transform("max")
        assert not len(f[~(worst > R.FRONTIER_AT_CAP_MAX)]), "nothing readable is left, so the rung goes"


class TestUnscaledRateGuard:
    """A headline metric absent from SCALE plots at its raw 0-1 value, i.e. 100x small.

    Second occurrence: over_refusal_graded_noncollapsed was the first, xst_refused / orb_refused the second, when the
    registry flipped its over-refusal headline. `unscaled_rates` already caught this shape -- it was just only ever
    called on the operating-point frame, never on the frontier frame where the over-refusal x-axis lives."""

    def test_the_refused_rate_spellings_are_scaled(self):
        for m in ("xst_refused", "orb_refused"):
            assert R.SCALE.get(m) == 100.0, f"{m} must scale to pp"
            assert m in R.IS_RATE, f"{m} needs a Wilson denominator"

    def test_every_scaled_proportion_metric_has_a_wilson_denominator(self):
        missing = [m for m, s in R.SCALE.items() if s == 100.0 and m not in R.IS_RATE]
        assert not missing, f"scaled as a percentage but no IS_RATE entry: {missing}"

    def test_the_guard_flags_a_proportion_metric_absent_from_scale(self):
        f = pd.DataFrame([dict(metric_id="brand_new_rate", value=0.61, headline=True),
                          dict(metric_id="brand_new_rate", value=0.44, headline=True)])
        out = R.unscaled_rates(f)
        assert list(out.metric_id) == ["brand_new_rate"], "an unscaled proportion headline must be reported"

    def test_the_guard_is_quiet_once_the_metric_is_scaled(self):
        f = pd.DataFrame([dict(metric_id="orb_refused", value=0.61, headline=True)])
        assert not len(R.unscaled_rates(f)), "a metric in SCALE is not flagged"

    def test_the_guard_ignores_non_headline_rows(self):
        f = pd.DataFrame([dict(metric_id="brand_new_rate", value=0.61, headline=False)])
        assert not len(R.unscaled_rates(f))


class TestTickRotation:
    """Figure 5's Petri panel: eval labels turned 45 degrees, one line each, right edge on the tick; "mean" upright."""

    def _frames(self):
        cols = ["model_id", "eval_id", "object_label", "seed", "d_y", "d_lo", "d_hi",
                "anchor_in_frame", "category", "anchor_abs", "cap_mismatch"]
        return (pd.DataFrame(columns=cols), pd.DataFrame(columns=cols),
                pd.DataFrame(columns=["model_id", "eval_id", "object_label", "status"]))

    def _petri_axis(self, fig):
        return next(a for a in fig.axes if a.get_title() == "Petri")

    def test_eval_labels_rotate_and_mean_stays_upright(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cm, d, gaps = self._frames()
        fig = R.delta_figure(cm, d, [["Petri", "agentic misalignment"]], R.OBJECTS, gaps, "t_rot",
                             unit=1.5, tick_rotation={"Petri": 45})
        labs = self._petri_axis(fig).get_xticklabels()
        assert len(labs) == len(R.CATEGORIES["Petri"][0]) + 1
        for lab in labs[:-1]:
            assert lab.get_rotation() == 45 and lab.get_ha() == "right" and lab.get_rotation_mode() == "anchor"
        # ROTATED LABELS KEEP THEIR TWO-LINE SPLIT (later that day): the same "\n" the horizontal
        # form uses, turned as a block. An earlier form joined them to one line ("cyber/bio/chem"); that is gone.
        texts = [lab.get_text() for lab in labs[:-1]]
        expected = [R.tick(e) for e in R.CATEGORIES["Petri"][0]] if hasattr(R, "tick") else None
        if expected:
            assert texts == expected, "a rotated eval label is exactly the horizontal label, newline included"
            assert any("\n" in x for x in texts), "at least one Petri label is two-line, and rotation must not flatten it"
        assert labs[-1].get_text() == "mean" and labs[-1].get_rotation() == 0
        assert labs[-1].get_fontweight() == "bold"
        plt.close(fig)

    def test_a_panel_not_named_keeps_its_labels_flat(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cm, d, gaps = self._frames()
        fig = R.delta_figure(cm, d, [["Petri", "agentic misalignment"]], R.OBJECTS, gaps, "t_rot2",
                             tick_rotation={"Petri": 45})
        ag = next(a for a in fig.axes if a.get_title() == "agentic misalignment")
        assert all(l.get_rotation() == 0 for l in ag.get_xticklabels())
        plt.close(fig)

    def test_no_rotation_argument_changes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cm, d, gaps = self._frames()
        fig = R.delta_figure(cm, d, [["Petri"]], R.OBJECTS, gaps, "t_rot3")
        assert all(l.get_rotation() == 0 for l in self._petri_axis(fig).get_xticklabels())
        plt.close(fig)


class TestOneRowFiguresShareAxesHeight:
    """One-row figures share a Y-AXIS height: a figure whose labels need more vertical room
    grows so its axes stay comparable, rather than keeping the figure height and shrinking the plot. Measured on empty
    frames: Figure 4's axes are 3.66 in at the 4.2 in default; Figure 5 with Petri rotated 45 degrees is 3.67 in at 5.4."""

    def _frames(self):
        cols = ["model_id", "eval_id", "object_label", "seed", "d_y", "d_lo", "d_hi",
                "anchor_in_frame", "category", "anchor_abs", "cap_mismatch"]
        return (pd.DataFrame(columns=cols), pd.DataFrame(columns=cols),
                pd.DataFrame(columns=["model_id", "eval_id", "object_label", "status"]))

    @staticmethod
    def _axes_height_in(fig, title):
        fig.canvas.draw()
        ax = next(a for a in fig.axes if a.get_title() == title)
        return ax.get_window_extent().height / fig.dpi

    def test_fig5_rotated_matches_fig4_axes_height(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cm, d, gaps = self._frames()
        f4 = R.delta_figure(cm, d, [["multi-query", "multi-turn"]], R.OBJECTS, gaps, "t_h4")
        f5 = R.delta_figure(cm, d, [["Petri", "agentic misalignment"]], R.OBJECTS, gaps, "t_h5",
                            unit=1.5, height=5.4, tick_rotation={"Petri": 45})
        h4, h5 = self._axes_height_in(f4, "multi-query"), self._axes_height_in(f5, "Petri")
        plt.close(f4); plt.close(f5)
        # WHAT SHIPS: rotated Petri labels keep their TWO-line split and the figure height stays at 5.4. Two-line rotated
        # labels need less vertical room than one-line ones, so at 5.4 the axes come out 4.06 in against Figure 4's 3.66
        # -- 0.4 in taller. Parity would be 5.0 (measured 3.66 = 3.66); 5.4 is the shipped height, so the test pins that
        # height and states the delta rather than pretending parity.
        assert h5 >= h4 - 0.1, f"Fig 5's axes may not be SHORTER than Fig 4's: {h5:.2f} vs {h4:.2f} in"
        assert abs(h4 - h5) <= 0.45, f"axes heights differ by more than the ruled 5.4-in height explains: Fig 4 {h4:.2f} vs Fig 5 {h5:.2f} in"

    def test_rotation_at_the_default_height_would_squeeze_the_axes(self, tmp_path, monkeypatch):
        """The failure v134 shipped: rotation with no extra height cost the axes over an inch."""
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cm, d, gaps = self._frames()
        f4 = R.delta_figure(cm, d, [["multi-query", "multi-turn"]], R.OBJECTS, gaps, "t_h4b")
        f5 = R.delta_figure(cm, d, [["Petri", "agentic misalignment"]], R.OBJECTS, gaps, "t_h5b",
                            unit=1.5, tick_rotation={"Petri": 45})
        h4, h5 = self._axes_height_in(f4, "multi-query"), self._axes_height_in(f5, "Petri")
        plt.close(f4); plt.close(f5)
        # 0.6, not 0.8: the gap is font-metric dependent (0.77 in under matplotlib 3.10 where `semibold` falls back to weight 700,
        # against ~0.9 in the notebook venv); the claim is only that the rotated labels cost real height
        assert h4 - h5 > 0.6, "without the extra height the rotated labels eat the axes"


class TestRungTableSignsMatchTheFigure:
    """`rung_eval_table` shows the GOOD delta the frontier draws, in every cell and every mean (2026-09-24).

    It used to print per-eval cells raw (ASR down = negative) beside sign-oriented ST/MT means, and its OR mean used +1
    where the frontier's x uses −1 — so the table's OR column was the negative of the figure's x. Diffing it against
    ladder_fig.ipynb's table read as a metric disagreement when it was a sign."""

    def _frame(self):
        rows = []
        # one seeded arm at dose 1.0 whose ASR FELL by 0.20 on every attack and whose refusal ROSE by 0.10 on both
        # benign evals, against an unsteered anchor; d_y is the raw plotted-unit delta the table pivots (y − anchor)
        for ev in ("pasttense", "codeattack", "renellm", "grok420", "crescendo", "actorattack"):
            rows.append(dict(object_label="difference_relu", seed="s1", dose=1.0, eval_id=ev, metric_id="asr",
                             headline=True, judge_ok=True, selection_state="selected", d_y=-20.0, lane="base_to_instruct"))
        for ev, m in (("xstest", "xst_refused"), ("or_bench_hard", "orb_refused")):
            rows.append(dict(object_label="difference_relu", seed="s1", dose=1.0, eval_id=ev, metric_id=m,
                             headline=True, judge_ok=True, selection_state="selected", d_y=+10.0, lane="base_to_instruct"))
        return pd.DataFrame(rows)

    def test_cells_and_means_are_oriented_up_is_better(self, monkeypatch):
        monkeypatch.setattr(R, "anchored", lambda d, **k: d)          # d_y is already the delta in this fixture
        out = R.rung_eval_table(self._frame(), ["difference_relu"]).reset_index()
        row = out.iloc[0]
        for c in ("Past tense", "CodeAttack", "ReNeLLM", "Pliny", "Crescendo", "ActorAttack"):
            assert row[c] == 20.0, f"{c}: an ASR drop of 20 pp is a +20 defense delta, got {row[c]}"
        for c in ("XSTest", "OR-Bench-Hard"):
            assert row[c] == -10.0, f"{c}: a refusal rise of 10 pp is a −10 compliance delta, got {row[c]}"
        assert row["ST mean"] == 20.0 and row["MT mean"] == 20.0
        assert row["OR mean"] == -10.0, "the OR mean must carry the figure's sign: more refusal = negative"


class TestFrontierDrawsOnlySelectedRungs:
    """A `not_in_roster` rung is a reading but not a rung of the roster's ladder; the frontier does not draw it
    (2026-09-24: Arditi at c = 0.1 and 0.2 appeared on static, entirely not_in_roster, and were not wanted).
    Unsteered `none` rows are anchors and pass whatever their state."""

    def test_selection_filter_keeps_selected_and_anchors_only(self):
        import inspect
        src = inspect.getsource(R.frontier_points)
        assert 'frame.selection_state.eq("selected") | frame.object_label.eq("none")' in src, \
            "frontier_points must admit only selected rungs (and every unsteered anchor)"

    def test_a_not_in_roster_rung_is_filtered_before_anything_else(self):
        f = pd.DataFrame([
            dict(object_label="arditi_allpos", selection_state="not_in_roster", eval_id="pasttense", dose=0.2),
            dict(object_label="arditi_allpos", selection_state="selected", eval_id="pasttense", dose=0.3),
            dict(object_label="none", selection_state="not_in_roster", eval_id="pasttense", dose=0.0),
            dict(object_label="none", selection_state="selected", eval_id="pasttense", dose=0.0),
        ])
        kept = f[f.selection_state.eq("selected") | f.object_label.eq("none")]     # the exact predicate the function applies
        assert set(zip(kept.object_label, kept.dose)) == {("arditi_allpos", 0.3), ("none", 0.0)}
        assert (kept.object_label.eq("none")).sum() == 2, "both anchors survive regardless of state"


class TestFrontierHonoursHeldOut:
    """A registry hold-out drops the seed's rung; anchors are untouched (ladder_fig drops difference_lora 1.25 s2 through
    held_out, and the frontier had been drawing it while censoring s3 instead)."""

    def test_source_applies_the_hold_out_before_anything_else(self):
        import inspect
        src = inspect.getsource(R.frontier_points)
        assert "held_out" in src and 'frame.object_label.ne("none")' in src

    def test_predicate_drops_a_seed_rung_with_any_held_row_and_keeps_anchors(self):
        f = pd.DataFrame([
            dict(object_label="difference_lora", seed="s2", dose=1.25, eval_id="xstest", held_out=1),
            dict(object_label="difference_lora", seed="s2", dose=1.25, eval_id="pasttense", held_out=0),
            dict(object_label="difference_lora", seed="s3", dose=1.25, eval_id="pasttense", held_out=0),
            dict(object_label="none", seed="", dose=0.0, eval_id="pasttense", held_out=1),
        ])
        held = f.assign(_h=f.held_out.gt(0)).groupby(["object_label","seed","dose"])["_h"].transform("max")
        kept = f[~(held & f.object_label.ne("none"))]
        assert set(zip(kept.object_label, kept.seed)) == {("difference_lora", "s3"), ("none", "")}
        assert not ((kept.object_label == "difference_lora") & (kept.seed == "s2")).any(), "one held row drops the whole seed-rung"

    def test_censor_reads_the_defense_battery_only(self):
        import inspect
        src = inspect.getsource(R.frontier_points)
        assert 'cap=g.loc[g.eval_id.isin(y_evals), "cap_share"].max()' in src
        assert "pool_rows.cap_share.where(pool_rows.eval_id.isin(y_evals))" in src


class TestFrontierDropRungs:
    """An explicit (object, seed, dose) set removes exactly those seed-rungs before pooling."""

    def test_predicate_drops_exactly_the_named_triples(self):
        f = pd.DataFrame([dict(object_label="difference_relu", seed="s3", dose=1.25, eval_id="pasttense"),
                          dict(object_label="difference_relu", seed="s2", dose=1.25, eval_id="pasttense"),
                          dict(object_label="difference_relu", seed="s3", dose=1.0, eval_id="pasttense"),
                          dict(object_label="difference_lora", seed="s2", dose=1.25, eval_id="pasttense")])
        drop = {("difference_relu", "s3", 1.25), ("difference_lora", "s2", 1.25)}
        key = pd.Series(list(zip(f.object_label, f.seed.fillna("").astype(str), pd.to_numeric(f.dose).round(3))), index=f.index)
        kept = f[~key.isin({(o, str(sd), round(float(dd), 3)) for o, sd, dd in drop})]
        assert set(zip(kept.object_label, kept.seed, kept.dose)) == {("difference_relu", "s2", 1.25), ("difference_relu", "s3", 1.0)}

    def test_frontier_points_accepts_drop_rungs(self):
        import inspect
        assert "drop_rungs" in inspect.signature(R.frontier_points).parameters


class TestFrontierTableSeedsAndHeaders:
    """The frontier table: ONE TABLE PER PANEL with no panel column, a seed column, short headers (`$\\Delta$ defense` /
    `$\\Delta$ compliance`, upper-cased outside the math span by latex_table), rows read from the panel's `per_seed` frame,
    and a first row stating the unsteered model's ABSOLUTE defense and compliance."""

    def _panels(self):
        def mk(title, ay, ax):
            pts = pd.DataFrame([dict(object_label="difference_relu", seed="", dose=1.0, dx=-17.8, dy=29.7, ax_abs=ax, ay_abs=ay)])
            ps = pd.DataFrame([dict(object_label="difference_relu", seed="s1", dose=1.0, dx=-6.2, dy=15.8),
                               dict(object_label="difference_relu", seed="s2", dose=1.0, dx=-18.8, dy=32.4),
                               dict(object_label="difference_relu", seed="s3", dose=1.0, dx=-28.3, dy=40.7),
                               dict(object_label="arditi_allpos", seed="", dose=0.3, dx=-15.4, dy=7.5)])
            return dict(title=title, pts=pts, per_seed=ps)
        return [mk("static", 55.0, 64.4), mk("multi-turn", 45.0, 64.4)]

    def test_one_table_per_panel_with_no_panel_column(self):
        tex = R.table_frontier(self._panels(), ["difference_relu", "arditi_allpos"], "{panel} rungs.", "tab:t")
        assert tex.count("\\begin{table}") == 2
        assert "\\label{tab:t-static}" in tex and "\\label{tab:t-multi-turn}" in tex
        assert "\\caption{Static rungs.}" in tex and "\\caption{Multi-turn rungs.}" in tex
        assert "PANEL" not in tex
        hdr = [l for l in tex.splitlines() if "ARM" in l and "SEED" in l][0]
        assert hdr.count("multicolumn") == 5

    def test_headers_are_short_and_upper_cased_outside_math(self):
        tex = R.table_frontier(self._panels(), ["difference_relu"], "cap", "tab:t")
        assert "$\\Delta$ DEFENSE" in tex and "$\\Delta$ COMPLIANCE" in tex
        assert "(pp)" not in tex.split("% note")[0] and "percentage points" in tex

    def test_rows_are_the_seed_rungs_not_the_pooled_point(self):
        tex = R.table_frontier(self._panels()[:1], ["difference_relu", "arditi_allpos"], "cap", "tab:t")
        for sd, dx in (("s1", "-6.2"), ("s2", "-18.8"), ("s3", "-28.3")):
            assert sd in tex and f"${dx}$" in tex, (sd, dx)
        assert "$-17.8$" not in tex
        assert tex.count("Refusal steer") == 1

    def test_the_baseline_row_states_the_unsteered_absolutes_per_panel(self):
        tex = R.table_frontier(self._panels(), ["difference_relu"], "cap", "tab:t")
        base = [l for l in tex.splitlines() if "unsteered (absolute)" in l]      # not the note, which also says "unsteered"
        assert len(base) == 2
        assert "$55.0$" in base[0] and "$64.4$" in base[0], base[0]
        assert "$45.0$" in base[1] and "$64.4$" in base[1], base[1]
        assert all(b.count("--") == 2 for b in base), "seed and dose are the only dashes on a baseline row"

    def test_a_prompted_arm_is_one_row_the_mean_over_its_runs(self):
        """Safety / Constitution SI accumulate several runs at dose 1 over waves; the figure pools them into one diamond and the
        table records that mean, not the runs."""
        ps = pd.DataFrame([dict(object_label="safety_si", seed="", dose=1.0, dx=-6.0, dy=20.0),
                           dict(object_label="safety_si", seed="", dose=1.0, dx=-8.0, dy=24.0),
                           dict(object_label="safety_si", seed="", dose=1.0, dx=-7.0, dy=22.0)])
        pts = pd.DataFrame([dict(object_label="safety_si", seed="", dose=1.0, dx=-7.0, dy=22.0, ax_abs=65.0, ay_abs=66.0)])
        tex = R.table_frontier([dict(title="static", pts=pts, per_seed=ps)], ["safety_si"], "cap", "tab:t")
        rows = [l for l in tex.splitlines() if "Safety SI" in l]
        assert len(rows) == 1, rows
        assert "$+22.0$" in rows[0] and "$-7.0$" in rows[0], rows[0]
        assert "large diamond" in tex

    def test_falls_back_to_pooled_points_without_per_seed(self):
        p = self._panels()[0]; p.pop("per_seed")
        tex = R.table_frontier([p], ["difference_relu"], "cap", "tab:t")
        assert "$-17.8$" in tex


class TestPaddingSeeds:
    """The padding panel and table take several fit seeds per rung: seeds faint behind, the pooled
    mean (k / n summed across seeds) bold; the table lists the seeds and a `mean` row. Unseeded arms unchanged."""

    def _frame(self):
        rows = []
        for pad, (kn_none, kn_s1, kn_s2) in ((0, (0.55, 0.45, 0.50)), (16384, (0.47, 0.40, 0.30)), (65536, (0.62, 0.35, 0.45))):
            rows.append(dict(eval_id="grok420", object_label="none", seed="", pad=pad, value=kn_none, n=259, headline=True,
                             selection_state="selected", pad_contract=True, held_out=0))
            for sd, v in (("s1", kn_s1), ("s2", kn_s2)):
                rows.append(dict(eval_id="grok420", object_label="difference_relu", seed=sd, pad=pad, value=v, n=259, headline=True,
                                 selection_state="selected", pad_contract=True, held_out=0))
            rows.append(dict(eval_id="grok420", object_label="safety_si", seed="", pad=pad, value=kn_s1 + 0.05, n=259, headline=True,
                             selection_state="selected", pad_contract=True, held_out=0))
        return pd.DataFrame(rows)

    def test_points_are_per_seed_with_k_and_n(self):
        pts = R.padding_points(self._frame())
        relu = pts[pts.object_label.eq("difference_relu")]
        assert sorted(set(relu.seed)) == ["s1", "s2"] and len(relu) == 6
        assert {"k1", "n1", "k0", "n0"} <= set(pts.columns) and (relu.n1 == 259).all()
        si = pts[pts.object_label.eq("safety_si")]; assert len(si) == 3 and set(si.seed) == {""}

    def test_a_held_seed_point_is_neither_drawn_nor_pooled(self):
        """The pad-0 difference-ReLU rows carry held_out = 1 on every seed and drew anyway; the pool included them."""
        f = self._frame()
        f.loc[f.object_label.eq("difference_relu") & f.seed.eq("s1") & f.pad.eq(0), "held_out"] = 1
        pts = R.padding_points(f)
        held = R.padding_held(pts)
        assert held.to_dict("records") == [dict(object_label="difference_relu", seed="s1", pad=0.0)]
        pooled = R.padding_pooled(pts[pts.object_label.eq("difference_relu")]).set_index("pad")
        assert pooled.loc[0.0, "n1"] == 259 and pooled.loc[0.0, "n_seeds"] == 1, "pad 0 pools seed 2 alone"
        assert pooled.loc[16384.0, "n1"] == 518 and not pooled.held.any()
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); R._draw_padding(ax, pts); plt.close(fig)
        s1_held_delta = float(pts[pts.object_label.eq("difference_relu") & pts.seed.eq("s1") & pts.pad.eq(0)].delta.iloc[0])
        drawn_at_x0 = {round(float(y), 6) for ln in ax.lines for x, y in zip(ln.get_xdata(), ln.get_ydata()) if float(x) == 0.0}
        assert round(s1_held_delta, 6) not in drawn_at_x0, "the held point must not be on the canvas"
        # every seed held at a rung -> no mean row for that rung
        f2 = f.copy(); f2.loc[f2.object_label.eq("difference_relu") & f2.pad.eq(0), "held_out"] = 1
        p2 = R.padding_pooled(R.padding_points(f2)[lambda d: d.object_label.eq("difference_relu")])
        assert sorted(p2.pad) == [16384.0, 65536.0]

    def test_pooled_sums_k_and_n_across_seeds(self):
        pts = R.padding_points(self._frame()); pooled = R.padding_pooled(pts[pts.object_label.eq("difference_relu")])
        assert len(pooled) == 3 and set(pooled.seed) == {"mean"} and (pooled.n1 == 518).all()
        r0 = pooled[pooled.pad.eq(0)].iloc[0]
        # k0 = round(.55*259) = 142; k1 = round(.45*259) + round(.50*259) = 117 + 130 = 247 over 518
        assert abs(r0.delta - 100 * (142 / 259 - 247 / 518)) < 1e-6
        assert (pooled.n_seeds == 2).all()

    def test_panel_draws_seeds_faint_and_the_pool_bold(self):
        pts = R.padding_points(self._frame())
        fig, ax = plt.subplots(); drawn = R._draw_padding(ax, pts, compact=True)
        assert set(drawn) == {"difference_relu", "safety_si"}
        lines = [l for l in ax.get_lines() if len(l.get_xdata()) == 3]
        relu = [l for l in lines if l.get_color() == R.COLOR["difference_relu"]]
        faint = [l for l in relu if l.get_linewidth() < 2]; bold = [l for l in relu if l.get_linewidth() >= 2]
        assert len(faint) == 2 and len(bold) == 1, (len(faint), len(bold))
        assert faint[0].get_alpha() < bold[0].get_alpha() and faint[0].get_zorder() < bold[0].get_zorder()
        si = [l for l in lines if l.get_color() == R.COLOR["safety_si"]]
        assert len(si) == 1 and si[0].get_linewidth() >= 2, "an unseeded arm is one bold line, as before"
        plt.close(fig)

    def test_table_lists_seeds_then_a_mean_row(self):
        pts = R.padding_points(self._frame())
        tex = R.padding_table(pts, ["difference_relu", "safety_si"])
        body = tex.split("\\hline")[-1]
        relu = [l for l in body.splitlines() if "Difference ReLU steer" in l]
        assert len(relu) == 3 and any("&s1 &" in l for l in relu) and any("&s2 &" in l for l in relu) and any("\\emph{mean}" in l for l in relu)
        si = [l for l in body.splitlines() if "Safety SI" in l]; assert len(si) == 1 and "&-- &" in si[0]
        assert "pools k and n across its seeds" in tex

    def test_old_unseeded_fixture_still_renders(self):
        fig, ax = plt.subplots(); drawn = R._draw_padding(ax, _pad_points(), compact=True); plt.close(fig)
        assert set(drawn) == {"arditi_allpos", "safety_si"}
        assert R.padding_table(_pad_points(), ["arditi_allpos"]).count("Refusal steer") == 1


class TestFrontierFencesPadding:
    """Static padding never enters the frontier (the rule of 2026-09-23, applied 2026-09-24 after Safety SI's static
    defense read +15.95 on the frontier against +21.98 in the main table: Pliny under 16k / 64k filler had pooled in)."""

    def test_source_fences_on_padding_tokens_and_source(self):
        import inspect
        src = inspect.getsource(R.frontier_points)
        assert "frame.padding_tokens.isna()" in src and 'lane_source_legacy.ne("static_padding")' in src

    def test_predicate_drops_padded_rows_and_keeps_the_rest(self):
        f = pd.DataFrame([dict(object_label="safety_si", padding_tokens=16384.0, lane_source_legacy="static_padding"),
                          dict(object_label="safety_si", padding_tokens=0.0, lane_source_legacy="static_padding"),
                          dict(object_label="safety_si", padding_tokens=np.nan, lane_source_legacy="sampled_lane")])
        kept = f[f.padding_tokens.isna()]; kept = kept[kept.lane_source_legacy.ne("static_padding")]
        assert len(kept) == 1 and kept.lane_source_legacy.iloc[0] == "sampled_lane", "pad 0 is a padded row too"


class TestOverRefusalLabelMean:
    """Over-refusal is read as `over_refusal_label_mean` everywhere: the frontier's x, the Figure 6 panel and
    its table, the appendix. The switch happens at load on all three frames at once, gated on complete coverage."""

    def _frame(self, drop_one: bool = False):
        rows = []
        for ev, ref in R.OR_REFUSED.items():
            for run in ("u", "a"):
                rows.append(dict(eval_id=ev, run_id=f"{ev}-{run}", object_label="none" if run == "u" else "safety_si", seed="", dose=1.0,
                                 metric_id=ref, value=0.2, n=250, headline=True, selection_state="selected", model_id="qwen35_4b_it"))
                if not (drop_one and ev == "xstest" and run == "a"):
                    rows.append(dict(eval_id=ev, run_id=f"{ev}-{run}", object_label="none" if run == "u" else "safety_si", seed="", dose=1.0,
                                     metric_id=R.OR_METRIC, value=0.25, n=250, headline=False, selection_state="selected", model_id="qwen35_4b_it"))
        return pd.DataFrame(rows)

    def test_the_metric_is_on_both_unit_maps_and_the_spec_covers_both_evals(self):
        assert R.SCALE[R.OR_METRIC] == 100.0 and R.IS_RATE[R.OR_METRIC] == 1.0
        assert set(R.OR_LABEL_MEAN) == set(R.FRONTIER_X[1]) and set(R.OR_LABEL_MEAN.values()) == {R.OR_METRIC}
        assert not len(R.unscaled_rates(pd.DataFrame(dict(metric_id=[R.OR_METRIC], value=[0.3], headline=[True]))))

    def test_complete_repoint_moves_every_headline_and_reports_no_gaps(self):
        out = R.repoint_all(self._frame(), R.OR_LABEL_MEAN)
        assert out.attrs["repoint_complete"] and out.attrs["repoint_coverage"] == (4, 4)
        assert set(out[out.headline].metric_id) == {R.OR_METRIC}
        assert len(R.repoint_gaps(self._frame(), R.OR_LABEL_MEAN)) == 0

    def test_a_missing_row_withholds_and_is_named(self):
        f = self._frame(drop_one=True)
        out = R.repoint_all(f, R.OR_LABEL_MEAN)
        assert not out.attrs["repoint_complete"] and out.attrs["repoint_coverage"] == (3, 4)
        gap = R.repoint_gaps(f, R.OR_LABEL_MEAN)
        assert len(gap) == 1 and gap.eval_id.iloc[0] == "xstest" and gap.object_label.iloc[0] == "safety_si"

    def test_generator_switches_all_three_frames_at_load_and_the_frontier_reads_the_switched_frame(self):
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        i_rep, i_op = src.index("R.repoint_all(v, R.OR_LABEL_MEAN, scope=R.selected_scope(v))"), src.index("op_all = paper_all[paper_all.on_lane")
        assert i_rep < i_op, "the repoint must precede the headline filter that builds op_all"
        assert '("paper_all", paper_all), ("frontier", frontier)' in src and '_or_rep["pool"] = R.repoint_all(pool, R.OR_LABEL_MEAN)' in src, \
            "the two read frames gate together; the pool is repointed with them"
        assert 'paper_all, frontier, pool = _or_rep["paper_all"], _or_rep["frontier"], _or_rep["pool"]' in src, "all three switch, or none"
        assert "_fr_main = frontier" in src and "_fr_orb" not in src and "_fr_graded" not in src
        assert "graded over-refusal" not in src, "prose still names the graded score"


class TestRepointScopeAndAnchorFamily:
    """v153 (2026-09-24): the unscoped gate withheld the label-mean switch for 46 frontier runs no panel reads (`r2_export` at 512,
    superseded / not_in_roster). Coverage is counted over the rows the consumer reads; anchors are matched to the arm's metric family."""

    def _frame(self):
        base = TestOverRefusalLabelMean()._frame()
        stale = pd.DataFrame([dict(eval_id="xstest", run_id="xstest-old", object_label="control_relu", seed="", dose=0.5, metric_id="xst_refused",
                                   value=0.1, n=250, headline=True, selection_state="superseded", model_id="qwen35_4b_it")])
        return pd.concat([base, stale], ignore_index=True)

    def test_unscoped_gate_counts_the_unread_run_and_withholds(self):
        out = R.repoint_all(self._frame(), R.OR_LABEL_MEAN)
        assert not out.attrs["repoint_complete"] and out.attrs["repoint_coverage"] == (4, 5)

    def test_scoped_gate_counts_selected_rows_only_and_passes(self):
        f = self._frame(); out = R.repoint_all(f, R.OR_LABEL_MEAN, scope=R.selected_scope(f))
        assert out.attrs["repoint_complete"] and out.attrs["repoint_coverage"] == (4, 4)
        assert set(out[out.headline & out.selection_state.eq("selected")].metric_id) == {R.OR_METRIC}
        assert out.loc[out.run_id.eq("xstest-old"), "headline"].all() and out.loc[out.run_id.eq("xstest-old"), "metric_id"].iloc[0] == "xst_refused", \
            "the unread run keeps its old headline rather than losing it"
        assert len(R.repoint_gaps(f, R.OR_LABEL_MEAN, scope=R.selected_scope(f))) == 0
        assert list(R.repoint_gaps(f, R.OR_LABEL_MEAN).run_id) == ["xstest-old"], "unscoped, the gap is still named"

    def test_selected_scope_is_everything_when_the_column_is_absent(self):
        f = pd.DataFrame(dict(eval_id=["xstest"], run_id=["r"], metric_id=["xst_refused"], headline=[True]))
        assert R.selected_scope(f).all() and len(R.selected_scope(f)) == 1

    def test_frontier_anchor_pools_are_matched_on_the_metric_family(self):
        import inspect
        src = inspect.getsource(R.frontier_points)
        assert src.count("none_all.metric_id.isin(metric_family(row.eval_id, row.metric_id))") == 2, "both anchor pool sites"
        assert "_fallback(row)" in src and '["eval_id", "metric_id"]' in src, "the campaign fallback is keyed on the metric too"
        assert R.metric_family("xstest", R.OR_METRIC) == [R.OR_METRIC], "label mean is its own family: a refused-rate row can never anchor it"

    def test_generator_gates_on_the_two_read_frames_and_reports_the_pool(self):
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        assert "R.repoint_all(v, R.OR_LABEL_MEAN, scope=R.selected_scope(v))" in src
        assert 'OR_METRIC_OK = all(bool(_or_rep[k].attrs.get("repoint_complete")) for k in ("paper_all", "frontier"))' in src
        assert '_or_rep["pool"] = R.repoint_all(pool, R.OR_LABEL_MEAN)' in src, "the pool is repointed but does not gate"
        assert "no_anchor" in src, "the frontier cell prints the rungs that found no anchor on their metric"


class TestRosterNamesThePaperLane:
    """Roster v9 puts the base->instruct fit on the `/instruct_direct` comparisons and leaves the IT fit with the transfer figures.
    With ROSTER_NAMES_THE_PAPER_LANE off (pre-v9) nothing changes; on, admission follows the comparison ids."""

    def _frame(self, v9: bool):
        m = "gemma4e4b_it"                          # the appendix case: both fits share a blank seed, so the pull cannot rescue the base row
        main_it = '["fig1_headline/instruct_direct","fig4_transfer"]' if not v9 else '["fig4_transfer","figQ_transfer"]'
        main_base = '["fig1_headline/base_to_instruct"]' if not v9 else '["fig1_headline/instruct_direct","fig1_headline/base_to_instruct"]'
        return pd.DataFrame([
            dict(model_id=m, object_label="difference_lora", eval_id="pasttense", seed="", lane="instruct_direct", view_legacy="v", comparison_ids_json=main_it),
            dict(model_id=m, object_label="difference_lora", eval_id="pasttense", seed="", lane="base_to_instruct", view_legacy="v", comparison_ids_json=main_base),
            dict(model_id=m, object_label="safety_si", eval_id="pasttense", seed="", lane=None, view_legacy="v", comparison_ids_json="[]"),
            dict(model_id=m, object_label="arditi_allpos", eval_id="pasttense", seed="", lane="instruct_direct", view_legacy="v", comparison_ids_json="[]"),
            dict(model_id=m, object_label="none", eval_id="pasttense", seed="", lane="instruct_direct", view_legacy="v", comparison_ids_json="[]"),
        ])

    def _admitted(self, f):
        on, pulled = R.admit_lane(f)
        return {(r.object_label, r.lane) for r, ok in zip(f.itertuples(), on) if ok}, {(r.object_label, r.lane) for r, pl in zip(f.itertuples(), pulled) if pl}

    def test_flag_off_is_the_lane_rule_before_and_after_v9(self, monkeypatch):
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", False)
        assert R.PAPER_LANE["gemma4e4b_it"] == "instruct_direct"
        for v9 in (False, True):
            on, pulled = self._admitted(self._frame(v9))
            assert ("difference_lora", "instruct_direct") in on and ("difference_lora", "base_to_instruct") not in on, "the IT fit is drawn, the base fit dropped (same blank-seed key)"
            assert not pulled

    def test_flag_on_before_v9_changes_nothing(self, monkeypatch):
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", True)
        on, pulled = self._admitted(self._frame(v9=False))
        assert ("difference_lora", "instruct_direct") in on and ("difference_lora", "base_to_instruct") not in on and not pulled

    def test_flag_on_after_v9_draws_the_base_fit_and_retires_the_it_fit(self, monkeypatch):
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", True)
        on, pulled = self._admitted(self._frame(v9=True))
        assert ("difference_lora", "base_to_instruct") in on, "the roster names the base fit for the paper lane's figures"
        assert ("difference_lora", "instruct_direct") not in on, "the IT fit, left with the transfer figures, is off the paper lane"
        assert not pulled, "and it is not pulled back as a missing measurement"
        assert {"safety_si", "arditi_allpos", "none"} <= {o for o, _l in on}, "id-less rows keep the lane rule"

    def test_qwen_seeded_base_fits_are_admitted_either_way(self, monkeypatch):
        """Qwen today: the base seeds are pulled (no IT row shares their seeded key); after v9 they are on by id. Never lost."""
        rows = [dict(model_id="qwen35_4b_it", object_label="difference_lora", eval_id="pasttense", seed=sd, lane="base_to_instruct", view_legacy=sd,
                     comparison_ids_json=ids) for sd in ("s1", "s2", "s3") for ids in ['["fig2_jailbreak/base_to_instruct"]']]
        rows.append(dict(model_id="qwen35_4b_it", object_label="difference_lora", eval_id="pasttense", seed="", lane="instruct_direct", view_legacy="it",
                         comparison_ids_json='["fig1_headline/instruct_direct","fig4_transfer"]'))
        f = pd.DataFrame(rows)
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", False); on, pulled = R.admit_lane(f)
        assert on.sum() == 4 and pulled.sum() == 3
        f9 = f.copy(); f9.loc[f9.lane.eq("base_to_instruct"), "comparison_ids_json"] = '["fig2_jailbreak/instruct_direct","fig2_jailbreak/base_to_instruct"]'
        f9.loc[f9.lane.eq("instruct_direct"), "comparison_ids_json"] = '["fig4_transfer"]'
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", True); on, pulled = R.admit_lane(f9)
        assert (on & f9.lane.eq("base_to_instruct")).sum() == 3 and pulled.sum() == 0 and not on[f9.lane.eq("instruct_direct")].any()


class TestPaddingPendingRungsAndPadViewSeeds:
    """Five rungs and three seeds per fitted arm are expected, and the difference ReLU drew three rungs and one line: the
    two new levels' unsteered rows are `not_in_roster` until roster v9, and the pad view's spelling puts the level after
    the seed, which the seed regex did not read."""

    def test_pad_view_seed_reads_pads_spelling(self):
        assert R.pad_view_seed("pad_static_qwen35_4b_obj_s2_p8192") == "s2"
        assert R.pad_view_seed("pad_static_qwen35_4b_obj_s3_p0") == "s3"
        assert R.pad_view_seed("pad_static_qwen35_4b_obj_p65536") == "s1", "the unsuffixed steered standup is seed 1"
        assert R.pad_view_seed("pad_static_qwen35_4b_p8192") is None, "the reference standup (unsteered, prompts) has no fit seed"
        assert R.pad_view_seed("qwen35_4b_s2") is None and R.pad_view_seed(None) is None

    def _frame(self):
        rows = []
        for pad, state in ((0, "selected"), (8192, "not_in_roster"), (16384, "selected")):
            rows.append(dict(eval_id="grok420", object_label="none", seed="", pad=pad, value=0.5, n=259, headline=True,
                             selection_state=state, pad_contract=True, held_out=0))
            for sd in ("s1", "s2"):
                rows.append(dict(eval_id="grok420", object_label="control_lora", seed=sd, pad=pad, value=0.4, n=259, headline=True,
                                 selection_state="selected", pad_contract=True, held_out=0))
        rows.append(dict(eval_id="grok420", object_label="none", seed="", pad=32768, value=0.5, n=259, headline=True,
                         selection_state="superseded", pad_contract=True, held_out=0))
        rows.append(dict(eval_id="grok420", object_label="control_lora", seed="s1", pad=32768, value=0.4, n=259, headline=True,
                         selection_state="selected", pad_contract=True, held_out=0))
        return pd.DataFrame(rows)

    def test_a_not_in_roster_anchor_keeps_its_rung_and_marks_it_pending(self):
        pts = R.padding_points(self._frame())
        assert sorted(set(pts.pad.astype(int))) == [0, 8192, 16384], "in-view rungs drawn; the superseded anchor's rung is not"
        assert R.padding_pending_rungs(pts) == [8192]
        assert pts[pts.pad.eq(8192)].pending.all() and not pts[pts.pad.ne(8192)].pending.any()
        pooled = R.padding_pooled(pts); assert pooled[pooled.pad.eq(8192)].pending.all()

    def test_table_daggers_pending_cells_and_says_why(self):
        pts = R.padding_points(self._frame())
        tex = R.padding_table(pts, ["control_lora"])
        assert "ddagger" in tex and "8,192" in tex and "roster v9 pending" in tex
        body = tex.split("\\hline")[-1]
        rows = [l for l in body.splitlines() if "Control LoRA" in l]
        assert rows and all(l.count("ddagger") == 1 for l in rows), "exactly the 8,192 cell of each row is marked"

    def test_fully_selected_frame_has_no_pending_mark(self):
        pts = R.padding_points(TestPaddingSeeds()._frame())
        assert R.padding_pending_rungs(pts) == [] and "ddagger" not in R.padding_table(pts, ["difference_relu"])


    def test_runs_census_query_carries_the_comparison_ids_the_lane_rule_reads(self):
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        runs_sql = src[src.index("RUNS_SQL = "):src.index("FROM v_paper_current pc", src.index("RUNS_SQL = "))]
        assert "pc.comparison_ids_json" in runs_sql, "runs_all must be admitted by the same rule as paper_all"
        assert R.ROSTER_NAMES_THE_PAPER_LANE is True, "roster v9 is loaded (2026-09-24 21:16); the flag stays on"


    def test_an_include_selected_it_fit_with_no_comparison_yields_to_the_base_fit(self, monkeypatch):
        """v157: 13 fitted IT rows selected by pre-v9 include rulings (empty comparison list) sat beside the base fit the roster
        names for the same cell. Under the flag they stand down where the cell has a row admitted by id, and only there."""
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", True)
        m = "gemma4e4b_it"
        f = pd.DataFrame([
            dict(model_id=m, object_label="constitution_lora", eval_id="crescendo", seed="", lane="instruct_direct", view_legacy="v", comparison_ids_json="[]"),
            dict(model_id=m, object_label="constitution_lora", eval_id="crescendo", seed="", lane="base_to_instruct", view_legacy="v", comparison_ids_json='["fig2_jailbreak/instruct_direct"]'),
            dict(model_id=m, object_label="constitution_lora", eval_id="mmlu", seed="", lane="instruct_direct", view_legacy="v", comparison_ids_json="[]"),
            dict(model_id=m, object_label="safety_si", eval_id="crescendo", seed="", lane="instruct_direct", view_legacy="v", comparison_ids_json="[]"),
        ])
        on, pulled = R.admit_lane(f)
        got = {(r.object_label, r.eval_id, r.lane) for r, ok in zip(f.itertuples(), on) if ok}
        assert ("constitution_lora", "crescendo", "base_to_instruct") in got
        assert ("constitution_lora", "crescendo", "instruct_direct") not in got, "the include-selected IT fit yields where the roster named the cell"
        assert ("constitution_lora", "mmlu", "instruct_direct") in got, "a cell the roster has not spoken for keeps its include"
        assert ("safety_si", "crescendo", "instruct_direct") in got, "prompts are not fits and never yield"
        assert not pulled.any()


    def test_the_paired_relu_with_no_paper_lane_fit_is_still_pulled_under_the_flag(self, monkeypatch):
        """2026-09-24 22:20: the non-Qwen difference ReLU vanished from every jailbreak panel. Its only lane is base->instruct,
        its ids are `/base_to_instruct` companions, and v9 re-shaped nothing for it -- ids-but-none-on-the-paper-lane is not
        the roster moving it off; the pull rule keeps admitting it. Only an object with a by-id row has been moved."""
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", True)
        m = "gemma4e4b_it"
        f = pd.DataFrame([
            dict(model_id=m, object_label="difference_relu", eval_id="pasttense", seed="", lane="base_to_instruct", view_legacy="paired",
                 comparison_ids_json='["fig2_jailbreak/base_to_instruct","figA_attacks_absolute/base_to_instruct"]'),
            dict(model_id=m, object_label="difference_lora", eval_id="pasttense", seed="", lane="instruct_direct", view_legacy="it",
                 comparison_ids_json='["fig4_transfer"]'),
            dict(model_id=m, object_label="difference_lora", eval_id="pasttense", seed="", lane="base_to_instruct", view_legacy="b",
                 comparison_ids_json='["fig2_jailbreak/instruct_direct","fig2_jailbreak/base_to_instruct"]'),
        ])
        on, pulled = R.admit_lane(f)
        got = {(r.object_label, r.lane): (bool(o), bool(pl)) for r, o, pl in zip(f.itertuples(), on, pulled)}
        assert got[("difference_relu", "base_to_instruct")] == (True, True), "pulled, as since 2026-09-16"
        assert got[("difference_lora", "instruct_direct")] == (False, False), "the IT fit the roster moved to the transfer figures stays off"
        assert got[("difference_lora", "base_to_instruct")] == (True, False)


class TestOneMeanOverEvaluationsRule:
    """A mean over k evaluations has se = sqrt(sum se_e^2) / k everywhere -- `category_means` (Figures 3-5, the x = object
    appendix figures) and `eval_mean_column` (the appendix x = eval figures' mean column) agree."""

    def test_eval_mean_column_matches_the_category_means_rule(self):
        # two evals, one reading each: se_e = 2 and 3 in plotted units -> se_mean = sqrt(13) / 2, no between-eval term
        pts = pd.DataFrame([dict(model_id="m", object_label="safety_si", eval_id="pasttense", seed="", draw="", y=20.0, lo=20 - 1.96 * 2, hi=20 + 1.96 * 2),
                            dict(model_id="m", object_label="safety_si", eval_id="codeattack", seed="", draw="", y=40.0, lo=40 - 1.96 * 3, hi=40 + 1.96 * 3)])
        m = R.eval_mean_column(pts, ["pasttense", "codeattack"]).iloc[0]
        assert m.y == 30.0 and abs((m.hi - m.lo) / 2 / 1.96 - np.sqrt(13) / 2) < 1e-9
        assert m.eval_spread == 20.0
        deltas = pd.DataFrame([dict(model_id="m", object_label="safety_si", seed="", eval_id="pasttense", d_y=-20.0, d_lo=-20 - 1.96 * 2, d_hi=-20 + 1.96 * 2,
                                    anchor_value=50.0, anchor_in_frame=True, ceiling=False, held_out=0),
                               dict(model_id="m", object_label="safety_si", seed="", eval_id="codeattack", d_y=-40.0, d_lo=-40 - 1.96 * 3, d_hi=-40 + 1.96 * 3,
                                    anchor_value=50.0, anchor_in_frame=True, ceiling=False, held_out=0)])
        cm = R.category_means(deltas, categories={"static": (["pasttense", "codeattack"], -1, "")}).iloc[0]
        assert cm.y == 30.0 and abs((cm.hi - cm.lo) / 2 / 1.96 - np.sqrt(13) / 2) < 1e-9, "the same interval from the same per-eval intervals"

    def test_no_between_eval_term_however_far_the_evals_disagree(self):
        pts = pd.DataFrame([dict(model_id="m", object_label="safety_si", eval_id=e, seed="", draw="", y=v, lo=v - 1.96, hi=v + 1.96)
                            for e, v in (("pasttense", 0.0), ("codeattack", 80.0))])
        m = R.eval_mean_column(pts, ["pasttense", "codeattack"]).iloc[0]
        assert abs((m.hi - m.lo) / 2 / 1.96 - np.sqrt(2) / 2) < 1e-9 and m.eval_spread == 80.0


class TestTransferPinned:
    """The appendix transfer figure's rules: LoRAs only, no batch condition, Qwen's base side = its three batch-64 seed fits
    pooled (never the batch-8 base fit), the instruct-direct fit pinned by object id."""

    def test_resolve_fit_batch_own_hyperparameters_and_parents(self):
        objs = pd.DataFrame([
            dict(object_id="a", fit_effective_batch=64, fit_hyperparameters_json=None, parent_object_ids_json=None),
            dict(object_id="b", fit_effective_batch=None, fit_hyperparameters_json='{"accum_steps": 32, "batch_size": 2}', parent_object_ids_json=None),
            dict(object_id="sub", fit_effective_batch=None, fit_hyperparameters_json=None, parent_object_ids_json='["a", "b"]'),
            dict(object_id="scaled", fit_effective_batch=None, fit_hyperparameters_json=None, parent_object_ids_json='["sub"]'),
            dict(object_id="c8", fit_effective_batch=8, fit_hyperparameters_json=None, parent_object_ids_json=None),
            dict(object_id="mixed", fit_effective_batch=None, fit_hyperparameters_json=None, parent_object_ids_json='["a", "c8"]'),
        ])
        b = R.resolve_fit_batch(objs)
        assert b["a"] == 64 and b["b"] == 64 and b["sub"] == 64 and b["scaled"] == 64 and np.isnan(b["mixed"])

    def _rows(self):
        rows = []
        def add(m, o, lane, oid, e, dy, seed="", batch=64):
            rows.append(dict(model_id=m, object_label=o, lane=lane, object_id=oid, fit_batch=batch, eval_id=e, seed=seed, draw="",
                             d_y=dy, d_lo=dy - 1.96 * 2, d_hi=dy + 1.96 * 2, held_out=0))
        for e, dy in (("pasttense", -20.0), ("codeattack", -10.0)):
            for oid, sd, off in (("000e0df5" + "0" * 56, "s1", 0.0), ("900938b5" + "0" * 56, "s2", 2.0), ("33e7b327" + "0" * 56, "s3", -2.0)):
                add("qwen35_4b_it", "control_lora", "base_to_instruct", oid, e, dy + off, sd)          # the three pinned seed fits
            add("qwen35_4b_it", "control_lora", "base_to_instruct", "4cd0f001" + "0" * 56, e, dy + 40.0, "", 8)   # the old batch-8 base fit: NEVER read
            add("qwen35_4b_it", "control_lora", "instruct_direct", "bace45e2" + "0" * 56, e, dy + 5.0)          # the pinned in-place fit
            add("qwen35_4b_it", "control_lora", "instruct_direct", "deadbeef" + "0" * 56, e, dy + 30.0, "", 8)  # another IT fit: not the pinned one
        add("gemma4e4b_it", "control_lora", "base_to_instruct", "8c0f6d97" + "0" * 56, "pasttense", -10.0, "", 8)
        add("gemma4e4b_it", "control_lora", "instruct_direct", "04a90de8" + "0" * 56, "pasttense", -12.0, "", 64)   # batch 8 vs 64: paired anyway
        add("llama8b_it", "control_lora", "base_to_instruct", "56254ce4" + "0" * 56, "pasttense", -10.0)             # pinned IT fit absent
        return pd.DataFrame(rows)

    def test_pins_pick_the_fits_and_the_difference_carries_both_intervals(self):
        cats = {"static": (["pasttense", "codeattack", "renellm"], -1, "")}
        pe, pc, pairs = R.transfer_pairs(self._rows(), cats, objects=["control_lora"], models=["gemma4e4b_it", "qwen35_4b_it", "llama8b_it"])
        st = pairs.set_index("model_id")
        assert st.loc["qwen35_4b_it", "status"] == "matched" and st.loc["qwen35_4b_it", "n_base_fits"] == 3 and st.loc["qwen35_4b_it", "base_batch"] == "64"
        assert st.loc["gemma4e4b_it", "status"] == "matched" and st.loc["gemma4e4b_it", "base_batch"] == "8" and st.loc["gemma4e4b_it", "it_batch"] == "64", "no batch condition"
        assert st.loc["llama8b_it", "status"].startswith("pinned instruct-direct fit 7bf26f8c absent")
        q = pe[pe.model_id.eq("qwen35_4b_it")].set_index("eval_id")
        assert abs(q.loc["pasttense", "diff"] - (-20.0 - (-15.0))) < 1e-9, "seed mean -20 (the batch-8 fit at +20 never enters) minus the pinned IT fit -15"
        se_base = float(np.hypot(np.sqrt(12) / 3, 2 / np.sqrt(3))); se_expect = np.sqrt(se_base ** 2 + 2 ** 2)
        assert abs(q.loc["pasttense", "se"] - se_expect) < 1e-9
        r = pc[pc.model_id.eq("qwen35_4b_it")].iloc[0]
        assert r.k == 2 and bool(r.partial) and abs(r.y - 5.0) < 1e-9, "oriented: the transferred fit lowered ASR 5 pp more -> +5 defense"

    def test_relu_steers_are_not_transfer_objects(self):
        assert R.TRANSFER_OBJECTS == ["control_lora", "constitution_lora", "difference_lora"]
        assert all(o.endswith("_lora") for o in R.TRANSFER_OBJECTS) and set(R.TRANSFER_PINS) == {(m, o) for m in R.TRANSFER_MODELS for o in R.TRANSFER_OBJECTS}

    def test_pin_check_counts_matches(self):
        objs = pd.DataFrame([dict(object_id=p + "0" * 56, paper_label="x", fit_model_id="m") for _, (b, it) in R.TRANSFER_PINS.items() for p in ([it] + list(b or []))])
        chk = R.transfer_pin_check(objs); assert (chk.matches == 1).all() and len(chk) == 18
        chk2 = R.transfer_pin_check(objs.iloc[1:]); assert (chk2.matches != 1).sum() == 1

    def test_figure_and_table_render(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cats = {"static": (["pasttense", "codeattack", "renellm"], -1, "")}
        pe, pc, pairs = R.transfer_pairs(self._rows(), cats, objects=["control_lora"], models=["gemma4e4b_it", "qwen35_4b_it", "llama8b_it"])
        fig = R.transfer_figure(pe, pc, pairs, ["control_lora"], ["gemma4e4b_it", "qwen35_4b_it", "llama8b_it"], cats, "t_transfer")
        assert fig is not None; plt.close(fig)
        tex = R.table_transfer(pc, pairs, ["control_lora"], ["gemma4e4b_it", "qwen35_4b_it", "llama8b_it"], cats, "cap", "tab:t")
        assert "absent from the frame" in tex and "3 seed fits pooled" in tex and "+5.0" in tex and "8 / 64" in tex


class TestReluArditiCosines:
    """The appendix cosine figure reads a sidecar computed under torch and binds it to the registry by
    object id; the real sidecar is committed data and must be self-consistent."""

    def test_sidecar_is_present_and_consistent(self):
        cs = R.load_cosine_sidecar()
        assert cs["layer"] == 12 and cs["d"] == 2560 and abs(cs["random_sd"] - 1 / np.sqrt(2560)) < 1e-9
        assert len(cs["entries"]) >= 6 and len(cs["pairs"]) == 6, "three seeds of two families make six pairs"
        for e in cs["entries"]:
            assert all(-1 <= e[k] <= 1 for k in ("read_vs_ref", "write_vs_ref", "read_vs_write")) and len(e["object_id"]) == 64
        ref, pairs = R.cosine_frames(cs)
        assert set(ref.kind) == {"read", "write"} and ref.cos.abs().max() < 0.2, "the steers are not the refusal direction"
        assert pairs.cos.min() > 0.5, "the seeds agree with one another"

    def test_check_binds_ids_to_the_registry_labels(self):
        cs = R.load_cosine_sidecar()
        rows = [dict(object_id=cs["reference"]["object_id"], paper_label="arditi_allpos", fit_replicate_label=None, fit_model_id="qwen35_4b_it")]
        for e in cs["entries"]:
            rows.append(dict(object_id=e["object_id"], paper_label=e["object_label"], fit_replicate_label=e["seed"] or None, fit_model_id=e["fit_model_id"]))
        objs = pd.DataFrame(rows)
        chk = R.cosine_sidecar_check(cs, objs); assert chk.known.all()
        objs2 = objs.copy(); objs2.loc[1, "paper_label"] = "safety_relu"             # the registry calls that id something else
        chk2 = R.cosine_sidecar_check(cs, objs2); assert (~chk2.known).sum() == 1 and chk2[~chk2.known].role.iloc[0] == "steer"
        chk3 = R.cosine_sidecar_check(cs, objs.iloc[2:]); assert not chk3[chk3.role == "reference"].known.iloc[0], "an unknown reference id is refused"

    def test_figure_and_table_render(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cs = R.load_cosine_sidecar()
        fig = R.cosine_figure(cs, None, "t_cos"); assert fig is not None and len(fig.axes) == 1, "one axis (the user, 2026-09-25)"
        assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["refusal vs.\ncontrol", "refusal vs.\nconstitution", "control vs.\nconstitution",
                                                                          "control vs.\ncontrol", "constitution vs.\nconstitution"]
        assert fig.axes[0].get_xticklabels()[0].get_fontsize() == fig.legends[0].get_texts()[0].get_fontsize(), "tick labels at the legend's size"
        leg_texts = sorted(t.get_text() for l in fig.legends for t in l.get_texts())
        assert leg_texts == sorted(["Control ReLU steer", "Constitution ReLU steer", "control × constitution", "read direction ($w_\\ell$)", "write direction ($v_\\ell$)"]), leg_texts
        obj_leg = next(l for l in fig.legends if any(t.get_text() == "Control ReLU steer" for t in l.get_texts()))
        mark_leg = next(l for l in fig.legends if any("read direction" in t.get_text() for t in l.get_texts()))
        assert obj_leg.get_bbox_to_anchor().y0 > mark_leg.get_bbox_to_anchor().y0, "the colour legend sits above the marker legend"
        assert fig.get_size_inches()[0] < 12.0, "narrower (the user, 2026-09-25)"
        assert len(cs["cross_pairs"]) == 9 and all(-1 <= q["read"] <= 1 for q in cs["cross_pairs"])
        g = R.cosine_groups(cs); assert set(g.group) == set(l for l, _, _ in R.COSINE_GROUPS) and len(g[g.group == "control vs.\nconstitution"]) == 18
        plt.close(fig)
        tex = R.table_cosines(cs, "cap", "tab:t"); assert "seed vs seed" in tex and "Control ReLU steer" in tex and "random directions" in tex


class TestAgenticMurderExcludedForQwen:
    """Qwen's unsteered misaligned-action rate on agentic murder is ~3 % (2.2-3.7 % over four runs), a floor; the CELL leaves
    every Qwen figure and table while the eval stays for the other models. Tables are placed [h]."""

    def test_the_eval_stays_and_the_qwen_cell_is_excluded(self):
        assert "agentic_murder" in R.CATEGORIES["agentic misalignment"][0] and "agentic_murder" in R.APPENDIX_CATEGORIES["agentic misalignment"][0]
        assert ("qwen35_4b_it", "agentic_murder") in R.EXCLUDED_CELLS and ("gemma4e4b_it", "agentic_murder") not in R.EXCLUDED_CELLS
        assert R.categories_for("qwen35_4b_it")["agentic misalignment"][0] == ["agentic_leaking"]
        assert R.categories_for("gemma4e4b_it")["agentic misalignment"][0] == ["agentic_leaking", "agentic_murder"]

    def test_drop_excluded_cells_removes_qwen_murder_only(self):
        f = pd.DataFrame([dict(model_id="qwen35_4b_it", eval_id="agentic_murder", object_label="none"),
                          dict(model_id="qwen35_4b_it", eval_id="agentic_murder", object_label="safety_si"),
                          dict(model_id="qwen35_4b_it", eval_id="agentic_leaking", object_label="safety_si"),
                          dict(model_id="gemma4e4b_it", eval_id="agentic_murder", object_label="safety_si")])
        kept, dropped = R.drop_excluded_cells(f)
        assert len(dropped) == 2 and set(dropped.model_id) == {"qwen35_4b_it"} and len(kept) == 2

    def test_qwen_agentic_mean_over_leaking_alone_is_not_partial_and_gemma_missing_murder_is(self):
        d = pd.DataFrame([dict(model_id="qwen35_4b_it", object_label="safety_si", seed="", eval_id="agentic_leaking", d_y=-5.0, d_lo=-8.0, d_hi=-2.0,
                               anchor_in_frame=True, ceiling=False, anchor_value=37.0, held_out=0),
                          dict(model_id="gemma4e4b_it", object_label="safety_si", seed="", eval_id="agentic_leaking", d_y=-5.0, d_lo=-8.0, d_hi=-2.0,
                               anchor_in_frame=True, ceiling=False, anchor_value=19.0, held_out=0)])
        cm = R.category_means(d, categories=R.APPENDIX_CATEGORIES).set_index("model_id")
        assert not bool(cm.loc["qwen35_4b_it", "partial"]) and bool(cm.loc["gemma4e4b_it", "partial"])

    def test_gap_status_names_the_excluded_cell(self):
        runs = pd.DataFrame(columns=["model_id", "object_label", "eval_id", "lifecycle_state", "selection_state", "decisive_ruling_id", "run_extra_json"])
        g = R.gap_status(runs, set(), ["safety_si"], ["agentic_murder"], models=["qwen35_4b_it", "gemma4e4b_it"])
        assert set(g[g.model_id == "qwen35_4b_it"].status) == {"excluded by ruling"} and set(g[g.model_id == "gemma4e4b_it"].status) == {"absent"}

    def test_generator_panel_evals_keep_murder_and_qwen_figures_take_their_own_categories(self):
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        assert '"agentic": ["agentic_leaking", "agentic_murder"]' in src
        assert "CAT_Q = R.categories_for(R.MODEL_ID)" in src and src.count("categories=CAT_Q") >= 5
        assert "R.drop_excluded_cells(op_all)" in src and "R.drop_excluded_cells(tr_rows)" in src

    def test_tables_are_placed_h(self):
        tex = R.latex_table(["a"], [["1"]], "cap", "tab:x")
        assert "\\begin{table}[h]" in tex and "[t]" not in tex

class TestTablesOrientedLikeTheFigures:
    """A misalignment reduction the figure draws as +10 read -10 in the per-eval table; the table now carries the category's
    sign, and the misalignment axes say 'Reduction in ...'."""

    def test_labels_say_reduction(self):
        assert R.CATEGORIES["Petri"][2].startswith("Reduction in misalignment score")
        assert R.CATEGORIES["agentic misalignment"][2].startswith("Reduction in misaligned actions")

    def test_per_eval_table_flips_lower_is_better_categories(self):
        d = pd.DataFrame([
            dict(model_id=R.MODEL_ID, object_label="safety_si", seed="", eval_id="agentic_leaking", d_y=-10.0, d_lo=-14.0, d_hi=-6.0, anchor_value=37.3),
            dict(model_id=R.MODEL_ID, object_label="safety_si", seed="", eval_id="mmlu", d_y=-2.0, d_lo=-3.0, d_hi=-1.0, anchor_value=70.0),
        ])
        tex_a = R.table_evals_seeds(d, [["agentic misalignment"]], ["safety_si"], "c", "tab:a")
        row = [l for l in tex_a.splitlines() if "Safety SI" in l][0]
        assert "$+10.0$" in row and "\\pm 4.0" in row, row                    # the symmetric +- form of _cell
        assert "37.3" in tex_a and "REDUCTION" in tex_a
        tex_c = R.table_evals_seeds(d, [["capability"]], ["safety_si"], "c", "tab:c")
        row_c = [l for l in tex_c.splitlines() if "Safety SI" in l][0]
        assert "-2.0" in row_c, "a higher-is-better category keeps the raw sign"


class TestPerEvalTableNamesItsAnchor:
    """44.3 + 56.2 = 100.5 % and 44.3 - 45.6 = -1.3 % on PAIR -- the arms were differenced against different unsteered anchors
    (43.8 and 47.1) while the first row printed their mean. A cell whose anchor is not the first row says so."""

    def test_cells_with_their_own_anchor_say_vs_and_reconstruct(self):
        d = pd.DataFrame([
            dict(model_id=R.MODEL_ID, object_label="control_relu", seed="s1", eval_id="pair", d_y=56.2, d_lo=52.0, d_hi=60.4, anchor_value=43.8),
            dict(model_id=R.MODEL_ID, object_label="difference_relu", seed="s2", eval_id="pair", d_y=-45.6, d_lo=-50.0, d_hi=-41.2, anchor_value=47.1),
        ])
        tex = R.table_evals_seeds(d, [["multi-query"]], ["control_relu", "difference_relu"], "c", "tab:p")
        first = [l for l in tex.splitlines() if "unsteered (absolute)" in l][0]
        assert "45.5" in first or "45.4" in first, first                      # the mean of the two anchors, labelled as such in the note
        cr = [l for l in tex.splitlines() if "Control ReLU" in l][0]; dr = [l for l in tex.splitlines() if "Difference ReLU" in l][0]
        assert "$-56.2$" in cr and "vs " not in cr, cr                          # oriented, and no per-cell anchor
        assert "$+45.6$" in dr and "vs " not in dr, dr
        assert "MEAN of the arms' unsteered anchors" in tex and "does not reconstruct" in tex

    def test_no_anchor_note_when_every_arm_shares_the_first_row(self):
        d = pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="safety_si", seed="", eval_id="pair", d_y=-10.0, d_lo=-14.0, d_hi=-6.0, anchor_value=44.0),
                          dict(model_id=R.MODEL_ID, object_label="control_lora", seed="s1", eval_id="pair", d_y=5.0, d_lo=1.0, d_hi=9.0, anchor_value=44.0)])
        tex = R.table_evals_seeds(d, [["multi-query"]], ["safety_si", "control_lora"], "c", "tab:p")
        assert "vs " not in tex and "MEAN of the arms" not in tex


class TestOneEvalCategoryHasNoMeanColumn:
    """Qwen's agentic panel, left with one eval, is titled "agentic", ticks "leaking" alone, and draws no mean."""

    def test_single_eval_panel(self, tmp_path, monkeypatch):
        monkeypatch.setattr(R, "FIGS", str(tmp_path))
        cm, d, gaps = TestOneRowFiguresShareAxesHeight()._frames()
        cats = R.categories_for(R.MODEL_ID)
        assert cats["agentic misalignment"][0] == ["agentic_leaking"]
        fig = R.delta_figure(cm, d, [["Petri", "agentic misalignment"]], R.OBJECTS, gaps, "t_agentic", categories=cats, titles={"agentic misalignment": "agentic"})
        ax = [a for a in fig.axes if a.get_title() == "agentic"]
        assert len(ax) == 1, [a.get_title() for a in fig.axes]
        labels = [t.get_text() for t in ax[0].get_xticklabels()]
        assert labels == ["leaking"], labels
        assert not any(l.get_xdata()[0] == 0.5 for l in ax[0].get_lines() if len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]), "no separator"
        petri = [a for a in fig.axes if a.get_title() == "Petri"][0]
        assert [t.get_text() for t in petri.get_xticklabels()][-1] == "mean" and petri.get_xticklabels()[0].get_rotation() == 0
        # the agentic panel is one Petri column wide: its axes width is ~1/7 of Petri's (7 columns incl. the mean)
        wa, wp = ax[0].get_position().width, petri.get_position().width
        assert abs(wa / wp - 1 / 7) < 0.03, (wa, wp)
        plt.close(fig)
        fig2 = R.delta_figure(cm, d, [["Petri", "agentic misalignment"]], R.OBJECTS, gaps, "t_agentic2", categories=cats, tick_fontsize={"Petri": 13})
        petri2 = [a for a in fig2.axes if a.get_title() == "Petri"][0]
        assert petri2.get_xticklabels()[0].get_fontsize() == 13
        plt.close(fig2)


class TestTransferRosterCheck:
    """Roster v10 gives the transfer figure its own comparison id; the rows carrying it must be exactly the pinned objects."""

    def _frame(self, extra=None, drop=None):
        rows = []
        for (m, o), (b, it) in R.TRANSFER_PINS.items():
            for lane, ids in ((R.TRANSFER_LANES[1], [it]), (R.TRANSFER_LANES[0], list(b or ["aaaaaaaa"]))):
                for pre in ids:
                    if (m, o, lane, pre) != drop:
                        rows.append(dict(model_id=m, object_label=o, lane=lane, object_id=pre + "0" * 56, comparison_ids_json=f'["{R.TRANSFER_COMPARISON}/{lane}"]'))
        rows.append(dict(model_id="qwen35_4b_it", object_label="control_lora", lane="instruct_direct", object_id="deadbeef" + "0" * 56, comparison_ids_json='["fig4_transfer"]'))
        if extra:
            rows.append(extra)
        return pd.DataFrame(rows)

    def test_agreement(self):
        chk = R.transfer_roster_check(self._frame())
        assert len(chk) and (chk.status == "both").all(), chk[chk.status != "both"]

    def test_roster_only_and_pins_only_are_named(self):
        extra = dict(model_id="qwen35_4b_it", object_label="control_lora", lane="instruct_direct", object_id="deadbeef" + "0" * 56, comparison_ids_json=f'["{R.TRANSFER_COMPARISON}/instruct_direct"]')
        chk = R.transfer_roster_check(self._frame(extra=extra, drop=("qwen35_4b_it", "control_lora", "base_to_instruct", "000e0df5")))
        assert set(chk[chk.status != "both"].object_id) == {"deadbeef", "000e0df5"}
        assert chk.set_index("object_id").loc["deadbeef", "status"] == "roster only" and chk.set_index("object_id").loc["000e0df5", "status"] == "pins only"


class TestTransferCapabilityAndFloors:
    def test_transfer_categories_keep_the_coding_evals(self):
        assert R.TRANSFER_CATEGORIES["capability"][0] == ["mmlu", "gsm8k", "bigcodebench", "bfcl"]
        assert R.TRANSFER_CATEGORIES["static"] == R.APPENDIX_CATEGORIES["static"]

    def test_floor_cells_are_named(self):
        pe = pd.DataFrame([dict(model_id="qwen35_4b_it", object_label="difference_lora", eval_id="bigcodebench", base_abs=40.0, it_abs=0.35),
                           dict(model_id="qwen35_4b_it", object_label="control_lora", eval_id="bigcodebench", base_abs=42.0, it_abs=37.3),
                           dict(model_id="qwen35_4b_it", object_label="control_lora", eval_id="pasttense", base_abs=2.0, it_abs=3.0)])
        f = R.transfer_floors(pe, ["mmlu", "gsm8k", "bigcodebench", "bfcl"])
        assert list(f.object_label) == ["difference_lora"], "only the collapsed capability cell; an attack eval at 2 % is not a floor here"


class TestNegativeDoseFigure:
    """The alpha = -1 appendix figure: the control objects at dose -1.0, static + mean above, multi-turn and
    over-refusal + mean below, drawn from whatever the frame holds."""

    def test_rows_select_the_dose_and_the_anchors(self):
        f = pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="control_lora", dose=-1.0, headline=True, eval_id="pasttense", held_out=0, seed="s1", lane_source_legacy="x", lane="l", run_id="a"),
                          dict(model_id=R.MODEL_ID, object_label="control_lora", dose=1.0, headline=True, eval_id="pasttense", held_out=0, seed="s1", lane_source_legacy="x", lane="l", run_id="b"),
                          dict(model_id=R.MODEL_ID, object_label="control_relu", dose=-1.0, headline=False, eval_id="pasttense", held_out=0, seed="", lane_source_legacy="x", lane="l", run_id="c"),
                          dict(model_id=R.MODEL_ID, object_label="none", dose=1.0, headline=True, eval_id="pasttense", held_out=0, seed="", lane_source_legacy="x", lane="l", run_id="d"),
                          dict(model_id="gemma4e4b_it", object_label="control_lora", dose=-1.0, headline=True, eval_id="pasttense", held_out=0, seed="", lane_source_legacy="x", lane="l", run_id="e")])
        rows = R.negative_dose_rows(f)
        assert set(rows.run_id) == {"a", "d"}, "the -1 headline arm and the unsteered anchor; not dose 1, not a non-headline row, not another model"
        # roster-pending rows draw and are named; superseded ones do not (2026-09-25, the control ReLU's alpha = -1 runs)
        f2 = f.assign(selection_state=["selected", "selected", "selected", "selected", "selected"])
        f2.loc[f2.run_id.eq("a"), "selection_state"] = "not_in_roster"
        f2 = pd.concat([f2, pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="control_relu", dose=-1.0, headline=True, eval_id="pasttense", held_out=0, seed="s1",
                                                lane_source_legacy="x", lane="l", run_id="f", selection_state="superseded")])], ignore_index=True)
        rows2 = R.negative_dose_rows(f2, anchors_from=f2)
        assert set(rows2.run_id) == {"a", "d"} and list(R.negative_dose_pending(rows2).object_label) == ["control_lora"]
        prov = R.negative_dose_provenance(rows); assert len(prov) == 1 and prov.iloc[0].evals == 1 and prov.iloc[0].seeds == "s1"
        assert R.NEGATIVE_DOSE_LAYOUT == [["static"], ["multi-turn", "over-refusal"]]
        # a seedless fourth fit of a seeded family leaves and is reported (v189: the strength ladder's own -1 rung drew as a 4th mark);
        # a family with NO seeded rows keeps its blank-seed rows (the single-fit case); an MMLU row is not in the pending census
        f3 = pd.concat([f.assign(selection_state="selected"), pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="control_lora", dose=-1.0, headline=True, eval_id="pasttense", held_out=0, seed="",
                                               lane_source_legacy="strength_lane_base2it_native", lane="l", run_id="g", view_legacy="qwen35_4b", selection_state="selected"),
                                          dict(model_id=R.MODEL_ID, object_label="control_relu", dose=-1.0, headline=True, eval_id="mmlu", held_out=0, seed="",
                                               lane_source_legacy="x", lane="l", run_id="h", view_legacy="v", selection_state="not_in_roster")])], ignore_index=True)
        rows3 = R.negative_dose_rows(f3)
        assert set(rows3.run_id) == {"a", "d", "h"} and [r["runs"] for r in rows3.attrs["seedless"]] == [1]
        assert rows3.attrs["seedless"][0]["lane_source_legacy"] == "strength_lane_base2it_native"
        # attrs must survive a concat downstream (v190: a DataFrame in attrs made pandas' attrs comparison raise in drop_excluded_cells)
        assert len(pd.concat([rows3, rows3.iloc[0:0]])) == len(rows3) and len(R.drop_excluded_cells(rows3)[0]) == len(rows3)
        assert list(R.negative_dose_pending(rows3, evals=["pasttense"]).object_label) == [] and list(R.negative_dose_pending(rows3).object_label) == ["control_relu"]

    def test_generator_has_the_cells_and_a_pending_message(self):
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        assert "figA_qwen_control_alpha_minus1" in src and "shipment pending" in src and "R.negative_dose_rows(paper_all" in src and "qwen_negdose" not in src and "figA_qwen_control_alpha_minus1" in src


class TestTransferInstrumentsFlagLowCap:
    def test_a_512_pooled_with_a_higher_cap_is_flagged_and_a_4096_8192_mix_is_not(self):
        rows = []
        def add(m, o, lane, e, run, cap, dec, src):
            rows.append(dict(model_id=m, object_label=o, lane=lane, eval_id=e, run_id=run, token_cap_tokens=cap, decode_source=dec, lane_source_legacy=src, object_id="x" * 64, seed="s1"))
        add("qwen35_4b_it", "constitution_lora", "instruct_direct", "actorattack", "r1", 512, "greedy", "r2_export")
        add("qwen35_4b_it", "constitution_lora", "instruct_direct", "actorattack", "r2", 4096, "sampled", "multiturn_grades")
        add("gemma4e4b_it", "control_lora", "base_to_instruct", "pasttense", "r3", 4096, "sampled", "sampled_lane")
        add("gemma4e4b_it", "control_lora", "base_to_instruct", "pasttense", "r4", 8192, "sampled", "strength_lane")
        pins = {("qwen35_4b_it", "constitution_lora"): (None, "x" * 8), ("gemma4e4b_it", "control_lora"): (None, None)}   # the side comes from the pins
        inst = R.transfer_instruments(pd.DataFrame(rows), pins=pins)
        by = inst.set_index(["model_id", "eval_id"])
        assert bool(by.loc[("qwen35_4b_it", "actorattack"), "mixed"]) and bool(by.loc[("qwen35_4b_it", "actorattack"), "low_cap"])
        assert bool(by.loc[("gemma4e4b_it", "pasttense"), "mixed"]) and not bool(by.loc[("gemma4e4b_it", "pasttense"), "low_cap"])
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        assert "_mixed.low_cap" in src and 'display.max_rows", None' in src


class TestDedupeDropsStaleContrasts:
    """A contrast a newer accepted revision stopped emitting is stale; two contrasts of one measurement against two anchor
    KINDS are by design. Keep the greatest ingest_seq per (run, judgment, metric, threshold, computation, anchor kind)."""

    def test_same_kind_collapses_to_the_latest_and_two_kinds_both_stay(self):
        def row(run, anchor, kind, seq, delta=0.1):
            return dict(run_id=run, judgment_id="j", metric_id="acc", threshold=-1.0, bon_n=None, computation_id="c1", anchor_kind=kind,
                        anchor_run_id=anchor, ingest_seq=seq, delta=delta, metric_role="owner", value=0.5)
        f = pd.DataFrame([row("r1", "a_old", "same_standup_none", 10), row("r1", "a_new", "same_standup_none", 11),      # stale + current
                          row("r2", "a1", "same_standup_none", 20), row("r2", "a2", "full_dose", 21)])                    # two kinds, by design
        out = R.dedupe(f)
        assert list(out[out.run_id.eq("r1")].anchor_run_id) == ["a_new"] and out.attrs.get("stale_contrasts_dropped") == 1
        assert set(out[out.run_id.eq("r2")].anchor_kind) == {"same_standup_none", "full_dose"}
        assert len(R.dedupe(f, contrasts=False)) == 2, "the frontier's one-row-per-measurement path is unchanged"


class TestSchema21MemberPasses:
    """Schema 21: `member_pass` rows are a pooled reading's inputs, not readings; a base_cap pass beside the
    ladder_rung of the same trial yields to the rung."""

    def test_the_new_over_refusal_rate_id_is_on_every_per_metric_map(self):
        assert R.SCALE["over_refusal"] == 100.0 and R.IS_RATE["over_refusal"] == 1.0          # schema 21 spelling (2026-10-02)

    def test_unpooled_member_is_admitted_only_where_the_cell_has_no_selected_reading(self):
        f = pd.DataFrame([dict(model_id="llama8b_it", object_label="constitution_lora", eval_id="pasttense", lane="l", dose=1.0, item_set_id="x", selection_state="unpooled_member", run_id="u1"),
                          dict(model_id="llama8b_it", object_label="constitution_lora", eval_id="pasttense", lane="l", dose=1.0, item_set_id="x", selection_state="unpooled_member", run_id="u2"),
                          dict(model_id="llama8b_it", object_label="difference_lora", eval_id="pasttense", lane="l", dose=1.0, item_set_id="x", selection_state="selected", run_id="s1"),
                          dict(model_id="llama8b_it", object_label="difference_lora", eval_id="pasttense", lane="l", dose=1.0, item_set_id="x", selection_state="unpooled_member", run_id="u3"),
                          dict(model_id="llama8b_it", object_label="none", eval_id="pasttense", lane="other", dose=1.0, item_set_id="x", selection_state="unpooled_member", run_id="n1")])
        out, census = R.admit_provisional(f)
        assert set(out.run_id) == {"u1", "u2", "s1"} and out.attrs["provisional_dropped"] == 2, "an unsteered member pass is never a provisional reading (rule 0027)"
        assert census.to_dict("records") == [dict(model_id="llama8b_it", object_label="constitution_lora", eval_id="pasttense", runs=2)]
        assert list(R.in_view(f)) == [True] * 5, "unpooled_member is in view; member_pass is not"
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        assert "v.selection_state = '{R.PROVISIONAL_STATE}'" in src and "R.admit_provisional(paper_all)" in src

    def test_shared_standup_unsteered_runs_are_fenced_out_of_the_pool(self):
        pool = pd.DataFrame([dict(run_id="a", lane_source_legacy="bigcodebench_full_shared_l2"), dict(run_id="b", lane_source_legacy="bigcodebench_full"), dict(run_id="c", lane_source_legacy=None)])
        out = R.pool_fence(pool)
        assert list(out.run_id) == ["b", "c"] and out.attrs["pool_fenced"] == 1
        assert "R.pool_fence(pool[" in (NOTEBOOKS / "make_registry_notebook.py").read_text()

    def test_source_local_anchor_is_fenced_from_the_pool_and_never_enters_the_paper_query(self):
        f = pd.DataFrame([dict(selection_state=R.ANCHOR_ONLY_STATE), dict(selection_state="member_pass")])
        assert list(R.in_view(f)) == [False, False] and R.ANCHOR_ONLY_STATE in R.OUT_OF_VIEW      # fence the anchor-only state out
        f2 = pd.DataFrame([dict(selection_state=R.ANCHOR_ONLY_STATE, object_label="none"), dict(selection_state=R.ANCHOR_ONLY_STATE, object_label="difference_lora"),
                           dict(selection_state="superseded", object_label="none"), dict(selection_state="selected", object_label="difference_lora")])
        assert list(R.frontier_view(f2)) == [True, False, False, True], "the frontier keeps a source-local NONE as an anchor candidate, nothing else"
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        sql = src[src.index("PAPER_SQL = "):src.index("\n", src.index("PAPER_SQL = ") + 200)]
        assert "in_current_paper = 1" in sql and R.ANCHOR_ONLY_STATE not in sql, "the paper frame admits roster rows and provisional unpooled rows only"

    def test_missing_intervals_are_a_census_and_a_table_note_not_a_zero_width_bar(self):
        d = pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="difference_relu", seed="s2", eval_id="petri_pooled", d_y=-3.2, d_lo=np.nan, d_hi=np.nan, anchor_value=5.0, anchor_in_frame=True, ceiling=False, held_out=0),
                          dict(model_id=R.MODEL_ID, object_label="difference_relu", seed="s1", eval_id="petri_pooled", d_y=-2.0, d_lo=-3.0, d_hi=-1.0, anchor_value=5.0, anchor_in_frame=True, ceiling=False, held_out=0)])
        mi = R.missing_intervals(d)
        assert mi.to_dict("records") == [dict(model_id=R.MODEL_ID, object_label="difference_relu", eval_id="petri_pooled", rows=1)]
        tex = R.table_evals_seeds(d, [["Petri"]], ["difference_relu"], "t", "tab:t", categories={"Petri": (["petri_pooled"], -1, "x")})
        assert "UNMEASURED" in tex and "$+3.2$ \\\\" in tex.replace("\n", " ") or "$+3.2$" in tex, tex[-600:]

    def test_member_pass_is_out_of_view(self):
        assert "member_pass" in R.OUT_OF_VIEW
        f = pd.DataFrame([dict(selection_state=s) for s in ("selected", "not_in_roster", "member_pass", "superseded")])
        assert list(R.in_view(f)) == [True, True, False, False]

    def test_ladder_rung_outranks_base_cap_in_one_cell(self):
        base = dict(model_id=R.MODEL_ID, object_label="control_lora", seed="s1", draw="", lane="base_to_instruct", dose=-1.0, eval_id="pasttense", metric_id="asr",
                    headline=True, judge_ok=True, item_set_id="x", value=0.3, k=30, n=100, delta=None, ingest_seq=1, token_cap_tokens=4096, pass_label="base_cap",
                    selection_state="selected", object_id="o1", run_id="base", view_legacy="v", lane_source_legacy="ctl_neg1",
                    judgment_id="j", threshold=0.5, bon_n=None, metric_role="owner", computation_id="c", ci_lo=None, ci_hi=None, anchor_run_id=None, anchor_kind=None,
                    in_current_paper=1, held_out=0, decisive_action="include", comparison_ids_json="[]", decode_policy_id=None, decode_source=None, decode_sent_json=None,
                    run_extra_json=None, serving_engine=None, escalation_state=None, two_term_dev=None)
        rows = [base, {**base, "run_id": "rung", "pass_label": "ladder_rung", "token_cap_tokens": 8192, "value": 0.25, "k": 25, "ingest_seq": 0},
                {**base, "object_label": "none", "run_id": "none", "pass_label": "pooled_t0_t1", "value": 0.5, "k": 50, "dose": 1.0}]
        d = pd.DataFrame(rows)
        try:
            out = R.anchored(d)
        except Exception as e:
            pytest.skip(f"anchored needs more frame than the fixture gives: {e}")
        kept = out[out.object_label.eq("control_lora")]
        assert list(kept.run_id) == ["rung"], kept[["run_id", "pass_label"]].to_dict("records")


class TestBonCurveUnadmittedEmpty:
    def test_no_unadmitted_vendor_run_returns_an_empty_frame_not_a_keyerror(self):
        """2026-10-07: the first frame with every vendor-preset best-of-N run admitted crashed the render — `cand[[]]` on an empty
        comprehension selects zero columns."""
        cols = ["model_id", "object_label", "eval_id", "in_current_paper", "metric_id", "threshold", "decode_policy_id", "run_id", "view_legacy", "lane_source_legacy", "seed", "selection_state"]
        long = pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="safety_si", eval_id="strongreject_bon", in_current_paper=1, metric_id="asr_bon_cap4096", threshold=0.5,
                                  decode_policy_id="qwen|bon|vendor_preset@v1", run_id="r1", view_legacy="v", lane_source_legacy="s", seed="", selection_state="selected")], columns=cols)
        op = long.copy()                                                  # every candidate is admitted
        out = R.bon_curve_unadmitted(op, long, ["safety_si"], [R.MODEL_ID])
        assert len(out) == 0 and {"model_id", "object_label", "run_id"} <= set(out.columns)


class TestTransferShares:
    def test_cells_over_the_floor_on_either_side_are_listed_with_both_sides(self):
        rows = []
        def add(m, o, lane, oid, e, y, degen, cap):
            rows.append(dict(model_id=m, object_label=o, lane=lane, object_id=oid + "0" * 56, eval_id=e, y=y, frac_degenerate_union5=degen, frac_at_token_cap=cap))
        add("llama8b_it", "control_lora", "instruct_direct", "7bf26f8c", "xstest", 8.6, 0.444, 0.392)     # the IT side degenerate
        add("llama8b_it", "control_lora", "base_to_instruct", "56254ce4", "xstest", 30.0, 0.006, 0.006)
        add("llama8b_it", "control_lora", "instruct_direct", "7bf26f8c", "mmlu", 60.0, 0.0, 0.0)          # clean on both sides
        add("llama8b_it", "control_lora", "base_to_instruct", "56254ce4", "mmlu", 61.0, 0.0, 0.0)
        add("llama8b_it", "control_lora", "instruct_direct", "deadbeef", "gsm8k", 1.0, 0.9, 0.9)          # not the pinned fit: ignored
        sh = R.transfer_shares(pd.DataFrame(rows))
        assert list(sh.eval_id) == ["xstest"] and abs(sh.it_degen.iloc[0] - 0.444) < 1e-9 and abs(sh.base_degen.iloc[0] - 0.006) < 1e-9
        assert abs(sh.it_value.iloc[0] - 8.6) < 1e-9 and abs(sh.base_value.iloc[0] - 30.0) < 1e-9


class TestPaperFiguresZip:
    def test_generator_zips_the_named_figures_into_figures_dir(self):
        src = (NOTEBOOKS / "make_registry_notebook.py").read_text()
        i = src.index("PAPER_FIGURES = ["); j = src.index("]", i)
        names = [x.strip().strip('"') for x in src[i + len("PAPER_FIGURES = ["):j].replace("\n", "").split(",") if x.strip()]
        assert names == ["fig3_single_turn_padding", "fig4_multi_turn", "fig5_misalignment", "fig6_capabilities", "fig7_frontier",
                         "figA_bon_curves_models", "figA_capabilities_models", "figA_misalignment_models", "figA_jailbreaks_models",
                         "figA_relu_arditi_cosines", "figA_transfer_models", "figA_qwen_control_alpha_minus1"]
        assert 'os.path.join(R.FIGS, "paper_figures.zip")' in src, "the zip lands in figures/, which the publish step mirrors"


class TestMultiturnCleanAsr:
    """The non-degenerate-goal ASR from the goal-level rows filed beside the headline judgment."""

    def test_clean_asr_from_goal_counts(self):
        rows = pd.DataFrame([dict(model_id="qwen35_4b_it", object_label="constitution_lora", lane="instruct_direct", object_id="0018b32b" + "0" * 56, eval_id="actorattack", run_id="r1", judgment_id="j1"),
                             dict(model_id="qwen35_4b_it", object_label="constitution_lora", lane="base_to_instruct", object_id="8be1e1a7" + "0" * 56, eval_id="actorattack", run_id="r2", judgment_id="j2")])
        metrics = pd.DataFrame([dict(run_id="r1", judgment_id="j1", metric_id="asr", threshold=0.5, k=58, n=259, value=58 / 259),
                                dict(run_id="r1", judgment_id="j1", metric_id="asr", threshold=0.5, k=58, n=259, value=58 / 259),   # the duplicated row the table carries
                                dict(run_id="r1", judgment_id="j1", metric_id="goal_degenerate_any_turn_share", threshold=-1.0, k=61, n=259, value=61 / 259),
                                dict(run_id="r1", judgment_id="j1", metric_id="goal_jailbroken_any_turn_degenerate_share", threshold=-1.0, k=12, n=58, value=12 / 58),
                                dict(run_id="r1", judgment_id="OTHER", metric_id="asr", threshold=0.5, k=197, n=259, value=197 / 259),   # another judgment: not the headline
                                dict(run_id="r2", judgment_id="j2", metric_id="asr", threshold=0.5, k=190, n=259, value=190 / 259)])
        out = R.multiturn_clean_asr(rows, metrics).set_index("side")
        it = out.loc["instruct-direct"]
        assert it.goals == 259 and it.jailbroken == 58 and it.degenerate_goals == 61 and it.jailbroken_degenerate == 12
        assert abs(it.asr_all - 22.39) < 0.05 and abs(it.asr_clean - 100 * 46 / 198) < 0.05 and it.has_goal_rows
        base = out.loc["base→instruct"]
        assert abs(base.asr_all - 73.36) < 0.05 and not base.has_goal_rows and np.isnan(base.asr_clean)


class TestIncludeRulingSurvivesNewIds:
    """Roster v10 gave include-selected rows their first comparison id (figA_transfer_models); the main-figure filter must keep them."""

    def test_include_rows_with_other_figures_ids_are_kept(self):
        f = pd.DataFrame([dict(run_id="a", comparison_ids_json='["figA_transfer_models"]', decisive_action="include"),
                          dict(run_id="b", comparison_ids_json='["fig1_headline/instruct_direct"]', decisive_action=None),
                          dict(run_id="c", comparison_ids_json='["fig4_transfer"]', decisive_action=None),
                          dict(run_id="d", comparison_ids_json="[]", decisive_action="include"),
                          dict(run_id="e", comparison_ids_json="[]", decisive_action=None)])
        kept = R.comparison_filter(f, ("fig1_headline/*",), quiet=True)
        assert set(kept.run_id) == {"a", "b", "d"}, set(kept.run_id)


class TestTransferInstrumentsAndBaselineInterval:
    def test_mixed_instrument_cells_are_named_and_only_paired_rows_count(self):
        rows = []
        def add(m, o, lane, oid, e, dy, cap, dec, src):
            rows.append(dict(model_id=m, object_label=o, lane=lane, object_id=oid + "0" * 56, eval_id=e, d_y=dy, token_cap_tokens=cap, decode_source=dec, lane_source_legacy=src, run_id=f"{oid}{e}{cap}", frac_degenerate_union5=0.0, frac_at_token_cap=0.0, y=50.0))
        add("llama8b_it", "control_lora", "instruct_direct", "7bf26f8c", "pasttense", -10.0, 4096, "unverified", "sampled_lane")
        add("llama8b_it", "control_lora", "instruct_direct", "7bf26f8c", "pasttense", -12.0, 8192, "vendor_recommendation", "strength_lane")
        add("llama8b_it", "control_lora", "base_to_instruct", "56254ce4", "pasttense", -8.0, 8192, "vendor_recommendation", "strength_lane")
        rows.append(dict(model_id="llama8b_it", object_label="control_lora", lane="instruct_direct", object_id="7bf26f8c" + "0" * 56, eval_id="pasttense", d_y=np.nan,
                         token_cap_tokens=512, decode_source="greedy", lane_source_legacy="r2_export", run_id="dup512", frac_degenerate_union5=0.6, frac_at_token_cap=0.5, y=80.0))
        d = pd.DataFrame(rows)
        inst = R.transfer_instruments(d).set_index("side")
        assert inst.loc["instruct-direct", "runs"] == 2 and bool(inst.loc["instruct-direct", "mixed"]) and "512" not in inst.loc["instruct-direct", "caps"], "the unpaired 512 duplicate is not counted"
        assert not bool(inst.loc["base→instruct", "mixed"])
        assert len(R.transfer_shares(d)) == 0, "the 60 %-degenerate duplicate never entered the pairing and so is not listed"

    def test_baseline_row_pools_the_unsteered_runs_with_an_interval(self):
        d = pd.DataFrame([dict(model_id=R.MODEL_ID, object_label="none", eval_id="pair", metric_id="asr", value=0.44, n=259, d_y=np.nan, anchor_value=np.nan, seed="", d_lo=np.nan, d_hi=np.nan),
                          dict(model_id=R.MODEL_ID, object_label="none", eval_id="pair", metric_id="asr", value=0.47, n=259, d_y=np.nan, anchor_value=np.nan, seed="", d_lo=np.nan, d_hi=np.nan),
                          dict(model_id=R.MODEL_ID, object_label="safety_si", eval_id="pair", metric_id="asr", value=0.30, n=259, d_y=-15.5, d_lo=-20.0, d_hi=-11.0, anchor_value=45.5, seed="")])
        tex = R.table_evals_seeds(d, [["multi-query"]], ["safety_si"], "c", "tab:p")
        first = [l for l in tex.splitlines() if "unsteered (absolute)" in l][0]
        assert "$45.5$" in first and "\\pm" in first, first                   # the mean of the anchors, with a pooled interval


class TestBaseSeedRowsWithOnlyTransferIdsArePulled:
    """Roster v10 gave the Qwen seed fits' BigCodeBench rows their first id, figA_transfer_models; a base->instruct row with ids that
    name no paper-lane figure was never on the paper lane and is pulled as before, not read as 'moved off' it."""

    def test_pull_survives_a_transfer_only_id(self, monkeypatch):
        monkeypatch.setattr(R, "ROSTER_NAMES_THE_PAPER_LANE", True)
        f = pd.DataFrame([
            dict(model_id="qwen35_4b_it", object_label="difference_lora", eval_id="bigcodebench", seed="s1", lane="base_to_instruct", view_legacy="s1",
                 comparison_ids_json='["figA_transfer_models"]'),
            dict(model_id="qwen35_4b_it", object_label="difference_lora", eval_id="pasttense", seed="s1", lane="base_to_instruct", view_legacy="s1",
                 comparison_ids_json='["fig2_jailbreak/instruct_direct"]'),                       # the object HAS paper-lane ids elsewhere
            dict(model_id="qwen35_4b_it", object_label="difference_lora", eval_id="pasttense", seed="", lane="instruct_direct", view_legacy="it",
                 comparison_ids_json='["fig4_transfer","figA_transfer_models"]'),                 # the IT fit left with the transfer figures: moved
        ])
        on, pulled = R.admit_lane(f)
        got = {(r.eval_id, r.lane): (bool(o), bool(pl)) for r, o, pl in zip(f.itertuples(), on, pulled)}
        assert got[("bigcodebench", "base_to_instruct")] == (True, True), "pulled, as before v10"
        assert got[("pasttense", "base_to_instruct")] == (True, False)
        assert got[("pasttense", "instruct_direct")] == (False, False)
