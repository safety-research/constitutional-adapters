"""The item-clustered pooled interval (decided 2026-11-19): a cell pooled over several decode draws is clustered by item.

A pooled cell - the unsteered runs every steered delta is measured against, a decode-only condition's dot, a fitted object's
re-decodes, the unsteered model's absolute cell - re-decodes the SAME items, so its interval is built over items (`pooled_ci`): each
item's mean over the draws, then the standard error over items. Pinned here on synthetic data:
  * the estimator: identical draws give one run's interval exactly (binary units, ratio units with several trials, stratified Petri,
    a unit missing in every run), independent decode noise shrinks it as 1 / sqrt(R), the coverage of the 95 % interval holds where
    the old rule (`hypot(sqrt(sum se_r^2) / R, sd / sqrt(R))`) under-covers, disjoint item sets combine as independent runs, a
    weighted mixture of anchor pools matches a brute-force influence sum, the per-stratum correction, the alignment guard, the
    degenerate case and the boundary case of a rate near 0 %;
  * the unit builders (`judge_rules.*_reading_units`) reproduce `judge_rules.reading` on every family and lane rule, and refuse what
    it refuses;
  * the notebook's readers with a fixture unit provider: `anchored` writes the anchor's clustered se on every row (se_D^2 = se_row^2 +
    se_anchor^2, the identity the SI-target figures invert), a one-run anchor is byte-identical to the old rule, `pool_draws`,
    `category_means`, `table_evals_seeds` and `pool_cell` read one interval per cell, a filtered cell pools its remaining members,
    a seed pool counts the shared anchor once, a run without item rows takes the marked identical-draws bound, and units that do not
    reproduce their reading stop the render;
  * a reading is held to its units on every call (a second reading of a run under the same unit key refuses, through `row_units`,
    `pool_se`'s cache and a pooled anchor), and a filed judged reading its item rows do not give takes the marked bound;
  * one fit's runs under two item-set ids of the same items are one cell for `pool_cell` and the appendix panels, as for
    `cell_interval`; the Wilson form keeps a goal's several correlated trials in the bar; a seed pool's anchor term is weighted by
    each eval's share of the category mean, and an anchor mixture by its weights;
The real item readers on three of the committed frame's anchor pools are pinned where the landing gate collects them
(tests/test_pooled_ci_gate.py); the committed paper frame's cells in test_registry_nb_pooled_ci_slow.py."""
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
import judge_rules as JR  # noqa: E402
import pooled_ci as P  # noqa: E402
import registry_nb as R  # noqa: E402


def units(values, b=None, ids=None, stratum=None, present=None) -> P.Units:
    v = np.asarray(values, dtype=float)
    ids = [str(i) for i in (ids if ids is not None else range(len(v)))]
    a = pd.Series(v, index=ids)
    bb = 1.0 if b is None else pd.Series(np.asarray(b, dtype=float), index=ids)
    st = None if stratum is None else pd.Series(list(stratum), index=ids)
    return P.make_units(a, bb, st, present)


def old_rule(ys, ses) -> float:
    """The notebook's pooling until 2026-11-19 (draw_stats / anchor_summary / pool_cell)."""
    ys, ses = np.asarray(ys, float), np.asarray(ses, float)
    n = len(ys)
    return float(np.hypot(np.sqrt((ses ** 2).sum()) / n, np.sqrt(ys.var(ddof=1) / n))) if n > 1 else float(ses[0])


def wilson_se(k, n) -> float:
    lo, hi = P.wilson(k, n)
    return (hi - lo) / 3.92


# ---- T1: identical draws are one reading ----------------------------------------------------------------------------------------
class TestIdenticalDraws:
    rng = np.random.default_rng(0)
    base = (rng.random(259) < 0.3).astype(float)

    @pytest.mark.parametrize("R_", [2, 3, 8])
    def test_binary_units_give_one_runs_interval_in_both_forms(self, R_):
        u = units(self.base)
        for form in P.FORMS:
            one = P.pooled([u], form=form, scale=100.0).se
            assert P.pooled([u] * R_, form=form, scale=100.0).se == pytest.approx(one, rel=1e-12)
        # the Wilson form IS the run's own Wilson interval at its n
        assert P.pooled([u] * R_, form="wilson", scale=100.0).se == pytest.approx(100 * wilson_se(self.base.sum(), 259), rel=1e-12)
        # the old rule was one run's / sqrt(R) for identical draws (the defect)
        own = 100 * wilson_se(self.base.sum(), 259)
        assert old_rule([self.base.mean() * 100] * R_, [own] * R_) == pytest.approx(own / np.sqrt(R_), rel=1e-12)

    def test_ratio_units_with_several_trials(self):
        a = self.rng.integers(0, 4, 100).astype(float)
        b = np.full(100, 3.0)
        u = units(a, b)
        assert P.pooled([u, u, u], form="wilson").se == pytest.approx(P.pooled([u], form="wilson").se, rel=1e-12)
        assert P.pooled([u, u, u], form="direct").se == pytest.approx(np.sqrt(P.cluster_var([u])), rel=1e-12)

    def test_stratified_petri_units(self):
        st = [f"b{i // 10}" for i in range(60)]
        u = units(self.rng.integers(1, 11, 60), ids=[f"b{i // 10}/s{i}" for i in range(60)], stratum=st)
        assert P.pooled([u] * 3, form="direct").se == pytest.approx(P.pooled([u], form="direct").se, rel=1e-12)

    def test_a_unit_missing_identically_in_every_run(self):
        u = units(self.base[:258], ids=range(258))
        assert P.pooled([u, u], form="wilson").se == pytest.approx(P.pooled([u], form="wilson").se, rel=1e-12)


# ---- T2 / T3: independent draws, coverage ----------------------------------------------------------------------------------------
def test_independent_decode_noise_shrinks_as_one_over_root_r():
    rng = np.random.default_rng(1)
    mean_se = {}
    for R_ in (1, 2, 4, 8):
        mean_se[R_] = np.mean([P.pooled([units((rng.random(259) < 0.3).astype(float)) for _ in range(R_)], form="direct").se for _ in range(400)])
    for R_ in (2, 4, 8):
        assert mean_se[R_] / mean_se[1] == pytest.approx(1 / np.sqrt(R_), rel=0.03)


def _coverage(noise: float, sims: int = 300, I: int = 259, R_: int = 3, seed: int = 2) -> tuple:
    rng = np.random.default_rng(seed)
    pop = rng.beta(0.5, 1.2, size=20000)
    truth = pop.mean()
    hit_new = hit_old = 0
    for _ in range(sims):
        q = pop[rng.integers(0, len(pop), I)]
        base = (rng.random(I) < q).astype(float)
        draws = [np.where(rng.random(I) < noise, (rng.random(I) < q).astype(float), base) for _ in range(R_)]
        us = [units(x) for x in draws]
        m = np.mean([x.mean() for x in draws])
        hit_new += abs(m - truth) <= 1.96 * P.pooled(us, form="wilson").se
        hit_old += abs(m - truth) <= 1.96 * old_rule([x.mean() for x in draws], [wilson_se(x.sum(), I) for x in draws])
    return hit_new / sims, hit_old / sims


@pytest.mark.parametrize("noise", [0.0, 1.0])
def test_the_95_interval_covers_and_the_old_rule_did_not_for_identical_draws(noise):
    new, old = _coverage(noise)
    assert 0.92 <= new <= 0.98, (noise, new)
    if noise == 0.0:
        assert old < 0.80, old                      # one run's interval / sqrt(3): the defect being fixed


# ---- T4: disjoint sets, partial overlap, weights, strata ---------------------------------------------------------------------------
def test_disjoint_item_sets_combine_as_independent_runs():
    rng = np.random.default_rng(3)
    a = units((rng.random(100) < 0.4).astype(float), ids=[f"a{i}" for i in range(100)])
    b = units((rng.random(150) < 0.2).astype(float), ids=[f"b{i}" for i in range(150)])
    w = np.array([0.5, 0.5])
    va, vb = P.cluster_var([a]), P.cluster_var([b])
    I = 250
    want = I / (I - 1) * (w[0] ** 2 * va * 99 / 100 + w[1] ** 2 * vb * 149 / 150)      # the global I / (I - 1) correction
    assert P.cluster_var([a, b], w) == pytest.approx(want, rel=1e-12)


def test_partial_overlap_lies_between_identical_and_disjoint():
    rng = np.random.default_rng(4)
    x = (rng.random(200) < 0.4).astype(float)
    same = P.pooled([units(x), units(x)], form="direct").se
    part = P.pooled([units(x[:150], ids=range(150)), units(x[50:], ids=range(50, 200))], form="direct").se
    disj = P.pooled([units(x[:100], ids=[f"a{i}" for i in range(100)]), units(x[100:], ids=[f"b{i}" for i in range(100)])], form="direct").se
    assert disj < part < same * 1.2


def test_a_weighted_mixture_of_three_pools_matches_the_brute_force_influence_sum():
    rng = np.random.default_rng(5)
    runs = [units((rng.random(80) < p).astype(float)) for p in (0.2, 0.3, 0.5, 0.4)]
    w = np.array([1 / 3, 1 / 6, 1 / 6, 1 / 3])
    psi = []
    for u in runs:
        t = u.table
        psi.append(((t.a - t.a.sum() / t.b.sum() * t.b) / t.b.sum()).to_numpy())
    p = (np.vstack(psi).T * w).sum(axis=1)
    assert P.cluster_var(runs, w) == pytest.approx(80 / 79 * (p ** 2).sum(), rel=1e-12)


def test_the_stratified_variance_corrects_each_stratum():
    rng = np.random.default_rng(6)
    ids = [f"b{i // 10}/s{i}" for i in range(60)]
    u = units(rng.integers(1, 11, 60), ids=ids, stratum=[i.split("/")[0] for i in ids])
    want = 0.0
    for s in range(6):
        g = u.table[u.table.stratum == f"b{s}"]
        psi = (g.a - g.a.sum() / g.b.sum() * g.b) / g.b.sum() / 6
        want += 10 / 9 * float((psi ** 2).sum())
    assert P.cluster_var([u]) == pytest.approx(want, rel=1e-12)


# ---- T5 / T6: the alignment guard, the degenerate and boundary cases ---------------------------------------------------------------
def test_runs_declaring_one_item_set_with_disjoint_ids_refuse():
    a = units([1, 0, 1, 0], ids=["g1", "g2", "g3", "g4"])
    b = units([1, 0, 1, 0], ids=["G1", "G2", "G3", "G4"])
    with pytest.raises(P.ItemIdsDisagree):
        P.check_alignment([a, b], ["set", "set"], ["r1", "r2"])
    assert P.check_alignment([a, b], ["set1", "set2"]) == 1.0          # two item sets: no claim to share items


def test_an_unscored_item_is_still_present_for_the_guard():
    a = units([1, 0, 1], ids=["g1", "g2", "g3"], present={"g1", "g2", "g3", "g4"})
    b = units([1, 0, 1, 1], ids=["g1", "g2", "g3", "g4"])
    assert P.check_alignment([a, b], ["s", "s"]) == 1.0


def test_identical_constant_draws_take_one_runs_own_interval_and_differing_ones_refuse():
    z = units(np.full(50, 4.0))
    assert P.pooled([z, z], form="direct", own_se=[0.3, 0.3]).se == 0.3
    with pytest.raises(P.DegenerateCell):
        P.pooled([z, units(np.full(50, 5.0))], form="direct", own_se=[0.3, 0.3])


def test_a_rate_near_zero_neither_collapses_nor_jumps():
    I = 259
    zero = units(np.zeros(I))
    one = np.zeros(I)
    one[7] = 1.0
    all_zero = P.pooled([zero] * 3, form="wilson", scale=100.0).se
    one_hit = P.pooled([zero, zero, units(one)], form="wilson", scale=100.0).se
    assert all_zero == pytest.approx(100 * wilson_se(0, I), rel=1e-12)      # all-zero draws: one run's Wilson interval
    assert 0 < one_hit < all_zero * 1.01 and one_hit > all_zero / 4            # one hit in one draw: a neighbouring interval, no collapse
    direct = P.pooled([zero, zero, units(one)], form="direct", scale=100.0).se
    assert direct < one_hit                                                   # the Wald form's collapse the Wilson form avoids


def test_the_bound_is_the_weighted_own_se_and_unmeasured_without_one():
    assert P.bound_se([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert P.bound_se([1.0, 3.0], [0.25, 0.75]) == pytest.approx(2.5)
    assert np.isnan(P.bound_se([1.0, np.nan]))


# ---- T8: the unit builders reproduce the readings ----------------------------------------------------------------------------------
def _sr_items(rng, goals=40, trials=2, samples=1, unscored=()):
    rows = []
    for g in range(goals):
        for t in range(trials):
            for s in range(samples):
                un = (g, t, s) in unscored
                rows.append(dict(item_id=f"g{g}", metric_id=JR.SR_UNSCORED if un else JR.SR_UNIT, trial_index=t, sample_index=s, turn_index=0,
                                 value=np.nan if un else float(rng.random()), label=None))
    return pd.DataFrame(rows)


def _ratio(ru: JR.ReadingUnits) -> float:
    if ru.stratum is None:
        return float(ru.a.sum() / ru.b.sum())
    return float(np.mean([ru.a[ru.stratum == s].sum() / ru.b[ru.stratum == s].sum() for s in sorted(set(ru.stratum))]))


class TestUnitBuildersReproduceTheReadings:
    rng = np.random.default_rng(7)

    @pytest.mark.parametrize("lane", ["seedrep_cluster", "sampled_lane"])
    def test_strongreject_per_cell_and_per_goal_pooled_lanes(self, lane):
        items = _sr_items(self.rng, goals=120, unscored={(3, 0, 0)})
        ctx = dict(metric_id="asr", threshold=0.5, lane_source_legacy=lane, ci_method="wilson_95", eval_id="pasttense")
        ru = JR.sr_reading_units(ctx, items)
        rd = JR.reading("strongreject", ctx, items)
        assert _ratio(ru) == pytest.approx(rd["value"], abs=1e-12) and ru.b.sum() == pytest.approx(rd["n"])
        assert "g3" in ru.present

    def test_best_of_n_at_n_reads_the_first_n_draws(self):
        items = _sr_items(self.rng, goals=30, trials=1, samples=8)
        ctx = dict(metric_id="asr_at_n4", threshold=0.5, lane_source_legacy="bon_stores", ci_method="wilson_95", eval_id="strongreject_bon")
        assert _ratio(JR.sr_reading_units(ctx, items)) == pytest.approx(JR.reading("strongreject", ctx, items)["value"], abs=1e-12)

    def test_too_many_undetermined_cells_refuse_as_the_reading_refuses(self):
        items = _sr_items(self.rng, goals=20, trials=1, unscored={(g, 0, 0) for g in range(5)})
        ctx = dict(metric_id="asr", threshold=0.99, lane_source_legacy="", ci_method="wilson_95", eval_id="pasttense")
        with pytest.raises(JR.NotDerivable):
            JR.reading("strongreject", ctx, items)
        with pytest.raises(JR.NotDerivable):
            JR.sr_reading_units(ctx, items)

    @pytest.mark.parametrize("lane,metric", [("r2_export", "over_refusal_label_mean"), ("relu_paired", "over_refusal_label_mean"),
                                             ("r2_export", "xst_refused")])
    def test_over_refusal_per_cell_and_per_item_pooled(self, lane, metric):
        labs = list(JR.LABEL_SCORE)
        rows = [dict(item_id=f"i{i}", metric_id="xst_label", trial_index=t, sample_index=0, turn_index=0, value=np.nan,
                     label=labs[int(self.rng.integers(0, 3))], subset="safe" if i < 40 else "unsafe") for i in range(50) for t in range(2)]
        items = pd.DataFrame(rows)
        ctx = dict(metric_id=metric, lane_source_legacy=lane, ci_method="normal_95_exact_item_variance", eval_id="xstest")
        ru = JR.cpi_reading_units(ctx, items)
        assert _ratio(ru) == pytest.approx(JR.reading("over_refusal", ctx, items)["value"], abs=1e-12)
        assert set(ru.a.index) <= {f"i{i}" for i in range(40)}

    def test_strength_reads(self):
        labs = list(JR.LABEL_SCORE)
        reads = pd.DataFrame(dict(item_id=[f"i{i}" for i in range(30)], at_cap=[i % 7 == 0 for i in range(30)],
                                  label=[labs[i % 3] for i in range(30)], state=["judged"] * 28 + ["censored"] * 2))
        for metric in ("over_refusal_label_mean", "over_refusal_graded", "xst_refused"):
            ctx = dict(metric_id=metric)
            assert _ratio(JR.strength_reading_units(ctx, reads)) == pytest.approx(JR.strength_reading(ctx, reads)["value"], abs=1e-12)

    def test_petri_behaviour_and_pooled_scores(self):
        beh = [f"petri_b{j}" for j in range(6)]
        scores = {f"{b}/s{i}#0#0": float(self.rng.integers(1, 11)) for b in beh for i in range(10)}
        scores["petri_b0/s3#0#0"] = np.nan
        one = {k: v for k, v in scores.items() if k.startswith("petri_b1/")}
        scored = lambda d: {k: v for k, v in d.items() if v == v}
        assert _ratio(JR.petri_reading_units(one, False)) == pytest.approx(R._petri_value(scored(one), False), abs=1e-12)
        ru = JR.petri_reading_units(scores, True)
        assert _ratio(ru) == pytest.approx(R._petri_value(scored(scores), True), abs=1e-12)
        assert "petri_b0/s3" in ru.present and "petri_b0/s3" not in ru.a.index

    def test_rule_scored_both_conventions_and_a_refusal(self):
        items = pd.DataFrame(dict(item_id=[f"q{i}" for i in range(10)], value=[1, 0, 1, 1, np.nan, 0, 1, 1, 0, 1]))
        in_n = JR.rule_reading_units(items, 6 / 10, 10)
        out_n = JR.rule_reading_units(items, 6 / 9, 9)
        assert in_n.b.sum() == 10 and out_n.b.sum() == 9 and "q4" in out_n.present
        with pytest.raises(JR.NotDerivable):
            JR.rule_reading_units(items, 0.5, 10)


# ---- T9 / T10: the notebook's readers with a fixture provider ------------------------------------------------------------------------
SEED_VIEWS = (("s1", "seedrep_battery", "d1/0"), ("qwen35_4b_s2", "seedrep_cluster", "d2/2"), ("qwen35_4b_s3", "seedrep_cluster", "d3/3"))
BASE = dict(model_id=R.MODEL_ID, seed="", draw="", lane=None, dose=1.0, eval_id="pasttense", metric_id="asr", headline=True, judge_ok=True,
            item_set_id="static_pasttense_259x2", value=0.3, k=None, n=120, delta=None, ingest_seq=1, token_cap_tokens=4096, pass_label="base_cap",
            selection_state="selected", object_id="obj", run_id="r", view_legacy="v", lane_source_legacy="seedrep_cluster", judgment_id="j",
            threshold=0.5, bon_n=None, metric_role="owner", computation_id="c", ci_lo=None, ci_hi=None, anchor_run_id=None, anchor_kind=None,
            in_current_paper=1, held_out=0, decisive_action="include", comparison_ids_json="[]",
            decode_policy_id="qwen35_4b_it|static|off|vendor_preset@v1", decode_source="model_card", decode_sent_json=None,
            run_extra_json=None, serving_engine=None, escalation_state=None, two_term_dev=None, judge_id="judge-50095",
            at_operating_point=True, pad=0.0, pad_contract=False)
N_ITEMS = 120


class World:
    """Synthetic runs with per-item outcomes: `rows` for the frame and a provider serving each run's units."""

    def __init__(self):
        self.outcomes, self.rows = {}, []

    def run(self, run_id, outcomes, **kw):
        x = np.asarray(outcomes, dtype=float)
        self.outcomes[run_id] = x
        r = {**BASE, **kw, "run_id": run_id, "value": float(x.mean()), "n": len(x)}
        r["k"] = float(x.sum())
        self.rows.append(r)
        return r

    def provider(self, row):
        x = self.outcomes.get(str(row["run_id"]))
        if x is None:
            raise P.UnitsUnavailable("not in the world")
        return units(x, ids=[f"g{i}" for i in range(len(x))])

    def frame(self):
        return pd.DataFrame(self.rows)


def draws_like(base, flips, rng):
    x = base.copy()
    idx = rng.choice(len(x), size=flips, replace=False)
    x[idx] = 1 - x[idx]
    return x


def static_world(identical=True, seed=8):
    rng = np.random.default_rng(seed)
    w = World()
    anchor = (rng.random(N_ITEMS) < 0.6).astype(float)
    si = (rng.random(N_ITEMS) < 0.2).astype(float)
    for i, (v, src, dr) in enumerate(SEED_VIEWS):
        w.run(f"none:{v}", anchor if identical else draws_like(anchor, 12, rng), object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr)
    for i, (v, src, dr) in enumerate(SEED_VIEWS):
        w.run(f"lora:{v}", (rng.random(N_ITEMS) < 0.4 + 0.05 * i).astype(float), object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct",
              object_id=f"lora{i}", view_legacy=v, lane_source_legacy=src)
    for i, (v, src, dr) in enumerate(SEED_VIEWS):
        w.run(f"si:{v}", si if identical else draws_like(si, 10, rng), object_label="safety_si", object_id="safety_si", view_legacy=v,
              lane_source_legacy=src, draw=dr)
    return w


CAT = {"static": (["pasttense"], -1, "defense")}


@pytest.fixture
def world():
    w = static_world()
    R.reset_pooled_ci(w.provider)
    return w


def test_the_anchor_takes_its_runs_clustered_se_and_every_row_keeps_the_identity(world):
    d = R.anchored(world.frame())
    row = d[d.object_label.eq("control_lora")].iloc[0]
    x = world.outcomes["none:s1"]
    assert row.anchor_draws == 3 and row.anchor_se_method.startswith("item-clustered")
    assert row.anchor_se == pytest.approx(100 * wilson_se(x.sum(), len(x)), rel=1e-12)      # identical draws: one run's interval
    for _, r in d[d.d_y.notna()].iterrows():
        se_d, se_row = (r.d_hi - r.d_lo) / 3.92, (r.hi - r.lo) / 3.92
        assert se_d ** 2 == pytest.approx(se_row ** 2 + r.anchor_se ** 2, rel=1e-10)          # the SI-target figures invert this
    assert R.POOLED_CI_REPORT["pools"] >= 1 and not R.POOLED_CI_REPORT["bound"]


def test_a_one_run_anchor_is_the_old_rule_byte_for_byte():
    w = World()
    rng = np.random.default_rng(9)
    w.run("none:s1", (rng.random(N_ITEMS) < 0.6).astype(float), object_label="none", view_legacy="s1", lane_source_legacy="seedrep_battery")
    w.run("lora:s1", (rng.random(N_ITEMS) < 0.4).astype(float), object_label="control_lora", seed="s1", lane="base_to_instruct", view_legacy="s1",
          lane_source_legacy="seedrep_battery")
    R.reset_pooled_ci(w.provider)
    r = R.anchored(w.frame()).iloc[0]
    a = R.intervals(w.frame()).set_index("run_id")
    se = np.hypot((a.hi - a.lo)["lora:s1"] / 3.92, float(np.sqrt((((a.hi - a.lo)[["none:s1"]] / 3.92) ** 2).sum()) / 1))
    assert (r.d_lo, r.d_hi) == (r.d_y - 1.96 * se, r.d_y + 1.96 * se)
    assert r.anchor_se_method == "one run"


def test_a_dot_of_identical_draws_reads_like_a_single_run_cell(world):
    d = R.anchored(world.frame())
    dot = R.pool_draws(d[d.object_label.eq("safety_si")])
    assert len(dot) == 1 and dot.n_draws.iloc[0] == 3
    x, a = world.outcomes["si:s1"], world.outcomes["none:s1"]
    want = np.hypot(100 * wilson_se(x.sum(), len(x)), 100 * wilson_se(a.sum(), len(a)))
    assert (dot.d_hi.iloc[0] - dot.d_lo.iloc[0]) / 3.92 == pytest.approx(want, rel=1e-10)


def test_every_reader_reads_the_one_cell_interval():
    w = static_world(identical=False)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    si = d[d.object_label.eq("safety_si")]
    c = R.cell_interval(si)
    dot = R.pool_draws(si).iloc[0]
    assert (dot.d_lo, dot.d_hi) == (c.lo, c.hi)
    cm = R.category_means(d, categories=CAT)
    m = cm[cm.object_label.eq("safety_si")].iloc[0]
    assert (m.hi - m.lo) / 3.92 == pytest.approx(c.se, rel=1e-12) and m.y == pytest.approx(-c.dy)
    pc = R.pool_cell(si, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    assert (pc.d_lo, pc.d_hi) == (c.lo, c.hi) and pc.n_pooled == 3
    tex = R.table_evals_seeds(d, [["static"]], ["safety_si"], "c", "tab:x", categories=CAT)
    assert f"{-c.dy:+.1f}" in tex.replace("$", "")


def test_a_filtered_cell_pools_its_remaining_members():
    w = static_world(identical=False)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    si = d[d.object_label.eq("safety_si")]
    two = si.iloc[:2]
    c2 = R.cell_interval(two)
    rows = [r for _, r in two.iterrows()]
    assert c2.se_steer == pytest.approx(R.pool_se(rows).se) and c2.dy == pytest.approx(two.d_y.mean())
    assert R.pool_draws(two).n_draws.iloc[0] == 2


def test_a_seed_pool_counts_the_shared_anchor_once(world):
    """The anchor once (2026-11-19); since 2026-11-21 the seeds' steered runs are item-clustered across the seeds (test_registry_nb_seed_ci.py
    pins that rule): the item term is the pooled se of the three seeds' units, plus the between-fit variance, plus the anchor's se once."""
    d = R.anchored(world.frame())
    cm = R.category_means(d, categories=CAT)
    lora = cm[cm.object_label.eq("control_lora")]
    m = R.pool_seeds(lora).iloc[0]
    rows = d[d.object_label.eq("control_lora")]
    items = P.pooled([world.provider(r) for _, r in rows.iterrows()], [1 / 3] * 3, "wilson", scale=100.0).se
    anchor = float(rows.anchor_se.iloc[0])
    want = np.sqrt(items ** 2 + lora.y.var(ddof=1) / 3 + anchor ** 2)
    assert (m.hi - m.lo) / 3.92 == pytest.approx(want, rel=1e-10)
    old_within = np.sqrt((((lora.hi - lora.lo) / 3.92) ** 2).sum()) / 3          # the anchor divided by three before 2026-11-19
    assert np.sqrt(old_within ** 2 + lora.y.var(ddof=1) / 3) < (m.hi - m.lo) / 3.92
    pc = R.pool_cell(rows, y="d_y", lo="d_lo", hi="d_hi").iloc[0]                     # the appendix's per-eval mark: the same rule
    assert (pc.d_hi - pc.d_lo) / 3.92 == pytest.approx(want, rel=1e-10)


def test_a_dot_against_three_different_one_run_anchors_takes_their_weighted_mixture():
    rng = np.random.default_rng(10)
    w = World()
    anchors = [w.run(f"none:{v}", (rng.random(N_ITEMS) < 0.5).astype(float), object_label="none", view_legacy=v, eval_id="bigcodebench",
                     metric_id="acc", item_set_id="bcb") for v, _s, _d in SEED_VIEWS]
    steer = [w.run(f"si:{v}", (rng.random(N_ITEMS) < 0.45).astype(float), object_label="safety_si", view_legacy=v, eval_id="bigcodebench",
                   metric_id="acc", item_set_id="bcb") for v, _s, _d in SEED_VIEWS]
    R.reset_pooled_ci(w.provider)
    ai = R.intervals(pd.DataFrame(anchors))
    si = R.intervals(pd.DataFrame(steer))
    rows = []
    for (_, a), (_, s) in zip(ai.iterrows(), si.iterrows()):
        R._UNIT_ROWS[R.unit_key(a)] = a
        rows.append({**s.to_dict(), "d_y": s.y - a.y, "d_lo": np.nan, "d_hi": np.nan, "route": "common anchor (1 run)", "anchor_units": R.unit_key(a)})
    c = R.cell_interval(pd.DataFrame(rows))
    us = [w.provider(a) for _, a in ai.iterrows()]
    assert R.combo_se(c.anchor) == pytest.approx(P.pooled(us, [1 / 3] * 3, "wilson", scale=100.0).se, rel=1e-12)
    assert c.se == pytest.approx(np.hypot(P.pooled([w.provider(s) for _, s in si.iterrows()], form="wilson", scale=100.0).se, R.combo_se(c.anchor)), rel=1e-12)


def test_the_unsteered_absolute_cell_pools_every_run_as_a_draw(world):
    pts = R.intervals(world.frame())
    none = pts[pts.object_label.eq("none")]
    pc = R.pool_cell(none)
    assert len(pc) == 1 and pc.n_pooled.iloc[0] == 3
    x = world.outcomes["none:s1"]
    assert (pc.hi.iloc[0] - pc.lo.iloc[0]) / 3.92 == pytest.approx(100 * wilson_se(x.sum(), len(x)), rel=1e-12)


def test_a_run_without_item_rows_takes_the_marked_bound_and_one_without_an_interval_is_unmeasured():
    w = static_world(identical=False)

    def partial(row):
        if str(row["run_id"]) == "none:s1":
            raise P.UnitsUnavailable("no item rows")
        return w.provider(row)

    R.reset_pooled_ci(partial)
    d = R.anchored(w.frame())
    r = d[d.object_label.eq("control_lora")].iloc[0]
    pool = R.intervals(w.frame())
    own = ((pool.hi - pool.lo) / 3.92)[pool.object_label.eq("none")].to_numpy()
    assert r.anchor_se == pytest.approx(own.mean(), rel=1e-12)
    assert r.anchor_se_method.startswith(R.BOUND_NOTE) and r.route == "common anchor (3 runs)"
    assert len(R.POOLED_CI_REPORT["bound"]) == 1
    assert R.BOUND_NOTE in R.pooled_ci_note(d)
    rows = [x for _, x in pool[pool.object_label.eq("none")].iterrows()]
    rows[1] = rows[1].copy()
    rows[1]["lo"] = rows[1]["hi"] = np.nan
    R._POOLS.clear()
    assert np.isnan(R.pool_se(rows).se)


def test_units_that_do_not_reproduce_their_reading_stop_the_render():
    """A reading this notebook re-derived from the very item rows its units read (`judge_source` "<judge> items") is held to them; a
    filed reading in the same position takes the marked bound instead (T11)."""
    w = static_world()
    for r in w.rows:
        r["judge_source"] = "minimax-m3 items"
    R.reset_pooled_ci(lambda row: units(np.zeros(N_ITEMS)) if row["object_label"] == "none" else w.provider(row))
    with pytest.raises(P.UnitsMismatch):
        R.anchored(w.frame())


def test_a_filed_contrast_row_is_unchanged_and_a_dot_without_own_intervals_gains_one():
    w = static_world(identical=False)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    si = d[d.object_label.eq("safety_si")].copy()
    si["lo"] = si["hi"] = np.nan
    si["d_lo"] = si["d_hi"] = np.nan
    one = R.cell_interval(si.iloc[:1])
    assert np.isnan(one.lo)                                                    # a single row keeps its (missing) interval
    c = R.cell_interval(si)
    assert np.isfinite(c.lo) and c.method.startswith("item-clustered")       # the dot's interval comes from its items
    fc = si.iloc[0].copy()
    fc["route"] = "common anchor (3 runs) " + R.FILED_CONTRAST_ROUTE
    fc["d_lo"], fc["d_hi"] = fc.d_y - 2.0, fc.d_y + 2.0
    c1 = R.cell_interval(pd.DataFrame([fc]))
    assert (c1.lo, c1.hi) == (fc.d_lo, fc.d_hi) and c1.anchor == {} and c1.se_steer == pytest.approx(4.0 / 3.92)


def test_a_mean_without_anchor_parts_pools_its_interval_whole():
    """A port's rows without their parts cannot be split: since 2026-11-21 a pool of several such takes the identical-seeds bound over the
    seeds' whole intervals (sum_s se_s / S, never narrower than the truth whatever the seeds' correlation) plus the between term, marked."""
    cm = pd.DataFrame([dict(model_id="m", category="static", object_label="o", seed=s, y=y, lo=y - 1.96, hi=y + 1.96, k=1, partial=False,
                            open=False, ceiling=False, at_cap=np.nan, degen=np.nan, anchors="", evals="e") for s, y in (("s1", 1.0), ("s2", 3.0))])
    m = R.pool_seeds(cm).iloc[0]
    assert (m.hi - m.lo) / 3.92 == pytest.approx(np.sqrt(1.0 + 2.0 / 2), rel=1e-12)          # the whole se once, plus between
    assert m.se_method.startswith(R.SEED_BOUND_NOTE)
    assert np.sqrt(2 / 4 + 2.0 / 2) < (m.hi - m.lo) / 3.92                                   # within + between, as before: the seeds as independent


def test_a_sidecar_registers_the_units_of_the_readings_it_computes(monkeypatch):
    """A reading a sidecar computes from its own payload (the control-pool file's runs; a lane's goal subset) takes its units from the
    same payload: the lane names its rows by `judge_source` and registers a builder (`UNIT_SOURCES`), which `item_units` reads first and
    `unit_key` keys apart from the registry's units of the same run."""
    src = "a lane's goal-subset payload"
    monkeypatch.setitem(R.UNIT_SOURCES, src, lambda row: JR.ReadingUnits(pd.Series([1.0, 0.0, 1.0, 1.0], index=list("abcd")),
                                                                        pd.Series(1.0, index=list("abcd")), None, frozenset("abcd")))
    row = pd.Series(dict(run_id="r1", eval_id="crescendo", metric_id="asr", judgment_id="j", threshold=0.5, bon_n=np.nan, judge_source=src,
                         value=0.75, collapsed=False))
    u = R.item_units(row)
    assert u.value == 0.75 and u.n == 4
    assert R.unit_key(row) != R.unit_key(row.drop("judge_source")) and R.unit_key(row).endswith(src)
    assert R.CTLSTOP_MAIN_SOURCE in R.UNIT_SOURCES


# ---- T11: a reading is held to its units on every call, and a filed reading the source cannot reproduce is marked ------------------
REDERIVED = "minimax-m3 items"                    # `rederive_judged_rows`' judge_source of a reading computed from its item rows


def _rederived_world(identical=False):
    w = static_world(identical=identical)
    for r in w.rows:
        r["judge_source"] = REDERIVED
    return w


def test_a_second_reading_under_a_cached_unit_key_is_refused_on_every_path():
    """A goal subset of a run read under the run's own unit key (no `judge_source` of its own) must not take the full run's units: the
    identity is checked on every call (`row_units`), the pool cache keys on the readings (`pool_se`), so `anchored` refuses a pooled
    anchor that carries the second reading."""
    w = _rederived_world()
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())                                                     # every unit key cached and registered
    none = [r for _, r in R.intervals(w.frame()).iterrows() if r.object_label == "none"]
    sub = none[0].copy()
    sub["value"], sub["n"] = sub["value"] / 2, 46
    with pytest.raises(P.UnitsMismatch):
        R.row_units(sub)                                                          # a cache hit, held to this row's reading
    same_v = none[0].copy()
    same_v["n"] = 46
    with pytest.raises(P.UnitsMismatch):
        R.row_units(same_v)                                                       # the same value over another n is another reading
    with pytest.raises(P.UnitsMismatch):
        R.pool_se([sub] + none[1:])                                               # the cached pool of these keys is not this pool's
    f2 = w.frame()
    f2.loc[f2.run_id.eq("none:s1"), "value"] = f2.loc[f2.run_id.eq("none:s1"), "value"] / 2
    with pytest.raises(P.UnitsMismatch):
        R.anchored(f2)                                                            # its anchor pool holds the second reading
    assert len(d)


def test_a_filed_judged_reading_its_items_do_not_reproduce_takes_the_marked_bound():
    """Under `primary` a judged row carries its producer's filed value (no re-derivation, `judge_source` None): a filed StrongREJECT
    reading of k / 259 whose judgment holds rows for 257 goals in the source cannot be item-clustered, so its pool takes the marked
    identical-draws bound; a re-derived reading in the same position stops the render."""
    w = static_world(identical=False)

    def short(row):
        u = w.provider(row)
        return u if row["object_label"] != "none" or str(row["run_id"]) != "none:s1" else units(w.outcomes["none:s1"][:-2], ids=[f"g{i}" for i in range(N_ITEMS - 2)])

    R.reset_pooled_ci(short)
    d = R.anchored(w.frame())
    r = d[d.object_label.eq("control_lora")].iloc[0]
    assert r.anchor_se_method.startswith(R.BOUND_NOTE)
    (rec,) = R.POOLED_CI_REPORT["bound"].values()
    assert rec["without_items"] == ["none:s1"] and "filed reading" in rec["why"][0]
    w2 = _rederived_world(identical=False)
    R.reset_pooled_ci(lambda row: units(w2.outcomes["none:s1"][:-2], ids=[f"g{i}" for i in range(N_ITEMS - 2)])
                      if str(row["run_id"]) == "none:s1" else w2.provider(row))
    with pytest.raises(P.UnitsMismatch):
        R.anchored(w2.frame())


@pytest.mark.parametrize("src,eid,petri_src,want", [
    ("minimax-m3 items", "pasttense", None, True), (f"minimax-m3 items ({R.SYNTH_SOURCE})", "pasttense", None, True),
    ("minimax-m3 record reads (sidecar)", "xstest", None, True), ("minimax-m3: filed value kept (judge-independent)", "pasttense", None, False),
    (None, "pasttense", None, False), (None, R.PETRI_POOLED, "glm-5.3 items (own items)", True), (None, R.PETRI_POOLED, None, False),
    (R.CTLSTOP_MAIN_SOURCE, "crescendo", None, True)])
def test_which_readings_are_held_to_the_identity(src, eid, petri_src, want):
    row = pd.Series(dict(judge_source=src, eval_id=eid, petri_judge_source=petri_src))
    assert R.reading_rederived(row) is want
    assert R.filed_judged_reading(row) is (not want)
    assert not R.filed_judged_reading(pd.Series(dict(judge_source=None, eval_id="mmlu")))       # rule-scored: held to its (value, n)


# ---- T12: one fit's runs under two item-set ids of the same items are one cell for every reader --------------------------------------
def _two_spellings_world():
    """A companion model's single fit with its multi-turn run filed twice, under two item-set ids of the same 120 goals (identical
    outcomes), against a three-run anchor."""
    rng = np.random.default_rng(13)
    w = World()
    anchor = (rng.random(N_ITEMS) < 0.5).astype(float)
    for v, src, dr in SEED_VIEWS:
        w.run(f"none:{v}", anchor, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id="crescendo", item_set_id="mt_goals")
    x = (rng.random(N_ITEMS) < 0.3).astype(float)
    for i, iset in enumerate(("jailbreak_goals", "multiturn_goals")):
        w.run(f"relu:{iset}", x, object_label="constitution_relu", object_id="relu", eval_id="crescendo", item_set_id=iset, draw=f"d{i}",
              lane="base_to_instruct", view_legacy="s1", lane_source_legacy="seedrep_battery")
    return w


def test_one_fit_under_two_item_set_ids_is_one_cell_for_pool_cell_and_category_panels():
    w = _two_spellings_world()
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("constitution_relu")]
    c = R.cell_interval(fit)                                                     # what category_means and table_evals_seeds read
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    assert (pc.d_hi - pc.d_lo) / 3.92 == pytest.approx(c.se, rel=1e-12) and pc.n_pooled == 2
    x = w.outcomes["relu:jailbreak_goals"]
    assert c.se_steer == pytest.approx(100 * wilson_se(x.sum(), len(x)), rel=1e-12)  # identical runs: one run's interval, never / sqrt(2)
    cp = R.pool_cell(R.pool_draws(fit, item_sets_apart=False), y="d_y", lo="d_lo", hi="d_hi").iloc[0]   # category_panels' path
    assert (cp.d_lo, cp.d_hi) == (pc.d_lo, pc.d_hi)
    assert len(R.pool_draws(fit)) == 2                                            # the fitted draw cell still keeps item sets apart
    pts = R.intervals(w.frame())
    ab = R.pool_cell(pts[pts.object_label.eq("constitution_relu")]).iloc[0]       # the absolute mark: the same cell
    assert (ab.hi - ab.lo) / 3.92 == pytest.approx(100 * wilson_se(x.sum(), len(x)), rel=1e-12)


def test_one_fit_under_two_item_set_spellings_that_share_no_ids_refuses():
    w = _two_spellings_world()
    base = w.provider

    def respelled(row):
        u = base(row)
        if str(row["run_id"]) == "relu:multiturn_goals":
            return units(w.outcomes["relu:multiturn_goals"], ids=[f"G{i}" for i in range(N_ITEMS)])
        return u

    R.reset_pooled_ci(respelled)
    d = R.anchored(w.frame())
    with pytest.raises(P.ItemIdsDisagree):
        R.pool_cell(d[d.object_label.eq("constitution_relu")], y="d_y", lo="d_lo", hi="d_hi")


# ---- T13: the Wilson form clusters a unit of several cells ------------------------------------------------------------------------------
def test_cells_var_is_the_clustered_variance_for_one_cell_units_and_the_binomial_one_for_several():
    rng = np.random.default_rng(14)
    u1 = units((rng.random(200) < 0.3).astype(float))
    assert P.cells_var(u1) == P.cluster_var([u1])                                  # one cell per unit: the same number, bit for bit
    a = rng.binomial(3, 0.3, 200).astype(float)
    u3 = units(a, b=np.full(200, 3.0))
    p, N = a.sum() / 600, 600
    assert P.cells_var(u3) == pytest.approx(N / (N - 1) * p * (1 - p) / N, rel=1e-12)
    with pytest.raises(ValueError):
        P.cells_var(units(np.full(10, 1.5), b=np.full(10, 3.0)))                    # not a count of binary cells


def _coverage_multi_trial(fresh: bool, sims: int = 400, I: int = 259, T: int = 3, R_: int = 3, seed: int = 15) -> float:
    rng = np.random.default_rng(seed)
    pop = rng.beta(0.4, 1.2, size=20000)
    truth = pop.mean()
    hit = 0
    for _ in range(sims):
        q = pop[rng.integers(0, len(pop), I)]
        base = rng.binomial(T, q).astype(float)
        us = [units(base if not fresh else rng.binomial(T, q).astype(float), b=np.full(I, float(T))) for _ in range(R_)]
        m = np.mean([u.value for u in us])
        hit += abs(m - truth) <= 1.96 * P.pooled(us, form="wilson").se
    return hit / sims


@pytest.mark.parametrize("fresh", [False, True])
def test_the_wilson_form_covers_when_each_goal_holds_several_correlated_trials(fresh):
    """Three trials per goal, correlated through the goal: the run's own Wilson interval counts 777 cells as independent, so measuring
    the design effect against the runs' clustered variance (as before) covered about 0.84; against the cells' binomial variance the
    within-goal clustering stays in the bar."""
    assert 0.92 <= _coverage_multi_trial(fresh) <= 0.98


# ---- T14: a seed pool's anchor term, weighted -------------------------------------------------------------------------------------------
def test_a_two_eval_category_seed_pool_counts_each_evals_anchor_once_divided_by_the_evals():
    rng = np.random.default_rng(16)
    w = World()
    for e in ("pasttense", "codeattack"):
        a = (rng.random(N_ITEMS) < 0.6).astype(float)
        for v, src, dr in SEED_VIEWS:
            w.run(f"none:{e}:{v}", np.where(rng.random(N_ITEMS) < 0.1, 1 - a, a), object_label="none", view_legacy=v, lane_source_legacy=src,
                  draw=dr, eval_id=e, item_set_id=e)
        for i, (v, src, dr) in enumerate(SEED_VIEWS):
            w.run(f"lora:{e}:{v}", (rng.random(N_ITEMS) < 0.3 + 0.05 * i).astype(float), object_label="control_lora", seed=f"s{i + 1}",
                  lane="base_to_instruct", object_id=f"lora{i}", view_legacy=v, lane_source_legacy=src, eval_id=e, item_set_id=e)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    cat = {"static": (["pasttense", "codeattack"], -1, "defense")}
    cm = R.category_means(d, categories=cat)
    lora = cm[cm.object_label.eq("control_lora")]
    m = R.pool_seeds(lora).iloc[0]
    rows = d[d.object_label.eq("control_lora")]
    n, k = 3, 2
    items2 = sum(P.pooled([w.provider(r) for _, r in rows[rows.eval_id == e].iterrows()], [1 / (n * k)] * n, "wilson", scale=100.0).se ** 2
                 for e in ("pasttense", "codeattack"))                         # the seeds item-clustered per eval (2026-11-21)
    anchor = rows.groupby("eval_id").anchor_se.first()
    want = np.sqrt(items2 + lora.y.var(ddof=1) / n + ((anchor / k) ** 2).sum())
    assert (m.hi - m.lo) / 3.92 == pytest.approx(want, rel=1e-10)


def test_combo_se_weights_a_non_uniform_mixture_of_anchor_runs():
    rng = np.random.default_rng(17)
    w = World()
    rows = [w.run(f"none:{v}", (rng.random(N_ITEMS) < 0.5).astype(float), object_label="none", view_legacy=v, eval_id="bigcodebench",
                  metric_id="acc", item_set_id="bcb") for v, _s, _d in SEED_VIEWS]
    R.reset_pooled_ci(w.provider)
    keys = []
    for _, a in R.intervals(pd.DataFrame(rows)).iterrows():
        R._UNIT_ROWS[R.unit_key(a)] = a
        keys.append(R.unit_key(a))
    wts = {keys[0]: 0.5, keys[1]: 0.25, keys[2]: 0.25}                             # two seed-matched pools sharing the first run
    us = [w.provider(pd.Series(r)) for r in rows]
    assert R.combo_se(wts) == pytest.approx(P.pooled(us, [0.5, 0.25, 0.25], "wilson", scale=100.0).se, rel=1e-12)
    assert R.combo_se(wts) != pytest.approx(P.pooled(us, None, "wilson", scale=100.0).se, rel=1e-6)
