"""The item-clustered FIT-SEED pool (decided 2026-11-21): a pool of fit seeds is item-clustered across the seeds.

Fit seeds (s1, s2, s3 of one object) are different fitted models read on the SAME items against the SAME anchor. Until this change a
seed pool combined the seeds' steered parts as sqrt(sum se_s^2) / S (their item samples read as independent), so identical seeds took
one seed's interval / sqrt(S). Now (`registry_nb.seed_pool`, `pooled_ci.seed_var`):

    V = V_items + V_fits + V_anchor

V_items the item-clustered variance of every seed's steered runs pooled at once (the cluster is the item, goal or scenario under every
seed), V_fits = s_y^2 / S the between-fit variance (kept: the seeds are a random sample of fits), V_anchor the shared anchor once. Means
do not change. Pinned here on synthetic data, with a fixture unit provider and no registry:
  * the estimator: identical seeds give one seed's interval exactly (binary, several trials, stratified Petri, a unit missing in every
    seed), disjoint item sets give the old within term up to the stated correction, the between-fit term remains, the identity
    B1^2 - old^2 = the seeds' item covariance, coverage, weights for a seed missing an eval and a seed cell of two draws, a Wilson-form
    pool of seeds at different rates against a brute-force influence sum, fits on different numbers of items (the per-cell effective n,
    its coverage, the padding form), a rate near 0 %;
  * the payload units of the control-pool appendix's own rows (`judge_rules.payload_reading_units`) reproduce its reductions;
  * the readers: `pool_seeds`, `pool_cell` (deltas and absolute readings), `category_panels`' path and `eval_mean_column` read one seed
    interval for identical seeds; a mean over evals of per-eval seed pools is ONE pool (the same fits on every eval); one fit is
    byte-identical; a run without units takes the marked bound, a member that cannot be split the opaque bound, fits constant on every
    unit an item term 0; fits whose item ids share nothing refused, the smallest overlap recorded; `appendix_common`, `padding_pooled` and
    `transfer_pairs` (the anchor netted, never paired across different runs; a lane on the bound named in the difference and the table)
    and the tables' notes of those bounds; the live-only judge tools read `pool_cell`.
The real item readers on committed-frame seed pools are pinned where the landing gate collects them (tests/test_pooled_ci_gate.py); the
committed paper frame's marks in test_registry_nb_pooled_ci_slow.py."""
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

Z = P.Z95                                   # the pooled intervals' z (`pooled_ci.wilson`)


# ---- helpers ------------------------------------------------------------------------------------------------------------------------
def units(values, b=None, ids=None, stratum=None, present=None) -> P.Units:
    v = np.asarray(values, dtype=float)
    ids = [str(i) for i in (ids if ids is not None else range(len(v)))]
    a = pd.Series(v, index=ids)
    bb = 1.0 if b is None else pd.Series(np.asarray(b, dtype=float), index=ids)
    st = None if stratum is None else pd.Series(list(stratum), index=ids)
    return P.make_units(a, bb, st, present)


def wilson_se(k, n, z=Z) -> float:
    p = k / n
    den = 1 + z * z / n
    h = z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / den
    return 2 * h / 3.92


def brute_var(tables: list, weights) -> float:
    """The clustered variance of sum_r w_r ybar_r by hand: binary one-cell units {id: outcome} per run, influence (x - ybar_r) / n_r,
    summed per id over the runs, n / (n - 1) over the union of ids."""
    ids = sorted({i for t in tables for i in t})
    psi = np.zeros(len(ids))
    pos = {i: j for j, i in enumerate(ids)}
    for t, w in zip(tables, weights):
        xs = np.array(list(t.values()), dtype=float)
        yb = xs.mean()
        for i, x in t.items():
            psi[pos[i]] += w * (x - yb) / len(t)
    n = len(ids)
    return n / (n - 1) * float((psi ** 2).sum())


def brute_wilson_se(tables: list, weights, scale=100.0) -> float:
    """A Wilson-form pool's se by hand: Wilson at n_eff = sum_r w_r n_r V_r / V over the normalised weights, V_r a run's own variance
    (n / deff, deff = V / sum_r w_r V_r, where the runs share n)."""
    w = np.asarray(weights, dtype=float)
    W = w.sum()
    wn = w / W
    v = brute_var(tables, wn)
    nv = float(sum(x * len(t) * brute_var([t], [1.0]) for x, t in zip(wn, tables)))
    p = float(sum(x * np.mean(list(t.values())) for x, t in zip(wn, tables)))
    n_eff = nv / v
    return wilson_se(p * n_eff, n_eff) * scale * W


def table(x, prefix="g") -> dict:
    return {f"{prefix}{i}": float(v) for i, v in enumerate(x)}


SEED_VIEWS = (("s1", "seedrep_battery", "d1/0"), ("qwen35_4b_s2", "seedrep_cluster", "d2/2"), ("qwen35_4b_s3", "seedrep_cluster", "d3/3"))
BASE = dict(model_id=R.MODEL_ID, seed="", draw="", lane=None, dose=1.0, eval_id="pasttense", metric_id="asr", headline=True, judge_ok=True,
            item_set_id="static_pasttense_259x2", value=0.3, k=None, n=120, delta=None, ingest_seq=1, token_cap_tokens=4096, pass_label="base_cap",
            selection_state="selected", object_id="obj", run_id="r", view_legacy="v", lane_source_legacy="seedrep_cluster", judgment_id="j",
            threshold=0.5, bon_n=None, metric_role="owner", computation_id="c", ci_lo=None, ci_hi=None, anchor_run_id=None, anchor_kind=None,
            in_current_paper=1, held_out=0, decisive_action="include", comparison_ids_json="[]",
            decode_policy_id="qwen35_4b_it|static|off|vendor_preset@v1", decode_source="model_card", decode_sent_json=None,
            run_extra_json=None, serving_engine=None, escalation_state=None, two_term_dev=None, judge_id="judge-50095",
            at_operating_point=True, pad=0.0, pad_contract=False)
N = 120
CAT = {"static": (["pasttense"], -1, "defense")}


class World:
    """Synthetic runs with per-item outcomes: `rows` for the frame and a provider serving each run's units (binary one-cell units by
    default; `strata` a Petri-like stratified score)."""

    def __init__(self):
        self.outcomes, self.rows, self.strata, self.missing = {}, [], {}, set()

    def run(self, run_id, outcomes, strata=None, **kw):
        x = np.asarray(outcomes, dtype=float)
        self.outcomes[run_id] = x
        if strata is not None:
            self.strata[run_id] = list(strata)
        r = {**BASE, **kw, "run_id": run_id, "value": float(x.mean()) if strata is None else float(pd.Series(x).groupby(list(strata)).mean().mean()),
             "n": len(x)}
        r["k"] = float(x.sum())
        self.rows.append(r)
        return r

    def provider(self, row):
        rid = str(row["run_id"])
        x = self.outcomes.get(rid)
        if x is None or rid in self.missing:
            raise P.UnitsUnavailable("not in the world")
        st = self.strata.get(rid)
        ids = [f"{s}/g{i}" for i, s in enumerate(st)] if st is not None else [f"g{i}" for i in range(len(x))]
        return units(x, ids=ids, stratum=st)

    def frame(self):
        return pd.DataFrame(self.rows)


def seed_world(seeds, anchors, eval_id="pasttense", metric_id="asr", item_set="static_pasttense_259x2", obj="control_lora", w=None, **kw):
    """`seeds` the steered outcome arrays (one run per seed, s1, s2, ...), `anchors` the unsteered runs' (one per seed view)."""
    w = w or World()
    for (v, src, dr), a in zip(SEED_VIEWS, anchors):
        w.run(f"none:{eval_id}:{v}", a, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id=eval_id,
              metric_id=metric_id, item_set_id=item_set, **kw)
    for i, x in enumerate(seeds):
        v, src, _dr = SEED_VIEWS[i % 3]
        w.run(f"{obj}:{eval_id}:s{i + 1}", x, object_label=obj, seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"{obj}{i}",
              view_legacy=v, lane_source_legacy=src, eval_id=eval_id, metric_id=metric_id, item_set_id=item_set, **kw)
    return w


def flipped(base, k, rng):
    x = base.copy()
    idx = rng.choice(len(x), size=k, replace=False)
    x[idx] = 1 - x[idx]
    return x


def old_seed_rule(ys, se_steer, anchor) -> float:
    """The seed pool until 2026-11-21: within sqrt(sum se_s^2) / S, between sd / sqrt(S), the anchor once."""
    ys, st = np.asarray(ys, float), np.asarray(se_steer, float)
    S = len(ys)
    return float(np.sqrt((st ** 2).sum() / S ** 2 + ys.var(ddof=1) / S + anchor ** 2))


def mark_se(r, lo="d_lo", hi="d_hi") -> float:
    return float((r[hi] - r[lo]) / 3.92)


@pytest.fixture
def fresh():
    R.reset_pooled_ci(R.no_item_rows)
    yield


# ---- T1: identical seeds sharing items are one seed's interval ------------------------------------------------------------------------
@pytest.mark.parametrize("S", [2, 3, 5])
@pytest.mark.parametrize("kind", ["binary", "trials", "petri", "missing"])
def test_identical_seeds_item_part_is_one_seeds_interval_in_both_forms(S, kind):
    rng = np.random.default_rng(100 + S)
    if kind == "binary":
        u = units((rng.random(259) < 0.3).astype(float))
    elif kind == "trials":
        u = units(rng.integers(0, 4, 120).astype(float), b=np.full(120, 3.0))
    elif kind == "petri":
        st = [f"b{i // 10}" for i in range(60)]
        u = units(rng.integers(1, 11, 60), ids=[f"b{i // 10}/s{i}" for i in range(60)], stratum=st)
    else:
        u = units((rng.random(258) < 0.3).astype(float), ids=range(258))
    w = np.full(S, 1.0 / S)
    for form in P.FORMS:
        if form == "wilson" and kind == "petri":
            continue
        one = P.pooled([u], form=form, scale=100.0).se
        items = P.pooled([u] * S, w, form=form, scale=100.0).se
        assert items == pytest.approx(one, rel=1e-12)
        assert P.seed_var(items ** 2, [u.value] * S) == pytest.approx(one ** 2, rel=1e-12)       # V_fits = 0: one seed's interval
        assert np.sqrt(sum(one ** 2 for _ in range(S))) / S == pytest.approx(one / np.sqrt(S), rel=1e-12)   # the old within term: / sqrt(S)


def test_identical_seeds_through_the_readers_are_one_seeds_interval_plus_the_anchor_once(fresh):
    rng = np.random.default_rng(1)
    x = (rng.random(N) < 0.4).astype(float)
    a = (rng.random(N) < 0.6).astype(float)
    w = seed_world([x, x, x], [a, a, a])
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("control_lora")]
    seed_se, anchor_se = 100 * wilson_se(x.sum(), N), 100 * wilson_se(a.sum(), N)
    want = float(np.hypot(seed_se, anchor_se))
    cm = R.category_means(d, categories=CAT)
    m = R.pool_seeds(cm[cm.object_label.eq("control_lora")]).iloc[0]
    assert (m.hi - m.lo) / 3.92 == pytest.approx(want, rel=1e-12) and m.se_method.startswith(R.SEED_CLUSTERED)
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    assert mark_se(pc) == pytest.approx(want, rel=1e-12)
    cp = R.pool_cell(R.pool_draws(fit, item_sets_apart=False), y="d_y", lo="d_lo", hi="d_hi").iloc[0]      # category_panels' path
    assert (cp.d_lo, cp.d_hi) == (pc.d_lo, pc.d_hi)
    old = old_seed_rule(fit.d_y, (fit.hi - fit.lo) / 3.92, anchor_se)                                       # the defect, pinned
    assert old == pytest.approx(np.sqrt(seed_se ** 2 / 3 + anchor_se ** 2), rel=1e-6) and old < want       # (the rows' own z: 1.959964)
    ab = R.pool_cell(R.intervals(w.frame())[lambda f: f.object_label.eq("control_lora")]).iloc[0]          # the absolute mark: no anchor
    assert (ab.hi - ab.lo) / 3.92 == pytest.approx(seed_se, rel=1e-12)


# ---- T2: independent seeds -------------------------------------------------------------------------------------------------------------
def test_disjoint_item_sets_give_the_old_within_term_up_to_the_stated_correction():
    rng = np.random.default_rng(2)
    sizes = (100, 150, 80)
    us = [units((rng.random(n) < 0.3 + 0.1 * j).astype(float), ids=[f"s{j}_{i}" for i in range(n)]) for j, n in enumerate(sizes)]
    S, Nall = len(us), sum(sizes)
    w = np.full(S, 1.0 / S)
    want = sum(P.cluster_var([u]) * (n - 1) / n for u, n in zip(us, sizes)) / S ** 2 * Nall / (Nall - 1)
    assert P.cluster_var(us, w) == pytest.approx(want, rel=1e-12)
    old_within2 = sum(P.cluster_var([u]) for u in us) / S ** 2
    assert P.cluster_var(us, w) == pytest.approx(old_within2, rel=0.02)          # the old rule, to the per-stratum correction's factor


def test_independent_outcomes_on_one_item_set_give_the_old_rule_in_expectation():
    rng = np.random.default_rng(3)
    S, I = 3, 259
    new, old = [], []
    for _ in range(400):
        us = [units((rng.random(I) < 0.3).astype(float)) for _ in range(S)]
        ys = [u.value for u in us]
        v_items = P.cluster_var(us, np.full(S, 1.0 / S))
        new.append(np.sqrt(P.seed_var(v_items, ys)))
        old.append(np.sqrt(sum(P.cluster_var([u]) for u in us) / S ** 2 + P.fit_var(ys)))
    assert 0.98 <= np.mean(new) / np.mean(old) <= 1.02


# ---- T3: the shared anchor once ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3])
def test_three_seeds_against_one_common_anchor_count_its_se_once(fresh, k):
    rng = np.random.default_rng(10 + k)
    a = (rng.random(N) < 0.6).astype(float)
    seeds = [(rng.random(N) < 0.3 + 0.07 * i).astype(float) for i in range(3)]
    anchors = [flipped(a, 9, rng) for _ in range(k)]
    w = World()
    for j, an in enumerate(anchors):
        w.run(f"none:{j}", an, object_label="none", view_legacy=f"v{j}", draw=f"d{j}")
    for i, x in enumerate(seeds):
        w.run(f"lora:s{i + 1}", x, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"o{i}")
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("control_lora")]
    anchor = float(fit.anchor_se.iloc[0])
    if k > 1:
        assert anchor == pytest.approx(brute_wilson_se([table(x) for x in anchors], [1 / k] * k), rel=1e-12)
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    items = brute_wilson_se([table(x) for x in seeds], [1 / 3] * 3)
    want = np.sqrt(items ** 2 + fit.d_y.var(ddof=1) / 3 + anchor ** 2)
    assert mark_se(pc) == pytest.approx(want, rel=1e-10)


def test_seed_matched_one_run_anchors_take_their_weighted_mixture_once(fresh):
    rng = np.random.default_rng(12)
    w = World()
    anchors = [(rng.random(N) < 0.5).astype(float) for _ in range(3)]
    seeds = [(rng.random(N) < 0.4 + 0.05 * i).astype(float) for i in range(3)]
    for i, ((v, src, dr), an, x) in enumerate(zip(SEED_VIEWS, anchors, seeds)):
        w.run(f"none:{v}", an, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, seed=f"s{i + 1}", eval_id="bigcodebench",
              metric_id="acc", item_set_id="bcb")
        w.run(f"lora:{v}", x, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"o{i}", view_legacy=v,
              lane_source_legacy=src, eval_id="bigcodebench", metric_id="acc", item_set_id="bcb")
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("control_lora")]
    assert set(fit.anchor_draws) == {1}                                              # each seed against its own seed's run
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    items = brute_wilson_se([table(x) for x in seeds], [1 / 3] * 3)
    mix = brute_wilson_se([table(x) for x in anchors], [1 / 3] * 3)                  # the three anchor runs, 1 / 3 each, clustered by item
    assert mark_se(pc) == pytest.approx(np.sqrt(items ** 2 + fit.d_y.var(ddof=1) / 3 + mix ** 2), rel=1e-10)


def test_a_two_eval_category_seed_pool_counts_each_evals_anchor_once_divided_by_the_evals(fresh):
    rng = np.random.default_rng(16)
    w = World()
    seeds_by_eval = {}
    for e in ("pasttense", "codeattack"):
        a = (rng.random(N) < 0.6).astype(float)
        seeds_by_eval[e] = [(rng.random(N) < 0.3 + 0.05 * i).astype(float) for i in range(3)]
        seed_world(seeds_by_eval[e], [np.where(rng.random(N) < 0.1, 1 - a, a) for _ in range(3)], eval_id=e, item_set=e, w=w)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    cm = R.category_means(d, categories={"static": (["pasttense", "codeattack"], -1, "defense")})
    lora = cm[cm.object_label.eq("control_lora")]
    m = R.pool_seeds(lora).iloc[0]
    rows = d[d.object_label.eq("control_lora")]
    k = 2
    items2 = sum(brute_wilson_se([table(x) for x in seeds_by_eval[e]], [1 / (3 * k)] * 3) ** 2 for e in seeds_by_eval)
    anchor = rows.groupby("eval_id").anchor_se.first()
    want = np.sqrt(items2 + lora.y.var(ddof=1) / 3 + ((anchor / k) ** 2).sum())
    assert (m.hi - m.lo) / 3.92 == pytest.approx(want, rel=1e-10)
    st = rows.assign(s=(rows.hi - rows.lo) / 3.92).groupby("seed").s.apply(lambda s: np.sqrt((s ** 2).sum()) / k)
    assert old_seed_rule(lora.y, st, np.sqrt(((anchor / k) ** 2).sum())) != pytest.approx(want, rel=1e-3)   # the old within term


# ---- T4 / T5: the between-fit term remains; the identity with the old rule ----------------------------------------------------------
def test_the_between_fit_term_remains_when_seeds_flip_a_fixed_set_of_items():
    rng = np.random.default_rng(4)
    base = (rng.random(259) < 0.3).astype(float)
    xs = [base.copy() for _ in range(3)]
    for s, x in enumerate(xs):
        x[20 * s:20 * s + 15 * (s + 1)] = 1.0                                       # a different fixed set of items per seed
    us = [units(x) for x in xs]
    ys = [u.value for u in us]
    v_items = P.cluster_var(us, np.full(3, 1 / 3))
    v = P.seed_var(v_items, ys)
    assert v == pytest.approx(v_items + np.var(ys, ddof=1) / 3, rel=1e-12) and v > v_items


def test_b1_squared_minus_the_old_rule_squared_is_the_seeds_item_covariance():
    rng = np.random.default_rng(5)
    for S in (2, 3, 4):
        common = rng.random(200)
        X = [(rng.random(200) < 0.2 + 0.6 * common).astype(float) for _ in range(S)]
        us = [units(x) for x in X]
        ys = [u.value for u in us]
        b1 = P.seed_var(P.cluster_var(us, np.full(S, 1 / S)), ys)
        old = sum(P.cluster_var([u]) for u in us) / S ** 2 + P.fit_var(ys)
        n = 200
        psi = [(x - x.mean()) / n for x in X]
        cov = sum(n / (n - 1) * float(psi[s] @ psi[t]) for s in range(S) for t in range(S) if s != t) / S ** 2
        assert b1 - old == pytest.approx(cov, rel=1e-12, abs=1e-15)
        assert cov > 0                                                             # seeds sharing hard and easy items covary positively


# ---- T6: coverage ------------------------------------------------------------------------------------------------------------------------
def _seed_coverage(identical: bool, I: int, mu: float, tg: float, sims: int = 300, S: int = 3, seed: int = 20261121) -> tuple:
    """Coverage of the 95 % interval of a three-seed mean, y_si ~ Bernoulli(expit(mu + g_i)) with items resampled per study and no fit
    effect, one run per seed (identical: the seeds' outcomes are one table). Deterministic for numpy's default_rng(seed) stream."""
    rng = np.random.default_rng(seed)
    expit = lambda v: 1 / (1 + np.exp(-v))
    truth = float(expit(mu + rng.normal(0, tg, 2_000_000)).mean())
    hit_new = hit_old = 0
    for _ in range(sims):
        p = expit(mu + rng.normal(0, tg, I))
        Y = np.tile((rng.random(I) < p).astype(float), (S, 1)) if identical else (rng.random((S, I)) < p[None, :]).astype(float)
        us = [units(Y[s]) for s in range(S)]
        ys = Y.mean(axis=1)
        m = ys.mean()
        items = P.pooled(us, np.full(S, 1 / S), "wilson").se
        new = np.sqrt(items ** 2 + P.fit_var(ys))
        se_s = [wilson_se(Y[s].sum(), I) for s in range(S)]
        old = np.sqrt(sum(x ** 2 for x in se_s) / S ** 2 + P.fit_var(ys))
        hit_new += abs(m - truth) <= 1.96 * new
        hit_old += abs(m - truth) <= 1.96 * old
    return hit_new / sims, hit_old / sims


def test_coverage_identical_seeds():
    new, old = _seed_coverage(identical=True, I=259, mu=-1.0, tg=1.5)
    assert 0.90 <= new <= 0.99, new
    assert old < 0.80, old                                                         # one seed's interval / sqrt(3): the defect


def test_coverage_near_deterministic_items_no_fit_effect():
    new, old = _seed_coverage(identical=False, I=382, mu=0.8, tg=4.0)
    assert new >= 0.92, new
    assert old < 0.92, old


# ---- T7: weights -------------------------------------------------------------------------------------------------------------------------
def test_a_seed_missing_an_eval_and_a_seed_cell_of_two_draws_against_a_brute_force_influence_sum(fresh):
    rng = np.random.default_rng(7)
    w = World()
    ev = ("pasttense", "codeattack")
    out = {}
    for e in ev:
        a = (rng.random(N) < 0.6).astype(float)
        for v, src, dr in SEED_VIEWS:
            w.run(f"none:{e}:{v}", a, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id=e, item_set_id=e)
    for s in (1, 2, 3):
        for e in ev:
            if s == 3 and e == "codeattack":
                continue                                                         # seed 3 misses the second eval: k_3 = 1
            draws = 2 if (s == 1 and e == "pasttense") else 1                    # seed 1's pasttense cell holds two decode draws
            for r in range(draws):
                x = (rng.random(N) < 0.25 + 0.1 * s).astype(float)
                w.run(f"lora:{e}:s{s}:{r}", x, object_label="control_lora", seed=f"s{s}", lane="base_to_instruct", object_id=f"o{s}",
                      eval_id=e, item_set_id=e, draw=f"t{r}")
                out.setdefault(e, []).append((s, r, draws, x))
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    cm = R.category_means(d, categories={"static": (list(ev), -1, "defense")})
    lora = cm[cm.object_label.eq("control_lora")].sort_values("seed")
    assert list(lora.k) == [2, 2, 1]
    m = R.pool_seeds(lora).iloc[0]
    S = 3
    k_s = {1: 2, 2: 2, 3: 1}
    items2 = 0.0
    for e, runs in out.items():
        tabs = [table(x) for _s, _r, _R, x in runs]
        wts = [1.0 / (S * k_s[s] * R_) for s, _r, R_, _x in runs]
        items2 += brute_wilson_se(tabs, wts) ** 2
    anchor2 = 0.0
    for e in ev:
        aw = sum(1.0 / (S * k_s[s]) for s in (1, 2, 3) if not (s == 3 and e == "codeattack"))
        anchor2 += (aw * float(d[(d.eval_id == e) & (d.object_label == "control_lora")].anchor_se.iloc[0])) ** 2
    want = np.sqrt(items2 + lora.y.var(ddof=1) / S + anchor2)
    assert (m.hi - m.lo) / 3.92 == pytest.approx(want, rel=1e-10)
    assert m.y == lora.y.mean()                                                   # the mean is the seeds' means averaged, as always


def test_a_wilson_form_pool_of_seeds_at_clearly_different_rates_against_a_brute_force_influence_sum(fresh):
    rng = np.random.default_rng(71)
    common = rng.random(N)
    seeds = [(common < q).astype(float) for q in (0.2, 0.4, 0.6)]                 # nested: seeds agree on the hard items
    a = (rng.random(N) < 0.5).astype(float)
    w = seed_world(seeds, [a, a, a])
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("control_lora")]
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    items = brute_wilson_se([table(x) for x in seeds], [1 / 3] * 3)
    want = np.sqrt(items ** 2 + fit.d_y.var(ddof=1) / 3 + float(fit.anchor_se.iloc[0]) ** 2)
    assert mark_se(pc) == pytest.approx(want, rel=1e-10)
    assert pc.d_y == fit.d_y.mean()


# ---- T7b: fits on different numbers of items (a 200-item head beside the 1,531-item pool) ----------------------------------------------
def _head_and_pool(x, ns):
    ids = [f"q{i}" for i in range(len(x))]
    return [units(x[:n], ids=ids[:n]) for n in ns]


def test_a_wilson_form_pool_of_fits_on_different_numbers_of_items_reads_their_clustered_variance():
    """Wilson's nominal variance at the cluster-effective n is the pool's clustered variance whatever the runs' n: n_eff is the per-cell
    variance the runs' own intervals read, weighted, over V. Reading n as sum w n over deff = V / sum w V_r took the runs' n and their own
    variances in separate weighted means, sum w n * sum w / n >= 1, and narrowed the bar by its square root (x0.66 here)."""
    rng = np.random.default_rng(81)
    x = (rng.random(1531) < 0.7).astype(float)
    us = _head_and_pool(x, (200, 1531, 1531))                                     # identical outcomes on the items they share
    w = np.full(3, 1 / 3)
    v = P.cluster_var(us, w)
    p = P.pooled(us, w, "wilson")
    nv = sum(c * u.n * P.cells_var(u) for c, u in zip(w, us))
    assert p.n_eff == pytest.approx(nv / v, rel=1e-12)
    assert p.se == pytest.approx(np.sqrt(v), rel=0.01)                            # the clustered variance, read in the Wilson form
    n_ref = sum(c * u.n for c, u in zip(w, us))
    vref = sum(c * P.cells_var(u) for c, u in zip(w, us))
    y = sum(c * u.value for c, u in zip(w, us))
    old = wilson_se(y * n_ref * vref / v, n_ref * vref / v)
    assert old / np.sqrt(v) == pytest.approx(0.66, abs=0.02)                      # the old effective n's narrowing
    same = P.pooled(_head_and_pool(x, (1531, 1531, 1531)), w, "wilson")          # one n: n / deff as before
    u = us[1]
    assert same.n_eff == pytest.approx(1531 * P.cells_var(u) / P.cluster_var([u] * 3, w), rel=1e-12)


def test_coverage_of_fits_on_different_numbers_of_items():
    rng = np.random.default_rng(20261122)
    q, ns, sims = 0.7, (200, 1531, 1531), 300
    w = np.full(3, 1 / 3)
    hit_new = hit_old = 0
    for _ in range(sims):
        us = _head_and_pool((rng.random(1531) < q).astype(float), ns)
        p = P.pooled(us, w, "wilson")
        hit_new += p.lo <= q <= p.hi
        vr = [P.cells_var(u) for u in us]
        n_old = p.n_eff * sum(c * u.n for c, u in zip(w, us)) * float(np.dot(w, vr)) / sum(c * u.n * r for c, u, r in zip(w, us, vr))
        lo, hi = P.wilson(sum(c * u.value for c, u in zip(w, us)) * n_old, n_old)
        hit_old += lo <= q <= hi
    assert 0.92 <= hit_new / sims <= 0.99
    assert hit_old / sims < 0.85                                                  # the old effective n under-covered (0.78 on this stream)


def test_padding_seeds_on_different_numbers_of_goals_read_the_per_cell_variance_over_v1(fresh):
    rng = np.random.default_rng(24)
    x = (rng.random(259) < 0.4).astype(float)
    a = (rng.random(259) < 0.55).astype(float)
    w = _pad_world([x[:80], x, flipped(x, 30, rng)], a)                           # one seed on an 80-goal head of the others' goals
    R.reset_pooled_ci(w.provider)
    pts = R.padding_points(w.frame())
    m = R.padding_pooled(pts).iloc[0]
    us = [w.provider(pd.Series(dict(run_id=f"lora:pad:s{s + 1}:0"))) for s in range(3)]
    ps = np.array([u.value for u in us])
    v1 = P.cluster_var(us, [1 / 3] * 3) + ps.var(ddof=1) / 3
    nv = np.mean([u.n * P.cells_var(u) for u in us])
    n_eff = nv / v1
    p1 = pts.k1.sum() / pts.n1.sum()
    l1, u1 = R.wilson(p1 * n_eff, n_eff)
    l0, u0 = R.wilson(int(a.sum()), 259)
    lo, hi = P.newcombe_bounds(a.mean(), l0, u0, p1, l1, u1)
    assert (m.lo, m.hi) == (pytest.approx(100 * lo, rel=1e-12), pytest.approx(100 * hi, rel=1e-12))
    n_old = np.mean([u.n for u in us]) * np.mean([P.cells_var(u) for u in us]) / v1
    assert n_old > 1.2 * n_eff                                                    # the old effective n read these seeds as more goals


# ---- T8: a rate near 0 % ---------------------------------------------------------------------------------------------------------------
def test_seeds_near_zero_neither_collapse_nor_jump():
    zero = units(np.zeros(259))
    one = units(np.r_[np.zeros(258), 1.0])
    w = np.full(3, 1 / 3)
    all0 = P.pooled([zero] * 3, w, "wilson", scale=100.0).se
    some = P.pooled([zero, zero, one], w, "wilson", scale=100.0).se
    assert all0 == pytest.approx(100 * wilson_se(0, 259), rel=1e-12)              # one seed's Wilson interval
    assert 0 < some < all0 * 1.01 and some > all0 / 4                             # a neighbouring interval, no collapse (as for draws)
    assert P.pooled([zero, zero, one], w, "direct", scale=100.0).se < some       # the Wald form's collapse the Wilson form avoids


# ---- T16: the control-pool file's own payloads -----------------------------------------------------------------------------------------
def _sr_entry(rng, goals=40, trials=2, flat=False):
    best = [[float(v) if rng.random() > 0.05 else None for v in rng.random(goals)] for _ in range(trials)]
    nu = [[3] * goals for _ in range(trials)]
    ns = [[3 if b is not None and rng.random() > 0.1 else 2 for b in bs] for bs in best]
    if flat:
        return dict(kind="sr_goal_best", item_set="x", best=best[0], n_units=nu[0], n_scored=ns[0])
    return dict(kind="sr_goal_best", item_set="x", trials=list(range(trials)), best=best, n_units=nu, n_scored=ns)


@pytest.mark.parametrize("flat", [False, True])
def test_payload_units_reproduce_the_strongreject_reduction(flat):
    rng = np.random.default_rng(8)
    e = _sr_entry(rng, flat=flat)
    for t in (0.5, 0.25):
        ru = JR.payload_reading_units("strongreject", e, t)
        red = JR.reduce("strongreject", JR._one_trial_flat(e), t)
        assert float(ru.a.sum()) == pytest.approx(red.k, abs=1e-12) and float(ru.b.sum()) == red.n
        assert float(ru.a.sum() / ru.b.sum()) == pytest.approx(red.value, abs=1e-12)
        if not flat:
            assert (ru.b > 1).any()                                                # a goal holds several determined trials: one cluster


def test_payload_units_reproduce_the_over_refusal_and_accuracy_reductions():
    rng = np.random.default_rng(9)
    lab = [int(c) if rng.random() > 0.1 else None for c in rng.integers(0, 3, 60)]
    sub = ["safe" if i % 2 else "unsafe" for i in range(60)]
    e = dict(kind="cpi_labels", item_set="x", label=lab, subset=sub)
    ru = JR.payload_reading_units("over_refusal", e)
    red = JR.reduce("over_refusal", e)
    assert float(ru.a.sum()) == pytest.approx(red.k, abs=1e-12) and float(ru.b.sum()) == red.n
    assert float(ru.a.sum() / ru.b.sum()) == pytest.approx(red.value, abs=1e-12)
    base = [int(v) if rng.random() > 0.1 else None for v in rng.integers(0, 2, 80)]
    drawn = [True if i % 7 == 0 else None for i in range(80)]
    rung = [int(v) if d_ else None for v, d_ in zip(rng.integers(0, 2, 80), drawn)]
    a = dict(kind="acc_items", item_set="x", base=base, rung=rung, rung_at_cap=drawn)
    ru = JR.payload_reading_units("rule", a)
    red = R.ctlstop_judging.reduce("rule", a)
    assert float(ru.a.sum()) == pytest.approx(red.k, abs=1e-12) and float(ru.b.sum()) == red.n
    assert len(ru.present) == 80


def test_a_file_rows_units_are_its_payloads_and_a_drifted_rule_stops_the_render(monkeypatch):
    rng = np.random.default_rng(91)
    e = _sr_entry(rng, goals=30, trials=1)
    for ns in e["n_scored"]:
        ns[:] = [3] * len(ns)
    side = {"rows": [dict(run_id="ctl:r1", object_id="o", rule="control_pool", slot="control_relu", seed="s1", eval_id="crescendo", role="drawn",
                          model_id=R.MODEL_ID, instrument_id="i", item_set_id="x")], "items": {"ctl:r1": e}}
    try:
        f = R.ctlstop_frame(side, 0.5)
    except KeyError:
        pytest.skip("the stored columns this synthetic row lacks")
    row = f.iloc[0]
    assert row.judge_source == R.CTLSTOP_PAYLOAD_SOURCE and R.reading_rederived(row)
    R.reset_pooled_ci(None)
    u = R.row_units(row)
    assert u.value == pytest.approx(float(row.value), abs=1e-12) and u.n == float(row.n)
    drifted = dict(e, best=[[None if b is None else min(1.0, b + 0.3) for b in bs] for bs in e["best"]])
    monkeypatch.setitem(R._CTL_OWN_PAYLOADS, "ctl:r1", ("strongreject", drifted))
    R.reset_pooled_ci(None)
    with pytest.raises(P.UnitsMismatch):
        R.row_units(row)
    # a file derived again with another payload of the run (the generator's wave tests): its rows' units follow the new payload
    R.reset_pooled_ci(None)
    f1 = R.ctlstop_frame(side, 0.5)
    R.row_units(f1.iloc[0])                                                       # cached under the run's unit key
    f2 = R.ctlstop_frame(dict(side, items={"ctl:r1": drifted}), 0.5)
    r2 = f2.iloc[0]
    assert float(r2.value) != float(f1.iloc[0].value)
    assert R.row_units(r2).value == pytest.approx(float(r2.value), abs=1e-12)


# ---- T9 / T10: the readers; one fit is byte-identical -----------------------------------------------------------------------------------
def test_a_single_seed_object_and_a_decode_only_dot_are_the_one_fit_rule_byte_for_byte(fresh):
    rng = np.random.default_rng(11)
    a = (rng.random(N) < 0.6).astype(float)
    w = seed_world([(rng.random(N) < 0.4).astype(float)], [a, flipped(a, 5, rng), flipped(a, 5, rng)])
    for v, src, dr in SEED_VIEWS:
        w.run(f"si:{v}", (rng.random(N) < 0.2).astype(float), object_label="safety_si", object_id="safety_si", view_legacy=v,
              lane_source_legacy=src, draw=dr)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    cm = R.category_means(d, categories=CAT)
    for o in ("control_lora", "safety_si"):
        g = cm[cm.object_label.eq(o)]
        m = R.pool_seeds(g).iloc[0]
        r = g.iloc[0]
        anchor = float(np.sqrt(sum(R.combo_se({k_: w_ for k_, w_ in kw}) ** 2 for _e, kw in r.anchor_terms))) if r.anchor_terms else 0.0
        se = float(np.sqrt(float(r.se_steer) ** 2 + 0.0 ** 2 + anchor ** 2))
        assert (m.lo, m.hi) == (m.y - 1.96 * se, m.y + 1.96 * se) and m.se_method == R.SEED_ONE_FIT
    one = d[d.object_label.eq("control_lora")]
    pc = R.pool_cell(one, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    assert (pc.d_lo, pc.d_hi) == (one.d_lo.iloc[0], one.d_hi.iloc[0])
    dot = d[d.object_label.eq("safety_si")]
    c = R.cell_interval(dot)
    pd_ = R.pool_cell(dot, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    assert (pd_.d_lo, pd_.d_hi) == (c.lo, c.hi)


def test_a_one_eval_mean_column_and_the_absolute_mark_read_one_seed_interval_for_identical_seeds(fresh):
    rng = np.random.default_rng(13)
    x = (rng.random(N) < 0.4).astype(float)
    a = (rng.random(N) < 0.6).astype(float)
    w = seed_world([x, x, x], [a, a, a])
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    pts = R.delta_points(d, ["pasttense"], ["control_lora"])
    mc = R.eval_mean_column(pts, ["pasttense"])
    one = pts.iloc[0]
    assert (mc.lo.iloc[0], mc.hi.iloc[0]) == (pytest.approx(one.lo, rel=1e-12), pytest.approx(one.hi, rel=1e-12))
    want = np.hypot(100 * wilson_se(x.sum(), N), 100 * wilson_se(a.sum(), N))
    assert (one.hi - one.lo) / 3.92 == pytest.approx(want, rel=1e-12)
    ab = R.pool_cell(R.flip(R.intervals(w.frame()))[lambda f: f.object_label.eq("control_lora")])
    assert (ab.hi.iloc[0] - ab.lo.iloc[0]) / 3.92 == pytest.approx(100 * wilson_se(x.sum(), N), rel=1e-12)


def _two_eval_world(rng, flip=8):
    w = World()
    by = {}
    for e in ("pasttense", "codeattack"):
        a = (rng.random(N) < 0.6).astype(float)
        base = (rng.random(N) < 0.35).astype(float)
        by[e] = [flipped(base, flip * (i + 1), rng) for i in range(3)]
        seed_world(by[e], [a, a, a], eval_id=e, item_set=e, w=w)
    return w, by


def test_a_mean_over_evals_of_per_eval_seed_pools_is_one_pool_over_the_same_fits(fresh):
    """The appendix's mean column (and a transfer category) averages per-eval seed pools; the fits are the same three on every eval, so
    the between-fit term is the variance of the fits' means over the panel's evals, never each eval's between term combined as if every
    eval had its own sample of fits."""
    rng = np.random.default_rng(14)
    w, by = _two_eval_world(rng)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    ev = ["pasttense", "codeattack"]
    pts = R.delta_points(d, ev, ["control_lora"])
    mc = R.eval_mean_column(pts, ev).iloc[0]
    assert mc.y == pytest.approx(pts.y.mean(), abs=0) and mc.se_method.startswith(R.SEED_CLUSTERED)
    fit = d[d.object_label.eq("control_lora")]
    k, S = 2, 3
    items2 = sum(brute_wilson_se([table(x) for x in by[e]], [1 / (k * S)] * S) ** 2 for e in ev)
    seed_means = (-fit.d_y).groupby(fit.seed).mean()                       # each fit's mean over the panel's evals (oriented)
    anchor2 = sum((float(fit[fit.eval_id == e].anchor_se.iloc[0]) / k) ** 2 for e in ev)
    want = np.sqrt(items2 + seed_means.var(ddof=1) / S + anchor2)
    assert (mc.hi - mc.lo) / 3.92 == pytest.approx(want, rel=1e-10)
    old = float(np.sqrt((((pts.hi - pts.lo) / 3.92) ** 2).sum()) / k)       # the per-eval pools combined: each eval's between term again
    assert old != pytest.approx(want, rel=1e-3)
    cm = R.category_means(d, categories={"static": (ev, -1, "defense")})
    ps = R.pool_seeds(cm[cm.object_label.eq("control_lora")]).iloc[0]       # the same pool as the category's mean mark
    assert (ps.hi - ps.lo) / 3.92 == pytest.approx(want, rel=1e-10) and ps.y == pytest.approx(mc.y, abs=1e-12)


# ---- T11: marks: a run without units, a member that cannot be split, constant fits; the silent skip gone -------------------------------
def test_a_seed_run_without_units_takes_the_marked_bound_and_without_an_interval_is_unmeasured(fresh):
    rng = np.random.default_rng(15)
    a = (rng.random(N) < 0.6).astype(float)
    seeds = [(rng.random(N) < 0.3 + 0.05 * i).astype(float) for i in range(3)]
    w = seed_world(seeds, [a, a, a])
    w.missing.add("control_lora:pasttense:s2")
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("control_lora")]
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    own = ((fit.hi - fit.lo) / 3.92).to_numpy()
    want = np.sqrt((own.mean()) ** 2 + fit.d_y.var(ddof=1) / 3 + float(fit.anchor_se.iloc[0]) ** 2)
    assert mark_se(pc) == pytest.approx(want, rel=1e-10)
    assert R.BOUND_NOTE in pc.se_method
    assert any(v["what"] == "seed pool" for v in R.POOLED_CI_REPORT["bound"].values())
    assert "Seed pools" in R.seed_pool_note(d) and R.BOUND_NOTE in R.seed_pool_note(d) and R.seed_pool_note(d) in R.cell_notes(d, ["pasttense"])
    f2 = fit.copy()
    f2.loc[f2.run_id.eq("control_lora:pasttense:s2"), ["lo", "hi", "d_lo", "d_hi"]] = np.nan
    R._SEED_POOLS.clear()
    R._POOLS.clear()
    pc2 = R.pool_cell(f2, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    assert np.isnan(pc2.d_lo) and pc2.se_method.startswith("unmeasured")


def test_a_seed_without_an_interval_but_with_units_is_clustered_and_the_silent_skip_is_gone(fresh):
    """Qwen's Difference ReLU on the Petri evals: one seed on a filed-contrast width, two with no interval at all, all three with
    transcripts. Their pool reads the three seeds' units; the old pool summed the missing se as 0 (the category mean's pandas sum)."""
    rng = np.random.default_rng(17)
    st = [f"b{i // 10}" for i in range(60)]
    a = rng.integers(2, 8, 60).astype(float)
    seeds = [rng.integers(1, 6, 60).astype(float) for _ in range(3)]
    w = World()
    e = "petri_self_preservation"
    for v, src, dr in SEED_VIEWS:
        w.run(f"none:{v}", a, strata=st, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id=e, metric_id="petri_score",
              item_set_id="petri", ci_lo=float(a.mean() - 0.3), ci_hi=float(a.mean() + 0.3))
    for i, x in enumerate(seeds):
        kw = dict(delta_lo=-0.5, delta_hi=0.5) if i == 0 else {}
        w.run(f"relu:s{i + 1}", x, strata=st, object_label="difference_relu", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"o{i}",
              eval_id=e, metric_id="petri_score", item_set_id="petri", **kw)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("difference_relu")].sort_values("seed")
    assert fit.d_lo.notna().tolist() == [True, False, False] and R.FILED_CONTRAST_ROUTE in fit.route.iloc[0]
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    us = [w.provider(r) for _, r in fit.iterrows()]
    items = P.pooled(us, [1 / 3] * 3, "direct").se
    want = np.sqrt(items ** 2 + fit.d_y.var(ddof=1) / 3 + float(fit.anchor_se.iloc[0]) ** 2)
    assert mark_se(pc) == pytest.approx(want, rel=1e-10) and pc.se_method.startswith(R.SEED_CLUSTERED)
    cm = R.category_means(d, categories={"misalignment": ([e], -1, "score")})     # (the Petri category reads the pooled eval)
    g = cm[cm.object_label.eq("difference_relu")]
    assert g.lo.isna().sum() == 2                                                 # a seed's own mean with no interval: unmeasured, not exact
    m = R.pool_seeds(g).iloc[0]
    assert (m.hi - m.lo) / 3.92 == pytest.approx(want, rel=1e-10)


def test_a_member_that_cannot_be_split_takes_the_opaque_bound(fresh):
    cm = pd.DataFrame([dict(model_id="m", category="static", object_label="o", seed=s, y=y, lo=y - 1.96 * se, hi=y + 1.96 * se, k=1,
                            partial=False, open=False, ceiling=False, at_cap=np.nan, degen=np.nan, anchors="", evals="e")
                       for s, y, se in (("s1", 1.0, 1.0), ("s2", 3.0, 2.0))])
    m = R.pool_seeds(cm).iloc[0]
    assert (m.hi - m.lo) / 3.92 == pytest.approx(np.sqrt(1.5 ** 2 + 2.0 / 2), rel=1e-12)    # sum_s se_s / S, plus the between term
    assert m.se_method.startswith(R.SEED_BOUND_NOTE) and R.POOLED_CI_REPORT["opaque_seed_pools"] >= 1
    old = np.sqrt((1 + 4) / 4 + 2.0 / 2)                                          # whole in the within term, as before
    assert old < (m.hi - m.lo) / 3.92


def test_fits_constant_on_every_unit_give_an_item_term_0_recorded_never_a_refusal(fresh):
    st = [f"b{i // 10}" for i in range(60)]
    rng = np.random.default_rng(18)
    a = rng.integers(2, 8, 60).astype(float)
    w = World()
    e = "petri_self_preservation"
    for v, src, dr in SEED_VIEWS:
        w.run(f"none:{v}", a, strata=st, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id=e, metric_id="petri_score",
              item_set_id="petri", ci_lo=float(a.mean() - 0.3), ci_hi=float(a.mean() + 0.3))
    for i, c in enumerate((1.0, 1.0, 2.0)):                                       # each fit constant; the third differs
        w.run(f"relu:s{i + 1}", np.full(60, c), strata=st, object_label="difference_relu", seed=f"s{i + 1}", lane="base_to_instruct",
              object_id=f"o{i}", eval_id=e, metric_id="petri_score", item_set_id="petri")
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("difference_relu")]
    pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
    want = np.sqrt(fit.d_y.var(ddof=1) / 3 + float(fit.anchor_se.iloc[0]) ** 2)
    assert mark_se(pc) == pytest.approx(want, rel=1e-12) and R.CONSTANT_ITEMS_NOTE in pc.se_method
    assert R.POOLED_CI_REPORT["constant_items"] >= 1
    rows = [r for _, r in fit.iterrows()]
    with pytest.raises(P.DegenerateCell):                                         # the decode-draw rule is unchanged
        R.pool_se(rows)


# ---- T12: appendix_common --------------------------------------------------------------------------------------------------------------
def test_appendix_common_rows_with_their_parts_give_pool_seeds_and_without_them_the_opaque_bound(fresh):
    import appendix_common as AC
    rng = np.random.default_rng(19)
    a = (rng.random(N) < 0.6).astype(float)
    seeds = [(rng.random(N) < 0.3 + 0.05 * i).astype(float) for i in range(3)]
    w = seed_world(seeds, [a, a, a])
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    fit = d[d.object_label.eq("control_lora")]
    rows = []
    for _, r in fit.iterrows():
        dp = R.delta_parts(r)
        rows.append(dict(object="control_lora", seed=r.seed, category="static", eval="pasttense", condition="c", delta=r.d_y, ci_lo=r.d_lo,
                         ci_hi=r.d_hi, se_steer=dp.se_steer, anchor_terms=dp.anchor_terms, steer_terms=dp.steer_terms,
                         split_anchor_terms=dp.split_anchor_terms))
    cats = {"static": (["pasttense"], -1, "defense")}
    pm = AC.paired_means(pd.DataFrame(rows), cats).iloc[0]
    cm = R.category_means(d, categories=CAT)
    ps = R.pool_seeds(cm[cm.object_label.eq("control_lora")]).iloc[0]
    assert (pm.hi - pm.lo) == pytest.approx(ps.hi - ps.lo, rel=1e-12) and pm.y == pytest.approx(ps.y, abs=1e-12)
    bare = pd.DataFrame(rows).drop(columns=["se_steer", "anchor_terms", "steer_terms", "split_anchor_terms"])
    pb = AC.paired_means(bare, cats).iloc[0]
    se_rows = ((fit.d_hi - fit.d_lo) / 3.92).to_numpy()
    assert (pb.hi - pb.lo) / 3.92 == pytest.approx(np.sqrt(se_rows.mean() ** 2 + fit.d_y.var(ddof=1) / 3), rel=1e-10)


# ---- T13: the padding means ------------------------------------------------------------------------------------------------------------
def _pad_world(seeds, anchor, pad=8192.0, drop_units=()):
    w = World()
    common = dict(eval_id="grok420", item_set_id="grok", pad=pad, pad_contract=True)
    w.run("none:pad", anchor, object_label="none", **common)
    for i, xs in enumerate(seeds):
        for r, x in enumerate(xs if isinstance(xs, list) else [xs]):
            w.run(f"lora:pad:s{i + 1}:{r}", x, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"o{i}",
                  draw=f"t{r}", **common)
    for rid in drop_units:
        w.missing.add(rid)
    return w


def test_identical_padding_seeds_give_one_seeds_newcombe_interval(fresh):
    rng = np.random.default_rng(20)
    x = (rng.random(259) < 0.45).astype(float)
    a = (rng.random(259) < 0.55).astype(float)
    w = _pad_world([x, x, x], a)
    R.reset_pooled_ci(w.provider)
    pts = R.padding_points(w.frame())
    m = R.padding_pooled(pts).iloc[0]
    k1, k0 = int(x.sum()), int(a.sum())
    lo, hi = R.newcombe_diff(k0, 259, k1, 259)
    assert (m.lo, m.hi) == (pytest.approx(100 * lo, rel=1e-12), pytest.approx(100 * hi, rel=1e-12))
    assert m.delta == 100.0 * (k0 / 259 - 3 * k1 / (3 * 259)) and m.k1 == 3 * k1 and m.n1 == 3 * 259
    lo_sum, hi_sum = R.newcombe_diff(k0, 259, 3 * k1, 3 * 259)                   # the old rule: one binomial sample of 777 goals
    assert hi_sum - lo_sum < m.hi / 100 - m.lo / 100


def test_padding_seeds_independent_by_goal_with_no_spread_give_the_summed_n_up_to_the_stated_correction(fresh):
    a = (np.arange(259) % 2 == 0).astype(float)
    xs = []
    rng = np.random.default_rng(21)
    for s in range(3):
        x = np.zeros(259)
        x[rng.choice(259, 100, replace=False)] = 1.0                               # equal rates, independent goals
        xs.append(x)
    w = _pad_world(xs, a)
    R.reset_pooled_ci(w.provider)
    m = R.padding_pooled(R.padding_points(w.frame())).iloc[0]
    us = [w.provider(pd.Series(dict(run_id=f"lora:pad:s{s + 1}:0"))) for s in range(3)]
    v1 = P.cluster_var(us, [1 / 3] * 3)
    n_eff = np.mean([u.n * P.cells_var(u) for u in us]) / v1                      # 259 / deff, deff = v1 / mean(V_r): one n
    p1 = 300 / 777
    l1, u1 = R.wilson(p1 * n_eff, n_eff)
    l0, u0 = R.wilson(int(a.sum()), 259)
    lo, hi = P.newcombe_bounds(a.mean(), l0, u0, p1, l1, u1)
    assert (m.lo, m.hi) == (pytest.approx(100 * lo, rel=1e-12), pytest.approx(100 * hi, rel=1e-12))
    lo_s, hi_s = R.newcombe_diff(int(a.sum()), 259, 300, 777)
    assert (m.hi - m.lo) / 100 == pytest.approx(hi_s - lo_s, rel=0.15)           # about the summed n, as the old rule read them


def test_a_spread_between_padding_seeds_widens_the_mean_and_a_seed_without_units_takes_the_marked_bound(fresh):
    rng = np.random.default_rng(22)
    common = rng.random(259)
    xs = [(common < q).astype(float) for q in (0.2, 0.45, 0.7)]
    a = (rng.random(259) < 0.5).astype(float)
    w = _pad_world(xs, a)
    R.reset_pooled_ci(w.provider)
    pts = R.padding_points(w.frame())
    m = R.padding_pooled(pts).iloc[0]
    k1 = int(sum(x.sum() for x in xs))
    lo_s, hi_s = R.newcombe_diff(int(a.sum()), 259, k1, 777)
    assert m.hi - m.lo > 2 * 100 * (hi_s - lo_s) and m.se_method.startswith("seed pool")
    w2 = _pad_world(xs, a, drop_units=("lora:pad:s2:0",))
    R.reset_pooled_ci(w2.provider)
    m2 = R.padding_pooled(R.padding_points(w2.frame())).iloc[0]
    assert R.SEED_BOUND_NOTE in m2.se_method and any(v["what"] == "padding mean" for v in R.POOLED_CI_REPORT["bound"].values())
    assert m2.delta == m.delta and np.isfinite(m2.lo)
    tab = R.padding_table(R.padding_points(w2.frame()))                          # the table's only mark of that bound
    assert "Mean rows with a seed's run without item rows (Control LoRA): the seeds' item term is the identical-seeds bound." in tab
    R.reset_pooled_ci(w.provider)
    assert "Mean rows with a seed's run without item rows" not in R.padding_table(R.padding_points(w.frame()))


def test_a_two_run_padding_seed_cell_weights_its_runs_half(fresh):
    rng = np.random.default_rng(23)
    xs = [[(rng.random(259) < 0.4).astype(float), (rng.random(259) < 0.4).astype(float)], (rng.random(259) < 0.5).astype(float),
          (rng.random(259) < 0.45).astype(float)]
    a = (rng.random(259) < 0.5).astype(float)
    w = _pad_world(xs, a)
    R.reset_pooled_ci(w.provider)
    pts = R.padding_points(w.frame())
    s1 = pts[pts.seed.eq("s1")].iloc[0]
    assert len(s1.steer_terms) == 2 and all(v == 0.5 for v in s1.steer_terms.values())
    m = R.padding_pooled(pts).iloc[0]
    tabs = [table(xs[0][0]), table(xs[0][1]), table(xs[1]), table(xs[2])]
    wts = [1 / 6, 1 / 6, 1 / 3, 1 / 3]
    ps = (pts.k1 / pts.n1).to_numpy()
    v1 = brute_var(tabs, wts) + ps.var(ddof=1) / 3
    n_eff = sum(x * len(t) * brute_var([t], [1.0]) for x, t in zip(wts, tabs)) / v1
    p1 = pts.k1.sum() / pts.n1.sum()
    l1, u1 = R.wilson(p1 * n_eff, n_eff)
    l0, u0 = R.wilson(int(a.sum()), 259)
    lo, hi = P.newcombe_bounds(a.mean(), l0, u0, p1, l1, u1)
    assert (m.lo, m.hi) == (pytest.approx(100 * lo, rel=1e-10), pytest.approx(100 * hi, rel=1e-10))


def test_a_padding_seed_whose_runs_declare_one_item_set_must_share_its_items(fresh):
    """The padding pool clusters its seeds' runs directly (`pooled_ci.cluster_var`), so it holds them to the alignment guard every other
    pool's runs meet (`pool_se`): runs declaring one item set share 99 % of their ids, or the render stops by name."""
    rng = np.random.default_rng(25)
    xs = [(rng.random(259) < 0.4).astype(float) for _ in range(3)]
    w = _pad_world(xs, (rng.random(259) < 0.5).astype(float))
    base_provider = w.provider

    def provider(row):
        u = base_provider(row)
        if str(row["run_id"]) != "lora:pad:s3:0":
            return u
        return units(u.table.a.to_numpy(), ids=[f"g{i}" if i < 130 else f"x{i}" for i in range(len(u.table))])
    R.reset_pooled_ci(provider)
    with pytest.raises(P.ItemIdsDisagree, match="item set grok"):
        R.padding_pooled(R.padding_points(w.frame()))
    R.reset_pooled_ci(w.provider)
    R.padding_pooled(R.padding_points(w.frame()))
    assert R.POOLED_CI_REPORT["alignment_min"] == 1.0 and R.POOLED_CI_REPORT["seed_overlap_min"] == 1.0


def test_padding_rows_without_their_runs_say_their_interval_is_not_a_seed_pools(fresh):
    pts = pd.DataFrame([dict(object_label="control_lora", seed=s, pad=0.0, delta=d_, lo=d_ - 2, hi=d_ + 2, defence=50.0, base_defence=40.0)
                        for s, d_ in (("s1", 1.0), ("s2", 3.0))])
    m = R.padding_pooled(pts).iloc[0]
    assert m.delta == 2.0 and "not a seed pool" in m.se_method


# ---- T14: the transfer difference --------------------------------------------------------------------------------------------------------
def _transfer_world(rng, same_anchor=True, it_kw=None, base_seeds=3):
    w = World()
    e = "pasttense"
    a = (rng.random(N) < 0.6).astype(float)
    a2 = flipped(a, 20, rng)
    for (v, src, dr) in SEED_VIEWS:
        w.run(f"none:{v}", a, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, lane="base_to_instruct" if not same_anchor else None)
    if not same_anchor:
        w.run("none:it", a2, object_label="none", view_legacy="it", lane_source_legacy="it_lane", draw="dit", lane="instruct_direct",
              item_set_id="static_pasttense_259x2b")
    base = []
    for i in range(base_seeds):
        x = (rng.random(N) < 0.3 + 0.05 * i).astype(float)
        base.append(x)
        w.run(f"base:s{i + 1}", x, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"b{i}", eval_id=e)
    xi = (rng.random(N) < 0.35).astype(float)
    w.run("it:1", xi, object_label="control_lora", lane="instruct_direct", object_id="it0", eval_id=e,
          **({"item_set_id": "static_pasttense_259x2b"} if not same_anchor else {}), **(it_kw or {}))
    return w, base, xi


def _pins(base_seeds):
    return {(R.MODEL_ID, "control_lora"): ([f"b{i}" for i in range(base_seeds)], "it0")}


def test_lanes_on_the_same_anchor_runs_net_the_anchor_to_zero(fresh):
    rng = np.random.default_rng(30)
    w, base, xi = _transfer_world(rng)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    pe, pc, _ = R.transfer_pairs(d, CAT, objects=["control_lora"], models=[R.MODEL_ID], pins=_pins(3))
    r = pe.iloc[0]
    fb = d[(d.lane == "base_to_instruct") & d.object_label.eq("control_lora")]
    fi = d[(d.lane == "instruct_direct") & d.object_label.eq("control_lora")]
    steer_b = np.sqrt(brute_wilson_se([table(x) for x in base], [1 / 3] * 3) ** 2 + fb.d_y.var(ddof=1) / 3)
    steer_i = float((fi.hi - fi.lo).iloc[0] / 3.92)                               # the one-fit lane: its row's own interval
    assert r.se == pytest.approx(np.hypot(steer_b, steer_i), rel=1e-10) and r.se_method == R.TRANSFER_NETTED
    assert r["diff"] == fb.d_y.mean() - fi.d_y.iloc[0]
    old = np.hypot(np.sqrt(steer_b ** 2 + fb.anchor_se.iloc[0] ** 2), np.hypot(steer_i, fi.anchor_se.iloc[0]))
    assert old > r.se                                                             # the anchor was counted twice
    c = pc.iloc[0]
    assert (c.hi - c.lo) / 3.92 == pytest.approx(r.se, rel=1e-10) and c.y == -r["diff"]


def test_lanes_on_different_anchor_runs_cluster_each_signed_mixture_apart_never_paired(fresh):
    rng = np.random.default_rng(31)
    w, base, xi = _transfer_world(rng, same_anchor=False)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    pe, _pc, _ = R.transfer_pairs(d, CAT, objects=["control_lora"], models=[R.MODEL_ID], pins=_pins(3))
    r = pe.iloc[0]
    fb = d[(d.lane == "base_to_instruct") & d.object_label.eq("control_lora")]
    fi = d[(d.lane == "instruct_direct") & d.object_label.eq("control_lora")]
    assert set(fb.anchor_units.iloc[0].split(";")).isdisjoint(fi.anchor_units.iloc[0].split(";"))
    steer_b = np.sqrt(brute_wilson_se([table(x) for x in base], [1 / 3] * 3) ** 2 + fb.d_y.var(ddof=1) / 3)
    steer_i = float((fi.hi - fi.lo).iloc[0] / 3.92)
    a_b, a_i = float(fb.anchor_se.iloc[0]), float(fi.anchor_se.iloc[0])
    assert r.se == pytest.approx(np.sqrt(steer_b ** 2 + steer_i ** 2 + a_b ** 2 + a_i ** 2), rel=1e-10)


def test_a_one_fit_lane_on_a_filed_contrast_width_reads_its_units_and_without_them_keeps_the_old_composition(fresh):
    st = [f"b{i // 10}" for i in range(60)]
    rng = np.random.default_rng(32)
    e = "petri_self_preservation"

    def world(const=False):
        w = World()
        a = rng.integers(2, 8, 60).astype(float)
        for v, src, dr in SEED_VIEWS:
            w.run(f"none:{v}", a, strata=st, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id=e,
                  metric_id="petri_score", item_set_id="petri", ci_lo=float(a.mean() - 0.3), ci_hi=float(a.mean() + 0.3))
        for i in range(3):
            x = rng.integers(1, 6, 60).astype(float)
            w.run(f"base:s{i + 1}", x, strata=st, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"b{i}",
                  eval_id=e, metric_id="petri_score", item_set_id="petri", ci_lo=float(x.mean() - 0.4), ci_hi=float(x.mean() + 0.4))
        xi = np.full(60, 1.0) if const else rng.integers(1, 6, 60).astype(float)
        w.run("it:1", xi, strata=st, object_label="control_lora", lane="instruct_direct", object_id="it0", eval_id=e, metric_id="petri_score",
              item_set_id="petri", delta_lo=-0.6, delta_hi=0.6)
        return w

    w = world()
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    cats = {"misalignment": ([e], -1, "score")}                                  # (the Petri category reads the pooled eval)
    pe, _c, _ = R.transfer_pairs(d, cats, objects=["control_lora"], models=[R.MODEL_ID], pins=_pins(3))
    r = pe.iloc[0]
    fi = d[(d.lane == "instruct_direct")]
    assert R.FILED_CONTRAST_ROUTE in fi.route.iloc[0] and r.se_method == R.TRANSFER_NETTED
    fb = d[(d.lane == "base_to_instruct") & d.object_label.eq("control_lora")]
    steer_b2 = P.pooled([w.provider(x) for _, x in fb.iterrows()], [1 / 3] * 3, "direct").se ** 2 + fb.d_y.var(ddof=1) / 3
    steer_i = P.pooled([w.provider(fi.iloc[0])], form="direct").se
    assert r.se == pytest.approx(np.sqrt(steer_b2 + steer_i ** 2), rel=1e-10)
    for missing, const in ((True, False), (False, True)):
        w2 = world(const=const)
        if missing:
            w2.missing.add("it:1")
        R.reset_pooled_ci(w2.provider)
        d2 = R.anchored(w2.frame())
        pe2, pc2, pairs2 = R.transfer_pairs(d2, cats, objects=["control_lora"], models=[R.MODEL_ID], pins=_pins(3))
        r2 = pe2.iloc[0]
        cb = R.pool_cell(d2[d2.lane.eq("base_to_instruct") & d2.object_label.eq("control_lora")], keys=("model_id", "object_label", "lane", "eval_id"),
                         y="d_y", lo="d_lo", hi="d_hi").iloc[0]
        fi2 = d2[d2.lane.eq("instruct_direct")].iloc[0]
        assert r2.se_method == R.TRANSFER_KEPT and np.isfinite(r2.se)
        assert r2.se == pytest.approx(np.hypot(mark_se(cb), mark_se(fi2)), rel=1e-12)          # the filed width, the other lane whole
        assert any(v["what"] == "transfer" for v in R.POOLED_CI_REPORT["bound"].values())
        assert pc2.iloc[0].se_method == R.TRANSFER_KEPT
        tab = R.table_transfer(pc2.assign(category="static"), pairs2, ["control_lora"], [R.MODEL_ID], {"static": cats["misalignment"]},
                               "Transfer.", "tab:t")                              # a category the table's LEVEL knows
        assert ("Cells whose steered part a lane's units cannot give (Qwen3.5-4B Control LoRA (static)): each lane's whole interval "
                "combined as independent.") in tab


def test_a_lane_with_a_run_without_item_rows_is_named_in_the_difference_and_the_transfer_table(fresh):
    rng = np.random.default_rng(34)
    w, base, xi = _transfer_world(rng)
    w.missing.add("base:s2")
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    pe, pc, pairs = R.transfer_pairs(d, CAT, objects=["control_lora"], models=[R.MODEL_ID], pins=_pins(3))
    want = f"{R.TRANSFER_NETTED} ({R.TRANSFER_LANE_BOUND})"
    assert pe.iloc[0].se_method == want and pc.iloc[0].se_method == want
    assert any("base:s2" in v["without_items"] for v in R.POOLED_CI_REPORT["bound"].values())   # recorded once, by the lane's first pool
    fb = d[(d.lane == "base_to_instruct") & d.object_label.eq("control_lora")]
    fi = d[(d.lane == "instruct_direct") & d.object_label.eq("control_lora")]
    own = ((fb.hi - fb.lo) / 3.92).to_numpy()
    steer_b2 = own.mean() ** 2 + fb.d_y.var(ddof=1) / 3                           # the lane's item term: the identical-draws bound
    assert pe.iloc[0].se == pytest.approx(np.hypot(np.sqrt(steer_b2), float((fi.hi - fi.lo).iloc[0] / 3.92)), rel=1e-10)
    tab = R.table_transfer(pc, pairs, ["control_lora"], [R.MODEL_ID], CAT, "Transfer.", "tab:t")
    assert ("Cells with a lane's run without item rows (Qwen3.5-4B Control LoRA (static)): that lane's item term is the identical-draws "
            "bound") in tab
    R.reset_pooled_ci(_transfer_world(np.random.default_rng(34))[0].provider)
    d0 = R.anchored(_transfer_world(np.random.default_rng(34))[0].frame())
    _pe0, pc0, pairs0 = R.transfer_pairs(d0, CAT, objects=["control_lora"], models=[R.MODEL_ID], pins=_pins(3))
    assert "a lane's run without item rows" not in R.table_transfer(pc0, pairs0, ["control_lora"], [R.MODEL_ID], CAT, "Transfer.", "tab:t")


def test_fits_whose_item_ids_share_nothing_are_refused_and_the_smallest_overlap_is_recorded(fresh):
    """A sidecar's units in another id namespace than the frame's, on rows declaring no item set (so the alignment guard of runs on one
    item set does not apply), would read the fits as independent samples under the item-clustered method's name: refused. A fit on part
    of the others' items is legitimate (the union is clustered) and its overlap is recorded."""
    rng = np.random.default_rng(35)
    a = (rng.random(N) < 0.6).astype(float)
    seeds = [(rng.random(N) < 0.3 + 0.05 * i).astype(float) for i in range(3)]
    for rename, raises in ((lambda i: f"pos{i}", True), (lambda i: f"g{i}" if i < N // 2 else f"other{i}", False)):
        w = seed_world(seeds, [a, a, a])
        for r in w.rows:
            if r["object_label"] == "control_lora":
                r["item_set_id"] = None                                           # a sidecar's rows: no declared item set
        base_provider = w.provider

        def provider(row, base_provider=base_provider, rename=rename):
            u = base_provider(row)
            if str(row["run_id"]) != "control_lora:pasttense:s2":
                return u
            return units(u.table.a.to_numpy(), ids=[rename(i) for i in range(len(u.table))])
        R.reset_pooled_ci(provider)
        d = R.anchored(w.frame())
        fit = d[d.object_label.eq("control_lora")]
        if raises:
            with pytest.raises(P.ItemIdsDisagree, match="share no item id"):
                R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi")
        else:
            pc = R.pool_cell(fit, y="d_y", lo="d_lo", hi="d_hi").iloc[0]
            assert pc.se_method.startswith(R.SEED_CLUSTERED)
            assert R.POOLED_CI_REPORT["seed_overlap_min"] == pytest.approx(0.5, abs=1 / N)
            assert "Control LoRA seed pool on pasttense" in R.POOLED_CI_REPORT["seed_overlap_at"]
    w = seed_world(seeds, [a, a, a])
    R.reset_pooled_ci(w.provider)
    R.pool_cell(R.anchored(w.frame()).query("object_label == 'control_lora'"), y="d_y", lo="d_lo", hi="d_hi")
    assert R.POOLED_CI_REPORT["seed_overlap_min"] == 1.0


def test_a_transfer_category_is_one_pool_per_lane_over_its_evals(fresh):
    rng = np.random.default_rng(33)
    w = World()
    by = {}
    for e in ("pasttense", "codeattack"):
        a = (rng.random(N) < 0.6).astype(float)
        for (v, src, dr) in SEED_VIEWS:
            w.run(f"none:{e}:{v}", a, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id=e, item_set_id=e)
        by[e] = [(rng.random(N) < 0.3 + 0.08 * i).astype(float) for i in range(3)]
        for i, x in enumerate(by[e]):
            w.run(f"base:{e}:s{i + 1}", x, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"b{i}", eval_id=e,
                  item_set_id=e)
        w.run(f"it:{e}", (rng.random(N) < 0.35).astype(float), object_label="control_lora", lane="instruct_direct", object_id="it0", eval_id=e,
              item_set_id=e)
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    ev = ["pasttense", "codeattack"]
    pe, pc, _ = R.transfer_pairs(d, {"static": (ev, -1, "defense")}, objects=["control_lora"], models=[R.MODEL_ID], pins=_pins(3))
    c = pc.iloc[0]
    fb = d[d.lane.eq("base_to_instruct") & d.object_label.eq("control_lora")]
    k, S = 2, 3
    items2 = sum(brute_wilson_se([table(x) for x in by[e]], [1 / (k * S)] * S) ** 2 for e in ev)
    fits2 = fb.d_y.groupby(fb.seed).mean().var(ddof=1) / S
    fi = d[d.lane.eq("instruct_direct") & d.object_label.eq("control_lora")].set_index("eval_id")
    it2 = sum((float(fi.hi[e] - fi.lo[e]) / 3.92 / k) ** 2 for e in ev)
    assert (c.hi - c.lo) / 3.92 == pytest.approx(np.sqrt(items2 + fits2 + it2), rel=1e-10)
    old = float(np.sqrt((pe.se ** 2).sum()) / k)
    assert old != pytest.approx((c.hi - c.lo) / 3.92, rel=1e-3)
    assert c.y == -pe["diff"].mean()


# ---- T15: the live-only judge tools read the render's rule --------------------------------------------------------------------------------
def test_the_live_only_judge_tools_pool_seeds_as_pool_cell_does(fresh):
    import judge_analysis as JA
    import petri_judge_analysis as PJA
    rng = np.random.default_rng(40)
    st = [f"b{i // 10}" for i in range(60)]
    w = World()
    a = rng.integers(2, 8, 60).astype(float)
    for e in R.PETRI_BEHAVIORS[:2]:
        for v, src, dr in SEED_VIEWS:
            w.run(f"none:{e}:{v}", a, strata=st, object_label="none", view_legacy=v, lane_source_legacy=src, draw=dr, eval_id=e,
                  metric_id="petri_score", item_set_id="petri", ci_lo=float(a.mean() - 0.3), ci_hi=float(a.mean() + 0.3))
        for i in range(3):
            x = rng.integers(1, 6, 60).astype(float)
            w.run(f"lora:{e}:s{i + 1}", x, strata=st, object_label="control_lora", seed=f"s{i + 1}", lane="base_to_instruct", object_id=f"o{i}",
                  eval_id=e, metric_id="petri_score", item_set_id="petri", ci_lo=float(x.mean() - 0.4), ci_hi=float(x.mean() + 0.4))
    R.reset_pooled_ci(w.provider)
    d = R.anchored(w.frame())
    pc = R.pool_cell(d[d.object_label.eq("control_lora")], y="d_y", lo="d_lo", hi="d_hi").set_index("eval_id")
    marks = JA.eval_marks(d, "Petri").set_index("column")
    po = PJA.pool_objects(d).set_index("eval_id")
    for e in R.PETRI_BEHAVIORS[:2]:
        assert (marks.loc[e, "hi"] - marks.loc[e, "lo"]) == pytest.approx(pc.loc[e, "d_hi"] - pc.loc[e, "d_lo"], rel=1e-12)
        assert (po.loc[e, "hi"] - po.loc[e, "lo"]) == pytest.approx(pc.loc[e, "d_hi"] - pc.loc[e, "d_lo"], rel=1e-12)
        assert marks.loc[e, "n_seeds"] == 3 and po.loc[e, "red"] == pytest.approx(-pc.loc[e, "d_y"], abs=1e-12)
