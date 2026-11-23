"""Every table's first row is ONE row (one per model in the all-model tables): the unsteered level of the quantity its figure's y axis
is a change in, AVERAGED OVER SEEDS; no cell is flagged for an arm whose own unsteered run differs from it; and every cell is the arm's
change against its OWN matched unsteered run, unchanged (the owner, 2026-11-14).

The level is defense success (100 - attack success), benign compliance (100 - over-refusal), the alignment score (11 - the 1-10
misalignment score, still 1-10 with 10 the most aligned), accuracy as it is (2026-11-11).
The average over seeds (`seed_average`): the mean over the fit seeds (s1, s2, s3) of each seed's level, a seed's level being the
unsteered reading the MOST of that seed's arms were differenced against, a plurality and not necessarily a majority (each arm one
vote, a ladder or a one-dot condition's draws included; a tie goes to the reading the whole column reads most, then to the mean, and
the captions point to the README for it); with no seed replicates, the reading the most of the arms were differenced against. An arm on another lane, cap or view keeps its own run and does not move the
first row. #524 printed instead a `vs a` beside every cell on another anchor (`own_anchor`) and one first row per seed in three
per-seed tables (`seed_split_reason`); a first cut of the owner's rule averaged each seed's DISTINCT readings, giving one paired-lane
arm half its seed's weight (2026-11-14). The tests below fail on both. Synthetic frames throughout, with the seeds' anchors
deliberately different: every arm's reading y is its own anchor plus its raw delta, and a cell's expected text is computed from the
frame, never read back from the table. In the committed render, two distillation-target tables print their one first row under a label
of their own (`OWN_FIRST_ROW`, which gives the reasons): the paper's Table 5, held byte-equal instead to its pin, the table the
manuscript must print, and the table on the 92 goals outside the training pool."""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import registry_nb as R  # noqa: E402

Q, G = R.MODEL_ID, "gemma4e4b_it"
PETRI0 = R.PETRI_BEHAVIORS[0]
TOP = {"static": 100.0, "multi-query": 100.0, "multi-turn": 100.0, "over-refusal": 100.0, "Petri": 11.0, "capability": None}
SIGN = {c: v[1] for c, v in R.CATEGORIES.items()}
PRINT = 0.05 + 1e-9         # a printed 1-dp number: the most it can sit from the exact value


def level(cat, v):
    """A raw reading as the figure's quantity: complemented where the category's registry metric is lower-is-better."""
    return v if TOP[cat] is None else TOP[cat] - v


def most_read(readings: list) -> float:
    """The level of a group of arms from [the unsteered reading each arm was differenced against]: the reading the most arms share,
    on a tie the mean of the tied readings."""
    vals = [round(v, 6) for v in readings]
    top = max(vals.count(v) for v in vals)
    return float(np.mean(sorted({v for v in vals if vals.count(v) == top})))


def seed_avg(by_seed: dict) -> float:
    """The expected first row from {seed: [the unsteered reading each of that seed's arms was differenced against]}."""
    return float(np.mean([most_read(v) for v in by_seed.values()]))


def row(model, obj, eval_id, y, anchor, seed="s1", half=4.0, **kw):
    """One anchored row: the arm's raw reading `y` against the unsteered `anchor`, its raw delta y - anchor with a +/- `half` interval."""
    d_y = y - anchor
    return dict(model_id=model, object_label=obj, seed=seed, draw="", eval_id=eval_id, y=y, lo=y - half, hi=y + half, d_y=d_y,
                d_lo=d_y - half, d_hi=d_y + half, anchor_in_frame=True, ceiling=False, anchor_value=anchor, held_out=0,
                route="common anchor (1 runs)", **kw)


def body(tex: str) -> list:
    """The tabular's rows as lists of cells."""
    inner = tex.split("\\\\ \\hline \\\\\n", 1)[1].split("\n\\end{tabular}", 1)[0]
    return [[c.strip() for c in ln.rstrip().removesuffix("\\\\").split(" &")] for ln in inner.splitlines() if ln.strip()]


def num(cell: str):
    """The value a cell prints (its first math span), or None for `--`."""
    m = re.search(r"\$([+-]?\d+(?:\.\d+)?)\$", cell)
    return float(m.group(1)) if m else None


def half_width(cell: str):
    m = re.search(r"\\pm (\d+(?:\.\d+)?)", cell)
    return float(m.group(1)) if m else None


def caption(tex: str) -> str:
    return re.search(r"\\caption\{(.*?)\}\n\\label", tex, re.S).group(1)


def cell_of(rows: list, *lead) -> list:
    return next(r for r in rows if r[:len(lead)] == list(lead))


def firsts(rows: list) -> list:
    return [r for r in rows if r[0] == R.BASELINE_ROW_LABEL]


def no_flags(tex: str):
    """No cell and no caption names another unsteered run: no `vs a`, in small type or in words."""
    assert "scriptsize vs" not in tex and "vs $" not in tex and not re.search(r"\bvs\b", caption(tex)), tex


def oriented(cat, d_y, d_lo, d_hi) -> str:
    """The cell an arm's single reading prints: its change oriented by the category's sign, with its interval."""
    s = SIGN[cat]
    lo, hi = sorted((s * d_lo, s * d_hi))
    return R._cell(s * d_y, lo, hi, signed=True)


def says_the_average_and_the_approximation(cap: str, quantity: str):
    """The caption states the first row's definition briefly and claims no exactness for first row + cell."""
    assert "averaged over seeds" in cap or "no seed replicates" in cap, cap
    assert "the unsteered reading the most " in cap and "differenced against" in cap and "reading most of" not in cap, cap
    assert cap.count(R.SEED_AVERAGE_TIES) == 1, cap                                  # the tie rule: one pointer to the README
    assert "against the arm's own unsteered run" in cap or "against its own unsteered run" in cap or "against the fit's own" in cap, cap
    assert f"approximates the" in cap and quantity in cap, cap
    assert "exactly where" in cap and "$=$" not in cap and "the anchor most" not in cap and "one per seed" not in cap, cap


# ---- the level map -----------------------------------------------------------------------------------------------------------------------
class TestTheLevelMap:
    def test_every_category_has_a_level_and_exactly_the_lower_is_better_ones_are_complemented(self):
        for cat, (_evals, sign, _label) in R.CATEGORIES.items():
            assert cat in R.LEVEL, cat
            assert (R.LEVEL[cat][0] is None) == (sign == +1), cat
            assert R.LEVEL[cat][0] == TOP[cat], cat

    def test_a_reading_and_its_interval_are_complemented_on_their_own_scale(self):
        assert R.level("static", 30.0) == 70.0 and R.level("over-refusal", 5.0) == 95.0
        assert R.level("Petri", 6.2) == pytest.approx(4.8)
        assert R.level("capability", 73.8) == 73.8
        assert R.level_interval("multi-turn", 20.0, 35.0) == (65.0, 80.0)          # (lo, hi) -> (100 - hi, 100 - lo)
        assert R.level_interval("Petri", 5.0, 7.5) == (3.5, 6.0)                     # (lo, hi) -> (11 - hi, 11 - lo)
        assert R.level_interval("capability", 1.0, 2.0) == (1.0, 2.0)

    def test_the_pooled_petri_score_belongs_to_petri(self):
        assert R.category_of("petri_pooled") == "Petri" and R.category_of("pair") == "multi-query" and R.category_of("nope") is None

    def test_the_figure_axes_name_the_quantities_that_rise_with_alignment(self):
        assert R.CATEGORIES["Petri"][2] == "Δ alignment score (1–10)"
        assert R.CATEGORIES["static"][2] == "Δ defense success (pp)" and R.CATEGORIES["over-refusal"][2] == "Δ benign compliance (pp)"


# ---- the average over seeds ----------------------------------------------------------------------------------------------------------------
class TestTheSeedAverage:
    def test_it_is_the_mean_over_seeds_of_each_seeds_unsteered_reading(self):
        # Pliny: s1's arms against runs at 4,096 tokens (44.8 % defense), s2's and s3's against runs at 8,192 (43.4 %)
        rd = [(sd, None, v, True) for sd, v in (("s1", 44.8), ("s2", 43.4), ("s3", 43.4)) for _arm in range(4)]
        assert R.seed_average(rd) == pytest.approx((44.8 + 43.4 + 43.4) / 3)
        assert R.seed_replicated(rd)

    def test_a_seeds_level_is_the_run_most_of_its_arms_read_and_the_one_arm_on_another_run_does_not_move_it(self):
        # s1: four arms on its own view's run (45.9), one on its paired view's run (46.8); s2 and s3 alike
        rd = ([("s1", "v1", 45.9, True)] * 4 + [("s1", "p1", 46.8, True)] + [("s2", "v2", 45.4, True)] * 4 + [("s2", "p2", 45.0, True)]
              + [("s3", "v3", 46.1, True)] * 4 + [("s3", "p3", 46.1, True)])
        assert R.seed_average(rd) == pytest.approx((45.9 + 45.4 + 46.1) / 3)
        # PAIR's shape in the committed snapshot: five arms per seed on the pool (56.18 % defense), the paired-write difference ReLU
        # on its lane's own run (52.90): the first row is the pool's level, not 54.5 (the distinct readings' mean)
        pair = [(sd, k, v, True) for sd in ("s1", "s2", "s3") for k, v, n in (("pool", 56.18, 5), ("paired", 52.90, 1)) for _ in range(n)]
        assert R.seed_average(pair) == pytest.approx(56.18)

    def test_a_tie_goes_to_the_run_the_whole_column_reads_most_and_only_then_to_the_mean(self):
        # each seed: the difference LoRA on the standup's run (3.0 % defense), the paired-write difference ReLU on its lane's (2.7);
        # the seedless conditions read the standup's run too, so every seed's tie goes to 3.0 (the distinct mean said 2.85)
        rd = [(sd, k, v, True) for sd in ("s1", "s2", "s3") for k, v in (("pool", 3.0), ("lane", 2.7))]
        rd += [("", "pool", 3.0, False)] * 3
        assert R.seed_average(rd) == pytest.approx(3.0)
        assert R.seed_average(rd[:6]) == pytest.approx(2.85), "with nothing to break the tie, the mean of the tied readings"
        assert R.seed_average([("s1", "a", 70.0, True), ("s1", "b", 72.0, True), ("s2", "c", 71.0, True)]) == pytest.approx((71.0 + 71.0) / 2)
        assert R.seed_average([("", "a", 70.0, True), ("", "b", 72.0, True), ("", "b", 72.0, True), ("", "a", 70.0, True)]) == pytest.approx(71.0)

    def test_a_seeds_level_is_the_plurality_reading_even_without_a_majority(self):
        # five arms per seed on four runs: two on A, one each on B, C and D. A is read by two of five, no majority, and still sets the
        # level; the captions say "the most", never "most of" (2026-11-14)
        rd = [(sd, k, v, True) for sd in ("s1", "s2") for k, v, n in (("A", 50.0, 2), ("B", 58.0, 1), ("C", 61.0, 1), ("D", 64.0, 1))
              for _ in range(n)]
        assert R.seed_average(rd) == pytest.approx(50.0)
        phrase = R.seed_average_phrase({True})
        assert "the unsteered reading the most arms of each seed" in phrase and "most of each" not in phrase, phrase
        assert phrase.endswith(f"; {R.SEED_AVERAGE_TIES})") and "README" in R.SEED_AVERAGE_TIES
        assert R.seed_average_phrase({False}, who="that model's") == (
            f"the unsteered reading the most of that model's arms were differenced against (no seed replicates; {R.SEED_AVERAGE_TIES})")

    def test_a_seedless_conditions_draws_are_one_vote(self):
        # a companion model: two fits on run A, a one-dot condition whose three draws all read run B
        rows = pd.DataFrame([dict(anchor_value=40.0, anchor_run="B", seed="")] * 3)
        draws = R.anchor_readings(rows, "static", fitted=False)
        assert draws == [("", "B", 60.0, False)]
        assert R.seed_average([("", "A", 55.0, True), ("", "A2", 55.0, True)] + draws) == pytest.approx((55.0 + 55.0 + 60.0) / 3), \
            "three distinct runs read once each: a three-way tie"
        assert R.seed_average([("", None, 55.0, True), ("", None, 55.0, True)] + draws) == pytest.approx(55.0), "two fits beat one draw set"

    def test_readings_are_told_apart_by_their_anchor_runs_and_by_level_only_without_them(self):
        two_runs_one_rate = [("", "run-a", 45.0, True), ("", "run-b", 45.0, True), ("", "run-c", 48.0, True)]
        assert R.seed_average(two_runs_one_rate) == pytest.approx(46.0), "two runs that read the same rate are two readings, a three-way tie"
        assert R.seed_average([("", None, 45.0, True), ("", None, 45.0, True), ("", None, 48.0, True)]) == pytest.approx(45.0)

    def test_seedless_conditions_count_only_where_there_are_no_seed_replicates(self):
        fitted = [("s1", "a", 70.0, True), ("s2", "a", 70.0, True), ("s3", "b", 73.0, True)]
        si = [("", "c", 60.0, False)]
        assert R.seed_average(fitted + si) == pytest.approx(71.0), "the one-dot condition is not a seed"
        assert R.seed_average([("s1", "a", 70.0, True)] + si) == pytest.approx(65.0), "one fit seed is no replicate: every reading counts"
        assert R.seed_average(si + [("", "d", 64.0, False)]) == pytest.approx(62.0)
        assert R.seed_average([("s1", "a", 70.0, True)] + si * 2) == pytest.approx(60.0), "without replicates the most-read reading"
        assert not R.seed_replicated([("s1", "a", 70.0, True)] + si) and R.seed_replicated(fitted)

    def test_no_level_is_none(self):
        assert R.seed_average([]) is None and R.seed_average([("s1", None, None, True), ("s2", None, np.nan, True)]) is None

    def test_the_old_choice_rules_are_gone(self):
        for name in ("reference", "own_anchor", "seed_split_reason", "OWN_ANCHOR_SENTENCE"):
            assert not hasattr(R, name), name


# ---- the per-eval, per-seed tables (Figures 3-6, the alpha = -1 table, the control-pool multi-turn table) --------------------------------
@pytest.mark.parametrize("cat,eval_id,anchor,phrase", [
    ("static", "pasttense", 35.2, "defense success rate (\\%; 100 $-$ attack success rate)"),
    ("multi-query", "pair", 44.3, "defense success rate (\\%; 100 $-$ attack success rate)"),
    ("multi-turn", "crescendo", 9.2, "defense success rate (\\%; 100 $-$ attack success rate)"),
    ("over-refusal", "xstest", 5.0, "benign compliance rate (\\%; 100 $-$ over-refusal rate)"),
    ("Petri", PETRI0, 7.0, "alignment score (1--10; 11 $-$ Petri misalignment score, so 10 is the most aligned)"),
    ("capability", "mmlu", 73.8, "accuracy or pass rate (\\%)"),
])
def test_per_eval_table_has_one_first_row_the_seed_average_and_every_cell_is_the_arms_own_change(cat, eval_id, anchor, phrase):
    """Three fit seeds on three different unsteered runs, a paired lane's run under one arm of every seed, and a one-dot condition on a
    run of its own: one first row, the mean over the seeds of the run two of each seed's three arms read (neither the paired lane's run
    nor the one-dot condition's moves it); every cell its arm's change against its own run, unmarked."""
    u = 0.2 if cat == "Petri" else 1.0
    seed_anchor = {"s1": anchor, "s2": anchor + 1.3 * u, "s3": anchor - 0.9 * u}
    lane = anchor + 2.6 * u
    rows_ = [row(Q, "safety_si", eval_id, anchor - 12.0 * u, anchor + 4.0 * u, seed="")]
    for sd, a in seed_anchor.items():
        rows_ += [row(Q, "control_lora", eval_id, a + 4.0 * u, a, seed=sd), row(Q, "constitution_lora", eval_id, a - 9.0 * u, a, seed=sd),
                  row(Q, "difference_relu", eval_id, lane - 7.0 * u, lane, seed=sd)]
    d = pd.DataFrame(rows_)
    objects = ["safety_si", "control_lora", "constitution_lora", "difference_relu"]
    tex = R.table_evals_seeds(d, [[cat]], objects, "Title.", "tab:t")
    rows = body(tex)
    k = 2 + R.categories_for(Q)[cat][0].index(eval_id)
    assert len(firsts(rows)) == 1 and rows[0][0] == R.BASELINE_ROW_LABEL and rows[0][1] == ""
    want = seed_avg({sd: [level(cat, a), level(cat, a), level(cat, lane)] for sd, a in seed_anchor.items()})
    assert want == pytest.approx(np.mean([level(cat, a) for a in seed_anchor.values()]))
    assert num(rows[0][k]) == pytest.approx(want, abs=PRINT), (cat, rows[0][k], want)
    for _, r in d.iterrows():
        sd = r.seed or "--"
        assert cell_of(rows, R._tex(R.LABEL[r.object_label]), sd)[k] == oriented(cat, r.d_y, r.d_lo, r.d_hi), (cat, r.object_label, sd)
    no_flags(tex)
    cap = caption(tex)
    assert cap.startswith("Title. First row: the unsteered model's " + phrase + ", averaged over seeds (the mean over fit seeds of the "
                          "unsteered reading the most arms of each seed were differenced against; ties broken as the notebook README "
                          "states).")
    says_the_average_and_the_approximation(cap, R.LEVEL[cat][1])
    name = R.LEVEL[cat][1]
    assert f"the change in {name} against the arm's own unsteered run ({R.LEVEL[cat][3]}; positive is better)" in cap
    assert f"so first row $+$ cell approximates the arm's {name} after the intervention, exactly where the arm's own unsteered run " \
           "equals the first row." in cap
    says_the_average_and_the_approximation(cap, name)


def test_per_eval_first_row_interval_is_complemented_with_its_value():
    """The first row's pooled interval (the unsteered runs pooled as draws, `pool_se`; a synthetic frame has no item rows, so the
    identical-draws bound: the mean of their Wilson intervals) moves with the value: centred on 100 - the anchor, half-width unchanged."""
    d = pd.DataFrame([dict(model_id=Q, object_label="none", eval_id="pair", metric_id="asr", value=0.44, n=259, d_y=np.nan, anchor_value=np.nan,
                           seed="", d_lo=np.nan, d_hi=np.nan),
                      dict(model_id=Q, object_label="none", eval_id="pair", metric_id="asr", value=0.47, n=259, d_y=np.nan, anchor_value=np.nan,
                           seed="", d_lo=np.nan, d_hi=np.nan),
                      dict(model_id=Q, object_label="safety_si", eval_id="pair", metric_id="asr", value=0.30, n=259, d_y=-15.5, d_lo=-20.0,
                           d_hi=-11.0, anchor_value=45.5, seed="", y=30.0)])
    first = body(R.table_evals_seeds(d, [["multi-query"]], ["safety_si"], "c", "tab:p"))[0]
    se = np.mean([100 * (R.wilson(p * 259, 259)[1] - R.wilson(p * 259, 259)[0]) / 3.92 for p in (0.44, 0.47)])
    assert num(first[3]) == pytest.approx(54.5) and half_width(first[3]) == pytest.approx(round(1.96 * se, 1)), first


def test_a_mixed_table_names_each_column_groups_level_and_averages_each_column_over_seeds():
    """The alpha = -1 table: static and multi-turn columns read defense success, the over-refusal columns benign compliance; on ReNeLLM
    s1's arms read runs at another cap (76.1 % defense where s2 and s3 read 76.3 %), which #524 printed as a `vs`."""
    cats = {c: R.CATEGORIES[c] for c in ("static", "multi-turn", "over-refusal")}
    seed_anchor = {"renellm": {"s1": 23.9, "s2": 23.7, "s3": 23.7}, "crescendo": {"s1": 0.8, "s2": 0.8, "s3": 0.8},
                   "xstest": {"s1": 4.7, "s2": 6.1, "s3": 5.2}}
    objects = ["control_lora", "control_relu"]
    d = pd.DataFrame([row(Q, o, e, a + 5.0 + 3 * j, a, seed=sd) for e, by in seed_anchor.items() for sd, a in by.items()
                      for j, o in enumerate(objects)])
    tex = R.table_evals_seeds(d, [["static", "multi-turn", "over-refusal"]], objects, "Alpha.", "tab:n", categories=cats)
    rows = body(tex)
    evals = [e for c in cats.values() for e in c[0]]
    cat_of = {e: c for c, v in cats.items() for e in v[0]}
    assert len(firsts(rows)) == 1
    for e, by in seed_anchor.items():
        k = 2 + evals.index(e)
        assert num(rows[0][k]) == pytest.approx(seed_avg({sd: [level(cat_of[e], a)] for sd, a in by.items()}), abs=PRINT), e
        for _, r in d[d.eval_id == e].iterrows():
            assert cell_of(rows, R._tex(R.LABEL[r.object_label]), r.seed)[k] == oriented(cat_of[e], r.d_y, r.d_lo, r.d_hi)
    no_flags(tex)
    cap = caption(tex)
    assert ("First row: the unsteered model's defense success rate (\\%; 100 $-$ attack success rate) on Past tense, CodeAttack, "
            "ReNeLLM, Pliny, Crescendo and ActorAttack and benign compliance rate (\\%; 100 $-$ over-refusal rate) on XSTest and "
            "OR-Bench-Hard, averaged over seeds (") in cap
    assert "so first row $+$ cell approximates the arm's value of that quantity after the intervention, exactly where" in cap


def pair_frame():
    """PAIR's shape in the committed snapshot: the seed fits against their standups' pool (43.8 % ASR), the paired-write difference
    ReLU against its lane's own unsteered run (47.1 %), the Safety SI against the pool; one Control ReLU fit fully jailbroken."""
    rows = []
    for i, sd in enumerate(("s1", "s2", "s3")):
        rows.append(row(Q, "control_relu", "pair", 100.0, 43.8, seed=sd, anchor_run="pool"))
        rows.append(row(Q, "control_lora", "pair", 43.8 + 53.1 + i, 43.8, seed=sd, anchor_run="pool"))
        rows.append(row(Q, "difference_relu", "pair", 47.1 - (5.8, 45.6, 40.2)[i], 47.1, seed=sd, anchor_run="paired-lane"))
    rows.append(row(Q, "safety_si", "pair", 43.8 - 19.6, 43.8, seed="", anchor_run="pool"))
    return pd.DataFrame(rows)


def test_pair_first_row_is_the_pool_most_arms_read_and_no_cell_is_flagged():
    """The case under the owner's rule of 2026-11-14: two of each seed's three arms read the pool (56.2 % defense), the
    paired-write difference ReLU its lane's run (52.9 %), so the first row is the pool's 56.2 (the distinct readings' mean, 54.6, gave
    the one paired-lane arm half the weight); the Difference ReLU steer's cells print their own change against the paired lane's run
    (+45.6 on s2), unmarked."""
    d = pair_frame()
    objects = ["safety_si", "control_relu", "control_lora", "difference_relu"]
    tex = R.table_evals_seeds(d, [["multi-query"]], objects, "PAIR.", "tab:p")
    rows = body(tex)
    k = 3                                                                            # Best-of-75, then PAIR
    assert len(firsts(rows)) == 1 and num(rows[0][k]) == pytest.approx(56.2, abs=PRINT), rows[0]
    assert cell_of(rows, "Difference ReLU steer", "s2")[k] == oriented("multi-query", -45.6, -49.6, -41.6)
    assert cell_of(rows, "Control ReLU steer", "s1")[k].startswith("$-56.2$")
    for _, r in d.iterrows():
        assert cell_of(rows, R._tex(R.LABEL[r.object_label]), r.seed or "--")[k] == oriented("multi-query", r.d_y, r.d_lo, r.d_hi)
    no_flags(tex)


@pytest.mark.parametrize("cat,eval_id,views,lane", [
    ("capability", "bigcodebench", {"s1": 45.9, "s2": 45.4, "s3": 46.1}, {"s1": 46.8, "s2": 45.0, "s3": 46.1}),
    ("over-refusal", "xstest", {"s1": 5.2, "s2": 6.0, "s3": 4.6}, {"s1": 4.7, "s2": 4.7, "s3": 4.7}),
])
def test_fit_seeds_on_their_own_views_average_into_one_first_row(cat, eval_id, views, lane):
    """BigCodeBench, BFCL, XSTest and OR-Bench-Hard read each fit seed against its own serving view's unsteered run, the paired difference
    ReLU against its paired view's (or its lane's one run), the one-dot conditions against the mean of the views their runs were served
    on. #524 printed a first row per seed and one for the one-dot conditions; now one, the mean over seeds of the view run most of each
    seed's arms read."""
    rows_ = []
    for sd, a in views.items():
        own = dict(anchor_views=f"qwen35_4b_{sd}", anchor_cap=2048.0, anchor_run=f"view-{sd}")
        for j, o in enumerate(("control_lora", "constitution_lora")):
            rows_.append(row(Q, o, eval_id, a + (-3.0 if cat == "capability" else 2.0) * (j + 1), a, seed=sd, **own))
        rows_.append(row(Q, "difference_relu", eval_id, lane[sd] + 7.0, lane[sd], seed=sd, anchor_run=f"paired-{sd}"))
        rows_.append(row(Q, "safety_si", eval_id, a + 4.0, a, seed="", **own))       # one run per view: one dot
    d = pd.DataFrame(rows_)
    objects = ["safety_si", "control_lora", "constitution_lora", "difference_relu"]
    tex = R.table_evals_seeds(d, [[cat]], objects, "Views.", "tab:v")
    rows = body(tex)
    evals = list(R.categories_for(Q)[cat][0])
    k = 2 + evals.index(eval_id)
    assert [r[1] for r in firsts(rows)] == [""], firsts(rows)
    assert num(rows[0][k]) == pytest.approx(np.mean([level(cat, a) for a in views.values()]), abs=PRINT), rows[0]
    si = d[d.object_label == "safety_si"]
    lo_, hi_ = R.cell_interval(si)[1:3]                         # the dot's interval (hand-built deltas: the identical-draws bound)
    assert cell_of(rows, "Safety SI", "--")[k] == R._cell(SIGN[cat] * si.d_y.mean(), *sorted((SIGN[cat] * lo_, SIGN[cat] * hi_)), signed=True)
    for _, r in d[d.object_label != "safety_si"].iterrows():
        assert cell_of(rows, R._tex(R.LABEL[r.object_label]), r.seed)[k] == oriented(cat, r.d_y, r.d_lo, r.d_hi)
    no_flags(tex)
    assert "First row: the unsteered model's " + R._level_phrase(cat) + ", averaged over seeds" in caption(tex)
    assert "serving views" not in caption(tex) and "-- is the conditions" not in caption(tex)


def test_seeds_read_at_different_generation_caps_average_into_one_first_row():
    """Pliny's shape in the committed snapshot: s1's arms against the unsteered runs at a 4,096-token cap (55.2 % ASR), s2's and s3's
    against those at 8,192 (56.6 %), and s1's Difference ReLU steer against the 8,192 runs too. #524 printed three first rows and named
    the caps in the caption; now one, the mean over seeds of the cap most of each seed's arms read (s1's one 8,192 arm does not move it)."""
    objects = ["control_lora", "constitution_lora", "control_relu", "constitution_relu", "difference_relu"]
    rows_ = []
    for sd, (a, c) in {"s1": (55.2, 4096.0), "s2": (56.6, 8192.0), "s3": (56.6, 8192.0)}.items():
        for j, o in enumerate(objects[:4]):
            rows_.append(row(Q, o, "grok420", a - 3.0 * (j + 1), a, seed=sd, anchor_cap=c, anchor_run=f"cap{int(c)}"))
        rows_.append(row(Q, "difference_relu", "grok420", 56.6 - 20.0, 56.6, seed=sd, anchor_cap=8192.0, anchor_run="cap8192"))
    d = pd.DataFrame(rows_)
    tex = R.table_evals_seeds(d, [["static"]], objects, "Static.", "tab:s")
    rows = body(tex)
    k = 2 + R.categories_for(Q)["static"][0].index("grok420")
    assert len(firsts(rows)) == 1 and num(rows[0][k]) == pytest.approx(100 - (55.2 + 56.6 + 56.6) / 3, abs=PRINT), rows[0]
    for _, r in d.iterrows():
        assert cell_of(rows, R._tex(R.LABEL[r.object_label]), r.seed)[k] == oriented("static", r.d_y, r.d_lo, r.d_hi)
    no_flags(tex)
    assert "generation cap" not in caption(tex) and "4,096" not in caption(tex)


def test_a_column_without_seed_replicates_takes_the_reading_most_arms_read_and_says_so():
    """Only seedless conditions (or one fit seed) in the frame: the reading most of the arms were differenced against (the pool, three
    of four; the vendor arm's run does not move it), and the caption says there are no seed replicates rather than claiming an average
    over seeds."""
    d = pd.DataFrame([row(Q, "safety_si", "pair", 30.0, 44.0, seed="", anchor_run="pool"),
                      row(Q, "constitution_si", "pair", 32.0, 44.0, seed="", anchor_run="pool"),
                      row(Q, "arditi_allpos", "pair", 20.0, 47.0, seed="", anchor_run="vendor"),
                      row(Q, "control_lora", "pair", 50.0, 44.0, seed="s1", anchor_run="pool")])
    objects = ["arditi_allpos", "safety_si", "constitution_si", "control_lora"]
    tex = R.table_evals_seeds(d, [["multi-query"]], objects, "PAIR.", "tab:p")
    rows = body(tex)
    assert len(firsts(rows)) == 1 and num(rows[0][3]) == pytest.approx(100 - 44.0, abs=PRINT), rows[0]
    no_flags(tex)
    cap = caption(tex)
    assert (", the unsteered reading the most of its arms were differenced against (no seed replicates; ties broken as the notebook "
            "README states).") in cap and "averaged" not in cap


# ---- the all-model category tables ---------------------------------------------------------------------------------------------------------
CAT_EVALS = {"static": ["pasttense", "codeattack"], "over-refusal": ["xstest", "or_bench_hard"], "Petri": ["petri_pooled"],
             "capability": ["mmlu", "gsm8k"]}
# the categories narrowed to the evals the frames carry, so no arm's mean is over fewer evaluations than its category names
# (Petri keeps its behaviors: its mean is the registry's pooled eval)
CATS_T = {c: (R.CATEGORIES[c][0] if c == "Petri" else CAT_EVALS[c], R.CATEGORIES[c][1], R.CATEGORIES[c][2]) for c in CAT_EVALS}


def model_frame(cat: str):
    """Qwen: a fitted arm with three seeds whose unsteered runs differ per seed, a one-dot condition on a run of its own; Gemma (no seed
    replicates): a LoRA and a ReLU steer on two different unsteered runs, and the one-dot condition on the LoRA's, so the LoRA's run is
    the one most of Gemma's arms read. Per (model, eval, seed) the anchors; readings that differ per eval."""
    u = 0.1 if cat == "Petri" else 1.0
    base = {e: (6.0 + j if cat == "Petri" else 30.0 + 5 * j) for j, e in enumerate(CAT_EVALS[cat])}
    shift = {"s1": 0.0, "s2": 1.6 * u, "s3": -0.7 * u}
    rows, anchors = [], {}
    for e, a in base.items():
        for sd, sh in shift.items():
            anchors[(Q, e, sd)] = a + sh
            rows.append(row(Q, "control_lora", e, a + sh + 3.0 * u, a + sh, seed=sd))
        rows.append(row(Q, "safety_si", e, a + 2.0 * u - 8.0 * u, a + 2.0 * u, seed=""))
        ag, ar = a + 10.0 * u, a + 12.5 * u
        anchors[(G, e, "lora")], anchors[(G, e, "relu")] = ag, ar
        rows += [row(G, "control_lora", e, ag + 4.0 * u, ag, seed=""), row(G, "control_relu", e, ar - 6.0 * u, ar, seed=""),
                 row(G, "safety_si", e, ag - 1.0 * u, ag, seed="")]
    return pd.DataFrame(rows), anchors


@pytest.mark.parametrize("cat", list(CAT_EVALS))
def test_category_table_first_row_is_each_models_seed_average_and_cells_are_unchanged(cat):
    d, anchors = model_frame(cat)
    cm = R.category_means(d, categories=CATS_T)
    objects = ["safety_si", "control_lora", "control_relu"]
    tex = R.table_categories_models(cm, cat, objects, [Q, G], f"{cat} across the four models.", "tab:c", categories=CATS_T)
    rows = body(tex)
    assert len(firsts(rows)) == 1 and rows[0][0] == R.BASELINE_ROW_LABEL
    evs = CAT_EVALS[cat]
    q = np.mean([level(cat, np.mean([anchors[(Q, e, sd)] for e in evs])) for sd in ("s1", "s2", "s3")])     # over Qwen's seeds
    g = level(cat, np.mean([anchors[(G, e, "lora")] for e in evs]))       # Gemma: the LoRA's run, read by two of its three arms
    assert num(rows[0][1]) == pytest.approx(q, abs=PRINT) and num(rows[0][2]) == pytest.approx(g, abs=PRINT), (cat, rows[0], q, g)
    pooled = R.pool_seeds(cm)
    for o in objects:
        for k, m in enumerate((Q, G), start=1):
            p = pooled[(pooled.model_id == m) & (pooled.object_label == o) & (pooled.category == cat)]
            want = R._cell(p.y.iat[0], p.lo.iat[0], p.hi.iat[0], signed=True) if len(p) else "--"
            assert cell_of(rows, R._tex(R.LABEL[o]))[k] == want, (cat, o, m)
    no_flags(tex)
    cap = caption(tex)
    assert cap.startswith(f"{cat} across the four models. First row: each model's unsteered " + R._level_phrase(cat))
    assert ("averaged over seeds (the mean over fit seeds of the unsteered reading the most arms of each seed were differenced against; "
            "where there are no seed replicates, the unsteered reading the most of that model's arms were differenced against; ties "
            "broken as the notebook README states).") in cap
    says_the_average_and_the_approximation(cap, R.LEVEL[cat][1])


def test_category_table_keeps_a_partial_arm_out_of_the_first_row_and_says_so():
    """Nemotron's constitution ReLU steer has no GSM8K reading: its capability mean is over three evaluations, and so is its anchor, so
    the first row reads the full-coverage arms alone (here Gemma's two LoRAs on two different runs, a tie: their mean)."""
    rows_ = [row(G, "control_lora", e, a + 4.0, a, seed="") for e, a in {"mmlu": 70.0, "gsm8k": 90.0}.items()]
    rows_ += [row(G, "constitution_lora", e, a - 6.0, a, seed="") for e, a in {"mmlu": 72.0, "gsm8k": 91.0}.items()]
    rows_ += [row(G, "constitution_relu", "mmlu", 60.0 - 3.0, 60.0, seed="")]                 # no GSM8K
    d = pd.DataFrame(rows_)
    cm = R.category_means(d, categories=CATS_T)
    objects = ["control_lora", "constitution_lora", "constitution_relu"]
    tex = R.table_categories_models(cm, "capability", objects, [G], "Capability.", "tab:c", categories=CATS_T)
    rows = body(tex)
    assert num(rows[0][1]) == pytest.approx((80.0 + 81.5) / 2, abs=PRINT), "the full-coverage arms' readings, not the partial arm's 60"
    assert cell_of(rows, "Constitution ReLU steer")[1] == R._cell(-3.0, -7.0, 1.0, signed=True)
    no_flags(tex)
    assert ("Gemma-4-E4B's Constitution ReLU steer (MMLU) is a mean over fewer of the category's evaluations, against its own "
            "unsteered level over them, and the first row leaves it out.") in caption(tex), caption(tex)


# ---- the tables under the absolute appendix figures ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("cat", list(CAT_EVALS))
def test_absolute_table_has_one_first_row_per_model_each_its_seed_average_and_oriented_unchanged_cells(cat):
    d, anchors = model_frame(cat)
    evals = CAT_EVALS[cat]
    tex = R.table_absolute(d, evals, ["none", "safety_si", "control_lora", "control_relu"], [Q, G], "Absolute.", "tab:a")
    rows = body(tex)
    assert [r[1] for r in firsts(rows)] == [R._tex(R.MODEL_LABEL[Q]), R._tex(R.MODEL_LABEL[G])]
    p = R.delta_points(d, evals, ["safety_si", "control_lora", "control_relu"])
    for k, e in enumerate(evals, start=2):
        q = np.mean([level(cat, anchors[(Q, e, sd)]) for sd in ("s1", "s2", "s3")])
        g = level(cat, anchors[(G, e, "lora")])
        assert num(cell_of(rows, R.BASELINE_ROW_LABEL, R._tex(R.MODEL_LABEL[Q]))[k]) == pytest.approx(q, abs=PRINT), (cat, e)
        assert num(cell_of(rows, R.BASELINE_ROW_LABEL, R._tex(R.MODEL_LABEL[G]))[k]) == pytest.approx(g, abs=PRINT), (cat, e)
        for _, r in p[p.eval_id == e].iterrows():
            assert cell_of(rows, R._tex(R.LABEL[r.object_label]), R._tex(R.MODEL_LABEL[r.model_id]))[k] == R._cell(r.y, r.lo, r.hi, signed=True)
    si = num(cell_of(rows, "Safety SI", R._tex(R.MODEL_LABEL[Q]))[2])
    assert (si > 0) == (cat != "capability"), "the Safety SI lowered every raw reading: better everywhere but capability"
    no_flags(tex)
    cap = caption(tex)
    assert cap.startswith("Absolute. First rows, one per model: each model's unsteered " + R._level_phrase(cat) + ", averaged over seeds")
    assert "so the model's first row $+$ cell approximates the arm's" in cap and "equals the model's first row" in cap
    says_the_average_and_the_approximation(cap, R.LEVEL[cat][1])


def test_absolute_table_takes_no_flip_switch():
    d, _ = model_frame("static")
    with pytest.raises(TypeError):
        R.table_absolute(d, ["pasttense"], ["none", "safety_si"], [Q], "c", "tab:a", flip_baseline=True)


def test_absolute_table_keeps_a_held_out_cell_out_of_the_first_row():
    d = pd.DataFrame([row(G, "safety_si", "pasttense", 45.0, 55.0, seed=""), row(G, "control_lora", "pasttense", 50.0, 52.0, seed=""),
                      row(G, "control_relu", "pasttense", 28.0, 40.0, seed="")])
    d["held_out"] = [0, 0, 1]
    tex = R.table_absolute(d, ["pasttense"], ["safety_si", "control_lora", "control_relu"], [G], "c", "tab:a")
    rows = body(tex)
    assert num(firsts(rows)[0][2]) == pytest.approx(100 - (55.0 + 52.0) / 2, abs=PRINT), "the held-out arm's 40 stays out"
    assert cell_of(rows, "Control ReLU steer")[2] == R._cell(12.0, 8.0, 16.0, signed=True) + "$^{\\S}$", "kept, marked held out"
    no_flags(tex)


# ---- the padding ladder ------------------------------------------------------------------------------------------------------------------
def test_padding_table_first_row_is_the_seed_average_at_each_rung_and_its_caption_says_so():
    """A seeded arm whose seeds read different same-rung runs and an unseeded condition on its own: per rung one first-row cell, the mean
    over the seeds (the unseeded condition is no seed)."""
    base = {0: {"s1": 44.8, "s2": 46.0, "s3": 45.1}, 8192: {"s1": 52.5, "s2": 52.5, "s3": 51.0}}
    pts = pd.DataFrame([dict(object_label="control_lora", seed=sd, pad=float(p), delta=-5.0, lo=-13.0, hi=3.0, defence=b - 5.0,
                             base_defence=b, held=False, pending=False) for p, by in base.items() for sd, b in by.items()]
                       + [dict(object_label="safety_si", seed="", pad=float(p), delta=19.7, lo=11.7, hi=27.7, defence=40.0 + 19.7,
                               base_defence=40.0, held=False, pending=False) for p in base])
    pts.attrs["eval_id"] = "grok420"
    tex = R.padding_table(pts, ["safety_si", "control_lora"])
    rows = body(tex)
    assert len(firsts(rows)) == 1
    for k, (p, by) in enumerate(base.items(), start=2):
        assert num(rows[0][k]) == pytest.approx(np.mean(list(by.values())), abs=PRINT), (p, rows[0])
    assert cell_of(rows, "Safety SI")[2] == R._cell(19.7, 11.7, 27.7, signed=True)
    assert cell_of(rows, "Control LoRA", "s2")[3] == R._cell(-5.0, -13.0, 3.0, signed=True)
    no_flags(tex)
    cap = caption(tex)
    assert cap.startswith("Padding ladder for Qwen3.5-4B. First row: the unsteered model's defense success rate (\\%; 100 $-$ attack "
                          "success rate) on Pliny at each filler rung, one column per rung, averaged over seeds (")
    assert "so first row $+$ cell approximates the defense success rate at that rung after the intervention, exactly where" in cap
    pts.attrs = {}
    assert " on Pliny" not in caption(R.padding_table(pts, ["safety_si"])), "an eval the points do not carry is not named"


# ---- the frontier ------------------------------------------------------------------------------------------------------------------------
def frontier_rows(specs):
    """Per-seed rungs (object, seed, dose, dy, dx, ay_abs, ax_abs)."""
    return pd.DataFrame([dict(object_label=o, seed=sd, dose=ds, dy=dy, dx=dx, ay_abs=ay, ax_abs=ax) for o, sd, ds, dy, dx, ay, ax in specs])


def test_frontier_first_row_is_the_seed_average_of_the_rungs_levels_and_rungs_are_unchanged():
    """The paired difference ReLU's ladder reads its lane's own unsteered run (2.7 % multi-turn defense where the LoRAs' rungs read 3.0;
    s3's difference LoRA compliance reads 60.0), the prompted diamond the mean of its runs'. One first row: per seed the level most of
    its rungs read, then the mean over seeds; the prompted rows and the Refusal steer are no seed."""
    ps = frontier_rows([("difference_lora", "s1", 1.0, 24.1, -12.1, 3.0, 59.6), ("difference_lora", "s2", 1.0, 25.0, -7.1, 3.0, 59.6),
                        ("difference_lora", "s3", 1.0, 41.1, -24.2, 3.0, 60.0),
                        ("control_lora", "s1", 1.0, 2.0, -1.0, 3.0, 59.6), ("control_lora", "s2", 1.0, 2.5, -1.5, 3.0, 59.6),
                        ("control_lora", "s3", 1.0, 3.0, -2.0, 3.0, 59.6),
                        ("difference_relu", "s1", 1.0, 15.8, -1.8, 2.7, 59.6), ("difference_relu", "s2", 1.0, 32.4, -14.1, 2.7, 60.0),
                        ("difference_relu", "s3", 1.0, 40.7, -25.1, 2.7, 59.6), ("arditi_allpos", "", 0.3, 7.5, -13.0, 9.0, 50.0),
                        ("safety_si", "", 1.0, 4.0, -7.0, 3.0, 59.6), ("safety_si", "", 1.0, 5.0, -6.0, 3.4, 59.6)])
    objects = ["arditi_allpos", "control_lora", "difference_lora", "difference_relu", "safety_si"]
    tex = R.table_frontier([dict(title="multi-turn", pts=ps.iloc[:1], per_seed=ps)], objects, "{panel} rungs.", "tab:f")
    rows = body(tex)
    assert len(firsts(rows)) == 1
    want_y = seed_avg({"s1": [3.0, 3.0, 2.7], "s2": [3.0, 3.0, 2.7], "s3": [3.0, 3.0, 2.7]})
    want_x = seed_avg({"s1": [59.6, 59.6, 59.6], "s2": [59.6, 59.6, 60.0], "s3": [60.0, 59.6, 59.6]})
    assert (want_y, want_x) == (pytest.approx(3.0), pytest.approx(59.6))
    assert num(rows[0][3]) == pytest.approx(want_y, abs=PRINT) and num(rows[0][4]) == pytest.approx(want_x, abs=PRINT), rows[0]
    assert cell_of(rows, "Difference ReLU steer", "s1")[3:5] == [R._cell(15.8, signed=True), R._cell(-1.8, signed=True)]
    assert cell_of(rows, "Safety SI")[3:5] == [R._cell(4.5, signed=True), R._cell(-6.5, signed=True)], "the diamond: its runs' mean"
    no_flags(tex)
    cap = caption(tex)
    assert cap.startswith("Multi-turn rungs. First row: the unsteered model, the figure's origin, its defense success rate (\\%; 100 $-$ "
                          "attack success rate")
    # a ladder votes once, so the caption counts arms, not rungs (the paired ReLU's four rungs do not outvote the LoRA's three)
    assert ("each averaged over seeds (the mean over fit seeds of the unsteered reading the most arms of each seed were differenced "
            "against; ties broken as the notebook README states)." in cap) and "rungs were differenced" not in cap
    assert "Each rung below: its change in each against its own unsteered run (pp; positive is better)" in cap
    assert "approximates the rung's defense success or benign compliance rate, exactly where" in cap and "$=$" not in cap


def test_frontier_ladders_vote_once_per_fit_so_the_first_row_is_the_standups_run():
    """The committed multi-turn frontier's shape: per seed the difference LoRA's three or four rungs on the standup's run (3.0 % defense)
    and the paired-write difference ReLU's four on its lane's (2.7), the Refusal steer's and the SIs' rows on the standup's run. Counted
    per rung the ReLU outvoted the LoRA on s2 and tied it on s1 and s3 (first row 2.8); a ladder is one fit, so each seed ties, and the
    tie goes to the run the column reads most: 3.0, the figure's origin."""
    specs = []
    for sd, n_lora in (("s1", 4), ("s2", 3), ("s3", 4)):
        specs += [("difference_lora", sd, 0.25 * (i + 1), 5.0 * i, -3.0 * i, 3.0, 59.6) for i in range(n_lora)]
        specs += [("difference_relu", sd, 0.25 * (i + 1), 4.0 * i, -2.0 * i, 2.7, 59.6) for i in range(4)]
    specs += [("arditi_allpos", "", d_, 2.0, -1.0, 3.0, 59.6) for d_ in (0.15, 0.3, 0.5)] + [("safety_si", "", 1.0, 4.4, -7.0, 3.0, 59.6)]
    ps = frontier_rows(specs)
    tex = R.table_frontier([dict(title="multi-turn", pts=ps.iloc[:1], per_seed=ps)], ["arditi_allpos", "difference_lora", "difference_relu",
                                                                                        "safety_si"], "{panel} rungs.", "tab:f")
    first = firsts(body(tex))
    assert len(first) == 1 and num(first[0][3]) == pytest.approx(3.0, abs=PRINT) and num(first[0][4]) == pytest.approx(59.6, abs=PRINT), first
    no_flags(tex)


# ---- the stopping-rule comparison --------------------------------------------------------------------------------------------------------
def rule_frames(anchors: dict, own: dict | None = None):
    """Both stopping rules' readings of every slot and seed; `own` moves one (rule, slot, seed) onto another anchor."""
    rows_pub, rows_new = [], []
    for o in R.CTLSTOP_SLOTS:
        for i, sd in enumerate(R.CTLSTOP_SEEDS):
            for e, a in anchors.items():
                ap = (own or {}).get(("published", o, sd), a)
                rows_pub.append(row(Q, o, e, ap - 1.0 - i, ap, seed=sd))
                rows_new.append(row(Q, o, e, a - 3.0 - i, a, seed=sd))
    return pd.DataFrame(rows_pub), pd.DataFrame(rows_new)


def test_rule_comparison_first_row_is_the_seed_average_and_its_mean_column_averages_the_attacks():
    """One published fit reads another unsteered run (97.0 % ASR on both attacks, where the rest read 99.2 and 94.9): its seed's
    level is still the run the rest of its fits read, and no change is flagged."""
    mt = {"multi-turn": R.CATEGORIES["multi-turn"]}
    anchors = {"crescendo": 99.2, "actorattack": 94.9}
    slot, sd = R.CTLSTOP_SLOTS[-1], R.CTLSTOP_SEEDS[1]
    pub, new = rule_frames(anchors, own={("published", slot, sd): 97.0})
    tex = R.table_rule_comparison_seeds(pub, new, R.category_means(pub, categories=mt), R.category_means(new, categories=mt), "Rules.",
                                        "tab:r", categories=mt)
    rows = body(tex)
    assert len(firsts(rows)) == 1
    n = 2 * len(R.CTLSTOP_SLOTS)                                                   # each seed's fits: every slot under both rules
    want = {e: seed_avg({s: [100 - a] * (n - (s == sd)) + ([100 - 97.0] if s == sd else []) for s in R.CTLSTOP_SEEDS}) for e, a in anchors.items()}
    assert want == {e: pytest.approx(100 - a) for e, a in anchors.items()}
    assert [num(c) for c in rows[0][2:4]] == [pytest.approx(want["crescendo"], abs=PRINT), pytest.approx(want["actorattack"], abs=PRINT)]
    assert num(rows[0][4]) == pytest.approx(np.mean(list(want.values())), abs=PRINT)
    moved = cell_of(rows, R._tex(R.LABEL[slot]), sd)[2].split(" / ")
    r = pub[(pub.object_label == slot) & (pub.seed == sd) & (pub.eval_id == "crescendo")].iloc[0]
    assert moved[0] == oriented("multi-turn", r.d_y, r.d_lo, r.d_hi), moved
    no_flags(tex)
    cap = caption(tex)
    assert cap.startswith("Rules. First row: the unsteered model's defense success rate (\\%; 100 $-$ attack success rate), averaged over "
                          "seeds (the mean over fit seeds of the unsteered reading the most fits of each seed were differenced against; "
                          "ties broken as the notebook README states); its mean column averages the category's evaluations.")
    assert "first row $+$ either change approximates that fit's defense success rate after the intervention, exactly where" in cap


# ---- the two forms the notebook does not render (kept to the same rule) -------------------------------------------------------------------
def test_unrendered_category_and_dose_tables_average_over_seeds_too():
    rows_ = []
    for e in ("pasttense", "codeattack"):
        for sd, a in {"s1": 30.0, "s2": 31.0, "s3": 32.5}.items():
            rows_ += [row(Q, "control_lora", e, a - 5.0, a, seed=sd), row(Q, "constitution_lora", e, a - 7.0, a, seed=sd),
                      row(Q, "difference_relu", e, a + 2.0 - 6.0, a + 2.0, seed=sd)]
    d = pd.DataFrame(rows_)
    objects = ["control_lora", "constitution_lora", "difference_relu"]
    tex = R.table_categories(R.category_means(d, categories=CATS_T), [["static"]], objects, "Static.", "tab:s", model=Q)
    rows = body(tex)
    assert len(firsts(rows)) == 1
    assert num(rows[0][1]) == pytest.approx(np.mean([100 - a for a in (30.0, 31.0, 32.5)]), abs=PRINT), rows[0]
    no_flags(tex)
    pts = pd.DataFrame([dict(object_label="difference_relu", seed=sd, dose=ds, dy=dy, ay_abs=ay)
                        for sd, ds, dy, ay in (("s1", 0.5, 3.0, 2.7), ("s2", 1.0, 15.8, 3.0), ("s3", 1.5, 30.0, 3.4))])
    t = R.table_dose_response(pts, ["difference_relu"], "Dose.", "tab:d")
    assert num(body(t)[0][3]) == pytest.approx((2.7 + 3.0 + 3.4) / 3, abs=PRINT) and "vs " not in t
    assert [c[3] for c in body(t)[1:]] == [R._cell(3.0, signed=True), R._cell(15.8, signed=True), R._cell(30.0, signed=True)]


# ---- the transfer table: no first row, and a caption that says what a cell is ------------------------------------------------------------
def test_transfer_caption_states_what_a_cell_is_its_units_and_its_direction():
    cats = {c: R.TRANSFER_CATEGORIES[c] for c in ("static", "Petri", "capability")}
    per_cat = pd.DataFrame([dict(model_id=G, object_label="control_lora", category=c, y=1.5, lo=0.5, hi=2.5, partial=False) for c in cats])
    pairs = pd.DataFrame([dict(model_id=G, object_label="control_lora", status="matched", base_batch=8, n_base_fits=1, it_batch=64)])
    tex = R.table_transfer(per_cat, pairs, ["control_lora"], [G], cats, "Transfer.", "tab:x")
    assert not firsts(body(tex))
    assert caption(tex) == ("Transfer. No first row: each cell is the base$\\to$instruct fit's change against the unsteered model minus the "
                            "instruct-direct fit's, each the mean over the category's evaluations, in defense success rate on static (pp), "
                            "alignment score on Petri (points) and accuracy or pass rate on capability (pp); positive means the transferred "
                            "fit did better, with its 95\\% interval.")


# ---- the committed render ----------------------------------------------------------------------------------------------------------------
def committed_tables() -> dict:
    """{label: LaTeX} of every table the committed notebook prints."""
    nb = json.loads((NOTEBOOKS / "registry_qwen_figures.ipynb").read_text())
    out = {}
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            txt = "".join(o.get("text", "")) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", ""))
            for t in re.findall(r"\\begin\{table\}.*?\\end\{table\}", txt, re.S):
                out[re.search(r"\\label\{([^}]*)\}", t).group(1)] = t
    return out


# the appendix analyses that read committed data files print the manuscript's booktabs tables, not registry_nb.latex_table's form; their
# first rows (the caperrors table's by registry_nb.seed_average) and captions are pinned by their own tests, paper/tests/test_appendix_*.py
DATA_FILE_TABLES = ("tab:corpus-audit", "tab:corpus-audit-full", "tab:app_evenodd_signs", "tab:caperrors", "tab:caperrors-bcb")

#: the two tables the committed render prints whose one first row carries a label of its own instead of `R.BASELINE_ROW_LABEL`, with
#: that label. Narrow on purpose: each must still print exactly one such row, as its first body row, and no row under the rule's label.
#:  * tab:app_si_distilled_mt is the paper's Table 5 (the distillation-target figure's table). It is printed in the manuscript's form,
#:    the first row `\emph{unsteered}` and the caption included, and held byte-equal instead to its pin, the table the manuscript
#:    must print (MANUSCRIPT_TABLE_5); the rule's caption phrases do not apply to it.
#:  * tab:qwen-si-target-outside-pool reads every arm on the 92 goals outside the distillation training pool, and its first row names
#:    that goal set (the unsteered level on those 92 goals, not on all 259 as the table above it); its caption keeps the rule's phrases.
OWN_FIRST_ROW = {R.SITARGET_PAPER_TABLE_LABEL: "\\emph{unsteered}",
                 "tab:qwen-si-target-outside-pool": f"\\emph{{unsteered (absolute, {R.SITARGET_OUTSIDE_POOL_N} goals)}}"}
#: Table 5 as the manuscript must print it, which is the committed render's (the manuscript's form without its `\rev{}`
#: revision-highlight wrapper)
MANUSCRIPT_TABLE_5 = pathlib.Path(__file__).resolve().parent / "data" / "si_target_paper_table.tex"


def seed_average_caption(lab: str, tex: str):
    """The rule's caption: the first row averaged over seeds (or no replicates), the plurality, the tie rule pointed to once."""
    cap = caption(tex)
    assert "one per seed" not in cap and ("averaged over seeds" in cap or "(no seed replicates" in cap), lab
    # the plurality, never "most of" (a majority it need not be), and the tie rule pointed to once
    assert "the unsteered reading the most " in cap and "reading most of" not in cap, lab
    assert cap.count(R.SEED_AVERAGE_TIES) == 1, lab


def test_the_committed_render_prints_one_first_row_per_table_or_model_and_no_flag_anywhere():
    tables = committed_tables()
    assert len(tables) >= 25 and "tab:qwen-capability-seeds" in tables and "tab:allmodels-absolute-capability" in tables
    # an exemption whose table the render no longer prints is stale, and is removed rather than left to excuse a later table
    assert set(OWN_FIRST_ROW) <= set(tables), sorted(set(OWN_FIRST_ROW) - set(tables))
    for lab, tex in tables.items():
        assert "scriptsize vs" not in tex and "vs $a$" not in tex, lab
        if lab in DATA_FILE_TABLES:
            continue
        rows = body(tex)
        fr = firsts(rows)
        if lab.startswith("tab:allmodels-absolute"):
            assert len(fr) == len({r[1] for r in fr}) >= 2, (lab, [r[1] for r in fr])          # one per model
        elif lab in OWN_FIRST_ROW:
            own = [r for r in rows if r[0] == OWN_FIRST_ROW[lab]]
            assert not fr and len(own) == 1 and rows[0][0] == OWN_FIRST_ROW[lab], (lab, fr, own)
            if lab == R.SITARGET_PAPER_TABLE_LABEL:
                paper = re.search(r"\\begin\{table\}.*?\\end\{table\}", MANUSCRIPT_TABLE_5.read_text(encoding="utf-8"), re.S).group(0)
                assert tex == paper, lab
            else:
                seed_average_caption(lab, tex)
        elif lab not in ("tab:allmodels-transfer", "tab:qwen-relu-arditi-cosines"):
            assert len(fr) == 1, (lab, fr)
            seed_average_caption(lab, tex)
