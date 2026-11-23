"""The table first rows on the notebook's full load chain (SISTEER_SLOW_TESTS=1): the committed render's numbers bound to the data.

The committed notebook's header and load cells and Figure 3's anchoring run against the source the environment names (the committed
paper-frame snapshot by default: about two minutes warm and about 4 GB of reassembled SQLite under paper/notebooks/_cache/, as the
control-pool slow module and tests/test_demo_objects.py's slow check). On the real anchored frame `d`, for each of the six Qwen per-seed tables
the notebook prints from it:
  * the committed render's table is exactly what `table_evals_seeds` prints from `d` now (the render is current with the code and data);
  * its ONE first row is, per eval, the mean over the fit seeds of the unsteered level the most of that seed's printed arms were
    differenced against (on a tie the mean of the tied levels), recomputed here from `d` with pandas alone, not through `seed_average`
    (the owner, 2026-11-14, after a first cut that averaged each seed's distinct readings);
  * every fitted arm's cell is its own change against its own unsteered run, oriented by the category: sign x d_y with its interval,
    read from `d` directly, so no cell is re-based onto the first row.
The control-pool appendix's two static tables, which the four declared control Pliny cells (read at 16,384, differenced against the
unsteered runs at 8,192) feed, are held to the same three checks on the lane cells' own frames, the first row's tie going to the reading
the whole column reads most as the README states.
Run it once on the final head and record its pass line."""
import collections
import contextlib
import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

SLOW = os.environ.get("SISTEER_SLOW_TESTS") == "1"
pytestmark = pytest.mark.skipif(not SLOW, reason="the notebook's full load chain on the committed snapshot (~2 min warm, about 4 GB cache): "
                                                 "SISTEER_SLOW_TESTS=1 opts in")
USES_ITEM_ROWS = True                     # the conftest's synthetic-frame provider stands down: the real frame's pooled cells read its item rows
REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
NOTEBOOK = NOTEBOOKS / "registry_qwen_figures.ipynb"
# (label, category, caption) of every Qwen per-seed table the notebook prints from `d` (make_registry_notebook.py, Figures 3, 4 and 6)
TABLES = [("tab:qwen-single-turn-seeds", "static", "Static defense for Qwen3.5-4B."),
          ("tab:qwen-multi-query-seeds", "multi-query", "Multi-query defense for Qwen3.5-4B."),
          ("tab:qwen-multi-turn-seeds", "multi-turn", "Multi-turn defense for Qwen3.5-4B."),
          ("tab:qwen-petri-seeds", "Petri", "Petri for Qwen3.5-4B."),
          ("tab:qwen-over-refusal-seeds", "over-refusal", "Over-refusal for Qwen3.5-4B."),
          ("tab:qwen-capability-seeds", "capability", "Capability for Qwen3.5-4B.")]
SEEDS = ("s1", "s2", "s3")
PRINT = 0.05 + 1e-9


def committed_tables() -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    out = {}
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            txt = "".join(o.get("text", "")) if o.get("output_type") == "stream" else "".join(o.get("data", {}).get("text/plain", ""))
            for t in re.findall(r"\\begin\{table\}.*?\\end\{table\}(?:\n% [^\n]*)*", txt, re.S):         # with its comment lines
                out[re.search(r"\\label\{([^}]*)\}", t).group(1)] = t
    return out


def body(tex: str) -> list:
    inner = tex.split("\\\\ \\hline \\\\\n", 1)[1].split("\n\\end{tabular}", 1)[0]
    return [[c.strip() for c in ln.rstrip().removesuffix("\\\\").split(" &")] for ln in inner.splitlines() if ln.strip()]


def num(cell: str):
    m = re.search(r"\$([+-]?\d+(?:\.\d+)?)\$", cell)
    return float(m.group(1)) if m else None


@pytest.fixture(scope="module")
def nb(notebook_rcparams):
    """The notebook namespace after the header and load cells and Figure 3's anchoring, as the control-pool slow module builds it."""
    import matplotlib
    matplotlib.use("Agg")
    saved_mod, saved_env, saved_cwd = sys.modules.get("registry_nb"), os.environ.get("REGISTRY_NB_SOURCE"), os.getcwd()
    saved_path = list(sys.path)
    try:
        os.chdir(NOTEBOOKS)
        os.environ["REGISTRY_NB_SOURCE"] = saved_env or "snapshot"
        sys.modules.pop("registry_nb", None)
        if str(NOTEBOOKS) not in sys.path:
            sys.path.insert(0, str(NOTEBOOKS))
        import registry_nb as R
        cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"}
        ns = {"__name__": "__table_levels_slow__"}
        with contextlib.redirect_stdout(sys.stderr):
            exec(cells["583e3f04"], ns)
            ns["R"] = R
            exec(cells["ec6aacd8"], ns)
            exec("d = R.anchored(op, pool=POOL_Q)", ns)
        ns["_committed"] = committed_tables()
        yield ns
    finally:
        os.chdir(saved_cwd)
        if saved_env is None:
            os.environ.pop("REGISTRY_NB_SOURCE", None)
        else:
            os.environ["REGISTRY_NB_SOURCE"] = saved_env
        sys.path[:] = saved_path                  # the notebook directory, and the SISTEER_SRC entry registry_nb inserts at import
        if saved_mod is None:                     # this module's fresh import never outlives it (tests/test_nb_snapshot_fallback.py)
            sys.modules.pop("registry_nb", None)
        else:
            sys.modules["registry_nb"] = saved_mod


@pytest.mark.parametrize("label,cat,cap", TABLES)
def test_the_committed_table_is_what_the_code_prints_from_the_data(nb, label, cat, cap):
    R, d = nb["R"], nb["d"]
    assert label in nb["_committed"], label
    assert R.table_evals_seeds(d, [[cat]], R.OBJECTS, cap, label) == nb["_committed"][label], label


@pytest.mark.parametrize("label,cat,cap", TABLES)
def test_the_first_row_is_the_mean_over_seeds_of_the_level_most_of_each_seeds_arms_read(nb, label, cat, cap):
    R, d = nb["R"], nb["d"]
    rows = body(nb["_committed"][label])
    firsts = [r for r in rows if r[0] == R.BASELINE_ROW_LABEL]
    assert len(firsts) == 1 and firsts[0][1] == "", firsts
    evals = list(R.categories_for(R.MODEL_ID)[cat][0])
    checked = 0
    for k, e in enumerate(evals, start=2):
        x = d[(d.model_id == R.MODEL_ID) & (d.eval_id == e) & d.d_y.notna() & d.anchor_value.notna() & d.object_label.isin(R.OBJECTS)]
        seed = x.seed.fillna("").astype(str).where(~x.object_label.isin(R.DECODE_ONLY), "")
        fitted = x[seed.isin(SEEDS)]
        if not len(fitted) or fitted.seed.nunique() < 2:
            continue                                                       # no seed replicates on this eval: not the case pinned here
        key = lambda k: ",".join(sorted({p.strip() for p in k.split(",") if p.strip()})) if isinstance(k, str) else ""
        run = fitted.anchor_run.map(key)
        lev = fitted.anchor_value.astype(float).map(lambda a: R.level(cat, a)).round(6)
        # A SEED'S TIE GOES TO THE TIED READING THE WHOLE COLUMN READS MOST, and failing that to the mean of the tied readings (#529's
        # rule, `seed_average`): every arm's reading votes once in the column, a seedless condition's draws once per distinct reading.
        # The re-cut frame holds such a tie (CodeAttack, seed 2: three arms on each of two readings).
        column, seen = collections.Counter(), set()
        for o, sd, r_, lv in zip(x.object_label, seed, x.anchor_run.map(key), x.anchor_value.astype(float).map(lambda a: R.level(cat, a)).round(6)):
            if sd == "" and (o, r_, lv) in seen:
                continue
            seen.add((o, r_, lv))
            column[(r_, lv)] += 1
        per_seed = []
        for sd in sorted(set(fitted.seed)):
            m = fitted.seed == sd
            n = collections.Counter(zip(run[m], lev[m]))
            tied = [k_ for k_, c in n.items() if c == max(n.values())]
            tied = [k_ for k_ in tied if column[k_] == max(column[t] for t in tied)]
            per_seed.append(float(np.mean(sorted({lv for (_r, lv) in tied}))))
        want = float(np.mean(per_seed))
        assert num(firsts[0][k]) == pytest.approx(want, abs=PRINT), (label, e, firsts[0][k], per_seed)
        checked += 1
    assert checked, label


@pytest.mark.parametrize("label,cat,cap", TABLES)
def test_every_fitted_cell_is_its_arms_own_oriented_change(nb, label, cat, cap):
    R, d = nb["R"], nb["d"]
    rows = body(nb["_committed"][label])
    sign = R.categories_for(R.MODEL_ID)[cat][1]
    evals = list(R.categories_for(R.MODEL_ID)[cat][0])
    x = d[(d.model_id == R.MODEL_ID) & d.eval_id.isin(evals) & d.d_y.notna() & d.object_label.isin(R.OBJECTS)
          & ~d.object_label.isin(R.DECODE_ONLY)]
    checked = 0
    for (o, sd, e), g in x.groupby([x.object_label, x.seed.fillna("").astype(str), x.eval_id]):
        lo, hi = sorted((sign * g.d_lo.mean(), sign * g.d_hi.mean()))
        row = next(r for r in rows if r[:2] == [R._tex(R.LABEL.get(o, o)), R._tex(sd or "--")])
        assert row[2 + evals.index(e)] == R._cell(sign * g.d_y.mean(), lo, hi, signed=True), (label, o, sd, e)
        checked += 1
    assert checked, label


# ---- the control-pool appendix's static tables -----------------------------------------------------------------------------------------
# The four declared control Pliny cells (#528: read at their 16,384 rung, differenced against the static panel's unsteered runs at 8,192)
# feed two tables the lane cells build from their own frames: the static per-seed table and the rule comparison's static table. They keep
# the same rule: one first row, the README's level recomputed here with pandas and collections alone, and every cell its arm's own change.
CTLSTOP_STATIC = "tab:qwen-ctlstop-static-seeds"
CTLSTOP_RULES_STATIC = "tab:qwen-ctlstop-rule-comparison-static-seeds"
LANE_CODE_CELLS = ("a6e0f2b8", "d2a7e5c9", "7fae15d6", "76fdab54")


@pytest.fixture(scope="module")
def lanes(nb, tmp_path_factory):
    """`nb` after the four control-pool lane code cells, as the committed notebook holds them (as the control-pool slow module runs
    them; the figures and tables they save go to a temporary directory)."""
    R = nb["R"]
    saved = R.FIGS
    cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"}
    try:
        R.FIGS = str(tmp_path_factory.mktemp("figs"))
        with contextlib.redirect_stdout(sys.stderr):
            exec("CAT_Q = R.categories_for(R.MODEL_ID)\ncm = R.category_means(d, categories=CAT_Q)", nb)
            exec(next(ln for ln in cells["ac902304"].splitlines() if ln.startswith("DCOLS = ")), nb)
            for cid in LANE_CODE_CELLS:
                exec(cells[cid], nb)
        yield nb
    finally:
        R.FIGS = saved


def _run_key(k) -> str:
    return ",".join(sorted({p.strip() for p in k.split(",") if p.strip()})) if isinstance(k, str) else ""


def rule_level(x, cat: str, R) -> float:
    """The README's first-row rule on the printed rows `x` of one column: per fit seed, the unsteered reading the most of that seed's
    fitted arms were differenced against (a reading = its runs and its level; a tie to the tied reading the whole column reads most,
    then the mean of the tied), averaged over the seeds; with no seed replicates, the same over every reading. A one-dot condition is
    seedless and its draws are one vote per distinct reading."""
    dot = x.object_label.isin(R.DECODE_ONLY)
    v = pd.DataFrame({"obj": x.object_label, "seed": x.seed.fillna("").astype(str).where(~dot, ""), "run": x.anchor_run.map(_run_key),
                      "lev": x.anchor_value.astype(float).map(lambda a: R.level(cat, a)).round(6), "one_dot": dot})
    v = pd.concat([v[~v.one_dot], v[v.one_dot].drop_duplicates(["obj", "seed", "run", "lev"])])
    column = collections.Counter(zip(v.run, v.lev))

    def most(g) -> float:
        n = collections.Counter(zip(g.run, g.lev))
        tied = [k for k, c in n.items() if c == max(n.values())]
        tied = [k for k in tied if column[k] == max(column[t] for t in tied)]
        return float(np.mean([lv for _r, lv in tied]))
    fit = v[v.seed.isin(SEEDS) & ~v.one_dot]
    if fit.seed.nunique() > 1:
        return float(np.mean([most(fit[fit.seed == sd]) for sd in sorted(set(fit.seed))]))
    return most(v)


def test_the_declared_pliny_cells_are_drawn_and_feed_both_static_tables(lanes):
    R, new = lanes["R"], lanes["ctl_st_new"]
    declared = new[new.declared_anchor_cap.notna() & new.d_y.notna()]
    assert len(declared) == 4 and set(declared.eval_id) == {"grok420"}, declared[["run_id", "eval_id", "object_label", "seed"]]
    assert set(declared.run_id) <= set(lanes["ctl_st_pts"].run_id) and set(declared.run_id) <= set(lanes["ctl_all_new"].run_id)


def test_the_control_pool_static_tables_are_what_the_code_prints_from_the_lane_frames(lanes):
    R, ns = lanes["R"], lanes
    assert R.table_evals_seeds(ns["ctl_st_pts"], [["static"]], R.OBJECTS,
                               "Static defense for Qwen3.5-4B with the control objects stopped on the control pool.", CTLSTOP_STATIC,
                               categories=ns["ctl_st_cats"], note=R.ctlstop_static_table_note(ns["ctl_st_new"])) == ns["_committed"][CTLSTOP_STATIC]
    assert R.table_rule_comparison_seeds(ns["ctl_pub_d"], ns["ctl_all_new"], ns["ctl_pub_cm"], ns["ctl_new_cm"],
                                         "Published and control-pool stopping, per seed, static (all its evaluations), for Qwen3.5-4B.",
                                         CTLSTOP_RULES_STATIC, categories={"static": ns["CTL_CATS"]["static"]}) == ns["_committed"][CTLSTOP_RULES_STATIC]


def test_the_control_pool_static_tables_first_row_is_the_rule_over_their_printed_arms(lanes):
    R, ns = lanes["R"], lanes
    evals = list(ns["ctl_st_cats"]["static"][0])
    pts = ns["ctl_st_pts"]
    st = body(ns["_committed"][CTLSTOP_STATIC])
    firsts = [r for r in st if r[0] == R.BASELINE_ROW_LABEL]
    assert len(firsts) == 1 and firsts[0][1] == "", firsts
    for k, e in enumerate(evals, start=2):
        x = pts[(pts.model_id == R.MODEL_ID) & (pts.eval_id == e) & pts.d_y.notna() & pts.anchor_value.notna() & pts.object_label.isin(R.OBJECTS)]
        assert num(firsts[0][k]) == pytest.approx(rule_level(x, "static", R), abs=PRINT), (CTLSTOP_STATIC, e, firsts[0][k])
    # the rule comparison: both rules' printed fits (the four refit slots, fit seeds), each rule's frame as the table reads it
    cmp_ = body(ns["_committed"][CTLSTOP_RULES_STATIC])
    firsts = [r for r in cmp_ if r[0] == R.BASELINE_ROW_LABEL]
    assert len(firsts) == 1 and firsts[0][1] == "", firsts
    frames, _cms = R._ctl_rule_frames(ns["ctl_pub_d"], ns["ctl_all_new"], ns["ctl_pub_cm"], ns["ctl_new_cm"], R.MODEL_ID, "static", evals)
    want = {}
    for k, e in enumerate(evals, start=2):
        x = pd.concat([f[(f.eval_id == e) & f.object_label.isin(R.CTLSTOP_SLOTS) & f.seed.astype(str).isin(R.CTLSTOP_SEEDS)
                         & f.anchor_value.notna()] for f in frames.values() if len(f)])
        want[e] = rule_level(x, "static", R)
        assert num(firsts[0][k]) == pytest.approx(want[e], abs=PRINT), (CTLSTOP_RULES_STATIC, e, firsts[0][k])
    assert num(firsts[0][2 + len(evals)]) == pytest.approx(np.mean(list(want.values())), abs=PRINT)


def test_every_control_pool_static_cell_is_its_arms_own_oriented_change(lanes):
    R, pts = lanes["R"], lanes["ctl_st_pts"]
    rows = body(lanes["_committed"][CTLSTOP_STATIC])
    evals = list(lanes["ctl_st_cats"]["static"][0])
    sign = lanes["ctl_st_cats"]["static"][1]
    x = pts[(pts.model_id == R.MODEL_ID) & pts.eval_id.isin(evals) & pts.d_y.notna() & pts.object_label.isin(R.OBJECTS)
            & ~pts.object_label.isin(R.DECODE_ONLY)]
    checked = 0
    for (o, sd, e), g in x.groupby([x.object_label, x.seed.fillna("").astype(str), x.eval_id]):
        lo, hi = sorted((sign * g.d_lo.mean(), sign * g.d_hi.mean()))
        row = next(r for r in rows if r[:2] == [R._tex(R.LABEL.get(o, o)), R._tex(sd or "--")])
        assert row[2 + evals.index(e)] == R._cell(sign * g.d_y.mean(), lo, hi, signed=True), (o, sd, e)
        checked += 1
    assert checked >= 20
