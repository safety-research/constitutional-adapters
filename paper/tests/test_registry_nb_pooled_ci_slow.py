"""The item-clustered pooled intervals on the committed paper frame (SISTEER_SLOW_TESTS=1), under the default policy.

The notebook's header and load cells run against the committed snapshot, then the two anchorings the figures read (`d`, Figures 3 to
6; `d_all`, the all-model appendix), their category means, and the control-pool appendix's main frames. Pinned:
  * MEANS DO NOT MOVE: every delta, reading, anchor value and category mean equals the render's before the change
    (`data/pooled_ci_baseline_means.json`, written from the frames of the render at the base commit, d3a60af77);
  * every pooled run's units reproduce its reading (the anchoring would have raised UnitsMismatch otherwise), the item ids of runs
    declaring one item set align (the smallest overlap is pinned), and exactly three anchors take the identical-draws bound: the
    Gemma and Nemotron MMLU anchors and the Llama GSM8K anchor (their runs carry no item rows in the frame), 28 delta rows;
  * a few intervals: the Qwen MMLU and OR-Bench-Hard anchors' se, the Constitution SI's MMLU dot and the Safety SI's CodeAttack dot;
  * the control-pool appendix's units from its file's payloads give each main run's clustered variance from the frame's item rows.
About two minutes warm and about 4 GB of reassembled SQLite under paper/notebooks/_cache/, as the other slow modules."""
import contextlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

SLOW = os.environ.get("SISTEER_SLOW_TESTS") == "1"
pytestmark = pytest.mark.skipif(not SLOW, reason="the notebook's full load chain on the committed snapshot (~2 min warm, about 4 GB cache): "
                                                 "SISTEER_SLOW_TESTS=1 opts in")
USES_ITEM_ROWS = True                     # the conftest's synthetic-frame provider stands down: these tests read the frame's item rows
REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
NOTEBOOK = NOTEBOOKS / "registry_qwen_figures.ipynb"
BASELINE = pathlib.Path(__file__).resolve().parent / "data" / "pooled_ci_baseline_means.json"
#: the anchors the frame holds no item rows for (model, eval): their runs come from exports outside the frame's item set
BOUND_ANCHORS = {("gemma4e4b_it", "mmlu"), ("nemotron3nano4b", "mmlu"), ("llama8b_it", "gsm8k")}
BOUND_ROWS = 28
TOL = 0.01


@pytest.fixture(scope="module")
def nb(notebook_rcparams):
    import matplotlib
    matplotlib.use("Agg")
    saved_mod, saved_env, saved_cwd = sys.modules.get("registry_nb"), os.environ.get("REGISTRY_NB_SOURCE"), os.getcwd()
    saved_path = list(sys.path)
    saved_judges = {k: os.environ.pop(k, None) for k in ("REGISTRY_NB_JUDGES", "REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE",
                                                        "REGISTRY_NB_PETRI_EXCLUDE")}
    try:
        os.chdir(NOTEBOOKS)
        os.environ["REGISTRY_NB_SOURCE"] = saved_env or "snapshot"
        sys.modules.pop("registry_nb", None)
        if str(NOTEBOOKS) not in sys.path:
            sys.path.insert(0, str(NOTEBOOKS))
        import registry_nb as R
        R.reset_pooled_ci(None)
        cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"}
        ns = {"__name__": "__pooled_ci_slow__"}
        with contextlib.redirect_stdout(sys.stderr):
            exec(cells["583e3f04"], ns)
            ns["R"] = R
            exec(cells["ec6aacd8"], ns)
            exec("d = R.anchored(op, pool=POOL_Q)\ncm = R.category_means(d, categories=R.categories_for(R.MODEL_ID))\n"
                 "d_all = R.anchored(op_all, pool=POOL_Q, setup='any')\ncm_all = R.category_means(d_all, categories=R.APPENDIX_CATEGORIES)\n", ns)
            ns["REPORT"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in R.POOLED_CI_REPORT.items()}
            exec("ctl_rows, ctl_pending, _ctl_file = R.ctlstop_load(R.SOURCE, allow_pending=True)\n"
                 "ctl_op, ctl_pool, ctl_main = R.ctlstop_main_frames(op, POOL_Q, d, _ctl_file)\n", ns)
        yield ns
    finally:
        os.chdir(saved_cwd)
        if saved_env is None:
            os.environ.pop("REGISTRY_NB_SOURCE", None)
        else:
            os.environ["REGISTRY_NB_SOURCE"] = saved_env
        for k, v in saved_judges.items():
            if v is not None:
                os.environ[k] = v
        sys.path[:] = saved_path                  # the notebook directory, and the SISTEER_SRC entry registry_nb inserts at import
        if saved_mod is None:                     # this module's fresh import never outlives it (tests/test_nb_snapshot_fallback.py)
            sys.modules.pop("registry_nb", None)
        else:
            sys.modules["registry_nb"] = saved_mod


def _key(*vals) -> str:
    return "|".join(str(v) for v in vals)


@pytest.mark.parametrize("name", ["d", "d_all"])
def test_no_delta_reading_or_anchor_moves(nb, name):
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["frames"][name]
    d = nb[name]
    got = {_key(r.model_id, r.object_label, r.seed, r.eval_id, r.metric_id, r.run_id): [r.y, r.d_y, r.anchor_value] for r in d.itertuples()}
    assert set(got) == set(base)
    for k, want in base.items():
        for g, w in zip(got[k], want):
            assert (w is None and pd.isna(g)) or g == pytest.approx(w, abs=1e-9), (k, got[k], want)


@pytest.mark.parametrize("name", ["cm", "cm_all"])
def test_no_category_mean_moves(nb, name):
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["frames"][name]
    got = {_key(r.model_id, r.category, r.object_label, r.seed): r.y for r in nb[name].itertuples()}
    assert set(got) == set(base)
    for k, w in base.items():
        assert (w is None and pd.isna(got[k])) or got[k] == pytest.approx(w, abs=1e-9), k


def test_every_pool_aligns_and_only_the_three_anchors_without_item_rows_take_the_bound(nb):
    R, rep = nb["R"], nb["REPORT"]
    assert rep["alignment_min"] == 1.0                    # every pool's runs on one item set hold the same items (unscored included)
    bound = {(v["model_id"], v["eval_id"]) for v in rep["bound"].values()}
    assert bound == BOUND_ANCHORS, bound
    assert all(v["what"] == "anchor" for v in rep["bound"].values())
    d, d_all = nb["d"], nb["d_all"]
    assert not d.anchor_se_method.astype(str).str.startswith(R.BOUND_NOTE).any()
    b = d_all[d_all.anchor_se_method.astype(str).str.startswith(R.BOUND_NOTE) & d_all.d_y.notna()]
    assert len(b) == BOUND_ROWS and set(zip(b.model_id, b.eval_id)) == BOUND_ANCHORS
    assert R.BOUND_NOTE in R.pooled_ci_note(b)


def _anchor_se(d, eval_id, k):
    x = d[(d.model_id == "qwen35_4b_it") & (d.eval_id == eval_id) & (d.anchor_draws == k)]
    vals = sorted(set(x.anchor_se.round(9)))
    assert len(vals) == 1, (eval_id, k, vals)
    return vals[0]


def test_pinned_intervals(nb):
    R, d = nb["R"], nb["d"]
    assert _anchor_se(d, "mmlu", 8) == pytest.approx(1.123, abs=TOL)
    assert _anchor_se(d, "or_bench_hard", 15) == pytest.approx(0.862, abs=TOL)
    dots = R.pool_draws(d[d.object_label.isin(R.DECODE_ONLY) & d.d_y.notna()]).set_index(["object_label", "eval_id"])
    for (o, e), (lo, hi) in {("constitution_si", "mmlu"): (-4.87, 1.42), ("safety_si", "codeattack"): (-17.83, -10.07)}.items():
        r = dots.loc[(o, e)]
        assert (r.d_lo, r.d_hi) == (pytest.approx(lo, abs=TOL), pytest.approx(hi, abs=TOL)), (o, e, r.d_lo, r.d_hi)
        assert r.n_draws == 3


def test_the_control_pool_files_units_give_the_frames_clustered_variance(nb):
    R = nb["R"]
    P = R._pooled_ci()
    ctl = nb["ctl_op"]
    ctl = ctl[ctl.judge_source.eq(R.CTLSTOP_MAIN_SOURCE)]
    frame = pd.concat([nb["op"], nb["POOL_Q"]], ignore_index=True)
    frame = R.intervals(R.dedupe(frame[frame.headline & frame.judge_ok]))
    by = {(str(r.run_id), str(r.metric_id), str(r.judgment_id)): r for _, r in frame.iterrows()}
    checked = 0
    for _, c in ctl.iterrows():
        m = by.get((str(c.run_id), str(c.metric_id), str(c.judgment_id)))
        if m is None or not np.isfinite(float(c.value)) or float(c.value) != pytest.approx(float(m.value), abs=1e-12):
            continue
        uc, um = R.row_units(c), R.row_units(m)
        assert P.cluster_var([uc]) == pytest.approx(P.cluster_var([um]), rel=1e-9), c.run_id
        assert uc.n == um.n
        checked += 1
    assert checked >= 50, checked
