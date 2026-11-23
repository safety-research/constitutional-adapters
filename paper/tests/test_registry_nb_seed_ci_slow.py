"""The item-clustered fit-seed pools on the committed paper frame (SISTEER_SLOW_TESTS=1).

The committed notebook's code cells run against the committed snapshot (every cell but the zip and the record, figures into a temporary
directory: about 16 minutes and about 4 GB of reassembled SQLite under paper/notebooks/_cache/), under the default policy, and the marks
the figures and tables draw from seed pools are recomputed from the render's frames: every mean mark (pool_seeds over every category-means
frame), the appendix's per-eval and absolute marks and their mean columns, the transfer differences per eval and per category, the padding
means and the generator figure's mean marks. Pinned:
  * MEANS DO NOT MOVE: every mark's mean equals the base render's EXACTLY (data/seed_ci_baseline.json, written from the frames of the
    render at the base commit), and the same marks exist;
  * NO MARK LOSES ITS INTERVAL: every mark with an interval at the base has one now (the Qwen Difference ReLU Petri marks gain one);
  * every seed pool is item-clustered: no seed-pool bound and no opaque seed pool under the default policy, the anchors' bounds as before
    (the three anchors whose runs carry no item rows in the frame), the two transfer cells kept in their old composition and no other,
    and every pool's fits on the same item ids;
  * a few intervals: the Difference ReLU Petri and Control ReLU capability mean marks, the Difference ReLU MMLU and Control LoRA CodeAttack
    per-eval marks, the Control LoRA padding mean at pad 0, a transfer cell, a mean column and a transfer category.
The filed-judges render (`primary`) completing from the committed frame, with its seed-pool bounds named, is the second fixture."""
import contextlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

SLOW = os.environ.get("SISTEER_SLOW_TESTS") == "1"
pytestmark = pytest.mark.skipif(not SLOW, reason="the whole notebook on the committed snapshot (~16 min per policy, about 4 GB cache): "
                                                 "SISTEER_SLOW_TESTS=1 opts in")
USES_ITEM_ROWS = True                     # the conftest's synthetic-frame provider stands down: these tests read the frame's item rows
REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
NOTEBOOK = NOTEBOOKS / "registry_qwen_figures.ipynb"
BASELINE = pathlib.Path(__file__).resolve().parent / "data" / "seed_ci_baseline.json"
#: the cells not executed: the figure zip and the record (they read and write figures/ and the cache listing only)
SKIP_CELLS = {"b7d4e9f2", "23555668"}
#: the anchors the frame holds no item rows for (model, eval), as test_registry_nb_pooled_ci_slow.py names them
BOUND_ANCHORS = {("gemma4e4b_it", "mmlu"), ("nemotron3nano4b", "mmlu"), ("llama8b_it", "gsm8k")}
ATTACKS = ["pasttense", "codeattack", "renellm", "grok420", "strongreject_bon", "pair", "crescendo", "actorattack"]
OR_EVALS = ["xstest", "or_bench_hard"]
TOL = 0.01
_MODS = ("registry_nb", "pooled_ci", "judge_rules", "appendix_common", "appendix_generator", "judge_analysis", "petri_judge_analysis",
         "appendix_geometry_evenodd")


def _run(policy: str, figs: pathlib.Path) -> dict:
    """Every code cell of the committed notebook in one namespace under `policy`, from a fresh import of this checkout's modules."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ("REGISTRY_NB_JUDGES", "REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_PETRI_EXCLUDE", "REGISTRY_NB_FIGS",
            "REGISTRY_NB_SOURCE")
    saved_env, saved_cwd, saved_path = {k: os.environ.get(k) for k in keys}, os.getcwd(), list(sys.path)
    #: every module of the notebook directory as it stood, so the fresh imports below never outlive the render
    #: (tests/test_nb_snapshot_fallback.py and the other slow fixtures put back the same); and every one of them imported FRESH, as a
    #: render's kernel imports them: a module an earlier test imported holds the `registry_nb` it imported then (the conftest's, with
    #: no item source), and a cell that imports it would read that one (appendix_geometry_evenodd registers its unit sources there)
    held = {m: sys.modules.get(m) for m in sorted(set(_MODS) | {p.stem for p in NOTEBOOKS.glob("*.py")})}
    for m in held:
        sys.modules.pop(m, None)
    show = plt.show
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ["REGISTRY_NB_SOURCE"] = saved_env["REGISTRY_NB_SOURCE"] or "snapshot"
        os.environ["REGISTRY_NB_FIGS"] = str(figs)
        if policy != "rejudge":
            os.environ["REGISTRY_NB_JUDGES"] = policy
        os.chdir(NOTEBOOKS)
        if str(NOTEBOOKS) not in sys.path:
            sys.path.insert(0, str(NOTEBOOKS))
        plt.show = lambda *a, **k: plt.close("all")
        ns = {"__name__": f"__seed_ci_slow_{policy}__"}
        cells = [c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        with contextlib.redirect_stdout(sys.stderr):
            for c in cells:
                if c.get("id") in SKIP_CELLS:
                    continue
                exec(compile("".join(c["source"]), f"<cell {c.get('id')}>", "exec"), ns)
                plt.close("all")
            ns["REPORT"] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in ns["R"].POOLED_CI_REPORT.items()}
        ns["MARKS"] = seed_marks(ns)
        return ns
    finally:
        plt.show = show
        os.chdir(saved_cwd)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.path[:] = saved_path                  # the notebook directory, and the SISTEER_SRC entry registry_nb inserts at import
        for m, mod in held.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod


def _key(*vals) -> str:
    return "|".join("" if (v is None or (isinstance(v, float) and v != v)) else str(v) for v in vals)


def seed_marks(ns: dict) -> dict:
    """{section: {key: (mean, lo, hi)}} of every mark the render draws from a seed pool, keyed as data/seed_ci_baseline.json."""
    R = ns["R"]
    out = {}
    mm = {}
    for tag, name in (("main", "cm"), ("all", "cm_all"), ("neg", "neg_cm"), ("ctl_st", "ctl_st_cm"), ("ctl_cap", "ctl_cap_cm"), ("ctl", "ctl_cm"),
                      ("ctl_pub", "ctl_pub_cm"), ("ctl_new", "ctl_new_cm"), ("ctl_m", "ctl_mcm")):
        if name in ns and len(ns[name]):
            for r in R.pool_seeds(ns[name]).itertuples():
                mm[_key(tag, r.model_id, r.category, r.object_label)] = (r.y, r.lo, r.hi)
    out["mean_marks"] = mm
    d_all = ns["d_all"]
    pc = R.pool_cell(d_all[d_all.d_y.notna()], y="d_y", lo="d_lo", hi="d_hi")
    out["eval_marks"] = {_key(r.model_id, r.object_label, r.eval_id): (r.d_y, r.d_lo, r.d_hi) for r in pc.itertuples()}
    mc, am = {}, {}
    for fig, evals in (("attacks", ATTACKS), ("capabilities", OR_EVALS + list(R.APPENDIX_CAPABILITY_EVALS))):
        pts = R.delta_points(d_all, evals, R.APPENDIX_OBJECTS)
        for cat, cols in R.split_by_category(evals):
            panel = pts[pts.eval_id.isin(cols)]
            for r in R.eval_mean_column(panel[~R.held_mask(panel)], cols).itertuples():
                mc[_key("delta", fig, cat, r.model_id, r.object_label)] = (r.y, r.lo, r.hi)
    for name, evals in (("pts_ab", ATTACKS), ("pts_acd", OR_EVALS + list(R.APPENDIX_CAPABILITY_EVALS))):
        src = ns["pts_ab"] if name == "pts_ab" else R.concat_plain(ns["pts_ac"], ns["pts_ad"])
        pts = R.pool_cell(src)
        for r in pts.itertuples():
            am[_key(name, r.model_id, r.object_label, r.eval_id)] = (r.y, r.lo, r.hi)
        for cat, cols in R.split_by_category(evals):
            panel = pts[pts.eval_id.isin(cols)]
            for r in R.eval_mean_column(panel[~R.held_mask(panel)], cols).itertuples():
                mc[_key("absolute", name, cat, r.model_id, r.object_label)] = (r.y, r.lo, r.hi)
    out["abs_marks"], out["mean_columns"] = am, mc
    tr_pe, tr_cat, _ = R.transfer_pairs(ns["tr_d"], R.TRANSFER_CATEGORIES, objects=R.TRANSFER_OBJECTS, models=R.TRANSFER_MODELS)
    out["transfer_eval"] = {_key(r.model_id, r.object_label, r.eval_id): (r.diff, r.lo, r.hi) for r in tr_pe.itertuples()}
    out["transfer_cat"] = {_key(r.model_id, r.object_label, r.category): (r.y, r.lo, r.hi) for r in tr_cat.itertuples()}
    pm = {}
    for o, g in ns["pad_pts"].groupby("object_label"):
        if len([s for s in set(g.seed.astype(str)) if s]) > 1:
            for r in R.padding_pooled(g).itertuples():
                pm[_key(o, r.pad)] = (r.delta, r.lo, r.hi)
    out["padding"] = pm
    AC, AG = sys.modules["appendix_common"], sys.modules["appendix_generator"]
    g = ns["gen_rows"]
    p = AC.paired_means(g[[c for c in list(AC.TIDY_COLUMNS) + list(AC.OPTIONAL_COLUMNS) if c in g.columns]], AG.CATEGORIES, "complete")
    out["generator"] = {_key(r.object, r.condition, r.category): (r.y, r.lo, r.hi) for r in p.itertuples()}
    return out


@pytest.fixture(scope="module")
def nb(tmp_path_factory, notebook_rcparams):
    return _run("rejudge", tmp_path_factory.mktemp("figs_rejudge"))


@pytest.fixture(scope="module")
def nb_primary(tmp_path_factory, notebook_rcparams):
    return _run("primary", tmp_path_factory.mktemp("figs_primary"))


def _baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))["marks"]


def test_every_seed_pool_marks_mean_equals_the_base_renders_exactly(nb):
    base, got = _baseline(), nb["MARKS"]
    assert set(base) == set(got)
    for sec, want in base.items():
        assert set(got[sec]) == set(want), (sec, sorted(set(want) ^ set(got[sec]))[:5])
        for k, w in want.items():
            g = got[sec][k][0]
            assert (w[0] is None and pd.isna(g)) or float(g) == w[0], (sec, k, g, w[0])


def test_no_mark_with_an_interval_at_the_base_loses_it(nb):
    base, got = _baseline(), nb["MARKS"]
    lost = [(sec, k) for sec, want in base.items() for k, w in want.items()
            if w[1] is not None and w[2] is not None and not (np.isfinite(float(got[sec][k][1])) and np.isfinite(float(got[sec][k][2])))]
    assert lost == []
    gained = [(sec, k) for sec, want in base.items() for k, w in want.items() if w[1] is None and np.isfinite(float(got[sec][k][1]))]
    assert any(k.startswith("qwen35_4b_it|difference_relu|petri") for _s, k in gained), gained[:10]


def test_every_seed_pool_is_item_clustered_and_the_anchor_bounds_are_as_before(nb):
    R, rep = nb["R"], nb["REPORT"]
    assert rep["opaque_seed_pools"] == 0 and rep["seed_pools"] > 300
    assert rep["seed_overlap_min"] == 1.0 and rep["alignment_min"] == 1.0    # every pool's fits hold the same item ids (subsets included)
    seed_bounds = {(v["what"], v["model_id"], v.get("object_label"), v["eval_id"]) for v in rep["bound"].values()
                   if v.get("what") in R.SEED_BOUND_WHATS}
    assert seed_bounds == KEPT_TRANSFER
    assert {(v["model_id"], v["eval_id"]) for v in rep["bound"].values() if v.get("what") not in R.SEED_BOUND_WHATS} == BOUND_ANCHORS
    tr_pe, tr_cat, _ = R.transfer_pairs(nb["tr_d"], R.TRANSFER_CATEGORIES, objects=R.TRANSFER_OBJECTS, models=R.TRANSFER_MODELS)
    kept = {("transfer", r.model_id, r.object_label, r.eval_id) for r in tr_pe.itertuples() if r.se_method == R.TRANSFER_KEPT}
    assert kept == KEPT_TRANSFER and set(tr_pe.se_method) == {R.TRANSFER_NETTED, R.TRANSFER_KEPT}
    assert set(tr_cat.se_method) == {R.TRANSFER_NETTED}                    # Petri's category reads the pooled score
    assert tr_pe.se.notna().all() and len(tr_pe) == 185


def test_the_sign_decomposition_pools_on_the_renders_registry_nb(nb):
    """The sign-decomposition cell's module is this render's fresh import, bound to the render's `registry_nb`: its unit sources are
    registered there, so its seed pools read the sidecar's payloads (item-clustered, counted in the report above) and none takes the
    identical-draws bound."""
    R, GE, rep = nb["R"], nb["GE"], nb["REPORT"]
    assert GE.R is R and set(GE.UNIT_SOURCE.values()) <= set(R.UNIT_SOURCES)
    assert not [v for v in rep["bound"].values() if "sign-decomposition" in str(v.get("what"))]
    cp = nb["_ge_cp"]                                                        # the cell's own category parts
    how = {(o, h) for o, g in cp.groupby("object") for q in GE.PARTS for h in g[f"{q}_how"]}
    assert all(h.startswith(R.SEED_ONE_FIT if o == "arditi_allpos" else R.SEED_CLUSTERED) for o, h in how), sorted(how)   # one fit: the Refusal steer


def _mark(nb, sec, key):
    y, lo, hi = nb["MARKS"][sec][key]
    return float(y), float(lo), float(hi)


#: (section, mark) -> (lo, hi): the render this change was measured on (the PR body quotes them)
PINS = {
    ("mean_marks", "main|qwen35_4b_it|Petri|difference_relu"): (2.6427, 4.2295),          # se 0.334 -> 0.405: all three seeds now enter
    ("mean_marks", "main|qwen35_4b_it|capability|control_relu"): (-13.9397, -10.1574),
    ("eval_marks", "qwen35_4b_it|difference_relu|mmlu"): (-5.9882, 0.1913),               # stops excluding 0
    ("eval_marks", "qwen35_4b_it|control_lora|codeattack"): (-0.0767, 17.5027),            # stops excluding 0
    ("eval_marks", "qwen35_4b_it|difference_relu|petri_pooled"): (-4.2295, -2.6427),       # no interval before
    ("padding", "control_lora|0.0"): (-75.4070, -46.1045),                                 # the between-fit term, new to the padding means
    ("transfer_eval", "llama8b_it|control_lora|grok420"): (1.4549, 12.2517),               # the anchor netted (it was counted twice)
    ("transfer_eval", "qwen35_4b_it|constitution_lora|pasttense"): (-5.4164, 2.3019),      # lanes on partly different anchor runs
    ("mean_columns", "delta|attacks|multi-turn|qwen35_4b_it|control_lora"): (0.2738, 6.9076),   # = the category's mean mark
    ("mean_columns", "delta|capabilities|over-refusal|qwen35_4b_it|constitution_lora"): (-14.9021, 0.1738),
    ("transfer_cat", "qwen35_4b_it|difference_lora|over-refusal"): (-0.0748, 19.3611),
    ("abs_marks", "pts_acd|qwen35_4b_it|control_relu|mmlu"): (66.7277, 71.3520),
    ("generator", "constitution_lora_1m|qwen_written|capability"): (-3.7385, 1.0474),
}
#: the transfer cells whose instruct-direct row (a Petri score at its floor on every scenario, no interval of its own) gives no steered
#: part from its units: they keep the old composition, marked (`transfer_se`)
KEPT_TRANSFER = {("transfer", "qwen35_4b_it", "control_lora", "petri_self_preservation"),
                 ("transfer", "qwen35_4b_it", "difference_lora", "petri_self_preservation")}


@pytest.mark.parametrize("sec,key,want", [(s, k, v) for (s, k), v in PINS.items()])
def test_pinned_intervals(nb, sec, key, want):
    y, lo, hi = _mark(nb, sec, key)
    assert (lo, hi) == (pytest.approx(want[0], abs=TOL), pytest.approx(want[1], abs=TOL)), (sec, key, lo, hi)


def test_the_filed_judges_render_completes_and_names_its_seed_pool_bounds(nb_primary):
    R, rep = nb_primary["R"], nb_primary["REPORT"]
    assert rep["opaque_seed_pools"] == 0
    seed_bounds = {(v["what"], v["model_id"], v.get("object_label"), v["eval_id"]) for v in rep["bound"].values()
                   if v.get("what") in R.SEED_BOUND_WHATS}
    assert {(w, m, e) for w, m, _o, e in seed_bounds if w != "transfer"} == PRIMARY_SEED_BOUNDS, sorted(seed_bounds)
    assert {b for b in seed_bounds if b[0] == "transfer"} == PRIMARY_KEPT_TRANSFER, sorted(seed_bounds)
    base, got = _baseline(), nb_primary["MARKS"]
    assert set(got) == set(base) and all(len(got[s]) for s in got)


#: the seed pools of the filed-judges render whose fits' runs have no item rows under the filed judges in the frame (what, model, eval)
#: (Qwen's Difference ReLU on the static attacks and Best-of-75: its runs' filed readings have no item rows under the filed judges)
PRIMARY_SEED_BOUNDS = ({("seed pool", "qwen35_4b_it", e) for e in ("codeattack", "grok420", "pasttense", "renellm", "strongreject_bon")}
                       | {("absolute seed pool", "qwen35_4b_it", e) for e in ("codeattack", "grok420", "pasttense", "renellm")})
PRIMARY_KEPT_TRANSFER = {("transfer", "qwen35_4b_it", "difference_lora", "petri_self_preservation")}
