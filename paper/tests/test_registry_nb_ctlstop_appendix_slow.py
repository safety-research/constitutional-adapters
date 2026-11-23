"""The control-pool stopping appendix figures: the SLOW half, on the notebook's full load chain (SISTEER_SLOW_TESTS=1).

The committed notebook's header and load cells and Figure 3's anchoring run against the source the environment names (the committed
paper-frame snapshot by default: about two minutes warm and about 4 GB of reassembled SQLite under paper/notebooks/_cache/, as
tests/test_demo_objects.py's slow check), then the four lane code cells run exactly as the generator writes them. The lane cells draw
every judged point from the appendix's own judge, whatever the render's policy: the main figures' runs they draw are read from the
committed file under that judge (`ctl_op`, `ctl_md`), so this module runs under any policy. Pinned here:
  * the main figures' runs under the appendix's judge are the runs the main figures draw (the run set, on the real frames, except
    what the file names as without a verdict), every judged row is that judge's, and the rule-scored evals' rows are `d`'s exactly;
  * each mirror's five unchanged objects ARE those rows: they equal `ctl_md` and its category means exactly, the capability mean
    being the two-eval mean of the same rows;
  * every refit row carries its class's anchor in those rows (the anchor RULE, re-checked on the real frames), except a static row
    read above every unsteered run's rung, which carries no delta and no anchor and is named unless the file declares it; a declared
    row carries the anchor of its declared cap's class, and the four declared Pliny cells and the two control objects' static means
    are drawn at their pinned values on the committed snapshot (at thresholds 0.5 and 0.75);
  * the main figures' per-row formula, d = y - A.y with se = hypot(own half-width, A.se), reproduces every multi-turn row of `d`
    exactly when each row's anchor is rebuilt from its own `anchor_run` list (the independent check of the anchoring the rows reuse);
  * the comparison's published control ReLU steer, control LoRA and difference LoRA are the main figures' runs under the appendix's
    judge, its published difference ReLU steer on the multi-turn attacks the six reruns;
  * the lane cells change no frame the figures above read (`op`, `d`, `cm`, `paper_all`, `frontier`, `POOL_Q`);
  * on the real frames, the comparison grid draws one mark per reading it differences and the gap glyph for each left without a delta;
  * on the real frames, each of the four figures the lane cells save is whole: the saved picture's outermost rows and columns are
    blank, so the canvas's edge cuts nothing (an axis label taller than its panel is what it once cut);
  * where the live registry is present, a fresh derivation equals the committed sidecar.
Under the filed judges at the paper's threshold, what the file says of the main-figure runs it does not read is held to the render
itself: a snapshot render draws exactly the runs the file's snapshot list names and leaves out exactly the runs the file names as
unread, and a live render draws none of those runs.
Run it once on the final head and record its pass line."""
import contextlib
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile

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
GENERATOR = NOTEBOOKS / "make_registry_notebook.py"
FRAMES = ("op", "d", "cm", "paper_all", "frontier", "POOL_Q")
LANE_CODE_CELLS = ("a6e0f2b8", "d2a7e5c9", "7fae15d6", "76fdab54")
LANE_FIGURES = ("figA_qwen_ctlstop_static", "figA_qwen_ctlstop_capabilities", "figA_qwen_ctlstop_multi_turn",
                "figA_qwen_ctlstop_rule_comparison")
EDGE_ROWS = 3                       # the outermost rows and columns of a saved picture that hold its blank margin only
LIVE_DB = pathlib.Path(os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")) / "registry.sqlite"


def generator_cell(cid: str) -> str:
    """A cell's source exactly as make_registry_notebook.py writes it (the generator strips the newlines around the raw string)."""
    src = GENERATOR.read_text(encoding="utf-8")
    m = re.search(r'^(?:md|code)\("%s", r("""|\'\'\')(.*?)\1\)' % cid, src, flags=re.M | re.S)
    assert m, cid
    return m.group(2).strip("\n")


def _fresh_registry_nb(mode: str):
    """Import registry_nb afresh under REGISTRY_NB_SOURCE=`mode` (the module resolves its source at import) and the filed judges
    (REGISTRY_NB_JUDGES=primary: these checks read the main figures' run set under the filed judges, as the sidecar's generator does;
    the notebook's default is now the paper's re-judges) with a figures directory of its own, returning the module the notebook cells
    will see; the caller restores sys.modules and the three variables (`_restore_env`)."""
    os.environ["REGISTRY_NB_SOURCE"] = mode
    os.environ["REGISTRY_NB_JUDGES"] = "primary"
    # a render under the filed judges may not write the paper's figures directory (registry_nb.announce_render_policy): a directory of
    # its own, as reproduce/10_figures.sh --judges primary gives it
    os.environ["REGISTRY_NB_FIGS"] = tempfile.mkdtemp(prefix="ctlstop_slow_figs_")
    sys.modules.pop("registry_nb", None)
    if str(NOTEBOOKS) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS))
    import registry_nb
    return registry_nb


def _restore_env(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _restore_modules(saved: dict) -> None:
    """Put back each module a fresh import replaced (or drop the fresh one when there was none), so no test module later reads this
    module's copy through sys.modules (`ctlstop_sidecar._R()`, the conftest): tests/test_nb_snapshot_fallback.py has the same restore."""
    for m, mod in saved.items():
        if mod is None:
            sys.modules.pop(m, None)
        else:
            sys.modules[m] = mod


@pytest.fixture(scope="module")
def nb(tmp_path_factory, notebook_rcparams):
    """The notebook namespace after the header and load cells and Figure 3's anchoring."""
    import matplotlib
    matplotlib.use("Agg")
    saved_mod, saved_env, saved_cwd = sys.modules.get("registry_nb"), os.environ.get("REGISTRY_NB_SOURCE"), os.getcwd()
    saved_vars = {k: os.environ.get(k) for k in ("REGISTRY_NB_SOURCE", "REGISTRY_NB_JUDGES", "REGISTRY_NB_FIGS")}
    saved_path = list(sys.path)
    try:
        os.chdir(NOTEBOOKS)
        R = _fresh_registry_nb(saved_env or "snapshot")
        R.FIGS = str(tmp_path_factory.mktemp("figs"))
        cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"}
        ns = {"__name__": "__ctlstop_slow__"}
        with contextlib.redirect_stdout(sys.stderr):
            exec(cells["583e3f04"], ns)
            ns["R"] = R
            exec(cells["ec6aacd8"], ns)
            exec("d = R.anchored(op, pool=POOL_Q)\nCAT_Q = R.categories_for(R.MODEL_ID)\ncm = R.category_means(d, categories=CAT_Q)", ns)
            exec(next(ln for ln in generator_cell("ac902304").splitlines() if ln.startswith("DCOLS = ")), ns)
        yield ns
    finally:
        os.chdir(saved_cwd)
        _restore_env(saved_vars)
        sys.path[:] = saved_path
        _restore_modules({"registry_nb": saved_mod})


def _digest(f: pd.DataFrame) -> str:
    """A frame's columns, dtypes, attrs keys and every cell (as text), hashed. The text is taken from a copy without attrs: `anchored`
    hangs a frame on attrs, and pandas compares attrs while it builds the copy, which two frames cannot answer."""
    h = hashlib.sha256()
    h.update(repr(list(f.columns)).encode()); h.update(repr(list(f.dtypes.astype(str))).encode()); h.update(repr(sorted(f.attrs)).encode())
    g = f.copy()
    g.attrs = {}
    h.update(pd.util.hash_pandas_object(g.astype(str), index=True).values.tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def ran(nb):
    """The namespace after the four lane code cells, with each frame's digest taken before and after them."""
    before = {k: _digest(nb[k]) for k in FRAMES}
    nb["_figs"] = nb["R"].FIGS                # where the lane cells save their figures (a later test points the module elsewhere)
    with contextlib.redirect_stdout(sys.stderr):
        for cid in LANE_CODE_CELLS:
            exec(generator_cell(cid), nb)
    nb["_digests"] = (before, {k: _digest(nb[k]) for k in FRAMES})
    return nb


def test_the_main_figures_runs_under_the_appendixs_judge_on_the_real_frames(ran):
    R, d, md, chk, rep = ran["R"], ran["d"], ran["ctl_md"], ran["_ctl_chk"], ran["ctl_main"]
    judged = [e for e, sp in R.CTLSTOP_EVALS.items() if sp["family"] != "rule"]
    rule = [e for e in R.CTLSTOP_EVALS if e not in judged]
    assert rep["read"] > 300 and rep["not_derivable"] == {} and chk["rows"] > 200
    if R.SR_THRESHOLD == R.SR_PAPER_THRESHOLD:
        assert chk["run_set_checked"]                                    # the cell raised otherwise; what is left out is named
    key = ["object_label", "seed", "draw", "eval_id", "run_id"]
    drawn = lambda f: {tuple(map(str, k)) for k in f[f.eval_id.isin(judged) & f.object_label.isin(R.OBJECTS) & f.d_y.notna()][key].itertuples(index=False, name=None)}
    assert drawn(md) <= drawn(d) and len(drawn(d) - drawn(md)) == len(chk["left_out"])
    jm = md[md.eval_id.isin(judged)]
    assert set(jm.judge_model) == {"minimax-m3"} and set(jm.judge_source) == {R.CTLSTOP_MAIN_SOURCE}
    assert set(jm.judge_id) <= set(R.CTLSTOP_JUDGES.values())
    a = md[md.eval_id.isin(rule)].sort_values(key).reset_index(drop=True)
    b = d[d.eval_id.isin(rule)].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) > 0
    for c in ("d_y", "d_lo", "d_hi", "anchor_value", "y", "lo", "hi", "anchor_run"):
        assert (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all(), c
    pend = set(ran["_ctl_file"].get("main_pending") or {})
    assert set(rep["no_judgment"]) <= pend and not set(md.run_id) & pend
    if not R.RENDER_POLICY:                 # the filed judges at the paper's threshold, as the file's lists of these runs were read
        if R.SOURCE == "snapshot":          # the snapshot's list is this render's, and it leaves out exactly the runs the file names as unread
            assert set(R.ctlstop_main_needs(d)) == set(ran["_ctl_file"]["main_needs"]["snapshot"]["runs"].split(" "))
            assert set(rep["no_judgment"]) == pend
        else:                               # a live render draws no run the file names as unread
            assert rep["no_judgment"] == {}


@pytest.mark.parametrize("mirror", [("ctl_st_d", "ctl_st_cm", ("static",)), ("ctl_cap_d", "ctl_cap_cm", ("over-refusal", "capability")),
                                    ("ctl_d", "ctl_cm", ("multi-query", "multi-turn"))])
def test_each_mirror_keeps_the_main_figures_five_objects_exactly(ran, mirror):
    R, d, cm = ran["R"], ran["ctl_md"], ran["ctl_mcm"]
    fd, fcm, cats = ran[mirror[0]], ran[mirror[1]], mirror[2]
    keep = [o for o in R.OBJECTS if o not in R.CTLSTOP_SLOTS]
    evals = [e for c in cats for e in ran["CTL_CATS"][c][0]]
    key = ["object_label", "seed", "draw", "eval_id", "run_id"]
    a = fd[fd.object_label.isin(keep) & fd.eval_id.isin(evals)].sort_values(key).reset_index(drop=True)
    b = d[d.object_label.isin(keep) & d.eval_id.isin(evals)].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) > 0
    for c in ("d_y", "d_lo", "d_hi", "anchor_value", "y", "lo", "hi"):
        assert (a[c].to_numpy() == b[c].to_numpy()).all(), c
    for cat in cats:
        ca = fcm[fcm.object_label.isin(keep) & fcm.category.eq(cat)].sort_values(["object_label", "seed"]).reset_index(drop=True)
        ref = cm if cat != "capability" else R.category_means(d[d.object_label.isin(keep)], categories={cat: ran["CTL_CATS"][cat]})
        cb = ref[ref.object_label.isin(keep) & ref.category.eq(cat)].sort_values(["object_label", "seed"]).reset_index(drop=True)
        assert len(ca) == len(cb) > 0 and all((ca[c].to_numpy() == cb[c].to_numpy()).all() for c in ("y", "lo", "hi", "anchor_abs")), cat


def test_every_refit_row_carries_its_class_anchor_on_the_real_frames(ran):
    R, d = ran["R"], ran["ctl_md"]
    new = R.concat_plain(ran["ctl_st_new"], ran["ctl_cap_new"], ran["ctl_new_d"])
    assert len(new) == len(ran["ctl_rows"][ran["ctl_rows"].rule.eq("control_pool") & ran["ctl_rows"].role.eq("drawn")])
    drawn = new[new.d_y.notna()]
    assert R.ctlstop_anchor_problems(new, d) == [] and drawn.anchor_in_frame.astype(bool).all()
    # the only rows left without a delta: a static reading above every unsteered run's rung that the file does not declare, named,
    # with no anchor on it; the file declares every such row of the committed data, so none is left
    left = R.ctlstop_rung_gap_rows(new)
    assert len(new) - len(drawn) == len(left) == len(R.ctlstop_rung_gap_notes(new))
    assert all(R.CTLSTOP_EVALS[r.eval_id]["drawn_cap"] == "run" and pd.isna(r.anchor_run) for r in left)
    declared = set(ran["_ctl_file"].get("anchor_declared") or {})
    assert {r.run_id for r in R.ctlstop_declared_rows(new)} == declared and not {r.run_id for r in left} & declared


#: THE FOUR DECLARED CONTROL-POOL PLINY CELLS ON THE COMMITTED SNAPSHOT, as the comparison draws them: each 16,384 reading's delta
#: against the static panel's unsteered runs at 8,192 (DECLARED_ANCHOR_RUNS; defense 83.14 at 0.5, 91.44 at 0.75) and its 95%
#: interval, oriented as the figure (+ is more defense), in pp; then each control object's static mean over seeds (the comparison's
#: mean column). The pins follow the committed snapshot frame, whose 8,192 unsteered pool is the live registry's six runs (a render
#: of the live registry gave the same values before the intervals became item-clustered; the intervals were re-pinned on the
#: snapshot on 2026-11-19, the deltas and the anchor unchanged; the two means' intervals again on 2026-11-21, when the seed pools
#: became item-clustered across the seeds, their means and the four cells unchanged).
DECLARED_ANCHOR_RUNS = frozenset({
    "full22:grok420:f71ed06cf77f:73b894fe5cb883a6", "full22:grok420:f71ed06cf77f:d7545f4c6dc9eaa3",
    "full22:grok420:f71ed06cf77f:dc6b08b40d594f87", "full22:grok420:f71ed06cf77f:f162022abe103d60",
    "sdfbase_full22:grok420:f71ed06cf77f:e5387db271547566", "sdfbase_full22:grok420:f71ed06cf77f:eb8a8625df1398a6",
})
DECLARED_PINS = {
    0.5: dict(cells={("control_lora", "s1"): (-50.71, -57.14, -44.27), ("control_relu", "s1"): (-55.34, -61.57, -49.12),
                     ("control_lora", "s2"): (-71.94, -76.86, -67.03), ("control_relu", "s3"): (-58.43, -64.48, -52.38)},
              anchor=83.14, means={"control_relu": (-21.03, -24.10, -17.97), "control_lora": (-16.33, -20.72, -11.93)}),
    0.75: dict(cells={("control_lora", "s1"): (-39.70, -46.07, -33.33), ("control_relu", "s1"): (-43.18, -49.55, -36.81),
                      ("control_lora", "s2"): (-62.10, -67.97, -56.23), ("control_relu", "s3"): (-39.70, -46.07, -33.33)},
               anchor=91.44, means={"control_relu": (-12.79, -14.77, -10.80), "control_lora": (-14.25, -17.17, -11.33)}),
}


def test_the_declared_pliny_cells_are_drawn_at_the_pinned_values_on_the_committed_snapshot(ran):
    """The four control-pool Pliny cells the file declares, on the committed snapshot's real frames: each is read at 16,384 and
    differenced against the six unsteered runs at 8,192 the static panel's fitted 8,192 rows take, at the pinned delta and interval;
    the two control objects' static means are whole (four evals per seed) and drawn at the pinned values."""
    R = ran["R"]
    if R.SOURCE != "snapshot" or R.SR_THRESHOLD not in DECLARED_PINS:
        pytest.skip(f"pinned on the committed snapshot at thresholds {sorted(DECLARED_PINS)}")
    pins = DECLARED_PINS[R.SR_THRESHOLD]
    sign = ran["CTL_CATS"]["static"][1]
    new = ran["ctl_st_new"]
    dec = {(r.object_label, str(r.seed)): r for r in R.ctlstop_declared_rows(new)}
    assert set(dec) == set(pins["cells"])
    fit8 = ran["ctl_md"]
    fit8 = fit8[fit8.eval_id.eq("grok420") & fit8.token_cap_tokens.eq(8192.0) & fit8.object_label.isin(R.CTLSTOP_BOUND_SLOTS) & fit8.d_y.notna()]
    for cell, (dy, lo, hi) in pins["cells"].items():
        r = dec[cell]
        assert (r.eval_id, r.token_cap_tokens, r.anchor_cap, bool(r.cap_mismatch)) == ("grok420", 16384.0, 8192.0, False)
        anchors = str(r.anchor_run).split(",")
        assert len(anchors) == len(set(anchors)) == 6 and set(anchors) == DECLARED_ANCHOR_RUNS, (cell, anchors)
        assert {str(r.anchor_run)} == set(fit8.anchor_run.astype(str))
        assert round(100.0 - r.anchor_value, 2) == pins["anchor"]
        got = (sign * r.d_y, min(sign * r.d_lo, sign * r.d_hi), max(sign * r.d_lo, sign * r.d_hi))
        assert tuple(round(x, 2) for x in got) == (dy, lo, hi), (cell, got)
    pairs = R.ctlstop_mean_pairs(ran["ctl_pub_cm"], ran["ctl_new_cm"], {"static": ran["CTL_CATS"]["static"]})["static"]
    for slot, (y, lo, hi) in pins["means"].items():
        m = pairs[slot]["control_pool"]
        assert m is not None and not pairs[slot]["why"].get("control_pool"), (slot, pairs[slot]["why"])
        assert (round(float(m.y), 2), round(float(m.lo), 2), round(float(m.hi), 2)) == (y, lo, hi), slot
    cm = ran["ctl_st_cm"]
    assert not cm[cm.object_label.isin(["control_relu", "control_lora"])].partial.any()


def test_the_main_figures_formula_reproduces_every_multi_turn_row(nb):
    """Each row's anchor rebuilt from its OWN `anchor_run` list (the unsteered rows of the frame and of the borrowable pool, by run id, in
    the listed order), then the main figures' rule: A.y the mean; A.se one run's own half-width / 1.96, several runs' item-clustered se
    (`pooled_ci.pooled` over the runs' units, Wilson at the cluster-effective n: the ASR runs carry Wilson intervals), or, where a run's
    reading has no item rows in the source, the identical-draws bound (the runs' own se averaged); d = y - A.y,
    se = hypot((hi - lo) / 3.92, A.se). Asserted to 1e-9."""
    R, d, op, POOL_Q = nb["R"], nb["d"], nb["op"], nb["POOL_Q"]
    none = pd.concat([op[op.object_label.eq("none") & op.headline & op.judge_ok], POOL_Q[POOL_Q.object_label.eq("none") & POOL_Q.headline & POOL_Q.judge_ok]],
                     ignore_index=True)
    none = none[none.metric_id.eq("asr") & pd.to_numeric(none.threshold, errors="coerce").eq(R.SR_THRESHOLD)].drop_duplicates(["run_id", "metric_id", "judgment_id"])
    mt = d[d.eval_id.isin(R.CTLSTOP_MULTI_TURN) & d.d_y.notna()]
    assert set(mt.eval_id) == set(R.CTLSTOP_MULTI_TURN)
    assert mt.route.astype(str).str.match(r"common anchor \(\d+ runs?\)$").all(), mt.route.value_counts().to_dict()
    for r in mt.itertuples():
        ids = str(r.anchor_run).split(",")
        rows = none[none.eval_id.eq(r.eval_id)].set_index("run_id").loc[ids]
        y = rows.value.astype(float) * 100.0
        own = rows.ci_lo.notna() & rows.ci_hi.notna()
        w = [R.wilson(v * n, n) for v, n in zip(rows.value.astype(float), rows.n.astype(float))]
        lo = pd.Series(np.where(own, rows.ci_lo.astype(float), [a for a, _ in w]), index=rows.index) * 100.0
        hi = pd.Series(np.where(own, rows.ci_hi.astype(float), [b for _, b in w]), index=rows.index) * 100.0
        k = len(rows)
        own_se = ((hi - lo) / 3.92).to_numpy()
        P = R._pooled_ci()
        if k == 1:
            se = float(own_se[0])
        else:
            iv = R.intervals(rows.reset_index())
            try:
                units = [R.row_units(x) for _, x in iv.iterrows()]
                se = P.pooled(units, form="wilson", scale=100.0).se
            except P.UnitsUnavailable:
                se = float(own_se.mean())
        ay = float(y.mean())
        dy = r.y - ay
        s = np.hypot((r.hi - r.lo) / 3.92, se)
        got, want = (r.anchor_value, r.d_y, r.d_lo, r.d_hi), (ay, dy, dy - 1.96 * s, dy + 1.96 * s)
        assert got == pytest.approx(want, abs=1e-9), (r.eval_id, r.object_label, r.seed)


def test_the_comparisons_published_rule_is_the_main_figures_rows_and_the_reruns(ran):
    R, d, pub = ran["R"], ran["ctl_md"], ran["ctl_pub_d"]
    slots = ["control_relu", "control_lora", "difference_lora"]
    evals = [e for c in ran["CTL_CATS"].values() for e in c[0]]
    key = ["object_label", "seed", "eval_id"]
    a = pub[pub.object_label.isin(slots)].sort_values(key).reset_index(drop=True)
    b = d[d.object_label.isin(slots) & d.eval_id.isin(evals)].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) > 0
    for c in ("run_id", "d_y", "d_lo", "d_hi", "anchor_value"):
        assert (a[c].to_numpy() == b[c].to_numpy()).all(), c
    v = pub[pub.object_label.eq("difference_relu") & pub.eval_id.isin(R.CTLSTOP_MULTI_TURN)]
    reruns = {R.CTLSTOP_RUNS[("published", "difference_relu", sd, e)] for sd in R.CTLSTOP_SEEDS for e in R.CTLSTOP_MULTI_TURN}
    assert set(v.run_id) <= reruns and not set(v.run_id) & set(d.run_id)         # never a main-figure row on the multi-turn attacks


def test_the_lane_cells_change_no_frame(ran):
    before, after = ran["_digests"]
    assert before == after, [k for k in FRAMES if before[k] != after[k]]


def test_comparison_marks_on_the_real_frames(ran, tmp_path):
    import matplotlib.pyplot as plt
    R = ran["R"]
    R.FIGS = str(tmp_path)
    fig = R.rule_comparison_figure(ran["ctl_pub_d"], ran["ctl_all_new"], ran["ctl_pub_cm"], ran["ctl_new_cm"], "t_real",
                                   categories=ran["CTL_CATS"], layout=R.CTLSTOP_GRID, pending=ran["ctl_pending"])
    gids = [c.get_gid() for ax in fig.axes for c in ax.collections if c.get_gid()]
    assert sum(1 for g in gids if g.startswith("ctlstop|control_pool|") and "|mean|" not in g) == int(ran["ctl_all_new"].d_y.notna().sum())
    assert sum(1 for g in gids if g.startswith("ctlstop-gap|control_pool|")) == len(R.ctlstop_rung_gap_rows(ran["ctl_all_new"]))
    assert sum(1 for g in gids if g.startswith("ctlstop|published|") and "|mean|" not in g) == len(R.pool_draws(ran["ctl_pub_d"]))
    assert sum(1 for g in gids if g.startswith("ctlstop-pending|control_pool|") and "|mean|" not in g) == sum(
        1 for p in ran["ctl_pending"] if p["rule"] == "control_pool")
    plt.close(fig)


@pytest.mark.parametrize("name", LANE_FIGURES)
def test_each_saved_figure_is_whole_on_the_real_frames(ran, name):
    """The picture a lane cell saved from the real frames, at the notebook's own text sizes: `save` leaves a blank margin around
    everything the canvas was fitted to, so ink on the outermost rows or columns is something the canvas's edge cut."""
    from PIL import Image
    png = pathlib.Path(ran["_figs"]) / f"{name}.png"
    assert png.is_file(), name
    ink = (np.asarray(Image.open(png).convert("RGB")) < 255).any(axis=2)
    assert dict(bottom=int(ink[-EDGE_ROWS:].sum()), top=int(ink[:EDGE_ROWS].sum()), left=int(ink[:, :EDGE_ROWS].sum()),
                right=int(ink[:, -EDGE_ROWS:].sum())) == dict(bottom=0, top=0, left=0, right=0)


@pytest.mark.skipif(not LIVE_DB.exists() or os.environ.get("REGISTRY_NB_SOURCE") == "snapshot",
                    reason="needs the live registry (and REGISTRY_NB_SOURCE not forced to snapshot)")
def test_live_derivation_equals_the_committed_sidecar():
    """The live check, on the real registry: a fresh derivation (the generator's `derive`, which a live render runs) equals the committed
    file and would be written without a refusal."""
    saved_mods = {m: sys.modules.get(m) for m in ("registry_nb", "ctlstop_sidecar")}
    saved_vars = {k: os.environ.get(k) for k in ("REGISTRY_NB_SOURCE", "REGISTRY_NB_JUDGES", "REGISTRY_NB_FIGS")}
    saved_path = list(sys.path)
    try:
        R = _fresh_registry_nb("live")
        assert R.SOURCE == "live"
        sys.modules.pop("ctlstop_sidecar", None)
        import ctlstop_sidecar as G
        committed = json.loads(pathlib.Path(R.CTLSTOP_SIDECAR).read_text(encoding="utf-8"))
        fresh, problems = G.derive(allow_pending=True, main_runs=R.ctlstop_main_runs(committed))
        assert problems == []
        assert len(R.ctlstop_sidecar_check(committed, fresh)) == 0
        frame, pending, _ = R.ctlstop_load("live", allow_pending=True, live=fresh, hold=False)      # the file, whatever the render's policy
        assert len(frame) == len(committed["rows"]) and pending == list(committed.get("pending") or [])
    finally:
        _restore_env(saved_vars)
        sys.path[:] = saved_path
        _restore_modules(saved_mods)
