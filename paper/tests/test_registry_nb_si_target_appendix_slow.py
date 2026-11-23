"""The distillation-target appendix figure: the SLOW half, on the notebook's full load chain (SISTEER_SLOW_TESTS=1).

The generator's header and load cells and Figure 3's anchoring run against the committed paper-frame snapshot (minutes, and a 2 GB
reassembled SQLite under paper/notebooks/_cache/, as tests/test_demo_objects.py's slow check), then the distillation-target code cell runs
exactly as the generator writes it. Pinned here:
  * under each run's filed judges (`primary`, set explicitly: the unset default is the paper's policy, whichever that is) the cell
    prints its one RENDER POLICY line, leaves an earlier render's copies as it found them (the zip cell's stale-file sweep removes
    them), draws nothing and changes no frame, and the paper table's cell prints and writes nothing;
  * under the re-judge judges, where the committed snapshot carries the figure's judge (MiniMax-M3 StrongREJECT v1; skipped, and named,
    while it does not: the snapshot is re-cut with the judge default), the corpus fits' and prompted instructions' rows the figure and
    tables read ARE Figure 4's (equal to `d`, run ids the pins), every distilled row carries Figure 4's anchor, Figure 4's per-row formula
    reproduces every drawn row on all goals and on the goals outside the training pool (the latter from the sidecar's per-goal best
    scores alone), every run passes the one-rubric check, the new cell changes no frame the figures above read, and the figure draws;
    the paper table's cell after it writes the paper's table (tab:app_si_distilled_mt) character for character, and no file either
    cell writes carries a render stamp or a StrongREJECT rubric version; under the item-clustered pooled intervals the full pass raises
    no UnitsMismatch, the lower row pools on its own unit source (SITARGET_OUTSIDE_SOURCE) and leaves Figure 4's anchor registrations
    as they were, and every anchor, prompted and corpus-fit run's payload units are its registry units; every row of the file names its
    payload as its unit source (SITARGET_FILE_SOURCE on all goals), so the distilled objects' fit-seed pools read the file's units, keyed
    by the goal ids, and every seed pool of the figure is item-clustered across its seeds (#543's rule), none on a bound;
  * where the live registry is present, a live derivation (the generator's own, under its pinned judges) equals the committed sidecar
    (rows, payloads, unscored replies, goals and pins) and a live load of it does not refuse (minutes: the frontier query over the pinned
    runs and the figure's judge's item rows).
Every load runs with REGISTRY_NB_STRENGTH_READS cleared, as the paper's render clears it (reproduce/10_figures.sh), so a full load under
the re-judge judges reads the strength lanes' XSTest reads the tree itself carries; a tree that carries none fails the re-judge half by
name rather than skipping it. Run it once on the final head and record its pass line, and again on any change that moves
`paper/snapshot/CURRENT` or rewrites the sidecar: the fast half checks the snapshot's row and judgment of every pinned run, and only
this half reads the item rows a snapshot render checks the payloads and readings against."""
import contextlib
import hashlib
import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

SLOW = os.environ.get("SISTEER_SLOW_TESTS") == "1"
pytestmark = pytest.mark.skipif(not SLOW, reason="the notebook's full load chain on the committed snapshot (minutes, 2 GB cache): "
                                                 "SISTEER_SLOW_TESTS=1 opts in")
USES_ITEM_ROWS = True                     # the conftest's synthetic-frame provider stands down: the real frame's pooled cells read its item rows
REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
GENERATOR = NOTEBOOKS / "make_registry_notebook.py"
NEW_CODE_CELL = "a2d84f6b"
TABLE_CELL = "bb79fa42"                                   # the paper's table of the figure (tab:app_si_distilled_mt), after it
PAPER_TABLE = REPO / "paper" / "tests" / "data" / "si_target_paper_table.tex"
FRAMES = ("op", "d", "cm", "paper_all", "frontier", "POOL_Q")
LIVE_DB = pathlib.Path(os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")) / "registry.sqlite"
JUDGE_ENV = ("REGISTRY_NB_SOURCE", "REGISTRY_NB_JUDGES", "REGISTRY_NB_FIGS", "REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE",
             "REGISTRY_NB_PETRI_EXCLUDE", "REGISTRY_NB_STRENGTH_READS")


def generator_cell(cid: str) -> str:
    """A cell's source exactly as make_registry_notebook.py writes it (the generator strips the newlines around the raw string)."""
    src = GENERATOR.read_text(encoding="utf-8")
    m = re.search(r'^(?:md|code)\("%s", r("""|\'\'\')(.*?)\1\)' % cid, src, flags=re.M | re.S)
    assert m, cid
    return m.group(2).strip("\n")


@contextlib.contextmanager
def _fresh(env: dict):
    """registry_nb imported afresh under `env` (the module resolves its source and judges at import); the module, the environment and
    the working directory are restored after."""
    saved_mod, saved_env, saved_cwd = sys.modules.get("registry_nb"), {k: os.environ.get(k) for k in JUDGE_ENV}, os.getcwd()
    try:
        for k in JUDGE_ENV:
            os.environ.pop(k, None)
        os.environ.update(env)
        os.chdir(NOTEBOOKS)
        sys.modules.pop("registry_nb", None)
        if str(NOTEBOOKS) not in sys.path:
            sys.path.insert(0, str(NOTEBOOKS))
        import registry_nb
        yield registry_nb
    finally:
        os.chdir(saved_cwd)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("registry_nb", None)
        if saved_mod is not None:
            sys.modules["registry_nb"] = saved_mod


def _load(R) -> dict:
    """The notebook namespace after the header and load cells and Figure 3's anchoring."""
    import matplotlib
    matplotlib.use("Agg")
    ns = {"__name__": "__sitarget_slow__"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(generator_cell("583e3f04"), ns)
        ns["R"] = R
        exec(generator_cell("ec6aacd8"), ns)
        exec("d = R.anchored(op, pool=POOL_Q)\nCAT_Q = R.categories_for(R.MODEL_ID)\ncm = R.category_means(d, categories=CAT_Q)", ns)
    return ns


def _digest(f: pd.DataFrame) -> str:
    """A frame's columns, dtypes, attrs keys and every cell (as text), hashed, from a copy without attrs (pandas compares attrs while it
    copies, which two frames under `==` cannot answer)."""
    h = hashlib.sha256()
    h.update(repr(list(f.columns)).encode()); h.update(repr(list(f.dtypes.astype(str))).encode()); h.update(repr(sorted(f.attrs)).encode())
    g = f.copy()
    g.attrs = {}
    h.update(pd.util.hash_pandas_object(g.astype(str), index=True).values.tobytes())
    return h.hexdigest()


def _run_cell(ns) -> dict:
    """The namespace after the new code cell and the paper table's cell after it, with each frame's digest taken before and after them
    and what each printed (`_printed`, `_printed_table`). The header cell rebinds `print` to the notebook's redacting print, which writes
    to sys.stdout, so the capture sees every line."""
    import io
    before = {k: _digest(ns[k]) for k in FRAMES}
    for cid, key in ((NEW_CODE_CELL, "_printed"), (TABLE_CELL, "_printed_table")):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(generator_cell(cid), ns)
        ns[key] = out.getvalue()
    ns["_digests"] = (before, {k: _digest(ns[k]) for k in FRAMES})
    return ns


# ---- under each run's filed judges -------------------------------------------------------------------------------------------------
def test_under_each_run_s_filed_judges_the_cell_prints_one_line_and_draws_nothing(tmp_path):
    """`primary` named explicitly (the unset default is the paper's policy, which draws). An earlier render's figure and table in the
    figures directory are left exactly as they were (name, content, modification stamp): they are the zip cell's stale-file sweep's to
    remove, which judges a file by its stamp."""
    with _fresh({"REGISTRY_NB_SOURCE": "snapshot", "REGISTRY_NB_JUDGES": "primary", "REGISTRY_NB_FIGS": str(tmp_path)}) as R:
        assert R.JUDGES == "primary"
        stale = [tmp_path / f"{R.SITARGET_FIGURE}.png", tmp_path / f"{R.SITARGET_FIGURE}_seeds.tex"]
        for f in stale:
            f.write_text("an earlier render", encoding="utf-8")
            os.utime(f, ns=(1_600_000_000_000_000_000,) * 2)
        kept = {f.name: (f.read_bytes(), f.stat().st_mtime_ns) for f in stale}
        ns = _run_cell(_load(R))
        printed = [ln for ln in ns["_printed"].splitlines() if ln.strip()]
        assert len(printed) == 1 and printed[0].startswith("RENDER POLICY · the distillation-target figure"), printed
        assert printed[0].endswith(": not drawn"), printed
        assert ns["_printed_table"] == "" and not (tmp_path / f"{R.SITARGET_PAPER_TABLE}.tex").exists()
        assert "st_d" not in ns
        assert {f.name: (f.read_bytes(), f.stat().st_mtime_ns) for f in tmp_path.glob(f"{R.SITARGET_FIGURE}*")} == kept
        before, after = ns["_digests"]
        assert before == after


# ---- under the re-judge judges, on the committed snapshot -------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    figs = tmp_path_factory.mktemp("sitarget_figs")
    with _fresh({"REGISTRY_NB_SOURCE": "snapshot", "REGISTRY_NB_JUDGES": "rejudge", "REGISTRY_NB_FIGS": str(figs)}) as R:
        with R.connect() as c:
            held = c.execute("SELECT COUNT(*) FROM judgment WHERE judge_id = ? AND status = 'complete'",
                             (R.SITARGET_EVAL_JUDGE["crescendo"].judge_id,)).fetchone()[0]
        if not held:
            pytest.skip("the committed snapshot carries no judgment of the figure's judge (MiniMax-M3 StrongREJECT v1): it is re-cut with "
                        "the judge default, and this half runs on that snapshot")
        try:
            loaded = _load(R)
        except R.StrengthReadsMissing as err:                                 # the half is due here, so a missing input fails it
            pytest.fail(f"the re-judge half's full load needs the strength lanes' XSTest reads and this tree has none to read "
                        f"({R.STRENGTH_READS_ENV} is cleared here, as the paper's render clears it): {err}")
        ns = _run_cell(loaded)
        ns["_figs"] = figs
        yield ns


def test_prompted_and_corpus_fit_rows_are_figure_4s(ran):
    R, d, st_d = ran["R"], ran["d"], ran["st_d"]
    labels = list(R.SITARGET_PROMPTED_SLOTS + R.SITARGET_REFERENCE_SLOTS)
    key = ["object_label", "seed", "eval_id", "run_id"]
    a = st_d[st_d.object_label.isin(labels)].sort_values(key).reset_index(drop=True)
    b = d[d.object_label.isin(labels) & d.eval_id.isin(R.SITARGET_EVALS) & d.d_y.notna()].sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) == 24
    for c in ("run_id", "d_y", "d_lo", "d_hi", "anchor_value", "anchor_run", "y", "lo", "hi", "judge_id", "judgment_id"):
        assert (a[c].to_numpy() == b[c].to_numpy()).all(), c
    assert set(a.judge_id) == {R.SITARGET_EVAL_JUDGE["crescendo"].judge_id}
    new = st_d[st_d.object_label.isin(R.SITARGET_NEW_SLOTS)]
    for e, g in new.groupby("eval_id"):
        assert set(g.anchor_run) == set(b[b.eval_id.eq(e)].anchor_run) and g.anchor_in_frame.astype(bool).all()
    assert len(new) == 36


def _none_rows(nb) -> pd.DataFrame:
    op, POOL_Q = nb["op"], nb["POOL_Q"]
    none = pd.concat([op[op.object_label.eq("none") & op.headline & op.judge_ok],
                      POOL_Q[POOL_Q.object_label.eq("none") & POOL_Q.headline & POOL_Q.judge_ok]], ignore_index=True)
    none = none[none.metric_id.eq("asr") & pd.to_numeric(none.threshold, errors="coerce").eq(nb["R"].SR_THRESHOLD)]
    return none.drop_duplicates(["run_id", "metric_id", "judgment_id"])


def _rule(R, rows: pd.DataFrame) -> tuple:
    """(A.y, A.se) by Figure 4's rule from the anchor's unsteered rows: A.y the mean of their rates (%), A.se one run's own half-width /
    1.96, several runs' item-clustered se (`pooled_ci.pooled` over each run's units from its judgment's item rows, `row_units`, Wilson at
    the cluster-effective n: the ASR runs carry Wilson intervals)."""
    y = rows.value.astype(float) * 100.0
    iv = R.intervals(rows.reset_index())
    if len(rows) == 1:
        return float(y.mean()), float((iv.hi.iat[0] - iv.lo.iat[0]) / 3.92)
    units = [R.row_units(x) for _, x in iv.iterrows()]
    return float(y.mean()), R._pooled_ci().pooled(units, form="wilson", scale=100.0).se


def test_figure_4_formula_reproduces_every_drawn_row_on_all_goals(ran):
    R, st_d = ran["R"], ran["st_d"]
    none = _none_rows(ran)
    for r in st_d[st_d.d_y.notna()].itertuples():
        ids = str(r.anchor_run).split(",")
        rows = none[none.eval_id.eq(r.eval_id)].set_index("run_id").loc[ids]
        ay, ase = _rule(R, rows)
        dy = r.y - ay
        s = np.hypot((r.hi - r.lo) / 3.92, ase)
        assert abs(r.anchor_value - ay) < 1e-9 and abs(r.d_y - dy) < 1e-9 and abs(r.d_lo - (dy - 1.96 * s)) < 1e-9, (r.eval_id, r.object_label, r.seed)
        assert abs(r.anchor_se - ase) < 1e-9 and r.anchor_se_method == "item-clustered (Wilson at the effective n)", (r.eval_id, r.object_label)


def _wilson(k: float, n: float, z: float = 1.959963984540054) -> tuple:
    """The Wilson score interval written out, at the StrongREJECT producers' z (scipy norm.ppf(0.975))."""
    p, den = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def _goal_units(R, entry: dict, t: float, idx, order: list):
    """The StrongREJECT rule's cluster units written out from a payload: one unit per determined goal of `idx` (a success, or every reply
    scored), a = 1 for a success and b = 1, keyed by its goal id; `present` every goal of `idx`."""
    a = {}
    for i in idx:
        hit = entry["best"][i] is not None and entry["best"][i] > t
        if hit or entry["n_scored"][i] == entry["n_units"][i]:
            a[order[i]] = float(hit)
    return R._pooled_ci().make_units(pd.Series(a, dtype=float), 1.0, None, [order[i] for i in idx])


def test_formula_reproduces_every_row_outside_the_training_pool(ran):
    """Each outside-pool row from the sidecar's per-goal best scores alone: the determined goals among the 92 and their successes, Wilson,
    against the subset rates of exactly the runs Figure 4's anchor pools in this render, the anchor's interval item-clustered over the 92
    goals from those runs' payloads (`_goal_units`, `pooled_ci.pooled`)."""
    R, d, st_out_d, gb, goals = ran["R"], ran["d"], ran["st_out_d"], ran["st_goal_best"], ran["st_goals"]
    idx = [i for i, c in enumerate(goals["outside_pool"]) if c == "1"]
    order = R.sitarget_goal_order(goals)
    t = R.SR_THRESHOLD

    def kn(rid):
        e = gb[rid]
        k = n = 0
        for i in idx:
            hit = e["best"][i] is not None and e["best"][i] > t
            if hit or e["n_scored"][i] == e["n_units"][i]:
                n += 1
                k += int(hit)
        return k, n

    assert sorted(st_out_d.run_id) == sorted(ran["st_rows"][ran["st_rows"].role.eq("arm")].run_id)
    for r in st_out_d.itertuples():
        ids = d[d.eval_id.eq(r.eval_id) & d.d_y.notna() & d.item_set_id.eq(R.SITARGET_ITEM_SET)].anchor_run.iat[0].split(",")
        ay = float(np.mean([100.0 * kn(a)[0] / kn(a)[1] for a in ids]))
        ase = R._pooled_ci().pooled([_goal_units(R, gb[a], t, idx, order) for a in ids], form="wilson", scale=100.0).se
        k, n = kn(r.run_id)
        lo, hi = _wilson(k, n)
        dy = 100.0 * k / n - ay
        s = np.hypot(100.0 * (hi - lo) / 3.92, ase)
        assert abs(r.d_y - dy) < 1e-9 and abs(r.d_lo - (dy - 1.96 * s)) < 1e-9 and abs(r.d_hi - (dy + 1.96 * s)) < 1e-9, r.run_id
        assert sorted(str(r.anchor_run).split(",")) == sorted(ids)


def test_every_run_is_on_its_eval_s_one_judge_and_the_cell_said_so(ran):
    R = ran["R"]
    assert R.sitarget_rubric_problems(ran["st_d"], ran["st_unsteered"]) == []
    for e in R.SITARGET_EVALS:
        assert f"sitarget rubric {e}: 18 refit, 6 corpus fit, 6 prompted" in ran["_printed"] and "0 mismatches" in ran["_printed"]


def test_the_new_cell_changes_no_frame(ran):
    before, after = ran["_digests"]
    assert before == after, [k for k in FRAMES if before[k] != after[k]]


def test_the_figure_and_its_tables_are_written(ran):
    R, figs = ran["R"], ran["_figs"]
    for ext in ("png", "pdf", "svg"):
        assert (figs / f"{R.SITARGET_FIGURE}.{ext}").is_file()
    assert (figs / f"{R.SITARGET_FIGURE}_seeds.tex").is_file() and (figs / f"{R.SITARGET_FIGURE}_seeds-outside-pool.tex").is_file()


def test_the_paper_table_is_the_paper_s_and_no_file_names_a_judge_version(ran):
    """The snapshot render's paper table (the cell after the figure's) is the paper's table character for character (paper/tests/data/
    si_target_paper_table.tex: the table the manuscript must print, in its form without the revision macro), printed and written beside
    the figure; and no file the two cells write carries a render stamp or names a StrongREJECT rubric version (the paper's render stamps
    nothing, and the generator comparison's lane-fixed registration comes after these cells)."""
    R, figs = ran["R"], ran["_figs"]
    want = PAPER_TABLE.read_text(encoding="utf-8")
    assert (figs / f"{R.SITARGET_PAPER_TABLE}.tex").read_text(encoding="utf-8") == want
    assert ran["_printed_table"].strip() == want.strip()
    written = sorted(figs.glob(f"{R.SITARGET_FIGURE}*.tex"))
    assert len(written) == 3, written
    for f in written:
        text = f.read_text(encoding="utf-8")
        assert "% judges" not in text and not re.search(r"strongreject v\d|StrongREJECT \(?v\d", text, re.I), f.name


# ---- the pooled intervals (item-clustered) on the real frames ------------------------------------------------------------------------
def test_a_full_pass_of_the_cells_raises_no_units_mismatch_and_pools_the_outside_rows_on_their_own_units(ran):
    """The two cells ran whole on the committed snapshot under the item-clustered intervals (`ran`: a UnitsMismatch anywhere in them would
    have failed the fixture). Every reading pooled since the render began - Figure 4's anchors and every pool the cells added - is held to
    its units again, and none is a mismatch; the pooled readers the cells call run again on their frames. Every row on the goals outside the
    pool names SITARGET_OUTSIDE_SOURCE, its anchor's unit keys are that source's, and its anchor's interval and each prompted instruction's
    over its trials are item-clustered over the 92 goals: no outside-pool pool takes the marked bound."""
    R, st_d, st_out_d, cats = ran["R"], ran["st_d"], ran["st_out_d"], ran["st_cats"]
    P = R._pooled_ci()
    mismatched = []
    for k, row in list(R._UNIT_ROWS.items()):
        try:
            R.row_units(row)
        except P.UnitsUnavailable:
            pass
        except P.UnitsMismatch as err:
            mismatched.append(f"{k}: {err}")
    assert not mismatched, mismatched[:5]
    for f in (st_d, st_out_d):
        R.pool_seeds(R.category_means(f, categories=cats))
        R.pool_draws(f[f.d_y.notna()])
    clustered = "item-clustered (Wilson at the effective n)"
    assert len(st_out_d) == 60 and st_out_d.judge_source.eq(R.SITARGET_OUTSIDE_SOURCE).all()
    assert all(k.endswith("|" + R.SITARGET_OUTSIDE_SOURCE) for ks in st_out_d.anchor_units for k in ks.split(";"))
    assert set(st_out_d.anchor_se_method) == {clustered}
    pr = R.pool_draws(st_out_d[st_out_d.object_label.isin(R.SITARGET_PROMPTED_SLOTS)])
    assert len(pr) == 4 and (pr.n_draws == 3).all() and pr.cell_method.eq(clustered).all()
    assert not [k for k in R.POOLED_CI_REPORT["bound"] if any(R.SITARGET_OUTSIDE_SOURCE in u for u in k)]


def test_figure_4s_anchor_registrations_survive_the_cells(ran):
    """No reading on the goals outside the pool sits under a unit key of a reading on all goals, so `anchored` registering the lower row's
    anchor replaced none of Figure 4's: after the cells, every anchor run Figure 4 pools on the multi-turn evals is registered as its
    reading on all 259 goals, the row Figure 4 differenced against."""
    R, d = ran["R"], ran["d"]
    stray = [k for k, r in R._UNIT_ROWS.items()
             if str(r.get("item_set_id")).endswith(R.SITARGET_OUTSIDE_TAG) and not k.endswith("|" + R.SITARGET_OUTSIDE_SOURCE)]
    assert not stray, stray
    ref = d[d.eval_id.isin(R.SITARGET_EVALS) & d.d_y.notna() & d.item_set_id.eq(R.SITARGET_ITEM_SET)]
    keys = {k for ks in ref.anchor_units for k in str(ks).split(";") if k}
    assert len(keys) >= 2 * 5
    for k in keys:
        r = R._UNIT_ROWS[k]
        assert r["item_set_id"] == R.SITARGET_ITEM_SET and r.get("judge_source") != R.SITARGET_OUTSIDE_SOURCE and float(r["n"]) > 250, k


def test_every_anchor_prompted_and_corpus_fit_run_s_payload_units_are_its_registry_units(ran):
    """THE PAYLOAD IS THE JUDGMENT, AS UNITS: for every run Figure 4's multi-turn anchor pools, every prompted run and every corpus fit's run
    (the registry readings whose units pool), the StrongREJECT rule over the sidecar's payload on all 259 goals, keyed by the goal ids
    (`sitarget_payload`, `judge_rules.payload_reading_units`: the file's unit source), gives exactly the units the render read from the
    run's judgment's item rows (`row_units`): the same goal ids, the same a and b. So the file's units (the same payload, on all goals or
    restricted to the goals outside the pool) are the judgment's, in the registry's namespace, and the fast half's sidecar-only provider
    pools as the render pools."""
    R, d, st_d, gb, goals = ran["R"], ran["d"], ran["st_d"], ran["st_goal_best"], ran["st_goals"]
    order, JR = R.sitarget_goal_order(goals), R._judge_rules()
    ref = d[d.eval_id.isin(R.SITARGET_EVALS) & d.d_y.notna() & d.item_set_id.eq(R.SITARGET_ITEM_SET)]
    rows = [R._UNIT_ROWS[k] for k in sorted({k for ks in ref.anchor_units for k in str(ks).split(";") if k})]
    assert len(rows) >= 2 * 5
    fig4 = [r for _, r in st_d[st_d.object_label.isin(R.SITARGET_PROMPTED_SLOTS + R.SITARGET_REFERENCE_SLOTS)].iterrows()]
    assert len(fig4) == 2 * 2 * 3 + 2 * 2 * 3
    rows += fig4
    for r in rows:
        assert R.unit_key(r).endswith("|registry"), r["run_id"]
        reg = R.row_units(r).table.sort_index()
        ru = JR.payload_reading_units("strongreject", R.sitarget_payload(gb[str(r["run_id"])], order), float(r["threshold"]))
        pay = R._pooled_ci().make_units(ru.a, ru.b, ru.stratum, ru.present).table.sort_index()
        assert reg[["a", "b"]].equals(pay[["a", "b"]]), r["run_id"]


def test_the_distilled_objects_seed_pools_read_the_file_s_units_item_clustered_across_their_seeds(ran):
    """#543's RULE ON THE REAL FRAMES (the fit-seed pools, `seed_pool`): every distilled object's row on all goals names its payload as its
    unit source (SITARGET_FILE_SOURCE) and every row on the goals outside the pool its own (SITARGET_OUTSIDE_SOURCE); the steered runs each
    distilled object's seed pool reads (the figure's mean marks, Table 5's seed means) are registered under those sources' keys, never
    the registry's, and the file's units of each distilled run equal the units the snapshot's item rows of its judgment give (the source
    reads the judgment, keyed by its goal ids). Every distilled and corpus-fit seed pool, on both rows, is item-clustered across its
    seeds (SEED_CLUSTERED); no pool on the multi-turn evals rests on a bound, none is opaque, and every pool's fits hold the same goal
    ids. A lane whose file rows lost their source would read the registry's key here (or take the marked bound where the source holds no
    item rows for a reading), and fails this test."""
    R, st_d, st_out_d, cats = ran["R"], ran["st_d"], ran["st_out_d"], ran["st_cats"]
    new = st_d[st_d.object_label.isin(R.SITARGET_NEW_SLOTS)]
    assert len(new) == 6 * 3 * 2 and new.judge_source.eq(R.SITARGET_FILE_SOURCE).all() and new.item_set_id.eq(R.SITARGET_ITEM_SET).all()
    assert st_out_d.judge_source.eq(R.SITARGET_OUTSIDE_SOURCE).all()
    for f, src, slots in ((st_d, R.SITARGET_FILE_SOURCE, R.SITARGET_NEW_SLOTS),
                          (st_out_d, R.SITARGET_OUTSIDE_SOURCE, R.SITARGET_NEW_SLOTS + R.SITARGET_REFERENCE_SLOTS)):
        cm = R.category_means(f, categories=cats)
        pooled = R.pool_seeds(cm).set_index("object_label")
        for slot in R.SITARGET_NEW_SLOTS + R.SITARGET_REFERENCE_SLOTS:
            assert pooled.loc[slot, "se_method"] == R.SEED_CLUSTERED and pooled.loc[slot, "n_seeds"] == 3, slot
            keys = {k for terms in cm[cm.object_label.eq(slot)].steer_terms for _e, kw in terms for k, _w in kw}
            assert len(keys) == 6, (slot, keys)
            if slot in slots:
                assert all(k.endswith("|" + src) for k in keys), (slot, sorted(keys)[:2])
                for k in keys:
                    row = R._UNIT_ROWS[k]
                    assert row["judge_source"] == src and row["item_set_id"].startswith(R.SITARGET_ITEM_SET), k
            else:
                assert all(k.endswith("|registry") for k in keys), (slot, sorted(keys)[:2])
    for _, r in new.iterrows():
        reg = R.item_units({**r.to_dict(), "judge_source": "minimax-m3 items"}).table.sort_index()     # its judgment's item rows
        assert R.row_units(r).table.sort_index()[["a", "b"]].equals(reg[["a", "b"]]), r.run_id
    rep = R.POOLED_CI_REPORT
    assert not [v for v in rep["bound"].values() if set(str(v["eval_id"]).split(",")) & set(R.SITARGET_EVALS)]
    assert rep["opaque_seed_pools"] == 0 and rep["seed_overlap_min"] == 1.0 and rep["alignment_min"] == 1.0


# ---- the live derivation ------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not LIVE_DB.exists() or os.environ.get("REGISTRY_NB_SOURCE") == "snapshot",
                    reason="needs the live registry (and REGISTRY_NB_SOURCE not forced to snapshot)")
def test_live_derivation_equals_the_committed_sidecar(tmp_path):
    """A live render's check, on the real registry: the generator's notebook namespace (its pinned judges) builds `frontier`, a live load of
    the committed sidecar does not refuse, and a fresh derivation equals the committed file (rows, payloads, unscored replies, goals, pins)."""
    saved_cwd = os.getcwd()
    try:
        with _fresh({"REGISTRY_NB_FIGS": str(tmp_path)}):
            sys.modules.pop("registry_nb", None)
            import si_target_sidecar as G
            ns = G.notebook_namespace()
            R = ns["R"]
            assert R.SOURCE == "live" and R.JUDGES == "rejudge"
            frame, gb, goals = R.sitarget_load("live", ns["frontier"])
            committed = json.loads(pathlib.Path(R.SITARGET_SIDECAR).read_text(encoding="utf-8"))
            assert len(frame) == len(committed["rows"]) and gb == committed["goal_best"] and goals == committed["goals"]
            fresh, problems, _new = G.derive(ns)
            assert problems == []
            for k in ("schema", "objects", "runs", "goals", "goal_best"):
                assert fresh.get(k) == committed.get(k), k
            assert fresh["provenance"]["unscored_replies"] == committed["provenance"]["unscored_replies"]
            assert len(R.sitarget_sidecar_check(committed["rows"], fresh["rows"])) == 0
    finally:
        os.chdir(saved_cwd)
