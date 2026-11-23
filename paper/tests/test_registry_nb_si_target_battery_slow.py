"""The distillation-target battery figure: the SLOW half, on the notebook's full load chain (SISTEER_SLOW_TESTS=1).

The generator's header and load cells and Figure 3's anchoring run against the committed paper-frame snapshot (minutes, and a 2 GB
reassembled SQLite under paper/notebooks/_cache/), then the multi-turn distillation-target cell and the battery cell run exactly as the
generator writes them. Pinned here:
  * under each run's filed judges (`primary`, set explicitly) the battery cell prints its one RENDER POLICY line, touches no file (an
    earlier render's copies are the zip cell's sweep to judge), draws nothing and changes no frame;
  * under the re-judge judges, where the committed snapshot carries the battery's judges (skipped, and named, while it does not: the
    snapshot is re-cut with the judge default), the prompted instructions' and the constitution and control corpus fits' rows the figure
    and tables read ARE the main figures' rows of `d` on every eval, Best-of-75 included; every distilled row carries its eval's anchor
    rule, a Best-of-75 one Figure 4's own pool; every run passes the one-rubric check; the cell changes no frame the figures above read;
    the figure and its five tables are written; under the item-clustered pooled intervals no pool the battery's marks rest on takes
    the identical-draws bound (the cell prints no `!! POOLED INTERVAL` line, records or reuses no bound pool and pools no opaque cell,
    and no bound pool on record holds a run the battery draws or differences against), and the main figures' anchor registrations
    (`_UNIT_ROWS`, which every later cell's pooled interval reads) are the same readings after the battery cell as before it; every
    distilled object's mean mark reads its seeds' runs through the battery file's own unit source (the multi-turn panel's through the
    multi-turn file's), item-clustered across the seeds (#543), and where the snapshot holds a battery run's item rows, the file's units
    of the reading equal theirs unit by unit;
  * where the live registry is present, a live derivation of the battery sidecar equals the committed file and a live load of it does
    not refuse.
Every load runs with REGISTRY_NB_STRENGTH_READS cleared, as the paper's render clears it. Run it once on the final head and record its
pass line."""
import contextlib
import hashlib
import io
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
MT_CELL, BATTERY_CELL = "a2d84f6b", "9d4a6f20"
FRAMES = ("op", "d", "cm", "paper_all", "frontier", "POOL_Q")
LIVE_DB = pathlib.Path(os.path.expanduser(os.environ.get("REGISTRY_ROOT") or "~/registry")) / "registry.sqlite"
JUDGE_ENV = ("REGISTRY_NB_SOURCE", "REGISTRY_NB_JUDGES", "REGISTRY_NB_FIGS", "REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE",
             "REGISTRY_NB_PETRI_EXCLUDE", "REGISTRY_NB_STRENGTH_READS", "REGISTRY_NB_SITARGET_PROVISIONAL")
#: the cell refuses a PROVISIONAL battery file unless the render is flagged provisional (a final file and the committed pending-registration
#: state - every BigCodeBench and BFCL reading pending, nothing else - draw unflagged); this half exercises the cell on whatever file is
#: committed (the final gate itself is the fast half's `test_the_final_render_refuses_a_provisional_file_and_only_an_explicit_flag_draws_one`)
_COMMITTED = json.loads((NOTEBOOKS / "data" / "qwen_si_target_battery_rows.json").read_text(encoding="utf-8"))
_REGISTRATION = [dict(slot=s, seed=sd, eval_id=e) for e in ("bigcodebench", "bfcl")
                 for s in ("sitarget_safety_lora", "sitarget_safety_relu", "sitarget_constitution_lora", "sitarget_constitution_relu",
                           "sitarget_c16_lora", "sitarget_c16_relu") for sd in ("s1", "s2", "s3")]
PROVISIONAL_ENV = ({"REGISTRY_NB_SITARGET_PROVISIONAL": "1"}
                   if "pending" in _COMMITTED and _COMMITTED["pending"] != _REGISTRATION else {})


def generator_cell(cid: str) -> str:
    src = GENERATOR.read_text(encoding="utf-8")
    m = re.search(r'^(?:md|code)\("%s", r("""|\'\'\')(.*?)\1\)' % cid, src, flags=re.M | re.S)
    assert m, cid
    return m.group(2).strip("\n")


@contextlib.contextmanager
def _fresh(env: dict):
    """registry_nb imported afresh under `env`; the module, the environment and the working directory are restored after."""
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
    import matplotlib
    matplotlib.use("Agg")
    ns = {"__name__": "__sitarget_battery_slow__"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(generator_cell("583e3f04"), ns)
        ns["R"] = R
        exec(generator_cell("ec6aacd8"), ns)
        exec("d = R.anchored(op, pool=POOL_Q)\nCAT_Q = R.categories_for(R.MODEL_ID)\ncm = R.category_means(d, categories=CAT_Q)", ns)
    return ns


def _digest(f: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(repr(list(f.columns)).encode()); h.update(repr(list(f.dtypes.astype(str))).encode()); h.update(repr(sorted(f.attrs)).encode())
    g = f.copy()
    g.attrs = {}
    h.update(pd.util.hash_pandas_object(g.astype(str), index=True).values.tobytes())
    return h.hexdigest()


NA = "<NA>"


def _canon(v) -> str:
    """A field's value as a reading reads it: a number as its float (an integer column that a concatenation widened to float reads the
    same), a missing value as NA, anything else as its text."""
    try:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return NA
    except (TypeError, ValueError):                                         # a list or array field: not a scalar
        return repr(v)
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return repr(float(v))
    return str(v)


def _pooled_state(R) -> tuple:
    """What the render's pooled intervals rest on at this moment: each pool on record as taking the identical-draws bound with its uses
    (`POOLED_CI_REPORT['bound']`; a pool that takes it again, cached or not, counts a use), the count of cells whose members' intervals
    could not be split (`opaque_cells`, the bound too), and every registered anchor run's row (`_UNIT_ROWS`), field by field (`_canon`)."""
    bound = {k: rec["uses"] for k, rec in R.POOLED_CI_REPORT["bound"].items()}
    rows = {k: {str(c): _canon(v) for c, v in r.items()} for k, r in R._UNIT_ROWS.items()}
    return bound, R.POOLED_CI_REPORT["opaque_cells"], rows


def _run_cells(ns) -> dict:
    """The namespace after the multi-turn cell and the battery cell, with each frame's digest and the pooled state (`_pooled_state`) taken
    before and after the battery cell, and what the battery cell printed."""
    with contextlib.redirect_stdout(io.StringIO()):
        exec(generator_cell(MT_CELL), ns)
    before, pooled = {k: _digest(ns[k]) for k in FRAMES}, _pooled_state(ns["R"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(generator_cell(BATTERY_CELL), ns)
    ns["_digests"] = (before, {k: _digest(ns[k]) for k in FRAMES})
    ns["_pooled"] = (pooled, _pooled_state(ns["R"]))
    ns["_printed"] = out.getvalue()
    return ns


def test_under_each_run_s_filed_judges_the_battery_cell_prints_one_line_and_draws_nothing(tmp_path):
    with _fresh({"REGISTRY_NB_SOURCE": "snapshot", "REGISTRY_NB_JUDGES": "primary", "REGISTRY_NB_FIGS": str(tmp_path)}) as R:
        stale = [tmp_path / f"{R.SITARGET_BATTERY_FIGURE}.png", tmp_path / f"{R.SITARGET_BATTERY_FIGURE}_seeds-static.tex"]
        for f in stale:
            f.write_text("an earlier render", encoding="utf-8")
        ns = _run_cells(_load(R))
        printed = [ln for ln in ns["_printed"].splitlines() if ln.strip()]
        assert len(printed) == 1 and printed[0].startswith("RENDER POLICY · the distillation-target battery figure"), printed
        assert printed[0].endswith(": not drawn"), printed
        assert "bt_d" not in ns and sorted(tmp_path.glob(f"{R.SITARGET_BATTERY_FIGURE}*")) == sorted(stale)     # the zip cell's sweep judges them
        before, after = ns["_digests"]
        assert before == after


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    figs = tmp_path_factory.mktemp("sitarget_battery_figs")
    with _fresh({"REGISTRY_NB_SOURCE": "snapshot", "REGISTRY_NB_JUDGES": "rejudge", "REGISTRY_NB_FIGS": str(figs), **PROVISIONAL_ENV}) as R:
        with R.connect() as c:
            held = {e: c.execute("SELECT COUNT(*) FROM judgment WHERE judge_id = ? AND status = 'complete'", (j.judge_id,)).fetchone()[0]
                    for e, j in R.SITARGET_EVAL_JUDGE.items()}
        if not all(held.values()):
            pytest.skip("the committed snapshot carries no judgment of some of the battery's judges "
                        f"({', '.join(sorted(e for e, n in held.items() if not n))}): it is re-cut with the judge default, and this "
                        "half runs on that snapshot")
        try:
            loaded = _load(R)
        except R.StrengthReadsMissing as err:
            pytest.fail(f"the re-judge half's full load needs the strength lanes' XSTest reads and this tree has none to read: {err}")
        ns = _run_cells(loaded)
        ns["_figs"] = figs
        yield ns


def test_corpus_fit_and_prompted_rows_are_the_main_figures_on_every_eval(ran):
    R, d, bt_d = ran["R"], ran["d"], ran["bt_d"]
    key = ["object_label", "seed", "eval_id", "run_id"]
    every = [e for c in R.SITARGET_BATTERY_CATEGORIES.values() for e in c[0]]
    for labels, evals in ((list(R.SITARGET_PROMPTED_SLOTS + R.SITARGET_REFERENCE_SLOTS), list(R.SITARGET_BATTERY_EVALS)),
                          (list(R.SITARGET_CONTROL_SLOTS), every)):
        a = bt_d[bt_d.object_label.isin(labels) & bt_d.eval_id.isin(evals)].sort_values(key).reset_index(drop=True)
        b = d[d.model_id.eq(R.MODEL_ID) & d.object_label.isin(labels) & d.eval_id.isin(evals) & d.d_y.notna()].sort_values(key).reset_index(drop=True)
        assert len(a) == len(b) and len(a) and set(a.eval_id) >= {"strongreject_bon"}, labels
        for c in ("run_id", "d_y", "d_lo", "d_hi", "anchor_value", "anchor_run", "y", "lo", "hi", "judge_id", "judgment_id"):
            # a missing value as a marker: pandas' str dtype keeps it as NaN, which never equals itself (a rule-scored eval's judge_id)
            assert (a[c].astype(str).fillna("<NA>").to_numpy() == b[c].astype(str).fillna("<NA>").to_numpy()).all(), (labels, c)


def test_the_distilled_best_of_75_rows_sit_on_figure_4_s_own_pool(ran):
    R, d, bt_new = ran["R"], ran["d"], ran["bt_new"]
    f4 = d[d.model_id.eq(R.MODEL_ID) & d.eval_id.eq("strongreject_bon") & d.d_y.notna() & d.object_label.isin(R.SITARGET_BOUND_SLOTS)]
    pools = {(R._sitarget_runs_of(r.anchor_run), round(float(r.anchor_value), 9)) for r in f4.itertuples()}
    assert len(pools) == 1
    (runs, value), = pools
    bon = bt_new[bt_new.eval_id.eq("strongreject_bon")]
    assert len(bon) == 18 and all(R._sitarget_runs_of(r.anchor_run) == runs and abs(float(r.anchor_value) - value) < 1e-9
                                  for r in bon.itertuples())


def test_every_distilled_row_carries_its_eval_s_anchor_rule_and_every_run_one_rubric(ran):
    R = ran["R"]
    assert R.sitarget_battery_anchor_problems(ran["bt_new"], ran["d"]) == []
    assert R.sitarget_battery_rubric_problems(ran["bt_d"], ran["bt_unsteered"]) == []
    assert "0 mismatches" in ran["_printed"] and "sitarget rubric strongreject_bon:" in ran["_printed"]
    assert "anchor runs, Figure 4's pool; 0 mismatches" in ran["_printed"]


def test_the_battery_cell_changes_no_frame(ran):
    before, after = ran["_digests"]
    assert before == after, [k for k in FRAMES if before[k] != after[k]]


def test_the_battery_figure_and_its_five_tables_are_written(ran):
    """The figure and its five tables are written, and no table carries a render stamp or names a StrongREJECT rubric version: the
    paper's render stamps nothing, and the generator comparison's lane-fixed registration comes after the battery's cells."""
    R, figs = ran["R"], ran["_figs"]
    for ext in ("png", "pdf", "svg"):
        assert (figs / f"{R.SITARGET_BATTERY_FIGURE}.{ext}").is_file()
    for cat in R.SITARGET_BATTERY_CATEGORIES:
        assert (figs / f"{R.SITARGET_BATTERY_FIGURE}_seeds-{cat}.tex").is_file(), cat
    written = sorted(figs.glob(f"{R.SITARGET_BATTERY_FIGURE}*.tex"))
    assert len(written) == len(R.SITARGET_BATTERY_CATEGORIES), written
    for f in written:
        text = f.read_text(encoding="utf-8")
        assert "% judges" not in text and not re.search(r"strongreject v\d|StrongREJECT \(?v\d", text, re.I), f.name


# ---- the pooled intervals (item-clustered) on the real frames ------------------------------------------------------------------------
def _keys(cells) -> set:
    return {k for ks in cells for k in str(ks).split(";") if k and k != "nan"}


def test_no_battery_pool_takes_the_identical_draws_bound(ran):
    """NO MARK OF THE BATTERY RESTS ON THE IDENTICAL-DRAWS BOUND: the battery cell prints no `!! POOLED INTERVAL` line, records no bound
    pool and takes none recorded before it again (each take counts a use), and pools no cell whose members' intervals cannot be split;
    and no bound pool on record anywhere in the render (Figure 3's anchoring of `d` and the multi-turn figure's cells included, which
    pooled the anchors the battery reads) holds a run the battery draws or differences against: its rows' own runs and their anchors'."""
    R, bt_d = ran["R"], ran["bt_d"]
    assert "!! POOLED INTERVAL" not in ran["_printed"]
    (bound0, opaque0, _rows0), (bound1, opaque1, _rows1) = ran["_pooled"]
    assert bound1 == bound0, {k: (bound0.get(k), bound1.get(k)) for k in set(bound0) | set(bound1) if bound0.get(k) != bound1.get(k)}
    assert opaque1 == opaque0
    drawn = bt_d[bt_d.model_id.eq(R.MODEL_ID) & bt_d.d_y.notna()]
    anchors = _keys(drawn.anchor_units)
    keys = {R.unit_key(r) for _, r in drawn.iterrows()} | anchors
    every = [e for c in R.SITARGET_BATTERY_CATEGORIES.values() for e in c[0]]
    assert {e for e, ks in zip(drawn.eval_id, drawn.anchor_units) if _keys([ks])} == set(every)    # every eval's anchors are in `keys`
    hit = [(rec["eval_id"], rec["what"], rec["runs"]) for k, rec in R.POOLED_CI_REPORT["bound"].items() if keys.intersection(k)]
    assert not hit, hit[:5]


def test_the_main_figures_anchor_registrations_are_unchanged_across_the_battery_cell(ran):
    """The battery differences its distilled rows with `anchored` on `op`'s unsteered rows, which registers each anchor run again (the
    latest registration of a unit key wins, and every later cell's pooled interval reads it). Every anchor run registered before the cell,
    each of the main figures' Qwen3.5-4B anchors in `d` among them (the battery's own anchors are all of them), is registered after it as
    the same reading, field for field: each field the row had holds the same value, and a field the new row adds (the battery rows'
    columns, which the concatenation brings) holds none.

    Since the fit-seed pools (#543) `_UNIT_ROWS` also holds every STEERED run a cell pools (`_steer_weights`), and every cell that draws a
    dot plot (`delta_figure`: the main figures' and this one) registers its steered runs again as its plotted points, the run's change
    in the panel's orientation in `y` / `lo` / `hi`. So a steered run registered before the cell (Figure 3's `category_means`, the
    multi-turn figure's cells) may be registered after it as the battery's point of the same run, but as the SAME READING: its run, metric,
    judgment, threshold, best-of-N, unit source, item set, value, k and n, and its model, eval and object, unchanged."""
    R, d, bt_d = ran["R"], ran["d"], ran["bt_d"]
    (_b0, _o0, rows0), (_b1, _o1, rows1) = ran["_pooled"]
    main = _keys(d[d.model_id.eq(R.MODEL_ID) & d.d_y.notna()].anchor_units)
    assert main and main <= set(rows0), sorted(main - set(rows0))[:5]
    assert _keys(bt_d.anchor_units) <= main, sorted(_keys(bt_d.anchor_units) - main)[:5]
    anchors = main | {k for f in ("d", "st_d", "st_out_d") if f in ran and "anchor_units" in ran[f].columns
                      for k in _keys(ran[f].anchor_units)}
    gone = sorted(set(rows0) - set(rows1))
    changed, reread = {}, {}
    reading = ("run_id", "metric_id", "judgment_id", "threshold", "bon_n", "judge_source", "item_set_id", "value", "k", "n", "model_id",
               "eval_id", "object_label")
    for k in set(rows0) & set(rows1):
        if k in anchors:
            diff = {c: (v, rows1[k].get(c)) for c, v in rows0[k].items() if rows1[k].get(c) != v}
            diff.update({c: (None, v) for c, v in rows1[k].items() if c not in rows0[k] and v != NA})
            if diff:
                changed[k] = diff
        else:
            diff = {c: (rows0[k].get(c, NA), rows1[k].get(c, NA)) for c in reading if rows0[k].get(c, NA) != rows1[k].get(c, NA)}
            if diff:
                reread[k] = diff
    assert len(anchors & set(rows0)) >= len(main) and set(rows0) - anchors          # both kinds are on record before the cell
    assert not gone and not changed and not reread, (gone[:5], len(changed), dict(list(changed.items())[:3]), len(reread),
                                                     dict(list(reread.items())[:3]))


def test_the_distilled_objects_mean_marks_read_the_file_s_units_item_clustered_across_their_seeds(ran):
    """#543's FIT-SEED POOLS ON THE REAL FRAMES: every distilled object's mean mark in every category of the battery figure (its
    `category_means`, `pool_seeds`) reads its seeds' steered runs through the battery file's own source (SITARGET_BATTERY_SOURCE; the
    multi-turn panel's through the multi-turn file's, SITARGET_FILE_SOURCE), never the registry's key, and is item-clustered across its
    three seeds. Every pool of the render holds its fits on the same item ids and its runs on one item set aligned (both 1.0)."""
    R, cm = ran["R"], ran["bt_cm"]
    pooled = R.pool_seeds(cm[cm.model_id.eq(R.MODEL_ID)])
    for slot in R.SITARGET_NEW_SLOTS:
        for cat in R.SITARGET_BATTERY_CATEGORIES:
            c = cm[cm.object_label.eq(slot) & cm.category.eq(cat)]
            keys = {k for terms in c.steer_terms for _e, kw in terms for k, _w in kw}
            src = R.SITARGET_FILE_SOURCE if cat == "multi-turn" else R.SITARGET_BATTERY_SOURCE
            n_evals = len(R.SITARGET_BATTERY_CATEGORIES[cat][0])
            assert len(keys) == 3 * n_evals and all(k.endswith("|" + src) for k in keys), (slot, cat, sorted(keys)[:2])
            p = pooled[pooled.object_label.eq(slot) & pooled.category.eq(cat)]
            assert len(p) == 1 and p.se_method.iat[0] == R.SEED_CLUSTERED and int(p.n_seeds.iat[0]) == 3, (slot, cat, p.se_method.tolist())
    rep = R.POOLED_CI_REPORT
    assert rep["seed_overlap_min"] == 1.0 and rep["alignment_min"] == 1.0 and rep["opaque_seed_pools"] == 0


def test_the_file_s_units_are_the_registry_s_where_the_snapshot_holds_the_run_s_item_rows(ran):
    """THE ITEM IDS ARE THE REGISTRY'S: for every distilled reading whose run's item rows the committed snapshot holds (the render's own
    source), the units the battery file's payload gives (SITARGET_BATTERY_SOURCE, keyed by the file's item list) equal the units the
    registry's item rows give the same reading (the row read without the source: its judgment's rows, or a rule-scored run's), unit by
    unit. The snapshot predates most battery runs; the read-only proof against the live registry covers every run."""
    R, bt_d = ran["R"], ran["bt_d"]
    P = R._pooled_ci()
    rows = bt_d[bt_d.object_label.isin(R.SITARGET_NEW_SLOTS) & bt_d.eval_id.isin(list(R.SITARGET_BATTERY_EVALS))]
    assert len(rows) == 216 and rows.judge_source.eq(R.SITARGET_BATTERY_SOURCE).all()
    same, unread = 0, 0
    for _, r in rows.iterrows():
        mine = R.item_units(r)
        bare = r.copy()
        bare["judge_source"] = None                                            # the run's reading as the registry's item rows give it
        try:
            theirs = R.item_units(bare)
        except (P.UnitsUnavailable, P.UnitsMismatch):
            unread += 1
            continue
        assert mine.table[["a", "b"]].sort_index().equals(theirs.table[["a", "b"]].sort_index()), r.run_id
        assert mine.present >= theirs.present, r.run_id
        same += 1
    assert same + unread == 216 and same > 0, (same, unread)
    print(f"battery file units equal to the snapshot's item-row units: {same} of 216 readings ({unread} without item rows in the snapshot)")


@pytest.mark.skipif(not LIVE_DB.exists() or os.environ.get("REGISTRY_NB_SOURCE") == "snapshot",
                    reason="needs the live registry (and REGISTRY_NB_SOURCE not forced to snapshot)")
def test_live_derivation_equals_the_committed_battery_sidecar(tmp_path):
    """A live render's check, on the real registry: a fresh derivation by the generator equals the committed file (every stored field,
    payload, item list and the pending list) and a live load of it does not refuse."""
    with _fresh({"REGISTRY_NB_SOURCE": "live", "REGISTRY_NB_JUDGES": "rejudge", "REGISTRY_NB_FIGS": str(tmp_path)}) as R:
        committed = json.loads(pathlib.Path(R.SITARGET_BATTERY_SIDECAR).read_text(encoding="utf-8"))
        assert _REGISTRATION == R.sitarget_battery_registration_pending()   # the module-level copy above is the notebook's state
        allow = "pending" in committed
        # THE DEFAULT RENDER'S LOAD: no provisional flag; a final file or the pending-registration state passes it, nothing else
        frame, pending, _sc = R.sitarget_battery_load(
            "live", allow_pending=R.sitarget_battery_state(committed) == R.SITARGET_STATE_PROVISIONAL)
        assert len(frame) == sum(r["role"] == "drawn" for r in committed["rows"]) and pending == list(committed.get("pending") or [])
        fresh, problems = R._sitarget_battery_generator().derive(allow_pending=allow)
        assert problems == []
        assert len(R.sitarget_battery_check(committed, fresh)) == 0
