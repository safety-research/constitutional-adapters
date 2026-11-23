"""The generator-comparison appendix figure (`figA_qwen_generator_ablation`) on the notebook's REAL frames: the slow half
(SISTEER_SLOW_TESTS=1), beside test_registry_nb_ctlstop_appendix_slow.py.

The notebook's header and load cells, Figure 3's anchoring and the four control-pool cells run as make_registry_notebook.py writes them,
on the source the environment names (the committed paper-frame snapshot by default: about two to three minutes and the reassembled
SQLite under paper/notebooks/_cache/), then the generator cell itself. The fast tests anchor on synthetic main-figure frames; this
module holds the same rules against the frames Figure 11 is drawn from, so a change to `registry_nb.anchored` or a refreshed snapshot
that breaks the figure's rows is caught before a render:
  * the committed file through the generator cell: every drawn reading anchored, once per condition, each condition's rows all
    returned and clean under the anchor rule (`registry_nb.ctlstop_anchor_problems` against ctl_md), every static reading with a
    delta (no rung gap), each (eval, cap, seed) group's anchor runs the ones Figure 11's fitted rows of the class carry, the table's
    `anchor cap` column flagging exactly the rows that carry cap_mismatch, the figure saved in three formats, the verdict printed;
  * a FULL file and a MID-WAVE file derived from the synthetic registry (`appendix_generator_world`: the later families filed as the
    eval wave will file them) anchored on the same frames: 36 of 36 rows per condition when full, every filed reading with a delta,
    the anchor rule held, every reading not yet filed drawn as a gap and named;
  * the day-0 canary's core: Figure 11's own control-pool readings of the eight later evaluations, relabelled to this figure's object
    and each corpus, take exactly the anchors, deltas and routes the control-pool cells gave them; both corpora in one
    `ctlstop_anchored` call refuse (the cell key has no condition).
Run it once on the final head, after the wave's file is written, and record its pass line."""
import contextlib
import io
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

SLOW = os.environ.get("SISTEER_SLOW_TESTS") == "1"
pytestmark = pytest.mark.skipif(not SLOW, reason="the notebook's full load chain on the committed snapshot (~3 min warm, about 4 GB cache): "
                                                 "SISTEER_SLOW_TESTS=1 opts in")
USES_ITEM_ROWS = True                     # the conftest's synthetic-frame provider stands down: the real frame's pooled cells read its item rows
REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
TESTS = pathlib.Path(__file__).resolve().parent
GENERATOR = NOTEBOOKS / "make_registry_notebook.py"
LANE_CODE_CELLS = ("a6e0f2b8", "d2a7e5c9", "7fae15d6", "76fdab54")
GEN_CELL = "431232a9"
FIGURE = "figA_qwen_generator_ablation"
#: the modules this module imports afresh under the render's source and restores after (the fast tests' copies stay untouched)
OWN_MODULES = ("registry_nb", "appendix_common", "ctlstop_sidecar", "appendix_generator_sidecar", "appendix_generator",
               "appendix_generator_world")
ENV = ("REGISTRY_NB_SOURCE", "REGISTRY_NB_FIGS", "APPENDIX_FIGS_MIRROR")
#: the day-0 canary's relabelling: one control-pool slot per seed
SLOT_BY_SEED = {"s1": "control_lora", "s2": "control_relu", "s3": "difference_lora"}
MIRROR_COLUMNS = ("d_y", "d_lo", "d_hi", "anchor_run", "anchor_value", "route", "cap_mismatch", "anchor_in_frame", "anchor_draws")


def generator_cell(cid: str) -> str:
    """A cell's source exactly as make_registry_notebook.py writes it (the generator strips the newlines around the raw string)."""
    src = GENERATOR.read_text(encoding="utf-8")
    m = re.search(r'^(?:md|code)\("%s", r("""|\'\'\')(.*?)\1\)' % cid, src, flags=re.M | re.S)
    assert m, cid
    return m.group(2).strip("\n")


@pytest.fixture(scope="module")
def nb(tmp_path_factory, notebook_rcparams):
    """The notebook namespace after the header and load cells, Figure 3's anchoring, DCOLS and the four control-pool cells."""
    import matplotlib
    matplotlib.use("Agg")
    saved_mods = {m: sys.modules.get(m) for m in OWN_MODULES}
    saved_env = {k: os.environ.get(k) for k in ENV}
    saved_cwd, saved_path = os.getcwd(), list(sys.path)
    figs = tmp_path_factory.mktemp("generator_figs")
    try:
        os.chdir(NOTEBOOKS)
        os.environ["REGISTRY_NB_SOURCE"] = saved_env["REGISTRY_NB_SOURCE"] or "snapshot"
        os.environ["REGISTRY_NB_FIGS"] = str(figs)
        os.environ.pop("APPENDIX_FIGS_MIRROR", None)
        for m in OWN_MODULES:
            sys.modules.pop(m, None)
        for p in (str(TESTS), str(NOTEBOOKS)):
            if p not in sys.path:
                sys.path.insert(0, p)
        ns = {"__name__": "__generator_slow__"}
        with contextlib.redirect_stdout(sys.stderr):
            exec(generator_cell("583e3f04"), ns)
            exec(generator_cell("ec6aacd8"), ns)
            exec("d = R.anchored(op, pool=POOL_Q)\nCAT_Q = R.categories_for(R.MODEL_ID)\ncm = R.category_means(d, categories=CAT_Q)", ns)
            exec(next(ln for ln in generator_cell("ac902304").splitlines() if ln.startswith("DCOLS = ")), ns)
            for cid in LANE_CODE_CELLS:
                exec(generator_cell(cid), ns)
        ns["_figs"] = figs
        yield ns
    finally:
        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for m, mod in saved_mods.items():
            if mod is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = mod


@pytest.fixture(scope="module")
def cell(nb):
    """The namespace after the generator cell, with what it printed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(generator_cell(GEN_CELL), nb)
    nb["_text"] = buf.getvalue()
    nb["G"] = sys.modules["appendix_generator_sidecar"]
    return nb


def _anchor_sets_are_figure_14s(R, gd: pd.DataFrame, md: pd.DataFrame) -> tuple:
    """Per (eval, cap, seed) group of anchored rows, its anchor runs against those of Figure 11's fitted rows of the class in ctl_md (the
    main figures' rows: on over-refusal the same eval and seed, else the same eval, cap class and item set); (compared, groups without
    a fitted row of the class)."""
    fitted = md[md.object_label.isin(R.CTLSTOP_BOUND_SLOTS) & md.d_y.notna()]
    compared, alone = 0, []
    for (e, cap, s), g in gd[gd.d_y.notna()].groupby(["eval_id", "token_cap_tokens", "seed"]):
        if e in R.SHARED_FRAME_ANCHOR:
            peers = fitted[fitted.eval_id.eq(e) & fitted.seed.astype(str).eq(str(s))]
        else:
            peers = fitted[fitted.eval_id.eq(e) & pd.to_numeric(fitted.token_cap_tokens, errors="coerce").isin(R.cap_class(e, cap))
                           & fitted.item_set_id.eq(g.item_set_id.iat[0])]
        if not len(peers):
            alone.append((e, cap, s))
            continue
        assert set(g.anchor_run.astype(str)) == set(peers.anchor_run.astype(str)), (e, cap, s)
        compared += 1
    return compared, alone


def _anchor_rule_per_condition(R, AG, gd: pd.DataFrame, md: pd.DataFrame, want: dict):
    for c in AG.CONDITIONS:
        part = gd[gd.condition.eq(c)]
        assert len(part) == want[c], (c, len(part), want[c])
        assert R.ctlstop_anchor_problems(part, md) == [], c


# ---- 1. the committed file through the generator cell --------------------------------------------------------------------------
def test_the_cell_anchors_the_committed_file_on_figure_14s_frames(cell):
    R, AG, G, md = cell["R"], cell["AG"], cell["G"], cell["ctl_md"]
    data, gd, rows = cell["gen_data"], cell["gen_d"], cell["gen_rows"]
    drawn = [r for r in data["rows"] if r["role"] == "drawn"]
    assert len(gd) == len(drawn) and set(gd.run_id) == {r["run_id"] for r in drawn}
    _anchor_rule_per_condition(R, AG, gd, md, {c: sum(G.CONDITION[r["arm"]] == c for r in drawn) for c in AG.CONDITIONS})
    st = gd[gd.eval_id.isin(G.STATIC)]
    assert len(st) == 24 and st.d_y.notna().all() and not st.route.astype(str).str.startswith(R.CTLSTOP_RUNG_GAP_ROUTE).any()
    compared, alone = _anchor_sets_are_figure_14s(R, gd, md)
    assert compared > 0 and not alone, alone
    assert len(rows) == 72 and int(rows.delta.notna().sum()) == len(drawn)
    t = AG.marks_table(rows)
    other = rows.delta.notna() & ~rows["eval"].isin(R.SHARED_FRAME_ANCHOR) & rows.cap_mismatch.astype(bool)
    assert ((t["anchor cap"] == "other") == other).all()
    shared = rows.delta.notna() & rows["eval"].isin(R.SHARED_FRAME_ANCHOR)   # the route this render's anchored rows took, bound or pooled
    assert (t["anchor cap"][shared] == rows.anchor_binding[shared]).all()
    assert set(rows.anchor_binding[shared]) <= {R.CTLSTOP_ANCHOR_BOUND, R.CTLSTOP_ANCHOR_POOLED}
    for ext in ("png", "pdf", "svg"):
        assert (cell["_figs"] / f"{FIGURE}.{ext}").stat().st_size > 1000
    text = cell["_text"]
    assert "VERDICT (pre-fixed" in text and "generator comparison:" in text and "anchor cap" in text
    if R.SR_THRESHOLD == 0.5:
        assert "-7.56 pp defense success, 95% -11.06 to -4.07" in text
    if data["pending"]:
        assert f"{len(data['pending'])} reading(s) not yet in the data" in text and "!! NOT DRAWN" in text


# ---- 2. full and mid-wave files on the same frames -------------------------------------------------------------------------------
@pytest.mark.parametrize("evals", [None, ("strongreject_bon", "mmlu", "gsm8k", "xstest")], ids=["full", "mid-wave"])
def test_a_file_the_wave_fills_anchors_every_filed_reading_on_figure_14s_frames(cell, evals):
    R, AG, G, md = cell["R"], cell["AG"], cell["G"], cell["ctl_md"]
    import appendix_generator_world as W
    c = W.registry(G.load_rows())
    W.file_wave(c, rng=np.random.default_rng(11), **({} if evals is None else {"evals": evals}))
    assert W.G is G                                      # one module: unpinned() lifts the pin this derivation reads
    with W.unpinned():
        data = G.derive(c)
    c.close()
    assert (data["pending"] == []) == (evals is None)
    gd = AG.anchor(AG.lane_frame(data), cell["ctl_op"], cell["ctl_pool"], md)
    drawn = [r for r in data["rows"] if r["role"] == "drawn"]
    want = {cc: sum(G.CONDITION[r["arm"]] == cc for r in drawn) for cc in AG.CONDITIONS}
    if evals is None:
        assert want == {"opus_written": 36, "qwen_written": 36}
    _anchor_rule_per_condition(R, AG, gd, md, want)
    assert gd.d_y.notna().all(), gd[gd.d_y.isna()][["eval_id", "seed", "token_cap_tokens", "route"]]
    compared, alone = _anchor_sets_are_figure_14s(R, gd, md)
    assert compared > 0
    for e, cap, s in alone:                              # no fitted row of the class to compare with (a GSM8K rung Figure 11 never read)
        g = gd[gd.eval_id.eq(e) & gd.token_cap_tokens.eq(cap) & gd.seed.astype(str).eq(str(s))]
        assert g.route.astype(str).str.startswith("common anchor").all(), (e, cap, s)
    rows = AG.tidy_rows(gd, data)
    assert int(rows.delta.notna().sum()) == len(drawn) == 72 - len([p for p in data["pending"] if p["arm"] != G.UNSTEERED])
    named = " ".join(AG.not_drawn(rows))
    assert (named == "") == (evals is None)
    for p in data["pending"]:
        assert f"s{p['seed']} {R.EVAL_LABEL[p['eval_id']]}" in named
    import matplotlib.pyplot as plt
    plt.close(AG.draw(rows))


# ---- 3. the day-0 canary's core --------------------------------------------------------------------------------------------------
def test_figure_14s_control_pool_readings_relabelled_take_the_mirrors_anchors(cell):
    R, AG, G = cell["R"], cell["AG"], cell["G"]
    op, pool, md, ctl_rows = cell["ctl_op"], cell["ctl_pool"], cell["ctl_md"], cell["ctl_rows"]
    eight = [e for e in R.CTLSTOP_EVALS if e not in G.STATIC]
    src = ctl_rows[ctl_rows.rule.eq("control_pool") & ctl_rows.role.eq("drawn") & ctl_rows.eval_id.isin(eight)]
    src = src[src.slot.eq(src.seed.map(SLOT_BY_SEED))]
    assert len(src) == len(eight) * 3, src.groupby(["eval_id", "seed"]).size()
    parts = []
    for c in AG.CONDITIONS:
        s = src.copy()
        s["source_run_id"], s["run_id"] = s.run_id, s.run_id + "#" + c
        s["object_label"], s["slot"], s["condition"] = AG.OBJECT, AG.OBJECT, c
        parts.append(s)
    syn = pd.concat(parts, ignore_index=True)
    gd = AG.anchor(syn, op, pool, md)
    _anchor_rule_per_condition(R, AG, gd, md, {c: len(eight) * 3 for c in AG.CONDITIONS})
    mirror = R.concat_plain(cell["ctl_st_new"], cell["ctl_new_d"], cell["ctl_cap_new"]).set_index("run_id")
    src_of = syn.set_index("run_id").source_run_id
    bad = []
    for r in gd.itertuples():
        m = mirror.loc[src_of[r.run_id]]
        for col in MIRROR_COLUMNS + (("anchor_binding",) if r.eval_id in R.SHARED_FRAME_ANCHOR else ()):
            a, b = getattr(r, col), m[col]
            same = (pd.isna(b) if not isinstance(b, str) else False) if (not isinstance(a, str) and pd.isna(a)) else a == b
            if not same:
                bad.append(f"{r.run_id} {col}: {a!r} / mirror {b!r}")
    assert not bad, bad[:10]
    with pytest.raises(AssertionError, match="anchoring kept"):
        R.ctlstop_anchored(op, syn, pool, md)
