"""The judge comparison reads its figures from the notebook's own figure map (`judge_compare.figure_map`, 2026-11-12).

Pins, on the committed README and generator and on synthetic renders (no registry, no kernel, no model call):
  * the derived map: every README figure is placed, the figure number is the README's (a shifted synthetic table renumbers with no
    code change), Fig. 10 draws neg_cm / neg_d, Fig. 12 draws its own four frames rather than Fig. 11's, and a README figure the
    generator does not draw refuses rather than being skipped.
  * the comparison: a figure whose frame a render did not dump is marked "not compared: frame not dumped", a judged figure whose frames
    have no shape the comparison reads refuses, Petri's pooled column is never an eval mark, and the mean-column verdict changes are
    counted once per distinct mark with how far each clears zero.
  * provenance: a render clears the inherited threshold, Petri judge, Petri exclusions and NB_REUSE_CACHE unless a flag passes them and
    records the policy, threshold, Petri judge, NB_REUSE_CACHE and the strength-reads files' sha256 in render.json. compare.json carries
    each render's policy, threshold, frame heads and item-row head per judge, and the Petri comparison pins REGISTRY_NB_JUDGES=primary.
"""
import argparse
import hashlib
import json
import os
import pathlib
import pickle
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import judge_analysis as JA  # noqa: E402
import judge_compare as JC  # noqa: E402
import petri_judge_compare as PJC  # noqa: E402
import registry_nb as R  # noqa: E402

Q = R.MODEL_ID
SR, PETRI = "strongreject", "petri_bloom_judge"


# ---- fakes: nbformat / nbclient, so nothing needs a kernel or the notebook extras ----------------------------------------------------
class _Cell(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


def _fake_nb(monkeypatch, seen: dict):
    nbf = types.ModuleType("nbformat")
    nbf.read = lambda path, as_version=4: types.SimpleNamespace(cells=[])
    nbf.write = lambda nb, path: pathlib.Path(path).write_text("{}")
    nbf.v4 = types.SimpleNamespace(new_code_cell=lambda src: _Cell(cell_type="code", source=src, outputs=[], id="dump"))
    nbc = types.ModuleType("nbclient")

    class NotebookClient:
        def __init__(self, nb, **kw):
            self.nb = nb

        def execute(self):
            seen["env"] = dict(os.environ)
            seen["cells"] = [c["source"] for c in self.nb.cells]

    nbc.NotebookClient = NotebookClient
    monkeypatch.setitem(sys.modules, "nbformat", nbf)
    monkeypatch.setitem(sys.modules, "nbclient", nbc)


@pytest.fixture
def env_restored():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


# ---- synthetic README / generator / renders ---------------------------------------------------------------------------------------
def readme(rows) -> str:
    body = "\n".join(f"| {p} | {w} | `{n}` |" for p, w, n in rows)
    return ("# notebook\n\n## Figure map\n\n| Paper figure | What it shows | File |\n|---|---|---|\n" + body +
            "\n| Tables 4 to 10 | numbers | LaTeX printed under the cells |\n\n## How to read the figures\n\ntext\n")


# the call shapes of make_registry_notebook.py, on frames this test builds
GEN = '''
code("a1", r"""
d = R.anchored(op, pool=POOL_Q)
cm = R.category_means(d, categories=CAT_Q)
gaps_q = R.gap_status(runs_id, R.drawn_set(d), R.OBJECTS, ALL_EVALS)
FIG4 = [["multi-query", "multi-turn"]]
R.delta_figure(cm, d, FIG4, R.OBJECTS, gaps_q, "fig4_multi_turn", categories=CAT_Q); plt.show()
FIG5 = [["Petri"]]
R.delta_figure(cm, d, FIG5, R.OBJECTS, gaps_q, "fig5_misalignment", categories=CAT_Q); plt.show()
""")
code("a2", r"""
if len(_prov):
    neg_d = R.anchored(neg_rows, pool=POOL_Q)
    neg_cm = R.category_means(neg_d, categories=NEG_CATS)
    neg_gaps = R.gap_status(runs_id, R.drawn_set(neg_d), R.NEGATIVE_DOSE_OBJECTS, NEG_EVALS)
    R.delta_figure(neg_cm, neg_d, R.NEGATIVE_DOSE_LAYOUT, R.NEGATIVE_DOSE_OBJECTS, neg_gaps, "figA_qwen_control_alpha_minus1", categories=NEG_CATS); plt.show()
""")
code("a3", r"""
ctl_cats = {"multi-turn": CAT_Q["multi-turn"]}
ctl_new_d = R.ctlstop_anchored(op, ctl_new, POOL_Q, d)
ctl_d = pd.concat([ctl_keep, ctl_new_d], ignore_index=True)
ctl_cm = R.category_means(ctl_d, categories=ctl_cats)
ctl_gaps = R.gap_status(runs_id, R.drawn_set(ctl_d), R.OBJECTS, ctl_evals)
R.delta_figure(ctl_cm, ctl_d, [["multi-turn"]], R.OBJECTS, ctl_gaps, "figA_qwen_ctlstop_multi_turn", categories=ctl_cats); plt.show()
ctl_pub_d = pd.concat([d, _ctl_pub_v], ignore_index=True)
ctl_pub_cm, ctl_new_cm = (R.category_means(f, categories=ctl_cats) for f in (ctl_pub_d, ctl_new_d))
R.rule_comparison_figure(ctl_pub_d, ctl_new_d, ctl_pub_cm, ctl_new_cm, "figA_qwen_ctlstop_rule_comparison", categories=ctl_cats,
                         pending=ctl_pending); plt.show()
""")
code("a4", r"""
d_all = R.anchored(op_all, pool=POOL_Q)
cm_all = R.category_means(d_all, categories=R.APPENDIX_CATEGORIES)
gaps_m = R.gap_status(runs_lane, R.drawn_set(d_all), R.APPENDIX_OBJECTS, ALL_EVALS)
JB_ALL = [["static", "multi-query", "multi-turn"]]
R.category_panels(cm_all, d_all, JB_ALL, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_m, "figA_jailbreaks_models", categories=R.APPENDIX_CATEGORIES); plt.show()
wd = R.weird_frame(op_all)
R.weird_figure(wd, "figW_weird"); plt.show()
""")
'''

FIGS = [("Fig. 4", "Multi-query and multi-turn", "fig4_multi_turn"), ("Fig. 5", "Petri", "fig5_misalignment"),
        ("Fig. 10", "Coefficient -1", "figA_qwen_control_alpha_minus1"), ("Fig. 11", "Control-pool stopping", "figA_qwen_ctlstop_multi_turn"),
        ("Fig. 12", "Published against control-pool", "figA_qwen_ctlstop_rule_comparison"), ("Fig. 13", "Jailbreaks, all models", "figA_jailbreaks_models")]


def _row(eval_id, obj, dy, half, rubric=SR, seed="s1", model=Q, anchor=50.0):
    """One delta row, in the registry's orientation (a defense gain is a negative d_y on an ASR eval)."""
    return dict(model_id=model, object_label=obj, seed=seed, eval_id=eval_id, d_y=dy, d_lo=dy - half, d_hi=dy + half, y=anchor + dy,
                anchor_value=anchor, anchor_in_frame=True, ceiling=False, judge_rubric_name=rubric, run_id=f"c:{eval_id}:{obj}:{seed}")


def _mt(obj, dy, half, **kw):
    return [_row("crescendo", obj, dy, half, **kw), _row("actorattack", obj, dy, half, **kw)]


def frames(side: str) -> dict:
    """One render's dump. Under the filed judges ("a") Qwen's control ReLU steer reduces multi-turn ASR significantly (+4 [+2.6, +5.4]
    after orientation) and not under the other ("b"); at coefficient -1 the reverse, with other values; the control-pool frames carry
    values of their own (Fig. 11: 20, Fig. 12 published: 10, control-pool: 30), the control-pool rows naming no rubric as the sidecar's
    do; Petri carries its six behaviours and the pooled score."""
    a = side == "a"
    d = pd.DataFrame(_mt("control_relu", -4 if a else -1, 2 if a else 3) + _mt("difference_lora", -30, 3) + _mt("control_lora", -2, 4)
                     + [_row(b, "difference_lora", -0.5 if a else -0.4, 0.4 if a else 0.5, rubric=PETRI, anchor=5.0) for b in R.PETRI_BEHAVIORS]
                     + [_row("petri_pooled", "difference_lora", -0.3 if a else -0.4, 0.25 if a else 0.35, rubric=PETRI, anchor=5.0)])
    neg_d = pd.DataFrame(_mt("control_relu", -1 if a else -6, 3 if a else 2) + _mt("control_lora", -1, 3))
    ctl_d = pd.DataFrame(_mt("control_relu", -20 if a else -21, 2) + _mt("difference_lora", -30, 3))
    ctl_pub_d = pd.DataFrame(_mt("control_relu", -10 if a else -11, 2) + _mt("difference_lora", -30, 3))
    ctl_new_d = pd.DataFrame(_mt("control_relu", -30 if a else -31, 2, rubric=np.nan) + _mt("difference_lora", -32, 3, rubric=np.nan))
    d_all = pd.concat([d, pd.DataFrame(_mt("control_relu", -5, 2, model="gemma4e4b_it"))], ignore_index=True)
    cats = R.APPENDIX_CATEGORIES
    out = dict(d=d, cm=R.category_means(d, categories=cats), neg_d=neg_d, neg_cm=R.category_means(neg_d, categories=cats),
               ctl_d=ctl_d, ctl_cm=R.category_means(ctl_d, categories=cats), ctl_pub_d=ctl_pub_d, ctl_new_d=ctl_new_d,
               ctl_pub_cm=R.category_means(ctl_pub_d, categories=cats), ctl_new_cm=R.category_means(ctl_new_d, categories=cats),
               d_all=d_all, cm_all=R.category_means(d_all, categories=cats), HEAD=1000,
               JUDGES="primary" if a else "rejudge", SOURCE="snapshot", JUDGE_REPORT={}, PETRI_JUDGE_REPORT={})
    if not a:
        out["JUDGE_REPORT"] = {"frames": [dict(spec="rejudge", threshold=0.5, judges={"j1": "minimax-m3 (strongreject)"},
                                               item_heads={"j1": 990}, frame_head=1000)]}
        out["PETRI_JUDGE_REPORT"] = {"frames": [dict(judge="j2", judge_model="glm-5.3", item_head=1005, frame_head=1000)]}
        out["SETTINGS"] = dict(JUDGES="rejudge", SR_THRESHOLD=0.5, PETRI_JUDGE=None)
    return out


def write_render(root: pathlib.Path, name: str, fr: dict) -> None:
    p = root / name
    (p / "figures").mkdir(parents=True)
    with open(p / "frames.pkl", "wb") as f:
        pickle.dump(fr, f)
    (p / "registry_qwen_figures.executed.ipynb").write_text("{}")
    (p / "render.json").write_text(json.dumps(dict(name=name, env={"REGISTRY_NB_JUDGES": fr["JUDGES"]})))


def run_compare(tmp_path, monkeypatch, figs=FIGS, drop=(), extra=None, pngs=()):
    _fake_nb(monkeypatch, {})
    root = tmp_path / "renders"
    fa, fb = frames("a"), frames("b")
    for k in drop:
        fa.pop(k), fb.pop(k)
    fa.update(extra or {})
    fb.update(extra or {})
    write_render(root, "base", fa)
    write_render(root, "other", fb)
    for side, name in pngs:                                     # figure files a render wrote, by (render, stem)
        from PIL import Image
        Image.new("RGB", (4, 4), "white").save(root / side / "figures" / f"{name}.png")
    rd, gen = tmp_path / "README.md", tmp_path / "gen.py"
    rd.write_text(readme(figs))
    gen.write_text(GEN)
    args = argparse.Namespace(renders=str(root), base="base", other="other", figs=str(tmp_path / "figs"), readme=str(rd), generator=str(gen))
    assert JA.compare(args) == 0
    return json.loads((tmp_path / "figs" / "compare.json").read_text())


def mean_mark(cat_rec: dict, obj: str, judge: str) -> float:
    return next(m["y"] for m in cat_rec["marks"] if m["object_label"] == obj and m["judge"] == judge)


# ---- the derived map, on the committed README and generator -------------------------------------------------------------------------
def test_committed_map_places_every_readme_figure_and_fig10_draws_the_negative_dose_frames():
    text = (NOTEBOOKS / "README.md").read_text()
    fmap = JC.figure_map()
    listed = re.findall(r"^\|\s*(Fig\.\s*\d+|Notebook only)\s*\|[^|\n]*\|\s*`([A-Za-z0-9_]+)`\s*\|\s*$", text, flags=re.M)
    assert [n for _, n in listed] == list(fmap)                        # every figure row, in the README's order
    neg = fmap["figA_qwen_control_alpha_minus1"]
    assert neg["paper"] == dict((n, p) for p, n in listed)["figA_qwen_control_alpha_minus1"]
    assert set(neg["frames"]) == {"neg_cm", "neg_d"}
    assert neg["layout_categories"] == ["static", "multi-turn", "over-refusal"]


def test_committed_map_gives_fig12_its_own_frames():
    fmap = JC.figure_map()
    assert fmap["figA_qwen_ctlstop_rule_comparison"]["frames"] == ["ctl_pub_d", "ctl_all_new", "ctl_pub_cm", "ctl_new_cm"]
    assert fmap["figA_qwen_ctlstop_multi_turn"]["frames"] == ["ctl_cm", "ctl_d"]
    assert fmap["figA_qwen_ctlstop_static"]["frames"] == ["ctl_st_cm", "ctl_st_pts"]        # the rows the static mirror draws
    assert fmap["figA_qwen_ctlstop_capabilities"]["frames"] == ["ctl_cap_cm", "ctl_cap_d"]
    assert fmap["fig3_single_turn_padding"]["hook_frames"] == ["pad_pts"]           # the padding panel's points, read by its hook
    assert set(fmap["fig7_frontier"]["frames"]) == {"pts_st", "pts_mt"}               # through the frontier's dict(pts=...) panels
    assert fmap["figA_qwen_si_target_battery"]["frames"] == ["bt_d", "bt_cm"]        # its gap table (sitarget_battery_gaps) is not a reading
    assert all("gaps" not in f for e in fmap.values() for f in e["frames"])          # a gap table is not a reading


def test_dump_cell_dumps_every_frame_the_map_draws(tmp_path, monkeypatch):
    fmap, parents = JC.figure_map(), JC.lineage()
    drawn = {f for e in fmap.values() for f in e["frames"] + e["hook_frames"]}
    g = {n: pd.DataFrame(dict(eval_id=["crescendo"], judge_rubric_name=[SR])) for n in drawn}
    g["frontier"] = pd.DataFrame(dict(eval_id=["xstest"], judge_rubric_name=["xstest_cpi"]))
    g.update(pd=pd, np=np, os=os, HEAD=7, R=types.SimpleNamespace(JUDGE_REPORT={}, PETRI_JUDGE_REPORT={}, JUDGES="primary", SOURCE="live",
                                                                  SR_THRESHOLD=0.5, SR_PAPER_THRESHOLD=0.5, PETRI_JUDGE=None, PETRI_EXCLUDE="",
                                                                  pooled_ci_state=lambda: {"unit_rows": {}, "units": {}, "unavailable": {}}))
    monkeypatch.setenv("JUDGE_COMPARE_DUMP", str(tmp_path / "frames.pkl"))
    exec(JC.dump_cell(fmap, parents), g)
    dumped = pickle.loads((tmp_path / "frames.pkl").read_bytes())
    assert drawn <= set(dumped) and {"neg_d", "neg_cm", "ctl_pub_d", "ctl_all_new", "ctl_pub_cm", "ctl_new_cm", "_cs", "_cs_check"} <= set(dumped)
    assert dumped["FRAME_RUBRICS"]["frontier"] == ["xstest_cpi"]                     # upstream frames: their rubrics, not their rows
    assert set(dumped["POOLED_CI_STATE"]) == {"unit_rows", "units", "unavailable"}   # the render's units, for its pooled cells
    assert "frontier" not in dumped
    assert json.loads((tmp_path / "settings.json").read_text())["SR_THRESHOLD"] == 0.5


# ---- the derived map, on synthetic tables: numbers and refusals ---------------------------------------------------------------------
def test_numbers_come_from_the_readme_map():
    shifted = [("Fig. 21", "Coefficient -1", "figA_qwen_control_alpha_minus1"), ("Fig. 4", "Multi-turn", "fig4_multi_turn"),
               ("Figure 17", "Jailbreaks", "figA_jailbreaks_models")]
    fmap = JC.figure_map(readme(shifted), GEN)
    assert [(e["paper"], n) for n, e in fmap.items()] == [(p, n) for p, _, n in shifted]
    assert fmap["figA_jailbreaks_models"]["what"] == "Jailbreaks"


def test_a_notebook_only_row_is_compared_and_a_row_that_says_more_is_left_out():
    """`Notebook only` alone marks a figure the paper does not print, compared like a numbered one; a first cell that says more than a
    figure number or `Notebook only` (the data-file figures, drawn with their own judge or none) is left out of the map."""
    rows = [("Notebook only", "Coefficient -1", "figA_qwen_control_alpha_minus1"), ("Fig. 4", "Multi-turn", "fig4_multi_turn"),
            ("Fig. 9, top (`fig:app_evenodd`)", "Sign decomposition", "figA_control_evenodd"),
            ("Notebook only (App. D.3 quotes its numbers)", "Forward passes", "figA_displacement")]
    fmap = JC.figure_map(readme(rows), GEN)
    assert [(e["paper"], n) for n, e in fmap.items()] == [("Notebook only", "figA_qwen_control_alpha_minus1"), ("Fig. 4", "fig4_multi_turn")]


def test_a_readme_figure_the_generator_does_not_draw_refuses():
    with pytest.raises(JC.FigureMapError, match="figZ_never_drawn"):
        JC.figure_map(readme(FIGS + [("Fig. 99", "Not drawn", "figZ_never_drawn")]), GEN)
    with pytest.raises(JC.FigureMapError, match="names no single figure"):
        JC.readme_figures(readme(FIGS).replace("`fig4_multi_turn`", "fig4_multi_turn and fig5"))


# ---- the comparison on synthetic renders -----------------------------------------------------------------------------------------
def test_compare_renumbers_from_the_readme_and_covers_fig10(tmp_path, monkeypatch):
    figs = [(p.replace("Fig. 1", "Fig. 3") if p.startswith("Fig. 1") else p, w, n) for p, w, n in FIGS]     # 10 -> 30, 11 -> 31, ...
    res = run_compare(tmp_path, monkeypatch, figs=figs)
    assert [res["figures"][n]["paper"] for _, _, n in figs] == [p for p, _, _ in figs]
    neg = res["figures"]["figA_qwen_control_alpha_minus1"]
    assert neg["paper"] == "Fig. 30" and neg["compared"] and neg["llm_judged"] is True
    mt = neg["categories"]["multi-turn"]
    assert mean_mark(mt, "control_relu", "a") == pytest.approx(1.0) and mean_mark(mt, "control_relu", "b") == pytest.approx(6.0)
    assert [(v["object"], v["verdict_a"], v["verdict_b"]) for v in mt["mean"]["verdict_changes"]] == [("control_relu", "0", "+")]


def test_compare_reads_fig12_from_its_own_frames(tmp_path, monkeypatch):
    res = run_compare(tmp_path, monkeypatch)
    f11, f12 = res["figures"]["figA_qwen_ctlstop_multi_turn"], res["figures"]["figA_qwen_ctlstop_rule_comparison"]
    assert mean_mark(f11["categories"]["multi-turn"], "control_relu", "a") == pytest.approx(20.0)
    blocks = list((f12.get("categories") or {}).values()) + [c for p in (f12.get("panels") or {}).values() for c in p["categories"].values()]
    drawn = sorted(round(m["y"], 6) for c in blocks for m in c["marks"] if m["object_label"] == "control_relu" and m["judge"] == "a")
    assert drawn == [10.0, 30.0]                         # published and control-pool, never Fig. 11's 20
    pub, new = f12["panels"]["ctl_pub_d"], f12["panels"]["ctl_new_d"]
    assert pub["means_frame"] == "ctl_pub_cm" and new["means_frame"] == "ctl_new_cm"
    assert mean_mark(pub["categories"]["multi-turn"], "control_relu", "a") == pytest.approx(10.0)
    assert mean_mark(new["categories"]["multi-turn"], "control_relu", "b") == pytest.approx(31.0)   # sidecar rows: no rubric, judged eval


def test_a_frame_the_render_did_not_dump_is_marked_not_compared(tmp_path, monkeypatch):
    res = run_compare(tmp_path, monkeypatch, drop=("neg_d", "neg_cm"))
    neg = res["figures"]["figA_qwen_control_alpha_minus1"]
    assert neg["compared"] is False and neg["not_compared"].startswith("not compared: frame not dumped")
    assert set(neg["frames_not_dumped"]) == {"neg_d", "neg_cm"}
    assert res["figures"]["fig4_multi_turn"]["compared"] is True


def test_a_judged_figure_with_no_comparable_frame_refuses(tmp_path, monkeypatch):
    weird = pd.DataFrame(dict(eval_id=["crescendo"], judge_rubric_name=[SR], value=[0.3]))
    with pytest.raises(JC.FigureMapError, match="figW_weird"):
        run_compare(tmp_path, monkeypatch, figs=FIGS + [("Fig. 40", "Weird", "figW_weird")], extra={"wd": weird})


def test_petri_pooled_column_is_the_mean_column_never_an_eval_mark(tmp_path, monkeypatch):
    marks = JA.eval_marks(frames("a")["d"], "Petri")
    assert set(marks.column) == set(R.PETRI_BEHAVIORS)
    res = run_compare(tmp_path, monkeypatch)
    petri = res["figures"]["fig5_misalignment"]["categories"]["Petri"]
    assert petri["evals"]["n_marks"] == len(R.PETRI_BEHAVIORS)
    assert all(v["column"] != R.POOLED_EVAL["Petri"] for v in petri["evals"]["verdict_changes"])
    assert mean_mark(petri, "difference_lora", "a") == pytest.approx(0.3)        # the pooled score, read once, in the mean column


def test_verdict_changes_are_counted_once_per_distinct_mark(tmp_path, monkeypatch):
    res = run_compare(tmp_path, monkeypatch)
    t = res["verdict_change_totals"]
    figs = {(c["object"], round(c["clearance"], 2)): c["figures"] for c in t["changes"]}
    # Qwen's control ReLU multi-turn mark under Fig. 4 and Fig. 13 is one change; Fig. 10's, at coefficient -1, is another
    assert t["entries"] == 3 and t["distinct"] == 2
    assert figs[("control_relu", 2.59)] == ["Fig. 4", "Fig. 13"]
    assert figs[("control_relu", round(6 - 1.96 * np.sqrt(2) * (4 / 3.92) / 2, 2))] == ["Fig. 10"]
    assert t["clearance_bands"]["pp"] == {"<= 1": 0, "1 to 3": 1, "> 3": 1, "?": 0}


def test_a_notebook_only_figure_is_named_by_its_file_in_the_distinct_marks():
    """Several figure-map rows read `Notebook only`, so a distinct mark names such a figure by its file, never by that shared word; a
    figure with a paper number keeps it."""
    assert JA.figure_label({"paper": "Fig. 4"}, "fig4_multi_turn") == "Fig. 4"
    assert JA.figure_label({"paper": JC.NOTEBOOK_ONLY}, "figA_qwen_ctlstop_multi_turn") == "figA_qwen_ctlstop_multi_turn"
    assert JA.figure_label({}, "figX") == "figX"


def test_compare_records_each_renders_policy_threshold_and_heads(tmp_path, monkeypatch):
    res = run_compare(tmp_path, monkeypatch)
    base, other = res["renders"]["base"], res["renders"]["other"]
    assert base["judges"] == "primary" and base["threshold"] == "not recorded by this render"
    assert base["item_heads"].startswith("none read")
    assert other["judges"] == "rejudge" and other["threshold"] == 0.5 and other["head"] == 1000
    assert other["item_heads"] == {"j1": {"judge": "minimax-m3 (strongreject)", "heads": [990]}}
    assert other["petri_item_heads"][0]["item_head"] == 1005
    assert other["items_older_than_frames"] == [dict(judge="j1", judge_name="minimax-m3 (strongreject)", item_head=990, frame_head=1000)]
    assert other["render"]["env"]["REGISTRY_NB_JUDGES"] == "rejudge"


# ---- render provenance ----------------------------------------------------------------------------------------------------------
def _render_args(out, **kw):
    a = dict(name="r1", out=str(out), source="live", judges="rejudge", reuse_cache=False, threshold=None, petri_judge=None, petri_exclude=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_render_env_clears_the_inherited_variables_and_records_the_strength_reads(tmp_path):
    f1, f2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    f1.write_bytes(b"one")
    f2.write_bytes(b"two")
    inherited = {"REGISTRY_NB_SR_THRESHOLD": "0.75", "REGISTRY_NB_PETRI_JUDGE": "glm-5.3", "REGISTRY_NB_PETRI_EXCLUDE": "/x.tsv",
                 "NB_REUSE_CACHE": "1", "REGISTRY_NB_STRENGTH_READS": f"{f1}{os.pathsep}{f2}"}
    env, cleared, prov = JC.render_env(_render_args(tmp_path), str(tmp_path / "r1"), inherited)
    assert set(cleared) == {"REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_PETRI_EXCLUDE", "NB_REUSE_CACHE"}
    assert not set(cleared) & set(env) and env["REGISTRY_NB_JUDGES"] == "rejudge"
    assert prov["NB_REUSE_CACHE"] is False and prov["REGISTRY_NB_SR_THRESHOLD"].startswith("unset")
    assert prov["REGISTRY_NB_STRENGTH_READS"] == [dict(path=str(f1), sha256=hashlib.sha256(b"one").hexdigest()),
                                                  dict(path=str(f2), sha256=hashlib.sha256(b"two").hexdigest())]
    env, cleared, prov = JC.render_env(_render_args(tmp_path, threshold=0.75, petri_judge="glm-5.3", reuse_cache=True), str(tmp_path / "r1"), inherited)
    assert env["REGISTRY_NB_SR_THRESHOLD"] == "0.75" and env["REGISTRY_NB_PETRI_JUDGE"] == "glm-5.3" and env["NB_REUSE_CACHE"] == "1"
    assert set(cleared) == {"REGISTRY_NB_PETRI_EXCLUDE"} and prov["REGISTRY_NB_SR_THRESHOLD"] == "0.75" and prov["NB_REUSE_CACHE"] is True


def test_render_runs_the_kernel_without_the_inherited_variables_and_writes_provenance(tmp_path, monkeypatch, env_restored):
    seen = {}
    _fake_nb(monkeypatch, seen)
    reads = tmp_path / "reads.parquet"
    reads.write_bytes(b"reads")
    for k, v in {"REGISTRY_NB_SR_THRESHOLD": "0.75", "REGISTRY_NB_PETRI_JUDGE": "glm-5.3", "REGISTRY_NB_PETRI_EXCLUDE": "/x.tsv",
                 "NB_REUSE_CACHE": "1", "REGISTRY_NB_STRENGTH_READS": str(reads)}.items():
        monkeypatch.setenv(k, v)
    JC.render(_render_args(tmp_path))
    leaked = [k for k in ("REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_PETRI_EXCLUDE", "NB_REUSE_CACHE") if k in seen["env"]]
    assert leaked == []                                  # (the kernel's whole environment is never printed: it holds credentials)
    assert seen["env"]["REGISTRY_NB_JUDGES"] == "rejudge"
    meta = json.loads((tmp_path / "r1" / "render.json").read_text())
    prov = meta["provenance"]
    assert prov["REGISTRY_NB_JUDGES"] == "rejudge" and prov["NB_REUSE_CACHE"] is False
    assert prov["REGISTRY_NB_STRENGTH_READS"] == [dict(path=str(reads), sha256=hashlib.sha256(b"reads").hexdigest())]
    assert meta["cleared"]["REGISTRY_NB_SR_THRESHOLD"] == "0.75"
    assert "neg_d" in seen["cells"][-1] and "FRAME_RUBRICS" in seen["cells"][-1]


def test_petri_render_pins_the_primary_policy(tmp_path, monkeypatch, env_restored):
    seen = {}
    _fake_nb(monkeypatch, seen)
    monkeypatch.setenv("REGISTRY_NB_JUDGES", "rejudge")
    monkeypatch.setenv("REGISTRY_NB_SR_THRESHOLD", "0.75")
    monkeypatch.setenv("NB_REUSE_CACHE", "1")
    PJC.render(argparse.Namespace(name="opus_live", out=str(tmp_path), source="live", judge="primary", exclude="", reuse_cache=False))
    assert seen["env"]["REGISTRY_NB_JUDGES"] == "primary"
    assert "REGISTRY_NB_SR_THRESHOLD" not in seen["env"] and "NB_REUSE_CACHE" not in seen["env"]     # cleared, as judge_compare clears them
    meta = json.loads((tmp_path / "opus_live" / "render.json").read_text())
    assert meta["env"]["REGISTRY_NB_JUDGES"] == "primary" and meta["cleared"] == {"REGISTRY_NB_SR_THRESHOLD": "0.75", "NB_REUSE_CACHE": "1"}
    env, cleared = PJC.render_env(argparse.Namespace(source="live", judge="glm-5.3", exclude="", reuse_cache=True), str(tmp_path / "x"),
                                  {"REGISTRY_NB_SR_THRESHOLD": "0.75", "NB_REUSE_CACHE": "1"})
    assert env["NB_REUSE_CACHE"] == "1" and cleared == {"REGISTRY_NB_SR_THRESHOLD": "0.75"}         # --reuse-cache passes it on


def test_compare_names_the_figure_files_the_readme_map_leaves_out(tmp_path, monkeypatch):
    res = run_compare(tmp_path, monkeypatch, pngs=[("base", "figA_x_delta_models"), ("other", "figA_y_absolute_models"),
                                                   ("other", FIGS[0][2])])
    assert res["figures_not_in_map"] == ["figA_x_delta_models", "figA_y_absolute_models"]          # a mapped figure is not listed
    assert "`figA_x_delta_models`" in (tmp_path / "figs" / "compare.md").read_text()
    assert JC.figures_not_in_map([str(tmp_path / "nowhere")], {}) == []


# ---- no module forces a registry source at import (the landing gate's host has no live registry) ---------------------------------------
BRANCH_MODULES = ("judge_compare.py", "judge_analysis.py", "judge_rules.py", "judge_rules_check.py", "judge_rules_bonstores_check.py",
                  "petri_judge_compare.py", "petri_judge_analysis.py")


def _module_level_env_writes(path: pathlib.Path) -> list:
    """The module-level statements of `path` that write os.environ (setdefault / update / pop / item assignment), never a function body."""
    import ast
    hits = []
    for s in ast.parse(path.read_text()).body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for n in ast.walk(s):
            target = n.func.value if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
                "setdefault", "update", "pop", "__setitem__") else None
            if isinstance(n, (ast.Assign, ast.AugAssign)):
                target = next((t.value for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Subscript)), None)
            if isinstance(target, ast.Attribute) and target.attr == "environ":
                hits.append(f"{path.name}:{n.lineno}")
    return hits


def test_no_comparison_module_writes_the_environment_at_import():
    assert [h for m in BRANCH_MODULES for h in _module_level_env_writes(NOTEBOOKS / m)] == []


def test_the_comparison_modules_import_on_a_host_with_no_registry(tmp_path):
    """The gate's condition: an empty HOME, no REGISTRY_ROOT, no REGISTRY_NB_SOURCE. Every comparison module imports, and the notebook
    module resolves its own default (the committed snapshot here), never a forced `live`."""
    import subprocess
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REGISTRY_", "NB_", "JUDGE_RULES_"))}
    env.update(HOME=str(tmp_path), MPLBACKEND="Agg")
    code = ("import judge_compare, judge_analysis, judge_rules, judge_rules_check, judge_rules_bonstores_check, petri_judge_compare, "
            "petri_judge_analysis, registry_nb as R; print('SOURCE=' + R.SOURCE)")
    out = subprocess.run([sys.executable, "-c", code], cwd=NOTEBOOKS, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "SOURCE=snapshot" in out.stdout


def test_the_petri_item_analysis_refuses_any_source_but_live(monkeypatch, tmp_path):
    import petri_judge_analysis as PJA
    monkeypatch.setattr(PJA.R, "SOURCE", "snapshot")
    with pytest.raises(RuntimeError, match="live registry"):
        PJA.item_frame(str(tmp_path / "served.tsv"))
