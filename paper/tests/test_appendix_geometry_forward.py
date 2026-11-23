"""appendix_geometry_forward: two judge-free geometry figures of the control-directions appendix (figA_refusal_enrichment, Figure 9's
lower half, and figA_displacement, drawn in the notebook only), drawn from the committed sidecar `data/appendix_geometry_forward.json`. No registry, no lane files: the
committed sidecar and synthetic mutations of it.

Pinned here:
  * the sidecar's contract: it loads; its stamp says judge-free; its provenance names the builder and every source file with a sha256
    and carries no absolute or home path; its 28 objects are distinct sha256 pins covering every role; and `geometry_forward_problems`
    names each way a sidecar can be broken (schema, judge stamp, a missing role, a wrong registry claim, a pooled KL that is not its
    seeds' mean, a hole in the projection table, a leaked path), which `load_geometry_forward` turns into a refusal;
  * the numbers the appendix text quotes, recomputed from the sidecar (the KL of the difference objects against theta_C, the
    refusal-projection drops at block 28's last template token and the sign-dependent parts over layers 15-30, the gate correlation
    and kappa_C, the enrichments, the signed cosines, the LoRA Frobenius cosines), so a rebuilt sidecar cannot drift from the text;
  * the canvas: every mark on every panel of both figures sits at the sidecar's value for it (scatter offsets, line data, interval
    segments, band rectangles), -theta_K and the -0.5 refusal steer are drawn open in their object's colour, the panel titles carry the
    text-set sizes and the unsteered projection levels, and `site=` moves panel (b); the refusal enrichment is one panel of two axes with no
    letter on the manuscript's canvas, its marks and grey bands the sidecar's, each mark in its object's colour and its family's shape
    (that its PNG is the manuscript's pixel for pixel is test_appendix_geometry_style's);
  * `registry_check` flags an unknown id, a wrong label and a wrong parent by name; both figures save png, pdf and svg into R.FIGS and
    every APPENDIX_FIGS_MIRROR directory; the module imports no torch (the notebook's environment has none)."""
import copy
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
NOTEBOOKS = REPO / "paper" / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import registry_nb as R  # noqa: E402
import appendix_common as AC  # noqa: E402
import appendix_geometry_forward as GF  # noqa: E402


@pytest.fixture(scope="module")
def gf():
    return GF.load_geometry_forward()


@pytest.fixture(autouse=True)
def _no_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIGS", str(tmp_path / "figs"))
    monkeypatch.delenv(AC.MIRROR_ENV, raising=False)
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def figs(gf, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("geomfigs"))
    fd = GF.displacement_figure(gf, out_dirs=[out])
    fr = GF.refusal_enrichment_figure(gf, out_dirs=[out])
    yield ([ax for sf in fd.subfigs for ax in sf.axes], list(fr.axes), out)
    plt.close(fd)
    plt.close(fr)


def _same(got, want, tol=1e-12):
    got, want = np.sort(np.asarray(got, float).ravel()), np.sort(np.asarray(want, float).ravel())
    assert got.shape == want.shape, (got.shape, want.shape)
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


def _scatter(ax):
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def _ys(ax):
    return np.concatenate([c.get_offsets()[:, 1] for c in _scatter(ax)])


def _segs(ax):
    return np.array([[s[:, 1].min(), s[:, 1].max()] for c in ax.collections if isinstance(c, LineCollection) for s in c.get_segments()])


def _lines(ax, n):
    return np.concatenate([np.asarray(l.get_ydata(), float) for l in ax.get_lines() if len(np.atleast_1d(l.get_ydata())) == n])


def _title(ax):
    return ax.get_title(loc="left")


# ---- 1. the sidecar's contract ---------------------------------------------------------------------------------------------------
def test_the_committed_sidecar_loads_is_stamped_judge_free_and_names_its_sources(gf):
    assert gf["schema"] == GF.SCHEMA and gf["judge"]["model"] == "none"
    pv = gf["provenance"]
    assert pv["script"] == GF.BUILDER and len(pv["script_sha256"]) == 64
    names = [f["name"] for f in pv["source_files"]]
    for need in ("c2_kl.csv", "c2_kl_per_eval.csv", "c3_projection_summary.csv", "c1_relu_layers.csv", "c1_lora_layers.csv", "refusal_directions_resid_post.pt",
                 "b2_summary.csv", "b2_relu_by_layer.csv", "b1_pairs_by_layer.csv", "b1_nulls.json", "shared/T1/shard_000.pt"):
        assert any(n.endswith(need) for n in names), need
    assert all(len(f["sha256"]) == 64 for f in pv["source_files"])
    assert not [s for s in GF._strings(gf) if s.startswith(("/", "~")) or "/home/" in s]
    assert pv["gpu_run"]["name"] == "geom-fwd-0074823" and len(pv["gpu_run"]["sanity_checks_passed"]) == 18
    assert GF.geometry_forward_problems(gf) == []


def test_every_object_is_a_distinct_sha256_pin_and_every_role_is_present(gf):
    objs = gf["objects"]
    ids = [o["object_id"] for o in objs]
    assert len(ids) == len(set(ids)) == 28
    roles = {(o["family"], o["role"], o["seed"]) for o in objs}
    for fam in GF.FAMILIES:
        for s in GF.SEEDS:
            assert {(fam, r, s) for r in ("C", "K", "mK", "V")} <= roles
    assert {("arditi", "pA", ""), ("arditi", "mA", ""), ("relu", "S", ""), ("lora", "S", "")} <= roles
    by = {(o["family"], o["role"], o["seed"]): o for o in objs}
    for fam in GF.FAMILIES:                                          # the derived objects name this seed's parents
        for s in GF.SEEDS:
            assert by[(fam, "mK", s)]["registry"]["parents"] == [by[(fam, "K", s)]["object_id"]]
            assert by[(fam, "V", s)]["registry"]["parents"] == [by[(fam, "C", s)]["object_id"], by[(fam, "K", s)]["object_id"]]
    assert by[("arditi", "mA", "")]["registry"]["parents"] == [by[("arditi", "pA", "")]["object_id"]]


def _drop_role(sc):
    sc["objects"] = [o for o in sc["objects"] if not (o["family"] == "lora" and o["role"] == "V" and o["seed"] == "2")]


def _wrong_label(sc):
    next(o for o in sc["objects"] if o["role"] == "K")["registry"]["paper_label"] = "constitution_relu"


def _pooled_kl(sc):
    rows = sc["displacement"]["kl"]["rows"]
    next(r for r in rows if r[:4] == ["relu", "T2", "V", "pooled"])[4] += 0.01


def _projection_hole(sc):
    sc["displacement"]["projection"]["delta"]["rows"].pop(17)


def _path_leak(sc):
    sc["provenance"]["source_files"][0]["name"] = "/home/someone/c2_kl.csv"


@pytest.mark.parametrize("mutate, message", [
    (lambda sc: sc.update(schema="other/1"), "schema"),
    (lambda sc: sc.update(judge={"model": "minimax-m3"}), "judge stamp"),
    (_drop_role, "objects: missing"),
    (_wrong_label, "registry claim"),
    (_pooled_kl, "not the mean of its seeds"),
    (_projection_hole, "projection.delta"),
    (_path_leak, "absolute or home paths"),
])
def test_problems_are_named_and_a_broken_file_is_refused(gf, tmp_path, mutate, message):
    sc = copy.deepcopy(gf)
    mutate(sc)
    probs = GF.geometry_forward_problems(sc)
    assert any(message in p for p in probs), probs
    p = tmp_path / "broken.json"
    p.write_text(json.dumps(sc))
    with pytest.raises(GF.GeometryForwardSidecarError, match="broken.json"):
        GF.load_geometry_forward(str(p))


# ---- 2. the numbers the text quotes ------------------------------------------------------------------------------------------------
def test_the_numbers_the_appendix_text_quotes(gf):
    t = GF.text_numbers(gf).set_index("key").value
    assert round(t["kl_relu_T2_V"], 2) == 0.46 and round(t["kl_relu_T2_C"], 2) == 0.75
    assert round(t["kl_lora_T2_V"], 2) == 0.52 and round(t["kl_lora_T2_C"], 2) == 0.75
    assert t["kl_V_below_C_cells"] == "12/12"
    assert t["kl_V_below_C_eval_cells"] == "36/36"                                  # "in every seed and evaluation"
    v, c, _ = t["first_token_kl_relu"]
    assert (round(v, 1), round(c, 1)) == (3.0, 2.0)                                 # "further away (3.0 against 2.0 nats)"
    assert t["kl_T1_refusal_steer"] == "0.039-0.053" and t["kl_T1_objects"] == "0.91-1.64"
    lo, hi = t["odd_frac_lora_seed_mean_range"]
    assert (round(lo, 2), round(hi, 2)) == (0.03, 0.12)                             # "+0.03 to +0.12 of the gap for the LoRA"
    per = t["odd_frac_relu_per_seed_range"]
    assert round(per["1"][1], 2) == 0.25 and round(per["3"][0], 2) == -0.31        # "single seeds +0.25 and -0.31"
    lo, hi = t["odd_frac_refusal_steer_range"]
    assert (round(lo, 2), round(hi, 2)) == (0.33, 0.53)
    assert round(t["unsteered_harmful_projection"], 1) == 17.8
    assert tuple(round(v) for v in t["drop_relu_pmK_pct"]) == (43, 78)              # "by 43-78 %"
    assert tuple(round(v) for v in t["drop_relu_C_pct"]) == (20, 35)
    assert tuple(round(v) for v in t["drop_relu_V_pct"]) == (51, 64)
    assert round(t["gate_corr_relu_V"], 2) == 0.28 and round(t["kappa_C_relu_V"], 2) == 0.18 and round(t["kappa_C_lora_V"], 2) == 0.14
    assert t["enrich_pc1_K_range"] == (0.94, 1.05) and (round(t["enrich_pc1_S_range"][0], 1), round(t["enrich_pc1_S_range"][1], 1)) == (2.1, 9.2)
    assert round(t["enrich_pc1_C_max"], 1) == 2.8 and t["enrich_span_K_write"] == 1.35 and t["enrich_span_S_write"] == 2.42
    assert t["haar_p_K_write_span"] <= 0.010 + 1e-12
    assert (round(t["signed_cos_K_pos-5"], 3), round(t["signed_cos_K_pos-1"], 3)) == (0.004, 0.008)
    assert (round(t["signed_cos_S_pos-5"], 3), round(t["signed_cos_S_pos-1"], 3)) == (0.028, 0.027)
    assert t["frobenius_CK_range"] == (0.047, 0.084) and t["n_refusal_directions"] == 279


def test_the_sign_dependent_fraction_is_the_odd_part_over_the_unsteered_gap_written_out(gf):
    d, u = GF.frame(gf, "displacement", "projection", "delta"), GF.frame(gf, "displacement", "projection", "unsteered")
    r = d[d.family.eq("relu") & d.seed.eq("1") & d["eval"].eq("arditi_harmful_val") & d.position.eq(-5) & d.layer.eq(27)].iloc[0]
    lv = u[u.position.eq(-5) & u.layer.eq(27)].set_index("eval").mean_proj
    want = (r.pK - r.mK) / 2 / (lv["arditi_harmful_val"] - lv["arditi_harmless_val"])
    of = GF.odd_fraction_frame(gf)
    got = of[of.family.eq("relu") & of.seed.eq("1") & of.layer.eq(27)].odd_frac.iat[0]
    assert math.isclose(got, want, rel_tol=1e-12) and round(got, 2) == 0.25
    assert set(of.layer) == set(range(15, 31))


# ---- 3. the canvas ---------------------------------------------------------------------------------------------------------------
def test_displacement_panels_draw_the_sidecars_values(gf, figs):
    A, _, _ = figs
    kl = GF.kl_frame(gf)
    for ax, st, n in zip(A[:2], ("T2", "T3"), (800, 400)):
        _same(_ys(ax), kl[kl.set.eq(st)].mean_kl)
        assert f"({n})" in _title(ax)
    pf = GF.projection_frame(gf)
    for ax, ev, lvl in zip(A[2:4], GF.EVALS, ("+17.8", "−5.8")):
        _same(_ys(ax), pf[pf["eval"].eq(ev)].delta)
        st = pf[pf["eval"].eq(ev) & pf.family.eq("arditi")]
        _same(_segs(ax), np.c_[st.delta - 1.96 * st.se, st.delta + 1.96 * st.se])
        assert _title(ax).endswith(f"unsteered {lvl}") and "(32)" in _title(ax)
    rw, lw = GF.writes_frames(gf)
    _same(_lines(A[4], 32), rw.corr_g)
    _same(_lines(A[5], 32), rw.kappa_C)
    _same(_ys(A[5]), lw.kappa_C)
    _same(_segs(A[5]), lw[["kappa_C_seed_min", "kappa_C_seed_max"]].values)


def test_negated_arms_are_open_in_their_objects_colour_and_filled_arms_are_filled(gf, figs):
    A, _, _ = figs
    pf = GF.projection_frame(gf)
    ax = A[2]
    marks = {}
    for c in _scatter(ax):
        y = round(float(c.get_offsets()[0, 1]), 9)
        marks.setdefault(y, []).append(c)
    for fam in GF.FAMILIES:
        ctl = to_hex(R.COLOR[f"control_{fam}"])
        for arm, open_ in (("pK", False), ("mK", True)):
            y = round(float(pf[pf.family.eq(fam) & pf.seed.eq("mean") & pf["eval"].eq("arditi_harmful_val") & pf.arm.eq(arm)].delta.iat[0]), 9)
            c = marks[y][0]
            face, edge = to_hex(c.get_facecolor()[0]), to_hex(c.get_edgecolor()[0])
            assert (face, edge) == (("#ffffff", ctl) if open_ else (ctl, "#ffffff")), (fam, arm, face, edge)
    for arm, open_ in (("pA", False), ("mA", True)):
        y = round(float(pf[pf.family.eq("arditi") & pf["eval"].eq("arditi_harmful_val") & pf.arm.eq(arm)].delta.iat[0]), 9)
        c = marks[y][0]
        steer = to_hex(R.COLOR["arditi_allpos"])
        assert to_hex((c.get_edgecolor() if open_ else c.get_facecolor())[0]) == steer


def _point_marks(ax):
    """The single-point Line2D marks of an axis (the refusal enrichment draws each mark as its own one-point line)."""
    return [ln for ln in ax.get_lines() if len(np.atleast_1d(ln.get_ydata())) == 1 and ln.get_marker() not in (None, "None", "")]


def test_refusal_enrichment_draws_the_sidecars_values(gf, figs):
    _, B, _ = figs
    en, band = GF.enrichment_frame(gf), GF.enrichment_band(gf)
    for ax, which in zip(B, ("span", "pc1")):
        _same([float(np.atleast_1d(ln.get_ydata())[0]) for ln in _point_marks(ax)], en[en.which.eq(which)].enrichment)
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        b = band[band.which.eq(which)]
        _same([r.get_y() for r in rects], b.lo)
        _same([r.get_y() + r.get_height() for r in rects], b.hi)
        assert ax.get_yscale() == "log"
    pc1 = en[en.which.eq("pc1") & en.seed.eq("mean")]
    assert (round(pc1[pc1.obj.eq("K")].enrichment.min(), 2), round(pc1[pc1.obj.eq("K")].enrichment.max(), 2)) == (0.94, 1.05)
    assert round(pc1[pc1.obj.eq("C") & pc1.component.eq("write")].enrichment.max(), 2) == 2.76     # the paper: enriched 2.8 times


def test_refusal_enrichment_marks_carry_the_objects_colour_and_the_familys_shape(gf, figs):
    """theta_C green, theta_K blue, the safety control yellow (appendix_geometry_style's palette), all filled with a white edge; a
    circle for a ReLU component, a square for the LoRA; a seed's mark smaller than its mean's, ENRICH_SEED_DX to its right."""
    import appendix_geometry_style as GS
    _, B, _ = figs
    en = GF.frame(gf, "refusal_subspace", "enrichment")
    for ax, which in zip(B, ("span", "pc1")):
        marks = _point_marks(ax)
        for i, (fam, comp, _lab) in enumerate(GF.COMPONENTS):
            for obj in ("C", "K", "S"):
                g = en[en.family.eq(fam) & en.component.eq(comp) & en.which.eq(which) & en.obj.eq(obj)]
                x = i + GF.OBJ_DX[obj]
                at = [m for m in marks if np.isclose(float(np.atleast_1d(m.get_xdata())[0]), x)]
                seeds = [m for m in marks if np.isclose(float(np.atleast_1d(m.get_xdata())[0]), x + GF.ENRICH_SEED_DX)]
                assert len(at) == 1 and len(seeds) == (0 if obj == "S" else 3), (which, fam, comp, obj)
                assert np.isclose(float(np.atleast_1d(at[0].get_ydata())[0]), g.enrichment.mean())
                for m in at + seeds:
                    assert m.get_marker() == GS.FAMILY_MARKER[fam]
                    assert to_hex(m.get_markerfacecolor()) == GS.OBJECT_COLOR[obj] and to_hex(m.get_markeredgecolor()) == "#ffffff"
                assert all(m.get_markersize() < at[0].get_markersize() for m in seeds)


def test_refusal_enrichment_is_one_panel_of_two_axes_with_no_letter(figs):
    """The lower half of Figure 9: two axes (the per-layer span, the first PC) on the manuscript's 5.5 x 2.29 in canvas, labelled over
    each axis, no panel letter, no principal-angle or signed-cosine axis."""
    _, B, _ = figs
    assert len(B) == 2
    labels = [[t.get_text() for t in ax.texts] for ax in B]
    assert labels == [["Per-layer refusal span (9 positions)"], ["First refusal PC (all layers, positions)"]]
    root = B[0].get_figure(root=True)
    assert tuple(root.get_size_inches()) == GF.ENRICH_CANVAS_IN
    assert not [t for t in root.texts if len(t.get_text()) == 1]
    assert not [ax for ax in B if "principal" in ax.get_ylabel().lower()]


def test_site_moves_panel_b(gf, tmp_path):
    pf = GF.projection_frame(gf, site=(24, -5))
    d = GF.frame(gf, "displacement", "projection", "delta")
    want = d[d.layer.eq(24) & d.position.eq(-5)]
    assert math.isclose(pf[pf.family.eq("relu") & pf.seed.eq("2") & pf["eval"].eq("arditi_harmless_val") & pf.arm.eq("C")].delta.iat[0],
                        want[want.family.eq("relu") & want.seed.eq("2") & want["eval"].eq("arditi_harmless_val")].C.iat[0])
    fig = GF.displacement_figure(gf, out_dirs=[str(tmp_path)], site=(24, -5))
    ax = [a for sf in fig.subfigs for a in sf.axes][2]
    assert "layer 24, position -5" in ax.get_ylabel()
    _same(_ys(ax), pf[pf["eval"].eq("arditi_harmful_val")].delta)


# ---- 4. registry binding, saving, environment ------------------------------------------------------------------------------------
def _objects_table(gf) -> pd.DataFrame:
    return pd.DataFrame([dict(object_id=o["object_id"], paper_label=o["registry"]["paper_label"], fit_model_id=o["registry"]["fit_model_id"],
                              fit_replicate_label=o["registry"]["fit_replicate_label"], derivation=o["registry"]["derivation"],
                              parent_object_ids_json=json.dumps(o["registry"]["parents"])) for o in gf["objects"]])


def test_registry_check_flags_an_unknown_id_a_wrong_label_and_a_wrong_parent(gf):
    t = _objects_table(gf)
    assert GF.registry_check(gf, t).known.all()
    k = next(o["object_id"] for o in gf["objects"] if o["role"] == "K")
    mk = next(o["object_id"] for o in gf["objects"] if o["role"] == "mK")
    s = next(o["object_id"] for o in gf["objects"] if o["role"] == "S")
    bad = t.copy()
    bad.loc[bad.object_id.eq(k), "paper_label"] = "constitution_relu"
    bad.loc[bad.object_id.eq(mk), "parent_object_ids_json"] = json.dumps(["0" * 64])
    bad = bad[bad.object_id.ne(s)]
    chk = GF.registry_check(gf, bad).set_index("object_id")
    assert chk.loc[k, "mismatch"] == "paper_label" and chk.loc[mk, "mismatch"] == "parents" and chk.loc[s, "mismatch"] == "not in the registry"
    assert int((~chk.known).sum()) == 3


def test_both_figures_save_three_formats_into_figs_and_every_mirror(gf, tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    monkeypatch.setenv(AC.MIRROR_ENV, str(mirror))
    GF.displacement_figure(gf)
    GF.refusal_enrichment_figure(gf)
    for d in (pathlib.Path(R.FIGS), mirror):
        for name in (GF.FIG_DISPLACEMENT, GF.FIG_ENRICHMENT):
            for ext in AC.FORMATS:
                assert (d / f"{name}.{ext}").is_file(), (d, name, ext)


def test_the_module_imports_no_torch():
    src = (NOTEBOOKS / "appendix_geometry_forward.py").read_text()
    assert "import torch" not in src and "from torch" not in src
