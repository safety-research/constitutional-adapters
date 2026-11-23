"""appendix_geometry_overlap: the overlap figure of the control-directions appendix (Figure 10, figA_cK_overlap), drawn from the committed
sidecar `data/appendix_geometry_overlap.json`. No registry, no figure-build files: the committed sidecar, the committed layer-12 cosine
data and synthetic mutations of the sidecar.

Pinned here:
  * the sidecar's contract: it loads; its stamp says judge-free; its provenance names the builder and every source file with a sha256
    and carries no absolute or home path; its 14 objects are distinct sha256 pins, one per family x role x seed, each the object
    appendix_geometry_forward pins for the same role and seed; and `geometry_overlap_problems` names each way a sidecar can be broken
    (schema, judge stamp, a missing object, a wrong registry claim, a missing pair, a share outside [0, 1], a null out of order, a
    layer-12 cosine that is not the cosine data's, a leaked path), which `load_geometry_overlap` turns into a refusal;
  * the numbers the appendix text quotes from Figure 10, recomputed from the sidecar (14% and 19% of the updates in the other's output
    space, 8.5-27.5% per layer, 0.16% for random subspaces, 0.1-0.5% for the unrelated LoRA, 47-68% between seeds, a write cosine of
    0.64 and an unrelated steer's of at most 0.07);
  * the canvas: every marker, bar, line, shading and band of both panels sits at the sidecar's value for it; theta_C's shares are filled
    and solid, theta_K's open and dashed; the safety LoRA's share in theta_C's or theta_K's span is in the table, not the figure; panel
    letters a and b on the manuscript's 5.5 x 4.44 in canvas (that its PNG is the manuscript's pixel for pixel is
    test_appendix_geometry_style's);
  * `registry_check` flags an unknown id and a wrong label by name; the figure saves png, pdf and svg into R.FIGS and every
    APPENDIX_FIGS_MIRROR directory; the module imports no torch (the notebook's environment has none)."""
import copy
import json
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
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

import registry_nb as R  # noqa: E402
import appendix_common as AC  # noqa: E402
import appendix_geometry_forward as GF  # noqa: E402
import appendix_geometry_overlap as GO  # noqa: E402


@pytest.fixture(scope="module")
def go():
    return GO.load_geometry_overlap()


@pytest.fixture(autouse=True)
def _no_repo_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIGS", str(tmp_path / "figs"))
    monkeypatch.delenv(AC.MIRROR_ENV, raising=False)
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def fig(go, tmp_path_factory):
    f = GO.overlap_figure(go, out_dirs=[str(tmp_path_factory.mktemp("overlap"))])
    yield f
    plt.close(f)


def _axes(fig):
    return list(fig.axes)


def _marks(ax):
    """Panel (a)'s mean markers: the marker-only lines (each series' markers are one Line2D, its connecting line another)."""
    return [ln for ln in ax.get_lines() if ln.get_linestyle() in ("None", "none", "") and ln.get_marker() not in (None, "None", "")]


def _same(got, want, tol=1e-12):
    got, want = np.sort(np.asarray(got, float).ravel()), np.sort(np.asarray(want, float).ravel())
    assert got.shape == want.shape, (got.shape, want.shape)
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


def _problems_after(go, mutate) -> list:
    d = copy.deepcopy(go)
    mutate(d)
    return GO.geometry_overlap_problems(d)


# ---- 1. the sidecar's contract ---------------------------------------------------------------------------------------------------
def test_the_committed_sidecar_loads_is_stamped_judge_free_and_names_its_sources(go):
    assert go["schema"] == GO.SCHEMA and go["judge"]["model"] == "none"
    pv = go["provenance"]
    assert pv["script"] == GO.BUILDER and len(pv["script_sha256"]) == 64
    names = [f["name"] for f in pv["source_files"]]
    assert names == ["figA_cK_overlap_pairs.csv", "figA_cK_overlap_null.csv", "figA_cK_overlap_values.csv",
                     "paper/notebooks/data/relu_arditi_cosines.json"]
    assert all(len(f["sha256"]) == 64 for f in pv["source_files"])
    assert not [s for s in GF._strings(go) if s.startswith(("/", "~")) or "/home/" in s]
    assert pv["checks"]["values_max_abs_diff"] < 1e-7 and "30 values" in pv["checks"]["layer12_cosines_equal"]
    assert go["lora_null"]["n_draws"] == 2000 and go["lora_null"]["seed"] == 1
    assert GO.geometry_overlap_problems(go) == []


def test_every_object_is_the_one_the_forward_sidecar_pins_for_its_role_and_seed(go):
    gf = GF.load_geometry_forward()
    fwd = {(o["family"], o["role"], o["seed"]): o["object_id"] for o in gf["objects"]}
    assert len(go["objects"]) == 14
    for o in go["objects"]:
        assert fwd[(o["family"], o["role"], o["seed"])] == o["object_id"], o["key"]
        assert o["registry"]["paper_label"] == GO.ROLE_LABEL[(o["family"], o["role"])]


def test_the_layer_12_cosines_are_the_cosine_datas(go):
    cos = json.loads(pathlib.Path(GO.COSINES).read_text())
    rc = GO.frame(go, "relu_cosine")
    rc = rc[rc.layer.eq(12)].set_index(["vector", "a", "b"]).cosine
    e = cos["cross_pairs"][0]
    assert e["object_a"] == "constitution_relu" and e["object_b"] == "control_relu"
    assert rc[("write", f"C_relu:{e['seed_a']}", f"K_relu:{e['seed_b']}")] == pytest.approx(e["write"], rel=GO.LAYER12_REL)
    assert GO.layer12_problems(go) == []


@pytest.mark.parametrize("mutate, phrase", [
    (lambda d: d.update(schema="appendix_geometry_overlap/0"), "schema"),
    (lambda d: d["judge"].update(model="minimax-m3"), "judge stamp"),
    (lambda d: d["objects"].pop(0), "objects:"),
    (lambda d: d["objects"][0]["registry"].update(paper_label="control_lora"), "registry claim"),
    (lambda d: d["lora_overlap"]["rows"].pop(), "lora_overlap must hold"),
    (lambda d: d["lora_overlap"]["rows"][0].__setitem__(6, 1.5), "share outside"),
    (lambda d: d["relu_cosine"]["rows"].pop(), "relu_cosine must hold"),
    (lambda d: d["lora_null"]["rows"][0].__setitem__(4, 9.0), "quantiles out of order"),
    (lambda d: d["lora_null"]["expected"].update(write=0.5), "lora_null.expected"),
    (lambda d: d["provenance"]["source_files"][0].update(sha256="x"), "source_files"),
    (lambda d: d["provenance"]["source_files"].append({"name": "/home/someone/x.csv", "sha256": "0" * 64}), "absolute or home paths"),
])
def test_each_way_to_break_the_sidecar_is_named(go, mutate, phrase):
    p = _problems_after(go, mutate)
    assert p and any(phrase in x for x in p), p


def test_a_layer_12_cosine_that_is_not_the_cosine_datas_is_refused(go, tmp_path):
    d = copy.deepcopy(go)
    rows = d["relu_cosine"]["rows"]
    i = next(k for k, r in enumerate(rows) if r[0] == "write" and r[1] == 12 and r[2] == "CK")
    rows[i][5] += 1e-3
    assert any("layer-12 write cosine" in x for x in GO.geometry_overlap_problems(d))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d))
    with pytest.raises(GO.GeometryOverlapSidecarError, match="layer-12"):
        GO.load_geometry_overlap(str(path))


# ---- 2. the numbers the text quotes ------------------------------------------------------------------------------------------------
def test_the_numbers_the_appendix_text_quotes(go):
    t = GO.text_numbers(go).set_index("key").value
    assert round(t["lora_write_CinK_mean_over_layers_pct"]) == 14 and round(t["lora_write_KinC_mean_over_layers_pct"]) == 19
    assert tuple(round(v, 1) for v in t["lora_write_CK_per_layer_pct"]) == (8.5, 27.5)
    assert round(t["lora_write_random_null_mean_pct"], 2) == 0.16
    lo, hi = t["lora_write_in_safety_per_layer_pct"]
    assert round(lo, 1) == 0.1 and round(hi, 1) == 0.5
    assert tuple(round(v) for v in t["lora_write_seed_ceiling_per_layer_pct"]) == (47, 68)
    assert round(t["relu_write_CK_max"], 2) == 0.64 and round(t["relu_write_safety_max"], 2) == 0.07


def test_the_band_is_the_envelope_of_the_six_drawn_objects_nulls(go):
    nl = GO.frame(go, "lora_null")
    b = GO.band_frame(go).set_index(["space", "layer"])
    g = nl[nl.measured.str[0].isin(["C", "K"]) & nl.space.eq("write") & nl.layer.eq(15)]
    assert len(g) == 6 and b.at[("write", 15), "lo"] == g["q0.025"].min() and b.at[("write", 15), "hi"] == g["q0.975"].max()
    assert not nl[nl.measured.eq("S_lora:-")].empty                            # the safety LoRA's null is kept, not drawn


def test_the_table_lists_every_series_and_marks_the_two_not_drawn(go):
    t = GO.table_overlap(go)
    assert len(t) == 2 * len(GO.LORA_SERIES) + 2 * len(GO.RELU_SERIES)
    assert set(t[~t.drawn].series) == set(GO.NOT_DRAWN)
    assert t.set_index(["panel", "space", "series"]).n.to_dict()[("a", "write", "CinK")] == 9


# ---- 3. the canvas ------------------------------------------------------------------------------------------------------------------
def test_panel_a_draws_every_series_mean_range_and_band(go, fig):
    s = GO.summary_frame(go)
    axes = _axes(fig)
    for ax, space in zip(axes[:2], ("write", "read")):
        a = s[s.panel.eq("a") & s.space.eq(space) & s.series.isin(list(GO.SERIES_A))]
        ys = np.concatenate([np.asarray(m.get_ydata(), float) for m in _marks(ax)])
        _same(ys, 100 * a["mean"])
        segs = np.array([[seg[:, 1].min(), seg[:, 1].max()] for c in ax.collections if isinstance(c, LineCollection) for seg in c.get_segments()])
        _same(segs, 100 * a[["min", "max"]].values)
        assert ax.get_yscale() == "log"
        band = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(band) == 1
        b = GO.band_frame(go)
        b = b[b.space.eq(space)]
        verts = band[0].get_paths()[0].vertices
        assert np.isclose(verts[:, 1].min(), 100 * b.lo.min()) and np.isclose(verts[:, 1].max(), 100 * b.hi.max())


def test_theta_c_shares_are_filled_and_theta_k_shares_open(go, fig):
    ax = _axes(fig)[0]
    s = GO.summary_frame(go)
    marks = {round(float(m.get_ydata()[0]), 9): m for m in _marks(ax)}
    for series, (_lab, ck, filled, off) in GO.SERIES_A.items():
        q = s[s.panel.eq("a") & s.space.eq("write") & s.series.eq(series)].sort_values("layer")
        m, col = marks[round(100 * float(q["mean"].iat[0]), 9)], to_hex(GO._colour(ck))
        np.testing.assert_allclose(m.get_xdata(), q.layer.values + off)
        face, edge = to_hex(m.get_markerfacecolor()), to_hex(m.get_markeredgecolor())
        assert (face, edge) == ((col, "#ffffff") if filled else ("#ffffff", col)), series
        assert m.get_marker() == "s"


def test_panel_b_draws_every_series_mean_and_range(go, fig):
    s = GO.summary_frame(go)
    for ax, vec in zip(_axes(fig)[2:], ("read", "write")):
        b = s[s.panel.eq("b") & s.space.eq(vec)]
        lines = [ln for ln in ax.get_lines() if len(ln.get_ydata()) == GO.N_LAYERS]
        assert len(lines) == len(GO.RELU_SERIES)
        _same(np.concatenate([ln.get_ydata() for ln in lines]), b["mean"])
        shades = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(shades) == len(GO.RELU_SERIES)
        for series in GO.RELU_SERIES:
            q = b[b.series.eq(series)]
            assert any(np.isclose(sh.get_paths()[0].vertices[:, 1].max(), q["max"].max()) for sh in shades), series
        assert len(ax.patches) == 1                                     # the grey band, +-2/sqrt(d)
        ys = ax.patches[0].get_path().transformed(ax.patches[0].get_patch_transform()).vertices[:, 1]
        assert np.isclose(ys.max(), GO.random_cosine_band(go)) and np.isclose(ys.min(), -GO.random_cosine_band(go))


def test_the_figure_has_panel_letters_a_and_b_and_four_axes(fig):
    """The manuscript's 5.5 x 4.44 in canvas: four axes, each labelled over its top-left corner, and the letters a and b."""
    axes = _axes(fig)
    assert tuple(fig.get_size_inches()) == GO.CANVAS_IN
    labels = [[t.get_text() for t in ax.texts if t.get_transform() == ax.transAxes] for ax in axes]
    assert labels == [["Output (write) space, span of B"], ["Input (read) space, span of A"], ["Read vectors (gate_w)"],
                      ["Write vectors (directions)"]]
    assert sorted(t.get_text() for t in fig.texts if len(t.get_text()) == 1) == ["a", "b"]


# ---- 4. registry binding, saving, environment ------------------------------------------------------------------------------------
def test_registry_check_flags_an_unknown_id_and_a_wrong_label(go):
    t = pd.DataFrame([dict(object_id=o["object_id"], paper_label=o["registry"]["paper_label"], fit_model_id=o["registry"]["fit_model_id"],
                           fit_replicate_label=o["registry"]["fit_replicate_label"], derivation=o["registry"]["derivation"],
                           parent_object_ids_json=json.dumps(o["registry"]["parents"])) for o in go["objects"]])
    assert GO.registry_check(go, t).known.all()
    k = next(o["object_id"] for o in go["objects"] if o["role"] == "K")
    s = next(o["object_id"] for o in go["objects"] if o["role"] == "S")
    bad = t.copy()
    bad.loc[bad.object_id.eq(k), "paper_label"] = "constitution_relu"
    bad = bad[bad.object_id.ne(s)]
    chk = GO.registry_check(go, bad).set_index("object_id")
    assert chk.loc[k, "mismatch"] == "paper_label" and chk.loc[s, "mismatch"] == "not in the registry" and int((~chk.known).sum()) == 2


def test_the_figure_saves_three_formats_into_figs_and_every_mirror(go, tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    monkeypatch.setenv(AC.MIRROR_ENV, str(mirror))
    GO.overlap_figure(go)
    for d in (pathlib.Path(R.FIGS), mirror):
        for ext in AC.FORMATS:
            assert (d / f"{GO.FIGURE}.{ext}").is_file(), (d, ext)


def test_the_module_imports_no_torch():
    src = (NOTEBOOKS / "appendix_geometry_overlap.py").read_text()
    assert "import torch" not in src and "from torch" not in src
